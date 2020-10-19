use std::fmt;
use std::rc::Rc;

use crate::analysis;
use crate::disasm::{Disassembler64, DestOperand, Operation};
use crate::exec_state::{self, Constraint, Memory, MergeStateCache};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::light_byteorder::ReadLittleEndian;
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandCtx, OperandType,
    ArithOpType,
};
use crate::{BinaryFile, VirtualAddress64};

const FLAGS_INDEX: usize = 0x10;
const STATE_OPERANDS: usize = FLAGS_INDEX + 6;

pub struct ExecutionState<'e> {
    // 16 registers, 10 xmm registers with 4 parts each, 6 flags
    state: [Operand<'e>; 0x10 + 0x6],
    xmm: Rc<[Operand<'e>; 0x10 * 4]>,
    cached_low_registers: CachedLowRegisters<'e>,
    memory: Memory<'e>,
    unresolved_constraint: Option<Constraint<'e>>,
    resolved_constraint: Option<Constraint<'e>>,
    /// A constraint where the address is resolved already
    memory_constraint: Option<Constraint<'e>>,
    ctx: OperandCtx<'e>,
    binary: Option<&'e BinaryFile<VirtualAddress64>>,
    /// Lazily update flags since a lot of instructions set them and
    /// they get discarded later.
    pending_flags: Option<(ArithOperand<'e>, MemAccessSize)>,
}

// Manual impl since derive adds an unwanted inline hint
impl<'e> Clone for ExecutionState<'e> {
    fn clone(&self) -> Self {
        Self {
            // Codegen optimization: memory cloning isn't a memcpy,
            // doing all memcpys at the end, after other function calls
            // or branching code generally avoids temporaries.
            memory: self.memory.clone(),
            cached_low_registers: self.cached_low_registers.clone(),
            xmm: self.xmm.clone(),
            state: self.state,
            resolved_constraint: self.resolved_constraint,
            unresolved_constraint: self.unresolved_constraint,
            memory_constraint: self.memory_constraint,
            ctx: self.ctx,
            binary: self.binary,
            pending_flags: self.pending_flags,
        }
    }
}

impl<'e> fmt::Debug for ExecutionState<'e> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExecutionStateX86_64")
    }
}

/// Caches eax/ax/al/ah resolving.
#[derive(Debug, Clone)]
struct CachedLowRegisters<'e> {
    registers: [[Option<Operand<'e>>; 3]; 0x10],
}

impl<'e> CachedLowRegisters<'e> {
    fn new() -> CachedLowRegisters<'e> {
        CachedLowRegisters {
            registers: [[None; 3]; 0x10],
        }
    }

    fn get_32(&self, register: u8) -> Option<Operand<'e>> {
        self.registers[register as usize][2]
    }

    fn get_16(&self, register: u8) -> Option<Operand<'e>> {
        self.registers[register as usize][0]
    }

    fn get_low8(&self, register: u8) -> Option<Operand<'e>> {
        self.registers[register as usize][1]
    }

    fn set_32(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][2] = Some(value);
    }

    fn set_16(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][0] = Some(value);
    }

    fn set_low8(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][1] = Some(value);
    }

    fn invalidate(&mut self, register: u8) {
        self.registers[register as usize] = [None; 3];
    }
}

/// Handles regular &'mut Operand assign for regs,
/// and the more complicated one for memory
enum Destination<'a, 'e> {
    Oper(&'a mut Operand<'e>),
    Register32(&'a mut Operand<'e>),
    Register16(&'a mut Operand<'e>),
    Register8High(&'a mut Operand<'e>),
    Register8Low(&'a mut Operand<'e>),
    Memory(&'a mut ExecutionState<'e>, Operand<'e>, MemAccessSize),
    Nop,
}

impl<'a, 'e> Destination<'a, 'e> {
    fn set(self, value: Operand<'e>, ctx: OperandCtx<'e>) {
        match self {
            Destination::Oper(o) => {
                *o = value;
            }
            Destination::Register32(o) => {
                // 32-bit register dest clears high bits (16- or 8-bit dests don't)
                let masked = if value.relevant_bits().end > 32 {
                    ctx.and_const(value, 0xffff_ffff)
                } else {
                    value
                };
                *o = masked;
            }
            Destination::Register16(o) => {
                let old = *o;
                let masked = if value.relevant_bits().end > 16 {
                    ctx.and_const(value, 0xffff)
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.end <= 16 {
                    masked
                } else {
                    ctx.or(
                        ctx.and_const(old, 0xffff_ffff_ffff_0000),
                        masked,
                    )
                };
                *o = new;
            }
            Destination::Register8High(o) => {
                let old = *o;
                let masked = if value.relevant_bits().end > 8 {
                    ctx.and_const(value, 0xff)
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.start >= 8 && old_bits.end <= 16 {
                    ctx.lsh_const(masked, 8)
                } else {
                    ctx.or(
                        ctx.and_const(old, 0xffff_ffff_ffff_00ff),
                        ctx.lsh_const(masked, 8),
                    )
                };
                *o = new;
            }
            Destination::Register8Low(o) => {
                let old = *o;
                let masked = if value.relevant_bits().end > 8 {
                    ctx.and_const(value, 0xff)
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.end <= 8 {
                    masked
                } else {
                    ctx.or(
                        ctx.and_const(old, 0xffff_ffff_ffff_ff00),
                        masked,
                    )
                };
                *o = new;
            }
            Destination::Memory(state, addr, size) => {
                if let Some((base, offset)) = Operand::const_offset(addr, ctx) {
                    let offset_8 = offset & 7;
                    let offset_rest = offset & !7;
                    if offset_8 != 0 {
                        let size_bits = size.bits() as u64;
                        let low_base = ctx.add_const(base, offset_rest);
                        let low_old = state.read_memory_impl(low_base, MemAccessSize::Mem64)
                            .unwrap_or_else(|| ctx.mem64(low_base));

                        let mask_low = offset_8 * 8;
                        let mask_high = (mask_low + size_bits).min(0x40);
                        let mask = !0u64 >> mask_low << mask_low <<
                            (0x40 - mask_high) >> (0x40 - mask_high);
                        let low_value = ctx.or(
                            ctx.and_const(
                                ctx.lsh_const(
                                    value,
                                    8 * offset_8,
                                ),
                                mask
                            ),
                            ctx.and_const(
                                low_old,
                                !mask,
                            ),
                        );
                        state.memory.set(low_base, low_value);
                        let needs_high = mask_low + size_bits > 0x40;
                        if needs_high {
                            let high_base = ctx.add_const(
                                base,
                                offset_rest.wrapping_add(8),
                            );
                            let high_old = state.read_memory_impl(high_base, MemAccessSize::Mem64)
                                .unwrap_or_else(|| ctx.mem64(high_base));
                            let mask = !0u64 >> (0x40 - (mask_low + size_bits - 0x40));
                            let high_value = ctx.or(
                                ctx.and_const(
                                    ctx.rsh_const(
                                        value,
                                        0x40 - 8 * offset_8,
                                    ),
                                    mask,
                                ),
                                ctx.and_const(
                                    high_old,
                                    !mask,
                                ),
                            );
                            state.memory.set(high_base, high_value);
                        }
                        return;
                    }
                }
                let value = match size {
                    MemAccessSize::Mem64 => value,
                    _ => {
                        let old = state.read_memory_impl(addr, MemAccessSize::Mem64)
                            .unwrap_or_else(|| ctx.mem64(addr));
                        let new_mask = match size {
                            MemAccessSize::Mem8 => 0xff,
                            MemAccessSize::Mem16 => 0xffff,
                            MemAccessSize::Mem32 | _ => 0xffff_ffff,
                        };
                        ctx.or(
                            ctx.and_const(value, new_mask),
                            ctx.and_const(old, !new_mask),
                        )
                    }
                };
                state.memory.set(addr, value);
            }
            Destination::Nop => (),
        }
    }
}

impl<'e> ExecutionStateTrait<'e> for ExecutionState<'e> {
    type VirtualAddress = VirtualAddress64;
    type Disassembler = Disassembler64<'e>;
    const WORD_SIZE: MemAccessSize = MemAccessSize::Mem64;

    fn maybe_convert_memory_immutable(&mut self, limit: usize) {
        self.memory.map.maybe_convert_immutable(limit);
    }

    fn add_resolved_constraint(&mut self, constraint: Constraint<'e>) {
        // If recently accessed memory location also has the same resolved value as this
        // constraint, add it to the constraint as well.
        // As it only works on recently accessed memory, it is very much a good-enough
        // heuristic and not a foolproof solution. A complete, but slower way would
        // be having a state merge function that is merges constraints depending on
        // what the values inside constraints were merged to.
        let ctx = self.ctx();
        if let OperandType::Arithmetic(ref arith) = *constraint.0.ty() {
            if arith.left.if_constant().is_some() {
                if let Some((addr, size)) =
                    self.memory.fast_reverse_lookup(ctx, arith.right, u64::max_value())
                {
                    let val = ctx.mem_variable_rc(size, addr);
                    self.add_memory_constraint(ctx.arithmetic(arith.ty, arith.left, val));
                }
            } else if arith.left.if_constant().is_some() {
                if let Some((addr, size)) =
                    self.memory.fast_reverse_lookup(ctx, arith.left, u64::max_value())
                {
                    let val = ctx.mem_variable_rc(size, addr);
                    self.add_memory_constraint(ctx.arithmetic(arith.ty, val, arith.right));
                }
            }
        }
        self.resolved_constraint = Some(constraint);
    }

    fn add_unresolved_constraint(&mut self, constraint: Constraint<'e>) {
        self.unresolved_constraint = Some(constraint);
    }

    fn move_to(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        let ctx = self.ctx();
        let resolved = self.resolve(value);
        let dest = self.get_dest_invalidate_constraints(dest);
        dest.set(resolved, ctx);
    }

    fn move_resolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        let ctx = self.ctx;
        self.unresolved_constraint = None;
        let dest = self.get_dest(dest, true);
        dest.set(value, ctx);
    }

    fn set_flags_resolved(&mut self, arith: &ArithOperand<'e>, size: MemAccessSize) {
        self.pending_flags = Some((*arith, size));
        // Could try to do smarter invalidation, but since in practice unresolved
        // constraints always are bunch of flags, invalidate it completely.
        self.unresolved_constraint = None;
    }

    #[inline]
    fn ctx(&self) -> OperandCtx<'e> {
        self.ctx
    }

    fn resolve(&mut self, operand: Operand<'e>) -> Operand<'e> {
        self.resolve(operand)
    }

    fn update(&mut self, operation: &Operation<'e>) {
        self.update(operation)
    }

    fn resolve_apply_constraints(&mut self, mut op: Operand<'e>) -> Operand<'e> {
        if let Some(ref constraint) = self.unresolved_constraint {
            op = constraint.apply_to(self.ctx, op);
        }
        let val = self.resolve(op);
        if let Some(ref constraint) = self.resolved_constraint {
            constraint.apply_to(self.ctx, val)
        } else {
            val
        }
    }

    fn read_memory(&mut self, address: Operand<'e>, size: MemAccessSize) -> Operand<'e> {
        self.read_memory_impl(address, size)
            .unwrap_or_else(|| self.ctx.mem_variable_rc(size, address))
    }

    fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.unresolve(val)
    }

    fn unresolve_memory(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.unresolve_memory(val)
    }

    fn merge_states(
        old: &mut Self,
        new: &mut Self,
        cache: &mut MergeStateCache<'e>,
    ) -> Option<Self> {
        merge_states(old, new, cache)
    }

    fn apply_call(&mut self, ret: VirtualAddress64) {
        let ctx = self.ctx;
        let rsp = ctx.register(4);
        self.move_to(
            &DestOperand::from_oper(rsp),
            ctx.sub_const(rsp, 8),
        );
        self.move_to(
            &DestOperand::from_oper(ctx.mem64(rsp)),
            ctx.constant(ret.0),
        );
    }

    fn initial_state(
        ctx: OperandCtx<'e>,
        binary: &'e BinaryFile<VirtualAddress64>,
    ) -> ExecutionState<'e> {
        ExecutionState::with_binary(binary, ctx)
    }

    fn find_functions_with_callers(file: &crate::BinaryFile<Self::VirtualAddress>)
        -> Vec<analysis::FuncCallPair<Self::VirtualAddress>>
    {
        crate::analysis::find_functions_with_callers_x86_64(file)
    }

    fn find_functions_from_calls(
        code: &[u8],
        section_base: Self::VirtualAddress,
        out: &mut Vec<Self::VirtualAddress>
    ) {
        crate::analysis::find_functions_from_calls_x86_64(code, section_base, out)
    }

    fn find_relocs(
        file: &BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<Self::VirtualAddress>, crate::OutOfBounds> {
        crate::analysis::find_relocs_x86_64(file)
    }

    fn function_ranges_from_exception_info(
        file: &crate::BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<(u32, u32)>, crate::OutOfBounds> {
        crate::analysis::function_ranges_from_exception_info_x86_64(file)
    }

    fn value_limits(&mut self, value: Operand<'e>) -> (u64, u64) {
        let mut result = (0, u64::max_value());
        if let Some(constraint) = self.resolved_constraint {
            let new = crate::exec_state::value_limits_recurse(constraint.0, value);
            result.0 = result.0.max(new.0);
            result.1 = result.1.min(new.1);
        }
        if let Some(constraint) = self.memory_constraint {
            let ctx = self.ctx();
            let transformed = ctx.transform(constraint.0, 8, |op| match *op.ty() {
                OperandType::Memory(ref mem) => Some(self.read_memory(mem.address, mem.size)),
                _ => None,
            });
            let new = crate::exec_state::value_limits_recurse(transformed, value);
            result.0 = result.0.max(new.0);
            result.1 = result.1.min(new.1);
        }
        result
    }

    unsafe fn clone_to(&self, out: *mut Self) {
        use std::ptr::write;
        let Self {
            memory,
            cached_low_registers,
            xmm,
            state,
            resolved_constraint,
            unresolved_constraint,
            memory_constraint,
            ctx,
            binary,
            pending_flags,
        } = self;
        write(&mut (*out).memory, memory.clone());
        write(&mut (*out).cached_low_registers, cached_low_registers.clone());
        write(&mut (*out).xmm, xmm.clone());
        write(&mut (*out).state, *state);
        write(&mut (*out).resolved_constraint, *resolved_constraint);
        write(&mut (*out).unresolved_constraint, *unresolved_constraint);
        write(&mut (*out).memory_constraint, *memory_constraint);
        write(&mut (*out).ctx, *ctx);
        write(&mut (*out).binary, *binary);
        write(&mut (*out).pending_flags, *pending_flags);
    }
}

impl<'e> ExecutionState<'e> {
    pub fn new<'b>(
        ctx: OperandCtx<'b>,
    ) -> ExecutionState<'b> {
        let dummy = ctx.const_0();
        let mut state = [dummy; STATE_OPERANDS];
        let mut xmm = Rc::new([dummy; 0x10 * 4]);
        let xmm_mut = Rc::make_mut(&mut xmm);
        for i in 0..16 {
            state[i] = ctx.register(i as u8);
            for j in 0..4 {
                xmm_mut[i * 4 + j] = ctx.xmm(i as u8, j as u8);
            }
        }
        for i in 0..5 {
            state[FLAGS_INDEX + i] = ctx.flag_by_index(i).clone();
        }
        // Direction
        state[FLAGS_INDEX + 5] = ctx.const_0();
        ExecutionState {
            state,
            xmm,
            cached_low_registers: CachedLowRegisters::new(),
            memory: Memory::new(),
            unresolved_constraint: None,
            resolved_constraint: None,
            memory_constraint: None,
            ctx,
            binary: None,
            pending_flags: None,
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile<VirtualAddress64>,
        ctx: OperandCtx<'b>,
    ) -> ExecutionState<'b> {
        let mut result = ExecutionState::new(ctx);
        result.binary = Some(binary);
        result
    }

    pub fn update(&mut self, operation: &Operation<'e>) {
        let ctx = self.ctx;
        match *operation {
            Operation::Move(ref dest, value, cond) => {
                let value = value.clone();
                if let Some(cond) = cond {
                    match self.resolve(cond).if_constant() {
                        Some(0) => (),
                        Some(_) => {
                            self.move_to(&dest, value);
                        }
                        None => {
                            self.get_dest_invalidate_constraints(&dest)
                                .set(ctx.new_undef(), ctx)
                        }
                    }
                } else {
                    self.move_to(&dest, value);
                }
            }
            Operation::MoveSet(ref moves) => {
                let resolved: Vec<Operand<'e>> = moves.iter()
                    .map(|x| self.resolve(x.1))
                    .collect();
                for (tp, val) in moves.iter().zip(resolved.into_iter()) {
                    self.get_dest_invalidate_constraints(&tp.0)
                        .set(val, ctx);
                }
            }
            Operation::Call(_) => {
                self.unresolved_constraint = None;
                if let Some(ref mut c) = self.resolved_constraint {
                    if c.invalidate_memory(ctx) == crate::exec_state::ConstraintFullyInvalid::Yes {
                        self.resolved_constraint = None
                    }
                }
                self.memory_constraint = None;
                static UNDEF_REGISTERS: &[u8] = &[0, 1, 2, 4, 8, 9, 10, 11];
                for &i in UNDEF_REGISTERS.iter() {
                    self.state[i as usize] = ctx.new_undef();
                    self.cached_low_registers.invalidate(i);
                }
                for i in 0..5 {
                    self.state[FLAGS_INDEX + i] = ctx.new_undef();
                }
                self.state[FLAGS_INDEX + Flag::Direction as usize] = ctx.const_0();
            }
            Operation::Special(_) => {
            }
            Operation::SetFlags(ref arith, size) => {
                let left = self.resolve(arith.left);
                let right = self.resolve(arith.right);
                let arith = ArithOperand {
                    left,
                    right,
                    ty: arith.ty,
                };
                self.pending_flags = Some((arith, size));
                // Could try to do smarter invalidation, but since in practice unresolved
                // constraints always are bunch of flags, invalidate it completely.
                self.unresolved_constraint = None;
            }
            Operation::Jump { .. } | Operation::Return(_) | Operation::Error(..) => (),
        }
    }

    /// Updates arithmetic flags if there are any pending flag updates.
    /// Must be called before accessing (both read or write) arithmetic flags.
    /// Obvously can just zero pending_flags if all flags are written over though.
    fn update_flags(&mut self) {
        if let Some((arith, size)) = self.pending_flags.take() {
            self.set_flags(arith, size);
        }
    }

    fn set_flags(
        &mut self,
        arith: ArithOperand<'e>,
        size: MemAccessSize,
    ) {
        use crate::operand::ArithOpType::*;
        let ctx = self.ctx;
        let result = ctx.arithmetic(arith.ty, arith.left, arith.right);
        match arith.ty {
            Add | Sub => {
                let sign_bit = match size {
                    MemAccessSize::Mem8 => 0x80u64,
                    MemAccessSize::Mem16 => 0x8000,
                    MemAccessSize::Mem32 => 0x8000_0000,
                    MemAccessSize::Mem64 => 0x8000_0000_0000_0000,
                };
                let mask = (sign_bit << 1).wrapping_sub(1);
                let left = ctx.and_const(arith.left, mask);
                let right = ctx.and_const(arith.right, mask);
                let result = ctx.and_const(result, mask);
                let carry;
                let overflow;
                if arith.ty == Add {
                    carry = ctx.gt(left, result);
                    // (right sge 0) == (left sgt result)
                    overflow = ctx.eq(
                        ctx.gt_const_left(sign_bit, right),
                        ctx.gt_signed(left, result, size),
                    );
                } else {
                    carry = ctx.gt(result, left);
                    // (right sge 0) == (result sgt left)
                    overflow = ctx.eq(
                        ctx.gt_const_left(sign_bit, right),
                        ctx.gt_signed(result, left, size),
                    );
                }
                self.state[FLAGS_INDEX + Flag::Carry as usize] = carry;
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = overflow;
                self.result_flags(result, size);
            }
            Xor | And | Or => {
                let zero = ctx.const_0();
                self.state[FLAGS_INDEX + Flag::Carry as usize] = zero;
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = zero;
                self.result_flags(result, size);
            }
            Lsh | Rsh  => {
                self.state[FLAGS_INDEX + Flag::Carry as usize] = ctx.new_undef();
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = ctx.new_undef();
                self.result_flags(result, size);
            }
            _ => {
                // NOTE: Relies on direction being index 5 so it won't change here
                for i in 0..5 {
                    self.state[FLAGS_INDEX + i] = ctx.new_undef();
                }
            }
        }
    }

    fn result_flags(
        &mut self,
        result: Operand<'e>,
        size: MemAccessSize,
    ) {
        let parity;
        let ctx = self.ctx;
        let zero = ctx.eq_const(result, 0);
        let sign_bit = match size {
            MemAccessSize::Mem8 => 0x80,
            MemAccessSize::Mem16 => 0x8000,
            MemAccessSize::Mem32 => 0x8000_0000,
            MemAccessSize::Mem64 => 0x8000_0000_0000_0000,
        };
        let sign = ctx.neq_const(
            ctx.and_const(
                result,
                sign_bit,
            ),
            0,
        );
        // Parity is defined to be just lowest byte so it doesn't need special handling for
        // 64 bits.
        parity = ctx.arithmetic(ArithOpType::Parity, result, ctx.const_0());
        self.state[FLAGS_INDEX + Flag::Zero as usize] = zero;
        self.state[FLAGS_INDEX + Flag::Sign as usize] = sign;
        self.state[FLAGS_INDEX + Flag::Parity as usize] = parity;
    }

    /// Makes all of memory undefined
    pub fn clear_memory(&mut self) {
        self.memory = Memory::new();
    }

    fn get_dest_invalidate_constraints<'s>(
        &'s mut self,
        dest: &DestOperand<'e>,
    ) -> Destination<'s, 'e> {
        let ctx = self.ctx();
        self.unresolved_constraint = match self.unresolved_constraint {
            Some(ref s) => s.invalidate_dest_operand(ctx, dest),
            None => None,
        };
        if let DestOperand::Memory(_) = dest {
            if let Some(ref mut s) = self.resolved_constraint {
                if s.invalidate_memory(ctx) == crate::exec_state::ConstraintFullyInvalid::Yes {
                    self.resolved_constraint = None;
                }
            }
            self.memory_constraint = None;
        }
        self.get_dest(dest, false)
    }

    fn get_dest<'s>(
        &'s mut self,
        dest: &DestOperand<'e>,
        dest_is_resolved: bool,
    ) -> Destination<'s, 'e> {
        match *dest {
            DestOperand::Register64(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Oper(&mut self.state[(reg.0 & 0xf) as usize])
            }
            DestOperand::Register32(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register32(&mut self.state[(reg.0 & 0xf) as usize])
            }
            DestOperand::Register16(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register16(&mut self.state[(reg.0 & 0xf) as usize])
            }
            DestOperand::Register8High(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register8High(&mut self.state[(reg.0 & 0xf) as usize])
            }
            DestOperand::Register8Low(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register8Low(&mut self.state[(reg.0 & 0xf) as usize])
            }
            DestOperand::Fpu(_) => Destination::Nop,
            DestOperand::Xmm(reg, word) => {
                let xmm = Rc::make_mut(&mut self.xmm);
                Destination::Oper(&mut xmm[(reg & 0xf) as usize * 4 + (word & 3) as usize])
            }
            DestOperand::Flag(flag) => {
                self.update_flags();
                Destination::Oper(&mut self.state[FLAGS_INDEX + flag as usize])
            }
            DestOperand::Memory(ref mem) => {
                let address = if dest_is_resolved {
                    mem.address.clone()
                } else {
                    self.resolve(mem.address)
                };
                Destination::Memory(self, address, mem.size)
            }
        }
    }

    /// Returns None if the value won't change.
    fn resolve_mem(&mut self, mem: &MemAccess<'e>) -> Option<Operand<'e>> {
        let address = self.resolve(mem.address);

        self.read_memory_impl(address, mem.size)
            .or_else(|| {
                // Just copy the input value if address didn't change
                if address == mem.address {
                    None
                } else {
                    Some(self.ctx.mem_variable_rc(mem.size, address))
                }
            })
    }

    /// Resolves memory operand for which the address is already resolved.
    ///
    /// Returns `None` if the memory at `address` hasn't changed.
    fn read_memory_impl(
        &mut self,
        address: Operand<'e>,
        size: MemAccessSize,
    ) -> Option<Operand<'e>> {
        let ctx = self.ctx;
        let mask = match size {
            MemAccessSize::Mem8 => 0xffu32,
            MemAccessSize::Mem16 => 0xffff,
            MemAccessSize::Mem32 => 0xffff_ffff,
            MemAccessSize::Mem64 => 0,
        };
        // Use 8-aligned addresses if there's a const offset
        if let Some((base, offset)) = Operand::const_offset(address, ctx) {
            let size_bytes = size.bits() / 8;
            let offset_8 = offset as u32 & 7;
            let offset_rest = offset & !7;
            if offset_8 != 0 {
                let low_base = ctx.add_const(base, offset_rest);
                let low = self.memory.get(low_base)
                    .or_else(|| {
                        // Avoid reading Mem64 if it's not necessary as it may go
                        // past binary end where a smaller read wouldn't
                        let size = match offset_8 + size_bytes {
                            1 => MemAccessSize::Mem8,
                            2 => MemAccessSize::Mem16,
                            3 | 4 => MemAccessSize::Mem32,
                            _ => MemAccessSize::Mem64,
                        };
                        self.resolve_binary_constant_mem(low_base, size)
                    })
                    .unwrap_or_else(|| ctx.mem64(low_base));
                let low = ctx.rsh_const(low, offset_8 as u64 * 8);
                let combined = if offset_8 + size_bytes > 8 {
                    let high_base = ctx.add_const(base, offset_rest.wrapping_add(8));
                    let high = self.memory.get(high_base)
                        .or_else(|| {
                            let size = match (offset_8 + size_bytes) - 8 {
                                1 => MemAccessSize::Mem8,
                                2 => MemAccessSize::Mem16,
                                3 | 4 => MemAccessSize::Mem32,
                                _ => MemAccessSize::Mem64,
                            };
                            self.resolve_binary_constant_mem(high_base, size)
                        })
                        .unwrap_or_else(|| ctx.mem64(high_base));
                    let high = ctx.lsh_const(high, (0x40 - offset_8 * 8) as u64);
                    ctx.or(low, high)
                } else {
                    low
                };
                let masked = if size != MemAccessSize::Mem64 {
                    ctx.and_const(combined, mask as u64)
                } else {
                    combined
                };
                return Some(masked);
            }
        }
        self.memory.get(address)
            .map(|operand| {
                if size != MemAccessSize::Mem64 {
                    ctx.and_const(operand, mask as u64)
                } else {
                    operand
                }
            })
            .or_else(|| self.resolve_binary_constant_mem(address, size))
    }

    fn resolve_binary_constant_mem(
        &self,
        address: Operand<'e>,
        size: MemAccessSize,
    ) -> Option<Operand<'e>> {
        if let Some(c) = address.if_constant() {
            let size_bytes = size.bits() / 8;
            // Simplify constants stored in code section (constant switch jumps etc)
            if let Some(end) = c.checked_add(size_bytes as u64) {
                let section = self.binary.and_then(|b| {
                    b.code_sections().find(|s| {
                        s.virtual_address.0 <= c &&
                            s.virtual_address.0 + s.virtual_size as u64 >= end
                    })
                });
                if let Some(section) = section {
                    let offset = (c - section.virtual_address.0) as usize;
                    let val = match size {
                        MemAccessSize::Mem8 => section.data[offset] as u64,
                        MemAccessSize::Mem16 => {
                            (&section.data[offset..]).read_u16().unwrap_or(0) as u64
                        }
                        MemAccessSize::Mem32 => {
                            (&section.data[offset..]).read_u32().unwrap_or(0) as u64
                        }
                        MemAccessSize::Mem64 => {
                            (&section.data[offset..]).read_u64().unwrap_or(0)
                        }
                    };
                    return Some(self.ctx.constant(val));
                }
            }
        }
        None
    }

    /// Checks cached/caches `reg & ff` masks.
    fn try_resolve_partial_register(
        &mut self,
        left: Operand<'e>,
        right: Operand<'e>,
    ) -> Option<Operand<'e>> {
        let ctx = self.ctx;
        let (const_op, c, other) = match left.if_constant() {
            Some(c) => (left, c, right),
            _ => match right.if_constant() {
                Some(c) => (right, c, left),
                _ => return None,
            }
        };
        let reg = other.if_register()?.0 & 0xf;
        if c <= 0xff {
            let op = match self.cached_low_registers.get_low8(reg) {
                None => {
                    let op = ctx.and_const(
                        self.state[reg as usize],
                        0xff,
                    );
                    self.cached_low_registers.set_low8(reg, op);
                    op
                }
                Some(x) => x,
            };
            if c == 0xff {
                Some(op)
            } else {
                Some(ctx.and(op, const_op))
            }
        } else if c <= 0xffff {
            let op = match self.cached_low_registers.get_16(reg) {
                None => {
                    let op = ctx.and_const(
                        self.state[reg as usize],
                        0xffff,
                    );
                    self.cached_low_registers.set_16(reg, op);
                    op
                }
                Some(x) => x,
            };
            if c == 0xffff {
                Some(op)
            } else {
                Some(ctx.and(op, const_op))
            }
        } else if c <= 0xffff_ffff {
            let op = match self.cached_low_registers.get_32(reg) {
                None => {
                    let op = ctx.and_const(
                        self.state[reg as usize],
                        0xffff_ffff
                    );
                    self.cached_low_registers.set_32(reg, op);
                    op
                }
                Some(x) => x,
            };
            if c == 0xffff_ffff {
                Some(op)
            } else {
                Some(ctx.and(op, const_op))
            }
        } else {
            None
        }
    }

    pub fn resolve(&mut self, value: Operand<'e>) -> Operand<'e> {
        match *value.ty() {
            OperandType::Register(reg) => {
                self.state[(reg.0 & 0xf) as usize]
            }
            OperandType::Xmm(reg, word) => {
                self.xmm[(reg & 0xf) as usize * 4 + (word & 3) as usize]
            }
            OperandType::Fpu(_) => value,
            OperandType::Flag(flag) => {
                self.update_flags();
                self.state[FLAGS_INDEX + flag as usize]
            }
            OperandType::Arithmetic(ref op) => {
                if op.ty == ArithOpType::And {
                    let r = self.try_resolve_partial_register(op.left, op.right);
                    if let Some(r) = r {
                        return r;
                    }
                };
                let left = self.resolve(op.left);
                let right = self.resolve(op.right);
                self.ctx.arithmetic(op.ty, left, right)
            }
            OperandType::ArithmeticFloat(ref op, size) => {
                let left = self.resolve(op.left);
                let right = self.resolve(op.right);
                self.ctx.float_arithmetic(op.ty, left, right, size)
            }
            OperandType::Constant(_) => value,
            OperandType::Custom(_) => value,
            OperandType::Memory(ref mem) => {
                self.resolve_mem(mem)
                    .unwrap_or_else(|| value)
            }
            OperandType::Undefined(_) => value,
            OperandType::SignExtend(val, from, to) => {
                let val = self.resolve(val);
                self.ctx.sign_extend(val, from, to)
            }
        }
    }

    /// Tries to find an register/memory address corresponding to a resolved value.
    pub fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        // TODO: Could also check xmm but who honestly uses it for unique storage
        for (reg, &val2) in self.state.iter().enumerate().take(0x10) {
            if val == val2 {
                return Some(self.ctx.register(reg as u8));
            }
        }
        None
    }

    /// Tries to find an memory address corresponding to a resolved value.
    pub fn unresolve_memory(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.memory.reverse_lookup(val).map(|x| self.ctx.mem64(x))
    }

    fn add_memory_constraint(&mut self, constraint: Operand<'e>) {
        if let Some(old) = self.memory_constraint {
            let ctx = self.ctx();
            self.memory_constraint = Some(Constraint(ctx.and(old.0, constraint)));
        } else {
            self.memory_constraint = Some(Constraint(constraint));
        }
    }
}


/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
pub fn merge_states<'a: 'r, 'r>(
    old: &'r mut ExecutionState<'a>,
    new: &'r mut ExecutionState<'a>,
    cache: &'r mut MergeStateCache<'a>,
) -> Option<ExecutionState<'a>> {
    old.update_flags();
    new.update_flags();

    fn check_eq<'e>(a: Operand<'e>, b: Operand<'e>) -> bool {
        a == b || a.is_undefined()
    }

    let ctx = old.ctx;
    fn merge<'e>(ctx: OperandCtx<'e>, a: Operand<'e>, b: Operand<'e>) -> Operand<'e> {
        match a == b || a.is_undefined() {
            true => a,
            false => ctx.new_undef(),
        }
    };

    let merged_ljec = if old.unresolved_constraint != new.unresolved_constraint {
        let mut result = None;
        // If one state has no constraint but matches the constrait of the other
        // state, the constraint should be kept on merge.
        if old.unresolved_constraint.is_none() {
            if let Some(ref con) = new.unresolved_constraint {
                // As long as we're working with flags, limiting to lowest bit
                // allows simplifying cases like (undef | 1)
                let lowest_bit = ctx.and_const(con.0, 1);
                match old.resolve_apply_constraints(lowest_bit).if_constant() {
                    Some(1) => result = Some(con.clone()),
                    _ => (),
                }
            }
        }
        if new.unresolved_constraint.is_none() {
            if let Some(ref con) = old.unresolved_constraint {
                // As long as we're working with flags, limiting to lowest bit
                // allows simplifying cases like (undef | 1)
                let lowest_bit = ctx.and_const(con.0, 1);
                match new.resolve_apply_constraints(lowest_bit).if_constant() {
                    Some(1) => result = Some(con.clone()),
                    _ => (),
                }
            }
        }
        Some(result)
    } else {
        None
    };
    let xmm_eq = {
        Rc::ptr_eq(&old.xmm, &new.xmm) ||
        old.xmm.iter().zip(new.xmm.iter())
            .all(|(&a, &b)| check_eq(a, b))
    };
    let resolved_constraint =
        exec_state::merge_constraint(ctx, old.resolved_constraint, new.resolved_constraint);
    let memory_constraint =
        exec_state::merge_constraint(ctx, old.memory_constraint, new.memory_constraint);
    let changed = (
            old.state.iter().zip(new.state.iter())
                .any(|(&a, &b)| !check_eq(a, b))
        ) || (
            if Rc::ptr_eq(&old.memory.map.map, &new.memory.map.map) {
                false
            } else {
                match cache.get_compare_result(&old.memory, &new.memory) {
                    Some(s) => s,
                    None => {
                        let result = old.memory.map.has_merge_changed(&new.memory.map);
                        cache.set_compare_result(&old.memory, &new.memory, result);
                        result
                    }
                }
            }
        ) ||
        !xmm_eq ||
        merged_ljec.as_ref().map(|x| *x != old.unresolved_constraint).unwrap_or(false) ||
        resolved_constraint != old.resolved_constraint ||
        memory_constraint != old.memory_constraint;
    if changed {
        let state = array_init::array_init(|i| merge(ctx, old.state[i], new.state[i]));
        let mut cached_low_registers = CachedLowRegisters::new();
        for i in 0..16 {
            let old_reg = &old.cached_low_registers.registers[i];
            let new_reg = &new.cached_low_registers.registers[i];
            for j in 0..old_reg.len() {
                if old_reg[j] == new_reg[j] {
                    // Doesn't merge things but sets them uncached if they differ
                    cached_low_registers.registers[i][j] = old_reg[j];
                }
            }
        }
        let mut xmm = old.xmm.clone();
        if !xmm_eq {
            let state = Rc::make_mut(&mut xmm);
            let old_xmm = &*old.xmm;
            let new_xmm = &*new.xmm;
            for i in 0..state.len() {
                state[i] = merge(ctx, old_xmm[i], new_xmm[i]);
            }
        }
        let memory = match cache.get_merge_result(&old.memory, &new.memory) {
            Some(s) => s,
            None => {
                let result = old.memory.merge(&new.memory, ctx);
                cache.set_merge_result(&old.memory, &new.memory, &result);
                result
            }
        };
        Some(ExecutionState {
            state,
            xmm,
            cached_low_registers,
            memory,
            unresolved_constraint: merged_ljec.unwrap_or_else(|| {
                // They were same, just use one from old
                old.unresolved_constraint.clone()
            }),
            resolved_constraint,
            memory_constraint,
            pending_flags: None,
            ctx,
            binary: old.binary,
        })
    } else {
        None
    }
}
