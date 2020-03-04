use std::fmt;

use crate::analysis;
use crate::disasm::{Disassembler32, DestOperand, Operation};
use crate::exec_state::{Constraint, Memory};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::light_byteorder::{ReadLittleEndian};
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
    ArithOpType,
};
use crate::{BinaryFile, VirtualAddress};

impl<'e> ExecutionStateTrait<'e> for ExecutionState<'e> {
    type VirtualAddress = VirtualAddress;
    type Disassembler = Disassembler32<'e>;

    fn maybe_convert_memory_immutable(&mut self) {
        self.memory.maybe_convert_immutable();
    }

    fn add_resolved_constraint(&mut self, constraint: Constraint<'e>) {
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

    fn ctx(&self) -> &'e OperandContext {
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

    fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.unresolve(val)
    }

    fn unresolve_memory(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.unresolve_memory(val)
    }

    fn merge_states(old: &mut Self, new: &mut Self) -> Option<Self> {
        merge_states(old, new)
    }

    fn apply_call(&mut self, ret: VirtualAddress) {
        let ctx = self.ctx;
        let esp = ctx.register_ref(4);
        self.move_to(
            &DestOperand::from_oper(esp),
            ctx.sub_const(esp, 4),
        );
        self.move_to(
            &DestOperand::from_oper(ctx.mem32(esp)),
            ctx.constant(ret.0 as u64),
        );
    }

    fn initial_state(
        ctx: &'e OperandContext,
        binary: &'e BinaryFile<VirtualAddress>,
    ) -> ExecutionState<'e> {
        let mut state = ExecutionState::with_binary(binary, ctx);

        // Set the return address to somewhere in 0x400000 range
        let return_address = ctx.mem32(ctx.register_ref(4));
        state.move_to(
            &DestOperand::from_oper(return_address),
            ctx.constant((binary.code_section().virtual_address.0 + 0x4230) as u64),
        );

        // Set the bytes above return address to 'call eax' to make it look like a legitmate call.
        state.move_to(
            &DestOperand::Memory(MemAccess {
                size: MemAccessSize::Mem16,
                address: ctx.sub_const(
                    return_address,
                    2,
                ),
            }),
            ctx.constant(0xd0ff),
        );
        state
    }

    fn find_functions_with_callers(file: &crate::BinaryFile<Self::VirtualAddress>)
        -> Vec<analysis::FuncCallPair<Self::VirtualAddress>>
    {
        crate::analysis::find_functions_with_callers_x86(file)
    }

    fn find_functions_from_calls(
        code: &[u8],
        section_base: Self::VirtualAddress,
        out: &mut Vec<Self::VirtualAddress>
    ) {
        crate::analysis::find_functions_from_calls_x86(code, section_base, out)
    }

    fn find_relocs(
        file: &BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<Self::VirtualAddress>, crate::OutOfBounds> {
        crate::analysis::find_relocs_x86(file)
    }

    fn value_limits(&self, value: Operand<'e>) -> (u64, u64) {
        if let Some(ref constraint) = self.resolved_constraint {
            crate::exec_state::value_limits_recurse(constraint.0, value)
        } else {
            (0, u64::max_value())
        }
    }
}

const FPU_REGISTER_INDEX: usize = 8;
const XMM_REGISTER_INDEX: usize = FPU_REGISTER_INDEX + 8;
const FLAGS_INDEX: usize = XMM_REGISTER_INDEX + 8 * 4;
const STATE_OPERANDS: usize = FLAGS_INDEX + 6;

#[derive(Clone)]
pub struct ExecutionState<'e> {
    // 8 registers, 8 fpu registers, 8 xmm registers with 4 parts each, 6 flags
    state: [Operand<'e>; 0x8 + 0x8 * 4 + 0x8 + 0x6],
    cached_low_registers: CachedLowRegisters<'e>,
    memory: Memory<'e>,
    resolved_constraint: Option<Constraint<'e>>,
    unresolved_constraint: Option<Constraint<'e>>,
    ctx: &'e OperandContext,
    binary: Option<&'e BinaryFile<VirtualAddress>>,
    pending_flags: Option<(ArithOperand<'e>, MemAccessSize)>,
}

impl<'e> fmt::Debug for ExecutionState<'e> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExecutionStateX86")
    }
}

/// Caches ax/al/ah resolving.
#[derive(Debug, Clone)]
struct CachedLowRegisters<'e> {
    registers: [[Option<Operand<'e>>; 2]; 8],
}

impl<'e> CachedLowRegisters<'e> {
    fn new() -> CachedLowRegisters<'e> {
        CachedLowRegisters {
            registers: [[None; 2]; 8],
        }
    }

    fn get_16(&self, register: u8) -> Option<Operand<'e>> {
        self.registers[register as usize][0]
    }

    fn get_low8(&self, register: u8) -> Option<Operand<'e>> {
        self.registers[register as usize][1]
    }

    fn set_16(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][0] = Some(value);
    }

    fn set_low8(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][1] = Some(value);
    }

    fn invalidate(&mut self, register: u8) {
        self.registers[register as usize] = [None; 2];
    }
}

/// Handles regular &'mut Operand assign for regs,
/// and the more complicated one for memory
enum Destination<'a, 'e> {
    Oper(&'a mut Operand<'e>),
    Register16(&'a mut Operand<'e>),
    Register8High(&'a mut Operand<'e>),
    Register8Low(&'a mut Operand<'e>),
    Memory(&'a mut ExecutionState<'e>, Operand<'e>, MemAccessSize),
}

/// Removes 0xffff_ffff masks from registers/undefined,
/// converts x + u64_const to x + u32_const.
///
/// The main idea is that it is assumed that arithmetic add/sub/mul won't overflow
/// when stored to operands. If the arithmetic result is used as a part of another
/// operation whose meaning changes based on high bits (div/mod/rsh/eq/gt), then
/// the result is explicitly masked. (Add / sub constants larger than u32::max will
/// be still truncated eagerly)
fn as_32bit_value<'e>(value: Operand<'e>, ctx: &'e OperandContext) -> Operand<'e> {
    if value.relevant_bits().end > 32 {
        if let Some((l, r)) = value.if_arithmetic_add() {
            // Uh, relying that simplification places constant on the right
            // always. Should be made a explicit guarantee.
            if let Some(c) = r.if_constant() {
                if c > u32::max_value() as u64 {
                    let truncated = c as u32 as u64;
                    return ctx.add_const(l, truncated);
                }
            }
        }
        if let Some(c) = value.if_constant() {
            if c > u32::max_value() as u64 {
                let truncated = c as u32 as u64;
                return ctx.constant(truncated);
            }
        }
        value.clone()
    } else {
        let const_mask = value.if_arithmetic_and()
            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()));
        if let Some((0xffff_ffff, other)) = const_mask {
            if other.is_undefined() || other.if_register().is_some() {
                other.clone()
            } else {
                value.clone()
            }
        } else {
            value.clone()
        }
    }
}

fn sext32_64(val: u32) -> u64 {
    val as i32 as i64 as u64
}

impl<'a, 'e> Destination<'a, 'e> {
    fn set(self, value: Operand<'e>, ctx: &'e OperandContext) {
        let value = as_32bit_value(value, ctx);
        match self {
            Destination::Oper(o) => {
                *o = value;
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
                        ctx.and_const(old, 0xffff_0000),
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
                        ctx.and_const(old, 0xffff_00ff),
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
                        ctx.and_const(old, 0xffff_ff00),
                        masked,
                    )
                };
                *o = new;
            }
            Destination::Memory(state, addr, size) => {
                if size == MemAccessSize::Mem64 {
                    // Split into two u32 sets
                    Destination::Memory(state, addr.clone(), MemAccessSize::Mem32)
                        .set(ctx.and_const(value, 0xffff_ffff), ctx);
                    let addr = ctx.add_const(addr, 4);
                    Destination::Memory(state, addr, MemAccessSize::Mem32)
                        .set(ctx.rsh_const(value, 32), ctx);
                    return;
                }
                if let Some((base, offset)) = Operand::const_offset(addr, ctx) {
                    let offset = offset as u32;
                    let offset_4 = offset & 3;
                    let offset_rest = sext32_64(offset & !3);
                    if offset_4 != 0 {
                        let size_bits = size.bits();
                        let low_base = ctx.add_const(base, offset_rest);
                        let low_old = state.resolve_mem(
                            &MemAccess {
                                address: low_base.clone(),
                                size: MemAccessSize::Mem32,
                            },
                        ).unwrap_or_else(|| ctx.mem32(low_base));

                        let mask_low = offset_4 * 8;
                        let mask_high = (mask_low + size_bits).min(0x20);
                        let mask = !0u32 >> mask_low << mask_low <<
                            (0x20 - mask_high) >> (0x20 - mask_high);
                        let low_value = ctx.or(
                            ctx.and_const(
                                ctx.lsh_const(
                                    value,
                                    8 * offset_4 as u64,
                                ),
                                mask as u64,
                            ),
                            ctx.and_const(
                                low_old,
                                !mask as u64,
                            ),
                        );
                        state.memory.set(low_base, low_value);
                        let needs_high = mask_low + size_bits > 0x20;
                        if needs_high {
                            let high_base = ctx.add_const(
                                base,
                                offset_rest.wrapping_add(4) & 0xffff_ffff,
                            );
                            let high_old = state.resolve_mem(
                                &MemAccess {
                                    address: high_base.clone(),
                                    size: MemAccessSize::Mem32,
                                },
                            ).unwrap_or_else(|| ctx.mem32(high_base));

                            let mask = !0u32 >> (0x20 - (mask_low + size_bits - 0x20));
                            let high_value = ctx.or(
                                ctx.and_const(
                                    ctx.rsh_const(
                                        value,
                                        (0x20 - 8 * offset_4) as u64,
                                    ),
                                    mask as u64,
                                ),
                                ctx.and_const(
                                    high_old,
                                    !mask as u64,
                                ),
                            );
                            state.memory.set(high_base, high_value);
                        }
                        return;
                    }
                }
                let value = if size == MemAccessSize::Mem32 {
                    value
                } else {
                    let old = state.resolve_mem(
                        &MemAccess {
                            address: addr.clone(),
                            size: MemAccessSize::Mem32,
                        },
                    ).unwrap_or_else(|| ctx.mem32(addr));
                    let new_mask = match size {
                        MemAccessSize::Mem8 => 0xff,
                        MemAccessSize::Mem16 => 0xffff,
                        _ => unreachable!(),
                    };
                    ctx.or(
                        ctx.and_const(value, new_mask),
                        ctx.and_const(old, !new_mask & 0xffff_ffff),
                    )
                };
                state.memory.set(addr, value);
            }
        }
    }
}

impl<'e> ExecutionState<'e> {
    pub fn new<'b>(
        ctx: &'b OperandContext,
    ) -> ExecutionState<'b> {
        let dummy = ctx.const_0();
        let mut state = [dummy; STATE_OPERANDS];
        for i in 0..8 {
            state[i] = ctx.register(i as u8);
            state[FPU_REGISTER_INDEX + i] = ctx.register_fpu(i as u8);
            for j in 0..4 {
                state[XMM_REGISTER_INDEX + i * 4 + j] = ctx.xmm(i as u8, j as u8);
            }
        }
        for i in 0..5 {
            state[FLAGS_INDEX + i] = ctx.flag_by_index(i).clone();
        }
        // Direction
        state[FLAGS_INDEX + 5] = ctx.const_0();
        ExecutionState {
            state,
            cached_low_registers: CachedLowRegisters::new(),
            memory: Memory::new(),
            resolved_constraint: None,
            unresolved_constraint: None,
            ctx,
            binary: None,
            pending_flags: None,
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile<VirtualAddress>,
        ctx: &'b OperandContext,
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
                static UNDEF_REGISTERS: &[u8] = &[0, 1, 2, 4];
                for &i in UNDEF_REGISTERS.iter() {
                    self.state[i as usize] = ctx.new_undef();
                    self.cached_low_registers.invalidate(i);
                }
                for i in 0..5 {
                    self.state[FLAGS_INDEX + i] = ctx.new_undef();
                }
                self.state[FLAGS_INDEX + Flag::Direction as usize] = ctx.const_0();
            }
            Operation::Jump { .. } => {
            }
            Operation::Return(_) => {
            }
            Operation::Special(ref code) => {
                if code == &[0xd9, 0xf6] {
                    // fdecstp
                    (&mut self.state[FPU_REGISTER_INDEX..][..8]).rotate_left(1);
                } else if code == &[0xd9, 0xf7] {
                    // fincstp
                    (&mut self.state[FPU_REGISTER_INDEX..][..8]).rotate_right(1);
                }
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
        }
    }

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
        let resolved_left = arith.left;
        let result = ctx.arithmetic(arith.ty, arith.left, arith.right);
        match arith.ty {
            Add => {
                let carry = ctx.gt(resolved_left, result);
                let overflow = ctx.gt_signed(resolved_left, result, size);
                self.state[FLAGS_INDEX + Flag::Carry as usize] = carry;
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = overflow;
                self.result_flags(result, size);
            }
            Sub => {
                let carry = ctx.gt(result, resolved_left);
                let overflow = ctx.gt_signed(result, resolved_left, size);
                self.state[FLAGS_INDEX + Flag::Carry as usize] = carry;
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = overflow;
                self.result_flags(result, size);
            }
            Xor | And | Or => {
                let zero = self.ctx.const_0();
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
        let parity = ctx.arithmetic(ArithOpType::Parity, result, ctx.const_0());
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
        if let Some(ref mut s) = self.resolved_constraint {
            if let DestOperand::Memory(_) = dest {
                if s.invalidate_memory(ctx) == crate::exec_state::ConstraintFullyInvalid::Yes {
                    self.resolved_constraint = None;
                }
            }
        }
        self.get_dest(dest, false)
    }

    fn get_dest<'s>(
        &'s mut self,
        dest: &DestOperand<'e>,
        dest_is_resolved: bool,
    ) -> Destination<'s, 'e> {
        match *dest {
            DestOperand::Register32(reg) | DestOperand::Register64(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Oper(&mut self.state[reg.0 as usize & 7])
            }
            DestOperand::Register16(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register16(&mut self.state[reg.0 as usize & 7])
            }
            DestOperand::Register8High(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register8High(&mut self.state[reg.0 as usize & 7])
            }
            DestOperand::Register8Low(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register8Low(&mut self.state[reg.0 as usize & 7])
            }
            DestOperand::Fpu(id) => {
                Destination::Oper(&mut self.state[FPU_REGISTER_INDEX + (id & 7) as usize])
            }
            DestOperand::Xmm(reg, word) => {
                Destination::Oper(&mut self.state[
                    XMM_REGISTER_INDEX + (reg & 7) as usize * 4 + (word & 3) as usize
                ])
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
    ///
    /// Empirical tests seem to imply that happens about 15~20% of time
    fn resolve_mem(&mut self, mem: &MemAccess<'e>) -> Option<Operand<'e>> {
        let ctx = self.ctx;
        let address = self.resolve(mem.address);
        if MemAccessSize::Mem64 == mem.size {
            // Split into 2 32-bit resolves
            let low_mem = MemAccess {
                address: mem.address,
                size: MemAccessSize::Mem32,
            };
            let high_mem = MemAccess {
                address: ctx.add_const(mem.address, 4),
                size: MemAccessSize::Mem32,
            };
            let low = self.resolve_mem(&low_mem);
            let high = self.resolve_mem(&high_mem);
            if low.is_none() && high.is_none() {
                return None;
            }
            let low = low
                .unwrap_or_else(|| ctx.mem32(low_mem.address));
            let high = high
                .unwrap_or_else(|| ctx.mem32(high_mem.address));
            return Some(ctx.or(low, high));
        }

        let mask = match mem.size {
            MemAccessSize::Mem8 => 0xffu32,
            MemAccessSize::Mem16 => 0xffff,
            MemAccessSize::Mem32 => 0,
            MemAccessSize::Mem64 => 0,
        };
        // Use 4-aligned addresses if there's a const offset
        if let Some((base, offset)) = Operand::const_offset(address, ctx) {
            let size_bytes = mem.size.bits() / 8;
            let offset = offset as u32;
            let offset_4 = offset & 3;
            let offset_rest = sext32_64(offset & !3);
            if offset_4 != 0 {
                let low_base = ctx.add_const(base, offset_rest);
                let low = self.memory.get(low_base)
                    .or_else(|| {
                        // Avoid reading Mem32 if it's not necessary as it may go
                        // past binary end where a smaller read wouldn't
                        let size = match offset_4 + size_bytes {
                            1 => MemAccessSize::Mem8,
                            2 => MemAccessSize::Mem16,
                            _ => MemAccessSize::Mem32,
                        };
                        self.resolve_binary_constant_mem(low_base, size)
                    })
                    .unwrap_or_else(|| ctx.mem32(low_base));
                let low = ctx.rsh_const(low, (offset_4 * 8) as u64);
                let combined = if offset_4 + size_bytes > 4 {
                    let high_base = ctx.add_const(
                        base,
                        offset_rest.wrapping_add(4) & 0xffff_ffff,
                    );
                    let high = self.memory.get(high_base)
                        .or_else(|| {
                            let size = match (offset_4 + size_bytes) - 4 {
                                1 => MemAccessSize::Mem8,
                                2 => MemAccessSize::Mem16,
                                _ => MemAccessSize::Mem32,
                            };
                            self.resolve_binary_constant_mem(high_base, size)
                        })
                        .unwrap_or_else(|| ctx.mem32(high_base));
                    let high = ctx.and_const(
                        ctx.lsh_const(
                            high,
                            (0x20 - offset_4 * 8) as u64,
                        ),
                        0xffff_ffff,
                    );
                    ctx.or(low, high)
                } else {
                    low
                };
                let masked = if mem.size != MemAccessSize::Mem32 {
                    ctx.and_const(combined, mask as u64)
                } else {
                    combined
                };
                return Some(masked);
            }
        }
        self.memory.get(address)
            .map(|operand| {
                if mask != 0 {
                    ctx.and_const(operand, mask as u64)
                } else {
                    operand
                }
            })
            .or_else(|| self.resolve_binary_constant_mem(address, mem.size))
            .or_else(|| {
                // Just copy the input value if address didn't change
                if address == mem.address {
                    None
                } else {
                    Some(ctx.mem_variable_rc(mem.size, address))
                }
            })
    }

    fn resolve_binary_constant_mem(
        &self,
        address: Operand<'e>,
        size: MemAccessSize,
    ) -> Option<Operand<'e>> {
        if let Some(c) = address.if_constant().map(|c| c as u32) {
            let size_bytes = size.bits() / 8;
            // Simplify constants stored in code section (constant switch jumps etc)
            if let Some(end) = c.checked_add(size_bytes) {
                let section = self.binary.and_then(|b| {
                    b.code_sections().find(|s| {
                        s.virtual_address.0 <= c && s.virtual_address.0 + s.virtual_size >= end
                    })
                });
                if let Some(section) = section {
                    let offset = (c - section.virtual_address.0) as usize;
                    let val = match size {
                        MemAccessSize::Mem8 => section.data[offset] as u32,
                        MemAccessSize::Mem16 => {
                            (&section.data[offset..]).read_u16().unwrap_or(0) as u32
                        }
                        MemAccessSize::Mem32 => {
                            (&section.data[offset..]).read_u32().unwrap_or(0)
                        }
                        MemAccessSize::Mem64 => unreachable!(),
                    };
                    return Some(self.ctx.constant(val as u64));
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
        let reg = other.if_register()?.0 & 0x7;
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
                        0xffff
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
        } else {
            None
        }
    }

    pub fn resolve(&mut self, value: Operand<'e>) -> Operand<'e> {
        match *value.ty() {
            OperandType::Register(reg) => {
                self.state[reg.0 as usize & 7]
            }
            OperandType::Xmm(reg, word) => {
                self.state[XMM_REGISTER_INDEX + (reg & 7) as usize * 4 + (word & 3) as usize]
            }
            OperandType::Fpu(id) => {
                self.state[FPU_REGISTER_INDEX + (id & 7) as usize]
            }
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
            OperandType::ArithmeticF32(ref op) => {
                let left = self.resolve(op.left);
                let right = self.resolve(op.right);
                self.ctx.f32_arithmetic(op.ty, left, right)
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
        for (reg, &val2) in self.state.iter().enumerate().take(8) {
            if val == val2 {
                return Some(self.ctx.register(reg as u8));
            }
        }
        None
    }

    /// Tries to find an memory address corresponding to a resolved value.
    pub fn unresolve_memory(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.memory.reverse_lookup(val).map(|x| self.ctx.mem32(x))
    }

    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    pub fn replace_memory(&mut self, new: Memory<'e>) {
        self.memory = new;
    }
}

/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
pub fn merge_states<'a: 'r, 'r>(
    old: &'r mut ExecutionState<'a>,
    new: &'r mut ExecutionState<'a>,
) -> Option<ExecutionState<'a>> {
    old.update_flags();
    new.update_flags();

    fn check_eq<'e>(a: Operand<'e>, b: Operand<'e>) -> bool {
        a == b || a.is_undefined()
    }
    fn check_memory_eq<'e>(a: &Memory<'e>, b: &Memory<'e>) -> bool {
        a.map.iter().all(|(&key, val)| {
            match contains_undefined(key.0) {
                true => true,
                false => match b.get(key.0) {
                    Some(b_val) => check_eq(*val, b_val),
                    None => true,
                },
            }
        })
    }

    let ctx = old.ctx;
    fn merge<'e>(ctx: &'e OperandContext, a: Operand<'e>, b: Operand<'e>) -> Operand<'e> {
        match a == b {
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
    let changed =
        old.state.iter().zip(new.state.iter())
            .any(|(&a, &b)| !check_eq(a, b)) ||
        !check_memory_eq(&old.memory, &new.memory) ||
        merged_ljec.as_ref().map(|x| *x != old.unresolved_constraint).unwrap_or(false) || (
            old.resolved_constraint.is_some() &&
            old.resolved_constraint != new.resolved_constraint
        );
    if changed {
        let state = array_init::array_init(|i| merge(ctx, old.state[i], new.state[i]));
        let mut cached_low_registers = CachedLowRegisters::new();
        for i in 0..8 {
            let old_reg = &old.cached_low_registers.registers[i];
            let new_reg = &new.cached_low_registers.registers[i];
            for j in 0..old_reg.len() {
                if old_reg[j] == new_reg[j] {
                    // Doesn't merge things but sets them uncached if they differ
                    cached_low_registers.registers[i][j] = old_reg[j];
                }
            }
        }
        Some(ExecutionState {
            state,
            cached_low_registers,
            memory: old.memory.merge(&new.memory, ctx),
            unresolved_constraint: merged_ljec.unwrap_or_else(|| {
                // They were same, just use one from old
                old.unresolved_constraint.clone()
            }),
            resolved_constraint: if old.resolved_constraint == new.resolved_constraint {
                old.resolved_constraint.clone()
            } else {
                None
            },
            pending_flags: None,
            ctx,
            binary: old.binary,
        })
    } else {
        None
    }
}

fn contains_undefined(oper: Operand<'_>) -> bool {
    oper.iter().any(|x| x.is_undefined())
}

#[test]
fn merge_state_constraints_eq() {
    let ctx = &crate::operand::OperandContext::new();
    let state_a = ExecutionState::new(ctx);
    let mut state_b = ExecutionState::new(ctx);
    let sign_eq_overflow_flag = ctx.eq(
        ctx.flag_o(),
        ctx.flag_s(),
    );
    let mut state_a = state_a.assume_jump_flag(sign_eq_overflow_flag, true);
    state_b.move_to(&DestOperand::from_oper(ctx.flag_o()), ctx.constant(1));
    state_b.move_to(&DestOperand::from_oper(ctx.flag_s()), ctx.constant(1));
    let merged = merge_states(&mut state_b, &mut state_a).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);
}

#[test]
fn merge_state_constraints_or() {
    let ctx = &crate::operand::OperandContext::new();
    let state_a = ExecutionState::new(ctx);
    let mut state_b = ExecutionState::new(ctx);
    let sign_or_overflow_flag = ctx.or(
        ctx.flag_o(),
        ctx.flag_s(),
    );
    let mut state_a = state_a.assume_jump_flag(sign_or_overflow_flag, true);
    state_b.move_to(&DestOperand::from_oper(ctx.flag_s()), ctx.constant(1));
    let merged = merge_states(&mut state_b, &mut state_a).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);
    // Should also happen other way, though then state_a must have something that is converted
    // to undef.
    let merged = merge_states(&mut state_a, &mut state_b).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);

    state_a.move_to(&DestOperand::from_oper(ctx.flag_c()), ctx.constant(1));
    let merged = merge_states(&mut state_a, &mut state_b).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);
}

#[test]
fn test_as_32bit_value() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.add(
        ctx.register(0),
        ctx.constant(0x1_0000_0000),
    );
    let expected = ctx.register(0);
    assert_eq!(as_32bit_value(op, ctx), expected);

    let op = ctx.and(
        ctx.register(0),
        ctx.constant(0xffff_ffff),
    );
    let expected = ctx.register(0);
    assert_eq!(as_32bit_value(op, ctx), expected);

    let op = ctx.constant(0x1_0000_0000);
    let expected = ctx.constant(0);
    assert_eq!(as_32bit_value(op, ctx), expected);
}
