use std::fmt;
use std::rc::Rc;

use crate::analysis;
use crate::disasm::{Disassembler64, DestOperand, Operation};
use crate::exec_state::{Constraint, InternMap, InternedOperand, Memory};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::light_byteorder::ReadLittleEndian;
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
    ArithOpType,
};
use crate::{BinaryFile, VirtualAddress64};

const XMM_REGISTER_INDEX: usize = 0x10;
const FLAGS_INDEX: usize = XMM_REGISTER_INDEX + 0x10 * 4;
const STATE_OPERANDS: usize = FLAGS_INDEX + 6;

#[derive(Clone)]
pub struct ExecutionState<'a> {
    // 16 registers, 10 xmm registers with 4 parts each, 6 flags
    state: [InternedOperand; 0x10 + 0x10 * 4 + 0x6],
    cached_low_registers: CachedLowRegisters,
    memory: Memory,
    unresolved_constraint: Option<Constraint>,
    resolved_constraint: Option<Constraint>,
    ctx: &'a OperandContext,
    binary: Option<&'a BinaryFile<VirtualAddress64>>,
    /// Lazily update flags since a lot of instructions set them and
    /// they get discarded later.
    pending_flags: Option<(ArithOperand, MemAccessSize)>,
}

impl<'e> fmt::Debug for ExecutionState<'e> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExecutionStateX86_64")
    }
}

/// Caches eax/ax/al/ah resolving.
/// InternedOperand(0) means that a register is not cached.
#[derive(Debug, Clone)]
struct CachedLowRegisters {
    registers: [[InternedOperand; 3]; 0x10],
}

impl CachedLowRegisters {
    fn new() -> CachedLowRegisters {
        CachedLowRegisters {
            registers: [[InternedOperand(0); 3]; 0x10],
        }
    }

    fn get_32(&self, register: u8) -> InternedOperand {
        self.registers[register as usize][2]
    }

    fn get_16(&self, register: u8) -> InternedOperand {
        self.registers[register as usize][0]
    }

    fn get_low8(&self, register: u8) -> InternedOperand {
        self.registers[register as usize][1]
    }

    fn set_32(&mut self, register: u8, value: InternedOperand) {
        self.registers[register as usize][2] = value;
    }

    fn set_16(&mut self, register: u8, value: InternedOperand) {
        self.registers[register as usize][0] = value;
    }

    fn set_low8(&mut self, register: u8, value: InternedOperand) {
        self.registers[register as usize][1] = value;
    }

    fn invalidate(&mut self, register: u8) {
        self.registers[register as usize] = [InternedOperand(0); 3];
    }
}

/// Handles regular &'mut Operand assign for regs,
/// and the more complicated one for memory
enum Destination<'a, 'e> {
    Oper(&'a mut InternedOperand),
    Register32(&'a mut InternedOperand),
    Register16(&'a mut InternedOperand),
    Register8High(&'a mut InternedOperand),
    Register8Low(&'a mut InternedOperand),
    Memory(&'a mut ExecutionState<'e>, Rc<Operand>, MemAccessSize),
    Nop,
}

impl<'a, 'e> Destination<'a, 'e> {
    fn set(self, value: Rc<Operand>, intern_map: &mut InternMap, ctx: &OperandContext) {
        match self {
            Destination::Oper(o) => {
                *o = intern_map.intern(value);
            }
            Destination::Register32(o) => {
                // 32-bit register dest clears high bits (16- or 8-bit dests don't)
                let masked = if value.relevant_bits().end > 32 {
                    ctx.and_const(&value, 0xffff_ffff)
                } else {
                    value
                };
                *o = intern_map.intern(masked);
            }
            Destination::Register16(o) => {
                let old = intern_map.operand(*o);
                let masked = if value.relevant_bits().end > 16 {
                    ctx.and_const(&value, 0xffff)
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.end <= 16 {
                    masked
                } else {
                    ctx.or(
                        &ctx.and_const(&old, 0xffff_ffff_ffff_0000),
                        &masked,
                    )
                };
                *o = intern_map.intern(new);
            }
            Destination::Register8High(o) => {
                let old = intern_map.operand(*o);
                let masked = if value.relevant_bits().end > 8 {
                    ctx.and_const(&value, 0xff)
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.start >= 8 && old_bits.end <= 16 {
                    ctx.lsh_const(&masked, 8)
                } else {
                    ctx.or(
                        &ctx.and_const(&old, 0xffff_ffff_ffff_00ff),
                        &ctx.lsh_const(&masked, 8),
                    )
                };
                *o = intern_map.intern(new);
            }
            Destination::Register8Low(o) => {
                let old = intern_map.operand(*o);
                let masked = if value.relevant_bits().end > 8 {
                    ctx.and_const(&value, 0xff)
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.end <= 8 {
                    masked
                } else {
                    ctx.or(
                        &ctx.and_const(&old, 0xffff_ffff_ffff_ff00),
                        &masked,
                    )
                };
                *o = intern_map.intern(new);
            }
            Destination::Memory(state, addr, size) => {
                if let Some((base, offset)) = Operand::const_offset(&addr, ctx) {
                    let offset_8 = offset & 7;
                    let offset_rest = offset & !7;
                    if offset_8 != 0 {
                        let size_bits = size.bits() as u64;
                        let low_base = ctx.add_const(&base, offset_rest);
                        let low_i = intern_map.intern(low_base.clone());
                        let low_old = state.resolve_mem(
                            &MemAccess {
                                address: low_base.clone(),
                                size: MemAccessSize::Mem64,
                            },
                            intern_map,
                        ).unwrap_or_else(|| ctx.mem64(&low_base));

                        let mask_low = offset_8 * 8;
                        let mask_high = (mask_low + size_bits).min(0x40);
                        let mask = !0u64 >> mask_low << mask_low <<
                            (0x40 - mask_high) >> (0x40 - mask_high);
                        let low_value = ctx.or(
                            &ctx.and_const(
                                &ctx.lsh_const(
                                    &value,
                                    8 * offset_8,
                                ),
                                mask
                            ),
                            &ctx.and_const(
                                &low_old,
                                !mask,
                            ),
                        );
                        state.memory.set(low_i, intern_map.intern(low_value));
                        let needs_high = mask_low + size_bits > 0x40;
                        if needs_high {
                            let high_base = ctx.add_const(
                                &base,
                                offset_rest.wrapping_add(8),
                            );
                            let high_i = intern_map.intern(high_base.clone());
                            let high_old = state.resolve_mem(
                                &MemAccess {
                                    address: high_base.clone(),
                                    size: MemAccessSize::Mem64,
                                },
                                intern_map,
                            ).unwrap_or_else(|| ctx.mem64(&high_base));
                            let mask = !0u64 >> (0x40 - (mask_low + size_bits - 0x40));
                            let high_value = ctx.or(
                                &ctx.and_const(
                                    &ctx.rsh_const(
                                        &value,
                                        0x40 - 8 * offset_8,
                                    ),
                                    mask,
                                ),
                                &ctx.and_const(
                                    &high_old,
                                    !mask,
                                ),
                            );
                            state.memory.set(high_i, intern_map.intern(high_value));
                        }
                        return;
                    }
                }
                let value = match size {
                    MemAccessSize::Mem64 => value,
                    _ => {
                        let old = state.resolve_mem(
                            &MemAccess {
                                address: addr.clone(),
                                size: MemAccessSize::Mem64,
                            },
                            intern_map,
                        ).unwrap_or_else(|| ctx.mem64(&addr));
                        let new_mask = match size {
                            MemAccessSize::Mem8 => 0xff,
                            MemAccessSize::Mem16 => 0xffff,
                            MemAccessSize::Mem32 | _ => 0xffff_ffff,
                        };
                        ctx.or(
                            &ctx.and_const(&value, new_mask),
                            &ctx.and_const(&old, !new_mask),
                        )
                    }
                };
                let addr = intern_map.intern(addr);
                state.memory.set(addr, intern_map.intern(value));
            }
            Destination::Nop => (),
        }
    }
}

impl<'a> ExecutionStateTrait<'a> for ExecutionState<'a> {
    type VirtualAddress = VirtualAddress64;
    type Disassembler = Disassembler64<'a>;

    fn maybe_convert_memory_immutable(&mut self) {
        self.memory.maybe_convert_immutable();
    }

    fn add_resolved_constraint(&mut self, constraint: Constraint) {
        self.resolved_constraint = Some(constraint);
    }

    fn add_unresolved_constraint(&mut self, constraint: Constraint) {
        self.unresolved_constraint = Some(constraint);
    }

    fn move_to(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap) {
        let ctx = self.ctx();
        let resolved = self.resolve(&value, i);
        let dest = self.get_dest_invalidate_constraints(dest, i);
        dest.set(resolved, i, ctx);
    }

    fn move_resolved(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap) {
        let ctx = self.ctx;
        self.unresolved_constraint = None;
        let dest = self.get_dest(dest, i, true);
        dest.set(value, i, ctx);
    }

    fn ctx(&self) -> &'a OperandContext {
        self.ctx
    }

    fn resolve(&mut self, operand: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand> {
        self.resolve(operand, i)
    }

    fn update(&mut self, operation: &Operation, intern_map: &mut InternMap) {
        self.update(operation, intern_map)
    }

    fn resolve_apply_constraints(&mut self, op: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand> {
        let stack_op;
        let mut op_ref = op;
        if let Some(ref constraint) = self.unresolved_constraint {
            stack_op = constraint.apply_to(self.ctx, op_ref);
            op_ref = &stack_op;
        }
        let val = self.resolve(op_ref, i);
        if let Some(ref constraint) = self.resolved_constraint {
            constraint.apply_to(self.ctx, &val)
        } else {
            val
        }
    }

    fn unresolve(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        self.unresolve(val, i)
    }

    fn unresolve_memory(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        self.unresolve_memory(val, i)
    }

    fn merge_states(old: &mut Self, new: &mut Self, i: &mut InternMap) -> Option<Self> {
        merge_states(old, new, i)
    }

    fn apply_call(&mut self, ret: VirtualAddress64, i: &mut InternMap) {
        let ctx = self.ctx;
        let rsp = ctx.register_ref(4);
        self.move_to(
            &DestOperand::from_oper(rsp),
            ctx.sub_const(&rsp, 8),
            i,
        );
        self.move_to(
            &DestOperand::from_oper(&ctx.mem64(rsp)),
            ctx.constant(ret.0),
            i,
        );
    }

    fn initial_state(
        ctx: &'a OperandContext,
        binary: &'a BinaryFile<VirtualAddress64>,
        interner: &mut InternMap,
    ) -> ExecutionState<'a> {
        ExecutionState::with_binary(binary, ctx, interner)
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

    fn value_limits(&self, value: &Rc<Operand>) -> (u64, u64) {
        if let Some(ref constraint) = self.resolved_constraint {
            crate::exec_state::value_limits_recurse(&constraint.0, value)
        } else {
            (0, u64::max_value())
        }
    }
}

impl<'e> ExecutionState<'e> {
    pub fn new<'b>(
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        let mut state = [InternedOperand(0); STATE_OPERANDS];
        for i in 0..16 {
            state[i] = interner.intern(ctx.register(i as u8));
            for j in 0..4 {
                state[XMM_REGISTER_INDEX + i * 4 + j] =
                    interner.intern(ctx.xmm(i as u8, j as u8));
            }
        }
        for i in 0..5 {
            state[FLAGS_INDEX + i] = interner.intern(ctx.flag_by_index(i).clone());
        }
        // Direction
        state[FLAGS_INDEX + 5] = interner.intern(ctx.const_0());
        ExecutionState {
            state,
            cached_low_registers: CachedLowRegisters::new(),
            memory: Memory::new(),
            unresolved_constraint: None,
            resolved_constraint: None,
            ctx,
            binary: None,
            pending_flags: None,
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile<VirtualAddress64>,
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        let mut result = ExecutionState::new(ctx, interner);
        result.binary = Some(binary);
        result
    }

    pub fn update(&mut self, operation: &Operation, intern_map: &mut InternMap) {
        let ctx = self.ctx;
        match operation {
            Operation::Move(dest, value, cond) => {
                let value = value.clone();
                if let Some(cond) = cond {
                    match self.resolve(cond, intern_map).if_constant() {
                        Some(0) => (),
                        Some(_) => {
                            self.move_to(&dest, value, intern_map);
                        }
                        None => {
                            self.get_dest_invalidate_constraints(&dest, intern_map)
                                .set(ctx.undefined_rc(), intern_map, ctx)
                        }
                    }
                } else {
                    self.move_to(&dest, value, intern_map);
                }
            }
            Operation::MoveSet(moves) => {
                let resolved: Vec<Rc<Operand>> = moves.iter()
                    .map(|x| self.resolve(&x.1, intern_map))
                    .collect();
                for (tp, val) in moves.iter().zip(resolved.into_iter()) {
                    self.get_dest_invalidate_constraints(&tp.0, intern_map)
                        .set(val, intern_map, ctx);
                }
            }
            Operation::Call(_) => {
                let mut ids = intern_map.many_undef(ctx, 13);
                self.unresolved_constraint = None;
                if let Some(ref mut c) = self.resolved_constraint {
                    if c.invalidate_memory(ctx) == crate::exec_state::ConstraintFullyInvalid::Yes {
                        self.resolved_constraint = None
                    }
                }
                static UNDEF_REGISTERS: &[u8] = &[0, 1, 2, 4, 8, 9, 10, 11];
                for &i in UNDEF_REGISTERS.iter() {
                    self.state[i as usize] = ids.next();
                    self.cached_low_registers.invalidate(i);
                }
                for i in 0..5 {
                    self.state[FLAGS_INDEX + i] = ids.next();
                }
                self.state[FLAGS_INDEX + Flag::Direction as usize] =
                    intern_map.intern(ctx.const_0());
            }
            Operation::Jump { .. } => {
            }
            Operation::Return(_) => {
            }
            Operation::Special(_) => {
            }
            Operation::SetFlags(arith, size) => {
                let left = self.resolve(&arith.left, intern_map);
                let right = self.resolve(&arith.right, intern_map);
                let arith = ArithOperand {
                    left,
                    right,
                    ty: arith.ty,
                };
                self.pending_flags = Some((arith, *size));
                // Could try to do smarter invalidation, but since in practice unresolved
                // constraints always are bunch of flags, invalidate it completely.
                self.unresolved_constraint = None;
            }
        }
    }

    /// Updates arithmetic flags if there are any pending flag updates.
    /// Must be called before accessing (both read or write) arithmetic flags.
    /// Obvously can just zero pending_flags if all flags are written over though.
    fn update_flags(&mut self, intern_map: &mut InternMap) {
        if let Some((arith, size)) = self.pending_flags.take() {
            self.set_flags(arith, size, intern_map);
        }
    }

    fn set_flags(
        &mut self,
        arith: ArithOperand,
        size: MemAccessSize,
        intern_map: &mut InternMap,
    ) {
        use crate::operand::ArithOpType::*;
        let resolved_left = &arith.left;

        let ctx = self.ctx;
        let result = ctx.arithmetic(arith.ty, &arith.left, &arith.right);
        match arith.ty {
            Add => {
                let overflow = ctx.gt_signed(resolved_left, &result, size);
                let carry = ctx.gt(resolved_left, &result);
                self.state[FLAGS_INDEX + Flag::Carry as usize] = intern_map.intern(carry);
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = intern_map.intern(overflow);
                self.result_flags(&result, size, intern_map);
            }
            Sub => {
                let overflow = ctx.gt_signed(resolved_left, &result, size);
                let carry = ctx.gt(&result, resolved_left);
                self.state[FLAGS_INDEX + Flag::Carry as usize] = intern_map.intern(carry);
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = intern_map.intern(overflow);
                self.result_flags(&result, size, intern_map);
            }
            Xor | And | Or => {
                let zero = intern_map.intern(ctx.const_0());
                self.state[FLAGS_INDEX + Flag::Carry as usize] = zero;
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = zero;
                self.result_flags(&result, size, intern_map);
            }
            Lsh | Rsh  => {
                let mut ids = intern_map.many_undef(ctx, 2);
                self.state[FLAGS_INDEX + Flag::Carry as usize] = ids.next();
                self.state[FLAGS_INDEX + Flag::Overflow as usize] = ids.next();
                self.result_flags(&result, size, intern_map);
            }
            _ => {
                let mut ids = intern_map.many_undef(ctx, 5);
                // NOTE: Relies on direction being index 5 so it won't change here
                for i in 0..5 {
                    self.state[FLAGS_INDEX + i] = ids.next();
                }
            }
        }
    }

    fn result_flags(
        &mut self,
        result: &Rc<Operand>,
        size: MemAccessSize,
        intern_map: &mut InternMap,
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
            &ctx.and_const(
                result,
                sign_bit,
            ),
            0,
        );
        // Parity is defined to be just lowest byte so it doesn't need special handling for
        // 64 bits.
        parity = ctx.arithmetic(ArithOpType::Parity, result, &ctx.const_0());
        self.state[FLAGS_INDEX + Flag::Zero as usize] = intern_map.intern(zero);
        self.state[FLAGS_INDEX + Flag::Sign as usize] = intern_map.intern(sign);
        self.state[FLAGS_INDEX + Flag::Parity as usize] = intern_map.intern(parity);
    }

    /// Makes all of memory undefined
    pub fn clear_memory(&mut self) {
        self.memory = Memory::new();
    }

    fn get_dest_invalidate_constraints<'s>(
        &'s mut self,
        dest: &DestOperand,
        interner: &mut InternMap,
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
        self.get_dest(dest, interner, false)
    }

    fn get_dest<'s>(
        &'s mut self,
        dest: &DestOperand,
        intern_map: &mut InternMap,
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
                Destination::Oper(&mut self.state[
                    XMM_REGISTER_INDEX + (reg & 0xf) as usize * 4 + (word & 3) as usize
                ])
            }
            DestOperand::Flag(flag) => {
                self.update_flags(intern_map);
                Destination::Oper(&mut self.state[FLAGS_INDEX + flag as usize])
            }
            DestOperand::Memory(ref mem) => {
                let address = if dest_is_resolved {
                    mem.address.clone()
                } else {
                    self.resolve(&mem.address, intern_map)
                };
                Destination::Memory(self, address, mem.size)
            }
        }
    }

    /// Returns None if the value won't change.
    fn resolve_mem(&mut self, mem: &MemAccess, i: &mut InternMap) -> Option<Rc<Operand>> {
        let ctx = self.ctx;
        let address = self.resolve(&mem.address, i);

        let mask = match mem.size {
            MemAccessSize::Mem8 => 0xffu32,
            MemAccessSize::Mem16 => 0xffff,
            MemAccessSize::Mem32 => 0xffff_ffff,
            MemAccessSize::Mem64 => 0,
        };
        // Use 8-aligned addresses if there's a const offset
        if let Some((base, offset)) = Operand::const_offset(&address, ctx) {
            let size_bytes = mem.size.bits() / 8;
            let offset_8 = offset as u32 & 7;
            let offset_rest = offset & !7;
            if offset_8 != 0 {
                let low_base = ctx.add_const(&base, offset_rest);
                let low = self.memory.get(i.intern(low_base.clone()))
                    .map(|x| i.operand(x))
                    .or_else(|| {
                        // Avoid reading Mem64 if it's not necessary as it may go
                        // past binary end where a smaller read wouldn't
                        let size = match offset_8 + size_bytes {
                            1 => MemAccessSize::Mem8,
                            2 => MemAccessSize::Mem16,
                            3 | 4 => MemAccessSize::Mem32,
                            _ => MemAccessSize::Mem64,
                        };
                        self.resolve_binary_constant_mem(&low_base, size)
                    })
                    .unwrap_or_else(|| ctx.mem64(&low_base));
                let low = ctx.rsh_const(&low, offset_8 as u64 * 8);
                let combined = if offset_8 + size_bytes > 8 {
                    let high_base = ctx.add_const(&base, offset_rest.wrapping_add(8));
                    let high = self.memory.get(i.intern(high_base.clone()))
                        .map(|x| i.operand(x))
                        .or_else(|| {
                            let size = match (offset_8 + size_bytes) - 8 {
                                1 => MemAccessSize::Mem8,
                                2 => MemAccessSize::Mem16,
                                3 | 4 => MemAccessSize::Mem32,
                                _ => MemAccessSize::Mem64,
                            };
                            self.resolve_binary_constant_mem(&high_base, size)
                        })
                        .unwrap_or_else(|| ctx.mem64(&high_base));
                    let high = ctx.lsh_const(&high, (0x40 - offset_8 * 8) as u64);
                    ctx.or(&low, &high)
                } else {
                    low
                };
                let masked = if mem.size != MemAccessSize::Mem64 {
                    ctx.and_const(&combined, mask as u64)
                } else {
                    combined
                };
                return Some(masked);
            }
        }
        self.memory.get(i.intern(address.clone()))
            .map(|interned| {
                let operand = i.operand(interned);
                if mem.size != MemAccessSize::Mem64 {
                    ctx.and_const(&operand, mask as u64)
                } else {
                    operand
                }
            })
            .or_else(|| self.resolve_binary_constant_mem(&address, mem.size))
            .or_else(|| {
                // Just copy the input value if address didn't change
                if Rc::ptr_eq(&address, &mem.address) {
                    None
                } else {
                    Some(ctx.mem_variable_rc(mem.size, &address))
                }
            })
    }

    fn resolve_binary_constant_mem(
        &self,
        address: &Rc<Operand>,
        size: MemAccessSize,
    ) -> Option<Rc<Operand>> {
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
        left: &Rc<Operand>,
        right: &Rc<Operand>,
        interner: &mut InternMap,
    ) -> Option<Rc<Operand>> {
        let ctx = self.ctx;
        let (const_op, c, other) = match left.ty {
            OperandType::Constant(c) => (left, c, right),
            _ => match right.ty {
                OperandType::Constant(c) => (right, c, left),
                _ => return None,
            }
        };
        let reg = other.if_register()?.0 & 0xf;
        if c <= 0xff {
            let interned = match self.cached_low_registers.get_low8(reg) {
                InternedOperand(0) => {
                    let op = ctx.and_const(
                        &interner.operand(self.state[reg as usize]),
                        0xff,
                    );
                    let interned = interner.intern(op);
                    self.cached_low_registers.set_low8(reg, interned);
                    interned
                }
                x => x,
            };
            let op = interner.operand(interned);
            if c == 0xff {
                Some(op)
            } else {
                Some(ctx.and(&op, const_op))
            }
        } else if c <= 0xffff {
            let interned = match self.cached_low_registers.get_16(reg) {
                InternedOperand(0) => {
                    let op = ctx.and_const(
                        &interner.operand(self.state[reg as usize]),
                        0xffff,
                    );
                    let interned = interner.intern(op);
                    self.cached_low_registers.set_16(reg, interned);
                    interned
                }
                x => x,
            };
            let op = interner.operand(interned);
            if c == 0xffff {
                Some(op)
            } else {
                Some(ctx.and(&op, const_op))
            }
        } else if c <= 0xffff_ffff {
            let interned = match self.cached_low_registers.get_32(reg) {
                InternedOperand(0) => {
                    let op = ctx.and_const(
                        &interner.operand(self.state[reg as usize]),
                        0xffff_ffff
                    );
                    let interned = interner.intern(op);
                    self.cached_low_registers.set_32(reg, interned);
                    interned
                }
                x => x,
            };
            let op = interner.operand(interned);
            if c == 0xffff_ffff {
                Some(op)
            } else {
                Some(ctx.and(&op, const_op))
            }
        } else {
            None
        }
    }

    fn resolve_inner(
        &mut self,
        value: &Rc<Operand>,
        interner: &mut InternMap,
    ) -> Rc<Operand> {
        match value.ty {
            OperandType::Register(reg) => {
                interner.operand(self.state[(reg.0 & 0xf) as usize])
            }
            OperandType::Xmm(reg, word) => {
                interner.operand(self.state[
                    XMM_REGISTER_INDEX + (reg & 0xf) as usize * 4 + (word & 3) as usize
                ])
            }
            OperandType::Fpu(_) => value.clone(),
            OperandType::Flag(flag) => {
                self.update_flags(interner);
                interner.operand(self.state[FLAGS_INDEX + flag as usize])
            }
            OperandType::Arithmetic(ref op) => {
                if op.ty == ArithOpType::And {
                    let r = self.try_resolve_partial_register(&op.left, &op.right, interner);
                    if let Some(r) = r {
                        return r;
                    }
                };
                let left = self.resolve(&op.left, interner);
                let right = self.resolve(&op.right, interner);
                self.ctx.arithmetic(op.ty, &left, &right)
            }
            OperandType::ArithmeticF32(ref op) => {
                let left = self.resolve(&op.left, interner);
                let right = self.resolve(&op.right, interner);
                self.ctx.f32_arithmetic(op.ty, &left, &right)
            }
            OperandType::Constant(_) => value.clone(),
            OperandType::Custom(_) => value.clone(),
            OperandType::Memory(ref mem) => {
                self.resolve_mem(mem, interner)
                    .unwrap_or_else(|| value.clone())
            }
            OperandType::Undefined(_) => value.clone(),
            OperandType::SignExtend(ref val, from, to) => {
                let val = self.resolve(val, interner);
                self.ctx.sign_extend(&val, from, to)
            }
        }
    }

    pub fn resolve(&mut self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
        let x = self.resolve_inner(value, interner);
        if x.is_simplified() {
            return x;
        }

        // Intern in case the simplification created a very deeply different operand tree,
        // as repeating the resolving would give an equal operand with different addresses.
        // Bandaid fix, not necessarily the best.

        // Assume that only resolving x + y or Mem[x + y] may cause heavy simplifications
        let needs_interning = match value.ty {
            OperandType::Arithmetic(..) => true,
            OperandType::Memory(ref mem) => match mem.address.ty {
                OperandType::Arithmetic(..) => true,
                _ => false,
            },
            _ => false,
        };

        if needs_interning {
            interner.intern_and_get(x)
        } else {
            x
        }
    }

    /// Tries to find an register/memory address corresponding to a resolved value.
    pub fn unresolve(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        // TODO: Could also check xmm but who honestly uses it for unique storage
        let interned = i.intern(val.clone());
        for (reg, &val) in self.state.iter().enumerate().take(0x10) {
            if interned == val {
                return Some(self.ctx.register(reg as u8));
            }
        }
        for (&key, &val) in &self.memory.map {
            if interned == val {
                return Some(i.operand(key));
            }
        }
        None
    }

    /// Tries to find an memory address corresponding to a resolved value.
    pub fn unresolve_memory(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        let interned = i.intern(val.clone());
        self.memory.reverse_lookup(interned).map(|x| self.ctx.mem64(&i.operand(x)))
    }
}


/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
pub fn merge_states<'a: 'r, 'r>(
    old: &'r mut ExecutionState<'a>,
    new: &'r mut ExecutionState<'a>,
    interner: &mut InternMap,
) -> Option<ExecutionState<'a>> {
    old.update_flags(interner);
    new.update_flags(interner);

    fn contains_undefined(oper: &Operand) -> bool {
        oper.iter().any(|x| match x.ty {
            OperandType::Undefined(_) => true,
            _ => false,
        })
    }

    let check_eq = |a: InternedOperand, b: InternedOperand| {
        a == b || a.is_undefined()
    };
    let check_memory_eq = |a: &Memory, b: &Memory, interner: &mut InternMap| {
        a.map.iter().all(|(&key, val)| {
            let oper = interner.operand(key);
            match contains_undefined(&oper) {
                true => true,
                false => match b.get(key) {
                    Some(b_val) => check_eq(*val, b_val),
                    None => true,
                },
            }
        })
    };

    let ctx = old.ctx;
    let merge = |a: InternedOperand, b: InternedOperand, i: &mut InternMap| -> InternedOperand {
        match a == b {
            true => a,
            false => i.new_undef(ctx),
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
                let lowest_bit = ctx.and_const(&con.0, 1);
                match old.resolve_apply_constraints(&lowest_bit, interner).if_constant() {
                    Some(1) => result = Some(con.clone()),
                    _ => (),
                }
            }
        }
        if new.unresolved_constraint.is_none() {
            if let Some(ref con) = old.unresolved_constraint {
                // As long as we're working with flags, limiting to lowest bit
                // allows simplifying cases like (undef | 1)
                let lowest_bit = ctx.and_const(&con.0, 1);
                match new.resolve_apply_constraints(&lowest_bit, interner).if_constant() {
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
        !check_memory_eq(&old.memory, &new.memory, interner) ||
        merged_ljec.as_ref().map(|x| *x != old.unresolved_constraint).unwrap_or(false) || (
            old.resolved_constraint.is_some() &&
            old.resolved_constraint != new.resolved_constraint
        );
    if changed {
        let state = array_init::array_init(|i| merge(old.state[i], new.state[i], interner));
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
        Some(ExecutionState {
            state,
            cached_low_registers,
            memory: old.memory.merge(&new.memory, interner, ctx),
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
