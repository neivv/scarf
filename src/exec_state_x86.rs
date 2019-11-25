use std::rc::Rc;

use byteorder::{ReadBytesExt, LE};

use crate::analysis;
use crate::disasm::{Disassembler32, DestOperand, Operation};
use crate::exec_state::{Constraint, InternMap, InternedOperand, Memory, XmmOperand};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
    ArithOpType,
};
use crate::{BinaryFile, VirtualAddress};

impl<'a> ExecutionStateTrait<'a> for ExecutionState<'a> {
    type VirtualAddress = VirtualAddress;
    type Disassembler = Disassembler32<'a>;

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
        let value = Operand::simplified(value);
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
        let mut stack_op;
        let mut op_ref = if op.is_simplified() {
            op
        } else {
            stack_op = Operand::simplified(op.clone());
            &stack_op
        };
        if let Some(ref constraint) = self.unresolved_constraint {
            stack_op = constraint.apply_to(op_ref);
            op_ref = &stack_op;
        }
        let val = self.resolve(op_ref, i);
        if let Some(ref constraint) = self.resolved_constraint {
            constraint.apply_to(&val)
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

    fn apply_call(&mut self, ret: VirtualAddress, i: &mut InternMap) {
        use crate::operand::operand_helpers::*;
        let ctx = self.ctx;
        let esp = ctx.register(4);
        self.move_to(
            &DestOperand::from_oper(&esp),
            operand_sub(esp.clone(), ctx.const_4()),
            i,
        );
        self.move_to(
            &DestOperand::from_oper(&mem32(esp)),
            ctx.constant(ret.0),
            i,
        );
    }

    fn initial_state(
        operand_ctx: &'a OperandContext,
        binary: &'a BinaryFile<VirtualAddress>,
        interner: &mut InternMap,
    ) -> ExecutionState<'a> {
        use crate::operand::operand_helpers::*;
        let mut state = ExecutionState::with_binary(binary, operand_ctx, interner);

        // Set the return address to somewhere in 0x400000 range
        let return_address = mem32(operand_ctx.register(4));
        state.move_to(
            &DestOperand::from_oper(&return_address),
            operand_ctx.constant(binary.code_section().virtual_address.0 + 0x4230),
            interner
        );

        // Set the bytes above return address to 'call eax' to make it look like a legitmate call.
        state.move_to(
            &DestOperand::from_oper(&mem_variable(MemAccessSize::Mem8,
                operand_sub(
                    return_address.clone(),
                    operand_ctx.const_1(),
                ),
            )),
            operand_ctx.constant(0xd0),
            interner
        );
        state.move_to(
            &DestOperand::from_oper(&mem_variable(MemAccessSize::Mem8,
                operand_sub(
                    return_address,
                    operand_ctx.const_2(),
                ),
            )),
            operand_ctx.const_ff(),
            interner
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
    ) -> Result<Vec<Self::VirtualAddress>, crate::Error> {
        crate::analysis::find_relocs_x86(file)
    }

    fn value_limits(&self, value: &Rc<Operand>) -> (u64, u64) {
        if let Some(ref constraint) = self.resolved_constraint {
            crate::exec_state::value_limits_recurse(&constraint.0, value)
        } else {
            (0, u64::max_value())
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExecutionState<'a> {
    registers: [InternedOperand; 0x8],
    cached_low_registers: CachedLowRegisters,
    xmm_registers: [XmmOperand; 0x8],
    fpu_registers: [InternedOperand; 0x8],
    flags: Flags,
    memory: Memory,
    resolved_constraint: Option<Constraint>,
    unresolved_constraint: Option<Constraint>,
    ctx: &'a OperandContext,
    code_sections: Vec<&'a crate::BinarySection<VirtualAddress>>,
    pending_flags: Option<ArithOperand>,
}

/// Caches ax/al/ah resolving.
/// InternedOperand(0) means that a register is not cached.
#[derive(Debug, Clone)]
struct CachedLowRegisters {
    registers: [[InternedOperand; 2]; 8],
}

impl CachedLowRegisters {
    fn new() -> CachedLowRegisters {
        CachedLowRegisters {
            registers: [[InternedOperand(0); 2]; 8],
        }
    }

    fn get_16(&self, register: u8) -> InternedOperand {
        self.registers[register as usize][0]
    }

    fn get_low8(&self, register: u8) -> InternedOperand {
        self.registers[register as usize][1]
    }

    fn set_16(&mut self, register: u8, value: InternedOperand) {
        self.registers[register as usize][0] = value;
    }

    fn set_low8(&mut self, register: u8, value: InternedOperand) {
        self.registers[register as usize][1] = value;
    }

    fn invalidate(&mut self, register: u8) {
        self.registers[register as usize] = [InternedOperand(0); 2];
    }
}

#[derive(Debug, Clone)]
pub struct Flags {
    zero: InternedOperand,
    carry: InternedOperand,
    overflow: InternedOperand,
    sign: InternedOperand,
    parity: InternedOperand,
    direction: InternedOperand,
}

/// Handles regular &'mut Operand assign for regs,
/// and the more complicated one for memory
enum Destination<'a> {
    Oper(&'a mut InternedOperand),
    Register16(&'a mut InternedOperand),
    Register8High(&'a mut InternedOperand),
    Register8Low(&'a mut InternedOperand),
    Pair(&'a mut InternedOperand, &'a mut InternedOperand),
    Memory(&'a mut Memory, Rc<Operand>, MemAccessSize),
}

/// Masks arithmetic operations to 32-bit, but keeps registers/undefined as they are.
/// Also removes 0xffff_ffff masks from them.
fn masked_to_32bit(value: &Rc<Operand>, ctx: &OperandContext) -> Rc<Operand> {
    use crate::operand::operand_helpers::*;
    // Allow undefined to be stored as is due to it mattering state merging
    // and interning. Hopefully won't cause issues..
    if value.relevant_bits().end > 32 {
        if value.is_undefined() || value.if_register().is_some() {
            value.clone()
        } else {
            operand_and(value.clone(), ctx.const_ffffffff())
        }
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

impl<'a> Destination<'a> {
    fn set(self, value: Rc<Operand>, intern_map: &mut InternMap, ctx: &OperandContext) {
        use crate::operand::operand_helpers::*;
        match self {
            Destination::Oper(o) => {
                let masked = masked_to_32bit(&value, ctx);
                *o = intern_map.intern(Operand::simplified(masked));
            }
            Destination::Register16(o) => {
                let old = intern_map.operand(*o);
                let masked = if value.relevant_bits().end > 16 {
                    operand_and(value, ctx.const_ffff())
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.end <= 16 {
                    Operand::simplified(masked)
                } else {
                    Operand::simplified(operand_or(
                        operand_and(old, ctx.const_ffff0000()),
                        masked,
                    ))
                };
                *o = intern_map.intern(new);
            }
            Destination::Register8High(o) => {
                let old = intern_map.operand(*o);
                let masked = if value.relevant_bits().end > 8 {
                    operand_and(value, ctx.const_ff())
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.start >= 8 && old_bits.end <= 16 {
                    Operand::simplified(operand_lsh(masked, ctx.const_8()))
                } else {
                    Operand::simplified(operand_or(
                        operand_and(old, ctx.const_ffff00ff()),
                        operand_lsh(masked, ctx.const_8())
                    ))
                };
                *o = intern_map.intern(new);
            }
            Destination::Register8Low(o) => {
                let old = intern_map.operand(*o);
                let masked = if value.relevant_bits().end > 8 {
                    operand_and(value, ctx.const_ff())
                } else {
                    value
                };
                let old_bits = old.relevant_bits();
                let new = if old_bits.end <= 8 {
                    Operand::simplified(masked)
                } else {
                    Operand::simplified(operand_or(
                        operand_and(old, ctx.const_ffffff00()),
                        masked,
                    ))
                };
                *o = intern_map.intern(new);
            }
            Destination::Pair(high, low) => {
                let (val_high, val_low) = Operand::pair(&value);
                *high = intern_map.intern(Operand::simplified(val_high));
                *low = intern_map.intern(Operand::simplified(val_low));
            }
            Destination::Memory(mem, addr, size) => {
                let addr = if addr.relevant_bits().end > 32 {
                    Operand::simplified(operand_and(addr, ctx.const_ffffffff()))
                } else {
                    addr
                };
                if size == MemAccessSize::Mem64 {
                    // Split into two u32 sets
                    Destination::Memory(mem, addr.clone(), MemAccessSize::Mem32)
                        .set(operand_and64(value.clone(), ctx.const_ffffffff()), intern_map, ctx);
                    let addr = operand_add(addr, ctx.const_4());
                    Destination::Memory(mem, addr, MemAccessSize::Mem32)
                        .set(operand_rsh64(value, ctx.constant(32)), intern_map, ctx);
                    return;
                }
                if let Some((base, offset)) = Operand::const_offset(&addr, ctx) {
                    let offset = offset as u32;
                    let offset_4 = offset & 3;
                    let offset_rest = offset & !3;
                    if offset_4 != 0 {
                        let size_bits = match size {
                            MemAccessSize::Mem32 => 32,
                            MemAccessSize::Mem16 => 16,
                            MemAccessSize::Mem8 => 8,
                            MemAccessSize::Mem64 => unreachable!(),
                        };
                        let low_base = Operand::simplified(
                            operand_add(base.clone(), ctx.constant(offset_rest))
                        );
                        let low_i = intern_map.intern(low_base.clone());
                        let low_old = mem.get(low_i)
                            .map(|x| intern_map.operand(x))
                            .unwrap_or_else(|| mem32(low_base));

                        let mask_low = offset_4 * 8;
                        let mask_high = (mask_low + size_bits).min(0x20);
                        let mask = !0 >> mask_low << mask_low <<
                            (0x20 - mask_high) >> (0x20 - mask_high);
                        let low_value = Operand::simplified(operand_or(
                            operand_and(
                                operand_lsh(
                                    value.clone(),
                                    ctx.constant(8 * offset_4),
                                ),
                                ctx.constant(mask),
                            ),
                            operand_and(
                                low_old,
                                ctx.constant(!mask),
                            ),
                        ));
                        mem.set(low_i, intern_map.intern(low_value));
                        let needs_high = mask_low + size_bits > 0x20;
                        if needs_high {
                            let high_base = Operand::simplified(
                                operand_add(
                                    base.clone(),
                                    ctx.constant(offset_rest.wrapping_add(4)),
                                )
                            );
                            let high_i = intern_map.intern(high_base.clone());
                            let high_old = mem.get(high_i)
                                .map(|x| intern_map.operand(x))
                                .unwrap_or_else(|| mem32(high_base));
                            let mask = !0 >> (0x20 - (mask_low + size_bits - 0x20));
                            let high_value = Operand::simplified(operand_or(
                                operand_and(
                                    operand_rsh(
                                        value,
                                        ctx.constant(0x20 - 8 * offset_4),
                                    ),
                                    ctx.constant(mask),
                                ),
                                operand_and(
                                    high_old,
                                    ctx.constant(!mask),
                                ),
                            ));
                            mem.set(high_i, intern_map.intern(high_value));
                        }
                        return;
                    }
                }
                let addr = intern_map.intern(Operand::simplified(addr));
                let value = Operand::simplified(match size {
                    MemAccessSize::Mem8 => operand_and(value, ctx.const_ff()),
                    MemAccessSize::Mem16 => operand_and(value, ctx.const_ffff()),
                    MemAccessSize::Mem32 => value,
                    MemAccessSize::Mem64 => unreachable!(),
                });
                mem.set(addr, intern_map.intern(value));
            }
        }
    }
}

impl Flags {
    fn initial(ctx: &OperandContext, interner: &mut InternMap) -> Flags {
        Flags {
            zero: interner.intern(ctx.flag_z()),
            carry: interner.intern(ctx.flag_c()),
            overflow: interner.intern(ctx.flag_o()),
            sign: interner.intern(ctx.flag_s()),
            parity: interner.intern(ctx.flag_p()),
            direction: interner.intern(ctx.const_0()),
        }
    }
}

impl<'a> ExecutionState<'a> {
    pub fn new<'b>(
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        let mut registers = [InternedOperand(0); 8];
        let mut fpu_registers = [InternedOperand(0); 8];
        let mut xmm_registers = [XmmOperand(
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
        ); 8];
        for i in 0..8 {
            registers[i] = interner.intern(ctx.register(i as u8));
            fpu_registers[i] = interner.intern(ctx.register_fpu(i as u8));
            xmm_registers[i] = XmmOperand::initial(i as u8, interner);
        }
        ExecutionState {
            registers,
            cached_low_registers: CachedLowRegisters::new(),
            xmm_registers,
            fpu_registers,
            flags: Flags::initial(ctx, interner),
            memory: Memory::new(),
            resolved_constraint: None,
            unresolved_constraint: None,
            ctx,
            code_sections: Vec::new(),
            pending_flags: None,
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile<VirtualAddress>,
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        let mut result = ExecutionState::new(ctx, interner);
        result.code_sections = binary.code_sections().collect();
        result
    }

    pub fn update(&mut self, operation: &Operation, intern_map: &mut InternMap) {
        let ctx = self.ctx;
        match operation {
            Operation::Move(dest, value, cond) => {
                let value = value.clone();
                if let Some(cond) = cond {
                    let cond = Operand::simplified(cond.clone());
                    match self.resolve(&cond, intern_map).if_constant() {
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
            Operation::Swap(left, right) => {
                let left_res = self.resolve(&left.as_operand(ctx), intern_map);
                let right_res = self.resolve(&right.as_operand(ctx), intern_map);
                self.get_dest_invalidate_constraints(&left, intern_map)
                    .set(right_res, intern_map, ctx);
                self.get_dest_invalidate_constraints(&right, intern_map)
                    .set(left_res, intern_map, ctx);
            }
            Operation::Call(_) => {
                let mut ids = intern_map.many_undef(ctx, 9);
                self.unresolved_constraint = None;
                if let Some(ref mut c) = self.resolved_constraint {
                    if c.invalidate_memory() == crate::exec_state::ConstraintFullyInvalid::Yes {
                        self.resolved_constraint = None
                    }
                }
                self.registers[0] = ids.next();
                self.registers[1] = ids.next();
                self.registers[2] = ids.next();
                self.registers[4] = ids.next();
                self.flags = Flags {
                    zero: ids.next(),
                    carry: ids.next(),
                    overflow: ids.next(),
                    sign: ids.next(),
                    parity: ids.next(),
                    direction: intern_map.intern(ctx.const_0()),
                };
            }
            Operation::Jump { .. } => {
            }
            Operation::Return(_) => {
            }
            Operation::Special(code) => {
                if code == &[0xd9, 0xf6] {
                    // fdecstp
                    self.fpu_registers.rotate_left(1);
                } else if code == &[0xd9, 0xf7] {
                    // fincstp
                    self.fpu_registers.rotate_right(1);
                }
            }
            Operation::SetFlags(arith, size) => {
                if *size != MemAccessSize::Mem32 {
                    return;
                }
                let left = self.resolve(&arith.left, intern_map);
                let right = self.resolve(&arith.right, intern_map);
                let arith = ArithOperand {
                    left,
                    right,
                    ty: arith.ty,
                };
                self.pending_flags = Some(arith);
                // Could try to do smarter invalidation, but since in practice unresolved
                // constraints always are bunch of flags, invalidate it completely.
                self.unresolved_constraint = None;
            }
        }
    }

    fn update_flags(&mut self, intern_map: &mut InternMap) {
        if let Some(arith) = self.pending_flags.take() {
            self.set_flags(arith, intern_map);
        }
    }

    fn set_flags(&mut self, arith: ArithOperand, intern_map: &mut InternMap) {
        use crate::operand::ArithOpType::*;
        use crate::operand_helpers::*;
        let resolved_left = &arith.left;
        let resolved_right = arith.right;
        let result = operand_arith(arith.ty, resolved_left.clone(), resolved_right);
        let result = Operand::simplified(result);
        match arith.ty {
            Add => {
                let carry = operand_gt(resolved_left.clone(), result.clone());
                let overflow = operand_gt_signed(resolved_left.clone(), result.clone());
                self.flags.carry = intern_map.intern(Operand::simplified(carry));
                self.flags.sign = intern_map.intern(Operand::simplified(overflow));
                self.result_flags(result, intern_map);
            }
            Sub => {
                let carry = operand_gt(result.clone(), resolved_left.clone());
                let overflow = operand_gt_signed(result.clone(), resolved_left.clone());
                self.flags.carry = intern_map.intern(Operand::simplified(carry));
                self.flags.sign = intern_map.intern(Operand::simplified(overflow));
                self.result_flags(result, intern_map);
            }
            Xor | And | Or => {
                let zero = intern_map.intern(self.ctx.const_0());
                self.flags.carry = zero;
                self.flags.overflow = zero;
                self.result_flags(result, intern_map);
            }
            Lsh | Rsh  => {
                let mut ids = intern_map.many_undef(self.ctx, 2);
                self.flags.carry = ids.next();
                self.flags.overflow = ids.next();
                self.result_flags(result, intern_map);
            }
            _ => {
                let mut ids = intern_map.many_undef(self.ctx, 5);
                self.flags.zero = ids.next();
                self.flags.carry = ids.next();
                self.flags.overflow = ids.next();
                self.flags.sign = ids.next();
                self.flags.parity = ids.next();
            }
        }
    }

    fn result_flags(&mut self, result: Rc<Operand>, intern_map: &mut InternMap) {
        use crate::operand_helpers::*;
        let ctx = self.ctx;
        let zero = Operand::simplified(operand_eq(result.clone(), ctx.const_0()));
        let sign = Operand::simplified(
            operand_ne64(
                ctx,
                operand_and64(
                    ctx.constant(0x8000_0000),
                    result.clone(),
                ),
                ctx.const_0(),
            )
        );
        let parity = Operand::simplified(
            operand_arith(ArithOpType::Parity, result, ctx.const_0())
        );
        self.flags.zero = intern_map.intern(zero);
        self.flags.sign = intern_map.intern(sign);
        self.flags.parity = intern_map.intern(parity);
    }

    /// Makes all of memory undefined
    pub fn clear_memory(&mut self) {
        self.memory = Memory::new();
    }

    fn get_dest_invalidate_constraints<'s>(
        &'s mut self,
        dest: &DestOperand,
        interner: &mut InternMap,
    ) -> Destination<'s> {
        self.unresolved_constraint = match self.unresolved_constraint {
            Some(ref s) => s.invalidate_dest_operand(dest),
            None => None,
        };
        if let Some(ref mut s) = self.resolved_constraint {
            if let DestOperand::Memory(_) = dest {
                if s.invalidate_memory() == crate::exec_state::ConstraintFullyInvalid::Yes {
                    self.resolved_constraint = None;
                }
            }
        }
        self.get_dest(dest, interner, false)
    }

    fn get_dest(
        &mut self,
        dest: &DestOperand,
        intern_map: &mut InternMap,
        dest_is_resolved: bool,
    ) -> Destination {
        match *dest {
            DestOperand::Register32(reg) | DestOperand::Register64(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Oper(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register16(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register16(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register8High(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register8High(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register8Low(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register8Low(&mut self.registers[reg.0 as usize])
            }
            DestOperand::PairEdxEax => {
                self.cached_low_registers.invalidate(0);
                self.cached_low_registers.invalidate(2);
                let (eax, rest) = self.registers.split_first_mut().unwrap();
                let edx = &mut rest[1];
                Destination::Pair(edx, eax)
            }
            DestOperand::Fpu(id) => {
                Destination::Oper(&mut self.fpu_registers[id as usize])
            }
            DestOperand::Xmm(reg, word) => {
                Destination::Oper(self.xmm_registers[reg as usize].word_mut(word))
            }
            DestOperand::Flag(flag) => {
                self.update_flags(intern_map);
                Destination::Oper(match flag {
                    Flag::Zero => &mut self.flags.zero,
                    Flag::Carry => &mut self.flags.carry,
                    Flag::Overflow => &mut self.flags.overflow,
                    Flag::Sign => &mut self.flags.sign,
                    Flag::Parity => &mut self.flags.parity,
                    Flag::Direction => &mut self.flags.direction,
                })
            }
            DestOperand::Memory(ref mem) => {
                let address = if dest_is_resolved {
                    mem.address.clone()
                } else {
                    self.resolve(&mem.address, intern_map)
                };
                Destination::Memory(&mut self.memory, address, mem.size)
            }
        }
    }

    fn resolve_mem(&mut self, mem: &MemAccess, i: &mut InternMap) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;

        let address = self.resolve(&mem.address, i);
        let address = if address.relevant_bits().end > 32 {
            Operand::simplified(operand_and(address, self.ctx.const_ffffffff()))
        } else {
            address
        };
        if MemAccessSize::Mem64 == mem.size {
            // Split into 2 32-bit resolves
            return operand_or64(
                operand_lsh64(
                    self.resolve_mem(&MemAccess {
                        address: operand_add64(mem.address.clone(), self.ctx.const_4()),
                        size: MemAccessSize::Mem32,
                    }, i),
                    self.ctx.constant(32),
                ),
                self.resolve_mem(&MemAccess {
                    address: mem.address.clone(),
                    size: MemAccessSize::Mem32,
                }, i),
            );
        }
        let size_bytes = match mem.size {
            MemAccessSize::Mem8 => 1,
            MemAccessSize::Mem16 => 2,
            MemAccessSize::Mem32 => 4,
            MemAccessSize::Mem64 => unreachable!(),
        };
        if let Some(c) = address.if_constant() {
            // Simplify constants stored in code section (constant switch jumps etc)
            if let Some(end) = c.checked_add(size_bytes) {
                let section = self.code_sections.iter().find(|s| {
                    s.virtual_address.0 <= c && s.virtual_address.0 + s.virtual_size >= end
                });
                if let Some(section) = section {
                    let offset = (c - section.virtual_address.0) as usize;
                    let val = match mem.size {
                        MemAccessSize::Mem8 => section.data[offset] as u32,
                        MemAccessSize::Mem16 => {
                            (&section.data[offset..]).read_u16::<LE>().unwrap_or(0) as u32
                        }
                        MemAccessSize::Mem32 => {
                            (&section.data[offset..]).read_u32::<LE>().unwrap_or(0)
                        }
                        MemAccessSize::Mem64 => unreachable!(),
                    };
                    return self.ctx.constant(val);
                }
            }
        }

        // Use 4-aligned addresses if there's a const offset
        if let Some((base, offset)) = Operand::const_offset(&address, self.ctx) {
            let offset = offset as u32;
            let offset_4 = offset & 3;
            let offset_rest = offset & !3;
            if offset_4 != 0 {
                let low_base = Operand::simplified(
                    operand_add(base.clone(), self.ctx.constant(offset_rest))
                );
                let low = self.memory.get(i.intern(low_base.clone()))
                    .map(|x| i.operand(x))
                    .unwrap_or_else(|| mem32(low_base));
                let low = operand_rsh(low, self.ctx.constant(offset_4 * 8));
                let combined = if offset_4 + size_bytes > 4 {
                    let high_base = Operand::simplified(
                        operand_add(base.clone(), self.ctx.constant(offset_rest.wrapping_add(4)))
                    );
                    let high = self.memory.get(i.intern(high_base.clone()))
                        .map(|x| i.operand(x))
                        .unwrap_or_else(|| mem32(high_base));
                    let high = operand_lsh(high, self.ctx.constant(0x20 - offset_4 * 8));
                    operand_or(low, high)
                } else {
                    low
                };
                let masked = match mem.size {
                    MemAccessSize::Mem8 => operand_and(combined, self.ctx.const_ff()),
                    MemAccessSize::Mem16 => operand_and(combined, self.ctx.const_ffff()),
                    MemAccessSize::Mem32 => combined,
                    MemAccessSize::Mem64 => unreachable!(),
                };
                return masked;
            }
        }
        self.memory.get(i.intern(address.clone()))
            .map(|interned| i.operand(interned))
            .unwrap_or_else(|| mem_variable_rc(mem.size, address))
    }

    /// Checks cached/caches `reg & ff` masks.
    /// Also checking ffff_ffff for just register directly
    fn try_resolve_partial_register(
        &mut self,
        left: &Rc<Operand>,
        right: &Rc<Operand>,
        interner: &mut InternMap,
    ) -> Option<Rc<Operand>> {
        use crate::operand::operand_helpers::*;
        let (const_op, c, other) = match left.ty {
            OperandType::Constant(c) => (left, c as u64, right),
            OperandType::Constant64(c) => (left, c, right),
            _ => match right.ty {
                OperandType::Constant(c) => (right, c as u64, left),
                OperandType::Constant64(c) => (right, c, left),
                _ => return None,
            }
        };
        let reg = other.if_register()?;
        if c <= 0xff {
            let interned = match self.cached_low_registers.get_low8(reg.0) {
                InternedOperand(0) => {
                    let op = operand_and(
                        interner.operand(self.registers[reg.0 as usize]),
                        self.ctx.const_ff(),
                    );
                    let interned = interner.intern(Operand::simplified(op));
                    self.cached_low_registers.set_low8(reg.0, interned);
                    interned
                }
                x => x,
            };
            let op = interner.operand(interned);
            if c == 0xff {
                Some(op)
            } else {
                Some(operand_and(op, const_op.clone()))
            }
        } else if c <= 0xffff {
            let interned = match self.cached_low_registers.get_16(reg.0) {
                InternedOperand(0) => {
                    let op = operand_and(
                        interner.operand(self.registers[reg.0 as usize]),
                        self.ctx.const_ffff(),
                    );
                    let interned = interner.intern(Operand::simplified(op));
                    self.cached_low_registers.set_16(reg.0, interned);
                    interned
                }
                x => x,
            };
            let op = interner.operand(interned);
            if c == 0xffff {
                Some(op)
            } else {
                Some(operand_and(op, const_op.clone()))
            }
        } else if c <= 0xffff_ffff {
            let op = interner.operand(self.registers[reg.0 as usize]);
            if c == 0xffff_ffff {
                Some(op)
            } else {
                Some(operand_and(op, const_op.clone()))
            }
        } else {
            None
        }
    }

    fn resolve_no_simplify(
        &mut self,
        value: &Rc<Operand>,
        interner: &mut InternMap,
    ) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;

        match value.ty {
            OperandType::Register(reg) => {
                interner.operand(self.registers[reg.0 as usize])
            }
            OperandType::Pair(ref high, ref low) => {
                pair(self.resolve(&high, interner), self.resolve(&low, interner))
            }
            OperandType::Xmm(reg, word) => {
                interner.operand(self.xmm_registers[reg as usize].word(word))
            }
            OperandType::Fpu(id) => {
                interner.operand(self.fpu_registers[id as usize])
            }
            OperandType::Flag(flag) => {
                self.update_flags(interner);
                interner.operand(match flag {
                    Flag::Zero => self.flags.zero,
                    Flag::Carry => self.flags.carry,
                    Flag::Overflow => self.flags.overflow,
                    Flag::Sign => self.flags.sign,
                    Flag::Parity => self.flags.parity,
                    Flag::Direction => self.flags.direction,
                }).clone()
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
                let ty = OperandType::Arithmetic(ArithOperand {
                    ty: op.ty,
                    left,
                    right,
                });
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::Arithmetic64(ref op) => {
                let ty = OperandType::Arithmetic64(ArithOperand {
                    ty: op.ty,
                    left: self.resolve(&op.left, interner),
                    right: self.resolve(&op.right, interner),
                });
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::ArithmeticF32(ref op) => {
                let ty = OperandType::ArithmeticF32(ArithOperand {
                    ty: op.ty,
                    left: self.resolve(&op.left, interner),
                    right: self.resolve(&op.right, interner),
                });
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::ArithmeticHigh(ref op) => {
                let ty = OperandType::ArithmeticHigh(ArithOperand {
                    ty: op.ty,
                    left: self.resolve(&op.left, interner),
                    right: self.resolve(&op.right, interner),
                });
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::Constant(_) => value.clone(),
            OperandType::Constant64(_) => value.clone(),
            OperandType::Custom(_) => value.clone(),
            OperandType::Memory(ref mem) => {
                self.resolve_mem(mem, interner)
            }
            OperandType::Undefined(_) => value.clone(),
            OperandType::SignExtend(ref val, from, to) => {
                let val = self.resolve(val, interner);
                let ty = OperandType::SignExtend(val, from, to);
                Operand::new_not_simplified_rc(ty)
            }
        }
    }

    pub fn resolve(&mut self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
        let x = self.resolve_no_simplify(value, interner);
        if x.is_simplified() {
            return x;
        }
        let operand = Operand::simplified(x);
        // Intern in case the simplification created a very deeply different operand tree,
        // as repeating the resolving would give an equal operand with different addresses.
        // Bandaid fix, not necessarily the best.
        interner.intern_and_get(operand)
    }

    /// Tries to find an register/memory address corresponding to a resolved value.
    pub fn unresolve(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        // TODO: Could also check xmm but who honestly uses it for unique storage
        let interned = i.intern(val.clone());
        for (reg, &val) in self.registers.iter().enumerate() {
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
        use crate::operand_helpers::*;
        let interned = i.intern(val.clone());
        self.memory.reverse_lookup(interned).map(|x| mem32(i.operand(x)))
    }

    pub fn memory(&self) -> &Memory {
        &self.memory
    }

    pub fn replace_memory(&mut self, new: Memory) {
        self.memory = new;
    }
}

/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
pub fn merge_states<'a: 'r, 'r>(
    old: &'r mut ExecutionState<'a>,
    new: &'r mut ExecutionState<'a>,
    interner: &mut InternMap,
) -> Option<ExecutionState<'a>> {
    use crate::operand::operand_helpers::*;

    old.update_flags(interner);
    new.update_flags(interner);

    let check_eq = |a: InternedOperand, b: InternedOperand| {
        a == b || a.is_undefined()
    };
    let check_xmm_eq = |a: &XmmOperand, b: &XmmOperand| {
        check_eq(a.0, b.0) &&
            check_eq(a.1, b.1) &&
            check_eq(a.2, b.2) &&
            check_eq(a.3, b.3)
    };
    let check_flags_eq = |a: &Flags, b: &Flags| {
        check_eq(a.zero, b.zero) &&
            check_eq(a.carry, b.carry) &&
            check_eq(a.overflow, b.overflow) &&
            check_eq(a.sign, b.sign) &&
            check_eq(a.parity, b.parity) &&
            check_eq(a.direction, b.direction)
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
    let merge_xmm = |a: &XmmOperand, b: &XmmOperand, i: &mut InternMap| -> XmmOperand {
        XmmOperand(
            merge(a.0, b.0, i),
            merge(a.1, b.1, i),
            merge(a.2, b.2, i),
            merge(a.3, b.3, i),
        )
    };

    let merged_ljec = if old.unresolved_constraint != new.unresolved_constraint {
        let mut result = None;
        // If one state has no constraint but matches the constrait of the other
        // state, the constraint should be kept on merge.
        if old.unresolved_constraint.is_none() {
            if let Some(ref con) = new.unresolved_constraint {
                // As long as we're working with flags, limiting to lowest bit
                // allows simplifying cases like (undef | 1)
                let lowest_bit = operand_and(ctx.const_1(), con.0.clone());
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
                let lowest_bit = operand_and(ctx.const_1(), con.0.clone());
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
        old.registers.iter().zip(new.registers.iter())
            .any(|(&a, &b)| !check_eq(a, b)) ||
        old.xmm_registers.iter().zip(new.xmm_registers.iter())
            .any(|(a, b)| !check_xmm_eq(a, b)) ||
        old.fpu_registers.iter().zip(new.fpu_registers.iter())
            .any(|(&a, &b)| !check_eq(a, b)) ||
        !check_flags_eq(&old.flags, &new.flags) ||
        !check_memory_eq(&old.memory, &new.memory, interner) ||
        merged_ljec.as_ref().map(|x| *x != old.unresolved_constraint).unwrap_or(false) || (
            old.resolved_constraint.is_some() &&
            old.resolved_constraint != new.resolved_constraint
        );
    if changed {
        let mut registers = [InternedOperand(0); 8];
        let mut cached_low_registers = CachedLowRegisters::new();
        let mut fpu_registers = [InternedOperand(0); 8];
        let mut xmm_registers = [XmmOperand(
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
        ); 8];
        let mut flags = Flags {
            zero: InternedOperand(0),
            carry: InternedOperand(0),
            overflow: InternedOperand(0),
            sign: InternedOperand(0),
            parity: InternedOperand(0),
            direction: InternedOperand(0),
        };
        {
            for i in 0..8 {
                registers[i] = merge(old.registers[i], new.registers[i], interner);
                fpu_registers[i] = merge(old.fpu_registers[i], new.fpu_registers[i], interner);
                xmm_registers[i] =
                    merge_xmm(&old.xmm_registers[i], &new.xmm_registers[i], interner);
                let old_reg = &old.cached_low_registers.registers[i];
                let new_reg = &new.cached_low_registers.registers[i];
                for j in 0..old_reg.len() {
                    if old_reg[j] == new_reg[j] {
                        // Doesn't merge things but sets them uncached if they differ
                        cached_low_registers.registers[i][j] = old_reg[j];
                    }
                }
            }
            let mut flags = [
                (&mut flags.zero, old.flags.zero, new.flags.zero),
                (&mut flags.carry, old.flags.carry, new.flags.carry),
                (&mut flags.overflow, old.flags.overflow, new.flags.overflow),
                (&mut flags.sign, old.flags.sign, new.flags.sign),
                (&mut flags.parity, old.flags.parity, new.flags.parity),
                (&mut flags.direction, old.flags.direction, new.flags.direction),
            ];
            for &mut (ref mut out, old, new) in &mut flags {
                **out = merge(old, new, interner);
            }
        }
        Some(ExecutionState {
            registers,
            cached_low_registers,
            fpu_registers,
            xmm_registers,
            flags,
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
            code_sections: old.code_sections.clone(),
        })
    } else {
        None
    }
}

fn contains_undefined(oper: &Operand) -> bool {
    oper.iter().any(|x| match x.ty {
        OperandType::Undefined(_) => true,
        _ => false,
    })
}

#[test]
fn merge_state_constraints_eq() {
    use crate::operand::operand_helpers::*;
    let mut i = InternMap::new();
    let ctx = crate::operand::OperandContext::new();
    let state_a = ExecutionState::new(&ctx, &mut i);
    let mut state_b = ExecutionState::new(&ctx, &mut i);
    let sign_eq_overflow_flag = Operand::simplified(operand_eq(
        ctx.flag_o(),
        ctx.flag_s(),
    ));
    let mut state_a = state_a.assume_jump_flag(&sign_eq_overflow_flag, true, &mut i);
    state_b.move_to(&DestOperand::from_oper(&ctx.flag_o()), constval(1), &mut i);
    state_b.move_to(&DestOperand::from_oper(&ctx.flag_s()), constval(1), &mut i);
    let merged = merge_states(&mut state_b, &mut state_a, &mut i).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);
}

#[test]
fn merge_state_constraints_or() {
    use crate::operand::operand_helpers::*;
    let mut i = InternMap::new();
    let ctx = crate::operand::OperandContext::new();
    let state_a = ExecutionState::new(&ctx, &mut i);
    let mut state_b = ExecutionState::new(&ctx, &mut i);
    let sign_or_overflow_flag = Operand::simplified(operand_or(
        ctx.flag_o(),
        ctx.flag_s(),
    ));
    let mut state_a = state_a.assume_jump_flag(&sign_or_overflow_flag, true, &mut i);
    state_b.move_to(&DestOperand::from_oper(&ctx.flag_s()), constval(1), &mut i);
    let merged = merge_states(&mut state_b, &mut state_a, &mut i).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);
    // Should also happen other way, though then state_a must have something that is converted
    // to undef.
    let merged = merge_states(&mut state_a, &mut state_b, &mut i).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);

    state_a.move_to(&DestOperand::from_oper(&ctx.flag_c()), constval(1), &mut i);
    let merged = merge_states(&mut state_a, &mut state_b, &mut i).unwrap();
    assert!(merged.unresolved_constraint.is_some());
    assert_eq!(merged.unresolved_constraint, state_a.unresolved_constraint);
}
