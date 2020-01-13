use std::rc::Rc;

use byteorder::{ReadBytesExt, LE};

use crate::analysis;
use crate::disasm::{Disassembler64, DestOperand, Operation};
use crate::exec_state::{Constraint, InternMap, InternedOperand, Memory, XmmOperand};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
    ArithOpType,
};
use crate::{BinaryFile, VirtualAddress64};

#[derive(Debug, Clone)]
pub struct ExecutionState<'a> {
    registers: [InternedOperand; 0x10],
    cached_low_registers: CachedLowRegisters,
    xmm_registers: [XmmOperand; 0x10],
    flags: Flags,
    memory: Memory,
    unresolved_constraint: Option<Constraint>,
    resolved_constraint: Option<Constraint>,
    ctx: &'a OperandContext,
    binary: Option<&'a BinaryFile<VirtualAddress64>>,
    /// Lazily update flags since a lot of instructions set them and
    /// they get discarded later.
    pending_flags: Option<(ArithOperand, MemAccessSize)>,
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
enum Destination<'a> {
    Oper(&'a mut InternedOperand),
    Register32(&'a mut InternedOperand),
    Register16(&'a mut InternedOperand),
    Register8High(&'a mut InternedOperand),
    Register8Low(&'a mut InternedOperand),
    Memory(&'a mut Memory, Rc<Operand>, MemAccessSize),
    Nop,
}

impl<'a> Destination<'a> {
    fn set(self, value: Rc<Operand>, intern_map: &mut InternMap, ctx: &OperandContext) {
        use crate::operand::operand_helpers::*;
        match self {
            Destination::Oper(o) => {
                *o = intern_map.intern(Operand::simplified(value));
            }
            Destination::Register32(o) => {
                // 32-bit register dest clears high bits (16- or 8-bit dests don't)
                let masked = if value.relevant_bits().end > 32 {
                    operand_and(value, ctx.const_ffffffff())
                } else {
                    value
                };
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
                        operand_and(old, ctx.constant(0xffff_ffff_ffff_0000)),
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
                        operand_and(old, ctx.constant(0xffff_ffff_ffff_00ff)),
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
                        operand_and(old, ctx.constant(0xffff_ffff_ffff_ff00)),
                        masked,
                    ))
                };
                *o = intern_map.intern(new);
            }
            Destination::Memory(mem, addr, size) => {
                if let Some((base, offset)) = Operand::const_offset(&addr, ctx) {
                    let offset_8 = offset & 7;
                    let offset_rest = offset & !7;
                    if offset_8 != 0 {
                        let size_bits = match size {
                            MemAccessSize::Mem64 => 64,
                            MemAccessSize::Mem32 => 32,
                            MemAccessSize::Mem16 => 16,
                            MemAccessSize::Mem8 => 8,
                        };
                        let low_base = Operand::simplified(
                            operand_add(base.clone(), ctx.constant(offset_rest))
                        );
                        let low_i = intern_map.intern(low_base.clone());
                        let low_old = mem.get(low_i)
                            .map(|x| intern_map.operand(x))
                            .unwrap_or_else(|| mem64(low_base));

                        let mask_low = offset_8 * 8;
                        let mask_high = (mask_low + size_bits).min(0x40);
                        let mask = !0 >> mask_low << mask_low <<
                            (0x40 - mask_high) >> (0x40 - mask_high);
                        let low_value = Operand::simplified(operand_or(
                            operand_and(
                                operand_lsh(
                                    value.clone(),
                                    ctx.constant(8 * offset_8),
                                ),
                                ctx.constant(mask),
                            ),
                            operand_and(
                                low_old,
                                ctx.constant(!mask),
                            ),
                        ));
                        mem.set(low_i, intern_map.intern(low_value));
                        let needs_high = mask_low + size_bits > 0x40;
                        if needs_high {
                            let high_base = Operand::simplified(
                                operand_add(
                                    base.clone(),
                                    ctx.constant(offset_rest.wrapping_add(8)),
                                )
                            );
                            let high_i = intern_map.intern(high_base.clone());
                            let high_old = mem.get(high_i)
                                .map(|x| intern_map.operand(x))
                                .unwrap_or_else(|| mem64(high_base));
                            let mask = !0 >> (0x40 - (mask_low + size_bits - 0x40));
                            let high_value = Operand::simplified(operand_or(
                                operand_and(
                                    operand_rsh(
                                        value,
                                        ctx.constant(0x80 - 8 * offset_8),
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
                    MemAccessSize::Mem32 => operand_and(value, ctx.const_ffffffff()),
                    MemAccessSize::Mem64 => value,
                });
                mem.set(addr, intern_map.intern(value));
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

    fn apply_call(&mut self, ret: VirtualAddress64, i: &mut InternMap) {
        use crate::operand::operand_helpers::*;
        let ctx = self.ctx;
        let rsp = ctx.register(4);
        self.move_to(
            &DestOperand::from_oper(&rsp),
            operand_sub(rsp.clone(), ctx.const_8()),
            i,
        );
        self.move_to(
            &DestOperand::from_oper(&mem64(rsp)),
            ctx.constant(ret.0),
            i,
        );
    }

    fn initial_state(
        operand_ctx: &'a OperandContext,
        binary: &'a BinaryFile<VirtualAddress64>,
        interner: &mut InternMap,
    ) -> ExecutionState<'a> {
        ExecutionState::with_binary(binary, operand_ctx, interner)
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
    ) -> Result<Vec<Self::VirtualAddress>, crate::Error> {
        crate::analysis::find_relocs_x86_64(file)
    }

    fn function_ranges_from_exception_info(
        file: &crate::BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<(u32, u32)>, crate::Error> {
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

impl<'a> ExecutionState<'a> {
    pub fn new<'b>(
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        let mut registers = [InternedOperand(0); 16];
        let mut xmm_registers = [XmmOperand(
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
        ); 16];
        for i in 0..16 {
            registers[i] = interner.intern(ctx.register(i as u8));
            xmm_registers[i] = XmmOperand::initial(i as u8, interner);
        }
        ExecutionState {
            registers,
            cached_low_registers: CachedLowRegisters::new(),
            xmm_registers,
            flags: Flags::initial(ctx, interner),
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
                    if c.invalidate_memory() == crate::exec_state::ConstraintFullyInvalid::Yes {
                        self.resolved_constraint = None
                    }
                }
                self.registers[0] = ids.next();
                self.registers[1] = ids.next();
                self.registers[2] = ids.next();
                self.registers[4] = ids.next();
                self.registers[8] = ids.next();
                self.registers[9] = ids.next();
                self.registers[10] = ids.next();
                self.registers[11] = ids.next();
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
        use crate::operand_helpers::*;
        let resolved_left = &arith.left;
        let resolved_right = arith.right;

        let ctx = self.ctx;
        let result = operand_arith(arith.ty, resolved_left.clone(), resolved_right);
        let result = Operand::simplified(result);
        match arith.ty {
            Add => {
                let overflow =
                    operand_gt_signed(resolved_left.clone(), result.clone(), size, ctx);
                let carry = operand_gt(resolved_left.clone(), result.clone());
                self.flags.carry = intern_map.intern(Operand::simplified(carry));
                self.flags.overflow = intern_map.intern(Operand::simplified(overflow));
                self.result_flags(result, size, intern_map);
            }
            Sub => {
                let overflow =
                    operand_gt_signed(resolved_left.clone(), result.clone(), size, ctx);
                let carry = operand_gt(result.clone(), resolved_left.clone());
                self.flags.carry = intern_map.intern(Operand::simplified(carry));
                self.flags.overflow = intern_map.intern(Operand::simplified(overflow));
                self.result_flags(result, size, intern_map);
            }
            Xor | And | Or => {
                let zero = intern_map.intern(ctx.const_0());
                self.flags.carry = zero;
                self.flags.overflow = zero;
                self.result_flags(result, size, intern_map);
            }
            Lsh | Rsh  => {
                let mut ids = intern_map.many_undef(ctx, 2);
                self.flags.carry = ids.next();
                self.flags.overflow = ids.next();
                self.result_flags(result, size, intern_map);
            }
            _ => {
                let mut ids = intern_map.many_undef(ctx, 5);
                self.flags.zero = ids.next();
                self.flags.carry = ids.next();
                self.flags.overflow = ids.next();
                self.flags.sign = ids.next();
                self.flags.parity = ids.next();
            }
        }
    }

    fn result_flags(
        &mut self,
        result: Rc<Operand>,
        size: MemAccessSize,
        intern_map: &mut InternMap,
    ) {
        use crate::operand_helpers::*;
        let parity;
        let ctx = self.ctx;
        let zero = Operand::simplified(operand_eq(result.clone(), ctx.const_0()));
        let sign_bit = match size {
            MemAccessSize::Mem8 => 0x80,
            MemAccessSize::Mem16 => 0x8000,
            MemAccessSize::Mem32 => 0x8000_0000,
            MemAccessSize::Mem64 => 0x8000_0000_0000_0000,
        };
        let sign = Operand::simplified(
            operand_ne(
                ctx,
                operand_and(
                    ctx.constant(sign_bit),
                    result.clone(),
                ),
                ctx.const_0(),
            )
        );
        // Parity is defined to be just lowest byte so it doesn't need special handling for
        // 64 bits.
        parity = Operand::simplified(
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
            DestOperand::Register64(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Oper(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register32(reg) => {
                self.cached_low_registers.invalidate(reg.0);
                Destination::Register32(&mut self.registers[reg.0 as usize])
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
            DestOperand::Fpu(_) => Destination::Nop,
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
        let size_bytes = match mem.size {
            MemAccessSize::Mem8 => 1,
            MemAccessSize::Mem16 => 2,
            MemAccessSize::Mem32 => 4,
            MemAccessSize::Mem64 => 8,
        };
        if let Some(c) = address.if_constant() {
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
                    let val = match mem.size {
                        MemAccessSize::Mem8 => section.data[offset] as u64,
                        MemAccessSize::Mem16 => {
                            (&section.data[offset..]).read_u16::<LE>().unwrap_or(0) as u64
                        }
                        MemAccessSize::Mem32 => {
                            (&section.data[offset..]).read_u32::<LE>().unwrap_or(0) as u64
                        }
                        MemAccessSize::Mem64 => {
                            (&section.data[offset..]).read_u64::<LE>().unwrap_or(0)
                        }
                    };
                    return self.ctx.constant(val);
                }
            }
        }

        // Use 8-aligned addresses if there's a const offset
        if let Some((base, offset)) = Operand::const_offset(&address, self.ctx) {
            let offset_8 = offset as u32 & 7;
            let offset_rest = offset & !7;
            if offset_8 != 0 {
                let low_base = Operand::simplified(
                    operand_add(base.clone(), self.ctx.constant(offset_rest))
                );
                let low = self.memory.get(i.intern(low_base.clone()))
                    .map(|x| i.operand(x))
                    .unwrap_or_else(|| mem64(low_base));
                let low = operand_rsh(low, self.ctx.constant(offset_8 as u64 * 8));
                let combined = if offset_8 + size_bytes > 8 {
                    let high_base = Operand::simplified(
                        operand_add(
                            base.clone(),
                            self.ctx.constant(offset_rest.wrapping_add(8)),
                        )
                    );
                    let high = self.memory.get(i.intern(high_base.clone()))
                        .map(|x| i.operand(x))
                        .unwrap_or_else(|| mem64(high_base));
                    let high = operand_lsh(
                        high,
                        self.ctx.constant((0x40 - offset_8 * 8) as u64),
                    );
                    operand_or(low, high)
                } else {
                    low
                };
                let masked = match mem.size {
                    MemAccessSize::Mem8 => operand_and(combined, self.ctx.const_ff()),
                    MemAccessSize::Mem16 => operand_and(combined, self.ctx.const_ffff()),
                    MemAccessSize::Mem32 => operand_and(combined, self.ctx.const_ffffffff()),
                    MemAccessSize::Mem64 => combined,
                };
                return masked;
            }
        }
        self.memory.get(i.intern(address.clone()))
            .map(|interned| i.operand(interned))
            .unwrap_or_else(|| mem_variable_rc(mem.size, address))
    }

    /// Checks cached/caches `reg & ff` masks.
    fn try_resolve_partial_register(
        &mut self,
        left: &Rc<Operand>,
        right: &Rc<Operand>,
        interner: &mut InternMap,
    ) -> Option<Rc<Operand>> {
        use crate::operand::operand_helpers::*;
        let (const_op, c, other) = match left.ty {
            OperandType::Constant(c) => (left, c, right),
            _ => match right.ty {
                OperandType::Constant(c) => (right, c, left),
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
            let interned = match self.cached_low_registers.get_32(reg.0) {
                InternedOperand(0) => {
                    let op = operand_and(
                        interner.operand(self.registers[reg.0 as usize]),
                        self.ctx.const_ffffffff(),
                    );
                    let interned = interner.intern(Operand::simplified(op));
                    self.cached_low_registers.set_32(reg.0, interned);
                    interned
                }
                x => x,
            };
            let op = interner.operand(interned);
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
        match value.ty {
            OperandType::Register(reg) => {
                interner.operand(self.registers[reg.0 as usize])
            }
            OperandType::Xmm(reg, word) => {
                interner.operand(self.xmm_registers[reg as usize].word(word))
            }
            OperandType::Fpu(_) => value.clone(),
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
            OperandType::ArithmeticF32(ref op) => {
                let ty = OperandType::ArithmeticF32(ArithOperand {
                    ty: op.ty,
                    left: self.resolve(&op.left, interner),
                    right: self.resolve(&op.right, interner),
                });
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::Constant(_) => value.clone(),
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
        self.memory.reverse_lookup(interned).map(|x| mem64(i.operand(x)))
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

    fn contains_undefined(oper: &Operand) -> bool {
        oper.iter().any(|x| match x.ty {
            OperandType::Undefined(_) => true,
            _ => false,
        })
    }

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
        !check_flags_eq(&old.flags, &new.flags) ||
        !check_memory_eq(&old.memory, &new.memory, interner) ||
        merged_ljec.as_ref().map(|x| *x != old.unresolved_constraint).unwrap_or(false) || (
            old.resolved_constraint.is_some() &&
            old.resolved_constraint != new.resolved_constraint
        );
    if changed {
        let mut registers = [InternedOperand(0); 16];
        let mut cached_low_registers = CachedLowRegisters::new();
        let mut xmm_registers = [XmmOperand(
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
        ); 16];
        let mut flags = Flags {
            zero: InternedOperand(0),
            carry: InternedOperand(0),
            overflow: InternedOperand(0),
            sign: InternedOperand(0),
            parity: InternedOperand(0),
            direction: InternedOperand(0),
        };
        {
            for i in 0..16 {
                registers[i] = merge(old.registers[i], new.registers[i], interner);
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
            binary: old.binary,
        })
    } else {
        None
    }
}
