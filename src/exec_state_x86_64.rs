use std::rc::Rc;

use byteorder::{ReadBytesExt, LE};

use crate::analysis;
use crate::disasm::{Disassembler64, DestOperand, Operation};
use crate::exec_state::{Constraint, InternMap, InternedOperand, Memory, XmmOperand};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
};
use crate::{BinaryFile, VirtualAddress64};

#[derive(Debug)]
pub struct ExecutionState<'a> {
    pub registers: [InternedOperand; 0x10],
    pub xmm_registers: [XmmOperand; 0x10],
    pub flags: Flags,
    pub memory: Memory,
    last_jump_extra_constraint: Option<Constraint>,
    ctx: &'a OperandContext,
    code_sections: Vec<&'a crate::BinarySection<VirtualAddress64>>,
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

/// Handles regular &'mut Operand assign for regs,
/// and the more complicated one for memory
enum Destination<'a> {
    Oper(&'a mut InternedOperand),
    Register32(&'a mut InternedOperand),
    Register16(&'a mut InternedOperand),
    Register8High(&'a mut InternedOperand),
    Register8Low(&'a mut InternedOperand),
    Pair(&'a mut InternedOperand, &'a mut InternedOperand),
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
                *o = intern_map.intern(Operand::simplified(
                    operand_and(value, ctx.const_ffffffff())
                ));
            }
            Destination::Register16(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(
                    operand_or64(
                        operand_and64(old, ctx.constant64(0xffff_ffff_ffff_0000)),
                        operand_and64(value, ctx.const_ffff()),
                    )
                ));
            }
            Destination::Register8High(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(operand_or64(
                    operand_and64(old, ctx.constant64(0xffff_ffff_ffff_00ff)),
                    operand_and64(operand_lsh(value, ctx.const_8()), ctx.const_ff00()),
                )));
            }
            Destination::Register8Low(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(
                    operand_or64(
                        operand_and64(old, ctx.constant64(0xffff_ffff_ffff_ff00)),
                        operand_and64(value, ctx.const_ff()),
                    )
                ));
            }
            Destination::Pair(high, low) => {
                let (val_high, val_low) = Operand::pair(&value);
                *high = intern_map.intern(Operand::simplified(val_high));
                *low = intern_map.intern(Operand::simplified(val_low));
            }
            Destination::Memory(mem, addr, size) => {
                if let Some((base, offset)) = Operand::const_offset(&addr, ctx) {
                    // TODO const_offset64
                    let offset = offset as u64;
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
                            operand_add64(base.clone(), ctx.constant64(offset_rest))
                        );
                        let low_i = intern_map.intern(low_base.clone());
                        let low_old = mem.get(low_i)
                            .map(|x| intern_map.operand(x))
                            .unwrap_or_else(|| mem64(low_base));

                        let mask_low = offset_8 * 8;
                        let mask_high = (mask_low + size_bits).min(0x40);
                        let mask = !0 >> mask_low << mask_low <<
                            (0x40 - mask_high) >> (0x40 - mask_high);
                        let low_value = Operand::simplified(operand_or64(
                            operand_and64(
                                operand_lsh64(
                                    value.clone(),
                                    ctx.constant(8 * offset_8 as u32),
                                ),
                                ctx.constant64(mask),
                            ),
                            operand_and64(
                                low_old,
                                ctx.constant64(!mask),
                            ),
                        ));
                        mem.set(low_i, intern_map.intern(low_value));
                        let needs_high = mask_low + size_bits > 0x40;
                        if needs_high {
                            let high_base = Operand::simplified(
                                operand_add64(
                                    base.clone(),
                                    ctx.constant64(offset_rest.wrapping_add(8)),
                                )
                            );
                            let high_i = intern_map.intern(high_base.clone());
                            let high_old = mem.get(high_i)
                                .map(|x| intern_map.operand(x))
                                .unwrap_or_else(|| mem64(high_base));
                            let mask = !0 >> (0x40 - (mask_low + size_bits - 0x40));
                            let high_value = Operand::simplified(operand_or64(
                                operand_and64(
                                    operand_rsh64(
                                        value,
                                        ctx.constant(0x80 - 8 * offset_8 as u32),
                                    ),
                                    ctx.constant64(mask),
                                ),
                                operand_and64(
                                    high_old,
                                    ctx.constant64(!mask),
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

    fn add_extra_constraint(&mut self, constraint: Constraint) {
        self.last_jump_extra_constraint = Some(constraint);
    }

    fn move_to(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap) {
        let ctx = self.ctx();
        let resolved = self.resolve(&value, i);
        let dest = self.get_dest_invalidate_constraints(dest, i);
        dest.set(resolved, i, ctx);
    }

    fn ctx(&self) -> &'a OperandContext {
        self.ctx
    }

    fn resolve(&self, operand: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand> {
        self.resolve(operand, i)
    }

    fn update(&mut self, operation: &Operation, intern_map: &mut InternMap) {
        self.update(operation, intern_map)
    }

    fn resolve_apply_constraints(&self, op: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand> {
        let mut stack_op;
        let mut op_ref = if op.is_simplified() {
            op
        } else {
            stack_op = Operand::simplified(op.clone());
            &stack_op
        };
        if let Some(ref constraint) = self.last_jump_extra_constraint {
            stack_op = constraint.apply_to(op_ref);
            op_ref = &stack_op;
        }
        self.resolve(op_ref, i)
    }

    fn unresolve(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        self.unresolve(val, i)
    }

    fn unresolve_memory(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>> {
        self.unresolve_memory(val, i)
    }

    fn merge_states(old: &Self, new: &Self, i: &mut InternMap) -> Option<Self> {
        merge_states(old, new, i)
    }

    fn apply_call(&mut self, ret: VirtualAddress64, i: &mut InternMap) {
        use crate::operand::operand_helpers::*;
        let ctx = self.ctx;
        let rsp = ctx.register64(4);
        self.move_to(
            &DestOperand::from_oper(&rsp),
            operand_sub64(rsp.clone(), ctx.const_8()),
            i,
        );
        self.move_to(
            &DestOperand::from_oper(&mem64(rsp)),
            ctx.constant64(ret.0),
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
            registers[i] = interner.intern(ctx.register64(i as u8));
            xmm_registers[i] = XmmOperand::initial(i as u8, interner);
        }
        ExecutionState {
            registers,
            xmm_registers,
            flags: Flags::initial(ctx, interner),
            memory: Memory::new(),
            last_jump_extra_constraint: None,
            ctx,
            code_sections: Vec::new(),
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile<VirtualAddress64>,
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
                // TODO: disassembly module should give the simplified values always
                let value = Operand::simplified(value.clone());
                if let Some(cond) = cond {
                    let cond = Operand::simplified(cond.clone());
                    match self.resolve_apply_constraints(&cond, intern_map).if_constant() {
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
                let left_res = self.resolve(&Rc::new(left.clone().into()), intern_map);
                let right_res = self.resolve(&Rc::new(right.clone().into()), intern_map);
                self.get_dest_invalidate_constraints(&left, intern_map)
                    .set(right_res, intern_map, ctx);
                self.get_dest_invalidate_constraints(&right, intern_map)
                    .set(left_res, intern_map, ctx);
            }
            Operation::Call(_) => {
                let mut ids = intern_map.many_undef(ctx, 13);
                self.last_jump_extra_constraint = None;
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
        }
    }

    /// Makes all of memory undefined
    pub fn clear_memory(&mut self) {
        self.memory = Memory::new();
    }

    pub fn move_no_resolve(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap) {
        let value = Operand::simplified(value);
        let ctx = self.ctx;
        let dest = self.get_dest_invalidate_constraints(dest, i);
        dest.set(value, i, ctx);
    }

    fn get_dest_invalidate_constraints<'s>(
        &'s mut self,
        dest: &DestOperand,
        interner: &mut InternMap,
    ) -> Destination<'s> {
        self.last_jump_extra_constraint = match self.last_jump_extra_constraint {
            Some(ref s) => s.invalidate_dest_operand(dest),
            None => None,
        };
        self.get_dest(dest, interner)
    }

    fn get_dest(&mut self, dest: &DestOperand, intern_map: &mut InternMap) -> Destination {
        match *dest {
            DestOperand::Register64(reg) => {
                Destination::Oper(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register(reg) => {
                Destination::Register32(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register16(reg) => {
                Destination::Register16(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register8High(reg) => {
                Destination::Register8High(&mut self.registers[reg.0 as usize])
            }
            DestOperand::Register8Low(reg) => {
                Destination::Register8Low(&mut self.registers[reg.0 as usize])
            }
            DestOperand::PairEdxEax => {
                let (eax, rest) = self.registers.split_first_mut().unwrap();
                let edx = &mut rest[1];
                Destination::Pair(edx, eax)
            }
            DestOperand::Fpu(_) => Destination::Nop,
            DestOperand::Xmm(reg, word) => {
                Destination::Oper(self.xmm_registers[reg as usize].word_mut(word))
            }
            DestOperand::Flag(flag) => Destination::Oper(match flag {
                Flag::Zero => &mut self.flags.zero,
                Flag::Carry => &mut self.flags.carry,
                Flag::Overflow => &mut self.flags.overflow,
                Flag::Sign => &mut self.flags.sign,
                Flag::Parity => &mut self.flags.parity,
                Flag::Direction => &mut self.flags.direction,
            }),
            DestOperand::Memory(ref mem) => {
                let address = self.resolve(&mem.address, intern_map);
                Destination::Memory(&mut self.memory, address, mem.size)
            }
        }
    }

    fn resolve_mem(&self, mem: &MemAccess, i: &mut InternMap) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;

        let address = self.resolve(&mem.address, i);
        let size_bytes = match mem.size {
            MemAccessSize::Mem8 => 1,
            MemAccessSize::Mem16 => 2,
            MemAccessSize::Mem32 => 4,
            MemAccessSize::Mem64 => 8,
        };
        if let Some(c) = address.if_constant64() {
            // Simplify constants stored in code section (constant switch jumps etc)
            if let Some(end) = c.checked_add(size_bytes as u64) {
                let section = self.code_sections.iter().find(|s| {
                    s.virtual_address.0 <= c &&
                        s.virtual_address.0 + s.virtual_size as u64 >= end
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
                    return self.ctx.constant64(val);
                }
            }
        }

        // Use 8-aligned addresses if there's a const offset
        if let Some((base, offset)) = Operand::const_offset(&address, self.ctx) {
            // TODO u64 Operand::const_offset
            let offset = offset as u64;
            let offset_8 = offset as u32 & 7;
            let offset_rest = offset & !7;
            if offset_8 != 0 {
                let low_base = Operand::simplified(
                    operand_add64(base.clone(), self.ctx.constant64(offset_rest))
                );
                let low = self.memory.get(i.intern(low_base.clone()))
                    .map(|x| i.operand(x))
                    .unwrap_or_else(|| mem64(low_base));
                let low = operand_rsh64(low, self.ctx.constant(offset_8 * 8));
                let combined = if offset_8 + size_bytes > 8 {
                    let high_base = Operand::simplified(
                        operand_add64(
                            base.clone(),
                            self.ctx.constant64(offset_rest.wrapping_add(8)),
                        )
                    );
                    let high = self.memory.get(i.intern(high_base.clone()))
                        .map(|x| i.operand(x))
                        .unwrap_or_else(|| mem64(high_base));
                    let high = operand_lsh64(high, self.ctx.constant(0x40 - offset_8 * 8));
                    operand_or64(low, high)
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

    fn resolve_no_simplify(&self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;

        match value.ty {
            OperandType::Register64(reg) => {
                interner.operand(self.registers[reg.0 as usize])
            }
            OperandType::Register(reg) => {
                operand_and(
                    interner.operand(self.registers[reg.0 as usize]),
                    self.ctx.const_ffffffff(),
                )
            }
            OperandType::Register16(reg) => {
                operand_and(
                    interner.operand(self.registers[reg.0 as usize]),
                    self.ctx.const_ffff(),
                )
            }
            OperandType::Register8High(reg) => {
                operand_rsh(
                    operand_and(
                        interner.operand(self.registers[reg.0 as usize]),
                        self.ctx.const_ff00(),
                    ),
                    self.ctx.const_8(),
                )
            },
            OperandType::Register8Low(reg) => {
                operand_and(
                    interner.operand(self.registers[reg.0 as usize]),
                    self.ctx.const_ff(),
                )
            },
            OperandType::Pair(ref high, ref low) => {
                pair(self.resolve(&high, interner), self.resolve(&low, interner))
            }
            OperandType::Xmm(reg, word) => {
                interner.operand(self.xmm_registers[reg as usize].word(word))
            }
            OperandType::Fpu(_) => value.clone(),
            OperandType::Flag(flag) => interner.operand(match flag {
                Flag::Zero => self.flags.zero,
                Flag::Carry => self.flags.carry,
                Flag::Overflow => self.flags.overflow,
                Flag::Sign => self.flags.sign,
                Flag::Parity => self.flags.parity,
                Flag::Direction => self.flags.direction,
            }).clone(),
            OperandType::Arithmetic(ref op) => {
                let ty = OperandType::Arithmetic(ArithOperand {
                    ty: op.ty,
                    left: self.resolve(&op.left, interner),
                    right: self.resolve(&op.right, interner),
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

    pub fn resolve(&self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
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

impl<'a> Clone for ExecutionState<'a> {
    fn clone(&self) -> ExecutionState<'a> {
        ExecutionState {
            registers: [
                self.registers[0],
                self.registers[1],
                self.registers[2],
                self.registers[3],
                self.registers[4],
                self.registers[5],
                self.registers[6],
                self.registers[7],
                self.registers[8],
                self.registers[9],
                self.registers[10],
                self.registers[11],
                self.registers[12],
                self.registers[13],
                self.registers[14],
                self.registers[15],
            ],
            xmm_registers: [
                self.xmm_registers[0],
                self.xmm_registers[1],
                self.xmm_registers[2],
                self.xmm_registers[3],
                self.xmm_registers[4],
                self.xmm_registers[5],
                self.xmm_registers[6],
                self.xmm_registers[7],
                self.xmm_registers[8],
                self.xmm_registers[9],
                self.xmm_registers[10],
                self.xmm_registers[11],
                self.xmm_registers[12],
                self.xmm_registers[13],
                self.xmm_registers[14],
                self.xmm_registers[15],
            ],
            flags: self.flags.clone(),
            memory: self.memory.clone(),
            last_jump_extra_constraint: self.last_jump_extra_constraint.clone(),
            ctx: self.ctx,
            code_sections: self.code_sections.clone(),
        }
    }
}

/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
pub fn merge_states<'a: 'r, 'r>(
    old: &'r ExecutionState<'a>,
    new: &'r ExecutionState<'a>,
    interner: &mut InternMap,
) -> Option<ExecutionState<'a>> {
    use crate::operand::operand_helpers::*;

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

    let merge = |a: InternedOperand, b: InternedOperand, i: &mut InternMap| -> InternedOperand {
        match a == b {
            true => a,
            false => i.new_undef(old.ctx),
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

    let merged_ljec = if old.last_jump_extra_constraint != new.last_jump_extra_constraint {
        let old_lje = &old.last_jump_extra_constraint;
        let new_lje = &new.last_jump_extra_constraint;
        Some(match (old_lje, new_lje, old, new) {
            // If one state has no constraint but matches the constrait of the other
            // state, the constraint should be kept on merge.
            (&None, &Some(ref con), state, _) | (&Some(ref con), &None, _, state) => {
                // As long as we're working with flags, limiting to lowest bit
                // allows simplifying cases like (undef | 1)
                let lowest_bit = operand_and(old.ctx.const_1(), con.0.clone());
                match state.resolve_apply_constraints(&lowest_bit, interner).if_constant() {
                    Some(1) => Some(con.clone()),
                    _ => None,
                }
            }
            _ => None,
        })
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
        merged_ljec.as_ref().map(|x| *x != old.last_jump_extra_constraint).unwrap_or(false);
    if changed {
        let mut registers = [InternedOperand(0); 16];
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
            xmm_registers,
            flags,
            memory: old.memory.merge(&new.memory, interner, old.ctx),
            last_jump_extra_constraint: merged_ljec.unwrap_or_else(|| {
                // They were same, just use one from old
                old.last_jump_extra_constraint.clone()
            }),
            ctx: old.ctx,
            code_sections: old.code_sections.clone(),
        })
    } else {
        None
    }
}