use std::collections::{hash_map, HashMap};
use std::mem;
use std::rc::Rc;

use byteorder::{ReadBytesExt, LE};
use fxhash::FxBuildHasher;

use crate::analysis;
use crate::disasm::{Disassembler32, DestOperand, Operation};
use crate::exec_state::{Constraint, InternMap, InternedOperand};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::operand::{
    ArithOperand, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
};
use crate::{BinaryFile, VirtualAddress};

impl<'a> ExecutionStateTrait<'a> for ExecutionState<'a> {
    type VirtualAddress = VirtualAddress;
    type Disassembler = Disassembler32<'a>;

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
}

#[derive(Debug)]
pub struct ExecutionState<'a> {
    pub registers: [InternedOperand; 0x8],
    pub xmm_registers: [XmmOperand; 0x8],
    pub flags: Flags,
    pub memory: Memory,
    last_jump_extra_constraint: Option<Constraint>,
    ctx: &'a OperandContext,
    code_sections: Vec<&'a crate::BinarySection<VirtualAddress>>,
}

#[derive(Debug, Clone, Copy)]
pub struct XmmOperand(InternedOperand, InternedOperand, InternedOperand, InternedOperand);

impl XmmOperand {
    fn initial(register: u8, interner: &mut InternMap) -> XmmOperand {
        XmmOperand(
            interner.intern(Rc::new(Operand::new_xmm(register, 0))),
            interner.intern(Rc::new(Operand::new_xmm(register, 1))),
            interner.intern(Rc::new(Operand::new_xmm(register, 2))),
            interner.intern(Rc::new(Operand::new_xmm(register, 3))),
        )
    }

    fn word(&self, idx: u8) -> InternedOperand {
        match idx {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            _ => unreachable!(),
        }
    }

    fn word_mut(&mut self, idx: u8) -> &mut InternedOperand {
        match idx {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            3 => &mut self.3,
            _ => unreachable!(),
        }
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
            ],
            flags: self.flags.clone(),
            memory: self.memory.clone(),
            last_jump_extra_constraint: self.last_jump_extra_constraint.clone(),
            ctx: self.ctx,
            code_sections: self.code_sections.clone(),
        }
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

#[derive(Debug, Clone)]
pub struct Memory {
    map: HashMap<InternedOperand, InternedOperand, FxBuildHasher>,
    /// Optimization for cases where memory gets large.
    /// The existing mapping can be moved to Rc, where cloning it is effectively free.
    immutable: Option<Rc<Memory>>,
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

struct MemoryIterUntilImm<'a> {
    iter: hash_map::Iter<'a, InternedOperand, InternedOperand>,
    immutable: &'a Option<Rc<Memory>>,
    limit: Option<&'a Memory>,
    in_immutable: bool,
}

pub struct MemoryIter<'a>(MemoryIterUntilImm<'a>);

impl Memory {
    pub fn new() -> Memory {
        Memory {
            map: HashMap::with_hasher(Default::default()),
            immutable: None,
        }
    }

    pub fn get(&self, address: InternedOperand) -> Option<InternedOperand> {
        let op = self.map.get(&address).cloned()
            .or_else(|| self.immutable.as_ref().and_then(|x| x.get(address)));
        op
    }

    pub fn set(&mut self, address: InternedOperand, value: InternedOperand) {
        self.map.insert(address, value);
    }

    /// Returns iterator yielding the interned operands.
    pub fn iter_interned(&mut self) -> MemoryIter {
        MemoryIter(self.iter_until_immutable(None))
    }

    pub fn len(&self) -> usize {
        self.map.len() + self.immutable.as_ref().map(|x| x.len()).unwrap_or(0)
    }

    /// The bool is true if the value is in immutable map
    fn get_with_immutable_info(
        &self,
        address: InternedOperand
    ) -> Option<(InternedOperand, bool)> {
        let op = self.map.get(&address).map(|&x| (x, false))
            .or_else(|| {
                self.immutable.as_ref().and_then(|x| x.get(address)).map(|x| (x, true))
            });
        op
    }

    /// Iterates until the immutable block.
    /// Only ref-equality is considered, not deep-eq.
    fn iter_until_immutable<'a>(&'a self, limit: Option<&'a Memory>) -> MemoryIterUntilImm<'a> {
        MemoryIterUntilImm {
            iter: self.map.iter(),
            immutable: &self.immutable,
            limit,
            in_immutable: false,
        }
    }

    fn common_immutable<'a>(&'a self, other: &'a Memory) -> Option<&'a Memory> {
        self.immutable.as_ref().and_then(|i| {
            let a = &**i as *const Memory;
            let mut pos = Some(other);
            while let Some(o) = pos {
                if o as *const Memory == a {
                    return pos;
                }
                pos = o.immutable.as_ref().map(|x| &**x);
            }
            i.common_immutable(other)
        })
    }

    fn has_before_immutable(&self, address: InternedOperand, immutable: &Memory) -> bool {
        if self as *const Memory == immutable as *const Memory {
            false
        } else if self.map.contains_key(&address) {
            true
        } else {
            self.immutable.as_ref()
                .map(|x| x.has_before_immutable(address, immutable))
                .unwrap_or(false)
        }
    }

    pub fn maybe_convert_immutable(&mut self) {
        if self.map.len() >= 64 {
            let map = mem::replace(&mut self.map, HashMap::with_hasher(Default::default()));
            let old_immutable = self.immutable.take();
            self.immutable = Some(Rc::new(Memory {
                map,
                immutable: old_immutable,
            }));
        }
    }

    /// Does a value -> key lookup (Finds an address containing value)
    pub fn reverse_lookup(&self, value: InternedOperand) -> Option<InternedOperand> {
        for (&key, &val) in &self.map {
            if value == val {
                return Some(key);
            }
        }
        if let Some(ref i) = self.immutable {
            i.reverse_lookup(value)
        } else {
            None
        }
    }
}

impl<'a> Iterator for MemoryIterUntilImm<'a> {
    /// The bool tells if we're at immutable parts of the calling operand or not
    type Item = (&'a InternedOperand, &'a InternedOperand, bool);
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(s) => Some((s.0, s.1, self.in_immutable)),
            None => {
                let at_limit = self.immutable.as_ref().map(|x| &**x as *const Memory) ==
                    self.limit.map(|x| x as *const Memory);
                if at_limit {
                    return None;
                }
                match *self.immutable {
                    Some(ref i) => {
                        *self = i.iter_until_immutable(self.limit);
                        self.in_immutable = true;
                        self.next()
                    }
                    None => None,
                }
            }
        }
    }
}

impl<'a> Iterator for MemoryIter<'a> {
    type Item = (InternedOperand, InternedOperand);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(&key, &val, _)| (key, val))
    }
}

impl<'a> Destination<'a> {
    fn set(self, value: Rc<Operand>, intern_map: &mut InternMap, ctx: &OperandContext) {
        use crate::operand::operand_helpers::*;
        match self {
            Destination::Oper(o) => {
                *o = intern_map.intern(Operand::simplified(value));
            }
            Destination::Register16(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(
                    operand_or(
                        operand_and(old, ctx.const_ffff0000()),
                        operand_and(value, ctx.const_ffff()),
                    )
                ));
            }
            Destination::Register8High(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(operand_or(
                    operand_and(old, ctx.const_ffff00ff()),
                    operand_and(operand_lsh(value, ctx.const_8()), ctx.const_ff00()),
                )));
            }
            Destination::Register8Low(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(
                    operand_or(
                        operand_and(old, ctx.const_ffffff00()),
                        operand_and(value, ctx.const_ff()),
                    )
                ));
            }
            Destination::Pair(high, low) => {
                let (val_high, val_low) = Operand::pair(&value);
                *high = intern_map.intern(Operand::simplified(val_high));
                *low = intern_map.intern(Operand::simplified(val_low));
            }
            Destination::Memory(mem, addr, size) => {
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
        let mut xmm_registers = [XmmOperand(
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
            InternedOperand(0),
        ); 8];
        for i in 0..8 {
            registers[i] = interner.intern(ctx.register(i as u8));
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
                let mut ids = intern_map.many_undef(ctx, 9);
                self.last_jump_extra_constraint = None;
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
            Operation::Special(_) => {
            }
        };
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
            DestOperand::Register(reg) => Destination::Oper(&mut self.registers[reg.0 as usize]),
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

    fn resolve_no_simplify(&self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;

        match value.ty {
            OperandType::Register(reg) | OperandType::Register64(reg) => {
                interner.operand(self.registers[reg.0 as usize])
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
        self.memory.reverse_lookup(interned).map(|x| mem32(i.operand(x)))
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
    let merge_memory = |a: &Memory, b: &Memory, i: &mut InternMap| -> Memory {
        if a.len() == 0 {
            return b.clone();
        }
        if b.len() == 0 {
            return a.clone();
        }
        let mut result = HashMap::with_hasher(Default::default());
        let imm_eq = a.immutable.as_ref().map(|x| &**x as *const Memory) ==
            b.immutable.as_ref().map(|x| &**x as *const Memory);
        if imm_eq {
            // Allows just checking a.map.iter() instead of a.iter()
            for (&key, &a_val) in &a.map {
                if let Some((b_val, is_imm)) = b.get_with_immutable_info(key) {
                    match a_val == b_val {
                        true => {
                            if !is_imm {
                                result.insert(key, a_val);
                            }
                        }
                        false => {
                            if is_imm {
                                result.insert(key, i.new_undef(old.ctx));
                            }
                        }
                    }
                }
            }
            Memory {
                map: result,
                immutable: a.immutable.clone(),
            }
        } else {
            // Not inserting undefined addresses here, as missing entry is effectively undefined.
            // Should be fine?
            //
            // a's immutable map is used as base, so one which exist there don't get inserted to
            // result, but if it has ones that should become undefined, the undefined has to be
            // inserted to the result instead.
            let common = a.common_immutable(b);
            for (&key, &b_val, _is_imm) in b.iter_until_immutable(common) {
                if let Some((a_val, is_imm)) = a.get_with_immutable_info(key) {
                    match a_val == b_val {
                        true => {
                            if !is_imm {
                                result.insert(key, a_val);
                            }
                        }
                        false => {
                            if is_imm {
                                result.insert(key, i.new_undef(old.ctx));
                            }
                        }
                    }
                }
            }
            // The result contains now anything that was in b's unique branch of the memory and
            // matched something in a.
            // However, it is possible that for address A, the common branch contains X and b's
            // unique branch has nothing, in which case b's value for A is X.
            // if a's unique branch has something for A, then A's value may be X (if the values
            // are equal), or undefined.
            //
            // Only checking both unique branches instead of walking through the common branch
            // ends up being faster in cases where the memory grows large.
            //
            // If there is no common branch, we don't have to do anything else.
            if let Some(common) = common {
                for (&key, &a_val, is_imm) in a.iter_until_immutable(Some(common)) {
                    if !b.has_before_immutable(key, common) {
                        if let Some(b_val) = common.get(key) {
                            match a_val == b_val {
                                true => {
                                    if !is_imm {
                                        result.insert(key, a_val);
                                    }
                                }
                                false => {
                                    if is_imm {
                                        result.insert(key, i.new_undef(old.ctx));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Memory {
                map: result,
                immutable: a.immutable.clone(),
            }
        }
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
        let mut registers = [InternedOperand(0); 8];
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
            memory: merge_memory(&old.memory, &new.memory, interner),
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
    let state_a = state_a.assume_jump_flag(&sign_eq_overflow_flag, true, &mut i);
    state_b.move_to(&DestOperand::from_oper(&ctx.flag_o()), constval(1), &mut i);
    state_b.move_to(&DestOperand::from_oper(&ctx.flag_s()), constval(1), &mut i);
    let merged = merge_states(&state_b, &state_a, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);
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
    let merged = merge_states(&state_b, &state_a, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);
    // Should also happen other way, though then state_a must have something that is converted
    // to undef.
    let merged = merge_states(&state_a, &state_b, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);

    state_a.move_to(&DestOperand::from_oper(&ctx.flag_c()), constval(1), &mut i);
    let merged = merge_states(&state_a, &state_b, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);
}
