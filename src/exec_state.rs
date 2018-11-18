use std::collections::{hash_map, HashMap};
use std::fmt;
use std::hash::BuildHasherDefault;
use std::mem;
use std::rc::Rc;

use byteorder::{ReadBytesExt, LE};
use fxhash::FxBuildHasher;

use disasm::{DestOperand, Operation};
use operand::{
    self, ArithOpType, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType,
    OperandDummyHasher,
};

/// The constraint is assumed to be something that can be substituted with 1 if met
/// (so constraint == constval(1))
#[derive(Debug, Clone, Eq, PartialEq)]
struct Constraint(Rc<Operand>);

impl Constraint {
    fn new(o: Rc<Operand>) -> Constraint {
        Constraint(o)
    }

    fn invalidate_dest_operand(&self, dest: &DestOperand) -> Option<Constraint> {
        match *dest {
            DestOperand::Register(reg) | DestOperand::Register16(reg) |
                DestOperand::Register8High(reg) | DestOperand::Register8Low(reg) =>
            {
                remove_matching_ands(&self.0, &mut |x| *x == OperandType::Register(reg))
            }
            DestOperand::PairEdxEax => None,
            DestOperand::Xmm(_, _) => {
                None
            }
            DestOperand::Flag(flag) => {
                remove_matching_ands(&self.0, &mut |x| *x == OperandType::Flag(flag))
            },
            DestOperand::Memory(_) => {
                None
            }
        }.map(Constraint::new)
    }

    fn apply_to(&self, oper: &mut Rc<Operand>) {
        let new = apply_constraint_split(&self.0, oper, true);
        *oper = Operand::simplified(new);
    }
}

/// Constraint invalidation helper
fn remove_matching_ands<F>(oper: &Rc<Operand>, fun: &mut F) -> Option<Rc<Operand>>
where F: FnMut(&OperandType) -> bool,
{
     match oper.ty {
        OperandType::Arithmetic(ArithOpType::And(ref l, ref r)) => {
            // Only going to try partially invalidate cases with logical ands
            if l.relevant_bits() != (0..1) || r.relevant_bits() != (0..1) {
                match oper.iter().any(|x| fun(&x.ty)) {
                    true => None,
                    false => Some(oper.clone()),
                }
            } else {
                let left = remove_matching_ands(l, fun);
                let right = remove_matching_ands(r, fun);
                match (left, right) {
                    (None, None) => None,
                    (Some(x), None) | (None, Some(x)) => Some(x),
                    (Some(l), Some(r)) => {
                        let op_ty = OperandType::Arithmetic(ArithOpType::And(l, r));
                        Some(Operand::new_not_simplified_rc(op_ty))
                    }
                }
            }
        },
        _ => match oper.iter().any(|x| fun(&x.ty)) {
            true => None,
            false => Some(oper.clone()),
        },
    }
}

/// Splits the constraint at ands that can be applied separately
/// Also can handle logical nots
fn apply_constraint_split(
    constraint: &Rc<Operand>,
    val: &Rc<Operand>,
    with: bool,
) -> Rc<Operand>{
    use operand::operand_helpers::*;
    match constraint.ty {
        OperandType::Arithmetic(ArithOpType::And(ref l, ref r)) => {
            let new = apply_constraint_split(l, val, with);
            apply_constraint_split(r, &new, with)
        }
        OperandType::Arithmetic(ArithOpType::Equal(ref l, ref r)) => {
            let other = match (&l.ty, &r.ty) {
                (&OperandType::Constant(0), _) => Some(r),
                (_, &OperandType::Constant(0)) => Some(l),
                _ => None
            };
            if let Some(other) = other {
                apply_constraint_split(other, val, !with)
            } else {
                // TODO OperandContext
                let subst_val = constval(if with { 1 } else { 0 });
                Operand::substitute(val, constraint, &subst_val)
            }
        }
        _ => {
            let subst_val = constval(if with { 1 } else { 0 });
            Operand::substitute(val, constraint, &subst_val)
        }
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
    code_sections: Vec<&'a crate::BinarySection>,
}

#[derive(Debug, Clone)]
pub struct XmmOperand(InternedOperand, InternedOperand, InternedOperand, InternedOperand);

impl XmmOperand {
    fn undefined(ctx: &OperandContext, interner: &mut InternMap) -> XmmOperand {
        XmmOperand(
            interner.new_undef(ctx),
            interner.new_undef(ctx),
            interner.new_undef(ctx),
            interner.new_undef(ctx),
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
                self.xmm_registers[0].clone(),
                self.xmm_registers[1].clone(),
                self.xmm_registers[2].clone(),
                self.xmm_registers[3].clone(),
                self.xmm_registers[4].clone(),
                self.xmm_registers[5].clone(),
                self.xmm_registers[6].clone(),
                self.xmm_registers[7].clone(),
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
        match op {
            Some(s) => match s.is_undefined() {
                true => None,
                false => Some(s),
            },
            None => None,
        }
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
        match op {
            Some(s) => match s.0.is_undefined() {
                true => None,
                false => Some(s),
            },
            None => None,
        }
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
        use operand::operand_helpers::*;
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
                if let Some((base, offset)) = Operand::const_offset(&addr, ctx) {
                    let offset_4 = offset & 3;
                    let offset_rest = offset & !3;
                    if offset_4 != 0 {
                        let size_bits = match size {
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
                });
                mem.set(addr, intern_map.intern(value));
            }
        }
    }
}

/// Since reanalyzing branches can give equivalent-but-separate `Operand` objects,
/// a single function's `Operand`s are interned to get guaranteed fast comparisions.
///
/// Generally anything that gets set as ExecutionState field gets interned here, other
/// intermediate values do not.
///
/// Additionally, Undefined(n) is always InternedOperand(!n)
#[derive(Clone)]
pub struct InternMap {
    pub map: Vec<Rc<Operand>>,
    reverse: HashMap<Rc<Operand>, InternedOperand, BuildHasherDefault<OperandDummyHasher>>,
}

impl fmt::Debug for InternMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InternMap({} entries)", self.map.len())
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InternedOperand(u32);

impl InternedOperand {
    pub fn is_undefined(&self) -> bool {
        self.0 > 0x8000_0000
    }
}

impl InternMap {
    pub fn new() -> InternMap {
        InternMap {
            map: Vec::new(),
            reverse: HashMap::with_hasher(Default::default()),
        }
    }

    pub fn intern(&mut self, val: Rc<Operand>) -> InternedOperand {
        if let OperandType::Undefined(id) = val.ty {
            InternedOperand(!id.0)
        } else {
            match self.reverse.entry(val.clone()) {
                hash_map::Entry::Occupied(e) => *e.get(),
                hash_map::Entry::Vacant(e) => {
                    let new = InternedOperand(self.map.len() as u32);
                    e.insert(new);
                    self.map.push(val);
                    new
                }
            }
        }
    }

    /// Faster than `self.intern(ctx.undefined_rc())`
    pub fn new_undef(&self, ctx: &OperandContext) -> InternedOperand {
        InternedOperand(!ctx.new_undefined_id())
    }

    pub fn operand(&self, val: InternedOperand) -> Rc<Operand> {
        if val.is_undefined() {
            let ty = OperandType::Undefined(operand::UndefinedId(!val.0));
            Operand::new_simplified_rc(ty)
        } else {
            self.map[val.0 as usize].clone()
        }
    }
}

impl Flags {
    fn undefined(ctx: &OperandContext, interner: &mut InternMap) -> Flags {
        let undef = |interner: &mut InternMap| interner.new_undef(ctx);
        Flags {
            zero: undef(interner),
            carry: undef(interner),
            overflow: undef(interner),
            sign: undef(interner),
            parity: undef(interner),
            direction: interner.intern(ctx.const_0()),
        }
    }
}


impl<'a> ExecutionState<'a> {
    pub fn new<'b>(
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        ExecutionState {
            registers: [
                interner.intern(ctx.register(0)),
                interner.intern(ctx.register(1)),
                interner.intern(ctx.register(2)),
                interner.intern(ctx.register(3)),
                interner.intern(ctx.register(4)),
                interner.intern(ctx.register(5)),
                interner.intern(ctx.register(6)),
                interner.intern(ctx.register(7)),
            ],
            xmm_registers: [
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
            ],
            flags: Flags::undefined(ctx, interner),
            memory: Memory::new(),
            last_jump_extra_constraint: None,
            ctx,
            code_sections: Vec::new(),
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile,
        ctx: &'b OperandContext,
        interner: &mut InternMap,
    ) -> ExecutionState<'b> {
        ExecutionState {
            registers: [
                interner.intern(ctx.register(0)),
                interner.intern(ctx.register(1)),
                interner.intern(ctx.register(2)),
                interner.intern(ctx.register(3)),
                interner.intern(ctx.register(4)),
                interner.intern(ctx.register(5)),
                interner.intern(ctx.register(6)),
                interner.intern(ctx.register(7)),
            ],
            xmm_registers: [
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
                XmmOperand::undefined(ctx, interner),
            ],
            flags: Flags::undefined(ctx, interner),
            memory: Memory::new(),
            last_jump_extra_constraint: None,
            ctx,
            code_sections: binary.code_sections().collect(),
        }
    }

    pub fn update(&mut self, operation: Operation, intern_map: &mut InternMap) {
        match operation {
            Operation::Move(dest, value, cond) => {
                // TODO: disassembly module should give the simplified values always
                let value = Operand::simplified(value);
                if let Some(cond) = cond {
                    let cond = Operand::simplified(cond);
                    match self.try_resolve_const(&cond, intern_map) {
                        Some(0) => (),
                        Some(_) => {
                            let resolved = self.resolve(&value, intern_map);
                            let dest = self.get_dest_invalidate_constraints(&dest, intern_map);
                            dest.set(resolved, intern_map, self.ctx);
                        }
                        None => {
                            self.get_dest_invalidate_constraints(&dest, intern_map)
                                .set(self.ctx.undefined_rc(), intern_map, self.ctx)
                        }
                    }
                } else {
                    let resolved = self.resolve(&value, intern_map);
                    let dest = self.get_dest_invalidate_constraints(&dest, intern_map);
                    dest.set(resolved, intern_map, self.ctx);
                }
            }
            Operation::Swap(left, right) => {
                // Shouldn't have to clone
                let left_res = self.resolve(&Rc::new(left.clone().into()), intern_map);
                let right_res = self.resolve(&Rc::new(right.clone().into()), intern_map);
                self.get_dest_invalidate_constraints(&left, intern_map)
                    .set(right_res, intern_map, self.ctx);
                self.get_dest_invalidate_constraints(&right, intern_map)
                    .set(left_res, intern_map, self.ctx);
            }
            Operation::Call(_) => {
                self.registers[0] = intern_map.new_undef(self.ctx);
                self.registers[1] = intern_map.new_undef(self.ctx);
                self.registers[2] = intern_map.new_undef(self.ctx);
                self.registers[4] = intern_map.new_undef(self.ctx);
                self.flags = Flags::undefined(self.ctx, intern_map);
            }
            Operation::Jump { .. } => {
            }
            Operation::Return(_) => {
            }
            Operation::Special(_) => {
            }
        };
    }

    /// Equivalent to `self.update(Operation::Move(dest, val, None), i)`
    pub fn move_to(&mut self, dest: DestOperand, val: Rc<Operand>, i: &mut InternMap) {
        self.update(Operation::Move(dest, val, None), i);
    }

    fn get_dest_invalidate_constraints(
        &mut self,
        dest: &DestOperand,
        interner: &mut InternMap,
    ) -> Destination {
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
                let (edx, _) = rest.split_first_mut().unwrap();
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

    fn resolve_arith(&self, op: &ArithOpType, i: &mut InternMap) -> ArithOpType {
        use operand::ArithOpType::*;
        match *op {
            Add(ref l, ref r) => {
                Add(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Sub(ref l, ref r) => {
                Sub(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Mul(ref l, ref r) => {
                Mul(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            SignedMul(ref l, ref r) => {
                SignedMul(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Div(ref l, ref r) => {
                Div(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Modulo(ref l, ref r) => {
                Modulo(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            And(ref l, ref r) => {
                And(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Or(ref l, ref r) => {
                Or(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Xor(ref l, ref r) => {
                Xor(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Lsh(ref l, ref r) => {
                Lsh(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Rsh(ref l, ref r) => {
                Rsh(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            RotateLeft(ref l, ref r) => {
                RotateLeft(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Equal(ref l, ref r) => {
                Equal(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            Not(ref x) => Not(self.resolve_no_simplify(x, i)),
            Parity(ref x) => Parity(self.resolve_no_simplify(x, i)),
            GreaterThan(ref l, ref r) => {
                GreaterThan(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
            GreaterThanSigned(ref l, ref r) => {
                GreaterThanSigned(self.resolve_no_simplify(l, i), self.resolve_no_simplify(r, i))
            }
        }
    }

    fn resolve_mem(&self, mem: &MemAccess, i: &mut InternMap) -> Rc<Operand> {
        use operand::operand_helpers::*;

        let address = self.resolve(&mem.address, i);
        let size_bytes = match mem.size {
            MemAccessSize::Mem8 => 1,
            MemAccessSize::Mem16 => 2,
            MemAccessSize::Mem32 => 4,
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
                };
                return masked;
            }
        }
        self.memory.get(i.intern(address.clone()))
            .map(|interned| i.operand(interned))
            .unwrap_or_else(|| mem_variable_rc(mem.size, address))
    }

    fn resolve_no_simplify(&self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
        use operand::operand_helpers::*;

        match value.ty {
            OperandType::Register(reg) => interner.operand(self.registers[reg.0 as usize]),
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
                let ty = OperandType::Arithmetic(self.resolve_arith(op, interner));
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::ArithmeticHigh(ref op) => {
                let ty = OperandType::ArithmeticHigh(self.resolve_arith(op, interner));
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::Constant(_) => value.clone(),
            OperandType::Memory(ref mem) => {
                self.resolve_mem(mem, interner)
            }
            OperandType::Undefined(_) => value.clone(),
        }
    }

    pub fn resolve(&self, value: &Rc<Operand>, interner: &mut InternMap) -> Rc<Operand> {
        Operand::simplified({
            let x = self.resolve_no_simplify(value, interner);
            x
        })
    }

    pub fn try_resolve_const(&self, condition: &Rc<Operand>, i: &mut InternMap) -> Option<u32> {
        let mut condition = condition.clone();
        if let Some(ref constraint) = self.last_jump_extra_constraint {
            constraint.apply_to(&mut condition);
        }
        self.resolve(&condition, i).if_constant()
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

    /// Returns state with the condition assumed to be true/false
    pub fn assume_jump_flag(
        &self,
        condition: &Operand,
        jump: bool,
        intern_map: &mut InternMap,
    ) -> ExecutionState<'a> {
        use operand::ArithOpType::*;
        use operand::operand_helpers::*;

        match (jump, &condition.ty) {
            (_, &OperandType::Arithmetic(Equal(ref left, ref right))) => {
                let mut state = self.clone();
                let (flag, flag_state) = match (&left.ty, &right.ty) {
                    (&OperandType::Flag(f), &OperandType::Constant(1)) |
                        (&OperandType::Constant(1), &OperandType::Flag(f)) => (f, jump as u32),
                    (&OperandType::Flag(f), &OperandType::Constant(0)) |
                        (&OperandType::Constant(0), &OperandType::Flag(f)) => (f, !jump as u32),
                    _ => {
                        let cond = match jump {
                            true => condition.clone().into(),
                            false => Operand::simplified(
                                operand_logical_not(condition.clone().into())
                            ),
                        };
                        state.last_jump_extra_constraint = Some(Constraint::new(cond));
                        return state;
                    }
                };
                let flag_operand = self.ctx.flag(flag);
                state.get_dest(&DestOperand::from_oper(&flag_operand), intern_map)
                    .set(self.ctx.constant(flag_state), intern_map, self.ctx);
                state
            }
            (false, &OperandType::Arithmetic(Or(ref left, ref right))) => {
                let mut state = self.clone();
                let cond = Operand::simplified(operand_and(
                    operand_logical_not(left.clone()),
                    operand_logical_not(right.clone()),
                ));
                state.last_jump_extra_constraint = Some(Constraint::new(cond));
                state
            }
            (true, &OperandType::Arithmetic(Or(ref left, ref right))) => {
                let mut state = self.clone();
                let cond = Operand::simplified(operand_or(
                    left.clone(),
                    right.clone()
                ));
                state.last_jump_extra_constraint = Some(Constraint::new(cond));
                state
            }
            (true, &OperandType::Arithmetic(And(ref left, ref right))) => {
                let mut state = self.clone();
                let cond = Operand::simplified(operand_and(
                    left.clone(),
                    right.clone(),
                ));
                state.last_jump_extra_constraint = Some(Constraint::new(cond));
                state
            }
            (false, &OperandType::Arithmetic(And(ref left, ref right))) => {
                let mut state = self.clone();
                let cond = Operand::simplified(operand_or(
                    operand_logical_not(left.clone()),
                    operand_logical_not(right.clone())
                ));
                state.last_jump_extra_constraint = Some(Constraint::new(cond));
                state
            }
            (_, _) => self.clone(),
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
    use operand::operand_helpers::*;

    let check_eq = |a: InternedOperand, b: InternedOperand, interner: &mut InternMap| {
        if a == b {
            true
        } else {
            let a = interner.operand(a);
            match a.ty {
                OperandType::Undefined(_) => true,
                _ => false,
            }
        }
    };
    let check_xmm_eq = |a: &XmmOperand, b: &XmmOperand, interner: &mut InternMap| {
        check_eq(a.0, b.0, interner) &&
            check_eq(a.1, b.1, interner) &&
            check_eq(a.2, b.2, interner) &&
            check_eq(a.3, b.3, interner)
    };
    let check_flags_eq = |a: &Flags, b: &Flags, interner: &mut InternMap| {
        check_eq(a.zero, b.zero, interner) &&
            check_eq(a.carry, b.carry, interner) &&
            check_eq(a.overflow, b.overflow, interner) &&
            check_eq(a.sign, b.sign, interner) &&
            check_eq(a.parity, b.parity, interner) &&
            check_eq(a.direction, b.direction, interner)
    };
    let check_memory_eq = |a: &Memory, b: &Memory, interner: &mut InternMap| {
        a.map.iter().all(|(&key, val)| {
            let oper = interner.operand(key);
            match contains_undefined(&oper) {
                true => true,
                false => match b.get(key) {
                    Some(b_val) => check_eq(*val, b_val, interner),
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
                match state.try_resolve_const(&lowest_bit, interner) {
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
            .any(|(&a, &b)| !check_eq(a, b, interner)) ||
        old.xmm_registers.iter().zip(new.xmm_registers.iter())
            .any(|(a, b)| !check_xmm_eq(a, b, interner)) ||
        !check_flags_eq(&old.flags, &new.flags, interner) ||
        !check_memory_eq(&old.memory, &new.memory, interner) ||
        merged_ljec.as_ref().map(|x| *x != old.last_jump_extra_constraint).unwrap_or(false);
    if changed {
        Some(ExecutionState {
            registers: [
                merge(old.registers[0], new.registers[0], interner),
                merge(old.registers[1], new.registers[1], interner),
                merge(old.registers[2], new.registers[2], interner),
                merge(old.registers[3], new.registers[3], interner),
                merge(old.registers[4], new.registers[4], interner),
                merge(old.registers[5], new.registers[5], interner),
                merge(old.registers[6], new.registers[6], interner),
                merge(old.registers[7], new.registers[7], interner),
            ],
            xmm_registers: [
                merge_xmm(&old.xmm_registers[0], &new.xmm_registers[0], interner),
                merge_xmm(&old.xmm_registers[1], &new.xmm_registers[1], interner),
                merge_xmm(&old.xmm_registers[2], &new.xmm_registers[2], interner),
                merge_xmm(&old.xmm_registers[3], &new.xmm_registers[3], interner),
                merge_xmm(&old.xmm_registers[4], &new.xmm_registers[4], interner),
                merge_xmm(&old.xmm_registers[5], &new.xmm_registers[5], interner),
                merge_xmm(&old.xmm_registers[6], &new.xmm_registers[6], interner),
                merge_xmm(&old.xmm_registers[7], &new.xmm_registers[7], interner),
            ],
            flags: Flags {
                zero: merge(old.flags.zero, new.flags.zero, interner),
                carry: merge(old.flags.carry, new.flags.carry, interner),
                overflow: merge(old.flags.overflow, new.flags.overflow, interner),
                sign: merge(old.flags.sign, new.flags.sign, interner),
                parity: merge(old.flags.parity, new.flags.parity, interner),
                direction: merge(old.flags.direction, new.flags.direction, interner),
            },
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
    use operand::operand_helpers::*;
    let mut i = InternMap::new();
    let ctx = ::operand::OperandContext::new();
    let state_a = ExecutionState::new(&ctx, &mut i);
    let mut state_b = ExecutionState::new(&ctx, &mut i);
    let sign_eq_overflow_flag = Operand::simplified(operand_eq(
        ctx.flag_o(),
        ctx.flag_s(),
    ));
    let state_a = state_a.assume_jump_flag(&sign_eq_overflow_flag, true, &mut i);
    state_b.move_to(DestOperand::from_oper(&ctx.flag_o()), constval(1), &mut i);
    state_b.move_to(DestOperand::from_oper(&ctx.flag_s()), constval(1), &mut i);
    let merged = merge_states(&state_b, &state_a, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);
}

#[test]
fn merge_state_constraints_or() {
    use operand::operand_helpers::*;
    let mut i = InternMap::new();
    let ctx = ::operand::OperandContext::new();
    let state_a = ExecutionState::new(&ctx, &mut i);
    let mut state_b = ExecutionState::new(&ctx, &mut i);
    let sign_or_overflow_flag = Operand::simplified(operand_or(
        ctx.flag_o(),
        ctx.flag_s(),
    ));
    let mut state_a = state_a.assume_jump_flag(&sign_or_overflow_flag, true, &mut i);
    state_b.move_to(DestOperand::from_oper(&ctx.flag_s()), constval(1), &mut i);
    let merged = merge_states(&state_b, &state_a, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);
    // Should also happen other way, though then state_a must have something that is converted
    // to undef.
    let merged = merge_states(&state_a, &state_b, &mut i);
    assert!(merged.is_none());

    state_a.move_to(DestOperand::from_oper(&ctx.flag_c()), constval(1), &mut i);
    let merged = merge_states(&state_a, &state_b, &mut i).unwrap();
    assert!(merged.last_jump_extra_constraint.is_some());
    assert_eq!(merged.last_jump_extra_constraint, state_a.last_jump_extra_constraint);
}

#[test]
fn apply_constraint() {
    use operand::operand_helpers::*;
    let ctx = ::operand::OperandContext::new();
    let constraint = Constraint(operand_eq(
        constval(0),
        operand_eq(
            operand_eq(
                ctx.flag_z(),
                constval(0),
            ),
            constval(0),
        ),
    ));
    let mut val = operand_or(
        operand_eq(
            operand_eq(
                ctx.flag_c(),
                constval(0),
            ),
            constval(0),
        ),
        operand_eq(
            operand_eq(
                ctx.flag_z(),
                constval(0),
            ),
            constval(0),
        ),
    );
    let old = val.clone();
    constraint.apply_to(&mut val);
    let eq = operand_eq(
        operand_eq(
            ctx.flag_c(),
            constval(0),
        ),
        constval(0),
    );
    assert_ne!(Operand::simplified(val.clone()), Operand::simplified(old));
    assert_eq!(Operand::simplified(val), Operand::simplified(eq));
}
