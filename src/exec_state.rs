use std::collections::{hash_map, HashMap};
use std::mem;
use std::rc::Rc;

use disasm::{DestOperand, Operation};
use operand::{self, ArithOpType, Flag, MemAccess, Operand, OperandContext, OperandType};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Unimplemented(desc: String) {
            display("Unimplemented {}", desc)
        }
    }
}

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
        apply_constraint_split(&self.0, oper);
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
fn apply_constraint_split(constraint: &Rc<Operand>, val: &mut Rc<Operand>) -> bool {
    use operand::operand_helpers::*;
    match constraint.ty {
        OperandType::Arithmetic(ArithOpType::And(ref l, ref r)) => {
            let mut changed = apply_constraint_split(l, val);
            changed |= apply_constraint_split(r, val);
            changed
        }
        OperandType::Arithmetic(ArithOpType::LogicalNot(ref l)) => {
            let changed = apply_constraint_split(l, val);
            if changed {
                *val = operand_logical_not(val.clone());
            }
            changed
        }
        _ => {
            apply_constraint_subpart(constraint, val)
        }
    }
}

macro_rules! apply_constraint_unary{
    ($constraint:expr, $val:expr, $out:expr, $fun:ident) => {{
        let mut inner = $val.clone();
        let changed = apply_constraint_subpart($constraint, &mut inner);
        if changed {
            *$out = Operand::new_not_simplified_rc(OperandType::Arithmetic($fun(inner)));
        }
        changed
    }}
}

macro_rules! apply_constraint_binary{
    ($constraint:expr, $l:expr, $r:expr, $out:expr, $fun:ident) => {{
        let mut left = $l.clone();
        let mut right = $r.clone();
        let mut changed = apply_constraint_subpart($constraint, &mut left);
        changed |= apply_constraint_subpart($constraint, &mut right);
        if changed {
            *$out = Operand::new_not_simplified_rc(OperandType::Arithmetic($fun(left, right)));
        }
        changed
    }}
}

fn apply_constraint_subpart(constraint: &Rc<Operand>, val: &mut Rc<Operand>) -> bool {
    use operand::ArithOpType::*;
    use operand::operand_helpers::*;

    if val == constraint {
        *val = constval(1);
        true
    } else {
        match val.ty.clone() {
            OperandType::Arithmetic(ref arith) => match *arith {
                Add(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Add),
                Sub(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Sub),
                Mul(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Mul),
                Div(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Div),
                Modulo(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Modulo),
                And(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, And),
                Or(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Or),
                Xor(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Xor),
                Lsh(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Lsh),
                Rsh(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Rsh),
                RotateLeft(ref l, ref r) => {
                    apply_constraint_binary!(constraint, l, r, val, RotateLeft)
                }
                Equal(ref l, ref r) => apply_constraint_binary!(constraint, l, r, val, Equal),
                GreaterThan(ref l, ref r) => {
                    apply_constraint_binary!(constraint, l, r, val, GreaterThan)
                }
                GreaterThanSigned(ref l, ref r) => {
                    apply_constraint_binary!(constraint, l, r, val, GreaterThanSigned)
                }
                SignedMul(ref l, ref r) => {
                    apply_constraint_binary!(constraint, l, r, val, SignedMul)
                }
                Not(ref l) => apply_constraint_unary!(constraint, l, val, Not),
                LogicalNot(ref l) => apply_constraint_unary!(constraint, l, val, LogicalNot),
                Parity(ref l) => apply_constraint_unary!(constraint, l, val, Parity),
            },
            _ => false,
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
                self.registers[0].clone(),
                self.registers[1].clone(),
                self.registers[2].clone(),
                self.registers[3].clone(),
                self.registers[4].clone(),
                self.registers[5].clone(),
                self.registers[6].clone(),
                self.registers[7].clone(),
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
            ctx: self.ctx.clone(),
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
    map: HashMap<InternedOperand, InternedOperand>,
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
    Memory(&'a mut Memory, Rc<Operand>),
}

struct MemoryIterUntilImm<'a> {
    iter: hash_map::Iter<'a, InternedOperand, InternedOperand>,
    immutable: &'a Option<Rc<Memory>>,
    limit: Option<&'a Memory>,
    in_immutable: bool,
}

impl Memory {
    fn new() -> Memory {
        Memory {
            map: HashMap::new(),
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
            let map = mem::replace(&mut self.map, HashMap::new());
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

impl<'a> Destination<'a> {
    fn set(self, value: Rc<Operand>, intern_map: &mut InternMap) -> Result<(), Error> {
        use operand::operand_helpers::*;
        match self {
            Destination::Oper(o) => {
                *o = intern_map.intern(Operand::simplified(value));
            }
            Destination::Register16(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(
                    operand_or(operand_and(old, constval(0xffff0000)), value)
                ));
            }
            Destination::Register8High(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(operand_or(
                    operand_and(old, constval(0xffff00ff)),
                    operand_lsh(value, constval(8))
                )));
            }
            Destination::Register8Low(o) => {
                let old = intern_map.operand(*o);
                *o = intern_map.intern(Operand::simplified(
                    operand_or(operand_and(old, constval(0xffffff00)), value)
                ));
            }
            Destination::Pair(high, low) => {
                let (val_high, val_low) = value.pair().expect("Assigning non-pair to pair");
                *high = intern_map.intern(Operand::simplified(val_high));
                *low = intern_map.intern(Operand::simplified(val_low));
            }
            Destination::Memory(mem, addr) => {
                let addr = intern_map.intern(Operand::simplified(addr));
                mem.map.insert(addr, intern_map.intern(Operand::simplified(value)));
            }
        }
        Ok(())
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
    reverse: HashMap<Rc<Operand>, InternedOperand>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct InternedOperand(u32);

impl InternedOperand {
    pub fn is_undefined(&self) -> bool {
        self.0 > 0x80000000
    }
}

impl InternMap {
    pub fn new() -> InternMap {
        InternMap {
            map: Vec::new(),
            reverse: HashMap::new(),
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
            direction: interner.intern(operand::operand_helpers::constval(0)),
        }
    }
}


impl<'a> ExecutionState<'a> {
    pub fn new<'b>(ctx: &'b OperandContext, interner: &mut InternMap) -> ExecutionState<'b> {
        ExecutionState {
            registers: [
                interner.intern(operand::operand_helpers::operand_register(0)),
                interner.intern(operand::operand_helpers::operand_register(1)),
                interner.intern(operand::operand_helpers::operand_register(2)),
                interner.intern(operand::operand_helpers::operand_register(3)),
                interner.intern(operand::operand_helpers::operand_register(4)),
                interner.intern(operand::operand_helpers::operand_register(5)),
                interner.intern(operand::operand_helpers::operand_register(6)),
                interner.intern(operand::operand_helpers::operand_register(7)),
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
        }
    }

    pub fn update(
        &mut self,
        operation: Operation,
        intern_map: &mut InternMap,
    ) -> Result<(), Error> {
        match operation {
            Operation::Move(dest, value, cond) => {
                // TODO: disassembly module should give the simplified values always
                let value = Operand::simplified(value);
                if let Some(cond) = cond {
                    let cond = Operand::simplified(cond);
                    match self.try_resolve_const(&cond, intern_map) {
                        Some(0) => (),
                        Some(_) => {
                            let resolved = self.resolve(&value, intern_map)?;
                            let dest = self.get_dest_invalidate_constraints(&dest, intern_map)?;
                            dest.set(resolved, intern_map)?;
                        }
                        None => {
                            self.get_dest_invalidate_constraints(&dest, intern_map)?
                                .set(self.ctx.undefined_rc(), intern_map)?
                        }
                    }
                } else {
                    let resolved = self.resolve(&value, intern_map)?;
                    let dest = self.get_dest_invalidate_constraints(&dest, intern_map)?;
                    dest.set(resolved, intern_map)?;
                }
            }
            Operation::Swap(left, right) => {
                // Shouldn't have to clone
                let left_res = self.resolve(&left.clone().into(), intern_map)?;
                let right_res = self.resolve(&right.clone().into(), intern_map)?;
                self.get_dest_invalidate_constraints(&left, intern_map)?
                    .set(right_res, intern_map)?;
                self.get_dest_invalidate_constraints(&right, intern_map)?
                    .set(left_res, intern_map)?;
            }
            Operation::Call(_) => {
                self.registers[0] = intern_map.new_undef(self.ctx);
                self.registers[1] = intern_map.new_undef(self.ctx);
                self.registers[2] = intern_map.new_undef(self.ctx);
                self.registers[4] = intern_map.new_undef(self.ctx);
                self.flags = Flags::undefined(self.ctx, intern_map);
            }
            x => return Err(Error::Unimplemented(format!("{:?}", x))),
        };
        Ok(())
    }

    fn get_dest_invalidate_constraints(
        &mut self,
        dest: &DestOperand,
        interner: &mut InternMap,
    ) -> Result<Destination, Error> {
        self.last_jump_extra_constraint = match self.last_jump_extra_constraint {
            Some(ref s) => s.invalidate_dest_operand(dest),
            None => None,
        };
        self.get_dest(dest, interner)
    }

    fn get_dest(
        &mut self,
        dest: &DestOperand,
        intern_map: &mut InternMap
    ) -> Result<Destination, Error> {
        Ok(match *dest {
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
                let address = self.resolve(&mem.address, intern_map)?;
                Destination::Memory(&mut self.memory, address)
            }
        })
    }

    fn resolve_arith(&self, op: &ArithOpType, i: &mut InternMap) -> Result<ArithOpType, Error> {
        use operand::ArithOpType::*;
        Ok(match *op {
            Add(ref l, ref r) => Add(self.resolve(l, i)?, self.resolve(r, i)?),
            Sub(ref l, ref r) => Sub(self.resolve(l, i)?, self.resolve(r, i)?),
            Mul(ref l, ref r) => Mul(self.resolve(l, i)?, self.resolve(r, i)?),
            SignedMul(ref l, ref r) => SignedMul(self.resolve(l, i)?, self.resolve(r, i)?),
            Div(ref l, ref r) => Div(self.resolve(l, i)?, self.resolve(r, i)?),
            Modulo(ref l, ref r) => Modulo(self.resolve(l, i)?, self.resolve(r, i)?),
            And(ref l, ref r) => And(self.resolve(l, i)?, self.resolve(r, i)?),
            Or(ref l, ref r) => Or(self.resolve(l, i)?, self.resolve(r, i)?),
            Xor(ref l, ref r) => Xor(self.resolve(l, i)?, self.resolve(r, i)?),
            Lsh(ref l, ref r) => Lsh(self.resolve(l, i)?, self.resolve(r, i)?),
            Rsh(ref l, ref r) => Rsh(self.resolve(l, i)?, self.resolve(r, i)?),
            RotateLeft(ref l, ref r) => RotateLeft(self.resolve(l, i)?, self.resolve(r, i)?),
            Equal(ref l, ref r) => Equal(self.resolve(l, i)?, self.resolve(r, i)?),
            Not(ref x) => Not(self.resolve(x, i)?),
            LogicalNot(ref x) => LogicalNot(self.resolve(x, i)?),
            Parity(ref x) => Parity(self.resolve(x, i)?),
            GreaterThan(ref l, ref r) => GreaterThan(self.resolve(l, i)?, self.resolve(r, i)?),
            GreaterThanSigned(ref l, ref r) => {
                GreaterThanSigned(self.resolve(l, i)?, self.resolve(r, i)?)
            }
        })
    }

    pub fn resolve(
        &self,
        value: &Operand,
        interner: &mut InternMap,
    ) -> Result<Rc<Operand>, Error> {
        use operand::operand_helpers::*;

        Ok(match value.ty {
            OperandType::Register(reg) => interner.operand(self.registers[reg.0 as usize]),
            OperandType::Register16(reg) => {
                operand_and(interner.operand(self.registers[reg.0 as usize]), constval(0xffff))
            }
            OperandType::Register8High(reg) => {
                operand_rsh(
                    operand_and(
                        interner.operand(self.registers[reg.0 as usize]),
                        constval(0xff00)
                    ),
                    constval(0x8),
                )
            },
            OperandType::Register8Low(reg) => {
                operand_and(
                    interner.operand(self.registers[reg.0 as usize].clone()),
                    constval(0xff)
                )
            },
            OperandType::Pair(ref high, ref low) => {
                pair(self.resolve(&high, interner)?, self.resolve(&low, interner)?)
            }
            OperandType::Xmm(reg, word) => {
                interner.operand(self.xmm_registers[reg as usize].word(word))
            }
            OperandType::Flag(flag) => interner.operand(match flag {
                Flag::Zero => self.flags.zero.clone(),
                Flag::Carry => self.flags.carry.clone(),
                Flag::Overflow => self.flags.overflow.clone(),
                Flag::Sign => self.flags.sign.clone(),
                Flag::Parity => self.flags.parity.clone(),
                Flag::Direction => self.flags.direction.clone(),
            }).clone(),
            OperandType::Arithmetic(ref op) => {
                let ty = OperandType::Arithmetic(self.resolve_arith(op, interner)?);
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::ArithmeticHigh(ref op) => {
                let ty = OperandType::ArithmeticHigh(self.resolve_arith(op, interner)?);
                Operand::new_not_simplified_rc(ty)
            }
            OperandType::Constant(x) => Operand::new_simplified_rc(OperandType::Constant(x)),
            OperandType::Memory(ref mem) => {
                let address = Operand::simplified(self.resolve(&mem.address, interner)?);
                self.memory.get(interner.intern(address.clone()))
                    .map(|interned| interner.operand(interned))
                    .unwrap_or_else(|| {
                        Operand::new_not_simplified_rc(OperandType::Memory(MemAccess {
                            address: address,
                            size: mem.size,
                        }))
                    })
            }
            OperandType::Undefined(x) => Operand::new_simplified_rc(OperandType::Undefined(x))
        })
    }

    pub fn try_resolve_const(&self, condition: &Operand, i: &mut InternMap) -> Option<u32> {
        let mut condition = Rc::new(condition.clone());
        if let Some(ref constraint) = self.last_jump_extra_constraint {
            constraint.apply_to(&mut condition);
        }
        match self.resolve(&condition, i) {
            Ok(operand) => {
                let simplified = Operand::simplified(operand.clone());
                //trace!("Resolve const {:#?} -> {:#?}", operand, simplified);
                match simplified.ty {
                    OperandType::Constant(c) => Some(c),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Returns state with the condition assumed to be true/false
    pub fn assume_jump_flag(
        &self,
        condition: &Operand,
        jump: bool,
        intern_map: &mut InternMap,
    ) -> Result<ExecutionState<'a>, Error> {
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
                            false => operand_logical_not(condition.clone().into()),
                        };
                        state.last_jump_extra_constraint = Some(Constraint::new(cond));
                        return Ok(state);
                    }
                };
                let flag_operand = Operand::new_simplified_rc(OperandType::Flag(flag));
                state.get_dest(&(*flag_operand).clone().into(), intern_map)?
                    .set(constval(flag_state), intern_map)?;
                Ok(state)
            }
            (false, &OperandType::Arithmetic(Or(ref left, ref right))) => {
                let mut state = self.assume_jump_flag(left, false, intern_map)?;
                state = state.assume_jump_flag(right, false, intern_map)?;
                Ok(state)
            }
            (true, &OperandType::Arithmetic(Or(ref left, ref right))) => {
                let mut state = self.clone();
                let cond = Operand::simplified(operand_or(
                    left.clone(),
                    right.clone()
                ));
                state.last_jump_extra_constraint = Some(Constraint::new(cond));
                Ok(state)
            }
            (true, &OperandType::Arithmetic(And(ref left, ref right))) => {
                let mut state = self.assume_jump_flag(left, true, intern_map)?;
                state = state.assume_jump_flag(right, true, intern_map)?;
                Ok(state)
            }
            (false, &OperandType::Arithmetic(And(ref left, ref right))) => {
                let mut state = self.clone();
                let cond = Operand::simplified(operand_or(
                    operand_logical_not(left.clone()),
                    operand_logical_not(right.clone())
                ));
                state.last_jump_extra_constraint = Some(Constraint::new(cond));
                Ok(state)
            }
            (_, &OperandType::Arithmetic(LogicalNot(ref left))) => {
                self.assume_jump_flag(left, !jump, intern_map)
            }
            (_, _) => Ok(self.clone()),
        }
    }
}

/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
pub fn merge_states<'a>(
    old: &ExecutionState<'a>,
    new: &ExecutionState<'a>,
    interner: &mut InternMap,
) -> Option<ExecutionState<'a>> {
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
        let mut result = HashMap::new();
        let imm_eq = a.immutable.as_ref().map(|x| &**x as *const Memory) ==
            b.immutable.as_ref().map(|x| &**x as *const Memory);
        if imm_eq {
            // Allows just checking a.map.iter() instead of a.iter()
            for (&key, &a_val) in a.map.iter() {
                match b.get_with_immutable_info(key) {
                    Some((b_val, is_imm)) => match a_val == b_val {
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
                    },
                    None => (),
                };
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
                match a.get_with_immutable_info(key) {
                    Some((a_val, is_imm)) => match a_val == b_val {
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
                    },
                    None => (),
                };
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
                        match common.get(key) {
                            Some(b_val) => match a_val == b_val {
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
                            None => (),
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

    let changed =
        old.registers.iter().zip(new.registers.iter())
            .any(|(&a, &b)| !check_eq(a, b, interner)) ||
        old.xmm_registers.iter().zip(new.xmm_registers.iter())
            .any(|(a, b)| !check_xmm_eq(a, b, interner)) ||
        !check_flags_eq(&old.flags, &new.flags, interner) ||
        !check_memory_eq(&old.memory, &new.memory, interner);
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
            last_jump_extra_constraint: {
                let eq = old.last_jump_extra_constraint == new.last_jump_extra_constraint;
                match eq {
                    true => old.last_jump_extra_constraint.clone(),
                    false => None,
                }
            },
            ctx: old.ctx,
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
