use std::collections::{hash_map, HashMap};
use std::hash::BuildHasherDefault;
use std::fmt;
use std::mem;
use std::ops::{Add};
use std::rc::Rc;

use fxhash::FxBuildHasher;

use crate::analysis;
use crate::disasm::{self, DestOperand, Instruction, Operation};
use crate::operand::{
    self, ArithOpType, ArithOperand, Operand, OperandType, OperandContext, OperandDummyHasher
};

/// A trait that does (most of) arch-specific state handling.
///
/// ExecutionState contains the CPU state that is simulated, so registers, flags.
pub trait ExecutionState<'a> : Clone {
    type VirtualAddress: VirtualAddress;
    // Technically ExecState shouldn't need to be linked to disassembly code,
    // but I didn't want an additional trait to sit between user-defined AnalysisState and
    // ExecutionState, so ExecutionState shall contain this detail as well.
    type Disassembler: Disassembler<'a, VirtualAddress = Self::VirtualAddress>;

    /// Bit of abstraction leak, but the memory structure is implemented as an partially
    /// immutable hashmap to keep clones not getting out of hand. This function is used to
    /// tell memory that it may be cloned soon, so the latest changes may be made
    /// immutable-shared if necessary.
    fn maybe_convert_memory_immutable(&mut self);
    /// Adds an additonal assumption that can't be represented by setting registers/etc.
    /// Resolved constraints are useful limiting possible values a variable can have
    /// (`value_limits`)
    fn add_resolved_constraint(&mut self, constraint: Constraint);
    /// Adds an additonal assumption that can't be represented by setting registers/etc.
    /// Unresolved constraints are useful for knowing that a jump chain such as `jg` followed by
    /// `jle` ends up always jumping at `jle`.
    fn add_unresolved_constraint(&mut self, constraint: Constraint);
    fn update(&mut self, operation: &Operation, i: &mut InternMap);
    fn move_to(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap);
    fn move_resolved(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap);
    fn ctx(&self) -> &'a OperandContext;
    fn resolve(&mut self, operand: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand>;
    fn resolve_apply_constraints(
        &mut self,
        operand: &Rc<Operand>,
        i: &mut InternMap,
    ) -> Rc<Operand>;
    fn unresolve(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>>;
    fn unresolve_memory(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>>;
    fn initial_state(
        operand_ctx: &'a OperandContext,
        binary: &'a crate::BinaryFile<Self::VirtualAddress>,
        interner: &mut InternMap,
    ) -> Self;
    fn merge_states(old: &mut Self, new: &mut Self, i: &mut InternMap) -> Option<Self>;

    /// Updates states as if the call instruction was executed (Push return address to stack)
    ///
    /// A separate function as calls are usually just stepped over.
    fn apply_call(&mut self, ret: Self::VirtualAddress, i: &mut InternMap);

    /// Creates an Mem[addr] with MemAccessSize of VirtualAddress size
    fn operand_mem_word(address: Rc<Operand>) -> Rc<Operand> {
        use crate::operand_helpers::*;
        if <Self::VirtualAddress as VirtualAddress>::SIZE == 4 {
            mem32(address)
        } else {
            mem64(address)
        }
    }

    /// Returns state with the condition assumed to be true/false.
    fn assume_jump_flag(
        &self,
        condition: &Rc<Operand>,
        jump: bool,
        i: &mut InternMap,
    ) -> Self {

        let ctx = self.ctx();
        match condition.ty {
            OperandType::Arithmetic(ref arith) => {
                let left = &arith.left;
                let right = &arith.right;
                match arith.ty {
                    ArithOpType::Equal => {
                        let mut state = self.clone();
                        let unresolved_cond = match jump {
                            true => condition.clone(),
                            false => ctx.eq_const(&condition, 0)
                        };
                        let resolved_cond = state.resolve(&unresolved_cond, i);
                        state.add_resolved_constraint(Constraint::new(resolved_cond));
                        let assignable_flag =
                            Operand::either(left, right, |x| {
                                x.if_constant().filter(|&c| c == 0)
                            })
                            .map(|(_, other)| {
                                other.if_arithmetic_eq()
                                    .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                                    .map(|(c, other)| (other, if c == 0 { jump } else { !jump }))
                                    .unwrap_or_else(|| (other, !jump))
                            })
                            .and_then(|(other, flag_state)| match other.ty {
                                OperandType::Flag(f) => Some((f, flag_state)),
                                _ => None,
                            });
                        if let Some((flag, flag_state)) = assignable_flag {
                            let constant = self.ctx().constant(flag_state as u64);
                            state.move_to(&DestOperand::Flag(flag), constant, i);
                        } else {
                            state.add_unresolved_constraint(Constraint::new(unresolved_cond));
                        }
                        state
                    }
                    ArithOpType::Or => {
                        if jump {
                            let mut state = self.clone();
                            let unresolved_cond = ctx.or(left, right);
                            let cond = state.resolve(&unresolved_cond, i);
                            state.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            state.add_resolved_constraint(Constraint::new(cond));
                            state
                        } else {
                            let mut state = self.clone();
                            let unresolved_cond = ctx.and(
                                &ctx.eq_const(left, 0),
                                &ctx.eq_const(right, 0),
                            );
                            let cond = state.resolve(&unresolved_cond, i);
                            state.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            state.add_resolved_constraint(Constraint::new(cond));
                            state
                        }
                    }
                    ArithOpType::And => {
                        if jump {
                            let mut state = self.clone();
                            let unresolved_cond = ctx.and(left, right);
                            let cond = state.resolve(&unresolved_cond, i);
                            state.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            state.add_resolved_constraint(Constraint::new(cond));
                            state
                        } else {
                            let mut state = self.clone();
                            let unresolved_cond = ctx.or(
                                &ctx.eq_const(left, 0),
                                &ctx.eq_const(right, 0),
                            );
                            let cond = state.resolve(&unresolved_cond, i);
                            state.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            state.add_resolved_constraint(Constraint::new(cond));
                            state
                        }
                    }
                    _ => self.clone(),
                }
            }
            _ => self.clone(),
        }
    }

    /// Returns smallest and largest (inclusive) value a *resolved* operand can have
    /// (Mainly meant to use extra constraint information)
    fn value_limits(&self, _value: &Rc<Operand>) -> (u64, u64) {
        (0, u64::max_value())
    }

    // Analysis functions, default to no-op
    fn find_functions_with_callers(_file: &crate::BinaryFile<Self::VirtualAddress>)
        -> Vec<analysis::FuncCallPair<Self::VirtualAddress>> { Vec::new() }

    fn find_functions_from_calls(
        _code: &[u8],
        _section_base: Self::VirtualAddress,
        _out: &mut Vec<Self::VirtualAddress>
    ) {
    }

    /// Returns function start and end addresses as a relative to base.
    ///
    /// The returned addresses are expected to be sorted and not have overlaps.
    /// (Though it currently trusts that the binary follows PE spec)
    fn function_ranges_from_exception_info(
        _file: &crate::BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<(u32, u32)>, crate::OutOfBounds> {
        Ok(Vec::new())
    }

    fn find_relocs(
        _file: &crate::BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<Self::VirtualAddress>, crate::OutOfBounds> {
        Ok(Vec::new())
    }
}

/// Either `scarf::VirtualAddress` in 32-bit or `scarf::VirtualAddress64` in 64-bit
pub trait VirtualAddress: Eq + PartialEq + Ord + PartialOrd + Copy + Clone + std::hash::Hash +
    fmt::LowerHex + fmt::UpperHex + fmt::Debug + Add<u32, Output = Self> +
    'static
{
    type Inner: fmt::LowerHex + fmt::UpperHex;
    const SIZE: u32;
    fn max_value() -> Self;
    fn inner(self) -> Self::Inner;
    fn from_u64(val: u64) -> Self;
    fn as_u64(self) -> u64;
}

impl VirtualAddress for crate::VirtualAddress {
    type Inner = u32;
    const SIZE: u32 = 4;
    fn max_value() -> Self {
        crate::VirtualAddress(!0)
    }

    fn inner(self) -> Self::Inner {
        self.0
    }

    fn from_u64(val: u64) -> Self {
        crate::VirtualAddress(val as u32)
    }

    fn as_u64(self) -> u64 {
        self.0 as u64
    }
}

impl VirtualAddress for crate::VirtualAddress64 {
    type Inner = u64;
    const SIZE: u32 = 8;
    fn max_value() -> Self {
        crate::VirtualAddress64(!0)
    }

    fn inner(self) -> Self::Inner {
        self.0
    }

    fn from_u64(val: u64) -> Self {
        crate::VirtualAddress64(val)
    }

    fn as_u64(self) -> u64 {
        self.0
    }
}

/// The constraint is assumed to be something that can be substituted with 1 if met
/// (so constraint == constval(1)).
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Constraint(pub Rc<Operand>);

impl Constraint {
    pub fn new(o: Rc<Operand>) -> Constraint {
        Constraint(Operand::simplified(o))
    }

    /// Invalidates any assumptions about memory
    pub(crate) fn invalidate_memory(&mut self) -> ConstraintFullyInvalid {
        let result = remove_matching_ands(&self.0, &mut |x| match x {
            OperandType::Memory(..) => true,
            _ => false,
        });
        match result {
            Some(s) => {
                self.0 = s;
                ConstraintFullyInvalid::No
            }
            None => ConstraintFullyInvalid::Yes,
        }
    }

    /// Invalidates any parts of the constraint that depend on unresolved dest.
    pub(crate) fn invalidate_dest_operand(&self, dest: &DestOperand) -> Option<Constraint> {
        match *dest {
            DestOperand::Register32(reg) | DestOperand::Register16(reg) |
                DestOperand::Register8High(reg) | DestOperand::Register8Low(reg) |
                DestOperand::Register64(reg) =>
            {
                remove_matching_ands(&self.0, &mut |x| *x == OperandType::Register(reg))
            }
            DestOperand::Xmm(_, _) => {
                None
            }
            DestOperand::Fpu(_) => {
                None
            }
            DestOperand::Flag(flag) => {
                remove_matching_ands(&self.0, &mut |x| *x == OperandType::Flag(flag))
            },
            DestOperand::Memory(_) => {
                // Assuming that everything may alias with memory
                remove_matching_ands(&self.0, &mut |x| match x {
                    OperandType::Memory(..) => true,
                    _ => false,
                })
            }
        }.map(Constraint::new)
     }

    pub(crate) fn apply_to(&self, oper: &Rc<Operand>) -> Rc<Operand> {
        let new = apply_constraint_split(&self.0, oper, true);
        Operand::simplified(new)
    }
}

#[derive(Eq, PartialEq, Copy, Clone)]
#[must_use]
pub enum ConstraintFullyInvalid {
    Yes,
    No,
}

/// Constraint invalidation helper
fn remove_matching_ands<F>(oper: &Rc<Operand>, fun: &mut F) -> Option<Rc<Operand>>
where F: FnMut(&OperandType) -> bool,
{
    if let Some((l, r)) = oper.if_arithmetic_and() {
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
                (Some(left), Some(right)) => {
                    let op_ty = OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::And,
                        left,
                        right,
                    });
                    Some(Operand::new_not_simplified_rc(op_ty))
                }
            }
        }
    } else {
        match oper.iter().any(|x| fun(&x.ty)) {
            true => None,
            false => Some(oper.clone()),
        }
    }
}

fn other_if_eq_zero(op: &Rc<Operand>) -> Option<&Rc<Operand>> {
    op.if_arithmetic_eq()
        .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant().filter(|&c| c == 0)))
        .map(|x| x.1)
}

/// Splits the constraint at ands that can be applied separately
/// Also can handle logical nots
fn apply_constraint_split(
    constraint: &Rc<Operand>,
    val: &Rc<Operand>,
    with: bool,
) -> Rc<Operand>{
    use crate::operand::operand_helpers::*;
    match constraint.ty {
        OperandType::Arithmetic(ref arith) if {
            arith.ty == ArithOpType::And &&
                arith.left.relevant_bits() == (0..1) &&
                arith.right.relevant_bits() == (0..1)
        } => {
            let new = apply_constraint_split(&arith.left, val, with);
            apply_constraint_split(&arith.right, &new, with)
        }
        OperandType::Arithmetic(ref arith) if {
            arith.ty == ArithOpType::Or &&
                arith.left.relevant_bits() == (0..1) &&
                arith.right.relevant_bits() == (0..1)
        } => {
            // If constraint is (x == 0) | (y == 0),
            // (x & y) can be replaced with 0
            let x = other_if_eq_zero(&arith.left);
            let y = other_if_eq_zero(&arith.right);
            if let (Some(x), Some(y)) = (x, y) {
                // Check also for (x & y) == 0 (Replaced with 1)
                let (cmp, subst) = other_if_eq_zero(val)
                    .map(|x| (x, 1))
                    .unwrap_or_else(|| (val, 0));
                if let Some((l, r)) = cmp.if_arithmetic_and() {
                    if (l == x && r == y) || (l == y && r == x) {
                        return constval(subst);
                    }
                }
            }
            let subst_val = constval(if with { 1 } else { 0 });
            Operand::substitute(val, constraint, &subst_val)
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Equal => {
            let (l, r) = (&arith.left, &arith.right);
            let other = Operand::either(l, r, |x| x.if_constant().filter(|&c| c == 0))
                .map(|x| x.1)
                .filter(|x| x.relevant_bits() == (0..1));
            if let Some(other) = other {
                apply_constraint_split(other, val, !with)
            } else {
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

/// Helper for ExecutionState::value_limits implementations with constraints
pub(crate) fn value_limits_recurse(constraint: &Rc<Operand>, value: &Rc<Operand>) -> (u64, u64) {
    match constraint.ty {
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let left = value_limits_recurse(&arith.left, value);
                    let right = value_limits_recurse(&arith.right, value);
                    return (left.0.max(right.0), (left.1.min(right.1)));
                }
                ArithOpType::GreaterThan => {
                    // 0 > x and x > u64_max should get simplified to 0
                    if let Some(c) = arith.left.if_constant() {
                        if is_subset(value, &arith.right) {
                            debug_assert!(c != 0);
                            return (0, c.wrapping_sub(1));
                        }
                    }
                    if let Some(c) = arith.right.if_constant() {
                        if is_subset(value, &arith.left) {
                            debug_assert!(c != u64::max_value());
                            return (c.wrapping_add(1), u64::max_value());
                        }
                    }
                }
                _ => (),
            }
        }
        _ => (),
    }
    (0, u64::max_value())
}

/// Returns true if sub == sup & some_const_mask (Eq is also fine)
/// Not really considering constants for this (value_limits_recurse)
fn is_subset(sub: &Rc<Operand>, sup: &Rc<Operand>) -> bool {
    if sub.ty.expr_size() > sup.ty.expr_size() {
        return false;
    }
    match sub.ty {
        OperandType::Memory(ref mem) => match sup.if_memory() {
            Some(mem2) => mem.address == mem2.address,
            None => false,
        }
        _ => sub == sup,
    }
}

/// Trait for disassembling instructions
pub trait Disassembler<'disasm_bytes> {
    type VirtualAddress: VirtualAddress;
    fn new(
        buf: &'disasm_bytes [u8],
        pos: usize,
        address: Self::VirtualAddress,
        // Should this use a separate lifetime for clarity?
        // 'disasm_bytes does still in practice always refer to &BinaryFile as well,
        // so it should be ok.
        ctx: &'disasm_bytes OperandContext,
    ) -> Self;
    fn next(&mut self) ->
        Result<Instruction<Self::VirtualAddress>, disasm::Error>;
    fn address(&self) -> Self::VirtualAddress;
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
pub struct InternedOperand(pub u32);

impl InternedOperand {
    pub fn is_undefined(self) -> bool {
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
                    let new = InternedOperand(self.map.len() as u32 + 1);
                    e.insert(new);
                    self.map.push(val);
                    new
                }
            }
        }
    }

    pub(crate) fn intern_and_get(&mut self, val: Rc<Operand>) -> Rc<Operand> {
        if let OperandType::Undefined(_) = val.ty {
            val
        } else {
            match self.reverse.entry(val.clone()) {
                hash_map::Entry::Occupied(e) => self.map[e.get().0 as usize - 1].clone(),
                hash_map::Entry::Vacant(e) => {
                    let new = InternedOperand(self.map.len() as u32 + 1);
                    e.insert(new);
                    self.map.push(val.clone());
                    val
                }
            }
        }
    }

    /// Faster than `self.intern(ctx.undefined_rc())`
    pub fn new_undef(&self, ctx: &OperandContext) -> InternedOperand {
        InternedOperand(!ctx.new_undefined_id())
    }

    pub(crate) fn many_undef(&self, ctx: &OperandContext, count: u32) -> ManyInternedUndef {
        let pos = ctx.alloc_undefined_ids(count);
        ManyInternedUndef {
            pos,
            limit: pos + count,
        }
    }

    pub fn operand(&self, val: InternedOperand) -> Rc<Operand> {
        if val.is_undefined() {
            let ty = OperandType::Undefined(operand::UndefinedId(!val.0));
            Operand::new_simplified_rc(ty)
        } else {
            self.map[val.0 as usize - 1].clone()
        }
    }
}

pub(crate) struct ManyInternedUndef {
    pos: u32,
    limit: u32,
}

impl ManyInternedUndef {
    pub fn next(&mut self) -> InternedOperand {
        assert!(self.pos < self.limit);
        let val = InternedOperand(!self.pos);
        self.pos += 1;
        val
    }
}

/// Contains memory state as addr -> value hashmap.
/// The ExecutionState is expected to take care of cases where memory is written with one
/// address and part of it is read at offset address, in practice by splitting accesses
/// to be word-sized, and any misaligned accesses become bitwise and-or combinations.
#[derive(Debug, Clone)]
pub struct Memory {
    pub(crate) map: HashMap<InternedOperand, InternedOperand, FxBuildHasher>,
    /// Optimization for cases where memory gets large.
    /// The existing mapping can be moved to Rc, where cloning it is effectively free.
    immutable: Option<Rc<Memory>>,
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
    pub fn iter_interned(&self) -> MemoryIter {
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

    pub fn merge(&self, new: &Memory, i: &mut InternMap, ctx: &OperandContext) -> Memory {
        let a = self;
        let b = new;
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
                                result.insert(key, i.new_undef(ctx));
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
                                result.insert(key, i.new_undef(ctx));
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
                                        result.insert(key, i.new_undef(ctx));
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

#[derive(Debug, Clone, Copy)]
pub struct XmmOperand(
    pub InternedOperand,
    pub InternedOperand,
    pub InternedOperand,
    pub InternedOperand,
);

impl XmmOperand {
    pub fn initial(ctx: &OperandContext, register: u8, interner: &mut InternMap) -> XmmOperand {
        XmmOperand(
            interner.intern(ctx.xmm(register, 0)),
            interner.intern(ctx.xmm(register, 1)),
            interner.intern(ctx.xmm(register, 2)),
            interner.intern(ctx.xmm(register, 3)),
        )
    }

    #[inline]
    pub fn word(&self, idx: u8) -> InternedOperand {
        match idx {
            0 => self.0,
            1 => self.1,
            2 => self.2,
            3 => self.3,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn word_mut(&mut self, idx: u8) -> &mut InternedOperand {
        match idx {
            0 => &mut self.0,
            1 => &mut self.1,
            2 => &mut self.2,
            3 => &mut self.3,
            _ => unreachable!(),
        }
    }
}

#[test]
fn apply_constraint() {
    let ctx = crate::operand::OperandContext::new();
    let constraint = Constraint(ctx.eq_const(
        &ctx.neq_const(
            ctx.flag_z(),
            0,
        ),
        0,
    ));
    let val = ctx.or(
        &ctx.neq_const(
            ctx.flag_c(),
            0,
        ),
        &ctx.neq_const(
            ctx.flag_z(),
            0,
        ),
    );
    let old = val.clone();
    let val = Operand::simplified(constraint.apply_to(&val));
    let eq = ctx.neq_const(
        ctx.flag_c(),
        0,
    );
    assert_ne!(val, old);
    assert_eq!(val, eq);
}

#[test]
fn apply_constraint_non1bit() {
    use crate::operand_helpers::*;
    // This shouldn't cause 0x8000_0000 to be optimized out
    let constraint = operand_eq(
        operand_and(
            constval(0x8000_0000),
            operand_register(1),
        ),
        constval(0),
    );
    let val = operand_and(
        constval(0x8000_0000),
        operand_sub(
            operand_register(1),
            operand_register(2),
        ),
    );
    let val = Operand::simplified(val);
    assert_eq!(Constraint(constraint).apply_to(&val), val);
}

#[test]
fn apply_constraint_or() {
    let ctx = &crate::operand::OperandContext::new();
    let constraint = ctx.or(
        &ctx.neq(
            ctx.flag_o(),
            ctx.flag_s(),
        ),
        &ctx.neq_const(
            ctx.flag_z(),
            0,
        ),
    );
    let val = ctx.eq_const(
        &ctx.and(
            &ctx.eq(
                ctx.flag_o(),
                ctx.flag_s(),
            ),
            &ctx.eq_const(
                ctx.flag_z(),
                0,
            ),
        ),
        0,
    );
    let applied = Constraint(constraint).apply_to(&val);
    let eq = ctx.constant(1);
    assert_eq!(Operand::simplified(applied), Operand::simplified(eq));
}
