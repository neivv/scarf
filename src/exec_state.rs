use std::collections::{hash_map, HashMap};
use std::hash::BuildHasherDefault;
use std::fmt;
use std::ops::{Add};
use std::rc::Rc;

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
    fn add_extra_constraint(&mut self, constraint: Constraint);
    fn update(&mut self, operation: &Operation, i: &mut InternMap);
    fn move_to(&mut self, dest: &DestOperand, value: Rc<Operand>, i: &mut InternMap);
    fn ctx(&self) -> &'a OperandContext;
    fn resolve(&self, operand: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand>;
    fn resolve_apply_constraints(&self, operand: &Rc<Operand>, i: &mut InternMap) -> Rc<Operand>;
    fn unresolve(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>>;
    fn unresolve_memory(&self, val: &Rc<Operand>, i: &mut InternMap) -> Option<Rc<Operand>>;
    fn initial_state(
        operand_ctx: &'a OperandContext,
        binary: &'a crate::BinaryFile<Self::VirtualAddress>,
        interner: &mut InternMap,
    ) -> Self;
    fn merge_states(old: &Self, new: &Self, i: &mut InternMap) -> Option<Self>;

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

    /// Creates either Arithmetic or Arithmetic64 based on VirtualAddress size
    fn operand_arith_word(ty: ArithOpType, left: Rc<Operand>, right: Rc<Operand>) -> Rc<Operand> {
        use crate::operand_helpers::*;
        if <Self::VirtualAddress as VirtualAddress>::SIZE == 4 {
            operand_arith(ty, left, right)
        } else {
            operand_arith64(ty, left, right)
        }
    }

    /// Returns state with the condition assumed to be true/false.
    fn assume_jump_flag(
        &self,
        condition: &Rc<Operand>,
        jump: bool,
        i: &mut InternMap,
    ) -> Self {
        use crate::operand::ArithOpType::*;
        use crate::operand::operand_helpers::*;

        match condition.ty {
            OperandType::Arithmetic(ref arith) => {
                let left = &arith.left;
                let right = &arith.right;
                match arith.ty {
                    Equal => {
                        Operand::either(left, right, |x| x.if_constant().filter(|&c| c == 0))
                            .and_then(|(_, other)| {
                                let (other, flag_state) = other.if_arithmetic_eq()
                                    .and_then(|(l, r)| {
                                        Operand::either(l, r, |x| x.if_constant())
                                            .map(|(c, other)| (other, if c == 0 { jump } else { !jump }))
                                    })
                                    .unwrap_or_else(|| (other, !jump));
                                match other.ty {
                                    OperandType::Flag(flag) => {
                                        let mut state = self.clone();
                                        let constant = self.ctx().constant(flag_state as u32);
                                        state.move_to(&DestOperand::Flag(flag), constant, i);
                                        Some(state)
                                    }
                                    _ => None,
                                }
                            })
                            .unwrap_or_else(|| {
                                let mut state = self.clone();
                                let cond = match jump {
                                    true => condition.clone(),
                                    false => Operand::simplified(
                                        operand_logical_not(condition.clone())
                                    ),
                                };
                                state.add_extra_constraint(Constraint::new(cond));
                                state
                            })
                    }
                    Or => {
                        if jump {
                            let mut state = self.clone();
                            let cond = Operand::simplified(operand_or(
                                left.clone(),
                                right.clone()
                            ));
                            state.add_extra_constraint(Constraint::new(cond));
                            state
                        } else {
                            let mut state = self.clone();
                            let cond = Operand::simplified(operand_and(
                                operand_logical_not(left.clone()),
                                operand_logical_not(right.clone()),
                            ));
                            state.add_extra_constraint(Constraint::new(cond));
                            state
                        }
                    }
                    And => {
                        if jump {
                            let mut state = self.clone();
                            let cond = Operand::simplified(operand_and(
                                left.clone(),
                                right.clone(),
                            ));
                            state.add_extra_constraint(Constraint::new(cond));
                            state
                        } else {
                            let mut state = self.clone();
                            let cond = Operand::simplified(operand_or(
                                operand_logical_not(left.clone()),
                                operand_logical_not(right.clone())
                            ));
                            state.add_extra_constraint(Constraint::new(cond));
                            state
                        }
                    }
                    _ => self.clone(),
                }
            }
            _ => self.clone(),
        }
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

    fn find_relocs(
        _file: &crate::BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<Self::VirtualAddress>, crate::Error> {
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
/// (so constraint == constval(1))
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Constraint(pub Rc<Operand>);

impl Constraint {
    pub fn new(o: Rc<Operand>) -> Constraint {
        Constraint(Operand::simplified(o))
    }

    pub(crate) fn invalidate_dest_operand(&self, dest: &DestOperand) -> Option<Constraint> {
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

/// Splits the constraint at ands that can be applied separately
/// Also can handle logical nots
fn apply_constraint_split(
    constraint: &Rc<Operand>,
    val: &Rc<Operand>,
    with: bool,
) -> Rc<Operand>{
    use crate::operand::operand_helpers::*;
    match constraint.ty {
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
            let new = apply_constraint_split(&arith.left, val, with);
            apply_constraint_split(&arith.right, &new, with)
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Equal => {
            let (l, r) = (&arith.left, &arith.right);
            let other = Operand::either(l, r, |x| x.if_constant().filter(|&c| c == 0))
                .map(|x| x.1);
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

/// Trait for disassembling instructions
pub trait Disassembler<'disasm_bytes> {
    type VirtualAddress: VirtualAddress;
    fn new(buf: &'disasm_bytes [u8], pos: usize, address: Self::VirtualAddress) -> Self;
    fn next(&mut self, ctx: &OperandContext) ->
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
                    let new = InternedOperand(self.map.len() as u32);
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
                hash_map::Entry::Occupied(e) => self.map[e.get().0 as usize].clone(),
                hash_map::Entry::Vacant(e) => {
                    let new = InternedOperand(self.map.len() as u32);
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
            self.map[val.0 as usize].clone()
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

#[test]
fn apply_constraint() {
    use crate::operand::operand_helpers::*;
    let ctx = crate::operand::OperandContext::new();
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
    let val = operand_or(
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
    let val = constraint.apply_to(&val);
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
