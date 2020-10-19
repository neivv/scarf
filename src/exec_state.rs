use std::collections::{hash_map, HashMap};
use std::fmt;
use std::mem;
use std::ops::{Add, Sub};
use std::rc::Rc;

use fxhash::FxBuildHasher;

use crate::analysis;
use crate::disasm::{DestOperand, Instruction, Operation};
use crate::operand::{
    ArithOpType, ArithOperand, Operand, OperandType, OperandCtx, OperandHashByAddress,
    MemAccessSize,
};
use crate::operand::slice_stack::Slice;

/// A trait that does (most of) arch-specific state handling.
///
/// ExecutionState contains the CPU state that is simulated, so registers, flags.
pub trait ExecutionState<'e> : Clone + 'e {
    type VirtualAddress: VirtualAddress;
    // Technically ExecState shouldn't need to be linked to disassembly code,
    // but I didn't want an additional trait to sit between user-defined AnalysisState and
    // ExecutionState, so ExecutionState shall contain this detail as well.
    type Disassembler: Disassembler<'e, VirtualAddress = Self::VirtualAddress>;
    const WORD_SIZE: MemAccessSize;

    /// Bit of abstraction leak, but the memory structure is implemented as an partially
    /// immutable hashmap to keep clones not getting out of hand. This function is used to
    /// tell memory that it may be cloned soon, so the latest changes may be made
    /// immutable-shared if necessary.
    fn maybe_convert_memory_immutable(&mut self, limit: usize);
    /// Adds an additonal assumption that can't be represented by setting registers/etc.
    /// Resolved constraints are useful limiting possible values a variable can have
    /// (`value_limits`)
    fn add_resolved_constraint(&mut self, constraint: Constraint<'e>);
    /// Adds an additonal assumption that can't be represented by setting registers/etc.
    /// Unresolved constraints are useful for knowing that a jump chain such as `jg` followed by
    /// `jle` ends up always jumping at `jle`.
    fn add_unresolved_constraint(&mut self, constraint: Constraint<'e>);
    fn update(&mut self, operation: &Operation<'e>);
    fn move_to(&mut self, dest: &DestOperand<'e>, value: Operand<'e>);
    fn move_resolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>);
    fn set_flags_resolved(&mut self, arith: &ArithOperand<'e>, size: MemAccessSize);
    fn ctx(&self) -> OperandCtx<'e>;
    fn resolve(&mut self, operand: Operand<'e>) -> Operand<'e>;
    fn resolve_apply_constraints(&mut self, operand: Operand<'e>) -> Operand<'e>;
    /// Reads memory for which the `address` is a resolved `Operand`.
    fn read_memory(&mut self, address: Operand<'e>, size: MemAccessSize) -> Operand<'e>;
    fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>>;
    fn unresolve_memory(&self, val: Operand<'e>) -> Option<Operand<'e>>;
    fn initial_state(
        operand_ctx: OperandCtx<'e>,
        binary: &'e crate::BinaryFile<Self::VirtualAddress>,
    ) -> Self;
    fn merge_states(
        old: &mut Self,
        new: &mut Self,
        cache: &mut MergeStateCache<'e>,
    ) -> Option<Self>;

    /// Updates states as if the call instruction was executed (Push return address to stack)
    ///
    /// A separate function as calls are usually just stepped over.
    fn apply_call(&mut self, ret: Self::VirtualAddress);

    /// Creates an Mem[addr] with MemAccessSize of VirtualAddress size
    fn operand_mem_word(ctx: OperandCtx<'e>, address: Operand<'e>) -> Operand<'e> {
        if <Self::VirtualAddress as VirtualAddress>::SIZE == 4 {
            ctx.mem32(address)
        } else {
            ctx.mem64(address)
        }
    }

    /// Updates state with the condition assumed to be true/false.
    fn assume_jump_flag(
        &mut self,
        condition: Operand<'e>,
        jump: bool,
    ) {
        let ctx = self.ctx();
        match *condition.ty() {
            OperandType::Arithmetic(ref arith) => {
                let left = arith.left;
                let right = arith.right;
                match arith.ty {
                    ArithOpType::Equal => {
                        let unresolved_cond = match jump {
                            true => condition.clone(),
                            false => ctx.eq_const(condition, 0)
                        };
                        let resolved_cond = self.resolve(unresolved_cond);
                        self.add_resolved_constraint(Constraint::new(resolved_cond));
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
                            .and_then(|(other, flag_state)| match *other.ty() {
                                OperandType::Flag(f) => Some((f, flag_state)),
                                _ => None,
                            });
                        if let Some((flag, flag_state)) = assignable_flag {
                            let constant = self.ctx().constant(flag_state as u64);
                            self.move_to(&DestOperand::Flag(flag), constant);
                        } else {
                            self.add_unresolved_constraint(Constraint::new(unresolved_cond));
                        }
                    }
                    ArithOpType::Or => {
                        if jump {
                            let unresolved_cond = ctx.or(left, right);
                            let cond = self.resolve(unresolved_cond);
                            self.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            self.add_resolved_constraint(Constraint::new(cond));
                        } else {
                            let unresolved_cond = ctx.and(
                                ctx.eq_const(left, 0),
                                ctx.eq_const(right, 0),
                            );
                            let cond = self.resolve(unresolved_cond);
                            self.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            self.add_resolved_constraint(Constraint::new(cond));
                        }
                    }
                    ArithOpType::And => {
                        if jump {
                            let unresolved_cond = ctx.and(left, right);
                            let cond = self.resolve(unresolved_cond);
                            self.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            self.add_resolved_constraint(Constraint::new(cond));
                        } else {
                            let unresolved_cond = ctx.or(
                                ctx.eq_const(left, 0),
                                ctx.eq_const(right, 0),
                            );
                            let cond = self.resolve(unresolved_cond);
                            self.add_unresolved_constraint(Constraint::new(unresolved_cond));
                            self.add_resolved_constraint(Constraint::new(cond));
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }

    /// Returns smallest and largest (inclusive) value a *resolved* operand can have
    /// (Mainly meant to use extra constraint information)
    fn value_limits(&mut self, _value: Operand<'e>) -> (u64, u64) {
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

    /// Equivalent to `out.write(self.clone())`, but may leave `out` partially
    /// overwritten if it panics.
    ///
    /// Useful for avoiding unnecessary memcpys.
    unsafe fn clone_to(&self, out: *mut Self);
}

/// Either `scarf::VirtualAddress` in 32-bit or `scarf::VirtualAddress64` in 64-bit
pub trait VirtualAddress: Eq + PartialEq + Ord + PartialOrd + Copy + Clone + std::hash::Hash +
    fmt::LowerHex + fmt::UpperHex + fmt::Debug + Add<u32, Output = Self> +
    Sub<u32, Output = Self> +
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
    #[inline]
    fn max_value() -> Self {
        crate::VirtualAddress(!0)
    }

    #[inline]
    fn inner(self) -> Self::Inner {
        self.0
    }

    #[inline]
    fn from_u64(val: u64) -> Self {
        crate::VirtualAddress(val as u32)
    }

    #[inline]
    fn as_u64(self) -> u64 {
        self.0 as u64
    }
}

impl VirtualAddress for crate::VirtualAddress64 {
    type Inner = u64;
    const SIZE: u32 = 8;
    #[inline]
    fn max_value() -> Self {
        crate::VirtualAddress64(!0)
    }

    #[inline]
    fn inner(self) -> Self::Inner {
        self.0
    }

    #[inline]
    fn from_u64(val: u64) -> Self {
        crate::VirtualAddress64(val)
    }

    #[inline]
    fn as_u64(self) -> u64 {
        self.0
    }
}

/// The constraint is assumed to be something that can be substituted with 1 if met
/// (so constraint == constval(1)).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Constraint<'e>(pub Operand<'e>);

impl<'e> Constraint<'e> {
    pub fn new(o: Operand<'e>) -> Constraint<'e> {
        Constraint(o)
    }

    /// Invalidates any assumptions about memory
    pub(crate) fn invalidate_memory(&mut self, ctx: OperandCtx<'e>) -> ConstraintFullyInvalid {
        let result = remove_matching_ands(ctx, self.0, &mut |x| match x {
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
    pub(crate) fn invalidate_dest_operand(
        self,
        ctx: OperandCtx<'e>,
        dest: &DestOperand<'e>,
    ) -> Option<Constraint<'e>> {
        match *dest {
            DestOperand::Register32(reg) | DestOperand::Register16(reg) |
                DestOperand::Register8High(reg) | DestOperand::Register8Low(reg) |
                DestOperand::Register64(reg) =>
            {
                remove_matching_ands(ctx, self.0, &mut |x| *x == OperandType::Register(reg))
            }
            DestOperand::Xmm(_, _) => {
                None
            }
            DestOperand::Fpu(_) => {
                None
            }
            DestOperand::Flag(flag) => {
                remove_matching_ands(ctx, self.0, &mut |x| *x == OperandType::Flag(flag))
            },
            DestOperand::Memory(_) => {
                // Assuming that everything may alias with memory
                remove_matching_ands(ctx, self.0, &mut |x| match x {
                    OperandType::Memory(..) => true,
                    _ => false,
                })
            }
        }.map(Constraint::new)
     }

    pub(crate) fn apply_to(self, ctx: OperandCtx<'e>, oper: Operand<'e>) -> Operand<'e> {
        apply_constraint_split(ctx, self.0, oper, true)
    }
}

#[derive(Eq, PartialEq, Copy, Clone)]
#[must_use]
pub enum ConstraintFullyInvalid {
    Yes,
    No,
}

/// Constraint invalidation helper
fn remove_matching_ands<'e, F>(
    ctx: OperandCtx<'e>,
    oper: Operand<'e>,
    fun: &mut F,
) -> Option<Operand<'e>>
where F: FnMut(&OperandType<'e>) -> bool,
{
    if let Some((l, r)) = oper.if_arithmetic_and() {
        // Only going to try partially invalidate cases with logical ands
        if l.relevant_bits() != (0..1) || r.relevant_bits() != (0..1) {
            match oper.iter().any(|x| fun(x.ty())) {
                true => None,
                false => Some(oper.clone()),
            }
        } else {
            let left = remove_matching_ands(ctx, l, fun);
            let right = remove_matching_ands(ctx, r, fun);
            match (left, right) {
                (None, None) => None,
                (Some(x), None) | (None, Some(x)) => Some(x),
                (Some(left), Some(right)) => Some(ctx.and(left, right)),
            }
        }
    } else {
        match oper.iter().any(|x| fun(x.ty())) {
            true => None,
            false => Some(oper.clone()),
        }
    }
}

fn other_if_eq_zero<'e>(op: Operand<'e>) -> Option<Operand<'e>> {
    op.if_arithmetic_eq()
        .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant().filter(|&c| c == 0)))
        .map(|x| x.1)
}

/// Splits the constraint at ands that can be applied separately
/// Also can handle logical nots
fn apply_constraint_split<'e>(
    ctx: OperandCtx<'e>,
    constraint: Operand<'e>,
    val: Operand<'e>,
    with: bool,
) -> Operand<'e> {
    match constraint.ty() {
        OperandType::Arithmetic(arith) if {
            arith.ty == ArithOpType::And &&
                arith.left.relevant_bits() == (0..1) &&
                arith.right.relevant_bits() == (0..1)
        } => {
            let new = apply_constraint_split(ctx, arith.left, val, with);
            apply_constraint_split(ctx, arith.right, new, with)
        }
        OperandType::Arithmetic(arith) if {
            arith.ty == ArithOpType::Or &&
                arith.left.relevant_bits() == (0..1) &&
                arith.right.relevant_bits() == (0..1)
        } => {
            // If constraint is (x == 0) | (y == 0),
            // (x & y) can be replaced with 0
            let x = other_if_eq_zero(arith.left);
            let y = other_if_eq_zero(arith.right);
            if let (Some(x), Some(y)) = (x, y) {
                // Check also for (x & y) == 0 (Replaced with 1)
                let (cmp, subst) = other_if_eq_zero(val)
                    .map(|x| (x, 1))
                    .unwrap_or_else(|| (val, 0));
                if let Some((l, r)) = cmp.if_arithmetic_and() {
                    if (l == x && r == y) || (l == y && r == x) {
                        return ctx.constant(subst);
                    }
                }
            }
            let subst_val = ctx.constant(if with { 1 } else { 0 });
            ctx.substitute(val, constraint, subst_val, 6)
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Equal => {
            let (l, r) = (arith.left, arith.right);
            let other = Operand::either(l, r, |x| x.if_constant().filter(|&c| c == 0))
                .map(|x| x.1)
                .filter(|x| x.relevant_bits() == (0..1));
            if let Some(other) = other {
                apply_constraint_split(ctx, other, val, !with)
            } else {
                let subst_val = ctx.constant(if with { 1 } else { 0 });
                ctx.substitute(val, constraint, subst_val, 6)
            }
        }
        _ => {
            let subst_val = ctx.constant(if with { 1 } else { 0 });
            ctx.substitute(val, constraint, subst_val, 6)
        }
    }
}

/// Helper for ExecutionState::value_limits implementations with constraints
pub(crate) fn value_limits_recurse<'e>(constraint: Operand<'e>, value: Operand<'e>) -> (u64, u64) {
    match constraint.ty() {
        OperandType::Arithmetic(arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let left = value_limits_recurse(arith.left, value);
                    let right = value_limits_recurse(arith.right, value);
                    return (left.0.max(right.0), (left.1.min(right.1)));
                }
                ArithOpType::GreaterThan => {
                    // 0 > x and x > u64_max should get simplified to 0
                    if let Some(c) = arith.left.if_constant() {
                        let (right, offset) = arith.right.if_arithmetic_sub()
                            .and_then(|(l, r)| Some((l, r.if_constant()?)))
                            .unwrap_or_else(|| (arith.right, 0));
                        if is_subset(value, right) {
                            debug_assert!(c != 0);
                            let low = offset;
                            if let Some(high) = c.wrapping_sub(1).checked_add(offset) {
                                return (low, high);
                            }
                        }
                    }
                    if let Some(c) = arith.right.if_constant() {
                        if is_subset(value, arith.left) {
                            debug_assert!(c != u64::max_value());
                            return (c.wrapping_add(1), u64::max_value());
                        }
                    }
                }
                ArithOpType::Sub => {
                    if let Some(low) = arith.right.if_constant() {
                        let (low_inner, high_inner) =
                            value_limits_recurse(constraint, arith.left);
                        if let Some(low) = low_inner.checked_add(low) {
                            if let Some(high) = high_inner.checked_add(low) {
                                return (low, high);
                            }
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
fn is_subset<'e>(sub: Operand<'e>, sup: Operand<'e>) -> bool {
    if sub.ty().expr_size() > sup.ty().expr_size() {
        return false;
    }
    match sub.ty() {
        OperandType::Memory(mem) => match sup.if_memory() {
            Some(mem2) => mem.address == mem2.address,
            None => false,
        }
        _ => sub == sup,
    }
}

/// Trait for disassembling instructions
pub trait Disassembler<'e> {
    type VirtualAddress: VirtualAddress;
    /// Creates a new Disassembler. It is expected that set_pos() is called
    /// afterwards before next().
    fn new(ctx: OperandCtx<'e>) -> Self;
    /// Seeks to a address.
    fn set_pos(
        &mut self,
        // Should this use a separate lifetime for clarity?
        // 'e does still in practice always refer to &BinaryFile as well,
        // so it should be ok.
        buf: &'e [u8],
        pos: usize,
        address: Self::VirtualAddress,
    );
    fn next<'s>(&'s mut self) -> Instruction<'s, 'e, Self::VirtualAddress>;
    fn address(&self) -> Self::VirtualAddress;
}

/// Contains memory state as addr -> value hashmap.
/// The ExecutionState is expected to take care of cases where memory is written with one
/// address and part of it is read at offset address, in practice by splitting accesses
/// to be word-sized, and any misaligned accesses become bitwise and-or combinations.
#[derive(Clone)]
pub struct Memory<'e> {
    pub(crate) map: MemoryMap<'e>,
    /// Caches value of last read/write
    cached_addr: Option<Operand<'e>>,
    cached_value: Option<Operand<'e>>,
}

#[derive(Clone)]
pub(crate) struct MemoryMap<'e> {
    /// Mutable map, cloned lazily on mutation if Rc is shared.
    /// Two memory cannot have ptr-equal `map` with different `immutable`
    ///
    /// It seems that this allow avoiding roughly 10% of memory merge calls, but also
    /// likely saves large amounts of time at exec state merges, for which about 90%
    /// won't result in changed state. (Doesn't mean that 90% of exec state merges can fast
    /// path ptr-eq compare this map though)
    ///
    /// Also if the additional indirection seems to ever hurt reads/writes, could have some
    /// temp struct that contains a direct (mutable) reference to the inner value that then
    /// can be used to read/write a lot at once.
    /// But I would believe that the instruction decoding/etc overhead is a lot greater
    /// than memory access during analysis.
    pub(crate) map: Rc<MemoryMapTopLevel<'e>>,
    /// Optimization for cases where memory gets large.
    /// The existing mapping can be moved to Rc, where cloning it is effectively free.
    immutable: Option<Rc<MemoryMap<'e>>>,
}

type MemoryMapTopLevel<'e> = HashMap<OperandHashByAddress<'e>, Operand<'e>, FxBuildHasher>;

struct MemoryIterUntilImm<'a, 'e> {
    iter: hash_map::Iter<'a, OperandHashByAddress<'e>, Operand<'e>>,
    immutable: &'a Option<Rc<MemoryMap<'e>>>,
    limit: Option<&'a MemoryMap<'e>>,
    in_immutable: bool,
}

impl<'e> Memory<'e> {
    pub fn new() -> Memory<'e> {
        Memory {
            map: MemoryMap {
                map: Rc::new(HashMap::with_hasher(Default::default())),
                immutable: None,
            },
            cached_addr: None,
            cached_value: None,
        }
    }

    pub fn get(&mut self, address: Operand<'e>) -> Option<Operand<'e>> {
        if Some(address) == self.cached_addr {
            return self.cached_value;
        }
        let result = self.map.get(address);
        self.cached_addr = Some(address);
        self.cached_value = result;
        result
    }

    pub fn set(&mut self, address: Operand<'e>, value: Operand<'e>) {
        self.map.set(address, value);
        self.cached_addr = Some(address);
        self.cached_value = Some(value);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Does a value -> key lookup (Finds an address containing value)
    pub fn reverse_lookup(&self, value: Operand<'e>) -> Option<Operand<'e>> {
        self.map.reverse_lookup(value)
    }

    /// Does a reverse lookup on last accessed memory address.
    ///
    /// Could probably cache like 4 previous accesses for slightly better results.
    pub(crate) fn fast_reverse_lookup(
        &self,
        ctx: OperandCtx<'e>,
        value: Operand<'e>,
        exec_state_mask: u64,
    ) -> Option<(Operand<'e>, MemAccessSize)> {
        fn check_or_part<'e>(
            this: &Memory<'e>,
            ctx: OperandCtx<'e>,
            part: Operand<'e>,
            value: Operand<'e>,
        ) -> Option<(Operand<'e>, MemAccessSize)> {
            static BYTES_TO_MEM_SIZE: [MemAccessSize; 8] = [
                MemAccessSize::Mem8, MemAccessSize::Mem16,
                MemAccessSize::Mem32, MemAccessSize::Mem32,
                MemAccessSize::Mem64, MemAccessSize::Mem64,
                MemAccessSize::Mem64, MemAccessSize::Mem64,
            ];
            if part == value {
                let size_bytes = part.relevant_bits().end.wrapping_sub(1) / 8;
                let mem_size = BYTES_TO_MEM_SIZE[(size_bytes & 7) as usize];
                Some((this.cached_addr?, mem_size))
            } else if let Some((l, r)) = part.if_arithmetic(ArithOpType::Lsh) {
                let offset = r.if_constant()?;
                let inner = check_or_part(this, ctx, l, value)?;
                let bytes = (offset as u8) >> 3 << 3;
                Some((ctx.add_const(inner.0, bytes as u64), inner.1))
            } else if let Some((l, r)) = part.if_arithmetic(ArithOpType::Or) {
                if l.relevant_bits().end < r.relevant_bits().start ||
                    r.relevant_bits().end < l.relevant_bits().start
                {
                    return None;
                }
                check_or_part(this, ctx, l, value)
                    .or_else(|| check_or_part(this, ctx, r, value))
            } else if let Some((l, r)) = part.if_arithmetic(ArithOpType::And) {
                let result = check_or_part(this, ctx, l, value)?;
                let mask = r.if_constant()?;
                let masked_size = if mask < 0x100 {
                    MemAccessSize::Mem8
                } else if mask < 0x1_0000 {
                    MemAccessSize::Mem16
                } else if mask < 0x1_0000_0000 {
                    MemAccessSize::Mem32
                } else {
                    MemAccessSize::Mem64
                };
                Some((result.0, masked_size))
            } else {
                None
            }
        }

        let (value, mask) = Operand::and_masked(value);
        let cached = if exec_state_mask != u64::max_value() {
            ctx.and_const(self.cached_value?, exec_state_mask)
        } else {
            self.cached_value?
        };
        let (cached, _) = Operand::and_masked(cached);
        let result = if value == cached {
            Some((self.cached_addr?, MemAccessSize::Mem64))
        } else if let Some((l, r)) = cached.if_arithmetic_or() {
            if l.relevant_bits().end < r.relevant_bits().start ||
                r.relevant_bits().end < l.relevant_bits().start
            {
                return None;
            }
            check_or_part(self, ctx, l, value)
                .or_else(|| check_or_part(self, ctx, r, value))
        } else {
            None
        };
        result.map(|(op, size)| {
            let masked_size = if mask < 0x100 {
                MemAccessSize::Mem8
            } else if mask < 0x1_0000 {
                MemAccessSize::Mem16
            } else if mask < 0x1_0000_0000 {
                MemAccessSize::Mem32
            } else {
                MemAccessSize::Mem64
            };
            if size.bits() > masked_size.bits() {
                (op, masked_size)
            } else {
                (op, size)
            }
        })
    }

    pub fn merge(&self, new: &Memory<'e>, ctx: OperandCtx<'e>) -> Memory<'e> {
        Memory {
            map: self.map.merge(&new.map, ctx),
            cached_addr: None,
            cached_value: None,
        }
    }
}

impl<'e> MemoryMap<'e> {
    pub fn get(&self, address: Operand<'e>) -> Option<Operand<'e>> {
        let op = self.map.get(&OperandHashByAddress(address)).cloned()
            .or_else(|| self.immutable.as_ref().and_then(|x| x.get(address)));
        op
    }

    pub fn set(&mut self, address: Operand<'e>, value: Operand<'e>) {
        let key = address.hash_by_address();
        // Don't insert a duplicate entry to mutable map if the immutable map already
        // has the correct key.
        //
        // Maybe avoiding making map mutable in the case where it doesn't contain
        // key and immutable has already correct value would be worth it?
        // It would avoid Rc::make_mut, but I feel that most of the time some other
        // instruction would require mutating th Rc immediately afterwards anyway.
        let map = Rc::make_mut(&mut self.map);
        if let Some(ref imm) = self.immutable {
            if imm.get(address) == Some(value) {
                map.remove(&key);
                return;
            }
        }
        map.insert(key, value);
    }

    pub fn len(&self) -> usize {
        self.map.len() + self.immutable.as_ref().map(|x| x.len()).unwrap_or(0)
    }

    /// The bool is true if the value is in immutable map
    fn get_with_immutable_info(
        &self,
        address: Operand<'e>
    ) -> Option<(Operand<'e>, bool)> {
        let op = self.map.get(&OperandHashByAddress(address)).map(|&x| (x, false))
            .or_else(|| {
                self.immutable.as_ref().and_then(|x| x.get(address)).map(|x| (x, true))
            });
        op
    }

    /// Iterates until the immutable block.
    /// Only ref-equality is considered, not deep-eq.
    fn iter_until_immutable<'a>(
        &'a self,
        limit: Option<&'a MemoryMap<'e>>,
    ) -> MemoryIterUntilImm<'a, 'e> {
        MemoryIterUntilImm {
            iter: self.map.iter(),
            immutable: &self.immutable,
            limit,
            in_immutable: false,
        }
    }

    fn common_immutable<'a>(&'a self, other: &'a MemoryMap<'e>) -> Option<&'a MemoryMap<'e>> {
        self.immutable.as_ref().and_then(|i| {
            let a = &**i as *const MemoryMap;
            let mut pos = Some(other);
            while let Some(o) = pos {
                if o as *const MemoryMap == a {
                    return pos;
                }
                pos = o.immutable.as_ref().map(|x| &**x);
            }
            i.common_immutable(other)
        })
    }

    pub fn maybe_convert_immutable(&mut self, limit: usize) {
        if self.map.len() >= limit {
            self.convert_immutable();
        }
    }

    fn convert_immutable(&mut self) {
        let map = mem::replace(
            &mut self.map,
            Rc::new(HashMap::with_hasher(Default::default())),
        );
        let old_immutable = self.immutable.take();
        self.immutable = Some(Rc::new(MemoryMap {
            map,
            immutable: old_immutable,
        }));
    }

    /// Does a value -> key lookup (Finds an address containing value)
    pub fn reverse_lookup(&self, value: Operand<'e>) -> Option<Operand<'e>> {
        for (&key, &val) in self.map.iter() {
            if value == val {
                return Some(key.0);
            }
        }
        if let Some(ref i) = self.immutable {
            i.reverse_lookup(value)
        } else {
            None
        }
    }

    pub fn merge(&self, new: &MemoryMap<'e>, ctx: OperandCtx<'e>) -> MemoryMap<'e> {
        // Merging memory is defined as:
        // If new & old values match, then the value is kept
        // Otherwise:
        //   If the address contains undefined, then it *may* be forgotten entirely
        //   If old is undefined, keep old value
        //   If old isn't undefined, generate a new value
        let a = self;
        let b = new;
        if Rc::ptr_eq(&a.map, &b.map) {
            return a.clone();
        }
        let mut result = HashMap::with_capacity_and_hasher(
            a.map.len().max(b.map.len()),
            Default::default(),
        );
        let a_empty = a.len() == 0;
        let b_empty = b.len() == 0;
        if (a_empty || b_empty) && a.immutable.is_none() && b.immutable.is_none() {
            let other = if a_empty { b } else { a };
            for (&key, &val) in other.map.iter() {
                if val.is_undefined() {
                    result.insert(key, val);
                } else {
                    result.insert(key, ctx.new_undef());
                }
            }
            return MemoryMap {
                map: Rc::new(result),
                immutable: a.immutable.clone(),
            };
        }
        let imm_eq = a.immutable.as_ref().map(|x| &**x as *const MemoryMap) ==
            b.immutable.as_ref().map(|x| &**x as *const MemoryMap);
        let result = if imm_eq {
            // Allows just checking a.map.iter() instead of a.iter()
            for (&key, &a_val) in a.map.iter() {
                if a_val.is_undefined() {
                    result.insert(key, a_val);
                } else {
                    if let Some((b_val, is_imm)) = b.get_with_immutable_info(key.0) {
                        match a_val == b_val {
                            true => {
                                if !is_imm {
                                    result.insert(key, a_val);
                                }
                            }
                            false => {
                                result.insert(key, ctx.new_undef());
                            }
                        }
                    } else {
                        result.insert(key, ctx.new_undef());
                    }
                }
            }
            'b_loop: for (&key, &b_val) in b.map.iter() {
                // This seems to be slightly faster than using entry()...
                // Maybe it's just something with different inlining decisions
                // that won't actually always be better, but it seems consistent
                // enough that I'm leaving this as is.
                if !result.contains_key(&key) {
                    let val = if b_val.is_undefined() {
                        b_val
                    } else {
                        if let Some((a_val, is_imm)) = a.get_with_immutable_info(key.0) {
                            if is_imm && a_val == b_val {
                                continue 'b_loop;
                            }
                        }
                        ctx.new_undef()
                    };
                    result.insert(key, val);
                }
            }
            MemoryMap {
                map: Rc::new(result),
                immutable: a.immutable.clone(),
            }
        } else {
            // a's immutable map is used as base, so one which exist there don't get inserted to
            // result, but if it has ones that should become undefined, the undefined has to be
            // inserted to the result instead.
            let common = a.common_immutable(b);
            for (&key, &b_val, b_is_imm) in b.iter_until_immutable(common) {
                if b_is_imm && b.get(key.0) != Some(b_val) {
                    // Wasn't newest value
                    continue;
                }
                if let Some((a_val, is_imm)) = a.get_with_immutable_info(key.0) {
                    match a_val == b_val {
                        true => {
                            if !is_imm {
                                result.insert(key, a_val);
                            }
                        }
                        false => {
                            if !a_val.is_undefined() {
                                result.insert(key, ctx.new_undef());
                            }
                        }
                    }
                } else {
                    if !key.0.contains_undefined() {
                        result.insert(key, ctx.new_undef());
                    }
                }
            }
            // The result contains now anything that was in b's unique branch of the memory.
            //
            // Repeat for a's unique branch.
            for (&key, &a_val, a_is_imm) in a.iter_until_immutable(common) {
                if !result.contains_key(&key) {
                    if !a_val.is_undefined() {
                        if a_is_imm && a.get(key.0) != Some(a_val) {
                            // Wasn't newest value
                            continue;
                        }
                        let needs_undef = if let Some(b_val) = b.get(key.0) {
                            a_val != b_val
                        } else {
                            true
                        };
                        if needs_undef {
                            // If the key with undefined was in imm, override its value,
                            // but otherwise just don't bother adding it back.
                            if !key.0.contains_undefined() || !a_is_imm {
                                result.insert(key, ctx.new_undef());
                            }
                        }
                    } else {
                        result.insert(key, a_val);
                    }
                }
            }
            MemoryMap {
                map: Rc::new(result),
                immutable: a.immutable.clone(),
            }
        };
        result
    }

    /// Check if there are any not equal fields that aren't undefined in `self`
    pub fn has_merge_changed(&self, b: &MemoryMap<'e>) -> bool {
        let a = self;
        if Rc::ptr_eq(&a.map, &b.map) {
            return false;
        }
        let common = a.common_immutable(b);
        for (&key, &a_val, is_imm) in a.iter_until_immutable(common) {
            if !key.0.contains_undefined() {
                if !a_val.is_undefined() {
                    if b.get(key.0) != Some(a_val) {
                        let was_newest_value = if !is_imm {
                            true
                        } else {
                            a.get(key.0) == Some(a_val)
                        };
                        if was_newest_value {
                            return true;
                        }
                    }
                }
            }
        }
        for (&key, &b_val, is_imm) in b.iter_until_immutable(common) {
            if !key.0.contains_undefined() {
                let different = match a.get(key.0) {
                    Some(a_val) => !a_val.is_undefined() && a_val != b_val,
                    None => true,
                };
                if different {
                    let was_newest_value = if !is_imm {
                        true
                    } else {
                        b.get(key.0) == Some(b_val)
                    };
                    if was_newest_value {
                        return true;
                    }
                }
            }
        }
        false
    }
}

impl<'a, 'e> Iterator for MemoryIterUntilImm<'a, 'e> {
    /// The bool tells if we're at immutable parts of the calling operand or not
    type Item = (&'a OperandHashByAddress<'e>, &'a Operand<'e>, bool);
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(s) => Some((s.0, s.1, self.in_immutable)),
            None => {
                let at_limit = self.immutable.as_ref().map(|x| &**x as *const MemoryMap) ==
                    self.limit.map(|x| x as *const MemoryMap);
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

pub fn merge_constraint<'e>(
    ctx: OperandCtx<'e>,
    old: Option<Constraint<'e>>,
    new: Option<Constraint<'e>>,
) -> Option<Constraint<'e>> {
    if old == new {
        return old;
    }
    let old = old?;
    let new = new?;
    let res = ctx.simplify_temp_stack().alloc(|mut old_parts| {
        collect_and_ops(old.0, &mut old_parts)?;
        ctx.simplify_temp_stack().alloc(|mut new_parts| {
            collect_and_ops(new.0, &mut new_parts)?;
            old_parts.retain(|old_val| {
                new_parts.iter().any(|&x| x == old_val)
            });
            join_ands(ctx, old_parts)
        })
    }).map(Constraint);
    res
}

fn join_ands<'e>(ctx: OperandCtx<'e>, parts: &[Operand<'e>]) -> Option<Operand<'e>> {
    let mut op = *parts.get(0)?;
    for &next in parts.iter().skip(1) {
        op = ctx.and(op, next);
    }
    Some(op)
}

fn collect_and_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e, Operand<'e>>,
) -> Option<()> {
    if let Some((l, r)) = s.if_arithmetic_and() {
        collect_and_ops(l, ops)?;
        collect_and_ops(r, ops)?;
    } else {
        ops.push(s).ok()?;
    }
    Some(()).filter(|()| ops.len() <= 8)
}

/// A cache which allows skipping some repeated work during merges.
///
/// For now, it caches last memory-changed result and
/// memory-merge results. Merge result caching is especially useful, as the
/// results will share their Rc pointer values, allowing deep comparisions
/// between them skipped - and even more merge caching.
///
/// This should be useful any time a branch has two destinations which aren't
/// reached from anywhere else, as analysis will store equal state to them
/// first time it reaches them, and then merge them again with same inputs.
/// Without this cache the merge would end up giving equal memory maps
/// which weren't stored in a same Rc, which would be unfortunate.
///
/// Large switches will obviously benefit even more from this.
pub struct MergeStateCache<'e> {
    last_compare: Option<MemoryOpCached<'e, bool>>,
    last_merge: Option<MemoryOpCached<'e, MemoryMap<'e>>>,
}

struct MemoryOpCached<'e, T> {
    // NOTE: This relies on MemoryMapTopLevel being always associated with same
    // immutable pointer.
    old: Rc<MemoryMapTopLevel<'e>>,
    new: Rc<MemoryMapTopLevel<'e>>,
    result: T,
}

impl<'e> MergeStateCache<'e> {
    pub fn new() -> MergeStateCache<'e> {
        MergeStateCache {
            last_compare: None,
            last_merge: None,
        }
    }

    pub fn get_compare_result(&self, old: &Memory<'e>, new: &Memory<'e>) -> Option<bool> {
        let cached = self.last_compare.as_ref()?;
        if Rc::ptr_eq(&cached.old, &old.map.map) && Rc::ptr_eq(&cached.new, &new.map.map) {
            Some(cached.result)
        } else {
            None
        }
    }

    pub fn set_compare_result(&mut self, old: &Memory<'e>, new: &Memory<'e>, result: bool) {
        self.last_compare = Some(MemoryOpCached {
            old: old.map.map.clone(),
            new: new.map.map.clone(),
            result,
        })
    }

    pub fn get_merge_result(&self, old: &Memory<'e>, new: &Memory<'e>) -> Option<Memory<'e>> {
        let cached = self.last_merge.as_ref()?;
        if Rc::ptr_eq(&cached.old, &old.map.map) && Rc::ptr_eq(&cached.new, &new.map.map) {
            Some(Memory {
                map: cached.result.clone(),
                cached_addr: None,
                cached_value: None,
            })
        } else {
            None
        }
    }

    pub fn set_merge_result(&mut self, old: &Memory<'e>, new: &Memory<'e>, result: &Memory<'e>) {
        self.last_merge = Some(MemoryOpCached {
            old: old.map.map.clone(),
            new: new.map.map.clone(),
            result: result.map.clone(),
        })
    }
}

#[test]
fn apply_constraint() {
    let ctx = &crate::operand::OperandContext::new();
    let constraint = Constraint(ctx.eq_const(
        ctx.neq_const(
            ctx.flag_z(),
            0,
        ),
        0,
    ));
    let val = ctx.or(
        ctx.neq_const(
            ctx.flag_c(),
            0,
        ),
        ctx.neq_const(
            ctx.flag_z(),
            0,
        ),
    );
    let old = val.clone();
    let val = constraint.apply_to(ctx, val);
    let eq = ctx.neq_const(
        ctx.flag_c(),
        0,
    );
    assert_ne!(val, old);
    assert_eq!(val, eq);
}

#[test]
fn apply_constraint_non1bit() {
    let ctx = &crate::operand::OperandContext::new();
    // This shouldn't cause 0x8000_0000 to be optimized out
    let constraint = ctx.eq(
        ctx.and(
            ctx.constant(0x8000_0000),
            ctx.register(1),
        ),
        ctx.constant(0),
    );
    let val = ctx.and(
        ctx.constant(0x8000_0000),
        ctx.sub(
            ctx.register(1),
            ctx.register(2),
        ),
    );
    assert_eq!(Constraint(constraint).apply_to(ctx, val), val);
}

#[test]
fn apply_constraint_or() {
    let ctx = &crate::operand::OperandContext::new();
    let constraint = ctx.or(
        ctx.neq(
            ctx.flag_o(),
            ctx.flag_s(),
        ),
        ctx.neq_const(
            ctx.flag_z(),
            0,
        ),
    );
    let val = ctx.eq_const(
        ctx.and(
            ctx.eq(
                ctx.flag_o(),
                ctx.flag_s(),
            ),
            ctx.eq_const(
                ctx.flag_z(),
                0,
            ),
        ),
        0,
    );
    let applied = Constraint(constraint).apply_to(ctx, val);
    let eq = ctx.constant(1);
    assert_eq!(applied, eq);
}

#[test]
fn merge_immutable_memory() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    a.set(ctx.constant(4), ctx.constant(8));
    a.set(ctx.constant(12), ctx.constant(8));
    b.set(ctx.constant(8), ctx.constant(15));
    b.set(ctx.constant(12), ctx.constant(8));
    a.map.convert_immutable();
    b.map.convert_immutable();
    let mut new = a.merge(&b, ctx);
    assert!(new.get(ctx.constant(4)).unwrap().is_undefined());
    assert!(new.get(ctx.constant(8)).unwrap().is_undefined());
    assert_eq!(new.get(ctx.constant(12)).unwrap(), ctx.constant(8));
}

#[test]
fn merge_memory_undef() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    let addr = ctx.sub_const(ctx.new_undef(), 8);
    a.set(addr, ctx.mem32(ctx.constant(4)));
    b.set(addr, ctx.mem32(ctx.constant(4)));
    a.map.convert_immutable();
    let mut new = a.merge(&b, ctx);
    assert_eq!(new.get(addr).unwrap(), ctx.mem32(ctx.constant(4)));
}

#[test]
fn merge_memory_equal() {
    // a has [addr] = 4 in top level, but 8 in immutable
    // b has [addr] = 4 in immutable
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    let addr = ctx.sub_const(ctx.register(4), 8);
    let addr2 = ctx.sub_const(ctx.register(4), 0xc);
    a.set(addr, ctx.constant(8));
    a.set(addr2, ctx.constant(0));
    b.set(addr, ctx.constant(4));
    a.map.convert_immutable();
    b.map.convert_immutable();
    a.set(addr, ctx.constant(4));
    a.set(addr2, ctx.constant(1));
    let mut new = a.merge(&b, ctx);
    assert_eq!(new.get(addr).unwrap(), ctx.constant(4));
    assert!(new.get(addr2).unwrap().is_undefined());
}

#[test]
fn equal_memory_no_need_to_merge() {
    // a has [addr] = 4 in top level, but 8 in immutable
    // b has [addr] = 4 in immutable
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    let addr = ctx.sub_const(ctx.register(4), 8);
    a.set(addr, ctx.constant(8));
    b.set(addr, ctx.constant(4));
    a.map.convert_immutable();
    b.map.convert_immutable();
    a.set(addr, ctx.constant(4));
    assert!(!a.map.has_merge_changed(&b.map));
}

#[test]
fn merge_memory_undef2() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    a.set(addr, ctx.mem32(ctx.constant(4)));
    a.map.convert_immutable();
    let mut b = a.clone();
    b.set(addr, ctx.mem32(ctx.constant(9)));
    b.map.convert_immutable();
    a.set(addr, ctx.new_undef());
    a.set(addr2, ctx.new_undef());
    let mut new = a.merge(&b, ctx);
    assert!(new.get(addr).unwrap().is_undefined());
    assert!(new.get(addr2).unwrap().is_undefined());
}

#[test]
fn value_limits_gt_range() {
    let ctx = &crate::operand::OperandContext::new();
    let constraint = ctx.gt_const_left(
        6,
        ctx.sub_const(
            ctx.register(0),
            2,
        ),
    );
    let (low, high) = value_limits_recurse(constraint, ctx.register(0));
    assert_eq!(low, 2);
    assert_eq!(high, 7);
}
