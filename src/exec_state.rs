//! Traits for abstracting over different CPU architecture, and code that can be shared
//! between them.
//!
//! Main points of interest are [`ExecutionState`] for the main architecture trait,
//! and [`VirtualAddress`] for the "Integer representing memory address of word size"
//! trait.

use std::collections::{HashMap};
use std::fmt;
use std::mem;
use std::ops::{Add, Sub};
use std::ptr;
use std::rc::Rc;

use arrayvec::ArrayVec;
use fxhash::FxBuildHasher;

use crate::analysis;
use crate::disasm::{FlagArith, DestOperand, FlagUpdate, Instruction, Operation};
use crate::operand::{
    ArithOperand, ArithOpType, Flag, Operand, OperandType, OperandCtx, OperandHashByAddress,
    MemAccessSize, MemAccess,
};
use crate::operand::slice_stack::Slice;
use crate::{BinaryFile};

/// A trait that does (most of) arch-specific state handling.
///
/// ExecutionState contains the CPU state that is simulated, so registers and memory.
///
/// It is also "the architecture trait" for generic code that wants to handle multiple
/// CPU architectures, usually as [`Analyzer::Exec`](analysis::Analyzer::Exec).
///
/// # Resolved and unresolved operands
///
/// `ExecutionState` can be considered a key-value map, where keys are registers and memory,
/// each containing a single value, [`Operand`]. But instead of providing explicit
/// `get_register` function, scarf uses `'unresolved' Operand -> 'resolved' Operand`
/// function [`resolve`](Self::resolve), which replaces all registers and memory in the
/// input `Operand` with values from `ExecutionState`, and returns the result.
///
/// Unresolved operands can be thought to refer the 'current' state, while resolved
/// operands refer to state at function entry, when the initial ExecutionState was
/// constructed.
/// Most of the the user-side analysis code will likely want to just deal with
/// resolved operands, as they represent same value regardless of what point in analysis
/// they've been constructed. Unresolved operands are mainly useful when needing to be
/// concerned with instruction-level details.
///
/// As an example, consider the following piece of x86-64 assembly code.
///
/// ```text
/// mov r8, rcx
/// add r8, 3
/// sub rcx, r8
/// ```
///
/// Scarf would represent these instructions as 3 `Operation::Move`s, using unresolved operands.
/// If the ExecutionState was just constructed and these are the first instructions executed,
/// The resolved values of rcx and r8 are simply the register itself, rcx and r8.
/// As the instructions get handled, scarf updates the values in ExecutionState:
///
/// ```text
/// mov r8, rcx
/// Unresolved: r8 = rcx
///     => `rcx` resolves to `rcx`
///     => Now resolved r8 = `rcx`
///
/// add r8, 3
/// Unresolved: r8 = r8 + 3
///     => `r8` resolves to `rcx`; `r8 + 3` resolves to `rcx + 3`
///     => Now resolved r8 = `rcx + 3`
///
/// sub rcx, r8
/// Unresolved: rcx = rcx - r8
///     => `rcx` resolves to `rcx`; `r8` resolves to `rcx + 3`; `rcx - r8` resolves to just `-3`
///     => Now resolved rcx = `-3` (To be precise, 0xffff_ffff_ffff_fffd)
///     => r8 still resolves to `rcx + 3`
/// ```
///
/// [`Operation`], which represents a (part of a) CPU instruction, without external state,
/// always contains unresolved operands, and resolving them is usually the first thing
/// that is done before inspecting them further.
///
/// When stating that a [`DestOperand`] is resolved or unresolved, the only behaviour changes
/// affect handling of `DestOperand::Memory` variant.
/// [`update`](Self::update) and [`move_to`](Self::move_to) consider it unresolved, and will
/// resolve it to get the memory address that gets written to.
/// [`move_resolved`](Self::move_resolved) will consider the memory address be already resolved.
/// Other variants of `DestOperand` do not change meaning between resolved or unresolved.
///
/// # Memory addresses
///
/// Scarf splits memory addresses to a 'base' [`Operand`] and 'offset' `u64`. ExecutionState
/// will make sure that any write and reads with equivalent base and overlapping offsets will
/// access the same values, simulating little-endian memory. Each unique base operand is
/// considered a completely separate memory space from each other, which generally is
/// conservatively correct, and currently there is no way (outside of duplicating memory writes)
/// to make two separate memory address bases to alias.
///
/// Internally the overlapping memory is implemented as a single `Operand` at each
/// 4- or 8-aligned offset, with bitwise ANDs, ORs, and shifts to make memory reading/writing
/// work without any additional handling required from outside ExecutionState. However, when
/// two states are [merged](../analysis/struct.FuncAnalysis.html#state-merging-and-loops),
/// the entire 4- or 8-byte range will be merged
/// to `Undefined`, even if only single byte of the range differed. This limitation does
/// not cause issues too often, but if a function stores multiple single-byte values
/// next to each other in memory, reading one of those bytes may give `Undefined` instead of
/// a more useful `Operand`.
///
/// See also [`MemAccess`] documentation for an overview of memory address invariants in scarf.
pub trait ExecutionState<'e> : Clone + 'e {
    type VirtualAddress: VirtualAddress;
    // Technically ExecState shouldn't need to be linked to disassembly code,
    // but I didn't want an additional trait to sit between user-defined AnalysisState and
    // ExecutionState, so ExecutionState shall contain this detail as well.
    type Disassembler: Disassembler<'e, VirtualAddress = Self::VirtualAddress>;
    const WORD_SIZE: MemAccessSize;

    /// Constructs the `ExecutionState`.
    fn initial_state(
        operand_ctx: OperandCtx<'e>,
        binary: &'e BinaryFile<Self::VirtualAddress>,
    ) -> Self;

    /// Returns the `OperandCtx` used by `self`.
    fn ctx(&self) -> OperandCtx<'e>;

    /// Converts unresolved input `Operand` to resolved `Operand`.
    ///
    /// That is, all register and memory `Operand`s in the input are replaced
    /// with what value the state has for this register or memory address.
    fn resolve(&mut self, operand: Operand<'e>) -> Operand<'e>;

    /// Converts unresolved input `MemAccess` to resolved `MemAccess`.
    ///
    /// Note that this is mostly equivalent function to `resolve`, just operating
    /// in `MemAccess` which saves some work and `Operand` allocations that resolving
    /// [`mem.address_op()`](MemAccess::address_op) would do.
    ///
    /// On the other hand, [`read_memory`](Self::read_memory) operates with a resolved address,
    /// allowing reading memory which may not be practical to reach from `resolve()` or
    /// `resolve_mem()`.
    fn resolve_mem(&mut self, mem: &MemAccess<'e>) -> MemAccess<'e>;

    /// As resolve, but applies constraints, which makes the operation slower
    /// but can produce more accurate results, especially for jump conditions.
    ///
    /// Constraints are usually added to state when analysis reaches a jump, for example
    /// a branch from `eax > 5` jump, will have a constraint that simplifies later `eax > 5`
    /// resolves to `1`, if resolve_apply_constraints is used.
    fn resolve_apply_constraints(&mut self, operand: Operand<'e>) -> Operand<'e>;

    /// Resolves a value of register.
    ///
    /// Quicker equivalent to `self.resolve(ctx.register(register))`.
    fn resolve_register(&mut self, register: u8) -> Operand<'e>;

    /// Resolves a value of flag.
    ///
    /// Quicker equivalent to `self.resolve(ctx.flag(flag))`.
    fn resolve_flag(&mut self, flag: Flag) -> Operand<'e>;

    /// Reads memory from a resolved `MemAccess`.
    ///
    /// Note that this differs from `resolve` and `resolve_mem`, which will always resolve
    /// the memory address, often ending up with a different result.
    ///
    /// For example, consider a function where we are interested in what the function
    /// writes to memory pointed by it input argument, passed in register rcx. Calling
    /// `resolve(Mem64[rcx])` or `resolve_mem({rcx, 0, Mem64})` on function return will
    /// first resolve rcx, which may no longer hold the input argument, and then read
    /// memory from there. Whereas `read_memory({rcx, 0, Mem64})` will read the
    /// memory pointed by input argument, no matter what the rcx register currently holds.
    ///
    /// While there is no a explicit `write_memory` method, [`move_resolved`](Self::move_resolved)
    /// with `DestOperand::Memory(mem)` will handle the corresponding write case.
    fn read_memory(&mut self, mem: &MemAccess<'e>) -> Operand<'e>;

    /// Applies changes from operation to the state.
    ///
    /// This is what the analysis code will do for each non-control-flow-related `Operation`
    /// generated, if the user does not prevent this by calling
    /// [`Control::skip_operation`](crate::analysis::Control::skip_operation).
    fn update(&mut self, operation: &Operation<'e>);

    /// Moves an unresolved value to unresolved destination.
    ///
    /// Convenience method for the equivalent but more verbose
    /// `update(Operation::Move(dest, value, None)`.
    fn move_to(&mut self, dest: &DestOperand<'e>, value: Operand<'e>);

    /// Moves a resolved value to (resolved) destination.
    ///
    /// This is the main way to update state when the value you have is
    /// already resolved. Note that if destination is a memory address, it will be
    /// considered to be resolved. To move a resolved value to unresolved memory address,
    /// resolve the `MemAccess` and call `move_resolved` with that.
    fn move_resolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>);

    /// Sets a register to resolved value.
    ///
    /// Shortcut for `self.move_resolved(&DestOperand::Register(register), value)`.
    fn set_register(&mut self, register: u8, value: Operand<'e>);

    /// Sets a flag to resolved value.
    ///
    /// Shortcut for `self.move_resolved(&DestOperand::Flag(register), value)`.
    fn set_flag(&mut self, flag: Flag, value: Operand<'e>);

    /// Sets flags to a value that has already been resolved.
    ///
    /// `carry` should contain resolved value of the carry flag, if `FlagUpdate.ty` is
    /// an operation taking carry as input (Adc or Sbb).
    fn set_flags_resolved(&mut self, flags: &FlagUpdate<'e>, carry: Option<Operand<'e>>);

    /// Updates state as if the call instruction was executed (Push return address to stack)
    ///
    /// A separate function as calls are usually just stepped over.
    fn apply_call(&mut self, ret: Self::VirtualAddress);

    /// Adds an additonal assumption that can't be represented by setting registers/etc.
    /// Resolved constraints are useful limiting possible values a variable can have
    /// ([`value_limits`](Self::value_limits))
    fn add_resolved_constraint(&mut self, constraint: Constraint<'e>);
    /// Adds an additonal assumption that can't be represented by setting registers/etc.
    /// Unresolved constraints are useful for knowing that a jump chain such as `jg` followed by
    /// `jle` ends up always jumping at `jle`.
    fn add_unresolved_constraint(&mut self, constraint: Constraint<'e>);
    fn add_resolved_constraint_from_unresolved(&mut self);

    /// Returns smallest and largest (inclusive) value a *resolved* `Operand` can have.
    ///
    /// This usually just returns (0, u64::MAX), but the state may be able to give
    /// better guess if a jump earlier has limited the range of possible values
    /// for the `Operand`.
    fn value_limits(&mut self, _value: Operand<'e>) -> (u64, u64) {
        (0, u64::MAX)
    }

    /// Merges two states to a new state, as described in
    /// [`FuncAnalysis documentation`](../analysis/struct.FuncAnalysis.html#state-merging-and-loops).
    ///
    /// Returns `None` if the merged state would be equivalent to `old`,
    /// otherwise the merged state is returned.
    fn merge_states(
        old: &mut Self,
        new: &mut Self,
        cache: &mut MergeStateCache<'e>,
    ) -> Option<Self>;

    /// Bit of abstraction leak, but the memory structure is implemented as an partially
    /// immutable hashmap to keep clones not getting out of hand. This function is used to
    /// tell memory that it may be cloned soon, so the latest changes may be made
    /// immutable-shared if necessary.
    fn maybe_convert_memory_immutable(&mut self, limit: usize);

    /// Tries to do reverse lookup to find some unresolved `Operand` for the
    /// resolved `val`.
    ///
    /// Does not provide good results; user code should implement unresolve
    /// that meets the accuracy/performance requirements they have itself.
    fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>>;

    /// Creates an `Mem[addr]` with MemAccessSize of VirtualAddress size.
    fn operand_mem_word(ctx: OperandCtx<'e>, address: Operand<'e>, offset: u64) -> Operand<'e> {
        if <Self::VirtualAddress as VirtualAddress>::SIZE == 4 {
            ctx.mem32(address, offset)
        } else {
            ctx.mem64(address, offset)
        }
    }

    // Analysis functions, default to no-op
    fn find_functions_with_callers(_file: &BinaryFile<Self::VirtualAddress>)
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
        _file: &BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<(u32, u32)>, crate::OutOfBounds> {
        Ok(Vec::new())
    }

    fn find_relocs(
        _file: &BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<Self::VirtualAddress>, crate::OutOfBounds> {
        Ok(Vec::new())
    }

    /// Equivalent to `out.write(self.clone())`, but may leave `out` partially
    /// overwritten if it panics.
    ///
    /// Useful for avoiding unnecessary memcpys.
    unsafe fn clone_to(&self, out: *mut Self);
}

/// Splits state in two, updating the one state with condition assumed be false
/// on jump, and the other state with condition assumed be true on jump.
///
/// Caller must make sure that `condition_resolved` == `state.resolve(condition)`
/// (Or resolve_apply_constraints).
///
/// Returns states in `(jump, no_jump)` order.
pub(crate) fn assume_jump_flag<'e, E>(
    state: E,
    condition: Operand<'e>,
    condition_resolved: Operand<'e>,
) -> (E, E)
where E: ExecutionState<'e>,
{
    let mut no_jump_state = state.clone();
    let mut jump_state = state;
    let ty = condition.ty();
    if let OperandType::Arithmetic(arith) = ty {
        if matches!(arith.ty, ArithOpType::Equal | ArithOpType::Or | ArithOpType::And) {
            let ctx = jump_state.ctx();

            let resolved_no_jump = ctx.eq_const(condition_resolved, 0);
            jump_state.add_resolved_constraint(Constraint::new(condition_resolved));
            no_jump_state.add_resolved_constraint(Constraint::new(resolved_no_jump));

            let mut do_unresolved_constraint = true;
            if arith.ty == ArithOpType::Equal {
                // Update relevant flag to 0/1 if reasonable, if not, then add constraint.
                if let OperandType::Flag(flag) = *arith.left.ty() {
                    if arith.right == ctx.const_0() {
                        jump_state.set_flag(flag, ctx.const_0());
                        no_jump_state.set_flag(flag, ctx.const_1());
                        do_unresolved_constraint = false;
                    }
                }
            }
            if do_unresolved_constraint {
                let unresolved_no_jump = ctx.eq_const(condition, 0);
                jump_state.add_unresolved_constraint(Constraint::new(condition));
                no_jump_state.add_unresolved_constraint(Constraint::new(unresolved_no_jump));
            }
        }
    } else if let OperandType::Flag(flag) = *ty {
        let ctx = jump_state.ctx();
        let resolved_no_jump = ctx.eq_const(condition_resolved, 0);
        jump_state.add_resolved_constraint(Constraint::new(condition_resolved));
        no_jump_state.add_resolved_constraint(Constraint::new(resolved_no_jump));
        jump_state.set_flag(flag, ctx.const_1());
        no_jump_state.set_flag(flag, ctx.const_0());
    }
    (jump_state, no_jump_state)
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

#[derive(Copy, Clone)]
pub(crate) enum FreezeOperation<'e> {
    Move(DestOperand<'e>, Operand<'e>),
    // Separate operand is carry, for adc/sbb.
    SetFlags(FlagUpdate<'e>, Option<Operand<'e>>),
}

pub(crate) fn flag_arith_to_op_arith(val: FlagArith) -> Option<ArithOpType> {
    static MAPPING: [Option<ArithOpType>; 12] = {
        let mut out = [None; 12];
        out[FlagArith::Add as usize] = Some(ArithOpType::Add);
        out[FlagArith::Adc as usize] = Some(ArithOpType::Add);
        out[FlagArith::Sub as usize] = Some(ArithOpType::Sub);
        out[FlagArith::Sbb as usize] = Some(ArithOpType::Sub);
        out[FlagArith::And as usize] = Some(ArithOpType::And);
        out[FlagArith::Or as usize] = Some(ArithOpType::Or);
        out[FlagArith::Xor as usize] = Some(ArithOpType::Xor);
        out[FlagArith::LeftShift as usize] = Some(ArithOpType::Lsh);
        out[FlagArith::RightShift as usize] = Some(ArithOpType::Rsh);
        out[FlagArith::RotateLeft as usize] = None;
        out[FlagArith::RotateRight as usize] = None;
        out[FlagArith::RightShiftArithmetic as usize] = None;
        out
    };
    MAPPING[val as usize]
}

pub(crate) fn carry_for_add_sub<'e>(
    ctx: OperandCtx<'e>,
    arith: &FlagUpdate<'e>,
    result: Operand<'e>,
    result_with_carry: Operand<'e>,
) -> Operand<'e> {
    use crate::disasm::FlagArith::*;

    let size = arith.size;
    let mask = size.mask();
    let left = ctx.and_const(arith.left, mask);
    let result = ctx.and_const(result, mask);
    if arith.ty == Add {
        ctx.gt(left, result)
    } else if arith.ty == Sub {
        ctx.gt(result, left)
    } else if arith.ty == Adc {
        // carry = (left > left + right) | (left > result)
        let result_with_carry = ctx.and_const(result_with_carry, mask);
        ctx.or(
            ctx.gt(left, result),
            ctx.gt(left, result_with_carry),
        )
    } else {
        // Sbb
        // carry = (left - right > left) | (result > left)
        let result_with_carry = ctx.and_const(result_with_carry, mask);
        ctx.or(
            ctx.gt(result, left),
            ctx.gt(result_with_carry, left),
        )
    }
}

pub(crate) fn overflow_for_add_sub<'e>(
    ctx: OperandCtx<'e>,
    arith: &FlagUpdate<'e>,
    result: Operand<'e>,
    result_with_carry: Operand<'e>,
) -> Operand<'e> {
    use crate::disasm::FlagArith::*;

    let size = arith.size;
    let mask = size.mask();
    let sign_bit = (mask >> 1).wrapping_add(1);
    let left = ctx.and_const(arith.left, mask);
    let right = ctx.and_const(arith.right, mask);
    let result = ctx.and_const(result, mask);
    if arith.ty == Add {
        // (right sge 0) == (left sgt result)
        ctx.eq(
            ctx.gt_const_left(sign_bit, right),
            ctx.gt_signed(left, result, size),
        )
    } else if arith.ty == Sub {
        // (right sge 0) == (result sgt left)
        ctx.eq(
            ctx.gt_const_left(sign_bit, right),
            ctx.gt_signed(result, left, size),
        )
    } else if arith.ty == Adc {
        // overflow = (right sge 0) == ((left sgt left + right) | (left sgt result))
        let result_with_carry = ctx.and_const(result_with_carry, mask);
        ctx.eq(
            ctx.gt_const_left(sign_bit, right),
            ctx.or(
                ctx.gt_signed(left, result, size),
                ctx.gt_signed(left, result_with_carry, size),
            ),
        )
    } else {
        // Sbb
        // overflow = (right sge 0) == ((left - right sgt left) | (result sgt left))
        let result_with_carry = ctx.and_const(result_with_carry, mask);
        ctx.eq(
            ctx.gt_const_left(sign_bit, right),
            ctx.or(
                ctx.gt_signed(result, left, size),
                ctx.gt_signed(result_with_carry, left, size),
            ),
        )
    }
}

/// Calculates x86 parity flag value, that is, 1 if number of set bits on low u8 is even,
/// and 0 if it is odd.
pub(crate) fn calculate_parity<'e>(ctx: OperandCtx<'e>, mut value: Operand<'e>) -> Operand<'e> {
    // Parity can be expressed as
    // 1 ^ (x & 1) ^ ((x >> 1) & 1) ^ ...
    // or alternatively, shorter as
    // a = ((x >> 4) ^ x)
    // b = ((a >> 2) ^ a)
    // result = ((b >> 1) ^ b)) == 0
    // Do even slightly faster version where temp values can be avoided when
    // relevant_bits is less than 8

    if let Some(c) = value.if_constant() {
        return match (c as u8).count_ones() & 1 == 0 {
            true => ctx.const_1(),
            false => ctx.const_0(),
        }
    }
    let mut rel_bits = value.relevant_bits();
    if rel_bits.start >= 8 {
        return ctx.const_1();
    }
    if rel_bits.end > 8 {
        value = ctx.and_const(value, 0xff);
        rel_bits.end = 8;
    }
    if rel_bits.start != 0 {
        value = ctx.rsh_const(value, rel_bits.start as u64);
    }
    while rel_bits.end > 1 {
        // 2 bit => 1
        // 3 bit => 2
        // 4 bit => 2
        // 5 bit => 3
        // 6 bit => 3
        // 7 bit => 4
        // 8 bit => 4
        let shift = rel_bits.end.wrapping_sub(rel_bits.start).wrapping_add(1) >> 1;
        value = ctx.xor(
            ctx.rsh_const(
                value,
                shift as u64,
            ),
            ctx.and_const(
                value,
                (1u32 << shift).wrapping_sub(1) as u64,
            )
        );
        rel_bits = value.relevant_bits();
    }
    if rel_bits.end == 0 {
        return ctx.const_1();
    }
    ctx.eq_const(
        value,
        0,
    )
}

pub(crate) struct FlagState<'a, 'e> {
    pub flags: &'a mut [Operand<'e>; 6],
    pub pending_flags: &'a mut PendingFlags<'e>,
}


fn operand_merge_eq<'e>(a: Operand<'e>, b: Operand<'e>) -> bool {
    a == b || a.is_undefined()
}

pub(crate) fn flags_merge_changed<'a, 'b, 'e>(
    old: &mut FlagState<'a, 'e>,
    new: &mut FlagState<'b, 'e>,
    ctx: OperandCtx<'e>,
) -> bool {
    // Direction
    if !operand_merge_eq(old.flags[5], new.flags[5]) {
        return true;
    }
    let mut checked_flags = 0u8;
    let mut check_pending_result_eq = false;
    if let Some(ref old_update) = old.pending_flags.update {
        if let Some(ref new_update) = new.pending_flags.update {
            if old_update == new_update && old.pending_flags.carry == new.pending_flags.carry {
                // Pending flags equal, no need to check any flags that are still pending
                // or were resolved from these.
                checked_flags |= old.pending_flags.flag_bits & new.pending_flags.flag_bits;
            } else {
                check_pending_result_eq = true;
            }
        }
    }
    if check_pending_result_eq {
        // If the flag inputs weren't equal, check if their results are
        if old.pending_flags.get_result(ctx).map(|x| x.result) ==
            new.pending_flags.get_result(ctx).map(|x| x.result)
        {
            checked_flags |= old.pending_flags.flag_bits & new.pending_flags.flag_bits;
        } else {
            // At least one flag in old is pending and result doesn't match,
            // it has changed. New being pending doesn't matter necessarily as
            // old may be undef
            if old.pending_flags.pending_bits != 0 {
                return true;
            }
        }
    }
    if checked_flags != 0x1f {
        for i in 0..5 {
            let mask = 1 << i;
            if checked_flags & mask == 0 {
                if old.pending_flags.pending_bits & mask != 0 {
                    return true;
                }
                if old.flags[i].is_undefined() {
                    continue;
                }
                if new.pending_flags.pending_bits & mask != 0 {
                    return true;
                }
                if new.flags[i] != old.flags[i] {
                    return true;
                }
            }
        }
    }
    false
}

pub(crate) fn merge_flags<'a, 'b, 'e>(
    old: &mut FlagState<'a, 'e>,
    new: &mut FlagState<'b, 'e>,
    out_flags: &mut [Operand<'e>; 6],
    ctx: OperandCtx<'e>,
) -> PendingFlags<'e> {
    if old.flags[5] == new.flags[5] {
        out_flags[5] = old.flags[5];
    } else {
        out_flags[5] = ctx.new_undef();
    }
    let mut flag_bits = 0u8;
    let mut pending_bits = 0u8;
    let mut check_pending_result_eq = false;
    let mut out_pending_flags = PendingFlags::new();
    if let Some(ref old_update) = old.pending_flags.update {
        if let Some(ref new_update) = new.pending_flags.update {
            if old_update == new_update && old.pending_flags.carry == new.pending_flags.carry {
                // Pending flags equal, no need to check any flags that are still pending
                // or were resolved from these.
                flag_bits = old.pending_flags.flag_bits & new.pending_flags.flag_bits;
                pending_bits = flag_bits &
                    old.pending_flags.pending_bits & new.pending_flags.pending_bits;
                out_pending_flags = *old.pending_flags;
            } else {
                check_pending_result_eq = true;
            }
        }
    }
    if check_pending_result_eq {
        // If the flag inputs weren't equal, check if their results are
        if old.pending_flags.get_result(ctx).map(|x| x.result) ==
            new.pending_flags.get_result(ctx).map(|x| x.result)
        {
            flag_bits = old.pending_flags.flag_bits & new.pending_flags.flag_bits;
            pending_bits = flag_bits &
                old.pending_flags.pending_bits & new.pending_flags.pending_bits;
            out_pending_flags = *old.pending_flags;
        }
    }
    out_pending_flags.flag_bits = flag_bits;
    out_pending_flags.pending_bits = pending_bits;
    if pending_bits != 0x1f {
        for i in 0..5 {
            let mask = 1 << i;
            if pending_bits & mask == 0 {
                if flag_bits & mask != 0 {
                    // This flag is in common flags but wasn't pending, so one of the
                    // old/new should have it calculated
                    if old.pending_flags.pending_bits & mask == 0 {
                        // Old has result calculated
                        out_flags[i] = old.flags[i];
                    } else {
                        // Otherwise it should be in new always
                        debug_assert!(new.pending_flags.pending_bits & mask == 0);
                        out_flags[i] = new.flags[i];
                    }
                } else {
                    let old_flag = old.flags[i];
                    if old.pending_flags.pending_bits & mask != 0 {
                        out_flags[i] = ctx.new_undef();
                    } else if old_flag.is_undefined() {
                        out_flags[i] = old_flag;
                    } else if new.pending_flags.pending_bits & mask != 0 ||
                        new.flags[i] != old_flag
                    {
                        out_flags[i] = ctx.new_undef();
                    } else {
                        out_flags[i] = old_flag;
                    }
                }
            }
        }
    }
    out_pending_flags
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PendingFlags<'e> {
    pub update: Option<FlagUpdate<'e>>,
    pub carry: Option<Operand<'e>>,
    pub result: Option<Option<PendingFlagsResult<'e>>>,
    /// Before reading flags, if this has a bit set, it means that the flag has to
    /// be calculated through pending_flags.
    pub pending_bits: u8,
    /// Which flags the current update applies to.
    /// Superset of pending_bits
    pub flag_bits: u8,
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct PendingFlagsResult<'e> {
    pub base_result: Operand<'e>,
    pub result: Operand<'e>,
}

impl<'e> PendingFlags<'e> {
    pub fn new() -> PendingFlags<'e> {
        PendingFlags {
            update: None,
            carry: None,
            result: None,
            pending_bits: 0,
            flag_bits: 0,
        }
    }

    pub fn reset(&mut self, update: &FlagUpdate<'e>, carry: Option<Operand<'e>>) {
        self.update = Some(*update);
        self.carry = carry;
        self.result = None;
        self.pending_bits = 0x1f;
        self.flag_bits = 0x1f;
    }

    pub fn is_pending(&self, flag: Flag) -> bool {
        self.pending_bits & (1 << flag as u32) != 0
    }

    pub fn clear_pending(&mut self, flag: Flag) {
        self.pending_bits &= !(1 << flag as u32);
    }

    pub fn make_non_pending(&mut self, flag: Flag) {
        self.pending_bits &= !(1 << flag as u32);
        self.flag_bits &= !(1 << flag as u32);
    }

    pub fn get_result(&mut self, ctx: OperandCtx<'e>) -> Option<PendingFlagsResult<'e>> {
        let update = &self.update;
        let carry = self.carry;
        *self.result.get_or_insert_with(|| {
            let arith = update.as_ref()?;
            let size = arith.size;
            if arith.left == arith.right && arith.ty == FlagArith::And {
                // Fast path for test x,x
                // As exec_state sub_fast_result already handles the most common
                // carry/zero for cmp, test x,x here is pretty common.
                let result = if arith.left.relevant_bits().end as u32 > size.bits() {
                    ctx.and_const(arith.left, size.mask())
                } else {
                    arith.left
                };
                return Some(PendingFlagsResult {
                    result,
                    base_result: result,
                });
            }
            let arith_ty = flag_arith_to_op_arith(arith.ty)?;

            let base_result =
                ctx.arithmetic_masked(arith_ty, arith.left, arith.right, size.mask());
            let mut result = if arith.ty == FlagArith::Adc {
                ctx.add(base_result, carry.unwrap_or_else(|| ctx.const_0()))
            } else if arith.ty == FlagArith::Sbb {
                ctx.sub(base_result, carry.unwrap_or_else(|| ctx.const_0()))
            } else {
                base_result
            };
            if result != base_result && size != MemAccessSize::Mem64 {
                result = ctx.and_const(result, size.mask());
            }
            Some(PendingFlagsResult {
                result,
                base_result,
            })
        })
    }

    /// Special case carry(x - y) => x < y, and zero(x - y) => x == y
    /// as those will be majority of flag assignments through cmp instructions,
    /// and avoiding creation of `x - y` operand followed by simplification
    /// to carry/zero ends up being faster.
    /// Overflow could probably be done too, not doing it for now.
    pub fn sub_fast_result(&self, ctx: OperandCtx<'e>, flag: Flag) -> Option<Operand<'e>> {
        let arith = self.update.as_ref()?;
        if arith.ty != FlagArith::Sub {
            return None;
        }

        if matches!(flag, Flag::Carry | Flag::Zero) {
            let mut left = arith.left;
            let mut right = arith.right;
            let mask_bits = arith.size.bits() as u8;
            if left.relevant_bits().end > mask_bits {
                left = ctx.and_const(left, arith.size.mask());
            }
            if right.relevant_bits().end > mask_bits {
                right = ctx.and_const(right, arith.size.mask());
            }
            if flag == Flag::Carry {
                return Some(ctx.gt(right, left));
            } else {
                // Zero
                return Some(ctx.eq(left, right));
            }
        }
        None
    }
}

/// Function to determine which flags add_resolved_constraint should consider checking
/// for equaling constraint. As the exec states delay realizing flags, checking all
/// flags unconditionally causes unnecessary work.
///
/// This is only returning flags that are may be set to one from assume_jump_flag
/// calling add_resolved_constraint. Trying to get good trade-off between work and
/// results.
pub(crate) fn flags_for_resolved_constraint_eq_check<'e>(
    flag_update: &FlagUpdate<'e>,
    arith: &ArithOperand<'e>,
    ctx: OperandCtx<'e>,
) -> &'static [Flag] {
    // This could return carry when arith is GreaterThan, or
    // Equal with FlagUpdate +/- 1/max, but hasn't been useful yet.
    let is_sign = arith.right == ctx.const_0() &&
        arith.left.if_arithmetic_eq()
            .filter(|x| x.1 == ctx.const_0())
            .and_then(|x| x.0.if_arithmetic_and()?.1.if_constant())
            .filter(|x| {
                matches!(x, 0x80 | 0x8000 | 0x8000_0000 | 0x8000_0000_0000_0000)
            })
            .is_some();
    if is_sign {
        return &[Flag::Sign, Flag::Zero];
    } else {
        if arith.right == ctx.const_0() ||
            matches!(flag_update.ty, FlagArith::Add | FlagArith::Sub | FlagArith::And)
        {
            return &[Flag::Zero];
        }
    }
    &[]
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
        let result = remove_matching_ands_memory(ctx, self.0);
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
                remove_matching_ands(ctx, self.0, ctx.register(reg))
            }
            DestOperand::Xmm(_, _) => {
                None
            }
            DestOperand::Fpu(_) => {
                None
            }
            DestOperand::Flag(flag) => {
                remove_matching_ands(ctx, self.0, ctx.flag(flag))
            },
            DestOperand::Memory(_) => {
                // Assuming that everything may alias with memory
                remove_matching_ands_memory(ctx, self.0)
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
fn remove_matching_ands<'e>(
    ctx: OperandCtx<'e>,
    oper: Operand<'e>,
    compare: Operand<'e>,
) -> Option<Operand<'e>> {
    let mut state = InvalidateConstraintState::new(oper, compare);
    if oper.relevant_bits() != (0..1) || oper.if_arithmetic_and().is_none() {
        // Only going to try partially invalidate cases with logical ands
        if state.check_remove(oper) == ConstraintRemoveCheckResult::Keep {
            return Some(oper);
        } else {
            return None;
        }
    }
    let mut next = Some(oper);
    while let Some(x) = next {
        let part;
        if let Some((l, r)) = x.if_arithmetic_and() {
            next = Some(l);
            part = r;
        } else {
            next = None;
            part = x;
        }
        if !state.check_and_part(part) {
            return None;
        }
    }
    state.build_and(ctx)
}

struct InvalidateConstraintState<'e> {
    build_and_buffer: ArrayVec<Operand<'e>, 8>,
    iter_buffer: ArrayVec<Operand<'e>, 16>,
    ops_checked: u32,
    // Allows skipping and building if the operand doesn't change
    all_checks_ok: bool,
    compare: Operand<'e>,
    oper: Operand<'e>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum ConstraintRemoveCheckResult {
    Remove,
    Keep,
    Limit,
}

impl<'e> InvalidateConstraintState<'e> {
    fn new(oper: Operand<'e>, compare: Operand<'e>) -> InvalidateConstraintState<'e> {
        InvalidateConstraintState {
            all_checks_ok: true,
            ops_checked: 0,
            compare,
            oper,
            iter_buffer: ArrayVec::new(),
            build_and_buffer: ArrayVec::new(),
        }
    }

    /// Returns true if the part should be invalidated, false if not
    fn check_remove(&mut self, op: Operand<'e>) -> ConstraintRemoveCheckResult {
        self.iter_buffer.clear();
        self.iter_buffer.push(op);
        while let Some(next) = self.iter_buffer.pop() {
            let mut part = next;
            'inner_loop: loop {
                if part == self.compare {
                    return ConstraintRemoveCheckResult::Remove;
                }
                if self.ops_checked >= 64 {
                    return ConstraintRemoveCheckResult::Limit;
                }
                self.ops_checked = self.ops_checked + 1;
                match *part.ty() {
                    OperandType::Arithmetic(ref arith) |
                        OperandType::ArithmeticFloat(ref arith, _) =>
                    {
                        part = arith.right;
                        if self.iter_buffer.is_full() {
                            return ConstraintRemoveCheckResult::Limit;
                        }
                        self.iter_buffer.push(arith.left);
                    },
                    OperandType::Memory(ref m) => {
                        part = m.address().0;
                    }
                    OperandType::SignExtend(val, _, _) => {
                        part = val;
                    }
                    _ => break 'inner_loop,
                }
            }
        }
        ConstraintRemoveCheckResult::Keep
    }

    /// Returns false if entire constraint should be invalidated
    fn check_and_part(&mut self, part: Operand<'e>) -> bool {
        match self.check_remove(part) {
            ConstraintRemoveCheckResult::Remove => {
                self.all_checks_ok = false;
                true
            }
            ConstraintRemoveCheckResult::Keep => {
                if self.build_and_buffer.is_full() {
                    false
                } else {
                    self.build_and_buffer.push(part);
                    true
                }
            }
            ConstraintRemoveCheckResult::Limit => false,
        }
    }

    fn build_and(&self, ctx: OperandCtx<'e>) -> Option<Operand<'e>> {
        if self.all_checks_ok {
            return Some(self.oper);
        }
        let mut result = match self.build_and_buffer.get(0) {
            Some(&s) => s,
            None => return None,
        };
        for &op in &self.build_and_buffer[1..] {
            result = ctx.and(result, op);
        }
        Some(result)
    }
}

/// Constraint invalidation helper; Removes parts of constraint referring to memory
fn remove_matching_ands_memory<'e>(
    ctx: OperandCtx<'e>,
    oper: Operand<'e>,
) -> Option<Operand<'e>> {
    if !oper.contains_memory() {
        return Some(oper);
    }
    if oper.relevant_bits() != (0..1) {
        // Only going to try partially invalidate cases with logical ands
        return None;
    }
    let mut next = Some(oper);
    let mut result = None;
    while let Some(x) = next {
        let part;
        if let Some((l, r)) = x.if_arithmetic_and() {
            next = Some(l);
            part = r;
        } else {
            next = None;
            part = x;
        }
        if !part.contains_memory() {
            if let Some(old) = result {
                result = Some(ctx.and(old, part));
            } else {
                result = Some(part);
            }
        }
    }
    result
}

fn other_if_eq_zero<'e>(op: Operand<'e>) -> Option<Operand<'e>> {
    op.if_arithmetic_eq()
        .filter(|(_, r)| r.if_constant() == Some(0))
        .map(|(l, _)| l)
}

/// Splits the constraint at ands that can be applied separately
/// Also can handle logical nots
fn apply_constraint_split<'e>(
    ctx: OperandCtx<'e>,
    constraint: Operand<'e>,
    val: Operand<'e>,
    with: bool,
) -> Operand<'e> {
    let zero = ctx.const_0();
    let one = ctx.const_1();
    match constraint.ty() {
        OperandType::Arithmetic(arith) if {
            ((arith.ty == ArithOpType::And && with == true) ||
                (arith.ty == ArithOpType::Or && with == false)) &&
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
                let (cmp, subst_1) = other_if_eq_zero(val)
                    .map(|x| (x, true))
                    .unwrap_or_else(|| (val, false));
                if let Some((l, r)) = cmp.if_arithmetic_and() {
                    if (l == x && r == y) || (l == y && r == x) {
                        return if subst_1 { one } else { zero };
                    }
                }
            }
            let subst_val = if with { one } else { zero };
            ctx.substitute(val, constraint, subst_val, 6)
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Equal => {
            if arith.right == zero {
                if arith.left.relevant_bits() == (0..1) {
                    apply_constraint_split(ctx, arith.left, val, !with)
                } else if with == true {
                    // (x | y) == 0 evaluates to 1, means that both x and y must be 0
                    let mut pos = arith.left;
                    let mut result = val;
                    while let Some((l, r)) = pos.if_arithmetic_or() {
                        result = apply_constraint_split(ctx, r, result, false);
                        pos = l;
                    }
                    apply_constraint_split(ctx, pos, result, false)
                } else {
                    // (arith.left == 0) is false, that is, arith.left != 0
                    // Transform any check of arith.left == 0 or (arith.left | x) == 0
                    // to 0
                    ctx.transform(val, 6, |op| {
                        let (l, r) = op.if_arithmetic_eq()?;
                        if r != zero {
                            return None;
                        }
                        let mut op = l;
                        if op == arith.left {
                            return Some(zero);
                        }
                        loop {
                            match op.if_arithmetic_or() {
                                Some((l, r)) => {
                                    if r == arith.left {
                                        return Some(zero);
                                    }
                                    op = l;
                                }
                                None => {
                                    if op == arith.left {
                                        return Some(zero);
                                    }
                                    break;
                                }
                            }
                        }
                        None
                    })
                }
            } else {
                let subst_val = if with { one } else { zero };
                ctx.substitute(val, constraint, subst_val, 6)
            }
        }
        _ => {
            let subst_val = if with { one } else { zero };
            ctx.substitute(val, constraint, subst_val, 6)
        }
    }
}

/// For constraint X, return Y:
/// Assumes that flags are 1bit, which isn't super set in stone elsewhere (yet)
///     flag == 0 => (flag, 0)
///     flag != 0 => (flag, 1)
///     flag == 1 => (flag, 1)
pub fn is_flag_const_constraint<'e>(
    ctx: OperandCtx<'e>,
    constraint: Operand<'e>,
) -> Option<(Flag, Operand<'e>)> {
    if let Some(flag) = constraint.if_flag() {
        return Some((flag, ctx.const_1()));
    }
    let (l, r) = constraint.if_arithmetic_eq()?;
    if let Some(flag) = l.if_flag() {
        if r == ctx.const_0() {
            return Some((flag, r));
        }
    }
    None
}

fn sign_extend(value: u64, from: MemAccessSize, to: MemAccessSize) -> u64 {
    if value > from.mask() / 2 {
        value | (to.mask() & !from.mask())
    } else {
        value
    }
}

/// Helper for ExecutionState::value_limits implementations with constraints
pub(crate) fn value_limits<'e>(constraint: Operand<'e>, value: Operand<'e>) -> (u64, u64) {
    let result = value_limits_recurse(constraint, value);
    if let Some((inner, from, to)) = value.if_sign_extend() {
        let inner_result = value_limits_recurse(constraint, inner);
        let inner_sext = (
            sign_extend(inner_result.0, from, to),
            sign_extend(inner_result.1, from, to),
        );
        (inner_sext.0.max(result.0), (inner_sext.1.min(result.1)))
    } else {
        result
    }
}

fn value_limits_recurse<'e>(constraint: Operand<'e>, value: Operand<'e>) -> (u64, u64) {
    match *constraint.ty() {
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let left = value_limits_recurse(arith.left, value);
                    let right = value_limits_recurse(arith.right, value);
                    return (left.0.max(right.0), (left.1.min(right.1)));
                }
                ArithOpType::GreaterThan => {
                    // 0 > x and x > u64_max should get simplified to 0
                    if let Some(c) = arith.left.if_constant() {
                        let (right_inner, right_mask) = Operand::and_masked(arith.right);
                        let (right, offset) = right_inner.if_sub_with_const()
                            .unwrap_or_else(|| (right_inner, 0));
                        if Operand::and_masked(value) == (right, right_mask) ||
                            (right_mask == u64::MAX && is_subset(value, right))
                        {
                            debug_assert!(c != 0);
                            let low = offset;
                            if let Some(high) = c.wrapping_sub(1).checked_add(offset) {
                                return (low, high);
                            }
                        }
                    }
                    if let Some(c) = arith.right.if_constant() {
                        if is_subset(value, arith.left) {
                            debug_assert!(c != u64::MAX);
                            return (c.wrapping_add(1), u64::MAX);
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
    (0, u64::MAX)
}

/// Returns true if sub == sup & some_const_mask (Eq is also fine)
/// Not really considering constants for this (value_limits_recurse)
fn is_subset<'e>(sub: Operand<'e>, sup: Operand<'e>) -> bool {
    if sub.ty().expr_size() > sup.ty().expr_size() {
        return false;
    }
    match sub.ty() {
        OperandType::Memory(mem) => match sup.if_memory() {
            Some(mem2) => mem.address() == mem2.address(),
            None => false,
        }
        _ => sub == sup,
    }
}

/// Trait for disassembling instructions
pub trait Disassembler<'e> {
    type VirtualAddress: VirtualAddress;
    /// Creates a new Disassembler.
    fn new(
        ctx: OperandCtx<'e>,
        binary: &'e BinaryFile<Self::VirtualAddress>,
        init_pos: Self::VirtualAddress,
    ) -> Self;
    /// Seeks to a address.
    fn set_pos(&mut self, address: Self::VirtualAddress) -> Result<(), ()>;
    fn next<'s>(&'s mut self) -> Instruction<'s, 'e, Self::VirtualAddress>;
    fn address(&self) -> Self::VirtualAddress;
}

/// Contains memory state as addr -> value hashmap.
/// The ExecutionState is expected to take care of cases where memory is written with one
/// address and part of it is read at offset address, in practice by splitting accesses
/// to be word-sized, and any misaligned accesses become bitwise and-or combinations.
#[derive(Clone)]
pub struct Memory<'e> {
    map: MemoryMap<'e>,
    /// Caches value of last read/write
    cached_addr: Option<(Operand<'e>, u64)>,
    cached_value: Option<Operand<'e>>,
}

#[derive(Clone)]
struct MemoryMap<'e> {
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
    map: Rc<MemoryMapTopLevel<'e>>,
    /// Optimization for cases where memory gets large.
    /// The existing mapping can be moved to Rc, where cloning it is effectively free.
    ///
    /// Every 4th immutable map gets previous 4 immutables merged together to one larger
    /// immutable map to make lookups that have to work through the chain of immutable maps
    /// faster.
    /// The non-merged chain is also kept avaiable to keep merges ideal.
    ///
    /// Example on how different maps point to each other:
    /// Numbers represent immutable_depth value.
    /// Going up/left means following immmutable, going right is following slower_immutable
    ///                      (0)
    ///                      (1)
    ///                      (2)
    ///              (3)---->(3)
    ///               |
    ///               |<-----(4)
    ///               |      (5)
    ///               |      (6)
    ///              (7)---->(7)
    ///               |
    ///               |<-----(8)
    ///               |      (9)
    ///               |      (10)
    ///              (11)--->(11)
    ///               |
    ///               |<-----(12)
    ///               |      (13)
    ///               |      (14)
    ///     (15)---->(15)--->(15)
    ///      |
    ///      |<--------------(16)
    ///      |        |      (17)
    ///      |        |      (18)
    ///      |       (19)--->(19)
    ///      |        |
    ///
    /// This layout allows memory.get() to just follow immutable, no matter which submap is
    /// used.
    immutable: Option<Rc<MemoryMap<'e>>>,
    slower_immutable: Option<Rc<MemoryMap<'e>>>,
    /// How long the immutable chain is.
    /// See above diagram for how it works with the merged maps
    immutable_depth: usize,
    /// Sum of map.len() of immutable map chain and own `map.len()`, only set when map
    /// is made immutable.
    immutable_length: usize,
    /// True if all values stored in `self.map` are `Undefined`.
    /// False does not guarantee that that there are not-undefined values though.
    all_undefined: bool,
}

type MemoryMapTopLevel<'e> = HashMap<(OperandHashByAddress<'e>, u64), Operand<'e>, FxBuildHasher>;

struct MemoryIterMapsUntilImm<'a, 'e> {
    map: Option<&'a MemoryMap<'e>>,
    limit: Option<&'a MemoryMap<'e>>,
}

impl<'e> Memory<'e> {
    pub fn new() -> Memory<'e> {
        Memory {
            map: MemoryMap {
                map: Rc::new(HashMap::with_hasher(Default::default())),
                immutable: None,
                slower_immutable: None,
                immutable_depth: 0,
                immutable_length: 0,
                all_undefined: true,
            },
            cached_addr: None,
            cached_value: None,
        }
    }

    pub fn get(&mut self, base: Operand<'e>, offset: u64) -> Option<Operand<'e>> {
        let address = (base, offset);
        if Some(address) == self.cached_addr {
            return self.cached_value;
        }
        let result = self.map.get(&(base.hash_by_address(), offset));
        self.cached_addr = Some(address);
        self.cached_value = result;
        result
    }

    pub fn set(&mut self, base: Operand<'e>, offset: u64, value: Operand<'e>) {
        let address = (base, offset);
        self.map.set((base.hash_by_address(), offset), value);
        self.cached_addr = Some(address);
        self.cached_value = Some(value);
    }

    /// Does a reverse lookup on last accessed memory address.
    ///
    /// Could probably cache like 4 previous accesses for slightly better results.
    /// *Unlike other Memory functions, this offset is returned in bytes, and requires
    /// caller to pass in word shift*
    pub(crate) fn fast_reverse_lookup(
        &self,
        ctx: OperandCtx<'e>,
        value: Operand<'e>,
        exec_state_mask: u64,
        word_shift: u32,
    ) -> Option<(Operand<'e>, u64, MemAccessSize)> {
        fn check_or_part<'e>(
            cached_addr: &(Operand<'e>, u64),
            part: Operand<'e>,
            value: Operand<'e>,
        ) -> Option<(Operand<'e>, u64, MemAccessSize)> {
            static BYTES_TO_MEM_SIZE: [MemAccessSize; 8] = [
                MemAccessSize::Mem8, MemAccessSize::Mem16,
                MemAccessSize::Mem32, MemAccessSize::Mem32,
                MemAccessSize::Mem64, MemAccessSize::Mem64,
                MemAccessSize::Mem64, MemAccessSize::Mem64,
            ];
            if part == value {
                let size_bytes = part.relevant_bits().end.wrapping_sub(1) / 8;
                let mem_size = BYTES_TO_MEM_SIZE[(size_bytes & 7) as usize];
                Some((cached_addr.0, cached_addr.1, mem_size))
            } else if let Some((l, r)) = part.if_arithmetic(ArithOpType::Lsh) {
                let offset = r.if_constant()?;
                let inner = check_or_part(cached_addr, l, value)?;
                let bytes = (offset as u8) >> 3 << 3;
                Some((inner.0, inner.1.wrapping_add(bytes as u64), inner.2))
            } else if let Some((l, r)) = part.if_arithmetic(ArithOpType::Or) {
                if l.relevant_bits().end < r.relevant_bits().start ||
                    r.relevant_bits().end < l.relevant_bits().start
                {
                    return None;
                }
                check_or_part(cached_addr, l, value)
                    .or_else(|| check_or_part(cached_addr, r, value))
            } else if let Some((l, r)) = part.if_arithmetic(ArithOpType::And) {
                let result = check_or_part(cached_addr, l, value)?;
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
                Some((result.0, result.1, masked_size))
            } else {
                None
            }
        }

        let (cached_base, cached_offset) = self.cached_addr?;
        let cached_offset = cached_offset << word_shift;
        let cached_value = self.cached_value?;

        let (value, mask) = Operand::and_masked(value);
        let cached = if exec_state_mask != u64::MAX {
            ctx.and_const(cached_value, exec_state_mask)
        } else {
            cached_value
        };
        let (cached, _) = Operand::and_masked(cached);
        let result = if value == cached {
            Some((cached_base, cached_offset, MemAccessSize::Mem64))
        } else if let Some((l, r)) = cached.if_arithmetic_or() {
            if l.relevant_bits().end < r.relevant_bits().start ||
                r.relevant_bits().end < l.relevant_bits().start
            {
                return None;
            }
            let cached_addr = (cached_base, cached_offset);
            check_or_part(&cached_addr, l, value)
                .or_else(|| check_or_part(&cached_addr, r, value))
        } else {
            None
        };
        result.map(|(op, offset, size)| {
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
                (op, offset, masked_size)
            } else {
                (op, offset, size)
            }
        })
    }

    /// Cheap check for if `self` and `other` share their hashmap.
    ///
    /// If true is returned, maps are equal; false can still mean
    /// that the maps are equal, it just can't be cheaply confirmed.
    pub fn is_same(&self, other: &Memory<'e>) -> bool {
        Rc::ptr_eq(&self.map.map, &other.map.map)
    }

    pub fn maybe_convert_immutable(&mut self, limit: usize) {
        self.map.maybe_convert_immutable(limit);
    }

    pub fn has_merge_changed(&self, b: &Memory<'e>) -> bool {
        self.map.has_merge_changed(&b.map)
    }
}

impl<'e> MemoryMap<'e> {
    pub fn get(&self, address: &(OperandHashByAddress<'e>, u64)) -> Option<Operand<'e>> {
        let op = self.map.get(address).cloned()
            .or_else(|| self.immutable.as_ref().and_then(|x| x.get(address)));
        op
    }

    fn get_no_immutable(&self, address: &(OperandHashByAddress<'e>, u64)) -> Option<Operand<'e>> {
        self.map.get(address).cloned()
    }

    pub fn set(&mut self, address: (OperandHashByAddress<'e>, u64), value: Operand<'e>) {
        // Don't insert a duplicate entry to mutable map if the immutable map already
        // has the correct key.
        //
        // Maybe avoiding making map mutable in the case where it doesn't contain
        // key and immutable has already correct value would be worth it?
        // It would avoid Rc::make_mut, but I feel that most of the time some other
        // instruction would require mutating th Rc immediately afterwards anyway.
        let map = Rc::make_mut(&mut self.map);
        if let Some(ref imm) = self.immutable {
            if imm.get(&address) == Some(value) {
                map.remove(&address);
                return;
            }
        }
        map.insert(address, value);
        self.all_undefined &= value.is_undefined();
    }

    fn is_empty(&self) -> bool {
        self.immutable.is_none() && self.map.is_empty()
    }

    /// The bool is true if the value is in immutable map
    fn get_with_immutable_info(
        &self,
        address: &(OperandHashByAddress<'e>, u64),
    ) -> Option<(Operand<'e>, bool)> {
        let op = self.map.get(address).map(|&x| (x, false))
            .or_else(|| {
                self.immutable.as_ref().and_then(|x| x.get(address)).map(|x| (x, true))
            });
        op
    }

    fn iter_maps_until_immutable<'a>(
        &'a self,
        limit: Option<&'a MemoryMap<'e>>,
    ) -> MemoryIterMapsUntilImm<'a, 'e> {
        match limit {
            Some(s) => if ptr::eq(self, s) {
                return MemoryIterMapsUntilImm {
                    map: None,
                    limit,
                };
            },
            None => (),
        };

        let pos = Self::slower_immutable_to_limit(self, limit);
        MemoryIterMapsUntilImm {
            map: Some(pos),
            limit,
        }
    }

    /// Goes through `map.slower_immutable` as much as needed to make sure that
    /// `limit` will be reachable from returned value
    fn slower_immutable_to_limit<'a>(
        map: &'a MemoryMap<'e>,
        limit: Option<&'a MemoryMap<'e>>,
    ) -> &'a MemoryMap<'e> {
        // Walk through slower_immutable if they exist to find the map that
        // doesn't go past limit
        let limit_depth = limit.map(|x| x.immutable_depth).unwrap_or(0);
        let mut pos = map;
        loop {
            let pos_depth = pos.immutable.as_ref().map(|x| x.immutable_depth).unwrap_or(0);
            if pos_depth >= limit_depth {
                return pos;
            }
            pos = match pos.slower_immutable.as_ref() {
                Some(s) => s,
                None => return pos,
            };
        }
    }

    fn common_immutable<'a>(&'a self, other: &'a MemoryMap<'e>) -> Option<&'a MemoryMap<'e>> {
        // The code afterwards doesn't work with some still-mutable maps
        // (E.g. immutable_depth == 3 but slower_immutable == None)
        // Still return a valid result if for some reason self.common_immutable(self)
        // is done, but then move to immutable if a map is mutable
        // Mutable maps where depth doesn't require slower_immutable are fine.
        if Rc::ptr_eq(&self.map, &other.map) {
            return Some(self);
        }
        let a = match self.immutable_depth & 0x3 == 0x3 && self.slower_immutable.is_none() {
            true => self.immutable.as_deref()?,
            false => self,
        };
        let b = match other.immutable_depth & 0x3 == 0x3 && other.slower_immutable.is_none() {
            true => other.immutable.as_deref()?,
            false => other,
        };

        let (mut greater, mut smaller) = match a.immutable_depth > b.immutable_depth {
            true => (a, b),
            false => (b, a),
        };

        while greater.immutable_depth != smaller.immutable_depth {
            let mut next = greater.immutable.as_ref();
            let mut next_depth = next.map(|x| x.immutable_depth).unwrap_or(0);
            while next_depth < smaller.immutable_depth {
                greater = greater.slower_immutable.as_ref()?;
                next = greater.immutable.as_ref();
                next_depth = next.map(|x| x.immutable_depth).unwrap_or(0);
            }
            greater = next?;
        }
        let mut prev_g_slow: Option<&Rc<MemoryMap<'e>>> = None;
        let mut prev_s_slow: Option<&Rc<MemoryMap<'e>>> = None;
        loop {
            if Rc::ptr_eq(&greater.map, &smaller.map) {
                // Consider two maps shaped like this which join at (5) and lower.
                //                      (0)
                //                      (1)
                //                      (2)
                //              (3)---->(3)
                //               |
                //               |<-----(4)
                //               |      (5)
                //               |      (6)
                //              (7)---->(7)
                // Since this loop has been walking through fast path, (7) isn't equal
                // while (3) is, but we'd want to return (5). So if slow paths exist,
                // descend into the slower path from last non-eq and keep looping
                if let Some(prev_g_slow) = prev_g_slow {
                    greater = &*prev_g_slow;
                    // Expected to be Some always too
                    smaller = &*prev_s_slow?;
                } else {
                    return Some(greater);
                }
            }
            prev_g_slow = greater.slower_immutable.as_ref();
            prev_s_slow = smaller.slower_immutable.as_ref();
            match greater.immutable.as_ref() {
                Some(s) => {
                    greater = s;
                    smaller = smaller.immutable.as_ref()?;
                }
                None => {
                    greater = prev_g_slow?;
                    prev_g_slow = greater.slower_immutable.as_ref();
                    smaller = prev_s_slow?;
                    prev_s_slow = smaller.slower_immutable.as_ref();
                }
            };
        }
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
        let immutable_depth = self.immutable_depth;
        let all_undefined = self.all_undefined;
        let immutable_length = old_immutable.as_ref().map(|x| x.immutable_length).unwrap_or(0) +
            map.len();
        self.immutable = Some(Rc::new(MemoryMap {
            map,
            immutable: old_immutable,
            slower_immutable: None,
            immutable_depth,
            immutable_length,
            all_undefined,
        }));
        self.all_undefined = true;
        self.immutable_depth = immutable_depth.wrapping_add(1);
        // Merge 4 immutables to one larger map, another layer if we're at 16
        // immutables etc.
        let mut n = 4usize;
        while self.immutable_depth & n.wrapping_sub(1) == 0 {
            // Calculate sum of 4 immutable map sizes
            let mut sum = 0usize;
            let mut pos = self.immutable.as_ref();
            for _ in 0..4 {
                match pos {
                    Some(next) => {
                        sum = sum.wrapping_add(next.map.len());
                        pos = next.immutable.as_ref();
                    }
                    None => break,
                }
            }
            // Actually build the map
            let mut merged_map = HashMap::with_capacity_and_hasher(sum, Default::default());
            let mut pos = self.immutable.as_ref();
            let mut all_undefined = true;
            for _ in 0..4 {
                match pos {
                    Some(next) => {
                        for (&k, &val) in next.map.iter() {
                            merged_map.entry(k).or_insert_with(|| val);
                        }
                        all_undefined &= next.all_undefined;
                        pos = next.immutable.as_ref();
                    }
                    None => break,
                }
            }
            let immutable = pos.cloned();
            let slower_immutable = self.immutable.take();
            let immutable_length = immutable.as_ref().map(|x| x.immutable_length).unwrap_or(0) +
                merged_map.len();
            self.immutable = Some(Rc::new(MemoryMap {
                map: Rc::new(merged_map),
                immutable,
                slower_immutable,
                immutable_depth,
                immutable_length,
                all_undefined,
            }));
            n = n << 2;
        }
    }

    fn merge(&self, new: &MemoryMap<'e>, ctx: OperandCtx<'e>) -> MemoryMap<'e> {
        // Merging memory is defined as:
        // If new & old values match, then the value is kept
        // Otherwise:
        //   If the address contains undefined, then it *may* be forgotten entirely
        //   If old is undefined, keep old value
        //   If old isn't undefined, but new is, *may* use new's value
        //      This is different from other exec state merging, but allows nicer perf
        //      when old is small and new is large with a lot of undefined
        //      Not sure if the "*may*" is ever false right now, ideally nobody
        //      should try relying on that.
        //   Otherwise generate a new undefined (obviously)
        let self_imm_len = self.immutable.as_ref().map(|x| x.immutable_length).unwrap_or(0);
        let new_imm_len = new.immutable.as_ref().map(|x| x.immutable_length).unwrap_or(0);
        let (a, b) = if new_imm_len > self_imm_len.wrapping_add(100) {
            // Take considerably larger map as a base if either is.
            // In general if the map sizes differ a lot, the larger one probably has been merged
            // already and contains undefined values which don't have to be recreated if it is
            // used as base.
            //
            // Maybe tracking amount of undefined values instead of all values
            // would be better guess though?
            (new, self)
        } else if self_imm_len > new_imm_len.wrapping_add(100) {
            (self, new)
        } else {
            // Take one with smaller immutable depth otherwise as base
            if self.immutable_depth > new.immutable_depth {
                (new, self)
            } else {
                (self, new)
            }
        };
        if Rc::ptr_eq(&a.map, &b.map) {
            return a.clone();
        }
        let mut result = HashMap::with_capacity_and_hasher(
            a.map.len().max(b.map.len()),
            Default::default(),
        );
        let a_empty = a.is_empty();
        let b_empty = b.is_empty();
        let result_immutable = a.immutable.clone();
        let result_immutable_len =
            result_immutable.as_ref().map(|x| x.immutable_length).unwrap_or(0);
        let slower_immutable = a.slower_immutable.clone();
        if (a_empty || b_empty) && a.immutable.is_none() && b.immutable.is_none() {
            let other = if a_empty { b } else { a };
            for (&key, &val) in other.map.iter() {
                if val.is_undefined() {
                    result.insert(key, val);
                } else {
                    result.insert(key, ctx.new_undef());
                }
            }
            let immutable_length = result_immutable_len + result.len();
            return MemoryMap {
                map: Rc::new(result),
                immutable: result_immutable,
                slower_immutable,
                immutable_depth: a.immutable_depth,
                immutable_length,
                all_undefined: true,
            };
        }
        let imm_eq = a.immutable.as_ref().map(|x| &**x as *const MemoryMap) ==
            b.immutable.as_ref().map(|x| &**x as *const MemoryMap);
        let result = if imm_eq {
            // Allows just checking a.map.iter() instead of a.iter()
            let mut all_undefined = true;
            for (key, &a_val) in a.map.iter() {
                if a_val.is_undefined() {
                    result.insert(*key, a_val);
                } else {
                    if let Some((b_val, is_imm)) = b.get_with_immutable_info(key) {
                        match a_val == b_val {
                            true => {
                                if !is_imm {
                                    all_undefined &= a_val.is_undefined();
                                    result.insert(*key, a_val);
                                }
                            }
                            false => {
                                if b_val.is_undefined() {
                                    result.insert(*key, b_val);
                                } else {
                                    result.insert(*key, ctx.new_undef());
                                }
                            }
                        }
                    } else {
                        result.insert(*key, ctx.new_undef());
                    }
                }
            }
            'b_loop: for (key, &b_val) in b.map.iter() {
                // This seems to be slightly faster than using entry()...
                // Maybe it's just something with different inlining decisions
                // that won't actually always be better, but it seems consistent
                // enough that I'm leaving this as is.
                if !result.contains_key(key) {
                    let val = if b_val.is_undefined() {
                        b_val
                    } else {
                        if let Some((a_val, is_imm)) = a.get_with_immutable_info(key) {
                            if is_imm && a_val == b_val {
                                continue 'b_loop;
                            }
                        }
                        ctx.new_undef()
                    };
                    result.insert(*key, val);
                }
            }
            let immutable_length = result_immutable_len + result.len();
            MemoryMap {
                map: Rc::new(result),
                immutable: result_immutable,
                slower_immutable,
                immutable_depth: a.immutable_depth,
                immutable_length,
                all_undefined,
            }
        } else {
            // a's immutable map is used as base, so one which exist there don't get inserted to
            // result, but if it has ones that should become undefined, the undefined has to be
            // inserted to the result instead.
            let common = a.common_immutable(b);
            let mut all_undefined = true;
            let mut b_is_imm = false;
            for map in b.iter_maps_until_immutable(common) {
                for (key, &b_val) in map.map.iter() {
                    if b_is_imm && b.get(key) != Some(b_val) {
                        // Wasn't newest value
                        continue;
                    }
                    if let Some((a_val, is_imm)) = a.get_with_immutable_info(key) {
                        match a_val == b_val {
                            true => {
                                if !is_imm {
                                    all_undefined &= a_val.is_undefined();
                                    result.insert(*key, a_val);
                                }
                            }
                            false => {
                                if !a_val.is_undefined() {
                                    if b_val.is_undefined() {
                                        result.insert(*key, b_val);
                                    } else {
                                        result.insert(*key, ctx.new_undef());
                                    }
                                }
                            }
                        }
                    } else {
                        if !key.0.0.contains_undefined() {
                            if b_val.is_undefined() {
                                result.insert(*key, b_val);
                            } else {
                                result.insert(*key, ctx.new_undef());
                            }
                        }
                    }
                }
                b_is_imm = true;
            }
            // The result contains now anything that was in b's unique branch of the memory.
            //
            // Repeat for a's unique branch.
            let mut a_is_imm = false;
            let a_mutable_all_undef = a.all_undefined;
            for map in a.iter_maps_until_immutable(common) {
                if a_mutable_all_undef && a_is_imm && map.all_undefined {
                    continue;
                }
                for (key, &a_val) in map.map.iter() {
                    if a_is_imm {
                        if a_val.is_undefined() {
                            if a_mutable_all_undef {
                                continue;
                            }
                            if let Some(nonimm) = a.get_no_immutable(key) {
                                if nonimm.is_undefined() {
                                    // Do nothing, the result will have 'key = a_val'
                                    // instead of 'key = nonimm', but both are undefined anyway
                                    continue;
                                }
                                // else continues to `if !result.contains_key(..)`
                            } else {
                                // Do nothing, if this is not the newest value it should've
                                // been skipped anyway, if this is then undef merged with
                                // anything will still be undef
                                continue;
                            }
                        } else {
                            if a.get(key) != Some(a_val) {
                                // Wasn't newest value
                                continue;
                            }
                        }
                    }
                    if !result.contains_key(key) {
                        let needs_undef = if let Some(b_val) = b.get(key) {
                            a_val != b_val
                        } else {
                            true
                        };
                        if needs_undef {
                            // If the key with undefined was in imm, override its value,
                            // but otherwise just don't bother adding it back.
                            if !key.0.0.contains_undefined() || !a_is_imm {
                                result.insert(*key, ctx.new_undef());
                            }
                        }
                    }
                }
                a_is_imm = true;
            }
            let immutable_length = result_immutable_len + result.len();
            let map = Rc::new(result);
            MemoryMap {
                map,
                immutable: result_immutable,
                slower_immutable,
                immutable_depth: a.immutable_depth,
                immutable_length,
                all_undefined,
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
        let mut is_imm = false;
        for map in a.iter_maps_until_immutable(common) {
            if !map.all_undefined {
                for (key, &a_val) in map.map.iter() {
                    if !key.0.0.contains_undefined() {
                        if !a_val.is_undefined() {
                            if b.get(key) != Some(a_val) {
                                let was_newest_value = if !is_imm {
                                    true
                                } else {
                                    a.get(key) == Some(a_val)
                                };
                                if was_newest_value {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            is_imm = true;
        }
        is_imm = false;
        for map in b.iter_maps_until_immutable(common) {
            for (key, &b_val) in map.map.iter() {
                if !key.0.0.contains_undefined() {
                    let different = match a.get(key) {
                        Some(a_val) => !a_val.is_undefined() && a_val != b_val,
                        None => true,
                    };
                    if different {
                        let was_newest_value = if !is_imm {
                            true
                        } else {
                            b.get(key) == Some(b_val)
                        };
                        if was_newest_value {
                            return true;
                        }
                    }
                }
            }
            is_imm = true;
        }
        false
    }
}

impl<'a, 'e> Iterator for MemoryIterMapsUntilImm<'a, 'e> {
    /// The bool tells if we're at immutable parts of the calling operand or not
    type Item = &'a MemoryMap<'e>;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.map?;
        let limit_ptr = self.limit.map(|x| x as *const MemoryMap<'e>).unwrap_or(ptr::null());
        self.map = next.immutable.as_deref()
            .filter(|&x| x as *const MemoryMap<'e>  != limit_ptr)
            .map(|x| MemoryMap::slower_immutable_to_limit(x, self.limit));
        Some(next)
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
    if let Some((old_l, old_r)) = old.0.if_arithmetic_eq() {
        if let Some((new_l, new_r)) = new.0.if_arithmetic_eq() {
            if old_r == ctx.const_0() && new_r == ctx.const_0() {
                if old_l.relevant_bits().end == 1 && new_l.relevant_bits().end == 1 {
                    // Use De Morgan's laws
                    let res = merge_constraint_inner(ctx, old_l, new_l, ArithOpType::Or)?;
                    return Some(Constraint(ctx.eq_const(res, 0)));
                }
            }
        }
    }
    let res = merge_constraint_inner(ctx, old.0, new.0, ArithOpType::And)?;
    Some(Constraint(res))
}

fn merge_constraint_inner<'e>(
    ctx: OperandCtx<'e>,
    old: Operand<'e>,
    new: Operand<'e>,
    ty: ArithOpType,
) -> Option<Operand<'e>> {
    ctx.simplify_temp_stack().alloc(|mut old_parts| {
        collect_ops(old, &mut old_parts, ty)?;
        ctx.simplify_temp_stack().alloc(|mut new_parts| {
            collect_ops(new, &mut new_parts, ty)?;
            old_parts.retain(|old_val| {
                new_parts.iter().any(|&x| x == old_val)
            });
            join_ops(ctx, old_parts, ty)
        })
    })
}

fn join_ops<'e>(
    ctx: OperandCtx<'e>,
    parts: &[Operand<'e>],
    ty: ArithOpType,
) -> Option<Operand<'e>> {
    let mut op = *parts.get(0)?;
    for &next in parts.iter().skip(1) {
        op = ctx.arithmetic(ty, op, next);
    }
    Some(op)
}

fn collect_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e, Operand<'e>>,
    ty: ArithOpType,
) -> Option<()> {
    if let Some((l, r)) = s.if_arithmetic(ty) {
        ops.push(r).ok()?;
        collect_ops(l, ops, ty)?;
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
    last_merge: Option<MemoryMergeCached<'e>>,
    /// Merge where the result has mutable map len == 0
    last_immutable_merge: Option<MemoryMergeCached<'e>>,
}

struct MemoryOpCached<'e, T> {
    // NOTE: This relies on MemoryMapTopLevel being always associated with same
    // immutable pointer.
    old: Rc<MemoryMapTopLevel<'e>>,
    new: Rc<MemoryMapTopLevel<'e>>,
    result: T,
}

struct MemoryMergeCached<'e> {
    old: MemoryMap<'e>,
    new: MemoryMap<'e>,
    result: MemoryMap<'e>,
}

impl<'e> MergeStateCache<'e> {
    pub fn new() -> MergeStateCache<'e> {
        MergeStateCache {
            last_compare: None,
            last_merge: None,
            last_immutable_merge: None,
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

    pub fn merge_memory(
        &mut self,
        old_base: &Memory<'e>,
        new_base: &Memory<'e>,
        ctx: OperandCtx<'e>,
    ) -> Memory<'e> {
        let mut old = &old_base.map;
        let mut new = &new_base.map;
        // Don't use empty outermost map when checking for cached inputs
        if old.map.len() == 0 {
            if let Some(ref imm) = old.immutable {
                old = &*imm;
            }
        }
        if new.map.len() == 0 {
            if let Some(ref imm) = new.immutable {
                new = &*imm;
            }
        }
        if let Some(ref cached) = self.last_merge {
            if Rc::ptr_eq(&cached.old.map, &old.map) && Rc::ptr_eq(&cached.new.map, &new.map) {
                return Memory {
                    map: cached.result.clone(),
                    cached_addr: None,
                    cached_value: None,
                };
            }
        }
        if let Some(ref cached) = self.last_immutable_merge {
            if Rc::ptr_eq(&cached.old.map, &old.map) && Rc::ptr_eq(&cached.new.map, &new.map) {
                return Memory {
                    map: cached.result.clone(),
                    cached_addr: None,
                    cached_value: None,
                };
            }
        }
        // This uses the possibly empty outermost maps because merge can't
        // handle immutable input properly.
        let mut result = old_base.map.merge(&new_base.map, ctx);
        result.maybe_convert_immutable(16);

        if result.map.len() == 0 {
            self.last_immutable_merge = Some(MemoryMergeCached {
                old: old.clone(),
                new: new.clone(),
                result: result.clone(),
            });
        }
        self.last_merge = Some(MemoryMergeCached {
            old: old.clone(),
            new: new.clone(),
            result: result.clone(),
        });
        Memory {
            map: result,
            cached_addr: None,
            cached_value: None,
        }
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
fn apply_constraint_2() {
    let ctx = &crate::operand::OperandContext::new();
    let constraint = Constraint(ctx.eq_const(
        ctx.eq_const(
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
    let eq = ctx.const_1();
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
    a.set(ctx.constant(4), 0, ctx.constant(8));
    a.set(ctx.constant(12), 0, ctx.constant(8));
    b.set(ctx.constant(8), 0, ctx.constant(15));
    b.set(ctx.constant(12), 0, ctx.constant(8));
    a.map.convert_immutable();
    b.map.convert_immutable();
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert!(new.get(ctx.constant(4), 0).unwrap().is_undefined());
    assert!(new.get(ctx.constant(8), 0).unwrap().is_undefined());
    assert_eq!(new.get(ctx.constant(12), 0).unwrap(), ctx.constant(8));
}

#[test]
fn merge_memory_undef() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    let addr = ctx.sub_const(ctx.new_undef(), 8);
    a.set(addr, 0, ctx.mem32(ctx.constant(4), 0));
    b.set(addr, 0, ctx.mem32(ctx.constant(4), 0));
    a.map.convert_immutable();
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert_eq!(new.get(addr, 0).unwrap(), ctx.mem32(ctx.constant(4), 0));
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
    a.set(addr, 0, ctx.constant(8));
    a.set(addr2, 0, ctx.constant(0));
    b.set(addr, 0, ctx.constant(4));
    a.map.convert_immutable();
    b.map.convert_immutable();
    a.set(addr, 0, ctx.constant(4));
    a.set(addr2, 0, ctx.constant(1));
    //  { addr: 8, addr2: 4 }       { addr: 4 }
    //          ^                      ^
    //  a { addr: 4, addr2: 1 }     b { }
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert_eq!(new.get(addr, 0).unwrap(), ctx.constant(4));
    assert!(new.get(addr2, 0).unwrap().is_undefined());
}

#[test]
fn equal_memory_no_need_to_merge() {
    // a has [addr] = 4 in top level, but 8 in immutable
    // b has [addr] = 4 in immutable
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    let addr = ctx.sub_const(ctx.register(4), 8);
    a.set(addr, 0, ctx.constant(8));
    b.set(addr, 0, ctx.constant(4));
    a.map.convert_immutable();
    b.map.convert_immutable();
    a.set(addr, 0, ctx.constant(4));
    assert!(!a.map.has_merge_changed(&b.map));
}

#[test]
fn merge_memory_undef2() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    a.set(addr, 0, ctx.mem32(ctx.constant(4), 0));
    a.map.convert_immutable();
    let mut b = a.clone();
    b.set(addr, 0, ctx.mem32(ctx.constant(9), 0));
    b.map.convert_immutable();
    a.set(addr, 0, ctx.new_undef());
    a.set(addr2, 0, ctx.new_undef());
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert!(new.get(addr2, 0).unwrap().is_undefined());
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
    let (low, high) = value_limits(constraint, ctx.register(0));
    assert_eq!(low, 2);
    assert_eq!(high, 7);
}

#[test]
fn merge_memory_undef3() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    a.set(addr, 0, ctx.mem32(ctx.constant(4), 0));
    a.map.convert_immutable();
    let mut b = a.clone();
    b.set(addr2, 0, ctx.new_undef());
    b.map.convert_immutable();
    a.set(addr, 0, ctx.new_undef());
    a.set(addr2, 0, ctx.new_undef());
    //          base { addr: mem32[4] }
    //          ^               ^
    //  b { addr2: ud }     a { addr: ud, addr2: ud }
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert!(new.get(addr2, 0).unwrap().is_undefined());
    let mut new = cache.merge_memory(&b, &a, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert!(new.get(addr2, 0).unwrap().is_undefined());
}

#[test]
fn merge_memory_undef4() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    a.set(addr, 0, ctx.mem32(ctx.constant(4), 0));
    a.map.convert_immutable();
    let mut b = a.clone();
    b.set(addr2, 0, ctx.new_undef());
    b.map.convert_immutable();
    a.set(addr, 0, ctx.constant(6));
    a.set(addr2, 0, ctx.constant(5));
    //          base { addr: mem32[4] }
    //          ^               ^
    //  b { addr2: ud }     a { addr: 6, addr2: 5 }
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert!(new.get(addr2, 0).unwrap().is_undefined());
    let mut new = cache.merge_memory(&b, &a, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert!(new.get(addr2, 0).unwrap().is_undefined());
}

#[test]
fn merge_memory_undef5() {
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    a.set(addr, 0, ctx.constant(1));
    a.set(addr2, 0, ctx.constant(2));
    a.map.convert_immutable();
    let b = a.clone();
    a.set(addr, 0, ctx.constant(6));
    //      { addr: 1, addr2: 2 }
    //      ^             ^
    //  b { }     a { addr: 6 }
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&a, &b, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert_eq!(new.get(addr2, 0).unwrap(), ctx.constant(2));
    let mut cache = MergeStateCache::new();
    let mut new = cache.merge_memory(&b, &a, ctx);
    assert!(new.get(addr, 0).unwrap().is_undefined());
    assert_eq!(new.get(addr2, 0).unwrap(), ctx.constant(2));
}

#[test]
fn merge_memory_undef6() {
    fn add_filler<'e>(mem: &mut Memory<'e>, ctx: OperandCtx<'e>, seed: u64) {
        for i in 0..64 {
            mem.set(ctx.constant(i * 0x100), 0, ctx.constant(seed + i));
        }
    }
    let ctx = &crate::operand::OperandContext::new();
    let mut a = Memory::new();
    let mut b = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    a.set(addr, 0, ctx.constant(3));
    a.map.convert_immutable();
    add_filler(&mut a, ctx, 0);
    a.map.convert_immutable();
    b.set(addr, 0, ctx.constant(2));
    b.map.convert_immutable();
    let mut c = b.clone();
    b.set(addr, 0, ctx.constant(3));
    b.map.convert_immutable();
    add_filler(&mut c, ctx, 8);
    c.set(addr2, 0, ctx.new_undef());

    // Test against broken cache logic
    // If cache takes advantage of a+b = m merge,
    // it should realize that addr in result is undefined

    //  { addr: 3 }    { addr: 2 } <-----
    //      ^             ^             ^
    // a { filler }   b { addr: 3 }  c { filler }
    //     ^             ^               ^
    //  m { } ~~~~~~~~~~~~               |
    //    ^                              |
    //  m2 { filler }                    |
    //    ^                              |
    //  result { } ~~~~~~~~~~~~~~~~~~~~~~~

    for &(swap_ab, swap_cm2) in &[(false, false), (false, true), (true, false), (true, true)] {
        println!("----------- {} {}", swap_ab, swap_cm2);
        let mut cache = MergeStateCache::new();
        let mut m = match swap_ab {
            false => cache.merge_memory(&a, &b, ctx),
            true => cache.merge_memory(&b, &a, ctx),
        };
        m.map.convert_immutable();
        let mut m2 = m.clone();
        add_filler(&mut m2, ctx, 16);
        m2.map.convert_immutable();

        let mut result = match swap_cm2 {
            false => cache.merge_memory(&m2, &c, ctx),
            true => cache.merge_memory(&c, &m2, ctx),
        };

        assert_eq!(a.get(addr, 0).unwrap(), ctx.constant(3));
        assert_eq!(b.get(addr, 0).unwrap(), ctx.constant(3));
        assert_eq!(m.get(addr, 0).unwrap(), ctx.constant(3));
        assert_eq!(c.get(addr, 0).unwrap(), ctx.constant(2));
        assert!(result.get(addr, 0).unwrap().is_undefined());
    }
}

#[test]
fn immutable_tree() {
    // Building 19-deep map and verifying it is correct
    //                      (0)
    //                      (1)
    //                      (2)
    //              (3)---->(3)
    //               |
    //               |<-----(4)
    //               |      (5)
    //               |      (6)
    //              (7)---->(7)
    //               |
    //               |<-----(8)
    //               |      (9)
    //               |      (10)
    //              (11)--->(11)
    //               |
    //               |<-----(12)
    //               |      (13)
    //               |      (14)
    //     (15)---->(15)--->(15)
    //      |
    //      |<--------------(16)
    //      |        |      (17)
    //      |        |      (18)
    //      |       (19)--->(19)
    let ctx = &crate::operand::OperandContext::new();
    let mut map = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let addr2 = ctx.sub_const(ctx.register(5), 16);
    for i in 0..20 {
        map.set(addr, 0, ctx.constant(i));
        if i == 3 {
            map.set(addr2, 0, ctx.constant(8));
        }
        map.map.convert_immutable();
    }

    let addr_key = &(addr.hash_by_address(), 0);
    let addr2_key = &(addr2.hash_by_address(), 0);

    // Just show the map for debugging the test
    let mut pos = &map.map;
    for i in (0..21).rev() {
        println!("--- {} ---", i);
        println!("{:?}", pos.get(addr_key));
        if matches!(i, 15) {
            pos = pos.slower_immutable.as_deref().unwrap();
            println!("{:?}", pos.get(addr_key));
            pos = pos.slower_immutable.as_deref().unwrap();
            println!("{:?}", pos.get(addr_key));
        }
        if matches!(i, 19 | 11 | 7 | 3) {
            pos = pos.slower_immutable.as_deref().unwrap();
            println!("{:?}", pos.get(addr_key));
        }
        if i != 0 {
            pos = pos.immutable.as_deref().unwrap();
        }
    }

    let mut pos = &map.map;
    let mut prev_fast = None;
    for i in (0..21).rev() {
        println!("--- {} ---", i);
        assert_eq!(pos.immutable_depth, i);
        if i == 20 {
            pos = pos.immutable.as_deref().unwrap();
            continue;
        }

        assert_eq!(pos.get(addr_key), Some(ctx.constant(i as u64)));
        if i >= 3 {
            assert_eq!(pos.get(addr2_key), Some(ctx.constant(8)));
        } else {
            assert_eq!(pos.get(addr2_key), None);
        }

        if matches!(i, 19 | 11 | 7 | 3) {
            prev_fast = pos.immutable.as_deref();
            pos = pos.slower_immutable.as_deref().unwrap();
            assert_eq!(pos.get(addr_key), Some(ctx.constant(i as u64)));
        }
        if i == 15 {
            pos = pos.slower_immutable.as_deref().unwrap();
            assert_eq!(pos.get(addr_key), Some(ctx.constant(i as u64)));

            prev_fast = pos.immutable.as_deref();
            pos = pos.slower_immutable.as_deref().unwrap();
            assert_eq!(pos.get(addr_key), Some(ctx.constant(i as u64)));
        }
        if matches!(i, 16 | 12 | 8 | 4) {
            let prev_fast = prev_fast.unwrap();
            assert!(Rc::ptr_eq(&pos.immutable.as_deref().unwrap().map, &prev_fast.map));
        }
        if i != 0 {
            pos = pos.immutable.as_deref().unwrap();
        }
    }
}

#[test]
fn test_common_immutable() {
    fn add_immutables<'e>(mem: &mut Memory<'e>, ctx: OperandCtx<'e>, count: usize) {
        for i in 0..count {
            mem.set(ctx.constant(444), 0, ctx.constant(i as u64));
            mem.map.convert_immutable();
        }
    }
    let ctx = &crate::operand::OperandContext::new();
    let mut mem = Memory::new();
    for i in 0..72 {
        println!("--- {} ---", i);
        mem.set(ctx.constant(999), 0, ctx.constant(i as u64));
        mem.map.convert_immutable();
        let mut a = mem.clone();
        let mut b = mem.clone();
        add_immutables(&mut a, ctx, i + 4);
        add_immutables(&mut b, ctx, i + 7);
        let common = a.map.common_immutable(&b.map).unwrap();
        assert!(Rc::ptr_eq(&common.map, &mem.map.immutable.as_deref().unwrap().map));
        let common = mem.map.common_immutable(&a.map).unwrap();
        assert!(Rc::ptr_eq(&common.map, &mem.map.immutable.as_deref().unwrap().map));
        let common = mem.map.common_immutable(&b.map).unwrap();
        assert!(Rc::ptr_eq(&common.map, &mem.map.immutable.as_deref().unwrap().map));
    }
    let common = mem.map.common_immutable(&mem.map).unwrap();
    assert!(Rc::ptr_eq(&common.map, &mem.map.map));
}

#[test]
fn value_limits_sext() {
    let ctx = &crate::operand::OperandContext::new();
    let constraint = ctx.gt_const_left(
        0x38,
        ctx.sub_const(
            ctx.mem8(ctx.register(0), 0),
            0x41,
        ),
    );
    let (low, high) = value_limits(constraint, ctx.mem8(ctx.register(0), 0));
    assert_eq!(low, 0x41);
    assert_eq!(high, 0x78);
    let (low, high) = value_limits(
        constraint,
        ctx.sign_extend(
            ctx.mem8(ctx.register(0), 0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
    );
    assert_eq!(low, 0x41);
    assert_eq!(high, 0x78);

    let constraint = ctx.gt_const_left(
        0x42,
        ctx.sub_const(
            ctx.mem8(ctx.register(0), 0),
            0x41,
        ),
    );
    let (low, high) = value_limits(constraint, ctx.mem8(ctx.register(0), 0));
    assert_eq!(low, 0x41);
    assert_eq!(high, 0x82);
    let (low, high) = value_limits(
        constraint,
        ctx.sign_extend(
            ctx.mem8(ctx.register(0), 0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
    );
    assert_eq!(low, 0x41);
    assert_eq!(high, 0xffff_ff82);
}

#[test]
fn test_merge_flags() {
    fn test<'e, F: FnOnce(&mut FlagState<'_, 'e>, &mut FlagState<'_, 'e>)>(
        ctx: OperandCtx<'e>,
        callback: F,
    ) {
        let mut flags1 = array_init::array_init(|i| ctx.register(i as u8));
        let mut flags2 = array_init::array_init(|i| ctx.register(8 + i as u8));
        flags1[5] = ctx.const_0();
        flags2[5] = ctx.const_0();
        let mut pending1 = PendingFlags::new();
        let mut pending2 = PendingFlags::new();
        let update = FlagUpdate {
            left: ctx.mem32c(0x100),
            right: ctx.mem32c(0x200),
            ty: FlagArith::Sub,
            size: MemAccessSize::Mem32,
        };
        pending1.reset(&update, None);
        pending2.reset(&update, None);

        let mut flag_state1 = FlagState {
            flags: &mut flags1,
            pending_flags: &mut pending1,
        };
        let mut flag_state2 = FlagState {
            flags: &mut flags2,
            pending_flags: &mut pending2,
        };
        callback(&mut flag_state1, &mut flag_state2);
    }

    let ctx = &crate::operand::OperandContext::new();
    test(ctx, |old, new| {
        // Should be equal, no changes
        assert!(flags_merge_changed(old, new, ctx) == false);
        let mut new_flags = array_init::array_init(|_| ctx.const_0());
        let new_pending = merge_flags(old, new, &mut new_flags, ctx);
        assert_eq!(new_pending.pending_bits, 0x1f);
        assert_eq!(new_pending.update, old.pending_flags.update);
    });
    test(ctx, |old, new| {
        new.pending_flags.clear_pending(Flag::Zero);
        old.pending_flags.clear_pending(Flag::Carry);
        let zero_carry_mask = (1 << Flag::Zero as u32) | (1 << Flag::Carry as u32);
        // Should assume that they're equal and new zero and old carry can be used
        assert!(flags_merge_changed(old, new, ctx) == false);
        let mut new_flags = array_init::array_init(|_| ctx.const_0());
        let new_pending = merge_flags(old, new, &mut new_flags, ctx);
        assert_eq!(new_pending.flag_bits, 0x1f);
        assert_eq!(new_pending.pending_bits, 0x1f & !zero_carry_mask);
        assert_eq!(new_pending.update, old.pending_flags.update);
        assert_eq!(new_flags[Flag::Zero as usize], new.flags[Flag::Zero as usize]);
        assert_eq!(new_flags[Flag::Carry as usize], old.flags[Flag::Carry as usize]);
    });
    test(ctx, |old, new| {
        new.pending_flags.clear_pending(Flag::Zero);
        old.pending_flags.make_non_pending(Flag::Carry);
        let carry_mask = 1 << Flag::Carry as u32;
        let zero_carry_mask = (1 << Flag::Zero as u32) | (1 << Flag::Carry as u32);
        // Now old carry has been assigned from somewhere else than pending FlagUpdate,
        // should be assumed that new carry isn't same
        assert!(flags_merge_changed(old, new, ctx) == true);
        let mut new_flags = array_init::array_init(|_| ctx.const_0());
        let new_pending = merge_flags(old, new, &mut new_flags, ctx);
        assert_eq!(new_pending.flag_bits, 0x1f & !carry_mask);
        assert_eq!(new_pending.pending_bits, 0x1f & !zero_carry_mask);
        assert_eq!(new_pending.update, old.pending_flags.update);
        assert_eq!(new_flags[Flag::Zero as usize], new.flags[Flag::Zero as usize]);
        assert!(new_flags[Flag::Carry as usize].is_undefined());
    });
    test(ctx, |old, new| {
        new.flags[Flag::Carry as usize] = ctx.constant(666);
        old.flags[Flag::Carry as usize] = ctx.constant(666);
        new.pending_flags.clear_pending(Flag::Carry);
        old.pending_flags.make_non_pending(Flag::Carry);
        let carry_mask = 1 << Flag::Carry as u32;
        // Carry sources are different but result is same, should not make undefined
        // or consider changed.
        assert!(flags_merge_changed(old, new, ctx) == false);
        let mut new_flags = array_init::array_init(|_| ctx.const_0());
        let new_pending = merge_flags(old, new, &mut new_flags, ctx);
        // This could also be 0x1f, but not going to care about it for now
        assert_eq!(new_pending.flag_bits, 0x1f & !carry_mask);
        assert_eq!(new_pending.pending_bits, 0x1f & !carry_mask);
        assert_eq!(new_pending.update, old.pending_flags.update);
        assert_eq!(new_flags[Flag::Carry as usize], ctx.constant(666));
    });
}

#[test]
fn test_iter_until_immutable() {
    // Building 19-deep map and verifying iter_until_immutable
    //                      (0)
    //                      (1)
    //                      (2)
    //              (3)---->(3)
    //               |
    //               |<-----(4)
    //               |      (5)
    //               |      (6)
    //              (7)---->(7)
    //               |
    //               |<-----(8)
    //               |      (9)
    //               |      (10)
    //              (11)--->(11)
    //               |
    //               |<-----(12)
    //               |      (13)
    //               |      (14)
    //     (15)---->(15)--->(15)
    //      |
    //      |<--------------(16)
    //      |        |      (17)
    //      |        |      (18)
    //      |       (19)--->(19)
    let ctx = &crate::operand::OperandContext::new();
    let mut map = Memory::new();
    let addr = ctx.sub_const(ctx.register(5), 8);
    let mut maps = Vec::new();
    for i in 0..20 {
        map.set(addr, 0, ctx.constant(i));
        map.map.convert_immutable();
        maps.push(map.clone());
    }

    let map19 = maps[19].map.immutable.as_ref().unwrap();
    assert_eq!(map19.immutable_depth, 19);
    let map14 = maps[14].map.immutable.as_ref().unwrap();
    assert_eq!(map14.immutable_depth, 14);
    let map12 = maps[12].map.immutable.as_ref().unwrap();
    assert_eq!(map12.immutable_depth, 12);
    let map11 = maps[11].map.immutable.as_ref().unwrap();
    assert_eq!(map11.immutable_depth, 11);
    let map3 = maps[3].map.immutable.as_ref().unwrap();
    assert_eq!(map3.immutable_depth, 3);
    let map1 = maps[1].map.immutable.as_ref().unwrap();
    assert_eq!(map1.immutable_depth, 1);

    let results = map19.iter_maps_until_immutable(Some(map14))
        .enumerate()
        .flat_map(|(i, x)| x.map.iter().map(move |x| (x.0, *x.1, i != 0)))
        .map(|x| ((x.0.0.0, x.0.1), x.1, x.2))
        .collect::<Vec<_>>();
    let cmp = vec![
        ((addr, 0u64), ctx.constant(19), false),
        ((addr, 0), ctx.constant(15), true),
    ];
    assert_eq!(results, cmp);

    let results = map19.iter_maps_until_immutable(Some(map12))
        .enumerate()
        .flat_map(|(i, x)| x.map.iter().map(move |x| (x.0, *x.1, i != 0)))
        .map(|x| ((x.0.0.0, x.0.1), x.1, x.2))
        .collect::<Vec<_>>();
    let cmp = vec![
        ((addr, 0u64), ctx.constant(19), false),
        ((addr, 0), ctx.constant(15), true),
        ((addr, 0), ctx.constant(14), true),
        ((addr, 0), ctx.constant(13), true),
    ];
    assert_eq!(results, cmp);

    let results = map19.iter_maps_until_immutable(Some(map11))
        .enumerate()
        .flat_map(|(i, x)| x.map.iter().map(move |x| (x.0, *x.1, i != 0)))
        .map(|x| ((x.0.0.0, x.0.1), x.1, x.2))
        .collect::<Vec<_>>();
    let cmp = vec![
        ((addr, 0u64), ctx.constant(19), false),
        ((addr, 0), ctx.constant(15), true),
    ];
    assert_eq!(results, cmp);

    let results = map19.iter_maps_until_immutable(Some(map3))
        .enumerate()
        .flat_map(|(i, x)| x.map.iter().map(move |x| (x.0, *x.1, i != 0)))
        .map(|x| ((x.0.0.0, x.0.1), x.1, x.2))
        .collect::<Vec<_>>();
    let cmp = vec![
        ((addr, 0u64), ctx.constant(19), false),
        ((addr, 0), ctx.constant(15), true),
        ((addr, 0), ctx.constant(11), true),
        ((addr, 0), ctx.constant(7), true),
    ];
    assert_eq!(results, cmp);

    let results = map19.iter_maps_until_immutable(Some(map1))
        .enumerate()
        .flat_map(|(i, x)| x.map.iter().map(move |x| (x.0, *x.1, i != 0)))
        .map(|x| ((x.0.0.0, x.0.1), x.1, x.2))
        .collect::<Vec<_>>();
    let cmp = vec![
        ((addr, 0u64), ctx.constant(19), false),
        ((addr, 0), ctx.constant(15), true),
        ((addr, 0), ctx.constant(11), true),
        ((addr, 0), ctx.constant(7), true),
        ((addr, 0), ctx.constant(3), true),
        ((addr, 0), ctx.constant(2), true),
    ];
    assert_eq!(results, cmp);

    let results = map14.iter_maps_until_immutable(Some(map1))
        .enumerate()
        .flat_map(|(i, x)| x.map.iter().map(move |x| (x.0, *x.1, i != 0)))
        .map(|x| ((x.0.0.0, x.0.1), x.1, x.2))
        .collect::<Vec<_>>();
    let cmp = vec![
        ((addr, 0u64), ctx.constant(14), false),
        ((addr, 0), ctx.constant(13), true),
        ((addr, 0), ctx.constant(12), true),
        ((addr, 0), ctx.constant(11), true),
        ((addr, 0), ctx.constant(7), true),
        ((addr, 0), ctx.constant(3), true),
        ((addr, 0), ctx.constant(2), true),
    ];
    assert_eq!(results, cmp);
}

#[test]
fn test_is_flag_const_constraint() {
    let ctx = &crate::operand::OperandContext::new();
    let flag = ctx.flag_z();
    let eq_0 = ctx.eq_const(flag, 0);
    let eq_1 = ctx.eq_const(flag, 1);
    let neq_0 = ctx.neq_const(flag, 0);
    assert_eq!(
        is_flag_const_constraint(ctx, eq_0).unwrap(),
        (Flag::Zero, ctx.constant(0)),
    );
    assert_eq!(
        is_flag_const_constraint(ctx, eq_1).unwrap(),
        (Flag::Zero, ctx.constant(1)),
    );
    assert_eq!(
        is_flag_const_constraint(ctx, neq_0).unwrap(),
        (Flag::Zero, ctx.constant(1)),
    );
}
