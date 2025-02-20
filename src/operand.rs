//! [`Operand`] and its supporting types.
//!
//! Main types here are [`OperandContext`] for the arena all operands are allocated in,
//! and [`Operand`], scarf's 'value/variable/expression' type.

mod intern;
#[cfg(test)]
mod normalize_tests;
mod simplify;
pub(crate) mod slice_stack;
#[cfg(test)]
mod simplify_tests;
mod util;

#[cfg(feature = "serde")]
mod deserialize;
#[cfg(feature = "serde")]
pub use self::deserialize::DeserializeOperand;

use std::cell::{Cell, RefCell, RefMut};
use std::cmp::{max, min, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::num::NonZeroU8;
use std::ops::Range;
use std::ptr;

use copyless::BoxHelper;
use fxhash::FxHashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::disasm::{self, Operation};
use crate::disasm_cache::{DisasmArch, DisasmCache};
use crate::exec_state;

use self::slice_stack::SliceStack;

/// `Operand` is the type of values in scarf.
///
/// It is an immutable reference type, free to copy and expected to be passed by value.
///
/// Different types of `Operand`s are listed in [`OperandType`] enum.
/// The main types of interest are:
///
/// - Single variables, such as `OperandType::Arch`, `OperandType::Custom`.
/// - Constant integers, `OperandType::Constant`.
/// - Expressions, `OperandType::Arithmetic`, using `Operand` as inputs for the expression,
/// and as such, able to have arbitrarily deep tree of subexpressions making up the `Operand`.
/// - Memory, `OperandType::Memory`, which is able to have any `Operand` representing the
/// address.
///
/// All `Operand`s are created through [`OperandContext`] arena, which interns the created
/// `Operand`s, allowing implementing equality comparison as a single reference equality
/// check. The lifetime `'e` refers to this backing memory in `OperandContext`
/// which the `Operand` reference points to.
///
/// # Simplification
///
/// Every created `Operand` which contains an expression is immediately simplified to a form
/// that scarf considers to be the simplest equivalent form of the expression.
///
/// For example, subtracting any `Operand` from itself is simplified to zero, while adding
/// any `Operand` to itself gets simplfied to multiplication by two:
///
/// ```rust
/// let ctx = &scarf::OperandContext::new();
/// // Create an `Operand` representing `rcx` register.
/// // x86-64 registers are ordered as 'rax rcx rdx rbx  rsp rbp rsi rdi  r8 r9 r10...
/// // As such, index of rcx is 1.
/// let rcx = ctx.register(1);
/// // Create an `Operand` representing `rcx - rcx` expression.
/// let sub_result = ctx.sub(rcx, rcx);
/// // Create an `Operand` representing constant 0.
/// let zero = ctx.constant(0);
/// // Subtraction result is also zero
/// assert_eq!(sub_result, zero);
///
/// // And `rcx + rcx` is same as `rcx * 2`
/// let add_result = ctx.add(rcx, rcx);
/// // Most arithmetic `Operand` creation functions have a variation
/// // which should be preferred when one of the operands is known to
/// // be constant.
/// // This is more concise and better for performance than the equivalent
/// // `ctx.mul(rcx, ctx.constant(2))`.
/// let mul_result = ctx.mul_const(rcx, 2);
/// assert_eq!(add_result, mul_result);
///
/// // We can also have an `Operand` expression for equivalency:
/// // `ArithOpType::Equal` evaluates to either 0 or 1 for the result.
/// let eq_result = ctx.eq(add_result, mul_result);
/// // Equivalent to `ctx.constant(1)`, but effectively free to evaluate.
/// let one = ctx.const_1();
/// assert_eq!(eq_result, one);
/// ```
///
/// All `Operand`s, and as such, arithmetic expressions of the `Operand`s are considered
/// 64-bit values. In general, arithmetic expressions wrap on overflow, and arithmetic smaller
/// than 64-bit is represented by doing bitwise AND to only keep the low bits.
///
/// **Note:** Overflowing shifts (Shifting by more than 64 bits) are defined to result in
/// zero, instead of the behaviour of x86 shift and rotate instructions on general-purpose
/// registers, where shift count is taken as modulo 64. Scarf represents those instructions
/// as `rax << (count & 63)`, and users generally do not have to think about this.
///
/// If scarf is given excessively complex arithmetic to simplify -- which is often
/// seen in highly unrolled hash functions, scarf rather gives up than never finishes the
/// simplification.
///
/// ## Canonicalization guarantees
///
/// Simplification tries to convert any equivalent expression to a single 'canonical' form,
/// to make equality comparisons useful. As the simplification rules get improved, this may
/// result in user code which matches against a specific form of arithmetic expression to break.
///
/// This does mean that updating scarf dependency can bring surprising breaking changes! Try to
/// have tests on any code that has to match on `Operand`s.
///
/// The following canonicalizations are guaranteed to be stable across scarf changes,
/// to allow code to be simpler and faster by not having to consider expression forms that are
/// never created.
///
/// - Any chain of an operation, where the operation is commutative (Swapping left/right does
/// not affect result; Mul/And/Or/Xor) will be simplified to have single operand on right
/// and rest of the chain on left:
///
///     `(((a ^ b) ^ c) ^ d) ^ e`.
/// - Expressions chains of multiple combined Add/Sub are simplified similar to above rule:
///
///     `(((a + b) - c) + d) + e`
///
///     No guarantee how adds and subs are ordered within the chain.
/// - If any of the above commutative/add-sub chains has a constant, it is guaranteed to be
/// the outermost right operand:
///
///     `(a & b) & ffff`
/// - Equality expressions containing a non-zero constant always have the constant alone on right,
/// with rest of the expression shuffled to left. If the constant is zero, and there is no
/// subtraction on left side, the zero is also guaranteed to be on left.
///
///     That is, `x - y == 0` is simplified to either `x == y` or `y == x` instead,
///     while `0 == x` is guaranteed be simplified to `x == 0`.
///
/// Consider using [`Operand::either`] for cases where the result you want may be on either
/// left- or right-hand operand of the expression.
///
/// # Matching
///
/// The most comprehensive way to match against different values of `Operand` is to
/// match on the [`OperandType`] enum, returned from [`Operand::ty`].
///
/// However, most of the time this leads to quite verbose code, especially when you want to
/// try match against just a single variant of arithmetic expression or constant.
///
/// `Operand` includes a large amount of matching methods, starting with `if_`, such as
/// [`if_constant`](Self::if_constant), returning `Option<u64>`,
/// [`if_arithmetic_add`](Self::if_arithmetic_add), returning
/// `Option<(Operand<'e>, Operand<'e>)>`, [`if_memory`](Self::if_memory),
/// returning `Option<&'e MemAccess<'e>>`.
///
/// Using the matching methods is usually a lot more concise than using `match` on
/// `OperandType`.
///
/// Note that the memory matching methods effectively have three tiers:
/// - [`if_memory`](Self::if_memory), returning [`MemAccess`]
///     if the `Operand` is memory operand at all.
/// - [`if_mem64`](Self::if_mem64), [`if_mem32`](Self::if_mem32),
///     and others, which still return `MemAccess`, but check
///     that `mem.size` is of expected size
/// - [`if_mem64_offset`](Self::if_mem64_offset), [`if_mem32_offset`](Self::if_mem32_offset),
///     and others, which check both `MemAccess` size.
///     and specific constant offset, commonly useful when matching on a structure field.
///
///     They return `Operand` for the base memory address.
#[derive(Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize), serde(transparent))]
pub struct Operand<'e>(&'e OperandBase<'e>, PhantomData<&'e mut &'e ()>);

/// Wrapper around `Operand` which implements `Hash` on the interned address.
/// Separate struct since hashing by address gives hashes that aren't stable
/// across executions or even separate `OperandContext`s.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct OperandHashByAddress<'e>(pub Operand<'e>);

#[cfg_attr(feature = "serde", derive(Serialize))]
pub(crate) struct OperandBase<'e> {
    ty: OperandType<'e>,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    relevant_bits: Range<u8>,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    flags: u8,
    /// Bits that are used to avoid some comparisons when checking against OperandType
    /// that has a subenum that gets commonly checked too.
    /// 0xff00 = Tag; None, Memory, Arithmetic, Register
    /// 0xff = Data; MemAccessSize for Memory, ArithOpType for Arithmetic
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    type_alt_tag: u16,
    /// Used to implement Ord in almost-always-constant time.
    ///
    /// - For OperandType variants that fit in u64,
    ///     this variable just holds the same data as the variant
    ///     that can be used to compare this.
    /// - For Arithmetic / ArithmeticFloat, this has high 5 bits set to ArithOpType
    ///     and rest are hash (stable across executions) of sort_order of the
    ///     left/right ops (ArithmeticFloat size is not included at all)
    /// - For Memory / SignExtend, this holds discriminant and sort_order
    ///     (Shifted right to fit in discriminant) of `base`
    ///     (Offset / size are not included)
    /// The values are picked so that there wouldn't be too many conflicts
    /// in average sort case, and that the sort order is 'sensible' for most
    /// operands (Registers sorted by their index, constants by constant value)
    /// And when there are conflicts (E.g. Mem32[rax], Mem8[rax], Mem32[rax + 8]
    /// have all same sort_order) it should be quick to check remaining variables.
    /// (In general you'd never compare two equal `OperandBase`s, since `Operand`
    /// comparison includes equality check already.)
    ///
    /// `sort_order` could be also used as input to hash, though it would require
    /// remembering to add data that is not included, with `Memory` and other
    /// large types.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    sort_order: u64,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    relevant_bits_mask: u64,
}

const TYPE_ALT_TAG_MASK: u16 = 0xff00;
const REGISTER_ALT_TAG: u16 = 0x100;
const MEM_ALT_TAG: u16 = 0x200;
const ARITH_ALT_TAG: u16 = 0x300;

const FLAG_CONTAINS_UNDEFINED: u8 = 0x1;
// For simplify_with_and_mask optimization.
// For example, ((x & ff) | y) & ff should remove the inner mask.
// Also now used with 32bit normalization to quickly check
// if there are constant and masks inside.
const FLAG_COULD_REMOVE_CONST_AND: u8 = 0x2;
// If not set, resolve(x) == x
// (Constants, undef, custom, and arithmetic using them)
const FLAG_NEEDS_RESOLVE: u8 = 0x4;
const FLAG_CONTAINS_MEMORY: u8 = 0x8;
const FLAG_32BIT_NORMALIZED: u8 = 0x10;
// The operand is add / sub with at least one of the terms
// having FLAG_32BIT_NORMALIZED set
const FLAG_PARTIAL_32BIT_NORMALIZED_ADD: u8 = 0x20;
// Can simplify_with_and_mask ever do anything.
// Other than when the mask makes the operand 0
// due to no overlapping bits.
const FLAG_CAN_SIMPLIFY_WITH_AND_MASK: u8 = 0x40;
const ALWAYS_INHERITED_FLAGS: u8 =
    FLAG_CONTAINS_UNDEFINED | FLAG_NEEDS_RESOLVE | FLAG_CONTAINS_MEMORY;

// Caches ([0] == 0) & ([1] == 0) => [2]
const SIMPLIFY_CACHE_AND_EQ_ZERO: usize = 0;
const SIMPLIFY_CACHE_SIZE: usize = SIMPLIFY_CACHE_AND_EQ_ZERO + 3;

impl<'e> Hash for OperandHashByAddress<'e> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ((self.0).0 as *const OperandBase<'e> as usize).hash(state)
    }
}

impl<'e> Eq for Operand<'e> { }

// Short-circuit the common case of aliasing pointers
impl<'e> PartialEq for Operand<'e> {
    #[inline]
    fn eq(&self, other: &Operand<'e>) -> bool {
        ptr::eq(self.0, other.0)
    }
}

// Short-circuit the common case of aliasing pointers
impl<'e> Ord for Operand<'e> {
    fn cmp(&self, other: &Operand<'e>) -> Ordering {
        if ptr::eq(self.0, other.0) {
            Ordering::Equal
        } else {
            let OperandBase {
                ref ty,
                relevant_bits: _,
                flags: _,
                type_alt_tag: _,
                sort_order,
                relevant_bits_mask: _,
            } = *self.0;

            let other_ty = &other.0.ty;
            let self_discr = ty.discriminant();
            let other_discr = other_ty.discriminant();
            let order = self_discr.cmp(&other_discr);
            if order != Ordering::Equal {
                return order;
            }

            let order = sort_order.cmp(&other.0.sort_order);
            if order != Ordering::Equal {
                return order;
            }
            ty.cmp_sort_order_was_eq(other_ty)
        }
    }
}

impl<'e> PartialOrd for Operand<'e> {
    fn partial_cmp(&self, other: &Operand<'e>) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    // Explicit lt implementation is bit faster for sorting which uses it
    fn lt(&self, other: &Operand<'e>) -> bool {
        let OperandBase {
            ref ty,
            relevant_bits: _,
            type_alt_tag: _,
            flags: _,
            sort_order,
            relevant_bits_mask: _,
        } = *self.0;

        let other_ty = &other.0.ty;
        let self_discr = ty.discriminant();
        let other_discr = other_ty.discriminant();
        if self_discr != other_discr {
            return self_discr < other_discr;
        }

        if sort_order != other.0.sort_order {
            return sort_order < other.0.sort_order;
        }

        if ptr::eq(self.0, other.0) {
            return false;
        }
        matches!(ty.cmp_sort_order_was_eq(other_ty), Ordering::Less)
    }
}

impl<'e> fmt::Debug for Operand<'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        /*
        f.debug_struct("Operand")
            .field("ty", &self.ty)
            .field("simplified", &self.simplified)
            .field("relevant_bits", &self.relevant_bits)
            .field("hash", &self.hash)
            .finish()
        */
        write!(f, "{}", self)
    }
}

impl<'e> fmt::Debug for OperandType<'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::OperandType::*;
        match self {
            Arch(r) => write!(f, "Arch({})", r.value()),
            Constant(r) => write!(f, "Constant({:x})", r),
            Custom(r) => write!(f, "Custom({:x})", r),
            Undefined(r) => write!(f, "Undefined_{:x}", r.0),
            Memory(r) => f.debug_tuple("Memory").field(r).finish(),
            Arithmetic(r) => f.debug_tuple("Arithmetic").field(r).finish(),
            ArithmeticFloat(r, size) => {
                f.debug_tuple("ArithmeticFloat").field(size).field(r).finish()
            }
            SignExtend(a, b, c) => {
                f.debug_tuple("SignExtend").field(a).field(b).field(c).finish()
            }
            Select(a, b, c) => {
                f.debug_tuple("Select").field(a).field(b).field(c).finish()
            }
        }
    }
}

impl<'e> fmt::Display for Operand<'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        display_operand(f, *self, 0)
    }
}

fn display_operand<'e>(f: &mut fmt::Formatter, op: Operand<'e>, recurse: u8) -> fmt::Result {
    use self::ArithOpType::*;
    if recurse >= 20 {
        match *op.ty() {
            OperandType::Memory(..) | OperandType::Arithmetic(..) |
                OperandType::ArithmeticFloat(..) | OperandType::SignExtend(..) |
                OperandType::Select(..) =>
            {
                return write!(f, "(...)");
            }
            _ => (),
        }
    }
    match *op.ty() {
        OperandType::Arch(r) => {
            if let Some(fmt) = r.arch_def.fmt {
                fmt(f, r.value())
            } else {
                write!(f, "Arch({:x})", r.value())
            }
        }
        OperandType::Constant(c) => write!(f, "{:x}", c),
        OperandType::Memory(ref mem) => {
            let (base, offset) = mem.address();
            if let Some(c) = mem.if_constant_address() {
                write!(f, "Mem{}[{:x}]", mem.size.bits(), c)
            } else {
                write!(f, "Mem{}[", mem.size.bits())?;
                display_operand(f, base, recurse + 1)?;
                if offset != 0 {
                    let (sign, offset) = match offset < 0x8000_0000_0000_0000 {
                        true => ('+', offset),
                        false => ('-', 0u64.wrapping_sub(offset)),
                    };
                    write!(f, " {} {:x}]", sign, offset)
                } else {
                    write!(f, "]")
                }
            }
        }
        OperandType::Undefined(id) => write!(f, "Undefined_{:x}", id.0),
        OperandType::Arithmetic(ref arith) | OperandType::ArithmeticFloat(ref arith, _) => {
            let l = arith.left;
            let r = arith.right;
            let (prefix, middle) = match arith.ty {
                Add => ("(", " + "),
                Sub => ("(", " - "),
                Mul => ("(", " * "),
                Div => ("(", " / "),
                Modulo => ("(", " % "),
                And => ("(", " & "),
                Or => ("(", " | "),
                Xor => ("(", " ^ "),
                Lsh => ("(", " << "),
                Rsh => ("(", " >> "),
                Equal => ("(", " == "),
                GreaterThan => ("(", " > "),
                SignedMul => ("mul_signed(", ", "),
                MulHigh => ("mul_high(", ", "),
                ToFloat => ("to_float(", ""),
                ToDouble => ("to_double(", ""),
                ToInt => ("to_int(", ""),
            };
            write!(f, "{}", prefix)?;
            display_operand(f, l, recurse + 1)?;
            if middle != "" {
                write!(f, "{}", middle)?;
                display_operand(f, r, recurse + 1)?;
            }
            write!(f, ")")?;
            match *op.ty() {
                OperandType::ArithmeticFloat(_, size) => {
                    write!(f, "[f{}]", size.bits())?;
                }
                _ => (),
            }
            Ok(())
        },
        OperandType::SignExtend(val, ref from, ref to) => {
            write!(f, "signext_{}_to_{}(", from.bits(), to.bits())?;
            display_operand(f, val, recurse + 1)?;
            write!(f, ")")
        }
        OperandType::Select(condition, yes, no) => {
            // select(condition ? yes : no)
            write!(f, "select(")?;
            for pair in [
                (condition, " ? "),
                (yes, " : "),
                (no, ")"),
            ] {
                display_operand(f, pair.0, recurse + 1)?;
                write!(f, "{}", pair.1)?;
            }
            Ok(())
        }
        OperandType::Custom(val) => {
            write!(f, "Custom_{:x}", val)
        }
    }
}

/// Different values an [`Operand`] can hold.
#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OperandType<'e> {
    /// A variable representing some CPU state, such as a register or a flag.
    #[cfg_attr(feature = "serde", serde(rename = "Register"))]
    Arch(ArchId<'e>),
    /// A constant integer.
    Constant(u64),
    /// A variable representing memory read. Scarf currently assumes memory to always be
    /// little-endian.
    Memory(MemAccess<'e>),
    /// Integer arithmetic expression of two `Operand`s.
    Arithmetic(ArithOperand<'e>),
    /// Float arithmetic expression of two `Operand`s.
    ///
    /// `MemAccessSize` is used to specify in what sized IEEE 754 floats (Effectively just
    /// 32- or 64-bit) the operation and input operands are. Note that constant inputs still
    /// use integer constant type `OperandType::Constant`, you will have to use [`f32::from_bits`]
    /// and [`f32::to_bits`] (Or equivalent `f64` functions for `MemAccessSize::Mem64`) to
    /// inspect constant inputs as floats.
    ArithmeticFloat(ArithOperand<'e>, MemAccessSize),
    /// An "completely unknown" variable, which is result of [merging two
    /// differing `Operand`s during a `FuncAnalysis`
    /// run](../analysis/struct.FuncAnalysis.html#state-merging-and-loops).
    Undefined(UndefinedId),
    /// Sign extends the input `Operand` from first `MemAccessSize` to second `MemAccessSize`.
    SignExtend(Operand<'e>, MemAccessSize, MemAccessSize),
    /// If-else / ternary operator: `if .0 != 0 { .1 } else { .2 }`.
    /// Note: Any nonzero value for condition is considered true, no limitation that
    /// condition has to be 0/1.
    Select(Operand<'e>, Operand<'e>, Operand<'e>),
    /// Arbitrary user-defined variable.
    ///
    /// This is never created by scarf itself, and the user code may assign any meaning
    /// to the integer value.
    Custom(u32),
}

#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ArithOperand<'e> {
    pub ty: ArithOpType,
    pub left: Operand<'e>,
    pub right: Operand<'e>,
}

/// Enum of arithmetic operations used by scarf [`Operand`]s. Used by
/// [`OperandType::Arithmetic and OperandType::ArithmeticFloat`](OperandType), which
/// in turn contain [`ArithOperand`], containing this enum.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ArithOpType {
    /// Addition. Wraps on overflow.
    Add = 0,
    /// Subtraction. Wraps on overflow.
    Sub,
    /// Multiplication. Wraps on overflow, though overflowing bits are accessible with `MulHigh`
    /// arithmetic using same inputs.
    Mul,
    /// High 64 bits of a 128-bit multiplication result.
    MulHigh,
    /// Signed multiplication. Does this even work?
    SignedMul,
    /// Division.
    Div,
    /// Modulo operation.
    Modulo,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Bitwise XOR.
    Xor,
    /// Left shift. Shifting by more than 64 results in 0.
    Lsh,
    /// Right shift. Shifting by more than 64 results in 0.
    Rsh,
    /// `1` if LHS is equal to RHS, otherwise 0.
    Equal,
    /// `1` if LHS is greater than RHS, otherwise 0. Inputs are treated as unsigned.
    GreaterThan,
    /// Converts 64-bit integer to 32-bit float (`OperandType::Arithmetic`),
    /// or 64-bit float to 32-bit float (`OperandType::Arithmetic`)
    ///
    /// Result is considered to be the raw integer representation of the float.
    ///
    /// `ArithOperand.right` is not used.
    ToFloat,
    /// Converts 64-bit integer to 64-bit float (`OperandType::Arithmetic`),
    /// or 32-bit float to 64-bit float (`OperandType::ArithmeticFloat`)
    ///
    /// Result is considered to be the raw integer representation of the float.
    ///
    /// `ArithOperand.right` is not used.
    ToDouble,
    /// Only meaningful with `OperandType::ArithmeticFloat`. Converts either 32- or 64-bit
    /// float to signed integer.
    ///
    /// The result integer size is same as input float size, with `0x8000_0000` or
    /// `0x8000_0000_0000_0000` representing overflow. 32-bit float to 64-bit integer conversion
    /// is represented as `ToInt(ToDouble(input))`
    ///
    /// `ArithOperand.right` is not used.
    ToInt,
}

impl<'e> ArithOperand<'e> {
    /// Returns true only for `ArithOpType::Equal` and `ArithOpType::GreaterThan`.
    pub fn is_compare_op(&self) -> bool {
        use self::ArithOpType::*;
        match self.ty {
            Equal | GreaterThan => true,
            _ => false,
        }
    }
}

/// Represents architecture (`ExecutionState`) -specific state.
///
/// The first 256 values are defined to be general purpose registers, from
/// `OperandCtx::register`. What state other values map (As well as what registers
/// map to what state) can vary between `ExecutionState` implementations.
#[cfg_attr(feature = "serde", derive(Serialize), serde(transparent))]
#[derive(Clone, Copy)]
pub struct ArchId<'e> {
    value: u32,
    #[cfg_attr(feature = "serde", serde(skip))]
    arch_def: &'e ArchDefinition<'e>,
}

impl<'e> ArchId<'e> {
    /// Gets the actual value. Values 0-255 represent registers that can
    /// be also created through `OperandCtx::register` and accessed with
    /// `Operand::if_register`
    #[inline]
    pub fn value(&self) -> u32 {
        self.value
    }

    /// Gets the value if it represents a register. (That is, the value is less than 256)
    #[inline]
    pub fn if_register(&self) -> Option<u8> {
        u8::try_from(self.value).ok()
    }

    fn if_tag_u8(&self, tag: u32) -> Option<u8> {
        if disasm::x86_arch_tag(self.value) == tag {
            Some(self.value as u8)
        } else {
            None
        }
    }

    #[inline]
    pub fn if_x86_register_32(&self) -> Option<u8> {
        self.if_tag_u8(disasm::X86_REGISTER32_TAG)
    }

    #[inline]
    pub fn if_x86_register_16(&self) -> Option<u8> {
        self.if_tag_u8(disasm::X86_REGISTER16_TAG)
    }

    #[inline]
    pub fn if_x86_register_8_low(&self) -> Option<u8> {
        self.if_tag_u8(disasm::X86_REGISTER8_LOW_TAG)
    }

    #[inline]
    pub fn if_x86_register_8_high(&self) -> Option<u8> {
        self.if_tag_u8(disasm::X86_REGISTER8_HIGH_TAG)
    }

    pub fn if_x86_flag(&self) -> Option<Flag> {
        if disasm::x86_arch_tag(self.value) == disasm::X86_FLAG_TAG {
            Some(Flag::x86_from_arch(self.value as u8))
        } else {
            None
        }
    }
}

impl<'e> Eq for ArchId<'e> { }

impl<'e> PartialEq for ArchId<'e> {
    #[inline]
    fn eq(&self, other: &ArchId<'e>) -> bool {
        self.value == other.value
    }
}

/// Newtype to distinguish each [`OperandType::Undefined`] from each other.
///
/// User code should rarely have need to use this.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct UndefinedId(#[cfg_attr(feature = "serde", serde(skip))] pub u32);

const FIRST_REGISTER_OP_INDEX: usize = 0;
const FIRST_FLAG_OP_INDEX: usize = FIRST_REGISTER_OP_INDEX + 0x10;
const FIRST_CONSTANT_OP_INDEX: usize = FIRST_FLAG_OP_INDEX + 6;
const SMALL_CONSTANT_COUNT: usize = 0x110;
const SIGN_AND_MASK_NEARBY_CONSTS: usize = 0x11;
const SIGN_AND_MASK_NEARBY_CONSTS_FIRST: usize = FIRST_CONSTANT_OP_INDEX + SMALL_CONSTANT_COUNT;
const COMMON_OPERANDS_COUNT: usize = SMALL_CONSTANT_COUNT + 0x10 + 0x6 +
    SIGN_AND_MASK_NEARBY_CONSTS * 5 + SIGN_AND_MASK_NEARBY_CONSTS / 2;

#[inline]
const fn sign_mask_const_index(group: usize) -> usize {
    SIGN_AND_MASK_NEARBY_CONSTS_FIRST + group * SIGN_AND_MASK_NEARBY_CONSTS
}

/// An arena for allocating and interning [`Operand`]s.
///
/// Usually the type alias to reference [`OperandCtx`] is used instead.
/// A variable of type `OperandContext` / `OperandCtx` is usually named `ctx`.
///
/// `OperandContext` is used to create `Operand`s, returning reference to already existing
/// `Operand` if it was already created, allowing equality comparisons be done with a simple
/// pointer equality check.
///
/// Additionally `OperandContext` makes sure that all created `Operand`s are simplified and
/// canonicalized. For example, `rcx - rcx` is simplified to a constant `0`, while `rax + rcx`
/// and `rcx + rax` return references to the same `Operand`. While the choice between
/// `rax + rcx` or `rcx + rax` for canonical form is not specified, some canonicalizations
/// [are guaranteed](./struct.Operand.html#canonicalization-guarantees),
/// and user code may rely on them to simplify matching of `Operand`s.
///
/// Most of the functions of `OperandContext` are for creating an [`Operand`] with specific
/// [`OperandType`], though there is also a [`mem_access`](Self::mem_access) to create
/// [`MemAccess`] (As it also has canonicalization requirements), and few general-use
/// functions
///
/// # `OperandContext` lifetime `'e`
///
/// The `'e` lifetime passed around everywhere in scarf ultimately refers to `Operand`s, which
/// are a reference to `&'e OperandContext<'e>`, usually shortened with a type alias to
/// [`OperandCtx<'e>`].
///
/// `'e` is an invariant lifetime, which generally makes it impossible to mix allocations from
/// two different `OperandContext`s. However, if you have `OperandCtx<'static>` or create two
/// separate `OperandContext`s in same scope, Rust allows assigning the same lifetime to them,
/// allowing the allocations to be mixed. While this is not expected to cause memory safety
/// issues, it will make scarf produce nonsensical results as equality comparisons are no
/// longer reliable. Try to avoid cases where there are two `OperandContext` with same lifetime.
///
/// A common use case for multiple `OperandContext`s is to use a shorter-lived one to analyze
/// functions (Which can allocate tens or hundrerds of thousands `Operand`s total), that will
/// be throwns away to release memory afterwards, and a longer-lived `OperandContext` to hold
/// `Operand` results that you are interested in.
/// `Operand`s can be copied across `OperandContext` with [`copy_operand`](Self::copy_operand).
///
/// # Cheap-to-access `Operand`s
///
/// In general, creating an `Operand` requires at least a hash table lookup / insertion to
/// make sure `Operand`s are equal if their references are equal. As such, avoiding
/// doing repeated hash table lookups by storing the returned `Operand` is preferrable.
/// However, few of the most common `Operand`s are already cached by `OperandContext`, and
/// any code having `OperandContext` reference can rely on retreiving these `Operand`s being
/// a single memory access.
///
/// The `Operand`s which are guaranteed to be cached are:
///
/// - `OperandType::Arch(n)`, for `n.value() < 16`
/// - `OperandType::Constant(n)`, for 0, 1, 2, 4, 8, through [`ctx.const_0()`](Self::const_0)
///     and other functions.
///
/// Some code uses this to micro-optimize cases where you want to check if `Operand` equals
/// to constant zero/one or constant register.
/// `op.if_constant() == 0` has to do two comparisons and two memory reads:
/// One to verify `OperandType::Constant` variant, second to verify that the constant is zero.
/// `op == ctx.const_0()` does just a single memory read to get the cached const_0, and
/// a single pointer comparison.
pub struct OperandContext<'e> {
    next_undefined: Cell<u32>,
    max_undefined: Cell<u32>,
    // Contains 0x10 registers, 0x6 flags, SMALL_CONSTANT_COUNT small constants,
    common_operands: Box<[OperandSelfRef; COMMON_OPERANDS_COUNT]>,
    // Mutable state that simplify code uses for caching.
    // Initialized to full of const_0()
    simplify_cache: [Cell<OperandSelfRef>; SIMPLIFY_CACHE_SIZE],
    interner: intern::Interner<'e>,
    const_interner: intern::ConstInterner<'e>,
    undef_interner: intern::UndefInterner,
    invariant_lifetime: PhantomData<&'e mut &'e ()>,
    simplify_temp_stack: SliceStack,
    // Keep buffer for ExecutionState's Operation::Freeze handling
    // Unless the user does something weird with intercepting the freeze operations,
    // this should be the only one that gets ever used
    // (Semi-ugly code organization for operand to slightly depend on exec_state,
    // but oh well)
    freeze_buffer: RefCell<Vec<exec_state::FreezeOperation<'e>>>,
    // Hashmap of other_op -> own_op, for large copy_operand() calls.
    // Key is usize, cast from other_op pointer.
    // Cleared after every copy_operand call.
    copy_operand_cache: RefCell<FxHashMap<usize, Operand<'e>>>,
    disasm_cache: RefCell<Option<Box<DisasmCache<'e>>>>,
    arch_def: &'e ArchDefinition<'e>,
}

pub struct ArchDefinition<'a> {
    /// Allows specifying different variable sizes for 65536-sized blocks
    /// of [`ArchId`]. So key is `arch_id.value() >> 16`. `ArchId` values that map to a larger
    /// key than this list is will get default treatment of 64 bit size.
    pub tag_definitions: &'a [ArchTagDefinition],
    /// Allows specifying custom `fmt::Display` for `ArchId` values.
    pub fmt: Option<fn(&mut fmt::Formatter, u32) -> fmt::Result>,
}

#[derive(Copy, Clone, Debug)]
pub struct ArchTagDefinition {
    /// How many bits large `Operand` with this `ArchId` tag is.
    /// Default is 64, but flags can use 1, and smaller registers may use 32 and such.
    /// 0 or values larger than 64 are not allowed.
    pub size: u8,
}

/// Convenience alias for [`OperandContext`] reference that avoids having to
/// type the `'e` lifetime twice.
///
/// This is the type used in all function parameters, `OperandContext` is only needed when
/// constructing it.
pub type OperandCtx<'e> = &'e OperandContext<'e>;

/// Represents an Operand<'e> stored in OperandContext.
/// Rust doesn't allow this sort of setup without unsafe code,
/// so have this sort of wrapper to somewhat separate these
/// unsafe casts from other operands.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct OperandSelfRef(*const ());

impl OperandSelfRef {
    #[inline]
    fn new<'e>(operand: Operand<'e>) -> OperandSelfRef {
        OperandSelfRef(operand.0 as *const OperandBase<'e> as *const ())
    }

    #[inline]
    unsafe fn cast<'e>(self) -> Operand<'e> {
        Operand(&*(self.0 as *const OperandBase<'e>), PhantomData)
    }
}

unsafe impl Send for OperandSelfRef {}

/// A single operand which carries all of its data with it.
///
/// Actually accessing the operand requires calling `operand()`, which
/// gives the operand with lifetime bound to this struct.
pub struct SelfOwnedOperand {
    #[allow(dead_code)]
    ctx: OperandContext<'static>,
    op: OperandSelfRef,
}

pub struct Iter<'e>(Option<IterState<'e>>);
pub struct IterNoMemAddr<'e>(Option<IterState<'e>>);

trait IterVariant<'e> {
    fn descend_to_mem_addr() -> bool;
    fn state<'b>(&'b mut self) -> &'b mut Option<IterState<'e>>;
}

impl<'e> IterVariant<'e> for Iter<'e> {
    fn descend_to_mem_addr() -> bool {
        true
    }

    fn state<'b>(&'b mut self) -> &'b mut Option<IterState<'e>> {
        &mut self.0
    }
}

fn iter_variant_next<'e, T: IterVariant<'e>>(s: &mut T) -> Option<Operand<'e>> {
    use self::OperandType::*;

    let inner = match s.state() {
        Some(ref mut s) => s,
        None => return None,
    };
    let next = inner.pos;

    match *next.ty() {
        Arithmetic(ref arith) | ArithmeticFloat(ref arith, _) => {
            inner.pos = arith.left;
            inner.stack.push(arith.right);
        },
        Memory(ref m) if T::descend_to_mem_addr() => {
            inner.pos = m.address().0;
        }
        SignExtend(val, _, _) => {
            inner.pos = val;
        }
        Select(a, b, c) => {
            inner.pos = a;
            inner.stack.reserve(2);
            inner.stack.push(b);
            inner.stack.push(c);
        }
        _ => {
            match inner.stack.pop() {
                Some(s) => inner.pos = s,
                _ => {
                    *s.state() = None;
                }
            }
        }
    }
    Some(next)
}

impl<'e> IterVariant<'e> for IterNoMemAddr<'e> {
    fn descend_to_mem_addr() -> bool {
        false
    }

    fn state<'b>(&'b mut self) -> &'b mut Option<IterState<'e>> {
        &mut self.0
    }
}

struct IterState<'e> {
    pos: Operand<'e>,
    stack: Vec<Operand<'e>>,
}

impl<'e> Iterator for Iter<'e> {
    type Item = Operand<'e>;
    fn next(&mut self) -> Option<Operand<'e>> {
        iter_variant_next(self)
    }
}

impl<'e> Iterator for IterNoMemAddr<'e> {
    type Item = Operand<'e>;
    fn next(&mut self) -> Option<Operand<'e>> {
        iter_variant_next(self)
    }
}

macro_rules! operand_context_const_methods {
    ($lt:lifetime, $($name:ident, $val:expr,)*) => {
        $(
            // Inline and a bit redundant implementation to allow
            // const_0() and other ones with less than 0x40 be
            // just an array read which is known to be in bounds.
            /// Cheap access to a small constant.
            #[inline]
            pub fn $name(&$lt self) -> Operand<$lt> {
                unsafe { self.common_operands[FIRST_CONSTANT_OP_INDEX + $val].cast() }
            }
        )*
    }
}

#[cfg(feature = "fuzz")]
thread_local! {
    static SIMPLIFICATION_INCOMPLETE: Cell<bool> = Cell::new(false);
}

#[cfg(feature = "fuzz")]
fn tls_simplification_incomplete() {
    SIMPLIFICATION_INCOMPLETE.with(|x| x.set(true));
}

#[cfg(feature = "fuzz")]
pub fn check_tls_simplification_incomplete() -> bool {
    SIMPLIFICATION_INCOMPLETE.with(|x| x.replace(false))
}

impl<'e> OperandContext<'e> {
    /// Creates a new OperandContext.
    ///
    /// This will immediately spend some time interning the most common `Operand`s that will
    /// be cheap to access later, maybe making this relatively slow operation compared to rest
    /// of scarf. See [docs on guarantees](./struct.OperandContext.html#cheap-to-access-operands)
    /// for `Operand`s that are guaranteed to be cheap to access afterwards.
    pub fn new() -> OperandContext<'e> {
        Self::new_with_arch(&crate::exec_state::X86_64_ARCH_DEFINITION)
    }

    pub fn new_with_arch(arch_def: &'e ArchDefinition<'e>) -> OperandContext<'e> {
        use std::ptr::null_mut;
        for tag_def in arch_def.tag_definitions.iter() {
            if tag_def.size == 0 || tag_def.size > 64 {
                panic!("Invalid arch definition tag size {}", tag_def.size);
            }
        }
        let common_operands =
            Box::alloc().init([OperandSelfRef(null_mut()); COMMON_OPERANDS_COUNT]);
        let mut result: OperandContext<'e> = OperandContext {
            next_undefined: Cell::new(0),
            max_undefined: Cell::new(0),
            common_operands,
            simplify_cache: array_init::array_init(|_| Cell::new(OperandSelfRef(null_mut()))),
            interner: intern::Interner::new(),
            const_interner: intern::ConstInterner::new(),
            undef_interner: intern::UndefInterner::new(),
            invariant_lifetime: PhantomData,
            simplify_temp_stack: SliceStack::new(),
            freeze_buffer: RefCell::new(Vec::new()),
            copy_operand_cache: RefCell::new(FxHashMap::default()),
            disasm_cache: RefCell::new(None),
            arch_def,
        };
        let common_operands = &mut result.common_operands;
        // Accessing interner here would force the invariant lifetime 'e to this stack frame.
        // Cast the interner reference to arbitrary lifetime to allow returning the result.
        let interner: &intern::Interner<'_> = unsafe { std::mem::transmute(&result.interner) };
        let const_interner: &intern::ConstInterner<'_> =
            unsafe { std::mem::transmute(&result.const_interner) };
        let base = FIRST_CONSTANT_OP_INDEX;
        for i in 0..SMALL_CONSTANT_COUNT {
            common_operands[base + i] = const_interner.add_uninterned(i as u64).self_ref();
        }
        let base = FIRST_REGISTER_OP_INDEX;
        for i in 0..0x10 {
            let arch_id = ArchId {
                value: i as u32,
                arch_def,
            };
            common_operands[base + i] =
                interner.add_uninterned(&OperandType::Arch(arch_id)).self_ref();
        }
        let base = FIRST_FLAG_OP_INDEX;
        let flags = [
            Flag::Zero, Flag::Carry, Flag::Overflow, Flag::Sign, Flag::Parity, Flag::Direction,
        ];
        for (i, &f) in flags.iter().enumerate() {
            let arch_id = ArchId {
                value: f as u32 | disasm::make_x86_arch_tag(disasm::X86_FLAG_TAG),
                arch_def,
            };
            common_operands[base + i] =
                interner.add_uninterned(&OperandType::Arch(arch_id)).self_ref();
        }
        let sign_mask_consts = [
            0x7fffu64, 0xffff, 0x7fff_ffff, 0xffff_ffff, 0x7fff_ffff_ffff_ffff,
            0xffff_ffff_ffff_ffff
        ];
        let mut pos = SIGN_AND_MASK_NEARBY_CONSTS_FIRST;
        for mid in sign_mask_consts {
            let count = if mid == 0xffff_ffff_ffff_ffff {
                SIGN_AND_MASK_NEARBY_CONSTS / 2
            } else {
                SIGN_AND_MASK_NEARBY_CONSTS
            };
            let start = mid.wrapping_sub(SIGN_AND_MASK_NEARBY_CONSTS as u64 / 2).wrapping_add(1);
            for val in 0..count {
                let n = start.wrapping_add(val as u64);
                common_operands[pos] = const_interner.add_uninterned(n as u64).self_ref();
                pos += 1;
            }
        }
        let zero = common_operands[FIRST_CONSTANT_OP_INDEX];
        for i in 0..SIMPLIFY_CACHE_SIZE {
            result.simplify_cache[i].set(zero);
        }
        result
    }

    /// Returns a struct that is used in conjuction with serde's `deserialize_seed` functions
    /// to allocate deserialized operands with lifetime of this `OperandContext`.
    #[cfg(feature = "serde")]
    pub fn deserialize_seed(&'e self) -> DeserializeOperand<'e> {
        DeserializeOperand(self)
    }

    /// Copies an operand referring to some other OperandContext to this OperandContext
    /// and returns the copied reference.
    pub fn copy_operand<'other>(&'e self, op: Operand<'other>) -> Operand<'e> {
        let self_arch_def = self.arch_def as *const _ as *const ();
        let mut ctx = CopyOperandCtx {
            arch_def: self_arch_def,
            recurse_limit: 32,
        };
        let ret = self.copy_operand_small(op, &mut ctx);
        if cfg!(debug_assertions) && !std::ptr::eq(ctx.arch_def, self_arch_def) {
            // Just a debug assertion since this is just similarly weird as Undefined copying.
            // Could allow cases where arch operand sizes are same though.
            panic!("Copied operand {op} to OperandCtx with different arch definition");
        }
        ret
    }

    fn copy_operand_small<'a: 'e + 'other, 'other>(
        &'e self,
        op: Operand<'other>,
        ctx: &mut CopyOperandCtx,
    ) -> Operand<'e> {
        if ctx.recurse_limit <= 1 {
            let mut map = self.copy_operand_cache.borrow_mut();
            if ctx.recurse_limit == 1 {
                map.clear();
                ctx.recurse_limit = 0;
            }
            return self.copy_operand_large(op, &mut map, ctx)
        }
        ctx.recurse_limit -= 1;
        let ty = match *op.ty() {
            OperandType::Arch(arch) => {
                ctx.arch_def = arch.arch_def as *const ArchDefinition<'_> as *const ();
                OperandType::Arch(ArchId {
                    value: arch.value,
                    arch_def: self.arch_def,
                })
            }
            OperandType::Constant(c) => OperandType::Constant(c),
            OperandType::Undefined(c) => OperandType::Undefined(c),
            OperandType::Custom(c) => OperandType::Custom(c),
            OperandType::Memory(ref mem) => OperandType::Memory(MemAccess {
                base: self.copy_operand_small(mem.base, ctx),
                offset: mem.offset,
                size: mem.size,
                const_base: mem.const_base,
            }),
            OperandType::Arithmetic(ref arith) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_small(arith.left, ctx),
                    right: self.copy_operand_small(arith.right, ctx),
                };
                OperandType::Arithmetic(arith)
            }
            OperandType::ArithmeticFloat(ref arith, size) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_small(arith.left, ctx),
                    right: self.copy_operand_small(arith.right, ctx),
                };
                OperandType::ArithmeticFloat(arith, size)
            }
            OperandType::SignExtend(a, b, c) => {
                OperandType::SignExtend(self.copy_operand_small(a, ctx), b, c)
            }
            OperandType::Select(a, b, c) => {
                OperandType::Select(
                    self.copy_operand_small(a, ctx),
                    self.copy_operand_small(b, ctx),
                    self.copy_operand_small(c, ctx),
                )
            }
        };
        self.intern_any(ty)
    }

    fn copy_operand_large<'a: 'e + 'other, 'other>(
        &'e self,
        op: Operand<'other>,
        cache: &mut FxHashMap<usize, Operand<'e>>,
        ctx: &mut CopyOperandCtx,
    ) -> Operand<'e> {
        let key = op.0 as *const OperandBase<'other> as usize;
        if let Some(&result) = cache.get(&key) {
            return result;
        }

        let ty = match *op.ty() {
            OperandType::Arch(arch) => {
                ctx.arch_def = arch.arch_def as *const ArchDefinition<'_> as *const ();
                OperandType::Arch(ArchId {
                    value: arch.value,
                    arch_def: self.arch_def,
                })
            }
            OperandType::Constant(c) => OperandType::Constant(c),
            OperandType::Undefined(c) => OperandType::Undefined(c),
            OperandType::Custom(c) => OperandType::Custom(c),
            OperandType::Memory(ref mem) => OperandType::Memory(MemAccess {
                base: self.copy_operand_large(mem.base, cache, ctx),
                offset: mem.offset,
                size: mem.size,
                const_base: mem.const_base,
            }),
            OperandType::Arithmetic(ref arith) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_large(arith.left, cache, ctx),
                    right: self.copy_operand_large(arith.right, cache, ctx),
                };
                OperandType::Arithmetic(arith)
            }
            OperandType::ArithmeticFloat(ref arith, size) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_large(arith.left, cache, ctx),
                    right: self.copy_operand_large(arith.right, cache, ctx),
                };
                OperandType::ArithmeticFloat(arith, size)
            }
            OperandType::SignExtend(a, b, c) => {
                OperandType::SignExtend(self.copy_operand_large(a, cache, ctx), b, c)
            }
            OperandType::Select(a, b, c) => {
                OperandType::Select(
                    self.copy_operand_large(a, cache, ctx),
                    self.copy_operand_large(b, cache, ctx),
                    self.copy_operand_large(c, cache, ctx),
                )
            }
        };
        let result = self.intern_any(ty);
        cache.insert(key, result);
        result
    }

    /// Interns operand on the default interner. Shouldn't be used for constants,
    /// arithmetic, or undefined
    fn intern(&'e self, ty: &OperandType<'e>) -> Operand<'e> {
        debug_assert!(
            match ty {
                OperandType::Constant(..) | OperandType::Undefined(..) |
                    OperandType::Arithmetic(..) => false,
                OperandType::Arch(r) => {
                    let is_fast_cased_reg = r.value() < 16;
                    let is_fast_cased_flag =
                        disasm::x86_arch_tag(r.value()) == disasm::X86_FLAG_TAG &&
                        r.value() < 6;
                    !is_fast_cased_reg && !is_fast_cased_flag
                }
                _ => true,
            },
            "General-purpose intern function called for OperandType with specialized interning",
        );

        self.interner.intern(ty)
    }

    fn intern_arith(
        &'e self,
        left: Operand<'e>,
        right: Operand<'e>,
        ty: ArithOpType,
    ) -> Operand<'e> {
        let arith = ArithOperand {
            ty,
            left,
            right,
        };
        let ty = OperandType::Arithmetic(arith);
        self.interner.intern(&ty)
    }

    fn intern_select(
        &'e self,
        condition: Operand<'e>,
        val_true: Operand<'e>,
        val_false: Operand<'e>,
    ) -> Operand<'e> {
        let ty = OperandType::Select(condition, val_true, val_false);
        self.interner.intern(&ty)
    }

    fn intern_any(&'e self, ty: OperandType<'e>) -> Operand<'e> {
        if let OperandType::Constant(c) = ty {
            self.constant(c)
        } else if let OperandType::Arch(r) = ty {
            self.arch(r.value())
        } else if let OperandType::Arithmetic(arith) = ty {
            self.intern_arith(arith.left, arith.right, arith.ty)
        } else {
            // Undefined has to go here since UndefInterner is just array that things get
            // pushed to and then looked up.. Maybe it would be better for copy_operand
            // to map existing Undefined to new undefined if it meets any?
            self.interner.intern(&ty)
        }
    }

    pub(crate) fn get_undef_pos(&'e self) -> u32 {
        self.next_undefined.get()
    }

    /// Resets next_undefined; keeps Undefined allocations low when
    /// same OperandCtx is used for several different analysises.
    ///
    /// User code should not rely on Undefined from different FuncAnalysis being
    /// distinguishable, ideally user code should avoid considering Operands with
    /// Undefined valid results.
    pub(crate) fn set_undef_pos(&'e self, val: u32) {
        self.next_undefined.set(val)
    }

    operand_context_const_methods! {
        'e,
        const_0, 0x0,
        const_1, 0x1,
        const_2, 0x2,
        const_4, 0x4,
        const_8, 0x8,
    }

    /// Cheap access to [`Flag::Zero`](OperandType) operand.
    #[inline]
    pub fn flag_z(&'e self) -> Operand<'e> {
        self.flag(Flag::Zero)
    }

    /// Cheap access to [`Flag::Carry`](OperandType) operand.
    #[inline]
    pub fn flag_c(&'e self) -> Operand<'e> {
        self.flag(Flag::Carry)
    }

    /// Cheap access to [`Flag::Overflow`](OperandType) operand.
    #[inline]
    pub fn flag_o(&'e self) -> Operand<'e> {
        self.flag(Flag::Overflow)
    }

    /// Cheap access to [`Flag::Sign`](OperandType) operand.
    #[inline]
    pub fn flag_s(&'e self) -> Operand<'e> {
        self.flag(Flag::Sign)
    }

    /// Cheap access to [`Flag::Parity`](OperandType) operand.
    #[inline]
    pub fn flag_p(&'e self) -> Operand<'e> {
        self.flag(Flag::Parity)
    }

    /// Cheap access to [`Flag::Direction`](OperandType) operand.
    #[inline]
    pub fn flag_d(&'e self) -> Operand<'e> {
        self.flag(Flag::Direction)
    }

    /// Cheap access to any [`OperandType::Flag`](OperandType) operand.
    #[inline]
    pub fn flag(&'e self, flag: Flag) -> Operand<'e> {
        self.flag_by_index(flag as usize)
    }

    pub(crate) fn flag_by_index(&'e self, index: usize) -> Operand<'e> {
        assert!(index < 6);
        unsafe { self.common_operands[FIRST_FLAG_OP_INDEX + index as usize].cast() }
    }

    /// Returns [`OperandType::Arch(num)`](OperandType) operand.
    ///
    /// If `num` is less than 16, the access is guaranteed to be cheap.
    #[inline]
    pub fn register(&'e self, num: u8) -> Operand<'e> {
        self.arch(num as u32)
    }

    #[inline]
    pub fn arch(&'e self, num: u32) -> Operand<'e> {
        if num < 0x10 {
            unsafe { self.common_operands[FIRST_REGISTER_OP_INDEX + num as usize].cast() }
        } else {
            self.arch_slow(num)
        }
    }

    /// Returns [`OperandType::Arch(id)`](OperandType) operand.
    fn arch_slow(&'e self, id: u32) -> Operand<'e> {
        const FIRST_COMMON_FLAG: u32 = disasm::make_x86_arch_tag(disasm::X86_FLAG_TAG);
        const LAST_COMMON_FLAG: u32 = FIRST_COMMON_FLAG + 6;
        if id >= FIRST_COMMON_FLAG && id < LAST_COMMON_FLAG {
            let index = FIRST_FLAG_OP_INDEX + (id - FIRST_COMMON_FLAG) as usize;
            unsafe { self.common_operands[index].cast() }
        } else {
            self.intern(&OperandType::Arch(ArchId {
                value: id,
                arch_def: self.arch_def,
            }))
        }
    }

    /// Returns [`OperandType::Custom(value)`](OperandType) operand.
    ///
    /// Custom operands are guaranteed to be never generated by scarf, allowing user code
    /// use them for arbitrary unknown variables of any meaning.
    pub fn custom(&'e self, value: u32) -> Operand<'e> {
        self.intern(&OperandType::Custom(value))
    }

    /// Returns [`OperandType::Constant(value)`](OperandType) operand.
    pub fn constant(&'e self, value: u64) -> Operand<'e> {
        if value < SMALL_CONSTANT_COUNT as u64 {
            unsafe { self.common_operands[FIRST_CONSTANT_OP_INDEX + value as usize].cast() }
        } else {
            // Fast path for values near sign bits or word masks
            // 7ff8 ..= 8008
            // fff8 ..= 1_0008
            // 7fff_fff8 ..= 8000_0008
            // ffff_fff8 ..= 1_0000_0008
            // 7fff_ffff_ffff_fff8 ..= 8000_0000_0000_0008
            // ffff_ffff_ffff_fff8 ..= ffff_ffff_ffff_ffff
            let index = ((value as u32 & 0x7fff).wrapping_add(0x8)) & 0x7fff;
            if index < SIGN_AND_MASK_NEARBY_CONSTS as u32 {
                let rest = value.wrapping_add(8) & 0xffff_ffff_ffff_8000;
                if rest.wrapping_sub(1) & rest == 0 {
                    let offset = match rest {
                        0x8000 => sign_mask_const_index(0),
                        0x1_0000 => sign_mask_const_index(1),
                        0x8000_0000 => sign_mask_const_index(2),
                        0x1_0000_0000 => sign_mask_const_index(3),
                        0x8000_0000_0000_0000 => sign_mask_const_index(4),
                        0x0 => sign_mask_const_index(5),
                        _ => return self.const_interner.intern(value),
                    };
                    let index = offset.wrapping_add(index as usize);
                    return unsafe { self.common_operands[index].cast() };
                }
            }
            self.const_interner.intern(value)
        }
    }

    /// Returns `Operand` for any arithmetic operation.
    pub fn arithmetic(
        &'e self,
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
    ) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_arith(left, right, ty, &mut simplify)
    }

    /// Returns `Operand` for any arithmetic operation, which will be then
    /// masked with a bitwise AND of `mask`.
    ///
    /// This may have better performance than the
    /// equivalent `ctx.and_const(ctx.arithmetic(ty, left, right), mask)`, as it is able
    /// to avoid allocating expressions that will be then immediately discarded.
    pub fn arithmetic_masked(
        &'e self,
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
        mask: u64,
    ) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_arith_masked(left, right, ty, mask, &mut simplify)
    }

    /// Returns `Operand` for a float arithmetic of any operation.
    pub fn float_arithmetic(
        &'e self,
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
        size: MemAccessSize,
    ) -> Operand<'e> {
        simplify::simplify_float_arith(left, right, ty, size, self)
    }

    /// Returns `Operand` for `left + right`.
    pub fn add(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_add_sub(left, right, false, u64::MAX, &mut simplify)
    }

    /// Returns `Operand` for `left - right`.
    pub fn sub(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_add_sub(left, right, true, u64::MAX, &mut simplify)
    }

    /// Returns `Operand` for `left * right`.
    pub fn mul(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_mul(left, right, &mut simplify)
    }

    /// Returns `Operand` for high 64 bits of 128-bit result of `left * right`.
    pub fn mul_high(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.arithmetic(ArithOpType::MulHigh, left, right)
    }

    /// Returns `Operand` for signed `left * right`.
    pub fn signed_mul(
        &'e self,
        left: Operand<'e>,
        right: Operand<'e>,
        _size: MemAccessSize,
    ) -> Operand<'e> {
        // TODO
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_mul(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left / right`.
    pub fn div(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.arithmetic(ArithOpType::Div, left, right)
    }

    /// Returns `Operand` for `left % right`.
    pub fn modulo(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.arithmetic(ArithOpType::Modulo, left, right)
    }

    /// Returns `Operand` for `left & right`.
    pub fn and(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_and(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    pub fn or(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_or(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    pub fn xor(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_xor(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    pub fn lsh(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_lsh(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    pub fn rsh(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_rsh(left, right, &mut simplify)
    }

    pub fn arithmetic_right_shift(
        &'e self,
        left: Operand<'e>,
        right: Operand<'e>,
        size: MemAccessSize,
    ) -> Operand<'e> {
        // Arithmetic shift shifts in the value of sign bit,
        // that can be represented as bitwise or of
        // `not(ffff...ffff << rhs >> rhs)`
        let sign_bit = 1u64 << (size.bits() - 1);
        let logical_rsh = self.rsh(left, right);
        let mask = (sign_bit << 1).wrapping_sub(1);
        let negative_shift_in_bits = if let Some(right) = right.if_constant() {
            let c = if right >= 64 {
                u64::MAX
            } else {
                (((mask << right) & mask) >> right) ^ mask
            };
            self.constant(c)
        } else {
            self.xor_const(
                self.rsh(
                    self.and_const(
                        self.lsh(
                            self.constant(mask),
                            right,
                        ),
                        mask,
                    ),
                    right,
                ),
                mask,
            )
        };
        let sign_bit_set = self.and_const(left, sign_bit);
        // Doing it as `(lhs >> rhs) | select(is_negative, sign_bit_shifted, 0)`
        self.or(
            self.select(sign_bit_set, negative_shift_in_bits, self.const_0()),
            logical_rsh,
        )
    }

    /// Returns `Operand` for `left == right`.
    pub fn eq(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_eq(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left != right`.
    pub fn neq(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.eq_const(self.eq(left, right), 0)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// Less than can be implemented by swapping `left` and `right`.
    ///
    /// Greater than or equal can be implemented by using `ctx.or(ctx.gt(a, b), ctx.eq(a, b))`
    pub fn gt(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_gt(left, right, &mut simplify)
    }

    /// Returns `Operand` for signed `left > right`.
    pub fn gt_signed(
        &'e self,
        left: Operand<'e>,
        right: Operand<'e>,
        size: MemAccessSize,
    ) -> Operand<'e> {
        let (mask, offset) = match size {
            MemAccessSize::Mem8 => (0xff, 0x80),
            MemAccessSize::Mem16 => (0xffff, 0x8000),
            MemAccessSize::Mem32 => (0xffff_ffff, 0x8000_0000),
            MemAccessSize::Mem64 => {
                let offset = 0x8000_0000_0000_0000;
                return self.gt(
                    self.add_const(left, offset),
                    self.add_const(right, offset),
                );
            }
        };
        let offset = self.constant(offset);
        self.gt(
            self.arithmetic_masked(
                ArithOpType::Add,
                left,
                offset,
                mask,
            ),
            self.arithmetic_masked(
                ArithOpType::Add,
                right,
                offset,
                mask,
            ),
        )
    }

    /// Returns `Operand` for `left + right`.
    pub fn add_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_add_const(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left - right`.
    pub fn sub_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_sub_const(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left - right`.
    pub fn sub_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_add_sub(left, right, true, u64::MAX, &mut simplify)
    }

    /// Returns `Operand` for `left * right`.
    pub fn mul_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        if right.wrapping_sub(1) & right == 0 {
            self.lsh_const(left, right.trailing_zeros() as u64)
        } else {
            let right = self.constant(right);
            let mut simplify = simplify::SimplifyCtx::new(self);
            simplify::simplify_mul(left, right, &mut simplify)
        }
    }

    /// Returns `Operand` for `left & right`.
    pub fn and_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify.and_const(left, right)
    }

    /// Returns `Operand` for `left | right`.
    pub fn or_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_or(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    pub fn xor_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_xor(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    pub fn lsh_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        if right >= 64 {
            self.const_0()
        } else {
            let mut simplify = simplify::SimplifyCtx::new(self);
            simplify::simplify_lsh_const(left, right as u8, &mut simplify)
        }
    }

    /// Returns `Operand` for `left << right`.
    pub fn lsh_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify::simplify_lsh(left, right, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    pub fn rsh_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        if right >= 64 {
            self.const_0()
        } else {
            let mut simplify = simplify::SimplifyCtx::new(self);
            simplify::simplify_rsh_const(left, right as u8, &mut simplify)
        }
    }

    /// Returns `Operand` for `left == right`.
    pub fn eq_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let mut simplify = simplify::SimplifyCtx::new(self);
        simplify.eq_const(left, right)
    }

    /// Returns `Operand` for `left != right`.
    ///
    /// Currently this is represented as `(left == right) == 0`.
    pub fn neq_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        // Avoid creating the inner eq if it is trivially going to be
        // removed again in outer eq simplify.
        if right == 0 && left.relevant_bits().end == 1 {
            return left;
        }
        self.eq_const(self.eq_const(left, right), 0)
    }

    /// Returns `Operand` for unsigned `left > right`.
    pub fn gt_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        self.gt(left, right)
    }

    /// Returns `Operand` for unsigned `left > right`.
    pub fn gt_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        self.gt(left, right)
    }

    /// Returns `Operand` for [`MemAccessSize::Mem64`] memory read, using given base and offset.
    #[inline]
    pub fn mem64(&'e self, base: Operand<'e>, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem64, base, offset)
    }

    /// Returns `Operand` for [`MemAccessSize::Mem32`] memory read, using given base and offset.
    #[inline]
    pub fn mem32(&'e self, base: Operand<'e>, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem32, base, offset)
    }

    /// Returns `Operand` for [`MemAccessSize::Mem16`] memory read, using given base and offset.
    #[inline]
    pub fn mem16(&'e self, base: Operand<'e>, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem16, base, offset)
    }

    /// Returns `Operand` for [`MemAccessSize::Mem8`] memory read, using given base and offset.
    #[inline]
    pub fn mem8(&'e self, base: Operand<'e>, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem8, base, offset)
    }

    /// Returns `Operand` for Mem64 with constant address.
    #[inline]
    pub fn mem64c(&'e self, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem64, self.const_0(), offset)
    }

    /// Returns `Operand` for Mem32 with constant address.
    #[inline]
    pub fn mem32c(&'e self, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem32, self.const_0(), offset)
    }

    /// Returns `Operand` for Mem16 with constant address.
    #[inline]
    pub fn mem16c(&'e self, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem16, self.const_0(), offset)
    }

    /// Returns `Operand` for Mem8 with constant address.
    #[inline]
    pub fn mem8c(&'e self, offset: u64) -> Operand<'e> {
        self.mem_any(MemAccessSize::Mem8, self.const_0(), offset)
    }

    /// Creates `Operand` referring to memory from parts that make up `MemAccess`.
    pub fn mem_any(
        &'e self,
        size: MemAccessSize,
        addr: Operand<'e>,
        offset: u64,
    ) -> Operand<'e> {
        let ty = OperandType::Memory(self.mem_access(addr, offset, size));
        self.intern(&ty)
    }

    /// Creates `Operand` referring to memory from `MemAccess`.
    pub fn memory(&'e self, mem: &MemAccess<'e>) -> Operand<'e> {
        let ty = OperandType::Memory(*mem);
        self.intern(&ty)
    }

    /// Creates `MemAccess` from base, offset, and size.
    pub fn mem_access(
        &'e self,
        base: Operand<'e>,
        offset: u64,
        size: MemAccessSize,
    ) -> MemAccess<'e> {
        let (base, offset2) = self.extract_add_sub_offset(base);
        MemAccess {
            base,
            offset: offset.wrapping_add(offset2),
            size,
            const_base: base == self.const_0(),
        }
    }

    /// Shortcut for `ctx.mem_access(base, offset, MemAccessSize::Mem8)`
    pub fn mem_access8(&'e self, base: Operand<'e>, offset: u64) -> MemAccess<'e> {
        self.mem_access(base, offset, MemAccessSize::Mem8)
    }

    /// Shortcut for `ctx.mem_access(base, offset, MemAccessSize::Mem16)`
    pub fn mem_access16(&'e self, base: Operand<'e>, offset: u64) -> MemAccess<'e> {
        self.mem_access(base, offset, MemAccessSize::Mem16)
    }

    /// Shortcut for `ctx.mem_access(base, offset, MemAccessSize::Mem16)`
    pub fn mem_access32(&'e self, base: Operand<'e>, offset: u64) -> MemAccess<'e> {
        self.mem_access(base, offset, MemAccessSize::Mem32)
    }

    /// Shortcut for `ctx.mem_access(base, offset, MemAccessSize::Mem16)`
    pub fn mem_access64(&'e self, base: Operand<'e>, offset: u64) -> MemAccess<'e> {
        self.mem_access(base, offset, MemAccessSize::Mem64)
    }

    /// Creates `Operand` representing `mem.address_op() + value` without creating
    /// intermediate `mem.address_op()`
    pub fn mem_add_op(&'e self, mem: &MemAccess<'e>, value: Operand<'e>) -> Operand<'e> {
        let (base, offset) = mem.address();
        let (rhs_base, rhs_offset) = self.extract_add_sub_offset(value);
        self.add_const(
            self.add(base, rhs_base),
            offset.wrapping_add(rhs_offset),
        )
    }

    /// Creates `Operand` representing `mem.address_op() - value` without creating
    /// intermediate `mem.address_op()`
    pub fn mem_sub_op(&'e self, mem: &MemAccess<'e>, value: Operand<'e>) -> Operand<'e> {
        let (base, offset) = mem.address();
        let (rhs_base, rhs_offset) = self.extract_add_sub_offset(value);
        // (x + c1) - (y + c2) == (x - y) + (c1 -c2)
        self.add_const(
            self.sub(base, rhs_base),
            offset.wrapping_sub(rhs_offset),
        )
    }

    /// Creates `Operand` representing `mem.address_op() + value` without creating
    /// intermediate `mem.address_op()`
    pub fn mem_add_const_op(&'e self, mem: &MemAccess<'e>, value: u64) -> Operand<'e> {
        let (base, offset) = mem.address();
        self.add_const(base, offset.wrapping_add(value))
    }

    /// Creates `Operand` representing `mem.address_op() - value` without creating
    /// intermediate `mem.address_op()`
    pub fn mem_sub_const_op(&'e self, mem: &MemAccess<'e>, value: u64) -> Operand<'e> {
        let (base, offset) = mem.address();
        self.add_const(base, offset.wrapping_sub(value))
    }

    pub(crate) fn extract_add_sub_offset(&'e self, value: Operand<'e>) -> (Operand<'e>, u64) {
        if let OperandType::Arithmetic(arith) = value.ty() {
            if arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub {
                if let Some(c) = arith.right.if_constant() {
                    let offset =
                        if arith.ty == ArithOpType::Add { c } else { 0u64.wrapping_sub(c) };
                    (arith.left, offset)
                } else {
                    (value, 0)
                }
            } else {
                (value, 0)
            }
        } else if let OperandType::Constant(c) = *value.ty() {
            (self.const_0(), c)
        } else {
            (value, 0)
        }
    }

    /// Creates an [`OperandType::SignExtend`] `Operand.`
    pub fn sign_extend(
        &'e self,
        val: Operand<'e>,
        from: MemAccessSize,
        to: MemAccessSize,
    ) -> Operand<'e> {
        simplify::simplify_sign_extend(val, from, to, self)
    }

    /// Creates an [`OperandType::Select`] `Operand.`
    /// `if condition != 0 { val_true } else { val_false }
    pub fn select(
        &'e self,
        condition: Operand<'e>,
        val_true: Operand<'e>,
        val_false: Operand<'e>,
    ) -> Operand<'e> {
        simplify::simplify_select(self, condition, val_true, val_false)
    }

    /// Creates a new `OperandType::Undefined(id)` `Operand`, where the `id` is
    /// unique until all currently ongoing
    /// [`FuncAnalysis::analyze`](crate::analysis::FuncAnalysis::analyze) calls have returned.
    ///
    /// See [FuncAnalysis docs](../analysis/struct.FuncAnalysis.html#state-merging-and-loops)
    /// for more information about how
    /// `OperandType::Undefined` is used by analysis to determine which branches have
    /// to be rechecked.
    pub fn new_undef(&'e self) -> Operand<'e> {
        let id = self.next_undefined.get();
        let next = id.wrapping_add(1);
        self.next_undefined.set(next);
        if id == self.max_undefined.get() {
            self.max_undefined.set(next);
            self.undef_interner.push(OperandBase {
                ty: OperandType::Undefined(UndefinedId(id)),
                relevant_bits: 0..64,
                flags: FLAG_CONTAINS_UNDEFINED | FLAG_32BIT_NORMALIZED,
                type_alt_tag: 0,
                sort_order: id as u64,
                relevant_bits_mask: u64::MAX,
            })
        } else {
            self.undef_interner.get(id as usize)
        }
    }

    /// Walks through sub-operands of an `Operand`, calling the provided callback
    /// that can choose to replace any of them with a new `Operand` by returning `Some()`.
    pub fn transform<F>(&'e self, oper: Operand<'e>, depth_limit: usize, mut f: F) -> Operand<'e>
    where F: FnMut(Operand<'e>) -> Option<Operand<'e>>
    {
        self.transform_internal(oper, depth_limit, &mut f)
    }

    fn transform_internal<F>(
        &'e self,
        oper: Operand<'e>,
        depth_limit: usize,
        f: &mut F,
    ) -> Operand<'e>
    where F: FnMut(Operand<'e>) -> Option<Operand<'e>>
    {
        if depth_limit == 0 {
            return oper;
        }
        if let Some(val) = f(oper) {
            return val;
        }
        match *oper.ty() {
            OperandType::Arithmetic(ref arith) => {
                let left = self.transform_internal(arith.left, depth_limit - 1, f);
                let right = self.transform_internal(arith.right, depth_limit - 1, f);
                if left == arith.left && right == arith.right {
                    oper
                } else {
                    self.arithmetic(arith.ty, left, right)
                }
            },
            OperandType::Memory(ref m) => {
                let (base, offset) = m.address();
                let new_base = self.transform_internal(base, depth_limit - 1, f);
                if base == new_base {
                    oper
                } else {
                    self.mem_any(m.size, new_base, offset)
                }
            }
            OperandType::SignExtend(val, from, to) => {
                let new_val = self.transform_internal(val, depth_limit - 1, f);
                if val == new_val {
                    oper
                } else {
                    self.sign_extend(new_val, from, to)
                }
            }
            OperandType::Select(a, b, c) => {
                let new_a = self.transform_internal(a, depth_limit - 1, f);
                let new_b = self.transform_internal(b, depth_limit - 1, f);
                let new_c = self.transform_internal(c, depth_limit - 1, f);
                if a == new_a && b == new_b && c == new_c {
                    oper
                } else {
                    self.select(a, b, c)
                }
            }
            _ => oper,
        }
    }

    pub fn substitute(
        &'e self,
        oper: Operand<'e>,
        val: Operand<'e>,
        with: Operand<'e>,
        depth_limit: usize,
    ) -> Operand<'e> {
        if let Some(mem) = val.if_memory() {
            // Transform also Mem16[mem.addr] to with & 0xffff if val is Mem32, etc.
            // I guess recursing inside mem.addr doesn't make sense here,
            // but didn't give it too much thought.
            self.transform(oper, depth_limit, |old| {
                old.if_memory()
                    .filter(|old| old.address() == mem.address())
                    .filter(|old| old.size.bits() <= mem.size.bits())
                    .map(|old| {
                        if mem.size == old.size || old.size == MemAccessSize::Mem64 {
                            with
                        } else {
                            let mask = match old.size {
                                MemAccessSize::Mem64 => unreachable!(),
                                MemAccessSize::Mem32 => 0xffff_ffff,
                                MemAccessSize::Mem16 => 0xffff,
                                MemAccessSize::Mem8 => 0xff,
                            };
                            self.and_const(with, mask)
                        }
                    })
            })
        } else {
            self.transform(oper, depth_limit, |old| match old == val {
                true => Some(with),
                false => None,
            })
        }
    }

    /// Gets counts of different types of operands interned.
    ///
    /// Intented for debug / diagnostic info.
    pub fn interned_counts(&self) -> InternedCounts {
        InternedCounts {
            other: self.interner.interned_count(),
            undefined: self.max_undefined.get() as usize,
            constant: self.const_interner.interned_count(),
        }
    }

    pub(crate) fn simplify_temp_stack(&'e self) -> &'e SliceStack {
        &self.simplify_temp_stack
    }

    #[inline]
    pub(crate) fn simplify_cache_get(&'e self, index: usize) -> Operand<'e> {
        unsafe { self.simplify_cache[index].get().cast() }
    }

    #[inline]
    pub(crate) fn simplify_cache_set(&'e self, index: usize, values: &[Operand<'e>]) {
        assert!(index + values.len() <= SIMPLIFY_CACHE_SIZE);
        for i in 0..values.len() {
            self.simplify_cache[index + i].set(values[i].self_ref());
        }
    }

    pub(crate) fn swap_freeze_buffer(&'e self, other: &mut Vec<exec_state::FreezeOperation<'e>>) {
        let mut own = self.freeze_buffer.borrow_mut();
        std::mem::swap(&mut *own, other);
    }

    fn disasm_cache_borrow_mut(&self, arch: DisasmArch) -> RefMut<'_, DisasmCache<'e>> {
        let val = self.disasm_cache.borrow_mut();
        RefMut::map(val, |x| {
            let inner = x.get_or_insert_with(|| Box::new(DisasmCache::new(arch)));
            inner.set_arch(arch);
            &mut **inner
        })
    }

    pub(crate) fn disasm_cache_read(
        &'e self,
        arch: DisasmArch,
        instruction: &[u8; 8],
        length: usize,
        out: &mut Vec<Operation<'e>>,
    ) -> bool {
        let cache = self.disasm_cache_borrow_mut(arch);
        if let Some(ops) = cache.get(instruction, length) {
            out.extend_from_slice(ops);
            true
        } else {
            false
        }
    }

    pub(crate) fn disasm_cache_write(
        &'e self,
        arch: DisasmArch,
        instruction: &[u8; 8],
        length: usize,
        data: &[Operation<'e>],
    ) {
        let mut cache = self.disasm_cache_borrow_mut(arch);
        cache.set(instruction, length, data);
    }

    /// Returns operand limited to low `size` bits.
    pub fn truncate(&'e self, operand: Operand<'e>, size: u8) -> Operand<'e> {
        let high = 64 - size;
        let mask = !0u64 << high >> high;
        self.and_const(operand, mask)
    }

    /// Converts operand to '32-bit normalized form'
    ///
    /// This function exists for 32-bit `ExecutionState`, which wants to mainly
    /// deal with `Operand`s without caring about their high 32 bits.
    ///
    /// This could be achieved with just `ctx.and_const(value, 0xffff_ffff)` on every
    /// `Operand`, but these masked values would then need to be constantly unwrapped.
    /// For values that don't use the entire 32 bits, such as `Mem32[400] << 8`, the
    /// mask would become just `0xffff_ff00` after simplification, and determining if
    /// the mask is just used intentionally or if it is just to mask out high bits
    /// becomes more complicated.
    ///
    /// So instead of always zeroing high bits, the following kinds of `Operand`s have
    /// their '32-bit normalized form' be as is, *without* an bitwise AND mask, even if
    /// rest of scarf can consider their high bits to possibly have a meaning. `Operand`s
    /// for which scarf can prove low 32 bits be always zero will become zero though, even
    /// if they would otherwise be included in the rules below. (E.g. `Mem32[x] << 0x20`)
    ///
    /// - Any 'pure 64-bit variable'; `Arch`, `Undefined`, `Custom`
    /// - Additions, subtractions, consisting only 32-bit normalized
    ///     `Operand`s
    /// - Multiplications, left shifts consisting 32-bit normalized `Operand` and a constant
    /// - *Most XORs* that have both halves 32-bit normalized. (Details left loosely defined
    ///     for now though)
    /// - `OperandType::ArithmeticFloat`
    ///
    /// If the `Operand` is `AND 0xffff_ffff` masked, but the value being masked
    /// meets above conditions, the mask is removed.
    ///
    /// If the `Operand` is `AND 0xffff_ffff` masked addition or subtraction, with
    /// some of the terms meeting the above conditions, those terms are moved outside
    /// of the AND mask. E.g. `(rax + (rcx ^ rdx)) & ffff_ffff` becomes
    /// `rax + ((rcx ^ rdx) & ffff_ffff)`.
    ///
    /// In other cases bitwise AND mask `0xffff_ffff` is applied. (Which does not change the
    /// input value if all its high bits are already zero)
    ///
    /// In the end, it is intended, though not necessarily guaranteed in complex cases,
    /// that `ctx.normalize_32bit(x) == ctx.normalize_32bit(x & ffff_ffff)`.
    #[inline]
    pub fn normalize_32bit(&'e self, value: Operand<'e>) -> Operand<'e> {
        if value.0.flags & FLAG_32BIT_NORMALIZED == 0 {
            self.normalize_32bit_main(value)
        } else {
            value
        }
    }

    fn normalize_32bit_main(&'e self, value: Operand<'e>) -> Operand<'e> {
        let mut value = value;
        if value.relevant_bits().end > 32 {
            value = self.and_const(value, 0xffff_ffff);
            if value.0.flags & FLAG_32BIT_NORMALIZED != 0 {
                return value;
            }
            // Else value should have removable and now
        }
        // Only 32-bit value passed here should be and mask
        // that can be removed. (Or mask + shift)
        if let Some(arith) = value.if_arithmetic_any() {
            if value.0.flags & (FLAG_32BIT_NORMALIZED | FLAG_PARTIAL_32BIT_NORMALIZED_ADD)
                == FLAG_PARTIAL_32BIT_NORMALIZED_ADD
            {
                self.normalize_32bit_add(value)
            } else {
                if arith.ty == ArithOpType::And {
                    if arith.left.0.flags &
                        (FLAG_32BIT_NORMALIZED | FLAG_PARTIAL_32BIT_NORMALIZED_ADD) ==
                            FLAG_PARTIAL_32BIT_NORMALIZED_ADD
                    {
                        return self.normalize_32bit_add(arith.left);
                    }
                    return arith.left;
                } else if matches!(arith.ty, ArithOpType::Lsh | ArithOpType::Mul) {
                    // `(x & mask) << shift` to `x << shift`
                    // (Expected that mask can be removed or else it would
                    // already be 32bit normalized)
                    if let Some(inner_arith) = arith.left.if_arithmetic_any() {
                        if let Some(mut shift) = arith.right.if_constant() {
                            if arith.ty == ArithOpType::Mul {
                                if shift & shift.wrapping_sub(1) != 0 {
                                    // Expected to be unreachable but avoid big errors
                                    // if there is non-power-of-two mul
                                    return value;
                                }
                                shift = shift.trailing_zeros() as u64;
                            }
                            if inner_arith.left.0.flags & FLAG_32BIT_NORMALIZED != 0 {
                                return self.lsh_const(inner_arith.left, shift);
                            }
                        }
                    }
                }
                // Expected to be unreachable?
                value
            }
        } else {
            // Expected to be unreachable
            value
        }
    }

    fn normalize_32bit_add(&'e self, value: Operand<'e>) -> Operand<'e> {
        if let Some(arith) = value.if_arithmetic_any() {
            // Swap add/sub constants be less than 8000_0000
            if let Some(c) = arith.right.if_constant() {
                let max = match arith.ty {
                    ArithOpType::Add => 0x8000_0000,
                    _ => 0x7fff_ffff,
                };
                if (c as u32) > max {
                    let swapped_c = 0u32.wrapping_sub(c as u32) as u64;
                    let left = self.normalize_32bit(arith.left);
                    if arith.ty == ArithOpType::Add {
                        return self.sub_const(left, swapped_c);
                    } else {
                        return self.add_const(left, swapped_c);
                    };
                }
            }
        }
        let mut needs_mask = None;
        let mut needs_is_sub = false;
        let mut doesnt_need_mask = self.const_0();
        let mut loop_count = 0;
        for _ in util::IterAddSubArithOps::new(value) {
            loop_count += 1;
            if loop_count >= 16 {
                // Too big, just mask the value and consider the work done.
                // slow::slow4 test especially ends up hitting edge cases with
                // massive (up to ~1000 ops) add/sub chains .
                #[cfg(feature = "fuzz")]
                tls_simplification_incomplete();
                return value;
            }
        }
        for (op, is_sub) in util::IterAddSubArithOps::new(value) {
            if op.0.flags & FLAG_32BIT_NORMALIZED == 0 {
                needs_mask = match needs_mask {
                    None => {
                        needs_is_sub = is_sub;
                        Some(op)
                    }
                    Some(old) => {
                        if is_sub {
                            if needs_is_sub {
                                Some(self.add(old, op))
                            } else {
                                Some(self.sub(old, op))
                            }
                        } else {
                            if needs_is_sub {
                                needs_is_sub = false;
                                Some(self.sub(op, old))
                            } else {
                                Some(self.add(old, op))
                            }
                        }
                    }
                };
            } else {
                if is_sub {
                    doesnt_need_mask = self.sub(doesnt_need_mask, op);
                } else {
                    doesnt_need_mask = self.add(doesnt_need_mask, op);
                }
            }
        }
        match needs_mask {
            Some(needs) => {
                let masked = self.and_const(needs, 0xffff_ffff);
                if needs_is_sub {
                    self.sub(doesnt_need_mask, masked)
                } else {
                    self.add(doesnt_need_mask, masked)
                }
            }
            None => doesnt_need_mask,
        }
    }
}

struct CopyOperandCtx {
    recurse_limit: u32,
    // *const ArchDefinition<'any>
    arch_def: *const (),
}

/// Contains counts of [`Operand`]s interned by a single [`OperandContext`].
///
/// May be useful for measuring performance / memory use of different approaches.
///
/// Created by [`OperandContext::interned_counts`].
pub struct InternedCounts {
    /// Amount of `OperandType::Undefined` operands interned.
    pub undefined: usize,
    /// Amount of `OperandType::Constant` operands interned.
    pub constant: usize,
    /// Amount of operands interned that are not included in other counts.
    pub other: usize,
}

impl InternedCounts {
    /// Amount of all operands interned.
    pub fn total(&self) -> usize {
        self.undefined + self.constant + self.other
    }
}

impl<'e> OperandType<'e> {
    /// Returns which bits the operand will use at most.
    fn calculate_relevant_bits(&self) -> Range<u8> {
        match *self {
            OperandType::Arithmetic(ref arith) => match arith.ty {
                ArithOpType::Equal | ArithOpType::GreaterThan => {
                    0..1
                }
                ArithOpType::Lsh | ArithOpType::Rsh => {
                    let left_bits = arith.left.relevant_bits();
                    let (min, max) = if let Some(c) = arith.right.if_constant() {
                        let c = c as u8 & 0x3f;
                        (c, c)
                    } else {
                        let right_bits = arith.right.relevant_bits();
                        if right_bits.end > 6 {
                            (0, 0x3f)
                        } else {
                            (0, (1u8 << right_bits.end).wrapping_sub(1))
                        }
                    };
                    if arith.ty == ArithOpType::Lsh {
                        let start = 64u8.min(left_bits.start + min as u8);
                        let end = 64u8.min(left_bits.end + max as u8);
                        start..end
                    } else {
                        let start = left_bits.start.saturating_sub(max as u8);
                        let end = left_bits.end.saturating_sub(min as u8);
                        start..end
                    }
                }
                ArithOpType::And => {
                    let rel_left = arith.left.relevant_bits();
                    let rel_right = arith.right.relevant_bits();
                    max(rel_left.start, rel_right.start)..min(rel_left.end, rel_right.end)
                }
                ArithOpType::Or | ArithOpType::Xor => {
                    let rel_left = arith.left.relevant_bits();
                    // Early exit if left uses all bits already
                    if rel_left == (0..64) {
                        return rel_left;
                    }
                    let rel_right = arith.right.relevant_bits();
                    min(rel_left.start, rel_right.start)..max(rel_left.end, rel_right.end)
                }
                ArithOpType::Add => {
                    // Add will only increase nonzero bits by one at most.
                    // And if the bits don't overlap they won't increase even by one.
                    // If rhs is constant use bit more accurate bitmask.
                    let left_relbits = arith.left.relevant_bits_mask();
                    let right_relbits = if let Some(c) = arith.right.if_constant() {
                        c
                    } else {
                        arith.right.relevant_bits_mask()
                    };
                    let max_value = left_relbits.checked_add(right_relbits)
                        .unwrap_or(u64::MAX);

                    let rel_left = arith.left.relevant_bits();
                    let rel_right = arith.right.relevant_bits();
                    let max_end = 64u8 - max_value.leading_zeros() as u8;
                    let lower_start = min(rel_left.start, rel_right.start);
                    lower_start..max_end
                }
                ArithOpType::Sub => {
                    // Sub will not add any nonzero bits to lower end, but higher end
                    // can be completely filled
                    let rel_left = arith.left.relevant_bits();
                    let rel_right = arith.right.relevant_bits();
                    min(rel_left.start, rel_right.start)..64
                }
                ArithOpType::Mul => {
                    let left_bits = arith.left.relevant_bits();
                    let right_bits = arith.right.relevant_bits();
                    if left_bits.start.wrapping_add(1) == left_bits.end ||
                        right_bits.start.wrapping_add(1) == right_bits.end
                    {
                        // If multiplying by value that has at most 1 bit shift,
                        // it is either multiplying by 0 or shifting left,
                        // in which case highest nonzero can be known be one less
                        // than otherwise.
                        // E.g. `x * (y & 0x8000)`
                        // Becomes either `0` or `x << 0xf`
                        // So result would be (x.start + 0xf, x.end + 0xf)
                        //      y relbits are (y.start == 0xf, y.end == 0x10)
                        //
                        // Instead of having two branches to figure out whether
                        // x is on left or right and using `x.end + y.start`,
                        // use `x.end + y.end - 1` which works both ways
                        let low = left_bits.start.wrapping_add(right_bits.start).min(64);
                        let high = left_bits.end
                            .wrapping_add(right_bits.end)
                            .wrapping_sub(1).min(64);
                        low..high
                    } else {
                        // 64 + 64 cannot overflow
                        let low = left_bits.start.wrapping_add(right_bits.start).min(64);
                        let high = left_bits.end.wrapping_add(right_bits.end).min(64);
                        low..high
                    }
                }
                ArithOpType::Modulo => {
                    let left_bits = arith.left.relevant_bits();
                    let right_bits = arith.right.relevant_bits();
                    // Modulo can only give a result as large as right,
                    // though if left is less than right, it only gives
                    // left
                    if arith.right.if_constant() == Some(0) {
                        0..64
                    } else {
                        0..(min(left_bits.end, right_bits.end))
                    }
                }
                ArithOpType::Div => {
                    if arith.right.if_constant() == Some(0) {
                        0..64
                    } else {
                        let left_bits = arith.left.relevant_bits();
                        0..left_bits.end
                    }
                }
                ArithOpType::ToFloat => 0..32,
                _ => 0..64,
            },
            OperandType::ArithmeticFloat(ref arith, size) => match arith.ty {
                // Arithmetic with f32 inputs is always assumed to be f32 output
                // Note that ToInt(f32) => i32 clamped, ToInt(f64) => i64 clamped
                // No way to specify ToInt(f32) => i64 other than ToInt(ToDouble(f32)),
                // guess that's good enough?
                ArithOpType::ToDouble => 0..64,
                ArithOpType::ToFloat => 0..32,
                ArithOpType::Equal | ArithOpType::GreaterThan => 0..1,
                _ => 0..(size.bits() as u8),
            }
            OperandType::Select(_, a, b) => {
                let a_bits = a.relevant_bits();
                let b_bits = b.relevant_bits();
                min(a_bits.start, b_bits.start)..max(a_bits.end, b_bits.end)
            }
            OperandType::Arch(arch) => {
                let tag = arch.value() >> 16;
                if let Some(tag_def) = arch.arch_def.tag_definitions.get(tag as usize) {
                    0..tag_def.size
                } else {
                    0..64
                }
            }
            // Note: constants not handled here; const_relevant_bits instead
            _ => 0..(self.expr_size().bits() as u8),
        }
    }

    fn const_relevant_bits(c: u64) -> Range<u8> {
        if c == 0 {
            0..0
        } else {
            let trailing = c.trailing_zeros() as u8;
            let leading = c.leading_zeros() as u8;
            trailing..(64u8.wrapping_sub(leading))
        }
    }

    fn flags(&self, relevant_bits: Range<u8>) -> u8 {
        use self::OperandType::*;
        // Value of FLAG_32BIT_NORMALIZED for types which are normalized when
        // explicitly masked with 0xffff_ffff
        let default_32_bit_normalized = if relevant_bits.end > 32 || relevant_bits.start >= 32 {
            0
        } else {
            FLAG_32BIT_NORMALIZED
        };
        match *self {
            Memory(ref mem) => {
                let can_simplify_with_and = match mem.size {
                    MemAccessSize::Mem8 => 0,
                    _ => FLAG_CAN_SIMPLIFY_WITH_AND_MASK,
                };
                (mem.address().0.0.flags & ALWAYS_INHERITED_FLAGS) | FLAG_NEEDS_RESOLVE |
                    FLAG_CONTAINS_MEMORY | default_32_bit_normalized | can_simplify_with_and
            }
            SignExtend(val, _, _) => {
                (val.0.flags & ALWAYS_INHERITED_FLAGS) |
                    default_32_bit_normalized | FLAG_CAN_SIMPLIFY_WITH_AND_MASK
            }
            Select(a, b, c) => {
                // Have to always set FLAG_CAN_SIMPLIFY_WITH_AND_MASK since
                // one of the inner values may be fully simplifiable to 0 by a mask.
                //
                // (Or could set it only when inner value relevant_bits aren't 0..0 or 0..64)
                ((a.0.flags | b.0.flags | c.0.flags) & ALWAYS_INHERITED_FLAGS) |
                    FLAG_CAN_SIMPLIFY_WITH_AND_MASK |
                    default_32_bit_normalized
            }
            Arithmetic(ref arith) => {
                let base = (arith.left.0.flags | arith.right.0.flags) & ALWAYS_INHERITED_FLAGS;
                let could_remove_const_and = if
                    arith.ty == ArithOpType::And && arith.right.if_constant().is_some()
                {
                    FLAG_COULD_REMOVE_CONST_AND | FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                } else {
                    match arith.ty {
                        ArithOpType::And | ArithOpType::Or | ArithOpType::Xor | ArithOpType::Add |
                            ArithOpType::Mul =>
                        {
                            // If relbits of left/right are not same, simplify_with_and_mask
                            // may be able to fully zero one of the operands, so
                            // FLAG_CAN_SIMPLIFY_WITH_AND_MASK has to be always set.
                            // If they are same, can just set if it either of children have it.
                            let child_flags = arith.left.0.flags | arith.right.0.flags;
                            match arith.left.relevant_bits() == arith.right.relevant_bits() {
                                true => {
                                    child_flags & (
                                        FLAG_COULD_REMOVE_CONST_AND |
                                        FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                                    )
                                }
                                false => {
                                    (child_flags & FLAG_COULD_REMOVE_CONST_AND) |
                                        FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                                }
                            }
                        }
                        ArithOpType::Sub => {
                            // If rhs of sub is value with only lowest 1 bit set, it cannot
                            // be simplified, because mask passed for sub rhs is 000..111 mask
                            // with all 0s below 1 bits set to 1.
                            // (At least with current implementation?)
                            let right_bits = arith.right.relevant_bits();
                            let left_flags = arith.left.0.flags;
                            if right_bits.end == 1 {
                                (left_flags & FLAG_COULD_REMOVE_CONST_AND) |
                                    FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                            } else {
                                let child_flags = left_flags | arith.right.0.flags;
                                (child_flags & FLAG_COULD_REMOVE_CONST_AND) |
                                    FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                            }
                        }
                        ArithOpType::Lsh => {
                            if arith.right.if_constant().is_none() {
                                0
                            } else {
                                if arith.left.if_memory().is_some() {
                                    (arith.left.0.flags & FLAG_COULD_REMOVE_CONST_AND) |
                                        FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                                } else {
                                    arith.left.0.flags &
                                        (
                                            FLAG_COULD_REMOVE_CONST_AND |
                                            FLAG_CAN_SIMPLIFY_WITH_AND_MASK
                                        )
                                }
                            }
                        }
                        ArithOpType::Rsh => {
                            if arith.right.if_constant().is_none() {
                                0
                            } else {
                                arith.left.0.flags &
                                    (FLAG_COULD_REMOVE_CONST_AND | FLAG_CAN_SIMPLIFY_WITH_AND_MASK)
                            }
                        }
                        _ => 0,
                    }
                };
                let normalized_32 = if relevant_bits.start >= 32 {
                    0
                } else {
                    Self::arithmetic_32_bit_normalized_flag(arith, default_32_bit_normalized)
                };
                base | could_remove_const_and | normalized_32
            }
            ArithmeticFloat(ref arith, _) => {
                (
                    (arith.left.0.flags | arith.right.0.flags) & ALWAYS_INHERITED_FLAGS
                ) | FLAG_32BIT_NORMALIZED
            }
            Arch(..) => {
                FLAG_NEEDS_RESOLVE | FLAG_32BIT_NORMALIZED
            }
            Custom(..) => FLAG_32BIT_NORMALIZED,
            // Note: undefined not handled here; new_undef instead
            // Note: constants not handled here; const_flags instead
            Undefined(..) | Constant(..) => 0,
        }
    }

    fn is_32bit_normalized_for_mul_shift(
        op: Operand<'e>,
        shift: u8,
    ) -> bool {
        let ty = op.ty();
        if op.relevant_bits().start.wrapping_add(shift) >= 32 {
            return false;
        }
        if let OperandType::Arithmetic(arith) = ty {
            match arith.ty {
                ArithOpType::Add | ArithOpType::Sub => {
                    if !Self::is_32bit_normalized_for_mul_shift(arith.right, shift) {
                        return false;
                    }
                    let mut iter = util::IterAddSubArithOps::new(arith.left);
                    if iter.any(|x| {
                        !Self::is_32bit_normalized_for_mul_shift(x.0, shift)
                    }) {
                        return false;
                    }
                    true
                }
                ArithOpType::Mul | ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let shift2 = match arith.ty {
                            ArithOpType::Lsh => c as u8,
                            _ => c.trailing_zeros() as u8,
                        };
                        Self::is_32bit_normalized_for_mul_shift(
                            arith.left,
                            shift.wrapping_add(shift2),
                        )
                    } else {
                        op.is_32bit_normalized()
                    }
                }
                ArithOpType::And => {
                    if let Some(c) = arith.right.if_constant() {
                        let shifted_c = c << shift;
                        // E.g. ((rax & ff) << 18) normalizes to (rax << 18)
                        // But any more limiting mask is already normalized.

                        // High bits of the shifted mask must be 0,
                        // otherwise the normalized form will be one without them.
                        let shifted_high_bits_zero = (shifted_c >> 32) as u32 == 0;

                        if !shifted_high_bits_zero {
                            return false;
                        }
                        // If shifted mask fills all bits of u32 when shifted in zeroes are
                        // changed to ones => nop mask => remove that mask when normalizing
                        let nop_mask_if_u32 = (
                                shifted_c as u32 |
                                    // Any shifted-in bits to 1
                                    (1u32.wrapping_shl(shift as u32)).wrapping_sub(1) |
                                    // Any bits that are known to be zero in arith.left to 1
                                    ((!(arith.left.relevant_bits_mask() as u32))
                                        .wrapping_shl(shift as u32))
                            ) == u32::MAX;
                        if nop_mask_if_u32 {
                            // ((rax & ff) << 18) to (rax << 18)
                            // but
                            // ((rax * rcx) << 18) to (((rax * rcx) & ff) << 18)
                            if Self::is_32bit_normalized_for_mul_shift(arith.left, shift) {
                                return false;
                            }
                        }
                        return true;
                    }
                    false
                }
                _ => {
                    // Require other arithmetic always be masked.
                    // Technically could keep recursing for or/xor but
                    // maybe better to not make this take excessively long?
                    false
                }
            }
        } else if let OperandType::Memory(mem) = ty {
            let min_shift = match mem.size {
                MemAccessSize::Mem8 => 64u8,
                MemAccessSize::Mem16 => 24,
                MemAccessSize::Mem32 => 16,
                MemAccessSize::Mem64 => 0,
            };
            shift < min_shift
        } else if let OperandType::SignExtend(_, from, _) = ty {
            let min_shift = match from {
                MemAccessSize::Mem8 => 24u8,
                MemAccessSize::Mem16 => 16,
                MemAccessSize::Mem32 => 64,
                MemAccessSize::Mem64 => 64,
            };
            shift < min_shift
        } else {
            true
        }
    }

    fn arithmetic_32_bit_normalized_flag(
        arith: &ArithOperand<'e>,
        default_32_bit_normalized: u8,
    ) -> u8 {
        match arith.ty {
            ArithOpType::And => {
                if let Some(c) = arith.right.if_constant() {
                    if arith.left.0.flags &
                        (FLAG_PARTIAL_32BIT_NORMALIZED_ADD | FLAG_32BIT_NORMALIZED) != 0
                    {
                        let left_relbits = arith.left.relevant_bits_mask();
                        if left_relbits as u32 == c as u32 {
                            // mask doesn't change anything for low 32 bits so
                            // arith.left is 32bit normalized value of this operand.
                            return 0;
                        }
                    }
                }
                ((arith.left.0.flags & arith.right.0.flags) & FLAG_32BIT_NORMALIZED) |
                    default_32_bit_normalized
            }
            // Would be nice to have or here too, but (x | y) & c may become
            // (x & c) | y which complicates things.
            ArithOpType::Xor => {
                if (arith.left.0.flags | arith.right.0.flags) & FLAG_COULD_REMOVE_CONST_AND == 0 {
                    (arith.left.0.flags & arith.right.0.flags) & FLAG_32BIT_NORMALIZED
                } else {
                    default_32_bit_normalized
                }
            }
            ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul |
                ArithOpType::Lsh =>
            {
                if matches!(arith.ty, ArithOpType::Add | ArithOpType::Sub) {
                    let mut result =
                        arith.left.0.flags & arith.right.0.flags & FLAG_32BIT_NORMALIZED;
                    if ((arith.left.0.flags | arith.right.0.flags) & FLAG_32BIT_NORMALIZED) |
                        (arith.left.0.flags & FLAG_PARTIAL_32BIT_NORMALIZED_ADD) != 0
                    {
                        result |= FLAG_PARTIAL_32BIT_NORMALIZED_ADD;
                    }
                    // Account for (x - 0xffff_ffff) becoming (x + 1) when
                    // 32-bit masked
                    if let Some(c) = arith.right.if_constant() {
                        let limit = match arith.ty {
                            ArithOpType::Add => 0x8000_0000u32,
                            _ => 0x7fff_ffff,
                        };
                        if c as u32 > limit {
                            return FLAG_PARTIAL_32BIT_NORMALIZED_ADD;
                        }
                    }
                    result
                } else {
                    let result = arith.left.0.flags & arith.right.0.flags & FLAG_32BIT_NORMALIZED;
                    if result != 0 {
                        // Mul or Lsh, if lhs is addition / subtraction,
                        // and rhs is constant, simplification may prove
                        // that some of the terms end up being irrelevant
                        // to low 32bits.
                        if let Some(c) = arith.right.if_constant() {
                            let shift = if arith.ty == ArithOpType::Lsh {
                                c as u8
                            } else {
                                c.trailing_zeros() as u8
                            };
                            if shift != 0 {
                                if !Self::is_32bit_normalized_for_mul_shift(arith.left, shift) {
                                    return 0;
                                }
                            }
                        } else {
                            // Nonconst mul / shifts are only normalized when less than 32 bit
                            // results.
                            return default_32_bit_normalized;
                        }
                    }
                    result
                }
            }
            _ => default_32_bit_normalized,
        }
    }

    #[inline]
    fn const_flags(value: u64) -> u8 {
        // Set FLAG_CAN_SIMPLIFY_WITH_AND_MASK when the value more than a single 1 bit
        let with_and_flag = if value.wrapping_sub(1) & value == 0 {
            0
        } else {
            FLAG_CAN_SIMPLIFY_WITH_AND_MASK
        };
        if value <= u32::MAX as u64 {
            FLAG_32BIT_NORMALIZED | with_and_flag
        } else {
            with_and_flag
        }
    }

    fn sort_order(&self) -> u64 {
        use OperandType::*;
        match *self {
            Memory(ref mem) => {
                let base = mem.address().0;
                let base_discriminant = base.ty().discriminant();
                ((base_discriminant as u64) << 59) | (base.0.sort_order >> 5)
            }
            SignExtend(base, _, _) => {
                let base_discriminant = base.ty().discriminant();
                ((base_discriminant as u64) << 59) | (base.0.sort_order >> 5)
            }
            Select(a, b, c) => {
                // Can this be anything useful? Maybe sort by a/b/c discriminants
                let hash = fxhash::hash64(&(a.0.sort_order, b.0.sort_order, c.0.sort_order));
                hash
            }
            Arithmetic(ref arith) | ArithmeticFloat(ref arith, _) => {
                let ty = arith.ty as u8;
                let hash = fxhash::hash64(&(arith.left.0.sort_order, arith.right.0.sort_order));
                ((ty as u64) << 59) | (hash >> 5)
            }
            // Shift these left so that Arithmetic / Memory sort_order won't
            // shift these out. Shifting left by 32 may compile on 32bit host to
            // just zeroing the low half without actually doing 64-bit shifts?
            Arch(a) => (a.value() as u64) << 32,
            Custom(a) => (a as u64) << 32,
            // Note: constants / undefined not handled here
            Constant(..) | Undefined(..) => 0,
        }
    }

    /// Separate function for comparison when fast path doesn't work
    /// (Give compiler chance to inline only fast path if it wants)
    ///
    /// Assumes that self and other have same `OperandType` variant
    /// and that their sort_order was equal
    fn cmp_sort_order_was_eq(&self, other: &OperandType<'e>) -> Ordering {
        if let OperandType::Memory(ref self_mem) = *self {
            if let OperandType::Memory(ref other_mem) = *other {
                return self_mem.cmp(other_mem);
            }
        } else if let OperandType::Arithmetic(ref self_arith) = *self {
            if let OperandType::Arithmetic(ref other_arith) = *other {
                // Note that ArithOperand (i.e. without sort_order)
                // comparison can give different result than OperandType comparison since
                // sort_order depends on both left and right, while ArithOperand
                // is left-then-right comparison.
                return self_arith.cmp(other_arith);
            }
        } else if let OperandType::ArithmeticFloat(ref self_arith, self_size) = *self {
            if let OperandType::ArithmeticFloat(ref other_arith, other_size) = *other {
                let order = self_arith.cmp(other_arith);
                if order != Ordering::Equal {
                    return order;
                }
                return self_size.cmp(&other_size);
            }
        } else if let OperandType::SignExtend(inner, from, to) = *self {
            if let OperandType::SignExtend(inner2, from2, to2) = *other {
                let order = inner.cmp(&inner2);
                if order != Ordering::Equal {
                    return order;
                }
                let sizes = ((from as u8) << 4) | (to as u8);
                let sizes2 = ((from2 as u8) << 4) | (to2 as u8);
                return sizes.cmp(&sizes2);
            }
        } else if let OperandType::Select(a, b, c) = *self {
            if let OperandType::Select(a2, b2, c2) = *other {
                return (a, b, c).cmp(&(a2, b2, c2));
            }
        }
        debug_assert!(false, "supposed to be unreachable");
        Ordering::Equal
    }

    #[inline]
    fn discriminant(&self) -> u8 {
        // This gets optimized to just Mem8[x] read as long
        // as ordering matches OperandType definition
        match *self {
            OperandType::Arch(..) => 0,
            OperandType::Constant(..) => 1,
            OperandType::Memory(..) => 2,
            OperandType::Arithmetic(..) => 3,
            OperandType::ArithmeticFloat(..) => 4,
            OperandType::Undefined(..) => 5,
            OperandType::SignExtend(..) => 6,
            OperandType::Select(..) => 7,
            OperandType::Custom(..) => 8,
        }
    }

    /// Returns whether the operand is 8, 16, 32, or 64 bits.
    /// Relevant with signed multiplication, usually operands can be considered
    /// zero-extended u32.
    pub fn expr_size(&self) -> MemAccessSize {
        use self::OperandType::*;
        match *self {
            Memory(ref mem) => mem.size,
            Arch(..) | Constant(..) | Arithmetic(..) | Undefined(..) |
                Custom(..) | ArithmeticFloat(..) | Select(..) => MemAccessSize::Mem64,
            SignExtend(_, _from, to) => to,
        }
    }

    /// NOTE: Unsafe code will rely on this value to skip enum discriminant comparison
    /// sometimes.
    fn alt_tag(&self) -> u16 {
        match *self {
            OperandType::Memory(ref mem) => MEM_ALT_TAG | mem.size as u16,
            OperandType::Arithmetic(ref arith) => ARITH_ALT_TAG | arith.ty as u16,
            OperandType::Arch(ref id) => {
                if id.value() < 256 {
                    REGISTER_ALT_TAG | (id.value() as u16)
                } else {
                    0
                }
            }
            _ => 0,
        }
    }
}

impl<'e> Operand<'e> {
    /// Creates a self ref for OperandContext.
    /// User has to unsafely make sure the OperandContext still exists when casting
    /// OperandSelfRef to Operand
    fn self_ref(self) -> OperandSelfRef {
        OperandSelfRef::new(self)
    }

    /// Returns reference to actual enum type carrying data of this `Operand`.
    #[inline]
    pub fn ty(self) -> &'e OperandType<'e> {
        &self.0.ty
    }

    #[inline]
    pub(crate) fn needs_resolve(self) -> bool {
        self.0.flags & FLAG_NEEDS_RESOLVE != 0
    }

    /// Returns what bits in this value are not guaranteed to be zero.
    ///
    /// End cannot be larger than 64.
    ///
    /// Can be also seen as trailing_zeros .. 64 - leading_zeros range
    #[inline]
    pub fn relevant_bits(self) -> Range<u8> {
        self.0.relevant_bits.clone()
    }

    #[inline]
    pub fn relevant_bits_mask(self) -> u64 {
        self.0.relevant_bits_mask
    }

    /// Returns `Some(c)` if `self.ty` is `OperandType::Constant(c)`
    #[inline]
    pub fn if_constant(self) -> Option<u64> {
        match *self.ty() {
            OperandType::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(c)` if `self.ty` is `OperandType::Custom(c)`
    #[inline]
    pub fn if_custom(self) -> Option<u32> {
        match *self.ty() {
            OperandType::Custom(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(r)` if `self.ty` is `OperandType::Arch(r)` with
    /// `r.value()` being less than 256.
    #[inline]
    pub fn if_register(self) -> Option<u8> {
        if self.0.type_alt_tag & TYPE_ALT_TAG_MASK == REGISTER_ALT_TAG {
            Some(self.0.type_alt_tag as u8)
        } else {
            None
        }
    }

    #[inline]
    pub fn if_arch_id(self) -> Option<u32> {
        match *self.ty() {
            OperandType::Arch(a) => Some(a.value()),
            _ => None,
        }
    }

    /// Returns `Some(f)` if `self.ty` is `OperandType::Flag(f)`
    pub fn x86_if_flag(self) -> Option<Flag> {
        let arch = self.if_arch_id()?;
        if disasm::x86_arch_tag(arch) == disasm::X86_FLAG_TAG {
            Some(Flag::x86_from_arch(arch as u8))
        } else {
            None
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)`
    #[inline]
    pub fn if_memory(self) -> Option<&'e MemAccess<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => Some(mem),
            _ => None,
        }
    }

    #[inline]
    fn if_mem_fast(self, size: MemAccessSize) -> Option<&'e MemAccess<'e>> {
        if self.0.type_alt_tag == (MEM_ALT_TAG | size as u16) {
            match *self.ty() {
                OperandType::Memory(ref mem) => {
                    debug_assert_eq!(mem.size, size);
                    Some(mem)
                }
                _ => unsafe {
                    debug_assert!(false, "unsafe code bug");
                    std::hint::unreachable_unchecked();
                }
            }
        } else {
            None
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem64`
    #[inline]
    pub fn if_mem64(self) -> Option<&'e MemAccess<'e>> {
        self.if_mem_fast(MemAccessSize::Mem64)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem32`
    #[inline]
    pub fn if_mem32(self) -> Option<&'e MemAccess<'e>> {
        self.if_mem_fast(MemAccessSize::Mem32)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem16`
    #[inline]
    pub fn if_mem16(self) -> Option<&'e MemAccess<'e>> {
        self.if_mem_fast(MemAccessSize::Mem16)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem8`
    #[inline]
    pub fn if_mem8(self) -> Option<&'e MemAccess<'e>> {
        self.if_mem_fast(MemAccessSize::Mem8)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem64`
    #[inline]
    pub fn if_mem64_offset(self, offset: u64) -> Option<Operand<'e>> {
        self.if_mem64()?.if_offset(offset)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem32`
    #[inline]
    pub fn if_mem32_offset(self, offset: u64) -> Option<Operand<'e>> {
        self.if_mem32()?.if_offset(offset)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem16`
    #[inline]
    pub fn if_mem16_offset(self, offset: u64) -> Option<Operand<'e>> {
        self.if_mem16()?.if_offset(offset)
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem8`
    #[inline]
    pub fn if_mem8_offset(self, offset: u64) -> Option<Operand<'e>> {
        self.if_mem8()?.if_offset(offset)
    }

    #[inline]
    fn check_arith_tag(self, ty: ArithOpType) -> bool {
        self.0.type_alt_tag == (ARITH_ALT_TAG | ty as u16)
    }

    /// Returns `Some((left, right))` if self.ty is `OperandType::Arithmetic { ty == ty }`
    #[inline]
    pub fn if_arithmetic(self, ty: ArithOpType) -> Option<(Operand<'e>, Operand<'e>)> {
        if self.check_arith_tag(ty) {
            match *self.ty() {
                OperandType::Arithmetic(ref arith) => {
                    debug_assert_eq!(arith.ty, ty);
                    Some((arith.left, arith.right))
                }
                _ => unsafe {
                    debug_assert!(false, "unsafe code bug");
                    std::hint::unreachable_unchecked();
                }
            }
        } else {
            None
        }
    }

    /// Returns `Some(arith)` if self.ty is `OperandType::Arithmetic(arith)`.
    #[inline]
    pub fn if_arithmetic_any(self) -> Option<&'e ArithOperand<'e>> {
        match *self.ty() {
            OperandType::Arithmetic(ref arith) => Some(arith),
            _ => None,
        }
    }

    /// Returns `true` if self.ty is `OperandType::Arithmetic { ty == ty }`
    #[inline]
    pub fn is_arithmetic(self, ty: ArithOpType) -> bool {
        self.check_arith_tag(ty)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Add(left, right))`
    #[inline]
    pub fn if_arithmetic_add(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Add)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Sub(left, right))`
    #[inline]
    pub fn if_arithmetic_sub(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Sub)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Mul(left, right))`
    #[inline]
    pub fn if_arithmetic_mul(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Mul)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::MulHigh(left, right))`
    #[inline]
    pub fn if_arithmetic_mul_high(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::MulHigh)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Equal(left, right))`
    #[inline]
    pub fn if_arithmetic_eq(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Equal)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::GreaterThan(left, right))`
    #[inline]
    pub fn if_arithmetic_gt(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::GreaterThan)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::And(left, right))`
    #[inline]
    pub fn if_arithmetic_and(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::And)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Or(left, right))`
    #[inline]
    pub fn if_arithmetic_or(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Or)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Xor(left, right))`
    #[inline]
    pub fn if_arithmetic_xor(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Xor)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Lsh(left, right))`
    #[inline]
    pub fn if_arithmetic_lsh(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Lsh)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Rsh(left, right))`
    #[inline]
    pub fn if_arithmetic_rsh(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Rsh)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic({ ty: arith, left, right: OperandType::Constant(right) })`
    pub fn if_arithmetic_with_const(self, arith: ArithOpType) -> Option<(Operand<'e>, u64)> {
        let (l, r) = self.if_arithmetic(arith)?;
        Some((l, r.if_constant()?))
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Add(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_add_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Add)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Sub(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_sub_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Sub)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Mul(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_mul_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Mul)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::MulHigh(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_mul_high_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::MulHigh)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Equal(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_eq_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Equal)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::GreaterThan(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_gt_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::GreaterThan)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::And(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_and_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::And)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Or(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_or_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Or)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Xor(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_xor_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Xor)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Lsh(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_lsh_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Lsh)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Rsh(left, OperandType::Constant(right)))`
    #[inline]
    pub fn if_rsh_with_const(self) -> Option<(Operand<'e>, u64)> {
        self.if_arithmetic_with_const(ArithOpType::Rsh)
    }

    /// Returns `Some((val, from, to))` if `self.ty` is
    /// `OperandType::SignExtend(val, from, to)`
    #[inline]
    pub fn if_sign_extend(self) -> Option<(Operand<'e>, MemAccessSize, MemAccessSize)> {
        match *self.ty() {
            OperandType::SignExtend(a, b, c) => Some((a, b, c)),
            _ => None,
        }
    }

    /// Returns `Some((condition, val_true, val_false))` if `self.ty` is
    /// `OperandType::Select(condition, val_true, val_false)`
    #[inline]
    pub fn if_select(self) -> Option<(Operand<'e>, Operand<'e>, Operand<'e>)> {
        match *self.ty() {
            OperandType::Select(a, b, c) => Some((a, b, c)),
            _ => None,
        }
    }

    /// Returns `Some((register, constant))` if operand is an and mask of register
    /// with constant.
    ///
    /// Useful for detecting 32-bit register which is represented as `Register(r) & ffff_ffff`.
    pub fn if_and_masked_register(self) -> Option<(u8, u64)> {
        let (l, r) = self.if_arithmetic_and()?;
        let reg = l.if_register()?;
        let c = r.if_constant()?;
        Some((reg, c))
    }

    /// Returns true if `self.ty() == OperandType::Undefined`.
    #[inline]
    pub fn is_undefined(self) -> bool {
        match self.ty() {
            OperandType::Undefined(_) => true,
            _ => false,
        }
    }

    /// Returns true if self or any child operand is `Undefined`.
    #[inline]
    pub fn contains_undefined(self) -> bool {
        self.0.flags & FLAG_CONTAINS_UNDEFINED != 0
    }

    /// Returns true if self or any child operand is `Memory`.
    #[inline]
    pub fn contains_memory(self) -> bool {
        self.0.flags & FLAG_CONTAINS_MEMORY != 0
    }

    #[inline]
    fn is_32bit_normalized(self) -> bool {
        self.0.flags & FLAG_32BIT_NORMALIZED != 0
    }

    /// If this value is known to be zero or known to be nonzero, returns Some(true/false)
    ///
    /// Used for select simplification / resolve fast path.
    pub(crate) fn is_known_bool(self) -> Option<bool> {
        if let Some(c) = self.if_constant() {
            Some(c != 0)
        } else {
            // Could have something smarter here but probably such trivially true selects won't
            // be generated.
            // self.if_or_with_const() would be always Some(true) but not really sure
            // why that would end up in a select.
            None
        }
    }

    /// Returns `(other, constant)` if operand is an and mask with constant,
    /// or just (self, u64::MAX) otherwise.
    pub fn and_masked(this: Operand<'e>) -> (Operand<'e>, u64) {
        this.if_and_with_const()
            .unwrap_or_else(|| (this, u64::MAX))
    }

    /// If either of `a` or `b` matches the filter-map `f`, return the mapped result and the other
    /// operand.
    pub fn either<F, T>(
        a: Operand<'e>,
        b: Operand<'e>,
        mut f: F,
    ) -> Option<(T, Operand<'e>)>
    where F: FnMut(Operand<'e>) -> Option<T>
    {
        f(a).map(|val| (val, b)).or_else(|| f(b).map(|val| (val, a)))
    }

    /// Returns an type implementing [`Iterator`], yielding `self`, followed by other
    /// `Operands` making up `self` in some unspecified order. If the same `Operand` is
    /// included in multiple branches of expression, it will be yielded mutiple times.
    ///
    /// Note that this needs to do a memory allocation to keep track of the iteration state,
    /// it is often better to match against expected form of expression than use `iter` to try
    /// find a specific suboperand.
    ///
    /// Additionally, scarf allows constructing `Operand` with `N^2` (not unique) suboperands
    /// in just `N` operations. This is often seen in hash functions, which shuffle bits of
    /// individual inputs without exactly discarding any of them. While scarf handles these
    /// massive `Operand`s fine, trying to iterate through them (Or even debug print them)
    /// will take excessive amounts of time. Code using `iter` that may be used in hash functions
    /// needs to have some limit after which iteration will be stopped.
    pub fn iter(self) -> Iter<'e> {
        Iter(Some(IterState {
            pos: self,
            stack: Vec::new(),
        }))
    }

    /// Returns an [`Iterator`] type similar to [`iter`](Self::iter), expect that it does not
    /// yield operands or their suboperands making up `OperandType::Memory` addresses.
    /// The operand with type `OperandType::Memory` is still yielded, it just is not recursed
    /// into.
    ///
    /// Same caveats that `iter` has apply here too.
    pub fn iter_no_mem_addr(self) -> IterNoMemAddr<'e> {
        IterNoMemAddr(Some(IterState {
            pos: self,
            stack: Vec::new(),
        }))
    }

    /// Converts this reference to `&'e OperandContext` to a struct
    /// which carries the backing `OperandContext` with it.
    ///
    /// As this means that a new `OperandContext` will be allocated and all operands this
    /// refers to will be copied there, the operation can be relatively slow.
    pub fn to_self_owned(self) -> SelfOwnedOperand {
        let ctx = OperandContext::new();
        // Somewhat weird but  as SelfOwnedOperand's point is not to have a lifetime dependency,
        // it has OperandContext<'static>. However, calling copy_operand here would mean
        // that there has to be &'static OperandContext<'static> which cannot be this local
        // variable that's being returned, so transmute it to have arbitrary different lifetime
        // for this call.
        let op = unsafe { std::mem::transmute::<_, OperandCtx<'_>>(&ctx) }
            .copy_operand(self).self_ref();
        SelfOwnedOperand {
            ctx,
            op,
        }
    }

    /// Returns this operand wrapped in a newtype that implements [`Hash`] by
    /// hashing the address of this reference.
    #[inline]
    pub fn hash_by_address(self) -> OperandHashByAddress<'e> {
        OperandHashByAddress(self)
    }

    /// Matches against:
    ///
    /// (x + c), returning (x, c),
    /// (x - c), returning (x, -c),
    /// c, returning (0, c)
    ///
    /// Probably should be removed.
    #[deprecated]
    pub fn const_offset(
        oper: Operand<'e>,
        ctx: OperandCtx<'e>,
    ) -> Option<(Operand<'e>, u64)> {
        match *oper.ty() {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
                if let Some(c) = arith.right.if_constant() {
                    return Some((arith.left, c));
                }
            }
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
                if let Some(c) = arith.right.if_constant() {
                    return Some((arith.left, 0u64.wrapping_sub(c)));
                }
            }
            OperandType::Constant(c) => {
                return Some((ctx.const_0(), c));
            }
            _ => (),
        }
        None
    }

    /// Generates operand from bytes, meant to help with fuzzing.
    ///
    /// Does not generate every variation of operands (skips fpu and such).
    ///
    /// TODO May be good to have this generate and-const masks a lot?
    #[cfg(feature = "fuzz")]
    pub fn from_fuzz_bytes(ctx: OperandCtx<'e>, bytes: &mut &[u8]) -> Option<Operand<'e>> {
        let read_u8 = |bytes: &mut &[u8]| -> Option<u8> {
            let &val = bytes.get(0)?;
            *bytes = &bytes[1..];
            Some(val)
        };
        let read_u64 = |bytes: &mut &[u8]| -> Option<u64> {
            let data: [u8; 8] = bytes.get(..8)?.try_into().unwrap();
            *bytes = &bytes[8..];
            Some(u64::from_le_bytes(data))
        };
        Some(match read_u8(bytes)? {
            0x0 => ctx.register(read_u8(bytes)? & 0xf),
            0x1 => ctx.xmm(read_u8(bytes)? & 0xf, read_u8(bytes)? & 0x3),
            0x2 => ctx.constant(read_u64(bytes)?),
            0x3 => {
                let size = match read_u8(bytes)? & 3 {
                    0 => MemAccessSize::Mem8,
                    1 => MemAccessSize::Mem16,
                    2 => MemAccessSize::Mem32,
                    _ => MemAccessSize::Mem64,
                };
                let inner = Operand::from_fuzz_bytes(ctx, bytes)?;
                ctx.mem_any(size, inner, 0)
            }
            0x4 => {
                let from = match read_u8(bytes)? & 3 {
                    0 => MemAccessSize::Mem8,
                    1 => MemAccessSize::Mem16,
                    2 => MemAccessSize::Mem32,
                    _ => MemAccessSize::Mem64,
                };
                let to = match read_u8(bytes)? & 3 {
                    0 => MemAccessSize::Mem8,
                    1 => MemAccessSize::Mem16,
                    2 => MemAccessSize::Mem32,
                    _ => MemAccessSize::Mem64,
                };
                let inner = Operand::from_fuzz_bytes(ctx, bytes)?;
                ctx.sign_extend(inner, from, to)
            }
            0x5 => {
                use self::ArithOpType::*;
                let left = Operand::from_fuzz_bytes(ctx, bytes)?;
                let right = Operand::from_fuzz_bytes(ctx, bytes)?;
                let ty = match read_u8(bytes)? {
                    0x0 => Add,
                    0x1 => Sub,
                    0x2 => Mul,
                    0x3 => SignedMul,
                    0x4 => Div,
                    0x5 => Modulo,
                    0x6 => And,
                    0x7 => Or,
                    0x8 => Xor,
                    0x9 => Lsh,
                    0xa => Rsh,
                    0xb => Equal,
                    0xd => GreaterThan,
                    0xe => ToFloat,
                    0xf => ToInt,
                    0x10 => ToDouble,
                    0x11 => MulHigh,
                    _ => return None,
                };
                if read_u8(bytes)? == 0 {
                    ctx.arithmetic(ty, left, right)
                } else {
                    let size = match read_u8(bytes)? & 3 {
                        0 => MemAccessSize::Mem32,
                        _ => MemAccessSize::Mem64,
                    };
                    ctx.float_arithmetic(ty, left, right, size)
                }
            }
            0x6 => ctx.custom(read_u64(bytes)? as u32),
            _ => return None,
        })
    }
}

impl SelfOwnedOperand {
    pub fn operand(&self) -> Operand<'_> {
        unsafe { self.op.cast() }
    }
}

impl fmt::Debug for SelfOwnedOperand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "SelfOwnedOperand({})", self.operand())
    }
}

impl Clone for SelfOwnedOperand {
    fn clone(&self) -> Self {
        self.operand().to_self_owned()
    }
}

#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Copy, Clone, Eq, Ord, PartialOrd)]
pub struct MemAccess<'e> {
    // Note: OperandType comparison relies on base being compared
    // first here with Ord derive
    base: Operand<'e>,
    offset: u64,
    pub size: MemAccessSize,
    /// true means that base == ctx.const_0()
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    const_base: bool,
}

impl<'e> PartialEq for MemAccess<'e> {
    #[inline]
    fn eq(&self, other: &MemAccess<'e>) -> bool {
        match *self {
            MemAccess { base, offset, size, const_base: _ } => {
                base == other.base &&
                    offset == other.offset &&
                    size == other.size
            }
        }
    }
}

impl<'e> fmt::Debug for MemAccess<'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let sign = if self.offset > 0x8000_0000_0000_0000 { "-" } else { "" };
        let abs = (self.offset as i64).wrapping_abs();
        write!(f, "MemAccess({}, {}{:x}, {:?})", self.base, sign, abs, self.size)
    }
}

impl<'e> MemAccess<'e> {
    /// Gets base address operand without add/sub offset, and the offset.
    pub fn address(&self) -> (Operand<'e>, u64) {
        (self.base, self.offset)
    }

    /// Joins the base address and offset to a single operand.
    ///
    /// Same as `ctx.add_const(self.address().0, self.address().1)`.
    pub fn address_op(&self, ctx: OperandCtx<'e>) -> Operand<'e> {
        ctx.add_const(self.base, self.offset)
    }

    /// If the memory has constant address (Base is `ctx.const_0()`), returns the offset
    pub fn if_constant_address(&self) -> Option<u64> {
        if self.const_base {
            Some(self.offset)
        } else {
            None
        }
    }

    /// Returns `Some(base)` only if offset is 0.
    ///
    /// Note that this means that any constant addresses will return `None`.
    pub fn if_no_offset(&self) -> Option<Operand<'e>> {
        if self.offset == 0 {
            Some(self.base)
        } else {
            None
        }
    }

    /// Returns `Some(base)` if `self.offset` equals `offset` parameter.
    pub fn if_offset(&self, compare: u64) -> Option<Operand<'e>> {
        if self.offset == compare {
            Some(self.base)
        } else {
            None
        }
    }

    /// Creates new `MemAccess` with address offset from `self` by `offset`
    /// and same size as before.
    pub fn with_offset(&self, offset: u64) -> MemAccess<'e> {
        MemAccess {
            base: self.base,
            offset: self.offset.wrapping_add(offset),
            size: self.size,
            const_base: self.const_base,
        }
    }

    /// Creates new `MemAccess` with address offset from `self` by `offset`
    /// and size set to `size`.
    pub fn with_offset_size(&self, offset: u64, size: MemAccessSize) -> MemAccess<'e> {
        MemAccess {
            base: self.base,
            offset: self.offset.wrapping_add(offset),
            size,
            const_base: self.const_base,
        }
    }

    /// Achieves the following without unnecessary interning:
    /// ```text
    /// self.address_op(ctx)
    ///     .if_arithmetic_add()
    ///     .and_then(|(l, r)| Operand::either(l, r, func))
    /// ```
    /// Though does behave bit differently in that if this `MemAccess` has non-zero
    /// offset, that constant will not be passed to `func`.
    pub fn if_add_either<F, T>(
        &self,
        ctx: OperandCtx<'e>,
        mut func: F,
    ) -> Option<(T, Operand<'e>)>
    where F: FnMut(Operand<'e>) -> Option<T>
    {
        let (base, offset) = self.address();
        if offset != 0 {
            if offset < 0x8000_0000_0000_0000 {
                if let Some(x) = func(base) {
                    Some((x, ctx.constant(offset)))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            base.if_arithmetic_add()
                .and_then(|(l, r)| Operand::either(l, r, &mut func))
        }
    }

    /// Achieves the following without unnecessary interning:
    /// ```text
    /// self.address_op(ctx)
    ///     .if_arithmetic_add()
    ///     .and_then(|(l, r)| Operand::either(l, r, func))
    ///     .map(|x| x.1)
    /// ```
    /// Though does behave bit differently in that if this `MemAccess` has non-zero
    /// offset, that constant will not be passed to `func`.
    pub fn if_add_either_other<F, T>(
        &self,
        ctx: OperandCtx<'e>,
        mut func: F,
    ) -> Option<Operand<'e>>
    where F: FnMut(Operand<'e>) -> Option<T>
    {
        let (base, offset) = self.address();
        if offset != 0 {
            if offset < 0x8000_0000_0000_0000 && func(base).is_some() {
                Some(ctx.constant(offset))
            } else {
                None
            }
        } else {
            base.if_arithmetic_add()
                .and_then(|(l, r)| Operand::either(l, r, &mut func))
                .map(|x| x.1)
        }
    }
}

/// MemAccessSize is enum for choosing between 8/16/32/64 bit size in scarf.
///
/// `MemAccessSize` is especially, as the name implies, used in [`MemAccess`], and
/// through that in any memory read or write operation. It does also show up as input
/// in some signed and floating-point arithmetic operations, which behave differently depending
/// on the operation size, as well as in the sign extension expression of [`Operand`],
/// [`OperandType::SignExtend`](OperandType).
///
/// Ultimately any read with size less than 64 bits (which is limit for scarf operations)
/// can be represented by doing a 64-bit read followed by an AND mask of bits you want to keep.
/// However, `MemAccessSize` is usually enough as the irregularly sized accesses are very rare.
///
/// (I think the odd ordering of placing Mem32 first in the enum was to make it 0, which may
/// end up generating slightly nicer assembly when Mem32 is common.)
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Hash, Eq, PartialEq, Copy, Debug, Ord, PartialOrd)]
pub enum MemAccessSize {
    Mem32 = 0,
    Mem16,
    Mem8,
    Mem64,
}

impl MemAccessSize {
    /// Returns bit size of `MemAccessSize`.
    ///
    /// ```rust
    /// use scarf::MemAccessSize;
    /// assert_eq!(MemAccessSize::Mem8.bits(), 8);
    /// assert_eq!(MemAccessSize::Mem16.bits(), 16);
    /// assert_eq!(MemAccessSize::Mem32.bits(), 32);
    /// assert_eq!(MemAccessSize::Mem64.bits(), 64);
    /// ```
    #[inline]
    pub fn bits(self) -> u32 {
        (match self {
            MemAccessSize::Mem64 => 64u8,
            MemAccessSize::Mem32 => 32,
            MemAccessSize::Mem16 => 16,
            MemAccessSize::Mem8 => 8,
        }) as u32
    }

    /// Returns byte size of `MemAccessSize`.
    ///
    /// ```rust
    /// use scarf::MemAccessSize;
    /// assert_eq!(MemAccessSize::Mem8.bytes(), 1);
    /// assert_eq!(MemAccessSize::Mem16.bytes(), 2);
    /// assert_eq!(MemAccessSize::Mem32.bytes(), 4);
    /// assert_eq!(MemAccessSize::Mem64.bytes(), 8);
    /// ```
    #[inline]
    pub fn bytes(self) -> u32 {
        (match self {
            MemAccessSize::Mem64 => 8u8,
            MemAccessSize::Mem32 => 4,
            MemAccessSize::Mem16 => 2,
            MemAccessSize::Mem8 => 1,
        }) as u32
    }

    /// Returns `self.bytes().trailing_bits()`
    ///
    /// ```rust
    /// use scarf::MemAccessSize;
    /// assert_eq!(MemAccessSize::Mem8.mul_div_shift(), 0);
    /// assert_eq!(MemAccessSize::Mem16.mul_div_shift(), 1);
    /// assert_eq!(MemAccessSize::Mem32.mul_div_shift(), 2);
    /// assert_eq!(MemAccessSize::Mem64.mul_div_shift(), 3);
    /// ```
    #[inline]
    pub fn mul_div_shift(self) -> u8 {
        match self {
            MemAccessSize::Mem64 => 3u8,
            MemAccessSize::Mem32 => 2,
            MemAccessSize::Mem16 => 1,
            MemAccessSize::Mem8 => 0,
        }
    }

    /// Returns the bitwise AND mask of `MemAccessSize`.
    ///
    /// ```rust
    /// use scarf::MemAccessSize;
    /// assert_eq!(MemAccessSize::Mem8.mask(), 0xff);
    /// assert_eq!(MemAccessSize::Mem16.mask(), 0xffff);
    /// assert_eq!(MemAccessSize::Mem32.mask(), 0xffff_ffff);
    /// assert_eq!(MemAccessSize::Mem64.mask(), 0xffff_ffff_ffff_ffff);
    /// ```
    #[inline]
    pub fn mask(self) -> u64 {
        (self.sign_bit() << 1).wrapping_sub(1)
    }

    /// Returns mask where each bit corresponds to a byte.
    ///
    /// ```rust
    /// use scarf::MemAccessSize;
    /// assert_eq!(MemAccessSize::Mem8.byte_mask().get(), 0b1);
    /// assert_eq!(MemAccessSize::Mem16.byte_mask().get(), 0b11);
    /// assert_eq!(MemAccessSize::Mem32.byte_mask().get(), 0b1111);
    /// assert_eq!(MemAccessSize::Mem64.byte_mask().get(), 0b1111_1111);
    /// ```
    #[inline]
    pub fn byte_mask(self) -> NonZeroU8 {
        const fn gen_values() -> [NonZeroU8; 4] {
            const fn unwrap(val: Option<NonZeroU8>) -> NonZeroU8 {
                match val {
                    Some(x) => x,
                    None => panic!(),
                }
            }
            let mut ret = [unwrap(NonZeroU8::new(1)); 4];
            ret[MemAccessSize::Mem8 as usize] = unwrap(NonZeroU8::new(1));
            ret[MemAccessSize::Mem16 as usize] = unwrap(NonZeroU8::new(0x3));
            ret[MemAccessSize::Mem32 as usize] = unwrap(NonZeroU8::new(0xf));
            ret[MemAccessSize::Mem64 as usize] = unwrap(NonZeroU8::new(0xff));
            ret
        }
        static VALUES: [NonZeroU8; 4] = gen_values();
        VALUES[self as usize]
    }

    /// Returns the sign bit of variables of this `MemAccessSize` (Highest mask bit).
    ///
    /// ```rust
    /// use scarf::MemAccessSize;
    /// assert_eq!(MemAccessSize::Mem8.sign_bit(), 0x80);
    /// assert_eq!(MemAccessSize::Mem16.sign_bit(), 0x8000);
    /// assert_eq!(MemAccessSize::Mem32.sign_bit(), 0x8000_0000);
    /// assert_eq!(MemAccessSize::Mem64.sign_bit(), 0x8000_0000_0000_0000);
    /// ```
    #[inline]
    pub fn sign_bit(self) -> u64 {
        1u64.wrapping_shl(self.bits().wrapping_sub(1))
    }
}

/// Enumeration of arithmetic result flags of CPU.
///
/// As of now, this is quite x86-specific, yet it does not contain all
/// flags of the CPU anyway (Such as the A flag).
///
/// [`ExecutionState`](crate::exec_state::ExecutionState) implementations store each flag in
/// a separate [`Operand`], instead of a single `Operand` representing EFLAGS register.
/// As such, each flag gets a separate variable in the `OperandType` enumeration too.
// Flags currently are cast to usize index when stored in ExecutionState,
// and the index is also used with pending_flags
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Eq, PartialEq, Copy, Debug, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum Flag {
    Zero = 0,
    Carry = 1,
    Overflow = 2,
    Sign = 3,
    Parity = 4,
    Direction = 5,
}

impl Flag {
    pub(crate) fn x86_from_arch(arch: u8) -> Flag {
        match arch & 7 {
            0 => Flag::Zero,
            1 => Flag::Carry,
            2 => Flag::Overflow,
            3 => Flag::Sign,
            4 => Flag::Parity,
            5 => Flag::Direction,
            _ => Flag::Zero,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn operand_iter() {
        use std::collections::HashSet;

        let ctx = OperandContext::new();
        let oper = ctx.and(
            ctx.sub(
                ctx.constant(1),
                ctx.register(6),
            ),
            ctx.eq(
                ctx.constant(77),
                ctx.register(4),
            ),
        );
        let opers = [
            oper.clone(),
            ctx.sub(ctx.constant(1), ctx.register(6)),
            ctx.constant(1),
            ctx.register(6),
            ctx.eq(ctx.constant(77), ctx.register(4)),
            ctx.constant(77),
            ctx.register(4),
        ];
        let mut seen = HashSet::new();
        for o in oper.iter() {
            let o = OperandHashByAddress(o);
            assert!(!seen.contains(&o));
            seen.insert(o);
        }
        for &o in &opers {
            assert!(seen.contains(&OperandHashByAddress(o)), "Didn't find {}", o);
        }
        assert_eq!(seen.len(), opers.len());
    }

    #[test]
    fn serialize_json() {
        use serde::de::DeserializeSeed;
        let ctx = &OperandContext::new();
        let op = ctx.and(
            ctx.register(6),
            ctx.mem32(
                ctx.sub(
                    ctx.constant(6),
                    ctx.register(3),
                ),
                0,
            ),
        );
        let json = serde_json::to_string(&op).unwrap();
        let mut des = serde_json::Deserializer::from_str(&json);
        let op2: Operand<'_> = ctx.deserialize_seed().deserialize(&mut des).unwrap();
        assert_eq!(op, op2);
    }

    #[test]
    fn serialize_json2() {
        use serde::de::DeserializeSeed;
        let ctx = &OperandContext::new();
        let op = ctx.float_arithmetic(
            ArithOpType::Sub,
            ctx.register(6),
            ctx.mem32(
                ctx.sub(
                    ctx.constant(6),
                    ctx.register(3),
                ),
                0,
            ),
            MemAccessSize::Mem32,
        );
        let json = serde_json::to_string(&op).unwrap();
        let mut des = serde_json::Deserializer::from_str(&json);
        let op2: Operand<'_> = ctx.deserialize_seed().deserialize(&mut des).unwrap();
        assert_eq!(op, op2);
    }

    #[test]
    fn serialize_json_fmt() {
        use serde::de::DeserializeSeed;
        // Verify that JSON serialization is stable to this
        // (Other details may not necessarily be)
        let json = r#"{
            "ty": {
                "Arithmetic": {
                    "ty": "And",
                    "left": {
                        "ty": {
                            "Register": 6
                        }
                    },
                    "right": {
                        "ty": {
                            "Memory": {
                                "base": {
                                    "ty": {
                                        "Arithmetic": {
                                            "ty": "Sub",
                                            "left": {
                                                "ty": {
                                                    "Constant": 6
                                                }
                                            },
                                            "right": {
                                                "ty": {
                                                    "Register": 3
                                                }
                                            }
                                        }
                                    }
                                },
                                "offset": 0,
                                "size": "Mem32"
                            }
                        }
                    }
                }
            }
        }"#;
        let ctx = &OperandContext::new();
        let op = ctx.and(
            ctx.register(6),
            ctx.mem32(
                ctx.sub(
                    ctx.constant(6),
                    ctx.register(3),
                ),
                0,
            ),
        );
        let mut des = serde_json::Deserializer::from_str(&json);
        let op2: Operand<'_> = ctx.deserialize_seed().deserialize(&mut des).unwrap();
        assert_eq!(op, op2);
    }

    #[test]
    fn serialize_binary() {
        use serde::de::DeserializeSeed;
        // Test binary roundtrip. Binary serialization doesn't any other stability
        // guarantees than working on same library version.
        let ctx = &OperandContext::new();
        let op = ctx.and(
            ctx.register(6),
            ctx.mem32(
                ctx.sub(
                    ctx.select(
                        ctx.eq(
                            ctx.register(35),
                            ctx.flag_z(),
                        ),
                        ctx.custom(2),
                        ctx.sign_extend(
                            ctx.arch(0x991406),
                            MemAccessSize::Mem16,
                            MemAccessSize::Mem32,
                        ),
                    ),
                    ctx.constant(6),
                ),
                0x999,
            ),
        );
        let serialized = postcard::to_allocvec(&op).unwrap();
        let mut des = postcard::Deserializer::from_bytes(&serialized);
        let op2: Operand<'_> = ctx.deserialize_seed().deserialize(&mut des).unwrap();
        assert_eq!(op, op2);
    }

    #[test]
    fn matching_funcs() {
        let ctx = OperandContext::new();
        assert_eq!(ctx.constant(5).if_constant(), Some(5));
        assert_eq!(ctx.constant(204692).if_constant(), Some(204692));
        assert_eq!(ctx.register(8).if_register(), Some(8));
        assert_eq!(ctx.custom(3).if_custom(), Some(3));

        let a = ctx.register(2);
        assert_eq!(
            ctx.sign_extend(a, MemAccessSize::Mem8, MemAccessSize::Mem32).if_sign_extend(),
            Some((a, MemAccessSize::Mem8, MemAccessSize::Mem32)),
        );
        assert_eq!(ctx.mem8(a, 8).if_mem8(), Some(&ctx.mem_access(a, 8, MemAccessSize::Mem8)));
        assert_eq!(ctx.mem16(a, 8).if_mem16(), Some(&ctx.mem_access(a, 8, MemAccessSize::Mem16)));
        assert_eq!(ctx.mem32(a, 8).if_mem32(), Some(&ctx.mem_access(a, 8, MemAccessSize::Mem32)));
        assert_eq!(ctx.mem64(a, 8).if_mem64(), Some(&ctx.mem_access(a, 8, MemAccessSize::Mem64)));
        assert_eq!(ctx.mem8(a, 8).if_mem8_offset(8), Some(a));
        assert_eq!(ctx.mem16(a, 8).if_mem16_offset(8), Some(a));
        assert_eq!(ctx.mem32(a, 8).if_mem32_offset(8), Some(a));
        assert_eq!(ctx.mem64(a, 8).if_mem64_offset(8), Some(a));
    }

    #[test]
    fn matching_funcs_arith() {
        let ctx = OperandContext::new();
        let a = ctx.register(5);
        let b = ctx.custom(9);
        let pair = Some((a, b));
        let pair2 = Some((b, a));
        assert!(
            ctx.add(a, b).if_arithmetic_add() == pair ||
            ctx.add(a, b).if_arithmetic_add() == pair2,
        );
        assert_eq!(ctx.sub(a, b).if_arithmetic_sub(), pair);
        assert!(
            ctx.mul(a, b).if_arithmetic_mul() == pair ||
            ctx.mul(a, b).if_arithmetic_mul() == pair2,
        );
        assert!(
            ctx.mul_high(a, b).if_arithmetic_mul_high() == pair ||
            ctx.mul_high(a, b).if_arithmetic_mul_high() == pair2,
        );
        assert_eq!(ctx.lsh(a, b).if_arithmetic_lsh(), pair);
        assert_eq!(ctx.rsh(a, b).if_arithmetic_rsh(), pair);
        assert!(
            ctx.or(a, b).if_arithmetic_or() == pair ||
            ctx.or(a, b).if_arithmetic_or() == pair2,
        );
        assert!(
            ctx.and(a, b).if_arithmetic_and() == pair ||
            ctx.and(a, b).if_arithmetic_and() == pair2,
        );
        assert!(
            ctx.xor(a, b).if_arithmetic_xor() == pair ||
            ctx.xor(a, b).if_arithmetic_xor() == pair2,
        );
        assert!(
            ctx.eq(a, b).if_arithmetic_eq() == pair ||
            ctx.eq(a, b).if_arithmetic_eq() == pair2,
        );
        assert_eq!(ctx.gt(a, b).if_arithmetic_gt(), pair);
    }

    #[test]
    fn matching_funcs_arith_const() {
        let ctx = OperandContext::new();
        let a = ctx.register(5);
        let b = 17u64;
        let pair = Some((a, b));
        assert_eq!(ctx.add_const(a, b).if_add_with_const(), pair);
        assert_eq!(ctx.sub_const(a, b).if_sub_with_const(), pair);
        assert_eq!(ctx.mul_const(a, b).if_mul_with_const(), pair);
        assert_eq!(ctx.and_const(a, b).if_and_with_const(), pair);
        assert_eq!(ctx.or_const(a, b).if_or_with_const(), pair);
        assert_eq!(ctx.xor_const(a, b).if_xor_with_const(), pair);
        assert_eq!(ctx.lsh_const(a, b).if_lsh_with_const(), pair);
        assert_eq!(ctx.rsh_const(a, b).if_rsh_with_const(), pair);
        assert_eq!(ctx.eq_const(a, b).if_eq_with_const(), pair);
        assert_eq!(ctx.gt_const(a, b).if_gt_with_const(), pair);
    }

    #[test]
    fn intern_registers() {
        for i in 0..256u32 {
            let ctx = OperandContext::new();
            let a = ctx.register(i as u8);
            assert_eq!(a.if_register(), Some(i as u8));
            assert_eq!(a, ctx.register(i as u8));
        }
    }

    #[test]
    fn intern_arch() {
        let ctx = OperandContext::new();
        for i in 0..256u32 {
            let a = ctx.register(i as u8);
            let b = ctx.arch(i);
            assert_eq!(a, b);
        }

        for i in 0..256u32 {
            // Had a bug with special handling of flag interning
            let _ = ctx.arch(0x50000 | i);
        }
    }
}

/// ```compile_fail
/// use scarf::{OperandCtx, OperandContext};
/// fn x<'e>(ctx: OperandCtx<'e>) {
///     let a = ctx.constant(1);
///     let ctx2 = OperandContext::new();
///     let b = ctx2.constant(2);
///     let sum = ctx.add(a, b);
/// }
/// ```
///
/// ```compile_fail
/// use scarf::{OperandCtx, OperandContext};
/// fn x<'e>(ctx: OperandCtx<'e>) {
///     let a = ctx.constant(1);
///     let ctx2 = OperandContext::new();
///     let b = ctx2.constant(2);
///     let sum = ctx2.add(a, b);
/// }
/// ```
///
/// A passing test in similar form as the above ones to make sure
/// the code doesn't fail for unrelated reasons.
/// ```
/// use scarf::{OperandCtx, OperandContext};
/// fn x<'e>(ctx: OperandCtx<'e>) {
///     let a = ctx.constant(1);
///     let b = ctx.constant(2);
///     let sum = ctx.add(a, b);
/// }
/// ```
#[cfg(doctest)]
extern {}
