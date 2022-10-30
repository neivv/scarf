//! [`Operand`] and its supporting types.
//!
//! Main types here are [`OperandContext`] for the arena all operands are allocated in,
//! and [`Operand`], scarf's 'value/variable/expression' type.

mod intern;
mod simplify;
pub(crate) mod slice_stack;
#[cfg(test)]
mod simplify_tests;
mod util;

#[cfg(feature = "serde")]
mod deserialize;
#[cfg(feature = "serde")]
pub use self::deserialize::DeserializeOperand;

use std::cell::{Cell, RefCell};
use std::cmp::{max, min, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::Range;
use std::ptr;

use copyless::BoxHelper;
use fxhash::FxHashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::exec_state;

use self::slice_stack::SliceStack;

/// `Operand` is the type of values in scarf.
///
/// It is an immutable reference type, free to copy and expected to be passed by value.
///
/// Different types of `Operand`s are listed in [`OperandType`] enum.
/// The main types of interest are:
///
/// - Single variables, such as `OperandType::Register`, `OperandType::Xmm`,
/// `OperandType::Custom`.
/// - Constant integers, `OperandType::Constant`.
/// - Expressions, `OperandType::Arithmetic`, using `Operand` as inputs for the expression,
/// and as such, able to have arbitrarily deep tree of subexpressions making up the `Operand`.
/// - Memory, `OperandType::Memory`, which is able to have any `Operand` representing the
/// address.
///
/// All `Operand`s are created through [`OperandContext`] arena, which interns the created
/// `Operand`s, allowing implementing equality comparision as a single reference equality
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
/// to make equality comparisions useful. As the simplification rules get improved, this may
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
    min_zero_bit_simplify_size: u8,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    relevant_bits: Range<u8>,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    flags: u8,
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
    /// comparision includes equality check already.)
    ///
    /// `sort_order` could be also used as input to hash, though it would require
    /// remembering to add data that is not included, with `Memory` and other
    /// large types.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    sort_order: u64,
}

const FLAG_CONTAINS_UNDEFINED: u8 = 0x1;
// For simplify_with_and_mask optimization.
// For example, ((x & ff) | y) & ff should remove the inner mask.
const FLAG_COULD_REMOVE_CONST_AND: u8 = 0x2;
// If not set, resolve(x) == x
// (Constants, undef, custom, and arithmetic using them)
const FLAG_NEEDS_RESOLVE: u8 = 0x4;
const FLAG_CONTAINS_MEMORY: u8 = 0x8;
const FLAG_32BIT_NORMALIZED: u8 = 0x10;
// The operand is add / sub with at least one of the terms
// having FLAG_32BIT_NORMALIZED set
const FLAG_PARTIAL_32BIT_NORMALIZED_ADD: u8 = 0x20;
const ALWAYS_INHERITED_FLAGS: u8 =
    FLAG_CONTAINS_UNDEFINED | FLAG_NEEDS_RESOLVE | FLAG_CONTAINS_MEMORY;

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
                min_zero_bit_simplify_size: _,
                relevant_bits: _,
                flags: _,
                sort_order,
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
            min_zero_bit_simplify_size: _,
            relevant_bits: _,
            flags: _,
            sort_order,
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
            .field("min_zero_bit_simplify_size", &self.min_zero_bit_simplify_size)
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
            Register(r) => write!(f, "Register({})", r),
            Xmm(r, x) => write!(f, "Xmm({}.{})", r, x),
            Fpu(r) => write!(f, "Fpu({})", r),
            Flag(r) => write!(f, "Flag({:?})", r),
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
        }
    }
}

impl<'e> fmt::Display for Operand<'e> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ArithOpType::*;

        match *self.ty() {
            OperandType::Register(r) => match r {
                0 => write!(f, "rax"),
                1 => write!(f, "rcx"),
                2 => write!(f, "rdx"),
                3 => write!(f, "rbx"),
                4 => write!(f, "rsp"),
                5 => write!(f, "rbp"),
                6 => write!(f, "rsi"),
                7 => write!(f, "rdi"),
                x => write!(f, "r{}", x),
            },
            OperandType::Xmm(reg, subword) => write!(f, "xmm{}.{}", reg, subword),
            OperandType::Fpu(reg) => write!(f, "fpu{}", reg),
            OperandType::Flag(flag) => match flag {
                Flag::Zero => write!(f, "z"),
                Flag::Carry => write!(f, "c"),
                Flag::Overflow => write!(f, "o"),
                Flag::Parity => write!(f, "p"),
                Flag::Sign => write!(f, "s"),
                Flag::Direction => write!(f, "d"),
            },
            OperandType::Constant(c) => write!(f, "{:x}", c),
            OperandType::Memory(ref mem) => {
                let (base, offset) = mem.address();
                if let Some(c) = mem.if_constant_address() {
                    write!(f, "Mem{}[{:x}]", mem.size.bits(), c)
                } else if offset == 0 {
                    write!(f, "Mem{}[{}]", mem.size.bits(), base)
                } else {
                    let (sign, offset) = match offset < 0x8000_0000_0000_0000 {
                        true => ('+', offset),
                        false => ('-', 0u64.wrapping_sub(offset)),
                    };
                    write!(f, "Mem{}[{} {} {:x}]", mem.size.bits(), base, sign, offset)
                }
            }
            OperandType::Undefined(id) => write!(f, "Undefined_{:x}", id.0),
            OperandType::Arithmetic(ref arith) | OperandType::ArithmeticFloat(ref arith, _) => {
                let l = arith.left;
                let r = arith.right;
                match arith.ty {
                    Add => write!(f, "({} + {})", l, r),
                    Sub => write!(f, "({} - {})", l, r),
                    Mul => write!(f, "({} * {})", l, r),
                    Div => write!(f, "({} / {})", l, r),
                    Modulo => write!(f, "({} % {})", l, r),
                    And => write!(f, "({} & {})", l, r),
                    Or => write!(f, "({} | {})", l, r),
                    Xor => write!(f, "({} ^ {})", l, r),
                    Lsh => write!(f, "({} << {})", l, r),
                    Rsh => write!(f, "({} >> {})", l, r),
                    Equal => write!(f, "({} == {})", l, r),
                    GreaterThan => write!(f, "({} > {})", l, r),
                    SignedMul => write!(f, "mul_signed({}, {})", l, r),
                    MulHigh => write!(f, "mul_high({}, {})", l, r),
                    ToFloat => write!(f, "to_float({})", l),
                    ToDouble => write!(f, "to_double({})", l),
                    ToInt => write!(f, "to_int({})", l),
                }?;
                match *self.ty() {
                    OperandType::ArithmeticFloat(_, size) => {
                        write!(f, "[f{}]", size.bits())?;
                    }
                    _ => (),
                }
                Ok(())
            },
            OperandType::SignExtend(ref val, ref from, ref to) => {
                write!(f, "signext_{}_to_{}({})", from.bits(), to.bits(), val)
            }
            OperandType::Custom(val) => {
                write!(f, "Custom_{:x}", val)
            }
        }
    }
}

/// Different values an [`Operand`] can hold.
#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum OperandType<'e> {
    /// A variable representing single general-purpose-register of a CPU.
    Register(u8),
    /// A variable representing single **32-bit word** of the SIMD register of a CPU.
    ///
    /// First `u8` is register number, second is word index (0 = lowest word).
    ///
    /// Scarf currently does not implement AVX, so no YMM or ZMM words, but they likely
    /// would just appear as word index 4 or greater.
    Xmm(u8, u8),
    /// A variable representing a FPU register, only implemented in 32-bit x86 at the moment.
    Fpu(u8),
    /// A variable representing a single flag of x86 EFLAGS. Currently this can hold a
    /// 32-bit value for some reason?? Scarf may be changed to assume that this can only
    /// be 1-bit value later on.
    ///
    /// In practice whenever [`ExecutionState`](crate::exec_state::ExecutionState) implementations
    /// write to a flag, they use either [`Eq`](ArithOpType) or [`GreaterThan`](ArithOpType)
    /// `Operand`s, or combinations of them resulting in 1-bit value anyway.
    Flag(Flag),
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
    Add,
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

/// Newtype to distinguish each [`OperandType::Undefined`] from each other.
///
/// User code should rarely have need to use this.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct UndefinedId(#[cfg_attr(feature = "serde", serde(skip))] pub u32);

const SMALL_CONSTANT_COUNT: usize = 0x110;
const SIGN_AND_MASK_NEARBY_CONSTS: usize = 0x11;
const SIGN_AND_MASK_NEARBY_CONSTS_FIRST: usize = SMALL_CONSTANT_COUNT + 0x10 + 0x6;
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
/// `Operand` if it was already created, allowing equality comparisions be done with a simple
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
/// issues, it will make scarf produce nonsensical results as equality comparisions are no
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
/// - `OperandType::Register(n)`, for `n < 16`
/// - `OperandType::Flag`
/// - `OperandType::Constant(n)`, for 0, 1, 2, 4, 8, through [`ctx.const_0()`](Self::const_0)
///     and other functions.
///
/// Some code uses this to micro-optimize cases where you want to check if `Operand` equals
/// to constant zero/one or constant register.
/// `op.if_constant() == 0` has to do two comparisions and two memory reads:
/// One to verify `OperandType::Constant` variant, second to verify that the constant is zero.
/// `op == ctx.const_0()` does just a single memory read to get the cached const_0, and
/// a single pointer comparision.
pub struct OperandContext<'e> {
    next_undefined: Cell<u32>,
    max_undefined: Cell<u32>,
    // Contains SMALL_CONSTANT_COUNT small constants, 0x10 registers, 0x6 flags
    common_operands: Box<[OperandSelfRef; COMMON_OPERANDS_COUNT]>,
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
                unsafe { self.common_operands[$val].cast() }
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
        use std::ptr::null_mut;
        let common_operands =
            Box::alloc().init([OperandSelfRef(null_mut()); COMMON_OPERANDS_COUNT]);
        let mut result: OperandContext<'e> = OperandContext {
            next_undefined: Cell::new(0),
            max_undefined: Cell::new(0),
            common_operands,
            interner: intern::Interner::new(),
            const_interner: intern::ConstInterner::new(),
            undef_interner: intern::UndefInterner::new(),
            invariant_lifetime: PhantomData,
            simplify_temp_stack: SliceStack::new(),
            freeze_buffer: RefCell::new(Vec::new()),
            copy_operand_cache: RefCell::new(FxHashMap::default()),
        };
        let common_operands = &mut result.common_operands;
        // Accessing interner here would force the invariant lifetime 'e to this stack frame.
        // Cast the interner reference to arbitrary lifetime to allow returning the result.
        let interner: &intern::Interner<'_> = unsafe { std::mem::transmute(&result.interner) };
        let const_interner: &intern::ConstInterner<'_> =
            unsafe { std::mem::transmute(&result.const_interner) };
        for i in 0..SMALL_CONSTANT_COUNT {
            common_operands[i] = const_interner.add_uninterned(i as u64).self_ref();
        }
        let base = SMALL_CONSTANT_COUNT;
        for i in 0..0x10 {
            common_operands[base + i] =
                interner.add_uninterned(OperandType::Register(i as u8)).self_ref();
        }
        let base = SMALL_CONSTANT_COUNT + 0x10;
        let flags = [
            Flag::Zero, Flag::Carry, Flag::Overflow, Flag::Parity, Flag::Sign, Flag::Direction,
        ];
        for (i, &f) in flags.iter().enumerate() {
            common_operands[base + i] = interner.add_uninterned(OperandType::Flag(f)).self_ref();
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
        self.copy_operand_small(op, &mut 32)
    }

    fn copy_operand_small<'other>(
        &'e self,
        op: Operand<'other>,
        recurse_limit: &mut u32,
    ) -> Operand<'e> {
        if *recurse_limit <= 1 {
            let mut map = self.copy_operand_cache.borrow_mut();
            if *recurse_limit == 1 {
                map.clear();
                *recurse_limit = 0;
            }
            return self.copy_operand_large(op, &mut map)
        }
        *recurse_limit -= 1;
        let ty = match *op.ty() {
            OperandType::Register(reg) => OperandType::Register(reg),
            OperandType::Xmm(a, b) => OperandType::Xmm(a, b),
            OperandType::Flag(f) => OperandType::Flag(f),
            OperandType::Fpu(f) => OperandType::Fpu(f),
            OperandType::Constant(c) => OperandType::Constant(c),
            OperandType::Undefined(c) => OperandType::Undefined(c),
            OperandType::Custom(c) => OperandType::Custom(c),
            OperandType::Memory(ref mem) => OperandType::Memory(MemAccess {
                base: self.copy_operand_small(mem.base, recurse_limit),
                offset: mem.offset,
                size: mem.size,
                const_base: mem.const_base,
            }),
            OperandType::Arithmetic(ref arith) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_small(arith.left, recurse_limit),
                    right: self.copy_operand_small(arith.right, recurse_limit),
                };
                OperandType::Arithmetic(arith)
            }
            OperandType::ArithmeticFloat(ref arith, size) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_small(arith.left, recurse_limit),
                    right: self.copy_operand_small(arith.right, recurse_limit),
                };
                OperandType::ArithmeticFloat(arith, size)
            }
            OperandType::SignExtend(a, b, c) => {
                OperandType::SignExtend(self.copy_operand_small(a, recurse_limit), b, c)
            }
        };
        self.intern_any(ty)
    }

    fn copy_operand_large<'other>(
        &'e self,
        op: Operand<'other>,
        cache: &mut FxHashMap<usize, Operand<'e>>
    ) -> Operand<'e> {
        let key = op.0 as *const OperandBase<'other> as usize;
        if let Some(&result) = cache.get(&key) {
            return result;
        }

        let ty = match *op.ty() {
            OperandType::Register(reg) => OperandType::Register(reg),
            OperandType::Xmm(a, b) => OperandType::Xmm(a, b),
            OperandType::Flag(f) => OperandType::Flag(f),
            OperandType::Fpu(f) => OperandType::Fpu(f),
            OperandType::Constant(c) => OperandType::Constant(c),
            OperandType::Undefined(c) => OperandType::Undefined(c),
            OperandType::Custom(c) => OperandType::Custom(c),
            OperandType::Memory(ref mem) => OperandType::Memory(MemAccess {
                base: self.copy_operand_large(mem.base, cache),
                offset: mem.offset,
                size: mem.size,
                const_base: mem.const_base,
            }),
            OperandType::Arithmetic(ref arith) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_large(arith.left, cache),
                    right: self.copy_operand_large(arith.right, cache),
                };
                OperandType::Arithmetic(arith)
            }
            OperandType::ArithmeticFloat(ref arith, size) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand_large(arith.left, cache),
                    right: self.copy_operand_large(arith.right, cache),
                };
                OperandType::ArithmeticFloat(arith, size)
            }
            OperandType::SignExtend(a, b, c) => {
                OperandType::SignExtend(self.copy_operand_large(a, cache), b, c)
            }
        };
        let result = self.intern_any(ty);
        cache.insert(key, result);
        result
    }

    /// Interns operand on the default interner. Shouldn't be used for constants or undefined
    fn intern(&'e self, ty: OperandType<'e>) -> Operand<'e> {
        self.interner.intern(ty)
    }

    fn intern_any(&'e self, ty: OperandType<'e>) -> Operand<'e> {
        if let OperandType::Constant(c) = ty {
            self.constant(c)
        } else if let OperandType::Register(r) = ty {
            self.register(r)
        } else if let OperandType::Flag(f) = ty {
            self.flag(f)
        } else {
            // Undefined has to go here since UndefInterner is just array that things get
            // pushed to and then looked up.. Maybe it would be better for copy_operand
            // to map existing Undefined to new undefined if it meets any?
            self.interner.intern(ty)
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
        unsafe { self.common_operands[SMALL_CONSTANT_COUNT + 0x10 + index as usize].cast() }
    }

    /// Returns [`OperandType::Register(num)`](OperandType) operand.
    ///
    /// If `num` is less than 16, the access is guaranteed to be cheap.
    #[inline]
    pub fn register(&'e self, num: u8) -> Operand<'e> {
        if num < 0x10 {
            unsafe { self.common_operands[SMALL_CONSTANT_COUNT + num as usize].cast() }
        } else {
            self.register_slow(num)
        }
    }

    fn register_slow(&'e self, num: u8) -> Operand<'e> {
        self.intern(OperandType::Register(num))
    }

    /// Returns [`OperandType::Fpu(index)`](OperandType) operand.
    pub fn register_fpu(&'e self, index: u8) -> Operand<'e> {
        self.intern(OperandType::Fpu(index))
    }

    /// Returns [`OperandType::Xmm(num, word)`](OperandType) operand.
    pub fn xmm(&'e self, num: u8, word: u8) -> Operand<'e> {
        self.intern(OperandType::Xmm(num, word))
    }

    /// Returns [`OperandType::Custom(value)`](OperandType) operand.
    ///
    /// Custom operands are guaranteed to be never generated by scarf, allowing user code
    /// use them for arbitrary unknown variables of any meaning.
    pub fn custom(&'e self, value: u32) -> Operand<'e> {
        self.intern(OperandType::Custom(value))
    }

    /// Returns [`OperandType::Constant(value)`](OperandType) operand.
    pub fn constant(&'e self, value: u64) -> Operand<'e> {
        if value < SMALL_CONSTANT_COUNT as u64 {
            unsafe { self.common_operands[value as usize].cast() }
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
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_arith(left, right, ty, self, &mut simplify)
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
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_arith_masked(left, right, ty, mask, self, &mut simplify)
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
        simplify::simplify_add_sub(left, right, false, self)
    }

    /// Returns `Operand` for `left - right`.
    pub fn sub(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_add_sub(left, right, true, self)
    }

    /// Returns `Operand` for `left * right`.
    pub fn mul(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_mul(left, right, self)
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
        simplify::simplify_mul(left, right, self)
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
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_and(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    pub fn or(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_or(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    pub fn xor(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_xor(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    pub fn lsh(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    pub fn rsh(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_rsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left == right`.
    pub fn eq(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_eq(left, right, self)
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
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_gt(left, right, self, &mut simplify)
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
        simplify::simplify_add_const(left, right, self)
    }

    /// Returns `Operand` for `left - right`.
    pub fn sub_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        simplify::simplify_sub_const(left, right, self)
    }

    /// Returns `Operand` for `left - right`.
    pub fn sub_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        simplify::simplify_add_sub(left, right, true, self)
    }

    /// Returns `Operand` for `left * right`.
    pub fn mul_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        simplify::simplify_mul(left, right, self)
    }

    /// Returns `Operand` for `left & right`.
    pub fn and_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_and_const(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    pub fn or_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_or(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    pub fn xor_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_xor(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    pub fn lsh_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        if right >= 64 {
            self.const_0()
        } else {
            let mut simplify = simplify::SimplifyWithZeroBits::default();
            simplify::simplify_lsh_const(left, right as u8, self, &mut simplify)
        }
    }

    /// Returns `Operand` for `left << right`.
    pub fn lsh_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    pub fn rsh_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        if right >= 64 {
            self.const_0()
        } else {
            let mut simplify = simplify::SimplifyWithZeroBits::default();
            simplify::simplify_rsh_const(left, right as u8, self, &mut simplify)
        }
    }

    /// Returns `Operand` for `left == right`.
    pub fn eq_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        simplify::simplify_eq_const(left, right, self)
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
        self.intern(ty)
    }

    /// Creates `Operand` referring to memory from `MemAccess`.
    pub fn memory(&'e self, mem: &MemAccess<'e>) -> Operand<'e> {
        let ty = OperandType::Memory(*mem);
        self.intern(ty)
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
                min_zero_bit_simplify_size: 0,
                relevant_bits: 0..64,
                flags: FLAG_CONTAINS_UNDEFINED | FLAG_32BIT_NORMALIZED,
                sort_order: id as u64,
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

    pub(crate) fn swap_freeze_buffer(&'e self, other: &mut Vec<exec_state::FreezeOperation<'e>>) {
        let mut own = self.freeze_buffer.borrow_mut();
        std::mem::swap(&mut *own, other);
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
    /// - Any 'pure 64-bit variable'; `Register`, `Undefined`, `Custom`
    /// - Additions and subtractions consisting only 32-bit normalized `Operand`s
    /// - Multiplications and left shifts consisting 32-bit normalized `Operand` and a constant
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
        // that can be removed.
        if let OperandType::Arithmetic(arith) = value.ty() {
            if arith.left.0.flags & (FLAG_32BIT_NORMALIZED | FLAG_PARTIAL_32BIT_NORMALIZED_ADD)
                == FLAG_PARTIAL_32BIT_NORMALIZED_ADD
            {
                self.normalize_32bit_add(arith.left)
            } else {
                arith.left
            }
        } else {
            // Expected to be unreachable
            value
        }
    }

    fn normalize_32bit_add(&'e self, value: Operand<'e>) -> Operand<'e> {
        let mut needs_mask = None;
        let mut needs_is_sub = false;
        let mut doesnt_need_mask = self.const_0();
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
    /// Returns the minimum size of a zero bit range required in simplify_with_zero_bits for
    /// anything to simplify.
    ///
    /// Relevant_bits argument should be acquired by calling self.relevant_bits()
    /// (Expected to be something that would be calculated otherwise by the caller anyway)
    fn min_zero_bit_simplify_size(&self, relevant_bits: Range<u8>) -> u8 {
        match *self {
            OperandType::Constant(_) => 0,
            // Mem32 can be simplified to Mem16 if highest bits are zero, etc
            OperandType::Memory(ref mem) => match mem.size {
                MemAccessSize::Mem8 => 8,
                MemAccessSize::Mem16 => 8,
                MemAccessSize::Mem32 => 16,
                MemAccessSize::Mem64 => 32,
            },
            OperandType::Register(_) | OperandType::Flag(_) | OperandType::Undefined(_) => 64,
            OperandType::Arithmetic(ref arith) => match arith.ty {
                ArithOpType::And | ArithOpType::Or | ArithOpType::Xor => {
                    min(
                        arith.left.0.min_zero_bit_simplify_size,
                        arith.right.0.min_zero_bit_simplify_size,
                    )
                }
                ArithOpType::Lsh | ArithOpType::Rsh => {
                    let right_bits = match arith.right.if_constant() {
                        Some(s) => 32u64.saturating_sub(s),
                        None => 32,
                    } as u8;
                    arith.left.0.min_zero_bit_simplify_size.min(right_bits)
                }
                // Could this be better than 0?
                ArithOpType::Add => 0,
                _ => relevant_bits.end - relevant_bits.start,
            }
            _ => 0,
        }
    }

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
                    // Add will only increase nonzero bits by one at most
                    let rel_left = arith.left.relevant_bits();
                    let rel_right = arith.right.relevant_bits();
                    let higher_end = max(rel_left.end, rel_right.end);
                    min(rel_left.start, rel_right.start)..min(higher_end + 1, 64)
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
                (mem.address().0.0.flags & ALWAYS_INHERITED_FLAGS) | FLAG_NEEDS_RESOLVE |
                    FLAG_CONTAINS_MEMORY | default_32_bit_normalized
            }
            SignExtend(val, _, _) => {
                (val.0.flags & ALWAYS_INHERITED_FLAGS) | default_32_bit_normalized
            }
            Arithmetic(ref arith) => {
                let base = (arith.left.0.flags | arith.right.0.flags) & ALWAYS_INHERITED_FLAGS;
                let could_remove_const_and = if
                    arith.ty == ArithOpType::And && arith.right.if_constant().is_some()
                {
                    FLAG_COULD_REMOVE_CONST_AND
                } else {
                    match arith.ty {
                        ArithOpType::And | ArithOpType::Or | ArithOpType::Xor | ArithOpType::Add |
                            ArithOpType::Sub | ArithOpType::Mul =>
                        {
                            (arith.left.0.flags | arith.right.0.flags) &
                                FLAG_COULD_REMOVE_CONST_AND
                        }
                        ArithOpType::Lsh | ArithOpType::Rsh => {
                            arith.left.0.flags & FLAG_COULD_REMOVE_CONST_AND
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
            Xmm(..) | Flag(..) | Fpu(..) | Register(..) => {
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
                _ => {
                    // Require all other arith with const mul / shift
                    // be masked.
                    // Not sure if there are patterns that 32-bit
                    // code uses which would benefit from this
                    // being less strict.
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
                default_32_bit_normalized
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
                            return 0;
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
                                    return default_32_bit_normalized;
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
        if value <= u32::MAX as u64 {
            FLAG_32BIT_NORMALIZED
        } else {
            0
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
            Arithmetic(ref arith) | ArithmeticFloat(ref arith, _) => {
                let ty = arith.ty as u8;
                let hash = fxhash::hash64(&(arith.left.0.sort_order, arith.right.0.sort_order));
                ((ty as u64) << 59) | (hash >> 5)
            }
            // Shift these left so that Arithmetic / Memory sort_order won't
            // shift these out. Shifting left by 32 may compile on 32bit host to
            // just zeroing the low half without actually doing 64-bit shifts?
            Register(a) | Fpu(a) => (a as u64) << 32,
            Flag(a) => (a as u64) << 32,
            Custom(a) => (a as u64) << 32,
            Xmm(a, b) => (((a as u64) << 8) | (b as u64)) << 32,
            // Note: constants / undefined not handled here
            Constant(..) | Undefined(..) => 0,
        }
    }

    /// Separate function for comparision when fast path doesn't work
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
                // comparision can give different result than OperandType comparision since
                // sort_order depends on both left and right, while ArithOperand
                // is left-then-right comparision.
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
        }
        debug_assert!(false, "supposed to be unreachable");
        Ordering::Equal
    }

    #[inline]
    fn discriminant(&self) -> u8 {
        // This gets optimized to just Mem8[x] read as long
        // as ordering matches OperandType definition
        match *self {
            OperandType::Register(..) => 0,
            OperandType::Xmm(..) => 1,
            OperandType::Fpu(..) => 2,
            OperandType::Flag(..) => 3,
            OperandType::Constant(..) => 4,
            OperandType::Memory(..) => 5,
            OperandType::Arithmetic(..) => 6,
            OperandType::ArithmeticFloat(..) => 7,
            OperandType::Undefined(..) => 8,
            OperandType::SignExtend(..) => 9,
            OperandType::Custom(..) => 10,
        }
    }

    /// Returns whether the operand is 8, 16, 32, or 64 bits.
    /// Relevant with signed multiplication, usually operands can be considered
    /// zero-extended u32.
    pub fn expr_size(&self) -> MemAccessSize {
        use self::OperandType::*;
        match *self {
            Memory(ref mem) => mem.size,
            Xmm(..) | Flag(..) | Fpu(..) => MemAccessSize::Mem32,
            Register(..) | Constant(..) | Arithmetic(..) | Undefined(..) |
                Custom(..) | ArithmeticFloat(..) => MemAccessSize::Mem64,
            SignExtend(_, _from, to) => to,
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

    pub fn relevant_bits_mask(self) -> u64 {
        let start = self.0.relevant_bits.start as u32;
        let end = self.0.relevant_bits.end as u32;
        if end >= 64 {
            !(1u64.wrapping_shl(start)
                .wrapping_sub(1))
        } else {
            1u64.wrapping_shl(end)
                .wrapping_sub(1)
                .wrapping_shr(start)
                .wrapping_shl(start)
        }
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

    /// Returns `Some(r)` if `self.ty` is `OperandType::Register(r)`
    #[inline]
    pub fn if_register(self) -> Option<u8> {
        match *self.ty() {
            OperandType::Register(r) => Some(r),
            _ => None,
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

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem64`
    #[inline]
    pub fn if_mem64(self) -> Option<&'e MemAccess<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem64 {
                true => Some(mem),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem32`
    #[inline]
    pub fn if_mem32(self) -> Option<&'e MemAccess<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem32 {
                true => Some(mem),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem16`
    #[inline]
    pub fn if_mem16(self) -> Option<&'e MemAccess<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem16 {
                true => Some(mem),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem8`
    #[inline]
    pub fn if_mem8(self) -> Option<&'e MemAccess<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem8 {
                true => Some(mem),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem64`
    #[inline]
    pub fn if_mem64_offset(self, offset: u64) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem64 {
                true => mem.if_offset(offset),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem32`
    #[inline]
    pub fn if_mem32_offset(self, offset: u64) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem32 {
                true => mem.if_offset(offset),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem16`
    #[inline]
    pub fn if_mem16_offset(self, offset: u64) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem16 {
                true => mem.if_offset(offset),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem8`
    #[inline]
    pub fn if_mem8_offset(self, offset: u64) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem8 {
                true => mem.if_offset(offset),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if self.ty is `OperandType::Arithmetic { ty == ty }`
    #[inline]
    pub fn if_arithmetic(
        self,
        ty: ArithOpType,
    ) -> Option<(Operand<'e>, Operand<'e>)> {
        match *self.ty() {
            OperandType::Arithmetic(ref arith) if arith.ty == ty => {
                Some((arith.left, arith.right))
            }
            _ => None,
        }
    }

    /// Returns `true` if self.ty is `OperandType::Arithmetic { ty == ty }`
    #[inline]
    pub fn is_arithmetic(
        self,
        ty: ArithOpType,
    ) -> bool {
        match *self.ty() {
            OperandType::Arithmetic(ref arith) => arith.ty == ty,
            _ => false,
        }
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

    /// Returns `Some((val, from, to))` if `self.ty` is
    /// `OperandType::SignExtend(val, from, to)`
    #[inline]
    pub fn if_sign_extend(self) -> Option<(Operand<'e>, MemAccessSize, MemAccessSize)> {
        match *self.ty() {
            OperandType::SignExtend(a, b, c) => Some((a, b, c)),
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

    /// Returns `(other, constant)` if operand is an and mask with constant,
    /// or just (self, u64::MAX) otherwise.
    pub fn and_masked(this: Operand<'e>) -> (Operand<'e>, u64) {
        this.if_arithmetic_and()
            .and_then(|(l, r)| Some((l, r.if_constant()?)))
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
            use std::convert::TryInto;
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
    // Note: OperandType comparision relies on base being compared
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
    Mem32,
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
    Carry,
    Overflow,
    Parity,
    Sign,
    Direction,
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
    fn test_normalize_32bit() {
        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.add(
            ctx.register(0),
            ctx.constant(0x1_0000_0000),
        );
        let expected = ctx.register(0);
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.or(
            ctx.register(2),
            ctx.lsh_const(
                ctx.register(1),
                32,
            ),
        );
        let expected = ctx.register(2);
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.and(
            ctx.register(0),
            ctx.constant(0xffff_ffff),
        );
        let expected = ctx.register(0);
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.and(
            ctx.custom(0),
            ctx.constant(0xffff_ffff),
        );
        let expected = ctx.custom(0);
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.constant(0x1_0000_0000);
        let expected = ctx.constant(0);
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.sub(
            ctx.register(0),
            ctx.constant(0xfb01_00fe),
        );
        let expected = ctx.add(
            ctx.register(0),
            ctx.constant(0x04fe_ff02),
        );
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.sub(
            ctx.register(0),
            ctx.constant(0x8000_0000),
        );
        let expected = ctx.add(
            ctx.register(0),
            ctx.constant(0x8000_0000),
        );
        assert_eq!(ctx.normalize_32bit(op), expected);
        assert_eq!(ctx.normalize_32bit(expected), expected);

        let op = ctx.add(
            ctx.register(0),
            ctx.constant(0x8000_0001),
        );
        let expected = ctx.sub(
            ctx.register(0),
            ctx.constant(0x7fff_ffff),
        );
        assert_eq!(ctx.normalize_32bit(op), expected);
        assert_eq!(ctx.normalize_32bit(expected), expected);

        let op = ctx.and(
            ctx.register(0),
            ctx.constant(0x7074_7676_7516_0007),
        );
        let expected = ctx.and(
            ctx.register(0),
            ctx.constant(0x7516_0007),
        );
        assert_eq!(ctx.normalize_32bit(op), expected);
    }

    #[test]
    fn normalize_32bit_complex_multiply_add() {
        // ((x & 0x8000_0000) + y) * 2 will only
        // have y * 2 in low 32 bits.
        // This is something that 32bit execstate probably wouldn't mind
        // getting wrong, but fixing it isn't too bad...

        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.mul_const(
            ctx.add(
                ctx.and_const(
                    ctx.register(0),
                    0x8000_0000,
                ),
                ctx.register(1),
            ),
            2,
        );
        // Should be rcx * 2 but simplification only is able to do this for
        // shifts and not multiplications right now.
        let expected = ctx.and_const(op, 0xffff_ffff);
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.lsh_const(
            ctx.add(
                ctx.and_const(
                    ctx.register(0),
                    0xffff_0000,
                ),
                ctx.register(1),
            ),
            0x10,
        );
        let expected = ctx.mul_const(
            ctx.register(1),
            0x10000,
        );
        assert_eq!(ctx.normalize_32bit(op), expected);

        // No change possible
        let op = ctx.mul_const(
            ctx.add(
                ctx.and_const(
                    ctx.register(0),
                    0xffff_0000,
                ),
                ctx.register(1),
            ),
            0x10001,
        );
        assert_eq!(ctx.normalize_32bit(op), op);

        let op = ctx.lsh_const(
            ctx.add(
                ctx.xmm(0, 0),
                ctx.mul_const(
                    ctx.register(0),
                    2,
                ),
            ),
            0x1f,
        );
        let expected = ctx.lsh_const(
            ctx.xmm(0, 0),
            0x1f,
        );
        assert_eq!(ctx.normalize_32bit(op), expected);
    }

    #[test]
    fn normalize_32bit_complex_multiply_and_const() {
        // Similar to above, but with and mask that could be simplified
        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.lsh_const(
            ctx.and_const(
                ctx.register(1),
                0x4_0004,
            ),
            0x10,
        );
        let expected = ctx.lsh_const(
            ctx.and_const(
                ctx.register(1),
                0x4,
            ),
            0x10,
        );
        assert_eq!(ctx.normalize_32bit(op), expected);

        let op = ctx.lsh_const(
            ctx.add(
                ctx.register(0),
                ctx.and_const(
                    ctx.register(1),
                    0x4_0004,
                ),
            ),
            0x10,
        );
        let expected = ctx.and_const(
            ctx.lsh_const(
                ctx.add(
                    ctx.register(0),
                    ctx.and_const(
                        ctx.register(1),
                        0x4,
                    ),
                ),
                0x10,
            ),
            0xffff_0000,
        );
        assert_eq!(ctx.normalize_32bit(op), expected);
    }

    #[test]
    fn normalize_32bit_complex_multiply_mem() {
        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.lsh_const(
            ctx.mem32c(
                0x14,
            ),
            0x10,
        );
        let expected = ctx.lsh_const(
            ctx.mem16c(
                0x14
            ),
            0x10,
        );
        assert_eq!(ctx.normalize_32bit(op), expected);
    }

    #[test]
    fn normalize_32bit_multiply_nonconst() {
        // This test case serves as a reason to limit 32bit normalization
        // of multiply / shift just to mul/shift by constants, which
        // are main thing that comes up in 32-bit addresses (index in array)
        //
        // If all multiplications are considered normalized like adds are,
        // ((((rbp + (rax * rax)) + (((rdx * 2) + 60002) * rcx)) * rbp) << 11)
        // constant 60002 can be simplified to 2
        // but whether that is possible is too hard to determine without any practical
        // gains.
        //
        // Of course this could just be accepted as a case where normalization won't
        // give same result as 0xffff_ffff mask

        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.lsh_const(
            ctx.mul(
                ctx.add(
                    ctx.add(
                        ctx.register(5),
                        ctx.mul(
                            ctx.register(0),
                            ctx.register(0),
                        ),
                    ),
                    ctx.mul(
                        ctx.add_const(
                            ctx.mul_const(
                                ctx.register(2),
                                2,
                            ),
                            0x60002,
                        ),
                        ctx.register(1),
                    ),
                ),
                ctx.register(5),
            ),
            0x11,
        );
        assert_eq!(
            ctx.normalize_32bit(op),
            ctx.normalize_32bit(
                ctx.and_const(
                    op,
                    0xffff_ffff,
                ),
            ),
        );
    }

    #[test]
    fn normalize_sign_extend() {
        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.lsh_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem16,
                MemAccessSize::Mem32,
            ),
            0x11,
        );
        let eq = ctx.lsh_const(
            ctx.and_const(
                ctx.register(0),
                0x7fff,
            ),
            0x11,
        );
        assert_eq!(ctx.normalize_32bit(op), eq);

        let op = ctx.lsh_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem16,
                MemAccessSize::Mem32,
            ),
            0x10,
        );
        let eq = ctx.lsh_const(
            ctx.and_const(
                ctx.register(0),
                0xffff,
            ),
            0x10,
        );
        assert_eq!(ctx.normalize_32bit(op), eq);
    }

    #[test]
    fn normalize_masked_xor_in_add() {
        let ctx = &crate::operand::OperandContext::new();
        let op = ctx.and_const(
            ctx.add(
                ctx.register(0),
                ctx.xor(
                    ctx.register(1),
                    ctx.register(2),
                ),
            ),
            0xffff_ffff,
        );
        let eq = ctx.add(
            ctx.register(0),
            ctx.and_const(
                ctx.xor(
                    ctx.register(1),
                    ctx.register(2),
                ),
                0xffff_ffff,
            ),
        );
        assert_eq!(ctx.normalize_32bit(op), eq);
    }

    #[test]
    fn normalize_masked_sub() {
        let ctx = &crate::operand::OperandContext::new();
        // rax - (rcx ^ rdx) and rax + ((0 - (rcx ^ rdx)) & ffff_ffff)
        // and rax - ((rcx ^ rdx) & ffff_ffff)
        // should be same normalized (Different with just simplification)
        let op = ctx.sub(
            ctx.register(0),
            ctx.xor(
                ctx.register(1),
                ctx.register(2),
            ),
        );
        let op_masked = ctx.sub(
            ctx.register(0),
            ctx.and_const(
                ctx.xor(
                    ctx.register(1),
                    ctx.register(2),
                ),
                0xffff_ffff,
            ),
        );
        let eq = ctx.add(
            ctx.register(0),
            ctx.and_const(
                ctx.sub(
                    ctx.constant(0),
                    ctx.xor(
                        ctx.register(1),
                        ctx.register(2),
                    ),
                ),
                0xffff_ffff,
            ),
        );
        assert_ne!(op, eq);
        assert_ne!(op, op_masked);
        assert_ne!(eq, op_masked);
        assert_eq!(ctx.normalize_32bit(op), ctx.normalize_32bit(eq));
        assert_eq!(ctx.normalize_32bit(op), ctx.normalize_32bit(op_masked));
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
