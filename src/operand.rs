mod intern;
mod simplify;
mod slice_stack;
#[cfg(test)]
mod simplify_tests;

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

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::bit_misc::{bits_overlap};

use self::slice_stack::SliceStack;

#[derive(Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize), serde(transparent))]
pub struct Operand<'e>(&'e OperandBase<'e>, PhantomData<&'e mut &'e ()>);

/// Wrapper around `Operand` which implements `Hash` on the interned address.
/// Separate struct since hashing by address gives hashes that aren't stable
/// across executions or even separate `OperandContext`s.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct OperandHashByAddress<'e>(pub Operand<'e>);

#[cfg_attr(feature = "serde", derive(Serialize))]
struct OperandBase<'e> {
    ty: OperandType<'e>,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    min_zero_bit_simplify_size: u8,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    relevant_bits: Range<u8>,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    flags: u8,
}

const FLAG_CONTAINS_UNDEFINED: u8 = 0x1;

impl<'e> Hash for OperandHashByAddress<'e> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ((self.0).0 as *const OperandBase<'e> as usize).hash(state)
    }
}

impl<'e> Eq for Operand<'e> { }

// Short-circuit the common case of aliasing pointers
impl<'e> PartialEq for Operand<'e> {
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
            } = *self.0;
            ty.cmp(&other.0.ty)
        }
    }
}

impl<'e> PartialOrd for Operand<'e> {
    fn partial_cmp(&self, other: &Operand<'e>) -> Option<Ordering> {
        Some(self.cmp(other))
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
            Register(r) => write!(f, "Register({})", r.0),
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
            OperandType::Register(r) => match r.0 {
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
            OperandType::Memory(ref mem) => write!(f, "Mem{}[{}]", match mem.size {
                MemAccessSize::Mem8 => "8",
                MemAccessSize::Mem16 => "16",
                MemAccessSize::Mem32 => "32",
                MemAccessSize::Mem64 => "64",
            }, mem.address),
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
                    Parity => write!(f, "parity({})", l),
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

#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum OperandType<'e> {
    Register(Register),
    Xmm(u8, u8),
    Fpu(u8),
    Flag(Flag),
    Constant(u64),
    Memory(MemAccess<'e>),
    Arithmetic(ArithOperand<'e>),
    ArithmeticFloat(ArithOperand<'e>, MemAccessSize),
    Undefined(UndefinedId),
    SignExtend(Operand<'e>, MemAccessSize, MemAccessSize),
    /// Arbitrary user-defined variable that does not compare equal with anything,
    /// and is guaranteed not to be generated by scarf's execution simulation.
    Custom(u32),
}

#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct ArithOperand<'e> {
    pub ty: ArithOpType,
    pub left: Operand<'e>,
    pub right: Operand<'e>,
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq, Ord, PartialOrd)]
pub enum ArithOpType {
    Add,
    Sub,
    Mul,
    SignedMul,
    Div,
    Modulo,
    And,
    Or,
    Xor,
    Lsh,
    Rsh,
    Equal,
    Parity,
    GreaterThan,
    ToFloat,
    ToDouble,
    ToInt,
}

impl<'e> ArithOperand<'e> {
    pub fn is_compare_op(&self) -> bool {
        use self::ArithOpType::*;
        match self.ty {
            Equal | GreaterThan => true,
            _ => false,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub struct UndefinedId(#[cfg_attr(feature = "serde", serde(skip))] pub u32);

pub struct OperandContext<'e> {
    next_undefined: Cell<u32>,
    // Contains 0x41 small constants, 0x10 registers, 0x6 flags
    common_operands: Box<[OperandSelfRef; 0x41 + 0x10 + 0x6]>,
    interner: intern::Interner<'e>,
    undef_interner: intern::UndefInterner,
    invariant_lifetime: PhantomData<&'e mut &'e ()>,
    simplify_temp_stack: SliceStack,
    // Caches registers "reg + x" where x can be in range -0x20..=0xdf
    // for registers other than rsp/rbp, and in range
    // -0x220..=0xdf for those two.
    // As such, it takes 0x100 * 6 + 0x300 * 2 = 3072 words of memory for 32-bit, and
    // 0x100 * 14 + 0x300 * 2 = 5120 words for 64-bit.
    // Size can be set with OperandCtx::resize_offset_cache
    offset_cache: RefCell<Vec<Option<Operand<'e>>>>,
}

/// Convenience alias for `OperandContext` reference that avoids having to
/// type the `'e` lifetime twice.
pub type OperandCtx<'e> = &'e OperandContext<'e>;

/// Represents an Operand<'e> stored in OperandContext.
/// Rust doesn't allow this sort of setup without unsafe code,
/// so have this sort of wrapper to somewhat separate these
/// unsafe casts from other operands.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct OperandSelfRef(*const ());

impl OperandSelfRef {
    fn new<'e>(operand: Operand<'e>) -> OperandSelfRef {
        OperandSelfRef(operand.0 as *const OperandBase<'e> as *const ())
    }

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
            inner.pos = m.address;
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
            pub fn $name(&$lt self) -> Operand<$lt> {
                self.constant($val)
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
    pub fn new() -> OperandContext<'e> {
        use std::ptr::null_mut;
        let common_operands = Box::alloc().init([OperandSelfRef(null_mut()); 0x41 + 0x10 + 0x6]);
        let mut result: OperandContext<'e> = OperandContext {
            next_undefined: Cell::new(0),
            common_operands,
            interner: intern::Interner::new(),
            undef_interner: intern::UndefInterner::new(),
            invariant_lifetime: PhantomData,
            simplify_temp_stack: SliceStack::new(),
            offset_cache: RefCell::new(Vec::new()),
        };
        let common_operands = &mut result.common_operands;
        // Accessing interner here would force the invariant lifetime 'e to this stack frame.
        // Cast the interner reference to arbitrary lifetime to allow returning the result.
        let interner: &intern::Interner<'_> = unsafe { std::mem::transmute(&result.interner) };
        for i in 0..0x41 {
            common_operands[i] = interner.intern(OperandType::Constant(i as u64)).self_ref();
        }
        let base = 0x41;
        for i in 0..0x10 {
            common_operands[base + i] =
                interner.intern(OperandType::Register(Register(i as u8))).self_ref();
        }
        let base = 0x41 + 0x10;
        common_operands[base + 0] =
            interner.intern(OperandType::Flag(Flag::Zero)).self_ref();
        common_operands[base + 1] =
            interner.intern(OperandType::Flag(Flag::Carry)).self_ref();
        common_operands[base + 2] =
            interner.intern(OperandType::Flag(Flag::Overflow)).self_ref();
        common_operands[base + 3] =
            interner.intern(OperandType::Flag(Flag::Parity)).self_ref();
        common_operands[base + 4] =
            interner.intern(OperandType::Flag(Flag::Sign)).self_ref();
        common_operands[base + 5] =
            interner.intern(OperandType::Flag(Flag::Direction)).self_ref();
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
        let ty = match *op.ty() {
            OperandType::Register(reg) => OperandType::Register(reg),
            OperandType::Xmm(a, b) => OperandType::Xmm(a, b),
            OperandType::Flag(f) => OperandType::Flag(f),
            OperandType::Fpu(f) => OperandType::Fpu(f),
            OperandType::Constant(c) => OperandType::Constant(c),
            OperandType::Undefined(c) => OperandType::Undefined(c),
            OperandType::Custom(c) => OperandType::Custom(c),
            OperandType::Memory(ref mem) => OperandType::Memory(MemAccess {
                address: self.copy_operand(mem.address),
                size: mem.size,
            }),
            OperandType::Arithmetic(ref arith) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand(arith.left),
                    right: self.copy_operand(arith.right),
                };
                OperandType::Arithmetic(arith)
            }
            OperandType::ArithmeticFloat(ref arith, size) => {
                let arith = ArithOperand {
                    ty: arith.ty,
                    left: self.copy_operand(arith.left),
                    right: self.copy_operand(arith.right),
                };
                OperandType::ArithmeticFloat(arith, size)
            }
            OperandType::SignExtend(a, b, c) => {
                OperandType::SignExtend(self.copy_operand(a), b, c)
            }
        };
        self.intern(ty)
    }

    operand_context_const_methods! {
        'e,
        const_0, 0x0,
        const_1, 0x1,
        const_2, 0x2,
        const_4, 0x4,
        const_8, 0x8,
        const_1f, 0x1f,
        const_20, 0x20,
        const_7f, 0x7f,
        const_ff, 0xff,
        const_7fff, 0x7fff,
        const_ff00, 0xff00,
        const_ffff, 0xffff,
        const_ffff0000, 0xffff0000,
        const_ffffff00, 0xffffff00,
        const_ffff00ff, 0xffff00ff,
        const_7fffffff, 0x7fffffff,
        const_ffffffff, 0xffffffff,
    }

    fn intern(&'e self, ty: OperandType<'e>) -> Operand<'e> {
        self.interner.intern(ty)
    }

    pub fn new_undef(&'e self) -> Operand<'e> {
        let id = self.next_undefined.get();
        self.next_undefined.set(id + 1);
        self.undef_interner.push(OperandType::Undefined(UndefinedId(id)))
    }

    pub fn flag_z(&'e self) -> Operand<'e> {
        self.flag(Flag::Zero)
    }

    pub fn flag_c(&'e self) -> Operand<'e> {
        self.flag(Flag::Carry)
    }

    pub fn flag_o(&'e self) -> Operand<'e> {
        self.flag(Flag::Overflow)
    }

    pub fn flag_s(&'e self) -> Operand<'e> {
        self.flag(Flag::Sign)
    }

    pub fn flag_p(&'e self) -> Operand<'e> {
        self.flag(Flag::Parity)
    }

    pub fn flag_d(&'e self) -> Operand<'e> {
        self.flag(Flag::Direction)
    }

    pub fn flag(&'e self, flag: Flag) -> Operand<'e> {
        self.flag_by_index(flag as usize)
    }

    pub(crate) fn flag_by_index(&'e self, index: usize) -> Operand<'e> {
        assert!(index < 6);
        unsafe { self.common_operands[0x41 + 0x10 + index as usize].cast() }
    }

    pub fn register(&'e self, index: u8) -> Operand<'e> {
        if index <= 0x10 {
            unsafe { self.common_operands[0x41 + index as usize].cast() }
        } else {
            self.intern(OperandType::Register(Register(index)))
        }
    }

    pub fn register_ref(&'e self, index: u8) -> Operand<'e> {
        self.register(index)
    }

    pub fn register_fpu(&'e self, index: u8) -> Operand<'e> {
        self.intern(OperandType::Fpu(index))
    }

    pub fn xmm(&'e self, num: u8, word: u8) -> Operand<'e> {
        self.intern(OperandType::Xmm(num, word))
    }

    pub fn custom(&'e self, value: u32) -> Operand<'e> {
        self.intern(OperandType::Custom(value))
    }

    pub fn constant(&'e self, value: u64) -> Operand<'e> {
        if value <= 0x40 {
            unsafe { self.common_operands[value as usize].cast() }
        } else {
            self.intern(OperandType::Constant(value))
        }
    }

    /// Returns operand limited to low `size` bits
    pub fn truncate(&'e self, operand: Operand<'e>, size: u8) -> Operand<'e> {
        let high = 64 - size;
        let mask = !0u64 << high >> high;
        self.and_const(operand, mask)
    }

    pub fn arithmetic(
        &'e self,
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
    ) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_arith(left, right, ty, self, &mut simplify)
    }

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
    ///
    /// The returned value is simplified.
    pub fn add(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_add_sub(left, right, false, self)
    }

    /// Returns `Operand` for `left - right`.
    ///
    /// The returned value is simplified.
    pub fn sub(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_add_sub(left, right, true, self)
    }

    /// Returns `Operand` for `left * right`.
    ///
    /// The returned value is simplified.
    pub fn mul(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_mul(left, right, self)
    }

    /// Returns `Operand` for signed `left * right`.
    ///
    /// The returned value is simplified.
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
    ///
    /// The returned value is simplified.
    pub fn div(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.arithmetic(ArithOpType::Div, left, right)
    }

    /// Returns `Operand` for `left % right`.
    ///
    /// The returned value is simplified.
    pub fn modulo(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.arithmetic(ArithOpType::Modulo, left, right)
    }

    /// Returns `Operand` for `left & right`.
    ///
    /// The returned value is simplified.
    pub fn and(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_and(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    ///
    /// The returned value is simplified.
    pub fn or(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_or(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    ///
    /// The returned value is simplified.
    pub fn xor(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_xor(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    ///
    /// The returned value is simplified.
    pub fn lsh(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    ///
    /// The returned value is simplified.
    pub fn rsh(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_rsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left == right`.
    ///
    /// The returned value is simplified.
    pub fn eq(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        simplify::simplify_eq(left, right, self)
    }

    /// Returns `Operand` for `left != right`.
    ///
    /// The returned value is simplified.
    pub fn neq(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.eq_const(self.eq(left, right), 0)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt(&'e self, left: Operand<'e>, right: Operand<'e>) -> Operand<'e> {
        self.arithmetic(ArithOpType::GreaterThan, left, right)
    }

    /// Returns `Operand` for signed `left > right`.
    ///
    /// The returned value is simplified.
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
        self.gt(
            self.and_const(
                self.add_const(left, offset),
                mask,
            ),
            self.and_const(
                self.add_const(right, offset),
                mask,
            ),
        )
    }

    /// Returns `Operand` for `left + right`.
    ///
    /// The returned value is simplified.
    pub fn add_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        simplify::simplify_add_const(left, right, self)
    }

    /// Returns `Operand` for `left - right`.
    ///
    /// The returned value is simplified.
    pub fn sub_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        simplify::simplify_sub_const(left, right, self)
    }

    /// Returns `Operand` for `left - right`.
    ///
    /// The returned value is simplified.
    pub fn sub_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        simplify::simplify_add_sub(left, right, true, self)
    }

    /// Returns `Operand` for `left * right`.
    ///
    /// The returned value is simplified.
    pub fn mul_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        simplify::simplify_mul(left, right, self)
    }

    /// Returns `Operand` for `left & right`.
    ///
    /// The returned value is simplified.
    pub fn and_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_and_const(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    ///
    /// The returned value is simplified.
    pub fn or_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_or(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    ///
    /// The returned value is simplified.
    pub fn xor_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_xor(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    ///
    /// The returned value is simplified.
    pub fn lsh_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    ///
    /// The returned value is simplified.
    pub fn lsh_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    ///
    /// The returned value is simplified.
    pub fn rsh_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_rsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left == right`.
    ///
    /// The returned value is simplified.
    pub fn eq_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        simplify::simplify_eq(left, right, self)
    }

    /// Returns `Operand` for `left != right`.
    ///
    /// The returned value is simplified.
    pub fn neq_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        self.eq_const(self.eq(left, right), 0)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt_const(&'e self, left: Operand<'e>, right: u64) -> Operand<'e> {
        let right = self.constant(right);
        self.gt(left, right)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt_const_left(&'e self, left: u64, right: Operand<'e>) -> Operand<'e> {
        let left = self.constant(left);
        self.gt(left, right)
    }

    pub fn mem64(&'e self, val: Operand<'e>) -> Operand<'e> {
        self.mem_variable_rc(MemAccessSize::Mem64, val)
    }

    pub fn mem32(&'e self, val: Operand<'e>) -> Operand<'e> {
        self.mem_variable_rc(MemAccessSize::Mem32, val)
    }

    pub fn mem16(&'e self, val: Operand<'e>) -> Operand<'e> {
        self.mem_variable_rc(MemAccessSize::Mem16, val)
    }

    pub fn mem8(&'e self, val: Operand<'e>) -> Operand<'e> {
        self.mem_variable_rc(MemAccessSize::Mem8, val)
    }

    pub fn mem_variable_rc(&'e self, size: MemAccessSize, addr: Operand<'e>) -> Operand<'e> {
        let ty = OperandType::Memory(MemAccess {
            address: addr,
            size,
        });
        self.intern(ty)
    }

    pub fn sign_extend(
        &'e self,
        val: Operand<'e>,
        from: MemAccessSize,
        to: MemAccessSize,
    ) -> Operand<'e> {
        simplify::simplify_sign_extend(val, from, to, self)
    }

    pub fn transform<F>(&'e self, oper: Operand<'e>, mut f: F) -> Operand<'e>
    where F: FnMut(Operand<'e>) -> Option<Operand<'e>>
    {
        self.transform_internal(oper, &mut f)
    }

    fn transform_internal<F>(&'e self, oper: Operand<'e>, f: &mut F) -> Operand<'e>
    where F: FnMut(Operand<'e>) -> Option<Operand<'e>>
    {
        if let Some(val) = f(oper) {
            return val;
        }
        match *oper.ty() {
            OperandType::Arithmetic(ref arith) => {
                let left = self.transform_internal(arith.left, f);
                let right = self.transform_internal(arith.right, f);
                if left == arith.left && right == arith.right {
                    oper
                } else {
                    self.arithmetic(arith.ty, left, right)
                }
            },
            OperandType::Memory(ref m) => {
                let address = self.transform_internal(m.address, f);
                if address == m.address {
                    oper
                } else {
                    self.mem_variable_rc(m.size, address)
                }
            }
            OperandType::SignExtend(val, from, to) => {
                let new_val = self.transform_internal(val, f);
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
    ) -> Operand<'e> {
        if let Some(mem) = val.if_memory() {
            // Transform also Mem16[mem.addr] to with & 0xffff if val is Mem32, etc.
            // I guess recursing inside mem.addr doesn't make sense here,
            // but didn't give it too much thought.
            self.transform(oper, |old| {
                old.if_memory()
                    .filter(|old| old.address == mem.address)
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
            self.transform(oper, |old| match old == val {
                true => Some(with),
                false => None,
            })
        }
    }

    /// Gets amount of operands interned. Intented for debug / diagnostic info.
    pub fn interned_count(&self) -> usize {
        self.interner.interned_count() + self.next_undefined.get() as usize
    }

    pub(crate) fn simplify_temp_stack(&'e self) -> &'e SliceStack {
        &self.simplify_temp_stack
    }

    pub(crate) fn resize_offset_cache(&'e self, register_count: usize) {
        // Optimization trick: resize_with using constant None is more likely to optimize
        // to memset than usual resize which likely optimizes to a one-at-time loop.
        let size = (register_count - 2) * 0x100 + 0x300 * 2;
        let mut offset_cache = self.offset_cache.borrow_mut();
        if offset_cache.len() < size {
            offset_cache.resize_with(size, || None);
        }
    }

    // Mainly for disasm. Maybe other components could benefit from this cache as well.
    pub(crate) fn register_offset_const(&'e self, register: u8, offset: i32) -> Operand<'e> {
        if offset < 0xe0 {
            if register == 4 || register == 5 {
                // rsp/rbp
                if offset >= -0x220 {
                    let index = (register - 4) as usize * 0x300 + (offset + 0x220) as usize;
                    if let Ok(mut offset_cache) = self.offset_cache.try_borrow_mut() {
                        if let Some(maybe_cached) = offset_cache.get_mut(index) {
                            return *maybe_cached.get_or_insert_with(|| {
                                self.add_const(self.register(register), offset as i64 as u64)
                            });
                        }
                    }
                }
            } else {
                if offset >= -0x20 {
                    // 0, 1, 2, 3, 6, 7, ...
                    let l1_index = if register < 4 {
                        register as usize
                    } else {
                        register as usize - 2
                    };
                    let index = 0x600 + l1_index * 0x100 + (offset + 0x20) as usize;
                    if let Ok(mut offset_cache) = self.offset_cache.try_borrow_mut() {
                        if let Some(maybe_cached) = offset_cache.get_mut(index) {
                            return *maybe_cached.get_or_insert_with(|| {
                                self.add_const(self.register(register), offset as i64 as u64)
                            });
                        }
                    }
                }
            }
        }
        // Uncached default
        self.add_const(self.register(register), offset as i64 as u64)
    }
}

impl<'e> OperandType<'e> {
    /// Returns the minimum size of a zero bit range required in simplify_with_zero_bits for
    /// anything to simplify.
    fn min_zero_bit_simplify_size(&self) -> u8 {
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
                _ => {
                    let rel_bits = self.calculate_relevant_bits();
                    rel_bits.end - rel_bits.start
                }
            }
            _ => 0,
        }
    }

    /// Returns which bits the operand will use at most.
    fn calculate_relevant_bits(&self) -> Range<u8> {
        match *self {
            OperandType::Memory(ref mem) => match mem.size {
                MemAccessSize::Mem8 => 0..8,
                MemAccessSize::Mem16 => 0..16,
                MemAccessSize::Mem32 => 0..32,
                MemAccessSize::Mem64 => 0..64,
            },
            OperandType::Arithmetic(ref arith) => match arith.ty {
                ArithOpType::Equal | ArithOpType::GreaterThan => {
                    0..1
                }
                ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let c = c & 0x3f;
                        let left_bits = arith.left.relevant_bits();
                        let start = min(64, left_bits.start + c as u8);
                        let end = min(64, left_bits.end + c as u8);
                        if start <= end {
                            start..end
                        } else {
                            0..0
                        }
                    } else {
                        0..64
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let c = c & 0x3f;
                        let left_bits = arith.left.relevant_bits();
                        let start = left_bits.start.saturating_sub(c as u8);
                        let end = left_bits.end.saturating_sub(c as u8);
                        if start <= end {
                            start..end
                        } else {
                            0..0
                        }
                    } else {
                        0..64
                    }
                }
                ArithOpType::And => {
                    let rel_left = arith.left.relevant_bits();
                    let rel_right = arith.right.relevant_bits();
                    if !bits_overlap(&rel_left, &rel_right) {
                        0..0
                    } else {
                        max(rel_left.start, rel_right.start)..min(rel_left.end, rel_right.end)
                    }
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
                ArithOpType::Mul => {
                    let left_bits = arith.left.relevant_bits();
                    let right_bits = arith.right.relevant_bits();
                    if left_bits == (0..0) || right_bits == (0..0) {
                        return 0..0;
                    }
                    // 64 + 64 cannot overflow
                    let low = left_bits.start.wrapping_add(right_bits.start).min(64);
                    let high = left_bits.end.wrapping_add(right_bits.end).min(64);
                    if low >= high {
                        0..0
                    } else {
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
                        arith.left.relevant_bits()
                    }
                }
                _ => 0..64,
            },
            OperandType::Constant(c) => {
                let trailing = c.trailing_zeros() as u8;
                let leading = c.leading_zeros() as u8;
                if 64 - leading < trailing {
                    0..0
                } else {
                    trailing..(64 - leading)
                }
            }
            _ => match self.expr_size() {
                MemAccessSize::Mem8 => 0..8,
                MemAccessSize::Mem16 => 0..16,
                MemAccessSize::Mem32 => 0..32,
                MemAccessSize::Mem64 => 0..64,
            },
        }
    }

    fn flags(&self) -> u8 {
        use self::OperandType::*;
        // Only flag at the moment is FLAG_CONTAINS_UNDEFINED
        match *self {
            Memory(ref mem) => mem.address.0.flags,
            SignExtend(val, _, _) => val.0.flags,
            Arithmetic(ref arith) | ArithmeticFloat(ref arith, _) => {
                arith.left.0.flags | arith.right.0.flags
            }
            Xmm(..) | Flag(..) | Fpu(..) | Register(..) | Constant(..) | Custom(..) => 0,
            Undefined(..) => FLAG_CONTAINS_UNDEFINED,
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

    pub fn ty(self) -> &'e OperandType<'e> {
        &self.0.ty
    }

    pub fn hash_by_address(self) -> OperandHashByAddress<'e> {
        OperandHashByAddress(self)
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
                ctx.mem_variable_rc(size, inner)
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
                ctx.sign_extend(inner, from, to);
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
                    0xc => Parity,
                    0xd => GreaterThan,
                    0xe => ToFloat,
                    0xf => ToInt,
                    _ => return None,
                };
                ctx.arithmetic(ty, left, right)
            }
            _ => return None,
        })
    }

    /// Returns true if self.ty() == OperandType::Undefined
    pub fn is_undefined(self) -> bool {
        match self.ty() {
            OperandType::Undefined(_) => true,
            _ => false,
        }
    }

    /// Returns true if self or any child operand is Undefined
    pub fn contains_undefined(self) -> bool {
        self.0.flags & FLAG_CONTAINS_UNDEFINED != 0
    }

    pub fn iter(self) -> Iter<'e> {
        Iter(Some(IterState {
            pos: self,
            stack: Vec::new(),
        }))
    }

    pub fn iter_no_mem_addr(self) -> IterNoMemAddr<'e> {
        IterNoMemAddr(Some(IterState {
            pos: self,
            stack: Vec::new(),
        }))
    }

    /// Returns what bits in this value are not guaranteed to be zero.
    ///
    /// End cannot be larger than 64.
    ///
    /// Can be also seen as trailing_zeros .. 64 - leading_zeros range
    pub fn relevant_bits(self) -> Range<u8> {
        self.0.relevant_bits.clone()
    }

    pub fn relevant_bits_mask(self) -> u64 {
        if self.0.relevant_bits.start >= self.0.relevant_bits.end {
            0
        } else {
            let low = self.0.relevant_bits.start;
            let high = 64 - self.0.relevant_bits.end;
            !0u64 << high >> high >> low << low
        }
    }

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

    /// Returns `Some(c)` if `self.ty` is `OperandType::Constant(c)`
    pub fn if_constant(self) -> Option<u64> {
        match *self.ty() {
            OperandType::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(c)` if `self.ty` is `OperandType::Custom(c)`
    pub fn if_custom(self) -> Option<u32> {
        match *self.ty() {
            OperandType::Custom(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(r)` if `self.ty` is `OperandType::Register(r)`
    pub fn if_register(self) -> Option<Register> {
        match *self.ty() {
            OperandType::Register(r) => Some(r),
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)`
    pub fn if_memory(self) -> Option<&'e MemAccess<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => Some(mem),
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem64`
    pub fn if_mem64(self) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem64 {
                true => Some(mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem32`
    pub fn if_mem32(self) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem32 {
                true => Some(mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem16`
    pub fn if_mem16(self) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem16 {
                true => Some(mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem8`
    pub fn if_mem8(self) -> Option<Operand<'e>> {
        match *self.ty() {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem8 {
                true => Some(mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if self.ty is `OperandType::Arithmetic { ty == ty }`
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
    pub fn if_arithmetic_add(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Add)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Sub(left, right))`
    pub fn if_arithmetic_sub(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Sub)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Mul(left, right))`
    pub fn if_arithmetic_mul(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Mul)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Equal(left, right))`
    pub fn if_arithmetic_eq(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Equal)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::GreaterThan(left, right))`
    pub fn if_arithmetic_gt(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::GreaterThan)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::And(left, right))`
    pub fn if_arithmetic_and(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::And)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Or(left, right))`
    pub fn if_arithmetic_or(self) -> Option<(Operand<'e>, Operand<'e>)> {
        self.if_arithmetic(ArithOpType::Or)
    }

    /// Returns `Some((register, constant))` if operand is an and mask of register
    /// with constant.
    ///
    /// Useful for detecting 32-bit register which is represented as `Register(r) & ffff_ffff`.
    pub fn if_and_masked_register(self) -> Option<(Register, u64)> {
        let (l, r) = self.if_arithmetic_and()?;
        let (reg, other) = Operand::either(l, r, |x| x.if_register())?;
        let other = other.if_constant()?;
        Some((reg, other))
    }

    /// Returns `(other, constant)` if operand is an and mask with constant,
    /// or just (self, u64::max_value())
    pub fn and_masked(this: Operand<'e>) -> (Operand<'e>, u64) {
        this.if_arithmetic_and()
            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
            .map(|(c, o)| (o, c))
            .unwrap_or_else(|| (this, u64::max_value()))
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
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct MemAccess<'e> {
    pub address: Operand<'e>,
    pub size: MemAccessSize,
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Hash, Eq, PartialEq, Copy, Debug, Ord, PartialOrd)]
pub enum MemAccessSize {
    Mem32,
    Mem16,
    Mem8,
    Mem64,
}

impl MemAccessSize {
    pub fn bits(self) -> u32 {
        match self {
            MemAccessSize::Mem64 => 64,
            MemAccessSize::Mem32 => 32,
            MemAccessSize::Mem16 => 16,
            MemAccessSize::Mem8 => 8,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Eq, PartialEq, Copy, Debug, Hash, Ord, PartialOrd)]
pub struct Register(pub u8);

// Flags currently are cast to usize index when stored in ExecutionState
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
                                "address": {
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
            ),
        );
        let mut des = serde_json::Deserializer::from_str(&json);
        let op2: Operand<'_> = ctx.deserialize_seed().deserialize(&mut des).unwrap();
        assert_eq!(op, op2);
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
