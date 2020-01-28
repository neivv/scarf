mod simplify;
#[cfg(test)]
mod simplify_tests;

use std::cell::Cell;
use std::cmp::{max, min, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::rc::Rc;

#[cfg(feature = "serde")]
use serde::{Deserializer, Deserialize, Serialize};

use crate::bit_misc::{bits_overlap};

#[cfg_attr(feature = "serde", derive(Serialize))]
#[derive(Clone, Eq)]
pub struct Operand {
    pub ty: OperandType,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    simplified: bool,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    hash: u64,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    min_zero_bit_simplify_size: u8,
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    relevant_bits: Range<u8>,
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Operand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Operand, D::Error> {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        const FIELDS: &[&str] = &["ty"];
        enum Field {
            Ty,
        }
        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`ty`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                        where E: de::Error
                    {
                        match value {
                            "ty" => Ok(Field::Ty),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct OperandVisitor;

        impl<'de> Visitor<'de> for OperandVisitor {
            type Value = Operand;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Operand")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Operand, V::Error>
                where V: SeqAccess<'de>
            {
                let ty = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let oper = Operand::new_not_simplified_rc(ty);
                Ok((*Operand::simplified(oper)).clone())
            }

            fn visit_map<V>(self, mut map: V) -> Result<Operand, V::Error>
                where V: MapAccess<'de>
            {
                let mut ty = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Ty => {
                            if ty.is_some() {
                                return Err(de::Error::duplicate_field("ty"));
                            }
                            ty = Some(map.next_value()?);
                        }
                    }
                }
                let ty = ty.ok_or_else(|| de::Error::missing_field("ty"))?;
                let oper = Operand::new_not_simplified_rc(ty);
                Ok((*Operand::simplified(oper)).clone())
            }
        }
        deserializer.deserialize_struct("Operand", FIELDS, OperandVisitor)
    }
}

impl Hash for Operand {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Operand {
            ty: _,
            simplified: _,
            min_zero_bit_simplify_size: _,
            relevant_bits: _,
            hash,
        } = *self;
        hash.hash(state)
    }
}

/// Horrible hasher that relies on data being hashed already being well-spread data
/// (Like Operands with their cached hash)
#[derive(Default)]
pub struct OperandDummyHasher {
    value: u64,
}

impl Hasher for OperandDummyHasher {
    fn finish(&self) -> u64 {
        self.value
    }

    fn write(&mut self, data: &[u8]) {
        for &x in data.iter().take(8) {
            self.value = (self.value << 8) | u64::from(x);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.value = value;
    }
}

// Short-circuit the common case of aliasing pointers
impl PartialEq for Operand {
    fn eq(&self, other: &Operand) -> bool {
        if other as *const Operand == self as *const Operand {
            true
        } else if self.hash != other.hash {
            false
        } else {
            let Operand {
                ref ty,
                min_zero_bit_simplify_size: _,
                simplified: _,
                relevant_bits: _,
                hash: _,
            } = *self;
            ty.eq(&other.ty)
        }
    }
}

// Short-circuit the common case of aliasing pointers
impl Ord for Operand {
    fn cmp(&self, other: &Operand) -> Ordering {
        if other as *const Operand == self as *const Operand {
            Ordering::Equal
        } else {
            let Operand {
                ref ty,
                min_zero_bit_simplify_size: _,
                simplified: _,
                relevant_bits: _,
                hash: _,
            } = *self;
            ty.cmp(&other.ty)
        }
    }
}

impl PartialOrd for Operand {
    fn partial_cmp(&self, other: &Operand) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for Operand {
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

impl fmt::Debug for OperandType {
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
            ArithmeticF32(r) => f.debug_tuple("ArithmeticF32").field(r).finish(),
            SignExtend(a, b, c) => {
                f.debug_tuple("SignExtend").field(a).field(b).field(c).finish()
            }
        }
    }
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ArithOpType::*;

        match self.ty {
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
            OperandType::Arithmetic(ref arith) | OperandType::ArithmeticF32(ref arith) => {
                let l = &arith.left;
                let r = &arith.right;
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
                    FloatToInt => write!(f, "float_to_int({})", l),
                    IntToFloat => write!(f, "int_to_float({})", l),
                }?;
                match self.ty {
                    OperandType::ArithmeticF32(..) => {
                        write!(f, "[f32]")?;
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

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum OperandType {
    Register(Register),
    Xmm(u8, u8),
    Fpu(u8),
    Flag(Flag),
    Constant(u64),
    Memory(MemAccess),
    Arithmetic(ArithOperand),
    ArithmeticF32(ArithOperand),
    Undefined(UndefinedId),
    SignExtend(Rc<Operand>, MemAccessSize, MemAccessSize),
    /// Arbitrary user-defined variable that does not compare equal with anything,
    /// and is guaranteed not to be generated by scarf's execution simulation.
    Custom(u32),
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ArithOperand {
    pub ty: ArithOpType,
    pub left: Rc<Operand>,
    pub right: Rc<Operand>,
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
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
    IntToFloat,
    FloatToInt,
}

impl ArithOperand {
    pub fn is_compare_op(&self) -> bool {
        use self::ArithOpType::*;
        match self.ty {
            Equal | GreaterThan => true,
            _ => false,
        }
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct UndefinedId(pub u32);

#[derive(Debug)]
pub struct OperandContext {
    next_undefined: Cell<u32>,
    globals: Rc<OperandCtxGlobals>,
}

#[derive(Debug)]
struct OperandCtxGlobals {
    constants: OperandCtxConstants,
    registers: [Rc<Operand>; 16],
    flags: [Rc<Operand>; 6],
}

pub struct Iter<'a>(Option<IterState<'a>>);
pub struct IterNoMemAddr<'a>(Option<IterState<'a>>);

trait IterVariant<'a> {
    fn descend_to_mem_addr() -> bool;
    fn state<'b>(&'b mut self) -> &'b mut Option<IterState<'a>>;
}

impl<'a> IterVariant<'a> for Iter<'a> {
    fn descend_to_mem_addr() -> bool {
        true
    }

    fn state<'b>(&'b mut self) -> &'b mut Option<IterState<'a>> {
        &mut self.0
    }
}

fn iter_variant_next<'a, T: IterVariant<'a>>(s: &mut T) -> Option<&'a Operand> {
    use self::OperandType::*;

    let inner = match s.state() {
        Some(ref mut s) => s,
        None => return None,
    };
    let next = inner.pos;

    match next.ty {
        Arithmetic(ref arith) | ArithmeticF32(ref arith) => {
            inner.pos = &arith.left;
            inner.stack.push(&arith.right);
        },
        Memory(ref m) if T::descend_to_mem_addr() => {
            inner.pos = &m.address;
        }
        SignExtend(ref val, _, _) => {
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

impl<'a> IterVariant<'a> for IterNoMemAddr<'a> {
    fn descend_to_mem_addr() -> bool {
        false
    }

    fn state<'b>(&'b mut self) -> &'b mut Option<IterState<'a>> {
        &mut self.0
    }
}

struct IterState<'a> {
    pos: &'a Operand,
    stack: Vec<&'a Operand>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Operand;
    fn next(&mut self) -> Option<&'a Operand> {
        iter_variant_next(self)
    }
}

impl<'a> Iterator for IterNoMemAddr<'a> {
    type Item = &'a Operand;
    fn next(&mut self) -> Option<&'a Operand> {
        iter_variant_next(self)
    }
}

macro_rules! operand_context_const_methods {
    ($($name:ident, $field:ident,)*) => {
        $(
            pub fn $name(&self) -> Rc<Operand> {
                self.globals.constants.$field.clone()
            }
        )*
    }
}

#[inline(never)]
fn make_constant_op(c: u32) -> Rc<Operand> {
    Operand::new_simplified_rc(OperandType::Constant(c as u64))
}

macro_rules! operand_ctx_constants {
    ($($name:ident: $value:expr,)*) => {
        #[derive(Debug)]
        struct OperandCtxConstants {
            small_consts: Vec<Rc<Operand>>,
            $(
                $name: Rc<Operand>,
            )*
        }

        impl OperandCtxConstants {
            fn new() -> OperandCtxConstants {
                OperandCtxConstants {
                    $(
                        $name: make_constant_op($value),
                    )*
                    small_consts: (0..0x41).map(|x| {
                        make_constant_op(x)
                    }).collect(),
                }
            }
        }
    }
}

operand_ctx_constants! {
    c_0: 0x0,
    c_1: 0x1,
    c_2: 0x2,
    c_4: 0x4,
    c_8: 0x8,
    c_1f: 0x1f,
    c_20: 0x20,
    c_7f: 0x7f,
    c_ff: 0xff,
    c_7fff: 0x7fff,
    c_ffff: 0xffff,
    c_ff00: 0xff00,
    c_ffffff00: 0xffff_ff00,
    c_ffff0000: 0xffff_0000,
    c_ffff00ff: 0xffff_00ff,
    c_7fffffff: 0x7fff_ffff,
    c_ffffffff: 0xffff_ffff,
}
thread_local! {
    static OPERAND_CTX_GLOBALS: Rc<OperandCtxGlobals> = Rc::new(OperandCtxGlobals::new());
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

#[inline(never)]
fn make_flag_op(i: usize) -> Rc<Operand> {
    let f = match i {
        0 => Flag::Zero,
        1 => Flag::Carry,
        2 => Flag::Overflow,
        3 => Flag::Parity,
        4 => Flag::Sign,
        5 => Flag::Direction,
        _ => unreachable!(),
    };
    Operand::new_simplified_rc(OperandType::Flag(f))
}

impl OperandCtxGlobals {
    fn new() -> OperandCtxGlobals {
        OperandCtxGlobals {
            constants: OperandCtxConstants::new(),
            flags: array_init::array_init(|i| make_flag_op(i)),
            registers: array_init::array_init(|i| {
                Operand::new_simplified_rc(OperandType::Register(Register(i as u8)))
            })
        }
    }
}

impl OperandContext {
    operand_context_const_methods! {
        const_7f, c_7f,
        const_ff, c_ff,
        const_7fff, c_7fff,
        const_ff00, c_ff00,
        const_ffff, c_ffff,
        const_ffff0000, c_ffff0000,
        const_ffffff00, c_ffffff00,
        const_ffff00ff, c_ffff00ff,
        const_7fffffff, c_7fffffff,
        const_ffffffff, c_ffffffff,
    }

    pub fn flag_z(&self) -> &Rc<Operand> {
        &self.globals.flags[Flag::Zero as usize]
    }

    pub fn flag_c(&self) -> &Rc<Operand> {
        &self.globals.flags[Flag::Carry as usize]
    }

    pub fn flag_o(&self) -> &Rc<Operand> {
        &self.globals.flags[Flag::Overflow as usize]
    }

    pub fn flag_s(&self) -> &Rc<Operand> {
        &self.globals.flags[Flag::Sign as usize]
    }

    pub fn flag_p(&self) -> &Rc<Operand> {
        &self.globals.flags[Flag::Parity as usize]
    }

    pub fn flag_d(&self) -> &Rc<Operand> {
        &self.globals.flags[Flag::Direction as usize]
    }

    pub fn flag(&self, flag: Flag) -> &Rc<Operand> {
        self.flag_by_index(flag as usize)
    }

    pub(crate) fn flag_by_index(&self, index: usize) -> &Rc<Operand> {
        &self.globals.flags[index]
    }

    pub fn register(&self, index: u8) -> Rc<Operand> {
        self.globals.registers[index as usize].clone()
    }

    pub fn register_ref(&self, index: u8) -> &Rc<Operand> {
        &self.globals.registers[index as usize]
    }

    pub fn register_fpu(&self, index: u8) -> Rc<Operand> {
        Operand::new_simplified_rc(OperandType::Fpu(index))
    }

    pub fn xmm(&self, num: u8, word: u8) -> Rc<Operand> {
        Operand::new_simplified_rc(OperandType::Xmm(num, word))
    }

    pub fn const_0(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[0].clone()
    }

    pub fn const_1(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[1].clone()
    }

    pub fn const_2(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[2].clone()
    }

    pub fn const_4(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[4].clone()
    }

    pub fn const_8(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[8].clone()
    }

    pub fn const_1f(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[0x1f].clone()
    }

    pub fn const_20(&self) -> Rc<Operand> {
        self.globals.constants.small_consts[0x20].clone()
    }

    pub fn custom(&self, value: u32) -> Rc<Operand> {
        Operand::new_simplified_rc(OperandType::Custom(value))
    }

    pub fn constant(&self, value: u64) -> Rc<Operand> {
        if value <= u32::max_value() as u64 {
            match value {
                0..=0x40 => {
                    self.globals.constants.small_consts[value as usize].clone()
                }
                0xff => self.const_ff(),
                _ => {
                    if value & 0x7f00 == 0x7f00 {
                        match value {
                            0x7fff => return self.const_7fff(),
                            0xff00 => return self.const_ff00(),
                            0xffff => return self.const_ffff(),
                            0xffff_ff00 => return self.const_ffffff00(),
                            0x7fff_ffff => return self.const_7fffffff(),
                            0xffff_ffff => return self.const_ffffffff(),
                            _ => (),
                        }
                    }
                    Operand::new_simplified_rc(OperandType::Constant(value))
                }
            }
        } else {
            Operand::new_simplified_rc(OperandType::Constant(value))
        }
    }

    pub fn new() -> OperandContext {
        OperandContext {
            next_undefined: Cell::new(0),
            globals: OPERAND_CTX_GLOBALS.with(|x| x.clone()),
        }
    }

    /// Returns first id allocated.
    pub fn alloc_undefined_ids(&self, count: u32) -> u32 {
        let id = self.next_undefined.get();
        // exec_state InternMap relies on this.
        assert!(id < u32::max_value() / 2 - count);
        self.next_undefined.set(id + count);
        id
    }

    pub fn new_undefined_id(&self) -> u32 {
        self.alloc_undefined_ids(1)
    }

    pub fn undefined_rc(&self) -> Rc<Operand> {
        let id = self.new_undefined_id();
        Operand::new_simplified_rc(OperandType::Undefined(UndefinedId(id)))
    }

    /// Returns operand limited to low `size` bits
    pub fn truncate(&self, operand: &Rc<Operand>, size: u8) -> Rc<Operand> {
        let high = 64 - size;
        let mask = !0u64 << high >> high;
        self.and_const(operand, mask)
    }

    pub fn arithmetic(
        &self,
        ty: ArithOpType,
        left: &Rc<Operand>,
        right: &Rc<Operand>,
    ) -> Rc<Operand> {
        let op = Operand::new_not_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty,
            left: left.clone(),
            right: right.clone(),
        }));
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplified_with_ctx(&op, self, &mut simplify)
    }

    pub fn f32_arithmetic(
        &self,
        ty: ArithOpType,
        left: &Rc<Operand>,
        right: &Rc<Operand>,
    ) -> Rc<Operand> {
        let op = Operand::new_not_simplified_rc(OperandType::ArithmeticF32(ArithOperand {
            ty,
            left: left.clone(),
            right: right.clone(),
        }));
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplified_with_ctx(&op, self, &mut simplify)
    }

    /// Returns `Operand` for `left + right`.
    ///
    /// The returned value is simplified.
    pub fn add(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        simplify::simplify_add_sub(left, right, false, self)
    }

    /// Returns `Operand` for `left - right`.
    ///
    /// The returned value is simplified.
    pub fn sub(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        simplify::simplify_add_sub(left, right, true, self)
    }

    /// Returns `Operand` for `left * right`.
    ///
    /// The returned value is simplified.
    pub fn mul(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        simplify::simplify_mul(left, right, self)
    }

    /// Returns `Operand` for signed `left * right`.
    ///
    /// The returned value is simplified.
    pub fn signed_mul(
        &self,
        left: &Rc<Operand>,
        right: &Rc<Operand>,
        _size: MemAccessSize,
    ) -> Rc<Operand> {
        // TODO
        simplify::simplify_mul(left, right, self)
    }

    /// Returns `Operand` for `left / right`.
    ///
    /// The returned value is simplified.
    pub fn div(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        self.arithmetic(ArithOpType::Div, left, right)
    }

    /// Returns `Operand` for `left % right`.
    ///
    /// The returned value is simplified.
    pub fn modulo(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        self.arithmetic(ArithOpType::Modulo, left, right)
    }

    /// Returns `Operand` for `left & right`.
    ///
    /// The returned value is simplified.
    pub fn and(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_and(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    ///
    /// The returned value is simplified.
    pub fn or(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_or(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    ///
    /// The returned value is simplified.
    pub fn xor(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_xor(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    ///
    /// The returned value is simplified.
    pub fn lsh(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    ///
    /// The returned value is simplified.
    pub fn rsh(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_rsh(left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left == right`.
    ///
    /// The returned value is simplified.
    pub fn eq(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        simplify::simplify_eq(left, right, self)
    }

    /// Returns `Operand` for `left != right`.
    ///
    /// The returned value is simplified.
    pub fn neq(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        self.eq_const(&self.eq(left, right), 0)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt(&self, left: &Rc<Operand>, right: &Rc<Operand>) -> Rc<Operand> {
        self.arithmetic(ArithOpType::GreaterThan, left, right)
    }

    /// Returns `Operand` for signed `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt_signed(
        &self,
        left: &Rc<Operand>,
        right: &Rc<Operand>,
        size: MemAccessSize,
    ) -> Rc<Operand> {
        let (mask, offset) = match size {
            MemAccessSize::Mem8 => (0xff, 0x80),
            MemAccessSize::Mem16 => (0xffff, 0x8000),
            MemAccessSize::Mem32 => (0xffff_ffff, 0x8000_0000),
            MemAccessSize::Mem64 => {
                let offset = 0x8000_0000_0000_0000;
                return self.gt(
                    &self.add_const(left, offset),
                    &self.add_const(&right, offset),
                );
            }
        };
        self.gt(
            &self.and_const(
                &self.add_const(&left, offset),
                mask,
            ),
            &self.and_const(
                &self.add_const(&right, offset),
                mask,
            ),
        )
    }

    /// Returns `Operand` for `left + right`.
    ///
    /// The returned value is simplified.
    pub fn add_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        simplify::simplify_add_sub(left, &right, false, self)
    }

    /// Returns `Operand` for `left - right`.
    ///
    /// The returned value is simplified.
    pub fn sub_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        simplify::simplify_add_sub(left, &right, true, self)
    }

    /// Returns `Operand` for `left - right`.
    ///
    /// The returned value is simplified.
    pub fn sub_const_left(&self, left: u64, right: &Rc<Operand>) -> Rc<Operand> {
        let left = self.constant(left);
        simplify::simplify_add_sub(&left, right, true, self)
    }

    /// Returns `Operand` for `left * right`.
    ///
    /// The returned value is simplified.
    pub fn mul_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        simplify::simplify_mul(left, &right, self)
    }

    /// Returns `Operand` for `left & right`.
    ///
    /// The returned value is simplified.
    pub fn and_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_and(left, &right, self, &mut simplify)
    }

    /// Returns `Operand` for `left | right`.
    ///
    /// The returned value is simplified.
    pub fn or_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_or(left, &right, self, &mut simplify)
    }

    /// Returns `Operand` for `left ^ right`.
    ///
    /// The returned value is simplified.
    pub fn xor_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_xor(left, &right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    ///
    /// The returned value is simplified.
    pub fn lsh_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(left, &right, self, &mut simplify)
    }

    /// Returns `Operand` for `left << right`.
    ///
    /// The returned value is simplified.
    pub fn lsh_const_left(&self, left: u64, right: &Rc<Operand>) -> Rc<Operand> {
        let left = self.constant(left);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_lsh(&left, right, self, &mut simplify)
    }

    /// Returns `Operand` for `left >> right`.
    ///
    /// The returned value is simplified.
    pub fn rsh_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplify_rsh(left, &right, self, &mut simplify)
    }

    /// Returns `Operand` for `left == right`.
    ///
    /// The returned value is simplified.
    pub fn eq_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        simplify::simplify_eq(left, &right, self)
    }

    /// Returns `Operand` for `left != right`.
    ///
    /// The returned value is simplified.
    pub fn neq_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        self.eq_const(&self.eq(left, &right), 0)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt_const(&self, left: &Rc<Operand>, right: u64) -> Rc<Operand> {
        let right = self.constant(right);
        self.gt(left, &right)
    }

    /// Returns `Operand` for unsigned `left > right`.
    ///
    /// The returned value is simplified.
    pub fn gt_const_left(&self, left: u64, right: &Rc<Operand>) -> Rc<Operand> {
        let left = self.constant(left);
        self.gt(&left, right)
    }

    pub fn mem64(&self, val: &Rc<Operand>) -> Rc<Operand> {
        self.mem_variable_rc(MemAccessSize::Mem64, val)
    }

    pub fn mem32(&self, val: &Rc<Operand>) -> Rc<Operand> {
        self.mem_variable_rc(MemAccessSize::Mem32, val)
    }

    pub fn mem16(&self, val: &Rc<Operand>) -> Rc<Operand> {
        self.mem_variable_rc(MemAccessSize::Mem16, val)
    }

    pub fn mem8(&self, val: &Rc<Operand>) -> Rc<Operand> {
        self.mem_variable_rc(MemAccessSize::Mem8, val)
    }

    pub fn mem_variable_rc(&self, size: MemAccessSize, val: &Rc<Operand>) -> Rc<Operand> {
        // Eagerly simplify these as the address cannot affect anything
        // this operand would get wrapped to.
        // Only drawback is that if this resulting operand is discarded before
        // it needed to be simplified, the work was wasted.
        // Though that should be a rare case.
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        Operand::new_simplified_rc(OperandType::Memory(MemAccess {
            address: simplify::simplified_with_ctx(val, self, &mut simplify),
            size,
        }))
    }

    pub fn sign_extend(
        &self,
        val: &Rc<Operand>,
        from: MemAccessSize,
        to: MemAccessSize,
    ) -> Rc<Operand> {
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        let val = simplify::simplified_with_ctx(val, self, &mut simplify);
        let op = Operand::new_not_simplified_rc(OperandType::SignExtend(val, from, to));
        let mut simplify = simplify::SimplifyWithZeroBits::default();
        simplify::simplified_with_ctx(&op, self, &mut simplify)
    }

    pub fn transform<F>(&self, oper: &Rc<Operand>, mut f: F) -> Rc<Operand>
    where F: FnMut(&Rc<Operand>) -> Option<Rc<Operand>>
    {
        self.transform_internal(&oper, &mut f)
    }

    fn transform_internal<F>(&self, oper: &Rc<Operand>, f: &mut F) -> Rc<Operand>
    where F: FnMut(&Rc<Operand>) -> Option<Rc<Operand>>
    {
        if let Some(val) = f(&oper) {
            return val;
        }
        match oper.ty {
            OperandType::Arithmetic(ref arith) => {
                let left = self.transform_internal(&arith.left, f);
                let right = self.transform_internal(&arith.right, f);
                if Rc::ptr_eq(&left, &arith.left) && Rc::ptr_eq(&right, &arith.right) {
                    oper.clone()
                } else {
                    self.arithmetic(arith.ty, &left, &right)
                }
            },
            OperandType::Memory(ref m) => {
                let address = self.transform_internal(&m.address, f);
                if Rc::ptr_eq(&address, &m.address) {
                    oper.clone()
                } else {
                    self.mem_variable_rc(m.size, &address)
                }
            }
            _ => oper.clone(),
        }
    }

    pub fn substitute(
        &self,
        oper: &Rc<Operand>,
        val: &Rc<Operand>,
        with: &Rc<Operand>,
    ) -> Rc<Operand> {
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
                            with.clone()
                        } else {
                            let mask = match old.size {
                                MemAccessSize::Mem64 => unreachable!(),
                                MemAccessSize::Mem32 => 0xffff_ffff,
                                MemAccessSize::Mem16 => 0xffff,
                                MemAccessSize::Mem8 => 0xff,
                            };
                            self.and_const(&with, mask)
                        }
                    })
            })
        } else {
            self.transform(oper, |old| match old == val {
                true => Some(with.clone()),
                false => None,
            })
        }
    }
}

impl OperandType {
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
                        arith.left.min_zero_bit_simplify_size,
                        arith.right.min_zero_bit_simplify_size,
                    )
                }
                ArithOpType::Lsh | ArithOpType::Rsh => {
                    let right_bits = match arith.right.if_constant() {
                        Some(s) => 32u64.saturating_sub(s),
                        None => 32,
                    } as u8;
                    arith.left.min_zero_bit_simplify_size.min(right_bits)
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

    /// Returns whether the operand is 8, 16, 32, or 64 bits.
    /// Relevant with signed multiplication, usually operands can be considered
    /// zero-extended u32.
    pub fn expr_size(&self) -> MemAccessSize {
        use self::OperandType::*;
        match *self {
            Memory(ref mem) => mem.size,
            Xmm(..) | Flag(..) | Fpu(..) | ArithmeticF32(..) => MemAccessSize::Mem32,
            Register(..) | Constant(..) | Arithmetic(..) | Undefined(..) |
                Custom(..) => MemAccessSize::Mem64,
            SignExtend(_, _from, to) => to,
        }
    }
}

impl Operand {
    fn new(ty: OperandType, simplified: bool) -> Operand {
        let hash = fxhash::hash64(&ty);
        Operand {
            simplified,
            hash,
            min_zero_bit_simplify_size: ty.min_zero_bit_simplify_size(),
            relevant_bits: ty.calculate_relevant_bits(),
            ty,
        }
    }

    /// Generates operand from bytes, meant to help with fuzzing.
    ///
    /// Does not generate every variation of operands (skips fpu and such).
    ///
    /// TODO May be good to have this generate and-const masks a lot?
    #[cfg(feature = "fuzz")]
    pub fn from_fuzz_bytes(ctx: &OperandContext, bytes: &mut &[u8]) -> Option<Rc<Operand>> {
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
                ctx.mem_variable_rc(size, &inner)
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
                ctx.sign_extend(&inner, from, to);
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
                    0xe => IntToFloat,
                    0xf => FloatToInt,
                    _ => return None,
                };
                ctx.arithmetic(ty, &left, &right)
            }
            _ => return None,
        })
    }

    // TODO: Should not be pub?
    pub(crate) fn new_simplified_rc(ty: OperandType) -> Rc<Operand> {
        Rc::new(Self::new(ty, true))
    }

    pub fn new_not_simplified_rc(ty: OperandType) -> Rc<Operand> {
        Rc::new(Self::new(ty, false))
    }

    pub(crate) fn is_simplified(&self) -> bool {
        self.simplified
    }

    pub fn is_undefined(&self) -> bool {
        match self.ty {
            OperandType::Undefined(_) => true,
            _ => false,
        }
    }

    pub fn iter(&self) -> Iter {
        Iter(Some(IterState {
            pos: self,
            stack: Vec::new(),
        }))
    }

    pub fn iter_no_mem_addr(&self) -> IterNoMemAddr {
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
    pub fn relevant_bits(&self) -> Range<u8> {
        self.relevant_bits.clone()
    }

    pub fn relevant_bits_mask(&self) -> u64 {
        if self.relevant_bits.start >= self.relevant_bits.end {
            0
        } else {
            let low = self.relevant_bits.start;
            let high = 64 - self.relevant_bits.end;
            !0u64 << high >> high >> low << low
        }
    }

    // "Simplify bitwise and: merge child ors"
    // Converts things like [x | const1, x | const2] to [x | (const1 & const2)]
    pub fn const_offset(oper: &Rc<Operand>, ctx: &OperandContext) -> Option<(Rc<Operand>, u64)> {
        // TODO: Investigate if this should be in `recurse`
        if let Some(c) = oper.if_constant() {
            return Some((ctx.const_0(), c));
        }

        fn recurse(oper: &Rc<Operand>) -> Option<u64> {
            // ehhh
            match oper.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
                    if let Some(c) = arith.left.if_constant() {
                        Some(recurse(&arith.right).unwrap_or(0).wrapping_add(c))
                    } else if let Some(c) = arith.right.if_constant() {
                        Some(recurse(&arith.left).unwrap_or(0).wrapping_add(c))
                    } else {
                        if let Some(c) = recurse(&arith.left) {
                            Some(recurse(&arith.right).unwrap_or(0).wrapping_add(c))
                        } else if let Some(c) = recurse(&arith.right) {
                            Some(recurse(&arith.left).unwrap_or(0).wrapping_add(c))
                        } else {
                            None
                        }
                    }

                }
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
                    if let Some(c) = arith.left.if_constant() {
                        Some(c.wrapping_sub(recurse(&arith.right).unwrap_or(0)))
                    } else if let Some(c) = arith.right.if_constant() {
                        Some(recurse(&arith.left).unwrap_or(0).wrapping_sub(c))
                    } else {
                        if let Some(c) = recurse(&arith.left) {
                            Some(c.wrapping_sub(recurse(&arith.right).unwrap_or(0)))
                        } else if let Some(c) = recurse(&arith.right) {
                            Some(recurse(&arith.left).unwrap_or(0).wrapping_sub(c))
                        } else {
                            None
                        }
                    }

                }
                _ => None,
            }
        }
        if let Some(offset) = recurse(oper) {
            let base = ctx.sub_const(oper, offset);
            Some((base, offset))
        } else {
            None
        }
    }

    pub fn simplified(s: Rc<Operand>) -> Rc<Operand> {
        if s.simplified {
            return s;
        }
        let ctx = &OperandContext::new();
        let mut swzb_ctx = simplify::SimplifyWithZeroBits::default();
        simplify::simplified_with_ctx(&s, ctx, &mut swzb_ctx)
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn transform<F>(oper: &Rc<Operand>, mut f: F) -> Rc<Operand>
    where F: FnMut(&Rc<Operand>) -> Option<Rc<Operand>>
    {
        Operand::simplified(Operand::transform_internal(&oper, &mut f))
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn transform_internal<F>(oper: &Rc<Operand>, f: &mut F) -> Rc<Operand>
    where F: FnMut(&Rc<Operand>) -> Option<Rc<Operand>>
    {
        if let Some(val) = f(&oper) {
            return val;
        }
        let sub = |oper: &Rc<Operand>, f: &mut F| Operand::transform_internal(oper, f);
        let ty = match oper.ty {
            OperandType::Arithmetic(ref arith) => OperandType::Arithmetic(ArithOperand {
                ty: arith.ty,
                left: sub(&arith.left, f),
                right: sub(&arith.right, f),
            }),
            OperandType::Memory(ref m) => {
                OperandType::Memory(MemAccess {
                    address: sub(&m.address, f),
                    size: m.size,
                })
            }
            ref x => x.clone(),
        };
        Operand::new_not_simplified_rc(ty)
    }

    #[deprecated]
    #[allow(deprecated)]
    pub fn substitute(oper: &Rc<Operand>, val: &Rc<Operand>, with: &Rc<Operand>) -> Rc<Operand> {
        if let Some(mem) = val.if_memory() {
            // Transform also Mem16[mem.addr] to with & 0xffff if val is Mem32, etc.
            // I guess recursing inside mem.addr doesn't make sense here,
            // but didn't give it too much thought.
            Operand::transform(oper, |old| {
                old.if_memory()
                    .filter(|old| old.address == mem.address)
                    .filter(|old| old.size.bits() <= mem.size.bits())
                    .map(|old| {
                        use crate::operand_helpers::*;
                        if mem.size == old.size || old.size == MemAccessSize::Mem64 {
                            with.clone()
                        } else {
                            let mask = constval(match old.size {
                                MemAccessSize::Mem64 => unreachable!(),
                                MemAccessSize::Mem32 => 0xffff_ffff,
                                MemAccessSize::Mem16 => 0xffff,
                                MemAccessSize::Mem8 => 0xff,
                            });
                            Operand::simplified(operand_and(with.clone(), mask))
                        }
                    })
            })
        } else {
            Operand::transform(oper, |old| match old == val {
                true => Some(with.clone()),
                false => None,
            })
        }
    }

    /// Returns `Some(c)` if `self.ty` is `OperandType::Constant(c)`
    pub fn if_constant(&self) -> Option<u64> {
        match self.ty {
            OperandType::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(r)` if `self.ty` is `OperandType::Register(r)`
    pub fn if_register(&self) -> Option<Register> {
        match self.ty {
            OperandType::Register(r) => Some(r),
            _ => None,
        }
    }

    /// Returns `Some(mem)` if `self.ty` is `OperandType::Memory(ref mem)`
    pub fn if_memory(&self) -> Option<&MemAccess> {
        match self.ty {
            OperandType::Memory(ref mem) => Some(mem),
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem64`
    pub fn if_mem64(&self) -> Option<&Rc<Operand>> {
        match self.ty {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem64 {
                true => Some(&mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem32`
    pub fn if_mem32(&self) -> Option<&Rc<Operand>> {
        match self.ty {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem32 {
                true => Some(&mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem16`
    pub fn if_mem16(&self) -> Option<&Rc<Operand>> {
        match self.ty {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem16 {
                true => Some(&mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some(mem.addr)` if `self.ty` is `OperandType::Memory(ref mem)` and
    /// `mem.size == MemAccessSize::Mem8`
    pub fn if_mem8(&self) -> Option<&Rc<Operand>> {
        match self.ty {
            OperandType::Memory(ref mem) => match mem.size == MemAccessSize::Mem8 {
                true => Some(&mem.address),
                false => None,
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if self.ty is `OperandType::Arithmetic { ty == ty }`
    pub fn if_arithmetic(
        &self,
        ty: ArithOpType,
    ) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        match self.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ty => {
                Some((&arith.left, &arith.right))
            }
            _ => None,
        }
    }

    /// Returns `true` if self.ty is `OperandType::Arithmetic { ty == ty }`
    pub fn is_arithmetic(
        &self,
        ty: ArithOpType,
    ) -> bool {
        match self.ty {
            OperandType::Arithmetic(ref arith) => arith.ty == ty,
            _ => false,
        }
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Add(left, right))`
    pub fn if_arithmetic_add(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::Add)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Sub(left, right))`
    pub fn if_arithmetic_sub(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::Sub)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Mul(left, right))`
    pub fn if_arithmetic_mul(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::Mul)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Equal(left, right))`
    pub fn if_arithmetic_eq(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::Equal)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::GreaterThan(left, right))`
    pub fn if_arithmetic_gt(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::GreaterThan)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::And(left, right))`
    pub fn if_arithmetic_and(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::And)
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Or(left, right))`
    pub fn if_arithmetic_or(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        self.if_arithmetic(ArithOpType::Or)
    }

    /// Returns `Some((register, constant))` if operand is an and mask of register
    /// with constant.
    ///
    /// Useful for detecting 32-bit register which is represented as `Register(r) & ffff_ffff`.
    pub fn if_and_masked_register(&self) -> Option<(Register, u64)> {
        let (l, r) = self.if_arithmetic_and()?;
        let (reg, other) = Operand::either(l, r, |x| x.if_register())?;
        let other = other.if_constant()?;
        Some((reg, other))
    }

    /// Returns `(other, constant)` if operand is an and mask with constant,
    /// or just (self, u64::max_value())
    pub fn and_masked(this: &Rc<Operand>) -> (&Rc<Operand>, u64) {
        this.if_arithmetic_and()
            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
            .map(|(c, o)| (o, c))
            .unwrap_or_else(|| (this, u64::max_value()))
    }

    /// If either of `a` or `b` matches the filter-map `f`, return the mapped result and the other
    /// operand.
    pub fn either<'a, F, T>(
        a: &'a Rc<Operand>,
        b: &'a Rc<Operand>,
        mut f: F,
    ) -> Option<(T, &'a Rc<Operand>)>
    where F: FnMut(&'a Rc<Operand>) -> Option<T>
    {
        f(a).map(|val| (val, b)).or_else(|| f(b).map(|val| (val, a)))
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct MemAccess {
    pub address: Rc<Operand>,
    pub size: MemAccessSize,
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Eq, PartialEq, Copy, Debug, Hash, Ord, PartialOrd)]
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

pub mod operand_helpers {
    use std::rc::Rc;

    use super::ArithOpType::*;
    use super::MemAccessSize::*;
    use super::{
        ArithOpType, ArithOperand, MemAccess, MemAccessSize, Operand, OperandContext,
        OperandType,
    };

    pub fn operand_register(num: u8) -> Rc<Operand> {
        OperandContext::new().register(num)
    }

    pub fn operand_arith(ty: ArithOpType, lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        Operand::new_not_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty,
            left: lhs,
            right: rhs,
        }))
    }

    pub fn operand_arith_f32(ty: ArithOpType, lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        Operand::new_not_simplified_rc(OperandType::ArithmeticF32(ArithOperand {
            ty,
            left: lhs,
            right: rhs,
        }))
    }

    pub fn operand_add(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Add, lhs, rhs)
    }

    pub fn operand_sub(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Sub, lhs, rhs)
    }

    pub fn operand_mul(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Mul, lhs, rhs)
    }

    pub fn operand_div(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Div, lhs, rhs)
    }

    pub fn operand_mod(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Modulo, lhs, rhs)
    }

    pub fn operand_and(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(And, lhs, rhs)
    }

    pub fn operand_eq(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Equal, lhs, rhs)
    }

    pub fn operand_ne(ctx: &OperandContext, lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_eq(operand_eq(lhs, rhs), ctx.const_0())
    }

    pub fn operand_gt(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(GreaterThan, lhs, rhs)
    }

    pub fn operand_or(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Or, lhs, rhs)
    }

    pub fn operand_xor(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Xor, lhs, rhs)
    }

    pub fn operand_lsh(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Lsh, lhs, rhs)
    }

    pub fn operand_rsh(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Rsh, lhs, rhs)
    }

    pub fn mem64(val: Rc<Operand>) -> Rc<Operand> {
        mem_variable_rc(Mem64, val)
    }

    pub fn mem32(val: Rc<Operand>) -> Rc<Operand> {
        mem_variable_rc(Mem32, val)
    }

    pub fn mem16(val: Rc<Operand>) -> Rc<Operand> {
        mem_variable_rc(Mem16, val)
    }

    pub fn mem8(val: Rc<Operand>) -> Rc<Operand> {
        mem_variable_rc(Mem8, val)
    }

    pub fn mem_variable_rc(size: MemAccessSize, val: Rc<Operand>) -> Rc<Operand> {
        // Eagerly simplify these as the address cannot affect anything
        // this operand would get wrapped to.
        // Only drawback is that if this resulting operand is discarded before
        // it needed to be simplified, the work was wasted.
        // Though that should be a rare case.
        Operand::new_simplified_rc(OperandType::Memory(MemAccess {
            address: Operand::simplified(val),
            size,
        }))
    }

    pub fn constval(num: u64) -> Rc<Operand> {
        OperandContext::new().constant(num)
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn operand_iter() {
        use std::collections::HashSet;

        let ctx = super::OperandContext::new();
        let oper = ctx.and(
            &ctx.sub(
                &ctx.constant(1),
                &ctx.register(6),
            ),
            &ctx.eq(
                &ctx.constant(77),
                &ctx.register(4),
            ),
        );
        let opers = [
            oper.clone(),
            ctx.sub(&ctx.constant(1), &ctx.register(6)),
            ctx.constant(1),
            ctx.register(6),
            ctx.eq(&ctx.constant(77), &ctx.register(4)),
            ctx.constant(77),
            ctx.register(4),
        ];
        let mut seen = HashSet::new();
        for o in oper.iter() {
            assert!(!seen.contains(o));
            seen.insert(o);
        }
        for o in &opers {
            assert!(seen.contains(&**o), "Didn't find {}", o);
        }
        assert_eq!(seen.len(), opers.len());
    }
}
