use std::cell::Cell;
use std::cmp::{max, min, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Range;
use std::rc::Rc;

use serde::{Deserialize as DeserializeTrait, Deserializer};
use serde_derive::{Deserialize, Serialize};

use crate::bit_misc::{bits_overlap, one_bit_ranges, zero_bit_ranges};
use crate::heapsort;
use crate::vec_drop_iter::VecDropIter;

#[derive(Clone, Eq, Serialize)]
pub struct Operand {
    pub ty: OperandType,
    #[serde(skip_serializing)]
    simplified: bool,
    #[serde(skip_serializing)]
    hash: u64,
    #[serde(skip_serializing)]
    min_zero_bit_simplify_size: u8,
    #[serde(skip_serializing)]
    relevant_bits: Range<u8>,
}

impl<'de> DeserializeTrait<'de> for Operand {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Operand, D::Error> {
        use serde::de::{self, MapAccess, SeqAccess, Visitor};

        const FIELDS: &[&str] = &["ty"];
        enum Field {
            Ty,
        }
        impl<'de> DeserializeTrait<'de> for Field {
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

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
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

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct ArithOperand {
    pub ty: ArithOpType,
    pub left: Rc<Operand>,
    pub right: Rc<Operand>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
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


#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
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
    flag_z: Rc<Operand>,
    flag_c: Rc<Operand>,
    flag_o: Rc<Operand>,
    flag_s: Rc<Operand>,
    flag_p: Rc<Operand>,
    flag_d: Rc<Operand>,
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
                        $name: Operand::new_simplified_rc(OperandType::Constant($value)),
                    )*
                    small_consts: (0..0x41).map(|x| {
                        Operand::new_simplified_rc(OperandType::Constant(x))
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

impl OperandCtxGlobals {
    fn new() -> OperandCtxGlobals {
        OperandCtxGlobals {
            constants: OperandCtxConstants::new(),
            flag_c: Operand::new_simplified_rc(OperandType::Flag(Flag::Carry)),
            flag_o: Operand::new_simplified_rc(OperandType::Flag(Flag::Overflow)),
            flag_p: Operand::new_simplified_rc(OperandType::Flag(Flag::Parity)),
            flag_z: Operand::new_simplified_rc(OperandType::Flag(Flag::Zero)),
            flag_s: Operand::new_simplified_rc(OperandType::Flag(Flag::Sign)),
            flag_d: Operand::new_simplified_rc(OperandType::Flag(Flag::Direction)),
            registers: [
                Operand::new_simplified_rc(OperandType::Register(Register(0))),
                Operand::new_simplified_rc(OperandType::Register(Register(1))),
                Operand::new_simplified_rc(OperandType::Register(Register(2))),
                Operand::new_simplified_rc(OperandType::Register(Register(3))),
                Operand::new_simplified_rc(OperandType::Register(Register(4))),
                Operand::new_simplified_rc(OperandType::Register(Register(5))),
                Operand::new_simplified_rc(OperandType::Register(Register(6))),
                Operand::new_simplified_rc(OperandType::Register(Register(7))),
                Operand::new_simplified_rc(OperandType::Register(Register(8))),
                Operand::new_simplified_rc(OperandType::Register(Register(9))),
                Operand::new_simplified_rc(OperandType::Register(Register(10))),
                Operand::new_simplified_rc(OperandType::Register(Register(11))),
                Operand::new_simplified_rc(OperandType::Register(Register(12))),
                Operand::new_simplified_rc(OperandType::Register(Register(13))),
                Operand::new_simplified_rc(OperandType::Register(Register(14))),
                Operand::new_simplified_rc(OperandType::Register(Register(15))),
            ],
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

    pub fn flag_z(&self) -> Rc<Operand> {
        self.globals.flag_z.clone()
    }

    pub fn flag_c(&self) -> Rc<Operand> {
        self.globals.flag_c.clone()
    }

    pub fn flag_o(&self) -> Rc<Operand> {
        self.globals.flag_o.clone()
    }

    pub fn flag_s(&self) -> Rc<Operand> {
        self.globals.flag_s.clone()
    }

    pub fn flag_p(&self) -> Rc<Operand> {
        self.globals.flag_p.clone()
    }

    pub fn flag_d(&self) -> Rc<Operand> {
        self.globals.flag_d.clone()
    }

    pub fn flag(&self, flag: Flag) -> Rc<Operand> {
        match flag {
            Flag::Zero => self.flag_z(),
            Flag::Carry => self.flag_c(),
            Flag::Overflow => self.flag_o(),
            Flag::Parity => self.flag_p(),
            Flag::Sign => self.flag_s(),
            Flag::Direction => self.flag_d(),
        }
    }

    pub fn register(&self, index: u8) -> Rc<Operand> {
        self.globals.registers[index as usize].clone()
    }

    pub fn register_fpu(&self, index: u8) -> Rc<Operand> {
        Operand::new_simplified_rc(OperandType::Fpu(index))
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
    pub fn truncate(&self, operand: Rc<Operand>, size: u8) -> Rc<Operand> {
        use self::operand_helpers::*;
        if operand.relevant_bits().end <= size {
            operand
        } else {
            let low = operand.relevant_bits().start;
            let high = 64 - size;
            let mask = !0u64 << high >> high >> low << low;
            Operand::simplified(operand_and(
                operand,
                self.constant(mask),
            ))
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
    pub fn from_fuzz_bytes(bytes: &mut &[u8]) -> Option<Rc<Operand>> {
        use self::operand_helpers::*;
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
            0x0 => operand_register(read_u8(bytes)? & 0xf),
            0x1 => operand_xmm(read_u8(bytes)? & 0xf, read_u8(bytes)? & 0x3),
            0x2 => constval(read_u64(bytes)?),
            0x3 => {
                let size = match read_u8(bytes)? & 3 {
                    0 => MemAccessSize::Mem8,
                    1 => MemAccessSize::Mem16,
                    2 => MemAccessSize::Mem32,
                    _ => MemAccessSize::Mem64,
                };
                let inner = Operand::from_fuzz_bytes(bytes)?;
                mem_variable_rc(size, inner)
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
                let inner = Operand::from_fuzz_bytes(bytes)?;
                Operand::new_not_simplified_rc(
                    OperandType::SignExtend(inner, from, to),
                )
            }
            0x5 => {
                use self::ArithOpType::*;
                let left = Operand::from_fuzz_bytes(bytes)?;
                let right = Operand::from_fuzz_bytes(bytes)?;
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
                operand_arith(ty, left, right)
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

    pub fn to_xmm_32(s: &Rc<Operand>, word: u8) -> Rc<Operand> {
        use self::operand_helpers::*;
        match s.ty {
            OperandType::Memory(ref mem) => match u64::from(word) {
                0 => s.clone(),
                x => mem32(operand_add(mem.address.clone(), constval(4 * x))),
            },
            OperandType::Register(reg) => {
                Operand::new_simplified_rc(OperandType::Xmm(reg.0, word))
            }
            _ => s.clone(),
        }
    }

    /// Return (low, high)
    pub fn to_xmm_64(s: &Rc<Operand>, word: u8) -> (Rc<Operand>, Rc<Operand>) {
        use self::operand_helpers::*;
        match s.ty {
            OperandType::Memory(ref mem) => match u64::from(word) {
                0 => {
                    let high = operand_add(mem.address.clone(), constval(4));
                    (s.clone(), mem32(high))
                }
                x => {
                    let low = operand_add(mem.address.clone(), constval(8 * x));
                    let high = operand_add(mem.address.clone(), constval(8 * x + 4));
                    (mem32(low), mem32(high))
                }
            },
            OperandType::Register(reg) => {
                let low = operand_xmm(reg.0, word * 2);
                let high = operand_xmm(reg.0, word * 2 + 1);
                (low, high)
            }
            _ => {
                if let Some((reg, _)) = s.if_and_masked_register() {
                    let low = operand_xmm(reg.0, word * 2);
                    let high = operand_xmm(reg.0, word * 2 + 1);
                    (low, high)
                } else {
                    #[cfg(debug_assertions)]
                    panic!("Cannot convert {} to 64-bit xmm", s);
                    #[cfg(not(debug_assertions))]
                    panic!("Cannot convert to 64-bit xmm");
                }
            }
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

    fn collect_add_ops(
        s: &Rc<Operand>,
        ops: &mut Vec<(Rc<Operand>, bool)>,
        out_mask: u64,
        negate: bool,
    ) {
        fn recurse(
            s: &Rc<Operand>,
            ops: &mut Vec<(Rc<Operand>, bool)>,
            out_mask: u64,
            negate: bool,
        )  {
            match s.ty {
                OperandType::Arithmetic(ref arith) if {
                    arith.ty == ArithOpType::Add || arith.ty== ArithOpType::Sub
                } => {
                    recurse(&arith.left, ops, out_mask, negate);
                    let negate_right = match arith.ty {
                        ArithOpType::Add => negate,
                        _ => !negate,
                    };
                    recurse(&arith.right, ops, out_mask, negate_right);
                }
                _ => {
                    let mut s = s.clone();
                    if !s.is_simplified() {
                        // Simplification can cause it to be an add
                        s = Operand::simplified(s);
                        if let OperandType::Arithmetic(ref arith) = s.ty {
                            if arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub {
                                recurse(&s, ops, out_mask, negate);
                                return;
                            }
                        }
                    }
                    if let Some((l, r)) = s.if_arithmetic_and() {
                        let const_other = Operand::either(l, r, |x| x.if_constant());
                        if let Some((c, other)) = const_other {
                            if c & out_mask == out_mask {
                                recurse(other, ops, out_mask, negate);
                                return;
                            }
                        }
                    }
                    ops.push((s, negate));
                }
            }
        }
        recurse(s, ops, out_mask, negate)
    }

    /// Unwraps a tree chaining arith operation to vector of the operands.
    ///
    /// Simplifies operands in process.
    ///
    /// If the limit is set, caller should verify that it was not hit (ops.len() > limit),
    /// as not all ops will end up being collected (TODO Probs should return result)
    fn collect_arith_ops(
        s: &Rc<Operand>,
        ops: &mut Vec<Rc<Operand>>,
        arith_type: ArithOpType,
        limit: usize,
        mut ctx_swzb: Option<(&OperandContext, &mut SimplifyWithZeroBits)>,
    ) {
        if ops.len() >= limit {
            if ops.len() == limit {
                ops.push(s.clone());
                #[cfg(feature = "fuzz")]
                tls_simplification_incomplete();
            }
            return;
        }
        match s.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == arith_type => {
                let ctx_swzb_ = ctx_swzb.as_mut().map(|x| (x.0, &mut *x.1));
                Operand::collect_arith_ops(&arith.left, ops, arith_type, limit, ctx_swzb_);
                Operand::collect_arith_ops(&arith.right, ops, arith_type, limit, ctx_swzb);
            }
            _ => {
                let mut s = s.clone();
                if !s.is_simplified() {
                    // Simplification can cause it to be what is being collected
                    s = match ctx_swzb {
                        Some((ctx, ref mut swzb)) => {
                            Operand::simplified_with_ctx(s, ctx, &mut *swzb)
                        }
                        None => Operand::simplified(s),
                    };
                    if let OperandType::Arithmetic(ref arith) = s.ty {
                        if arith.ty == arith_type {
                            Operand::collect_arith_ops(&s, ops, arith_type, limit, ctx_swzb);
                            return;
                        }
                    }
                }
                ops.push(s);
            }
        }
    }

    fn collect_mul_ops(s: &Rc<Operand>, ops: &mut Vec<Rc<Operand>>) {
        Operand::collect_arith_ops(s, ops, ArithOpType::Mul, usize::max_value(), None);
    }

    fn collect_and_ops(
        s: &Rc<Operand>,
        ops: &mut Vec<Rc<Operand>>,
        limit: usize,
        ctx: &OperandContext,
        swzb: &mut SimplifyWithZeroBits,
    ) {
        Operand::collect_arith_ops(s, ops, ArithOpType::And, limit, Some((ctx, swzb)));
    }

    fn collect_or_ops(
        s: &Rc<Operand>,
        ops: &mut Vec<Rc<Operand>>,
        ctx: &OperandContext,
        swzb: &mut SimplifyWithZeroBits,
    ) {
        Operand::collect_arith_ops(s, ops, ArithOpType::Or, usize::max_value(), Some((ctx, swzb)));
    }

    fn collect_xor_ops(
        s: &Rc<Operand>,
        ops: &mut Vec<Rc<Operand>>,
        limit: usize,
        ctx: &OperandContext,
        swzb: &mut SimplifyWithZeroBits,
    ) {
        Operand::collect_arith_ops(s, ops, ArithOpType::Xor, limit, Some((ctx, swzb)));
    }

    // "Simplify bitwise and: merge child ors"
    // Converts things like [x | const1, x | const2] to [x | (const1 & const2)]
    fn simplify_and_merge_child_ors(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) {
        use self::operand_helpers::*;
        fn or_const(op: &Rc<Operand>) -> Option<(u64, &Rc<Operand>)> {
            match op.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Or => {
                    Operand::either(&arith.left, &arith.right, |x| x.if_constant())
                }
                _ => None,
            }
        }

        let mut iter = VecDropIter::new(ops);
        while let Some(mut op) = iter.next() {
            let mut new = None;
            if let Some((mut constant, val)) = or_const(&op) {
                let mut second = iter.duplicate();
                while let Some(other_op) = second.next_removable() {
                    let mut remove = false;
                    if let Some((other_constant, other_val)) = or_const(&other_op) {
                        if other_val == val {
                            constant &= other_constant;
                            remove = true;
                        }
                    }
                    if remove {
                        other_op.remove();
                    }
                }
                let constant = ctx.constant(constant);
                let oper = operand_or(val.clone(), constant);
                new = Some(Operand::simplified(oper));
            }
            if let Some(new) = new {
                *op = new;
            }
        }
    }

    // "Simplify bitwise or: xor merge"
    // Converts [x, y] to [x ^ y] where x and y don't have overlapping
    // relevant bit ranges. Then ideally the xor can simplify further.
    // Technically valid for any non-overlapping x and y, but limit transformation
    // to cases where x and y are xors.
    fn simplify_or_merge_xors(
        ops: &mut Vec<Rc<Operand>>,
        ctx: &OperandContext,
        swzb: &mut SimplifyWithZeroBits,
    ) {
        fn is_xor(op: &Rc<Operand>) -> bool {
            match op.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Xor => true,
                _ => false,
            }
        }

        let mut iter = VecDropIter::new(ops);
        while let Some(mut op) = iter.next() {
            let mut new = None;
            if is_xor(&op) {
                let mut second = iter.duplicate();
                let bits = op.relevant_bits();
                while let Some(other_op) = second.next_removable() {
                    if is_xor(&other_op) {
                        let other_bits = other_op.relevant_bits();
                        if !bits_overlap(&bits, &other_bits) {
                            new = Some(simplify_xor(&op, &other_op, ctx, swzb));
                            other_op.remove();
                        }
                    }
                }
            }
            if let Some(new) = new {
                *op = new;
            }
        }
    }

    /// "Simplify bitwise or: merge child ands"
    /// Converts things like [x & const1, x & const2] to [x & (const1 | const2)]
    ///
    /// Also used by xors with only_nonoverlapping true
    fn simplify_or_merge_child_ands(
        ops: &mut Vec<Rc<Operand>>,
        ctx: &OperandContext,
        swzb_ctx: &mut SimplifyWithZeroBits,
        only_nonoverlapping: bool,
    ) {
        fn and_const(op: &Rc<Operand>) -> Option<(u64, &Rc<Operand>)> {
            match op.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                    Operand::either(&arith.left, &arith.right, |x| x.if_constant())
                }
                OperandType::Memory(ref mem) => match mem.size {
                    MemAccessSize::Mem8 => Some((0xff, op)),
                    MemAccessSize::Mem16 => Some((0xffff, op)),
                    MemAccessSize::Mem32 => Some((0xffff_ffff, op)),
                    MemAccessSize::Mem64 => Some((0xffff_ffff_ffff_ffff, op)),
                }
                _ => {
                    let bits = op.relevant_bits();
                    if bits != (0..64) && bits.start < bits.end {
                        let low = bits.start;
                        let high = 64 - bits.end;
                        Some((!0 >> low << low << high >> high, op))
                    } else {
                        None
                    }
                }
            }
        }

        if ops.len() > 16 {
            // The loop below is quadratic complexity, being especially bad
            // if there are lot of masked xors, so give up if there are more
            // ops than usual code would have.
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
            return;
        }

        let mut iter = VecDropIter::new(ops);
        while let Some(mut op) = iter.next() {
            let mut new = None;
            if let Some((mut constant, val)) = and_const(&op) {
                let mut second = iter.duplicate();
                let mut new_val = val.clone();
                while let Some(other_op) = second.next_removable() {
                    let mut remove = false;
                    if let Some((other_constant, other_val)) = and_const(&other_op) {
                        let result = if only_nonoverlapping && other_constant & constant != 0 {
                            None
                        } else {
                            try_merge_ands(other_val, val, other_constant, constant, ctx, swzb_ctx)
                        };
                        if let Some(merged) = result {
                            constant |= other_constant;
                            new_val = merged;
                            remove = true;
                        }
                    }
                    if remove {
                        other_op.remove();
                    }
                }
                new = Some(simplify_and(&new_val, &ctx.constant(constant), ctx, swzb_ctx));
            }
            if let Some(new) = new {
                *op = new;
            }
        }
    }

    // Simplify or: merge comparisions
    // Converts
    // (c > x) | (c == x) to (c + 1 > x),
    // (x > c) | (x == c) to (x > c + 1).
    // Cannot do for values that can overflow, so just limit it to constants for now.
    // (Well, could do (c + 1 > x) | (c == max_value), but that isn't really simpler)
    fn simplify_or_merge_comparisions(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) {
        use self::operand_helpers::*;

        #[derive(Eq, PartialEq, Copy, Clone)]
        enum MatchType {
            ConstantGreater,
            ConstantLess,
            Equal,
        }

        fn check_match(op: &Rc<Operand>) -> Option<(u64, &Rc<Operand>, MatchType)> {
            match op.ty {
                OperandType::Arithmetic(ref arith) => {
                    let left = &arith.left;
                    let right = &arith.right;
                    match arith.ty {
                        ArithOpType::Equal => {
                            let (c, other) = Operand::either(left, right, |x| x.if_constant())?;
                            return Some((c, other, MatchType::Equal));
                        }
                        ArithOpType::GreaterThan => {
                            if let Some(c) = left.if_constant() {
                                return Some((c, right, MatchType::ConstantGreater));
                            }
                            if let Some(c) = right.if_constant() {
                                return Some((c, left, MatchType::ConstantLess));
                            }
                        }
                        _ => (),
                    }
                }
                _ => (),
            }
            None
        }

        let mut iter = VecDropIter::new(ops);
        while let Some(mut op) = iter.next() {
            let mut new = None;
            if let Some((c, x, ty)) = check_match(&op) {
                let mut second = iter.duplicate();
                while let Some(other_op) = second.next_removable() {
                    let mut remove = false;
                    if let Some((c2, x2, ty2)) = check_match(&other_op) {
                        if c == c2 && x == x2 {
                            match (ty, ty2) {
                                (MatchType::ConstantGreater, MatchType::Equal) |
                                    (MatchType::Equal, MatchType::ConstantGreater) =>
                                {
                                    // min/max edge cases can be handled by gt simplification,
                                    // don't do them here.
                                    if let Some(new_c) = c.checked_add(1) {
                                        let merged = operand_gt(ctx.constant(new_c), x.clone());
                                        new = Some(Operand::simplified(merged));
                                        remove = true;
                                    }
                                }
                                (MatchType::ConstantLess, MatchType::Equal) |
                                    (MatchType::Equal, MatchType::ConstantLess) =>
                                {
                                    if let Some(new_c) = c.checked_sub(1) {
                                        let merged = operand_gt(x.clone(), ctx.constant(new_c));
                                        new = Some(Operand::simplified(merged));
                                        remove = true;
                                    }
                                }
                                _ => (),
                            }
                        }
                    }
                    if remove {
                        other_op.remove();
                        break;
                    }
                }
            }
            if let Some(new) = new {
                *op = new;
            }
        }
    }

    pub fn const_offset(oper: &Rc<Operand>, ctx: &OperandContext) -> Option<(Rc<Operand>, u64)> {
        use crate::operand::operand_helpers::*;

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
            let base = Operand::simplified(operand_sub(oper.clone(), ctx.constant(offset)));
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
        let mut swzb_ctx = SimplifyWithZeroBits::default();
        Operand::simplified_with_ctx(s, ctx, &mut swzb_ctx)
    }

    fn simplified_with_ctx(
        s: Rc<Operand>,
        ctx: &OperandContext,
        swzb_ctx: &mut SimplifyWithZeroBits,
    ) -> Rc<Operand> {
        use crate::operand_helpers::*;
        if s.simplified {
            return s;
        }
        let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
        match s.clone().ty {
            OperandType::Arithmetic(ref arith) => {
                let left = &arith.left;
                let right = &arith.right;
                match arith.ty {
                    ArithOpType::Add | ArithOpType::Sub => {
                        let is_sub = arith.ty == ArithOpType::Sub;
                        simplify_add_sub(left, right, is_sub, ctx)
                    }
                    ArithOpType::Mul => simplify_mul(left, right, ctx),
                    ArithOpType::And => simplify_and(left, right, ctx, swzb_ctx),
                    ArithOpType::Or => simplify_or(left, right, ctx, swzb_ctx),
                    ArithOpType::Xor => simplify_xor(left, right, ctx, swzb_ctx),
                    ArithOpType::Lsh => simplify_lsh(left, right, ctx, swzb_ctx),
                    ArithOpType::Rsh => simplify_rsh(left, right, ctx, swzb_ctx),
                    ArithOpType::Equal => simplify_eq(left, right, ctx),
                    ArithOpType::GreaterThan => {
                        let mut left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
                        if left == right {
                            return ctx.const_0();
                        }
                        let (left_inner, mask) = Operand::and_masked(&left);
                        let (right_inner, mask2) = Operand::and_masked(&right);
                        // Can simplify x - y > x to y > x if mask starts from bit 0
                        let mask_is_continuous_from_0 = mask.wrapping_add(1) & mask == 0;
                        if mask == mask2 && mask_is_continuous_from_0 {
                            // TODO collect_add_ops would be more complete
                            if let OperandType::Arithmetic(ref arith) = left_inner.ty {
                                if arith.ty == ArithOpType::Sub {
                                    if arith.left == *right_inner {
                                        left = if mask == u64::max_value() {
                                            arith.right.clone()
                                        } else {
                                            let c = ctx.constant(mask);
                                            simplify_and(&arith.right, &c, ctx, swzb_ctx)
                                        };
                                    }
                                }
                            }
                        }
                        match (left.if_constant(), right.if_constant()) {
                            (Some(a), Some(b)) => match a > b {
                                true => return ctx.const_1(),
                                false => return ctx.const_0(),
                            },
                            (Some(c), None) => {
                                if c == 0 {
                                    return ctx.const_0();
                                }
                                // max > x if x != max
                                let relbit_mask = right.relevant_bits_mask();
                                if c == relbit_mask {
                                    let op = operand_ne(ctx, left, right);
                                    return Operand::simplified_with_ctx(op, ctx, swzb_ctx)
                                }
                            }
                            (None, Some(c)) => {
                                // x > 0 if x != 0
                                if c == 0 {
                                    let op = operand_ne(ctx, left, right);
                                    return Operand::simplified_with_ctx(op, ctx, swzb_ctx)
                                }
                                let relbit_mask = left.relevant_bits_mask();
                                if c == relbit_mask {
                                    return ctx.const_0();
                                }
                            }
                            _ => (),
                        }
                        let arith = ArithOperand {
                            ty: ArithOpType::GreaterThan,
                            left,
                            right,
                        };
                        Operand::new_simplified_rc(OperandType::Arithmetic(arith))
                    }
                    ArithOpType::Div | ArithOpType::Modulo => {
                        let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
                        if let Some(r) = right.if_constant() {
                            if r == 0 {
                                // Use 0 / 0 for any div by zero
                                let arith = ArithOperand {
                                    ty: arith.ty,
                                    left: right.clone(),
                                    right,
                                };
                                return Operand::new_simplified_rc(OperandType::Arithmetic(arith));
                            }
                            if arith.ty == ArithOpType::Modulo {
                                // If x % y == x if y > x
                                if r > left.relevant_bits_mask() {
                                    return left;
                                }
                                if let Some(l) = left.if_constant() {
                                    return ctx.constant(l % r);
                                }
                            } else {
                                // Div, x / y == 0 if y > x
                                if r > left.relevant_bits_mask() {
                                    return ctx.const_0();
                                }
                                if let Some(l) = left.if_constant() {
                                    return ctx.constant(l / r);
                                }
                                // x / 1 == x
                                if r == 1 {
                                    return left;
                                }
                            }
                        }
                        if left.if_constant() == Some(0) {
                            return ctx.const_0();
                        }
                        if left == right {
                            // x % x == 0, x / x = 1
                            if arith.ty == ArithOpType::Modulo {
                                return ctx.const_0();
                            } else {
                                return ctx.const_1();
                            }
                        }
                        let arith = ArithOperand {
                            ty: arith.ty,
                            left,
                            right,
                        };
                        Operand::new_simplified_rc(OperandType::Arithmetic(arith))
                    }
                    ArithOpType::Parity => {
                        let val = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        if let Some(c) = val.if_constant() {
                            return match (c as u8).count_ones() & 1 == 0 {
                                true => ctx.const_1(),
                                false => ctx.const_0(),
                            }
                        } else {
                            let ty = OperandType::Arithmetic(ArithOperand {
                                ty: ArithOpType::Parity,
                                left: val,
                                right: ctx.const_0(),
                            });
                            Operand::new_simplified_rc(ty)
                        }
                    }
                    ArithOpType::FloatToInt => {
                        let val = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        if let Some(c) = val.if_constant() {
                            use byteorder::{ReadBytesExt, WriteBytesExt, LE};
                            let mut buf = [0; 4];
                            (&mut buf[..]).write_u32::<LE>(c as u32).unwrap();
                            let float = (&buf[..]).read_f32::<LE>().unwrap();
                            let overflow = float > i32::max_value() as f32 ||
                                float < i32::min_value() as f32;
                            let int = if overflow {
                                0x8000_0000
                            } else {
                                float as i32 as u32
                            };
                            ctx.constant(int as u64)
                        } else {
                            let ty = OperandType::Arithmetic(ArithOperand {
                                ty: arith.ty,
                                left: val,
                                right: ctx.const_0(),
                            });
                            Operand::new_simplified_rc(ty)
                        }
                    }
                    ArithOpType::IntToFloat => {
                        let val = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        if let Some(c) = val.if_constant() {
                            use byteorder::{ReadBytesExt, WriteBytesExt, LE};
                            let mut buf = [0; 4];
                            (&mut buf[..]).write_f32::<LE>(c as i32 as f32).unwrap();
                            let float = (&buf[..]).read_u32::<LE>().unwrap();
                            ctx.constant(float as u64)
                        } else {
                            let ty = OperandType::Arithmetic(ArithOperand {
                                ty: arith.ty,
                                left: val,
                                right: ctx.const_0(),
                            });
                            Operand::new_simplified_rc(ty)
                        }
                    }
                    _ => {
                        let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
                        let ty = OperandType::Arithmetic(ArithOperand {
                            ty: arith.ty,
                            left,
                            right,
                        });
                        Operand::new_simplified_rc(ty)
                    }
                }
            }
            OperandType::Memory(ref mem) => {
                Operand::new_simplified_rc(OperandType::Memory(MemAccess {
                    address: Operand::simplified_with_ctx(mem.address.clone(), ctx, swzb_ctx),
                    size: mem.size,
                }))
            }
            OperandType::SignExtend(ref val, from, to) => {
                if from.bits() >= to.bits() {
                    return ctx.const_0();
                }
                let val = Operand::simplified_with_ctx(val.clone(), ctx, swzb_ctx);
                // Shouldn't be 64bit constant since then `from` would already be Mem64
                // Obviously such thing could be built, but assuming disasm/users don't..
                if let Some(val) = val.if_constant() {
                    let (ext, mask) = match from {
                        MemAccessSize::Mem8 => (val & 0x80 != 0, 0xff),
                        MemAccessSize::Mem16 => (val & 0x8000 != 0, 0xffff),
                        MemAccessSize::Mem32 | _ => (val & 0x8000_0000 != 0, 0xffff_ffff),
                    };
                    let val = val & mask;
                    if ext {
                        let new = match to {
                            MemAccessSize::Mem16 => (0xffff & !mask) | val as u64,
                            MemAccessSize::Mem32 => (0xffff_ffff & !mask) | val as u64,
                            MemAccessSize::Mem64 | _ => {
                                (0xffff_ffff_ffff_ffff & !mask) | val as u64
                            }
                        };
                        ctx.constant(new)
                    } else {
                        ctx.constant(val)
                    }
                } else {
                    let ty = OperandType::SignExtend(val, from, to);
                    Operand::new_simplified_rc(ty)
                }
            }
            _ => mark_self_simplified(s),
        }
    }

    pub fn transform<F>(oper: &Rc<Operand>, mut f: F) -> Rc<Operand>
    where F: FnMut(&Rc<Operand>) -> Option<Rc<Operand>>
    {
        Operand::simplified(Operand::transform_internal(&oper, &mut f))
    }

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

/// Return (offset, len, value_offset)
fn is_offset_mem(
    op: &Rc<Operand>,
    ctx: &OperandContext,
) -> Option<(Rc<Operand>, (u64, u32, u32))> {
    match op.ty {
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Lsh => {
            if let Some(c) = arith.right.if_constant() {
                if c & 0x7 == 0 && c < 0x40 {
                    let bytes = (c / 8) as u32;
                    return is_offset_mem(&arith.left, ctx)
                        .map(|(x, (off, len, val_off))| {
                            (x, (off, len, val_off + bytes))
                        });
                }
            }
            None
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Rsh => {
            if let Some(c) = arith.right.if_constant() {
                if c & 0x7 == 0 && c < 0x40 {
                    let bytes = (c / 8) as u32;
                    return is_offset_mem(&arith.left, ctx)
                        .and_then(|(x, (off, len, val_off))| {
                            if bytes < len {
                                let off = off.wrapping_add(bytes as u64);
                                Some((x, (off, len - bytes, val_off)))
                            } else {
                                None
                            }
                        });
                }
            }
            None
        }
        OperandType::Memory(ref mem) => {
            let len = match mem.size {
                MemAccessSize::Mem64 => 8,
                MemAccessSize::Mem32 => 4,
                MemAccessSize::Mem16 => 2,
                MemAccessSize::Mem8 => 1,
            };

            Some(Operand::const_offset(&mem.address, ctx)
                .map(|(val, off)| (val, (off, len, 0)))
                .unwrap_or_else(|| (mem.address.clone(), (0, len, 0))))
        }
        _ => None,
    }
}

fn try_merge_memory(
    val: &Rc<Operand>,
    shift: (u64, u32, u32),
    other_shift: (u64, u32, u32),
    ctx: &OperandContext,
) -> Option<Rc<Operand>> {
    use self::operand_helpers::*;
    let (shift, other_shift) = match (shift.2, other_shift.2) {
        (0, 0) => return None,
        (0, _) => (shift, other_shift),
        (_, 0) => (other_shift, shift),
        _ => return None,
    };
    let (off1, len1, _) = shift;
    let (off2, len2, val_off2) = other_shift;
    if off1.wrapping_add(len1 as u64) != off2 || len1 != val_off2 {
        return None;
    }
    let addr = operand_add(val.clone(), ctx.constant(off1));
    let oper = match (len1 + len2).min(4) {
        1 => mem_variable_rc(MemAccessSize::Mem8, addr),
        2 => mem_variable_rc(MemAccessSize::Mem16, addr),
        3 => operand_and(
            mem_variable_rc(MemAccessSize::Mem32, addr),
            ctx.constant(0x00ff_ffff),
        ),
        4 => mem_variable_rc(MemAccessSize::Mem32, addr),
        _ => return None,
    };
    Some(oper)
}

/// Simplify or: merge memory
/// Converts (Mem32[x] >> 8) | (Mem32[x + 4] << 18) to Mem32[x + 1]
/// Also used for xor since x ^ y == x | y if x and y do not overlap at all.
fn simplify_or_merge_mem(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) {
    let mut iter = VecDropIter::new(ops);
    while let Some(mut op) = iter.next() {
        let mut new = None;
        if let Some((val, shift)) = is_offset_mem(&op, ctx) {
            let mut second = iter.duplicate();
            while let Some(other_op) = second.next_removable() {
                let mut remove = false;
                if let Some((other_val, other_shift)) = is_offset_mem(&other_op, ctx) {
                    if val == other_val {
                        let result = try_merge_memory(&val, other_shift, shift, ctx);
                        if let Some(merged) = result {
                            new = Some(Operand::simplified(merged));
                            remove = true;
                        }
                    }
                }
                if remove {
                    other_op.remove();
                    break;
                }
            }
        }
        if let Some(new) = new {
            *op = new;
        }
    }
}

fn add_sub_ops_to_tree(
    ops: &mut Vec<(Rc<Operand>, bool)>,
    ctx: &OperandContext,
) -> Rc<Operand> {
    use self::ArithOpType::*;

    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    // Place non-negated terms last so the simplified result doesn't become
    // (0 - x) + y
    heapsort::sort_by(ops, |&(ref a_val, a_neg), &(ref b_val, b_neg)| {
        (b_neg, b_val) < (a_neg, a_val)
    });
    let mut tree = match ops.pop() {
        Some((s, neg)) => {
            match neg {
                false => mark_self_simplified(s),
                true => {
                    let arith = ArithOperand {
                        ty: Sub,
                        left: ctx.const_0(),
                        right: s,
                    };
                    Operand::new_simplified_rc(OperandType::Arithmetic(arith))
                }
            }
        }
        None => return ctx.const_0(),
    };
    while let Some((op, neg)) = ops.pop() {
        let arith = ArithOperand {
            ty: if neg { Sub } else { Add },
            left: tree,
            right: op,
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    tree
}

fn simplify_add_sub(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    is_sub: bool,
    ctx: &OperandContext,
) -> Rc<Operand> {
    let mut ops = simplify_add_sub_ops(left, right, is_sub, u64::max_value(), ctx);
    add_sub_ops_to_tree(&mut ops, ctx)
}

fn simplify_mul(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    let mut ops = vec![];
    Operand::collect_mul_ops(left, &mut ops);
    Operand::collect_mul_ops(right, &mut ops);
    let mut const_product = ops.iter().flat_map(|x| x.if_constant())
        .fold(1u64, |product, x| product.wrapping_mul(x));
    if const_product == 0 {
        return ctx.const_0();
    }
    ops.retain(|x| x.if_constant().is_none());
    if ops.is_empty() {
        return ctx.constant(const_product);
    }
    heapsort::sort(&mut ops);
    if const_product != 1 {
        let mut changed;
        // Apply constant c * (x + y) => (c * x + c * y) as much as possible.
        // (This repeats at least if (c * x + c * y) => c * y due to c * x == 0)
        loop {
            changed = false;
            for i in 0..ops.len() {
                if simplify_mul_should_apply_constant(&ops[i]) {
                    let new = simplify_mul_apply_constant(&ops[i], const_product, ctx);
                    ops.swap_remove(i);
                    Operand::collect_mul_ops(&new, &mut ops);
                    changed = true;
                    break;
                }
                let new = simplify_mul_try_mul_constants(&ops[i], const_product, ctx);
                if let Some(new) = new {
                    ops.swap_remove(i);
                    Operand::collect_mul_ops(&new, &mut ops);
                    changed = true;
                    break;
                }
            }
            if changed {
                const_product = ops.iter().flat_map(|x| x.if_constant())
                    .fold(1u64, |product, x| product.wrapping_mul(x));
                ops.retain(|x| x.if_constant().is_none());
                heapsort::sort(&mut ops);
                if const_product == 0 {
                    return ctx.const_0();
                } else if const_product == 1 {
                    break;
                }
            } else {
                break;
            }
        }
        if !changed {
            ops.push(ctx.constant(const_product));
        }
    }
    match ops.len() {
        0 => return ctx.const_1(),
        1 => return ops.remove(0),
        _ => (),
    };
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_1());
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::Mul,
            left: tree,
            right: op,
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    tree
}

// For converting c * (x + y) to (c * x + c * y)
fn simplify_mul_should_apply_constant(op: &Operand) -> bool {
    fn inner(op: &Operand) -> bool {
        match op.ty {
            OperandType::Arithmetic(ref arith) => match arith.ty {
                ArithOpType::Add | ArithOpType::Sub => {
                    inner(&arith.left) && inner(&arith.right)
                }
                ArithOpType::Mul => {
                    Operand::either(&arith.left, &arith.right, |x| x.if_constant()).is_some()
                }
                _ => false,
            },
            OperandType::Constant(_) => true,
            _ => false,
        }
    }
    match op.ty {
        OperandType::Arithmetic(ref arith) if {
            arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub
        } => {
            inner(&arith.left) && inner(&arith.right)
        }
        _ => false,
    }
}

fn simplify_mul_apply_constant(op: &Rc<Operand>, val: u64, ctx: &OperandContext) -> Rc<Operand> {
    use self::operand_helpers::*;
    let constant = ctx.constant(val);
    fn inner(op: &Rc<Operand>, constant: &Rc<Operand>) -> Rc<Operand> {
        match op.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
                operand_add(inner(&arith.left, constant), inner(&arith.right, constant))
            }
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
                operand_sub(inner(&arith.left, constant), inner(&arith.right, constant))
            }
            _ => Operand::simplified(operand_mul(constant.clone(), op.clone())),
        }
    }
    let new = inner(op, &constant);
    Operand::simplified(new)
}

// For converting c * (c2 + y) to (c_mul_c2 + c * y)
fn simplify_mul_try_mul_constants(
    op: &Operand,
    c: u64,
    ctx: &OperandContext,
) -> Option<Rc<Operand>> {
    use self::operand_helpers::*;
    match op.ty {
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
            Operand::either(&arith.left, &arith.right, |x| x.if_constant())
                .map(|(c2, other)| {
                    let multiplied = ctx.constant(c2.wrapping_mul(c));
                    operand_add(multiplied, operand_mul(ctx.constant(c), other.clone()))
                })
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
            match (&arith.left.ty, &arith.right.ty) {
                (&OperandType::Constant(c2), _) => {
                    let multiplied = ctx.constant(c2.wrapping_mul(c));
                    Some(operand_sub(multiplied, operand_mul(ctx.constant(c), arith.right.clone())))
                }
                (_, &OperandType::Constant(c2)) => {
                    let multiplied = ctx.constant(c2.wrapping_mul(c));
                    Some(operand_sub(operand_mul(ctx.constant(c), arith.left.clone()), multiplied))
                }
                _ => None
            }
        }
        _ => None,
    }.map(|op| Operand::simplified(op))
}

fn simplify_add_sub_ops(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    is_sub: bool,
    mask: u64,
    ctx: &OperandContext,
) -> Vec<(Rc<Operand>, bool)> {
    let mut ops = Vec::new();
    Operand::collect_add_ops(left, &mut ops, mask, false);
    Operand::collect_add_ops(right, &mut ops, mask, is_sub);
    simplify_collected_add_sub_ops(&mut ops, ctx);
    ops
}

fn simplify_collected_add_sub_ops(
    ops: &mut Vec<(Rc<Operand>, bool)>,
    ctx: &OperandContext,
) {
    let const_sum = ops.iter()
        .flat_map(|&(ref x, neg)| x.if_constant().map(|x| (x, neg)))
        .fold(0u64, |sum, (x, neg)| match neg {
            false => sum.wrapping_add(x),
            true => sum.wrapping_sub(x),
        });
    ops.retain(|&(ref x, _)| x.if_constant().is_none());

    heapsort::sort(ops);
    simplify_add_merge_muls(ops, ctx);
    let new_consts = simplify_add_merge_masked_reverting(ops);
    let const_sum = const_sum.wrapping_add(new_consts);
    if ops.is_empty() {
        if const_sum != 0 {
            ops.push((ctx.constant(const_sum), false));
        }
        return;
    }

    if const_sum != 0 {
        if const_sum > 0x8000_0000_0000_0000 {
            ops.push((ctx.constant(0u64.wrapping_sub(const_sum)), true));
        } else {
            ops.push((ctx.constant(const_sum), false));
        }
    }
}

fn simplify_add_merge_masked_reverting(ops: &mut Vec<(Rc<Operand>, bool)>) -> u64 {
    // Shouldn't need as complex and_const as other places use
    fn and_const(op: &Rc<Operand>) -> Option<(u64, &Rc<Operand>)> {
        match op.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                Operand::either(&arith.left, &arith.right, |x| x.if_constant())
            }
            _ => None,
        }
    }

    fn check_vals(a: &Rc<Operand>, b: &Rc<Operand>) -> bool {
        if let Some((l, r)) = a.if_arithmetic_sub() {
            if l.if_constant() == Some(0) && r == b {
                return true;
            }
        }
        if let Some((l, r)) = b.if_arithmetic_sub() {
            if l.if_constant() == Some(0) && r == a {
                return true;
            }
        }
        false
    }

    if ops.is_empty() {
        return 0;
    }
    let mut sum = 0u64;
    let mut i = 0;
    'outer: while i + 1 < ops.len() {
        let op = &ops[i].0;
        if let Some((constant, val)) = and_const(&op) {
            // Only try merging when mask's low bits are all ones and nothing else is
            if constant.wrapping_add(1).count_ones() <= 1 && ops[i].1 == false {
                let mut j = i + 1;
                while j < ops.len() {
                    let other_op = &ops[j].0;
                    if let Some((other_constant, other_val)) = and_const(&other_op) {
                        let ok = other_constant == constant &&
                            check_vals(val, other_val) &&
                            ops[j].1 == false;
                        if ok {
                            sum = sum.wrapping_add(constant.wrapping_add(1));
                            // Skips i += 1, removes j first to not move i
                            ops.swap_remove(j);
                            ops.swap_remove(i);
                            continue 'outer;
                        }
                    }
                    j += 1;
                }
            }
        }
        i += 1;
    }
    sum
}

/// Returns a better approximation of relevant bits in addition.
///
/// Eq uses this to avoid unnecessary masking, as relbits
/// for addition aren't completely stable depending on order.
///
/// E.g. since
/// add_relbits(x, y) = min(x.low, y.low) .. max(x.hi, y.hi) + 1
/// (bool + bool) = 0..2
/// (bool + u8) = 0..9
/// (bool + bool) + u8 = 0..9
/// (bool + u8) + bool = 0..10
///
/// First return value is relbit mask for positive terms, second is for negative terms.
fn relevant_bits_for_eq(ops: &Vec<(Rc<Operand>, bool)>) -> (u64, u64) {
    let mut sizes = ops.iter().map(|x| (x.1, x.0.relevant_bits())).collect::<Vec<_>>();
    heapsort::sort_by(&mut sizes, |(a_neg, a_bits), (b_neg, b_bits)| {
        (a_neg, a_bits.end) < (b_neg, b_bits.end)
    });
    let mut iter = sizes.iter();
    let mut pos_bits = 64..0;
    let mut neg_bits = 64..0;
    while let Some(next) = iter.next() {
        let bits = next.1.clone();
        if next.0 == true {
            neg_bits = bits;
            while let Some(next) = iter.next() {
                let bits = next.1.clone();
                neg_bits.start = min(bits.start, neg_bits.start);
                neg_bits.end =
                    max(neg_bits.end.wrapping_add(1), bits.end.wrapping_add(1)).min(64);
            }
            break;
        }
        if pos_bits.end == 0 {
            pos_bits = bits;
        } else {
            pos_bits.start = min(bits.start, pos_bits.start);
            pos_bits.end = max(pos_bits.end.wrapping_add(1), bits.end.wrapping_add(1)).min(64);
        }
    }
    let pos_mask = if pos_bits.end == 0 {
        0
    } else {
        let low = pos_bits.start;
        let high = 64 - pos_bits.end;
        !0u64 << high >> high >> low << low
    };
    let neg_mask = if neg_bits.end == 0 {
        0
    } else {
        let low = neg_bits.start;
        let high = 64 - neg_bits.end;
        !0u64 << high >> high >> low << low
    };
    (pos_mask, neg_mask)
}

fn simplify_eq(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
) -> Rc<Operand> {
    use self::operand_helpers::*;

    // Possibly common enough to be worth the early exit
    if left == right {
        return ctx.const_1();
    }
    let left = &Operand::simplified(left.clone());
    let right = &Operand::simplified(right.clone());
    // Well, maybe they are equal now???
    if left == right {
        return ctx.const_1();
    }
    // Equality is just bit comparision without overflow semantics, even though
    // this also uses x == y => x - y == 0 property to simplify it.
    let shared_mask = left.relevant_bits_mask() | right.relevant_bits_mask();
    let add_sub_mask = if shared_mask == 0 {
        u64::max_value()
    } else {
        u64::max_value() >> shared_mask.leading_zeros()
    };
    let mut ops = simplify_add_sub_ops(left, right, true, add_sub_mask, ctx);
    if ops.is_empty() {
        return ctx.const_1();
    }
    // Since with eq the negations can be reverted, canonicalize eq
    // as sorting parts, and making the last one positive, swapping all
    // negations if it isn't yet.
    // Last one since the op tree is constructed in reverse in the end.
    //
    // Sorting without the mask, hopefully is valid way to keep
    // ordering stable.
    heapsort::sort_by(&mut ops, |a, b| Operand::and_masked(&a.0) < Operand::and_masked(&b.0));
    if ops[ops.len() - 1].1 == true {
        for op in &mut ops {
            op.1 = !op.1;
        }
    }
    let mark_self_simplified = |s: &Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    match ops.len() {
        0 => ctx.const_1(),
        1 => match ops[0].0.ty {
            OperandType::Constant(0) => ctx.const_1(),
            OperandType::Constant(_) => ctx.const_0(),
            _ => {
                if let Some((left, right)) = ops[0].0.if_arithmetic_eq() {
                    // Check for (x == 0) == 0
                    let either_const = Operand::either(&left, &right, |x| x.if_constant());
                    if let Some((0, other)) = either_const {
                        let is_compare = match other.ty {
                            OperandType::Arithmetic(ref arith) => arith.is_compare_op(),
                            _ => false,
                        };
                        if is_compare {
                            return other.clone();
                        }
                    }
                }
                // Simplify (x << c2) == 0 to x if c2 cannot shift any bits out
                // Or ((x << c2) & c3) == 0 to (x & (c3 >> c2)) == 0
                // Or ((x >> c2) & c3) == 0 to (x & (c3 << c2)) == 0
                let (masked, mask) = Operand::and_masked(&ops[0].0);
                let mask = mask & add_sub_mask;
                match masked.ty {
                    OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Lsh => {
                        if let Some(c2) = arith.right.if_constant() {
                            let new = simplify_and(
                                &arith.left,
                                &ctx.constant(mask.wrapping_shr(c2 as u32)),
                                ctx,
                                &mut SimplifyWithZeroBits::default(),
                            );
                            return simplify_eq(&new, &ctx.const_0(), ctx);
                        }
                    }
                    OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Rsh => {
                        if let Some(c2) = arith.right.if_constant() {
                            let new = simplify_and(
                                &arith.left,
                                &ctx.constant(mask.wrapping_shl(c2 as u32)),
                                ctx,
                                &mut SimplifyWithZeroBits::default(),
                            );
                            return simplify_eq(&new, &ctx.const_0(), ctx);
                        }
                    }
                    _ => ()
                }
                let mut op = mark_self_simplified(&ops[0].0);
                let relbits = op.relevant_bits_mask();
                if add_sub_mask & relbits != relbits {
                    let constant = ctx.constant(add_sub_mask);
                    op = simplify_and(&op, &constant, ctx, &mut SimplifyWithZeroBits::default());
                }

                let arith = ArithOperand {
                    ty: ArithOpType::Equal,
                    left: op,
                    right: ctx.const_0(),
                };
                Operand::new_simplified_rc(OperandType::Arithmetic(arith))
            }
        },
        2 => {
            let first_const = ops[0].0.if_constant().is_some();

            let (left, right) = match first_const {
                // ops[1] isn't const, so make it to not need sub(0, x)
                true => match ops[1].1 {
                    false => (&ops[0], &ops[1]),
                    true => (&ops[1], &ops[0]),
                },
                // Otherwise just make ops[0] not need sub
                _ => match ops[0].1 {
                    false => (&ops[1], &ops[0]),
                    true => (&ops[0], &ops[1]),
                },
            };
            let mask = add_sub_mask;
            let make_op = |op: &Rc<Operand>, negate: bool| -> Rc<Operand> {
                match negate {
                    false => {
                        let relbit_mask = op.relevant_bits_mask();
                        if relbit_mask & mask != relbit_mask {
                            simplify_and(
                                op,
                                &ctx.constant(mask),
                                ctx,
                                &mut SimplifyWithZeroBits::default(),
                            )
                        } else {
                            mark_self_simplified(op)
                        }
                    }
                    true => {
                        let mut op = operand_sub(ctx.const_0(), op.clone());
                        if mask != u64::max_value() {
                            op = operand_and(op, ctx.constant(mask));
                        }
                        Operand::simplified(op)
                    }
                }
            };
            let left = make_op(&left.0, !left.1);
            let right = make_op(&right.0, right.1);
            simplify_eq_2_ops(left, right, ctx)
        },
        _ => {
            let (left_rel_bits, right_rel_bits) = relevant_bits_for_eq(&ops);
            // Construct a + b + c == d + e + f
            // where left side has all non-negated terms,
            // and right side has all negated terms (Negation forgotten as they're on the right)
            let mut left_tree = match ops.iter().position(|x| x.1 == false) {
                Some(i) => {
                    let op = ops.swap_remove(i).0;
                    mark_self_simplified(&op)
                }
                None => ctx.const_0(),
            };
            let mut right_tree = match ops.iter().position(|x| x.1 == true) {
                Some(i) => {
                    let op = ops.swap_remove(i).0;
                    mark_self_simplified(&op)
                }
                None => ctx.const_0(),
            };
            while let Some((op, neg)) = ops.pop() {
                if !neg {
                    let arith = ArithOperand {
                        ty: ArithOpType::Add,
                        left: left_tree,
                        right: op,
                    };
                    left_tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
                } else {
                    let arith = ArithOperand {
                        ty: ArithOpType::Add,
                        left: right_tree,
                        right: op,
                    };
                    right_tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
                }
            }
            if add_sub_mask & left_rel_bits != left_rel_bits {
                left_tree = simplify_and(
                    &left_tree,
                    &ctx.constant(add_sub_mask),
                    ctx,
                    &mut SimplifyWithZeroBits::default(),
                );
            }
            if add_sub_mask & right_rel_bits != right_rel_bits {
                right_tree = simplify_and(
                    &right_tree,
                    &ctx.constant(add_sub_mask),
                    ctx,
                    &mut SimplifyWithZeroBits::default(),
                );
            }
            let arith = ArithOperand {
                ty: ArithOpType::Equal,
                left: left_tree,
                right: right_tree,
            };
            let ty = OperandType::Arithmetic(arith);
            Operand::new_simplified_rc(ty)
        }
    }
}

fn simplify_eq_2_ops(
    left: Rc<Operand>,
    right: Rc<Operand>,
    ctx: &OperandContext,
) -> Rc<Operand> {
    fn mask_maskee(x: &Rc<Operand>) -> Option<(u64, &Rc<Operand>)> {
        match x.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                Operand::either(&arith.left, &arith.right, |x| x.if_constant())
            }
            OperandType::Memory(ref mem) => {
                match mem.size {
                    MemAccessSize::Mem8 => Some((0xff, x)),
                    MemAccessSize::Mem16 => Some((0xffff, x)),
                    _ => None,
                }
            }
            _ => None,
        }
    };

    let (left, right) = match left < right {
        true => (left, right),
        false => (right, left),
    };

    if let Some((c, other)) = Operand::either(&left, &right, |x| x.if_constant()) {
        if c == 1 {
            // Simplify compare == 1 to compare
            match other.ty {
                OperandType::Arithmetic(ref arith) => {
                    if arith.is_compare_op() {
                        return other.clone();
                    }
                }
                _ => (),
            }
        }
    }
    // Try to prove (x & mask) == ((x + c) & mask) true/false.
    // If c & mask == 0, it's true if c & mask2 == 0, otherwise unknown
    //    mask2 is mask, where 0-bits whose next bit is 1 are switched to 1.
    // If c & mask == mask, it's unknown, unless mask contains the bit 0x1, in which
    // case it's false
    // Otherwise it's false.
    //
    // This can be deduced from how binary addition works; for digit to not change, the
    // added digit needs to either be 0, or 1 with another 1 carried from lower digit's
    // addition.
    //
    // TODO is this necessary anymore?
    // Probably could be simpler to do just with relevant_bits?
    {
        let left_const = mask_maskee(&left);
        let right_const = mask_maskee(&right);
        if let (Some((mask1, l)), Some((mask2, r))) = (left_const, right_const) {
            if mask1 == mask2 {
                let add_const = simplify_eq_masked_add(l).map(|(c, other)| (other, r, c))
                    .or_else(|| {
                        simplify_eq_masked_add(r).map(|(c, other)| (other, l, c))
                    });
                if let Some((a, b, added_const)) = add_const {
                    let a = simplify_with_and_mask(
                        a,
                        mask1,
                        ctx,
                        &mut SimplifyWithZeroBits::default(),
                    );
                    if a == *b {
                        match added_const & mask1 {
                            0 => {
                                // TODO
                            }
                            x if x == mask1 => {
                                if mask1 & 1 == 1 {
                                    return ctx.const_0();
                                }
                            }
                            _ => return ctx.const_0(),
                        }
                    }
                }
            }
        }
    }
    let arith = ArithOperand {
        ty: ArithOpType::Equal,
        left,
        right,
    };
    Operand::new_simplified_rc(OperandType::Arithmetic(arith))
}

fn simplify_eq_masked_add(operand: &Rc<Operand>) -> Option<(u64, &Rc<Operand>)> {
    match operand.ty {
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
            arith.left.if_constant().map(|c| (c, &arith.right))
                .or_else(|| arith.left.if_constant().map(|c| (c, &arith.left)))
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
            arith.left.if_constant().map(|c| (0u64.wrapping_sub(c), &arith.right))
                .or_else(|| arith.right.if_constant().map(|c| (0u64.wrapping_sub(c), &arith.left)))
        }
        _ => None,
    }
}

// Tries to merge (a & a_mask) | (b & b_mask) to (a_mask | b_mask) & result
fn try_merge_ands(
    a: &Rc<Operand>,
    b: &Rc<Operand>,
    a_mask: u64,
    b_mask: u64,
    ctx: &OperandContext,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<Rc<Operand>>{
    if a == b {
        return Some(a.clone());
    }
    if let Some(a) = a.if_constant() {
        if let Some(b) = b.if_constant() {
            return Some(ctx.constant(a | b));
        }
    }
    if let Some((val, shift)) = is_offset_mem(a, ctx) {
        if let Some((other_val, other_shift)) = is_offset_mem(b, ctx) {
            if val == other_val {
                let result = try_merge_memory(&val, other_shift, shift, ctx);
                if let Some(merged) = result {
                    return Some(merged);
                }
            }
        }
    }
    match (&a.ty, &b.ty) {
        (&OperandType::Arithmetic(ref c), &OperandType::Arithmetic(ref d)) => {
            if c.ty == ArithOpType::Xor && d.ty == ArithOpType::Xor {
                try_merge_ands(&c.left, &d.left, a_mask, b_mask, ctx, swzb).and_then(|left| {
                    try_merge_ands(&c.right, &d.right, a_mask, b_mask, ctx, swzb).map(|right| (left, right))
                }).or_else(|| try_merge_ands(&c.left, &d.right, a_mask, b_mask, ctx, swzb).and_then(|first| {
                    try_merge_ands(&c.right, &d.left, a_mask, b_mask, ctx, swzb).map(|second| (first, second))
                })).map(|(first, second)| {
                    Operand::simplified(simplify_xor(&first, &second, ctx, swzb))
                })
            } else {
                None
            }
        }
        (&OperandType::Memory(ref a_mem), &OperandType::Memory(ref b_mem)) => {
            // Can treat Mem16[x], Mem8[x] as Mem16[x], Mem16[x]
            if a_mem.address == b_mem.address {
                let check_mask = |op: &Rc<Operand>, mask: u64, ok: &Rc<Operand>| {
                    if op.relevant_bits().end >= 64 - mask.leading_zeros() as u8 {
                        Some(ok.clone())
                    } else {
                        None
                    }
                };
                match (a_mem.size, b_mem.size) {
                    (MemAccessSize::Mem64, _) => check_mask(b, b_mask, a),
                    (_, MemAccessSize::Mem64) => check_mask(a, a_mask, b),
                    (MemAccessSize::Mem32, _) => check_mask(b, b_mask, a),
                    (_, MemAccessSize::Mem32) => check_mask(a, a_mask, b),
                    (MemAccessSize::Mem16, _) => check_mask(b, b_mask, a),
                    (_, MemAccessSize::Mem16) => check_mask(a, a_mask, b),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Fast path for cases where `register_like OP constant` doesn't simplify more than that.
/// Requires that main simplification always places constant on the right for consistency.
/// (Not all do it as of writing this).
///
/// Note also that with and, it has to be first checked that `register_like & constant` != 0
/// before calling this function.
fn check_quick_arith_simplify<'a>(
    left: &'a Rc<Operand>,
    right: &'a Rc<Operand>,
) -> Option<(&'a Rc<Operand>, &'a Rc<Operand>)> {
    let (c, other) = if left.if_constant().is_some() {
        (left, right)
    } else if right.if_constant().is_some() {
        (right, left)
    } else {
        return None;
    };
    match other.ty {
        OperandType::Register(_) | OperandType::Xmm(_, _) | OperandType::Fpu(_) |
            OperandType::Flag(_) | OperandType::Undefined(_) | OperandType::Custom(_) =>
        {
            Some((other, c))
        }
        _ => None,
    }
}

fn simplify_and(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    if !bits_overlap(&left.relevant_bits(), &right.relevant_bits()) {
        return ctx.const_0();
    }
    let const_other = Operand::either(left, right, |x| x.if_constant());
    if let Some((c, other)) = const_other {
        if let Some((l, r)) = check_quick_arith_simplify(left, right) {
            let left_bits = l.relevant_bits();
            let right = if left_bits.end != 64 {
                let mask = (1 << left_bits.end) - 1;
                let c = r.if_constant().unwrap_or(0);
                if c & mask == mask {
                    return l.clone();
                }
                ctx.constant(c & mask)
            } else {
                r.clone()
            };
            let arith = ArithOperand {
                ty: ArithOpType::And,
                left: l.clone(),
                right,
            };
            return Operand::new_simplified_rc(OperandType::Arithmetic(arith));
        }
        let other_relbit_mask = other.relevant_bits_mask();
        if c & other_relbit_mask == other_relbit_mask {
            return Operand::simplified_with_ctx(other.clone(), ctx, swzb_ctx);
        }
    }

    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());

    let mut ops = vec![];
    Operand::collect_and_ops(left, &mut ops, 30, ctx, swzb_ctx);
    Operand::collect_and_ops(right, &mut ops, 30, ctx, swzb_ctx);
    if ops.len() > 30 {
        // This is likely some hash function being unrolled, give up
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: left.clone(),
            right: right.clone(),
        };
        return Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    let mut const_remain = !0u64;
    // Keep second mask in form 00000011111 (All High bits 0, all low bits 1),
    // as that allows simplifying add/sub/mul a bit more
    let mut low_const_remain = !0u64;
    loop {
        for op in &ops {
            let relevant_bits = op.relevant_bits();
            if relevant_bits.start == 0 {
                let shift = (64 - relevant_bits.end) & 63;
                low_const_remain = low_const_remain << shift >> shift;
            }
        }
        const_remain = ops.iter()
            .map(|op| match op.ty {
                OperandType::Constant(c) => c,
                _ => op.relevant_bits_mask(),
            })
            .fold(const_remain, |sum, x| sum & x);
        ops.retain(|x| x.if_constant().is_none());
        if ops.is_empty() || const_remain == 0 {
            return ctx.constant(const_remain);
        }
        let crem_high_zeros = const_remain.leading_zeros();
        low_const_remain = low_const_remain << crem_high_zeros >> crem_high_zeros;

        heapsort::sort(&mut ops);
        ops.dedup();
        simplify_and_remove_unnecessary_ors(&mut ops, const_remain);

        // Prefer (rax & 0xff) << 1 over (rax << 1) & 0x1fe.
        // Should this be limited to only when all ops are lsh?
        // Or ops.len() == 1.
        // Can't think of any cases now where this early break
        // would hurt though.
        let skip_simplifications = ops.iter().all(|x| {
            x.relevant_bits_mask() == const_remain
        });
        if skip_simplifications {
            break;
        }

        let mut ops_changed = false;
        if low_const_remain != !0 && low_const_remain != const_remain {
            vec_filter_map(&mut ops, |op| {
                let new = simplify_with_and_mask(&op, low_const_remain, ctx, swzb_ctx);
                if let Some(c) = new.if_constant() {
                    if c & const_remain != const_remain {
                        const_remain &= c;
                        ops_changed = true;
                    }
                    None
                } else {
                    if new != op {
                        ops_changed = true;
                    }
                    Some(new)
                }
            });
        }
        if const_remain != !0 {
            vec_filter_map(&mut ops, |op| {
                let new = simplify_with_and_mask(&op, const_remain, ctx, swzb_ctx);
                if let Some(c) = new.if_constant() {
                    if c & const_remain != const_remain {
                        const_remain &= c;
                        ops_changed = true;
                    }
                    None
                } else {
                    if new != op {
                        ops_changed = true;
                    }
                    Some(new)
                }
            });
        }
        if ops.is_empty() {
            break;
        }
        for bits in zero_bit_ranges(const_remain) {
            vec_filter_map(&mut ops, |op| {
                simplify_with_zero_bits(&op, &bits, ctx, swzb_ctx)
                    .and_then(|x| match x.if_constant() {
                        Some(0) => None,
                        _ => Some(x),
                    })
            });
            // Unlike the other is_empty check above this returns 0, since if zero bit filter
            // removes all remaining ops, the result is 0 even with const_remain != 0
            // (simplify_with_zero_bits is defined to return None instead of Some(const(0)),
            // and obviously constant & 0 == 0)
            if ops.is_empty() {
                return ctx.const_0();
            }
        }
        // Simplify (x | y) & mask to (x | (y & mask)) if mask is useless to x
        let mut const_remain_necessary = true;
        for i in 0..ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic_or() {
                let left_mask = match l.if_constant() {
                    Some(c) => c,
                    None => l.relevant_bits_mask(),
                };
                let right_mask = match r.if_constant() {
                    Some(c) => c,
                    None => r.relevant_bits_mask(),
                };
                let left_needs_mask = left_mask & const_remain != left_mask;
                let right_needs_mask = right_mask & const_remain != right_mask;
                if !left_needs_mask && right_needs_mask && left_mask != const_remain {
                    let constant = ctx.constant(const_remain & right_mask);
                    let masked = simplify_and(&r, &constant, ctx, swzb_ctx);
                    let new = simplify_or(&l, &masked, ctx, swzb_ctx);
                    ops[i] = new;
                    const_remain_necessary = false;
                    ops_changed = true;
                } else if left_needs_mask && !right_needs_mask && right_mask != const_remain {
                    let constant = ctx.constant(const_remain & left_mask);
                    let masked = simplify_and(&l, &constant, ctx, swzb_ctx);
                    let new = simplify_or(&r, &masked, ctx, swzb_ctx);
                    ops[i] = new;
                    const_remain_necessary = false;
                    ops_changed = true;
                }
            }
        }
        if !const_remain_necessary {
            // All ops were masked with const remain, so it should not be useful anymore
            const_remain = u64::max_value();
        }

        let mut new_ops = vec![];
        for i in 0..ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic_and() {
                Operand::collect_and_ops(l, &mut new_ops, usize::max_value(), ctx, swzb_ctx);
                Operand::collect_and_ops(r, &mut new_ops, usize::max_value(), ctx, swzb_ctx);
            } else if let Some(c) = ops[i].if_constant() {
                if c & const_remain != const_remain {
                    ops_changed = true;
                    const_remain &= c;
                }
            }
        }

        for op in &mut ops {
            let mask = op.relevant_bits_mask();
            if mask & const_remain != const_remain {
                ops_changed = true;
                const_remain &= mask;
            }
        }
        ops.retain(|x| x.if_constant().is_none());
        if new_ops.is_empty() && !ops_changed {
            break;
        }
        ops.retain(|x| x.if_arithmetic_and().is_none());
        ops.extend(new_ops);
    }
    Operand::simplify_and_merge_child_ors(&mut ops, ctx);

    // Replace not(x) & not(y) with not(x | y)
    if ops.len() >= 2 {
        let neq_compare_count = ops.iter().filter(|x| is_neq_compare(x)).count();
        if neq_compare_count >= 2 {
            let mut neq_ops = Vec::with_capacity(neq_compare_count);
            for op in &mut ops {
                if is_neq_compare(op) {
                    if let Some((l, _)) = op.if_arithmetic_eq() {
                        neq_ops.push(l.clone());
                    }
                }
            }
            let or = simplify_or_ops(neq_ops, ctx, swzb_ctx);
            let not = simplify_eq(&or, &ctx.const_0(), ctx);
            ops.retain(|x| !is_neq_compare(x));
            insert_sorted(&mut ops, not);
        }
    }

    let relevant_bits = ops.iter().fold(!0, |bits, op| {
        bits & op.relevant_bits_mask()
    });
    // Don't use a const mask which has all 1s for relevant bits.
    let final_const_remain = if const_remain & relevant_bits == relevant_bits {
        0
    } else {
        const_remain & relevant_bits
    };
    match ops.len() {
        0 => return ctx.constant(final_const_remain),
        1 if final_const_remain == 0 => return ops.remove(0),
        _ => (),
    };
    heapsort::sort(&mut ops);
    ops.dedup();
    simplify_and_remove_unnecessary_ors(&mut ops, const_remain);
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: tree,
            right: op,
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    // Make constant always be on right of simplified and
    if final_const_remain != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: tree,
            right: ctx.constant(final_const_remain),
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    tree
}

fn is_neq_compare(op: &Rc<Operand>) -> bool {
    match op.if_arithmetic_eq() {
        Some((l, r)) => match l.ty {
            OperandType::Arithmetic(ref a) => a.is_compare_op() && r.if_constant() == Some(0),
            _ => false,
        },
        _ => false,
    }
}

fn insert_sorted(ops: &mut Vec<Rc<Operand>>, new: Rc<Operand>) {
    let insert_pos = match ops.binary_search(&new) {
        Ok(i) | Err(i) => i,
    };
    ops.insert(insert_pos, new);
}

/// Transform (x | y | ...) & x => x
fn simplify_and_remove_unnecessary_ors(
    ops: &mut Vec<Rc<Operand>>,
    const_remain: u64,
) {
    fn contains_or(op: &Rc<Operand>, check: &Rc<Operand>) -> bool {
        if let Some((l, r)) = op.if_arithmetic_or() {
            if l == check || r == check {
                true
            } else {
                contains_or(l, check) || contains_or(r, check)
            }
        } else {
            false
        }
    }

    fn contains_or_const(op: &Rc<Operand>, check: u64) -> bool {
        if let Some((_, r)) = op.if_arithmetic_or() {
            if let Some(c) = r.if_constant() {
                c & check == check
            } else {
                false
            }
        } else {
            false
        }
    }

    let mut pos = 0;
    while pos < ops.len() {
        let mut j = 0;
        while j < ops.len() {
            if j == pos {
                j += 1;
                continue;
            }
            if contains_or(&ops[j], &ops[pos]) {
                ops.remove(j);
                // `j` can be before or after `pos`,
                // depending on that `pos` may need to be decremented
                if j < pos {
                    pos -= 1;
                }
            } else {
                j += 1;
            }
        }
        pos += 1;
    }
    for j in (0..ops.len()).rev() {
        if contains_or_const(&ops[j], const_remain) {
            ops.swap_remove(j);
        }
    }
}

fn simplify_or(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
    swzb: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    let left_bits = left.relevant_bits();
    let right_bits = right.relevant_bits();
    // x | 0 early exit
    if left_bits.start >= left_bits.end {
        return Operand::simplified_with_ctx(right.clone(), ctx, swzb);
    }
    if right_bits.start >= right_bits.end {
        return Operand::simplified_with_ctx(left.clone(), ctx, swzb);
    }
    if let Some((l, r)) = check_quick_arith_simplify(left, right) {
        let r_const = r.if_constant().unwrap_or(0);
        let left_bits = l.relevant_bits_mask();
        if left_bits & r_const == left_bits {
            return r.clone();
        }
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: l.clone(),
            right: r.clone(),
        };
        return Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }

    let mut ops = vec![];
    Operand::collect_or_ops(left, &mut ops, ctx, swzb);
    Operand::collect_or_ops(right, &mut ops, ctx, swzb);
    simplify_or_ops(ops, ctx, swzb)
}

fn simplify_or_ops(
    mut ops: Vec<Rc<Operand>>,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    let ops = &mut ops;
    let mut const_val = 0;
    loop {
        const_val = ops.iter().flat_map(|x| x.if_constant())
            .fold(const_val, |sum, x| sum | x);
        ops.retain(|x| x.if_constant().is_none());
        if ops.is_empty() || const_val == u64::max_value() {
            return ctx.constant(const_val);
        }
        heapsort::sort(ops);
        ops.dedup();
        let mut const_val_changed = false;
        if const_val != 0 {
            vec_filter_map(ops, |op| {
                let new = simplify_with_and_mask(&op, !const_val, ctx, swzb_ctx);
                if let Some(c) = new.if_constant() {
                    if c | const_val != const_val {
                        const_val |= c;
                        const_val_changed = true;
                    }
                    None
                } else {
                    Some(new)
                }
            });
        }
        for bits in one_bit_ranges(const_val) {
            vec_filter_map(ops, |op| simplify_with_one_bits(&op, &bits, ctx));
        }
        Operand::simplify_or_merge_child_ands(ops, ctx, swzb_ctx, false);
        Operand::simplify_or_merge_xors(ops, ctx, swzb_ctx);
        simplify_or_merge_mem(ops, ctx);
        Operand::simplify_or_merge_comparisions(ops, ctx);

        let mut new_ops = vec![];
        for i in 0..ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic_or() {
                Operand::collect_or_ops(l, &mut new_ops, ctx, swzb_ctx);
                Operand::collect_or_ops(r, &mut new_ops, ctx, swzb_ctx);
            } else if let Some(c) = ops[i].if_constant() {
                if c | const_val != const_val {
                    const_val |= c;
                    const_val_changed = true;
                }
            }
        }
        ops.retain(|x| x.if_constant().is_none());
        if new_ops.is_empty() && !const_val_changed {
            break;
        }
        ops.retain(|x| x.if_arithmetic_or().is_none());
        ops.extend(new_ops);
    }
    heapsort::sort(ops);
    ops.dedup();
    match ops.len() {
        0 => return ctx.constant(const_val),
        1 if const_val == 0 => return ops.remove(0),
        _ => (),
    };
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: tree,
            right: op,
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    if const_val != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: tree,
            right: ctx.constant(const_val),
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    tree
}

/// Counts xor ops, descending into x & c masks, as
/// simplify_rsh/lsh do that as well.
/// Too long xors should not be tried to be simplified in shifts.
fn simplify_shift_is_too_long_xor(ops: &[Rc<Operand>]) -> bool {
    fn count(op: &Rc<Operand>) -> usize {
        match op.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                if arith.right.if_constant().is_some() {
                    count(&arith.left)
                } else {
                    1
                }
            }
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Xor => {
                count(&arith.left) + count(&arith.right)
            }
            _ => 1,
        }
    }

    const LIMIT: usize = 16;
    if ops.len() > LIMIT {
        return true;
    }
    let mut sum = 0;
    for op in ops {
        sum += count(op);
        if sum > LIMIT {
            break;
        }
    }
    sum > LIMIT
}

fn simplify_lsh(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
    let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
    let default = || {
        let arith = ArithOperand {
            ty: ArithOpType::Lsh,
            left: left.clone(),
            right: right.clone(),
        };
        Operand::new_simplified_rc(OperandType::Arithmetic(arith))
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left.clone();
    } else if constant >= 64 - u64::from(left.relevant_bits().start) {
        return ctx.const_0();
    }
    let zero_bits = (64 - constant as u8)..64;
    match simplify_with_zero_bits(&left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => {
            if s != left {
                return simplify_lsh(&s, &right, ctx, swzb_ctx);
            }
        }
    }
    match left.ty {
        OperandType::Constant(a) => ctx.constant(a << constant as u8),
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    // Simplify (x & mask) << c to (x << c) & (mask << c)
                    if let Some(c) = arith.right.if_constant() {
                        let high = 64 - zero_bits.start;
                        let low = left.relevant_bits().start;
                        let no_op_mask = !0u64 >> low << low << high >> high;

                        let new = simplify_lsh(&arith.left, &right, ctx, swzb_ctx);
                        if c == no_op_mask {
                            return new;
                        } else {
                            let constant = &ctx.constant(c << constant);
                            return simplify_and(&new, constant, ctx, swzb_ctx);
                        }
                    }
                    let arith = ArithOperand {
                        ty: ArithOpType::Lsh,
                        left: left.clone(),
                        right: right.clone(),
                    };
                    Operand::new_simplified_rc(OperandType::Arithmetic(arith))
                }
                ArithOpType::Xor => {
                    // Try to simplify any parts of the xor separately
                    let mut ops = vec![];
                    Operand::collect_xor_ops(&left, &mut ops, 16, ctx, swzb_ctx);
                    if simplify_shift_is_too_long_xor(&ops) {
                        // Give up on dumb long xors
                        default()
                    } else {
                        for op in &mut ops {
                            *op = simplify_lsh(op, &right, ctx, swzb_ctx);
                        }
                        simplify_xor_ops(&mut ops, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Mul => {
                    if constant < 0x10 {
                        // Prefer (x * y * 4) over ((x * y) << 2),
                        // especially since usually there's already a constant there.
                        let multiply_constant = 1 << constant;
                        simplify_mul(&left, &ctx.constant(multiply_constant), ctx)
                    } else {
                        default()
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(inner_const) = arith.right.if_constant() {
                        let sum = inner_const.saturating_add(constant);
                        if sum < 64 {
                            simplify_lsh(&arith.left, &ctx.constant(sum), ctx, swzb_ctx)
                        } else {
                            ctx.const_0()
                        }
                    } else {
                        default()
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(rsh_const) = arith.right.if_constant() {
                        let diff = rsh_const as i8 - constant as i8;
                        if rsh_const >= 64 {
                            return ctx.const_0();
                        }
                        let mask = (!0u64 >> rsh_const) << constant;
                        let tmp;
                        let val = match diff {
                            0 => &arith.left,
                            // (x >> rsh) << lsh, rsh > lsh
                            x if x > 0 => {
                                tmp = simplify_rsh(
                                    &arith.left,
                                    &ctx.constant(x as u64),
                                    ctx,
                                    swzb_ctx,
                                );
                                &tmp
                            }
                            // (x >> rsh) << lsh, lsh > rsh
                            x => {
                                tmp = simplify_lsh(
                                    &arith.left,
                                    &ctx.constant(x.abs() as u64),
                                    ctx,
                                    swzb_ctx,
                                );
                                &tmp
                            }
                        };
                        let relbit_mask = val.relevant_bits_mask();
                        if relbit_mask & mask != relbit_mask {
                            simplify_and(val, &ctx.constant(mask), ctx, swzb_ctx)
                        } else {
                            // Should be trivially true based on above let but
                            // assert to prevent regressions
                            debug_assert!(val.is_simplified());
                            val.clone()
                        }
                    } else {
                        default()
                    }
                }
                _ => default(),
            }
        }
        _ => default(),
    }
}


fn simplify_rsh(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    use self::operand_helpers::*;

    let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
    let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
    let default = || {
        let arith = ArithOperand {
            ty: ArithOpType::Rsh,
            left: left.clone(),
            right: right.clone(),
        };
        Operand::new_simplified_rc(OperandType::Arithmetic(arith))
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left.clone();
    } else if constant >= left.relevant_bits().end.into() {
        return ctx.const_0();
    }
    let zero_bits = 0..(constant as u8);
    match simplify_with_zero_bits(&left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => {
            if s != left {
                return simplify_rsh(&s, &right, ctx, swzb_ctx);
            }
        }
    }

    match left.ty {
        OperandType::Constant(a) => ctx.constant(a >> constant),
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let const_other =
                        Operand::either(&arith.left, &arith.right, |x| x.if_constant());
                    if let Some((c, other)) = const_other {
                        let low = zero_bits.end;
                        let high = 64 - other.relevant_bits().end;
                        let no_op_mask = !0u64 >> low << low << high >> high;
                        if c == no_op_mask {
                            let new = simplify_rsh(&other, &right, ctx, swzb_ctx);
                            return new;
                        }
                        // `(x & c) >> constant` can be simplified to
                        // `(x >> constant) & (c >> constant)
                        // With lsh/rsh it can simplify further,
                        // but do it always for canonicalization
                        let new = simplify_rsh(&other, &right, ctx, swzb_ctx);
                        let new =
                            simplify_and(&new, &ctx.constant(c >> constant), ctx, swzb_ctx);
                        return new;
                    }
                    let arith = ArithOperand {
                        ty: ArithOpType::Rsh,
                        left: left.clone(),
                        right: right.clone(),
                    };
                    Operand::new_simplified_rc(OperandType::Arithmetic(arith))
                }
                ArithOpType::Xor => {
                    // Try to simplify any parts of the xor separately
                    let mut ops = vec![];
                    Operand::collect_xor_ops(&left, &mut ops, 16, ctx, swzb_ctx);
                    if simplify_shift_is_too_long_xor(&ops) {
                        // Give up on dumb long xors
                        default()
                    } else {
                        for op in &mut ops {
                            *op = simplify_rsh(op, &right, ctx, swzb_ctx);
                        }
                        simplify_xor_ops(&mut ops, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(lsh_const) = arith.right.if_constant() {
                        if lsh_const >= 64 {
                            return ctx.const_0();
                        }
                        let diff = constant as i8 - lsh_const as i8;
                        let mask = (!0u64 << lsh_const) >> constant;
                        let tmp;
                        let val = match diff {
                            0 => &arith.left,
                            // (x << rsh) >> lsh, rsh > lsh
                            x if x > 0 => {
                                tmp = simplify_rsh(
                                    &arith.left,
                                    &ctx.constant(x as u64),
                                    ctx,
                                    swzb_ctx,
                                );
                                &tmp
                            }
                            // (x << rsh) >> lsh, lsh > rsh
                            x => {
                                tmp = simplify_lsh(
                                    &arith.left,
                                    &ctx.constant(x.abs() as u64),
                                    ctx,
                                    swzb_ctx,
                                );
                                &tmp
                            }
                        };
                        let relbit_mask = val.relevant_bits_mask();
                        if relbit_mask & mask != relbit_mask {
                            simplify_and(val, &ctx.constant(mask), ctx, swzb_ctx)
                        } else {
                            // Should be trivially true based on above let but
                            // assert to prevent regressions
                            debug_assert!(val.is_simplified());
                            val.clone()
                        }
                    } else {
                        default()
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(inner_const) = arith.right.if_constant() {
                        let sum = inner_const.saturating_add(constant);
                        if sum < 64 {
                            simplify_rsh(&arith.left, &ctx.constant(sum), ctx, swzb_ctx)
                        } else {
                            ctx.const_0()
                        }
                    } else {
                        default()
                    }
                }
                _ => default(),
            }
        },
        OperandType::Memory(ref mem) => {
            match mem.size {
                MemAccessSize::Mem64 => {
                    if constant >= 56 {
                        let addr = operand_add(mem.address.clone(), ctx.constant(7));
                        let c = ctx.constant(constant - 56);
                        let new = mem_variable_rc(MemAccessSize::Mem8, addr);
                        return simplify_rsh(&new, &c, ctx, swzb_ctx);
                    } else if constant >= 48 {
                        let addr = operand_add(mem.address.clone(), ctx.constant(6));
                        let c = ctx.constant(constant - 48);
                        let new = mem_variable_rc(MemAccessSize::Mem16, addr);
                        return simplify_rsh(&new, &c, ctx, swzb_ctx);
                    } else if constant >= 32 {
                        let addr = operand_add(mem.address.clone(), ctx.const_4());
                        let c = ctx.constant(constant - 32);
                        let new = mem_variable_rc(MemAccessSize::Mem32, addr);
                        return simplify_rsh(&new, &c, ctx, swzb_ctx);
                    }
                }
                MemAccessSize::Mem32 => {
                    if constant >= 24 {
                        let addr = operand_add(mem.address.clone(), ctx.constant(3));
                        let c = ctx.constant(constant - 24);
                        let new = mem_variable_rc(MemAccessSize::Mem8, addr);
                        return simplify_rsh(&new, &c, ctx, swzb_ctx);
                    } else if constant >= 16 {
                        let addr = operand_add(mem.address.clone(), ctx.const_2());
                        let c = ctx.constant(constant - 16);
                        let new = mem_variable_rc(MemAccessSize::Mem16, addr);
                        return simplify_rsh(&new, &c, ctx, swzb_ctx);
                    }
                }
                MemAccessSize::Mem16 => {
                    if constant >= 8 {
                        let addr = operand_add(mem.address.clone(), ctx.const_1());
                        let c = ctx.constant(constant - 8);
                        let new = mem_variable_rc(MemAccessSize::Mem8, addr);
                        return simplify_rsh(&new, &c, ctx, swzb_ctx);
                    }
                }
                _ => (),
            }
            default()
        }
        _ => default(),
    }
}

fn vec_filter_map<T, F: FnMut(T) -> Option<T>>(vec: &mut Vec<T>, mut fun: F) {
    for _ in 0..vec.len() {
        let val = vec.pop().unwrap();
        if let Some(new) = fun(val) {
            vec.insert(0, new);
        }
    }
}

fn should_stop_with_and_mask(swzb_ctx: &mut SimplifyWithZeroBits) -> bool {
    if swzb_ctx.with_and_mask_count > 80 {
        #[cfg(feature = "fuzz")]
        tls_simplification_incomplete();
        true
    } else {
        false
    }
}

/// Convert and(x, mask) to x
fn simplify_with_and_mask(
    op: &Rc<Operand>,
    mask: u64,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    if op.relevant_bits_mask() & mask == 0 {
        return ctx.const_0();
    }
    if should_stop_with_and_mask(swzb_ctx) {
        return op.clone();
    }
    swzb_ctx.with_and_mask_count += 1;
    let op = simplify_with_and_mask_inner(op, mask, ctx, swzb_ctx);
    op
}

fn simplify_with_and_mask_inner(
    op: &Rc<Operand>,
    mask: u64,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    use self::operand_helpers::*;
    match op.ty {
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    if let Some(c) = arith.right.if_constant() {
                        let self_mask = mask & arith.left.relevant_bits_mask();
                        if c == self_mask {
                            return arith.left.clone();
                        } else if c & self_mask == 0 {
                            return ctx.const_0();
                        }
                    }
                    let simplified_left =
                        simplify_with_and_mask(&arith.left, mask, ctx, swzb_ctx);
                    let simplified_right =
                        simplify_with_and_mask(&arith.right, mask, ctx, swzb_ctx);
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op.clone();
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op.clone()
                    } else {
                        let op = simplify_and(&simplified_left, &simplified_right, ctx, swzb_ctx);
                        simplify_with_and_mask(&op, mask, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Or => {
                    let simplified_left =
                        simplify_with_and_mask(&arith.left, mask, ctx, swzb_ctx);
                    if let Some(c) = simplified_left.if_constant() {
                        if mask & c == mask & arith.right.relevant_bits_mask() {
                            return simplified_left;
                        }
                    }
                    let simplified_right =
                        simplify_with_and_mask(&arith.right, mask, ctx, swzb_ctx);
                    if let Some(c) = simplified_right.if_constant() {
                        if mask & c == mask & arith.left.relevant_bits_mask() {
                            return simplified_right;
                        }
                    }
                    // Possibly common to get zeros here
                    if simplified_left.if_constant() == Some(0) {
                        return simplified_right;
                    }
                    if simplified_right.if_constant() == Some(0) {
                        return simplified_left;
                    }
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op.clone();
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op.clone()
                    } else {
                        simplify_or(&simplified_left, &simplified_right, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let left = simplify_with_and_mask(&arith.left, mask >> c, ctx, swzb_ctx);
                        if left == arith.left {
                            op.clone()
                        } else {
                            let op = operand_lsh(left, arith.right.clone());
                            Operand::simplified(op)
                        }
                    } else {
                        op.clone()
                    }
                }
                ArithOpType::Xor | ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul => {
                    if arith.ty != ArithOpType::Xor {
                        // The mask can be applied separately to left and right if
                        // any of the unmasked bits input don't affect masked bits in result.
                        // For add/sub/mul, a bit can only affect itself and more
                        // significant bits.
                        //
                        // First, check if relevant bits start of either operand >= mask end,
                        // in which case the operand cannot affect result at all and we can
                        // just return the other operand simplified with the mask.
                        //
                        // Otherwise check if mask has has all low bits 1 and all high bits 0,
                        // and apply left/right separately.
                        //
                        // Assuming it is 00001111...
                        // Adding 1 makes a valid mask to overflow to 10000000...
                        // Though the 1 bit can be carried out so count_ones is 1 or 0.
                        let mask_end_bit = 64 - mask.leading_zeros() as u8;
                        let other = Operand::either(&arith.left, &arith.right, |x| {
                            if x.relevant_bits().start >= mask_end_bit { Some(()) } else { None }
                        }).map(|((), other)| other);
                        if let Some(other) = other {
                            return simplify_with_and_mask(other, mask, ctx, swzb_ctx);
                        }
                        let ok = mask.wrapping_add(1).count_ones() <= 1;
                        if !ok {
                            return op.clone();
                        }
                    }
                    let simplified_left =
                        simplify_with_and_mask(&arith.left, mask, ctx, swzb_ctx);
                    let simplified_right =
                        simplify_with_and_mask(&arith.right, mask, ctx, swzb_ctx);
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op.clone();
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op.clone()
                    } else {
                        let op = operand_arith(arith.ty, simplified_left, simplified_right);
                        let op = Operand::simplified_with_ctx(op, ctx, swzb_ctx);
                        // The result may simplify again, for example with mask 0x1
                        // Mem16[x] + Mem32[x] + Mem8[x] => 3 * Mem8[x] => 1 * Mem8[x]
                        simplify_with_and_mask(&op, mask, ctx, swzb_ctx)
                    }
                }
                _ => op.clone(),
            }
        }
        OperandType::Memory(ref mem) => {
            let mask = match mem.size {
                MemAccessSize::Mem8 => mask & 0xff,
                MemAccessSize::Mem16 => mask & 0xffff,
                MemAccessSize::Mem32 => mask & 0xffff_ffff,
                MemAccessSize::Mem64 => mask,
            };
            // Try to do conversions such as Mem32[x] & 00ff_ff00 => Mem16[x + 1] << 8,
            // but also Mem32[x] & 003f_5900 => (Mem16[x + 1] & 3f59) << 8.

            // Round down to 8 -> convert to bytes
            let mask_low = mask.trailing_zeros() / 8;
            // Round up to 8 -> convert to bytes
            let mask_high = (64 - mask.leading_zeros() + 7) / 8;
            if mask_high <= mask_low {
                return op.clone();
            }
            let mask_size = mask_high - mask_low;
            let mem_size = mem.size.bits();
            let new_size;
            if mask_size <= 1 && mem_size > 8 {
                new_size = MemAccessSize::Mem8;
            } else if mask_size <= 2 && mem_size > 16 {
                new_size = MemAccessSize::Mem16;
            } else if mask_size <= 4 && mem_size > 32 {
                new_size = MemAccessSize::Mem32;
            } else {
                return op.clone();
            }
            let new_addr = if mask_low == 0 {
                mem.address.clone()
            } else {
                operand_add(mem.address.clone(), ctx.constant(mask_low as u64))
            };
            let mem = mem_variable_rc(new_size, new_addr);
            let shifted = if mask_low == 0 {
                mem
            } else {
                operand_lsh(mem, ctx.constant(mask_low as u64 * 8))
            };
            Operand::simplified(shifted)
        }
        OperandType::Constant(c) => if c & mask != c {
            ctx.constant(c & mask)
        } else {
            op.clone()
        }
        _ => op.clone(),
    }
}

#[derive(Default)]
struct SimplifyWithZeroBits {
    simplify_count: u8,
    with_and_mask_count: u8,
    /// simplify_with_zero_bits can cause a lot of recursing in xor
    /// simplification with has functions, stop simplifying if a limit
    /// is hit.
    xor_recurse: u8,
}

/// Simplifies `op` when the bits in the range `bits` are guaranteed to be zero.
/// Returning `None` is considered same as `Some(constval(0))` (The value gets optimized out in
/// bitwise and).
///
/// Bits are assumed to be in 0..64 range
fn simplify_with_zero_bits(
    op: &Rc<Operand>,
    bits: &Range<u8>,
    ctx: &OperandContext,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<Rc<Operand>> {
    use self::operand_helpers::*;
    if op.min_zero_bit_simplify_size > bits.end - bits.start || bits.start >= bits.end {
        return Some(op.clone());
    }
    let relevant_bits = op.relevant_bits();
    // Check if we're setting all nonzero bits to zero
    if relevant_bits.start >= bits.start && relevant_bits.end <= bits.end {
        return None;
    }
    // Check if we're zeroing bits that were already zero
    if relevant_bits.start >= bits.end || relevant_bits.end <= bits.start {
        return Some(op.clone());
    }

    let recurse_check = match op.ty {
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And | ArithOpType::Or | ArithOpType::Xor |
                    ArithOpType::Lsh | ArithOpType::Rsh => true,
                _ => false,
            }
        }
        _ => false,
    };

    fn should_stop(swzb: &mut SimplifyWithZeroBits) -> bool {
        if swzb.simplify_count > 40 {
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
            true
        } else {
            false
        }
    }

    if recurse_check {
        if swzb.xor_recurse > 4 {
            swzb.simplify_count = u8::max_value();
        }
        if should_stop(swzb) {
            return Some(op.clone());
        } else {
            swzb.simplify_count += 1;
        }
    }

    match op.ty {
        OperandType::Arithmetic(ref arith) => {
            let left = &arith.left;
            let right = &arith.right;
            match arith.ty {
                ArithOpType::And => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    if should_stop(swzb) {
                        return Some(op.clone());
                    }
                    return match simplified_left {
                        Some(l) => {
                            let simplified_right =
                                simplify_with_zero_bits(right, bits, ctx, swzb);
                            if should_stop(swzb) {
                                return Some(op.clone());
                            }
                            match simplified_right {
                                Some(r) => {
                                    if l == *left && r == *right {
                                        Some(op.clone())
                                    } else {
                                        Some(simplify_and(&l, &r, ctx, swzb))
                                    }
                                }
                                None => None,
                            }
                        }
                        None => None,
                    };
                }
                ArithOpType::Or => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    let simplified_right = simplify_with_zero_bits(right, bits, ctx, swzb);
                    if should_stop(swzb) {
                        return Some(op.clone());
                    }
                    return match (simplified_left, simplified_right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l == *left && r == *right {
                                Some(op.clone())
                            } else {
                                Some(simplify_or(&l, &r, ctx, swzb))
                            }
                        }
                    };
                }
                ArithOpType::Xor => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    let simplified_right = simplify_with_zero_bits(right, bits, ctx, swzb);
                    if should_stop(swzb) {
                        return Some(op.clone());
                    }
                    return match (simplified_left, simplified_right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l == *left && r == *right {
                                Some(op.clone())
                            } else {
                                swzb.xor_recurse += 1;
                                let result = simplify_xor(&l, &r, ctx, swzb);
                                swzb.xor_recurse -= 1;
                                Some(result)
                            }
                        }
                    };
                }
                ArithOpType::Lsh => {
                    if let Some(c) = right.if_constant() {
                        if bits.end >= 64 && bits.start <= c as u8 {
                            return None;
                        } else {
                            let low = bits.start.saturating_sub(c as u8);
                            let high = bits.end.saturating_sub(c as u8);
                            if low >= high {
                                return Some(op.clone());
                            }
                            let result = simplify_with_zero_bits(left, &(low..high), ctx, swzb);
                            if let Some(result) =  result {
                                if result != *left {
                                    return Some(simplify_lsh(&result, right, ctx, swzb));
                                }
                            }
                        }
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = right.if_constant() {
                        if bits.start == 0 && c as u8 >= (64 - bits.end) {
                            return None;
                        } else {
                            let low = bits.start.saturating_add(c as u8).min(64);
                            let high = bits.end.saturating_add(c as u8).min(64);
                            if low >= high {
                                return Some(op.clone());
                            }
                            let result1 = if bits.end == 64 {
                                let mask_high = 64 - low;
                                let mask = !0u64 >> c << c << mask_high >> mask_high;
                                simplify_with_and_mask(left, mask, ctx, swzb)
                            } else {
                                left.clone()
                            };
                            let result2 =
                                simplify_with_zero_bits(&result1, &(low..high), ctx, swzb);
                            if let Some(result2) =  result2 {
                                if result2 != *left {
                                    return Some(
                                        simplify_rsh(&result2, right, ctx, swzb)
                                    );
                                }
                            } else if result1 != *left {
                                return Some(simplify_rsh(&result1, right, ctx, swzb));
                            }
                        }
                    }
                }
                _ => (),
            }
        }
        OperandType::Constant(c) => {
            let low = bits.start;
            let high = 64 - bits.end;
            let mask = !(!0u64 >> low << low << high >> high);
            let new_val = c & mask;
            return match new_val {
                0 => None,
                c => Some(ctx.constant(c)),
            };
        }
        OperandType::Memory(ref mem) => {
            if bits.start == 0 && bits.end >= relevant_bits.end {
                return None;
            } else if bits.end == 64 {
                if bits.start <= 8 && relevant_bits.end > 8 {
                    return Some(mem_variable_rc(MemAccessSize::Mem8, mem.address.clone()));
                } else if bits.start <= 16 && relevant_bits.end > 16 {
                    return Some(mem_variable_rc(MemAccessSize::Mem16, mem.address.clone()));
                } else if bits.start <= 32 && relevant_bits.end > 32 {
                    return Some(mem_variable_rc(MemAccessSize::Mem32, mem.address.clone()));
                }
            }
        }
        _ => (),
    }
    Some(op.clone())
}

/// Simplifies `op` when the bits in the range `bits` are guaranteed to be one.
/// Returning `None` means that `op | constval(bits) == constval(bits)`
fn simplify_with_one_bits(
    op: &Rc<Operand>,
    bits: &Range<u8>,
    ctx: &OperandContext,
) -> Option<Rc<Operand>> {
    fn check_useless_and_mask<'a>(
        left: &'a Rc<Operand>,
        right: &'a Rc<Operand>,
        bits: &Range<u8>,
    ) -> Option<&'a Rc<Operand>> {
        // one_bits | (other & c) can be transformed to other & (c | one_bits)
        // if c | one_bits is all ones for other's relevant bits, const mask
        // can be removed.
        let const_other = Operand::either(left, right, |x| x.if_constant());
        if let Some((c, other)) = const_other {
            let low = bits.start;
            let high = 64 - bits.end;
            let mask = !0u64 >> low << low << high >> high;
            let nop_mask = other.relevant_bits_mask();
            if c | mask == nop_mask {
                return Some(other);
            }
        }
        None
    }

    use self::operand_helpers::*;
    if bits.start >= bits.end {
        return Some(op.clone());
    }
    let default = || {
        let relevant_bits = op.relevant_bits();
        match relevant_bits.start >= bits.start && relevant_bits.end <= bits.end {
            true => None,
            false => Some(op.clone()),
        }
    };
    match op.ty {
        OperandType::Arithmetic(ref arith) => {
            let left = &arith.left;
            let right = &arith.right;
            match arith.ty {
                ArithOpType::And => {
                    if let Some(other) = check_useless_and_mask(left, right, bits) {
                        return simplify_with_one_bits(other, bits, ctx);
                    }
                    let left = simplify_with_one_bits(left, bits, ctx);
                    let right = simplify_with_one_bits(right, bits, ctx);
                    match (left, right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => {
                            let low = bits.start;
                            let high = 64 - bits.end;
                            let mask = !0u64 >> low << low << high >> high;
                            if mask == s.relevant_bits_mask() {
                                Some(s)
                            } else {
                                Some(Operand::simplified(operand_and(ctx.constant(mask), s)))
                            }
                        }
                        (Some(l), Some(r)) => {
                            if l != arith.left || r != arith.right {
                                if let Some(other) = check_useless_and_mask(&l, &r, bits) {
                                    return simplify_with_one_bits(other, bits, ctx);
                                }
                                let new = Operand::simplified(operand_and(l, r));
                                if new == *op {
                                    Some(new)
                                } else {
                                    simplify_with_one_bits(&new, bits, ctx)
                                }
                            } else {
                                Some(op.clone())
                            }
                        }
                    }
                }
                ArithOpType::Or => {
                    let left = simplify_with_one_bits(left, bits, ctx);
                    let right = simplify_with_one_bits(right, bits, ctx);
                    match (left, right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l != arith.left || r != arith.right {
                                let new = Operand::simplified(operand_or(l, r));
                                if new == *op {
                                    Some(new)
                                } else {
                                    simplify_with_one_bits(&new, bits, ctx)
                                }
                            } else {
                                Some(op.clone())
                            }
                        }
                    }
                }
                _ => default(),
            }
        }
        OperandType::Constant(c) => {
            let low = bits.start;
            let high = 64 - bits.end;
            let mask = !0u64 >> low << low << high >> high;
            let new_val = c | mask;
            match new_val & !mask {
                0 => None,
                c => Some(ctx.constant(c)),
            }
        }
        OperandType::Memory(ref mem) => {
            let max_bits = op.relevant_bits();
            if bits.start == 0 && bits.end >= max_bits.end {
                None
            } else if bits.end >= max_bits.end {
                if bits.start <= 8 && max_bits.end > 8 {
                    Some(mem_variable_rc(MemAccessSize::Mem8, mem.address.clone()))
                } else if bits.start <= 16 && max_bits.end > 16 {
                    Some(mem_variable_rc(MemAccessSize::Mem16, mem.address.clone()))
                } else if bits.start <= 32 && max_bits.end > 32 {
                    Some(mem_variable_rc(MemAccessSize::Mem32, mem.address.clone()))
                } else {
                    Some(op.clone())
                }
            } else {
                Some(op.clone())
            }
        }
        _ => default(),
    }
}

/// Merges things like [2 * b, a, c, b, c] to [a, 3 * b, 2 * c]
fn simplify_add_merge_muls(
    ops: &mut Vec<(Rc<Operand>, bool)>,
    ctx: &OperandContext,
) {
    fn count_equivalent_opers(ops: &[(Rc<Operand>, bool)], equiv: &Operand) -> Option<u64> {
        ops.iter().map(|&(ref o, neg)| {
            let (mul, val) = o.if_arithmetic_mul()
                .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                .unwrap_or_else(|| (1, o));
            match *equiv == **val {
                true => if neg { 0u64.wrapping_sub(mul) } else { mul },
                false => 0,
            }
        }).fold(None, |sum, next| if next != 0 {
            Some(sum.unwrap_or(0).wrapping_add(next))
        } else {
            sum
        })
    }

    let mut pos = 0;
    while pos < ops.len() {
        let merged = {
            let (self_mul, op) = ops[pos].0.if_arithmetic_mul()
                .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                .unwrap_or_else(|| (1, &ops[pos].0));

            let others = count_equivalent_opers(&ops[pos + 1..], op);
            if let Some(others) = others {
                let self_mul = if ops[pos].1 { 0u64.wrapping_sub(self_mul) } else { self_mul };
                let sum = self_mul.wrapping_add(others);
                if sum == 0 {
                    Some(None)
                } else {
                    Some(Some((sum, op.clone())))
                }
            } else {
                None
            }
        };
        match merged {
            Some(Some((sum, equiv))) => {
                let mut other_pos = pos + 1;
                while other_pos < ops.len() {
                    let is_equiv = ops[other_pos].0
                        .if_arithmetic_mul()
                        .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                        .map(|(_, other)| *other == equiv)
                        .unwrap_or_else(|| ops[other_pos].0 == equiv);
                    if is_equiv {
                        ops.remove(other_pos);
                    } else {
                        other_pos += 1;
                    }
                }
                let negate = sum > 0x8000_0000_0000_0000;
                let sum = if negate { (!sum).wrapping_add(1) } else { sum };
                ops[pos].0 = simplify_mul(&equiv, &ctx.constant(sum), ctx);
                ops[pos].1 = negate;
                pos += 1;
            }
            // Remove everything matching
            Some(None) => {
                let (op, _) = ops.remove(pos);
                let equiv = op.if_arithmetic_mul()
                    .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                    .map(|(_, other)| other)
                    .unwrap_or_else(|| &op);
                let mut other_pos = pos;
                while other_pos < ops.len() {
                    if ops[other_pos].0 == *equiv {
                        ops.remove(other_pos);
                    } else {
                        other_pos += 1;
                    }
                }
            }
            None => {
                pos += 1;
            }
        }
    }
}

fn simplify_xor(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
    swzb: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    let left_bits = left.relevant_bits();
    let right_bits = right.relevant_bits();
    // x ^ 0 early exit
    if left_bits.start >= left_bits.end {
        return Operand::simplified(right.clone());
    }
    if right_bits.start >= right_bits.end {
        return Operand::simplified(left.clone());
    }
    if let Some((l, r)) = check_quick_arith_simplify(left, right) {
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: l.clone(),
            right: r.clone(),
        };
        return Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    let mut ops = vec![];
    Operand::collect_xor_ops(left, &mut ops, 30, ctx, swzb);
    Operand::collect_xor_ops(right, &mut ops, 30, ctx, swzb);
    if ops.len() > 30 {
        // This is likely some hash function being unrolled, give up
        // Also set swzb to stop everything
        swzb.simplify_count = u8::max_value();
        swzb.with_and_mask_count = u8::max_value();
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: left.clone(),
            right: right.clone(),
        };
        return Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    simplify_xor_ops(&mut ops, ctx, swzb)
}

fn simplify_xor_try_extract_constant(
    op: &Rc<Operand>,
    ctx: &OperandContext,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<(Rc<Operand>, u64)> {
    use self::operand_helpers::*;

    fn recurse(op: &Rc<Operand>) -> Option<(Rc<Operand>, u64)> {
        match op.ty {
            OperandType::Arithmetic(ref arith) => {
                match arith.ty {
                    ArithOpType::And => {
                        let left = recurse(&arith.left);
                        let right = recurse(&arith.right);
                        return match (left, right) {
                            (None, None) => None,
                            (Some(a), None) => {
                                Some((operand_and(a.0, arith.right.clone()), a.1))
                            }
                            (None, Some(a)) => {
                                Some((operand_and(a.0, arith.left.clone()), a.1))
                            }
                            (Some(a), Some(b)) => {
                                Some((operand_and(a.0, b.0), a.1 ^ b.1))
                            }
                        };
                    }
                    ArithOpType::Xor => {
                        if let Some(c) = arith.right.if_constant() {
                            return Some((arith.left.clone(), c));
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
        None
    }

    let (l, r) = op.if_arithmetic_and()?;
    let and_mask = r.if_constant()?;
    let (new, c) = recurse(l)?;
    let new = simplify_and(&new, r, ctx, swzb);
    Some((new, c & and_mask))
}

fn simplify_xor_ops(
    ops: &mut Vec<Rc<Operand>>,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());

    let mut const_val = 0;
    loop {
        const_val = ops.iter().flat_map(|x| x.if_constant())
            .fold(const_val, |sum, x| sum ^ x);
        ops.retain(|x| x.if_constant().is_none());
        heapsort::sort(&mut *ops);
        simplify_xor_remove_reverting(ops);
        simplify_or_merge_mem(ops, ctx); // Yes, this is supposed to stay valid for xors.
        Operand::simplify_or_merge_child_ands(ops, ctx, swzb_ctx, true);
        if ops.is_empty() {
            return ctx.constant(const_val);
        }

        let mut ops_changed = false;
        for i in 0..ops.len() {
            let op = &ops[i];
            // Convert c1 ^ (y | z) to c1 ^ z ^ y if y & z == 0
            if let Some((l, r)) = op.if_arithmetic_or() {
                let l_bits = match l.if_constant() {
                    Some(c) => c,
                    None => Operand::and_masked(l).1 & l.relevant_bits_mask(),
                };
                let r_bits = match r.if_constant() {
                    Some(c) => c,
                    None => Operand::and_masked(r).1 & r.relevant_bits_mask(),
                };
                if l_bits & r_bits == 0 {
                    let const_other = Operand::either(l, r, |x| x.if_constant());
                    if let Some((c, other)) = const_other {
                        const_val ^= c;
                        ops[i] = other.clone();
                    } else {
                        let l = l.clone();
                        let r = r.clone();
                        ops[i] = l;
                        ops.push(r);
                    }
                    ops_changed = true;
                }
            }
        }
        for i in 0..ops.len() {
            let result = simplify_xor_try_extract_constant(&ops[i], ctx, swzb_ctx);
            if let Some((new, constant)) = result {
                ops[i] = new;
                const_val ^= constant;
                ops_changed = true;
            }
        }
        let mut new_ops = vec![];
        for i in 0..ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic(ArithOpType::Xor) {
                Operand::collect_xor_ops(l, &mut new_ops, usize::max_value(), ctx, swzb_ctx);
                Operand::collect_xor_ops(r, &mut new_ops, usize::max_value(), ctx, swzb_ctx);
            }
        }
        if new_ops.is_empty() && !ops_changed {
            heapsort::sort(&mut *ops);
            break;
        }
        ops.retain(|x| x.if_arithmetic(ArithOpType::Xor).is_none());
        ops.extend(new_ops);
    }

    match ops.len() {
        0 => return ctx.constant(const_val),
        1 if const_val == 0 => return ops.remove(0),
        _ => (),
    };
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: tree,
            right: op,
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    // Make constant always be on topmost right branch
    if const_val != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: tree,
            right: ctx.constant(const_val),
        };
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(arith));
    }
    tree
}

/// Assumes that `ops` is sorted.
fn simplify_xor_remove_reverting(ops: &mut Vec<Rc<Operand>>) {
    let mut first_same = ops.len() as isize - 1;
    let mut pos = first_same - 1;
    while pos >= 0 {
        let pos_u = pos as usize;
        let first_u = first_same as usize;
        if ops[pos_u] == ops[first_u] {
            ops.remove(first_u);
            ops.remove(pos_u);
            first_same -= 2;
            if pos > first_same {
                pos = first_same
            }
        } else {
            first_same = pos;
        }
        pos -= 1;
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct MemAccess {
    pub address: Rc<Operand>,
    pub size: MemAccessSize,
}

#[derive(Clone, Eq, PartialEq, Copy, Debug, Hash, Ord, PartialOrd, Deserialize, Serialize)]
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

#[derive(Clone, Eq, PartialEq, Copy, Debug, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub struct Register(pub u8);

#[derive(Clone, Eq, PartialEq, Copy, Debug, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub enum Flag {
    Zero,
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

    pub fn operand_xmm(num: u8, word: u8) -> Rc<Operand> {
        Operand::new_simplified_rc(OperandType::Xmm(num, word))
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

    pub fn operand_signed_mul(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(SignedMul, lhs, rhs)
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

    pub fn operand_not(lhs: Rc<Operand>) -> Rc<Operand> {
        operand_xor(lhs, constval(0xffff_ffff))
    }

    pub fn operand_logical_not(lhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Equal, constval(0), lhs)
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
        Operand::new_not_simplified_rc(OperandType::Memory(MemAccess {
            address: val,
            size,
        }))
    }

    pub fn constval(num: u64) -> Rc<Operand> {
        OperandContext::new().constant(num)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn check_simplification_consistency(op: Rc<Operand>) {
        let simplified = Operand::simplified(op);
        let bytes = bincode::serialize(&simplified).unwrap();
        let back: Rc<Operand> = Rc::new(bincode::deserialize(&bytes).unwrap());
        assert_eq!(simplified, back);
    }

    #[test]
    fn simplify_add_sub() {
        use super::operand_helpers::*;
        let op1 = operand_add(constval(5), operand_sub(operand_register(2), constval(5)));
        assert_eq!(Operand::simplified(op1), operand_register(2));
        // (5 * r2) + (5 - (5 + r2)) == (5 * r2) - r2
        let op1 = operand_add(
            operand_mul(constval(5), operand_register(2)),
            operand_sub(
                constval(5),
                operand_add(constval(5), operand_register(2)),
            )
        );
        let op2 = operand_sub(
            operand_mul(constval(5), operand_register(2)),
            operand_register(2),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(op2));
    }

    #[test]
    fn simplify_add_sub_repeat_operands() {
        use super::operand_helpers::*;
        // x - (x - 4) == 4
        let op1 = operand_sub(
            operand_register(2),
            operand_sub(
                operand_register(2),
                constval(4),
            )
        );
        let op2 = constval(4);
        assert_eq!(Operand::simplified(op1), Operand::simplified(op2));
    }

    #[test]
    fn simplify_mul() {
        use super::operand_helpers::*;
        let op1 = operand_add(constval(5), operand_sub(operand_register(2), constval(5)));
        assert_eq!(Operand::simplified(op1), operand_register(2));
        // (5 * r2) + (5 - (5 + r2)) == (5 * r2) - r2
        let op1 = operand_mul(
            operand_mul(constval(5), operand_register(2)),
            operand_mul(
                constval(5),
                operand_add(constval(5), operand_register(2)),
            )
        );
        let op2 = operand_mul(
            constval(25),
            operand_mul(
                operand_register(2),
                operand_add(constval(5), operand_register(2)),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(op2));
    }

    #[test]
    fn simplify_and_or_chain() {
        use super::operand_helpers::*;
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        // ((((w | (mem8[z] << 8)) & 0xffffff00) | mem8[y]) & 0xffff00ff) ==
        //     ((w & 0xffffff00) | mem8[y]) & 0xffff00ff
        let op1 = operand_and(
            operand_or(
                operand_and(
                    operand_or(
                        operand_register(4),
                        operand_lsh(
                            mem8(operand_register(3)),
                            constval(8),
                        ),
                    ),
                    constval(0xffffff00),
                ),
                mem8(operand_register(2)),
            ),
            constval(0xffff00ff),
        );
        let op2 = operand_and(
            operand_or(
                operand_and(
                    operand_register(4),
                    constval(0xffffff00),
                ),
                mem8(operand_register(2)),
            ),
            constval(0xffff00ff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(op2));
    }

    #[test]
    fn simplify_and() {
        use super::operand_helpers::*;
        // x & x == x
        let op1 = operand_and(
            operand_register(4),
            operand_register(4),
        );
        assert_eq!(Operand::simplified(op1), operand_register(4));
    }

    #[test]
    fn simplify_and_constants() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffff),
            operand_and(
                constval(0xf3),
                operand_register(4),
            ),
        );
        let op2 = operand_and(
            constval(0xf3),
            operand_register(4),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(op2));
    }

    #[test]
    fn simplify_or() {
        use super::operand_helpers::*;
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        // mem8[x] | 0xff == 0xff
        let op1 = operand_or(
            mem8(operand_register(2)),
            constval(0xff),
        );
        assert_eq!(Operand::simplified(op1), constval(0xff));
        // (y == z) | 1 == 1
        let op1 = operand_or(
            operand_eq(
                operand_register(3),
                operand_register(4),
            ),
            constval(1),
        );
        assert_eq!(Operand::simplified(op1), constval(1));
    }

    #[test]
    fn simplify_xor() {
        use super::operand_helpers::*;
        // x ^ x ^ x == x
        let op1 = operand_xor(
            operand_register(1),
            operand_xor(
                operand_register(1),
                operand_register(1),
            ),
        );
        assert_eq!(Operand::simplified(op1), operand_register(1));
        let op1 = operand_xor(
            operand_register(1),
            operand_xor(
                operand_register(2),
                operand_register(1),
            ),
        );
        assert_eq!(Operand::simplified(op1), operand_register(2));
    }

    #[test]
    fn simplify_eq() {
        use super::operand_helpers::*;
        // Simplify (x == y) == 1 to x == y
        let op1 = operand_eq(constval(5), operand_register(2));
        let eq1 = operand_eq(constval(1), operand_eq(constval(5), operand_register(2)));
        // Simplify (x == y) == 0 == 0 to x == y
        let op2 = operand_eq(
            constval(0),
            operand_eq(
                constval(0),
                operand_eq(constval(5), operand_register(2)),
            ),
        );
        let eq2 = operand_eq(constval(5), operand_register(2));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_eq2() {
        use super::operand_helpers::*;
        // Simplify (x == y) == 0 == 0 == 0 to (x == y) == 0
        let op1 = operand_eq(
            constval(0),
            operand_eq(
                constval(0),
                operand_eq(
                    constval(0),
                    operand_eq(constval(5), operand_register(2)),
                ),
            ),
        );
        let eq1 = operand_eq(operand_eq(constval(5), operand_register(2)), constval(0));
        let ne1 = operand_eq(constval(5), operand_register(2));
        assert_eq!(Operand::simplified(op1.clone()), Operand::simplified(eq1));
        assert_ne!(Operand::simplified(op1), Operand::simplified(ne1));
    }

    #[test]
    fn simplify_gt() {
        use super::operand_helpers::*;
        let op1 = operand_gt(constval(4), constval(2));
        let op2 = operand_gt(constval(4), constval(!2));
        assert_eq!(Operand::simplified(op1), constval(1));
        assert_eq!(Operand::simplified(op2), constval(0));
    }

    #[test]
    fn simplify_const_shifts() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(constval(0x55), constval(0x4));
        let op2 = operand_rsh(constval(0x55), constval(0x4));
        let op3 = operand_and(
            operand_lsh(constval(0x55), constval(0x1f)),
            constval(0xffff_ffff),
        );
        let op4 = operand_lsh(constval(0x55), constval(0x1f));
        assert_eq!(Operand::simplified(op1), constval(0x550));
        assert_eq!(Operand::simplified(op2), constval(0x5));
        assert_eq!(Operand::simplified(op3), constval(0x8000_0000));
        assert_eq!(Operand::simplified(op4), constval(0x2a_8000_0000));
    }

    #[test]
    fn simplify_or_parts() {
        use super::operand_helpers::*;
        let op1 = operand_or(
            operand_and(
                mem32(operand_register(4)),
                constval(0xffff0000),
            ),
            operand_and(
                mem32(operand_register(4)),
                constval(0x0000ffff),
            )
        );
        let op2 = operand_or(
            operand_and(
                mem32(operand_register(4)),
                constval(0xffff00ff),
            ),
            operand_and(
                mem32(operand_register(4)),
                constval(0x0000ffff),
            )
        );
        let op3 = operand_or(
            operand_and(
                operand_register(4),
                constval(0x00ff00ff),
            ),
            operand_and(
                operand_register(4),
                constval(0x0000ffff),
            )
        );
        let eq3 = operand_and(
            operand_register(4),
            constval(0x00ffffff),
        );
        assert_eq!(Operand::simplified(op1), mem32(operand_register(4)));
        assert_eq!(Operand::simplified(op2), mem32(operand_register(4)));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
    }

    #[test]
    fn simplify_and_parts() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            operand_or(
                operand_register(4),
                constval(0xffff0000),
            ),
            operand_or(
                operand_register(4),
                constval(0x0000ffff),
            )
        );
        let op2 = operand_and(
            operand_or(
                operand_register(4),
                constval(0x00ff00ff),
            ),
            operand_or(
                operand_register(4),
                constval(0x0000ffff),
            )
        );
        let eq2 = operand_or(
            operand_register(4),
            constval(0x000000ff),
        );
        assert_eq!(Operand::simplified(op1), operand_register(4));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_lsh_or_rsh() {
        use super::operand_helpers::*;
        let op1 = operand_rsh(
            operand_or(
                operand_and(
                    operand_register(4),
                    constval(0xffff),
                ),
                operand_lsh(
                    operand_and(
                        operand_register(5),
                        constval(0xffff),
                    ),
                    constval(0x10),
                ),
            ),
            constval(0x10),
        );
        let eq1 = operand_and(
            operand_register(5),
            constval(0xffff),
        );
        let op2 = operand_rsh(
            operand_or(
                operand_and(
                    operand_register(4),
                    constval(0xffff),
                ),
                operand_or(
                    operand_lsh(
                        operand_and(
                            operand_register(5),
                            constval(0xffff),
                        ),
                        constval(0x10),
                    ),
                    operand_and(
                        operand_register(1),
                        constval(0xffff),
                    ),
                ),
            ),
            constval(0x10),
        );
        let eq2 = operand_and(
            operand_register(5),
            constval(0xffff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_and_or_bug() {
        fn operand_rol(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
            // rol(x, y) == (x << y) | (x >> (32 - y))
            operand_or(
                operand_lsh(lhs.clone(), rhs.clone()),
                operand_rsh(lhs, operand_sub(constval(32), rhs)),
            )
        }

        use super::operand_helpers::*;
        let op = operand_and(
            operand_or(
                operand_lsh(
                    operand_xor(
                        operand_rsh(
                            operand_register(1),
                            constval(0x10),
                        ),
                        operand_add(
                            operand_and(
                                constval(0xffff),
                                operand_register(1),
                            ),
                            operand_rol(
                                operand_and(
                                    constval(0xffff),
                                    operand_register(2),
                                ),
                                constval(1),
                            ),
                        ),
                    ),
                    constval(0x10),
                ),
                operand_and(
                    constval(0xffff),
                    operand_register(1),
                ),
            ),
            constval(0xffff),
        );
        let eq = operand_and(
            operand_register(1),
            constval(0xffff),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
    }

    #[test]
    fn simplify_pointless_and_masks() {
        use super::operand_helpers::*;
        let op = operand_and(
            operand_rsh(
                mem32(operand_register(1)),
                constval(0x10),
            ),
            constval(0xffff),
        );
        let eq = operand_rsh(
            mem32(operand_register(1)),
            constval(0x10),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
    }

    #[test]
    fn simplify_add_x_x() {
        use super::operand_helpers::*;
        let op = operand_add(
            operand_register(1),
            operand_register(1),
        );
        let eq = operand_mul(
            operand_register(1),
            constval(2),
        );
        let op2 = operand_add(
            operand_sub(
                operand_add(
                    operand_register(1),
                    operand_register(1),
                ),
                operand_register(1),
            ),
            operand_add(
                operand_register(1),
                operand_register(1),
            ),
        );
        let eq2 = operand_mul(
            operand_register(1),
            constval(3),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_add_x_x_64() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op = operand_add(
            ctx.register(1),
            ctx.register(1),
        );
        let eq = operand_mul(
            ctx.register(1),
            constval(2),
        );
        let neq = ctx.register(1);
        let op2 = operand_add(
            operand_sub(
                operand_add(
                    ctx.register(1),
                    ctx.register(1),
                ),
                ctx.register(1),
            ),
            operand_add(
                ctx.register(1),
                ctx.register(1),
            ),
        );
        let eq2 = operand_mul(
            ctx.register(1),
            constval(3),
        );
        assert_eq!(Operand::simplified(op.clone()), Operand::simplified(eq));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_ne!(Operand::simplified(op), Operand::simplified(neq));
    }

    #[test]
    fn simplify_and_xor_const() {
        use super::operand_helpers::*;
        let op = operand_and(
            constval(0xffff),
            operand_xor(
                constval(0x12345678),
                operand_register(1),
            ),
        );
        let eq = operand_and(
            constval(0xffff),
            operand_xor(
                constval(0x5678),
                operand_register(1),
            ),
        );
        let op2 = operand_and(
            constval(0xffff),
            operand_or(
                constval(0x12345678),
                operand_register(1),
            ),
        );
        let eq2 = operand_and(
            constval(0xffff),
            operand_or(
                constval(0x5678),
                operand_register(1),
            ),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_mem_access_and() {
        use super::operand_helpers::*;
        let op = operand_and(
            constval(0xffff),
            mem32(constval(0x123456)),
        );
        let eq = mem_variable_rc(MemAccessSize::Mem16, constval(0x123456));
        let op2 = operand_and(
            constval(0xfff),
            mem32(constval(0x123456)),
        );
        let eq2 = operand_and(
            constval(0xfff),
            mem_variable_rc(MemAccessSize::Mem16, constval(0x123456)),
        );
        assert_ne!(Operand::simplified(op2.clone()), Operand::simplified(eq.clone()));
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_and_or_bug2() {
        use super::operand_helpers::*;
        let op = operand_and(
            operand_or(
                constval(1),
                operand_and(
                    constval(0xffffff00),
                    operand_register(1),
                ),
            ),
            constval(0xff),
        );
        let ne = operand_and(
            operand_or(
                constval(1),
                operand_register(1),
            ),
            constval(0xff),
        );
        let eq = constval(1);
        assert_ne!(Operand::simplified(op.clone()), Operand::simplified(ne));
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
    }

    #[test]
    fn simplify_adjacent_ands_advanced() {
        use super::operand_helpers::*;
        let op = operand_and(
            constval(0xffff),
            operand_sub(
                operand_register(0),
                operand_or(
                    operand_and(
                        constval(0xff00),
                        operand_xor(
                            operand_xor(
                                constval(0x4200),
                                operand_register(1),
                            ),
                            mem_variable_rc(MemAccessSize::Mem16, operand_register(2)),
                        ),
                    ),
                    operand_and(
                        constval(0xff),
                        operand_xor(
                            operand_xor(
                                constval(0xa6),
                                operand_register(1),
                            ),
                            mem_variable_rc(MemAccessSize::Mem8, operand_register(2)),
                        ),
                    ),
                ),
            ),
        );
        let eq = operand_and(
            constval(0xffff),
            operand_sub(
                operand_register(0),
                operand_and(
                    constval(0xffff),
                    operand_xor(
                        operand_xor(
                            constval(0x42a6),
                            operand_register(1),
                        ),
                        mem_variable_rc(MemAccessSize::Mem16, operand_register(2)),
                    ),
                ),
            ),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
    }

    #[test]
    fn simplify_shifts() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(
            operand_rsh(
                operand_and(
                    operand_register(1),
                    constval(0xff00),
                ),
                constval(8),
            ),
            constval(8),
        );
        let eq1 = operand_and(
            operand_register(1),
            constval(0xff00),
        );
        let op2 = operand_rsh(
            operand_lsh(
                operand_and(
                    operand_register(1),
                    constval(0xff),
                ),
                constval(8),
            ),
            constval(8),
        );
        let eq2 = operand_and(
            operand_register(1),
            constval(0xff),
        );
        let op3 = operand_rsh(
            operand_lsh(
                operand_and(
                    operand_register(1),
                    constval(0xff),
                ),
                constval(8),
            ),
            constval(7),
        );
        Operand::simplified(op3.clone());
        let eq3 = operand_lsh(
            operand_and(
                operand_register(1),
                constval(0xff),
            ),
            constval(1),
        );
        let op4 = operand_rsh(
            operand_lsh(
                operand_and(
                    operand_register(1),
                    constval(0xff),
                ),
                constval(7),
            ),
            constval(8),
        );
        let eq4 = operand_rsh(
            operand_and(
                operand_register(1),
                constval(0xff),
            ),
            constval(1),
        );
        let op5 = operand_rsh(
            operand_and(
                mem32(operand_register(1)),
                constval(0xffff0000),
            ),
            constval(0x10),
        );
        let eq5 = operand_rsh(
            mem32(operand_register(1)),
            constval(0x10),
        );
        let op6 = operand_rsh(
            operand_and(
                mem32(operand_register(1)),
                constval(0xffff1234),
            ),
            constval(0x10),
        );
        let eq6 = operand_rsh(
            mem32(operand_register(1)),
            constval(0x10),
        );
        let op7 = operand_and(
            operand_lsh(
                operand_and(
                    mem32(constval(1)),
                    constval(0xffff),
                ),
                constval(0x10),
            ),
            constval(0xffff_ffff),
        );
        let eq7 = operand_and(
            operand_lsh(
                mem32(constval(1)),
                constval(0x10),
            ),
            constval(0xffff_ffff),
        );
        let op8 = operand_rsh(
            operand_and(
                operand_register(1),
                constval(0xffff_ffff_ffff_0000),
            ),
            constval(0x10),
        );
        let eq8 = operand_rsh(
            operand_register(1),
            constval(0x10),
        );
        let op9 = operand_rsh(
            operand_and(
                operand_register(1),
                constval(0xffff0000),
            ),
            constval(0x10),
        );
        let ne9 = operand_rsh(
            operand_register(1),
            constval(0x10),
        );
        let op10 = operand_rsh(
            operand_and(
                operand_register(1),
                constval(0xffff_ffff_ffff_1234),
            ),
            constval(0x10),
        );
        let eq10 = operand_rsh(
            operand_register(1),
            constval(0x10),
        );
        let op11 = operand_rsh(
            operand_and(
                operand_register(1),
                constval(0xffff_1234),
            ),
            constval(0x10),
        );
        let ne11 = operand_rsh(
            operand_register(1),
            constval(0x10),
        );
        let op12 = operand_lsh(
            operand_and(
                mem32(constval(1)),
                constval(0xffff),
            ),
            constval(0x10),
        );
        let ne12 = operand_lsh(
            mem32(constval(1)),
            constval(0x10),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
        assert_eq!(Operand::simplified(op4), Operand::simplified(eq4));
        assert_eq!(Operand::simplified(op5), Operand::simplified(eq5));
        assert_eq!(Operand::simplified(op6), Operand::simplified(eq6));
        assert_eq!(Operand::simplified(op7), Operand::simplified(eq7));
        assert_eq!(Operand::simplified(op8), Operand::simplified(eq8));
        assert_ne!(Operand::simplified(op9), Operand::simplified(ne9));
        assert_eq!(Operand::simplified(op10), Operand::simplified(eq10));
        assert_ne!(Operand::simplified(op11), Operand::simplified(ne11));
        assert_ne!(Operand::simplified(op12), Operand::simplified(ne12));
    }

    #[test]
    fn simplify_mem_zero_bits() {
        let mem16 = |x| mem_variable_rc(MemAccessSize::Mem16, x);
        use super::operand_helpers::*;
        let op1 = operand_rsh(
            operand_or(
                mem16(operand_register(0)),
                operand_lsh(
                    mem16(operand_register(1)),
                    constval(0x10),
                ),
            ),
            constval(0x10),
        );
        let eq1 = mem16(operand_register(1));
        let op2 = operand_and(
            operand_or(
                mem16(operand_register(0)),
                operand_lsh(
                    mem16(operand_register(1)),
                    constval(0x10),
                ),
            ),
            constval(0xffff0000),
        );
        let eq2 = operand_lsh(
            mem16(operand_register(1)),
            constval(0x10),
        );
        let op3 = operand_and(
            operand_or(
                mem16(operand_register(0)),
                operand_lsh(
                    mem16(operand_register(1)),
                    constval(0x10),
                ),
            ),
            constval(0xffff),
        );
        let eq3 = mem16(operand_register(0));
        let op4 = operand_or(
            operand_or(
                mem16(operand_register(0)),
                operand_lsh(
                    mem16(operand_register(1)),
                    constval(0x10),
                ),
            ),
            constval(0xffff0000),
        );
        let eq4 = operand_or(
            mem16(operand_register(0)),
            constval(0xffff0000),
        );

        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
        assert_eq!(Operand::simplified(op4), Operand::simplified(eq4));
    }

    #[test]
    fn simplify_mem_16_hi_or_mem8() {
        let mem16 = |x| mem_variable_rc(MemAccessSize::Mem16, x);
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        use super::operand_helpers::*;
        let op1 = operand_or(
            mem8(operand_register(1)),
            operand_and(
                mem16(operand_register(1)),
                constval(0xff00),
            ),
        );
        let eq1 = mem16(operand_register(1));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_xor_and() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0x7ffffff0),
            operand_xor(
                operand_and(
                    constval(0x7ffffff0),
                    operand_register(0),
                ),
                operand_register(1),
            ),
        );
        let eq1 = operand_and(
            constval(0x7ffffff0),
            operand_xor(
                operand_register(0),
                operand_register(1),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_large_and_xor_chain() {
        // Check that this executes in reasonable time
        use super::OperandContext;
        use super::operand_helpers::*;

        let ctx = OperandContext::new();
        let mut chain = ctx.undefined_rc();
        for _ in 0..20 {
            chain = operand_xor(
                operand_and(
                    constval(0x7fffffff),
                    chain.clone(),
                ),
                operand_and(
                    constval(0x7fffffff),
                    operand_xor(
                        operand_and(
                            constval(0x7fffffff),
                            chain.clone(),
                        ),
                        ctx.undefined_rc(),
                    ),
                ),
            );
            chain = Operand::simplified(chain);
            Operand::simplified(
                operand_rsh(
                    chain.clone(),
                    constval(1),
                ),
            );
        }
    }

    #[test]
    fn simplify_merge_adds_as_mul() {
        use super::operand_helpers::*;
        let op = operand_add(
            operand_mul(
                operand_register(1),
                constval(2),
            ),
            operand_register(1),
        );
        let eq = operand_mul(
            operand_register(1),
            constval(3),
        );
        let op2 = operand_add(
            operand_mul(
                operand_register(1),
                constval(2),
            ),
            operand_mul(
                operand_register(1),
                constval(8),
            ),
        );
        let eq2 = operand_mul(
            operand_register(1),
            constval(10),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_merge_and_xor() {
        use super::operand_helpers::*;
        let mem16 = |x| mem_variable_rc(MemAccessSize::Mem16, x);
        let op = operand_or(
            operand_and(
                operand_xor(
                    operand_xor(
                        mem32(constval(1234)),
                        constval(0x1123),
                    ),
                    mem32(constval(3333)),
                ),
                constval(0xff00),
            ),
            operand_and(
                operand_xor(
                    operand_xor(
                        mem32(constval(1234)),
                        constval(0x666666),
                    ),
                    mem32(constval(3333)),
                ),
                constval(0xff),
            ),
        );
        let eq = operand_and(
            constval(0xffff),
            operand_xor(
                operand_xor(
                    mem16(constval(1234)),
                    mem16(constval(3333)),
                ),
                constval(0x1166),
            ),
        );
        assert_eq!(Operand::simplified(op), Operand::simplified(eq));
    }

    #[test]
    fn simplify_and_or_const() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            operand_or(
                constval(0x38),
                operand_register(1),
            ),
            constval(0x28),
        );
        let eq1 = constval(0x28);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_sub_eq_zero() {
        use super::operand_helpers::*;
        // 0 == (x - y) is same as x == y
        let op1 = operand_eq(
            constval(0),
            operand_sub(
                mem32(operand_register(1)),
                mem32(operand_register(2)),
            ),
        );
        let eq1 = operand_eq(
            mem32(operand_register(1)),
            mem32(operand_register(2)),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_x_eq_x_add() {
        use super::operand_helpers::*;
        // The register 2 can be ignored as there is no way for the addition to cause lowest
        // byte to be equal to what it was. If the constant addition were higher than 0xff,
        // then it couldn't be simplified (effectively the high unknown is able to cause unknown
        // amount of reduction in the constant's effect, but looping the lowest byte around
        // requires a multiple of 0x100 to be added)
        let op1 = operand_eq(
            operand_and(
                operand_or(
                    operand_and(
                        operand_register(2),
                        constval(0xffffff00),
                    ),
                    operand_and(
                        operand_register(1),
                        constval(0xff),
                    ),
                ),
                constval(0xff),
            ),
            operand_and(
                operand_add(
                    operand_or(
                        operand_and(
                            operand_register(2),
                            constval(0xffffff00),
                        ),
                        operand_and(
                            operand_register(1),
                            constval(0xff),
                        ),
                    ),
                    constval(1),
                ),
                constval(0xff),
            ),
        );
        let eq1 = constval(0);
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        let op2 = operand_eq(
            mem8(constval(555)),
            operand_and(
                operand_add(
                    operand_or(
                        operand_and(
                            operand_register(2),
                            constval(0xffffff00),
                        ),
                        mem8(constval(555)),
                    ),
                    constval(1),
                ),
                constval(0xff),
            ),
        );
        let eq2 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_overflowing_shifts() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(
            operand_rsh(
                operand_register(1),
                constval(0x55),
            ),
            constval(0x22),
        );
        let eq1 = constval(0);
        let op2 = operand_rsh(
            operand_lsh(
                operand_register(1),
                constval(0x55),
            ),
            constval(0x22),
        );
        let eq2 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_and_not_mem32() {
        use super::operand_helpers::*;
        let mem16 = |x| mem_variable_rc(MemAccessSize::Mem16, x);
        let op1 = operand_and(
            operand_not(
                mem32(constval(0x123)),
            ),
            constval(0xffff),
        );
        let eq1 = operand_and(
            operand_not(
                mem16(constval(0x123)),
            ),
            constval(0xffff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_eq_consts() {
        use super::operand_helpers::*;
        let op1 = operand_eq(
            constval(0),
            operand_and(
                operand_add(
                    constval(1),
                    operand_register(1),
                ),
                constval(0xffffffff),
            ),
        );
        let eq1 = operand_eq(
            constval(0xffffffff),
            operand_and(
                operand_register(1),
                constval(0xffffffff),
            ),
        );
        let op2 = operand_eq(
            constval(0),
            operand_add(
                constval(1),
                operand_register(1),
            ),
        );
        let eq2 = operand_eq(
            constval(0xffff_ffff_ffff_ffff),
            operand_register(1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_add_mul() {
        use super::operand_helpers::*;
        let op1 = operand_mul(
            constval(4),
            operand_add(
                constval(5),
                operand_mul(
                    operand_register(0),
                    constval(3),
                ),
            ),
        );
        let eq1 = operand_add(
            constval(20),
            operand_mul(
                operand_register(0),
                constval(12),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_bool_oper() {
        use super::operand_helpers::*;
        let op1 = operand_eq(
            constval(0),
            operand_eq(
                operand_gt(
                    operand_register(0),
                    operand_register(1),
                ),
                constval(0),
            ),
        );
        let eq1 = operand_gt(
            operand_register(0),
            operand_register(1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_gt2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            operand_sub(
                ctx.register(5),
                ctx.register(2),
            ),
            ctx.register(5),
        );
        let eq1 = operand_gt(
            ctx.register(2),
            ctx.register(5),
        );
        // Checking for signed gt requires sign == overflow, unlike
        // unsigned where it's just carry == 1
        let op2 = operand_gt(
            operand_and(
                operand_add(
                    operand_sub(
                        ctx.register(5),
                        ctx.register(2),
                    ),
                    ctx.constant(0x8000_0000),
                ),
                ctx.constant(0xffff_ffff),
            ),
            operand_and(
                operand_add(
                    ctx.register(5),
                    ctx.constant(0x8000_0000),
                ),
                ctx.constant(0xffff_ffff),
            ),
        );
        let ne2 = operand_gt(
            operand_and(
                operand_add(
                    ctx.register(2),
                    ctx.constant(0x8000_0000),
                ),
                ctx.constant(0xffff_ffff),
            ),
            operand_and(
                operand_add(
                    ctx.register(5),
                    ctx.constant(0x8000_0000),
                ),
                ctx.constant(0xffff_ffff),
            ),
        );
        let op3 = operand_gt(
            operand_sub(
                ctx.register(5),
                ctx.register(2),
            ),
            ctx.register(5),
        );
        let eq3 = operand_gt(
            ctx.register(2),
            ctx.register(5),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_ne!(Operand::simplified(op2), Operand::simplified(ne2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
    }

    #[test]
    fn simplify_mem32_rsh() {
        use super::operand_helpers::*;
        let mem16 = |x| mem_variable_rc(MemAccessSize::Mem16, x);
        let op1 = operand_rsh(
            mem32(constval(0x123)),
            constval(0x10),
        );
        let eq1 = mem16(constval(0x125));
        let op2 = operand_rsh(
            mem32(constval(0x123)),
            constval(0x11),
        );
        let eq2 = operand_rsh(
            mem16(constval(0x125)),
            constval(0x1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_mem_or() {
        use super::operand_helpers::*;
        let mem16 = |x| mem_variable_rc(MemAccessSize::Mem16, x);
        let op1 = operand_or(
            operand_rsh(
                mem32(
                    operand_add(
                        operand_register(0),
                        constval(0x120),
                    ),
                ),
                constval(0x8),
            ),
            operand_lsh(
                mem32(
                    operand_add(
                        operand_register(0),
                        constval(0x124),
                    ),
                ),
                constval(0x18),
            ),
        );
        let eq1 = mem32(
            operand_add(
                operand_register(0),
                constval(0x121),
            ),
        );
        let op2 = operand_or(
            mem16(
                operand_add(
                    operand_register(0),
                    constval(0x122),
                ),
            ),
            operand_lsh(
                mem16(
                    operand_add(
                        operand_register(0),
                        constval(0x124),
                    ),
                ),
                constval(0x10),
            ),
        );
        let eq2 = mem32(
            operand_add(
                operand_register(0),
                constval(0x122),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_rsh_and() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffff),
            operand_rsh(
                operand_or(
                    constval(0x123400),
                    operand_and(
                        operand_register(1),
                        constval(0xff000000),
                    ),
                ),
                constval(8),
            ),
        );
        let eq1 = constval(0x1234);
        let op2 = operand_and(
            constval(0xffff0000),
            operand_lsh(
                operand_or(
                    constval(0x123400),
                    operand_and(
                        operand_register(1),
                        constval(0xff),
                    ),
                ),
                constval(8),
            ),
        );
        let eq2 = constval(0x12340000);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_mem32_or() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            operand_or(
                operand_lsh(
                    operand_register(2),
                    constval(0x18),
                ),
                operand_rsh(
                    operand_or(
                        constval(0x123400),
                        operand_and(
                            mem32(operand_register(1)),
                            constval(0xff000000),
                        ),
                    ),
                    constval(8),
                ),
            ),
            constval(0xffff),
        );
        let eq1 = constval(0x1234);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_mem_bug() {
        use super::operand_helpers::*;
        let op1 = operand_or(
            operand_rsh(
                constval(0),
                constval(0x10),
            ),
            operand_and(
                operand_lsh(
                    operand_lsh(
                        operand_and(
                            operand_add(
                                constval(0x20),
                                operand_register(4),
                            ),
                            constval(0xffff_ffff),
                        ),
                        constval(0x10),
                    ),
                    constval(0x10),
                ),
                constval(0xffff_ffff),
            ),
        );
        let eq1 = constval(0);
        let op2 = operand_or(
            operand_rsh(
                constval(0),
                constval(0x10),
            ),
            operand_lsh(
                operand_lsh(
                    operand_add(
                        constval(0x20),
                        operand_register(4),
                    ),
                    constval(0x20),
                ),
                constval(0x20),
            ),
        );
        let eq2 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_and_or_rsh() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffffff00),
            operand_or(
                operand_rsh(
                    mem32(operand_register(1)),
                    constval(0x18),
                ),
                operand_rsh(
                    mem32(operand_register(4)),
                    constval(0x18),
                ),
            ),
        );
        let eq1 = constval(0);
        let op2 = operand_and(
            constval(0xffffff00),
            operand_or(
                operand_rsh(
                    operand_register(1),
                    constval(0x18),
                ),
                operand_rsh(
                    operand_register(4),
                    constval(0x18),
                ),
            ),
        );
        let ne2 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_ne!(Operand::simplified(op2), Operand::simplified(ne2));
    }

    #[test]
    fn simplify_and_or_rsh_mul() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xff000000),
            operand_or(
                constval(0xfe000000),
                operand_rsh(
                    operand_and(
                        operand_mul(
                            operand_register(2),
                            operand_register(1),
                        ),
                        constval(0xffff_ffff),
                    ),
                    constval(0x18),
                ),
            ),
        );
        let eq1 = constval(0xfe000000);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_mem_misalign2() {
        use super::operand_helpers::*;
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        let op1 = operand_or(
            operand_rsh(
                mem32(
                    operand_register(1),
                ),
                constval(0x8),
            ),
            operand_lsh(
                mem8(
                    operand_add(
                        constval(0x4),
                        operand_register(1),
                    ),
                ),
                constval(0x18),
            ),
        );
        let eq1 = mem32(
            operand_add(
                operand_register(1),
                constval(1),
            ),
        );
        let op2 = operand_or(
            operand_rsh(
                mem32(
                    operand_sub(
                        operand_register(1),
                        constval(0x4),
                    ),
                ),
                constval(0x8),
            ),
            operand_lsh(
                mem8(
                    operand_register(1),
                ),
                constval(0x18),
            ),
        );
        let eq2 = mem32(
            operand_sub(
                operand_register(1),
                constval(3),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_and_shift_overflow_bug() {
        use super::operand_helpers::*;
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        let op1 = operand_and(
            operand_or(
                operand_rsh(
                    operand_or(
                        operand_rsh(
                            mem8(operand_register(1)),
                            constval(7),
                        ),
                        operand_and(
                            constval(0xff000000),
                            operand_lsh(
                                mem8(operand_register(2)),
                                constval(0x11),
                            ),
                        ),
                    ),
                    constval(0x10),
                ),
                operand_lsh(
                    mem32(operand_register(4)),
                    constval(0x10),
                ),
            ),
            constval(0xff),
        );
        let eq1 = operand_and(
            operand_rsh(
                operand_or(
                    operand_rsh(
                        mem8(operand_register(1)),
                        constval(7),
                    ),
                    operand_and(
                        constval(0xff000000),
                        operand_lsh(
                            mem8(operand_register(2)),
                            constval(0x11),
                        ),
                    ),
                ),
                constval(0x10),
            ),
            constval(0xff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_mul_add() {
        use super::operand_helpers::*;
        let op1 = operand_mul(
            constval(0xc),
            operand_add(
                constval(0xc),
                operand_register(1),
            ),
        );
        let eq1 = operand_add(
            constval(0x90),
            operand_mul(
                constval(0xc),
                operand_register(1),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_masks() {
        use super::operand_helpers::*;
        // One and can be removed since zext(u8) + zext(u8) won't overflow u32
        let op1 = operand_and(
            constval(0xff),
            operand_add(
                operand_and(
                    constval(0xff),
                    operand_register(1),
                ),
                operand_and(
                    constval(0xff),
                    operand_add(
                        operand_and(
                            constval(0xff),
                            operand_register(1),
                        ),
                        operand_and(
                            constval(0xff),
                            operand_add(
                                operand_register(4),
                                operand_and(
                                    constval(0xff),
                                    operand_register(1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let eq1 = operand_and(
            constval(0xff),
            operand_add(
                operand_and(
                    constval(0xff),
                    operand_register(1),
                ),
                operand_add(
                    operand_and(
                        constval(0xff),
                        operand_register(1),
                    ),
                    operand_and(
                        constval(0xff),
                        operand_add(
                            operand_register(4),
                            operand_and(
                                constval(0xff),
                                operand_register(1),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let eq1b = operand_and(
            constval(0xff),
            operand_add(
                operand_mul(
                    constval(2),
                    operand_and(
                        constval(0xff),
                        operand_register(1),
                    ),
                ),
                operand_and(
                    constval(0xff),
                    operand_add(
                        operand_register(4),
                        operand_and(
                            constval(0xff),
                            operand_register(1),
                        ),
                    ),
                ),
            ),
        );

        let op2 = operand_and(
            constval(0x3fffffff),
            operand_add(
                operand_and(
                    constval(0x3fffffff),
                    operand_register(1),
                ),
                operand_and(
                    constval(0x3fffffff),
                    operand_add(
                        operand_and(
                            constval(0x3fffffff),
                            operand_register(1),
                        ),
                        operand_and(
                            constval(0x3fffffff),
                            operand_add(
                                operand_register(4),
                                operand_and(
                                    constval(0x3fffffff),
                                    operand_register(1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let eq2 = operand_and(
            constval(0x3fffffff),
            operand_add(
                operand_and(
                    constval(0x3fffffff),
                    operand_register(1),
                ),
                operand_add(
                    operand_and(
                        constval(0x3fffffff),
                        operand_register(1),
                    ),
                    operand_and(
                        constval(0x3fffffff),
                        operand_add(
                            operand_register(4),
                            operand_and(
                                constval(0x3fffffff),
                                operand_register(1),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let op3 = operand_and(
            constval(0x7fffffff),
            operand_add(
                operand_and(
                    constval(0x7fffffff),
                    operand_register(1),
                ),
                operand_and(
                    constval(0x7fffffff),
                    operand_add(
                        operand_and(
                            constval(0x7fffffff),
                            operand_register(1),
                        ),
                        operand_and(
                            constval(0x7fffffff),
                            operand_add(
                                operand_register(4),
                                operand_and(
                                    constval(0x7fffffff),
                                    operand_register(1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let eq3 = operand_and(
            constval(0x7fffffff),
            operand_add(
                operand_and(
                    constval(0x7fffffff),
                    operand_register(1),
                ),
                operand_add(
                    operand_and(
                        constval(0x7fffffff),
                        operand_register(1),
                    ),
                    operand_and(
                        constval(0x7fffffff),
                        operand_add(
                            operand_register(4),
                            operand_and(
                                constval(0x7fffffff),
                                operand_register(1),
                            ),
                        ),
                    ),
                ),
            ),
        );
        assert_eq!(Operand::simplified(op1.clone()), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1b));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
    }

    #[test]
    fn simplify_and_masks2() {
        use super::operand_helpers::*;
        // One and can be removed since zext(u8) + zext(u8) won't overflow u32
        let op1 = operand_and(
            constval(0xff),
            operand_add(
                operand_mul(
                    constval(2),
                    operand_and(
                        constval(0xff),
                        operand_register(1),
                    ),
                ),
                operand_and(
                    constval(0xff),
                    operand_add(
                        operand_mul(
                            constval(2),
                            operand_and(
                                constval(0xff),
                                operand_register(1),
                            ),
                        ),
                        operand_and(
                            constval(0xff),
                            operand_add(
                                operand_register(4),
                                operand_and(
                                    constval(0xff),
                                    operand_register(1),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        );

        let eq1 = operand_and(
            constval(0xff),
            operand_add(
                operand_mul(
                    constval(4),
                    operand_and(
                        constval(0xff),
                        operand_register(1),
                    ),
                ),
                operand_and(
                    constval(0xff),
                    operand_add(
                        operand_register(4),
                        operand_and(
                            constval(0xff),
                            operand_register(1),
                        ),
                    ),
                ),
            ),
        );

        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_xor_and_xor() {
        use super::operand_helpers::*;
        // c1 ^ ((x ^ c1) & c2) == x & c2 if c2 & c1 == c1
        // (Effectively going to transform c1 ^ (y & c2) == (y ^ (c1 & c2)) & c2)
        let op1 = operand_xor(
            constval(0x423),
            operand_and(
                constval(0xfff),
                operand_xor(
                    constval(0x423),
                    operand_register(1),
                ),
            ),
        );
        let eq1 = operand_and(
            constval(0xfff),
            operand_register(1),
        );

        let op2 = operand_xor(
            constval(0x423),
            operand_or(
                operand_and(
                    constval(0xfff),
                    operand_xor(
                        constval(0x423),
                        mem32(operand_register(1)),
                    ),
                ),
                operand_and(
                    constval(0xffff_f000),
                    mem32(operand_register(1)),
                ),
            )
        );
        let eq2 = mem32(operand_register(1));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_or_mem_bug2() {
        use super::operand_helpers::*;
        let op = operand_or(
            operand_and(
                operand_rsh(
                    mem32(operand_sub(operand_register(2), constval(0x1))),
                    constval(8),
                ),
                constval(0x00ff_ffff),
            ),
            operand_and(
                mem32(operand_sub(operand_register(2), constval(0x14))),
                constval(0xff00_0000),
            ),
        );
        // Just checking that this doesn't panic
        let _ = Operand::simplified(op);
    }

    #[test]
    fn simplify_panic() {
        use super::operand_helpers::*;
        let mem8 = |x| mem_variable_rc(MemAccessSize::Mem8, x);
        let op1 = operand_and(
            constval(0xff),
            operand_rsh(
                operand_add(
                    constval(0x1d),
                    operand_eq(
                        operand_eq(
                            operand_and(
                                constval(1),
                                mem8(operand_register(3)),
                            ),
                            constval(0),
                        ),
                        constval(0),
                    ),
                ),
                constval(8),
            ),
        );
        let eq1 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn shift_xor_parts() {
        use super::operand_helpers::*;
        let op1 = operand_rsh(
            operand_xor(
                constval(0xffe60000),
                operand_xor(
                    operand_lsh(mem16(operand_register(5)), constval(0x10)),
                    mem32(operand_register(5)),
                ),
            ),
            constval(0x10),
        );
        let eq1 = operand_xor(
            constval(0xffe6),
            operand_xor(
                mem16(operand_register(5)),
                operand_rsh(mem32(operand_register(5)), constval(0x10)),
            ),
        );
        let op2 = operand_lsh(
            operand_xor(
                constval(0xffe6),
                mem16(operand_register(5)),
            ),
            constval(0x10),
        );
        let eq2 = operand_xor(
            constval(0xffe60000),
            operand_lsh(mem16(operand_register(5)), constval(0x10)),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn lea_mul_9() {
        use super::operand_helpers::*;
        let base = Operand::simplified(operand_add(
            constval(0xc),
            operand_and(
                constval(0xffff_ff7f),
                mem32(operand_register(1)),
            ),
        ));
        let op1 = operand_add(
            base.clone(),
            operand_mul(
                base.clone(),
                constval(8),
            ),
        );
        let eq1 = operand_mul(
            base,
            constval(9),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn lsh_mul() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(
            operand_mul(
                constval(0x9),
                mem32(operand_register(1)),
            ),
            constval(0x2),
        );
        let eq1 = operand_mul(
            mem32(operand_register(1)),
            constval(0x24),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn lea_mul_negative() {
        use super::operand_helpers::*;
        let base = Operand::simplified(operand_sub(
            mem16(operand_register(3)),
            constval(1),
        ));
        let op1 = operand_add(
            constval(0x1234),
            operand_mul(
                base,
                constval(0x4),
            ),
        );
        let eq1 = operand_add(
            constval(0x1230),
            operand_mul(
                mem16(operand_register(3)),
                constval(0x4),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn and_u32_max() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffff_ffff),
            constval(0xffff_ffff),
        );
        let eq1 = constval(0xffff_ffff);
        let op2 = operand_and(
            constval(!0),
            constval(!0),
        );
        let eq2 = constval(!0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn and_64() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffff_ffff_ffff),
            constval(0x12456),
        );
        let eq1 = constval(0x12456);
        let op2 = operand_and(
            mem32(operand_register(0)),
            constval(!0),
        );
        let eq2 = mem32(operand_register(0));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn short_and_is_32() {
        use super::operand_helpers::*;
        let op1 = Operand::simplified(operand_and(
            mem32(operand_register(0)),
            mem32(operand_register(1)),
        ));
        match op1.ty {
            OperandType::Arithmetic(..) => (),
            _ => panic!("Simplified was {}", op1),
        }
    }

    #[test]
    fn and_32bit() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffff_ffff),
            mem32(operand_register(1)),
        );
        let eq1 = mem32(operand_register(1));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn mem8_mem32_shift_eq() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xff),
            operand_rsh(
                mem32(operand_add(operand_register(1), constval(0x4c))),
                constval(0x8),
            ),
        );
        let eq1 = mem8(operand_add(operand_register(1), constval(0x4d)));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn or_64() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = Operand::simplified(operand_or(
            constval(0xffff_0000_0000),
            constval(0x12456),
        ));
        let eq1 = constval(0xffff_0001_2456);
        let op2 = Operand::simplified(operand_or(
            ctx.register(0),
            constval(0),
        ));
        let eq2 = ctx.register(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn lsh_64() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(
            constval(0x4),
            constval(0x28),
        );
        let eq1 = constval(0x0000_0400_0000_0000);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn xor_64() {
        use super::operand_helpers::*;
        let op1 = operand_xor(
            constval(0x4000_0000_0000),
            constval(0x6000_0000_0000),
        );
        let eq1 = constval(0x2000_0000_0000);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn eq_64() {
        use super::operand_helpers::*;
        let op1 = operand_eq(
            constval(0x40),
            constval(0x00),
        );
        let eq1 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn and_bug_64() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            operand_and(
                constval(0xffff_ffff),
                operand_rsh(
                    mem8(
                        operand_add(
                            constval(0xf105b2a),
                            operand_and(
                                constval(0xffff_ffff),
                                operand_add(
                                    operand_register(1),
                                    constval(0xd6057390),
                                ),
                            ),
                        ),
                    ),
                    constval(0xffffffffffffffda),
                ),
            ),
            constval(0xff),
        );
        let eq1 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_gt_or_eq() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            operand_gt(
                constval(0x5),
                operand_register(1),
            ),
            operand_eq(
                constval(0x5),
                operand_register(1),
            ),
        );
        let eq1 = operand_gt(
            constval(0x6),
            operand_register(1),
        );
        let op2 = operand_or(
            operand_gt(
                constval(0x5),
                ctx.register(1),
            ),
            operand_eq(
                constval(0x5),
                ctx.register(1),
            ),
        );
        // Confirm that 6 > rcx isn't 6 > ecx
        let ne2 = operand_gt(
            constval(0x6),
            operand_and(
                ctx.register(1),
                constval(0xffff_ffff),
            ),
        );
        let eq2 = operand_gt(
            constval(0x6),
            ctx.register(1),
        );
        let op3 = operand_or(
            operand_gt(
                constval(0x5_0000_0000),
                ctx.register(1),
            ),
            operand_eq(
                constval(0x5_0000_0000),
                ctx.register(1),
            ),
        );
        let eq3 = operand_gt(
            constval(0x5_0000_0001),
            ctx.register(1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_ne!(Operand::simplified(op2.clone()), Operand::simplified(ne2));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
    }

    #[test]
    fn pointless_gt() {
        use super::operand_helpers::*;
        let op1 = operand_gt(
            constval(0),
            operand_register(0),
        );
        let eq1 = constval(0);
        let op2 = operand_gt(
            operand_and(
                operand_register(0),
                constval(0xffff_ffff),
            ),
            constval(u32::max_value() as u64),
        );
        let eq2 = constval(0);
        let op3 = operand_gt(
            operand_register(0),
            constval(u64::max_value()),
        );
        let eq3 = constval(0);
        let op4 = operand_gt(
            operand_register(0),
            constval(u32::max_value() as u64),
        );
        let ne4 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
        assert_ne!(Operand::simplified(op4), Operand::simplified(ne4));
    }

    #[test]
    fn and_64_to_32() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_register(0),
            constval(0xf9124),
        );
        let eq1 = operand_and(
            operand_register(0),
            constval(0xf9124),
        );
        let op2 = operand_and(
            operand_add(
                ctx.register(0),
                ctx.register(2),
            ),
            constval(0xf9124),
        );
        let eq2 = operand_and(
            operand_add(
                operand_register(0),
                operand_register(2),
            ),
            constval(0xf9124),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_bug_xor_and_u32_max() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let unk = ctx.undefined_rc();
        let op1 = operand_xor(
            operand_and(
                mem32(unk.clone()),
                constval(0xffff_ffff),
            ),
            constval(0xffff_ffff),
        );
        let eq1 = operand_xor(
            mem32(unk.clone()),
            constval(0xffff_ffff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn operand_iter() {
        use std::collections::HashSet;

        use super::operand_helpers::*;
        let oper = operand_and(
            operand_sub(
                constval(5),
                operand_register(6),
            ),
            operand_eq(
                constval(77),
                operand_register(4),
            ),
        );
        let opers = [
            oper.clone(),
            operand_sub(constval(5), operand_register(6)),
            constval(5),
            operand_register(6),
            operand_eq(constval(77), operand_register(4)),
            constval(77),
            operand_register(4),
        ];
        let mut seen = HashSet::new();
        for o in oper.iter() {
            assert!(!seen.contains(o));
            seen.insert(o);
        }
        for o in &opers {
            assert!(seen.contains(&**o));
        }
        assert_eq!(seen.len(), opers.len());
    }

    #[test]
    fn simplify_eq_64_to_32() {
        use super::operand_helpers::*;
        let op1 = operand_eq(
            operand_register(0),
            constval(0),
        );
        let eq1 = operand_eq(
            operand_register(0),
            constval(0),
        );
        let op2 = operand_eq(
            operand_register(0),
            operand_register(2),
        );
        let eq2 = operand_eq(
            operand_register(0),
            operand_register(2),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_read_middle_u16_from_mem32() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xffff),
            operand_rsh(
                mem32(constval(0x11230)),
                constval(8),
            ),
        );
        let eq1 = mem16(constval(0x11231));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_unnecessary_shift_in_eq_zero() {
        use super::operand_helpers::*;
        let op1 = operand_eq(
            operand_lsh(
                operand_and(
                    mem8(operand_register(4)),
                    constval(8),
                ),
                constval(0xc),
            ),
            constval(0),
        );
        let eq1 = operand_eq(
            operand_and(
                mem8(operand_register(4)),
                constval(8),
            ),
            constval(0),
        );
        let op2 = operand_eq(
            operand_rsh(
                operand_and(
                    mem8(operand_register(4)),
                    constval(8),
                ),
                constval(1),
            ),
            constval(0),
        );
        let eq2 = operand_eq(
            operand_and(
                mem8(operand_register(4)),
                constval(8),
            ),
            constval(0),
        );
        let op3 = operand_eq(
            operand_and(
                mem8(operand_register(4)),
                constval(8),
            ),
            constval(0),
        );
        let ne3 = operand_eq(
            mem8(operand_register(4)),
            constval(0),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_ne!(Operand::simplified(op3), Operand::simplified(ne3));
    }

    #[test]
    fn simplify_unnecessary_and_in_shifts() {
        use super::operand_helpers::*;
        let op1 = operand_rsh(
            operand_and(
                operand_lsh(
                    mem8(constval(0x100)),
                    constval(0xd),
                ),
                constval(0x1f0000),
            ),
            constval(0x10),
        );
        let eq1 = operand_rsh(
            mem8(constval(0x100)),
            constval(0x3),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_set_bit_masked() {
        use super::operand_helpers::*;
        let op1 = operand_or(
            operand_and(
                mem16(constval(0x1000)),
                constval(0xffef),
            ),
            constval(0x10),
        );
        let eq1 = operand_or(
            mem16(constval(0x1000)),
            constval(0x10),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_masked_mul_lsh() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(
            operand_and(
                operand_mul(
                    mem32(constval(0x1000)),
                    constval(9),
                ),
                constval(0x3fff_ffff),
            ),
            constval(0x2),
        );
        let eq1 = operand_and(
            operand_mul(
                mem32(constval(0x1000)),
                constval(0x24),
            ),
            constval(0xffff_ffff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_inner_masks_on_arith() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            ctx.constant(0xff),
            operand_add(
                ctx.register(4),
                operand_and(
                    ctx.constant(0xff),
                    ctx.register(1),
                ),
            ),
        );
        let eq1 = operand_and(
            ctx.constant(0xff),
            operand_add(
                ctx.register(4),
                ctx.register(1),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_add_to_const_0() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_add(
            operand_add(
                ctx.constant(1),
                mem32(ctx.constant(0x5000)),
            ),
            ctx.constant(u64::max_value()),
        );
        let eq1 = mem32(ctx.constant(0x5000));
        let op2 = operand_and(
            operand_add(
                operand_add(
                    ctx.constant(1),
                    mem32(ctx.constant(0x5000)),
                ),
                ctx.constant(0xffff_ffff),
            ),
            ctx.constant(0xffff_ffff),
        );
        let eq2 = mem32(ctx.constant(0x5000));
        let op3 = operand_and(
            operand_add(
                operand_add(
                    ctx.constant(1),
                    mem32(ctx.constant(0x5000)),
                ),
                ctx.constant(0xffff_ffff),
            ),
            ctx.constant(0xffff_ffff),
        );
        let eq3 = mem32(ctx.constant(0x5000));
        let op4 = operand_add(
            operand_add(
                ctx.constant(1),
                mem32(ctx.constant(0x5000)),
            ),
            ctx.constant(0xffff_ffff_ffff_ffff),
        );
        let eq4 = mem32(ctx.constant(0x5000));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
        assert_eq!(Operand::simplified(op4), Operand::simplified(eq4));
    }

    #[test]
    fn simplify_sub_self_masked() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let ud = ctx.undefined_rc() ;
        let op1 = operand_sub(
            operand_and(
                ud.clone(),
                ctx.const_ffffffff(),
            ),
            operand_and(
                ud.clone(),
                ctx.const_ffffffff(),
            ),
        );
        let eq1 = ctx.const_0();
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_rsh() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_rsh(
            operand_and(
                mem8(ctx.constant(0x900)),
                ctx.constant(0xf8),
            ),
            ctx.constant(3),
        );
        let eq1 = operand_rsh(
            mem8(ctx.constant(0x900)),
            ctx.constant(3),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_64() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            operand_lsh(
                operand_and(
                    ctx.constant(0),
                    ctx.const_ffffffff(),
                ),
                ctx.constant(0x20),
            ),
            operand_and(
                ctx.register(0),
                ctx.const_ffffffff(),
            ),
        );
        let eq1 = operand_and(
            ctx.register(0),
            ctx.const_ffffffff(),
        );
        let ne1 = ctx.register(0);
        assert_eq!(Operand::simplified(op1.clone()), Operand::simplified(eq1));
        assert_ne!(Operand::simplified(op1), Operand::simplified(ne1));
    }

    #[test]
    fn gt_same() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            ctx.register(6),
            ctx.register(6),
        );
        let eq1 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_useless_and_mask() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_lsh(
                operand_and(
                    ctx.register(0),
                    ctx.constant(0xff),
                ),
                ctx.constant(1),
            ),
            ctx.constant(0x1fe),
        );
        let eq1 = operand_lsh(
            operand_and(
                ctx.register(0),
                ctx.constant(0xff),
            ),
            ctx.constant(1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_gt_masked() {
        use super::operand_helpers::*;
        // x - y > x => y > x,
        // just with a mask
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            operand_and(
                operand_sub(
                    ctx.register(0),
                    ctx.register(1),
                ),
                ctx.constant(0x1ff),
            ),
            operand_and(
                ctx.register(0),
                ctx.constant(0x1ff),
            ),
        );
        let eq1 = operand_gt(
            operand_and(
                ctx.register(1),
                ctx.constant(0x1ff),
            ),
            operand_and(
                ctx.register(0),
                ctx.constant(0x1ff),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn cannot_simplify_mask_sub() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_sub(
            operand_and(
                operand_sub(
                    ctx.constant(0x4234),
                    ctx.register(0),
                ),
                ctx.constant(0xffff_ffff),
            ),
            ctx.constant(0x1ff),
        );
        let op1 = Operand::simplified(op1);
        // Cannot move the outer sub inside and
        assert!(op1.if_arithmetic_sub().is_some());
    }

    #[test]
    fn simplify_bug_panic() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_and(
                operand_sub(
                    ctx.constant(0),
                    ctx.register(1),
                ),
                ctx.constant(0xffff_ffff),
            ),
            ctx.constant(0),
        );
        // Doesn't simplify, but used to cause a panic
        let _ = Operand::simplified(op1);
    }

    #[test]
    fn simplify_lsh_and_rsh() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_rsh(
            operand_and(
                operand_lsh(
                    ctx.register(1),
                    ctx.constant(0x10),
                ),
                ctx.constant(0xffff_0000),
            ),
            ctx.constant(0x10),
        );
        let eq1 = operand_and(
            ctx.register(1),
            ctx.constant(0xffff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_lsh_and_rsh2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_sub(
                mem16(ctx.register(2)),
                operand_lsh(
                    mem16(ctx.register(1)),
                    ctx.constant(0x9),
                ),
            ),
            ctx.constant(0xffff),
        );
        let eq1 = operand_rsh(
            operand_and(
                operand_lsh(
                    operand_sub(
                        mem16(ctx.register(2)),
                        operand_lsh(
                            mem16(ctx.register(1)),
                            ctx.constant(0x9),
                        ),
                    ),
                    ctx.constant(0x10),
                ),
                ctx.constant(0xffff0000),
            ),
            ctx.constant(0x10),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_ne_shifted_and() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_and(
                operand_lsh(
                    mem8(ctx.register(2)),
                    ctx.constant(0x8),
                ),
                ctx.constant(0x800),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_and(
                mem8(ctx.register(2)),
                ctx.constant(0x8),
            ),
            ctx.constant(0),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn xor_shift_bug() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_rsh(
            operand_and(
                operand_xor(
                    ctx.constant(0xffff_a987_5678),
                    operand_lsh(
                        operand_rsh(
                            mem8(ctx.constant(0x223345)),
                            ctx.constant(3),
                        ),
                        ctx.constant(0x10),
                    ),
                ),
                ctx.constant(0xffff_0000),
            ),
            ctx.constant(0x10),
        );
        let eq1 = operand_xor(
            ctx.constant(0xa987),
            operand_rsh(
                mem8(ctx.constant(0x223345)),
                ctx.constant(3),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_shift_and() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_rsh(
            operand_and(
                ctx.register(0),
                ctx.constant(0xffff_0000),
            ),
            ctx.constant(0x10),
        );
        let eq1 = operand_and(
            operand_rsh(
                ctx.register(0),
                ctx.constant(0x10),
            ),
            ctx.constant(0xffff),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_rotate_mask() {
        // This is mainly useful to make sure rol32(reg32, const) substituted
        // with mem32 is same as just rol32(mem32, const)
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_or(
                operand_rsh(
                    operand_and(
                        ctx.register(0),
                        ctx.constant(0xffff_ffff),
                    ),
                    ctx.constant(0xb),
                ),
                operand_lsh(
                    operand_and(
                        ctx.register(0),
                        ctx.constant(0xffff_ffff),
                    ),
                    ctx.constant(0x15),
                ),
            ),
            ctx.constant(0xffff_ffff),
        );
        let op1 = Operand::simplified(op1);
        let subst = Operand::substitute(&op1, &ctx.register(0), &mem32(ctx.constant(0x1234)));
        let with_mem = operand_and(
            operand_or(
                operand_rsh(
                    mem32(ctx.constant(0x1234)),
                    ctx.constant(0xb),
                ),
                operand_lsh(
                    mem32(ctx.constant(0x1234)),
                    ctx.constant(0x15),
                ),
            ),
            ctx.constant(0xffff_ffff),
        );
        let with_mem = Operand::simplified(with_mem);
        assert_eq!(subst, with_mem);
    }

    #[test]
    fn simplify_add_sub_to_zero() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_add(
            operand_and(
                ctx.register(0),
                ctx.constant(0xffff_ffff),
            ),
            operand_and(
                operand_sub(
                    ctx.constant(0),
                    ctx.register(0),
                ),
                ctx.constant(0xffff_ffff),
            ),
        );
        let eq1 = ctx.constant(0x1_0000_0000);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_less_or_eq() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        // not(c > x) & not(x == c) => not(c + 1 > x)
        // (Same as not((c > x) | (x == c)))
        let op1 = operand_and(
            operand_eq(
                ctx.constant(0),
                operand_gt(
                    ctx.constant(5),
                    ctx.register(1),
                ),
            ),
            operand_eq(
                ctx.constant(0),
                operand_eq(
                    ctx.constant(5),
                    ctx.register(1),
                ),
            ),
        );
        let eq1 = operand_eq(
            ctx.constant(0),
            operand_gt(
                ctx.constant(6),
                ctx.register(1),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_eq_consistency() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = Operand::simplified(operand_eq(
            operand_add(
                ctx.register(1),
                ctx.register(2),
            ),
            ctx.register(3),
        ));
        let eq1a = Operand::simplified(operand_eq(
            operand_sub(
                ctx.register(3),
                ctx.register(2),
            ),
            ctx.register(1),
        ));
        let eq1b = Operand::simplified(operand_eq(
            operand_add(
                ctx.register(2),
                ctx.register(1),
            ),
            ctx.register(3),
        ));
        assert_eq!(op1, eq1a);
        assert_eq!(op1, eq1b);
    }

    #[test]
    fn simplify_mul_consistency() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_mul(
            ctx.register(1),
            operand_add(
                ctx.register(0),
                ctx.register(0),
            ),
        );
        let eq1 = operand_mul(
            operand_mul(
                ctx.constant(2),
                ctx.register(0),
            ),
            ctx.register(1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_sub_add_2_bug() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_add(
            operand_sub(
                ctx.register(1),
                operand_add(
                    ctx.register(2),
                    ctx.register(2),
                ),
            ),
            ctx.register(3),
        );
        let eq1 = operand_add(
            operand_sub(
                ctx.register(1),
                operand_mul(
                    ctx.register(2),
                    ctx.constant(2),
                ),
            ),
            ctx.register(3),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_mul_consistency2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_mul(
            ctx.constant(0x50505230402c2f4),
            operand_add(
                ctx.constant(0x100ffee),
                ctx.register(0),
            ),
        );
        let eq1 = operand_add(
            ctx.constant(0xcdccaa4f6ec24ad8),
            operand_mul(
                ctx.constant(0x50505230402c2f4),
                ctx.register(0),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_eq_consistency2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            operand_eq(
                ctx.register(0),
                ctx.register(0),
            ),
            operand_eq(
                ctx.register(0),
                ctx.register(1),
            ),
        );
        let eq1 = operand_eq(
            operand_eq(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.constant(0),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_fully() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            mem64(ctx.register(0)),
            mem8(ctx.register(0)),
        );
        let eq1 = mem8(ctx.register(0));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_mul_consistency3() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        // 100 * r8 * (r6 + 8) => r8 * (100 * (r6 + 8)) => r8 * ((100 * r6) + 800)
        let op1 = operand_mul(
            operand_mul(
                operand_mul(
                    ctx.register(8),
                    operand_add(
                        ctx.register(6),
                        ctx.constant(0x8),
                    ),
                ),
                ctx.constant(0x10),
            ),
            ctx.constant(0x10),
        );
        let eq1 = operand_mul(
            ctx.register(8),
            operand_add(
                operand_mul(
                    ctx.register(6),
                    ctx.constant(0x100),
                ),
                ctx.constant(0x800),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency1() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            operand_add(
                ctx.register(0),
                operand_sub(
                    operand_or(
                        ctx.register(2),
                        operand_or(
                            operand_add(
                                ctx.register(0),
                                ctx.register(0),
                            ),
                            ctx.register(1),
                        ),
                    ),
                    ctx.register(0),
                ),
            ),
            ctx.register(5),
        );
        let eq1 = operand_or(
            ctx.register(2),
            operand_or(
                operand_mul(
                    ctx.register(0),
                    ctx.constant(2),
                ),
                operand_or(
                    ctx.register(1),
                    ctx.register(5),
                ),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_mul_consistency4() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_mul(
            operand_mul(
                operand_mul(
                    operand_add(
                        operand_mul(
                            operand_add(
                                ctx.register(1),
                                ctx.register(2),
                            ),
                            ctx.register(8),
                        ),
                        ctx.constant(0xb02020202020200),
                    ),
                    ctx.constant(0x202020202020202),
                ),
                ctx.constant(0x200000000000000),
            ),
            operand_mul(
                ctx.register(0),
                ctx.register(8),
            ),
        );
        let eq1 = operand_mul(
            operand_mul(
                ctx.register(0),
                operand_mul(
                    ctx.register(8),
                    ctx.register(8),
                ),
            ),
            operand_mul(
                operand_add(
                    ctx.register(1),
                    ctx.register(2),
                ),
                ctx.constant(0x400000000000000),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_consistency1() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            mem64(ctx.register(0)),
            operand_and(
                operand_or(
                    ctx.constant(0xfd0700002ff4004b),
                    mem8(ctx.register(5)),
                ),
                ctx.constant(0x293b00be00),
            ),
        );
        let eq1 = operand_eq(
            mem64(ctx.register(0)),
            ctx.constant(0x2b000000),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_consistency2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            mem8(ctx.register(0)),
            operand_or(
                operand_and(
                    ctx.constant(0xfeffffffffffff24),
                    operand_add(
                        ctx.register(0),
                        ctx.constant(0x2fbfb01ffff0000),
                    ),
                ),
                ctx.constant(0xf3fb000091010e00),
            ),
        );
        let eq1 = operand_and(
            mem8(ctx.register(0)),
            operand_and(
                ctx.register(0),
                ctx.constant(0x24),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_mul_consistency5() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_mul(
            operand_mul(
                operand_add(
                    operand_add(
                        ctx.register(0),
                        ctx.register(0),
                    ),
                    ctx.constant(0x25000531004000),
                ),
                operand_add(
                    operand_mul(
                        ctx.constant(0x4040405f6020405),
                        ctx.register(0),
                    ),
                    ctx.constant(0x25000531004000),
                ),
            ),
            ctx.constant(0xe9f4000000000000),
        );
        let eq1 = operand_mul(
            operand_mul(
                ctx.register(0),
                ctx.register(0),
            ),
            ctx.constant(0xc388000000000000)
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_consistency3() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_or(
                operand_or(
                    mem16(ctx.register(0)),
                    ctx.constant(0x4eff0001004107),
                ),
                ctx.constant(0x231070100fa00de),
            ),
            ctx.constant(0x280000d200004010),
        );
        let eq1 = ctx.constant(0x4010);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_consistency4() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_and(
                operand_or(
                    operand_xmm(4, 1),
                    ctx.constant(0x1e04ffffff0000),
                ),
                operand_xmm(0, 1),
            ),
            ctx.constant(0x40ffffffffffff60),
        );
        let eq1 = operand_and(
            Operand::simplified(
                operand_and(
                    operand_or(
                        ctx.constant(0xffff0000),
                        operand_xmm(4, 1),
                    ),
                    ctx.constant(0xffffff60),
                ),
            ),
            operand_xmm(0, 1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            operand_and(
                ctx.constant(0xfe05000080000025),
                operand_or(
                    operand_xmm(0, 1),
                    ctx.constant(0xf3fbfb01ffff0000),
                ),
            ),
            ctx.constant(0xf3fb0073_00000000),
        );
        let eq1 = operand_or(
            ctx.constant(0xf3fb0073_80000000),
            operand_and(
                operand_xmm(0, 1),
                ctx.constant(0x25),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_add_consistency1() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_add(
            operand_add(
                operand_and(
                    ctx.constant(0xfeffffffffffff24),
                    operand_or(
                        operand_xmm(0, 1),
                        ctx.constant(0xf3fbfb01ffff0000),
                    ),
                ),
                operand_and(
                    ctx.constant(0xfeffffffffffff24),
                    operand_or(
                        operand_xmm(0, 1),
                        ctx.constant(0xf301fc01ffff3eff)
                    ),
                ),
            ),
            ctx.custom(3),
        );
        let eq1 = operand_add(
            Operand::simplified(operand_add(
                operand_and(
                    ctx.constant(0xfeffffffffffff24),
                    operand_or(
                        operand_xmm(0, 1),
                        ctx.constant(0xf3fbfb01ffff0000),
                    ),
                ),
                operand_and(
                    ctx.constant(0xfeffffffffffff24),
                    operand_or(
                        operand_xmm(0, 1),
                        ctx.constant(0xf301fc01ffff3eff)
                    ),
                ),
            )),
            ctx.custom(3),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_add_simple() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_sub(
            ctx.constant(1),
            ctx.constant(4),
        );
        let eq1 = ctx.constant(0xffff_ffff_ffff_fffd);
        let op2 = operand_add(
            operand_sub(
                ctx.constant(0xf40205051a02c2f4),
                ctx.register(0),
            ),
            ctx.register(0),
        );
        let eq2 = ctx.constant(0xf40205051a02c2f4);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
    }

    #[test]
    fn simplify_1bit_sum() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        // Since the gt makes only at most LSB of the sum to be considered,
        // the multiplication isn't used at all and the sum
        // Mem8[rax] + Mem16[rax] + Mem32[rax] can become 3 * Mem8[rax] which
        // can just be replaced with Mem8[rax]
        let op1 = operand_and(
            operand_gt(
                ctx.register(5),
                ctx.register(4),
            ),
            operand_add(
                operand_add(
                    operand_mul(
                        ctx.constant(6),
                        ctx.register(0),
                    ),
                    operand_add(
                        mem8(ctx.register(0)),
                        mem32(ctx.register(1)),
                    ),
                ),
                operand_add(
                    mem16(ctx.register(0)),
                    mem64(ctx.register(0)),
                ),
            ),
        );
        let eq1 = operand_and(
            operand_gt(
                ctx.register(5),
                ctx.register(4),
            ),
            operand_add(
                mem8(ctx.register(0)),
                mem8(ctx.register(1)),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_masked_add() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        // Cannot move the constant out of and since
        // (fffff + 1400) & ffff6ff24 == 101324, but
        // (fffff & ffff6ff24) + 1400 == 71324
        let op1 = operand_and(
            operand_add(
                mem32(ctx.register(0)),
                ctx.constant(0x1400),
            ),
            ctx.constant(0xffff6ff24),
        );
        let ne1 = operand_add(
            operand_and(
                mem32(ctx.register(0)),
                ctx.constant(0xffff6ff24),
            ),
            ctx.constant(0x1400),
        );
        let op2 = operand_add(
            ctx.register(1),
            operand_and(
                operand_add(
                    mem32(ctx.register(0)),
                    ctx.constant(0x1400),
                ),
                ctx.constant(0xffff6ff24),
            ),
        );
        let ne2 = operand_add(
            ctx.register(1),
            operand_add(
                operand_and(
                    mem32(ctx.register(0)),
                    ctx.constant(0xffff6ff24),
                ),
                ctx.constant(0x1400),
            ),
        );
        assert_ne!(Operand::simplified(op1), Operand::simplified(ne1));
        assert_ne!(Operand::simplified(op2), Operand::simplified(ne2));
    }

    #[test]
    fn simplify_masked_add2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        // Cannot move the constant out of and since
        // (fffff + 1400) & ffff6ff24 == 101324, but
        // (fffff & ffff6ff24) + 1400 == 71324
        let op1 = operand_add(
            ctx.constant(0x4700000014fef910),
            operand_and(
                operand_add(
                    mem32(ctx.register(0)),
                    ctx.constant(0x1400),
                ),
                ctx.constant(0xffff6ff24),
            ),
        );
        let ne1 = operand_add(
            ctx.constant(0x4700000014fef910),
            operand_add(
                operand_and(
                    mem32(ctx.register(0)),
                    ctx.constant(0xffff6ff24),
                ),
                ctx.constant(0x1400),
            ),
        );
        let op1 = Operand::simplified(op1);
        assert_ne!(op1, Operand::simplified(ne1));
        assert!(
            op1.iter().any(|x| {
                x.if_arithmetic_add()
                    .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                    .filter(|&(c, other)| c == 0x1400 && other.if_memory().is_some())
                    .is_some()
            }),
            "Op1 was simplified wrong: {}", op1,
        );
    }

    #[test]
    fn simplify_masked_add3() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_add(
            operand_and(
                ctx.constant(0xffff),
                operand_or(
                    ctx.register(0),
                    ctx.register(1),
                ),
            ),
            operand_and(
                ctx.constant(0xffff),
                operand_or(
                    ctx.register(0),
                    ctx.register(1),
                ),
            ),
        );
        let op1 = Operand::simplified(op1);
        assert!(op1.relevant_bits().end > 16, "Operand wasn't simplified correctly {}", op1);
    }

    #[test]
    fn simplify_or_consistency3() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_or(
                ctx.constant(0x854e00e501001007),
                mem16(ctx.register(0)),
            ),
            ctx.constant(0x28004000d2000010),
        );
        let eq1 = operand_and(
            mem8(ctx.register(0)),
            ctx.constant(0x10),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_eq_consistency3() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_and(
                Operand::new_not_simplified_rc(
                    OperandType::SignExtend(
                        ctx.constant(0x2991919191910000),
                        MemAccessSize::Mem8,
                        MemAccessSize::Mem16,
                    ),
                ),
                ctx.register(1),
            ),
            mem8(ctx.register(2)),
        );
        let eq1 = operand_eq(
            mem8(ctx.register(2)),
            ctx.constant(0),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency4() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            operand_and(
                operand_or(
                    ctx.constant(0x80000000000002),
                    operand_xmm(2, 1),
                ),
                mem16(ctx.register(0)),
            ),
            ctx.constant(0x40ffffffff3fff7f),
        );
        let eq1 = operand_or(
            operand_and(
                operand_xmm(2, 1),
                mem8(ctx.register(0)),
            ),
            ctx.constant(0x40ffffffff3fff7f),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_infinite_recurse_bug() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            operand_and(
                operand_or(
                    ctx.constant(0x100),
                    operand_and(
                        operand_mul(
                            ctx.constant(4),
                            ctx.register(0),
                        ),
                        ctx.constant(0xffff_fe00),
                    ),
                ),
                mem32(ctx.register(0)),
            ),
            ctx.constant(0xff_ffff_fe00),
        );
        let _ = Operand::simplified(op1);
    }

    #[test]
    fn simplify_eq_consistency4() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_add(
                ctx.constant(0x7014b0001050500),
                mem32(ctx.register(0)),
            ),
            mem32(ctx.register(1)),
        );
        let op1 = Operand::simplified(op1);
        assert!(
            op1.iter().any(|x| x.if_arithmetic_and().is_some()) == false,
            "Operand didn't simplify correctly: {}", op1,
        );
    }

    #[test]
    fn simplify_eq_consistency5() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            mem32(ctx.register(1)),
            operand_add(
                ctx.constant(0x5a00000001),
                mem8(ctx.register(0)),
            ),
        );
        let op1 = Operand::simplified(op1);
        assert!(
            op1.iter().any(|x| x.if_arithmetic_and().is_some()) == false,
            "Operand didn't simplify correctly: {}", op1,
        );
    }

    #[test]
    fn simplify_eq_consistency6() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            ctx.register(0),
            operand_add(
                ctx.register(0),
                operand_rsh(
                    mem16(ctx.register(0)),
                    ctx.constant(5),
                ),
            ),
        );
        let eq1a = operand_eq(
            ctx.constant(0),
            operand_rsh(
                mem16(ctx.register(0)),
                ctx.constant(5),
            ),
        );
        let eq1b = operand_eq(
            ctx.constant(0),
            operand_and(
                mem16(ctx.register(0)),
                ctx.constant(0xffe0),
            ),
        );
        assert_eq!(Operand::simplified(op1.clone()), Operand::simplified(eq1a));
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1b));
    }

    #[test]
    fn simplify_eq_consistency7() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            ctx.constant(0x4000000000570000),
            operand_add(
                mem32(ctx.register(0)),
                operand_add(
                    mem32(ctx.register(1)),
                    ctx.constant(0x7e0000fffc01),
                ),
            ),
        );
        let eq1 = operand_eq(
            ctx.constant(0x3fff81ffff5703ff),
            operand_add(
                mem32(ctx.register(0)),
                mem32(ctx.register(1)),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_zero_eq_zero() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            ctx.constant(0),
            ctx.constant(0),
        );
        let eq1 = ctx.constant(1);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_xor_consistency1() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_xor(
            operand_xor(
                ctx.register(1),
                ctx.constant(0x5910e010000),
            ),
            operand_and(
                operand_or(
                    ctx.register(2),
                    ctx.constant(0xf3fbfb01ffff0000),
                ),
                ctx.constant(0x1ffffff24),
            ),
        );
        let eq1 = operand_xor(
            ctx.register(1),
            operand_xor(
                ctx.constant(0x590f1fe0000),
                operand_and(
                    ctx.constant(0xff24),
                    ctx.register(2),
                ),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_xor_consistency2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_xor(
            operand_xor(
                ctx.register(1),
                ctx.constant(0x59100010e00),
            ),
            operand_and(
                operand_or(
                    ctx.register(2),
                    ctx.constant(0xf3fbfb01ffff7e00),
                ),
                ctx.constant(0x1ffffff24),
            ),
        );
        let eq1 = operand_xor(
            ctx.register(1),
            operand_xor(
                ctx.constant(0x590fffe7000),
                operand_and(
                    ctx.constant(0x8124),
                    ctx.register(2),
                ),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency5() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0xffff7024_ffffffff),
            operand_and(
                mem64(ctx.constant(0x100)),
                ctx.constant(0x0500ff04_ffff0000),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0xffff7024ffffffff),
            operand_lsh(
                mem8(ctx.constant(0x105)),
                ctx.constant(0x28),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency6() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0xfffeffffffffffff),
            operand_and(
                operand_xmm(0, 0),
                ctx.register(0),
            ),
        );
        let eq1 = ctx.constant(0xfffeffffffffffff);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_useless_mod() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_mod(
            operand_xmm(0, 0),
            ctx.constant(0x504ff04ff0000),
        );
        let eq1 = operand_xmm(0, 0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency7() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0xffffffffffffff41),
            operand_and(
                operand_xmm(0, 0),
                operand_or(
                    ctx.register(0),
                    ctx.constant(0x504ffffff770000),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0xffffffffffffff41),
            operand_and(
                operand_xmm(0, 0),
                ctx.register(0),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_or_consistency8() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0x40ff_ffff_ffff_3fff),
            operand_and(
                mem64(ctx.register(0)),
                operand_or(
                    operand_xmm(0, 0),
                    ctx.constant(0x0080_0000_0000_0002),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0x40ff_ffff_ffff_3fff),
            operand_and(
                mem16(ctx.register(0)),
                operand_xmm(0, 0),
            ),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_consistency5() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_and(
                mem8(ctx.register(1)),
                operand_or(
                    ctx.constant(0x22),
                    operand_xmm(0, 0),
                ),
            ),
            ctx.constant(0x23),
        );
        check_simplification_consistency(op1);
    }

    #[test]
    fn simplify_and_consistency6() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_and(
                ctx.constant(0xfeffffffffffff24),
                operand_or(
                    ctx.constant(0xf3fbfb01ffff0000),
                    operand_xmm(0, 0),
                ),
            ),
            operand_or(
                ctx.constant(0xf3fb000091010e03),
                mem8(ctx.register(1)),
            ),
        );
        check_simplification_consistency(op1);
    }

    #[test]
    fn simplify_or_consistency9() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0x47000000140010ff),
            operand_or(
                mem16(ctx.constant(0x100)),
                operand_or(
                    ctx.constant(0x2a00000100100730),
                    operand_mul(
                        ctx.constant(0x2000000000),
                        operand_xmm(4, 1),
                    ),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0x6f000001141017ff),
            operand_or(
                operand_lsh(
                    mem8(ctx.constant(0x101)),
                    ctx.constant(8),
                ),
                operand_mul(
                    ctx.constant(0x2000000000),
                    operand_xmm(4, 1),
                ),
            ),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_and_consistency7() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_and(
                ctx.constant(0xfeffffffffffff24),
                operand_or(
                    ctx.constant(0xf3fbfb01ffff0000),
                    operand_xmm(0, 0),
                ),
            ),
            operand_or(
                ctx.constant(0xc04ffff6efef1f6),
                mem8(ctx.register(1)),
            ),
        );
        check_simplification_consistency(op1);
    }

    #[test]
    fn simplify_and_consistency8() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_and(
                operand_add(
                    ctx.constant(0x5050505000001),
                    ctx.register(0),
                ),
                operand_mod(
                    ctx.register(4),
                    ctx.constant(0x3ff0100000102),
                ),
            ),
            ctx.constant(0x3ff01000001),
        );
        let eq1 = operand_and(
            Operand::simplified(
                operand_and(
                    operand_add(
                        ctx.constant(0x5050505000001),
                        ctx.register(0),
                    ),
                    operand_mod(
                        ctx.register(4),
                        ctx.constant(0x3ff0100000102),
                    ),
                ),
            ),
            ctx.constant(0x3ff01000001),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        let eq1b = operand_and(
            operand_and(
                operand_add(
                    ctx.constant(0x5050505000001),
                    ctx.register(0),
                ),
                operand_mod(
                    ctx.register(4),
                    ctx.constant(0x3ff0100000102),
                ),
            ),
            ctx.constant(0x50003ff01000001),
        );
        let eq1b = Operand::simplified(eq1b);
        assert_eq!(op1, eq1);
        assert_eq!(op1, eq1b);
    }

    #[test]
    fn simplify_eq_consistency8() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_add(
                operand_add(
                    operand_add(
                        ctx.constant(1),
                        operand_mod(
                            ctx.register(0),
                            mem8(ctx.register(0)),
                        ),
                    ),
                    operand_div(
                        operand_eq(
                            ctx.register(0),
                            ctx.register(1),
                        ),
                        operand_eq(
                            ctx.register(4),
                            ctx.register(6),
                        ),
                    ),
                ),
                operand_eq(
                    ctx.register(4),
                    ctx.register(5),
                ),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_add(
                Operand::simplified(
                    operand_add(
                        operand_add(
                            ctx.constant(1),
                            operand_mod(
                                ctx.register(0),
                                mem8(ctx.register(0)),
                            ),
                        ),
                        operand_div(
                            operand_eq(
                                ctx.register(0),
                                ctx.register(1),
                            ),
                            operand_eq(
                                ctx.register(4),
                                ctx.register(6),
                            ),
                        ),
                    ),
                ),
                operand_eq(
                    ctx.register(4),
                    ctx.register(5),
                ),
            ),
            ctx.constant(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_gt_consistency1() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            operand_sub(
                operand_gt(
                    operand_sub(
                        ctx.register(0),
                        ctx.register(0),
                    ),
                    ctx.register(5),
                ),
                ctx.register(0),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_eq(
                ctx.register(0),
                ctx.constant(0),
            ),
            ctx.constant(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_add_consistency2() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_add(
            operand_add(
                operand_or(
                    operand_sub(
                        ctx.register(1),
                        operand_add(
                            ctx.register(0),
                            ctx.register(0),
                        ),
                    ),
                    ctx.constant(0),
                ),
                operand_add(
                    ctx.register(0),
                    ctx.register(0),
                ),
            ),
            ctx.register(0),
        );
        let eq1 = operand_add(
            ctx.register(0),
            ctx.register(1),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_and_consistency9() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_and(
                operand_add(
                    ctx.register(1),
                    ctx.constant(0x50007f3fbff0000),
                ),
                operand_or(
                    operand_xmm(0, 1),
                    ctx.constant(0xf3fbfb01ffff0000),
                ),
            ),
            ctx.constant(0x6080e6300000000),
        );
        let eq1 = operand_and(
            operand_add(
                ctx.register(1),
                ctx.constant(0x50007f3fbff0000),
            ),
            ctx.constant(0x2080a0100000000),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_bug_infloop1() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0xe20040ffe000e500),
            operand_xor(
                ctx.constant(0xe20040ffe000e500),
                operand_or(
                    ctx.register(0),
                    ctx.register(1),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0xe20040ffe000e500),
            operand_or(
                ctx.register(0),
                ctx.register(1),
            ),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_eq_consistency9() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_and(
                operand_xmm(0, 0),
                ctx.constant(0x40005ff000000ff),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_and(
                operand_xmm(0, 0),
                ctx.constant(0xff),
            ),
            ctx.constant(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_eq_consistency10() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            operand_sub(
                operand_mod(
                    operand_xmm(0, 0),
                    ctx.register(0),
                ),
                operand_sub(
                    ctx.constant(0),
                    operand_mod(
                        ctx.register(1),
                        ctx.constant(0),
                    ),
                ),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_eq(
                operand_add(
                    operand_mod(
                        operand_xmm(0, 0),
                        ctx.register(0),
                    ),
                    operand_mod(
                        ctx.register(1),
                        ctx.constant(0),
                    ),
                ),
                ctx.constant(0),
            ),
            ctx.constant(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
        assert!(op1.iter().any(|x| match x.if_arithmetic(ArithOpType::Modulo) {
            Some((a, b)) => a.if_constant() == Some(0) && b.if_constant() == Some(0),
            None => false,
        }), "0 / 0 disappeared: {}", op1);
    }

    #[test]
    fn simplify_eq_consistency11() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_gt(
            operand_lsh(
                operand_sub(
                    operand_xmm(0, 0),
                    ctx.register(0),
                ),
                ctx.constant(1),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_eq(
                operand_and(
                    operand_sub(
                        operand_xmm(0, 0),
                        ctx.register(0),
                    ),
                    ctx.constant(0x7fff_ffff_ffff_ffff),
                ),
                ctx.constant(0),
            ),
            ctx.constant(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_eq_consistency12() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_and(
                operand_add(
                    operand_xmm(0, 0),
                    ctx.register(0),
                ),
                operand_add(
                    ctx.constant(1),
                    ctx.constant(0),
                ),
            ),
            ctx.constant(0),
        );
        let eq1 = operand_eq(
            operand_and(
                operand_add(
                    operand_xmm(0, 0),
                    ctx.register(0),
                ),
                ctx.constant(1),
            ),
            ctx.constant(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_eq_consistency13() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_eq(
            operand_add(
                operand_sub(
                    operand_and(
                        ctx.constant(0xffff),
                        ctx.register(1),
                    ),
                    mem8(ctx.register(0)),
                ),
                operand_xmm(0, 0),
            ),
            operand_add(
                operand_and(
                    ctx.register(3),
                    ctx.constant(0x7f),
                ),
                operand_xmm(0, 0),
            ),
        );
        let eq1 = operand_eq(
            operand_and(
                ctx.constant(0xffff),
                ctx.register(1),
            ),
            operand_add(
                operand_and(
                    ctx.register(3),
                    ctx.constant(0x7f),
                ),
                mem8(ctx.register(0)),
            ),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_and_consistency10() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            ctx.constant(0x5ffff05b700),
            operand_and(
                operand_or(
                    operand_xmm(1, 0),
                    ctx.constant(0x5ffffffff00),
                ),
                operand_or(
                    ctx.register(0),
                    ctx.constant(0x5ffffffff0000),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0x5ffff050000),
            operand_and(
                ctx.register(0),
                ctx.constant(0xb700),
            ),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_xor_consistency3() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0x200ffffff7f),
            operand_xor(
                operand_or(
                    operand_xmm(1, 0),
                    ctx.constant(0x20000ff20ffff00),
                ),
                operand_or(
                    ctx.register(0),
                    ctx.constant(0x5ffffffff0000),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0x200ffffff7f),
            operand_xor(
                operand_xor(
                    operand_xmm(1, 0),
                    ctx.constant(0x20000ff00000000),
                ),
                operand_or(
                    ctx.register(0),
                    ctx.constant(0x5fdff00000000),
                ),
            ),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_or_consistency10() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0xffff_ffff_ffff),
            operand_or(
                operand_xor(
                    operand_xmm(1, 0),
                    ctx.register(0),
                ),
                ctx.register(0),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0xffff_ffff_ffff),
            ctx.register(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_xor_consistency4() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_xor(
            operand_xor(
                ctx.constant(0x07ff_ffff_0000_0000),
                operand_or(
                    ctx.register(1),
                    operand_and(
                        operand_xmm(1, 0),
                        ctx.constant(0xff),
                    ),
                ),
            ),
            ctx.register(0),
        );
        let eq1 = operand_xor(
            Operand::simplified(
                operand_xor(
                    ctx.constant(0x07ff_ffff_0000_0000),
                    operand_or(
                        ctx.register(1),
                        operand_and(
                            operand_xmm(1, 0),
                            ctx.constant(0xff),
                        ),
                    ),
                ),
            ),
            ctx.register(0),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_or_consistency11() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0xb7ff),
            operand_or(
                operand_xor(
                    ctx.register(1),
                    operand_xmm(1, 0),
                ),
                operand_and(
                    operand_or(
                        operand_xmm(1, 3),
                        ctx.constant(0x5ffffffff00),
                    ),
                    operand_or(
                        ctx.constant(0x5ffffffff7800),
                        ctx.register(1),
                    ),
                ),
            ),
        );
        let eq1 = operand_or(
            ctx.constant(0x5ffffffffff),
            ctx.register(1),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_and_consistency11() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            ctx.constant(0xb7ff),
            operand_and(
                operand_xmm(1, 0),
                operand_or(
                    operand_or(
                        operand_xmm(2, 0),
                        ctx.constant(0x5ffffffff00),
                    ),
                    ctx.register(1),
                ),
            ),
        );
        check_simplification_consistency(op1);
    }

    #[test]
    fn simplify_and_consistency12() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            ctx.constant(0x40005ffffffffff),
            operand_xmm(1, 0),
        );
        let eq1 = operand_xmm(1, 0);
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_or_consistency12() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_or(
            ctx.constant(0x500000000007fff),
            operand_and(
                operand_xmm(1, 0),
                operand_or(
                    operand_and(
                        operand_xmm(1, 3),
                        ctx.constant(0x5ffffffff00),
                    ),
                    ctx.register(0),
                ),
            )
        );
        check_simplification_consistency(op1);
    }

    #[test]
    fn simplify_or_consistency13() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_xmm(1, 0),
            operand_or(
                operand_add(
                    operand_xmm(1, 0),
                    ctx.constant(0x8ff00000000),
                ),
                ctx.register(0),
            ),
        );
        let eq1 = operand_xmm(1, 0);
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_xor_consistency5() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_xor(
            operand_xmm(1, 0),
            operand_or(
                operand_xor(
                    operand_xmm(1, 0),
                    operand_xmm(1, 1),
                ),
                ctx.constant(0x600000000000000),
            ),
        );
        let eq1 = operand_xor(
            operand_xmm(1, 1),
            ctx.constant(0x600000000000000),
        );
        let op1 = Operand::simplified(op1);
        let eq1 = Operand::simplified(eq1);
        assert_eq!(op1, eq1);
    }

    #[test]
    fn simplify_and_consistency13() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            ctx.constant(0x50000000000b7ff),
            operand_and(
                operand_xmm(1, 0),
                operand_or(
                    operand_add(
                        ctx.register(0),
                        ctx.register(0),
                    ),
                    operand_mod(
                        ctx.register(0),
                        ctx.constant(0xff0000),
                    ),
                ),
            )
        );
        check_simplification_consistency(op1);
    }

    #[test]
    fn simplify_and_panic() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            operand_xmm(1, 0),
            operand_and(
                operand_or(
                    operand_xmm(1, 0),
                    ctx.constant(0x5ffffffff00),
                ),
                operand_arith(ArithOpType::Parity, ctx.register(0), ctx.constant(0)),
            )
        );
        let _ = Operand::simplified(op1);
    }

    #[test]
    fn simplify_and_consistency14() {
        use super::operand_helpers::*;
        let ctx = &OperandContext::new();
        let op1 = operand_and(
            ctx.register(0),
            operand_and(
                operand_gt(
                    operand_xmm(1, 0),
                    ctx.constant(0),
                ),
                operand_gt(
                    operand_xmm(1, 4),
                    ctx.constant(0),
                ),
            )
        );
        check_simplification_consistency(op1);
    }
}
