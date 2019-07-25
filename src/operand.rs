use std::cell::Cell;
use std::cmp::{max, min, Ordering};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Range;
use std::rc::Rc;

use serde::{Deserialize as DeserializeTrait, Deserializer};
use serde_derive::{Deserialize, Serialize};

use crate::bit_misc::{bits_overlap, one_bit_ranges, zero_bit_ranges};
use crate::vec_drop_iter::VecDropIter;

#[derive(Debug, Clone, Eq, Serialize)]
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

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::ArithOpType::*;

        match self.ty {
            OperandType::Register(r) => match r.0 {
                0 => write!(f, "eax"),
                1 => write!(f, "ecx"),
                2 => write!(f, "edx"),
                3 => write!(f, "ebx"),
                4 => write!(f, "esp"),
                5 => write!(f, "ebp"),
                6 => write!(f, "esi"),
                7 => write!(f, "edi"),
                x => write!(f, "r32_{}", x),
            },
            OperandType::Register64(r) => match r.0 {
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
            OperandType::Register16(r) => match r.0 {
                0 => write!(f, "ax"),
                1 => write!(f, "cx"),
                2 => write!(f, "dx"),
                3 => write!(f, "bx"),
                4 => write!(f, "sp"),
                5 => write!(f, "bp"),
                6 => write!(f, "si"),
                7 => write!(f, "di"),
                x => write!(f, "r16_{}", x),
            },
            OperandType::Register8High(r) => match r.0 {
                0 => write!(f, "ah"),
                1 => write!(f, "ch"),
                2 => write!(f, "dh"),
                3 => write!(f, "bh"),
                x => write!(f, "r8hi_{}", x),
            },
            OperandType::Register8Low(r) => match r.0 {
                0 => write!(f, "al"),
                1 => write!(f, "cl"),
                2 => write!(f, "dl"),
                3 => write!(f, "bl"),
                x => write!(f, "r8lo_{}", x),
            },
            OperandType::Pair(ref hi, ref low) => write!(f, "{}:{}", hi, low),
            OperandType::Xmm(reg, subword) => write!(f, "xmm{}.{}", reg, subword),
            OperandType::Flag(flag) => match flag {
                Flag::Zero => write!(f, "z"),
                Flag::Carry => write!(f, "c"),
                Flag::Overflow => write!(f, "o"),
                Flag::Parity => write!(f, "p"),
                Flag::Sign => write!(f, "s"),
                Flag::Direction => write!(f, "d"),
            },
            OperandType::Constant(c) => write!(f, "{:x}", c),
            OperandType::Constant64(c) => write!(f, "{:x}", c),
            OperandType::Memory(ref mem) => write!(f, "Mem{}[{}]", match mem.size {
                MemAccessSize::Mem8 => "8",
                MemAccessSize::Mem16 => "16",
                MemAccessSize::Mem32 => "32",
                MemAccessSize::Mem64 => "64",
            }, mem.address),
            OperandType::Undefined(id) => write!(f, "Undefined_{:x}", id.0),
            OperandType::Arithmetic(ref arith) | OperandType::Arithmetic64(ref arith) => {
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
                    GreaterThanSigned => write!(f, "gt_signed({}, {})", l, r),
                    SignedMul => write!(f, "mul_signed({}, {})", l, r),
                    Parity => write!(f, "parity({})", l),
                }?;
                if let OperandType::Arithmetic64(..) = self.ty {
                    write!(f, "[64]")?;
                }
                Ok(())
            },
            OperandType::ArithmeticHigh(ref arith) => {
                // TODO: Should honestly just have format on ArithOperand
                let fmt = Operand::new_not_simplified_rc(OperandType::Arithmetic(arith.clone()));
                write!(f, "{}.high", fmt)
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Deserialize, Serialize)]
pub enum OperandType {
    Register(Register),
    Register16(Register),
    Register8High(Register),
    Register8Low(Register),
    Register64(Register),
    // For div, as it sets eax to (edx:eax / x), and edx to (edx:eax % x)
    Pair(Rc<Operand>, Rc<Operand>),
    Xmm(u8, u8),
    Flag(Flag),
    Constant(u32),
    Constant64(u64),
    Memory(MemAccess),
    Arithmetic(ArithOperand),
    Arithmetic64(ArithOperand),
    Undefined(UndefinedId),
    // The high 32 bits that usually are discarded in a airthmetic operation,
    // but relevant for 64-bit multiplications.
    ArithmeticHigh(ArithOperand),
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
    GreaterThanSigned,
}

impl ArithOperand {
    pub fn is_compare_op(&self) -> bool {
        use self::ArithOpType::*;
        match self.ty {
            Equal | GreaterThan | GreaterThanSigned => true,
            _ => false,
        }
    }

    /// If self has 64-bit operands, returns a new ArithOperand with high bits discarded.
    ///
    /// Helper function for simplification. Take care with behaviour of Arith64 -> Arith
    /// conversion.
    fn truncate_operands_to_32bit(&self, ctx: &OperandContext) -> Option<ArithOperand> {
        let trunc = |operand: &Rc<Operand>| -> Rc<Operand> {
            match operand.ty {
                OperandType::Constant64(c) => ctx.constant(c as u32),
                OperandType::Register64(r) => ctx.register(r.0),
                OperandType::Arithmetic64(_) => {
                    self::operand_helpers::operand_and64(operand.clone(), ctx.const_ffffffff())
                }
                _ => operand.clone(),
            }
        };
        let first_needs_trunc = match self.left.ty.expr_size() {
            MemAccessSize::Mem8 | MemAccessSize::Mem16 | MemAccessSize::Mem32 => false,
            MemAccessSize::Mem64 => true,
        };
        let second_needs_trunc = match self.right.ty.expr_size() {
            MemAccessSize::Mem8 | MemAccessSize::Mem16 | MemAccessSize::Mem32 => false,
            MemAccessSize::Mem64 => true,
        };
        if first_needs_trunc || second_needs_trunc {
            let left = if first_needs_trunc {
                trunc(&self.left)
            } else {
                self.left.clone()
            };
            let right = if second_needs_trunc {
                trunc(&self.right)
            } else {
                self.right.clone()
            };
            Some(ArithOperand {
                left,
                right,
                ty: self.ty,
            })
        } else {
            None
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
    registers: [Rc<Operand>; 8],
    registers16: [Rc<Operand>; 8],
    // Contains the nonexisting esp/ebp/esi/edi low ones for simplicity
    registers8_low: [Rc<Operand>; 8],
    registers8_high: [Rc<Operand>; 4],
    registers64: [Rc<Operand>; 8],
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

    let inner = match mem::replace(s.state(), None) {
        Some(s) => s,
        None => return None,
    };
    let next = inner.pos;

    *s.state() = match next.ty {
        Arithmetic(ref arith) | ArithmeticHigh(ref arith) => {
            Some(IterState {
                pos: &arith.left,
                rhs: Some(Box::new(IterState {
                    pos: &arith.right,
                    rhs: inner.rhs,
                })),
            })
        },
        Memory(ref m) => {
            Some(IterState {
                pos: &m.address,
                rhs: inner.rhs,
            })
        }
        Pair(ref hi, ref low) => {
            Some(IterState {
                pos: hi,
                rhs: Some(Box::new(IterState {
                    pos: low,
                    rhs: inner.rhs,
                })),
            })
        }
        _ => inner.rhs.map(|x| *x),
    };
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
    rhs: Option<Box<IterState<'a>>>,
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
            ],
            registers16: [
                Operand::new_simplified_rc(OperandType::Register16(Register(0))),
                Operand::new_simplified_rc(OperandType::Register16(Register(1))),
                Operand::new_simplified_rc(OperandType::Register16(Register(2))),
                Operand::new_simplified_rc(OperandType::Register16(Register(3))),
                Operand::new_simplified_rc(OperandType::Register16(Register(4))),
                Operand::new_simplified_rc(OperandType::Register16(Register(5))),
                Operand::new_simplified_rc(OperandType::Register16(Register(6))),
                Operand::new_simplified_rc(OperandType::Register16(Register(7))),
            ],
            registers8_low: [
                Operand::new_simplified_rc(OperandType::Register8Low(Register(0))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(1))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(2))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(3))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(4))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(5))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(6))),
                Operand::new_simplified_rc(OperandType::Register8Low(Register(7))),
            ],
            registers8_high: [
                Operand::new_simplified_rc(OperandType::Register8High(Register(0))),
                Operand::new_simplified_rc(OperandType::Register8High(Register(1))),
                Operand::new_simplified_rc(OperandType::Register8High(Register(2))),
                Operand::new_simplified_rc(OperandType::Register8High(Register(3))),
            ],
            registers64: [
                Operand::new_simplified_rc(OperandType::Register64(Register(0))),
                Operand::new_simplified_rc(OperandType::Register64(Register(1))),
                Operand::new_simplified_rc(OperandType::Register64(Register(2))),
                Operand::new_simplified_rc(OperandType::Register64(Register(3))),
                Operand::new_simplified_rc(OperandType::Register64(Register(4))),
                Operand::new_simplified_rc(OperandType::Register64(Register(5))),
                Operand::new_simplified_rc(OperandType::Register64(Register(6))),
                Operand::new_simplified_rc(OperandType::Register64(Register(7))),
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

    pub fn register16(&self, index: u8) -> Rc<Operand> {
        self.globals.registers16[index as usize].clone()
    }

    pub fn register8_low(&self, index: u8) -> Rc<Operand> {
        self.globals.registers8_low[index as usize].clone()
    }

    pub fn register8_high(&self, index: u8) -> Rc<Operand> {
        self.globals.registers8_high[index as usize].clone()
    }

    pub fn register64(&self, index: u8) -> Rc<Operand> {
        self.globals.registers64[index as usize].clone()
    }

    pub fn reg_variable_size(&self, reg: Register, size: MemAccessSize) -> Rc<Operand> {
        match size {
            MemAccessSize::Mem64 => self.register64(reg.0),
            MemAccessSize::Mem32 => self.register(reg.0),
            MemAccessSize::Mem16 => self.register16(reg.0),
            MemAccessSize::Mem8 => if reg.0 >= 4 {
                self.register8_high(reg.0 - 4)
            } else {
                self.register8_low(reg.0)
            },
        }
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

    pub fn constant(&self, value: u32) -> Rc<Operand> {
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
}

impl fmt::Debug for OperandType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::OperandType::*;
        match self {
            Register(r) => write!(f, "Register({})", r.0),
            Register16(r) => write!(f, "Register16({})", r.0),
            Register8High(r) => write!(f, "Register8High({})", r.0),
            Register8Low(r) => write!(f, "Register8Low({})", r.0),
            Register64(r) => write!(f, "Register64({})", r.0),
            Xmm(r, x) => write!(f, "Xmm({}.{})", r, x),
            Flag(r) => write!(f, "Flag({:?})", r),
            Constant(r) => write!(f, "Constant({:x})", r),
            Constant64(r) => write!(f, "Constant64({:x})", r),
            Undefined(r) => write!(f, "Undefined_{:x}", r.0),
            Pair(hi, low) => {
                f.debug_tuple("Pair")
                    .field(hi)
                    .field(low)
                    .finish()
            }
            Memory(r) => f.debug_tuple("Memory").field(r).finish(),
            Arithmetic(r) => f.debug_tuple("Arithmetic").field(r).finish(),
            Arithmetic64(r) => f.debug_tuple("Arithmetic64").field(r).finish(),
            ArithmeticHigh(r) => f.debug_tuple("ArithmeticHigh").field(r).finish(),
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
            OperandType::Register(_) | OperandType::Flag(_) | OperandType::Undefined(_) => 32,
            OperandType::Arithmetic(ref arith) => match arith.ty {
                ArithOpType::And | ArithOpType::Or | ArithOpType::Xor => {
                    min(
                        arith.left.min_zero_bit_simplify_size,
                        arith.right.min_zero_bit_simplify_size,
                    )
                }
                ArithOpType::Lsh | ArithOpType::Rsh => {
                    let right_bits = match arith.right.if_constant() {
                        Some(s) => 32u32.saturating_sub(s),
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
                ArithOpType::Equal | ArithOpType::GreaterThan | ArithOpType::GreaterThanSigned => {
                    0..1
                }
                ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let c = c & 0x1f;
                        let left_bits = arith.left.relevant_bits();
                        let start = min(32, left_bits.start + c as u8);
                        let end = min(32, left_bits.end + c as u8);
                        if start <= end {
                            start..end
                        } else {
                            0..0
                        }
                    } else {
                        0..32
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let c = c & 0x1f;
                        let left_bits = arith.left.relevant_bits();
                        let start = left_bits.start.saturating_sub(c as u8);
                        let end = left_bits.end.saturating_sub(c as u8);
                        if start <= end {
                            start..end
                        } else {
                            0..0
                        }
                    } else {
                        0..32
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
                    if rel_left == (0..32) {
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
                    if higher_end >= 32 {
                        // Can overflow, so first bit can be nonzero
                        0..32
                    } else {
                        // Otherwise the highest nonzero bit is just one greater than before.
                        min(rel_left.start, rel_right.start)..(higher_end + 1)
                    }
                }
                ArithOpType::Mul => {
                    Operand::either(&arith.left, &arith.right, |x| x.if_constant())
                        .map(|(c, other)| {
                            if c == 0 {
                                0..0
                            } else {
                                let other_bits = other.relevant_bits();
                                let high = other_bits.end + (32 - c.leading_zeros() as u8 - 1);
                                if high > 32 {
                                    0..32
                                } else {
                                    other_bits.start..high
                                }
                            }
                        })
                        .unwrap_or(0..32)
                }
                _ => 0..32,
            },
            OperandType::Arithmetic64(_) => {
                0..64
            }
            OperandType::Constant(c) => {
                let trailing = c.trailing_zeros() as u8;
                let leading = c.leading_zeros() as u8;
                if 32 - leading < trailing {
                    0..0
                } else {
                    trailing..(32 - leading)
                }
            }
            OperandType::Constant64(c) => {
                let trailing = c.trailing_zeros() as u8;
                let leading = c.leading_zeros() as u8;
                if 64 - leading < trailing {
                    0..0
                } else {
                    trailing..(64 - leading)
                }
            }
            OperandType::Register64(..) => 0..64,
            _ => 0..32,
        }
    }

    /// Returns whether the operand is 8, 16, 32, or 64 bits.
    /// Relevant with signed multiplication, usually operands can be considered
    /// zero-extended u32.
    pub fn expr_size(&self) -> MemAccessSize {
        use self::OperandType::*;
        match *self {
            Memory(ref mem) => mem.size,
            Register(..) | Arithmetic(..) | Pair(..) | Xmm(..) | Flag(..) | Constant(..) |
                Undefined(..) | ArithmeticHigh(..) => MemAccessSize::Mem32,
            Register16(..) => MemAccessSize::Mem16,
            Register8High(..) | Register8Low(..) => MemAccessSize::Mem8,
            Register64(..) | Constant64(..) | Arithmetic64(..) => MemAccessSize::Mem64,
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

    fn new_simplified(ty: OperandType) -> Operand {
        Self::new(ty, true)
    }

    pub fn new_xmm(register: u8, word_id: u8) -> Operand {
        Operand::new_simplified(OperandType::Xmm(register, word_id))
    }

    pub fn new_not_simplified(ty: OperandType) -> Operand {
        Self::new(ty, false)
    }

    // TODO: Should not be pub?
    pub(crate) fn new_simplified_rc(ty: OperandType) -> Rc<Operand> {
        Self::new_simplified(ty).into()
    }

    pub fn new_not_simplified_rc(ty: OperandType) -> Rc<Operand> {
        Self::new_not_simplified(ty).into()
    }

    pub(crate) fn is_simplified(&self) -> bool {
        self.simplified
    }

    pub fn pair(s: &Rc<Operand>) -> (Rc<Operand>, Rc<Operand>) {
        use self::operand_helpers::*;
        match s.ty {
            OperandType::Pair(ref a, ref b) => (a.clone(), b.clone()),
            OperandType::Arithmetic(ref arith) => {
                let high_ty = OperandType::ArithmeticHigh(arith.clone());
                let high = Operand::new_not_simplified_rc(high_ty);
                (Operand::simplified(high), s.clone())
            }
            _ => {
                (constval(0), s.clone())
            }
        }
    }

    pub fn to_xmm_32(s: &Rc<Operand>, word: u8) -> Rc<Operand> {
        use self::operand_helpers::*;
        match s.ty {
            OperandType::Memory(ref mem) => match u32::from(word) {
                0 => s.clone(),
                x => mem32(operand_add(mem.address.clone(), constval(4 * x))),
            },
            OperandType::Register(reg) => {
                Operand::new_simplified(OperandType::Xmm(reg.0, word)).into()
            }
            _ => s.clone(),
        }
    }

    /// Return (low, high)
    pub fn to_xmm_64(s: &Rc<Operand>, word: u8) -> (Rc<Operand>, Rc<Operand>) {
        use self::operand_helpers::*;
        match s.ty {
            OperandType::Memory(ref mem) => match u32::from(word) {
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
                let low = Operand::new_simplified(OperandType::Xmm(reg.0, word * 2)).into();
                let high = Operand::new_simplified(OperandType::Xmm(reg.0, word * 2 + 1)).into();
                (low, high)
            }
            _ => panic!("Invalid value passed to to_xmm_64: {:?}", s),
        }
    }

    pub fn iter(&self) -> Iter {
        Iter(Some(IterState {
            pos: self,
            rhs: None,
        }))
    }

    pub fn iter_no_mem_addr(&self) -> IterNoMemAddr {
        IterNoMemAddr(Some(IterState {
            pos: self,
            rhs: None,
        }))
    }

    /// Returns which bits the operand will use at most.
    pub fn relevant_bits(&self) -> Range<u8> {
        self.relevant_bits.clone()
    }

    fn collect_add_ops(s: &Rc<Operand>, ops: &mut Vec<(Rc<Operand>, bool)>, negate: bool) {
        match s.ty {
            OperandType::Arithmetic(ref arith) if {
                arith.ty == ArithOpType::Add || arith.ty== ArithOpType::Sub
            } => {
                Operand::collect_add_ops(&arith.left, ops, negate);
                let negate_right = match arith.ty {
                    ArithOpType::Add => negate,
                    _ => !negate,
                };
                Operand::collect_add_ops(&arith.right, ops, negate_right);
            }
            _ => {
                let mut s = s.clone();
                if !s.is_simplified() {
                    // Simplification can cause it to be an add
                    s = Operand::simplified(s);
                    if let OperandType::Arithmetic(ref arith) = s.ty {
                        if arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub {
                            Operand::collect_add_ops(&s, ops, negate);
                            return;
                        }
                    }
                }
                ops.push((s, negate));
            }
        }
    }

    fn collect_mul_ops(s: &Rc<Operand>, ops: &mut Vec<Rc<Operand>>) {
        match s.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Mul => {
                Operand::collect_mul_ops(&arith.left, ops);
                Operand::collect_mul_ops(&arith.right, ops);
            }
            _ => {
                ops.push(Operand::simplified(s.clone()));
            }
        }
    }

    fn collect_signed_mul_ops(s: &Rc<Operand>, ops: &mut Vec<Rc<Operand>>) {
        match s.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::SignedMul => {
                Operand::collect_signed_mul_ops(&arith.left, ops);
                Operand::collect_signed_mul_ops(&arith.right, ops);
            }
            _ => {
                ops.push(Operand::simplified(s.clone()));
            }
        }
    }

    fn collect_and_ops(s: &Rc<Operand>, ops: &mut Vec<Rc<Operand>>) {
        match s.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                Operand::collect_and_ops(&arith.left, ops);
                Operand::collect_and_ops(&arith.right, ops);
            }
            _ => {
                ops.push(Operand::simplified(s.clone()));
            }
        }
    }

    fn collect_or_ops(s: &Rc<Operand>, ops: &mut Vec<Rc<Operand>>) {
        match s.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Or => {
                Operand::collect_or_ops(&arith.left, ops);
                Operand::collect_or_ops(&arith.right, ops);
            }
            _ => {
                ops.push(Operand::simplified(s.clone()));
            }
        }
    }

    fn collect_xor_ops(s: &Rc<Operand>, ops: &mut Vec<Rc<Operand>>) {
        match s.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Xor => {
                Operand::collect_xor_ops(&arith.left, ops);
                Operand::collect_xor_ops(&arith.right, ops);
            }
            _ => {
                ops.push(Operand::simplified(s.clone()));
            }
        }
    }

    // "Simplify bitwise and: merge child ors"
    // Converts things like [x | const1, x | const2] to [x | (const1 & const2)]
    fn simplify_and_merge_child_ors(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) {
        use self::operand_helpers::*;
        fn or_const(op: &Rc<Operand>) -> Option<(u32, &Rc<Operand>)> {
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
                new = Some(Operand::simplified(operand_or(val.clone(), ctx.constant(constant))));
            }
            if let Some(new) = new {
                *op = new;
            }
        }
    }

    // "Simplify bitwise or: merge child ands"
    // Converts things like [x & const1, x & const2] to [x & (const1 | const2)]
    fn simplify_or_merge_child_ands(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) {
        fn and_const(op: &Rc<Operand>) -> Option<(u32, &Rc<Operand>)> {
            match op.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                    Operand::either(&arith.left, &arith.right, |x| x.if_constant())
                }
                OperandType::Memory(ref mem) => match mem.size {
                    MemAccessSize::Mem8 => Some((0xff, op)),
                    MemAccessSize::Mem16 => Some((0xffff, op)),
                    _ => None,
                }
                _ => {
                    let bits = op.relevant_bits();
                    if bits != (0..32) && bits.start < bits.end {
                        let low = bits.start;
                        let high = 32 - bits.end;
                        Some((!0 >> low << low << high >> high, op))
                    } else {
                        None
                    }
                }
            }
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
                        let result =
                            try_merge_ands(other_val, val, other_constant, constant, ctx);
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
                new = Some(
                    simplify_and(
                        &new_val,
                        &ctx.constant(constant),
                        ctx,
                        &mut SimplifyWithZeroBits::default(),
                    )
                );
            }
            if let Some(new) = new {
                *op = new;
            }
        }
    }

    // Simplify or: merge memory
    // Converts (Mem32[x] >> 8) | (Mem32[x + 4] << 18) to Mem32[x + 1]
    fn simplify_or_merge_mem(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) {
        use self::operand_helpers::*;

        // Return (offset, len, value_offset)
        fn is_offset_mem(
            op: &Rc<Operand>,
            ctx: &OperandContext,
        ) -> Option<(Rc<Operand>, (u32, u32, u32))> {
            match op.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        if c & 0x7 == 0 && c < 0x20 {
                            let bytes = c / 8;
                            return is_offset_mem(&arith.left, ctx)
                                .map(|(x, (off, len, val_off))| (x, (off, len, val_off + bytes)));
                        }
                    }
                    None
                }
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Rsh => {
                    if let Some(c) = arith.right.if_constant() {
                        if c & 0x7 == 0 && c < 0x20 {
                            let bytes = c / 8;
                            return is_offset_mem(&arith.left, ctx)
                                .and_then(|(x, (off, len, val_off))| {
                                    if bytes < len {
                                        Some((x, (off.wrapping_add(bytes), len - bytes, val_off)))
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

        fn try_merge(
            val: &Rc<Operand>,
            shift: (u32, u32, u32),
            other_shift: (u32, u32, u32),
            ctx: &OperandContext,
        ) -> Option<Rc<Operand>> {
            let (shift, other_shift) = match (shift.2, other_shift.2) {
                (0, 0) => return None,
                (0, _) => (shift, other_shift),
                (_, 0) => (other_shift, shift),
                _ => return None,
            };
            let (off1, len1, _) = shift;
            let (off2, len2, val_off2) = other_shift;
            if off1.wrapping_add(len1) != off2 || len1 != val_off2 {
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

        let mut iter = VecDropIter::new(ops);
        while let Some(mut op) = iter.next() {
            let mut new = None;
            if let Some((val, shift)) = is_offset_mem(&op, ctx) {
                let mut second = iter.duplicate();
                while let Some(other_op) = second.next_removable() {
                    let mut remove = false;
                    if let Some((other_val, other_shift)) = is_offset_mem(&other_op, ctx) {
                        if val == other_val {
                            let result = try_merge(&val, other_shift, shift, ctx);
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

    pub fn const_offset(oper: &Rc<Operand>, ctx: &OperandContext) -> Option<(Rc<Operand>, u32)> {
        use crate::operand::operand_helpers::*;

        fn recurse(oper: &Rc<Operand>) -> Option<u32> {
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
        if s.simplified {
            return s;
        }
        let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
        match s.clone().ty {
            OperandType::Arithmetic(ref arith) => {
                let truncated;
                // Arithmetic (32-bit) is defined to have operands truncated to 32-bit.
                // The motivation being that Register64(0) & 0xffff_ffff != Register64(0),
                // and as such all the existing arith simplifications would have to
                // be reviewed for similar cases.
                let arith = if let Some(trunc) = arith.truncate_operands_to_32bit(ctx) {
                    truncated = trunc;
                    &truncated
                } else {
                    arith
                };
                let left = &arith.left;
                let right = &arith.right;
                match arith.ty {
                    ArithOpType::Add | ArithOpType::Sub => {
                        let is_sub = arith.ty == ArithOpType::Sub;
                        simplify_add_sub(left, right, is_sub, ctx)
                    }
                    ArithOpType::Mul => simplify_mul(left, right, ctx),
                    ArithOpType::SignedMul => simplify_signed_mul(left, right, ctx),
                    ArithOpType::And => simplify_and(left, right, ctx, swzb_ctx),
                    ArithOpType::Or => simplify_or(left, right, ctx),
                    ArithOpType::Xor => simplify_xor(left, right, ctx),
                    ArithOpType::Equal => simplify_eq(left, right, ctx),
                    ArithOpType::GreaterThan => {
                        let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
                        let l = left.clone();
                        let r = right.clone();
                        match (&l.ty, &r.ty) {
                            (&OperandType::Constant(a), &OperandType::Constant(b)) => if a > b {
                                return ctx.const_1();
                            } else {
                                return ctx.const_0();
                            },
                            (&OperandType::Arithmetic(ref arith), _) => {
                                if arith.ty == ArithOpType::Sub {
                                    if arith.left == r {
                                        let ty = OperandType::Arithmetic(ArithOperand {
                                            ty: ArithOpType::GreaterThan,
                                            left: arith.right.clone(),
                                            right: r.clone(),
                                        });
                                        return Operand::new_simplified_rc(ty);
                                    }
                                }
                            }
                            _ => (),
                        }
                        let ty = OperandType::Arithmetic(ArithOperand {
                            ty: ArithOpType::GreaterThan,
                            left,
                            right,
                        });
                        Operand::new_simplified_rc(ty)
                    }
                    ArithOpType::GreaterThanSigned => {
                        let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
                        let l = left.clone();
                        let r = right.clone();
                        if let (Some(a), Some(b)) = (l.if_constant(), r.if_constant()) {
                            match a as i32 > b as i32 {
                                true => return ctx.const_1(),
                                false => return ctx.const_0(),
                            }
                        }
                        let ty = OperandType::Arithmetic(ArithOperand {
                            ty: ArithOpType::GreaterThanSigned,
                            left,
                            right,
                        });
                        Operand::new_simplified_rc(ty)
                    }
                    ArithOpType::Lsh => simplify_lsh(left, right, ctx, swzb_ctx),
                    ArithOpType::Rsh => simplify_rsh(left, right, ctx, swzb_ctx),
                    ArithOpType::Div | ArithOpType::Modulo => {
                        let left = Operand::simplified_with_ctx(left.clone(), ctx, swzb_ctx);
                        let left = match left.ty {
                            OperandType::Pair(ref high, ref low) => {
                                match high.if_constant() == Some(0) {
                                    true => low.clone(),
                                    false => left.clone(),
                                }
                            }
                            _ => left.clone(),
                        };

                        let right = Operand::simplified_with_ctx(right.clone(), ctx, swzb_ctx);
                        let ty = OperandType::Arithmetic(ArithOperand {
                            ty: arith.ty,
                            left,
                            right,
                        });
                        Operand::new_simplified_rc(ty)
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
                }
            }
            OperandType::Memory(ref mem) => {
                Operand::new_simplified_rc(OperandType::Memory(MemAccess {
                    address: Operand::simplified_with_ctx(mem.address.clone(), ctx, swzb_ctx),
                    size: mem.size,
                }))
            }
            _ => mark_self_simplified(s),
        }
    }

    pub fn transform<F>(oper: &Rc<Operand>, mut f: F) -> Rc<Operand>
    where F: FnMut(&Rc<Operand>) -> Option<Rc<Operand>>
    {
        Operand::transform_internal(&oper, &mut f)
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
        fn mem_bits(val: MemAccessSize) -> u8 {
            match val {
                MemAccessSize::Mem64 => 64,
                MemAccessSize::Mem32 => 32,
                MemAccessSize::Mem16 => 16,
                MemAccessSize::Mem8 => 8,
            }
        }
        if let Some(mem) = val.if_memory() {
            // Transform also Mem16[mem.addr] to with & 0xffff if val is Mem32, etc.
            // I guess recursing inside mem.addr doesn't make sense here,
            // but didn't give it too much thought.
            Operand::transform(oper, |old| {
                old.if_memory()
                    .filter(|old| old.address == mem.address)
                    .filter(|old| mem_bits(old.size) <= mem_bits(mem.size))
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
    pub fn if_constant(&self) -> Option<u32> {
        match self.ty {
            OperandType::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Returns `Some(c)` if `self.ty` is `OperandType::Constant(c)` *or*
    /// `OperandType::Constant64(c)`
    pub fn if_constant64(&self) -> Option<u64> {
        match self.ty {
            OperandType::Constant(c) => Some(c as u64),
            OperandType::Constant64(c) => Some(c),
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

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Add(left, right))`
    pub fn if_arithmetic_add(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        match self.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
                Some((&arith.left, &arith.right))
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Mul(left, right))`
    pub fn if_arithmetic_mul(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        match self.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Mul => {
                Some((&arith.left, &arith.right))
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Equal(left, right))`
    pub fn if_arithmetic_eq(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        match self.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Equal => {
                Some((&arith.left, &arith.right))
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::And(left, right))`
    pub fn if_arithmetic_and(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        match self.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::And => {
                Some((&arith.left, &arith.right))
            },
            _ => None,
        }
    }

    /// Returns `Some((left, right))` if `self.ty` is
    /// `OperandType::Arithmetic(ArithOpType::Or(left, right))`
    pub fn if_arithmetic_or(&self) -> Option<(&Rc<Operand>, &Rc<Operand>)> {
        match self.ty {
            OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Or => {
                Some((&arith.left, &arith.right))
            },
            _ => None,
        }
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

fn simplify_add_sub(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    is_sub: bool,
    ctx: &OperandContext,
) -> Rc<Operand> {
    use self::ArithOpType::*;
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    let mut ops = simplify_add_sub_ops(left, right, is_sub, ctx);
    let mut tree = match ops.pop() {
        Some((s, neg)) => match neg {
            false => mark_self_simplified(s),
            true => {
                let op_ty = OperandType::Arithmetic(ArithOperand {
                    ty: Sub,
                    left: ctx.const_0(),
                    right: s,
                });
                Operand::new_simplified_rc(op_ty)
            }
        },
        None => ctx.const_0(),
    };
    while let Some((op, neg)) = ops.pop() {
        let op_ty = OperandType::Arithmetic(ArithOperand {
            ty: if neg { Sub } else { Add},
            left: tree,
            right: op,
        });
        tree = Operand::new_simplified_rc(op_ty);
    }
    tree
}

fn simplify_mul(left: &Rc<Operand>, right: &Rc<Operand>, ctx: &OperandContext) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    let mut ops = vec![];
    Operand::collect_mul_ops(left, &mut ops);
    Operand::collect_mul_ops(right, &mut ops);
    let const_product = ops.iter().flat_map(|x| x.if_constant())
        .fold(1u32, |product, x| product.wrapping_mul(x));
    if const_product == 0 {
        return ctx.const_0();
    }
    ops.retain(|x| x.if_constant().is_none());
    if ops.is_empty() {
        return ctx.constant(const_product);
    }
    ops.sort();
    if const_product != 1 {
        if ops.len() == 1 {
            if simplify_mul_should_apply_constant(&ops[0]) {
                let op = ops.swap_remove(0);
                return simplify_mul_apply_constant(&op, const_product, ctx);
            }
            let new = simplify_mul_try_mul_constants(&ops[0], const_product, ctx);
            if let Some(new) = new {
                return new;
            }
        }
        ops.push(ctx.constant(const_product));
    }
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_1());
    while let Some(op) = ops.pop() {
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty: ArithOpType::Mul,
            left: tree,
            right: op,
        }));
    }
    tree
}

fn simplify_signed_mul(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    let mut ops = vec![];
    Operand::collect_signed_mul_ops(left, &mut ops);
    Operand::collect_signed_mul_ops(right, &mut ops);
    let const_product = ops.iter().flat_map(|x| x.if_constant())
        .fold(1i32, |product, x| product.wrapping_mul(x as i32)) as u32;
    if const_product == 0 {
        return ctx.const_0();
    }
    ops.retain(|x| x.if_constant().is_none());
    if ops.is_empty() {
        return ctx.constant(const_product);
    }
    ops.sort();
    // If there are no small operands, equivalent to unsigned multiply
    // Maybe could assume if there's even one 32-bit operand? As having different
    // size operands is sketchy.
    let all_32bit = ops.iter().all(|x| x.ty.expr_size() == MemAccessSize::Mem32);
    if const_product != 1 {
        if ops.len() == 1 {
            if simplify_mul_should_apply_constant(&ops[0]) {
                let op = ops.swap_remove(0);
                return simplify_mul_apply_constant(&op, const_product, ctx);
            }
            let new = simplify_mul_try_mul_constants(&ops[0], const_product, ctx);
            if let Some(new) = new {
                return new;
            }
        }
        ops.push(ctx.constant(const_product));
    }
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_1());
    while let Some(op) = ops.pop() {
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty: if all_32bit { ArithOpType::Mul } else { ArithOpType::SignedMul },
            left: tree,
            right: op,
        }));
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

fn simplify_mul_apply_constant(op: &Rc<Operand>, val: u32, ctx: &OperandContext) -> Rc<Operand> {
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
    c: u32,
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
    }
}

fn simplify_add_sub_ops(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    is_sub: bool,
    ctx: &OperandContext,
) -> Vec<(Rc<Operand>, bool)> {
    let mut ops = Vec::new();
    Operand::collect_add_ops(left, &mut ops, false);
    Operand::collect_add_ops(right, &mut ops, is_sub);
    let const_sum = ops.iter()
        .flat_map(|&(ref x, neg)| x.if_constant().map(|x| (x, neg)))
        .fold(0u32, |sum, (x, neg)| match neg {
            false => sum.wrapping_add(x),
            true => sum.wrapping_sub(x),
        });
    ops.retain(|&(ref x, _)| x.if_constant().is_none());
    ops.sort();
    simplify_add_merge_muls(&mut ops, ctx);
    if ops.is_empty() {
        return vec![(ctx.constant(const_sum), false)];
    }
    if const_sum != 0 {
        if const_sum > 0x8000_0000 {
            ops.push((ctx.constant(0u32.wrapping_sub(const_sum)), true));
        } else {
            ops.push((ctx.constant(const_sum), false));
        }
    }
    // Place non-negated terms last so the simplified result doesn't become
    // (0 - x) + y
    ops.sort_by(|&(ref a_val, a_neg), &(ref b_val, b_neg)| {
        (b_neg, b_val).cmp(&(a_neg, a_val))
    });
    ops
}

fn simplify_eq(left: &Rc<Operand>, right: &Rc<Operand>, ctx: &OperandContext) -> Rc<Operand> {
    use self::operand_helpers::*;

    let (left, right) = match left < right {
        true => (left, right),
        false => (right, left),
    };

    let mut ops = simplify_add_sub_ops(left, right, true, ctx);
    let mark_self_simplified = |s: &Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());
    match ops.len() {
        0 => ctx.const_1(),
        1 => match ops[0].0.ty {
            OperandType::Constant(0) => ctx.const_1(),
            OperandType::Constant(_) => ctx.const_0(),
            _ => {
                if let OperandType::Arithmetic(ref arith) = ops[0].0.ty {
                    if arith.ty == ArithOpType::Equal {
                        // Check for (x == 0) == 0
                        let either_const =
                            Operand::either(&arith.left, &arith.right, |x| x.if_constant());
                        if let Some((0, other)) = either_const {
                            if let OperandType::Arithmetic(ref arith) = other.ty {
                                if arith.is_compare_op() {
                                    return other.clone();
                                }
                            }
                        }
                    }
                }
                let op = mark_self_simplified(&ops[0].0);
                let ty = OperandType::Arithmetic(ArithOperand {
                    ty: ArithOpType::Equal,
                    left: op,
                    right: ctx.const_0(),
                });
                Operand::new_simplified_rc(ty)
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
            let left = match left.1 {
                true => mark_self_simplified(&left.0),
                false => Operand::simplified(operand_sub(ctx.const_0(), left.0.clone())),
            };
            let right = match right.1 {
                false => mark_self_simplified(&right.0),
                true => Operand::simplified(operand_sub(ctx.const_0(), right.0.clone())),
            };
            simplify_eq_2_ops(left, right, ctx)
        },
        _ => {
            let mut tree = match ops.pop() {
                Some((s, neg)) => match neg {
                    false => mark_self_simplified(&s),
                    true => {
                        let op_ty = OperandType::Arithmetic(ArithOperand {
                            ty: ArithOpType::Sub,
                            left: ctx.const_0(),
                            right: s,
                        });
                        Operand::new_simplified_rc(op_ty)
                    }
                },
                None => ctx.const_0(),
            };
            while let Some((op, neg)) = ops.pop() {
                let op_ty = OperandType::Arithmetic(ArithOperand {
                    ty: if neg { ArithOpType::Sub } else { ArithOpType::Add },
                    left: tree,
                    right: op,
                });
                tree = Operand::new_simplified_rc(op_ty);
            }
            let ty = match tree.ty {
                OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
                    OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::Equal,
                        left: arith.left.clone(),
                        right: arith.right.clone(),
                    })
                }
                _ => {
                    OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::Equal,
                        left: tree.clone(),
                        right: ctx.const_0(),
                    })
                }
            };
            Operand::new_simplified_rc(ty)
        }
    }
}

fn simplify_eq_2_ops(left: Rc<Operand>, right: Rc<Operand>, ctx: &OperandContext) -> Rc<Operand> {
    fn mask_maskee(x: &Rc<Operand>) -> Option<(u32, &Rc<Operand>)> {
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
            if let OperandType::Arithmetic(ref arith) = other.ty {
                if arith.is_compare_op() {
                    return other.clone();
                }
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
                    let a = simplify_with_and_mask(a, mask1, ctx);
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
    let ty = OperandType::Arithmetic(ArithOperand {
        ty: ArithOpType::Equal,
        left,
        right,
    });
    Operand::new_simplified_rc(ty)
}

fn simplify_eq_masked_add(operand: &Rc<Operand>) -> Option<(u32, &Rc<Operand>)> {
    match operand.ty {
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
            arith.left.if_constant().map(|c| (c, &arith.right))
                .or_else(|| arith.left.if_constant().map(|c| (c, &arith.left)))
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
            arith.left.if_constant().map(|c| (0u32.wrapping_sub(c), &arith.right))
                .or_else(|| arith.right.if_constant().map(|c| (0u32.wrapping_sub(c), &arith.left)))
        }
        _ => None,
    }
}

// Tries to merge (a & a_mask) | (b & b_mask) to (a_mask | b_mask) & result
fn try_merge_ands(
    a: &Rc<Operand>,
    b: &Rc<Operand>,
    a_mask: u32,
    b_mask: u32,
    ctx: &OperandContext,
) -> Option<Rc<Operand>>{
    use self::operand_helpers::*;
    if a == b {
        return Some(a.clone());
    } else if a_mask & b_mask != 0 {
        return None;
    }
    match (&a.ty, &b.ty) {
        (&OperandType::Arithmetic(ref a), &OperandType::Arithmetic(ref b)) => {
            if a.ty == ArithOpType::Xor && b.ty == ArithOpType::Xor {
                try_merge_ands(&a.left, &b.left, a_mask, b_mask, ctx).and_then(|left| {
                    try_merge_ands(&a.right, &b.right, a_mask, b_mask, ctx).map(|right| (left, right))
                }).or_else(|| try_merge_ands(&a.left, &b.right, a_mask, b_mask, ctx).and_then(|first| {
                    try_merge_ands(&a.right, &b.left, a_mask, b_mask, ctx).map(|second| (first, second))
                })).map(|(first, second)| {
                    Operand::simplified(operand_xor(first, second))
                })
            } else {
                None
            }
        }
        (&OperandType::Constant(a), &OperandType::Constant(b)) => Some(ctx.constant(a | b)),
        (&OperandType::Memory(ref a_mem), &OperandType::Memory(ref b_mem)) => {
            // Can treat Mem16[x], Mem8[x] as Mem16[x], Mem16[x]
            if a_mem.address == b_mem.address {
                let check_mask = |op: &Rc<Operand>, mask: u32, ok: &Rc<Operand>| {
                    if op.relevant_bits().end >= 32 - mask.leading_zeros() as u8 {
                        Some(ok.clone())
                    } else {
                        None
                    }
                };
                match (a_mem.size, b_mem.size) {
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

fn simplify_and(
    left: &Rc<Operand>,
    right: &Rc<Operand>,
    ctx: &OperandContext,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());

    let mut ops = vec![];
    Operand::collect_and_ops(left, &mut ops);
    Operand::collect_and_ops(right, &mut ops);
    let mut const_remain = !0u32;
    loop {
        const_remain = ops.iter().flat_map(|x| x.if_constant())
            .fold(const_remain, |sum, x| sum & x);
        ops.retain(|x| x.if_constant().is_none());
        if ops.is_empty() {
            if const_remain == !0 {
                return ctx.const_0();
            } else {
                return ctx.constant(const_remain);
            }
        }
        ops.sort();
        ops.dedup();
        if const_remain != !0 {
            vec_filter_map(&mut ops, |op| {
                let new = simplify_with_and_mask(&op, const_remain, ctx);
                if let OperandType::Constant(c) = new.ty {
                    const_remain &= c;
                    None
                } else {
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
        let mut new_ops = vec![];
        for op in &ops {
            if let Some((l, r)) = op.if_arithmetic_and() {
                Operand::collect_and_ops(l, &mut new_ops);
                Operand::collect_and_ops(r, &mut new_ops);
            }
        }
        if new_ops.is_empty() {
            break;
        }
        ops.retain(|x| x.if_arithmetic_and().is_none());
        ops.extend(new_ops);
    }
    Operand::simplify_and_merge_child_ors(&mut ops, ctx);
    let relevant_bits = ops.iter().fold(!0, |bits, op| {
        let relevant_bits = op.relevant_bits();
        let low = relevant_bits.start;
        let high = 32 - relevant_bits.end;
        let mask = !0 >> low << low << high >> high;
        bits & mask
    });
    // Don't push a const mask which has all 1s for relevant bits.
    if const_remain & relevant_bits != relevant_bits {
        ops.push(ctx.constant(const_remain));
    }
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty: ArithOpType::And,
            left: tree,
            right: op,
        }));
    }
    tree
}

fn simplify_or(left: &Rc<Operand>, right: &Rc<Operand>, ctx: &OperandContext) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());

    let mut ops = vec![];
    Operand::collect_or_ops(left, &mut ops);
    Operand::collect_or_ops(right, &mut ops);
    let const_val = ops.iter().flat_map(|x| x.if_constant())
        .fold(0u32, |sum, x| sum | x);
    ops.retain(|x| x.if_constant().is_none());
    if ops.is_empty() {
        return ctx.constant(const_val);
    }
    ops.sort();
    ops.dedup();
    for bits in one_bit_ranges(const_val) {
        vec_filter_map(&mut ops, |op| simplify_with_one_bits(&op, &bits, ctx));
    }
    Operand::simplify_or_merge_child_ands(&mut ops, ctx);
    Operand::simplify_or_merge_mem(&mut ops, ctx);
    if const_val != 0 {
        ops.push(ctx.constant(const_val));
    }
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty: ArithOpType::Or,
            left: tree,
            right: op,
        }));
    }
    tree
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
        let ty = OperandType::Arithmetic(ArithOperand {
            ty: ArithOpType::Lsh,
            left: left.clone(),
            right: right.clone(),
        });
        Operand::new_simplified_rc(ty)
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left.clone();
    } else if constant >= 0x20 {
        return ctx.const_0();
    } else {
        let zero_bits = (0x20 - constant as u8)..32;
        match simplify_with_zero_bits(&left, &zero_bits, ctx, swzb_ctx) {
            None => return ctx.const_0(),
            Some(s) => {
                if s != left {
                    return simplify_lsh(&s, &right, ctx, swzb_ctx);
                }
            }
        }
    }
    match left.ty {
        OperandType::Constant(a) => ctx.constant(a << (constant & 0x1f)),
        OperandType::Arithmetic(ref arith) => match arith.ty {
            ArithOpType::And => {
                let zero_bits = (0x20 - constant as u8)..32;
                let mut ops = vec![];
                Operand::collect_and_ops(&left, &mut ops);
                let low = zero_bits.start;
                let high = 32 - zero_bits.end;
                let no_op_mask = !(!0 >> low << low << high >> high);
                ops.retain(|x| match x.ty {
                    OperandType::Constant(c) => c != no_op_mask,
                    _ => true,
                });
                ops.sort();
                let single_operand = ops.len() == 1;
                let mut tree = match ops.pop() {
                    Some(s) => s,
                    None => return ctx.const_0(),
                };
                while let Some(op) = ops.pop() {
                    tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::And,
                        left: tree,
                        right: op,
                    }));
                }
                // If we got rid of the or, the remaining operand may simplify further,
                // otherwise avoid recursion.
                if single_operand {
                    simplify_lsh(&tree, &right, ctx, swzb_ctx)
                } else {
                    let ty = OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::Lsh,
                        left: tree,
                        right: right.clone(),
                    });
                    Operand::new_simplified_rc(ty)
                }
            }
            ArithOpType::Xor => {
                // Try to simplify any parts of the xor separately
                let mut ops = vec![];
                Operand::collect_xor_ops(&left, &mut ops);
                if ops.len() > 16 {
                    // Give up on dumb long xors
                    default()
                } else {
                    for op in &mut ops {
                        *op = simplify_lsh(op, &right, ctx, swzb_ctx);
                    }
                    simplify_xor_ops(&mut ops, ctx)
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
                    if sum < 0x20 {
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
                    if rsh_const >= 0x20 {
                        return ctx.const_0();
                    }
                    let mask = (!0u32 >> rsh_const) << constant;
                    let tmp;
                    let val = match diff {
                        0 => &arith.left,
                        // (x >> rsh) << lsh, rsh > lsh
                        x if x > 0 => {
                            tmp = simplify_rsh(
                                &arith.left,
                                &ctx.constant(x as u32),
                                ctx,
                                swzb_ctx,
                            );
                            &tmp
                        }
                        // (x >> rsh) << lsh, lsh > rsh
                        x => {
                            tmp = simplify_lsh(
                                &arith.left,
                                &ctx.constant(x.abs() as u32),
                                ctx,
                                swzb_ctx,
                            );
                            &tmp
                        }
                    };
                    simplify_and(&val, &ctx.constant(mask), ctx, swzb_ctx)
                } else {
                    default()
                }
            }
            _ => default(),
        },
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
        let ty = OperandType::Arithmetic(ArithOperand {
            ty: ArithOpType::Rsh,
            left: left.clone(),
            right: right.clone(),
        });
        Operand::new_simplified_rc(ty)
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left.clone();
    } else if constant >= 0x20 {
        return ctx.const_0();
    } else {
        let zero_bits = 0..((constant & 0x1f) as u8);
        match simplify_with_zero_bits(&left, &zero_bits, ctx, swzb_ctx) {
            None => return ctx.const_0(),
            Some(s) => {
                if s != left {
                    return simplify_rsh(&s, &right, ctx, swzb_ctx);
                }
            }
        }
    }

    match left.ty {
        OperandType::Constant(a) => ctx.constant(a >> constant),
        OperandType::Arithmetic(ref arith) => match arith.ty {
            ArithOpType::And => {
                let zero_bits = 0..(constant as u8);
                let mut ops = vec![];
                Operand::collect_and_ops(&left, &mut ops);
                let low = zero_bits.start;
                let high = 32 - zero_bits.end;
                let no_op_mask = !(!0 >> low << low << high >> high);
                ops.retain(|x| match x.ty {
                    OperandType::Constant(c) => c != no_op_mask,
                    _ => true,
                });
                ops.sort();
                let single_operand = ops.len() == 1;
                let mut tree = match ops.pop() {
                    Some(s) => s,
                    None => return ctx.const_0(),
                };
                while let Some(op) = ops.pop() {
                    tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::And,
                        left: tree,
                        right: op,
                    }));
                }
                // If we got rid of the or, the remaining operand may simplify further,
                // otherwise avoid recursion.
                if single_operand {
                    simplify_rsh(&tree, &right, ctx, swzb_ctx)
                } else {
                    let ty = OperandType::Arithmetic(ArithOperand {
                        ty: ArithOpType::Rsh,
                        left: tree,
                        right: right.clone(),
                    });
                    Operand::new_simplified_rc(ty)
                }
            }
            ArithOpType::Xor => {
                // Try to simplify any parts of the xor separately
                let mut ops = vec![];
                Operand::collect_xor_ops(&left, &mut ops);
                if ops.len() > 16 {
                    // Give up on dumb long xors
                    default()
                } else {
                    for op in &mut ops {
                        *op = simplify_rsh(op, &right, ctx, swzb_ctx);
                    }
                    simplify_xor_ops(&mut ops, ctx)
                }
            }
            ArithOpType::Lsh => {
                if let Some(lsh_const) = arith.right.if_constant() {
                    if lsh_const >= 0x20 {
                        return ctx.const_0();
                    }
                    let diff = constant as i8 - lsh_const as i8;
                    let mask = (!0u32 << lsh_const) >> constant;
                    let tmp;
                    let val = match diff {
                        0 => &arith.left,
                        // (x << rsh) >> lsh, rsh > lsh
                        x if x > 0 => {
                            tmp = simplify_rsh(&arith.left, &ctx.constant(x as u32), ctx, swzb_ctx);
                            &tmp
                        }
                        // (x << rsh) >> lsh, lsh > rsh
                        x => {
                            tmp =
                                simplify_lsh(&arith.left, &ctx.constant(x.abs() as u32), ctx, swzb_ctx);
                            &tmp
                        }
                    };
                    simplify_and(val, &ctx.constant(mask), ctx, swzb_ctx)
                } else {
                    default()
                }
            }
            ArithOpType::Rsh => {
                if let Some(inner_const) = arith.right.if_constant() {
                    let sum = inner_const.saturating_add(constant);
                    if sum < 0x20 {
                        simplify_rsh(&arith.left, &ctx.constant(sum), ctx, swzb_ctx)
                    } else {
                        ctx.const_0()
                    }
                } else {
                    default()
                }
            }
            _ => default(),
        },
        OperandType::Memory(ref mem) => {
            match mem.size {
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

/// Convert and(x, mask) to x
fn simplify_with_and_mask(op: &Rc<Operand>, mask: u32, ctx: &OperandContext) -> Rc<Operand> {
    use self::operand_helpers::*;
    match op.ty {
        OperandType::Arithmetic(ref arith) => match arith.ty {
            ArithOpType::And => {
                match (&arith.left.ty, &arith.right.ty) {
                    (&OperandType::Constant(c), _) => if c == mask {
                        return arith.right.clone();
                    } else if c & mask == 0 {
                        return ctx.const_0();
                    },
                    (_, &OperandType::Constant(c)) => if c == mask {
                        return arith.left.clone();
                    } else if c & mask == 0 {
                        return ctx.const_0();
                    },
                    _ => (),
                }
                let simplified_left = simplify_with_and_mask(&arith.left, mask, ctx);
                let simplified_right = simplify_with_and_mask(&arith.right, mask, ctx);
                if simplified_left == arith.left && simplified_right == arith.right {
                    op.clone()
                } else {
                    Operand::simplified(operand_and(simplified_left, simplified_right))
                }
            }
            ArithOpType::Or => {
                let simplified_left = simplify_with_and_mask(&arith.left, mask, ctx);
                if let OperandType::Constant(c) = simplified_left.ty {
                    if mask & c == mask {
                        return simplified_left;
                    }
                }
                let simplified_right = simplify_with_and_mask(&arith.right, mask, ctx);
                if let OperandType::Constant(c) = simplified_right.ty {
                    if mask & c == mask {
                        return simplified_right;
                    }
                }
                if simplified_left == arith.left && simplified_right == arith.right {
                    op.clone()
                } else {
                    Operand::simplified(operand_or(simplified_left, simplified_right))
                }
            }
            ArithOpType::Xor => {
                let simplified_left = simplify_with_and_mask(&arith.left, mask, ctx);
                let simplified_right = simplify_with_and_mask(&arith.right, mask, ctx);
                if simplified_left == arith.left && simplified_right == arith.right {
                    op.clone()
                } else {
                    Operand::simplified(operand_xor(simplified_left, simplified_right))
                }
            }
            _ => op.clone(),
        },
        _ => op.clone(),
    }
}

#[derive(Default)]
struct SimplifyWithZeroBits {
    simplify_count: u32,
}

/// Simplifies `op` when the bits in the range `bits` are guaranteed to be zero.
/// Returning `None` is considered same as `Some(constval(0))` (The value gets optimized out in
/// bitwise and).
fn simplify_with_zero_bits(
    op: &Rc<Operand>,
    bits: &Range<u8>,
    ctx: &OperandContext,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<Rc<Operand>> {
    fn try_remove_const_and_mask<'a>(
        op: &'a Rc<Operand>,
        bits: &Range<u8>,
    ) -> Option<&'a Rc<Operand>> {
        op.if_arithmetic_and()
            .and_then(|(l, r)| Operand::either(l, r, |x| {
                x.if_constant().and_then(|constant| {
                    let rel_bits = x.relevant_bits();
                    let low = rel_bits.start;
                    let high = 32 - rel_bits.end;
                    let mask = !0 >> low << low << high >> high;
                    if constant == mask {
                        Some(())
                    } else {
                        None
                    }
                })
            }))
            .and_then(|((), other)| {
                let other_bits = other.relevant_bits();
                // Can remove the constant mask if the addition won't overflow
                // without it. Any bits the const mask would clear will be cleared
                // since `bits` are zero.
                let can_remove_const = bits.end == 32 &&
                    other_bits.end < 32 &&
                    other_bits.end > bits.start &&
                    other_bits.start <= bits.start;
                if can_remove_const {
                    Some(other)
                } else {
                    None
                }
            })
    }

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
        OperandType::Arithmetic(ref arith) => match arith.ty {
            ArithOpType::And | ArithOpType::Or | ArithOpType::Xor |
                ArithOpType::Lsh | ArithOpType::Rsh => true,
            _ => false,
        },
        _ => false,
    };

    if recurse_check {
        if swzb.simplify_count > 40 {
            // Give up
            return Some(op.clone());
        } else {
            swzb.simplify_count += 1;
        }
    }

    let result = match op.ty {
        OperandType::Arithmetic(ref arith) => {
            let left = &arith.left;
            let right = &arith.right;
            match arith.ty {
                ArithOpType::And => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    match simplified_left {
                        Some(l) => {
                            let simplified_right =
                                simplify_with_zero_bits(right, bits, ctx, swzb);
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
                    }
                }
                ArithOpType::Or => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    let simplified_right = simplify_with_zero_bits(right, bits, ctx, swzb);
                    match (simplified_left, simplified_right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l == *left && r == *right {
                                Some(op.clone())
                            } else {
                                Some(simplify_or(&l, &r, ctx))
                            }
                        }
                    }
                }
                ArithOpType::Xor => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    let simplified_right = simplify_with_zero_bits(right, bits, ctx, swzb);
                    match (simplified_left, simplified_right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l == *left && r == *right {
                                Some(op.clone())
                            } else {
                                Some(simplify_xor(&l, &r, ctx))
                            }
                        }
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(c) = right.if_constant() {
                        if bits.end >= 0x20 && bits.start <= c as u8 {
                            None
                        } else {
                            let low = bits.start.saturating_sub(c as u8);
                            let high = bits.end.saturating_sub(c as u8);
                            if low >= high {
                                return Some(op.clone());
                            }
                            let result = simplify_with_zero_bits(left, &(low..high), ctx, swzb);
                            if let Some(result) =  result {
                                if result != *left {
                                    Some(simplify_lsh(&result, right, ctx, swzb))
                                } else {
                                    Some(op.clone())
                                }
                            } else {
                                Some(op.clone())
                            }
                        }
                    } else {
                        Some(op.clone())
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = right.if_constant() {
                        if bits.start == 0 && c as u8 >= (0x20 - bits.end) {
                            None
                        } else {
                            let low = bits.start.saturating_add(c as u8).min(0x20);
                            let high = bits.end.saturating_add(c as u8).min(0x20);
                            if low >= high {
                                return Some(op.clone());
                            }
                            let result = simplify_with_zero_bits(left, &(low..high), ctx, swzb);
                            if let Some(result) =  result {
                                if result != *left {
                                    Some(simplify_rsh(&result, right, ctx, swzb))
                                } else {
                                    Some(op.clone())
                                }
                            } else {
                                Some(op.clone())
                            }
                        }
                    } else {
                        Some(op.clone())
                    }
                }
                ArithOpType::Add => {
                    let new_left = try_remove_const_and_mask(left, bits);
                    let new_right = try_remove_const_and_mask(right, bits);
                    if new_left.is_some() || new_right.is_some() {
                        let new_left = new_left.unwrap_or(&left);
                        let new_right = new_right.unwrap_or(&right);
                        Some(simplify_add_sub(new_left, new_right, false, ctx))
                    } else {
                        Some(op.clone())
                    }
                }
                _ => Some(op.clone()),
            }
        }
        OperandType::Constant(c) => {
            let low = bits.start;
            let high = 32 - bits.end;
            let mask = !(!0 >> low << low << high >> high);
            let new_val = c & mask;
            match new_val {
                0 => None,
                c => Some(ctx.constant(c)),
            }
        }
        OperandType::Memory(ref mem) => {
            if bits.start == 0 && bits.end >= relevant_bits.end {
                None
            } else if bits.end == 32 {
                if bits.start <= 8 && relevant_bits.end > 8 {
                    Some(mem_variable_rc(MemAccessSize::Mem8, mem.address.clone()))
                } else if bits.start <= 16 && relevant_bits.end > 16 {
                    Some(mem_variable_rc(MemAccessSize::Mem16, mem.address.clone()))
                } else {
                    Some(op.clone())
                }
            } else {
                Some(op.clone())
            }
        }
        _ => Some(op.clone()),
    };
    result
}

/// Simplifies `op` when the bits in the range `bits` are guaranteed to be one.
/// Returning `None` means that `op | constval(bits) == constval(bits)`
fn simplify_with_one_bits(
    op: &Rc<Operand>,
    bits: &Range<u8>,
    ctx: &OperandContext,
) -> Option<Rc<Operand>> {
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
                    let left = simplify_with_one_bits(left, bits, ctx);
                    let right = simplify_with_one_bits(right, bits, ctx);
                    match (left, right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => {
                            let low = bits.start;
                            let high = 32 - bits.end;
                            let mask = !0 >> low << low << high >> high;
                            Some(Operand::simplified(operand_and(ctx.constant(mask), s)))
                        }
                        (Some(l), Some(r)) => Some(Operand::simplified(operand_and(l, r))),
                    }
                }
                ArithOpType::Or => {
                    let left = simplify_with_one_bits(left, bits, ctx);
                    let right = simplify_with_one_bits(right, bits, ctx);
                    match (left, right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => Some(Operand::simplified(operand_or(l, r))),
                    }
                }
                _ => default(),
            }
        }
        OperandType::Constant(c) => {
            let low = bits.start;
            let high = 32 - bits.end;
            let mask = !0 >> low << low << high >> high;
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
            } else if bits.end == 32 {
                if bits.start <= 8 && max_bits.end > 8 {
                    Some(mem_variable_rc(MemAccessSize::Mem8, mem.address.clone()))
                } else if bits.start <= 16 && max_bits.end > 16 {
                    Some(mem_variable_rc(MemAccessSize::Mem16, mem.address.clone()))
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
fn simplify_add_merge_muls(ops: &mut Vec<(Rc<Operand>, bool)>, ctx: &OperandContext) {
    use self::operand_helpers::*;

    fn count_equivalent_opers(ops: &[(Rc<Operand>, bool)], equiv: &Operand) -> u32 {
        ops.iter().map(|&(ref o, neg)| {
            let (mul, val) = o.if_arithmetic_mul()
                .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                .unwrap_or_else(|| (1, o));
            match *equiv == **val {
                true => if neg { 0u32.wrapping_sub(mul) } else { mul },
                false => 0,
            }
        }).fold(0, |sum, next| sum.wrapping_add(next))
    }

    let mut pos = 0;
    while pos < ops.len() {
        let merged = {
            let (self_mul, op) = ops[pos].0.if_arithmetic_mul()
                .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                .unwrap_or_else(|| (1, &ops[pos].0));

            let others = count_equivalent_opers(&ops[pos + 1..], op);
            if others != 0 {
                let self_mul = if ops[pos].1 { 0u32.wrapping_sub(self_mul) } else { self_mul };
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
                    let is_equiv = match ops[other_pos].0.ty {
                        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Mul => {
                            match (&arith.left.ty, &arith.right.ty) {
                                (&OperandType::Constant(_), _) => arith.right == equiv,
                                (_, &OperandType::Constant(_)) => arith.left == equiv,
                                _ => ops[other_pos].0 == equiv,
                            }
                        }
                        _ => ops[other_pos].0 == equiv,
                    };
                    if is_equiv {
                        ops.remove(other_pos);
                    } else {
                        other_pos += 1;
                    }
                }
                if sum > 0x8000_0000 {
                    let sum = !sum.wrapping_add(1);
                    ops[pos].0 = Operand::simplified(operand_mul(ctx.constant(sum), equiv));
                    ops[pos].1 = true;
                } else {
                    ops[pos].0 = Operand::simplified(operand_mul(ctx.constant(sum), equiv));
                    ops[pos].1 = false;
                }
                pos += 1;
            }
            // Remove everything matching
            Some(None) => {
                let (op, _) = ops.remove(pos);
                let equiv = match op.ty {
                    OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Mul => {
                        match (&arith.left.ty, &arith.right.ty) {
                            (&OperandType::Constant(_), _) => &arith.right,
                            (_, &OperandType::Constant(_)) => &arith.left,
                            _ => &op,
                        }
                    },
                    _ => &op,
                };
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

fn simplify_xor(left: &Rc<Operand>, right: &Rc<Operand>, ctx: &OperandContext) -> Rc<Operand> {
    let mut ops = vec![];
    Operand::collect_xor_ops(left, &mut ops);
    Operand::collect_xor_ops(right, &mut ops);
    simplify_xor_ops(&mut ops, ctx)
}

fn simplify_xor_ops(ops: &mut Vec<Rc<Operand>>, ctx: &OperandContext) -> Rc<Operand> {
    let mark_self_simplified = |s: Rc<Operand>| Operand::new_simplified_rc(s.ty.clone());

    let const_val = ops.iter().flat_map(|x| x.if_constant())
        .fold(0u32, |sum, x| sum ^ x);
    ops.retain(|x| x.if_constant().is_none());
    ops.sort();
    simplify_xor_remove_reverting(ops);
    if ops.is_empty() {
        return ctx.constant(const_val);
    }
    if const_val != 0 {
        if ops.len() == 1 {
            // Try some conversations
            let op = &ops[0];
            // Convert c1 ^ (y & c2) == (y ^ (c1 & c2)) & c2
            if let Some((l, r)) = op.if_arithmetic_and() {
                let vals = match (&l.ty, &r.ty) {
                    (&OperandType::Constant(c), _) => Some((l, c, r)),
                    (_, &OperandType::Constant(c)) => Some((r, c, l)),
                    _ => None,
                };
                if let Some((and_const, c, other)) = vals {
                    if const_val & c == 0 {
                        return op.clone();
                    } else {
                        return simplify_and(
                            and_const,
                            &simplify_xor(&ctx.constant(const_val & c), other, ctx),
                            ctx,
                            &mut SimplifyWithZeroBits::default(),
                        );
                    }
                }
            }
            // Convert c1 ^ ((c2 & x) | y) to (c2 & x) | (y ^ c1)
            // if c1 & c2 == 0
            if let Some((l, r)) = op.if_arithmetic_or() {
                let vals = Operand::either(l, r, |x| {
                    x.if_arithmetic_and()
                        .and_then(|(l, r)| {
                            Operand::either(l, r, |x| {
                                x.if_constant().filter(|c| c & const_val == 0)
                            })
                        })
                        .map(|_| x)
                });
                if let Some((lhs, rhs)) = vals {
                    return simplify_or(
                        lhs,
                        &simplify_xor(rhs, &ctx.constant(const_val), ctx),
                        ctx,
                    );
                }
            }
        }
        ops.push(ctx.constant(const_val));
    }
    let mut tree = ops.pop().map(mark_self_simplified)
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        tree = Operand::new_simplified_rc(OperandType::Arithmetic(ArithOperand {
            ty: ArithOpType::Xor,
            left: tree,
            right: op,
        }));
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
        OperandType, Register,
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

    pub fn operand_arith64(ty: ArithOpType, lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        Operand::new_not_simplified_rc(OperandType::Arithmetic64(ArithOperand {
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

    pub fn operand_gt(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(GreaterThan, lhs, rhs)
    }

    pub fn operand_gt_signed(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(GreaterThanSigned, lhs, rhs)
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

    pub fn operand_rol(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        // rol(x, y) == (x << y) | (x >> (32 - y))
        operand_or(
            operand_lsh(lhs.clone(), rhs.clone()),
            operand_rsh(lhs, operand_sub(constval(32), rhs)),
        )
    }

    pub fn operand_not(lhs: Rc<Operand>) -> Rc<Operand> {
        operand_xor(lhs, constval(0xffff_ffff))
    }

    pub fn operand_logical_not(lhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith(Equal, constval(0), lhs)
    }

    pub fn operand_add64(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith64(Add, lhs, rhs)
    }

    pub fn operand_and64(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith64(And, lhs, rhs)
    }

    pub fn operand_or64(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith64(Or, lhs, rhs)
    }

    pub fn operand_lsh64(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith64(Lsh, lhs, rhs)
    }

    pub fn operand_rsh64(lhs: Rc<Operand>, rhs: Rc<Operand>) -> Rc<Operand> {
        operand_arith64(Rsh, lhs, rhs)
    }

    pub fn mem32_norc(val: Rc<Operand>) -> Operand {
        mem_variable(Mem32, val)
    }

    pub fn mem32(val: Rc<Operand>) -> Rc<Operand> {
        mem32_norc(val).into()
    }

    pub fn mem16(val: Rc<Operand>) -> Rc<Operand> {
        mem_variable_rc(Mem16, val)
    }

    pub fn mem8(val: Rc<Operand>) -> Rc<Operand> {
        mem_variable_rc(Mem8, val)
    }

    pub fn mem_variable(size: MemAccessSize, val: Rc<Operand>) -> Operand {
        Operand::new_not_simplified(OperandType::Memory(MemAccess {
            address: val,
            size,
        }))
    }

    pub fn mem_variable_rc(size: MemAccessSize, val: Rc<Operand>) -> Rc<Operand> {
        mem_variable(size, val).into()
    }

    pub fn constval(num: u32) -> Rc<Operand> {
        OperandContext::new().constant(num)
    }

    pub fn pair_edx_eax() -> Rc<Operand> {
        Operand::new_simplified_rc(OperandType::Pair(
            Operand::new_simplified_rc(OperandType::Register(Register(2))),
            Operand::new_simplified_rc(OperandType::Register(Register(0))),
        ))
    }

    pub fn pair(high: Rc<Operand>, low: Rc<Operand>) -> Rc<Operand> {
        Operand::new_not_simplified_rc(OperandType::Pair(high, low))
    }

}

#[cfg(test)]
mod test {
    use super::*;

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
    fn simplify_signed_mul() {
        // It's same as unsigned until sign extension is needed
        use super::operand_helpers::*;
        let op1 = operand_add(constval(5), operand_sub(operand_register(2), constval(5)));
        assert_eq!(Operand::simplified(op1), operand_register(2));
        // (5 * r2) + (5 - (5 + r2)) == (5 * r2) - r2
        let op1 = operand_signed_mul(
            operand_signed_mul(constval(5), operand_register(2)),
            operand_signed_mul(
                constval(5),
                operand_add(constval(5), operand_register(2)),
            )
        );
        let eq1 = operand_signed_mul(
            constval(25),
            operand_signed_mul(
                operand_register(2),
                operand_add(constval(5), operand_register(2)),
            ),
        );
        let op2 = operand_signed_mul(
            operand_register(1),
            operand_register(2),
        );
        let eq2 = operand_mul(
            operand_register(1),
            operand_register(2),
        );
        let op3 = operand_signed_mul(
            Operand::new_simplified_rc(OperandType::Register16(Register(1))),
            Operand::new_simplified_rc(OperandType::Register16(Register(2))),
        );
        let ne3 = operand_mul(
            Operand::new_simplified_rc(OperandType::Register16(Register(1))),
            Operand::new_simplified_rc(OperandType::Register16(Register(2))),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_ne!(Operand::simplified(op3), Operand::simplified(ne3));
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
    fn simplify_gt_signed() {
        use super::operand_helpers::*;
        let op1 = operand_gt_signed(constval(4), constval(2));
        let op2 = operand_gt_signed(constval(4), constval(!2));
        assert_eq!(Operand::simplified(op1), constval(1));
        assert_eq!(Operand::simplified(op2), constval(1));
    }

    #[test]
    fn simplify_const_shifts() {
        use super::operand_helpers::*;
        let op1 = operand_lsh(constval(0x55), constval(0x4));
        let op2 = operand_rsh(constval(0x55), constval(0x4));
        let op3 = operand_lsh(constval(0x55), constval(0x1f));
        assert_eq!(Operand::simplified(op1), constval(0x550));
        assert_eq!(Operand::simplified(op2), constval(0x5));
        assert_eq!(Operand::simplified(op3), constval(0x8000_0000));
    }

    #[test]
    fn simplify_or_parts() {
        use super::operand_helpers::*;
        let op1 = operand_or(
            operand_and(
                operand_register(4),
                constval(0xffff0000),
            ),
            operand_and(
                operand_register(4),
                constval(0x0000ffff),
            )
        );
        let op2 = operand_or(
            operand_and(
                operand_register(4),
                constval(0xffff00ff),
            ),
            operand_and(
                operand_register(4),
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
        assert_eq!(Operand::simplified(op1), operand_register(4));
        assert_eq!(Operand::simplified(op2), operand_register(4));
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
                operand_register(1),
                constval(0x10),
            ),
            constval(0xffff),
        );
        let eq = operand_rsh(
            operand_register(1),
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
                operand_register(1),
                constval(0xffff0000),
            ),
            constval(0x10),
        );
        let eq5 = operand_rsh(
            operand_register(1),
            constval(0x10),
        );
        let op6 = operand_rsh(
            operand_and(
                operand_register(1),
                constval(0xffff1234),
            ),
            constval(0x10),
        );
        let eq6 = operand_rsh(
            operand_register(1),
            constval(0x10),
        );
        let op7 = operand_lsh(
            operand_and(
                operand_register(1),
                constval(0xffff),
            ),
            constval(0x10),
        );
        let eq7 = operand_lsh(
            operand_register(1),
            constval(0x10),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_eq!(Operand::simplified(op2), Operand::simplified(eq2));
        assert_eq!(Operand::simplified(op3), Operand::simplified(eq3));
        assert_eq!(Operand::simplified(op4), Operand::simplified(eq4));
        assert_eq!(Operand::simplified(op5), Operand::simplified(eq5));
        assert_eq!(Operand::simplified(op6), Operand::simplified(eq6));
        assert_eq!(Operand::simplified(op7), Operand::simplified(eq7));
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
                operand_register(1),
                operand_register(2),
            ),
        );
        let eq1 = operand_eq(
            operand_register(1),
            operand_register(2),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_x_eq_x_add() {
        use super::operand_helpers::*;
        // The register 2 can be igned as there is no way for the addition to cause lowest
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
            operand_add(
                constval(1),
                operand_register(1),
            ),
        );
        let eq1 = operand_eq(
            constval(0xffffffff),
            operand_register(1),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
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
        let op1 = operand_gt(
            operand_sub(
                operand_register(5),
                operand_register(2),
            ),
            operand_register(5),
        );
        let eq1 = operand_gt(
            operand_register(2),
            operand_register(5),
        );
        // Checking for signed gt requires sign == overflow, unlike
        // unsigned where it's just carry == 1
        let op2 = operand_gt_signed(
            operand_sub(
                operand_register(5),
                operand_register(2),
            ),
            operand_register(5),
        );
        let ne2 = operand_gt_signed(
            operand_register(2),
            operand_register(5),
        );
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
        assert_ne!(Operand::simplified(op2), Operand::simplified(ne2));
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
            operand_lsh(
                operand_lsh(
                    operand_add(
                        constval(0x20),
                        operand_register(4),
                    ),
                    constval(0x10),
                ),
                constval(0x10),
            ),
        );
        let eq1 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_or_rsh() {
        use super::operand_helpers::*;
        let op1 = operand_and(
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
        let eq1 = constval(0);
        assert_eq!(Operand::simplified(op1), Operand::simplified(eq1));
    }

    #[test]
    fn simplify_and_or_rsh_mul() {
        use super::operand_helpers::*;
        let op1 = operand_and(
            constval(0xff000000),
            operand_or(
                constval(0xfe000000),
                operand_rsh(
                    operand_mul(
                        operand_register(2),
                        operand_register(1),
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

        // This cannot be simplified since it would change behaviour if, say
        // reg(1) == 0x7000_0000 and reg(4) == 0, if the mask is removed, it would become
        // 0x5000_0001
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

        let ne3 = operand_and(
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
        assert_ne!(Operand::simplified(op3), Operand::simplified(ne3));
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
                        operand_register(1),
                    ),
                ),
                operand_and(
                    constval(0xffff_f000),
                    operand_register(1),
                ),
            )
        );
        let eq2 = operand_register(1);
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
}
