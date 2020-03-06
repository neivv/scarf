use std::fmt;

use serde::{Deserializer, Deserialize};
use serde::de::{self, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor};

use super::{MemAccess, MemAccessSize, ArithOperand, Operand, OperandCtx, OperandType};

pub struct DeserializeOperand<'e>(pub(crate) OperandCtx<'e>);
pub struct DeserializeOperandType<'e>(OperandCtx<'e>);
pub struct DeserializeMemory<'e>(OperandCtx<'e>);
pub struct DeserializeArith<'e>(OperandCtx<'e>);

impl<'de, 'e> DeserializeSeed<'de> for DeserializeOperand<'e> {
    type Value = Operand<'e>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Operand<'e>, D::Error> {
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

        struct OperandVisitor<'e>(OperandCtx<'e>);

        impl<'de, 'e> Visitor<'de> for OperandVisitor<'e> {
            type Value = Operand<'e>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Operand")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
                where V: SeqAccess<'de>
            {
                let ty = seq.next_element_seed(DeserializeOperandType(self.0))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(self.0.intern(ty))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
                where V: MapAccess<'de>
            {
                let mut ty = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Ty => {
                            if ty.is_some() {
                                return Err(de::Error::duplicate_field("ty"));
                            }
                            ty = Some(map.next_value_seed(DeserializeOperandType(self.0))?);
                        }
                    }
                }
                let ty = ty.ok_or_else(|| de::Error::missing_field("ty"))?;
                Ok(self.0.intern(ty))
            }
        }
        deserializer.deserialize_struct("Operand", FIELDS, OperandVisitor(self.0))
    }
}

impl<'de, 'e> DeserializeSeed<'de> for DeserializeOperandType<'e> {
    type Value = OperandType<'e>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        const VARIANTS: &[&str] = &[
            "Register", "Xmm", "Fpu", "Flag", "Constant", "Memory", "Arithmetic", "ArithmeticF32",
            "Undefined", "SignExtend", "Custom",
        ];

        #[derive(Copy, Clone)]
        enum Variant {
            Register,
            Xmm,
            Fpu,
            Flag,
            Constant,
            Memory,
            Arithmetic,
            ArithmeticF32,
            Undefined,
            SignExtend,
            Custom,
        }

        impl<'de> Deserialize<'de> for Variant {
            fn deserialize<D>(deserializer: D) -> Result<Variant, D::Error>
                where D: Deserializer<'de>
            {
                struct VariantVisitor;

                impl<'de> Visitor<'de> for VariantVisitor {
                    type Value = Variant;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`ty`")
                    }

                    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
                        where E: de::Error
                    {
                        match value {
                            0 => Ok(Variant::Register),
                            1 => Ok(Variant::Xmm),
                            2 => Ok(Variant::Fpu),
                            3 => Ok(Variant::Flag),
                            4 => Ok(Variant::Constant),
                            5 => Ok(Variant::Memory),
                            6 => Ok(Variant::Arithmetic),
                            7 => Ok(Variant::ArithmeticF32),
                            8 => Ok(Variant::Undefined),
                            9 => Ok(Variant::SignExtend),
                            10 => Ok(Variant::Custom),
                            x => Err(de::Error::invalid_value(
                                de::Unexpected::Unsigned(x),
                                &"Invalid variant id",
                            )),
                        }
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                        where E: de::Error
                    {
                        match value {
                            "Register" => Ok(Variant::Register),
                            "Xmm" => Ok(Variant::Xmm),
                            "Fpu" => Ok(Variant::Fpu),
                            "Flag" => Ok(Variant::Flag),
                            "Constant" => Ok(Variant::Constant),
                            "Memory" => Ok(Variant::Memory),
                            "Arithmetic" => Ok(Variant::Arithmetic),
                            "ArithmeticF32" => Ok(Variant::ArithmeticF32),
                            "Undefined" => Ok(Variant::Undefined),
                            "SignExtend" => Ok(Variant::SignExtend),
                            "Custom" => Ok(Variant::Custom),
                            _ => Err(de::Error::unknown_variant(value, VARIANTS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(VariantVisitor)
            }
        }

        struct XmmVisitor;

        impl<'de> Visitor<'de> for XmmVisitor {
            type Value = (u8, u8);

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("(u8, u8)")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
                where V: SeqAccess<'de>
            {
                Ok((
                    seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?,
                    seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?,
                ))
            }
        }

        struct SextVisitor<'e>(OperandCtx<'e>);

        impl<'de, 'e> Visitor<'de> for SextVisitor<'e> {
            type Value = (Operand<'e>, MemAccessSize, MemAccessSize);

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("(Operand, MemAccessSize, MemAccessSize)")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
                where V: SeqAccess<'de>
            {
                let operand = seq.next_element_seed(DeserializeOperand(self.0))?;
                Ok((
                    operand.ok_or_else(|| de::Error::invalid_length(0, &self))?,
                    seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?,
                    seq.next_element()?.ok_or_else(|| de::Error::invalid_length(0, &self))?,
                ))
            }
        }

        struct OperandTypeVisitor<'e>(OperandCtx<'e>);

        impl<'de, 'e> Visitor<'de> for OperandTypeVisitor<'e> {
            type Value = OperandType<'e>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("enum OperandType")
            }

            fn visit_enum<V>(self, e: V) -> Result<OperandType<'e>, V::Error>
                where V: EnumAccess<'de>
            {
                let (variant, v) = e.variant()?;
                match variant {
                    Variant::Register => Ok(OperandType::Register(v.newtype_variant()?)),
                    Variant::Fpu => Ok(OperandType::Fpu(v.newtype_variant()?)),
                    Variant::Flag => Ok(OperandType::Flag(v.newtype_variant()?)),
                    Variant::Constant => Ok(OperandType::Constant(v.newtype_variant()?)),
                    Variant::Undefined => Ok(OperandType::Undefined(v.newtype_variant()?)),
                    Variant::Custom => Ok(OperandType::Custom(v.newtype_variant()?)),
                    Variant::Xmm => {
                        let (a, b) = v.tuple_variant(2, XmmVisitor)?;
                        Ok(OperandType::Xmm(a, b))
                    }
                    Variant::Memory => {
                        let mem = v.newtype_variant_seed(DeserializeMemory(self.0))?;
                        Ok(OperandType::Memory(mem))
                    }
                    Variant::Arithmetic | Variant::ArithmeticF32 => {
                        let arith = v.newtype_variant_seed(DeserializeArith(self.0))?;
                        if let Variant::Arithmetic = variant {
                            Ok(OperandType::Arithmetic(arith))
                        } else {
                            Ok(OperandType::ArithmeticF32(arith))
                        }
                    }
                    Variant::SignExtend => {
                        let (a, b, c) = v.tuple_variant(3, SextVisitor(self.0))?;
                        Ok(OperandType::SignExtend(a, b, c))
                    }
                }
            }
        }

        deserializer.deserialize_enum("OperandType", VARIANTS, OperandTypeVisitor(self.0))
    }
}

impl<'de, 'e> DeserializeSeed<'de> for DeserializeMemory<'e> {
    type Value = MemAccess<'e>;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        const FIELDS: &[&str] = &["address", "size"];
        enum Field {
            Address,
            Size,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("Memory field")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                        where E: de::Error
                    {
                        match value {
                            "address" => Ok(Field::Address),
                            "size" => Ok(Field::Size),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct MemoryVisitor<'e>(OperandCtx<'e>);

        impl<'de, 'e> Visitor<'de> for MemoryVisitor<'e> {
            type Value = MemAccess<'e>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct MemAccess")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
                where V: SeqAccess<'de>
            {
                let address = seq.next_element_seed(DeserializeOperand(self.0))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let size = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(MemAccess {
                    address,
                    size,
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
                where V: MapAccess<'de>
            {
                let mut address = None;
                let mut size = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Address => {
                            if address.is_some() {
                                return Err(de::Error::duplicate_field("address"));
                            }
                            address = Some(map.next_value_seed(DeserializeOperand(self.0))?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                    }
                }
                let address = address.ok_or_else(|| de::Error::missing_field("address"))?;
                let size = size.ok_or_else(|| de::Error::missing_field("size"))?;
                Ok(MemAccess {
                    address,
                    size,
                })
            }
        }

        deserializer.deserialize_struct("MemAccess", FIELDS, MemoryVisitor(self.0))
    }
}

impl<'de, 'e> DeserializeSeed<'de> for DeserializeArith<'e> {
    type Value = ArithOperand<'e>;
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        const FIELDS: &[&str] = &["ty", "left", "right"];
        enum Field {
            Ty,
            Left,
            Right,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                where D: Deserializer<'de>
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("Arith field")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                        where E: de::Error
                    {
                        match value {
                            "ty" => Ok(Field::Ty),
                            "left" => Ok(Field::Left),
                            "right" => Ok(Field::Right),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct ArithVisitor<'e>(OperandCtx<'e>);

        impl<'de, 'e> Visitor<'de> for ArithVisitor<'e> {
            type Value = ArithOperand<'e>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct ArithOperand")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
                where V: SeqAccess<'de>
            {
                let ty = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let left = seq.next_element_seed(DeserializeOperand(self.0))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let right = seq.next_element_seed(DeserializeOperand(self.0))?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                Ok(ArithOperand {
                    ty,
                    left,
                    right
                })
            }

            fn visit_map<V>(self, mut map: V) -> Result<Self::Value, V::Error>
                where V: MapAccess<'de>
            {
                let mut ty = None;
                let mut left = None;
                let mut right = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Left => {
                            if left.is_some() {
                                return Err(de::Error::duplicate_field("left"));
                            }
                            left = Some(map.next_value_seed(DeserializeOperand(self.0))?);
                        }
                        Field::Right => {
                            if right.is_some() {
                                return Err(de::Error::duplicate_field("right"));
                            }
                            right = Some(map.next_value_seed(DeserializeOperand(self.0))?);
                        }
                        Field::Ty => {
                            if ty.is_some() {
                                return Err(de::Error::duplicate_field("ty"));
                            }
                            ty = Some(map.next_value()?);
                        }
                    }
                }
                let ty = ty.ok_or_else(|| de::Error::missing_field("ty"))?;
                let left = left.ok_or_else(|| de::Error::missing_field("left"))?;
                let right = right.ok_or_else(|| de::Error::missing_field("right"))?;
                Ok(ArithOperand {
                    ty,
                    left,
                    right
                })
            }
        }

        deserializer.deserialize_struct("ArithOperand", FIELDS, ArithVisitor(self.0))
    }
}
