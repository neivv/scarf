use std::fmt;

use serde::{Deserializer, Deserialize};
use serde::de::{self, DeserializeSeed, EnumAccess, MapAccess, SeqAccess, VariantAccess, Visitor};

use super::{MemAccess, MemAccessSize, ArithOperand, Operand, OperandCtx};

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
                Ok(ty)
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
                Ok(ty)
            }
        }
        deserializer.deserialize_struct("Operand", FIELDS, OperandVisitor(self.0))
    }
}

impl<'de, 'e> DeserializeSeed<'de> for DeserializeOperandType<'e> {
    type Value = Operand<'e>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        const VARIANTS: &[&str] = &[
            "Register", "Xmm", "Fpu", "Flag", "Constant", "Memory", "Arithmetic", "ArithmeticFloat",
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
            ArithmeticFloat,
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
                            7 => Ok(Variant::ArithmeticFloat),
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
                            "ArithmeticFloat" => Ok(Variant::ArithmeticFloat),
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

        struct ArithFloatVisitor<'e>(OperandCtx<'e>);

        impl<'de, 'e> Visitor<'de> for ArithFloatVisitor<'e> {
            type Value = (ArithOperand<'e>, MemAccessSize);

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("(ArithOperand, MemAccessSize)")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Self::Value, V::Error>
                where V: SeqAccess<'de>
            {
                let arith = seq.next_element_seed(DeserializeArith(self.0))?;
                Ok((
                    arith.ok_or_else(|| de::Error::invalid_length(0, &self))?,
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
            type Value = Operand<'e>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("enum OperandType")
            }

            fn visit_enum<V>(self, e: V) -> Result<Operand<'e>, V::Error>
                where V: EnumAccess<'de>
            {
                let (variant, v) = e.variant()?;
                match variant {
                    Variant::Register => {
                        let r: u8 = v.newtype_variant()?;
                        Ok(self.0.register(r))
                    }
                    Variant::Fpu => {
                        let r: u8 = v.newtype_variant()?;
                        Ok(self.0.register_fpu(r))
                    }
                    Variant::Flag => {
                        let flag = v.newtype_variant()?;
                        Ok(self.0.flag(flag))
                    }
                    Variant::Constant => {
                        let c = v.newtype_variant()?;
                        Ok(self.0.constant(c))
                    }
                    Variant::Undefined => {
                        // TODO: Not sure if undefined should deserialize to given id
                        // or if this only should guarantee that two undefs with same id
                        // in input are serialized to a new but same id.
                        //
                        // For now this returns new id every time
                        Ok(self.0.new_undef())
                    }
                    Variant::Custom => {
                        let c = v.newtype_variant()?;
                        Ok(self.0.custom(c))
                    }
                    Variant::Xmm => {
                        let (a, b) = v.tuple_variant(2, XmmVisitor)?;
                        Ok(self.0.xmm(a, b))
                    }
                    Variant::Memory => {
                        let mem = v.newtype_variant_seed(DeserializeMemory(self.0))?;
                        Ok(self.0.mem_variable_rc(mem.size, mem.address))
                    }
                    Variant::Arithmetic => {
                        let arith = v.newtype_variant_seed(DeserializeArith(self.0))?;
                        Ok(self.0.arithmetic(arith.ty, arith.left, arith.right))
                    }
                    Variant::ArithmeticFloat => {
                        let (a, b) = v.tuple_variant(2, ArithFloatVisitor(self.0))?;
                        Ok(self.0.float_arithmetic(a.ty, a.left, a.right, b))
                    }
                    Variant::SignExtend => {
                        let (a, b, c) = v.tuple_variant(3, SextVisitor(self.0))?;
                        Ok(self.0.sign_extend(a, b, c))
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
