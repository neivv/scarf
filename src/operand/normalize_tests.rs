use super::*;

trait CtxExt<'e> {
    fn normalize(&'e self, op: Operand<'e>) -> Operand<'e>;
}

impl<'e> CtxExt<'e> for OperandContext<'e> {
    fn normalize(&'e self, op: Operand<'e>) -> Operand<'e> {
        let result = self.normalize_32bit(op);
        let masked = self.and_const(op, 0xffff_ffff);
        let result_masked = self.normalize_32bit(masked);
        assert_eq!(
            result, result_masked,
            "Different results when zero masked, masked was {}", masked
        );
        assert!(
            result.is_32bit_normalized(),
            "{} became {} but normalize flag is not set", op, result,
        );
        assert!(
            result_masked.is_32bit_normalized(),
            "(masked) {} became {} but normalize flag is not set", masked, result_masked,
        );
        result
    }
}

#[test]
fn test_normalize_32bit() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.add(
        ctx.register(0),
        ctx.constant(0x1_0000_0000),
    );
    let expected = ctx.register(0);
    assert_eq!(ctx.normalize(op), expected);

    let op = ctx.or(
        ctx.register(2),
        ctx.lsh_const(
            ctx.register(1),
            32,
        ),
    );
    let expected = ctx.register(2);
    assert_eq!(ctx.normalize(op), expected);

    let op = ctx.and(
        ctx.register(0),
        ctx.constant(0xffff_ffff),
    );
    let expected = ctx.register(0);
    assert_eq!(ctx.normalize(op), expected);

    let op = ctx.and(
        ctx.custom(0),
        ctx.constant(0xffff_ffff),
    );
    let expected = ctx.custom(0);
    assert_eq!(ctx.normalize(op), expected);

    let op = ctx.constant(0x1_0000_0000);
    let expected = ctx.constant(0);
    assert_eq!(ctx.normalize(op), expected);

    let op = ctx.sub(
        ctx.register(0),
        ctx.constant(0xfb01_00fe),
    );
    let expected = ctx.add(
        ctx.register(0),
        ctx.constant(0x04fe_ff02),
    );
    assert_eq!(ctx.normalize(op), expected);

    let op = ctx.sub(
        ctx.register(0),
        ctx.constant(0x8000_0000),
    );
    let expected = ctx.add(
        ctx.register(0),
        ctx.constant(0x8000_0000),
    );
    assert_eq!(ctx.normalize(op), expected);
    assert_eq!(ctx.normalize(expected), expected);

    let op = ctx.add(
        ctx.register(0),
        ctx.constant(0x8000_0001),
    );
    let expected = ctx.sub(
        ctx.register(0),
        ctx.constant(0x7fff_ffff),
    );
    assert_eq!(ctx.normalize(op), expected);
    assert_eq!(ctx.normalize(expected), expected);

    let op = ctx.and(
        ctx.register(0),
        ctx.constant(0x7074_7676_7516_0007),
    );
    let expected = ctx.and(
        ctx.register(0),
        ctx.constant(0x7516_0007),
    );
    assert_eq!(ctx.normalize(op), expected);
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
    let expected = ctx.mul_const(ctx.register(1), 2);
    assert_eq!(ctx.normalize(op), expected);

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
    assert_eq!(ctx.normalize(op), expected);

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
    assert_eq!(ctx.normalize(op), op);

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
    assert_eq!(ctx.normalize(op), expected);
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
    assert_eq!(ctx.normalize(op), expected);

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
    let expected = ctx.lsh_const(
        ctx.add(
            ctx.register(0),
            ctx.and_const(
                ctx.register(1),
                0x4,
            ),
        ),
        0x10,
    );
    assert_eq!(ctx.normalize(op), expected);
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
    assert_eq!(ctx.normalize(op), expected);
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
    assert!(
        ctx.normalize(op).0.flags & FLAG_32BIT_NORMALIZED != 0,
        "Normalized to {} but result doesn't have normalize flag",
        ctx.normalize(op),
    );
    let masked = ctx.and_const(op, 0xffff_ffff);
    assert!(
        ctx.normalize(masked).0.flags & FLAG_32BIT_NORMALIZED != 0,
        "Normalized masked to {} but result doesn't have normalize flag",
        ctx.normalize(masked),
    );
    assert_eq!(
        ctx.normalize(op),
        ctx.normalize(masked),
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
    let op2 = ctx.lsh_const(
        ctx.and_const(
            ctx.register(0),
            0x7fff,
        ),
        0x11,
    );
    let eq = ctx.lsh_const(
        ctx.register(0),
        0x11,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op2), eq);

    let op = ctx.lsh_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0x10,
    );
    let eq = ctx.lsh_const(
        ctx.register(0),
        0x10,
    );
    assert_eq!(ctx.normalize(op), eq);
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
        ctx.xor(
            ctx.register(1),
            ctx.register(2),
        ),
    );
    assert_eq!(ctx.normalize(op), eq);
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
    assert_eq!(ctx.normalize(op), ctx.normalize(eq));
    assert_eq!(ctx.normalize(op), ctx.normalize(op_masked));
}

#[test]
fn normalize_shift() {
    let ctx = &crate::operand::OperandContext::new();
    // ((rax & ff) << 18) to (rax << 18)
    let op = ctx.lsh_const(
        ctx.and_const(
            ctx.register(0),
            0xff,
        ),
        0x18,
    );
    let op_masked = ctx.and_const(op, 0xffff_ffff);
    let eq = ctx.lsh_const(
        ctx.register(0),
        0x18,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op_masked), eq);

    // ((rax * rcx) << 18) to (((rax * rcx) & ff) << 18)
    // (Other way around since non-const multiplication)
    let op = ctx.lsh_const(
        ctx.and_const(
            ctx.mul(
                ctx.register(0),
                ctx.register(1),
            ),
            0xff,
        ),
        0x18,
    );
    let op_masked = ctx.and_const(op, 0xffff_ffff);
    let eq = ctx.lsh_const(
        ctx.and_const(
            ctx.mul(
                ctx.register(0),
                ctx.register(1),
            ),
            0xff,
        ),
        0x18,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op_masked), eq);
}

#[test]
fn normalize_mul_power_of_two() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.mul_const(
        ctx.and_const(
            ctx.register(0),
            0x7fff_ffff,
        ),
        2,
    );
    let eq = ctx.mul_const(
        ctx.register(0),
        2,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(eq), eq);
}

#[test]
fn normalize_shift_index() {
    let ctx = &crate::operand::OperandContext::new();
    // ((((rax * 12) + rdx) & ff) << 18) to (((rax * 12) + rdx) << 18)
    let op = ctx.lsh_const(
        ctx.and_const(
            ctx.add(
                ctx.mul_const(
                    ctx.register(0),
                    12,
                ),
                ctx.register(2),
            ),
            0xff,
        ),
        0x18,
    );
    let op_masked = ctx.and_const(op, 0xffff_ffff);
    let eq = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.register(0),
                12,
            ),
            ctx.register(2),
        ),
        0x18,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op_masked), eq);
}

#[test]
fn normalize_shift_index_mem() {
    let ctx = &crate::operand::OperandContext::new();
    // ((((rax * 12) + Mem8[rdx]) & ff) << 18)
    // and (((rax * 12) + Mem16[rdx]) << 18)
    // to (((rax * 12) + Mem8[rdx]) << 18)
    let op = ctx.lsh_const(
        ctx.and_const(
            ctx.add(
                ctx.mul_const(
                    ctx.register(0),
                    12,
                ),
                ctx.mem8(ctx.register(2), 0),
            ),
            0xff,
        ),
        0x18,
    );
    let op16 = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.register(0),
                12,
            ),
            ctx.mem16(ctx.register(2), 0),
        ),
        0x18,
    );
    let op_masked = ctx.and_const(op, 0xffff_ffff);
    let op16_masked = ctx.and_const(op16, 0xffff_ffff);
    let eq = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.register(0),
                12,
            ),
            ctx.mem8(ctx.register(2), 0),
        ),
        0x18,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op_masked), eq);
    assert_eq!(ctx.normalize(eq), eq);
    assert_eq!(ctx.normalize(op16), eq);
    assert_eq!(ctx.normalize(op16_masked), eq);
}

#[test]
fn normalize_shift_index_mem_2() {
    let ctx = &crate::operand::OperandContext::new();
    // ((((Mem8[rax] * 12) + rdx) & ff) << 18)
    // and (((Mem16[rax] * 12) + rdx) << 18)
    // to (((Mem8[rax] * 12) + rdx) << 18)
    let op = ctx.lsh_const(
        ctx.and_const(
            ctx.add(
                ctx.mul_const(
                    ctx.mem8(ctx.register(0), 0),
                    12,
                ),
                ctx.register(2),
            ),
            0xff,
        ),
        0x18,
    );
    let op16 = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.mem16(ctx.register(0), 0),
                12,
            ),
            ctx.register(2),
        ),
        0x18,
    );
    let op_masked = ctx.and_const(op, 0xffff_ffff);
    let op16_masked = ctx.and_const(op16, 0xffff_ffff);
    let eq = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.mem8(ctx.register(0), 0),
                12,
            ),
            ctx.register(2),
        ),
        0x18,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op_masked), eq);
    assert_eq!(ctx.normalize(eq), eq);
    assert_eq!(ctx.normalize(op16), ctx.normalize(op16_masked));
    assert_eq!(ctx.normalize(op16), eq);
    assert_eq!(ctx.normalize(op16_masked), eq);
}

#[test]
fn normalize_mul_power_of_two_index() {
    let ctx = &crate::operand::OperandContext::new();
    // ((((rax * 12) + rdx) & 3fff_ffff) << 2) to (((rax * 12) + rdx) << 2)
    let op = ctx.lsh_const(
        ctx.and_const(
            ctx.add(
                ctx.mul_const(
                    ctx.register(0),
                    12,
                ),
                ctx.register(2),
            ),
            0x3fff_ffff,
        ),
        0x2,
    );
    let op_masked = ctx.and_const(op, 0xffff_ffff);
    let eq = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.register(0),
                12,
            ),
            ctx.register(2),
        ),
        0x2,
    );
    assert_eq!(ctx.normalize(op), eq);
    assert_eq!(ctx.normalize(op_masked), eq);
}

#[test]
fn normalize_bitop() {
    let ctx = &crate::operand::OperandContext::new();
    let add = ctx.add_const(
        ctx.register(0),
        0x1234567890,
    );
    let xor = ctx.xor_const(
        add,
        0xaaffaaff,
    );
    let xor_eq = ctx.xor_const(
        ctx.add_const(
            ctx.register(0),
            0x34567890,
        ),
        0xaaffaaff,
    );
    assert_eq!(ctx.normalize(xor), xor_eq);
}

#[test]
fn normalize_xor_2() {
    let ctx = &crate::operand::OperandContext::new();
    // Xor without constant
    // Becomes ((((rax + rax)[f32] >> 1) & 7ffffffe) ^ (rax * 2))
    // If this is considered normalized (Technically could be),
    // there's issue that and masking makes it
    // (((rax + rax)[f32] >> 1) ^ (rax * 2)) & fffffffe
    // instead, and determining that to not be normalized would
    // require walking through xor with and mask and realizing that
    // it's not needed for (rax * 2)
    //
    // So just making non-constant xors not be normalized
    // if they don't fit in u32. May be changed later.
    let op1 = ctx.mul_const(
        ctx.xor(
            ctx.rsh_const(
                ctx.float_arithmetic(
                    ArithOpType::Add,
                    ctx.register(0),
                    ctx.register(0),
                    MemAccessSize::Mem32,
                ),
                2,
            ),
            ctx.register(0),
        ),
        2,
    );
    assert_eq!(ctx.normalize(op1), ctx.and_const(op1, 0xffff_ffff));
}

#[test]
fn normalize_bitop_or() {
    let ctx = &crate::operand::OperandContext::new();
    let add = ctx.add_const(
        ctx.register(0),
        0x1234567890,
    );
    let or = ctx.or_const(
        add,
        0xaaffaaff,
    );
    let or_eq = ctx.or_const(
        ctx.add_const(
            ctx.register(0),
            0x34567890,
        ),
        0xaaffaaff,
    );
    assert_eq!(ctx.normalize(or), ctx.normalize(or_eq));
}

#[test]
fn normalize_shifted_mul2() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.lsh_const(
        ctx.add(
            ctx.mul_const(
                ctx.register(0),
                2,
            ),
            ctx.mul_const(
                ctx.xmm(0, 0),
                2,
            ),
        ),
        0x11,
    );
    assert_eq!(ctx.normalize(op), op);
}

#[test]
fn normalize_with_mask() {
    let ctx = &crate::operand::OperandContext::new();
    // Mask in `Mem32[rax] & ffff_f00f` should not prevent
    // the outer mask from being removed.
    let op = ctx.and_const(
        ctx.mul_const(
            ctx.and_const(
                ctx.mem32(ctx.register(0), 0),
                0xffff_f00f,
            ),
            0x78,
        ),
        0xffff_fff8,
    );

    let eq = ctx.mul_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0),
            0x1fff_f00f,
        ),
        0x78,
    );
    assert_eq!(ctx.normalize(op), eq);
}

#[test]
fn normalize_mul_large_const() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.mul_const(
        ctx.add(
            ctx.register(0),
            ctx.register(1),
        ),
        0xa600_0000,
    );
    assert_eq!(ctx.normalize(op), op);
}

#[test]
fn normalize_or_with_shift() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.lsh_const(
        ctx.or(
            ctx.lsh_const(
                ctx.and_const(
                    ctx.mem8(ctx.register(0), 0),
                    0x5,
                ),
                0x8,
            ),
            ctx.and_const(
                ctx.mul(
                    ctx.register(0),
                    ctx.register(0),
                ),
                0x4000_0000,
            ),
        ),
        0x8,
    );
    let eq = ctx.lsh_const(
        ctx.and_const(
            ctx.mem8(ctx.register(0), 0),
            0x5,
        ),
        0x10,
    );
    assert_eq!(ctx.normalize(op), eq);
}

#[test]
fn normalize_sub1() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.sub_const(
        ctx.and_const(
            ctx.register(1),
            1,
        ),
        0x1,
    );
    assert_eq!(ctx.normalize(op), op);
}

#[test]
fn normalize_sub7f() {
    let ctx = &crate::operand::OperandContext::new();
    let op = ctx.sub_const(
        ctx.and_const(
            ctx.register(1),
            1,
        ),
        0x7f,
    );
    assert_eq!(ctx.normalize(op), op);
}
