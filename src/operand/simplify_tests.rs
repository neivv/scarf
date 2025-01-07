use super::*;

use crate::exec_state::{OperandCtxExtX86};

fn check_simplification_consistency<'e>(ctx: OperandCtx<'e>, op: Operand<'e>) {
    use serde::de::DeserializeSeed;
    let bytes = serde_json::to_vec(&op).unwrap();
    let mut de = serde_json::Deserializer::from_slice(&bytes);
    let back: Operand<'e> = ctx.deserialize_seed().deserialize(&mut de).unwrap();
    assert_eq!(op, back);
}

#[test]
fn simplify_add_sub() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(ctx.constant(5), ctx.sub(ctx.register(2), ctx.constant(5)));
    assert_eq!(op1, ctx.register(2));
    // (5 * r2) + (5 - (5 + r2)) == (5 * r2) - r2
    let op1 = ctx.add(
        ctx.mul(ctx.constant(5), ctx.register(2)),
        ctx.sub(
            ctx.constant(5),
            ctx.add(ctx.constant(5), ctx.register(2)),
        ),
    );
    let op2 = ctx.sub(
        ctx.mul(ctx.constant(5), ctx.register(2)),
        ctx.register(2),
    );
    assert_eq!(op1, op2);
}

#[test]
fn simplify_add_sub_repeat_operands() {
    let ctx = &OperandContext::new();
    // x - (x - 4) == 4
    let op1 = ctx.sub(
        ctx.register(2),
        ctx.sub(
            ctx.register(2),
            ctx.constant(4),
        )
    );
    let op2 = ctx.constant(4);
    assert_eq!(op1, op2);
}

#[test]
fn simplify_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(ctx.constant(5), ctx.sub(ctx.register(2), ctx.constant(5)));
    assert_eq!(op1, ctx.register(2));
    // (5 * r2) + (5 - (5 + r2)) == (5 * r2) - r2
    let op1 = ctx.mul(
        ctx.mul(ctx.constant(5), ctx.register(2)),
        ctx.mul(
            ctx.constant(5),
            ctx.add(ctx.constant(5), ctx.register(2)),
        )
    );
    let op2 = ctx.mul(
        ctx.constant(25),
        ctx.mul(
            ctx.register(2),
            ctx.add(ctx.constant(5), ctx.register(2)),
        ),
    );
    assert_eq!(op1, op2);
}

#[test]
fn simplify_and_or_chain() {
    let ctx = &OperandContext::new();
    // ((((w | (ctx.mem8[z] << 8)) & 0xffffff00) | mem8[y]) & 0xffff00ff) ==
    //     ((w & 0xffffff00) | mem8[y]) & 0xffff00ff
    let op1 = ctx.and(
        ctx.or(
            ctx.and(
                ctx.or(
                    ctx.register(4),
                    ctx.lsh(
                        ctx.mem8(ctx.register(3), 0),
                        ctx.constant(8),
                    ),
                ),
                ctx.constant(0xffffff00),
            ),
            ctx.mem8(ctx.register(2), 0),
        ),
        ctx.constant(0xffff00ff),
    );
    let op2 = ctx.and(
        ctx.or(
            ctx.and(
                ctx.register(4),
                ctx.constant(0xffffff00),
            ),
            ctx.mem8(ctx.register(2), 0),
        ),
        ctx.constant(0xffff00ff),
    );
    assert_eq!(op1, op2);
}

#[test]
fn simplify_and() {
    let ctx = &OperandContext::new();
    // x & x == x
    let op1 = ctx.and(
        ctx.register(4),
        ctx.register(4),
    );
    assert_eq!(op1, ctx.register(4));
}

#[test]
fn simplify_and_constants() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffff),
        ctx.and(
            ctx.constant(0xf3),
            ctx.register(4),
        ),
    );
    let op2 = ctx.and(
        ctx.constant(0xf3),
        ctx.register(4),
    );
    assert_eq!(op1, op2);
}

#[test]
fn simplify_or() {
    let ctx = &OperandContext::new();
    // mem8[x] | 0xff == 0xff
    let op1 = ctx.or(
        ctx.mem8(ctx.register(2), 0),
        ctx.constant(0xff),
    );
    assert_eq!(op1, ctx.constant(0xff));
    // (y == z) | 1 == 1
    let op1 = ctx.or(
        ctx.eq(
            ctx.register(3),
            ctx.register(4),
        ),
        ctx.constant(1),
    );
    assert_eq!(op1, ctx.constant(1));
}

#[test]
fn simplify_xor() {
    let ctx = &OperandContext::new();
    // x ^ x ^ x == x
    let op1 = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.register(1),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, ctx.register(1));
    let op1 = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.register(2),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, ctx.register(2));
}

#[test]
fn simplify_eq() {
    let ctx = &OperandContext::new();
    // Simplify (x == y) == 1 to x == y
    let op1 = ctx.eq(ctx.constant(5), ctx.register(2));
    let eq1 = ctx.eq(ctx.constant(1), ctx.eq(ctx.constant(5), ctx.register(2)));
    // Simplify (x == y) == 0 == 0 to x == y
    let op2 = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.constant(0),
            ctx.eq(ctx.constant(5), ctx.register(2)),
        ),
    );
    let eq2 = ctx.eq(ctx.constant(5), ctx.register(2));
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_eq2() {
    let ctx = &OperandContext::new();
    // Simplify (x == y) == 0 == 0 == 0 to (x == y) == 0
    let op1 = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.constant(0),
            ctx.eq(
                ctx.constant(0),
                ctx.eq(ctx.constant(5), ctx.register(2)),
            ),
        ),
    );
    let eq1 = ctx.eq(ctx.eq(ctx.constant(5), ctx.register(2)), ctx.constant(0));
    let ne1 = ctx.eq(ctx.constant(5), ctx.register(2));
    assert_eq!(op1, eq1);
    assert_ne!(op1, ne1);
}

#[test]
fn simplify_gt() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(ctx.constant(4), ctx.constant(2));
    let op2 = ctx.gt(ctx.constant(4), ctx.constant(!2));
    assert_eq!(op1, ctx.constant(1));
    assert_eq!(op2, ctx.constant(0));
}

#[test]
fn simplify_const_shifts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(ctx.constant(0x55), ctx.constant(0x4));
    let op2 = ctx.rsh(ctx.constant(0x55), ctx.constant(0x4));
    let op3 = ctx.and(
        ctx.lsh(ctx.constant(0x55), ctx.constant(0x1f)),
        ctx.constant(0xffff_ffff),
    );
    let op4 = ctx.lsh(ctx.constant(0x55), ctx.constant(0x1f));
    assert_eq!(op1, ctx.constant(0x550));
    assert_eq!(op2, ctx.constant(0x5));
    assert_eq!(op3, ctx.constant(0x8000_0000));
    assert_eq!(op4, ctx.constant(0x2a_8000_0000));
}

#[test]
fn simplify_or_parts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.mem32(ctx.register(4), 0),
            ctx.constant(0xffff0000),
        ),
        ctx.and(
            ctx.mem32(ctx.register(4), 0),
            ctx.constant(0x0000ffff),
        )
    );
    let op2 = ctx.or(
        ctx.and(
            ctx.mem32(ctx.register(4), 0),
            ctx.constant(0xffff00ff),
        ),
        ctx.and(
            ctx.mem32(ctx.register(4), 0),
            ctx.constant(0x0000ffff),
        )
    );
    let op3 = ctx.or(
        ctx.and(
            ctx.register(4),
            ctx.constant(0x00ff00ff),
        ),
        ctx.and(
            ctx.register(4),
            ctx.constant(0x0000ffff),
        )
    );
    let eq3 = ctx.and(
        ctx.register(4),
        ctx.constant(0x00ffffff),
    );
    assert_eq!(op1, ctx.mem32(ctx.register(4), 0));
    assert_eq!(op2, ctx.mem32(ctx.register(4), 0));
    assert_eq!(op3, eq3);
}

#[test]
fn simplify_and_parts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.register(4),
            ctx.constant(0xffff0000),
        ),
        ctx.or(
            ctx.register(4),
            ctx.constant(0x0000ffff),
        )
    );
    let op2 = ctx.and(
        ctx.or(
            ctx.register(4),
            ctx.constant(0x00ff00ff),
        ),
        ctx.or(
            ctx.register(4),
            ctx.constant(0x0000ffff),
        )
    );
    let eq2 = ctx.or(
        ctx.register(4),
        ctx.constant(0x000000ff),
    );
    assert_eq!(op1, ctx.register(4));
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_lsh_or_rsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.or(
            ctx.and(
                ctx.register(4),
                ctx.constant(0xffff),
            ),
            ctx.lsh(
                ctx.and(
                    ctx.register(5),
                    ctx.constant(0xffff),
                ),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.and(
        ctx.register(5),
        ctx.constant(0xffff),
    );
    let op2 = ctx.rsh(
        ctx.or(
            ctx.and(
                ctx.register(4),
                ctx.constant(0xffff),
            ),
            ctx.or(
                ctx.lsh(
                    ctx.and(
                        ctx.register(5),
                        ctx.constant(0xffff),
                    ),
                    ctx.constant(0x10),
                ),
                ctx.and(
                    ctx.register(1),
                    ctx.constant(0xffff),
                ),
            ),
        ),
        ctx.constant(0x10),
    );
    let eq2 = ctx.and(
        ctx.register(5),
        ctx.constant(0xffff),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_and_or_bug() {
    fn rol<'e>(ctx: OperandCtx<'e>, lhs: Operand<'e>, rhs: Operand<'e>) -> Operand<'e> {
        // rol(x, y) == (x << y) | (x >> (32 - y))
        ctx.or(
            ctx.lsh(lhs, rhs),
            ctx.rsh(lhs, ctx.sub(ctx.constant(32), rhs)),
        )
    }

    let ctx = &OperandContext::new();
    let op = ctx.and(
        ctx.or(
            ctx.lsh(
                ctx.xor(
                    ctx.rsh(
                        ctx.register(1),
                        ctx.constant(0x10),
                    ),
                    ctx.add(
                        ctx.and(
                            ctx.constant(0xffff),
                            ctx.register(1),
                        ),
                        rol(
                            ctx,
                            ctx.and(
                                ctx.constant(0xffff),
                                ctx.register(2),
                            ),
                            ctx.constant(1),
                        ),
                    ),
                ),
                ctx.constant(0x10),
            ),
            ctx.and(
                ctx.constant(0xffff),
                ctx.register(1),
            ),
        ),
        ctx.constant(0xffff),
    );
    let eq = ctx.and(
        ctx.register(1),
        ctx.constant(0xffff),
    );
    assert_eq!(op, eq);
}

#[test]
fn simplify_pointless_and_masks() {
    let ctx = &OperandContext::new();
    let op = ctx.and(
        ctx.rsh(
            ctx.mem32(ctx.register(1), 0),
            ctx.constant(0x10),
        ),
        ctx.constant(0xffff),
    );
    let eq = ctx.rsh(
        ctx.mem32(ctx.register(1), 0),
        ctx.constant(0x10),
    );
    assert_eq!(op, eq);
}

#[test]
fn simplify_add_x_x() {
    let ctx = &OperandContext::new();
    let op = ctx.add(
        ctx.register(1),
        ctx.register(1),
    );
    let eq = ctx.mul(
        ctx.register(1),
        ctx.constant(2),
    );
    let op2 = ctx.add(
        ctx.sub(
            ctx.add(
                ctx.register(1),
                ctx.register(1),
            ),
            ctx.register(1),
        ),
        ctx.add(
            ctx.register(1),
            ctx.register(1),
        ),
    );
    let eq2 = ctx.mul(
        ctx.register(1),
        ctx.constant(3),
    );
    assert_eq!(op, eq);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_add_x_x_64() {
    let ctx = &OperandContext::new();
    let op = ctx.add(
        ctx.register(1),
        ctx.register(1),
    );
    let eq = ctx.mul(
        ctx.register(1),
        ctx.constant(2),
    );
    let neq = ctx.register(1);
    let op2 = ctx.add(
        ctx.sub(
            ctx.add(
                ctx.register(1),
                ctx.register(1),
            ),
            ctx.register(1),
        ),
        ctx.add(
            ctx.register(1),
            ctx.register(1),
        ),
    );
    let eq2 = ctx.mul(
        ctx.register(1),
        ctx.constant(3),
    );
    assert_eq!(op, eq);
    assert_eq!(op2, eq2);
    assert_ne!(op, neq);
}

#[test]
fn simplify_and_xor_const() {
    let ctx = &OperandContext::new();
    let op = ctx.and(
        ctx.constant(0xffff),
        ctx.xor(
            ctx.constant(0x12345678),
            ctx.register(1),
        ),
    );
    let eq = ctx.and(
        ctx.constant(0xffff),
        ctx.xor(
            ctx.constant(0x5678),
            ctx.register(1),
        ),
    );
    let op2 = ctx.and(
        ctx.constant(0xffff),
        ctx.or(
            ctx.constant(0x12345678),
            ctx.register(1),
        ),
    );
    let eq2 = ctx.and(
        ctx.constant(0xffff),
        ctx.or(
            ctx.constant(0x5678),
            ctx.register(1),
        ),
    );
    assert_eq!(op, eq);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_mem_access_and() {
    let ctx = &OperandContext::new();
    let op = ctx.and(
        ctx.constant(0xffff),
        ctx.mem32(ctx.const_0(), 0x123456),
    );
    let eq = ctx.mem16(ctx.const_0(), 0x123456);
    let op2 = ctx.and(
        ctx.constant(0xfff),
        ctx.mem32(ctx.const_0(), 0x123456),
    );
    let eq2 = ctx.and(
        ctx.constant(0xfff),
        ctx.mem16(ctx.const_0(), 0x123456),
    );
    assert_ne!(op2, eq);
    assert_eq!(op, eq);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_and_or_bug2() {
    let ctx = &OperandContext::new();
    let op = ctx.and(
        ctx.or(
            ctx.constant(1),
            ctx.and(
                ctx.constant(0xffffff00),
                ctx.register(1),
            ),
        ),
        ctx.constant(0xff),
    );
    let ne = ctx.and(
        ctx.or(
            ctx.constant(1),
            ctx.register(1),
        ),
        ctx.constant(0xff),
    );
    let eq = ctx.constant(1);
    assert_ne!(op, ne);
    assert_eq!(op, eq);
}

#[test]
fn simplify_adjacent_ands_advanced() {
    let ctx = &OperandContext::new();
    let op = ctx.and(
        ctx.constant(0xffff),
        ctx.sub(
            ctx.register(0),
            ctx.or(
                ctx.and(
                    ctx.constant(0xff00),
                    ctx.xor(
                        ctx.xor(
                            ctx.constant(0x4200),
                            ctx.register(1),
                        ),
                        ctx.mem16(ctx.register(2), 0),
                    ),
                ),
                ctx.and(
                    ctx.constant(0xff),
                    ctx.xor(
                        ctx.xor(
                            ctx.constant(0xa6),
                            ctx.register(1),
                        ),
                        ctx.mem8(ctx.register(2), 0),
                    ),
                ),
            ),
        ),
    );
    let eq = ctx.and(
        ctx.constant(0xffff),
        ctx.sub(
            ctx.register(0),
            ctx.and(
                ctx.constant(0xffff),
                ctx.xor(
                    ctx.xor(
                        ctx.constant(0x42a6),
                        ctx.register(1),
                    ),
                    ctx.mem16(ctx.register(2), 0),
                ),
            ),
        ),
    );
    assert_eq!(op, eq);
}

#[test]
fn simplify_shifts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(
        ctx.rsh(
            ctx.and(
                ctx.register(1),
                ctx.constant(0xff00),
            ),
            ctx.constant(8),
        ),
        ctx.constant(8),
    );
    let eq1 = ctx.and(
        ctx.register(1),
        ctx.constant(0xff00),
    );
    let op2 = ctx.rsh(
        ctx.lsh(
            ctx.and(
                ctx.register(1),
                ctx.constant(0xff),
            ),
            ctx.constant(8),
        ),
        ctx.constant(8),
    );
    let eq2 = ctx.and(
        ctx.register(1),
        ctx.constant(0xff),
    );
    let op3 = ctx.rsh(
        ctx.lsh(
            ctx.and(
                ctx.register(1),
                ctx.constant(0xff),
            ),
            ctx.constant(8),
        ),
        ctx.constant(7),
    );
    let eq3 = ctx.lsh(
        ctx.and(
            ctx.register(1),
            ctx.constant(0xff),
        ),
        ctx.constant(1),
    );
    let op4 = ctx.rsh(
        ctx.lsh(
            ctx.and(
                ctx.register(1),
                ctx.constant(0xff),
            ),
            ctx.constant(7),
        ),
        ctx.constant(8),
    );
    let eq4 = ctx.rsh(
        ctx.and(
            ctx.register(1),
            ctx.constant(0xff),
        ),
        ctx.constant(1),
    );
    let op5 = ctx.rsh(
        ctx.and(
            ctx.mem32(ctx.register(1), 0),
            ctx.constant(0xffff0000),
        ),
        ctx.constant(0x10),
    );
    let eq5 = ctx.rsh(
        ctx.mem32(ctx.register(1), 0),
        ctx.constant(0x10),
    );
    let op6 = ctx.rsh(
        ctx.and(
            ctx.mem32(ctx.register(1), 0),
            ctx.constant(0xffff1234),
        ),
        ctx.constant(0x10),
    );
    let eq6 = ctx.rsh(
        ctx.mem32(ctx.register(1), 0),
        ctx.constant(0x10),
    );
    let op7 = ctx.and(
        ctx.lsh(
            ctx.and(
                ctx.mem32(ctx.const_0(), 1),
                ctx.constant(0xffff),
            ),
            ctx.constant(0x10),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq7 = ctx.and(
        ctx.lsh(
            ctx.mem32(ctx.const_0(), 1),
            ctx.constant(0x10),
        ),
        ctx.constant(0xffff_ffff),
    );
    let op8 = ctx.rsh(
        ctx.and(
            ctx.register(1),
            ctx.constant(0xffff_ffff_ffff_0000),
        ),
        ctx.constant(0x10),
    );
    let eq8 = ctx.rsh(
        ctx.register(1),
        ctx.constant(0x10),
    );
    let op9 = ctx.rsh(
        ctx.and(
            ctx.register(1),
            ctx.constant(0xffff0000),
        ),
        ctx.constant(0x10),
    );
    let ne9 = ctx.rsh(
        ctx.register(1),
        ctx.constant(0x10),
    );
    let op10 = ctx.rsh(
        ctx.and(
            ctx.register(1),
            ctx.constant(0xffff_ffff_ffff_1234),
        ),
        ctx.constant(0x10),
    );
    let eq10 = ctx.rsh(
        ctx.register(1),
        ctx.constant(0x10),
    );
    let op11 = ctx.rsh(
        ctx.and(
            ctx.register(1),
            ctx.constant(0xffff_1234),
        ),
        ctx.constant(0x10),
    );
    let ne11 = ctx.rsh(
        ctx.register(1),
        ctx.constant(0x10),
    );
    let op12 = ctx.lsh(
        ctx.and(
            ctx.mem32(ctx.const_0(), 1),
            ctx.constant(0xffff),
        ),
        ctx.constant(0x10),
    );
    let ne12 = ctx.lsh(
        ctx.mem32(ctx.const_0(), 1),
        ctx.constant(0x10),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_eq!(op4, eq4);
    assert_eq!(op5, eq5);
    assert_eq!(op6, eq6);
    assert_eq!(op7, eq7);
    assert_eq!(op8, eq8);
    assert_ne!(op9, ne9);
    assert_eq!(op10, eq10);
    assert_ne!(op11, ne11);
    assert_ne!(op12, ne12);
}

#[test]
fn simplify_mem_zero_bits() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.or(
            ctx.mem16(ctx.register(0), 0),
            ctx.lsh(
                ctx.mem16(ctx.register(1), 0),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.mem16(ctx.register(1), 0);
    let op2 = ctx.and(
        ctx.or(
            ctx.mem16(ctx.register(0), 0),
            ctx.lsh(
                ctx.mem16(ctx.register(1), 0),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xffff0000),
    );
    let eq2 = ctx.lsh(
        ctx.mem16(ctx.register(1), 0),
        ctx.constant(0x10),
    );
    let op3 = ctx.and(
        ctx.or(
            ctx.mem16(ctx.register(0), 0),
            ctx.lsh(
                ctx.mem16(ctx.register(1), 0),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xffff),
    );
    let eq3 = ctx.mem16(ctx.register(0), 0);
    let op4 = ctx.or(
        ctx.or(
            ctx.mem16(ctx.register(0), 0),
            ctx.lsh(
                ctx.mem16(ctx.register(1), 0),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xffff0000),
    );
    let eq4 = ctx.or(
        ctx.mem16(ctx.register(0), 0),
        ctx.constant(0xffff0000),
    );

    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_eq!(op4, eq4);
}

#[test]
fn simplify_mem_16_hi_or_mem8() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.mem8(ctx.register(1), 0),
        ctx.and(
            ctx.mem16(ctx.register(1), 0),
            ctx.constant(0xff00),
        ),
    );
    let eq1 = ctx.mem16(ctx.register(1), 0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_xor_and() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0x7ffffff0),
        ctx.xor(
            ctx.and(
                ctx.constant(0x7ffffff0),
                ctx.register(0),
            ),
            ctx.register(1),
        ),
    );
    let eq1 = ctx.and(
        ctx.constant(0x7ffffff0),
        ctx.xor(
            ctx.register(0),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_large_and_xor_chain() {
    // Check that this executes in reasonable time
    use super::OperandContext;

    let ctx = OperandContext::new();
    let mut chain = ctx.new_undef();
    for _ in 0..20 {
        chain = ctx.xor(
            ctx.and(
                ctx.constant(0x7fffffff),
                chain,
            ),
            ctx.and(
                ctx.constant(0x7fffffff),
                ctx.xor(
                    ctx.and(
                        ctx.constant(0x7fffffff),
                        chain,
                    ),
                    ctx.new_undef(),
                ),
            ),
        );
        let _ = ctx.rsh(
            chain,
            ctx.constant(1),
        );
    }
}

#[test]
fn simplify_merge_adds_as_mul() {
    let ctx = &OperandContext::new();
    let op = ctx.add(
        ctx.mul(
            ctx.register(1),
            ctx.constant(2),
        ),
        ctx.register(1),
    );
    let eq = ctx.mul(
        ctx.register(1),
        ctx.constant(3),
    );
    let op2 = ctx.add(
        ctx.mul(
            ctx.register(1),
            ctx.constant(2),
        ),
        ctx.mul(
            ctx.register(1),
            ctx.constant(8),
        ),
    );
    let eq2 = ctx.mul(
        ctx.register(1),
        ctx.constant(10),
    );
    assert_eq!(op, eq);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_merge_and_xor() {
    let ctx = &OperandContext::new();
    let op = ctx.or(
        ctx.and(
            ctx.xor(
                ctx.xor(
                    ctx.mem32(ctx.const_0(), 1234),
                    ctx.constant(0x1123),
                ),
                ctx.mem32(ctx.const_0(), 3333),
            ),
            ctx.constant(0xff00),
        ),
        ctx.and(
            ctx.xor(
                ctx.xor(
                    ctx.mem32(ctx.const_0(), 1234),
                    ctx.constant(0x666666),
                ),
                ctx.mem32(ctx.const_0(), 3333),
            ),
            ctx.constant(0xff),
        ),
    );
    let eq = ctx.and(
        ctx.constant(0xffff),
        ctx.xor(
            ctx.xor(
                ctx.mem16(ctx.const_0(), 1234),
                ctx.mem16(ctx.const_0(), 3333),
            ),
            ctx.constant(0x1166),
        ),
    );
    assert_eq!(op, eq);
}

#[test]
fn simplify_and_or_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.constant(0x38),
            ctx.register(1),
        ),
        ctx.constant(0x28),
    );
    let eq1 = ctx.constant(0x28);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_sub_eq_zero() {
    let ctx = &OperandContext::new();
    // 0 == (x - y) is same as x == y
    let op1 = ctx.eq(
        ctx.constant(0),
        ctx.sub(
            ctx.mem32(ctx.register(1), 0),
            ctx.mem32(ctx.register(2), 0),
        ),
    );
    let eq1 = ctx.eq(
        ctx.mem32(ctx.register(1), 0),
        ctx.mem32(ctx.register(2), 0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_x_eq_x_add() {
    let ctx = &OperandContext::new();
    // The register 2 can be ignored as there is no way for the addition to cause lowest
    // byte to be equal to what it was. If the constant addition were higher than 0xff,
    // then it couldn't be simplified (effectively the high unknown is able to cause unknown
    // amount of reduction in the constant's effect, but looping the lowest byte around
    // requires a multiple of 0x100 to be added)
    let op1 = ctx.eq(
        ctx.and(
            ctx.or(
                ctx.and(
                    ctx.register(2),
                    ctx.constant(0xffffff00),
                ),
                ctx.and(
                    ctx.register(1),
                    ctx.constant(0xff),
                ),
            ),
            ctx.constant(0xff),
        ),
        ctx.and(
            ctx.add(
                ctx.or(
                    ctx.and(
                        ctx.register(2),
                        ctx.constant(0xffffff00),
                    ),
                    ctx.and(
                        ctx.register(1),
                        ctx.constant(0xff),
                    ),
                ),
                ctx.constant(1),
            ),
            ctx.constant(0xff),
        ),
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.eq(
        ctx.mem8(ctx.const_0(), 555),
        ctx.and(
            ctx.add(
                ctx.or(
                    ctx.and(
                        ctx.register(2),
                        ctx.constant(0xffffff00),
                    ),
                    ctx.mem8(ctx.const_0(), 555),
                ),
                ctx.constant(1),
            ),
            ctx.constant(0xff),
        ),
    );
    let eq2 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_overflowing_shifts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(
        ctx.rsh(
            ctx.register(1),
            ctx.constant(0x55),
        ),
        ctx.constant(0x22),
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.rsh(
        ctx.lsh(
            ctx.register(1),
            ctx.constant(0x55),
        ),
        ctx.constant(0x22),
    );
    let eq2 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_and_not_mem32() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.xor_const(
            ctx.mem32(ctx.const_0(), 0x123),
            0xffff_ffff_ffff_ffff,
        ),
        0xffff,
    );
    let eq1 = ctx.and_const(
        ctx.xor_const(
            ctx.mem16(ctx.const_0(), 0x123),
            0xffff_ffff_ffff_ffff,
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.constant(0),
        ctx.and(
            ctx.add(
                ctx.constant(1),
                ctx.register(1),
            ),
            ctx.constant(0xffffffff),
        ),
    );
    let eq1 = ctx.eq(
        ctx.constant(0xffffffff),
        ctx.and(
            ctx.register(1),
            ctx.constant(0xffffffff),
        ),
    );
    let op2 = ctx.eq(
        ctx.constant(0),
        ctx.add(
            ctx.constant(1),
            ctx.register(1),
        ),
    );
    let eq2 = ctx.eq(
        ctx.constant(0xffff_ffff_ffff_ffff),
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_add_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul(
        ctx.constant(4),
        ctx.add(
            ctx.constant(5),
            ctx.mul(
                ctx.register(0),
                ctx.constant(3),
            ),
        ),
    );
    let eq1 = ctx.add(
        ctx.constant(20),
        ctx.mul(
            ctx.register(0),
            ctx.constant(12),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_bool_oper() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.constant(0),
        ),
    );
    let eq1 = ctx.gt(
        ctx.register(0),
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.sub(
            ctx.register(5),
            ctx.register(2),
        ),
        ctx.register(5),
    );
    let eq1 = ctx.gt(
        ctx.register(2),
        ctx.register(5),
    );
    // Checking for signed gt requires sign == overflow, unlike
    // unsigned where it's just carry == 1
    let op2 = ctx.gt(
        ctx.and(
            ctx.add(
                ctx.sub(
                    ctx.register(5),
                    ctx.register(2),
                ),
                ctx.constant(0x8000_0000),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.and(
            ctx.add(
                ctx.register(5),
                ctx.constant(0x8000_0000),
            ),
            ctx.constant(0xffff_ffff),
        ),
    );
    let ne2 = ctx.gt(
        ctx.and(
            ctx.add(
                ctx.register(2),
                ctx.constant(0x8000_0000),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.and(
            ctx.add(
                ctx.register(5),
                ctx.constant(0x8000_0000),
            ),
            ctx.constant(0xffff_ffff),
        ),
    );
    let op3 = ctx.gt(
        ctx.sub(
            ctx.register(5),
            ctx.register(2),
        ),
        ctx.register(5),
    );
    let eq3 = ctx.gt(
        ctx.register(2),
        ctx.register(5),
    );
    assert_eq!(op1, eq1);
    assert_ne!(op2, ne2);
    assert_eq!(op3, eq3);
}

#[test]
fn simplify_mem32_rsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.mem32(ctx.const_0(), 0x123),
        ctx.constant(0x10),
    );
    let eq1 = ctx.mem16(ctx.const_0(), 0x125);
    let op2 = ctx.rsh(
        ctx.mem32(ctx.const_0(), 0x123),
        ctx.constant(0x11),
    );
    let eq2 = ctx.rsh(
        ctx.mem16(ctx.const_0(), 0x125),
        ctx.constant(0x1),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_mem_or() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.rsh(
            ctx.mem64(
                ctx.register(0),
                0x120,
            ),
            ctx.constant(0x8),
        ),
        ctx.lsh(
            ctx.mem64(
                ctx.register(0),
                0x128,
            ),
            ctx.constant(0x38),
        ),
    );
    let eq1 = ctx.mem64(
        ctx.register(0),
        0x121,
    );
    let op2 = ctx.or(
        ctx.mem16(
            ctx.register(0),
            0x122,
        ),
        ctx.lsh(
            ctx.mem16(
                ctx.register(0),
                0x124,
            ),
            ctx.constant(0x10),
        ),
    );
    let eq2 = ctx.mem32(
        ctx.register(0),
        0x122,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_rsh_and() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffff),
        ctx.rsh(
            ctx.or(
                ctx.constant(0x123400),
                ctx.and(
                    ctx.register(1),
                    ctx.constant(0xff000000),
                ),
            ),
            ctx.constant(8),
        ),
    );
    let eq1 = ctx.constant(0x1234);
    let op2 = ctx.and(
        ctx.constant(0xffff0000),
        ctx.lsh(
            ctx.or(
                ctx.constant(0x123400),
                ctx.and(
                    ctx.register(1),
                    ctx.constant(0xff),
                ),
            ),
            ctx.constant(8),
        ),
    );
    let eq2 = ctx.constant(0x12340000);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_mem32_or() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.lsh(
                ctx.register(2),
                ctx.constant(0x18),
            ),
            ctx.rsh(
                ctx.or(
                    ctx.constant(0x123400),
                    ctx.and(
                        ctx.mem32(ctx.register(1), 0),
                        ctx.constant(0xff000000),
                    ),
                ),
                ctx.constant(8),
            ),
        ),
        ctx.constant(0xffff),
    );
    let eq1 = ctx.constant(0x1234);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_mem_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.rsh(
            ctx.constant(0),
            ctx.constant(0x10),
        ),
        ctx.and(
            ctx.lsh(
                ctx.lsh(
                    ctx.and(
                        ctx.add(
                            ctx.constant(0x20),
                            ctx.register(4),
                        ),
                        ctx.constant(0xffff_ffff),
                    ),
                    ctx.constant(0x10),
                ),
                ctx.constant(0x10),
            ),
            ctx.constant(0xffff_ffff),
        ),
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.or(
        ctx.rsh(
            ctx.constant(0),
            ctx.constant(0x10),
        ),
        ctx.lsh(
            ctx.lsh(
                ctx.add(
                    ctx.constant(0x20),
                    ctx.register(4),
                ),
                ctx.constant(0x20),
            ),
            ctx.constant(0x20),
        ),
    );
    let eq2 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_and_or_rsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffffff00),
        ctx.or(
            ctx.rsh(
                ctx.mem32(ctx.register(1), 0),
                ctx.constant(0x18),
            ),
            ctx.rsh(
                ctx.mem32(ctx.register(4), 0),
                ctx.constant(0x18),
            ),
        ),
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.and(
        ctx.constant(0xffffff00),
        ctx.or(
            ctx.rsh(
                ctx.register(1),
                ctx.constant(0x18),
            ),
            ctx.rsh(
                ctx.register(4),
                ctx.constant(0x18),
            ),
        ),
    );
    let ne2 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_ne!(op2, ne2);
}

#[test]
fn simplify_and_or_rsh_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xff000000),
        ctx.or(
            ctx.constant(0xfe000000),
            ctx.rsh(
                ctx.and(
                    ctx.mul(
                        ctx.register(2),
                        ctx.register(1),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.constant(0x18),
            ),
        ),
    );
    let eq1 = ctx.constant(0xfe000000);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mem_misalign2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.rsh(
            ctx.mem32(ctx.register(1), 0),
            ctx.constant(0x8),
        ),
        ctx.lsh(
            ctx.mem8(
                ctx.register(1),
                4,
            ),
            ctx.constant(0x18),
        ),
    );
    let eq1 = ctx.mem32(
        ctx.register(1),
        1,
    );
    let op2 = ctx.or(
        ctx.rsh(
            ctx.mem32(
                ctx.sub(
                    ctx.register(1),
                    ctx.constant(0x4),
                ),
                0,
            ),
            ctx.constant(0x8),
        ),
        ctx.lsh(
            ctx.mem8(ctx.register(1), 0),
            ctx.constant(0x18),
        ),
    );
    let eq2 = ctx.mem32(
        ctx.sub(
            ctx.register(1),
            ctx.constant(3),
        ),
        0,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_and_shift_overflow_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.rsh(
                ctx.or(
                    ctx.rsh(
                        ctx.mem8(ctx.register(1), 0),
                        ctx.constant(7),
                    ),
                    ctx.and(
                        ctx.constant(0xff000000),
                        ctx.lsh(
                            ctx.mem8(ctx.register(2), 0),
                            ctx.constant(0x11),
                        ),
                    ),
                ),
                ctx.constant(0x10),
            ),
            ctx.lsh(
                ctx.mem32(ctx.register(4), 0),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xff),
    );
    let eq1 = ctx.and(
        ctx.rsh(
            ctx.or(
                ctx.rsh(
                    ctx.mem8(ctx.register(1), 0),
                    ctx.constant(7),
                ),
                ctx.and(
                    ctx.constant(0xff000000),
                    ctx.lsh(
                        ctx.mem8(ctx.register(2), 0),
                        ctx.constant(0x11),
                    ),
                ),
            ),
            ctx.constant(0x10),
        ),
        ctx.constant(0xff),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mul_add() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul(
        ctx.constant(0xc),
        ctx.add(
            ctx.constant(0xc),
            ctx.register(1),
        ),
    );
    let eq1 = ctx.add(
        ctx.constant(0x90),
        ctx.mul(
            ctx.constant(0xc),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_masks() {
    let ctx = &OperandContext::new();
    // One and can be removed since zext(u8) + zext(u8) won't overflow u32
    let op1 = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.and(
                ctx.constant(0xff),
                ctx.register(1),
            ),
            ctx.and(
                ctx.constant(0xff),
                ctx.add(
                    ctx.and(
                        ctx.constant(0xff),
                        ctx.register(1),
                    ),
                    ctx.and(
                        ctx.constant(0xff),
                        ctx.add(
                            ctx.register(4),
                            ctx.and(
                                ctx.constant(0xff),
                                ctx.register(1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );

    let eq1 = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.and(
                ctx.constant(0xff),
                ctx.register(1),
            ),
            ctx.add(
                ctx.and(
                    ctx.constant(0xff),
                    ctx.register(1),
                ),
                ctx.and(
                    ctx.constant(0xff),
                    ctx.add(
                        ctx.register(4),
                        ctx.and(
                            ctx.constant(0xff),
                            ctx.register(1),
                        ),
                    ),
                ),
            ),
        ),
    );

    let eq1b = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.mul(
                ctx.constant(2),
                ctx.and(
                    ctx.constant(0xff),
                    ctx.register(1),
                ),
            ),
            ctx.and(
                ctx.constant(0xff),
                ctx.add(
                    ctx.register(4),
                    ctx.and(
                        ctx.constant(0xff),
                        ctx.register(1),
                    ),
                ),
            ),
        ),
    );

    let op2 = ctx.and(
        ctx.constant(0x3fffffff),
        ctx.add(
            ctx.and(
                ctx.constant(0x3fffffff),
                ctx.register(1),
            ),
            ctx.and(
                ctx.constant(0x3fffffff),
                ctx.add(
                    ctx.and(
                        ctx.constant(0x3fffffff),
                        ctx.register(1),
                    ),
                    ctx.and(
                        ctx.constant(0x3fffffff),
                        ctx.add(
                            ctx.register(4),
                            ctx.and(
                                ctx.constant(0x3fffffff),
                                ctx.register(1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );

    let eq2 = ctx.and(
        ctx.constant(0x3fffffff),
        ctx.add(
            ctx.and(
                ctx.constant(0x3fffffff),
                ctx.register(1),
            ),
            ctx.add(
                ctx.and(
                    ctx.constant(0x3fffffff),
                    ctx.register(1),
                ),
                ctx.and(
                    ctx.constant(0x3fffffff),
                    ctx.add(
                        ctx.register(4),
                        ctx.and(
                            ctx.constant(0x3fffffff),
                            ctx.register(1),
                        ),
                    ),
                ),
            ),
        ),
    );

    let op3 = ctx.and(
        ctx.constant(0x7fffffff),
        ctx.add(
            ctx.and(
                ctx.constant(0x7fffffff),
                ctx.register(1),
            ),
            ctx.and(
                ctx.constant(0x7fffffff),
                ctx.add(
                    ctx.and(
                        ctx.constant(0x7fffffff),
                        ctx.register(1),
                    ),
                    ctx.and(
                        ctx.constant(0x7fffffff),
                        ctx.add(
                            ctx.register(4),
                            ctx.and(
                                ctx.constant(0x7fffffff),
                                ctx.register(1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );

    let eq3 = ctx.and(
        ctx.constant(0x7fffffff),
        ctx.add(
            ctx.and(
                ctx.constant(0x7fffffff),
                ctx.register(1),
            ),
            ctx.add(
                ctx.and(
                    ctx.constant(0x7fffffff),
                    ctx.register(1),
                ),
                ctx.and(
                    ctx.constant(0x7fffffff),
                    ctx.add(
                        ctx.register(4),
                        ctx.and(
                            ctx.constant(0x7fffffff),
                            ctx.register(1),
                        ),
                    ),
                ),
            ),
        ),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1, eq1b);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
}

#[test]
fn simplify_and_masks2() {
    let ctx = &OperandContext::new();
    // One and can be removed since zext(u8) + zext(u8) won't overflow u32
    let op1 = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.mul(
                ctx.constant(2),
                ctx.and(
                    ctx.constant(0xff),
                    ctx.register(1),
                ),
            ),
            ctx.and(
                ctx.constant(0xff),
                ctx.add(
                    ctx.mul(
                        ctx.constant(2),
                        ctx.and(
                            ctx.constant(0xff),
                            ctx.register(1),
                        ),
                    ),
                    ctx.and(
                        ctx.constant(0xff),
                        ctx.add(
                            ctx.register(4),
                            ctx.and(
                                ctx.constant(0xff),
                                ctx.register(1),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    );

    let eq1 = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.mul(
                ctx.constant(4),
                ctx.and(
                    ctx.constant(0xff),
                    ctx.register(1),
                ),
            ),
            ctx.and(
                ctx.constant(0xff),
                ctx.add(
                    ctx.register(4),
                    ctx.and(
                        ctx.constant(0xff),
                        ctx.register(1),
                    ),
                ),
            ),
        ),
    );

    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_and_xor() {
    let ctx = &OperandContext::new();
    // c1 ^ ((x ^ c1) & c2) == x & c2 if c2 & c1 == c1
    // (Effectively going to transform c1 ^ (y & c2) == (y ^ (c1 & c2)) & c2)
    let op1 = ctx.xor(
        ctx.constant(0x423),
        ctx.and(
            ctx.constant(0xfff),
            ctx.xor(
                ctx.constant(0x423),
                ctx.register(1),
            ),
        ),
    );
    let eq1 = ctx.and(
        ctx.constant(0xfff),
        ctx.register(1),
    );

    let op2 = ctx.xor(
        ctx.constant(0x423),
        ctx.or(
            ctx.and(
                ctx.constant(0xfff),
                ctx.xor(
                    ctx.constant(0x423),
                    ctx.mem32(ctx.register(1), 0),
                ),
            ),
            ctx.and(
                ctx.constant(0xffff_f000),
                ctx.mem32(ctx.register(1), 0),
            ),
        )
    );
    let eq2 = ctx.mem32(ctx.register(1), 0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_or_mem_bug2() {
    let ctx = &OperandContext::new();
    // Just checking that this doesn't panic
    let _ = ctx.or(
        ctx.and(
            ctx.rsh(
                ctx.mem32(ctx.sub(ctx.register(2), ctx.constant(0x1)), 0),
                ctx.constant(8),
            ),
            ctx.constant(0x00ff_ffff),
        ),
        ctx.and(
            ctx.mem32(ctx.sub(ctx.register(2), ctx.constant(0x14)), 0),
            ctx.constant(0xff00_0000),
        ),
    );
}

#[test]
fn simplify_panic() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xff),
        ctx.rsh(
            ctx.add(
                ctx.constant(0x1d),
                ctx.eq(
                    ctx.eq(
                        ctx.and(
                            ctx.constant(1),
                            ctx.mem8(ctx.register(3), 0),
                        ),
                        ctx.constant(0),
                    ),
                    ctx.constant(0),
                ),
            ),
            ctx.constant(8),
        ),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn shift_xor_parts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.xor(
            ctx.constant(0xffe60000),
            ctx.xor(
                ctx.lsh(ctx.mem16(ctx.register(5), 0), ctx.constant(0x10)),
                ctx.mem32(ctx.register(5), 0),
            ),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.xor(
        ctx.constant(0xffe6),
        ctx.xor(
            ctx.mem16(ctx.register(5), 0),
            ctx.rsh(ctx.mem32(ctx.register(5), 0), ctx.constant(0x10)),
        ),
    );
    let op2 = ctx.lsh(
        ctx.xor(
            ctx.constant(0xffe6),
            ctx.mem16(ctx.register(5), 0),
        ),
        ctx.constant(0x10),
    );
    let eq2 = ctx.xor(
        ctx.constant(0xffe60000),
        ctx.lsh(ctx.mem16(ctx.register(5), 0), ctx.constant(0x10)),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn lea_mul_9() {
    let ctx = &OperandContext::new();
    let base = ctx.add(
        ctx.constant(0xc),
        ctx.and(
            ctx.constant(0xffff_ff7f),
            ctx.mem32(ctx.register(1), 0),
        ),
    );
    let op1 = ctx.add(
        base,
        ctx.mul(
            base,
            ctx.constant(8),
        ),
    );
    let eq1 = ctx.mul(
        base,
        ctx.constant(9),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn lsh_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(
        ctx.mul(
            ctx.constant(0x9),
            ctx.mem32(ctx.register(1), 0),
        ),
        ctx.constant(0x2),
    );
    let eq1 = ctx.mul(
        ctx.mem32(ctx.register(1), 0),
        ctx.constant(0x24),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn lea_mul_negative() {
    let ctx = &OperandContext::new();
    let base = ctx.sub(
        ctx.mem16(ctx.register(3), 0),
        ctx.constant(1),
    );
    let op1 = ctx.add(
        ctx.constant(0x1234),
        ctx.mul(
            base,
            ctx.constant(0x4),
        ),
    );
    let eq1 = ctx.add(
        ctx.constant(0x1230),
        ctx.mul(
            ctx.mem16(ctx.register(3), 0),
            ctx.constant(0x4),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_u32_max() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffff_ffff),
        ctx.constant(0xffff_ffff),
    );
    let eq1 = ctx.constant(0xffff_ffff);
    let op2 = ctx.and(
        ctx.constant(!0),
        ctx.constant(!0),
    );
    let eq2 = ctx.constant(!0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn and_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffff_ffff_ffff),
        ctx.constant(0x12456),
    );
    let eq1 = ctx.constant(0x12456);
    let op2 = ctx.and(
        ctx.mem32(ctx.register(0), 0),
        ctx.constant(!0),
    );
    let eq2 = ctx.mem32(ctx.register(0), 0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn short_and_is_32() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.mem32(ctx.register(0), 0),
        ctx.mem32(ctx.register(1), 0),
    );
    match op1.ty() {
        OperandType::Arithmetic(..) => (),
        _ => panic!("Simplified was {}", op1),
    }
}

#[test]
fn and_32bit() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffff_ffff),
        ctx.mem32(ctx.register(1), 0),
    );
    let eq1 = ctx.mem32(ctx.register(1), 0);
    assert_eq!(op1, eq1);
}

#[test]
fn mem8_mem32_shift_eq() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xff),
        ctx.rsh(
            ctx.mem32(ctx.register(1), 0x4c),
            ctx.constant(0x8),
        ),
    );
    let eq1 = ctx.mem8(ctx.register(1), 0x4d);
    assert_eq!(op1, eq1);
}

#[test]
fn or_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xffff_0000_0000),
        ctx.constant(0x12456),
    );
    let eq1 = ctx.constant(0xffff_0001_2456);
    let op2 = ctx.or(
        ctx.register(0),
        ctx.constant(0),
    );
    let eq2 = ctx.register(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn lsh_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(
        ctx.constant(0x4),
        ctx.constant(0x28),
    );
    let eq1 = ctx.constant(0x0000_0400_0000_0000);
    assert_eq!(op1, eq1);
}

#[test]
fn xor_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.constant(0x4000_0000_0000),
        ctx.constant(0x6000_0000_0000),
    );
    let eq1 = ctx.constant(0x2000_0000_0000);
    assert_eq!(op1, eq1);
}

#[test]
fn eq_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.constant(0x40),
        ctx.constant(0x00),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn and_bug_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.constant(0xffff_ffff),
            ctx.rsh(
                ctx.mem8(
                    ctx.add(
                        ctx.constant(0xf105b2a),
                        ctx.and(
                            ctx.constant(0xffff_ffff),
                            ctx.add(
                                ctx.register(1),
                                ctx.constant(0xd6057390),
                            ),
                        ),
                    ),
                    0,
                ),
                ctx.constant(0xffffffffffffffda),
            ),
        ),
        ctx.constant(0xff),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt_or_eq() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.gt(
            ctx.constant(0x5),
            ctx.register(1),
        ),
        ctx.eq(
            ctx.constant(0x5),
            ctx.register(1),
        ),
    );
    let eq1 = ctx.gt(
        ctx.constant(0x6),
        ctx.register(1),
    );
    let op2 = ctx.or(
        ctx.gt(
            ctx.constant(0x5),
            ctx.register(1),
        ),
        ctx.eq(
            ctx.constant(0x5),
            ctx.register(1),
        ),
    );
    // Confirm that 6 > rcx isn't 6 > ecx
    let ne2 = ctx.gt(
        ctx.constant(0x6),
        ctx.and(
            ctx.register(1),
            ctx.constant(0xffff_ffff),
        ),
    );
    let eq2 = ctx.gt(
        ctx.constant(0x6),
        ctx.register(1),
    );
    let op3 = ctx.or(
        ctx.gt(
            ctx.constant(0x5_0000_0000),
            ctx.register(1),
        ),
        ctx.eq(
            ctx.constant(0x5_0000_0000),
            ctx.register(1),
        ),
    );
    let eq3 = ctx.gt(
        ctx.constant(0x5_0000_0001),
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
    assert_ne!(op2, ne2);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
}

#[test]
fn pointless_gt() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.constant(0),
        ctx.register(0),
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.gt(
        ctx.and(
            ctx.register(0),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq2 = ctx.constant(0);
    let op3 = ctx.gt(
        ctx.register(0),
        ctx.constant(u64::MAX),
    );
    let eq3 = ctx.constant(0);
    let op4 = ctx.gt(
        ctx.register(0),
        ctx.constant(0xffff_ffff),
    );
    let ne4 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_ne!(op4, ne4);
}

#[test]
fn and_64_to_32() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.register(0),
        ctx.constant(0xf9124),
    );
    let eq1 = ctx.and(
        ctx.register(0),
        ctx.constant(0xf9124),
    );
    let op2 = ctx.and(
        ctx.add(
            ctx.register(0),
            ctx.register(2),
        ),
        ctx.constant(0xf9124),
    );
    let eq2 = ctx.and(
        ctx.add(
            ctx.register(0),
            ctx.register(2),
        ),
        ctx.constant(0xf9124),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_bug_xor_and_u32_max() {
    let ctx = &OperandContext::new();
    let unk = ctx.new_undef();
    let op1 = ctx.xor(
        ctx.and(
            ctx.mem32(unk, 0),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq1 = ctx.xor(
        ctx.mem32(unk, 0),
        ctx.constant(0xffff_ffff),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_64_to_32() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.register(0),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.register(0),
        ctx.constant(0),
    );
    let op2 = ctx.eq(
        ctx.register(0),
        ctx.register(2),
    );
    let eq2 = ctx.eq(
        ctx.register(0),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_read_middle_u16_from_mem32() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xffff),
        ctx.rsh(
            ctx.mem32(ctx.const_0(), 0x11230),
            ctx.constant(8),
        ),
    );
    let eq1 = ctx.mem16(ctx.const_0(), 0x11231);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_unnecessary_shift_in_eq_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.lsh(
            ctx.and(
                ctx.mem8(ctx.register(4), 0),
                ctx.constant(8),
            ),
            ctx.constant(0xc),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(4), 0),
            ctx.constant(8),
        ),
        ctx.constant(0),
    );
    let op2 = ctx.eq(
        ctx.rsh(
            ctx.and(
                ctx.mem8(ctx.register(4), 0),
                ctx.constant(8),
            ),
            ctx.constant(1),
        ),
        ctx.constant(0),
    );
    let eq2 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(4), 0),
            ctx.constant(8),
        ),
        ctx.constant(0),
    );
    let op3 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(4), 0),
            ctx.constant(8),
        ),
        ctx.constant(0),
    );
    let ne3 = ctx.eq(
        ctx.mem8(ctx.register(4), 0),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_ne!(op3, ne3);
}

#[test]
fn simplify_unnecessary_and_in_shifts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.and(
            ctx.lsh(
                ctx.mem8(ctx.constant(0x100), 0),
                ctx.constant(0xd),
            ),
            ctx.constant(0x1f0000),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.rsh(
        ctx.mem8(ctx.constant(0x100), 0),
        ctx.constant(0x3),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_set_bit_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.mem16(ctx.constant(0x1000), 0),
            ctx.constant(0xffef),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.or(
        ctx.mem16(ctx.constant(0x1000), 0),
        ctx.constant(0x10),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_masked_mul_lsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(
        ctx.and(
            ctx.mul(
                ctx.mem32(ctx.constant(0x1000), 0),
                ctx.constant(9),
            ),
            ctx.constant(0x3fff_ffff),
        ),
        ctx.constant(0x2),
    );
    let eq1 = ctx.and(
        ctx.mul(
            ctx.mem32(ctx.constant(0x1000), 0),
            ctx.constant(0x24),
        ),
        ctx.constant(0xffff_ffff),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_inner_masks_on_arith() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.register(4),
            ctx.and(
                ctx.constant(0xff),
                ctx.register(1),
            ),
        ),
    );
    let eq1 = ctx.and(
        ctx.constant(0xff),
        ctx.add(
            ctx.register(4),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_add_to_const_0() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(
        ctx.add(
            ctx.constant(1),
            ctx.mem32(ctx.constant(0x5000), 0),
        ),
        ctx.constant(u64::MAX),
    );
    let eq1 = ctx.mem32(ctx.constant(0x5000), 0);
    let op2 = ctx.and(
        ctx.add(
            ctx.add(
                ctx.constant(1),
                ctx.mem32(ctx.constant(0x5000), 0),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq2 = ctx.mem32(ctx.constant(0x5000), 0);
    let op3 = ctx.and(
        ctx.add(
            ctx.add(
                ctx.constant(1),
                ctx.mem32(ctx.constant(0x5000), 0),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq3 = ctx.mem32(ctx.constant(0x5000), 0);
    let op4 = ctx.add(
        ctx.add(
            ctx.constant(1),
            ctx.mem32(ctx.constant(0x5000), 0),
        ),
        ctx.constant(0xffff_ffff_ffff_ffff),
    );
    let eq4 = ctx.mem32(ctx.constant(0x5000), 0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_eq!(op4, eq4);
}

#[test]
fn simplify_sub_self_masked() {
    let ctx = &OperandContext::new();
    let ud = ctx.new_undef() ;
    let op1 = ctx.sub(
        ctx.and(
            ud,
            ctx.constant(0xffff_ffff),
        ),
        ctx.and(
            ud,
            ctx.constant(0xffff_ffff),
        ),
    );
    let eq1 = ctx.const_0();
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_rsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.and(
            ctx.mem8(ctx.constant(0x900), 0),
            ctx.constant(0xf8),
        ),
        ctx.constant(3),
    );
    let eq1 = ctx.rsh(
        ctx.mem8(ctx.constant(0x900), 0),
        ctx.constant(3),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_64() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.lsh(
            ctx.and(
                ctx.constant(0),
                ctx.constant(0xffff_ffff),
            ),
            ctx.constant(0x20),
        ),
        ctx.and(
            ctx.register(0),
            ctx.constant(0xffff_ffff),
        ),
    );
    let eq1 = ctx.and(
        ctx.register(0),
        ctx.constant(0xffff_ffff),
    );
    let ne1 = ctx.register(0);
    assert_eq!(op1, eq1);
    assert_ne!(op1, ne1);
}

#[test]
fn gt_same() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.register(6),
        ctx.register(6),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_useless_and_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.lsh(
            ctx.and(
                ctx.register(0),
                ctx.constant(0xff),
            ),
            ctx.constant(1),
        ),
        ctx.constant(0x1fe),
    );
    let eq1 = ctx.lsh(
        ctx.and(
            ctx.register(0),
            ctx.constant(0xff),
        ),
        ctx.constant(1),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt_masked() {
    // x - y > x => y > x,
    // just with a mask
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.and(
            ctx.sub(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.constant(0x1ff),
        ),
        ctx.and(
            ctx.register(0),
            ctx.constant(0x1ff),
        ),
    );
    let eq1 = ctx.gt(
        ctx.and(
            ctx.register(1),
            ctx.constant(0x1ff),
        ),
        ctx.and(
            ctx.register(0),
            ctx.constant(0x1ff),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn cannot_simplify_mask_sub() {
    let ctx = &OperandContext::new();
    let op1 = ctx.sub(
        ctx.and(
            ctx.sub(
                ctx.constant(0x4234),
                ctx.register(0),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0x1ff),
    );
    // Cannot move the outer sub inside and
    assert!(op1.if_arithmetic_sub().is_some());
}

#[test]
fn simplify_bug_panic() {
    let ctx = &OperandContext::new();
    // Doesn't simplify, but used to cause a panic
    let _ = ctx.eq(
        ctx.and(
            ctx.sub(
                ctx.constant(0),
                ctx.register(1),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0),
    );
}

#[test]
fn simplify_lsh_and_rsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.and(
            ctx.lsh(
                ctx.register(1),
                ctx.constant(0x10),
            ),
            ctx.constant(0xffff_0000),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.and(
        ctx.register(1),
        ctx.constant(0xffff),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_lsh_and_rsh2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.sub(
            ctx.mem16(ctx.register(2), 0),
            ctx.lsh(
                ctx.mem16(ctx.register(1), 0),
                ctx.constant(0x9),
            ),
        ),
        ctx.constant(0xffff),
    );
    let eq1 = ctx.rsh(
        ctx.and(
            ctx.lsh(
                ctx.sub(
                    ctx.mem16(ctx.register(2), 0),
                    ctx.lsh(
                        ctx.mem16(ctx.register(1), 0),
                        ctx.constant(0x9),
                    ),
                ),
                ctx.constant(0x10),
            ),
            ctx.constant(0xffff0000),
        ),
        ctx.constant(0x10),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_ne_shifted_and() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.and(
            ctx.lsh(
                ctx.mem8(ctx.register(2), 0),
                ctx.constant(0x8),
            ),
            ctx.constant(0x800),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(2), 0),
            ctx.constant(0x8),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn xor_shift_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.and(
            ctx.xor(
                ctx.constant(0xffff_a987_5678),
                ctx.lsh(
                    ctx.rsh(
                        ctx.mem8(ctx.constant(0x223345), 0),
                        ctx.constant(3),
                    ),
                    ctx.constant(0x10),
                ),
            ),
            ctx.constant(0xffff_0000),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.xor(
        ctx.constant(0xa987),
        ctx.rsh(
            ctx.mem8(ctx.constant(0x223345), 0),
            ctx.constant(3),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_shift_and() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh(
        ctx.and(
            ctx.register(0),
            ctx.constant(0xffff_0000),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.and(
        ctx.rsh(
            ctx.register(0),
            ctx.constant(0x10),
        ),
        ctx.constant(0xffff),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_rotate_mask() {
    // This is mainly useful to make sure rol32(reg32, const) substituted
    // with mem32 is same as just rol32(mem32, const)
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.rsh(
                ctx.and(
                    ctx.register(0),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.constant(0xb),
            ),
            ctx.lsh(
                ctx.and(
                    ctx.register(0),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.constant(0x15),
            ),
        ),
        ctx.constant(0xffff_ffff),
    );
    let subst = ctx.substitute(op1, ctx.register(0), ctx.mem32(ctx.const_0(), 0x1234), 100);
    let with_mem = ctx.and(
        ctx.or(
            ctx.rsh(
                ctx.mem32(ctx.const_0(), 0x1234),
                ctx.constant(0xb),
            ),
            ctx.lsh(
                ctx.mem32(ctx.const_0(), 0x1234),
                ctx.constant(0x15),
            ),
        ),
        ctx.constant(0xffff_ffff),
    );
    assert_eq!(subst, with_mem);
}

#[test]
fn simplify_add_sub_to_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(
        ctx.and(
            ctx.register(0),
            ctx.constant(0xffff_ffff),
        ),
        ctx.and(
            ctx.sub(
                ctx.constant(0),
                ctx.register(0),
            ),
            ctx.constant(0xffff_ffff),
        ),
    );
    let eq1 = ctx.constant(0x1_0000_0000);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_less_or_eq() {
    let ctx = &OperandContext::new();
    // not(c > x) & not(x == c) => not(c + 1 > x)
    // (Same as not((c > x) | (x == c)))
    let op1 = ctx.and(
        ctx.eq(
            ctx.constant(0),
            ctx.gt(
                ctx.constant(5),
                ctx.register(1),
            ),
        ),
        ctx.eq(
            ctx.constant(0),
            ctx.eq(
                ctx.constant(5),
                ctx.register(1),
            ),
        ),
    );
    let eq1 = ctx.eq(
        ctx.constant(0),
        ctx.gt(
            ctx.constant(6),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.add(
            ctx.register(1),
            ctx.register(2),
        ),
        ctx.register(3),
    );
    let eq1a = ctx.eq(
        ctx.sub(
            ctx.register(3),
            ctx.register(2),
        ),
        ctx.register(1),
    );
    let eq1b = ctx.eq(
        ctx.add(
            ctx.register(2),
            ctx.register(1),
        ),
        ctx.register(3),
    );
    assert_eq!(op1, eq1a);
    assert_eq!(op1, eq1b);
}

#[test]
fn simplify_mul_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul(
        ctx.register(1),
        ctx.add(
            ctx.register(0),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.mul(
        ctx.mul(
            ctx.constant(2),
            ctx.register(0),
        ),
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_sub_add_2_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(
        ctx.sub(
            ctx.register(1),
            ctx.add(
                ctx.register(2),
                ctx.register(2),
            ),
        ),
        ctx.register(3),
    );
    let eq1 = ctx.add(
        ctx.sub(
            ctx.register(1),
            ctx.mul(
                ctx.register(2),
                ctx.constant(2),
            ),
        ),
        ctx.register(3),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mul_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul(
        ctx.constant(0x50505230402c2f4),
        ctx.add(
            ctx.constant(0x100ffee),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.add(
        ctx.constant(0xcdccaa4f6ec24ad8),
        ctx.mul(
            ctx.constant(0x50505230402c2f4),
            ctx.register(0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.eq(
            ctx.register(0),
            ctx.register(0),
        ),
        ctx.eq(
            ctx.register(0),
            ctx.register(1),
        ),
    );
    let eq1 = ctx.eq(
        ctx.eq(
            ctx.register(0),
            ctx.register(1),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_fully() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.mem64(ctx.register(0), 0),
        ctx.mem8(ctx.register(0), 0),
    );
    let eq1 = ctx.mem8(ctx.register(0), 0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mul_consistency3() {
    let ctx = &OperandContext::new();
    // 100 * r8 * (r6 + 8) => r8 * (100 * (r6 + 8)) => r8 * ((100 * r6) + 800)
    let op1 = ctx.mul(
        ctx.mul(
            ctx.mul(
                ctx.register(8),
                ctx.add(
                    ctx.register(6),
                    ctx.constant(0x8),
                ),
            ),
            ctx.constant(0x10),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.mul(
        ctx.register(8),
        ctx.add(
            ctx.mul(
                ctx.register(6),
                ctx.constant(0x100),
            ),
            ctx.constant(0x800),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.add(
            ctx.register(0),
            ctx.sub(
                ctx.or(
                    ctx.register(2),
                    ctx.or(
                        ctx.add(
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
    let eq1 = ctx.or(
        ctx.register(2),
        ctx.or(
            ctx.mul(
                ctx.register(0),
                ctx.constant(2),
            ),
            ctx.or(
                ctx.register(1),
                ctx.register(5),
            ),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mul_consistency4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul(
        ctx.mul(
            ctx.mul(
                ctx.add(
                    ctx.mul(
                        ctx.add(
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
        ctx.mul(
            ctx.register(0),
            ctx.register(8),
        ),
    );
    let eq1 = ctx.mul(
        ctx.mul(
            ctx.register(0),
            ctx.mul(
                ctx.register(8),
                ctx.register(8),
            ),
        ),
        ctx.mul(
            ctx.add(
                ctx.register(1),
                ctx.register(2),
            ),
            ctx.constant(0x400000000000000),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.mem64(ctx.register(0), 0),
        ctx.and(
            ctx.or(
                ctx.constant(0xfd0700002ff4004b),
                ctx.mem8(ctx.register(5), 0),
            ),
            ctx.constant(0x293b00be00),
        ),
    );
    let eq1 = ctx.eq(
        ctx.mem64(ctx.register(0), 0),
        ctx.constant(0x2b000000),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.mem8(ctx.register(0), 0),
        ctx.or(
            ctx.and(
                ctx.constant(0xfeffffffffffff24),
                ctx.add(
                    ctx.register(0),
                    ctx.constant(0x2fbfb01ffff0000),
                ),
            ),
            ctx.constant(0xf3fb000091010e00),
        ),
    );
    let eq1 = ctx.and(
        ctx.mem8(ctx.register(0), 0),
        ctx.and(
            ctx.register(0),
            ctx.constant(0x24),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mul_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul(
        ctx.mul(
            ctx.add(
                ctx.add(
                    ctx.register(0),
                    ctx.register(0),
                ),
                ctx.constant(0x25000531004000),
            ),
            ctx.add(
                ctx.mul(
                    ctx.constant(0x4040405f6020405),
                    ctx.register(0),
                ),
                ctx.constant(0x25000531004000),
            ),
        ),
        ctx.constant(0xe9f4000000000000),
    );
    let eq1 = ctx.mul(
        ctx.mul(
            ctx.register(0),
            ctx.register(0),
        ),
        ctx.constant(0xc388000000000000)
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.or(
                ctx.mem16(ctx.register(0), 0),
                ctx.constant(0x4eff0001004107),
            ),
            ctx.constant(0x231070100fa00de),
        ),
        ctx.constant(0x280000d200004010),
    );
    let eq1 = ctx.constant(0x4010);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.or(
                ctx.xmm(4, 1),
                ctx.constant(0x1e04ffffff0000),
            ),
            ctx.xmm(0, 1),
        ),
        ctx.constant(0x40ffffffffffff60),
    );
    let eq1 = ctx.and(
        ctx.and(
            ctx.or(
                ctx.constant(0xffff0000),
                ctx.xmm(4, 1),
            ),
            ctx.constant(0xffffff60),
        ),
        ctx.xmm(0, 1),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.constant(0xfe05000080000025),
            ctx.or(
                ctx.xmm(0, 1),
                ctx.constant(0xf3fbfb01ffff0000),
            ),
        ),
        ctx.constant(0xf3fb0073_00000000),
    );
    let eq1 = ctx.or(
        ctx.constant(0xf3fb0073_80000000),
        ctx.and(
            ctx.xmm(0, 1),
            ctx.constant(0x25),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_add_simple() {
    let ctx = &OperandContext::new();
    let op1 = ctx.sub(
        ctx.constant(1),
        ctx.constant(4),
    );
    let eq1 = ctx.constant(0xffff_ffff_ffff_fffd);
    let op2 = ctx.add(
        ctx.sub(
            ctx.constant(0xf40205051a02c2f4),
            ctx.register(0),
        ),
        ctx.register(0),
    );
    let eq2 = ctx.constant(0xf40205051a02c2f4);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_1bit_sum() {
    let ctx = &OperandContext::new();
    // Since the gt makes only at most LSB of the sum to be considered,
    // the multiplication isn't used at all and the sum
    // Mem8[rax] + Mem16[rax] + Mem32[rax] can become 3 * Mem8[rax] which
    // can just be replaced with Mem8[rax]
    let op1 = ctx.and(
        ctx.gt(
            ctx.register(5),
            ctx.register(4),
        ),
        ctx.add(
            ctx.add(
                ctx.mul(
                    ctx.constant(6),
                    ctx.register(0),
                ),
                ctx.add(
                    ctx.mem8(ctx.register(0), 0),
                    ctx.mem32(ctx.register(1), 0),
                ),
            ),
            ctx.add(
                ctx.mem16(ctx.register(0), 0),
                ctx.mem64(ctx.register(0), 0),
            ),
        ),
    );
    let eq1 = ctx.and(
        ctx.gt(
            ctx.register(5),
            ctx.register(4),
        ),
        ctx.add(
            ctx.mem8(ctx.register(0), 0),
            ctx.mem8(ctx.register(1), 0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_masked_add() {
    let ctx = &OperandContext::new();
    // Cannot move the constant out of and since
    // (fffff + 1400) & ffff6ff24 == 101324, but
    // (fffff & ffff6ff24) + 1400 == 71324
    let op1 = ctx.and(
        ctx.add(
            ctx.mem32(ctx.register(0), 0),
            ctx.constant(0x1400),
        ),
        ctx.constant(0xffff6ff24),
    );
    let ne1 = ctx.add(
        ctx.and(
            ctx.mem32(ctx.register(0), 0),
            ctx.constant(0xffff6ff24),
        ),
        ctx.constant(0x1400),
    );
    let op2 = ctx.add(
        ctx.register(1),
        ctx.and(
            ctx.add(
                ctx.mem32(ctx.register(0), 0),
                ctx.constant(0x1400),
            ),
            ctx.constant(0xffff6ff24),
        ),
    );
    let ne2 = ctx.add(
        ctx.register(1),
        ctx.add(
            ctx.and(
                ctx.mem32(ctx.register(0), 0),
                ctx.constant(0xffff6ff24),
            ),
            ctx.constant(0x1400),
        ),
    );
    assert_ne!(op1, ne1);
    assert_ne!(op2, ne2);
}

#[test]
fn simplify_masked_add2() {
    let ctx = &OperandContext::new();
    // Cannot move the constant out of and since
    // (fffff + 1400) & ffff6ff24 == 101324, but
    // (fffff & ffff6ff24) + 1400 == 71324
    let op1 = ctx.add(
        ctx.constant(0x4700000014fef910),
        ctx.and(
            ctx.add(
                ctx.mem32(ctx.register(0), 0),
                ctx.constant(0x1400),
            ),
            ctx.constant(0xffff6ff24),
        ),
    );
    let ne1 = ctx.add(
        ctx.constant(0x4700000014fef910),
        ctx.add(
            ctx.and(
                ctx.mem32(ctx.register(0), 0),
                ctx.constant(0xffff6ff24),
            ),
            ctx.constant(0x1400),
        ),
    );
    assert_ne!(op1, ne1);
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
    let ctx = &OperandContext::new();
    let op1 = ctx.add(
        ctx.and(
            ctx.constant(0xffff),
            ctx.or(
                ctx.register(0),
                ctx.register(1),
            ),
        ),
        ctx.and(
            ctx.constant(0xffff),
            ctx.or(
                ctx.register(0),
                ctx.register(1),
            ),
        ),
    );
    assert!(op1.relevant_bits().end > 16, "Operand wasn't simplified correctly {}", op1);
}

#[test]
fn simplify_or_consistency3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.constant(0x854e00e501001007),
            ctx.mem16(ctx.register(0), 0),
        ),
        ctx.constant(0x28004000d2000010),
    );
    let eq1 = ctx.and(
        ctx.mem8(ctx.register(0), 0),
        ctx.constant(0x10),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.and(
            ctx.sign_extend(
                ctx.constant(0x2991919191910000),
                MemAccessSize::Mem8,
                MemAccessSize::Mem16,
            ),
            ctx.register(1),
        ),
        ctx.mem8(ctx.register(2), 0),
    );
    let eq1 = ctx.eq(
        ctx.mem8(ctx.register(2), 0),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.or(
                ctx.constant(0x80000000000002),
                ctx.xmm(2, 1),
            ),
            ctx.mem16(ctx.register(0), 0),
        ),
        ctx.constant(0x40ffffffff3fff7f),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.xmm(2, 1),
            ctx.mem8(ctx.register(0), 0),
        ),
        ctx.constant(0x40ffffffff3fff7f),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_infinite_recurse_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.or(
                ctx.constant(0x100),
                ctx.and(
                    ctx.mul(
                        ctx.constant(4),
                        ctx.register(0),
                    ),
                    ctx.constant(0xffff_fe00),
                ),
            ),
            ctx.mem32(ctx.register(0), 0),
        ),
        ctx.constant(0xff_ffff_fe00),
    );
    let _ = op1;
}

#[test]
fn simplify_eq_consistency4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.add(
            ctx.constant(0x7014b0001050500),
            ctx.mem32(ctx.register(0), 0),
        ),
        ctx.mem32(ctx.register(1), 0),
    );
    assert_eq!(op1, ctx.const_0());
}

#[test]
fn simplify_eq_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.mem32(ctx.register(1), 0),
        ctx.add(
            ctx.constant(0x5a00000001),
            ctx.mem8(ctx.register(0), 0),
        ),
    );
    assert_eq!(op1, ctx.const_0());
}

#[test]
fn simplify_eq_consistency6() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.register(0),
        ctx.add(
            ctx.register(0),
            ctx.rsh(
                ctx.mem16(ctx.register(0), 0),
                ctx.constant(5),
            ),
        ),
    );
    let eq1a = ctx.eq(
        ctx.constant(0),
        ctx.rsh(
            ctx.mem16(ctx.register(0), 0),
            ctx.constant(5),
        ),
    );
    let eq1b = ctx.eq(
        ctx.constant(0),
        ctx.and(
            ctx.mem16(ctx.register(0), 0),
            ctx.constant(0xffe0),
        ),
    );
    assert_eq!(op1, eq1a);
    assert_eq!(op1, eq1b);
}

#[test]
fn simplify_eq_consistency7() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.constant(0x4000000000570000),
        ctx.add(
            ctx.mem32(ctx.register(0), 0),
            ctx.add(
                ctx.mem32(ctx.register(1), 0),
                ctx.constant(0x7e0000fffc01),
            ),
        ),
    );
    let eq1 = ctx.eq(
        ctx.constant(0x3fff81ffff5703ff),
        ctx.add(
            ctx.mem32(ctx.register(0), 0),
            ctx.mem32(ctx.register(1), 0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_zero_eq_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.constant(0),
        ctx.constant(0),
    );
    let eq1 = ctx.constant(1);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_consistency1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.xor(
            ctx.register(1),
            ctx.constant(0x5910e010000),
        ),
        ctx.and(
            ctx.or(
                ctx.register(2),
                ctx.constant(0xf3fbfb01ffff0000),
            ),
            ctx.constant(0x1ffffff24),
        ),
    );
    let eq1 = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.constant(0x590f1fe0000),
            ctx.and(
                ctx.constant(0xff24),
                ctx.register(2),
            ),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.xor(
            ctx.register(1),
            ctx.constant(0x59100010e00),
        ),
        ctx.and(
            ctx.or(
                ctx.register(2),
                ctx.constant(0xf3fbfb01ffff7e00),
            ),
            ctx.constant(0x1ffffff24),
        ),
    );
    let eq1 = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.constant(0x590fffe7000),
            ctx.and(
                ctx.constant(0x8124),
                ctx.register(2),
            ),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xffff7024_ffffffff),
        ctx.and(
            ctx.mem64(ctx.const_0(), 0x100),
            ctx.constant(0x0500ff04_ffff0000),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0xffff7024ffffffff),
        ctx.lsh(
            ctx.mem8(ctx.const_0(), 0x105),
            ctx.constant(0x28),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency6() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xfffeffffffffffff),
        ctx.and(
            ctx.xmm(0, 0),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.constant(0xfffeffffffffffff);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_useless_mod() {
    let ctx = &OperandContext::new();
    let op1 = ctx.modulo(
        ctx.xmm(0, 0),
        ctx.constant(0x504ff04ff0000),
    );
    let eq1 = ctx.xmm(0, 0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency7() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xffffffffffffff41),
        ctx.and(
            ctx.xmm(0, 0),
            ctx.or(
                ctx.register(0),
                ctx.constant(0x504ffffff770000),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0xffffffffffffff41),
        ctx.and(
            ctx.xmm(0, 0),
            ctx.register(0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency8() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0x40ff_ffff_ffff_3fff),
        ctx.and(
            ctx.mem64(ctx.register(0), 0),
            ctx.or(
                ctx.xmm(0, 0),
                ctx.constant(0x0080_0000_0000_0002),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0x40ff_ffff_ffff_3fff),
        ctx.and(
            ctx.mem16(ctx.register(0), 0),
            ctx.xmm(0, 0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.mem8(ctx.register(1), 0),
            ctx.or(
                ctx.constant(0x22),
                ctx.xmm(0, 0),
            ),
        ),
        ctx.constant(0x23),
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_and_consistency6() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.constant(0xfeffffffffffff24),
            ctx.or(
                ctx.constant(0xf3fbfb01ffff0000),
                ctx.xmm(0, 0),
            ),
        ),
        ctx.or(
            ctx.constant(0xf3fb000091010e03),
            ctx.mem8(ctx.register(1), 0),
        ),
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_or_consistency9() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0x47000000140010ff),
        ctx.or(
            ctx.mem16(ctx.const_0(), 0x100),
            ctx.or(
                ctx.constant(0x2a00000100100730),
                ctx.mul(
                    ctx.constant(0x2000000000),
                    ctx.xmm(4, 1),
                ),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0x6f000001141017ff),
        ctx.or(
            ctx.lsh(
                ctx.mem8(ctx.const_0(), 0x101),
                ctx.constant(8),
            ),
            ctx.mul(
                ctx.constant(0x2000000000),
                ctx.xmm(4, 1),
            ),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency7() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.constant(0xfeffffffffffff24),
            ctx.or(
                ctx.constant(0xf3fbfb01ffff0000),
                ctx.xmm(0, 0),
            ),
        ),
        ctx.or(
            ctx.constant(0xc04ffff6efef1f6),
            ctx.mem8(ctx.register(1), 0),
        ),
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_and_consistency8() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.add(
                ctx.constant(0x5050505000001),
                ctx.register(0),
            ),
            ctx.modulo(
                ctx.register(4),
                ctx.constant(0x3ff0100000102),
            ),
        ),
        ctx.constant(0x3ff01000001),
    );
    let eq1 = ctx.and(
        ctx.and(
            ctx.add(
                ctx.constant(0x5050505000001),
                ctx.register(0),
            ),
            ctx.modulo(
                ctx.register(4),
                ctx.constant(0x3ff0100000102),
            ),
        ),
        ctx.constant(0x50003ff01000001),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency8() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.add(
            ctx.add(
                ctx.add(
                    ctx.constant(1),
                    ctx.modulo(
                        ctx.register(0),
                        ctx.mem8(ctx.register(0), 0),
                    ),
                ),
                ctx.div(
                    ctx.eq(
                        ctx.register(0),
                        ctx.register(1),
                    ),
                    ctx.eq(
                        ctx.register(4),
                        ctx.register(6),
                    ),
                ),
            ),
            ctx.eq(
                ctx.register(4),
                ctx.register(5),
            ),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.add(
            ctx.add(
                ctx.add(
                    ctx.constant(1),
                    ctx.modulo(
                        ctx.register(0),
                        ctx.mem8(ctx.register(0), 0),
                    ),
                ),
                ctx.div(
                    ctx.eq(
                        ctx.register(0),
                        ctx.register(1),
                    ),
                    ctx.eq(
                        ctx.register(4),
                        ctx.register(6),
                    ),
                ),
            ),
            ctx.eq(
                ctx.register(4),
                ctx.register(5),
            ),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt_consistency1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.sub(
            ctx.gt(
                ctx.sub(
                    ctx.register(0),
                    ctx.register(0),
                ),
                ctx.register(5),
            ),
            ctx.register(0),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.eq(
            ctx.register(0),
            ctx.constant(0),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_add_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(
        ctx.add(
            ctx.or(
                ctx.sub(
                    ctx.register(1),
                    ctx.add(
                        ctx.register(0),
                        ctx.register(0),
                    ),
                ),
                ctx.constant(0),
            ),
            ctx.add(
                ctx.register(0),
                ctx.register(0),
            ),
        ),
        ctx.register(0),
    );
    let eq1 = ctx.add(
        ctx.register(0),
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency9() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.and(
            ctx.add(
                ctx.register(1),
                ctx.constant(0x50007f3fbff0000),
            ),
            ctx.or(
                ctx.xmm(0, 1),
                ctx.constant(0xf3fbfb01ffff0000),
            ),
        ),
        ctx.constant(0x6080e6300000000),
    );
    let eq1 = ctx.and(
        ctx.add(
            ctx.register(1),
            ctx.constant(0x50007f3fbff0000),
        ),
        ctx.constant(0x2080a0100000000),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_bug_infloop1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xe20040ffe000e500),
        ctx.xor(
            ctx.constant(0xe20040ffe000e500),
            ctx.or(
                ctx.register(0),
                ctx.register(1),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0xe20040ffe000e500),
        ctx.or(
            ctx.register(0),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency9() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.and(
            ctx.xmm(0, 0),
            ctx.constant(0x40005ff000000ff),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.xmm(0, 0),
            ctx.constant(0xff),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency10() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.sub(
            ctx.modulo(
                ctx.xmm(0, 0),
                ctx.register(0),
            ),
            ctx.sub(
                ctx.constant(0),
                ctx.modulo(
                    ctx.register(1),
                    ctx.constant(0),
                ),
            ),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.eq(
            ctx.add(
                ctx.modulo(
                    ctx.xmm(0, 0),
                    ctx.register(0),
                ),
                ctx.modulo(
                    ctx.register(1),
                    ctx.constant(0),
                ),
            ),
            ctx.constant(0),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
    assert!(op1.iter().any(|x| match x.if_arithmetic(ArithOpType::Modulo) {
        Some((a, b)) => a.if_constant() == Some(0) && b.if_constant() == Some(0),
        None => false,
    }), "0 / 0 disappeared: {}", op1);
}

#[test]
fn simplify_eq_consistency11() {
    let ctx = &OperandContext::new();
    // x << 1 != 0 and x & 0x7fff_ffff_ffff_ffff != 0
    // are both true iff any of the 63 lowest bits is nonzero
    let op1 = ctx.gt(
        ctx.lsh(
            ctx.sub(
                ctx.xmm(0, 0),
                ctx.register(0),
            ),
            ctx.constant(1),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.eq(
            ctx.and(
                ctx.sub(
                    ctx.xmm(0, 0),
                    ctx.register(0),
                ),
                ctx.constant(0x7fff_ffff_ffff_ffff),
            ),
            ctx.constant(0),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency12() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.and(
            ctx.add(
                ctx.xmm(0, 0),
                ctx.register(0),
            ),
            ctx.add(
                ctx.constant(1),
                ctx.constant(0),
            ),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.add(
                ctx.xmm(0, 0),
                ctx.register(0),
            ),
            ctx.constant(1),
        ),
        ctx.constant(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_consistency13() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.add(
            ctx.sub(
                ctx.and(
                    ctx.constant(0xffff),
                    ctx.register(1),
                ),
                ctx.mem8(ctx.register(0), 0),
            ),
            ctx.xmm(0, 0),
        ),
        ctx.add(
            ctx.and(
                ctx.register(3),
                ctx.constant(0x7f),
            ),
            ctx.xmm(0, 0),
        ),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.constant(0xffff),
            ctx.register(1),
        ),
        ctx.add(
            ctx.and(
                ctx.register(3),
                ctx.constant(0x7f),
            ),
            ctx.mem8(ctx.register(0), 0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency10() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0x5ffff05b700),
        ctx.and(
            ctx.or(
                ctx.xmm(1, 0),
                ctx.constant(0x5ffffffff00),
            ),
            ctx.or(
                ctx.register(0),
                ctx.constant(0x5ffffffff0000),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0x5ffff050000),
        ctx.and(
            ctx.register(0),
            ctx.constant(0xb700),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_consistency3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0x200ffffff7f),
        ctx.xor(
            ctx.or(
                ctx.xmm(1, 0),
                ctx.constant(0x20000ff20ffff00),
            ),
            ctx.or(
                ctx.register(0),
                ctx.constant(0x5ffffffff0000),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0x200ffffff7f),
        ctx.xor(
            ctx.xor(
                ctx.xmm(1, 0),
                ctx.constant(0x20000ff00000000),
            ),
            ctx.or(
                ctx.register(0),
                ctx.constant(0x5fdff00000000),
            ),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency10() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xffff_ffff_ffff),
        ctx.or(
            ctx.xor(
                ctx.xmm(1, 0),
                ctx.register(0),
            ),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0xffff_ffff_ffff),
        ctx.register(0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency11() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0xb7ff),
        ctx.or(
            ctx.xor(
                ctx.register(1),
                ctx.xmm(1, 0),
            ),
            ctx.and(
                ctx.or(
                    ctx.xmm(1, 3),
                    ctx.constant(0x5ffffffff00),
                ),
                ctx.or(
                    ctx.constant(0x5ffffffff7800),
                    ctx.register(1),
                ),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0x5ffffffffff),
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency11() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xb7ff),
        ctx.and(
            ctx.xmm(1, 0),
            ctx.or(
                ctx.or(
                    ctx.xmm(2, 0),
                    ctx.constant(0x5ffffffff00),
                ),
                ctx.register(1),
            ),
        ),
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_and_consistency12() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0x40005ffffffffff),
        ctx.xmm(1, 0),
    );
    let eq1 = ctx.xmm(1, 0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_consistency12() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.constant(0x500000000007fff),
        ctx.and(
            ctx.xmm(1, 0),
            ctx.or(
                ctx.and(
                    ctx.xmm(1, 3),
                    ctx.constant(0x5ffffffff00),
                ),
                ctx.register(0),
            ),
        )
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_or_consistency13() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.xmm(1, 0),
        ctx.or(
            ctx.add(
                ctx.xmm(1, 0),
                ctx.constant(0x8ff00000000),
            ),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.xmm(1, 0);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.xmm(1, 0),
        ctx.or(
            ctx.xor(
                ctx.xmm(1, 0),
                ctx.xmm(1, 1),
            ),
            ctx.constant(0x600000000000000),
        ),
    );
    let eq1 = ctx.xor(
        ctx.xmm(1, 1),
        ctx.constant(0x600000000000000),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency13() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0x50000000000b7ff),
        ctx.and(
            ctx.xmm(1, 0),
            ctx.or(
                ctx.add(
                    ctx.register(0),
                    ctx.register(0),
                ),
                ctx.modulo(
                    ctx.register(0),
                    ctx.constant(0xff0000),
                ),
            ),
        )
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_and_consistency14() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.register(0),
        ctx.and(
            ctx.gt(
                ctx.xmm(1, 0),
                ctx.constant(0),
            ),
            ctx.gt(
                ctx.xmm(1, 4),
                ctx.constant(0),
            ),
        )
    );
    check_simplification_consistency(ctx, op1);
}

#[test]
fn simplify_gt3() {
    let ctx = &OperandContext::new();
    // x - y + z > x + z => (x + z) - y > x + z => y > x + z
    let op1 = ctx.gt(
        ctx.add(
            ctx.sub(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.constant(5),
        ),
        ctx.add(
            ctx.register(0),
            ctx.constant(5),
        ),
    );
    let eq1 = ctx.gt(
        ctx.register(1),
        ctx.add(
            ctx.register(0),
            ctx.constant(5),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_sub_to_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.sub(
        ctx.mul(
            ctx.constant(0x2),
            ctx.register(0),
        ),
        ctx.mul(
            ctx.constant(0x2),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.const_0();
    assert_eq!(op1, eq1);
}

#[test]
fn masked_gt_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.sub(
            ctx.and(
                ctx.register(2),
                ctx.constant(0xffff_ffff),
            ),
            ctx.constant(2),
        ),
        ctx.and(
            ctx.register(2),
            ctx.constant(0xffff_ffff),
        ),
    );
    let eq1 = ctx.gt(
        ctx.constant(2),
        ctx.and(
            ctx.register(2),
            ctx.constant(0xffff_ffff),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_sign_extend() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.sign_extend(
            ctx.register(2),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        ctx.constant(0xffff),
    );
    let eq1 = ctx.and(
        ctx.register(2),
        ctx.constant(0xffff),
    );
    let op2 = ctx.and(
        ctx.sign_extend(
            ctx.mem16(ctx.const_0(), 0x100),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        ctx.constant(0xffff),
    );
    let eq2 = ctx.mem16(ctx.const_0(), 0x100);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn gt_masked_mem() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.mem8(ctx.register(0), 0),
                ctx.constant(0xc),
            ),
            0xff,
        ),
        ctx.mem8(ctx.register(0), 0),
    );
    let eq1 = ctx.gt(
        ctx.constant(0xc),
        ctx.mem8(ctx.register(0), 0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_masked_mem2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.mem8(ctx.register(0), 0),
                ctx.constant(0xc),
            ),
            0xffff,
        ),
        ctx.mem8(ctx.register(0), 0),
    );
    let eq1 = ctx.gt(
        ctx.constant(0xc),
        ctx.mem8(ctx.register(0), 0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_x_const() {
    // x + 5 > x + 9
    // (x + 9) - 4 > x + 9
    // 4 > x + 9 (-6 >= x >= -9)
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.add(
            ctx.register(0),
            ctx.constant(5),
        ),
        ctx.add(
            ctx.register(0),
            ctx.constant(9),
        ),
    );
    let eq1 = ctx.gt(
        ctx.constant(4),
        ctx.add(
            ctx.register(0),
            ctx.constant(9),
        ),
    );
    assert_eq!(op1, eq1);

    // x + 2 > x + 3
    // (x + 3) - 1 > x + 3
    // 1 > x + 3 (x == -3)
    let op1 = ctx.gt(
        ctx.add(
            ctx.register(0),
            ctx.constant(2),
        ),
        ctx.add(
            ctx.register(0),
            ctx.constant(3),
        ),
    );
    let eq1 = ctx.eq(
        ctx.constant(-3i64 as u64),
        ctx.register(0),
    );
    assert_eq!(op1, eq1);

    // x + 2 > 7
    // 7 - (5 - x) > 7
    // 5 - x > 7
    let op1 = ctx.gt(
        ctx.sub(
            ctx.constant(5),
            ctx.register(0),
        ),
        ctx.constant(7),
    );
    let eq1 = ctx.gt(
        ctx.add_const(
            ctx.register(0),
            2,
        ),
        ctx.constant(7),
    );
    assert_eq!(op1, eq1);

    let op1 = ctx.gt(
        ctx.sub(
            ctx.sub(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.constant(5),
        ),
        ctx.constant(0x7fff_ffff),
    );
    let eq1 = ctx.gt(
        ctx.add(
            ctx.sub(
                ctx.register(1),
                ctx.register(0),
            ),
            ctx.constant(0x8000_0004),
        ),
        ctx.constant(0x7fff_ffff),
    );
    assert_eq!(op1, eq1);
    // Op1 is better form than eq1
    assert!(
        op1.if_arithmetic_gt().unwrap().0.iter().any(|x| x.if_constant() == Some(5)),
        "Bad canonicalization: {}", op1,
    );
}

#[test]
fn eq_1bit() {
    let ctx = &OperandContext::new();
    let op1 = ctx.neq_const(
        ctx.or(
            ctx.eq(
                ctx.register(3),
                ctx.register(2),
            ),
            ctx.eq(
                ctx.register(6),
                ctx.register(1),
            ),
        ),
        0,
    );
    let eq1 = ctx.or(
        ctx.eq(
            ctx.register(3),
            ctx.register(2),
        ),
        ctx.eq(
            ctx.register(6),
            ctx.register(1),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn x_is_0_or_1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.eq_const(
            ctx.register(2),
            0,
        ),
        ctx.eq_const(
            ctx.register(2),
            1,
        ),
    );
    let eq1 = ctx.gt(
        ctx.constant(2),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt_const_left() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.sub(
            ctx.register(2),
            ctx.constant(5),
        ),
        ctx.register(2),
    );
    let eq1 = ctx.gt(
        ctx.constant(5),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
    let op1 = ctx.gt(
        ctx.sub(
            ctx.register(2),
            ctx.constant(0xc000_0000_0000_0000),
        ),
        ctx.register(2),
    );
    let eq1 = ctx.gt(
        ctx.constant(0xc000_0000_0000_0000),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_signed() {
    // sign(x - y) != overflow(x - y) => gt_signed(y, x)
    let ctx = &OperandContext::new();
    // Effectively x = r0 & ffff_ffff, y = r1 & ffff_ffff,
    // but arith is being always masked later so no masks needed here.
    let arith = ctx.sub(
        ctx.register(0),
        ctx.register(1),
    );
    let op1 = ctx.neq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.and_const(ctx.register(1), 0xffff_ffff)),
            ctx.gt_signed(arith, ctx.register(0), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let op1_xor = ctx.xor(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.and_const(ctx.register(1), 0xffff_ffff)),
            ctx.gt_signed(arith, ctx.register(0), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        ctx.register(1),
        ctx.register(0),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1_xor, eq1);
}

#[test]
fn gt_signed_nomask() {
    // like gt_signed, but does not have mask for y at one point.
    let ctx = &OperandContext::new();
    let arith = ctx.sub(
        ctx.register(0),
        ctx.register(1),
    );
    // Consider x (r0) = 8, y (r1) = 1_0000_0000
    // arith = ffff_ffff_0000_0008 = 8
    //      (ctx.gt_signed will mask lhs/rhs so `arith` is always masked to just 8)
    // 0 sgt 8 should be 0, but op1 becomes 1 due to missing mask.

    // 1 != 0 => 1
    let op1 = ctx.neq(
        // 0 == 0 => 1
        ctx.eq(
            // -- no mask here -- => 8000_0000 > 1_0000_0000 => 0
            ctx.gt_const_left(0x8000_0000, ctx.register(1)),
            // 8 sgt 8 => 0
            ctx.gt_signed(arith, ctx.register(0), MemAccessSize::Mem32),
        ),
        // 0 != 0 => 0
        ctx.neq_const(
            // 0
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let op1_xor = ctx.xor(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.register(1)),
            ctx.gt_signed(arith, ctx.register(0), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        ctx.register(1),
        ctx.register(0),
        MemAccessSize::Mem32,
    );
    assert_ne!(op1, eq1);
    assert_ne!(op1_xor, eq1);
    assert_eq!(op1, op1_xor);
}

#[test]
fn gt_signed2() {
    // sign(x - y) != overflow(x - y) => gt_signed(y, x) with a constant
    let ctx = &OperandContext::new();
    let arith = ctx.sub(
        ctx.add_const(ctx.register(0), 1),
        ctx.constant(0x50),
    );
    let op1 = ctx.neq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.constant(0x50)),
            ctx.gt_signed(arith, ctx.add_const(ctx.register(0), 1), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let op1_xor = ctx.xor(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.constant(0x50)),
            ctx.gt_signed(arith, ctx.add_const(ctx.register(0), 1), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        ctx.constant(0x50),
        ctx.add_const(ctx.register(0), 1),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1_xor, eq1);
}

#[test]
fn gt_signed3() {
    // Equivalent expressions
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.constant(0x8000_0050),
        ctx.and_const(
            ctx.sub(
                ctx.constant(0x4e),
                ctx.register(0),
            ),
            0xffff_ffff,
        ),
    );
    let eq1 = ctx.gt(
        ctx.constant(0x8000_0050),
        ctx.and_const(
            ctx.add(
                ctx.constant(0x8000_0001),
                ctx.register(0),
            ),
            0xffff_ffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_signed4() {
    // sign(x - y) != overflow(x - y) => gt_signed(y, x),
    //      with y == constant and x == (var - constant)
    let ctx = &OperandContext::new();
    let arith = ctx.sub(
        ctx.sub_const(ctx.register(0), 1),
        ctx.constant(0x50),
    );
    let op1 = ctx.neq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.constant(0x50)),
            ctx.gt_signed(arith, ctx.sub_const(ctx.register(0), 1), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let op1_xor = ctx.xor(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.constant(0x50)),
            ctx.gt_signed(arith, ctx.sub_const(ctx.register(0), 1), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        ctx.constant(0x50),
        ctx.sub_const(ctx.register(0), 1),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1_xor, eq1);
}

#[test]
fn gt_signed5() {
    // sign(x - y) == overflow(x - y) => gt_signed(y, x) == 0
    let ctx = &OperandContext::new();
    let arith = ctx.sub(
        ctx.register(0),
        ctx.register(1),
    );
    let op1 = ctx.eq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.and_const(ctx.register(1), 0xffff_ffff)),
            ctx.gt_signed(arith, ctx.register(0), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.eq_const(
        ctx.gt_signed(
            ctx.register(1),
            ctx.register(0),
            MemAccessSize::Mem32,
        ),
        0,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_signed5_nomask() {
    // Can't simplify due to missing mask.
    // Effectively same case as gt_signed_nomask, just == 0 added to op1 / eq1,
    // see that for more details.
    let ctx = &OperandContext::new();
    let arith = ctx.sub(
        ctx.register(0),
        ctx.register(1),
    );
    let op1 = ctx.eq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.register(1)),
            ctx.gt_signed(arith, ctx.register(0), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.eq_const(
        ctx.gt_signed(
            ctx.register(1),
            ctx.register(0),
            MemAccessSize::Mem32,
        ),
        0,
    );
    assert_ne!(op1, eq1);
}

#[test]
fn gt_signed6() {
    // sign(x - y) == overflow(x - y) => gt_signed(y, x) == 0 => gt_signed(x, y - 1)
    let ctx = &OperandContext::new();
    let arith = ctx.sub(
        ctx.add_const(ctx.register(0), 1),
        ctx.constant(0x50),
    );
    let op1 = ctx.eq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, ctx.constant(0x50)),
            ctx.gt_signed(arith, ctx.add_const(ctx.register(0), 1), MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        ctx.add_const(ctx.register(0), 1),
        ctx.constant(0x4f),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_signed7() {
    // sign(x - y) != overflow(x - y) => gt_signed(y, x),
    //      with masked y
    let ctx = &OperandContext::new();
    let x = ctx.mem32(ctx.register(0), 0);
    let y = ctx.and_const(
        ctx.register(1),
        0xffff_ffff,
    );
    let arith = ctx.sub(x, y);
    let op1 = ctx.neq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, y),
            ctx.gt_signed(arith, x, MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let op1_xor = ctx.xor(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, y),
            ctx.gt_signed(arith, x, MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        y,
        x,
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1_xor, eq1);
}

#[test]
fn gt_signed8() {
    // sign(x - y) != overflow(x - y) => gt_signed(y, x),
    //      with masked x
    let ctx = &OperandContext::new();
    let x = ctx.and_const(
        ctx.register(1),
        0xffff_ffff,
    );
    let y = ctx.mem32(ctx.register(0), 0);
    let arith = ctx.sub(x, y);
    let op1 = ctx.neq(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, y),
            ctx.gt_signed(arith, x, MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let op1_xor = ctx.xor(
        ctx.eq(
            ctx.gt_const_left(0x8000_0000, y),
            ctx.gt_signed(arith, x, MemAccessSize::Mem32),
        ),
        ctx.neq_const(
            ctx.and_const(
                arith,
                0x8000_0000,
            ),
            0,
        ),
    );
    let eq1 = ctx.gt_signed(
        y,
        x,
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1_xor, eq1);
}

#[test]
fn merge_mem_or() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.mem32(ctx.const_0(), 0x1230),
            0xff_ffff,
        ),
        ctx.lsh_const(
            ctx.mem8(ctx.const_0(), 0x1233),
            0x18,
        ),
    );
    let eq1 = ctx.mem32(ctx.const_0(), 0x1230);
    assert_eq!(op1, eq1);
}

#[test]
fn gt_neq() {
    let ctx = &OperandContext::new();
    // not(20 > x) => x > 1f
    let op1 = ctx.eq_const(
        ctx.gt_const_left(
            0x20,
            ctx.register(1),
        ),
        0,
    );
    let eq1 = ctx.gt_const(
        ctx.register(1),
        0x1f,
    );
    let op2 = ctx.eq_const(
        ctx.gt_const(
            ctx.register(1),
            0x20,
        ),
        0,
    );
    let eq2 = ctx.gt_const_left(
        0x21,
        ctx.register(1),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn lsh_or_lsh_rsh() {
    let ctx = &OperandContext::new();
    // ((Mem8 << 8) | (Mem16 << 10)) >> 8 to Mem8 | (Mem16 << 8)
    // ((x << 8) | (y << 10)) >> 8 in general isn't so simple as the
    // lsh may be useful for cutting out high bits
    let op1 = ctx.rsh_const(
        ctx.or(
            ctx.or(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(1), 0),
                    8,
                ),
                ctx.lsh_const(
                    ctx.mem16(ctx.register(2), 0),
                    0x10,
                ),
            ),
            ctx.lsh_const(
                ctx.mem8(ctx.register(3), 0),
                0x18,
            ),
        ),
        8,
    );
    let eq1 = ctx.or(
        ctx.or(
            ctx.mem8(ctx.register(1), 0),
            ctx.lsh_const(
                ctx.mem16(ctx.register(2), 0),
                0x8,
            ),
        ),
        ctx.lsh_const(
            ctx.mem8(ctx.register(3), 0),
            0x10,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn remove_and_from_gt() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const_left(
        60,
        ctx.and_const(
            ctx.sub_const(
                ctx.mem16(ctx.register(6), 0),
                8,
            ),
            0xffff_ffff,
        ),
    );
    let eq1 = ctx.gt_const_left(
        60,
        ctx.sub_const(
            ctx.mem16(ctx.register(6), 0),
            8,
        ),
    );
    let op2 = ctx.gt_const_left(
        600,
        ctx.and_const(
            ctx.sub_const(
                ctx.mem32(ctx.register(6), 0),
                0x800000,
            ),
            0xf_ffff_ffff,
        ),
    );
    let eq2 = ctx.gt_const_left(
        600,
        ctx.sub_const(
            ctx.mem32(ctx.register(6), 0),
            0x800000,
        ),
    );
    let op3 = ctx.gt_const_left(
        600,
        ctx.and_const(
            ctx.sub_const(
                ctx.mem32(ctx.register(6), 0),
                0x800000,
            ),
            0xffff_ffff,
        ),
    );
    let eq3 = ctx.gt_const_left(
        600,
        ctx.sub_const(
            ctx.mem32(ctx.register(6), 0),
            0x800000,
        ),
    );
    // For u32 = 0, op4 => 0xd000_0000 > 0xc000_0000,
    // but unmasked ne4 => 0xd000_0000 > 0xffff_ffff_c000_0000
    let op4 = ctx.gt_const_left(
        0xd000_0000,
        ctx.and_const(
            ctx.sub_const(
                ctx.mem32(ctx.register(6), 0),
                0x4000_0000,
            ),
            0xffff_ffff,
        ),
    );
    let ne4 = ctx.gt_const_left(
        0xd000_0000,
        ctx.sub_const(
            ctx.mem32(ctx.register(6), 0),
            0x4000_0000,
        ),
    );
    let op5 = ctx.gt(
        ctx.mem16(ctx.register(4), 0),
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(6), 0),
                ctx.mem16(ctx.register(1), 0),
            ),
            0xffff_ffff,
        ),
    );
    let eq5 = ctx.gt(
        ctx.mem16(ctx.register(4), 0),
        ctx.sub(
            ctx.mem16(ctx.register(6), 0),
            ctx.mem16(ctx.register(1), 0),
        ),
    );
    let op6 = ctx.gt(
        ctx.mem16(ctx.register(4), 0),
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(6), 0),
                ctx.mem32(ctx.register(1), 0),
            ),
            0xffff_ffff,
        ),
    );
    let ne6 = ctx.gt(
        ctx.mem16(ctx.register(4), 0),
        ctx.sub(
            ctx.mem16(ctx.register(6), 0),
            ctx.mem32(ctx.register(1), 0),
        ),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_ne!(op4, ne4);
    assert_eq!(op5, eq5);
    assert_ne!(op6, ne6);
}

#[test]
fn gt_or_eq_sub_const() {
    // (6 > (x - 3)) | (x == 9) => 7 > (x - 3)
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.gt_const_left(
            6,
            ctx.sub_const(
                ctx.register(5),
                3,
            ),
        ),
        ctx.eq_const(
            ctx.register(5),
            9,
        ),
    );
    let eq1 = ctx.gt_const_left(
        7,
        ctx.sub_const(
            ctx.register(5),
            3,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn lsh_to_mul_when_sensible() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh_const(
        ctx.add_const(
            ctx.mul_const(
                ctx.mem32(ctx.register(0), 0),
                0x3,
            ),
            0x1234,
        ),
        4,
    );
    let eq1 = ctx.add_const(
        ctx.mul_const(
            ctx.mem32(ctx.register(0), 0),
            0x30,
        ),
        0x12340,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_sub_to_masked_add() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.mul_const(
                ctx.mem16(ctx.register(0), 0),
                0x2,
            ),
            0xf000,
        ),
        0xffff,
    );
    let eq1 = ctx.and_const(
        ctx.add_const(
            ctx.mul_const(
                ctx.mem16(ctx.register(0), 0),
                0x2,
            ),
            0x1000,
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn mul_large_const_to_lsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul_const(
        ctx.register(1),
        0x2_0000,
    );
    let eq1 = ctx.lsh_const(
        ctx.register(1),
        0x11,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn cannot_split_rsh_with_add() {
    let ctx = &OperandContext::new();
    // ecx = 1 => op1 == 1, ne1 == 0
    let op1 = ctx.rsh_const(
        ctx.add_const(
            ctx.register(1),
            0xffff_ffff,
        ),
        0x20,
    );
    let ne1 = ctx.rsh_const(
        ctx.register(1),
        0x20,
    );
    assert_ne!(op1, ne1);
}

#[test]
fn masked_sub_to_masked_add2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.rsh_const(
            ctx.sub_const(
                ctx.mul_const(
                    ctx.mem32(ctx.register(0), 0),
                    0x2,
                ),
                0xf000_0000,
            ),
            0x10,
        ),
        0xffff,
    );
    let eq1 = ctx.and_const(
        ctx.rsh_const(
            ctx.add_const(
                ctx.mul_const(
                    ctx.mem32(ctx.register(0), 0),
                    0x2,
                ),
                0x1000_0000,
            ),
            0x10,
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_low_word_of_operation() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.add_const(
            ctx.or(
                ctx.rsh_const(
                    ctx.mem32(ctx.const_0(), 0x4242),
                    0xb,
                ),
                ctx.and_const(
                    ctx.lsh_const(
                        ctx.mem16(ctx.const_0(), 0x4242),
                        0x15,
                    ),
                    0xffe0_0000,
                ),
            ),
            0xd124_ec43,
        ),
        0xffff,
    );
    let eq1 = ctx.and_const(
        ctx.add_const(
            ctx.rsh_const(
                ctx.mem32(ctx.const_0(), 0x4242),
                0xb,
            ),
            0xec43,
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn lsh_rsh() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.lsh_const(
            ctx.add_const(
                ctx.mem32(ctx.const_0(), 0x4242),
                0x1234,
            ),
            0x10,
        ),
        0x10,
    );
    let eq1 = ctx.add_const(
        ctx.mem32(ctx.const_0(), 0x4242),
        0x1234,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn mul_lsh() {
    // Check that simplifying this won't infloop
    let ctx = &OperandContext::new();
    for i in 0..64 {
        ctx.mul_const(
            ctx.mul(
                ctx.register(0),
                ctx.register(1),
            ),
            1 << i,
        );
    }
}

#[test]
fn invalid_lsh_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh(
        ctx.mem8(ctx.register(1), 0),
        ctx.constant(0xffff_ffff_ffff_ff83),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.xor(
                ctx.register(1),
                ctx.register(2),
            ),
            0xffff,
        ),
        ctx.and_const(
            ctx.register(1),
            0xffff,
        ),
    );
    let eq1 = ctx.and_const(ctx.register(2), 0xffff);
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors2() {
    let ctx = &OperandContext::new();
    // ((x ^ Mem16[y]) & ffff) ^ ((z ^ Mem16[y]) & 1ffff)
    // => (x & ffff) ^ (z & 1ffff)
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.xor(
                ctx.register(1),
                ctx.mem16(ctx.register(2), 0),
            ),
            0xffff,
        ),
        ctx.and_const(
            ctx.xor(
                ctx.register(3),
                ctx.mem16(ctx.register(2), 0),
            ),
            0x1_ffff,
        ),
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.register(1),
            0xffff,
        ),
        ctx.and_const(
            ctx.register(3),
            0x1_ffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors3() {
    let ctx = &OperandContext::new();
    // ((x & ff) ^ (y & ffff) ^ Mem16[z])
    // => ((x & ff) ^ y ^ Mem16[z]) & ffff
    // (Canonicalize to keep the and mask outside if reasonable)
    let op1 = ctx.xor(
        ctx.xor(
            ctx.and_const(
                ctx.register(0),
                0xff,
            ),
            ctx.and_const(
                ctx.register(1),
                0xffff,
            ),
        ),
        ctx.mem16(ctx.register(2), 0),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.xor(
                ctx.and_const(
                    ctx.register(0),
                    0xff,
                ),
                ctx.register(1),
            ),
            ctx.mem16(ctx.register(2), 0),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn medium_sized_and_simplify() {
    let ctx = &OperandContext::new();
    let key1 = ctx.rsh_const(
        ctx.sub_const(
            ctx.or(
                ctx.rsh_const(
                    ctx.mem32(ctx.register(0), 0),
                    0xb,
                ),
                ctx.lsh_const(
                    ctx.mem32(ctx.register(0), 0),
                    0x15,
                ),
            ),
            0x20f1359f,
        ),
        0x10,
    );
    let key2 = ctx.sub(
        ctx.mul_const(
            ctx.mem16(ctx.register(1), 0),
            2,
        ),
        ctx.lsh_const(
            ctx.mem8(ctx.register(2), 0),
            9,
        ),
    );
    let key3 = ctx.xor(
        ctx.mem16(ctx.register(3), 0),
        ctx.mem16(ctx.register(4), 0),
    );
    let joined = ctx.and_const(
        ctx.xor(
            key1,
            ctx.xor(
                key2,
                key3,
            ),
        ),
        0xffff,
    );
    let key4 = ctx.and_const(
        ctx.sub(
            ctx.and_const(
                ctx.xor_const(
                    ctx.xor(
                        ctx.mul_const(
                            ctx.mem16(ctx.register(5), 0),
                            2,
                        ),
                        ctx.sub_const(
                            ctx.and_const(
                                ctx.rsh_const(
                                    ctx.mem32(ctx.register(6), 0),
                                    0xa,
                                ),
                                0x3ffffe,
                            ),
                            0x6b3e,
                        ),
                    ),
                    0x6700,
                ),
                0x1fffe,
            ),
            ctx.lsh_const(
                ctx.mem16(ctx.register(7), 0),
                9,
            ),
        ),
        0xfffffffffffe,
    );
    let joined2 = ctx.xor(
        joined,
        key4,
    );
    let joined3 = ctx.xor_const(joined2, 0x8335);
    let masked = ctx.and_const(joined3, 0xffff);
    let alt_masked = ctx.xor(
        ctx.and_const(
            joined,
            0xffff,
        ),
        ctx.xor_const(
            ctx.and_const(
                key4,
                0xffff,
            ),
            0x8335,
        ),
    );

    assert_eq!(masked, alt_masked);
}

#[test]
fn masked_xors4() {
    let ctx = &OperandContext::new();
    // (((x << 2) & fffe) ^ (y & ffff) ^ Mem16[z])
    // => ((x << 2) ^ y ^ Mem16[z]) & ffff
    let op1 = ctx.xor(
        ctx.xor(
            ctx.and_const(
                ctx.lsh_const(
                    ctx.register(0),
                    2,
                ),
                0xfffe,
            ),
            ctx.and_const(
                ctx.register(1),
                0xffff,
            ),
        ),
        ctx.mem16(ctx.register(2), 0),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.xor(
                ctx.lsh_const(
                    ctx.register(0),
                    2,
                ),
                ctx.register(1),
            ),
            ctx.mem16(ctx.register(2), 0),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors5() {
    let ctx = &OperandContext::new();
    // ((x ^ z) & ff) ^ ((x ^ y) & ff00) ^ (w & ffff)
    // => (z & ff) ^ (y & ff00) ^ ((x ^ w) & ffff)
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.xor(
                ctx.register(0),
                ctx.register(2),
            ),
            0xff,
        ),
        ctx.xor(
            ctx.and_const(
                ctx.xor(
                    ctx.register(0),
                    ctx.register(1),
                ),
                0xff00,
            ),
            ctx.and_const(
                ctx.register(3),
                0xffff,
            ),
        ),
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.register(2),
            0xff,
        ),
        ctx.xor(
            ctx.and_const(
                ctx.register(1),
                0xff00,
            ),
            ctx.and_const(
                ctx.xor(
                    ctx.register(0),
                    ctx.register(3),
                ),
                0xffff,
            ),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors6() {
    let ctx = &OperandContext::new();
    // (Mem8[x] - 34) & ff) ^ (Mem16[x] - 1234 & ff00)
    // => (Mem16[x] - 1234) & ffff
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.mem8(ctx.register(0), 0),
                ctx.constant(0x34),
            ),
            0xff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(0), 0),
                ctx.constant(0x1234),
            ),
            0xff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.mem16(ctx.register(0), 0),
            ctx.constant(0x1234),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors7() {
    let ctx = &OperandContext::new();
    // ((Mem8[x] * 2) - 34) & ff) ^ ((Mem16[x] * 2) - 1234 & ff00)
    // => ((Mem16[x] * 2) - 1234) & ffff
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.mul_const(
                    ctx.mem8(ctx.register(0), 0),
                    2,
                ),
                ctx.constant(0x34),
            ),
            0xff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.mul_const(
                    ctx.mem16(ctx.register(0), 0),
                    2,
                ),
                ctx.constant(0x1234),
            ),
            0xff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.mul_const(
                ctx.mem16(ctx.register(0), 0),
                2,
            ),
            ctx.constant(0x1234),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors8() {
    let ctx = &OperandContext::new();
    // ((Mem8[x] & 80) - 34) & ff) ^ ((Mem16[x] & 8080) - 1234 & ff00)
    // => ((Mem16[x] & 8080) - 1234) & ffff
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.and_const(
                    ctx.mem8(ctx.register(0), 0),
                    0x80,
                ),
                ctx.constant(0x34),
            ),
            0xff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.and_const(
                    ctx.mem16(ctx.register(0), 0),
                    0x8080,
                ),
                ctx.constant(0x1234),
            ),
            0xff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.and_const(
                ctx.mem16(ctx.register(0), 0),
                0x8080,
            ),
            ctx.constant(0x1234),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn lsh_rsh2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.lsh_const(
            ctx.mem32(ctx.const_0(), 0x34),
            0x4,
        ),
        0x10,
    );
    let eq1 = ctx.rsh_const(
        ctx.mem32(ctx.const_0(), 0x34),
        0xc,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xors9() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.mem32(ctx.register(1), 0),
            0xffff,
        ),
        ctx.mem8(ctx.register(3), 0),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.mem32(ctx.register(1), 0),
            ctx.mem8(ctx.register(3), 0),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
    let op2 = ctx.xor(
        ctx.and_const(
            ctx.xor(
                ctx.mem32(ctx.register(1), 0),
                ctx.constant(0x1111),
            ),
            0xffff,
        ),
        ctx.constant(0x4f4f),
    );
    let eq2 = ctx.and_const(
        ctx.xor(
            ctx.mem32(ctx.register(1), 0),
            ctx.constant(0x5e5e),
        ),
        0xffff,
    );
    assert_eq!(op2, eq2);
}

#[test]
fn masked_xors10() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub_const(
                ctx.mem32(ctx.register(1), 0),
                0x1234,
            ),
            0xffff,
        ),
        ctx.constant(0xe4e4),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.sub_const(
                ctx.mem32(ctx.register(1), 0),
                0x1234,
            ),
            ctx.constant(0xe4e4),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_and_ne() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.gt_const(
            ctx.and_const(
                ctx.sub_const(
                    ctx.register(1),
                    0xd,
                ),
                0xffff_ffff,
            ),
            0xc,
        ),
        ctx.neq_const(
            ctx.register(1),
            0x1a,
        ),
    );
    let eq1 = ctx.gt_const(
        ctx.and_const(
            ctx.sub_const(
                ctx.register(1),
                0xd,
            ),
            0xffff_ffff,
        ),
        0xd,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn eq_simplify_const_right() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.register(6),
        6,
    );
    let op2 = ctx.eq_const(
        ctx.eq_const(
            ctx.register(6),
            6,
        ),
        0,
    );
    let op3 = ctx.eq_const(
        ctx.and_const(
            ctx.register(6),
            0xffff,
        ),
        6,
    );
    assert_eq!(op1.if_arithmetic_eq().unwrap().1, ctx.constant(6));
    assert_eq!(op2.if_arithmetic_eq().unwrap().1, ctx.constant(0));
    assert_eq!(op2.if_arithmetic_eq().unwrap().0.if_arithmetic_eq().unwrap().1, ctx.constant(6));
    assert_eq!(op3.if_arithmetic_eq().unwrap().1, ctx.constant(6));
}

#[test]
fn simplify_eq3() {
    let ctx = &OperandContext::new();
    let sub = ctx.sub(
        ctx.mem32(ctx.const_0(), 0x29a),
        ctx.mem32(ctx.register(2), 0),
    );
    let op1 = ctx.eq_const(
        sub,
        0x4d2,
    );
    assert_eq!(op1.if_arithmetic_eq().unwrap().0, sub);
    assert_eq!(op1.if_arithmetic_eq().unwrap().1, ctx.constant(0x4d2));
}

#[test]
fn simplify_eq4() {
    let ctx = &OperandContext::new();
    let and = ctx.and_const(
        ctx.sub(
            ctx.mem32(ctx.const_0(), 0x29a),
            ctx.mem32(ctx.register(2), 0),
        ),
        0xffff_ffff,
    );
    let op1 = ctx.eq_const(
        and,
        0x4d2,
    );
    assert_eq!(op1.if_arithmetic_eq().unwrap().0, and);
    assert_eq!(op1.if_arithmetic_eq().unwrap().1, ctx.constant(0x4d2));
}

#[test]
fn simplify_or_gt_not() {
    let ctx = &OperandContext::new();
    // Can't simplify (2 > (x & 0xffff_ffff)) | x == 2 to (3 > (x & 0xffff_ffff))
    // as that would change result x == 0x1_0000_0002
    let op1 = ctx.or(
        ctx.gt_const_left(
            2,
            ctx.and_const(
                ctx.register(0),
                0xffff_ffff,
            ),
        ),
        ctx.eq_const(
            ctx.register(0),
            2,
        ),
    );
    let eq1 = ctx.gt_const_left(
        3,
        ctx.and_const(
            ctx.register(0),
            0xffff_ffff,
        ),
    );
    assert_ne!(op1, eq1);
}

#[test]
fn simplify_eq5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.sub_const(
                ctx.register(1),
                2,
            ),
            0xffff_ffff,
        ),
        0,
    );
    let ne1 = ctx.eq_const(
        ctx.register(1),
        2,
    );
    let eq1 = ctx.eq_const(
        ctx.and_const(
            ctx.register(1),
            0xffff_ffff,
        ),
        2,
    );
    assert_ne!(op1, ne1);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq6() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.add_const(
                ctx.mem32(ctx.const_0(), 0x100),
                0x1,
            ),
            0xffff_ffff,
        ),
        0,
    );
    let eq1 = ctx.eq_const(
        ctx.mem32(ctx.const_0(), 0x100),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq7() {
    let ctx = &OperandContext::new();
    let cmp = ctx.and_const(
        ctx.sub_const_left(
            0,
            ctx.gt_const(
                ctx.mem32(ctx.const_0(), 0x100),
                3,
            ),
        ),
        0xffff_ffff,
    );
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.add_const(
                cmp,
                1,
            ),
            0xffff_ffff,
        ),
        0,
    );
    let eq1 = ctx.gt_const(
        ctx.mem32(ctx.const_0(), 0x100),
        3,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_mem_merge() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.xor_const(
            ctx.mem32(
                ctx.mem64(ctx.const_0(), 0x80),
                0x18,
            ),
            0x45451212,
        ),
        ctx.xor_const(
            ctx.xor(
                ctx.lsh_const(
                    ctx.mem32(
                        ctx.mem64(ctx.const_0(), 0x80),
                        0x1c,
                    ),
                    0x20,
                ),
                ctx.lsh_const(
                    ctx.mem32(
                        ctx.mem64(ctx.const_0(), 0x80),
                        0x18,
                    ),
                    0x20,
                ),
            ),
            0x00451200,
        ),
    );
    let eq1 = ctx.xor_const(
        ctx.xor(
            ctx.lsh_const(
                ctx.mem32(
                    ctx.mem64(ctx.const_0(), 0x80),
                    0x18,
                ),
                0x20,
            ),
            ctx.mem64(
                ctx.mem64(ctx.const_0(), 0x80),
                0x18,
            ),
        ),
        0x45000012,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mul_rsh() {
    // Simplify (x * 422) >> 4 to ((x * 221) >> 3) & 0fff_ffff_ffff_ffff
    // etc
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.mul_const(
            ctx.register(0),
            0x554
        ),
        5,
    );
    let eq1 = ctx.and_const(
        ctx.rsh_const(
            ctx.mul_const(
                ctx.register(0),
                0x155
            ),
            3,
        ),
        0x07ff_ffff_ffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn merge_or_complex() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.sub_const(
                ctx.and_const(
                    ctx.rsh_const(
                        ctx.mul_const(
                            ctx.mem16(ctx.register(0), 0),
                            0x21dd,
                        ),
                        2,
                    ),
                    0xffe,
                ),
                0x67d,
            ),
            0xf00,
        ),
        ctx.and_const(
            ctx.sub_const(
                ctx.and_const(
                    ctx.rsh_const(
                        ctx.mul_const(
                            ctx.mem16(ctx.register(0), 0),
                            0x21dd,
                        ),
                        2,
                    ),
                    0xffe,
                ),
                0x67d,
            ),
            0xff,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.rsh_const(
                    ctx.mul_const(
                        ctx.mem16(ctx.register(0), 0),
                        0x21dd,
                    ),
                    2,
                ),
                0xffe,
            ),
            0x67d,
        ),
        0xfff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn merge_xor_complex() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.add_const(
                ctx.or(
                    ctx.rsh_const(
                        ctx.mem32(ctx.register(0), 0),
                        0xb,
                    ),
                    ctx.lsh_const(
                        ctx.mem8(ctx.register(0), 0),
                        0x15,
                    ),
                ),
                0x643e18,
            ),
            0xff_ffff,
        ),
        ctx.and_const(
            ctx.add_const(
                ctx.or(
                    ctx.rsh_const(
                        ctx.mem32(ctx.register(0), 0),
                        0xb,
                    ),
                    ctx.lsh_const(
                        ctx.mem16(ctx.register(0), 0),
                        0x15,
                    ),
                ),
                0x48643e18,
            ),
            0xff00_0000,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.add_const(
            ctx.or(
                ctx.rsh_const(
                    ctx.mem32(ctx.register(0), 0),
                    0xb,
                ),
                ctx.lsh_const(
                    ctx.mem16(ctx.register(0), 0),
                    0x15,
                ),
            ),
            0x48643e18,
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn merge_xor_complex2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub_const(
                ctx.mem16(ctx.register(0), 0),
                0x6400,
            ),
            0xff00,
        ),
        ctx.and_const(
            ctx.mem8(ctx.register(0), 0),
            0xff,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.mem16(ctx.register(0), 0),
            0x6400,
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_sub_mask() {
    let ctx = &OperandContext::new();
    // If the x - y subtraction overflows, it'll be greater than x
    // whether it gets masked or not (As long as the mask >= x)
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub_const(
                ctx.and_const(
                    ctx.register(0),
                    0xffff,
                ),
                0x50,
            ),
            0xffff_ffff,
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );
    let eq1 = ctx.gt(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                0xffff,
            ),
            0x50,
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );

    let op2 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.and_const(
                    ctx.register(0),
                    0xffff,
                ),
                ctx.mem32(ctx.register(2), 0),
            ),
            0xffff_ffff,
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );
    let eq2 = ctx.gt(
        ctx.sub(
            ctx.and_const(
                ctx.register(0),
                0xffff,
            ),
            ctx.mem32(ctx.register(2), 0),
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );

    // This can't simplified like above since (3 - 1_0000_0000) & ffff_ffff == 3
    let op3 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.and_const(
                    ctx.register(0),
                    0xffff,
                ),
                ctx.mem64(ctx.register(2), 0),
            ),
            0xffff_ffff,
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );
    let ne3 = ctx.gt(
        ctx.sub(
            ctx.and_const(
                ctx.register(0),
                0xffff,
            ),
            ctx.mem64(ctx.register(2), 0),
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );
    let eq3 = ctx.gt(
        ctx.sub(
            ctx.and_const(
                ctx.register(0),
                0xffff,
            ),
            ctx.mem32(ctx.register(2), 0),
        ),
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
    );

    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_ne!(op3, ne3);
    assert_eq!(op3, eq3);
}

#[test]
fn signed_gt_normalize() {
    let ctx = &OperandContext::new();
    // Mem32 sgt 50 in two different ways
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.constant(0x50),
                ctx.mem32(ctx.register(0), 0),
            ),
            0xffff_ffff,
        ),
        ctx.constant(0x8000_0050),
    );
    let eq1 = ctx.gt(
        ctx.constant(0x7fff_ffaf),
        ctx.sub(
            ctx.mem32(ctx.register(0), 0),
            ctx.constant(0x51),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn merge_signed_gt_ne() {
    let ctx = &OperandContext::new();
    // (Mem32 sgt 50) & Mem32 != 51
    let op1 = ctx.and(
        ctx.gt(
            ctx.and_const(
                ctx.sub(
                    ctx.constant(0x50),
                    ctx.mem32(ctx.register(0), 0),
                ),
                0xffff_ffff,
            ),
            ctx.constant(0x8000_0050),
        ),
        ctx.neq_const(
            ctx.mem32(ctx.register(0), 0),
            0x51,
        ),
    );
    // Mem32 sgt 51
    let eq1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.constant(0x51),
                ctx.mem32(ctx.register(0), 0),
            ),
            0xffff_ffff,
        ),
        ctx.constant(0x8000_0051),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_sub_mask2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.gt_const(
            ctx.and_const(
                ctx.custom(0),
                0xffff,
            ),
            0x50,
        ),
        ctx.neq_const(
            ctx.and_const(
                ctx.custom(0),
                0xffff,
            ),
            0x51,
        ),
    );
    let eq1 = ctx.gt_const(
        ctx.and_const(
            ctx.custom(0),
            0xffff,
        ),
        0x51,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn signed_gt_normalize2() {
    let ctx = &OperandContext::new();
    // Mem32 sgt 50 in two different ways
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.constant(0x50),
                ctx.mem32(ctx.register(0), 0),
            ),
            0xffff_ffff,
        ),
        ctx.constant(0x8000_0050),
    );
    let eq1 = ctx.gt(
        ctx.and_const(
            ctx.add_const(
                ctx.mem32(ctx.register(0), 0),
                0x8000_0000,
            ),
            0xffff_ffff,
        ),
        ctx.and_const(
            ctx.add_const(
                ctx.constant(0x50),
                0x8000_0000,
            ),
            0xffff_ffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_normalize3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const(
        ctx.and_const(
            ctx.sub_const(
                ctx.register(0),
                1,
            ),
            0xffff_ffff,
        ),
        0,
    );
    let eq1 = ctx.neq_const(
        ctx.and_const(
            ctx.register(0),
            0xffff_ffff,
        ),
        1,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_normalize4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const(
        ctx.and_const(
            ctx.sub_const(
                ctx.register(0),
                0x51,
            ),
            0xffff_ffff,
        ),
        0x7fff_ffaf,
    );
    let eq1 = ctx.gt_const_left(
        0x8000_0050,
        ctx.and_const(
            ctx.add_const(
                ctx.register(0),
                0x7fff_ffff,
            ),
            0xffff_ffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_merge_shifted_mem() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.lsh_const(
            ctx.mem8(
                ctx.register(1),
                0xd,
            ),
            0x18
        ),
        ctx.lsh_const(
            ctx.mem8(
                ctx.register(1),
                0xc,
            ),
            0x10
        ),
    );
    let eq1 = ctx.lsh_const(
        ctx.mem16(
            ctx.register(1),
            0xc,
        ),
        0x10,
    );
    let op2 = ctx.or(
        ctx.lsh_const(
            ctx.mem8(ctx.const_0(), 0xd),
            0x18
        ),
        ctx.lsh_const(
            ctx.mem8(ctx.const_0(), 0xc),
            0x10
        ),
    );
    let eq2 = ctx.lsh_const(
        ctx.mem16(ctx.const_0(), 0xc),
        0x10,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_or_merge_shifted_mem2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.lsh_const(
            ctx.mem8(
                ctx.register(1),
                0xf,
            ),
            0x18
        ),
        ctx.and_const(
            ctx.mem32(
                ctx.register(1),
                0xc,
            ),
            0xf0_fff0,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.mem32(
            ctx.register(1),
            0xc,
        ),
        0xfff0_fff0,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_merge_shifted_mem3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.lsh_const(
            ctx.mem8(
                ctx.register(1),
                0xf,
            ),
            0x18
        ),
        ctx.and_const(
            ctx.or(
                ctx.register(0),
                ctx.mem32(
                    ctx.register(1),
                    0xc,
                ),
            ),
            0xff_ffff,
        ),
    );
    let eq1 = ctx.or(
        ctx.mem32(
            ctx.register(1),
            0xc,
        ),
        ctx.and_const(
            ctx.register(0),
            0xff_ffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_merge() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.lsh_const(
                ctx.sub(
                    ctx.mem16(ctx.register(1), 0),
                    ctx.mem16(ctx.register(2), 0),
                ),
                0x10,
            ),
            0x00ff_ffff,
        ),
        ctx.and_const(
            ctx.lsh_const(
                ctx.sub(
                    ctx.mem16(ctx.register(1), 0),
                    ctx.mem16(ctx.register(2), 0),
                ),
                0x10,
            ),
            0xff00_0000,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.lsh_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.mem16(ctx.register(2), 0),
            ),
            0x10,
        ),
        0xffff_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_merge2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.xor_const(
                    ctx.mem16(ctx.register(2), 0),
                    0x1234,
                ),
            ),
            0x00ff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.xor_const(
                    ctx.mem16(ctx.register(2), 0),
                    0x1234,
                ),
            ),
            0xff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.mem16(ctx.register(1), 0),
            ctx.xor_const(
                ctx.mem16(ctx.register(2), 0),
                0x1234,
            ),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_merge3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.xor_const(
                    ctx.sub(
                        ctx.mem16(ctx.register(2), 0),
                        ctx.mem16(ctx.register(3), 0),
                    ),
                    0x1234,
                ),
            ),
            0x00ff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.xor_const(
                    ctx.sub(
                        ctx.mem16(ctx.register(2), 0),
                        ctx.mem16(ctx.register(3), 0),
                    ),
                    0x1234,
                ),
            ),
            0xff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.mem16(ctx.register(1), 0),
            ctx.xor_const(
                ctx.sub(
                    ctx.mem16(ctx.register(2), 0),
                    ctx.mem16(ctx.register(3), 0),
                ),
                0x1234,
            ),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_merge4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.xor_const(
                    ctx.and_const(
                        ctx.mem16(ctx.register(2), 0),
                        0xfffe,
                    ),
                    0x1234,
                ),
            ),
            0x00ff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(1), 0),
                ctx.xor_const(
                    ctx.and_const(
                        ctx.mem16(ctx.register(2), 0),
                        0xfffe,
                    ),
                    0x1234,
                ),
            ),
            0xff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.mem16(ctx.register(1), 0),
            ctx.xor_const(
                ctx.and_const(
                    ctx.mem16(ctx.register(2), 0),
                    0xfffe,
                ),
                0x1234,
            ),
        ),
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_merge5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.lsh_const(
                    ctx.sub(
                        ctx.mem16(ctx.register(1), 0),
                        ctx.mem16(ctx.register(4), 0),
                    ),
                    0x10,
                ),
                ctx.register(2),
            ),
            0x00ff_ffff,
        ),
        ctx.and_const(
            ctx.sub(
                ctx.lsh_const(
                    ctx.sub(
                        ctx.mem16(ctx.register(1), 0),
                        ctx.mem16(ctx.register(4), 0),
                    ),
                    0x10,
                ),
                ctx.register(2),
            ),
            0xff00_0000,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.lsh_const(
                ctx.sub(
                    ctx.mem16(ctx.register(1), 0),
                    ctx.mem16(ctx.register(4), 0),
                ),
                0x10,
            ),
            ctx.register(2),
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn canonicalize_demorgan() {
    // prefer (x & y) == 0 over (x == 0) | (y == 0)
    // and (x | y) == 0 over (x == 0) & (y == 0)
    // when x and y are 1bit expressions
    // (No real reason why this way, other than it's less operands)
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.eq_const(
            ctx.eq(
                ctx.register(0),
                ctx.register(1),
            ),
            0,
        ),
        ctx.eq_const(
            ctx.gt(
                ctx.register(3),
                ctx.register(4),
            ),
            0,
        ),
    );
    let eq1 = ctx.eq_const(
        ctx.and(
            ctx.eq(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.gt(
                ctx.register(3),
                ctx.register(4),
            ),
        ),
        0,
    );

    let op2 = ctx.and(
        ctx.eq_const(
            ctx.eq(
                ctx.register(0),
                ctx.register(1),
            ),
            0,
        ),
        ctx.eq_const(
            ctx.gt(
                ctx.register(3),
                ctx.register(4),
            ),
            0,
        ),
    );
    let eq2 = ctx.eq_const(
        ctx.or(
            ctx.eq(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.gt(
                ctx.register(3),
                ctx.register(4),
            ),
        ),
        0,
    );
    assert!(
        eq1.if_arithmetic_eq().and_then(|x| x.0.if_arithmetic_and()).is_some(),
        "Bad canonicalization for {}", eq1,
    );
    assert_eq!(op1, eq1);
    assert!(
        eq2.if_arithmetic_eq().and_then(|x| x.0.if_arithmetic_or()).is_some(),
        "Bad canonicalization for {}", eq2,
    );
    assert_eq!(op2, eq2);
}

#[test]
fn or_const_eq_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.or(
            ctx.register(0),
            ctx.constant(0x6560),
        ),
        0,
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.eq_const(
        ctx.and_const(
            ctx.or(
                ctx.register(0),
                ctx.constant(0x6560),
            ),
            0xffff_ffff,
        ),
        0,
    );
    let eq2 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn or_const_eq_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.or(
            ctx.register(0),
            ctx.constant(0x6560),
        ),
        0x1200,
    );
    let eq1 = ctx.constant(0);
    let op2 = ctx.eq_const(
        ctx.and_const(
            ctx.or(
                ctx.register(0),
                ctx.constant(0x6560),
            ),
            0xffff_ffff,
        ),
        0x1200,
    );
    let eq2 = ctx.constant(0);
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_mul_high() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul_high(
        ctx.constant(0x5555_6666),
        ctx.mem32(ctx.register(5), 0),
    );
    assert_eq!(op1, ctx.const_0());
    let op2 = ctx.mul_high(
        ctx.constant(0x5555_6666_7777_8888),
        ctx.constant(0x5555_6666_1111_2222),
    );
    assert_eq!(op2, ctx.constant(0x1c71d27d12345d4b));
}

#[test]
fn simplify_sub_sext() {
    // sext32_64((x - y) & ffff_ffff) => x - y
    // when x relevant_bits end <= 31, and y <= 31.
    //
    // Example for y relbits end > 31:
    // 50 - 9000_0000 = 0xffff_ffff_7000_0050
    // wouldn't get sign extended after masked
    // (Currently not simplified at all)
    // Similarly for x bits end > 31
    // 9000_0000 - 50 => 8fff_ffb0 gets sign extended
    // but it wasn't just as a result of the subtraction.
    // And it cannot be converted to only sext x, as
    // 8000_0001 - 50 would become wrong then.
    let ctx = &OperandContext::new();
    let op1 = ctx.sign_extend(
        ctx.and_const(
            ctx.sub(
                ctx.mem16(ctx.register(0), 0),
                ctx.constant(0x555),
            ),
            0xffff_ffff,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    let eq1 = ctx.sub(
        ctx.mem16(ctx.register(0), 0),
        ctx.constant(0x555),
    );
    assert_eq!(op1, eq1);

    let op1 = ctx.sign_extend(
        ctx.and_const(
            ctx.sub(
                ctx.constant(0x555),
                ctx.mem32(ctx.register(0), 0),
            ),
            0xffff_ffff,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    match *op1.ty() {
        OperandType::SignExtend(val, MemAccessSize::Mem32, MemAccessSize::Mem64) => {
            assert_eq!(val, ctx.and_const(
                ctx.sub(
                    ctx.constant(0x555),
                    ctx.mem32(ctx.register(0), 0),
                ),
                0xffff_ffff,
            ));
        }
        _ => panic!("Bad simplify to {}", op1),
    }

    let op1 = ctx.sign_extend(
        ctx.and_const(
            ctx.sub(
                ctx.mem32(ctx.register(0), 0),
                ctx.constant(0x555),
            ),
            0xffff_ffff,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    match *op1.ty() {
        OperandType::SignExtend(val, MemAccessSize::Mem32, MemAccessSize::Mem64) => {
            assert_eq!(val, ctx.and_const(
                ctx.sub(
                    ctx.mem32(ctx.register(0), 0),
                    ctx.constant(0x555),
                ),
                0xffff_ffff,
            ));
        }
        _ => panic!("Bad simplify to {}", op1),
    }
}

#[test]
fn simplify_sub_sext2() {
    // sext32_64((sext_to_32(x) - y) & ffff_ffff) => sext_to_64(x) - y
    // As tested above, it isn't guaranteed to be doable for every x,
    // but if `sext_to_32(x) - y` never changes sign from neg -> pos it is fine.
    // (pos -> neg is ok)
    // (Could also be applied for `(a - b) - y` etc.)
    let ctx = &OperandContext::new();
    let op1 = ctx.sign_extend(
        ctx.and_const(
            ctx.sub(
                ctx.sign_extend(
                    ctx.mem8(ctx.register(0), 0),
                    MemAccessSize::Mem8,
                    MemAccessSize::Mem32,
                ),
                ctx.constant(0x555),
            ),
            0xffff_ffff,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    let eq1 = ctx.sub(
        ctx.sign_extend(
            ctx.mem8(ctx.register(0), 0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem64,
        ),
        ctx.constant(0x555),
    );
    assert_eq!(op1, eq1);

    // Ok, `y` is at most 0x3fff_ffff which won't be able to signed overflow
    // when subtracted from 0xffff_ff80
    let op1 = ctx.sign_extend(
        ctx.and_const(
            ctx.sub(
                ctx.sign_extend(
                    ctx.mem8(ctx.register(0), 0),
                    MemAccessSize::Mem8,
                    MemAccessSize::Mem32,
                ),
                ctx.rsh_const(
                    ctx.mem32(ctx.register(5), 0),
                    2,
                ),
            ),
            0xffff_ffff,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    let eq1 = ctx.sub(
        ctx.sign_extend(
            ctx.mem8(ctx.register(0), 0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem64,
        ),
        ctx.rsh_const(
            ctx.mem32(ctx.register(5), 0),
            2,
        ),
    );
    assert_eq!(op1, eq1);

    // Not ok, `y` is at most 0x7fff_ffff.
    let op1 = ctx.sign_extend(
        ctx.and_const(
            ctx.sub(
                ctx.sign_extend(
                    ctx.mem8(ctx.register(0), 0),
                    MemAccessSize::Mem8,
                    MemAccessSize::Mem32,
                ),
                ctx.rsh_const(
                    ctx.mem32(ctx.register(5), 0),
                    1,
                ),
            ),
            0xffff_ffff,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    match *op1.ty() {
        OperandType::SignExtend(val, MemAccessSize::Mem32, MemAccessSize::Mem64) => {
            assert_eq!(val, ctx.and_const(
                ctx.sub(
                    ctx.sign_extend(
                        ctx.mem8(ctx.register(0), 0),
                        MemAccessSize::Mem8,
                        MemAccessSize::Mem32,
                    ),
                    ctx.rsh_const(
                        ctx.mem32(ctx.register(5), 0),
                        1,
                    ),
                ),
                0xffff_ffff,
            ));
        }
        _ => panic!("Bad simplify to {}", op1),
    }
}

#[test]
fn sext_eq() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
        0x66,
    );
    let eq1 = ctx.eq_const(
        ctx.register(0),
        0x66,
    );
    let op2 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
        0xffff_ff99,
    );
    let eq2 = ctx.eq_const(
        ctx.register(0),
        0x99,
    );
    let op3 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0x99,
    );
    let eq3 = ctx.eq_const(
        ctx.register(0),
        0x99,
    );
    let op4 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0x9999_9999,
    );
    let op5 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0xffff_7fff,
    );
    let op6 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0x8000,
    );
    let op7 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0x7fff,
    );
    let eq7 = ctx.eq_const(
        ctx.register(0),
        0x7fff,
    );
    let op8 = ctx.eq_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        0xffff_8000,
    );
    let eq8 = ctx.eq_const(
        ctx.register(0),
        0x8000,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_eq!(op4, ctx.const_0());
    assert_eq!(op5, ctx.const_0());
    assert_eq!(op6, ctx.const_0());
    assert_eq!(op7, eq7);
    assert_eq!(op8, eq8);
}

#[test]
fn sext_gt_const_right() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
        0x66,
    );
    let eq1 = ctx.gt_const(
        ctx.register(0),
        0x66,
    );
    let op2 = ctx.gt_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
        0x99,
    );
    let eq2 = ctx.gt_const(
        ctx.register(0),
        0x7f,
    );
    let op3 = ctx.gt_const(
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
        0xffff_ff99,
    );
    let eq3 = ctx.gt_const(
        ctx.register(0),
        0x99,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
}

#[test]
fn sext_gt_const_left() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const_left(
        0x66,
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
    );
    let eq1 = ctx.gt_const_left(
        0x66,
        ctx.register(0),
    );
    let op2 = ctx.gt_const_left(
        0x99,
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
    );
    let eq2 = ctx.gt_const_left(
        0x80,
        ctx.register(0),
    );
    let op3 = ctx.gt_const_left(
        0xffff_ff99,
        ctx.sign_extend(
            ctx.register(0),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
    );
    let eq3 = ctx.gt_const_left(
        0x99,
        ctx.register(0),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
}

#[test]
fn eq_masked_sub() {
    // (x - 77) & ffff_ffff == 22
    // => x == 99
    // when x is 32bit value
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.sub_const(
                    ctx.mem32(ctx.register(6), 0),
                    0x41,
                ),
                0xffff_ffff
            ),
            0x37,
        ),
        0x0,
    );
    let eq1 = ctx.eq_const(
        ctx.mem32(ctx.register(6), 0),
        0x78
    );
    assert_eq!(op1, eq1);
}

#[test]
fn const_gt_sext_sub_const() {
    let ctx = &OperandContext::new();
    // Valid range 0x33 .. 0x55 => sext unnecessary
    let op1 = ctx.gt_const_left(
        0x22,
        ctx.sub_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem8,
                MemAccessSize::Mem32,
            ),
            0x33,
        ),
    );
    let eq1 = ctx.gt_const_left(
        0x22,
        ctx.sub_const(
            ctx.register(0),
            0x33,
        ),
    );
    // Valid range post sext 0x77 .. 0x99 => change to 0x77 .. 0x80
    let op2 = ctx.gt_const_left(
        0x22,
        ctx.sub_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem8,
                MemAccessSize::Mem32,
            ),
            0x77,
        ),
    );
    let eq2 = ctx.gt_const_left(
        0x9,
        ctx.sub_const(
            ctx.register(0),
            0x77,
        ),
    );
    // Valid range post sext 0x77 .. 0xffff_ff99 => change to 0x77 .. 0x99
    let op3 = ctx.gt_const_left(
        0xffff_ff22,
        ctx.sub_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem8,
                MemAccessSize::Mem32,
            ),
            0x77,
        ),
    );
    let eq3 = ctx.gt_const_left(
        0x22,
        ctx.sub_const(
            ctx.register(0),
            0x77,
        ),
    );
    // Valid range post sext 0x80 .. 0xffff_ff7f => zero
    let op4 = ctx.gt_const_left(
        0xffff_feff,
        ctx.sub_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem8,
                MemAccessSize::Mem32,
            ),
            0x80,
        ),
    );
    let eq4 = ctx.const_0();
    // Valid range post sext 0xffff_ff55 .. 0xffff_ff99 => change to 0x80 .. 0x99
    let op5 = ctx.gt_const_left(
        0x44,
        ctx.sub_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem8,
                MemAccessSize::Mem32,
            ),
            0xffff_ff55,
        ),
    );
    let eq5 = ctx.gt_const_left(
        0x19,
        ctx.sub_const(
            ctx.register(0),
            0x80,
        ),
    );
    // Valid range post sext 0xffff_ff88 .. 0xffff_ff99 => change to 0x88 .. 0x99
    let op6 = ctx.gt_const_left(
        0x11,
        ctx.sub_const(
            ctx.sign_extend(
                ctx.register(0),
                MemAccessSize::Mem8,
                MemAccessSize::Mem32,
            ),
            0xffff_ff88,
        ),
    );
    let eq6 = ctx.gt_const_left(
        0x11,
        ctx.sub_const(
            ctx.register(0),
            0x88,
        ),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
    assert_eq!(op4, eq4);
    assert_eq!(op5, eq5);
    assert_eq!(op6, eq6);
}

#[test]
fn sext_chain() {
    let ctx = &OperandContext::new();
    let op1 = ctx.sign_extend(
        ctx.sign_extend(
            ctx.and_const(
                ctx.register(0),
                0xffff,
            ),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    let eq1 = ctx.sign_extend(
        ctx.and_const(
            ctx.register(0),
            0xffff,
        ),
        MemAccessSize::Mem16,
        MemAccessSize::Mem64,
    );
    let op2 = ctx.sign_extend(
        ctx.sign_extend(
            ctx.and_const(
                ctx.register(0),
                0xff,
            ),
            MemAccessSize::Mem8,
            MemAccessSize::Mem32,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    let eq2 = ctx.sign_extend(
        ctx.and_const(
            ctx.register(0),
            0xff,
        ),
        MemAccessSize::Mem8,
        MemAccessSize::Mem64,
    );
    let op3 = ctx.sign_extend(
        ctx.sign_extend(
            ctx.and_const(
                ctx.register(0),
                0xff,
            ),
            MemAccessSize::Mem8,
            MemAccessSize::Mem16,
        ),
        MemAccessSize::Mem32,
        MemAccessSize::Mem64,
    );
    let eq3 = ctx.sign_extend(
        ctx.and_const(
            ctx.register(0),
            0xff,
        ),
        MemAccessSize::Mem8,
        MemAccessSize::Mem16,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
    assert_eq!(op3, eq3);
}

#[test]
fn f32_arith_unnecessary_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.float_arithmetic(
            ArithOpType::Add,
            ctx.register(0),
            ctx.register(1),
            MemAccessSize::Mem32,
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.float_arithmetic(
        ArithOpType::Add,
        ctx.register(0),
        ctx.register(1),
        MemAccessSize::Mem32,
    );
    let op2 = ctx.and_const(
        ctx.float_arithmetic(
            ArithOpType::ToInt,
            ctx.mem32(ctx.register(0), 0),
            ctx.const_0(),
            MemAccessSize::Mem32,
        ),
        0xffff_ffff,
    );
    let eq2 = ctx.float_arithmetic(
        ArithOpType::ToInt,
        ctx.mem32(ctx.register(0), 0),
        ctx.const_0(),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_sub_const_left() {
    let ctx = &OperandContext::new();
    let op1 = ctx.sub(
        ctx.sub(
            ctx.constant(4),
            ctx.register(2),
        ),
        ctx.constant(6),
    );
    let eq1 = ctx.sub(
        ctx.sub(
            ctx.constant(0),
            ctx.register(2),
        ),
        ctx.constant(2),
    );
    assert_eq!(op1, eq1);

    let op2 = ctx.add(
        ctx.sub(
            ctx.constant(0),
            ctx.register(2),
        ),
        ctx.constant(6),
    );
    let eq2 = ctx.sub(
        ctx.constant(6),
        ctx.register(2),
    );
    assert_eq!(op2, eq2);
}

#[test]
fn simplify_mul_masked_eq() {
    let ctx = &OperandContext::new();
    // Effectively (rax & f0) << 1 == 0 to (rax & f0) == 0
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.mul_const(
                ctx.and_const(
                    ctx.register(0),
                    0xf0,
                ),
                2,
            ),
            0xfffe,
        ),
        0,
    );
    let eq1 = ctx.eq_const(
        ctx.and_const(
            ctx.register(0),
            0xf0,
        ),
        0,
    );
    assert_eq!(op1, eq1);

    // (rax << 1) & f0 == 0 to (rax & 78) == 0
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.mul_const(
                ctx.register(0),
                2,
            ),
            0xf0,
        ),
        0,
    );
    let eq1 = ctx.eq_const(
        ctx.and_const(
            ctx.register(0),
            0x78,
        ),
        0,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_u32_range() {
    let ctx = &OperandContext::new();
    // (d > ((rcx - d) & ffffffff)) | ((rcx & ffffffff) == 1a)
    //  => (e > ((rcx - d) & ffffffff))
    let op1 = ctx.or(
        ctx.gt_const_left(
            0xd,
            ctx.and_const(
                ctx.sub_const(
                    ctx.register(1),
                    0xd,
                ),
                0xffff_ffff,
            ),
        ),
        ctx.eq_const(
            ctx.and_const(ctx.register(1), 0xffff_ffff),
            0x1a,
        ),
    );
    let eq1 = ctx.gt_const_left(
        0xe,
        ctx.and_const(
            ctx.sub_const(
                ctx.register(1),
                0xd,
            ),
            0xffff_ffff,
        ),
    );

    assert_eq!(op1, eq1);
}

#[test]
fn simplify_u8_range() {
    let ctx = &OperandContext::new();
    // ((f > (((rcx >> 10) & ff) - 2)) | (((rcx >> 10) & ff) == 11))
    //  => (10 > (((rcx >> 10) && ff) - 2))
    let val = ctx.and_const(ctx.rsh_const(ctx.register(1), 0x10), 0xff);
    let op1 = ctx.or(
        ctx.gt_const_left(
            0xf,
            ctx.sub_const(
                val,
                0x2,
            ),
        ),
        ctx.eq_const(
            val,
            0x11,
        ),
    );
    let eq1 = ctx.gt_const_left(
        0x10,
        ctx.sub_const(
            val,
            0x2,
        ),
    );

    assert_eq!(op1, eq1);
}

#[test]
fn simplify_mask_hole() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.register(0),
            0xffff_ffff_ffff_00ff,
        ),
        ctx.and_const(
            ctx.register(0),
            0xff00,
        ),
    );
    assert_eq!(op1, ctx.register(0));

    let op1 = ctx.or(
        ctx.and_const(
            ctx.mem64c(0x100),
            0xffff_ffff_ffff_00ff,
        ),
        ctx.and_const(
            ctx.mem64c(0x100),
            0xff00,
        ),
    );
    assert_eq!(op1, ctx.mem64c(0x100));

    let op1 = ctx.or(
        ctx.and_const(
            ctx.mem64c(0x100),
            0xffff_ffff_ffff_00ff,
        ),
        ctx.and_const(
            ctx.mem64c(0x100),
            0xf000,
        ),
    );
    assert_eq!(op1, ctx.and_const(ctx.mem64c(0x100), 0xffff_ffff_ffff_f0ff));

    let op1 = ctx.or(
        ctx.and_const(
            ctx.mem64c(0x100),
            0xff,
        ),
        ctx.and_const(
            ctx.mem64c(0x100),
            0xfff0_0000,
        ),
    );
    assert_eq!(op1, ctx.and_const(ctx.mem64c(0x100), 0xfff0_00ff));
}

#[test]
fn simplify_xor_merge_mem() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.xor(
            ctx.mem16c(0x12331e),
            ctx.xor(
                ctx.mem64c(0x5000),
                ctx.lsh_const(ctx.mem64c(0x123320), 0x10),
            ),
        ),
        ctx.constant(0x12345),
    );
    let eq1 = ctx.xor(
        ctx.xor(
            ctx.mem64c(0x12331e),
            ctx.mem64c(0x5000),
        ),
        ctx.constant(0x12345),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_merge_bug() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.xor(
                ctx.custom(1),
                ctx.lsh_const(
                    ctx.mem16c(0xc0),
                    0x20,
                ),
            ),
            0xffff_ffff_0000_0000,
        ),
        ctx.and_const(
            ctx.xor(
                ctx.custom(1),
                ctx.mem16c(0xc0),
            ),
            0xffff_ffff,
        ),
    );
    let eq1 = ctx.xor(
        ctx.xor(
            ctx.and_const(
                ctx.lsh_const(
                    ctx.mem16c(0xc0),
                    0x20,
                ),
                0xffff_ffff_0000_0000,
            ),
            ctx.and_const(
                ctx.mem16c(0xc0),
                0xffff_ffff,
            ),
        ),
        ctx.custom(1),
    );

    assert_eq!(op1, eq1);
}

#[test]
fn simplify_lsh_and_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh_const(
        ctx.and_const(
            ctx.register(0),
            0xffff_ffff,
        ),
        0x20,
    );
    let eq1 = ctx.lsh_const(
        ctx.register(0),
        0x20,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_rsh_and_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.and_const(
            ctx.register(0),
            0xffff_ffff_0000_0000,
        ),
        0x20,
    );
    let eq1 = ctx.rsh_const(
        ctx.register(0),
        0x20,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_redundant_mask1() {
    // ((((rax & 80000000) == 0) - 1) & 80000000) is just (rax & 80000000)
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.eq_const(
                ctx.and_const(
                    ctx.register(0),
                    0x8000_0000,
                ),
                0,
            ),
            1,
        ),
        0x8000_0000,
    );
    let eq1 = ctx.and_const(
        ctx.register(0),
        0x8000_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_redundant_mask2() {
    // ((((rax & 80000000) == 0) - 1) & 40000000) => (rax & 80000000) >> 1
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.eq_const(
                ctx.and_const(
                    ctx.register(0),
                    0x8000_0000,
                ),
                0,
            ),
            1,
        ),
        0x4000_0000,
    );
    let eq1 = ctx.rsh_const(
        ctx.and_const(
            ctx.register(0),
            0x8000_0000,
        ),
        1,
    );
    assert_eq!(op1, eq1);
    // ((((rax & 8000_0001) == 0) - 1) & 8_0000_0010) => (rax & 8000_0001) << 4
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.eq_const(
                ctx.and_const(
                    ctx.register(0),
                    0x8000_0000,
                ),
                0,
            ),
            1,
        ),
        0x8_0000_0000,
    );
    let eq1 = ctx.lsh_const(
        ctx.and_const(
            ctx.register(0),
            0x8000_0000,
        ),
        4,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt_always_false() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const(
        ctx.and_const(
            ctx.register(8),
            0xffff_ffff,
        ),
        0x7fff_ffff_ffff_ffff,
    );
    let eq1 = ctx.const_0();
    assert_eq!(op1, eq1);
    let op1 = ctx.gt_const(
        ctx.and_const(
            ctx.register(8),
            0xfff,
        ),
        0xfff,
    );
    let eq1 = ctx.const_0();
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_gt_always_true() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt_const_left(
        0x7fff_ffff_ffff_ffff,
        ctx.and_const(
            ctx.register(8),
            0xffff_ffff,
        ),
    );
    let eq1 = ctx.const_1();
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_trivial_or_join() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.register(1),
            0xffff,
        ),
        ctx.and_const(
            ctx.register(1),
            0xffff0000,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.register(1),
        0xffffffff,
    );
    assert_eq!(op1, eq1);
    let op1 = ctx.or(
        ctx.and_const(
            ctx.register(1),
            0xffff_ffff_0000_0000,
        ),
        ctx.and_const(
            ctx.register(1),
            0xffff_ffff,
        ),
    );
    let eq1 = ctx.register(1);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_join_of_adds() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.add(
                ctx.mem32(ctx.register(9), 0),
                ctx.mem32(ctx.register(10), 0),
            ),
            0xffff,
        ),
        ctx.and_const(
            ctx.add(
                ctx.mem32(ctx.register(9), 0),
                ctx.mem32(ctx.register(10), 0),
            ),
            0xffff_0000,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.add(
            ctx.mem32(ctx.register(9), 0),
            ctx.mem32(ctx.register(10), 0),
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_join_of_sub_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.sub_const(
                ctx.register(1),
                0x140,
            ),
            0xff,
        ),
        ctx.and_const(
            ctx.sub_const(
                ctx.register(1),
                0x140,
            ),
            0xffff_ff00,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.register(1),
            0x140,
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_join_of_add_const_mem() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.add_const(
                ctx.mem32(ctx.register(1), 0),
                0x10000,
            ),
            0xffff,
        ),
        ctx.and_const(
            ctx.add_const(
                ctx.mem32(ctx.register(1), 0),
                0x10000,
            ),
            0xffff_0000,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.add_const(
            ctx.mem32(ctx.register(1), 0),
            0x10000,
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_eq_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.mul_const(
            ctx.register(1),
            0x3,
        ),
        ctx.mul_const(
            ctx.register(1),
            0x9,
        ),
    );
    let eq1 = ctx.eq_const(
        ctx.mul_const(
            ctx.register(1),
            0x6,
        ),
        0,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn int_to_float_is_32bit() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.arithmetic(
            ArithOpType::ToFloat,
            ctx.register(1),
            ctx.const_0(),
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.arithmetic(
        ArithOpType::ToFloat,
        ctx.register(1),
        ctx.const_0(),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn float_compares_are_bool() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.float_arithmetic(
            ArithOpType::Equal,
            ctx.xmm(0, 0),
            ctx.xmm(0, 1),
            MemAccessSize::Mem32,
        ),
        1,
    );
    let eq1 = ctx.float_arithmetic(
        ArithOpType::Equal,
        ctx.xmm(0, 0),
        ctx.xmm(0, 1),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
    let op1 = ctx.and_const(
        ctx.float_arithmetic(
            ArithOpType::GreaterThan,
            ctx.xmm(0, 0),
            ctx.xmm(0, 1),
            MemAccessSize::Mem32,
        ),
        1,
    );
    let eq1 = ctx.float_arithmetic(
        ArithOpType::GreaterThan,
        ctx.xmm(0, 0),
        ctx.xmm(0, 1),
        MemAccessSize::Mem32,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn mem_mask_eq_mask() {
    let ctx = &OperandContext::new();
    // (Mem32[100] & 8888_0000) == 8888_0000
    // => (Mem16[102] & 8888) == 8888
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.mem32c(0x100),
            0x8888_0000,
        ),
        0x8888_0000,
    );
    let eq1 = ctx.eq_const(
        ctx.and_const(
            ctx.mem32c(0x102),
            0x8888,
        ),
        0x8888,
    );
    // ((Mem16[102] >> 3) & 1111) == 1111
    let eq2 = ctx.eq_const(
        ctx.and_const(
            ctx.rsh_const(
                ctx.mem32c(0x102),
                3,
            ),
            0x1111,
        ),
        0x1111,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1, eq2);
    // Check also that canonical form is (_ == 8888)
    // (No shifts)
    let (_, r) = op1.if_arithmetic_eq().unwrap();
    assert_eq!(r, ctx.constant(0x8888));
}

#[test]
fn xor_or() {
    // a ^ (a | b) => !a & b, where ab are booleans
    // 00 => 0 ^ 0 = 0
    // 01 => 0 ^ 1 = 1
    // 10 => 1 ^ 1 = 0
    // 11 => 1 ^ 1 = 0
    // Ends up being part of float compare -> lahf -> test ah -> parity chain
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.or(
            ctx.eq_const(ctx.register(6), 7),
            ctx.eq_const(ctx.register(1), 0),
        ),
        ctx.eq_const(ctx.register(1), 0),
    );
    let eq1 = ctx.and(
        ctx.neq_const(ctx.register(1), 0),
        ctx.eq_const(ctx.register(6), 7),
    );
    assert_eq!(op1, eq1);
    // Check also that canonical form is and
    let _ = op1.if_arithmetic_and().unwrap();

    let f_eq = ctx.float_arithmetic(
        ArithOpType::Equal,
        ctx.mem16c(0x500),
        ctx.xmm(0, 0),
        MemAccessSize::Mem32,
    );
    let op1 = ctx.xor(
        ctx.or(
            ctx.or(
                ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
                ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
            ),
            f_eq,
        ),
        ctx.or(
            ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
            ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
        ),
    );
    let eq1 = ctx.and(
        ctx.eq_const(
            ctx.or(
                ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
                ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
            ),
            0,
        ),
        f_eq,
    );
    assert_eq!(op1, eq1);

    // Check all variations of 2-op or as a
    let op1 = ctx.xor(
        ctx.or(
            ctx.or(
                ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
                ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
            ),
            f_eq,
        ),
        ctx.or(
            f_eq,
            ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
        ),
    );
    let eq1 = ctx.and(
        ctx.eq_const(
            ctx.or(
                ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
                f_eq,
            ),
            0,
        ),
        ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
    );
    assert_eq!(op1, eq1);

    let op1 = ctx.xor(
        ctx.or(
            ctx.or(
                ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
                ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
            ),
            f_eq,
        ),
        ctx.or(
            f_eq,
            ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
        ),
    );
    let eq1 = ctx.and(
        ctx.eq_const(
            ctx.or(
                ctx.eq_const(ctx.mem16c(0x500), 0x7f80),
                f_eq,
            ),
            0,
        ),
        ctx.eq_const(ctx.xmm(0, 0), 0x7f80_0000),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn add_greater_or_eq() {
    let ctx = &OperandContext::new();

    // (a > x + 400) | (a == x + 400)
    // => (a > x + 3ff)
    let lhs = ctx.register(1);
    let rhs = ctx.add_const(
        ctx.mem16c(0x500),
        0x400,
    );
    let gt = ctx.gt(lhs, rhs);
    let eq = ctx.eq(lhs, rhs);
    let op1 = ctx.or(gt, eq);
    let eq1 = ctx.gt(
        lhs,
        ctx.add_const(
            ctx.mem16c(0x500),
            0x3ff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn add_greater_or_eq_masked() {
    let ctx = &OperandContext::new();

    // (a > x + 400) | (a == x + 400)
    // => (a > x + 3ff)
    let lhs = ctx.and_const(ctx.register(1), 0xffff_ffff);
    let rhs = ctx.add_const(
        ctx.mem16c(0x500),
        0x400,
    );
    let gt = ctx.gt(lhs, rhs);
    let eq = ctx.eq(lhs, rhs);
    let op1 = ctx.or(gt, eq);
    let eq1 = ctx.gt(
        lhs,
        ctx.add_const(
            ctx.mem16c(0x500),
            0x3ff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn add_greater_or_eq_no_change() {
    let ctx = &OperandContext::new();

    // (a > x + 400) | (a == x + 500)
    // Won't simplify
    let lhs = ctx.register(1);
    let rhs = ctx.add_const(
        ctx.mem16c(0x500),
        0x400,
    );
    let gt = ctx.gt(lhs, rhs);
    let eq = ctx.eq(lhs, ctx.add_const(ctx.mem16c(0x500), 0x500));
    let op1 = ctx.or(gt, eq);
    let (l, r) = op1.if_arithmetic_or()
        .unwrap_or_else(|| panic!("{op1} is not or"));
    assert!((l == gt && r == eq) || (l == eq && r == gt));
}

#[test]
fn or_self() {
    let ctx = &OperandContext::new();

    let op1 = ctx.or(
        ctx.register(0),
        ctx.register(0),
    );
    let eq1 = ctx.register(0);
    assert_eq!(op1, eq1);
}

#[test]
fn lsh_relbits() {
    let ctx = &OperandContext::new();

    // Unknown shift but at most 7 bits,
    // so it'll always get masked out
    let op1 = ctx.and_const(
        ctx.lsh(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 0),
                8,
            ),
            ctx.and_const(
                ctx.register(4),
                3,
            ),
        ),
        0xff00_00ff,
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn rsh_relbits() {
    let ctx = &OperandContext::new();

    // Unknown shift but at most 7 bits,
    // so it'll always get masked out
    let op1 = ctx.and_const(
        ctx.rsh(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 0),
                16,
            ),
            ctx.and_const(
                ctx.register(4),
                3,
            ),
        ),
        0xff00_00ff,
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn zero_shift_any() {
    let ctx = &OperandContext::new();

    let op1 = ctx.lsh(
        ctx.constant(0),
        ctx.register(5)
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);

    let op1 = ctx.rsh(
        ctx.constant(0),
        ctx.register(5)
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn or_and_to_const_bug() {
    let ctx = &OperandContext::new();

    // Result is just const1 & const2, bits of xmm that aren't set to 1 by or
    // are set to 0 by and
    let op1 = ctx.and_const(
        ctx.or_const(
            ctx.xmm(0, 0),
            0x0103_3501_0025_c700,
        ),
        0x0000_1401_0025_c700,
    );
    let eq1 = ctx.constant(0x0000_1401_0025_c700);
    assert_eq!(op1, eq1);
}

#[test]
fn mul_simplify_to_zero() {
    let ctx = &OperandContext::new();

    let op1 = ctx.mul_const(
        ctx.and_const(
            ctx.register(0),
            0xb_0000_0000,
        ),
        0x0400_0b00_0000_0000,
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);

    let op1 = ctx.mul(
        ctx.mul(
            ctx.and_const(
                ctx.register(0),
                0xffff_0000,
            ),
            ctx.and_const(
                ctx.register(0),
                0xffff_0000,
            ),
        ),
        ctx.mul(
            ctx.and_const(
                ctx.register(0),
                0xffff_0000,
            ),
            ctx.and_const(
                ctx.register(0),
                0xffff_0000,
            ),
        ),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn and_simplify_to_zero() {
    let ctx = &OperandContext::new();

    // rax must be both 8 and less than 5 => becomes 0
    let op1 = ctx.and(
        ctx.and(
            ctx.register(0),
            ctx.eq_const(
                ctx.register(0),
                8,
            ),
        ),
        ctx.gt_const_left(
            5,
            ctx.register(0),
        ),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn or_cmp_1bit_mask_consistency() {
    let ctx = &OperandContext::new();

    // (rax > rcx) | (rdx & 1)
    // is same as
    // ((rax > rcx) | rdx) & 1
    let op1 = ctx.or(
        ctx.gt(
            ctx.register(0),
            ctx.register(1),
        ),
        ctx.and_const(
            ctx.register(2),
            1,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.gt(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.register(2),
        ),
        1,
    );
    assert_eq!(op1, eq1);

    // ((rax > rcx) | ((rdx | Custom(2)) & 1)) | (r8 == r9)
    // is same as
    // ((((rax > rcx) | rdx) | Custom(2)) | (r8 == r9)) & 1
    let op1 = ctx.or(
        ctx.or(
            ctx.gt(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.and_const(
                ctx.or(
                    ctx.register(2),
                    ctx.custom(2),
                ),
                1,
            ),
        ),
        ctx.eq(
            ctx.register(8),
            ctx.register(9),
        ),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.or(
                ctx.or(
                    ctx.gt(
                        ctx.register(0),
                        ctx.register(1),
                    ),
                    ctx.register(2),
                ),
                ctx.custom(2),
            ),
            ctx.eq(
                ctx.register(8),
                ctx.register(9),
            ),
        ),
        1,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn xor_cmp_1bit_mask_consistency() {
    let ctx = &OperandContext::new();

    // (rax > rcx) ^ (rdx & 1)
    // is same as
    // ((rax > rcx) ^ rdx) & 1
    let op1 = ctx.xor(
        ctx.gt(
            ctx.register(0),
            ctx.register(1),
        ),
        ctx.and_const(
            ctx.register(2),
            1,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.gt(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.register(2),
        ),
        1,
    );

    assert_eq!(op1, eq1);

    // And the canonical form is ((rax > rcx) ^ rdx) & 1
    // This is different from or; moving and masks to be outermost
    // with xor was considered ideal. Maybe or should do the same.
    op1.if_arithmetic(ArithOpType::And).expect("Did not canonicalize to and");

    // ((rax > rcx) ^ ((rdx ^ Custom(2)) & 1)) ^ (r8 == r9)
    // is same as
    // ((((rax > rcx) ^ rdx) ^ Custom(2)) ^ (r8 == r9)) & 1
    let op1 = ctx.xor(
        ctx.xor(
            ctx.gt(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.and_const(
                ctx.xor(
                    ctx.register(2),
                    ctx.custom(2),
                ),
                1,
            ),
        ),
        ctx.eq(
            ctx.register(8),
            ctx.register(9),
        ),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.xor(
                ctx.xor(
                    ctx.gt(
                        ctx.register(0),
                        ctx.register(1),
                    ),
                    ctx.register(2),
                ),
                ctx.custom(2),
            ),
            ctx.eq(
                ctx.register(8),
                ctx.register(9),
            ),
        ),
        1,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_with_mul_inside_xor() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.xmm(0, 0),
        ctx.xor_const(
            ctx.mul(
                ctx.and_const(
                    ctx.register(1),
                    0x603060001060002,
                ),
                ctx.register(0),
            ),
            0x520230010090002,
        ),
    );
    let eq1 = ctx.and(
        ctx.xmm(0, 0),
        ctx.xor_const(
            ctx.mul(
                ctx.and_const(
                    ctx.register(1),
                    0x0106_0002,
                ),
                ctx.register(0),
            ),
            0x1009_0002,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_or_with_mem_merge() {
    let ctx = &OperandContext::new();
    // Mem16 can be moved inside to (Mem32[rax] & c500_ffff)
    // and the c500_ffff and c500_61ff masks have same effect
    // due to or constant setting differing bits to one.
    // (Simpler simplification without mem merge won't hit the inconsistency bug)
    let op1 = ctx.or(
        ctx.mem16(ctx.register(0), 0),
        ctx.or_const(
            ctx.and_const(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(0), 3),
                    0x18,
                ),
                0xc5000000,
            ),
            0x2019e00,
        ),
    );
    let eq1 = ctx.or_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0),
            0xc50061ff,
        ),
        0x2019e00,
    );
    let eq2 = ctx.or_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0),
            0xc500ffff,
        ),
        0x2019e00,
    );
    assert_eq!(eq2, eq1);
    assert_eq!(op1, eq1);
}

#[test]
fn div_relevant_bits() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.div(
            ctx.constant(0x5000_0002),
            ctx.register(1),
        ),
        1,
    );
    assert_ne!(op1, ctx.const_0());

    let op1 = ctx.and_const(
        ctx.div(
            ctx.constant(0x5000_0002),
            ctx.register(1),
        ),
        0x8000_0000,
    );
    assert_eq!(op1, ctx.const_0());
}

#[test]
fn simplify_xor_with_masks1() {
    let ctx = &OperandContext::new();
    // Effectively ((Mem32[2] << 8) ^ (Mem32[2] << 8) ^ esi) & ffff_ffff
    // So memory shifts cancel out.
    let op1 = ctx.xor(
        ctx.lsh_const(
            ctx.and_const(
                ctx.mem32c(2),
                0xff_ffff,
            ),
            8,
        ),
        ctx.and_const(
            ctx.xor(
                ctx.lsh_const(ctx.mem32c(2), 8),
                ctx.register(6),
            ),
            0xffff_ffff,
        ),
    );
    assert_eq!(op1, ctx.and_const(ctx.register(6), 0xffff_ffff));
}

#[test]
fn simplify_xor_with_masks2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.xor(
            ctx.and_const(
                ctx.register(0),
                0xffff_ff00,
            ),
            ctx.and_const(
                ctx.register(1),
                0xffff_0000,
            ),
        ),
        ctx.xor(
            ctx.and_const(
                ctx.register(0),
                0x00ff_ffff,
            ),
            ctx.and_const(
                ctx.register(1),
                0x00ff_ffff,
            ),
        ),
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.register(0),
            0xff00_00ff,
        ),
        ctx.and_const(
            ctx.register(1),
            0xff00_ffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn canonicalize_shifted_masked_op() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh_const(
        ctx.and_const(
            ctx.register(0),
            0xff_ffff,
        ),
        8,
    );
    let eq1 = ctx.and_const(
        ctx.lsh_const(
            ctx.register(0),
            8,
        ),
        0xffff_ff00,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn canonicalize_u32_sub_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.mul_const(
        ctx.sub_const(
            ctx.register(0),
            0xd,
        ),
        4,
    );
    let eq1 = ctx.sub_const(
        ctx.mul_const(
            ctx.register(0),
            0x4,
        ),
        0xd * 4,
    );
    assert_eq!(op1, eq1);

    let ctx = &OperandContext::new();
    let op1 = ctx.mul_const(
        ctx.and_const(
            ctx.sub_const(
                ctx.register(0),
                0xd,
            ),
            0xffff_ffff,
        ),
        4,
    );
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.mul_const(
                ctx.register(0),
                0x4,
            ),
            0xd * 4,
        ),
        0x3_ffff_fffc,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_with_masks3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.or(
            ctx.and_const(
                ctx.xor_const(
                    ctx.and_const(
                        ctx.register(0),
                        0xff_ff00,
                    ),
                    0xff,
                ),
                0xff_ffff,
            ),
            ctx.and_const(
                ctx.register(0),
                0xff00_0000,
            ),
        ),
        ctx.and_const(
            ctx.xor_const(
                ctx.xor(
                    ctx.and_const(ctx.register(0), 0xffff_ff00),
                    ctx.register(6),
                ),
                0xff,
            ),
            0xffff_ffff,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.register(6),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_with_masks4() {
    let ctx = &OperandContext::new();

    let a = ctx.lsh_const(
        ctx.xor(
            ctx.mem8(ctx.register(1), 3),
            ctx.rsh_const(
                ctx.register(0),
                0x18,
            ),
        ),
        0x18,
    );
    let b = ctx.xor(
        ctx.lsh_const(
            ctx.mem8(ctx.register(1), 3),
            0x18,
        ),
        ctx.register(0),
    );
    let op1 = ctx.xor(
        a,
        b,
    );
    let eq1 = ctx.and_const(
        ctx.register(0),
        0x00ff_ffff,
    );
    assert_eq!(op1, eq1);

    // base and second are otherwise same, but
    // base clears low 0x18 bytes of rax
    let base = ctx.lsh_const(
        ctx.or(
            ctx.xor(
                ctx.mem8(ctx.register(1), 3),
                ctx.rsh_const(
                    ctx.register(0),
                    0x18,
                ),
            ),
            ctx.xor(
                ctx.register(3),
                ctx.mem8(ctx.register(2), 3),
            ),
        ),
        0x18,
    );
    let second = ctx.or(
        ctx.xor(
            ctx.lsh_const(
                ctx.mem8(ctx.register(1), 3),
                0x18,
            ),
            ctx.register(0),
        ),
        ctx.lsh_const(
            ctx.xor(
                ctx.register(3),
                ctx.mem8(ctx.register(2), 3),
            ),
            0x18,
        ),
    );
    let op1 = ctx.xor(
        base,
        second,
    );
    let eq1 = ctx.and_const(
        ctx.register(0),
        0x00ff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_with_masks5() {
    let ctx = &OperandContext::new();

    let base = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.mem32(ctx.register(1), 1),
            ),
            ctx.register(6),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(2), 0),
            ctx.register(6),
        ),
    );
    let base_high = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(2), 3),
                    0x18,
                ),
                ctx.lsh_const(
                    ctx.mem8(ctx.register(1), 4),
                    0x18,
                ),
            ),
            ctx.register(6),
        ),
        ctx.xor(
            ctx.lsh_const(
                ctx.mem32(ctx.register(2), 3),
                0x18,
            ),
            ctx.register(6),
        ),
    );
    let op1 = ctx.xor(
        ctx.and_const(base, 0xff_ffff),
        ctx.and_const(
            base_high,
            0xff00_0000,
        ),
    );
    let eq1 = ctx.and_const(base, 0xffff_ffff);
    assert_eq!(op1, eq1);

    // Check that similar op with Mem16[rdx] ^ (Mem8[rdx] << 0x18)
    // doesn't become Mem32[rdx]
    let base = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.mem16(ctx.register(2), 0),
                ctx.mem32(ctx.register(1), 1),
            ),
            ctx.register(6),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(9), 0),
            ctx.register(6),
        ),
    );
    let base_high = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(2), 3),
                    0x18,
                ),
                ctx.lsh_const(
                    ctx.mem8(ctx.register(1), 4),
                    0x18,
                ),
            ),
            ctx.register(6),
        ),
        ctx.xor(
            ctx.lsh_const(
                ctx.mem32(ctx.register(9), 3),
                0x18,
            ),
            ctx.register(6),
        ),
    );
    let op1 = ctx.xor(
        ctx.and_const(base, 0xff_ffff),
        ctx.and_const(
            base_high,
            0xff00_0000,
        ),
    );
    let ne1 = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.mem32(ctx.register(1), 1),
            ),
            ctx.register(6),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(9), 0),
            ctx.register(6),
        ),
    );
    assert_ne!(op1, ctx.and_const(ne1, 0xffff_ffff));

    let eq1 = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.and_const(
                    ctx.mem32(ctx.register(2), 0),
                    0xff00_ffff,
                ),
                ctx.mem32(ctx.register(1), 1),
            ),
            ctx.register(6),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(9), 0),
            ctx.register(6),
        ),
    );
    assert_eq!(op1, ctx.and_const(eq1, 0xffff_ffff));
}

#[test]
fn simplify_xor_with_masks6() {
    let ctx = &OperandContext::new();

    let high1 = ctx.lsh_const(
        ctx.or(
            ctx.xor(
                ctx.xor(
                    ctx.mem8(ctx.register(2), 3),
                    ctx.mem8(ctx.register(1), 4),
                ),
                ctx.rsh_const(
                    ctx.register(6),
                    0x18,
                ),
            ),
            ctx.xor(
                ctx.mem8(ctx.register(2), 3),
                ctx.rsh_const(
                    ctx.register(6),
                    0x18,
                ),
            ),
        ),
        0x18,
    );
    let high2 = ctx.and_const(
        ctx.or(
            ctx.xor(
                ctx.xor(
                    ctx.lsh_const(
                        ctx.mem8(ctx.register(2), 3),
                        0x18,
                    ),
                    ctx.lsh_const(
                        ctx.mem8(ctx.register(1), 4),
                        0x18,
                    ),
                ),
                ctx.register(6),
            ),
            ctx.xor(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(2), 3),
                    0x18,
                ),
                ctx.register(6),
            ),
        ),
        0xff00_0000,
    );
    let op1 = ctx.xor(high1, high2);
    let eq1 = ctx.and_const(
        ctx.register(6),
        0xffff_ffff_0000_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_xor_with_masks7() {
    let ctx = &OperandContext::new();

    let high = ctx.lsh_const(
        ctx.and_const(
            ctx.or(
                ctx.xor(
                    ctx.xor(
                        ctx.mem8(ctx.register(2), 3),
                        ctx.mem8(ctx.register(1), 4),
                    ),
                    ctx.rsh_const(
                        ctx.register(6),
                        0x18,
                    ),
                ),
                ctx.xor(
                    ctx.mem8(ctx.register(2), 3),
                    ctx.rsh_const(
                        ctx.register(6),
                        0x18,
                    ),
                ),
            ),
            0xff,
        ),
        0x18,
    );
    let low = ctx.and_const(
        ctx.or(
            ctx.xor(
                ctx.xor(
                    ctx.mem32(ctx.register(2), 0),
                    ctx.mem32(ctx.register(1), 1),
                ),
                ctx.register(6),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.register(6),
            ),
        ),
        0x00ff_ffff,
    );
    let op1 = ctx.xor(high, low);
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.xor(
                ctx.xor(
                    ctx.mem32(ctx.register(2), 0),
                    ctx.mem32(ctx.register(1), 1),
                ),
                ctx.register(6),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.register(6),
            ),
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn canonicalize_and_mul_poweroftwo() {
    let ctx = &OperandContext::new();

    let op1 = ctx.mul_const(
        ctx.and_const(
            ctx.register(4),
            0xff,
        ),
        0x4,
    );
    let eq1 = ctx.and_const(
        ctx.mul_const(
            ctx.register(4),
            0x4,
        ),
        0x3ff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn xor_rotate_mem_crash() {
    let ctx = &OperandContext::new();

    let _ = ctx.xor(
        ctx.rsh_const(
            ctx.mem32(ctx.register(0), 4),
            0x8,
        ),
        ctx.lsh_const(
            ctx.mem8(ctx.register(0), 4),
            0x18,
        ),
    );
}

#[test]
fn xor_mem64_crash() {
    let ctx = &OperandContext::new();

    let op1 = ctx.or(
        ctx.lsh_const(
            ctx.and_const(
                ctx.mem8(ctx.register(0), 7),
                0xfe,
            ),
            0x38,
        ),
        ctx.mem64(ctx.register(0), 0),
    );
    assert_eq!(op1, ctx.mem64(ctx.register(0), 0));
}

#[test]
fn mul_masked_reduce_inner_mask() {
    let ctx = &OperandContext::new();

    // (x & ffff_f00f) * 0x28 is same as
    // ((x & ffff_f00f) << 3) + ((x & ffff_f00f) << 5)
    // Meaning that high 3 bits of the mask become useless
    // when masked with ffff_ffff
    let op1 = ctx.and_const(
        ctx.mul_const(
            ctx.and_const(
                ctx.register(0),
                0xffff_f00f,
            ),
            0x28,
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.and_const(
        ctx.mul_const(
            ctx.and_const(
                ctx.register(0),
                0x1fff_f00f,
            ),
            0x28,
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn mul_large_const_bug() {
    let ctx = &OperandContext::new();

    let op1 = ctx.and_const(
        ctx.mul_const(
            ctx.register(0),
            0xa600_0000,
        ),
        0xffff_ffff,
    );
    // Check that it at least isn't (rax & mask), mul should still be kept somewhere
    let (l, _) = op1.if_and_with_const().unwrap_or_else(|| panic!("Invalid op {}", op1));
    l.if_mul_with_const().unwrap_or_else(|| panic!("Invalid op {}", op1));
}

#[test]
fn mul_simplify_crash() {
    let ctx = &OperandContext::new();
    let _ = ctx.mul(
        ctx.mul_const(
            ctx.and(
                ctx.or(
                    ctx.xmm(0, 0),
                    ctx.register(1),
                ),
                ctx.xmm(1, 0),
            ),
            0x0100_0000_0000_0000,
        ),
        ctx.register(0),
    );
}

#[test]
fn and_unnecesary_const_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.or(
            ctx.register(1),
            ctx.mem32(ctx.register(2), 0),
        ),
        ctx.mem32(ctx.register(3), 0),
    );
    // Check that there are no constant ffff_ffff mask added uselessly
    let (l, r) = op1.if_arithmetic_and().unwrap_or_else(|| panic!("Incorrect op {op1}"));
    let ((), other) = Operand::either(l, r, |x| match x.if_mem32() {
        Some(..) => Some(()),
        _ => None,
    }).unwrap_or_else(|| panic!("Incorrect op {op1}"));
    let (l, r) = other.if_arithmetic_or().unwrap_or_else(|| panic!("Incorrect op {op1}"));
    let (_, other) = Operand::either(l, r, |x| x.if_register())
        .unwrap_or_else(|| panic!("Incorrect op {op1}"));
    assert_eq!(other, ctx.mem32(ctx.register(2), 0));
}

#[test]
fn and_unnecesary_const_mask_2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.register(1),
                0x20,
            ),
            ctx.or_const(
                ctx.mem8(ctx.register(0), 0),
                0x50200,
            ),
        ),
        0xffff_ffff
    );
    let eq1 = ctx.or_const(
        ctx.mem8(ctx.register(0), 0),
        0x50200,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn xor_const_inside_and_chain() {
    let ctx = &OperandContext::new();
    // Shouldn't simplify
    let op1 = ctx.xor_const(
        ctx.and_const(
            ctx.and(
                ctx.register(0),
                ctx.xor_const(
                    ctx.register(1),
                    0xff,
                ),
            ),
            0xffff_ffff,
        ),
        0xff,
    );
    (|| {
        let x = match op1.if_and_with_const()? {
            (x, 0xffff_ffff) => x,
            _ => return None,
        };
        let x = match x.if_xor_with_const()? {
            (x, 0xff) => x,
            _ => return None,
        };
        x.if_arithmetic_and()
    })().unwrap_or_else(|| panic!("Bad simplify {op1}"));
}

#[test]
fn masked_or_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.register(1),
                0x20,
            ),
            ctx.or(
                ctx.and_const(
                    ctx.mem16(ctx.register(0), 0),
                    0x3a32,
                ),
                ctx.and_const(
                    ctx.mem16(ctx.register(1), 0),
                    0x3aff,
                ),
            ),
        ),
        0xffff_ffff
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.mem16(ctx.register(0), 0),
            0x3a32,
        ),
        ctx.and_const(
            ctx.mem16(ctx.register(1), 0),
            0x3aff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xor_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.xor(
            ctx.lsh_const(
                ctx.register(1),
                0x20,
            ),
            ctx.xor(
                ctx.and_const(
                    ctx.mem16(ctx.register(0), 0),
                    0x3a32,
                ),
                ctx.and_const(
                    ctx.mem16(ctx.register(1), 0),
                    0x3aff,
                ),
            ),
        ),
        0xffff_ffff
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.mem16(ctx.register(0), 0),
            0x3a32,
        ),
        ctx.and_const(
            ctx.mem16(ctx.register(1), 0),
            0x3aff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_with_const_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or_const(
        ctx.and_const(
            ctx.or(
                ctx.lsh_const(
                    ctx.register(1),
                    0x20,
                ),
                ctx.or(
                    ctx.register(2),
                    ctx.register(3),
                ),
            ),
            !0x12341234u64 & 0xffff_ffff,
        ),
        0x1234_1234,
    );
    let eq1 = ctx.and_const(
        ctx.or_const(
            ctx.or(
                ctx.lsh_const(
                    ctx.register(1),
                    0x20,
                ),
                ctx.or(
                    ctx.register(2),
                    ctx.register(3),
                ),
            ),
            0x12341234,
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_shifts() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.lsh_const(
            ctx.and_const(
                ctx.register(1),
                0xffff,
            ),
            0x10,
        ),
        ctx.and_const(
            ctx.rsh_const(
                ctx.register(2),
                0x10,
            ),
            0xffff,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.register(1),
                0x10,
            ),
            ctx.and_const(
                ctx.rsh_const(
                    ctx.register(2),
                    0x10,
                ),
                0xffff,
            ),
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_xor_subset() {
    let ctx = &OperandContext::new();
    // (x ^ y) | x => x | y
    let op1 = ctx.or(
        ctx.xor(
            ctx.xor(
                ctx.register(3),
                ctx.mem32(ctx.register(1), 0),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.register(5),
            ),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(2), 0),
            ctx.register(5),
        ),
    );
    let eq1 = ctx.or(
        ctx.xor(
            ctx.register(3),
            ctx.mem32(ctx.register(1), 0),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(2), 0),
            ctx.register(5),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_xor_to_and() {
    let ctx = &OperandContext::new();
    // (x ^ y) ^ (x | y) => x & y
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.or(
                ctx.xor(
                    ctx.register(3),
                    ctx.mem32(ctx.register(1), 0),
                ),
                ctx.xor(
                    ctx.mem32(ctx.register(2), 0),
                    ctx.register(5),
                ),
            ),
            0xffff_ffff,
        ),
        ctx.xor(
            ctx.xor(
                ctx.register(3),
                ctx.mem32(ctx.register(1), 0),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.register(5),
            ),
        ),
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.and(
                ctx.xor(
                    ctx.register(3),
                    ctx.mem32(ctx.register(1), 0),
                ),
                ctx.xor(
                    ctx.mem32(ctx.register(2), 0),
                    ctx.register(5),
                ),
            ),
            0xffff_ffff,
        ),
        ctx.and_const(
            ctx.xor(
                ctx.register(3),
                ctx.register(5),
            ),
            0xffff_ffff_0000_0000,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_xor_and_to_xor() {
    let ctx = &OperandContext::new();
    // (x & y) ^ (x | y) => x ^ y
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.or(
                ctx.xor(
                    ctx.register(3),
                    ctx.mem32(ctx.register(1), 0),
                ),
                ctx.xor(
                    ctx.mem32(ctx.register(2), 0),
                    ctx.register(5),
                ),
            ),
            0xffff_ffff,
        ),
        ctx.and(
            ctx.xor(
                ctx.register(3),
                ctx.mem32(ctx.register(1), 0),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(2), 0),
                ctx.register(5),
            ),
        ),
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.xor(
                ctx.xor(
                    ctx.register(3),
                    ctx.mem32(ctx.register(1), 0),
                ),
                ctx.xor(
                    ctx.mem32(ctx.register(2), 0),
                    ctx.register(5),
                ),
            ),
            0xffff_ffff,
        ),
        ctx.and_const(
            ctx.and(
                ctx.register(3),
                ctx.register(5),
            ),
            0xffff_ffff_0000_0000,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_merge_complex() {
    let ctx = &OperandContext::new();
    // (x & y) ^ (x | y) => x ^ y
    let op1 = ctx.or(
        ctx.and(
            ctx.xor(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(1), 3),
                    0x18,
                ),
                ctx.lsh_const(
                    ctx.mem8(ctx.register(2), 3),
                    0x18,
                ),
            ),
            ctx.xor(
                ctx.lsh_const(
                    ctx.mem8(ctx.register(3), 3),
                    0x18,
                ),
                ctx.register(5),
            ),
        ),
        ctx.and_const(
            ctx.and(
                ctx.xor(
                    ctx.mem32(ctx.register(1), 0),
                    ctx.mem32(ctx.register(2), 0),
                ),
                ctx.xor(
                    ctx.mem32(ctx.register(3), 0),
                    ctx.register(5),
                ),
            ),
            0x00ff_ffff,
        ),
    );
    let eq1 = ctx.and(
        ctx.xor(
            ctx.mem32(ctx.register(1), 0),
            ctx.mem32(ctx.register(2), 0),
        ),
        ctx.xor(
            ctx.mem32(ctx.register(3), 0),
            ctx.register(5),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_or_const() {
    let ctx = &OperandContext::new();
    // (x | y | (c1 | c2)) & c2  => c2
    let op1 = ctx.and_const(
        ctx.or_const(
            ctx.or(
                ctx.register(5),
                ctx.mem32(ctx.register(1), 0),
            ),
            0x5550,
        ),
        0x4400,
    );
    let eq1 = ctx.constant(0x4400);
    assert_eq!(op1, eq1);
}

#[test]
fn and_and_xor() {
    let ctx = &OperandContext::new();
    // (x & y) & (x ^ y) => 0
    let op1 = ctx.and(
        ctx.and(
            ctx.xor(
                ctx.mem32(ctx.register(1), 0),
                ctx.mem32(ctx.register(2), 0),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(3), 0),
                ctx.register(5),
            ),
        ),
        ctx.xor(
            ctx.xor(
                ctx.mem32(ctx.register(1), 0),
                ctx.mem32(ctx.register(2), 0),
            ),
            ctx.xor(
                ctx.mem32(ctx.register(3), 0),
                ctx.register(5),
            ),
        ),
    );
    let eq1 = ctx.constant(0);
    assert_eq!(op1, eq1);
}

#[test]
fn and_and_or() {
    let ctx = &OperandContext::new();
    // (x & y) & (x | y) => x & y
    // In general x & (x | y) => x
    let op1 = ctx.and(
        ctx.and(
            ctx.or(
                ctx.mem32(ctx.register(1), 0),
                ctx.mem32(ctx.register(2), 0),
            ),
            ctx.or(
                ctx.mem32(ctx.register(3), 0),
                ctx.register(5),
            ),
        ),
        ctx.or(
            ctx.or(
                ctx.mem32(ctx.register(1), 0),
                ctx.mem32(ctx.register(2), 0),
            ),
            ctx.or(
                ctx.mem32(ctx.register(3), 0),
                ctx.register(5),
            ),
        ),
    );
    let eq1 = ctx.and(
        ctx.or(
            ctx.mem32(ctx.register(1), 0),
            ctx.mem32(ctx.register(2), 0),
        ),
        ctx.or(
            ctx.mem32(ctx.register(3), 0),
            ctx.register(5),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_and_xor_2() {
    let ctx = &OperandContext::new();
    // x & !(x ^ y) => x & y
    let op1 = ctx.and(
        ctx.mem8(ctx.register(1), 0),
        ctx.xor_const(
            ctx.xor(
                ctx.mem8(ctx.register(0), 0),
                ctx.mem8(ctx.register(1), 0),
            ),
            0xff,
        ),
    );
    let eq1 = ctx.and(
        ctx.mem8(ctx.register(0), 0),
        ctx.mem8(ctx.register(1), 0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_merge_masks() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.mul_const(
                ctx.register(0),
                2,
            ),
            ctx.register(1),
        ),
        0x019e_0130,
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.register(1),
            0x019e_0130,
        ),
        ctx.mul_const(
            ctx.and_const(
                ctx.register(0),
                0x00cf_0098,
            ),
            2,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_with_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 0),
                8,
            ),
            ctx.register(1),
        ),
        0x5_1100,
    );
    let eq1 = ctx.or(
        ctx.lsh_const(
            ctx.and_const(
                ctx.mem8(ctx.register(0), 0),
                0x11,
            ),
            8,
        ),
        ctx.and_const(
            ctx.register(1),
            0x5_1100,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_or_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.and_const(
                ctx.mem32(ctx.register(0), 0),
                0x5_0130,
            ),
            ctx.mem16(ctx.register(1), 0),
        ),
        0x5_01ff,
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0),
            0x5_0130,
        ),
        ctx.and_const(
            ctx.mem16(ctx.register(1), 0),
            0x1ff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_or_consistency3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 1),
                8,
            ),
            ctx.mem8(ctx.register(1), 0),
        ),
        0x9eff,
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 1),
                8,
            ),
            0x9e00,
        ),
        ctx.mem8(ctx.register(1), 0),
    );
    let eq2 = ctx.or(
        ctx.and_const(
            ctx.mem16(ctx.register(0), 0),
            0x9e00,
        ),
        ctx.mem8(ctx.register(1), 0),
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1, eq2);
}

#[test]
fn masked_shift_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.lsh_const(
            ctx.mem8(ctx.register(0), 1),
            8,
        ),
        0x9eff,
    );
    let eq1 = ctx.and_const(
        ctx.mem16(ctx.register(0), 0),
        0x9e00,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_or_consistency4() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 1),
                8,
            ),
            ctx.lsh_const(
                ctx.and_const(
                    ctx.register(3),
                    0x0280_0100,
                ),
                1,
            ),
        ),
        0x0701_9e00,
    );
    let eq1 = ctx.or(
        ctx.lsh_const(
            ctx.and_const(
                ctx.register(3),
                0x0280_0100,
            ),
            1,
        ),
        ctx.lsh_const(
            ctx.and_const(
                ctx.mem8(ctx.register(0), 1),
                0x9e,
            ),
            8,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_or_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.mem8(ctx.register(0), 1),
                8,
            ),
            ctx.and(
                ctx.register(0),
                ctx.mul_const(
                    ctx.register(0),
                    0x0800_04000,
                ),
            ),
        ),
        0xc701_9e00,
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.and(
                ctx.register(0),
                ctx.mul_const(
                    ctx.register(0),
                    0x0800_04000,
                ),
            ),
            0xc701_9c00,
        ),
        ctx.lsh_const(
            ctx.and_const(
                ctx.mem8(ctx.register(0), 1),
                0x9e,
            ),
            8,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xor_with_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.and_const(
            ctx.lsh_const(
                ctx.mem8(ctx.register(2), 1),
                1,
            ),
            0x77,
        ),
        ctx.constant(0x37),
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.lsh_const(
                ctx.mem8(ctx.register(2), 1),
                1,
            ),
            ctx.constant(0x37),
        ),
        0x77,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_or_with_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.lsh_const(
                ctx.mem8(ctx.register(2), 1),
                1,
            ),
            0x77,
        ),
        ctx.constant(0x37),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.lsh_const(
                ctx.mem8(ctx.register(2), 1),
                1,
            ),
            ctx.constant(0x37),
        ),
        0x77,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn eq_always_zero_with_sub() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.and_const(
            ctx.sub(
                ctx.const_0(),
                ctx.lsh_const(
                    ctx.register(5),
                    0x18,
                ),
            ),
            0xffff_ffff,
        ),
        0x1,
    );
    let eq1 = ctx.const_0();
    assert_eq!(op1, eq1);
}

#[test]
fn or_xor_same_with_mask() {
    let ctx = &OperandContext::new();
    // (x ^ y) | x => x | y,
    // the mask won't affect Mem8 here
    let op1 = ctx.or(
        ctx.and_const(
            ctx.xor(
                ctx.mem8(ctx.register(1), 0),
                ctx.register(0),
            ),
            0x00ff_33ff_00ff_33ff,
        ),
        ctx.mem8(ctx.register(1), 0),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.mem8(ctx.register(1), 0),
            ctx.register(0),
        ),
        0x00ff_33ff_00ff_33ff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_xor_same_with_mask2() {
    let ctx = &OperandContext::new();
    // ((x ^ y) & z) | x => x | (y & z),
    let op1 = ctx.or(
        ctx.and_const(
            ctx.xor(
                ctx.register(2),
                ctx.register(0),
            ),
            0x00ff_33ff_00ff_33ff,
        ),
        ctx.register(2),
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.register(0),
            0x00ff_33ff_00ff_33ff,
        ),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_xor_same_with_mask3() {
    let ctx = &OperandContext::new();
    // ((x ^ y) & z) | x => x | (y & z),
    // (Even if the and isn't with a constant mask)
    let op1 = ctx.or(
        ctx.and(
            ctx.and(
                ctx.xor(
                    ctx.register(2),
                    ctx.register(0),
                ),
                ctx.constant(0x00ff_33ff_00ff_33ff),
            ),
            ctx.register(5),
        ),
        ctx.register(2),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.and(
                ctx.register(0),
                ctx.constant(0x00ff_33ff_00ff_33ff),
            ),
            ctx.register(5),
        ),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_masked_or1() {
    let ctx = &OperandContext::new();
    // ((x | y) & z) | x => x | (y & z),
    // (Even if the and isn't with a constant mask)
    let op1 = ctx.or(
        ctx.and(
            ctx.or(
                ctx.register(2),
                ctx.register(0),
            ),
            ctx.constant(0x00ff_33ff_00ff_33ff),
        ),
        ctx.register(2),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.register(0),
            ctx.constant(0x00ff_33ff_00ff_33ff),
        ),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_masked_or2() {
    let ctx = &OperandContext::new();
    // ((x | y) & z) | x => x | (y & z),
    // (Even if the and isn't with a constant mask)
    let op1 = ctx.or(
        ctx.and(
            ctx.and(
                ctx.or(
                    ctx.register(2),
                    ctx.register(0),
                ),
                ctx.constant(0x00ff_33ff_00ff_33ff),
            ),
            ctx.register(5),
        ),
        ctx.register(2),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.and(
                ctx.register(0),
                ctx.constant(0x00ff_33ff_00ff_33ff),
            ),
            ctx.register(5),
        ),
        ctx.register(2),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn multiple_or_and() {
    let ctx = &OperandContext::new();
    // (r0 & r1) | ((r0 & r9) | r8)
    // => ((r0 & r1) | (r0 & r9)) | r8
    // => (r0 & (r1 | r9)) | r8
    let op1 = ctx.or(
        ctx.and(
            ctx.register(1),
            ctx.register(0),
        ),
        ctx.or(
            ctx.and(
                ctx.register(9),
                ctx.register(0),
            ),
            ctx.register(8),
        ),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.register(0),
            ctx.or(
                ctx.register(1),
                ctx.register(9),
            ),
        ),
        ctx.register(8),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_or_consistency6() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.mem16(ctx.register(0), 0),
            0x9e00,
        ),
        ctx.and_const(
            ctx.register(2),
            0xc79e_0100,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.and_const(
                ctx.mem16(ctx.register(0), 0),
                0x9e00,
            ),
            ctx.and_const(
                ctx.register(2),
                0xc79e_0100,
            ),
        ),
        0xc79e_9f00,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn multiple_or_and2() {
    let ctx = &OperandContext::new();
    // (r15 & r0) | (r0 & r1) | r15
    // => ((r15 | r1) & r0) | r15
    // => (r1 & r0) | r15
    let op1 = ctx.or(
        ctx.and(
            ctx.register(15),
            ctx.register(0),
        ),
        ctx.or(
            ctx.and(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.register(15),
        ),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.register(1),
            ctx.register(0),
        ),
        ctx.register(15),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn multiple_or_and3() {
    let ctx = &OperandContext::new();
    // (r0 & 53) | (r0 & r1) | r15
    // => ((r1 | 53) & r0) | r15
    let op1 = ctx.or(
        ctx.and(
            ctx.register(0),
            ctx.constant(0x53),
        ),
        ctx.or(
            ctx.and(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.register(15),
        ),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.or(
                ctx.register(1),
                ctx.constant(0x53),
            ),
            ctx.register(0),
        ),
        ctx.register(15),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_mem_high_byte() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or_const(
            ctx.mem32(ctx.register(0), 0),
            0x0000_00c7_01ea_2000,
        ),
        0x0006_0000_0600_0000,
    );
    let eq1 = ctx.and_const(
        ctx.or_const(
            ctx.lsh_const(
                ctx.and_const(
                    ctx.mem8(ctx.register(0), 3),
                    0x6,
                ),
                0x18,
            ),
            0x0000_0001_0000_0000,
        ),
        0xffff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_add_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.add_const(
                ctx.mem16(ctx.register(3), 0),
                0x0016_0005_0200,
            ),
            ctx.register(4),
        ),
        0x0701_0763_c700,
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.add_const(
                ctx.mem16(ctx.register(3), 0),
                0x0016_0005_0200,
            ),
            0x0701_0763_c700,
        ),
        ctx.and_const(
            ctx.register(4),
            0x0701_0763_c700,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn xor_verify_no_mask_remove() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.xor(
            ctx.xor(
                ctx.and_const(
                    ctx.register(0),
                    0x1ff,
                ),
                ctx.and_const(
                    ctx.register(1),
                    0xff,
                ),
            ),
            ctx.register(9),
        ),
        0xffff,
    );
    let ne1 = ctx.and_const(
        ctx.xor(
            ctx.xor(
                ctx.register(0),
                ctx.and_const(
                    ctx.register(1),
                    0xff,
                ),
            ),
            ctx.register(9),
        ),
        0xffff,
    );
    assert_ne!(op1, ne1);
}

#[test]
fn and_or_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.or(
                ctx.register(15),
                ctx.and(
                    ctx.register(0),
                    ctx.register(1),
                ),
            ),
            ctx.register(6),
        ),
        ctx.register(15),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.and(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.register(6),
        ),
        ctx.register(15),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn unnecessary_and_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.and(
                ctx.or_const(
                    ctx.mem16(ctx.register(3), 0),
                    0x0600_0000,
                ),
                ctx.xmm(0, 0),
            ),
            ctx.register(0),
        ),
        0x1000_c79e_0100,
    );
    let high = ctx.or(
        op1,
        ctx.lsh_const(
            ctx.register(0),
            0x20,
        ),
    );
    let x = ctx.and_const(high, 0xffff_ffff);
    let y = ctx.and_const(op1, 0xffff_ffff);
    assert_eq!(x, y);
}

#[test]
fn simplify_mul_const_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.mul_const(
            ctx.register(1),
            0x8000_0000_0000_0200,
        ),
        0x0010_0000,
    );
    let eq1 = ctx.and_const(
        ctx.mul_const(
            ctx.register(1),
            0x0200,
        ),
        0x0010_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn sub_incorrect_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                0x5_0600_0000,
            ),
            0x2_fa00_0000,
        ),
        0x0107_c79e_0100,
    );
    let op1 = ctx.and_const(op1, 0xffff_ffff);
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                0x0600_0000,
            ),
            0xfa00_0000,
        ),
        0xc79e_0100,
    );
    // Somewhat unexpectedly this ends up being equal too
    // (Can be verified by hand since (rax & 0x0600_0000) has only 4 different values it can have
    let eq2 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                0x0600_0000,
            ),
            0x0200_0000,
        ),
        0x0600_0000,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1, eq2);
}

#[test]
fn complex_or_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and_const(
            ctx.or(
                ctx.and_const(
                    ctx.lsh_const(
                        ctx.add_const(
                            ctx.register(0),
                            1,
                        ),
                        9,
                    ),
                    0x0600_0000,
                ),
                ctx.mem16(ctx.register(0), 0),
            ),
            0x0400_0100,
        ),
        ctx.and_const(
            ctx.or(
                ctx.and_const(
                    ctx.register(0),
                    0x0600_0000,
                ),
                ctx.mem16(ctx.register(15), 0),
            ),
            0x0400_0100,
        ),
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.or(
                ctx.and_const(
                    ctx.lsh_const(
                        ctx.add_const(
                            ctx.register(0),
                            1,
                        ),
                        9,
                    ),
                    0x0600_0000,
                ),
                ctx.mem16(ctx.register(0), 0),
            ),
            ctx.or(
                ctx.and_const(
                    ctx.register(0),
                    0x0600_0000,
                ),
                ctx.mem16(ctx.register(15), 0),
            ),
        ),
        0x0400_0100,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn add_cant_remove_inner_mask() {
    let ctx = &OperandContext::new();
    // E.g. 0f0f + 0ff0 = 1eff,
    // but 0f0f + 0fff = 1f0e
    let op1 = ctx.and_const(
        ctx.add(
            ctx.mem16(ctx.register(0), 0),
            ctx.and_const(
                ctx.mem16(ctx.register(2), 0),
                0xfff0,
            ),
        ),
        0xff00,
    );
    let ne1 = ctx.and_const(
        ctx.add(
            ctx.mem16(ctx.register(0), 0),
            ctx.mem16(ctx.register(2), 0),
        ),
        0xff00,
    );
    assert_ne!(op1, ne1);
}

#[test]
fn masked_sub_remove_inner_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                0x0600_0000,
            ),
            0x0200_0000,
        ),
        0x0600_0000,
    );
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.register(0),
            0x0200_0000,
        ),
        0x0600_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_sub_consistency() {
    let ctx = &OperandContext::new();
    // rhs of the subtraction has bit 0x1 known be 0, so it won't propagate up
    // and mask can be moved out.
    let op1 = ctx.and_const(
        ctx.xor(
            ctx.sub(
                ctx.and_const(
                    ctx.register(0),
                    0xfffe,
                ),
                ctx.lsh_const(
                    ctx.register(8),
                    9,
                ),
            ),
            ctx.register(5),
        ),
        0xffff,
    );
    let eq1 = ctx.xor(
        ctx.and_const(
            ctx.sub(
                ctx.register(0),
                ctx.lsh_const(
                    ctx.register(8),
                    9,
                ),
            ),
            0xfffe,
        ),
        ctx.and_const(
            ctx.register(5),
            0xffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xor_consistency2() {
    let ctx = &OperandContext::new();
    // c will have mask fff0, but it can just use mask ffff
    // and mask can be moved out.
    let a = ctx.register(1);
    let b = ctx.and_const(
        ctx.register(0),
        0xfff0,
    );
    let c = ctx.sub(
        ctx.lsh_const(
            ctx.register(8),
            8,
        ),
        ctx.lsh_const(
            ctx.register(4),
            4,
        ),
    );
    let op1 = ctx.and_const(
        ctx.xor(
            a,
            ctx.xor(b, c),
        ),
        0xffff,
    );
    let op2 = ctx.xor(
        b,
        ctx.and_const(
            ctx.xor(a, c),
            0xffff,
        ),
    );
    let op3 = ctx.xor(
        ctx.and_const(
            c,
            0xffff,
        ),
        ctx.and_const(
            ctx.xor(a, b),
            0xffff,
        ),
    );
    assert_eq!(op1, op2);
    assert_eq!(op1, op3);
}

#[test]
fn masked_add_sub_const_consistency() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.add_const(
                ctx.and_const(
                    ctx.register(0),
                    0x0600_0000,
                ),
                0x0600_0000,
            ),
            ctx.register(1),
        ),
        0x8700_0100,
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.and_const(
                ctx.sub_const(
                    ctx.and_const(
                        ctx.register(0),
                        0x0600_0000,
                    ),
                    0x0200_0000,
                ),
                0x0600_0000,
            ),
            ctx.register(1),
        ),
        0x8700_0100,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn add_no_overlapping_bits_useless_mask() {
    let ctx = &OperandContext::new();
    // Can't overflow past ffff_ffff since no bits overlap
    let op1 = ctx.and_const(
        ctx.add(
            ctx.and_const(
                ctx.register(0),
                0xffff_0000,
            ),
            ctx.mem16(ctx.register(0), 0),
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.add(
        ctx.and_const(
            ctx.register(0),
            0xffff_0000,
        ),
        ctx.mem16(ctx.register(0), 0),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_add_const_unnecessary_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.add_const(
            ctx.and_const(
                ctx.register(0),
                0xff_ffff,
            ),
            0xfc00_0000,
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.add_const(
        ctx.and_const(
            ctx.register(0),
            0xff_ffff,
        ),
        0xfc00_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_add_sub_const_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.add(
        ctx.register(1),
        ctx.and(
            ctx.xmm(0, 0),
            ctx.add_const(
                ctx.and_const(
                    ctx.register(0),
                    0xff_ffff,
                ),
                0x7c00_0000,
            ),
        ),
    );
    let eq1 = ctx.add(
        ctx.register(1),
        ctx.and_const(
            ctx.and(
                ctx.xmm(0, 0),
                ctx.sub_const(
                    ctx.and_const(
                        ctx.register(0),
                        0xff_ffff,
                    ),
                    0x0400_0000,
                ),
            ),
            0x7fff_ffff,
        ),
    );
    let ne1 = ctx.add(
        ctx.register(1),
        ctx.and(
            ctx.xmm(0, 0),
            ctx.sub_const(
                ctx.and_const(
                    ctx.register(0),
                    0xff_ffff,
                ),
                0x0400_0000,
            ),
        ),
    );
    assert_eq!(op1, eq1);
    assert_ne!(op1, ne1);
}

#[test]
fn masked_ors_useless_outer_mask() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.or(
            ctx.and_const(
                ctx.register(0),
                0x0100_00ff,
            ),
            ctx.and_const(
                ctx.register(1),
                0x1201_9e00,
            ),
        ),
        ctx.mem8(ctx.register(8), 0),
    );
    let eq1 = ctx.and_const(
        op1,
        0x1301_9eff,
    );
    assert_eq!(op1, eq1);
    // Check that op1 doesn't get masked since it doesn't need one
    assert!(op1.if_arithmetic_or().is_some(), "Should be or, was {op1}");
}

#[test]
fn shifts_causing_unnecessary_mask() {
    let ctx = &OperandContext::new();
    let base = ctx.xor(
        ctx.rsh_const(
            ctx.register(14),
            0x10,
        ),
        ctx.rsh_const(
            ctx.sub_const(
                ctx.mem32(ctx.register(4), 0),
                0x2e3913bd,
            ),
            0x10,
        ),
    );
    let op1 = ctx.or(
        ctx.lsh_const(
            ctx.and_const(
                base,
                0xffff,
            ),
            0x10,
        ),
        ctx.and_const(
            ctx.register(11),
            0xffff,
        ),
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.xor(
                ctx.register(14),
                ctx.sub_const(
                    ctx.mem32(ctx.register(4), 0),
                    0x2e3913bd,
                ),
            ),
            0xffff_0000,
        ),
        ctx.and_const(
            ctx.register(11),
            0xffff,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_xor_shifts_simplify() {
    let ctx = &OperandContext::new();
    let base = ctx.xor(
        ctx.rsh_const(
            ctx.register(14),
            0x10,
        ),
        ctx.rsh_const(
            ctx.sub_const(
                ctx.mem32(ctx.register(4), 0),
                0x2e3913bd,
            ),
            0x10,
        ),
    );
    let op1 = ctx.lsh_const(
        ctx.and_const(
            base,
            0xffff,
        ),
        0x10,
    );
    let eq1 = ctx.and_const(
        ctx.xor(
            ctx.register(14),
            ctx.sub_const(
                ctx.mem32(ctx.register(4), 0),
                0x2e3913bd,
            ),
        ),
        0xffff_0000,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn neq_0_flag() {
    let ctx = &OperandContext::new();
    let op1 = ctx.neq_const(
        ctx.flag_c(),
        0,
    );
    let eq1 = ctx.flag_c();
    assert_eq!(op1, eq1);

    let op1 = ctx.neq_const(
        ctx.and_const(
            ctx.register(0),
            1,
        ),
        0,
    );
    let eq1 = ctx.and_const(
        ctx.register(0),
        1,
    );
    assert_eq!(op1, eq1);

    let op1 = ctx.neq_const(
        ctx.or(
            ctx.flag_c(),
            ctx.flag_z(),
        ),
        0,
    );
    let eq1 = ctx.or(
        ctx.flag_c(),
        ctx.flag_z(),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn neq_0_flag_and_eq() {
    let ctx = &OperandContext::new();
    // ((o == s) & (z == 0)) == 0
    // is same as
    // (((o == s) == 0) | z
    let op1 = ctx.eq_const(
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
    let eq1 = ctx.or(
        ctx.neq(
            ctx.flag_o(),
            ctx.flag_s(),
        ),
        ctx.flag_z(),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn neq_0_reg_and_eq() {
    let ctx = &OperandContext::new();
    // ((o == s) & (r0 == 0)) == 0
    // should not change
    let op1 = ctx.eq_const(
        ctx.and(
            ctx.eq(
                ctx.flag_o(),
                ctx.flag_s(),
            ),
            ctx.eq_const(
                ctx.register(0),
                0,
            ),
        ),
        0,
    );
    let eq1 = ctx.or(
        ctx.neq(
            ctx.flag_o(),
            ctx.flag_s(),
        ),
        ctx.neq_const(
            ctx.register(0),
            0,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn neq_0_rsh_and_eq() {
    let ctx = &OperandContext::new();
    // ((o == s) & ((rax >> 3f) == 0)) == 0
    // is same as
    // (((o == s) == 0) | (rax >> 3f)
    let op1 = ctx.eq_const(
        ctx.and(
            ctx.eq(
                ctx.flag_o(),
                ctx.flag_s(),
            ),
            ctx.eq_const(
                ctx.rsh_const(
                    ctx.register(0),
                    0x3f,
                ),
                0,
            ),
        ),
        0,
    );
    let eq1 = ctx.or(
        ctx.neq(
            ctx.flag_o(),
            ctx.flag_s(),
        ),
        ctx.rsh_const(
            ctx.register(0),
            0x3f,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn rsh_3f_is_high_bit_eq_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.register(0),
        0x3f,
    );
    let eq1 = ctx.neq_const(
        ctx.and_const(
            ctx.register(0),
            0x8000_0000_0000_0000,
        ),
        0,
    );
    assert_eq!(op1, eq1);

    let op1 = ctx.rsh_const(
        ctx.mem32(ctx.register(0), 0),
        0x1f,
    );
    let eq1 = ctx.neq_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0),
            0x8000_0000,
        ),
        0,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn or_with_mem8_high_bit_add_eq_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq_const(
        ctx.or(
            ctx.eq_const(
                ctx.and_const(
                    ctx.add(
                        ctx.mem8(ctx.register(0), 0),
                        ctx.mem8(ctx.register(1), 0),
                    ),
                    0x80,
                ),
                0,
            ),
            ctx.eq(
                ctx.mem8(ctx.register(8), 0),
                ctx.mem8(ctx.register(9), 0),
            ),
        ),
        0,
    );
    let eq1 = ctx.and(
        ctx.neq_const(
            ctx.and_const(
                ctx.add(
                    ctx.mem8(ctx.register(0), 0),
                    ctx.mem8(ctx.register(1), 0),
                ),
                0x80,
            ),
            0,
        ),
        ctx.neq(
            ctx.mem8(ctx.register(8), 0),
            ctx.mem8(ctx.register(9), 0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_sub_zero_lhs() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub(
            ctx.and_const(
                ctx.register(0),
                0x2_0000_0000,
            ),
            ctx.mem16(ctx.register(1), 0),
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.and_const(
        ctx.sub(
            ctx.constant(0),
            ctx.mem16(ctx.register(1), 0),
        ),
        0xffff_ffff,
    );

    assert_eq!(op1, eq1);
}

#[test]
fn low_bit_with_or_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or_const(
        ctx.and_const(
            ctx.mem8(ctx.register(0), 0),
            1,
        ),
        0x7fff_fffe,
    );
    let eq1 = ctx.or_const(
        ctx.mem8(ctx.register(0), 0),
        0x7fff_fffe,
    );

    assert_eq!(op1, eq1);
}

#[test]
fn unnecessary_mask_from_sub_const() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                1,
            ),
            0x7f,
        ),
        0xffff_ffff,
    );
    let eq1 = ctx.add_const(
        ctx.and_const(
            ctx.register(0),
            1,
        ),
        0xffff_ff81,
    );

    assert_eq!(op1, eq1);
}

#[test]
fn simplify_masked_mul_const() {
    let ctx = &OperandContext::new();
    // Multiplication shifts everything by at least 8 to left,
    // making high bits of or const be past and mask.
    let op1 = ctx.and_const(
        ctx.mul_const(
            ctx.or_const(
                ctx.mem16(ctx.register(1), 0),
                0x10_0035_0005,
            ),
            0x500,
        ),
        0x10_0500_0000,
    );
    let eq1 = ctx.and_const(
        ctx.mul_const(
            ctx.or_const(
                ctx.mem16(ctx.register(1), 0),
                0x0035_0005,
            ),
            0x500,
        ),
        0x0500_0000,
    );

    assert_eq!(op1, eq1);
}

#[test]
fn equivalent_masked_shifted_memory() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.mem32(ctx.register(0), 0x101),
        0xff00_ff00,
    );
    let eq1 = ctx.lsh_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0x102),
            0xff00_ff,
        ),
        8,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn equivalent_masked_shifted_memory2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.mem64(ctx.register(0), 0x101),
        0x00ff_ff00_ff00,
    );
    let eq1 = ctx.lsh_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0x102),
            0xffff00_ff,
        ),
        8,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn equivalent_masked_shifted_memory3() {
    let ctx = &OperandContext::new();
    let op1 = ctx.lsh_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0x101),
            0xff00_ff00,
        ),
        4,
    );
    let eq1 = ctx.lsh_const(
        ctx.and_const(
            ctx.mem32(ctx.register(0), 0x102),
            0xff00_ff,
        ),
        12,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn eq_zero_xor_1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor_const(
        ctx.gt(
            ctx.register(1),
            ctx.register(0),
        ),
        1,
    );
    let eq1 = ctx.eq_const(
        ctx.gt(
            ctx.register(1),
            ctx.register(0),
        ),
        0,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn neq_xor_1() {
    let ctx = &OperandContext::new();
    let op1 = ctx.xor(
        ctx.gt(
            ctx.register(1),
            ctx.register(0),
        ),
        ctx.gt(
            ctx.register(2),
            ctx.register(4),
        ),
    );
    let eq1 = ctx.neq(
        ctx.gt(
            ctx.register(1),
            ctx.register(0),
        ),
        ctx.gt(
            ctx.register(2),
            ctx.register(4),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_xor() {
    let ctx = &OperandContext::new();
    // (x & y) ^ x => (x & !y)
    let x = ctx.xor(
        ctx.register(0),
        ctx.register(4),
    );
    let y = ctx.xor(
        ctx.register(3),
        ctx.register(9),
    );
    let op1 = ctx.xor(
        ctx.and(x, y),
        x,
    );
    let eq1 = ctx.and(
        x,
        ctx.xor_const(
            y,
            u64::MAX,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_xor_not() {
    let ctx = &OperandContext::new();
    // (!x & y) ^ x => (x | y)
    let x = ctx.xor(
        ctx.register(0),
        ctx.register(4),
    );
    let y = ctx.xor(
        ctx.register(3),
        ctx.register(9),
    );
    let op1 = ctx.xor(
        ctx.and(
            ctx.xor_const(x, u64::MAX),
            y,
        ),
        x,
    );
    let eq1 = ctx.or(x, y);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_xor_both() {
    let ctx = &OperandContext::new();
    // (x & y) ^ (x ^ y) => (x | y)
    let x = ctx.xor(
        ctx.register(0),
        ctx.register(4),
    );
    let y = ctx.xor(
        ctx.register(3),
        ctx.register(9),
    );
    let op1 = ctx.xor(
        ctx.and(x, y),
        ctx.xor(x, y),
    );
    let eq1 = ctx.or(x, y);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_xor_both2() {
    let ctx = &OperandContext::new();
    // (x & y) ^ (x ^ y) => (x | y)
    let x = ctx.xor(
        ctx.register(6),
        ctx.mem32(ctx.register(2), 0),
    );
    let y = ctx.xor(
        ctx.mem32(ctx.register(1), 0),
        ctx.mem32(ctx.register(4), 0),
    );
    let op1 = ctx.xor(
        ctx.and(x, y),
        ctx.xor(x, y),
    );
    let eq1 = ctx.or(x, y);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_xor() {
    let ctx = &OperandContext::new();
    // (x | y) ^ x => (!x & y)
    let x = ctx.xor(
        ctx.register(0),
        ctx.register(4),
    );
    let y = ctx.xor(
        ctx.register(3),
        ctx.register(9),
    );
    let op1 = ctx.xor(
        ctx.or(x, y),
        x,
    );
    let eq1 = ctx.and(
        ctx.xor_const(
            x,
            u64::MAX,
        ),
        y,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_xor_not2() {
    let ctx = &OperandContext::new();
    // (!x & y) ^ y => (x & y)
    let x = ctx.xor(
        ctx.register(0),
        ctx.register(4),
    );
    let y = ctx.xor(
        ctx.register(3),
        ctx.register(9),
    );
    let op1 = ctx.xor(
        ctx.and(
            ctx.xor_const(x, u64::MAX),
            y,
        ),
        y,
    );
    let eq1 = ctx.and(x, y);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_or_xor_both() {
    let ctx = &OperandContext::new();
    // (x | y) ^ (x ^ y) => (x & y)
    let x = ctx.xor(
        ctx.register(0),
        ctx.register(4),
    );
    let y = ctx.xor(
        ctx.register(3),
        ctx.register(9),
    );
    let op1 = ctx.xor(
        ctx.or(x, y),
        ctx.xor(x, y),
    );
    let eq1 = ctx.and(x, y);
    assert_eq!(op1, eq1);
}

#[test]
fn shifted_or_mem_nop() {
    let ctx = &OperandContext::new();
    let value = ctx.or(
        ctx.mem32(ctx.register(1), 1),
        ctx.register(6),
    );
    let high = ctx.and_const(value, 0xff00_0000);
    let high_shifted = ctx.rsh_const(high, 0x18);
    let high_shifted_back = ctx.lsh_const(high_shifted, 0x18);
    assert_eq!(high, high_shifted_back);
}

#[test]
fn shifted_or_mem_should_remove_shift() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.or(
            ctx.mem32(ctx.register(1), 1),
            ctx.mem32(ctx.register(5), 0),
        ),
        8,
    );
    let eq1 = ctx.and_const(
        ctx.or(
            ctx.mem32(ctx.register(1), 2),
            ctx.mem32(ctx.register(5), 1),
        ),
        0xff_ffff,
    );
    let eq2 = ctx.and_const(
        op1,
        0xff_ffff,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1, eq2);
}

#[test]
fn shifted_mem_should_remove_shift() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.mem32(ctx.register(1), 1),
        8,
    );
    let eq1 = ctx.and_const(
        ctx.mem32(ctx.register(1), 2),
        0xff_ffff
    );
    let eq2 = ctx.and_const(
        op1,
        0xff_ffff,
    );
    assert_eq!(op1, eq1);
    assert_eq!(op1, eq2);
}

#[test]
fn merge_or_xor_cant_merge() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.add_const(
            ctx.register(0),
            0x6600,
        ),
        0xff00,
    );
    let op2 = ctx.and_const(
        ctx.add_const(
            ctx.register(0),
            0x6610,
        ),
        0xffff_0000,
    );
    let or = ctx.or(op1, op2);
    let xor = ctx.xor(op1, op2);
    let ne1 = ctx.and_const(
        ctx.add_const(
            ctx.register(0),
            0x6610,
        ),
        0xffff_ff00,
    );
    let ne2 = ctx.and_const(
        ctx.add_const(
            ctx.register(0),
            0x6600,
        ),
        0xffff_ff00,
    );
    assert_ne!(or, ne1);
    assert_ne!(or, ne2);
    assert_ne!(xor, ne1);
    assert_ne!(xor, ne2);
}

#[test]
fn merge_or_xor_cant_merge_mul() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.mul_const(
            ctx.register(0),
            0x6600,
        ),
        0xff00,
    );
    let op2 = ctx.and_const(
        ctx.mul_const(
            ctx.register(0),
            0x6610,
        ),
        0xffff_0000,
    );
    let or = ctx.or(op1, op2);
    let xor = ctx.xor(op1, op2);
    let ne1 = ctx.and_const(
        ctx.mul_const(
            ctx.register(0),
            0x6610,
        ),
        0xffff_ff00,
    );
    let ne2 = ctx.and_const(
        ctx.mul_const(
            ctx.register(0),
            0x6600,
        ),
        0xffff_ff00,
    );
    assert_ne!(or, ne1);
    assert_ne!(or, ne2);
    assert_ne!(xor, ne1);
    assert_ne!(xor, ne2);
}

#[test]
fn incorrect_and_simplify() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.or(
            ctx.and_const(
                ctx.sub_const(
                    ctx.and_const(
                        ctx.register(0),
                        0x40,
                    ),
                    0x6,
                ),
                0x7e,
            ),
            ctx.mem8(ctx.register(0), 0),
        ),
        0xf,
    );
    let eq1 = ctx.or(
        ctx.and_const(
            ctx.sub_const(
                ctx.constant(0),
                0x6,
            ),
            0xf,
        ),
        ctx.and_const(
            ctx.mem8(ctx.register(0), 0),
            0xf,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn incorrect_rsh_simplfiy() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.or(
            ctx.or(
                ctx.lsh_const(
                    ctx.register(5),
                    0x38,
                ),
                ctx.and_const(
                    ctx.register(1),
                    0x00ff_00ff_00ff_00ff,
                ),
            ),
            ctx.constant(
                0x0000_0600_0400_0200,
            ),
        ),
        0x20,
    );
    let eq1 = ctx.or(
        ctx.or(
            ctx.rsh_const(
                ctx.lsh_const(
                    ctx.register(5),
                    0x38,
                ),
                0x20,
            ),
            ctx.and_const(
                ctx.rsh_const(
                    ctx.register(1),
                    0x20,
                ),
                0x00ff_00ff,
            ),
        ),
        ctx.constant(
            0x0000_0600,
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn same_comparison_eq() {
    let ctx = &OperandContext::new();
    let cmp = ctx.gt_const(
        ctx.register(0),
        0x5000,
    );
    let op1 = ctx.or(cmp, cmp);
    let eq1 = cmp;
    assert_eq!(op1, eq1);
}

#[test]
fn greater_or_equal_with_add() {
    let ctx = &OperandContext::new();
    // (x + 777 > y) | (y - x == 777)
    // => (x + 778 > y)
    let op1 = ctx.or(
        ctx.gt(
            ctx.add_const(
                ctx.mem64c(0x3000),
                0x777,
            ),
            ctx.mul(
                ctx.mem32(ctx.register(0), 0),
                ctx.mem32(ctx.register(1), 0),
            ),
        ),
        ctx.eq(
            ctx.constant(0x777),
            ctx.sub(
                ctx.mul(
                    ctx.mem32(ctx.register(0), 0),
                    ctx.mem32(ctx.register(1), 0),
                ),
                ctx.mem64c(0x3000),
            ),
        ),
    );
    let eq1 = ctx.gt(
        ctx.add_const(
            ctx.mem64c(0x3000),
            0x778,
        ),
        ctx.mul(
            ctx.mem32(ctx.register(0), 0),
            ctx.mem32(ctx.register(1), 0),
        ),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn invalid_demorgan_bug() {
    let ctx = &OperandContext::new();
    let left = ctx.eq_const(
        ctx.and_const(
            ctx.register(0),
            0xffff_ffff,
        ),
        0,
    );
    let right = ctx.neq_const(
        ctx.and_const(
            ctx.register(0),
            0x8000_0000,
        ),
        0,
    );
    let or = ctx.or(left, right);
    assert_eq!(
        ctx.substitute(left, ctx.register(0), ctx.constant(2), 8),
        ctx.constant(0),
        "{left}",
    );
    assert_eq!(
        ctx.substitute(right, ctx.register(0), ctx.constant(2), 8),
        ctx.constant(0),
        "{right}",
    );
    assert_eq!(
        ctx.substitute(or, ctx.register(0), ctx.constant(2), 8),
        ctx.constant(0),
        "{or}",
    );
}

#[test]
fn valid_demorgan_and_to_or() {
    let ctx = &OperandContext::new();
    let left = ctx.eq_const(
        ctx.and_const(
            ctx.register(1),
            0xffff_ffff,
        ),
        0,
    );
    let right = ctx.neq_const(
        ctx.and_const(
            ctx.register(0),
            0x8000_0000,
        ),
        0,
    );
    let and = ctx.and(left, right);
    // Unlike invalid_demorgan_bug test, where
    // (x_32bit == 0 | y_1bit) can't be converted to (x_32bit & !y_1bit) == 0
    // here (x_32bit == 0 & y_1bit) can be converted to (x_32bit | !y_1bit) == 0.
    let eq1 = ctx.eq_const(
        ctx.or(
            ctx.and_const(
                ctx.register(1),
                0xffff_ffff,
            ),
            ctx.eq_const(
                ctx.and_const(
                    ctx.register(0),
                    0x8000_0000,
                ),
                0,
            ),
        ),
        0,
    );
    assert_eq!(and, eq1);

    let left2 = ctx.eq_const(
        ctx.or(
            ctx.or(
                ctx.eq(
                    ctx.flag_o(),
                    ctx.flag_s(),
                ),
                ctx.flag_c(),
            ),
            ctx.flag_z(),
        ),
        0,
    );
    let right2 = ctx.eq_const(
        ctx.register(0),
        0,
    );
    let and2 = ctx.and(left2, right2);
    let eq2 = ctx.eq_const(
        ctx.or(
            ctx.register(0),
            ctx.or(
                ctx.or(
                    ctx.eq(
                        ctx.flag_o(),
                        ctx.flag_s(),
                    ),
                    ctx.flag_c(),
                ),
                ctx.flag_z(),
            ),
        ),
        0,
    );
    assert_eq!(and2, eq2);

    // Same but with different ordering
    // (rax == 0) & ((o == s) == 0) & ((z | c) == 0)
    let op3 = ctx.and(
        ctx.and(
            ctx.eq_const(
                ctx.register(0),
                0,
            ),
            ctx.eq_const(
                ctx.eq(
                    ctx.flag_o(),
                    ctx.flag_s(),
                ),
                0,
            ),
        ),
        ctx.eq_const(
            ctx.or(
                ctx.flag_c(),
                ctx.flag_z(),
            ),
            0,
        ),
    );
    assert_eq!(op3, eq2);
}

#[test]
fn not_gt_is_le() {
    let ctx = &OperandContext::new();
    let a = ctx.and_const(ctx.register(0), 0xff);
    let b = ctx.mem8(ctx.register(6), 0);
    let op1 = ctx.or(
        ctx.gt(a, b),
        ctx.eq(a, b),
    );
    let eq1 = ctx.eq_const(
        ctx.gt(b, a),
        0,
    );
    assert_eq!(op1, eq1);

    let op1 = ctx.eq_const(
        ctx.or(
            ctx.gt(a, b),
            ctx.eq(a, b),
        ),
        0,
    );
    let eq1 = ctx.gt(b, a);
    assert_eq!(op1, eq1);
}

#[test]
fn select_same() {
    let ctx = &OperandContext::new();
    let op1 = ctx.select(
        ctx.register(1),
        ctx.register(2),
        ctx.register(2),
    );
    let eq1 = ctx.register(2);
    assert_eq!(op1, eq1);
}

#[test]
fn asr_to_shr() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and_const(
        ctx.rsh_const(
            ctx.register(1),
            1,
        ),
        0x7fff_ffff,
    );
    let eq1 = ctx.and_const(
        ctx.arithmetic_right_shift(
            ctx.register(1),
            ctx.constant(1),
            MemAccessSize::Mem32,
        ),
        0x7fff_ffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_shifted_memory() {
    let ctx = &OperandContext::new();
    let op1 = ctx.rsh_const(
        ctx.and(
            ctx.lsh_const(
                ctx.mem32(ctx.register(0), 0x100),
                0x20,
            ),
            ctx.lsh_const(
                ctx.mem32(ctx.register(0), 0x200),
                0x20,
            ),
        ),
        0x20,
    );
    let eq1 = ctx.and(
        ctx.mem32(ctx.register(0), 0x100),
        ctx.mem32(ctx.register(0), 0x200),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn masked_and_add_to_mul() {
    let ctx = &OperandContext::new();
    let dh = ctx.and_const(
        ctx.register(2),
        0xff00,
    );
    let op1 = ctx.and_const(
        ctx.add(
            ctx.and_const(
                ctx.add(
                    ctx.register(1),
                    dh,
                ),
                0xffff,
            ),
            dh,
        ),
        0xffff,
    );
    let eq1 = ctx.arithmetic_masked(
        ArithOpType::Add,
        ctx.and_const(
            ctx.add(
                ctx.register(1),
                dh,
            ),
            0xffff,
        ),
        dh,
        0xffff,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_or_with_reverting_variable_mask() {
    let ctx = &OperandContext::new();
    // (x ^ ffff_ffff) & ((x & y) | z)
    // where x doesn't have high bits set can be simplified to
    // (x ^ ffff_ffff) & z
    let x = ctx.mem32(ctx.register(1), 0);
    let y = ctx.register(5);
    let z = ctx.register(3);
    let op1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        ctx.or(
            ctx.and(x, y),
            z,
        ),
    );
    let eq1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        z,
    );
    assert_eq!(op1, eq1);
    // x ^ ffff_ffff inside
    let x = ctx.xor_const(x, 0xffff_ffff);
    let op1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        ctx.or(
            ctx.and(x, y),
            z,
        ),
    );
    let eq1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        z,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn and_with_reverting_variable_mask() {
    let ctx = &OperandContext::new();
    // (x ^ ffff_ffff) & x == 0
    let x = ctx.mem32(ctx.register(1), 0);
    let op1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        x,
    );
    let eq1 = ctx.const_0();
    assert_eq!(op1, eq1);
}

#[test]
fn and_or_with_reverting_variable_mask2() {
    let ctx = &OperandContext::new();
    // (x ^ ffff_ffff) & ((x & y) | z)
    // where x has high bits set can be simplified to
    // (x ^ ffff_ffff) & ((x & ffff_ffff_0000_0000 & y) | z)
    // which is not necessarily worth it (maybe when simplify_with_and_mask is used?),
    // but if y & ffff_ffff_0000_0000 == 0 then it can be simplified to
    // (x ^ ffff_ffff) & z without any extra context
    let x = ctx.register(1);
    let y = ctx.mem32(ctx.register(5), 0);
    let z = ctx.register(3);
    let op1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        ctx.or(
            ctx.and(x, y),
            z,
        ),
    );
    let eq1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        z,
    );
    assert_eq!(op1, eq1);
    // x ^ ffff_ffff inside
    let x = ctx.xor_const(x, 0xffff_ffff);
    let op1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        ctx.or(
            ctx.and(x, y),
            z,
        ),
    );
    let eq1 = ctx.and(
        ctx.xor_const(x, 0xffff_ffff),
        z,
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_high_constant() {
    let ctx = &OperandContext::new();
    // 9 > (x + 9)
    // is true for x = -1 ..= -9
    // So x > ffff...fff6
    let x = ctx.register(1);
    let op1 = ctx.gt(
        ctx.constant(9),
        ctx.add_const(x, 9),
    );
    let eq1 = ctx.gt(
        x,
        ctx.constant(0xffff_ffff_ffff_fff6),
    );
    assert_eq!(op1, eq1);

    // Similarly 9 > ((x + 9) & ffff_ffff)
    // is true for (x & ffff_ffff) = -1 ..= -9
    // So x & ffff_ffff > ffff_fff6
    let op1 = ctx.gt(
        ctx.constant(9),
        ctx.and_const(ctx.add_const(x, 9), 0xffff_ffff),
    );
    let eq1 = ctx.gt(
        ctx.and_const(x, 0xffff_ffff),
        ctx.constant(0xffff_fff6),
    );
    assert_eq!(op1, eq1);
}
