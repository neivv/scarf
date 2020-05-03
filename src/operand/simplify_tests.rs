use super::*;

fn check_simplification_consistency<'e>(ctx: OperandCtx<'e>, op: Operand<'e>) {
    let config = bincode::config();
    let bytes = config.serialize(&op).unwrap();
    let back: Operand<'e> = config.deserialize_seed(ctx.deserialize_seed(), &bytes).unwrap();
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
                        ctx.mem8(ctx.register(3)),
                        ctx.constant(8),
                    ),
                ),
                ctx.constant(0xffffff00),
            ),
            ctx.mem8(ctx.register(2)),
        ),
        ctx.constant(0xffff00ff),
    );
    let op2 = ctx.and(
        ctx.or(
            ctx.and(
                ctx.register(4),
                ctx.constant(0xffffff00),
            ),
            ctx.mem8(ctx.register(2)),
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
        ctx.mem8(ctx.register(2)),
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
            ctx.mem32(ctx.register(4)),
            ctx.constant(0xffff0000),
        ),
        ctx.and(
            ctx.mem32(ctx.register(4)),
            ctx.constant(0x0000ffff),
        )
    );
    let op2 = ctx.or(
        ctx.and(
            ctx.mem32(ctx.register(4)),
            ctx.constant(0xffff00ff),
        ),
        ctx.and(
            ctx.mem32(ctx.register(4)),
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
    assert_eq!(op1, ctx.mem32(ctx.register(4)));
    assert_eq!(op2, ctx.mem32(ctx.register(4)));
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
            ctx.mem32(ctx.register(1)),
            ctx.constant(0x10),
        ),
        ctx.constant(0xffff),
    );
    let eq = ctx.rsh(
        ctx.mem32(ctx.register(1)),
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
        ctx.mem32(ctx.constant(0x123456)),
    );
    let eq = ctx.mem16(ctx.constant(0x123456));
    let op2 = ctx.and(
        ctx.constant(0xfff),
        ctx.mem32(ctx.constant(0x123456)),
    );
    let eq2 = ctx.and(
        ctx.constant(0xfff),
        ctx.mem16(ctx.constant(0x123456)),
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
                        ctx.mem16(ctx.register(2)),
                    ),
                ),
                ctx.and(
                    ctx.constant(0xff),
                    ctx.xor(
                        ctx.xor(
                            ctx.constant(0xa6),
                            ctx.register(1),
                        ),
                        ctx.mem8(ctx.register(2)),
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
                    ctx.mem16(ctx.register(2)),
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
            ctx.mem32(ctx.register(1)),
            ctx.constant(0xffff0000),
        ),
        ctx.constant(0x10),
    );
    let eq5 = ctx.rsh(
        ctx.mem32(ctx.register(1)),
        ctx.constant(0x10),
    );
    let op6 = ctx.rsh(
        ctx.and(
            ctx.mem32(ctx.register(1)),
            ctx.constant(0xffff1234),
        ),
        ctx.constant(0x10),
    );
    let eq6 = ctx.rsh(
        ctx.mem32(ctx.register(1)),
        ctx.constant(0x10),
    );
    let op7 = ctx.and(
        ctx.lsh(
            ctx.and(
                ctx.mem32(ctx.constant(1)),
                ctx.constant(0xffff),
            ),
            ctx.constant(0x10),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq7 = ctx.and(
        ctx.lsh(
            ctx.mem32(ctx.constant(1)),
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
            ctx.mem32(ctx.constant(1)),
            ctx.constant(0xffff),
        ),
        ctx.constant(0x10),
    );
    let ne12 = ctx.lsh(
        ctx.mem32(ctx.constant(1)),
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
            ctx.mem16(ctx.register(0)),
            ctx.lsh(
                ctx.mem16(ctx.register(1)),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.mem16(ctx.register(1));
    let op2 = ctx.and(
        ctx.or(
            ctx.mem16(ctx.register(0)),
            ctx.lsh(
                ctx.mem16(ctx.register(1)),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xffff0000),
    );
    let eq2 = ctx.lsh(
        ctx.mem16(ctx.register(1)),
        ctx.constant(0x10),
    );
    let op3 = ctx.and(
        ctx.or(
            ctx.mem16(ctx.register(0)),
            ctx.lsh(
                ctx.mem16(ctx.register(1)),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xffff),
    );
    let eq3 = ctx.mem16(ctx.register(0));
    let op4 = ctx.or(
        ctx.or(
            ctx.mem16(ctx.register(0)),
            ctx.lsh(
                ctx.mem16(ctx.register(1)),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xffff0000),
    );
    let eq4 = ctx.or(
        ctx.mem16(ctx.register(0)),
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
        ctx.mem8(ctx.register(1)),
        ctx.and(
            ctx.mem16(ctx.register(1)),
            ctx.constant(0xff00),
        ),
    );
    let eq1 = ctx.mem16(ctx.register(1));
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
                    ctx.mem32(ctx.constant(1234)),
                    ctx.constant(0x1123),
                ),
                ctx.mem32(ctx.constant(3333)),
            ),
            ctx.constant(0xff00),
        ),
        ctx.and(
            ctx.xor(
                ctx.xor(
                    ctx.mem32(ctx.constant(1234)),
                    ctx.constant(0x666666),
                ),
                ctx.mem32(ctx.constant(3333)),
            ),
            ctx.constant(0xff),
        ),
    );
    let eq = ctx.and(
        ctx.constant(0xffff),
        ctx.xor(
            ctx.xor(
                ctx.mem16(ctx.constant(1234)),
                ctx.mem16(ctx.constant(3333)),
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
            ctx.mem32(ctx.register(1)),
            ctx.mem32(ctx.register(2)),
        ),
    );
    let eq1 = ctx.eq(
        ctx.mem32(ctx.register(1)),
        ctx.mem32(ctx.register(2)),
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
        ctx.mem8(ctx.constant(555)),
        ctx.and(
            ctx.add(
                ctx.or(
                    ctx.and(
                        ctx.register(2),
                        ctx.constant(0xffffff00),
                    ),
                    ctx.mem8(ctx.constant(555)),
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
            ctx.mem32(ctx.constant(0x123)),
            0xffff_ffff_ffff_ffff,
        ),
        0xffff,
    );
    let eq1 = ctx.and_const(
        ctx.xor_const(
            ctx.mem16(ctx.constant(0x123)),
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
        ctx.mem32(ctx.constant(0x123)),
        ctx.constant(0x10),
    );
    let eq1 = ctx.mem16(ctx.constant(0x125));
    let op2 = ctx.rsh(
        ctx.mem32(ctx.constant(0x123)),
        ctx.constant(0x11),
    );
    let eq2 = ctx.rsh(
        ctx.mem16(ctx.constant(0x125)),
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
            ctx.mem32(
                ctx.add(
                    ctx.register(0),
                    ctx.constant(0x120),
                ),
            ),
            ctx.constant(0x8),
        ),
        ctx.lsh(
            ctx.mem32(
                ctx.add(
                    ctx.register(0),
                    ctx.constant(0x124),
                ),
            ),
            ctx.constant(0x18),
        ),
    );
    let eq1 = ctx.mem32(
        ctx.add(
            ctx.register(0),
            ctx.constant(0x121),
        ),
    );
    let op2 = ctx.or(
        ctx.mem16(
            ctx.add(
                ctx.register(0),
                ctx.constant(0x122),
            ),
        ),
        ctx.lsh(
            ctx.mem16(
                ctx.add(
                    ctx.register(0),
                    ctx.constant(0x124),
                ),
            ),
            ctx.constant(0x10),
        ),
    );
    let eq2 = ctx.mem32(
        ctx.add(
            ctx.register(0),
            ctx.constant(0x122),
        ),
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
                        ctx.mem32(ctx.register(1)),
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
                ctx.mem32(ctx.register(1)),
                ctx.constant(0x18),
            ),
            ctx.rsh(
                ctx.mem32(ctx.register(4)),
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
            ctx.mem32(
                ctx.register(1),
            ),
            ctx.constant(0x8),
        ),
        ctx.lsh(
            ctx.mem8(
                ctx.add(
                    ctx.constant(0x4),
                    ctx.register(1),
                ),
            ),
            ctx.constant(0x18),
        ),
    );
    let eq1 = ctx.mem32(
        ctx.add(
            ctx.register(1),
            ctx.constant(1),
        ),
    );
    let op2 = ctx.or(
        ctx.rsh(
            ctx.mem32(
                ctx.sub(
                    ctx.register(1),
                    ctx.constant(0x4),
                ),
            ),
            ctx.constant(0x8),
        ),
        ctx.lsh(
            ctx.mem8(
                ctx.register(1),
            ),
            ctx.constant(0x18),
        ),
    );
    let eq2 = ctx.mem32(
        ctx.sub(
            ctx.register(1),
            ctx.constant(3),
        ),
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
                        ctx.mem8(ctx.register(1)),
                        ctx.constant(7),
                    ),
                    ctx.and(
                        ctx.constant(0xff000000),
                        ctx.lsh(
                            ctx.mem8(ctx.register(2)),
                            ctx.constant(0x11),
                        ),
                    ),
                ),
                ctx.constant(0x10),
            ),
            ctx.lsh(
                ctx.mem32(ctx.register(4)),
                ctx.constant(0x10),
            ),
        ),
        ctx.constant(0xff),
    );
    let eq1 = ctx.and(
        ctx.rsh(
            ctx.or(
                ctx.rsh(
                    ctx.mem8(ctx.register(1)),
                    ctx.constant(7),
                ),
                ctx.and(
                    ctx.constant(0xff000000),
                    ctx.lsh(
                        ctx.mem8(ctx.register(2)),
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
                    ctx.mem32(ctx.register(1)),
                ),
            ),
            ctx.and(
                ctx.constant(0xffff_f000),
                ctx.mem32(ctx.register(1)),
            ),
        )
    );
    let eq2 = ctx.mem32(ctx.register(1));
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
                ctx.mem32(ctx.sub(ctx.register(2), ctx.constant(0x1))),
                ctx.constant(8),
            ),
            ctx.constant(0x00ff_ffff),
        ),
        ctx.and(
            ctx.mem32(ctx.sub(ctx.register(2), ctx.constant(0x14))),
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
                            ctx.mem8(ctx.register(3)),
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
                ctx.lsh(ctx.mem16(ctx.register(5)), ctx.constant(0x10)),
                ctx.mem32(ctx.register(5)),
            ),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.xor(
        ctx.constant(0xffe6),
        ctx.xor(
            ctx.mem16(ctx.register(5)),
            ctx.rsh(ctx.mem32(ctx.register(5)), ctx.constant(0x10)),
        ),
    );
    let op2 = ctx.lsh(
        ctx.xor(
            ctx.constant(0xffe6),
            ctx.mem16(ctx.register(5)),
        ),
        ctx.constant(0x10),
    );
    let eq2 = ctx.xor(
        ctx.constant(0xffe60000),
        ctx.lsh(ctx.mem16(ctx.register(5)), ctx.constant(0x10)),
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
            ctx.mem32(ctx.register(1)),
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
            ctx.mem32(ctx.register(1)),
        ),
        ctx.constant(0x2),
    );
    let eq1 = ctx.mul(
        ctx.mem32(ctx.register(1)),
        ctx.constant(0x24),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn lea_mul_negative() {
    let ctx = &OperandContext::new();
    let base = ctx.sub(
        ctx.mem16(ctx.register(3)),
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
            ctx.mem16(ctx.register(3)),
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
        ctx.mem32(ctx.register(0)),
        ctx.constant(!0),
    );
    let eq2 = ctx.mem32(ctx.register(0));
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn short_and_is_32() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.mem32(ctx.register(0)),
        ctx.mem32(ctx.register(1)),
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
        ctx.mem32(ctx.register(1)),
    );
    let eq1 = ctx.mem32(ctx.register(1));
    assert_eq!(op1, eq1);
}

#[test]
fn mem8_mem32_shift_eq() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.constant(0xff),
        ctx.rsh(
            ctx.mem32(ctx.add(ctx.register(1), ctx.constant(0x4c))),
            ctx.constant(0x8),
        ),
    );
    let eq1 = ctx.mem8(ctx.add(ctx.register(1), ctx.constant(0x4d)));
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
        ctx.constant(u32::max_value() as u64),
    );
    let eq2 = ctx.constant(0);
    let op3 = ctx.gt(
        ctx.register(0),
        ctx.constant(u64::max_value()),
    );
    let eq3 = ctx.constant(0);
    let op4 = ctx.gt(
        ctx.register(0),
        ctx.constant(u32::max_value() as u64),
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
            ctx.mem32(unk),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq1 = ctx.xor(
        ctx.mem32(unk),
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
            ctx.mem32(ctx.constant(0x11230)),
            ctx.constant(8),
        ),
    );
    let eq1 = ctx.mem16(ctx.constant(0x11231));
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_unnecessary_shift_in_eq_zero() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.lsh(
            ctx.and(
                ctx.mem8(ctx.register(4)),
                ctx.constant(8),
            ),
            ctx.constant(0xc),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(4)),
            ctx.constant(8),
        ),
        ctx.constant(0),
    );
    let op2 = ctx.eq(
        ctx.rsh(
            ctx.and(
                ctx.mem8(ctx.register(4)),
                ctx.constant(8),
            ),
            ctx.constant(1),
        ),
        ctx.constant(0),
    );
    let eq2 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(4)),
            ctx.constant(8),
        ),
        ctx.constant(0),
    );
    let op3 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(4)),
            ctx.constant(8),
        ),
        ctx.constant(0),
    );
    let ne3 = ctx.eq(
        ctx.mem8(ctx.register(4)),
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
                ctx.mem8(ctx.constant(0x100)),
                ctx.constant(0xd),
            ),
            ctx.constant(0x1f0000),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.rsh(
        ctx.mem8(ctx.constant(0x100)),
        ctx.constant(0x3),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_set_bit_masked() {
    let ctx = &OperandContext::new();
    let op1 = ctx.or(
        ctx.and(
            ctx.mem16(ctx.constant(0x1000)),
            ctx.constant(0xffef),
        ),
        ctx.constant(0x10),
    );
    let eq1 = ctx.or(
        ctx.mem16(ctx.constant(0x1000)),
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
                ctx.mem32(ctx.constant(0x1000)),
                ctx.constant(9),
            ),
            ctx.constant(0x3fff_ffff),
        ),
        ctx.constant(0x2),
    );
    let eq1 = ctx.and(
        ctx.mul(
            ctx.mem32(ctx.constant(0x1000)),
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
            ctx.mem32(ctx.constant(0x5000)),
        ),
        ctx.constant(u64::max_value()),
    );
    let eq1 = ctx.mem32(ctx.constant(0x5000));
    let op2 = ctx.and(
        ctx.add(
            ctx.add(
                ctx.constant(1),
                ctx.mem32(ctx.constant(0x5000)),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq2 = ctx.mem32(ctx.constant(0x5000));
    let op3 = ctx.and(
        ctx.add(
            ctx.add(
                ctx.constant(1),
                ctx.mem32(ctx.constant(0x5000)),
            ),
            ctx.constant(0xffff_ffff),
        ),
        ctx.constant(0xffff_ffff),
    );
    let eq3 = ctx.mem32(ctx.constant(0x5000));
    let op4 = ctx.add(
        ctx.add(
            ctx.constant(1),
            ctx.mem32(ctx.constant(0x5000)),
        ),
        ctx.constant(0xffff_ffff_ffff_ffff),
    );
    let eq4 = ctx.mem32(ctx.constant(0x5000));
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
            ctx.const_ffffffff(),
        ),
        ctx.and(
            ud,
            ctx.const_ffffffff(),
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
            ctx.mem8(ctx.constant(0x900)),
            ctx.constant(0xf8),
        ),
        ctx.constant(3),
    );
    let eq1 = ctx.rsh(
        ctx.mem8(ctx.constant(0x900)),
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
                ctx.const_ffffffff(),
            ),
            ctx.constant(0x20),
        ),
        ctx.and(
            ctx.register(0),
            ctx.const_ffffffff(),
        ),
    );
    let eq1 = ctx.and(
        ctx.register(0),
        ctx.const_ffffffff(),
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
            ctx.mem16(ctx.register(2)),
            ctx.lsh(
                ctx.mem16(ctx.register(1)),
                ctx.constant(0x9),
            ),
        ),
        ctx.constant(0xffff),
    );
    let eq1 = ctx.rsh(
        ctx.and(
            ctx.lsh(
                ctx.sub(
                    ctx.mem16(ctx.register(2)),
                    ctx.lsh(
                        ctx.mem16(ctx.register(1)),
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
                ctx.mem8(ctx.register(2)),
                ctx.constant(0x8),
            ),
            ctx.constant(0x800),
        ),
        ctx.constant(0),
    );
    let eq1 = ctx.eq(
        ctx.and(
            ctx.mem8(ctx.register(2)),
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
                        ctx.mem8(ctx.constant(0x223345)),
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
            ctx.mem8(ctx.constant(0x223345)),
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
    let subst = ctx.substitute(op1, ctx.register(0), ctx.mem32(ctx.constant(0x1234)));
    let with_mem = ctx.and(
        ctx.or(
            ctx.rsh(
                ctx.mem32(ctx.constant(0x1234)),
                ctx.constant(0xb),
            ),
            ctx.lsh(
                ctx.mem32(ctx.constant(0x1234)),
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
        ctx.mem64(ctx.register(0)),
        ctx.mem8(ctx.register(0)),
    );
    let eq1 = ctx.mem8(ctx.register(0));
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
        ctx.mem64(ctx.register(0)),
        ctx.and(
            ctx.or(
                ctx.constant(0xfd0700002ff4004b),
                ctx.mem8(ctx.register(5)),
            ),
            ctx.constant(0x293b00be00),
        ),
    );
    let eq1 = ctx.eq(
        ctx.mem64(ctx.register(0)),
        ctx.constant(0x2b000000),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_and_consistency2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.mem8(ctx.register(0)),
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
        ctx.mem8(ctx.register(0)),
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
                ctx.mem16(ctx.register(0)),
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
                    ctx.mem8(ctx.register(0)),
                    ctx.mem32(ctx.register(1)),
                ),
            ),
            ctx.add(
                ctx.mem16(ctx.register(0)),
                ctx.mem64(ctx.register(0)),
            ),
        ),
    );
    let eq1 = ctx.and(
        ctx.gt(
            ctx.register(5),
            ctx.register(4),
        ),
        ctx.add(
            ctx.mem8(ctx.register(0)),
            ctx.mem8(ctx.register(1)),
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
            ctx.mem32(ctx.register(0)),
            ctx.constant(0x1400),
        ),
        ctx.constant(0xffff6ff24),
    );
    let ne1 = ctx.add(
        ctx.and(
            ctx.mem32(ctx.register(0)),
            ctx.constant(0xffff6ff24),
        ),
        ctx.constant(0x1400),
    );
    let op2 = ctx.add(
        ctx.register(1),
        ctx.and(
            ctx.add(
                ctx.mem32(ctx.register(0)),
                ctx.constant(0x1400),
            ),
            ctx.constant(0xffff6ff24),
        ),
    );
    let ne2 = ctx.add(
        ctx.register(1),
        ctx.add(
            ctx.and(
                ctx.mem32(ctx.register(0)),
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
                ctx.mem32(ctx.register(0)),
                ctx.constant(0x1400),
            ),
            ctx.constant(0xffff6ff24),
        ),
    );
    let ne1 = ctx.add(
        ctx.constant(0x4700000014fef910),
        ctx.add(
            ctx.and(
                ctx.mem32(ctx.register(0)),
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
            ctx.mem16(ctx.register(0)),
        ),
        ctx.constant(0x28004000d2000010),
    );
    let eq1 = ctx.and(
        ctx.mem8(ctx.register(0)),
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
        ctx.mem8(ctx.register(2)),
    );
    let eq1 = ctx.eq(
        ctx.mem8(ctx.register(2)),
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
            ctx.mem16(ctx.register(0)),
        ),
        ctx.constant(0x40ffffffff3fff7f),
    );
    let eq1 = ctx.or(
        ctx.and(
            ctx.xmm(2, 1),
            ctx.mem8(ctx.register(0)),
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
            ctx.mem32(ctx.register(0)),
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
            ctx.mem32(ctx.register(0)),
        ),
        ctx.mem32(ctx.register(1)),
    );
    assert!(
        op1.iter().any(|x| x.if_arithmetic_and().is_some()) == false,
        "Operand didn't simplify correctly: {}", op1,
    );
}

#[test]
fn simplify_eq_consistency5() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.mem32(ctx.register(1)),
        ctx.add(
            ctx.constant(0x5a00000001),
            ctx.mem8(ctx.register(0)),
        ),
    );
    assert!(
        op1.iter().any(|x| x.if_arithmetic_and().is_some()) == false,
        "Operand didn't simplify correctly: {}", op1,
    );
}

#[test]
fn simplify_eq_consistency6() {
    let ctx = &OperandContext::new();
    let op1 = ctx.eq(
        ctx.register(0),
        ctx.add(
            ctx.register(0),
            ctx.rsh(
                ctx.mem16(ctx.register(0)),
                ctx.constant(5),
            ),
        ),
    );
    let eq1a = ctx.eq(
        ctx.constant(0),
        ctx.rsh(
            ctx.mem16(ctx.register(0)),
            ctx.constant(5),
        ),
    );
    let eq1b = ctx.eq(
        ctx.constant(0),
        ctx.and(
            ctx.mem16(ctx.register(0)),
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
            ctx.mem32(ctx.register(0)),
            ctx.add(
                ctx.mem32(ctx.register(1)),
                ctx.constant(0x7e0000fffc01),
            ),
        ),
    );
    let eq1 = ctx.eq(
        ctx.constant(0x3fff81ffff5703ff),
        ctx.add(
            ctx.mem32(ctx.register(0)),
            ctx.mem32(ctx.register(1)),
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
            ctx.mem64(ctx.constant(0x100)),
            ctx.constant(0x0500ff04_ffff0000),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0xffff7024ffffffff),
        ctx.lsh(
            ctx.mem8(ctx.constant(0x105)),
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
            ctx.mem64(ctx.register(0)),
            ctx.or(
                ctx.xmm(0, 0),
                ctx.constant(0x0080_0000_0000_0002),
            ),
        ),
    );
    let eq1 = ctx.or(
        ctx.constant(0x40ff_ffff_ffff_3fff),
        ctx.and(
            ctx.mem16(ctx.register(0)),
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
            ctx.mem8(ctx.register(1)),
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
            ctx.mem8(ctx.register(1)),
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
            ctx.mem16(ctx.constant(0x100)),
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
                ctx.mem8(ctx.constant(0x101)),
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
            ctx.mem8(ctx.register(1)),
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
                        ctx.mem8(ctx.register(0)),
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
                        ctx.mem8(ctx.register(0)),
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
                ctx.mem8(ctx.register(0)),
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
            ctx.mem8(ctx.register(0)),
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
fn simplify_and_panic() {
    let ctx = &OperandContext::new();
    let op1 = ctx.and(
        ctx.xmm(1, 0),
        ctx.and(
            ctx.or(
                ctx.xmm(1, 0),
                ctx.constant(0x5ffffffff00),
            ),
            ctx.arithmetic(ArithOpType::Parity, ctx.register(0), ctx.constant(0)),
        )
    );
    let _ = op1;
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
            ctx.mem16(ctx.constant(0x100)),
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
        ),
        ctx.constant(0xffff),
    );
    let eq2 = ctx.mem16(ctx.constant(0x100));
    assert_eq!(op1, eq1);
    assert_eq!(op2, eq2);
}

#[test]
fn gt_masked_mem() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.mem8(ctx.register(0)),
                ctx.constant(0xc),
            ),
            0xff,
        ),
        ctx.mem8(ctx.register(0)),
    );
    let eq1 = ctx.gt(
        ctx.constant(0xc),
        ctx.mem8(ctx.register(0)),
    );
    assert_eq!(op1, eq1);
}

#[test]
fn gt_masked_mem2() {
    let ctx = &OperandContext::new();
    let op1 = ctx.gt(
        ctx.and_const(
            ctx.sub(
                ctx.mem8(ctx.register(0)),
                ctx.constant(0xc),
            ),
            0xffff,
        ),
        ctx.mem8(ctx.register(0)),
    );
    let eq1 = ctx.gt(
        ctx.constant(0xc),
        ctx.mem8(ctx.register(0)),
    );
    assert_eq!(op1, eq1);
}
