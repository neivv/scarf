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

#[test]
fn simplify_gt3() {
    use super::operand_helpers::*;
    let ctx = &OperandContext::new();
    // x - y + z > x + z => (x + z) - y > x + z => y > x + z
    let op1 = operand_gt(
        operand_add(
            operand_sub(
                ctx.register(0),
                ctx.register(1),
            ),
            ctx.constant(5),
        ),
        operand_add(
            ctx.register(0),
            ctx.constant(5),
        ),
    );
    let eq1 = operand_gt(
        ctx.register(1),
        operand_add(
            ctx.register(0),
            ctx.constant(5),
        ),
    );
    let op1 = Operand::simplified(op1);
    let eq1 = Operand::simplified(eq1);
    assert_eq!(op1, eq1);
}

#[test]
fn simplify_sub_to_zero() {
    use super::operand_helpers::*;
    let ctx = &OperandContext::new();
    let op1 = operand_sub(
        operand_mul(
            ctx.constant(0x2),
            ctx.register(0),
        ),
        operand_mul(
            ctx.constant(0x2),
            ctx.register(0),
        ),
    );
    let eq1 = ctx.const_0();
    let op1 = Operand::simplified(op1);
    let eq1 = Operand::simplified(eq1);
    assert_eq!(op1, eq1);
}
