use std::collections::HashMap;
use std::io::{self, Write};

use crate::cfg::{Cfg, CfgState, CfgOutEdges, NodeLink};
use crate::exec_state::VirtualAddress;
use crate::operand::{ArithOpType, MemAccessSize, Operand, OperandType, OperandContext};

pub fn write<'e, W: Write, S: CfgState>(
    ctx: &'e OperandContext,
    cfg: &mut Cfg<'e, S>,
    out: &mut W,
) -> Result<(), io::Error> {
    writeln!(out, "digraph func {{")?;
    let mut nodes = HashMap::new();
    let mut node_name_pos = 0;
    cfg.calculate_distances();
    let mut cycles = if cfg.nodes().count() < 500 {
        cfg.cycles()
    } else {
        Vec::new()
    };
    cycles.sort();
    for n in cfg.nodes() {
        let node_name = next_node_name(&mut node_name_pos);
        let mut label = format!(
            "{:x} -> {:x}\nDistance: {}",
            n.address,
            n.node.end_address,
            n.node.distance,
        );
        for cycle in cycles.iter().filter(|x| x[0].address() == n.address) {
            use std::fmt::Write;
            let cycle = cycle.iter().map(|x| x.address().as_u64()).collect::<Vec<_>>();
            write!(label, "\nCycle {:x?}", cycle).unwrap();
        }
        writeln!(out, "  {} [label=\"{}\"];", node_name, label)?;
        nodes.insert(n.address, node_name);
    }
    for n in cfg.nodes() {
        let node_name = nodes.get(&n.address).expect("Broken graph");
        let mut print = |node: &NodeLink<S::VirtualAddress>, cond| print_out_edge(
            out,
            &node_name,
            node.address(),
            &nodes,
            &mut node_name_pos,
            cond
        );

        match n.node.out_edges {
            CfgOutEdges::Single(ref node) => {
                print(node, None)?;
            }
            CfgOutEdges::Branch(ref default, ref cond) => {
                print(default, None)?;
                print(&cond.node, Some(pretty_print_condition(ctx, cond.condition)))?;
            }
            CfgOutEdges::None => (),
            CfgOutEdges::Switch(ref cases, _) => {
                for (i, node) in cases.iter().enumerate() {
                    print(&node, Some(i.to_string()))?;
                }
            }
        }
    }
    writeln!(out, "}}")?;
    Ok(())
}

fn print_out_edge<W: Write, Va: VirtualAddress>(
    out: &mut W,
    src: &str,
    addr: Va,
    nodes: &HashMap<Va, String>,
    node_name_pos: &mut u32,
    name: Option<String>,
) -> Result<(), io::Error> {
    let node_name;
    let dest = if addr == Va::max_value() {
        node_name = next_node_name(node_name_pos);
        writeln!(out, "  {} [label=\"???\"];", node_name)?;
        &node_name
    } else {
        nodes.get(&addr).expect("Broken graph")
    };
    if let Some(name) = name {
        writeln!(out, "  {} -> {} [color=\"green\"][label=\"{}\"];", src, dest, name)?;
    } else {
        writeln!(out, "  {} -> {};", src, dest)?;
    }
    Ok(())
}

fn next_node_name(pos: &mut u32) -> String {
    *pos += 1;
    let mut val = *pos;
    let mut out = String::new();
    while val != 0 || out.is_empty() {
        out.insert(0, (b'a' + ((val - 1) % 25) as u8) as char);
        val = (val - 1) / 25;
    }
    out
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum Comparision {
    Equal,
    NotEqual,
    LessThan,
    GreaterOrEqual,
    GreaterThan,
    LessOrEqual,
    SignedLessThan,
    SignedGreaterOrEqual,
    SignedGreaterThan,
    SignedLessOrEqual,
}

#[derive(Debug, Eq, PartialEq)]
struct ComparisionGuess<'e> {
    comparision: Comparision,
    left: Operand<'e>,
    right: Operand<'e>,
    size: MemAccessSize,
}

/// Guesses comparision size by simply just returning smallest
/// possible size that both a and b fit in
fn compare_size<'e>(a: Operand<'e>, b: Operand<'e>) -> MemAccessSize {
    let a_bits = a.relevant_bits();
    let b_bits = b.relevant_bits();
    if a_bits.end <= 8 && b_bits.end <= 8 {
        MemAccessSize::Mem8
    } else if a_bits.end <= 16 && b_bits.end <= 16 {
        MemAccessSize::Mem16
    } else if a_bits.end <= 32 && b_bits.end <= 32 {
        MemAccessSize::Mem32
    } else {
        MemAccessSize::Mem64
    }
}

/// Attempts to understand what comparision the operand represents.
/// E.g. signed x < y doesn't have an operand and can be represented
/// as ((y + 8000_0000) > (x + 8000_0000)) & ((x == y) == 0) == 0.
/// In practice it's likely whatever flag_s == flag_o for x - y
/// looks like, likely not the above 'easy' to understand form.
///
/// Returned value has bitwise and masks removed from left/right,
/// using `size` field to show what they were.
fn comparision_from_operand<'e>(
    ctx: &'e OperandContext,
    oper: Operand<'e>,
) -> Option<ComparisionGuess<'e>> {
    match oper.ty() {
        OperandType::Arithmetic(arith) => {
            let l = arith.left;
            let r = arith.right;
            match arith.ty {
                ArithOpType::Equal => {
                    let other = match (l.ty(), r.ty()) {
                        (&OperandType::Constant(0), _) => Some(r),
                        (_, &OperandType::Constant(0)) => Some(l),
                        _ => None,
                    };
                    if let Some(other) = other {
                        if let Some(inner) = comparision_from_operand(ctx, other) {
                            let inverted = match inner.comparision {
                                Comparision::Equal => Comparision::NotEqual,
                                Comparision::NotEqual => Comparision::Equal,
                                Comparision::LessThan => Comparision::GreaterOrEqual,
                                Comparision::GreaterOrEqual => Comparision::LessThan,
                                Comparision::GreaterThan => Comparision::LessOrEqual,
                                Comparision::LessOrEqual => Comparision::GreaterThan,
                                Comparision::SignedLessThan => Comparision::SignedGreaterOrEqual,
                                Comparision::SignedGreaterOrEqual => Comparision::SignedLessThan,
                                Comparision::SignedGreaterThan => Comparision::SignedLessOrEqual,
                                Comparision::SignedLessOrEqual => Comparision::SignedGreaterThan,
                            };
                            Some(ComparisionGuess {
                                comparision: inverted,
                                ..inner
                            })
                        } else {
                            let (other, size) = extract_mask_single(other);
                            let (l2, r2) = compare_base_op(ctx, other).decide(ctx);
                            Some(ComparisionGuess {
                                comparision: Comparision::Equal,
                                left: l2.clone(),
                                right: r2.clone(),
                                size,
                            })
                        }
                    } else {
                        if let Some(left) = comparision_from_operand(ctx, l) {
                            let right = comparision_from_operand(ctx, r)
                                .filter(|r| r.size == left.size);
                            if let Some(right) = right {
                                let lc = left.comparision;
                                let ll = left.left;
                                let lr = left.right;
                                let rc = right.comparision;
                                let rl = right.left;
                                let rr = right.right;
                                let result = sign_flag_check(ctx, lc, ll, lr, left.size)
                                    .and_then(|x| {
                                        Some((x, overflow_flag_check(ctx, rc, rl, rr, right.size)?))
                                    })
                                    .and_then(|(op1, op2)| {
                                        op1.decide_with_lhs(ctx, op2.0)
                                            .filter(|op1_new| *op1_new == op2)
                                    })
                                    .or_else(|| {
                                        sign_flag_check(ctx, rc, rl, rr, right.size)
                                            .and_then(|x| {
                                                Some((
                                                    x,
                                                    overflow_flag_check(ctx, lc, ll, lr, left.size)?,
                                                ))
                                            })
                                            .and_then(|(op1, op2)| {
                                                op1.decide_with_lhs(ctx, op2.0)
                                                    .filter(|op1_new| *op1_new == op2)
                                            })
                                    });
                                if let Some(result) = result {
                                    return Some(ComparisionGuess {
                                        comparision: Comparision::SignedGreaterOrEqual,
                                        left: result.0,
                                        right: result.1,
                                        size: left.size,
                                    });
                                }
                            }
                        }
                        let (l, r, size) = extract_masks(l, r);
                        Some(ComparisionGuess {
                            comparision: Comparision::Equal,
                            left: l.clone(),
                            right: r.clone(),
                            size,
                        })
                    }
                }
                ArithOpType::GreaterThan => {
                    let (left, right, size) = extract_masks(l, r);
                    if let Some(op) = carry_flag_check(ctx, Comparision::GreaterThan, left, right) {
                        if check_signed_lt_zero(op.0, size) {
                            Some(ComparisionGuess {
                                comparision: Comparision::SignedLessThan,
                                left: op.1,
                                right: ctx.constant(0),
                                size: size,
                            })
                        } else {
                            // Maybe this check doesn't belong here?
                            // Maybe start of ArithOpType::GreaterThan match arm should do
                            // the y > x + 8000_0000 check that overflow_flag_check
                            // currently does?
                            //
                            // Note: Intentionally using top-level left/right
                            // and not Some(op) from carry_flag_check.
                            // (More reason to do this check elsewhere)
                            let result = overflow_flag_check(
                                ctx,
                                Comparision::GreaterThan,
                                left,
                                right,
                                size,
                            );
                            if let Some((left, right)) = result {
                                Some(ComparisionGuess {
                                    comparision: Comparision::SignedLessThan,
                                    left,
                                    right,
                                    size,
                                })
                            } else {
                                Some(ComparisionGuess {
                                    comparision: Comparision::LessThan,
                                    left: op.0,
                                    right: op.1,
                                    size,
                                })
                            }
                        }
                    } else {
                        // I'm not sure why these are other way around here compared to
                        // above check_signed_lt_zero.
                        if check_signed_lt_zero(right, size) {
                            Some(ComparisionGuess {
                                comparision: Comparision::SignedLessThan,
                                left: left.clone(),
                                right: ctx.constant(0),
                                size,
                            })
                        } else {
                            Some(ComparisionGuess {
                                comparision: Comparision::GreaterThan,
                                left: left.clone(),
                                right: right.clone(),
                                size,
                            })
                        }
                    }
                }
                ArithOpType::Or => {
                    if let Some(left) = comparision_from_operand(ctx, l) {
                        if let Some(right) = comparision_from_operand(ctx, r) {
                            let size = if left.size.bits() > right.size.bits() {
                                left.size
                            } else {
                                right.size
                            };
                            let lc = left.comparision;
                            let ll = left.left;
                            let lr = left.right;
                            let rc = right.comparision;
                            let rl = right.left;
                            let rr = right.right;
                            let ops = signed_less_check(lc, ll, lr)
                                .and_then(|op1| {
                                    zero_flag_check(ctx, rc, rl, rr).map(|op2| (op1, op2))
                                })
                                .or_else(|| {
                                    signed_less_check(rc, rl, rr)
                                        .and_then(|op1| {
                                            zero_flag_check(ctx, lc, ll, lr)
                                                .map(|op2| (op1, op2))
                                        })
                                });
                            if let Some((op1, op2)) = ops {
                                if let Some((_left, right)) = op2.decide_with_lhs(ctx, op1.0) {
                                    if op1.1 == right {
                                        return Some(ComparisionGuess {
                                            comparision: Comparision::SignedLessOrEqual,
                                            left: op1.0,
                                            right: op1.1,
                                            size,
                                        });
                                    }
                                }
                            }
                            let ops = unsigned_less_check(lc, ll, lr)
                                .and_then(|op1| {
                                    zero_flag_check(ctx, rc, rl, rr).map(|op2| (op1, op2))
                                })
                                .or_else(|| {
                                    unsigned_less_check(rc, rl, rr)
                                        .and_then(|op1| {
                                            zero_flag_check(ctx, lc, ll, lr)
                                                .map(|op2| (op1, op2))
                                        })
                                });
                            if let Some((op1, op2)) = ops {
                                if let Some((_left, right)) = op2.decide_with_lhs(ctx, op1.0) {
                                    if op1.1 == right {
                                        return Some(ComparisionGuess {
                                            comparision: Comparision::LessOrEqual,
                                            left: op1.0,
                                            right: op1.1,
                                            size,
                                        });
                                    }
                                }
                            }
                        }
                    }
                    None
                }
                _ => None,
            }
        },
        _ => None,
    }
}

fn check_signed_lt_zero(
    constant: Operand<'_>,
    size: MemAccessSize,
) -> bool {
    let int_max = match size {
        MemAccessSize::Mem8 => 0x7f,
        MemAccessSize::Mem16 => 0x7fff,
        MemAccessSize::Mem32 => 0x7fff_ffff,
        MemAccessSize::Mem64 => 0x7fff_ffff_ffff_ffff,
    };
    constant.if_constant() == Some(int_max)
}

fn zero_flag_check<'e>(
    ctx: &'e OperandContext,
    comp: Comparision,
    l: Operand<'e>,
    r: Operand<'e>,
) -> Option<CompareOperands<'e>> {
    if comp == Comparision::Equal {
        let mut ops = Vec::new();
        collect_add_ops(l, &mut ops, false);
        collect_add_ops(r, &mut ops, true);
        for &mut (ref mut op, ref mut negate) in &mut ops {
            if let Some(c) = op.if_constant() {
                if c > 0x8000_0000 && *negate == false {
                    *op = ctx.constant(0u64.wrapping_sub(c));
                    *negate = true;
                }
            }
        }
        Some(CompareOperands::UncertainEitherWay(ops))
    } else {
        None
    }
}

/// Return CompareOperands for `x` if comp(l, r) is equivalent to `sign(x)`
/// (`signed_less(x, 0)` or `unsigned_less(i32_max, x)`)
fn sign_flag_check<'e>(
    ctx: &'e OperandContext,
    comp: Comparision,
    l: Operand<'e>,
    r: Operand<'e>,
    size: MemAccessSize,
) -> Option<CompareOperands<'e>> {
    let int_max = match size {
        MemAccessSize::Mem8 => 0x7f,
        MemAccessSize::Mem16 => 0x7fff,
        MemAccessSize::Mem32 => 0x7fff_ffff,
        MemAccessSize::Mem64 => 0x7fff_ffff_ffff_ffff,
    };
    match (comp, l.ty(), r.ty()) {
        (Comparision::GreaterThan, _, &OperandType::Constant(c)) => {
            if c == int_max {
                Some(compare_base_op(ctx, l))
            } else {
                None
            }
        }
        (Comparision::LessThan, &OperandType::Constant(c), _) => {
            if c == int_max {
                Some(compare_base_op(ctx, r))
            } else {
                None
            }
        }
        (Comparision::SignedLessThan, _, &OperandType::Constant(0)) => {
            Some(compare_base_op(ctx, l))
        }
        (Comparision::SignedGreaterOrEqual, &OperandType::Constant(0), _) => {
            Some(compare_base_op(ctx, r))
        }
        _ => None,
    }
}

fn carry_flag_check<'e>(
    ctx: &'e OperandContext,
    comp: Comparision,
    l: Operand<'e>,
    r: Operand<'e>,
) -> Option<(Operand<'e>, Operand<'e>)> {
    match comp {
        Comparision::GreaterThan => {
            let op = compare_base_op(ctx, l);
            if let Some(op) = op.decide_with_lhs(ctx, r) {
                Some(op)
            } else {
                Some((r.clone(), l.clone()))
            }
        }
        Comparision::LessThan => {
            let op = compare_base_op(ctx, r);
            if let Some(op) = op.decide_with_lhs(ctx, l) {
                Some(op)
            } else {
                Some((l.clone(), r.clone()))
            }
        }
        _ => None,
    }
}

/// Left, right, size
type ExtractedMask<'e> = (Operand<'e>, Operand<'e>, MemAccessSize);

/// Figures out if left and right are parts of a smaller than 64-bit comparision.
fn extract_masks<'e>(left: Operand<'e>, right: Operand<'e>) -> ExtractedMask<'e> {
    let size = compare_size(left, right);
    let left = Operand::and_masked(left).0;
    let right = Operand::and_masked(right).0;
    (left, right, size)
}

fn extract_mask_single<'e>(op: Operand<'e>) -> (Operand<'e>, MemAccessSize) {
    let size = match op.relevant_bits().end {
        x if x > 32 => MemAccessSize::Mem64,
        x if x > 16 => MemAccessSize::Mem32,
        x if x > 8 => MemAccessSize::Mem16,
        _ => MemAccessSize::Mem8,
    };
    let a = Operand::and_masked(op).0;
    (a, size)
}

/// Check either
///
/// #1
/// `(x - y) sgt x`
///
/// or
///
/// #2
/// `y ugt (x + 8000_0000)`
///     => `(x - y + 8000_0000) ugt (x + 8000_0000)`    (*)
///     => `(x - y) sgt x`
///
/// or
///
/// #3
/// `(x + c) ugt (x + 8000_0000)`
///     => `(x - (8000_0000 - c)) sgt x`   (y = 8000_0000 - c)
///
/// (*) Undoes `(x - y) ugt x` <=> `y ugt x` simplification
///
/// as ways to implement overflow flag set.
///
/// returns (x, y).
///
/// Bitwise and masks should be removed and `size` should be provided
/// by caller.
fn overflow_flag_check<'e>(
    ctx: &'e OperandContext,
    comp: Comparision,
    l: Operand<'e>,
    r: Operand<'e>,
    size: MemAccessSize,
) -> Option<(Operand<'e>, Operand<'e>)> {
    let sign_bit = match size {
        MemAccessSize::Mem8 => 0x80,
        MemAccessSize::Mem16 => 0x8000,
        MemAccessSize::Mem32 => 0x8000_0000,
        MemAccessSize::Mem64 => 0x8000_0000_0000_0000,
    };
    match comp {
        Comparision::SignedGreaterThan => {
            let op = compare_base_op(ctx, l);
            if let Some(op) = op.decide_with_lhs(ctx, r) {
                Some((op.0, op.1))
            } else {
                Some((r.clone(), l.clone()))
            }
        }
        Comparision::SignedLessThan => {
            let op = compare_base_op(ctx, r);
            if let Some(op) = op.decide_with_lhs(ctx, l) {
                Some((op.0, op.1))
            } else {
                Some((l.clone(), r.clone()))
            }
        }
        Comparision::GreaterThan => {
            let mut ops = Vec::new();
            collect_add_ops(r, &mut ops, false);
            let constant_idx = ops.iter().position(|x| {
                x.0.if_constant() == Some(sign_bit) && x.1 == false
            });
            if let Some(i) = constant_idx {
                ops.remove(i);
            } else {
                return None;
            }
            // Matching #3 if left contains all of ops,
            // #2 otherwise.
            let mut left_ops = Vec::new();
            collect_add_ops(l, &mut left_ops, false);
            let left_has_all_ops = ops.iter().all(|r| {
                match left_ops.iter().position(|l| l == r) {
                    Some(s) => {
                        left_ops.remove(s);
                        true
                    }
                    None => false,
                }
            });
            let left_constant = match left_ops.len() {
                0 => Some(0),
                1 => left_ops[0].0.if_constant().map(|c| match left_ops[0].1 {
                    true => 0u64.wrapping_sub(c),
                    false => c,
                }),
                _ => None,
            };
            let tree = add_operands_to_tree(ctx, ops);
            if left_has_all_ops && left_constant.is_some() {
                let constant = sign_bit.wrapping_sub(left_constant.unwrap_or(0));
                Some((tree, ctx.constant(constant)))
            } else {
                Some((tree, l.clone()))
            }
        }
        _ => None,
    }
}

fn signed_less_check<'e>(
    comp: Comparision,
    l: Operand<'e>,
    r: Operand<'e>,
) -> Option<(Operand<'e>, Operand<'e>)> {
    match comp {
        Comparision::SignedLessThan => Some((l, r)),
        Comparision::SignedGreaterOrEqual => Some((r, l)),
        _ => None,
    }
}

fn unsigned_less_check<'e>(
    comp: Comparision,
    l: Operand<'e>,
    r: Operand<'e>,
) -> Option<(Operand<'e>, Operand<'e>)> {
    match comp {
        Comparision::LessThan => Some((l, r)),
        Comparision::GreaterOrEqual => Some((r, l)),
        _ => None,
    }
}

/// Used to represent partially resolved `cmp left, right` where
/// what is left and what is right could not be determined.
/// Zero and sign flag checks have to return this as, for example,
/// in the expression `zero = (a - b - c == 0)` it isn't possible
/// to determine if it originally was `cmp a, (b + c)` or
/// `cmp (a - b), c` or some other combination.
///
/// CompareOperands::decide picks something it is able to, but if
/// the left side can be known (carry/overflow flags can give better
/// guess for lhs/rhs), CompareOperands::decide_with_lhs returns
/// Some((lhs, rhs)) if it is possible to resolve the ambiguous
/// comparision that CompareOperands stores.
/// To be clear, on success the returned 'lhs' is always same as the
/// one given as input. (Maybe the function should be only return
/// Some(rhs)).
#[derive(Debug, Clone, Eq, PartialEq)]
enum CompareOperands<'e> {
    /// Both left and right were possible to figure out.
    /// (In practice, that means that there was no subtraction;
    /// left is entire expression and right is 0)
    Certain(Operand<'e>, Operand<'e>),
    // bool negate
    Uncertain(Vec<(Operand<'e>, bool)>),
    // bool negate, for symmetric operations like eq
    UncertainEitherWay(Vec<(Operand<'e>, bool)>),
}

impl<'e> CompareOperands<'e> {
    fn decide(self, ctx: &'e OperandContext) -> (Operand<'e>, Operand<'e>) {
        match self {
            CompareOperands::Certain(l, r) => (l, r),
            CompareOperands::Uncertain(mut opers) |
                CompareOperands::UncertainEitherWay(mut opers) =>
            {
                let first_negative = opers.iter().position(|x| x.1 == true);
                let rhs = match first_negative {
                    Some(s) => opers.remove(s).0,
                    None => ctx.constant(0),
                };
                let tree = add_operands_to_tree(ctx, opers);
                (tree, rhs)
            }
        }
    }

    fn decide_with_lhs(
        self,
        ctx: &'e OperandContext,
        lhs: Operand<'e>,
    ) -> Option<(Operand<'e>, Operand<'e>)> {
        match self {
            CompareOperands::Certain(l, r) => {
                if lhs == l {
                    Some((l, r))
                } else {
                    None
                }
            }
            CompareOperands::Uncertain(mut opers) => {
                let mut lhs_opers = Vec::new();
                collect_add_ops(lhs, &mut lhs_opers, false);
                for required_lhs in lhs_opers {
                    if let Some(pos) = opers.iter().position(|x| x == &required_lhs) {
                        opers.remove(pos);
                    } else {
                        return None;
                    }
                }
                // What is negative in lhs is positive in rhs
                for &mut (_, ref mut neg) in &mut opers {
                    *neg = !*neg;
                }
                let rhs = add_operands_to_tree(ctx, opers);
                Some((lhs.clone(), rhs))
            }
            CompareOperands::UncertainEitherWay(mut opers) => {
                CompareOperands::Uncertain(opers.clone()).decide_with_lhs(ctx, lhs)
                    .or_else(|| {
                        for &mut (_, ref mut neg) in &mut opers {
                            *neg = !*neg;
                        }
                        CompareOperands::Uncertain(opers).decide_with_lhs(ctx, lhs)
                    })
            }
        }
    }
}

fn add_operands_to_tree<'e>(
    ctx: &'e OperandContext,
    mut opers: Vec<(Operand<'e>, bool)>,
) -> Operand<'e> {
    let mut tree = opers.pop().map(|(op, neg)| match neg {
        true => ctx.sub(ctx.constant(0), op),
        false => op,
    }).unwrap_or_else(|| ctx.constant(0));
    while let Some((op, neg)) = opers.pop() {
        match neg {
            true => tree = ctx.sub(tree, op),
            false => tree = ctx.add(tree, op),
        }
    }
    tree
}

fn compare_base_op<'e>(ctx: &'e OperandContext, op: Operand<'e>) -> CompareOperands<'e> {
    match op.ty() {
        OperandType::Arithmetic(arith) if {
            arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub
        } => {
            let mut ops = Vec::new();
            collect_add_ops(arith.left, &mut ops, false);
            let negate = match arith.ty {
                ArithOpType::Add => false,
                ArithOpType::Sub => true,
                _ => unreachable!(),
            };
            collect_add_ops(arith.right, &mut ops, negate);
            for &mut (ref mut op, ref mut negate) in &mut ops {
                if let Some(c) = op.if_constant() {
                    if c > 0x8000_0000 && *negate == false {
                        *op = ctx.constant(0u64.wrapping_sub(c));
                        *negate = true;
                    }
                }
            }
            CompareOperands::Uncertain(ops)
        }
        _ => CompareOperands::Certain(op.clone(), ctx.constant(0)),
    }
}

fn collect_add_ops<'e>(s: Operand<'e>, ops: &mut Vec<(Operand<'e>, bool)>, negate: bool) {
    match s.ty() {
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Add => {
            collect_add_ops(arith.left, ops, negate);
            collect_add_ops(arith.right, ops, negate);
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Sub => {
            collect_add_ops(arith.left, ops, negate);
            collect_add_ops(arith.right, ops, !negate);
        }
        _ => {
            ops.push((s, negate));
        }
    }
}

fn pretty_print_condition<'e>(ctx: &'e OperandContext, cond: Operand<'e>) -> String {
    if let Some(comp) = comparision_from_operand(ctx, cond) {
        let left = comp.left;
        let right = comp.right;
        let bits = comp.size.bits();
        match comp.comparision {
            Comparision::Equal => format!("{} == {}", left, right),
            Comparision::NotEqual => format!("{} != {}", left, right),
            Comparision::LessThan => format!("{} < {}", left, right),
            Comparision::GreaterOrEqual => format!("{} >= {}", left, right),
            Comparision::GreaterThan => format!("{} > {}", left, right),
            Comparision::LessOrEqual => format!("{} <= {}", left, right),
            Comparision::SignedLessThan => format!("signed_{}({} < {})", bits, left, right),
            Comparision::SignedGreaterOrEqual => format!("signed_{}({} >= {})", bits, left, right),
            Comparision::SignedGreaterThan => format!("signed_{}({} > {})", bits, left, right),
            Comparision::SignedLessOrEqual => format!("signed_{}({} <= {})", bits, left, right),
        }
    } else {
        cond.to_string()
    }
}

#[test]
fn recognize_compare_operands_32() {
    let ctx = &OperandContext::new();
    let op = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.and(
                    ctx.sub(
                        ctx.register(2),
                        ctx.register(4),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.constant(0x7fff_ffff),
            ),
            ctx.gt(
                ctx.and(
                    ctx.add(
                        ctx.sub(
                            ctx.register(2),
                            ctx.register(4),
                        ),
                        ctx.constant(0x8000_0000),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.and(
                    ctx.add(
                        ctx.register(2),
                        ctx.constant(0x8000_0000),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
            ),
        ),
    );

    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::SignedLessThan,
        left: ctx.register(2),
        right: ctx.register(4),
        size: MemAccessSize::Mem32,
    });

    // Comparing against zero
    let op = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.and(
                    ctx.register(2),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.constant(0x7fff_ffff),
            ),
            ctx.gt(
                ctx.and(
                    ctx.add(
                        ctx.register(2),
                        ctx.constant(0x8000_0000),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.and(
                    ctx.add(
                        ctx.register(2),
                        ctx.constant(0x8000_0000),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
            ),
        ),
    );
    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::SignedLessThan,
        left: ctx.register(2),
        right: ctx.constant(0),
        size: MemAccessSize::Mem32,
    });

    // Comparing with sub
    let op = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.and(
                    ctx.sub(
                        ctx.register(4),
                        ctx.add(
                            ctx.register(2),
                            ctx.constant(123),
                        ),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.constant(0x7fff_ffff),
            ),
            ctx.gt(
                ctx.and(
                    ctx.add(
                        ctx.sub(
                            ctx.register(4),
                            ctx.add(
                                ctx.register(2),
                                ctx.constant(123),
                            ),
                        ),
                        ctx.constant(0x8000_0000),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
                ctx.and(
                    ctx.add(
                        ctx.sub(
                            ctx.register(4),
                            ctx.register(2),
                        ),
                        ctx.constant(0x8000_0000),
                    ),
                    ctx.constant(0xffff_ffff),
                ),
            ),
        ),
    );
    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::SignedLessThan,
        left: ctx.sub(
            ctx.register(4),
            ctx.register(2),
        ),
        right: ctx.constant(123),
        size: MemAccessSize::Mem32,
    });
}

#[test]
fn recognize_compare_operands_64() {
    let ctx = &OperandContext::new();
    let op = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.sub(
                    ctx.register(2),
                    ctx.register(4),
                ),
                ctx.constant(0x7fff_ffff_ffff_ffff),
            ),
            ctx.gt(
                ctx.add(
                    ctx.sub(
                        ctx.register(2),
                        ctx.register(4),
                    ),
                    ctx.constant(0x8000_0000_0000_0000),
                ),
                ctx.add(
                    ctx.register(2),
                    ctx.constant(0x8000_0000_0000_0000),
                ),
            ),
        ),
    );

    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::SignedLessThan,
        left: ctx.register(2),
        right: ctx.register(4),
        size: MemAccessSize::Mem64,
    });

    // Comparing against zero
    let op = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.register(2),
                ctx.constant(0x7fff_ffff_ffff_ffff),
            ),
            ctx.gt(
                ctx.add(
                    ctx.register(2),
                    ctx.constant(0x8000_0000_0000_0000),
                ),
                ctx.add(
                    ctx.register(2),
                    ctx.constant(0x8000_0000_0000_0000),
                ),
            ),
        ),
    );
    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::SignedLessThan,
        left: ctx.register(2),
        right: ctx.constant(0),
        size: MemAccessSize::Mem64,
    });

    // Comparing with sub
    let op = ctx.eq(
        ctx.constant(0),
        ctx.eq(
            ctx.gt(
                ctx.sub(
                    ctx.register(4),
                    ctx.add(
                        ctx.register(2),
                        ctx.constant(123),
                    ),
                ),
                ctx.constant(0x7fff_ffff_ffff_ffff),
            ),
            ctx.gt(
                ctx.add(
                    ctx.sub(
                        ctx.register(4),
                        ctx.add(
                            ctx.register(2),
                            ctx.constant(123),
                        ),
                    ),
                    ctx.constant(0x8000_0000_0000_0000),
                ),
                ctx.add(
                    ctx.sub(
                        ctx.register(4),
                        ctx.register(2),
                    ),
                    ctx.constant(0x8000_0000_0000_0000),
                ),
            ),
        ),
    );
    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::SignedLessThan,
        left: ctx.sub(
            ctx.register(4),
            ctx.register(2),
        ),
        right: ctx.constant(123),
        size: MemAccessSize::Mem64,
    });
}

#[test]
fn recognize_compare_operands_unsigned() {
    let ctx = &OperandContext::new();
    let op = ctx.or(
        ctx.eq(
            ctx.constant(0),
            ctx.eq(
                ctx.constant(0),
                ctx.gt(
                    ctx.and(
                        ctx.sub(
                            ctx.sub(
                                ctx.mem32(ctx.register(2)),
                                ctx.constant(1234),
                            ),
                            ctx.mem32(ctx.constant(666)),
                        ),
                        ctx.constant(0xffff_ffff),
                    ),
                    ctx.and(
                        ctx.sub(
                            ctx.mem32(ctx.register(2)),
                            ctx.mem32(ctx.constant(666)),
                        ),
                        ctx.constant(0xffff_ffff),
                    ),
                ),
            ),
        ),
        ctx.eq(
            ctx.constant(0),
            ctx.and(
                ctx.sub(
                    ctx.sub(
                        ctx.mem32(ctx.register(2)),
                        ctx.constant(1234),
                    ),
                    ctx.mem32(ctx.constant(666)),
                ),
                ctx.constant(0xffff_ffff),
            ),
        ),
    );
    let comp = comparision_from_operand(ctx, op).unwrap();
    assert_eq!(comp, ComparisionGuess {
        comparision: Comparision::LessOrEqual,
        left: ctx.sub(
            ctx.mem32(ctx.register(2)),
            ctx.mem32(ctx.constant(666)),
        ),
        right: ctx.constant(1234),
        size: MemAccessSize::Mem32,
    });
}
