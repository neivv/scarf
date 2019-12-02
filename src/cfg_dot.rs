use std::collections::HashMap;
use std::io::{self, Write};
use std::rc::Rc;

use crate::cfg::{Cfg, CfgState, CfgOutEdges, NodeLink};
use crate::exec_state::VirtualAddress;
use crate::operand::{ArithOpType, Operand, OperandType};

pub fn write<W: Write, S: CfgState>(cfg: &mut Cfg<S>, out: &mut W) -> Result<(), io::Error> {
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
                print(&cond.node, Some(pretty_print_condition(&cond.condition)))?;
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

fn comparision_from_operand(
    oper: &Rc<Operand>
) -> Option<(Comparision, Rc<Operand>, Rc<Operand>)> {
    use crate::operand::operand_helpers::*;
    match oper.ty {
        OperandType::Arithmetic(ref arith) => {
            let l = &arith.left;
            let r = &arith.right;
            match arith.ty {
                ArithOpType::Equal => {
                    let other = match (&l.ty, &r.ty) {
                        (&OperandType::Constant(0), _) => Some(r),
                        (_, &OperandType::Constant(0)) => Some(l),
                        _ => None,
                    };
                    if let Some(other) = other {
                        if let Some((c, l, r)) = comparision_from_operand(&other) {
                            let inverted = match c {
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
                            Some((inverted, l, r))
                        } else {
                            let (l2, r2) = compare_base_op(&other).decide();
                            Some((Comparision::Equal, l2, r2))
                        }
                    } else {
                        if let Some((lc, ll, lr)) = comparision_from_operand(l) {
                            if let Some((rc, rl, rr)) = comparision_from_operand(r) {
                                let result = sign_flag_check(lc, &ll, &lr)
                                    .and_then(|x| Some((x, overflow_flag_check(rc, &rl, &rr)?)))
                                    .and_then(|(op1, op2)| {
                                        op1.decide_with_lhs(&op2.0)
                                            .filter(|op1_new| *op1_new == op2)
                                    })
                                    .or_else(|| {
                                        sign_flag_check(rc, &rl, &rr)
                                            .and_then(|x| {
                                                Some((x, overflow_flag_check(lc, &ll, &lr)?))
                                            })
                                            .and_then(|(op1, op2)| {
                                                op1.decide_with_lhs(&op2.0)
                                                    .filter(|op1_new| *op1_new == op2)
                                            })
                                    });
                                if let Some(op1) = result {
                                    return Some((
                                        Comparision::SignedGreaterOrEqual,
                                        op1.0,
                                        op1.1,
                                    ));
                                }
                            }
                        }
                        Some((Comparision::Equal, l.clone(), r.clone()))
                    }
                }
                ArithOpType::GreaterThan => {
                    if let Some(op) = carry_flag_check(Comparision::GreaterThan, l, r) {
                        if op.0.ty == OperandType::Constant(0x7fff_ffff) {
                            Some((Comparision::SignedLessThan, op.1, constval(0)))
                        } else {
                            // Maybe this check doesn't belong here?
                            // Maybe start of ArithOpType::GreaterThan match arm should do
                            // the y > x + 8000_0000 check that overflow_flag_check
                            // currently does?
                            if let Some(op) = overflow_flag_check(Comparision::GreaterThan, l, r) {
                                Some((Comparision::SignedLessThan, op.0, op.1))
                            } else {
                                Some((Comparision::LessThan, op.0, op.1))
                            }
                        }
                    } else {
                        if r.ty == OperandType::Constant(0x7fff_ffff) {
                            // yes, using SignedLessThan here as well
                            Some((Comparision::SignedLessThan, l.clone(), constval(0)))
                        } else {
                            Some((Comparision::GreaterThan, l.clone(), r.clone()))
                        }
                    }
                }
                ArithOpType::Or => {
                    if let Some((lc, ll, lr)) = comparision_from_operand(l) {
                        if let Some((rc, rl, rr)) = comparision_from_operand(r) {
                            let ops = signed_less_check(lc, &ll, &lr)
                                .and_then(|op1| zero_flag_check(rc, &rl, &rr).map(|op2| (op1, op2)))
                                .or_else(|| {
                                    signed_less_check(rc, &rl, &rr)
                                        .and_then(|op1| {
                                            zero_flag_check(lc, &ll, &lr).map(|op2| (op1, op2))
                                        })
                                });
                            if let Some((op1, op2)) = ops {
                                if let Some((_left, right)) = op2.decide_with_lhs(&op1.0) {
                                    if op1.1 == right {
                                        return Some((
                                            Comparision::SignedLessOrEqual,
                                            op1.0,
                                            op1.1,
                                        ));
                                    }
                                }
                            }
                            let ops = unsigned_less_check(lc, &ll, &lr)
                                .and_then(|op1| zero_flag_check(rc, &rl, &rr).map(|op2| (op1, op2)))
                                .or_else(|| {
                                    unsigned_less_check(rc, &rl, &rr)
                                        .and_then(|op1| {
                                            zero_flag_check(lc, &ll, &lr).map(|op2| (op1, op2))
                                        })
                                });
                            if let Some((op1, op2)) = ops {
                                if let Some((_left, right)) = op2.decide_with_lhs(&op1.0) {
                                    if op1.1 == right {
                                        return Some((
                                            Comparision::LessOrEqual,
                                            op1.0,
                                            op1.1,
                                        ));
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

fn zero_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<CompareOperands> {
    use crate::operand::operand_helpers::*;
    if comp == Comparision::Equal {
        let mut ops = Vec::new();
        collect_add_ops(l, &mut ops, false);
        collect_add_ops(r, &mut ops, true);
        for &mut (ref mut op, ref mut negate) in &mut ops {
            if let Some(c) = op.if_constant() {
                if c > 0x8000_0000 && *negate == false {
                    *op = constval(0u64.wrapping_sub(c));
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
fn sign_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<CompareOperands> {
    match (comp, &l.ty, &r.ty) {
        (Comparision::GreaterThan, _, &OperandType::Constant(0x7fff_ffff)) => {
            Some(compare_base_op(l))
        }
        (Comparision::LessThan, &OperandType::Constant(0x7fff_ffff), _) => {
            Some(compare_base_op(r))
        }
        (Comparision::SignedLessThan, _, &OperandType::Constant(0)) => {
            Some(compare_base_op(l))
        }
        (Comparision::SignedGreaterOrEqual, &OperandType::Constant(0), _) => {
            Some(compare_base_op(r))
        }
        _ => None,
    }
}

/// Carry / unsigned wraparound is done for subtraction (cmp) as
/// flag_carry = x - y > x
/// Extract (y, x).
fn carry_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<(Rc<Operand>, Rc<Operand>)> {
    match comp {
        Comparision::GreaterThan => {
            let op = compare_base_op(l);
            if let Some(op) = op.decide_with_lhs(r) {
                Some(op)
            } else {
                Some((r.clone(), l.clone()))
            }
        }
        Comparision::LessThan => {
            let op = compare_base_op(r);
            if let Some(op) = op.decide_with_lhs(l) {
                Some(op)
            } else {
                Some((l.clone(), r.clone()))
            }
        }
        _ => None,
    }
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
/// returns (x, y)
fn overflow_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<(Rc<Operand>, Rc<Operand>)> {
    use crate::operand_helpers::*;
    match comp {
        Comparision::SignedGreaterThan => {
            let op = compare_base_op(l);
            if let Some(op) = op.decide_with_lhs(r) {
                Some(op)
            } else {
                Some((r.clone(), l.clone()))
            }
        }
        Comparision::SignedLessThan => {
            let op = compare_base_op(r);
            if let Some(op) = op.decide_with_lhs(l) {
                Some(op)
            } else {
                Some((l.clone(), r.clone()))
            }
        }
        Comparision::GreaterThan => {
            let mut ops = Vec::new();
            collect_add_ops(r, &mut ops, false);
            let constant_idx = ops.iter().position(|x| {
                x.0.if_constant() == Some(0x8000_0000) && x.1 == false
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
            let tree = add_operands_to_tree(ops);
            if left_has_all_ops && left_constant.is_some() {
                let constant = 0x8000_0000u64.wrapping_sub(left_constant.unwrap_or(0));
                Some((tree, constval(constant)))
            } else {
                Some((tree, l.clone()))
            }
        }
        _ => None,
    }
}

fn signed_less_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<(Rc<Operand>, Rc<Operand>)> {
    match comp {
        Comparision::SignedLessThan => Some((l.clone(), r.clone())),
        Comparision::SignedGreaterOrEqual => Some((r.clone(), l.clone())),
        _ => None,
    }
}

fn unsigned_less_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<(Rc<Operand>, Rc<Operand>)> {
    match comp {
        Comparision::LessThan => Some((l.clone(), r.clone())),
        Comparision::GreaterOrEqual => Some((r.clone(), l.clone())),
        _ => None,
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum CompareOperands {
    Certain(Rc<Operand>, Rc<Operand>),
    // bool negate
    Uncertain(Vec<(Rc<Operand>, bool)>),
    // bool negate, for symmetric operations like eq
    UncertainEitherWay(Vec<(Rc<Operand>, bool)>),
}

impl CompareOperands {
    // Returns (l, r) of eq comparision???
    fn decide(self) -> (Rc<Operand>, Rc<Operand>) {
        use crate::operand::operand_helpers::*;
        match self {
            CompareOperands::Certain(l, r) => (l, r),
            CompareOperands::Uncertain(mut opers) |
                CompareOperands::UncertainEitherWay(mut opers) =>
            {
                let first_negative = opers.iter().position(|x| x.1 == true);
                let rhs = match first_negative {
                    Some(s) => opers.remove(s).0,
                    None => constval(0),
                };
                let tree = add_operands_to_tree(opers);
                (tree, rhs)
            }
        }
    }

    /// Lhs is expected to be `x` in `x < x - y`, the one without addition,
    /// while `self` is expected to be longer one `x - y`.
    /// In that case it extracts `x` and `y`.
    /// Not necessarily left hand of operation if operation is LessThan. etc.
    /// Return (x, y) if matching.
    fn decide_with_lhs(self, lhs: &Rc<Operand>) -> Option<(Rc<Operand>, Rc<Operand>)> {
        match self {
            CompareOperands::Certain(l, r) => {
                if Operand::simplified(lhs.clone()) == Operand::simplified(l.clone()) {
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
                let rhs = add_operands_to_tree(opers);
                Some((lhs.clone(), rhs))
            }
            CompareOperands::UncertainEitherWay(mut opers) => {
                CompareOperands::Uncertain(opers.clone()).decide_with_lhs(lhs)
                    .or_else(|| {
                        for &mut (_, ref mut neg) in &mut opers {
                            *neg = !*neg;
                        }
                        CompareOperands::Uncertain(opers).decide_with_lhs(lhs)
                    })
            }
        }
    }
}

fn add_operands_to_tree(mut opers: Vec<(Rc<Operand>, bool)>) -> Rc<Operand> {
    use crate::operand::operand_helpers::*;
    let mut tree = opers.pop().map(|(op, neg)| match neg {
        true => operand_sub(constval(0), op),
        false => op,
    }).unwrap_or_else(|| constval(0));
    while let Some((op, neg)) = opers.pop() {
        match neg {
            true => tree = operand_sub(tree, op),
            false => tree = operand_add(tree, op),
        }
    }
    Operand::simplified(tree)
}

fn compare_base_op(op: &Rc<Operand>) -> CompareOperands {
    use crate::operand::operand_helpers::*;
    match op.ty {
        OperandType::Arithmetic(ref arith) if {
            arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub
        } => {
            let mut ops = Vec::new();
            collect_add_ops(&arith.left, &mut ops, false);
            let negate = match arith.ty {
                ArithOpType::Add => false,
                ArithOpType::Sub => true,
                _ => unreachable!(),
            };
            collect_add_ops(&arith.right, &mut ops, negate);
            for &mut (ref mut op, ref mut negate) in &mut ops {
                if let OperandType::Constant(c) = op.ty {
                    if c > 0x8000_0000 && *negate == false {
                        *op = constval(0u64.wrapping_sub(c));
                        *negate = true;
                    }
                }
            }
            CompareOperands::Uncertain(ops)
        }
        _ => CompareOperands::Certain(op.clone(), constval(0)),
    }
}

fn collect_add_ops(s: &Rc<Operand>, ops: &mut Vec<(Rc<Operand>, bool)>, negate: bool) {
    match s.ty {
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Add => {
            collect_add_ops(&arith.left, ops, negate);
            collect_add_ops(&arith.right, ops, negate);
        }
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Sub => {
            collect_add_ops(&arith.left, ops, negate);
            collect_add_ops(&arith.right, ops, !negate);
        }
        _ => {
            ops.push((s.clone(), negate));
        }
    }
}

fn pretty_print_condition(cond: &Rc<Operand>) -> String {
    if let Some((comp, left, right)) = comparision_from_operand(cond) {
        match comp {
            Comparision::Equal => format!("{} == {}", left, right),
            Comparision::NotEqual => format!("{} != {}", left, right),
            Comparision::LessThan => format!("{} < {}", left, right),
            Comparision::GreaterOrEqual => format!("{} >= {}", left, right),
            Comparision::GreaterThan => format!("{} > {}", left, right),
            Comparision::LessOrEqual => format!("{} <= {}", left, right),
            Comparision::SignedLessThan => format!("signed({} < {})", left, right),
            Comparision::SignedGreaterOrEqual => format!("signed({} >= {})", left, right),
            Comparision::SignedGreaterThan => format!("signed({} > {})", left, right),
            Comparision::SignedLessOrEqual => format!("signed({} <= {})", left, right),
        }
    } else {
        cond.to_string()
    }
}

#[test]
fn recognize_compare_operands() {
    use crate::operand::operand_helpers::*;
    let op = Operand::simplified(operand_eq(
        constval(0),
        operand_eq(
            operand_gt(
                operand_sub(
                    operand_register(2),
                    operand_register(4),
                ),
                constval(0x7fff_ffff),
            ),
            operand_gt(
                operand_and(
                    operand_add(
                        operand_sub(
                            operand_register(2),
                            operand_register(4),
                        ),
                        constval(0x8000_0000),
                    ),
                    constval(0xffff_ffff),
                ),
                operand_and(
                    operand_add(
                        operand_register(2),
                        constval(0x8000_0000),
                    ),
                    constval(0xffff_ffff),
                ),
            ),
        ),
    ));

    let (comp, left, right) = comparision_from_operand(&op).unwrap();
    assert_eq!(comp, Comparision::SignedLessThan);
    assert_eq!(left, operand_register(2));
    assert_eq!(right, operand_register(4));

    // Comparing against zero
    let op = Operand::simplified(operand_eq(
        constval(0),
        operand_eq(
            operand_gt(
                operand_register(2),
                constval(0x7fff_ffff),
            ),
            operand_gt(
                operand_and(
                    operand_add(
                        operand_register(2),
                        constval(0x8000_0000),
                    ),
                    constval(0xffff_ffff),
                ),
                operand_and(
                    operand_add(
                        operand_register(2),
                        constval(0x8000_0000),
                    ),
                    constval(0xffff_ffff),
                ),
            ),
        ),
    ));
    let (comp, left, right) = comparision_from_operand(&op).unwrap();
    assert_eq!(comp, Comparision::SignedLessThan);
    assert_eq!(left, operand_register(2));
    assert_eq!(right, constval(0));

    // Comparing with sub
    let op = Operand::simplified(operand_eq(
        constval(0),
        operand_eq(
            operand_gt(
                operand_sub(
                    operand_register(4),
                    operand_add(
                        operand_register(2),
                        constval(123),
                    ),
                ),
                constval(0x7fff_ffff),
            ),
            operand_gt(
                operand_and(
                    operand_add(
                        operand_sub(
                            operand_register(4),
                            operand_add(
                                operand_register(2),
                                constval(123),
                            ),
                        ),
                        constval(0x8000_0000),
                    ),
                    constval(0xffff_ffff),
                ),
                operand_and(
                    operand_add(
                        operand_sub(
                            operand_register(4),
                            operand_register(2),
                        ),
                        constval(0x8000_0000),
                    ),
                    constval(0xffff_ffff),
                ),
            ),
        ),
    ));
    let (comp, left, right) = comparision_from_operand(&op).unwrap();
    assert_eq!(comp, Comparision::SignedLessThan);
    assert_eq!(left, operand_sub(
        operand_register(4),
        operand_register(2),
    ));
    assert_eq!(right, constval(123));
}

#[test]
fn recognize_compare_operands_unsigned() {
    use crate::operand::operand_helpers::*;
    let op = Operand::simplified(operand_or(
        operand_eq(
            constval(0),
            operand_eq(
                constval(0),
                operand_gt(
                    operand_sub(
                        operand_sub(
                            mem32(operand_register(2)),
                            constval(1234),
                        ),
                        mem32(constval(666)),
                    ),
                    operand_sub(
                        mem32(operand_register(2)),
                        mem32(constval(666)),
                    ),
                ),
            ),
        ),
        operand_eq(
            constval(0),
            operand_sub(
                operand_sub(
                    mem32(operand_register(2)),
                    constval(1234),
                ),
                mem32(constval(666)),
            ),
        ),
    ));
    let (comp, left, right) = comparision_from_operand(&op).unwrap();
    assert_eq!(comp, Comparision::LessOrEqual);
    assert_eq!(left, operand_sub(
        mem32(operand_register(2)),
        mem32(constval(666)),
    ));
    assert_eq!(right, constval(1234));
}
