extern crate clap;
extern crate scarf;

use std::collections::HashMap;
use std::ffi::OsStr;
use std::rc::Rc;

use scarf::{Operand, OperandType, VirtualAddress};
use scarf::operand::ArithOpType;

fn main() {
    let matches = clap::App::new("scarf-cfg")
        .arg(clap::Arg::with_name("path")
            .index(1)
            .value_name("FILE")
            .required(true)
            .help("Selects the binary"))
        .arg(clap::Arg::with_name("addr")
            .index(2)
            .value_name("ADDRESS")
            .required(true)
            .help("Function start address (in hex)"))
        .arg(clap::Arg::with_name("destructive_calls")
            .long("destructive_calls")
            .required(false)
            .takes_value(false)
            .help("Assumes that calls may write to any pointer passed to them"))
        .get_matches();
    let file = matches.value_of_os("path").unwrap();
    // Address is with default base
    let addr = matches.value_of("addr").unwrap();
    let addr = u32::from_str_radix(&addr, 16).expect("Address wasn't hex");
    let destructive_calls = matches.is_present("destructive_calls");
    let binary = scarf::parse(file).unwrap();
    let ctx = scarf::operand::OperandContext::new();
    let analysis = scarf::analysis::FuncAnalysis::new(&binary, &ctx, VirtualAddress(addr));
    let mut was_called = false;
    let (cfg, errors) = analysis.finish_with_changes(|op, state, _, _| {
        if was_called {
            was_called = false;
            if destructive_calls {
                state.memory = scarf::exec_state::Memory::new();
            }
        }
        match *op {
            scarf::Operation::Call(..) => {
                was_called = true;
            }
            _ => (),
        }
    });
    for (addr, e) in errors {
        eprintln!("{:08x}: {}", addr.0, e);
    }
    println!("digraph func_{:x} {{", addr);
    let mut nodes = HashMap::new();
    let mut node_name_pos = 0;
    for (&address, node) in cfg.nodes() {
        let node_name = next_node_name(&mut node_name_pos);
        println!("  {} [label=\"{:08x} -> {:08x}\"];", node_name, address.0, node.end_address.0);
        nodes.insert(address, node_name);
    }
    for (&address, node) in cfg.nodes() {
        let node_name = nodes.get(&address).expect("Broken graph");
        if let Some(ref out) = node.out_edges {
            print_out_edge(&node_name, out.default, &nodes, &mut node_name_pos, None);
            if let Some((cond_dest, ref cond)) = out.cond {
                print_out_edge(
                    &node_name,
                    cond_dest,
                    &nodes,
                    &mut node_name_pos,
                    Some(pretty_print_condition(cond)),
                );
            }
        }
    }
    println!("}}");
}

fn print_out_edge(
    src: &str,
    addr: VirtualAddress,
    nodes: &HashMap<VirtualAddress, String>,
    node_name_pos: &mut u32,
    name: Option<String>,
) {
    let node_name;
    let dest = if addr == VirtualAddress(!0) {
        node_name = next_node_name(node_name_pos);
        println!("  {} [label=\"???\"];", node_name);
        &node_name
    } else {
        nodes.get(&addr).expect("Broken graph")
    };
    if let Some(name) = name {
        println!("  {} -> {} [color=\"green\"][label=\"{}\"];", src, dest, name);
    } else {
        println!("  {} -> {};", src, dest);
    }
}

fn next_node_name(pos: &mut u32) -> String {
    *pos += 1;
    let mut val = *pos;
    let mut out = String::new();
    while val != 0 || out.is_empty() {
        out.insert(0, ('a' as u8 + ((val - 1) % 25) as u8) as char);
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
    use scarf::operand_helpers::*;
    match oper.ty {
        OperandType::Arithmetic(ref arith) => match *arith {
            ArithOpType::Equal(ref l, ref r) => {
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
                            if let Some(op1) = sign_flag_check(lc, &ll, &lr) {
                                if let Some(op2) = overflow_flag_check(rc, &rl, &rr) {
                                    if let Some(op1) = op1.decide_with_lhs(&op2.0) {
                                        if op1 == op2 {
                                            return Some((
                                                Comparision::SignedGreaterOrEqual,
                                                op1.0,
                                                op1.1,
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Some((Comparision::Equal, l.clone(), r.clone()))
                }
            }
            ArithOpType::GreaterThan(ref l, ref r) => {
                if let Some(op) = carry_flag_check(Comparision::GreaterThan, l, r) {
                    if op.0.ty == OperandType::Constant(0x7fffffff) {
                        Some((Comparision::SignedLessThan, op.1, constval(0)))
                    } else {
                        Some((Comparision::LessThan, op.0, op.1))
                    }
                } else {
                    if r.ty == OperandType::Constant(0x7fffffff) {
                        // yes, using SignedLessThan here as well
                        Some((Comparision::SignedLessThan, l.clone(), constval(0)))
                    } else {
                        Some((Comparision::GreaterThan, l.clone(), r.clone()))
                    }
                }
            }
            ArithOpType::GreaterThanSigned(ref l, ref r) => {
                if let Some(op) = overflow_flag_check(Comparision::SignedGreaterThan, l, r) {
                    Some((Comparision::SignedLessThan, op.0, op.1))
                } else {
                    Some((Comparision::SignedGreaterThan, l.clone(), r.clone()))
                }
            }
            ArithOpType::Or(ref l, ref r) => {
                if let Some((lc, ll, lr)) = comparision_from_operand(l) {
                    if let Some((rc, rl, rr)) = comparision_from_operand(r) {
                        if let Some(op1) = signed_less_check(lc, &ll, &lr) {
                            if let Some(op2) = zero_flag_check(rc, &rl, &rr) {
                                if op1 == op2 {
                                    return Some((
                                        Comparision::SignedLessOrEqual,
                                        op1.0,
                                        op1.1,
                                    ));
                                }
                            }
                        }
                        if let Some(op1) = unsigned_less_check(lc, &ll, &lr) {
                            if let Some(op2) = zero_flag_check(rc, &rl, &rr) {
                                if op1 == op2 {
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
        },
        _ => None,
    }
}

fn zero_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<(Rc<Operand>, Rc<Operand>)> {
    if comp == Comparision::Equal {
        Some((l.clone(), r.clone()))
    } else {
        None
    }
}

fn sign_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<CompareOperands> {
    match (comp, &l.ty, &r.ty) {
        (Comparision::GreaterThan, _, &OperandType::Constant(0x7fffffff)) => {
            Some(compare_base_op(l))
        }
        (Comparision::LessThan, &OperandType::Constant(0x7fffffff), _) => {
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

fn overflow_flag_check(
    comp: Comparision,
    l: &Rc<Operand>,
    r: &Rc<Operand>
) -> Option<(Rc<Operand>, Rc<Operand>)> {
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
}

impl CompareOperands {
    fn decide(self) -> (Rc<Operand>, Rc<Operand>) {
        use scarf::operand::operand_helpers::*;
        match self {
            CompareOperands::Certain(l, r) => (l, r),
            CompareOperands::Uncertain(mut opers) => {
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
        }
    }
}

fn add_operands_to_tree(mut opers: Vec<(Rc<Operand>, bool)>) -> Rc<Operand> {
    use scarf::operand::operand_helpers::*;
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
    use scarf::operand::operand_helpers::*;
    match op.ty {
        OperandType::Arithmetic(ArithOpType::Add(ref l, ref r)) |
            OperandType::Arithmetic(ArithOpType::Sub(ref l, ref r)) =>
        {
            let mut ops = Vec::new();
            collect_add_ops(l, &mut ops, false);
            let negate = match op.ty {
                OperandType::Arithmetic(ArithOpType::Add(_, _)) => false,
                OperandType::Arithmetic(ArithOpType::Sub(_, _)) => true,
                _ => unreachable!(),
            };
            collect_add_ops(r, &mut ops, negate);
            for &mut (ref mut op, ref mut negate) in &mut ops {
                if let OperandType::Constant(c) = op.ty {
                    if c > 0x80000000 && *negate == false {
                        *op = constval(0u32.wrapping_sub(c));
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
        OperandType::Arithmetic(ArithOpType::Add(ref left, ref right)) => {
            collect_add_ops(left, ops, negate);
            collect_add_ops(right, ops, negate);
        }
        OperandType::Arithmetic(ArithOpType::Sub(ref left, ref right)) => {
            collect_add_ops(left, ops, negate);
            collect_add_ops(right, ops, !negate);
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
    use scarf::operand_helpers::*;
    let op = Operand::simplified(operand_eq(
        constval(0),
        operand_eq(
            operand_gt(
                operand_sub(
                    operand_register(2),
                    operand_register(4),
                ),
                constval(0x7fffffff),
            ),
            operand_gt_signed(
                operand_sub(
                    operand_register(2),
                    operand_register(4),
                ),
                operand_register(2),
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
                constval(0x7fffffff),
            ),
            operand_gt_signed(
                operand_register(2),
                operand_register(2),
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
                constval(0x7fffffff),
            ),
            operand_gt_signed(
                operand_sub(
                    operand_register(4),
                    operand_add(
                        operand_register(2),
                        constval(123),
                    ),
                ),
                operand_sub(
                    operand_register(4),
                    operand_register(2),
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
    use scarf::operand_helpers::*;
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
