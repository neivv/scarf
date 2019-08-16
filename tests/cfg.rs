extern crate byteorder;
extern crate scarf;

mod helpers;

use std::ffi::OsStr;
use byteorder::{ReadBytesExt, LittleEndian};

use scarf::analysis;
use scarf::cfg::CfgOutEdges;
use scarf::{
    BinaryFile, VirtualAddress
};

type Analysis<'a> =
    analysis::FuncAnalysis<'a, scarf::ExecutionStateX86<'a>, analysis::DefaultState>;

#[test]
fn switch_cfg() {
    let (binary, func) = load_test(0);
    let ctx = scarf::operand::OperandContext::new();
    let analysis = Analysis::new(&binary, &ctx, func);
    let (mut cfg, errors) = analysis.finish();
    cfg.calculate_distances();
    assert!(errors.is_empty());
    for node in cfg.nodes() {
        println!("Node {:?}", node);
    }
    let switch_outs = cfg.nodes().filter_map(|x| match x.node.out_edges {
        CfgOutEdges::Switch(ref cases, _) => Some((x, cases.clone())),
        _ => None,
    }).collect::<Vec<_>>();
    assert_eq!(switch_outs.len(), 1);
    assert_eq!(switch_outs[0].0.node.distance, 2);
    assert_eq!(switch_outs[0].0.node.distance, 2);
    assert_eq!(switch_outs[0].1.len(), 5);
}

#[test]
fn undecideable() {
    let (binary, func) = load_test(1);
    let ctx = scarf::operand::OperandContext::new();
    let analysis = Analysis::new(&binary, &ctx, func);
    let (mut cfg, errors) = analysis.finish();
    assert!(errors.is_empty());

    let mut dummy = Vec::new();
    scarf::cfg_dot::write(&mut cfg, &mut dummy).unwrap();
}

#[test]
fn switch_but_constant_case() {
    let (binary, func) = load_test(2);
    let ctx = scarf::operand::OperandContext::new();
    let analysis = Analysis::new(&binary, &ctx, func);
    let (mut cfg, errors) = analysis.finish();
    cfg.calculate_distances();
    assert!(errors.is_empty());
    for node in cfg.nodes() {
        println!("Node {:?}", node);
    }
    for x in cfg.nodes() {
        match x.node.out_edges {
            CfgOutEdges::Switch(_, _) => {
                panic!("The switch case was supposed to be constant, so just a single jump");
            }
            _ => (),
        }
    }
}

fn load_test(idx: usize) -> (BinaryFile<VirtualAddress>, VirtualAddress) {
    let mut binary = helpers::raw_bin(OsStr::new("test_inputs/cfg.bin")).unwrap();
    let code_section = binary.code_section();
    let offset = (&code_section.data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = code_section.virtual_address + offset;
    let switch_table_addr = code_section.virtual_address + (code_section.virtual_size - 5 * 4);
    binary.set_relocs((0..5).map(|i| switch_table_addr + i * 4).collect());
    (binary, func)
}
