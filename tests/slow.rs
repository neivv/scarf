mod helpers;

use std::ffi::OsStr;

use scarf::analysis;

use byteorder::{ReadBytesExt, LittleEndian};

#[test]
fn hash_func() {
    test(0);
}

#[test]
fn hash_func2() {
    test(1);
}

#[test]
fn hash_func3() {
    test(2);
}

#[test]
fn hash_func64_1() {
    test64(0);
}

#[test]
fn slow4() {
    test(3);
}

struct DummyAnalyzer;
impl<'e> analysis::Analyzer<'e> for DummyAnalyzer {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86<'e>;
}

struct DummyAnalyzer64;
impl<'e> analysis::Analyzer<'e> for DummyAnalyzer64 {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86_64<'e>;
}

fn test(idx: usize) {
    let binary = helpers::raw_bin(OsStr::new("test_inputs/slow.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    let ctx = scarf::operand::OperandContext::new();
    let mut analysis = analysis::FuncAnalysis::new(&binary, &ctx, func);
    analysis.analyze(&mut DummyAnalyzer);
    assert!(analysis.errors.is_empty(), "Had errors: {:#?}", analysis.errors);
}

fn test64(idx: usize) {
    let binary = helpers::raw_bin_64(OsStr::new("test_inputs/slow_64.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    let ctx = scarf::operand::OperandContext::new();
    let mut analysis = analysis::FuncAnalysis::new(&binary, &ctx, func);
    analysis.analyze(&mut DummyAnalyzer64);
    assert!(analysis.errors.is_empty(), "Had errors: {:#?}", analysis.errors);
}
