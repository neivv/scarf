mod helpers;

use std::ffi::OsStr;

use scarf::analysis::{self, Control};
use scarf::Operation;

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

#[test]
fn slow5() {
    test(4);
}

#[test]
fn slow6() {
    test(5);
}

struct DummyAnalyzer;
impl<'e> analysis::Analyzer<'e> for DummyAnalyzer {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86<'e>;
    fn operation(&mut self, _control: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        if let Operation::Error(e) = *op {
            panic!("Disassembly error {}", e);
        }
    }
}

struct DummyAnalyzer64;
impl<'e> analysis::Analyzer<'e> for DummyAnalyzer64 {
    type State = analysis::DefaultState;
    type Exec = scarf::ExecutionStateX86_64<'e>;
    fn operation(&mut self, _control: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        if let Operation::Error(e) = *op {
            panic!("Disassembly error {}", e);
        }
    }
}

fn test(idx: usize) {
    let binary = helpers::raw_bin(OsStr::new("test_inputs/slow.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    let ctx = scarf::operand::OperandContext::new();
    let mut analysis = analysis::FuncAnalysis::new(&binary, &ctx, func);
    analysis.analyze(&mut DummyAnalyzer);
}

fn test64(idx: usize) {
    let binary = helpers::raw_bin_64(OsStr::new("test_inputs/slow_64.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    let ctx = scarf::operand::OperandContext::new();
    let mut analysis = analysis::FuncAnalysis::new(&binary, &ctx, func);
    analysis.analyze(&mut DummyAnalyzer64);
}
