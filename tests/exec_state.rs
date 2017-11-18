extern crate byteorder;
extern crate scarf;

use std::ffi::OsStr;
use std::io::Read;
use std::rc::Rc;

use byteorder::{ReadBytesExt, LittleEndian};

use scarf::{BinaryFile, BinarySection, ExecutionState, Operand, Operation, VirtualAddress};
use scarf::analysis;
use scarf::operand_helpers::*;

#[test]
fn movzx() {
    test(0, &[
         (operand_register(0), constval(1)),
         (operand_register(1), constval(0xfffd)),
         (operand_register(2), constval(0xfffd)),
    ]);
}

#[test]
fn movsx() {
    test(1, &[
         (operand_register(0), constval(1)),
         (operand_register(1), constval(0xfffd)),
         (operand_register(2), constval(0xfffffffd)),
    ]);
}

#[test]
fn movzx_mem() {
    test(2, &[
         (operand_register(0), constval(0x90)),
    ]);
}

#[test]
fn movsx_mem() {
    test(3, &[
         (operand_register(0), constval(0xffffff90)),
    ]);
}

fn test(idx: usize, changes: &[(Rc<Operand>, Rc<Operand>)]) {
    let binary = raw_bin(OsStr::new("test_inputs/exec_state.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    let ctx = scarf::operand::OperandContext::new();
    let mut interner = scarf::exec_state::InternMap::new();
    let state = ExecutionState::new(&ctx, &mut interner);
    let mut expected_state = state.clone();
    for &(ref op, ref val) in changes {
        let op = Operation::Move((**op).clone().into(), val.clone(), None);
        expected_state.update(op, &mut interner);
    }
    let mut analysis =
        analysis::FuncAnalysis::with_state(&binary, &ctx, func, state, interner.clone());
    let mut end_state = None;
    while let Some(mut branch) = analysis.next_branch() {
        let mut ops = branch.operations();
        while let Some((op, state, _addr, i)) = ops.next() {
            println!("{:?}", op);
            if let Operation::Return(_) = *op {
                end_state = Some((state.clone(), i.clone()));
            }
        }
    }
    let (end_state, mut end_i) = end_state.unwrap();
    for i in 0..8 {
        let expected = expected_state.resolve(&operand_register(i), &mut interner);
        let end = end_state.resolve(&operand_register(i), &mut end_i);
        assert_eq!(expected, end, "Register {}", i);
    }
    println!("{:?}", analysis.errors);
    assert!(analysis.errors.is_empty());
}

fn raw_bin(filename: &OsStr) -> Result<BinaryFile, scarf::Error> {
    let mut file = std::fs::File::open(filename)?;
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    Ok(scarf::raw_bin(VirtualAddress(0x00400000), vec![BinarySection {
        name: {
            // ugh
            let mut x = [0; 8];
            for (out, &val) in x.iter_mut().zip(b".text\0\0\0".iter()) {
                *out = val;
            }
            x
        },
        virtual_address: VirtualAddress(0x401000),
        virtual_size: buf.len() as u32,
        data: buf,
    }]))
}
