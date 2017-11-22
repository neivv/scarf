extern crate byteorder;
extern crate scarf;

use std::ffi::OsStr;
use std::io::Read;
use std::rc::Rc;

use byteorder::{ReadBytesExt, LittleEndian};

use scarf::{
    BinaryFile, BinarySection, DestOperand, ExecutionState, Operand, Operation, VirtualAddress
};
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

#[test]
fn f6_f7_and_const() {
    test_inline(&[
        0xc7, 0x00, 0x00, 0x45, 0x00, 0x00, // mov [eax], 4500
        0xf7, 0x00, 0x45, 0x40, 0x00, 0x00, // test [eax], 4045
        0x0f, 0x95, 0xc1, // setne cl
        0x0f, 0xb6, 0xc9, // movzx ecx, cl
        0xf6, 0x00, 0x45, // test byte [eax], 45
        0x0f, 0x95, 0xc0, // setne al
        0x0f, 0xb6, 0xc0, // movzx eax, al
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0)),
         (operand_register(1), constval(1)),
    ]);
}

fn test_inner(file: &BinaryFile, func: VirtualAddress, changes: &[(Rc<Operand>, Rc<Operand>)]) {
    let ctx = scarf::operand::OperandContext::new();
    let mut interner = scarf::exec_state::InternMap::new();
    let state = ExecutionState::new(&ctx, &mut interner);
    let mut expected_state = state.clone();
    for &(ref op, ref val) in changes {
        let op = Operation::Move(DestOperand::from_oper(op), val.clone(), None);
        expected_state.update(op, &mut interner);
    }
    let mut analysis =
        analysis::FuncAnalysis::with_state(file, &ctx, func, state, interner.clone());
    let mut end_state = None;
    while let Some(mut branch) = analysis.next_branch() {
        let mut ops = branch.operations();
        while let Some((op, state, addr, i)) = ops.next() {
            println!("@ {:x} {:?}", addr.0, op);
            if let Operation::Return(_) = *op {
                end_state = Some((state.clone(), i.clone()));
            }
        }
    }
    println!("{:?}", analysis.errors);
    assert!(analysis.errors.is_empty());
    let (end_state, mut end_i) = end_state.unwrap();
    for i in 0..8 {
        let expected = expected_state.resolve(&operand_register(i), &mut interner);
        let end = end_state.resolve(&operand_register(i), &mut end_i);
        assert_eq!(expected, end, "Register {}", i);
    }
}

fn test_inline(code: &[u8], changes: &[(Rc<Operand>, Rc<Operand>)]) {
    let binary = scarf::raw_bin(VirtualAddress(0x00400000), vec![BinarySection {
        name: {
            // ugh
            let mut x = [0; 8];
            for (out, &val) in x.iter_mut().zip(b".text\0\0\0".iter()) {
                *out = val;
            }
            x
        },
        virtual_address: VirtualAddress(0x401000),
        virtual_size: code.len() as u32,
        data: code.into(),
    }]);
    test_inner(&binary, binary.code_section().virtual_address, changes);
}

fn test(idx: usize, changes: &[(Rc<Operand>, Rc<Operand>)]) {
    let binary = raw_bin(OsStr::new("test_inputs/exec_state.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    test_inner(&binary, func, changes);
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
