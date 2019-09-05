extern crate byteorder;
extern crate scarf;

#[allow(dead_code)] mod helpers;

use std::rc::Rc;

use scarf::{
    BinaryFile, BinarySection, DestOperand, Operand, Operation, VirtualAddress64
};
use scarf::analysis::{self, Control};
use scarf::ExecutionStateX86_64 as ExecutionState;
use scarf::operand_helpers::*;
use scarf::operand::OperandType;

#[test]
fn test_basic() {
    test_inline(&[
        0xb8, 0x20, 0x00, 0x00, 0x00, // mov eax, 20
        0x48, 0x83, 0xc0, 0x39, // add rax, 39
        0xa8, 0x62, // test al, 62
        0x75, 0x01, // jne ret
        0xcc, // int3
        0xc3, // ret
    ], &[
         (operand_register64(0), constval(0x59)),
    ]);
}

#[test]
fn test_xor_high() {
    test_inline(&[
        0xb9, 0x04, 0x00, 0x00, 0x00, // mov ecx, 4
        0x30, 0xfd, // xor ch, bh
        0x30, 0xfd, // xor ch, bh
        0xc3, // ret
    ], &[
         (operand_register64(1), constval(0x4)),
    ]);
}

#[test]
fn test_neg_mem8() {
    test_inline(&[
        0xc6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, 0x04, // mov byte [rbp + 4ea2d325], 4
        0xf6, 0x9d, 0x25, 0xd3, 0xa2, 0x4e, // neg byte [rbp + 4ea2d325]
        0x0f, 0xb6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, // movzx eax, byte [rbp + 4ea2d325]
        0xc3, // ret
    ], &[
         (operand_register64(0), constval(0xfc)),
    ]);
}

struct CollectEndState<'e> {
    end_state: Option<(ExecutionState<'e>, scarf::exec_state::InternMap)>,
}

impl<'e> analysis::Analyzer<'e> for CollectEndState<'e> {
    type State = analysis::DefaultState;
    type Exec = ExecutionState<'e>;
    fn operation(&mut self, control: &mut Control<'e, '_, '_, Self>, op: &Operation) {
        println!("@ {:x} {:#?}", control.address(), op);
        if let Operation::Return(_) = *op {
            let (state, i) = control.exec_state();
            self.end_state = Some((state.clone(), i.clone()));
        }
    }
}

fn test_inner(
    file: &BinaryFile<VirtualAddress64>,
    func: VirtualAddress64,
    changes: &[(Rc<Operand>, Rc<Operand>)],
) {
    let ctx = scarf::operand::OperandContext::new();
    let mut interner = scarf::exec_state::InternMap::new();
    let state = ExecutionState::with_binary(file, &ctx, &mut interner);
    let mut expected_state = state.clone();
    for &(ref op, ref val) in changes {
        let op = Operation::Move(DestOperand::from_oper(op), val.clone(), None);
        expected_state.update(&op, &mut interner);
    }
    let mut analysis =
        analysis::FuncAnalysis::with_state(file, &ctx, func, state, interner.clone());
    let mut collect_end_state = CollectEndState {
        end_state: None,
    };
    analysis.analyze(&mut collect_end_state);

    println!("{:?}", analysis.errors);
    assert!(analysis.errors.is_empty());
    let (end_state, mut end_i) = collect_end_state.end_state.unwrap();
    for i in 0..16 {
        let expected = expected_state.resolve(&operand_register64(i), &mut interner);
        let end = end_state.resolve(&operand_register64(i), &mut end_i);
        if end.iter().any(|x| match x.ty {
            OperandType::Undefined(_) => true,
            _ => false,
        }) {
            let expected_is_ud = match expected.ty {
                OperandType::Undefined(_) => true,
                _ => false,
            };
            assert!(expected_is_ud);
        } else {
            println!("{:#?}", end);
            assert_eq!(expected, end, "Register {}: got {} expected {}", i, end, expected);
        }
    }
}

fn test_inline(code: &[u8], changes: &[(Rc<Operand>, Rc<Operand>)]) {
    let binary = scarf::raw_bin(VirtualAddress64(0x00400000), vec![BinarySection {
        name: {
            // ugh
            let mut x = [0; 8];
            for (out, &val) in x.iter_mut().zip(b".text\0\0\0".iter()) {
                *out = val;
            }
            x
        },
        virtual_address: VirtualAddress64(0x401000),
        virtual_size: code.len() as u32,
        data: code.into(),
    }]);
    test_inner(&binary, binary.code_section().virtual_address, changes);
}
