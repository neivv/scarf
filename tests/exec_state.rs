extern crate byteorder;
extern crate scarf;

mod helpers;

use std::ffi::OsStr;
use std::rc::Rc;

use byteorder::{ReadBytesExt, LittleEndian};

use scarf::{
    BinaryFile, BinarySection, DestOperand, Operand, Operation, VirtualAddress
};
use scarf::analysis;
use scarf::ExecutionStateX86 as ExecutionState;
use scarf::operand_helpers::*;
use scarf::operand::OperandType;

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

#[test]
fn movsx_16_self() {
    test_inline(&[
        0xb8, 0x94, 0x12, 0x00, 0x00, // mov eax, 1294
        0x66, 0x0f, 0xbe, 0xc0, // movsx ax, al
        0x31, 0xc9, // xor ecx, ecx
        0x31, 0xf6, // xor esi, esi
        0x66, 0x0f, 0xbe, 0xc8, // movsx cx, al
        0x88, 0x02, // mov [edx], al
        0x66, 0x0f, 0xbe, 0x32, // movsx si, [edx]
        0x66, 0x0f, 0xbe, 0x12, // movsx dx, [edx]
        0x81, 0xe2, 0xff, 0xff, 0x00, 0x00, // and edx, ffff
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0xff94)),
         (operand_register(1), constval(0xff94)),
         (operand_register(2), constval(0xff94)),
         (operand_register(6), constval(0xff94)),
    ]);
}

#[test]
fn cmp_const_16() {
    test_inline(&[
        0x66, 0xc7, 0x03, 0x05, 0x05, // mov word [ebx], 0505
        0x31, 0xc0, // xor eax, eax
        0x66, 0x81, 0x3b, 0xef, 0xbb, // cmp word [ebx], bbef
        0x0f, 0x92, 0xc0, // setb al
        0xc3, //ret
    ], &[
         (operand_register(0), constval(1)),
    ]);
}

#[test]
fn add_i8() {
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x83, 0xc0, 0xfb, // add eax, 0xfffffffb
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0xfffffffb)),
    ]);
}

#[test]
fn shld() {
    test_inline(&[
        0xb9, 0x23, 0x01, 0x00, 0xff, // mov ecx, 0xff000123
        0xb8, 0x00, 0x00, 0x00, 0x40, // mov eax, 0x40000000
        0x0f, 0xa4, 0xc1, 0x04, // shld ecx, eax, 4
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0x40000000)),
         (operand_register(1), constval(0xf0001234)),
    ]);
}

#[test]
fn double_jbe() {
    test_inline(&[
        0xb8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0x76, 0x03, // jbe _jbe2
        0x77, 0x09, // ja _end
        0xcc, // int3
        // _jbe2:
        0x76, 0x06, // jbe _end
        0xb8, 0x04, 0x00, 0x00, 0x00, // mov eax, 4
        0xcc, // int3
        0xc3, // ret
    ], &[
         (operand_register(0), constval(1)),
    ]);
}

#[test]
fn shr_mem_10() {
    test_inline(&[
        0xc7, 0x45, 0xf4, 0x78, 0x56, 0x34, 0x12, // mov [ebp - 0xc], 0x12345678
        0xc1, 0x6d, 0xf4, 0x10, // shr [ebp - 0xc], 0x10
        0x8b, 0x45, 0xf4, // mov eax, [ebp - 0xc]
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0x1234)),
    ]);
}

#[test]
fn shr_mem_10_b() {
    // Memory address (base + ffffffff)
    test_inline(&[
        0xc7, 0x45, 0xff, 0x78, 0x56, 0x34, 0x12, // mov [ebp - 0x1], 0x12345678
        0xc1, 0x6d, 0xff, 0x10, // shr [ebp - 0x1], 0x10
        0x8b, 0x45, 0xff, // mov eax, [ebp - 0x1]
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0x1234)),
    ]);
}

#[test]
fn read_ffffffff() {
    test_inline(&[
        0xa1, 0xff, 0xff, 0xff, 0xff, // mov eax, [ffffffff]
        0xc3, //ret
    ], &[
         (operand_register(0), mem32(constval(0xffff_ffff))),
    ]);
}

#[test]
fn read_this() {
    test_inline(&[
        0xa1, 0x00, 0x10, 0x40, 0x00, // mov eax, [401000]
        0x8b, 0x0d, 0x0e, 0x10, 0x40, 0x00, // mov ecx, [40100e]
        0x8b, 0x15, 0x0f, 0x10, 0x40, 0x00, // mov edx, [40100f]
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0x4010_00a1)),
         (operand_register(1), constval(0xc300_4010)),
         (operand_register(2), mem32(constval(0x0040_100f))),
    ]);
}

#[test]
fn je_jne_with_memory_write() {
    test_inline(&[
        0x8b, 0x03, // mov eax, [ebx]
        0x01, 0xc8, // add eax, ecx
        0x74, 0x05, // je ret
        0x89, 0x01, // mov [ecx], eax
        0x75, 0x01, // jne ret
        0xcc, // int3
        0x31, 0xc0, // xor eax, eax
        0xc3, //ret
    ], &[
        (operand_register(0), constval(0)),
    ]);
}

#[test]
fn not_is_xor() {
    test_inline(&[
        0xf7, 0xd0, // not eax
        0x83, 0xf0, 0xff, // xor eax, ffff_ffff
        0x66, 0xf7, 0xd1, // not cx
        0x81, 0xf1, 0xff, 0xff, 0x00, 0x00, // xor ecx, ffff
        0xc3, //ret
    ], &[
        (operand_register(0), operand_register(0)),
        (operand_register(1), operand_register(1)),
    ]);
}

#[test]
fn jge_jge() {
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x39, 0xc8, // cmp eax, ecx
        0x7d, 0x03, // jge next
        0x7c, 0x04, // jl ret
        0xcc, // int3
        // next:
        0x7d, 0x01, // jge ret
        0xcc, // int3
        0xc3, //ret
    ], &[
        (operand_register(0), constval(0)),
    ]);
}

#[test]
fn inc_dec_flags() {
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x40, // inc eax
        0x75, 0x01, // jne skip
        0xcc, // int3
        0x48, // dec eax
        0x74, 0x01, // jne skip
        0xcc, // int3
        0x48, // dec eax
        0x73, 0x01, // jnc skip
        0xcc, // int3
        0xc3, //ret
    ], &[
        (operand_register(0), constval(0xffff_ffff)),
    ]);
}

#[test]
fn jo_jno_sometimes_undef() {
    // jo is guaranteed to not be taken if esi == 0
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x85, 0xf6, // test esi, esi
        0x74, 0x02, // je skip
        0x39, 0xf9, // cmp ecx, edi
        // skip:
        0x70, 0x06, // jo end
        0xc0, 0xec, 0x00, // shr ah, 0
        0x71, 0x01, // jno end
        0xcc, // int3
        0x31, 0xc0, // xor eax, eax
        0xc3, //ret
    ], &[
        (operand_register(0), constval(0)),
    ]);
}

#[test]
fn call_removes_constraints() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0x31, 0xdb, // xor ebx, ebx
        0x85, 0xf6, // test esi, esi
        0x7d, 0x08, // jge end
        0xe8, 0x00, 0x00, 0x00, 0x00, // call x
        0x7c, 0x01, // jl end
        0x43, // inc ebx
        // end
        0xeb, 0x00, // jmp ret
        0xc3, //ret
    ], &[
        (operand_register(0), ctx.undefined_rc()),
        (operand_register(1), ctx.undefined_rc()),
        (operand_register(2), ctx.undefined_rc()),
        (operand_register(3), ctx.undefined_rc()),
        (operand_register(4), ctx.undefined_rc()),
    ]);
}

#[test]
fn div_mod() {
    test_inline(&[
        0x33, 0xd2, // xor edx, edx
        0xb9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0xf7, 0xf1, // div ecx
        0xc3, //ret
    ], &[
        (operand_register(1), constval(7)),
        (operand_register(2), operand_mod(operand_register(0), constval(7))),
        (operand_register(0), operand_div(operand_register(0), constval(7))),
    ]);
}

#[test]
fn cmp_mem8() {
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x04, 0x95, // add al, 95
        0x89, 0x01, // mov [ecx], eax
        0x80, 0x39, 0x94, // cmp byte [ecx], 94
        0x77, 0x01, // ja ok
        0xcc,
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0x95)),
    ]);
}

fn test_inner(
    file: &BinaryFile<VirtualAddress>,
    func: VirtualAddress,
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
    let mut end_state = None;
    while let Some(mut branch) = analysis.next_branch() {
        let mut ops = branch.operations();
        while let Some((op, state, addr, i)) = ops.next() {
            println!("@ {:x} {:#?}", addr.0, op);
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
            assert_eq!(expected, end, "Register {}: got {} expected {}", i, end, expected);
        }
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
    let binary = helpers::raw_bin(OsStr::new("test_inputs/exec_state.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    test_inner(&binary, func, changes);
}
