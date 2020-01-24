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
         (operand_register(0), constval(0x59)),
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
         (operand_register(1), constval(0x4)),
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
         (operand_register(0), constval(0xfc)),
    ]);
}

#[test]
fn test_neg_mem8_dummy_rex_r() {
    test_inline(&[
        0xc6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, 0x04, // mov byte [rbp + 4ea2d325], 4
        0x4e, 0xf6, 0x9d, 0x25, 0xd3, 0xa2, 0x4e, // neg byte [rbp + 4ea2d325]
        0x0f, 0xb6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, // movzx eax, byte [rbp + 4ea2d325]
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0xfc)),
    ]);
}

#[test]
fn test_new_8bit_regs() {
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x31, 0xf6, // xor esi, esi
        0x4d, 0x31, 0xc9, // xor r9, r9
        0x4d, 0x31, 0xff, // xor r15, r15
        0x66, 0xb8, 0xfe, 0x02, // mov ax, 2fe
        0x66, 0xbe, 0xfe, 0x02, // mov si, 2fe
        0x66, 0x41, 0xb9, 0xfe, 0x02, // mov r9w, 2fe
        0x66, 0x41, 0xbf, 0xfe, 0x02, // mov r9w, 2fe
        0x04, 0x05, // add al, 5
        0x80, 0xc4, 0x05, // add ah, 5
        0x40, 0x80, 0xc6, 0x05, // add sil, 5
        0x41, 0x80, 0xc1, 0x05, // add r9b, 5
        0x41, 0x80, 0xc7, 0x05, // add r15b, 5
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0x703)),
         (operand_register(6), constval(0x203)),
         (operand_register(9), constval(0x203)),
         (operand_register(15), constval(0x203)),
    ]);
}

#[test]
fn test_64bit_regs() {
    test_inline(&[
        0x49, 0xbf, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, // mov r15, c_0000_000c
        0x31, 0xc0, // xor eax, eax
        0x48, 0x8d, 0x88, 0x88, 0x00, 0x00, 0x00, // lea rcx, [rax + 88]
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0)),
         (operand_register(1), constval(0x88)),
         (operand_register(15), constval(0xc_0000_000c)),
    ]);
}

#[test]
fn test_btr() {
    test_inline(&[
        0xb8, 0xff, 0xff, 0x00, 0x00, // mov eax, ffff
        0xb9, 0x08, 0x00, 0x00, 0x00, // mov ecx, 8
        0x0f, 0xb3, 0xc8, // btr eax, ecx
        0x73, 0x06, // jnc int3
        0x0f, 0xba, 0xf0, 0x11, // btr eax, 11
        0x73, 0x01, // jnc end
        0xcc, // int3
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0xfeff)),
         (operand_register(1), constval(0x8)),
    ]);
}

#[test]
fn test_movsxd() {
    test_inline(&[
        0xc7, 0x44, 0x24, 0x28, 0x44, 0xff, 0x44, 0xff, // mov dword [rsp + 28], ff44ff44
        0x4c, 0x63, 0x5c, 0x24, 0x28, // movsxd r11, dword [rsp + 28]
        0xc3, // ret
    ], &[
         (operand_register(11), constval(0xffff_ffff_ff44_ff44)),
    ]);
}

#[test]
fn movaps() {
    test_inline(&[
        0xc7, 0x04, 0x24, 0x45, 0x45, 0x45, 0x45, // mov dword [rsp], 45454545
        0xc7, 0x44, 0x24, 0x0c, 0x25, 0x25, 0x25, 0x25, // mov dword [rsp], 25252525
        0x0f, 0x28, 0x04, 0x24, // movaps xmm0, [rsp]
        0x0f, 0x29, 0x44, 0x24, 0x20, // movaps [rsp+20], xmm0
        0x8b, 0x44, 0x24, 0x20, // mov eax, [rsp + 20]
        0x8b, 0x4c, 0x24, 0x2c, // mov ecx, [rsp + 2c]
        0xc3, //ret
    ], &[
         (operand_register(0), constval(0x45454545)),
         (operand_register(1), constval(0x25252525)),
    ]);
}

#[test]
fn test_bt() {
    test_inline(&[
        0xb8, 0xff, 0xff, 0x00, 0x00, // mov eax, ffff
        0x48, 0xc1, 0xe0, 0x20, // shl rax, 20
        0xb9, 0x28, 0x00, 0x00, 0x00, // mov ecx, 28
        0x0f, 0xa3, 0xc8, // bt eax, ecx
        0x73, 0x01, // jnc end
        0xcc, // int3
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0xffff_0000_0000)),
         (operand_register(1), constval(0x28)),
    ]);
}

#[test]
fn test_xadd() {
    test_inline(&[
        0xb8, 0x20, 0x00, 0x00, 0x00, // mov eax, 20
        0xc7, 0x04, 0x24, 0x13, 0x00, 0x00, 0x00, // mov dword [rsp], 13
        0xf0, 0x0f, 0xc1, 0x04, 0x24, // lock xadd dword [rsp], eax
        0x8b, 0x0c, 0x24, // mov ecx, [rsp]
        0xba, 0x05, 0x00, 0x00, 0x00, // mov edx, 5
        0x0f, 0xc1, 0xd2, // xadd edx, edx
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0x13)),
         (operand_register(1), constval(0x33)),
         (operand_register(2), constval(0xa)),
    ]);
}

#[test]
fn test_switch() {
    let ctx = scarf::operand::OperandContext::new();
    // 2 cases to ok, 3rd fake
    // rcx is undef if the cases are run
    test_inline(&[
        0x31, 0xc9, // xor ecx, ecx
        0x48, 0x83, 0xf8, 0x02, // cmp rax, 2
        0x73, 0x28, // jae end
        0x48, 0x8d, 0x0d, 0x12, 0x00, 0x00, 0x00, // lea rcx, [switch_table]
        0x0f, 0xb7, 0x0c, 0x41, // movzx ecx, word [rcx + rax * 2]
        0x48, 0x8d, 0x05, 0x0d, 0x00, 0x00, 0x00, // lea rax, [fail]
        0x48, 0x01, 0xc8, // add rax, rcx
        0x31, 0xc9, // xor ecx, ecx
        0xff, 0xe0, // jmp rax
        // switch_table:
        0x01, 0x00, // case1 - fail
        0x04, 0x00, // case2 - fail
        0x00, 0x00, // fail - fail
        // fail:
        0xcc, // int3
        // case1:
        0x83, 0xc1, 0x06, // add ecx, 6
        // case2:
        0x83, 0xc1, 0x06, // add ecx, 6
        // (Since the test system doesn't merge end states from differend blocks)
        0xeb, 0x00, // jmp end
        // end:
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0)),
         (operand_register(1), ctx.undefined_rc()),
    ]);
}

#[test]
fn test_negative_offset() {
    test_inline(&[
        0xc7, 0x00, 0x05, 0x00, 0x00, 0x00, // mov dword [rax], 5
        0x48, 0x83, 0xc0, 0x04, // add rax, 4
        0x8b, 0x48, 0xfc, // mov ecx, [rax - 4]
        0x48, 0x05, 0x00, 0x01, 0x00, 0x00, // add rax, 100
        0x03, 0x88, 0xfc, 0xfe, 0xff, 0xff, // add ecx, [rax - 104]
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0)),
         (operand_register(1), constval(0xa)),
    ]);
}

#[test]
fn test_negative_offset2() {
    test_inline(&[
        0xc7, 0x04, 0x10, 0x05, 0x00, 0x00, 0x00, // mov dword [rax + rdx], 5
        0x48, 0x83, 0xc0, 0x04, // add rax, 4
        0x8b, 0x4c, 0x10, 0xfc, // mov ecx, [rax + rdx - 4]
        0x48, 0x05, 0x00, 0x01, 0x00, 0x00, // add rax, 100
        0x03, 0x8c, 0x10, 0xfc, 0xfe, 0xff, 0xff, // add ecx, [rax + rdx - 104]
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0)),
         (operand_register(1), constval(0xa)),
    ]);
}

#[test]
fn lazy_flag_constraint_invalidation() {
    let ctx = scarf::operand::OperandContext::new();
    // Had a bug that lazy flag updates didn't invalidate extra constraints
    // (So the unrelated cmp -> ja would always be taken)
    test_inline(&[
        0x31, 0xf6, // xor esi, esi
        0x39, 0xc8, // cmp eax, ecx
        0x76, 0x07, // jbe ret_
        0x39, 0xca, // cmp edx, ecx
        0x77, 0x03, // ja ret_
        0x83, 0xc6, 0x01, // add esi, 1
        0xeb, 0x00, // jmp ret
        0xc3, // ret
    ], &[
         (operand_register(6), ctx.undefined_rc()),
    ]);
}

#[test]
fn punpcklbw() {
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x0c, 0x24, // mov [rsp], rcx
        0x66, 0x0f, 0x60, 0x04, 0x24, // punpcklbw xmm0, [rsp]
        0x0f, 0x11, 0x04, 0x24, // movups [rsp], xmm0
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]
        0xc3, // ret
    ], &[
         (operand_register(0), constval(0x5511_4422_3333_2244)),
         (operand_register(1), constval(0x9912_8834_7756_6678)),
    ]);
}

#[test]
fn overflow_not_set_bug() {
    let ctx = scarf::operand::OperandContext::new();
    // Had a bug that made the `jl maybe` never be taken
    test_inline(&[
        0xbf, 0x01, 0x00, 0x00, 0x00, // mov edi, 1
        0x0f, 0xbf, 0x46, 0x16, // movsx eax, word [rsi + 16]
        0x89, 0xc1, // mov ecx, eax,
        0xc1, 0xf9, 0x05, // sar ecx, 5
        0x85, 0xc9, // test ecx, ecx
        0x79, 0x04, // jns more
        0x33, 0xc9, // xor ecx, ecx
        0xeb, 0x12, // jmp end
        // more:
        0x0f, 0xb7, 0x82, 0xe6, 0x00, 0x00, 0x00, // movzx eax word [rdx + e6]
        0x3b, 0xc8, // cmp ecx, eax
        0x7c, 0x02, // jl maybe
        0xeb, 0x07, // jmp end
        0xbf, 0x02, 0x00, 0x00, 0x00, // mov edi, 2
        0xeb, 0x00, // jmp end
        // end:
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.undefined_rc()),
         (ctx.register(1), ctx.undefined_rc()),
         (ctx.register(7), ctx.undefined_rc()),
    ]);
}

#[test]
fn switch_case_count2() {
    let ctx = scarf::operand::OperandContext::new();
    // 3 cases to ok, 4th fake
    // rdx, rsi, rdi, rbp are undef if the cases are run
    test_inline(&[
        // base:
        0x49, 0x83, 0xf8, 0x02, // cmp r8, 2
        0x76, 0x04, // jbe switch
        0x31, 0xd2, // xor edx, edx
        0xeb, 0x2d, // jmp end
        // switch:
        0x48, 0x89, 0xc8, // mov rax, rcx
        0x4c, 0x8d, 0x0d, 0xec, 0xff, 0xff, 0xff, // lea r9 [base]
        0x43, 0x8b, 0x8c, 0x81, 0x21, 0x00, 0x00, 0x00, // mov ecx, [r9 + r8 * 4 + switch_table - base]
        0x4c, 0x01, 0xc9, // add rcx, r9
        0xff, 0xe1, // jmp rcx
        // switch_table:
        0x31, 0x00, 0x00, 0x00,
        0x33, 0x00, 0x00, 0x00,
        0x35, 0x00, 0x00, 0x00,
        0x3a, 0x00, 0x00, 0x00,
        // case0:
        0x31, 0xf6, // xor esi, esi
        // case1:
        0x31, 0xff, // xor edi, edi
        // case2:
        0x31, 0xed, // xor rbp, rbp
        // end
        0xeb, 0x00,
        0xc3, // ret
        // case3_fake
        0xcc, // int3
    ], &[
         (operand_register(0), ctx.undefined_rc()),
         (operand_register(1), ctx.undefined_rc()),
         (operand_register(2), ctx.undefined_rc()),
         (operand_register(5), ctx.undefined_rc()),
         (operand_register(6), ctx.undefined_rc()),
         (operand_register(7), ctx.undefined_rc()),
         (operand_register(9), ctx.undefined_rc()),
    ]);
}

#[test]
fn switch_case_count3() {
    let ctx = scarf::operand::OperandContext::new();
    // 2 cases to ok, 3rd fake
    // rdx, rsi, rdi, rbp are undef if the cases are run
    test_inline(&[
        // base:
        0x49, 0x83, 0xf8, 0x01, // cmp r8, 1
        0x76, 0x04, // jbe switch
        0x31, 0xd2, // xor edx, edx
        0xeb, 0x2d, // jmp end
        // switch:
        0x48, 0x89, 0xc8, // mov rax, rcx
        0x4c, 0x8d, 0x0d, 0xec, 0xff, 0xff, 0xff, // lea r9 [base]
        0x43, 0x8b, 0x8c, 0x81, 0x21, 0x00, 0x00, 0x00, // mov ecx, [r9 + r8 * 4 + switch_table - base]
        0x4c, 0x01, 0xc9, // add rcx, r9
        0xff, 0xe1, // jmp rcx
        // switch_table:
        0x31, 0x00, 0x00, 0x00,
        0x33, 0x00, 0x00, 0x00,
        0x3a, 0x00, 0x00, 0x00,
        0x3a, 0x00, 0x00, 0x00,
        // case0:
        0x31, 0xf6, // xor esi, esi
        // case1:
        0x31, 0xff, // xor edi, edi
        0x31, 0xed, // xor rbp, rbp
        // end
        0xeb, 0x00,
        0xc3, // ret
        // case2_fake, case3_fake
        0xcc, // int3
    ], &[
         (operand_register(0), ctx.undefined_rc()),
         (operand_register(1), ctx.undefined_rc()),
         (operand_register(2), ctx.undefined_rc()),
         (operand_register(5), ctx.undefined_rc()),
         (operand_register(6), ctx.undefined_rc()),
         (operand_register(7), ctx.undefined_rc()),
         (operand_register(9), ctx.undefined_rc()),
    ]);
}

#[test]
fn dec_flags() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0x33, 0xc0, // xor eax, eax
        0x33, 0xd2, // xor eax, eax
        0x4d, 0x85, 0xc0, // test r8, r8
        0x74, 0x0c, // je set_edx_1
        0x49, 0xff, 0xc8, // dec r8
        0x74, 0x02, // je set_eax_1
        0xeb, 0x08, // jmp end
        // set_eax_1
        0x83, 0xc8, 0x01, // or eax, 1
        0xeb, 0x03, // jmp end
        // set_edx_1
        0x83, 0xca, 0x01, // or eax, 1
        // end:
        0xeb, 0x00,
        0xc3, // ret
    ], &[
         (operand_register(0), ctx.undefined_rc()),
         (operand_register(2), ctx.undefined_rc()),
         (operand_register(8), ctx.undefined_rc()),
    ]);
}

#[test]
fn sub_flags() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0x85, 0xd2, // test edx, edx
        0x74, 0x0f, // je set_ecx_1
        0x83, 0xea, 0x01, // sub edx, 1
        0x74, 0x05, // je set_eax_1
        0x83, 0xcb, 0x01, // or ebx, 1
        0xeb, 0x08, // jmp end
        // set_eax_1
        0x83, 0xc8, 0x01, // or eax, 1
        0xeb, 0x03, // jmp end
        // set_ecx_1
        0x83, 0xc9, 0x01, // or ecx, 1
        // end:
        0xeb, 0x00,
        0xc3, // ret
    ], &[
         (operand_register(0), ctx.undefined_rc()),
         (operand_register(1), ctx.undefined_rc()),
         (operand_register(2), ctx.undefined_rc()),
         (operand_register(3), ctx.undefined_rc()),
    ]);
}

#[test]
fn bswap() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x49, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov r9, 99887766_55443322
        0x48, 0x0f, 0xc9, // bswap rcx
        0x41, 0x0f, 0xc9, // bswap r9d
        0xc3, // ret
    ], &[
         (operand_register(1), ctx.constant(0x22334455_66778899)),
         (operand_register(9), ctx.constant(0x22334455)),
    ]);
}

#[test]
fn sbb() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0xb9, 0x01, 0x00, 0x00, 0x00, // mov ecx, 1
        0x83, 0xf8, 0x01, // cmp eax, 1 (c = 1)
        0x19, 0xc1, // sbb ecx, eax (ecx = 1 - 0 - 1 = 0)
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0)),
    ]);
}

#[test]
fn adc() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x83, 0xf8, 0x01, // cmp eax, 1 (c = 1)
        0x11, 0xc0, // adc eax, eax (eax = 0 + 0 + 1 = 1)
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(1)),
    ]);
}

#[test]
fn movd() {
    let ctx = scarf::operand::OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0x66, 0x0f, 0x6e, 0x04, 0xe4, // movd xmm0, [esp]
        0x0f, 0x11, 0x44, 0xe4, 0x10, // movups [esp + 10], xmm0
        0x8b, 0x44, 0xe4, 0x10, // mov eax, [esp + 10]
        0x8b, 0x4c, 0xe4, 0x14, // mov ecx, [esp + 14]
        0x8b, 0x54, 0xe4, 0x18, // mov edx, [esp + 18]
        0x8b, 0x5c, 0xe4, 0x1c, // mov ebx, [esp + 1c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x12345678)),
         (ctx.register(1), ctx.constant(0)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0)),
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
        if let Operation::Move(_, val, _) = op {
            println!("Resolved is {}", control.resolve(val));
        }
        if let Operation::Jump { condition, .. } = op {
            println!("Resolved condition is {}", control.resolve(condition));
        }
        if let Operation::Return(_) = op {
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
    let (mut end_state, mut end_i) = collect_end_state.end_state.unwrap();
    for i in 0..16 {
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
            assert!(expected_is_ud, "Register {}: got undef {} expected {}", i, end, expected);
        } else {
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
