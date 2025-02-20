extern crate byteorder;
extern crate scarf;

#[allow(dead_code)] mod helpers;
mod shared_x86;

use std::ffi::OsStr;

use byteorder::{ReadBytesExt, LittleEndian};

use scarf::{
    BinaryFile, BinarySection, DestOperand, Operand, Operation, OperandContext, VirtualAddress64,
    MemAccessSize,
};
use scarf::analysis::{self, Control};
use scarf::ExecutionStateX86_64 as ExecutionState;
use scarf::exec_state::ExecutionState as _;
use scarf::exec_state::{OperandCtxExtX86};

#[test]
fn test_basic() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0x20, 0x00, 0x00, 0x00, // mov eax, 20
        0x48, 0x83, 0xc0, 0x39, // add rax, 39
        0xa8, 0x62, // test al, 62
        0x75, 0x01, // jne ret
        0xcc, // int3
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x59)),
    ]);
}

#[test]
fn test_xor_high() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb9, 0x04, 0x00, 0x00, 0x00, // mov ecx, 4
        0x30, 0xfd, // xor ch, bh
        0x30, 0xfd, // xor ch, bh
        0xc3, // ret
    ], &[
         (ctx.register(1), ctx.constant(0x4)),
    ]);
}

#[test]
fn test_neg_mem8() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, 0x04, // mov byte [rbp + 4ea2d325], 4
        0xf6, 0x9d, 0x25, 0xd3, 0xa2, 0x4e, // neg byte [rbp + 4ea2d325]
        0x0f, 0xb6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, // movzx eax, byte [rbp + 4ea2d325]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xfc)),
    ]);
}

#[test]
fn test_neg_mem8_dummy_rex_r() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, 0x04, // mov byte [rbp + 4ea2d325], 4
        0x4e, 0xf6, 0x9d, 0x25, 0xd3, 0xa2, 0x4e, // neg byte [rbp + 4ea2d325]
        0x0f, 0xb6, 0x85, 0x25, 0xd3, 0xa2, 0x4e, // movzx eax, byte [rbp + 4ea2d325]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xfc)),
    ]);
}

#[test]
fn test_new_8bit_regs() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(0x703)),
         (ctx.register(6), ctx.constant(0x203)),
         (ctx.register(9), ctx.constant(0x203)),
         (ctx.register(15), ctx.constant(0x203)),
    ]);
}

#[test]
fn test_64bit_regs() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x49, 0xbf, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, // mov r15, c_0000_000c
        0x31, 0xc0, // xor eax, eax
        0x48, 0x8d, 0x88, 0x88, 0x00, 0x00, 0x00, // lea rcx, [rax + 88]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x88)),
         (ctx.register(15), ctx.constant(0xc_0000_000c)),
    ]);
}

#[test]
fn test_btr() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(0xfeff)),
         (ctx.register(1), ctx.constant(0x8)),
    ]);
}

#[test]
fn test_movsxd() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x44, 0x24, 0x28, 0x44, 0xff, 0x44, 0xff, // mov dword [rsp + 28], ff44ff44
        0x4c, 0x63, 0x5c, 0x24, 0x28, // movsxd r11, dword [rsp + 28]
        0xc3, // ret
    ], &[
         (ctx.register(11), ctx.constant(0xffff_ffff_ff44_ff44)),
    ]);
}

#[test]
fn movaps() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0x24, 0x45, 0x45, 0x45, 0x45, // mov dword [rsp], 45454545
        0xc7, 0x44, 0x24, 0x0c, 0x25, 0x25, 0x25, 0x25, // mov dword [rsp], 25252525
        0x0f, 0x28, 0x04, 0x24, // movaps xmm0, [rsp]
        0x0f, 0x29, 0x44, 0x24, 0x20, // movaps [rsp+20], xmm0
        0x8b, 0x44, 0x24, 0x20, // mov eax, [rsp + 20]
        0x8b, 0x4c, 0x24, 0x2c, // mov ecx, [rsp + 2c]
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0x45454545)),
         (ctx.register(1), ctx.constant(0x25252525)),
    ]);
}

#[test]
fn test_bt() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0xff, 0xff, 0x00, 0x00, // mov eax, ffff
        0x48, 0xc1, 0xe0, 0x20, // shl rax, 20
        0xb9, 0x28, 0x00, 0x00, 0x00, // mov ecx, 28
        0x0f, 0xa3, 0xc8, // bt eax, ecx
        0x73, 0x01, // jnc end
        0xcc, // int3
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xffff_0000_0000)),
         (ctx.register(1), ctx.constant(0x28)),
    ]);
}

#[test]
fn test_xadd() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0x20, 0x00, 0x00, 0x00, // mov eax, 20
        0xc7, 0x04, 0x24, 0x13, 0x00, 0x00, 0x00, // mov dword [rsp], 13
        0xf0, 0x0f, 0xc1, 0x04, 0x24, // lock xadd dword [rsp], eax
        0x8b, 0x0c, 0x24, // mov ecx, [rsp]
        0xba, 0x05, 0x00, 0x00, 0x00, // mov edx, 5
        0x0f, 0xc1, 0xd2, // xadd edx, edx
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x13)),
         (ctx.register(1), ctx.constant(0x33)),
         (ctx.register(2), ctx.constant(0xa)),
    ]);
}

#[test]
fn test_switch() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.new_undef()),
    ]);
}

#[test]
fn test_negative_offset() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x00, 0x05, 0x00, 0x00, 0x00, // mov dword [rax], 5
        0x48, 0x83, 0xc0, 0x04, // add rax, 4
        0x8b, 0x48, 0xfc, // mov ecx, [rax - 4]
        0x48, 0x05, 0x00, 0x01, 0x00, 0x00, // add rax, 100
        0x03, 0x88, 0xfc, 0xfe, 0xff, 0xff, // add ecx, [rax - 104]
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0xa)),
    ]);
}

#[test]
fn test_negative_offset2() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0x10, 0x05, 0x00, 0x00, 0x00, // mov dword [rax + rdx], 5
        0x48, 0x83, 0xc0, 0x04, // add rax, 4
        0x8b, 0x4c, 0x10, 0xfc, // mov ecx, [rax + rdx - 4]
        0x48, 0x05, 0x00, 0x01, 0x00, 0x00, // add rax, 100
        0x03, 0x8c, 0x10, 0xfc, 0xfe, 0xff, 0xff, // add ecx, [rax + rdx - 104]
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0xa)),
    ]);
}

#[test]
fn lazy_flag_constraint_invalidation() {
    let ctx = &OperandContext::new();
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
         (ctx.register(6), ctx.new_undef()),
    ]);
}

#[test]
fn punpcklbw() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(0x5511_4422_3333_2244)),
         (ctx.register(1), ctx.constant(0x9912_8834_7756_6678)),
    ]);
}

#[test]
fn punpcklwd() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x0c, 0x24, // mov [rsp], rcx
        0x66, 0x0f, 0x61, 0x04, 0x24, // punpcklwd xmm0, [rsp]
        0x0f, 0x11, 0x04, 0x24, // movups [rsp], xmm0
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x5544_1122_3322_3344)),
         (ctx.register(1), ctx.constant(0x9988_1234_7766_5678)),
    ]);
}

#[test]
fn punpckldq() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x0c, 0x24, // mov [rsp], rcx
        0x66, 0x0f, 0x62, 0x04, 0x24, // punpckldq xmm0, [rsp]
        0x0f, 0x11, 0x04, 0x24, // movups [rsp], xmm0
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x5544_3322_1122_3344)),
         (ctx.register(1), ctx.constant(0x9988_7766_1234_5678)),
    ]);
}

#[test]
fn punpcklqdq() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x0c, 0x24, // mov [rsp], rcx
        0x66, 0x0f, 0x6c, 0x04, 0x24, // punpcklqdq xmm0, [rsp]
        0x0f, 0x11, 0x04, 0x24, // movups [rsp], xmm0
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x12345678_11223344)),
         (ctx.register(1), ctx.constant(0x99887766_55443322)),
    ]);
}

#[test]
fn unpcklps() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x0c, 0x24, // mov [rsp], rcx
        0x0f, 0x14, 0x04, 0x24, // unpcklps xmm0, [rsp]
        0x0f, 0x11, 0x04, 0x24, // movups [rsp], xmm0
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x5544_3322_1122_3344)),
         (ctx.register(1), ctx.constant(0x9988_7766_1234_5678)),
    ]);
}

#[test]
fn unpcklpd() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x0c, 0x24, // mov [rsp], rcx
        0x66, 0x0f, 0x14, 0x04, 0x24, // unpcklpd xmm0, [rsp]
        0x0f, 0x11, 0x04, 0x24, // movups [rsp], xmm0
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x12345678_11223344)),
         (ctx.register(1), ctx.constant(0x99887766_55443322)),
    ]);
}

#[test]
fn unpack_high() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0x44, 0x33, 0x22, 0x11, 0x78, 0x56, 0x34, 0x12, // mov rax, 12345678_11223344
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x48, 0x89, 0x04, 0x24, // mov [rsp], rax
        0x48, 0x89, 0x4c, 0x24, 0x08, // mov [rsp + 8], rcx
        0xf3, 0x0f, 0x6f, 0x04, 0x24, // movdqu xmm0, [rsp]
        0x48, 0xb8, 0x99, 0x00, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0xff, // mov rax, ffaabbcc_ddee0099
        0x48, 0xb9, 0x55, 0x44, 0xaa, 0xff, 0x34, 0x12, 0xcd, 0xab, // mov rcx, abcd1234_ffaa4455
        0x48, 0x89, 0x04, 0x24, // mov [rsp], rax
        0x48, 0x89, 0x4c, 0x24, 0x08, // mov [rsp + 8], rcx
        0xf3, 0x0f, 0x6f, 0x0c, 0x24, // movdqu xmm1, [rsp]

        0x66, 0x0f, 0x7f, 0xc2, // movdqa xmm2, xmm0
        0x66, 0x0f, 0x68, 0xd1, // punpckhbw xmm2, xmm1
        0x66, 0x0f, 0x7f, 0x14, 0x24, // movdqa [rsp], xmm2
        0x48, 0x8b, 0x04, 0x24, // mov rax, [rsp]
        0x48, 0x8b, 0x4c, 0x24, 0x08, // mov rcx, [rsp + 8]

        0x66, 0x0f, 0x7f, 0xc2, // movdqa xmm2, xmm0
        0x66, 0x0f, 0x69, 0xd1, // punpckhwd xmm2, xmm1
        0x66, 0x0f, 0x7f, 0x14, 0x24, // movdqa [rsp], xmm2
        0x48, 0x8b, 0x14, 0x24, // mov rdx, [rsp]
        0x48, 0x8b, 0x5c, 0x24, 0x08, // mov rbx, [rsp + 8]

        0x66, 0x0f, 0x7f, 0xc2, // movdqa xmm2, xmm0
        0x66, 0x0f, 0x6a, 0xd1, // punpckhdq xmm2, xmm1
        0x66, 0x0f, 0x7f, 0x14, 0x24, // movdqa [rsp], xmm2
        0x48, 0x8b, 0x34, 0x24, // mov rsi, [rsp]
        0x48, 0x8b, 0x7c, 0x24, 0x08, // mov rdi, [rsp + 8]

        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xff55_aa44_4433_5522)),
         (ctx.register(1), ctx.constant(0xab99_cd88_1277_3466)),
         (ctx.register(2), ctx.constant(0xffaa_5544_4455_3322)),
         (ctx.register(3), ctx.constant(0xabcd_9988_1234_7766)),
         (ctx.register(6), ctx.constant(0xffaa4455_55443322)),
         (ctx.register(7), ctx.constant(0xabcd1234_99887766)),
    ]);
}

#[test]
fn switch_case_count2() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(5), ctx.new_undef()),
         (ctx.register(6), ctx.new_undef()),
         (ctx.register(7), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
    ]);
}

#[test]
fn switch_case_count3() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(5), ctx.new_undef()),
         (ctx.register(6), ctx.new_undef()),
         (ctx.register(7), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
    ]);
}

#[test]
fn dec_flags() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(8), ctx.new_undef()),
    ]);
}

#[test]
fn sub_flags() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(3), ctx.new_undef()),
    ]);
}

#[test]
fn bswap() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov rcx, 99887766_55443322
        0x49, 0xb9, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, // mov r9, 99887766_55443322
        0x48, 0x0f, 0xc9, // bswap rcx
        0x41, 0x0f, 0xc9, // bswap r9d
        0xc3, // ret
    ], &[
         (ctx.register(1), ctx.constant(0x22334455_66778899)),
         (ctx.register(9), ctx.constant(0x22334455)),
    ]);
}

#[test]
fn mov_al() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb0, 0x55, // mov al, 55
        0xa2, 0x44, 0x33, 0x22, 0x11, 0xdd, 0xcc, 0xbb, 0xaa, // mov [aabbccdd11223344], al
        0x48, 0xb9, 0x44, 0x33, 0x22, 0x11, 0xdd, 0xcc, 0xbb, 0xaa, // mov rcx, aabbccdd11223344
        0x0f, 0xb6, 0x09, // movzx ecx, byte [rcx]
        0x31, 0xc0, // xor eax, eax
        0xa0, 0x88, 0x77, 0x66, 0x55, 0x78, 0x56, 0x34, 0x12, // mov al, [1234567855667788]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem8(ctx.const_0(), 0x1234567855667788)),
         (ctx.register(1), ctx.constant(0x55)),
    ]);
}

#[test]
fn test_eax_after_call() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x4d, 0x31, 0xff, // xor r15, r15
        0x85, 0xc0, // test eax, eax
        0x75, 0x0c, // jne end
        0xe8, 0x00, 0x00, 0x00, 0x00, // call x
        0x85, 0xc0, // test eax, eax
        0x74, 0x03, // je end
        0x41, 0xb7, 0x01, // mov r15b, 1
        // end
        0xeb, 0x00,
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(8), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
         (ctx.register(10), ctx.new_undef()),
         (ctx.register(11), ctx.new_undef()),
         (ctx.register(15), ctx.new_undef()),
    ]);
}

#[test]
fn switch_u32_op() {
    let ctx = &OperandContext::new();
    // 3 cases to ok, 4th fake
    // rsi, rdi  are undef if the cases are run
    test_inline(&[
        // base:
        0x45, 0x89, 0xc5, // mov r13d, r8d
        0x41, 0x81, 0xfd, 0x04, 0x01, 0x00, 0x00, // cmp r13d, 104
        0x77, 0x32, // ja end
        0x74, 0x30, // je end
        0x41, 0x83, 0xfd, 0x02, // cmp r13d, 2
        0x77, 0x2a, // ja end

        0x4c, 0x8d, 0x0d, 0xe5, 0xff, 0xff, 0xff, // lea r9 [base]
        0x43, 0x8b, 0x8c, 0xa9, 0x28, 0x00, 0x00, 0x00, // mov ecx, [r9 + r13 * 4 + switch_table - base]
        0x4c, 0x01, 0xc9, // add rcx, r9
        0xff, 0xe1, // jmp rcx
        // switch_table:
        0x38, 0x00, 0x00, 0x00,
        0x38, 0x00, 0x00, 0x00,
        0x3c, 0x00, 0x00, 0x00,
        0x41, 0x00, 0x00, 0x00,
        // case0,1:
        0x31, 0xf6, // xor esi, esi
        0xeb, 0x02, // jmp end
        // case2:
        0x31, 0xff, // xor edi, edi
        // end
        0xeb, 0x00,
        0xc3, // ret
        // case3_fake
        0xcc, // int3
    ], &[
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(6), ctx.new_undef()),
         (ctx.register(7), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
         (ctx.register(13), ctx.and(ctx.register(8), ctx.constant(0xffff_ffff))),
    ]);
}

#[test]
fn movd_to_reg() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x02, 0x00, 0x00, 0x00, // mov [esp], 2
        0xc7, 0x44, 0xe4, 0x04, 0x04, 0x00, 0x00, 0x00, // mov [esp + 4], 4
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x7e, 0xc0, // movd eax, xmm0
        0x66, 0x48, 0x0f, 0x7e, 0xc1, // movq rcx, xmm0
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x2)),
         (ctx.register(1), ctx.constant(0x4_0000_0002)),
    ]);
}

#[test]
fn move_mem_in_parts() {
    let ctx = &OperandContext::new();
    // Issue was that eax was being assigned esi,
    // and esi was zeroed afterwards, but the small stores
    // kept assuming that unmodified value of `Mem32[eax]` == `Mem32[0]`
    test_inline(&[
        0x48, 0x89, 0xf0, // mov rax, rsi
        0x31, 0xf6, // xor esi, esi
        0x31, 0xc9, // xor ecx, ecx
        0x88, 0x08, // mov [eax], cl
        0x66, 0x8b, 0x8e, 0x30, 0x12, 0x00, 0x00, // mov cx, [rsi + 1230]
        0x66, 0x89, 0x08, // mov [eax], cx
        0x8a, 0x8e, 0x32, 0x12, 0x00, 0x00, // mov cl, [rsi + 1232]
        0x88, 0x48, 0x02, // mov [eax + 2], cl
        0x8a, 0x8e, 0x33, 0x12, 0x00, 0x00, // mov cl, [rsi + 1233]
        0x88, 0x48, 0x03, // mov [eax + 3], cl
        0x31, 0xc9, // xor ecx, ecx
        0x8b, 0x00, // mov eax, [rax]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem32(ctx.const_0(), 0x1230)),
         (ctx.register(1), ctx.constant(0)),
         (ctx.register(6), ctx.constant(0)),
    ]);
}

#[test]
fn movzx_movsx_high_reg2() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x41, 0xb9, 0xf0, 0x90, 0x00, 0x00, // mov r9d, 90f0
        0x41, 0xbc, 0x80, 0x90, 0x00, 0x00, // mov r12d, 9080
        0x41, 0x0f, 0xb6, 0xc1, // movzx eax, r9b
        0x41, 0x0f, 0xb6, 0xcc, // movzx ecx, r12b
        0x41, 0x0f, 0xbe, 0xd1, // movsx edx, r9b
        0x41, 0x0f, 0xbe, 0xdc, // movsx ebx, r12b
        0x4d, 0x0f, 0xbe, 0xc9, // movsx r9, r9b
        0x4d, 0x0f, 0xb6, 0xd4, // movzx r10, r12b
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xf0)),
         (ctx.register(1), ctx.constant(0x80)),
         (ctx.register(2), ctx.constant(0xffff_fff0)),
         (ctx.register(3), ctx.constant(0xffff_ff80)),
         (ctx.register(9), ctx.constant(0xffff_ffff_ffff_fff0)),
         (ctx.register(10), ctx.constant(0x80)),
         (ctx.register(12), ctx.constant(0x9080)),
    ]);
}

#[test]
fn jump_eq() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0x34, 0x12, 0x00, 0x00, // mov eax, 1234
        0xb9, 0x03, 0x00, 0x00, 0x00, // mov ecx, 3
        0x8b, 0x00, // mov eax, [eax]
        0x3b, 0xc8, // cmp ecx, eax (carry set if eax > ecx)
        0x1b, 0xc0, // sbb eax, eax (eax = ffff_ffff if carry set)
        0xff, 0xc0, // inc eax (eax = (eax > ecx) == 0)
        0x85, 0xc0, // test eax, eax
        0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
        0x0f, 0x94, 0xc0, // sete al (eax = eax > ecx)
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.gt(ctx.mem32(ctx.const_0(), 0x1234), ctx.constant(3))),
         (ctx.register(1), ctx.constant(3)),
    ]);
}

#[test]
fn xmm_u128_left_shift2() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0x04, 0xe4, // mov [esp], eax
        0x89, 0x4c, 0xe4, 0x04, // mov [esp + 4], ecx
        0x89, 0x54, 0xe4, 0x08, // mov [esp + 8], edx
        0x89, 0x5c, 0xe4, 0x0c, // mov [esp + c], ebx
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xf8, 0x0a, // pslldq xmm0, a
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0)),
        (ctx.register(1), ctx.constant(0)),
        (ctx.register(2), ctx.and_const(ctx.lsh_const(ctx.register(0), 0x10), 0xffff_ffff)),
        (ctx.register(3), ctx.or(
            ctx.and_const(ctx.rsh_const(ctx.register(0), 0x10), 0xffff),
            ctx.and_const(ctx.lsh_const(ctx.register(1), 0x10), 0xffff_0000),
        )),
    ]);
}

#[test]
fn xmm_u128_right_shift2() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0x04, 0xe4, // mov [esp], eax
        0x89, 0x4c, 0xe4, 0x04, // mov [esp + 4], ecx
        0x89, 0x54, 0xe4, 0x08, // mov [esp + 8], edx
        0x89, 0x5c, 0xe4, 0x0c, // mov [esp + c], ebx
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xd8, 0x0a, // psrldq xmm0, a
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.or(
            ctx.and_const(ctx.rsh_const(ctx.register(2), 0x10), 0xffff),
            ctx.and_const(ctx.lsh_const(ctx.register(3), 0x10), 0xffff_0000),
        )),
        (ctx.register(1), ctx.and_const(ctx.rsh_const(ctx.register(3), 0x10), 0xffff)),
        (ctx.register(2), ctx.constant(0)),
        (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn absolute_address() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0x8b, 0x0c, 0x25, 0x34, 0x12, 0x00, 0x00, // mov rcx, [1234]
        0xc3, // ret
    ], &[
         (ctx.register(1), ctx.mem64(ctx.const_0(), 0x1234)),
    ]);
}

#[test]
fn test_switch_cases_in_memory() {
    let ctx = &OperandContext::new();
    // 2 cases to ok, 3rd fake
    // rcx is undef if the cases are run
    test(0, &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.constant(0)),
    ]);
}

#[test]
fn lea_sizes() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x48, 0xb8, 0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12, // mov rax, 1234567890abcdef
        0x67, 0x8d, 0x08, // lea ecx, [eax]
        0x67, 0x48, 0x8d, 0x10, // lea rdx, [eax]
        0x8d, 0x30, // lea esi, [rax]
        0x48, 0x8d, 0x38, // lea rdi, [rax]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0x1234567890abcdef)),
        (ctx.register(1), ctx.constant(0x90abcdef)),
        (ctx.register(2), ctx.constant(0x90abcdef)),
        (ctx.register(6), ctx.constant(0x90abcdef)),
        (ctx.register(7), ctx.constant(0x1234567890abcdef)),
    ]);
}

#[test]
fn cvtsi2sd_or_ss_also_r64() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xb8, 0x56, 0x34, 0x12, 0x80, // mov eax, 80123456
        0xf3, 0x0f, 0x2a, 0xc0, // cvtsi2ss xmm0, eax
        0xf3, 0x48, 0x0f, 0x2a, 0xc8, // cvtsi2ss xmm1, rax
        0xf2, 0x0f, 0x2a, 0xd0, // cvtsi2sd xmm2, eax
        0xf2, 0x48, 0x0f, 0x2a, 0xd8, // cvtsi2sd xmm3, rax
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0x80123456)),
        (ctx.xmm(0, 0), ctx.constant(0xCEFFDB97)),
        (ctx.xmm(1, 0), ctx.constant(0x4F001234)),
        (ctx.xmm(2, 0), ctx.constant(0xEA800000)),
        (ctx.xmm(2, 1), ctx.constant(0xC1DFFB72)),
        (ctx.xmm(3, 0), ctx.constant(0x8AC00000)),
        (ctx.xmm(3, 1), ctx.constant(0x41E00246)),
    ]);
}

#[test]
fn cvtsd2si() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xb8, 0x00, 0x20, 0x67, 0xc0, // mov eax, c0672000
        0x48, 0xc1, 0xe0, 0x20, // shl rax, 20
        0x66, 0x48, 0x0f, 0x6e, 0xc0, // movq xmm0, rax
        0xf2, 0x48, 0x0f, 0x2d, 0xc0, // cvtsd2si rax, xmm0
        0xf2, 0x0f, 0x2d, 0xc8, // cvtsd2si ecx, xmm0
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0xffff_ffff_ffff_ff47)),
        (ctx.register(1), ctx.constant(0xffff_ff47)),
        (ctx.xmm(0, 0), ctx.constant(0x0)),
        (ctx.xmm(0, 1), ctx.constant(0xc0672000)),
        (ctx.xmm(0, 2), ctx.constant(0x0)),
        (ctx.xmm(0, 3), ctx.constant(0x0)),
    ]);
}

#[test]
fn switch_different_resolved_constraints_on_branch_end() {
    let ctx = &OperandContext::new();
    // 3 cases to ok, 4th fake
    // Contains multiple different branches going to switch_start with different
    // resolved values but same unresolved values so switch_start could still have
    // useful constraint.
    test(1, &[
         (ctx.register(0), ctx.mem64(ctx.new_undef(), 0)),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(6), ctx.new_undef()),
         (ctx.register(7), ctx.new_undef()),
         (ctx.register(8), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
         (ctx.register(13), ctx.mem32(ctx.mem64(ctx.new_undef(), 0), 8)),
    ]);
}

#[test]
fn switch_u32_with_sub() {
    let ctx = &OperandContext::new();
    test(2, &[
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(6), ctx.new_undef()),
         (ctx.register(7), ctx.new_undef()),
         (ctx.register(8), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
    ]);
}

#[test]
fn call_clears_pending_flags() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x85, 0xc0, // test eax, eax
        0xff, 0xd1, // call ecx
        0xb9, 0x00, 0x00, 0x00, 0x00, // mov ecx, 0
        0x74, 0x03, // je skip_add
        0x83, 0xc1, 0x03, // add ecx, 3
        // skip_add:
        0xeb, 0x00,
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(8), ctx.new_undef()),
         (ctx.register(9), ctx.new_undef()),
         (ctx.register(10), ctx.new_undef()),
         (ctx.register(11), ctx.new_undef()),
    ]);
}

#[test]
fn rdtsc_is_32_bits() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x0f, 0x31, // rdtsc
        0x48, 0xc1, 0xe8, 0x20, // shr rax, 20
        0x48, 0xc1, 0xea, 0x20, // shr rdx, 20
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.const_0()),
         (ctx.register(2), ctx.const_0()),
    ]);
}

#[test]
fn misc_coverage() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x03, 0x05, 0xfa, 0xff, 0x0f, 0x00, // add eax, [rip + 0010_0000] = [0050_1002]
        0xc3, // ret
    ], &[
         (
            ctx.register(0),
            ctx.mem32c(0x0050_1002),
         ),
    ]);
}

#[test]
fn mov_eax_eax() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0xc0, // mov eax, eax
        0x66, 0x89, 0xc9, // mov cx, cx
        0x88, 0xd2, // mov dl, dl
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.and_const(ctx.register(0), 0xffff_ffff)),
    ]);
}

#[test]
fn u16_push() {
    let ctx = &OperandContext::new();

    test_inline(&[
        // Pushes only u16 (Probs illegal in any OS ABI stack pointer requirements)
        0x66, 0x6a, 0xfd, // push -3
        0x66, 0x68, 0x34, 0x12, // push 1234
        0x66, 0x68, 0x11, 0x11, // push 1111
        0x66, 0x6a, 0xfe, // push -2
        0x58, // pop rax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xfffd_1234_1111_fffe)),
    ]);
}

#[test]
fn not_binary_mem_read() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0x48, 0x8b, 0x80, 0x00, 0x10, 0x40, 0x00, // mov rax, [rax + 401000]
        0xc6, 0x81, 0x00, 0x10, 0x40, 0x00, 0x05, // mov byte [rcx + 401000], 5
        0x48, 0x8b, 0x89, 0x00, 0x10, 0x40, 0x00, // mov rcx, [rcx + 401000]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem64(ctx.register(0), 0x401000)),
         (
            ctx.register(1),
            ctx.or(
                ctx.and_const(
                    ctx.mem64(ctx.register(1), 0x401000),
                    0xffff_ffff_ffff_ff00,
                ),
                ctx.constant(5),
            ),
         ),
    ]);
}

#[test]
fn mem_oversized_writes() {
    // Verify that memory writes values larger than the MemAccess size won't break
    // following memory.
    let ctx = &OperandContext::new();
    let mut state = ExecutionState::new(ctx);
    let dest = |addr, size| DestOperand::from_oper(ctx.mem_any(size, ctx.register(1), addr));
    state.move_resolved(&dest(0x100, MemAccessSize::Mem32), ctx.constant(0));
    state.move_resolved(&dest(0x104, MemAccessSize::Mem32), ctx.constant(0));
    state.move_resolved(&dest(0x108, MemAccessSize::Mem32), ctx.constant(0));
    state.move_resolved(&dest(0x10c, MemAccessSize::Mem32), ctx.constant(0));
    state.move_resolved(&dest(0x100, MemAccessSize::Mem8), ctx.custom(0));
    state.move_resolved(&dest(0x105, MemAccessSize::Mem16), ctx.custom(1));
    state.move_resolved(&dest(0x109, MemAccessSize::Mem32), ctx.custom(2));

    assert_eq!(
        state.resolve(ctx.mem32(ctx.register(1), 0x100)),
        ctx.and_const(
            ctx.custom(0),
            0xff,
        ),
    );
    assert_eq!(
        state.resolve(ctx.mem32(ctx.register(1), 0x104)),
        ctx.lsh_const(
            ctx.and_const(
                ctx.custom(1),
                0xffff,
            ),
            8,
        ),
    );
    assert_eq!(
        state.resolve(ctx.mem32(ctx.register(1), 0x108)),
        ctx.lsh_const(
            ctx.and_const(
                ctx.custom(2),
                0xffff_ff,
            ),
            8,
        ),
    );
    assert_eq!(
        state.resolve(ctx.mem32(ctx.register(1), 0x10c)),
        ctx.rsh_const(
            ctx.and_const(
                ctx.custom(2),
                0xffff_ffff,
            ),
            0x18,
        ),
    );
}

#[test]
fn switch_negative_cases() {
    let ctx = &OperandContext::new();

    test(3, &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(3), ctx.new_undef()),
    ]);
}

struct CollectEndState<'e> {
    end_state: Option<ExecutionState<'e>>,
}

impl<'e> analysis::Analyzer<'e> for CollectEndState<'e> {
    type State = analysis::DefaultState;
    type Exec = ExecutionState<'e>;
    fn operation(&mut self, control: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        println!("@ {:x} {:#?}", control.address(), op);
        match *op {
            Operation::Move(_, val) => {
                let val = control.resolve(val);
                println!("Resolved is {}", val);
            }
            Operation::ConditionalMove(_, val, cond) => {
                let val = control.resolve(val);
                let cond = control.resolve(cond);
                println!("Resolved is {} cond {}", val, cond);
            }
            Operation::Jump { condition, .. } => {
                println!("Resolved cond is {}", control.resolve(condition));
                println!(
                    "Resolved cond is {} (Constraints applied)",
                    control.resolve_apply_constraints(condition),
                );
            }
            Operation::Return(_) => {
                let state = control.exec_state();
                self.end_state = Some(state.clone());
            }
            Operation::Error(e) => {
                panic!("Disassembly error {}", e);
            }
            _ => (),
        }
    }
}

#[test]
fn read_in_virtual_only_bytes() {
    // Reading at data that is in text.virtual_address, text.virtual_size
    // range but not in data bytes.
    // Just regression test against crashing, it could also result in 0
    // or partially known value, but not implementing that right now.
    let file = &make_binary_with_virtual_size(&[
        0xb8, 0x00, 0x10, 0x40, 0x00, // mov eax, 00401000
        // Read past data end
        0x8b, 0x48, 0x20, // mov ecx, [eax + 20]
        // Reads two last bytes and two not-in-data bytes (10, c3, xx, xx)
        0x8b, 0x48, 0x10, // mov eax, [eax + 10]
        0xc3, // ret
    ], 0x1000);
    let func = file.code_section().virtual_address;
    let ctx = &OperandContext::new();

    let state = ExecutionState::with_binary(file, ctx);
    let mut analysis = analysis::FuncAnalysis::with_state(file, ctx, func, state);
    let mut collect_end_state = CollectEndState {
        end_state: None,
    };
    analysis.analyze(&mut collect_end_state);
}

fn test_inner<'e, 'b>(
    file: &'e BinaryFile<VirtualAddress64>,
    func: VirtualAddress64,
    changes: &[(Operand<'b>, Operand<'b>)],
    init: &[(Operand<'b>, Operand<'b>)],
    xmm: bool,
) {
    let ctx = &OperandContext::new();
    let expected = changes.iter().map(|&(a, b)| {
        (ctx.copy_operand(a), ctx.copy_operand(b))
    }).collect::<Vec<_>>();
    let init = init.iter().map(|&(a, b)| {
        (ctx.copy_operand(a), ctx.copy_operand(b))
    }).collect::<Vec<_>>();

    let mut state = ExecutionState::with_binary(file, ctx);
    for &(op, val) in &init {
        state.move_resolved(&DestOperand::from_oper(op), val);
    }
    let mut analysis = analysis::FuncAnalysis::with_state(file, ctx, func, state);
    let mut collect_end_state = CollectEndState {
        end_state: None,
    };
    analysis.analyze(&mut collect_end_state);

    let mut end_state = collect_end_state.end_state.unwrap();
    for i in 0..16 {
        let reg = ctx.register(i);
        if expected.iter().any(|x| x.0 == reg) {
            continue;
        }
        let end = end_state.resolve(reg);
        assert_eq!(end, reg, "Register {}: got {} expected {}", i, end, reg);
    }
    if xmm {
        for i in 0..16 {
            for j in 0..4 {
                let reg = ctx.xmm(i, j);
                if expected.iter().any(|x| x.0 == reg) {
                    continue;
                }
                let end = end_state.resolve(reg);
                assert_eq!(end, reg, "XMM {}: got {} expected {}", i, end, reg);
            }
        }
    }
    for &(op, val) in &expected {
        let val2 = end_state.resolve(op);
        if val2.contains_undefined() {
            let replace = ctx.custom(0x100);
            let cmp1 =
                ctx.transform(val, 16, |x| if x.is_undefined() { Some(replace) } else { None });
            let cmp2 =
                ctx.transform(val2, 16, |x| if x.is_undefined() { Some(replace) } else { None });
            assert_eq!(cmp1, cmp2, "Operand {op}: got {} expected {}", val2, val);
        } else {
            assert_eq!(val, val2, "Operand {op}: got {val2} expected {val}");
        }
    }
}

fn make_binary_with_virtual_size(code: &[u8], virtual_size: u32) -> BinaryFile<VirtualAddress64> {
    scarf::raw_bin(VirtualAddress64(0x00400000), vec![BinarySection {
        name: *b".text\0\0\0",
        virtual_address: VirtualAddress64(0x401000),
        virtual_size,
        data: code.into(),
    }])
}

fn make_binary(code: &[u8]) -> BinaryFile<VirtualAddress64> {
    make_binary_with_virtual_size(code, code.len() as u32)
}

fn test_inline_xmm<'e>(code: &[u8], changes: &[(Operand<'e>, Operand<'e>)]) {
    let binary = make_binary(code);
    test_inner(&binary, binary.code_section().virtual_address, changes, &[], true);
}

fn test_inline<'e>(code: &[u8], changes: &[(Operand<'e>, Operand<'e>)]) {
    let binary = make_binary(code);
    test_inner(&binary, binary.code_section().virtual_address, changes, &[], false);
}

fn test_inline_with_init<'e>(
    code: &[u8],
    changes: &[(Operand<'e>, Operand<'e>)],
    init: &[(Operand<'e>, Operand<'e>)],
) {
    let binary = make_binary(code);
    test_inner(&binary, binary.code_section().virtual_address, changes, init, false);
}

fn test<'b>(idx: usize, changes: &[(Operand<'b>, Operand<'b>)]) {
    let binary = helpers::raw_bin_64(OsStr::new("test_inputs/exec_state_x86_64.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 8..]).read_u64::<LittleEndian>().unwrap();
    let func = VirtualAddress64(offset);
    test_inner(&binary, func, changes, &[], false);
}
