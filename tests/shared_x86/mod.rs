//! 32/64-bit x86 tests

use super::{test_inline, test_inline_xmm, test_inline_with_init};

use scarf::{OperandCtx, OperandContext, Operand};

fn is_64bit() -> bool {
    use scarf::exec_state::VirtualAddress;
    <super::ExecutionState as scarf::ExecutionState<'_>>::VirtualAddress::SIZE == 8
}

fn mask_if_64bit<'e>(ctx: OperandCtx<'e>, left: Operand<'e>, right: u64) -> Operand<'e> {
    if !is_64bit() {
        left
    } else {
        ctx.and_const(left, right)
    }
}

#[test]
fn overflow_not_set_bug() {
    let ctx = &OperandContext::new();
    // Had a bug that made the `jl maybe` never be taken
    test_inline(&[
        0xbf, 0x01, 0x00, 0x00, 0x00, // mov edi, 1
        0x0f, 0xbf, 0x46, 0x16, // movsx eax, word [esi + 16]
        0x89, 0xc1, // mov ecx, eax,
        0xc1, 0xf9, 0x05, // sar ecx, 5
        0x85, 0xc9, // test ecx, ecx
        0x79, 0x04, // jns more
        0x33, 0xc9, // xor ecx, ecx
        0xeb, 0x12, // jmp end
        // more:
        0x0f, 0xb7, 0x82, 0xe6, 0x00, 0x00, 0x00, // movzx eax word [edx + e6]
        0x3b, 0xc8, // cmp ecx, eax
        0x7c, 0x02, // jl maybe
        0xeb, 0x07, // jmp end
        0xbf, 0x02, 0x00, 0x00, 0x00, // mov edi, 2
        0xeb, 0x00, // jmp end
        // end:
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(7), ctx.new_undef()),
    ]);
}

#[test]
fn sbb() {
    let ctx = &OperandContext::new();
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
    let ctx = &OperandContext::new();
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
    let ctx = &OperandContext::new();
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

#[test]
fn partial_overwrite() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x00, 0x78, 0x56, 0x34, 0x12, // mov [eax], 12345678
        0xc6, 0x00, 0x99, // mov byte [eax], 99
        0xc7, 0x41, 0x01, 0x78, 0x56, 0x34, 0x12, // mov [ecx + 1], 12345678
        0xc6, 0x41, 0x01, 0xaa, // mov byte [ecx + 1], aa
        0x8b, 0x00, // mov eax, [eax]
        0x8b, 0x49, 0x01, // mov ecx, [ecx + 1]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x12345699)),
         (ctx.register(1), ctx.constant(0x123456aa)),
    ]);
}

// Exact behaviour on the case of code being executed is not
// specified, but unrelated bytes in code section are assumed
// to be modifiable and the modifications are assumed to be readable back.
#[test]
fn modifying_code_section() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0x11, 0x10, 0x40, 0x00, // mov eax, 00401011 (&bytes)
        0x66, 0xc7, 0x40, 0x04, 0x88, 0x77, // mov [eax + 4], word 7788
        0x8b, 0x08, // mov ecx, [eax]
        0x8b, 0x40, 0x04, // mov eax, [eax + 4]
        0xc3, // ret
        // bytes:
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
    ], &[
         (ctx.register(0), ctx.constant(0x18177788)),
         (ctx.register(1), ctx.constant(0x14131211)),
    ]);
}

#[test]
fn sse_i32_to_f64() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x02, 0x00, 0x00, 0x00, // mov [esp], 2
        0xc7, 0x44, 0xe4, 0x04, 0x05, 0x00, 0x00, 0x00, // mov [esp + 4], 5
        0xc7, 0x44, 0xe4, 0x08, 0x63, 0x00, 0x00, 0x00, // mov [esp + 8], 63
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0xf3, 0x0f, 0xe6, 0xc8, // cvtdq2pd xmm1, xmm0
        0x0f, 0x11, 0x0c, 0xe4, // movups [esp], xmm1
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x4000_0000)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0x4014_0000)),
    ]);
}

#[test]
fn sse_i32_to_f64_2() {
    let ctx = &OperandContext::new();
    // test cvtdq2pd with same input and out
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x02, 0x00, 0x00, 0x00, // mov [esp], 2
        0xc7, 0x44, 0xe4, 0x04, 0x05, 0x00, 0x00, 0x00, // mov [esp + 4], 5
        0xc7, 0x44, 0xe4, 0x08, 0x63, 0x00, 0x00, 0x00, // mov [esp + 8], 63
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0xf3, 0x0f, 0xe6, 0xc0, // cvtdq2pd xmm0, xmm0
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x4000_0000)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0x4014_0000)),
    ]);
}

#[test]
fn sse_f64_to_i32() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x00, 0x00, 0x00, 0x00, // mov [esp], 0
        0xc7, 0x44, 0xe4, 0x04, 0x00, 0x00, 0x00, 0x40, // mov [esp + 4], 4000_0000
        0xc7, 0x44, 0xe4, 0x08, 0x00, 0x00, 0x00, 0x00, // mov [esp + 8], 0
        0xc7, 0x44, 0xe4, 0x0c, 0x00, 0x00, 0x14, 0x40, // mov [esp + c], 4014_0000
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0xf2, 0x0f, 0xe6, 0xc8, // cvtpd2dq xmm1, xmm0
        0x0f, 0x11, 0x0c, 0xe4, // movups [esp], xmm1
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(2)),
         (ctx.register(1), ctx.constant(5)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn sse_f64_to_i32_2() {
    let ctx = &OperandContext::new();
    // test cvtdq2pd with same input and out
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x00, 0x00, 0x00, 0x00, // mov [esp], 0
        0xc7, 0x44, 0xe4, 0x04, 0x00, 0x00, 0x00, 0x40, // mov [esp + 4], 4000_0000
        0xc7, 0x44, 0xe4, 0x08, 0x00, 0x00, 0x00, 0x00, // mov [esp + 8], 0
        0xc7, 0x44, 0xe4, 0x0c, 0x00, 0x00, 0x14, 0x40, // mov [esp + c], 4014_0000
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0xf2, 0x0f, 0xe6, 0xc0, // cvtpd2dq xmm0, xmm0
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(2)),
         (ctx.register(1), ctx.constant(5)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn sse_packed_shift1() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x23, 0x01, 0x00, 0x00, // mov [esp], 123
        0xc7, 0x44, 0xe4, 0x04, 0x56, 0x04, 0x00, 0x00, // mov [esp + 4], 456
        0xc7, 0x44, 0xe4, 0x08, 0x89, 0x07, 0x00, 0x00, // mov [esp + 8], 789
        0xc7, 0x44, 0xe4, 0x0c, 0xbc, 0x0a, 0x00, 0x00, // mov [esp + c], abc
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xf0, 0x25, // psllq xmm0, 0x25
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x2460)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0xf120)),
    ]);
}

#[test]
fn sse_packed_shift2() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x23, 0x01, 0x00, 0x00, // mov [esp], 123
        0xc7, 0x44, 0xe4, 0x04, 0x56, 0x04, 0x00, 0x00, // mov [esp + 4], 456
        0xc7, 0x44, 0xe4, 0x08, 0x89, 0x07, 0x00, 0x00, // mov [esp + 8], 789
        0xc7, 0x44, 0xe4, 0x0c, 0xbc, 0x0a, 0x00, 0x00, // mov [esp + c], abc
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xd0, 0x25, // psrlq xmm0, 0x25
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x22)),
         (ctx.register(1), ctx.constant(0)),
         (ctx.register(2), ctx.constant(0x55)),
         (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn sse_packed_shift3() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x23, 0x01, 0x00, 0x00, // mov [esp], 123
        0xc7, 0x44, 0xe4, 0x04, 0x56, 0x04, 0x00, 0x00, // mov [esp + 4], 456
        0xc7, 0x44, 0xe4, 0x08, 0x89, 0x07, 0x00, 0x00, // mov [esp + 8], 789
        0xc7, 0x44, 0xe4, 0x0c, 0xbc, 0x0a, 0x00, 0x00, // mov [esp + c], abc
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xf8, 0x07, // pslldq xmm0, 0x7
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x2300_0000)),
         (ctx.register(2), ctx.constant(0x5600_0001)),
         (ctx.register(3), ctx.constant(0x8900_0004)),
    ]);
}

#[test]
fn sse_packed_shift4() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x23, 0x01, 0x00, 0x00, // mov [esp], 123
        0xc7, 0x44, 0xe4, 0x04, 0x56, 0x04, 0x00, 0x00, // mov [esp + 4], 456
        0xc7, 0x44, 0xe4, 0x08, 0x89, 0x07, 0x00, 0x00, // mov [esp + 8], 789
        0xc7, 0x44, 0xe4, 0x0c, 0xbc, 0x0a, 0x00, 0x00, // mov [esp + c], abc
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xd8, 0x07, // psrldq xmm0, 0x7
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x0007_8900)),
         (ctx.register(1), ctx.constant(0x000a_bc00)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn sse_f32_f64_arith() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x02, 0x00, 0x00, 0x00, // mov [esp], 2
        0xc7, 0x44, 0xe4, 0x04, 0x05, 0x00, 0x00, 0x00, // mov [esp + 4], 5
        0xc7, 0x44, 0xe4, 0x08, 0x89, 0x07, 0x00, 0x00, // mov [esp + 8], 789
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x0f, 0x5b, 0xc0, // cvtd1ps xmm0, xmm0
        0x0f, 0x10, 0xc8, // movups xmm1, xmm0
        0x0f, 0x58, 0xc1, // addps xmm0, xmm1
        0x0f, 0x5a, 0xc0, // cvtps2pd xmm0, xmm0
        0x0f, 0x5a, 0xc9, // cvtps2pd xmm1, xmm1
        0x66, 0x0f, 0x59, 0xc1, // mulpd xmm0, xmm1
        0x66, 0x0f, 0x11, 0x04, 0xe4, // movupd [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x40200000)),
         (ctx.register(2), ctx.constant(0)),
         (ctx.register(3), ctx.constant(0x40490000)),
    ]);
}

#[test]
fn overflow_flag() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x31, 0xc9, // xor ecx, ecx
        0x31, 0xd2, // xor edx, edx
        0x31, 0xdb, // xor ebx, ebx
        0xb1, 0x80, // mov cl, 80
        0x28, 0xc8, // sub al, cl
        0x0f, 0x90, 0xc2, // seto dl (0 - -80 = overflow)
        0x31, 0xc0, // xor eax, eax
        0x00, 0xc8, // add al, cl
        0x0f, 0x90, 0xc3, // seto bl (0 + -80 = no overflow)
        0xb1, 0x80, // mov cl, 0
        0x28, 0xc8, // sub al, cl
        0x0f, 0x90, 0xc6, // seto dh (0 - 0 = no overflow)
        0x00, 0xc8, // add al, cl
        0x0f, 0x90, 0xc7, // seto bh (0 + 0 = no overflow)
        0x31, 0xc0, // xor eax, eax
        0x89, 0xd6, // mov esi, edx (0x00_01)
        0x89, 0xdf, // mov edi, ebx (0x00_00)

        0x31, 0xd2, // xor edx, edx
        0x31, 0xdb, // xor ebx, ebx
        0xb1, 0x80, // mov cl, 80
        0xb0, 0xff, // mov al, ff
        0x28, 0xc8, // sub al, cl
        0x0f, 0x90, 0xc2, // seto dl (-1 - -80 = no overflow)
        0xb0, 0xff, // mov al, ff
        0x00, 0xc8, // add al, cl
        0x0f, 0x90, 0xc3, // seto bl (-1 + -80 = overflow)
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x7f)),
         (ctx.register(1), ctx.constant(0x80)),
         (ctx.register(2), ctx.constant(0x0)),
         (ctx.register(3), ctx.constant(0x1)),
         (ctx.register(6), ctx.constant(0x1)),
         (ctx.register(7), ctx.constant(0x0)),
    ]);
}

#[test]
fn sign_flag_u32() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x31, 0xd2, // xor edx, edx
        0xb9, 0x00, 0x00, 0x00, 0x80, // mov ecx, 8000_0000
        0x01, 0xc8, // add eax, ecx
        0x0f, 0x98, 0xc2, // sets dl
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x8000_0000)),
         (ctx.register(1), ctx.constant(0x8000_0000)),
         (ctx.register(2), ctx.constant(1)),
    ]);
}

#[test]
fn movzx_movsx_high_reg() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xbb, 0xf0, 0xe0, 0xd0, 0xc0, // mov ebx, c0d0e0f0
        0x0f, 0xb6, 0xc3, // movzx eax, bl
        0x0f, 0xb6, 0xcf, // movzx ecx, bh
        0x0f, 0xbe, 0xd3, // movsx edx, bl
        0x0f, 0xbe, 0xdf, // movsx ebx, bh
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xf0)),
         (ctx.register(1), ctx.constant(0xe0)),
         (ctx.register(2), ctx.constant(0xffff_fff0)),
         (ctx.register(3), ctx.constant(0xffff_ffe0)),
    ]);
}

#[test]
fn merge_mem_to_undef() {
    let ctx = &OperandContext::new();
    // eax should be [esp], but ecx is undefined
    test_inline(&[
        0x8b, 0x44, 0xe4, 0x04, // mov eax, [esp + 4]
        0x89, 0x4c, 0xe4, 0x04, // mov [esp + 4], ecx
        0x85, 0xc0, // test eax, eax
        0x74, 0x06, // je end
        0x89, 0x54, 0xe4, 0x04, // mov [esp + 4], edx
        0xeb, 0x00, // jmp end
        // end:
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x31, 0xd2, // xor edx, edx
        0x39, 0xc8, // cmp eax, ecx
        0x0f, 0x94, 0xc2, // sete dl
        0x31, 0xd2, // xor edx, edx
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.mem32(ctx.register(4), 4)),
        (ctx.register(1),
            if is_64bit() { ctx.and_const(ctx.new_undef(), 0xffff_ffff) } else { ctx.new_undef() },
        ),
        (ctx.register(2), ctx.constant(0)),
    ]);
}

#[test]
fn movdqa() {
    let ctx = &OperandContext::new();
    // eax should be [esp], but ecx is undefined
    test_inline(&[
        0xc7, 0x44, 0xe4, 0x00, 0x01, 0x00, 0x00, 0x00, // mov [esp], 1
        0xc7, 0x44, 0xe4, 0x08, 0x02, 0x00, 0x00, 0x00, // mov [esp + 8], 2
        0xc7, 0x44, 0xe4, 0x04, 0x03, 0x00, 0x00, 0x00, // mov [esp + 4], 3
        0x66, 0x0f, 0x6f, 0x2c, 0xe4, // mov xmm5, [esp]
        0x66, 0x0f, 0x7f, 0x6c, 0xe4, 0x10, // mov [esp + 1], xmm5
        0x8b, 0x44, 0xe4, 0x10, // mov eax, [esp + 10]
        0x8b, 0x4c, 0xe4, 0x14, // mov ecx, [esp + 14]
        0x8b, 0x54, 0xe4, 0x18, // mov edx, [esp + 18]
        0x8b, 0x5c, 0xe4, 0x1c, // mov ebx, [esp + 1c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(1)),
         (ctx.register(1), ctx.constant(3)),
         (ctx.register(2), ctx.constant(2)),
         (ctx.register(3), ctx.mem32(ctx.register(4), 0xc)),
    ]);
}

#[test]
fn jump_constraint_missed1() {
    let ctx = &OperandContext::new();
    // When reaching the `jg bad`, both skip_loop and out-of-loop branches
    // aren't going to jump. And as such jle branch should always jump
    // jg means Z = 0 & S = O
    // When coming from skip_loop, it knows that eax = 0 => edx = 0 => Z = 1,
    // from loop Z = 1 due to preceding jne
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x39, 0xd0, // cmp eax, edx
        0x73, 0x08, // jae skip_loop
        0x29, 0xc2, // sub edx, eax
        // loop:
        0x83, 0xea, 0x01, // sub edx, 1
        0x75, 0xfb, // jne loop
        0x90, // nop
        // skip_loop:
        0x7f, 0x02, // jg bad
        0x7e, 0x01, // jle end
        // bad
        0xcc, // int3
        // end:
        0x31, 0xd2, // xor edx, edx
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(2), ctx.constant(0)),
    ]);
}

#[test]
fn jump_constraint_missed2() {
    let ctx = &OperandContext::new();
    // Same as above but no nop before skip_loop
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x39, 0xd0, // cmp eax, edx
        0x73, 0x07, // jae skip_loop
        0x29, 0xc2, // sub edx, eax
        // loop:
        0x83, 0xea, 0x01, // sub edx, 1
        0x75, 0xfb, // jne loop
        // skip_loop:
        0x7f, 0x02, // jg bad
        0x7e, 0x01, // jle end
        // bad
        0xcc, // int3
        // end:
        0x31, 0xd2, // xor edx, edx
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(2), ctx.constant(0)),
    ]);
}

#[test]
fn xmm_u128_left_shift1() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0x04, 0xe4, // mov [esp], eax
        0x89, 0x4c, 0xe4, 0x04, // mov [esp + 4], ecx
        0x89, 0x54, 0xe4, 0x08, // mov [esp + 8], edx
        0x89, 0x5c, 0xe4, 0x0c, // mov [esp + c], ebx
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xf8, 0x04, // pslldq xmm0, 4
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0)),
        (ctx.register(1), mask_if_64bit(ctx, ctx.register(0), 0xffff_ffff)),
        (ctx.register(2), mask_if_64bit(ctx, ctx.register(1), 0xffff_ffff)),
        (ctx.register(3), mask_if_64bit(ctx, ctx.register(2), 0xffff_ffff)),
    ]);
}

#[test]
fn xmm_u128_right_shift1() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0x04, 0xe4, // mov [esp], eax
        0x89, 0x4c, 0xe4, 0x04, // mov [esp + 4], ecx
        0x89, 0x54, 0xe4, 0x08, // mov [esp + 8], edx
        0x89, 0x5c, 0xe4, 0x0c, // mov [esp + c], ebx
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x73, 0xd8, 0x04, // pslrdq xmm0, 4
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
        (ctx.register(0), mask_if_64bit(ctx, ctx.register(1), 0xffff_ffff)),
        (ctx.register(1), mask_if_64bit(ctx, ctx.register(2), 0xffff_ffff)),
        (ctx.register(2), mask_if_64bit(ctx, ctx.register(3), 0xffff_ffff)),
        (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn keep_constraint() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x85, 0xc0, // test eax, eax
        0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
        // loop:
        0x7f, 0x03, // jg cont
        0x7e, 0x01, // jle cont
        0xcc, // int3
        // cont:
        0x83, 0xc0, 0x01, // add eax, 1
        0x83, 0xf8, 0x04, // cmp eax, 4
        0x72, 0xf3, // jb loop
        0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn stc_jbe() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xf9, // stc
        0x76, 0x01, // jbe end
        0xcc, // int3,
        // end:
        0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn eq_minus_one() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x8b, 0x41, 0x24, // mov eax, [ecx + 24]
        0x83, 0xf8, 0xff, // cmp eax, -1
        0x0f, 0x94, 0xc0, // sete al,
        0x0f, 0xb6, 0xc0, // movzx eax, al
        0x66, 0x8b, 0x49, 0x24, // mov cx, word [ecx + 24]
        0x66, 0x83, 0xf9, 0xff, // cmp cx, -1
        0x0f, 0x94, 0xc1, // sete cl,
        0x0f, 0xb6, 0xc9, // movzx ecx, cl
        0xc3, // ret

    ], &[
        (ctx.register(0),
            ctx.eq_const(ctx.mem32(ctx.register(1), 0x24), 0xffff_ffff)),
        (ctx.register(1),
            ctx.eq_const(ctx.mem16(ctx.register(1), 0x24), 0xffff)),
    ]);
}

#[test]
fn or_minus_one_eax() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x83, 0xc8, 0xff, // or eax, ffff_ffff
        0xc3, // ret

    ], &[
        (ctx.register(0), ctx.constant(0xffff_ffff)),
    ]);
}

#[test]
fn neg_sets_flags() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0x00, 0x00, 0x00, 0x00, // mov eax, 0
        0xb9, 0x00, 0x00, 0x00, 0x00, // mov ecx, 0
        0xba, 0x00, 0x00, 0x00, 0x00, // mov edx, 0
        0xf6, 0xd8, // neg al
        0x0f, 0x93, 0xc2, // setae dl
        0x0f, 0x94, 0xc1, // sete cl
        0xb8, 0x80, 0x00, 0x00, 0x00, // mov eax, 80
        0xf6, 0xd8, // neg al
        0x0f, 0x90, 0xc0, // seto al
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0x01)),
        (ctx.register(1), ctx.constant(0x01)),
        (ctx.register(2), ctx.constant(0x01)),
    ]);
}

#[test]
fn add_minus_one_jne() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x83, 0xc2, 0xff, // add edx, ffff_ffff
        0x75, 0xfb, // jne start
        // start2:
        0x66, 0x81, 0xc1, 0xff, 0xff, // add cx, ffff
        0x75, 0xf9, // jne start2
        0x31, 0xc9, // xor ecx, ecx
        0x31, 0xd2, // xor edx, edx
        0xc3, // ret

    ], &[
        (ctx.register(1), ctx.constant(0)),
        (ctx.register(2), ctx.constant(0)),
    ]);
}

#[test]
fn cvtsi2sd_or_ss() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xb8, 0x56, 0x34, 0x12, 0x80, // mov eax, 80123456
        0xf3, 0x0f, 0x2a, 0xc0, // cvtsi2ss xmm0, eax
        0xf2, 0x0f, 0x2a, 0xd0, // cvtsi2sd xmm2, eax
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0x80123456)),
        (ctx.xmm(0, 0), ctx.constant(0xCEFFDB97)),
        (ctx.xmm(2, 0), ctx.constant(0xEA800000)),
        (ctx.xmm(2, 1), ctx.constant(0xC1DFFB72)),
    ]);
}

#[test]
fn pmovmskb() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0xc7, 0x44, 0xe4, 0x04, 0x80, 0x80, 0x80, 0x80, // mov [esp + 4], 80808080
        0xc7, 0x44, 0xe4, 0x08, 0x22, 0x11, 0x99, 0x88, // mov [esp + 8], 88991122
        0xc7, 0x44, 0xe4, 0x0c, 0x08, 0xef, 0xcd, 0xab, // mov [esp + c], ABCDEF08
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0xd7, 0xc0, // pmovmskb eax, xmm0
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(0xecf0)),
    ]);
}

#[test]
fn pmovsx() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0xc7, 0x44, 0xe4, 0x04, 0x80, 0x80, 0x80, 0x80, // mov [esp + 4], 80808080
        0xc7, 0x44, 0xe4, 0x08, 0x22, 0x11, 0x99, 0x88, // mov [esp + 8], 88991122
        0xc7, 0x44, 0xe4, 0x0c, 0x08, 0xef, 0xcd, 0xab, // mov [esp + c], ABCDEF08
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x38, 0x20, 0xc8, // pmovsxbw xmm1, xmm0
        0x66, 0x0f, 0x38, 0x21, 0xd0, // pmovsxbd xmm2, xmm0
        0x66, 0x0f, 0x38, 0x22, 0xd8, // pmovsxbq xmm3, xmm0
        0x66, 0x0f, 0x38, 0x23, 0xe0, // pmovsxwd xmm4, xmm0
        0x66, 0x0f, 0x38, 0x24, 0xe8, // pmovsxwq xmm5, xmm0
        0x66, 0x0f, 0x38, 0x25, 0xf0, // pmovsxdq xmm6, xmm0
        0xc3, // ret
    ], &[
        (ctx.xmm(0, 0), ctx.constant(0x12345678)),
        (ctx.xmm(0, 1), ctx.constant(0x80808080)),
        (ctx.xmm(0, 2), ctx.constant(0x88991122)),
        (ctx.xmm(0, 3), ctx.constant(0xabcdef08)),

        (ctx.xmm(1, 0), ctx.constant(0x00560078)),
        (ctx.xmm(1, 1), ctx.constant(0x00120034)),
        (ctx.xmm(1, 2), ctx.constant(0xff80ff80)),
        (ctx.xmm(1, 3), ctx.constant(0xff80ff80)),

        (ctx.xmm(2, 0), ctx.constant(0x00000078)),
        (ctx.xmm(2, 1), ctx.constant(0x00000056)),
        (ctx.xmm(2, 2), ctx.constant(0x00000034)),
        (ctx.xmm(2, 3), ctx.constant(0x00000012)),

        (ctx.xmm(3, 0), ctx.constant(0x00000078)),
        (ctx.xmm(3, 1), ctx.constant(0x00000000)),
        (ctx.xmm(3, 2), ctx.constant(0x00000056)),
        (ctx.xmm(3, 3), ctx.constant(0x00000000)),

        (ctx.xmm(4, 0), ctx.constant(0x00005678)),
        (ctx.xmm(4, 1), ctx.constant(0x00001234)),
        (ctx.xmm(4, 2), ctx.constant(0xffff8080)),
        (ctx.xmm(4, 3), ctx.constant(0xffff8080)),

        (ctx.xmm(5, 0), ctx.constant(0x00005678)),
        (ctx.xmm(5, 1), ctx.constant(0x00000000)),
        (ctx.xmm(5, 2), ctx.constant(0x00001234)),
        (ctx.xmm(5, 3), ctx.constant(0x00000000)),

        (ctx.xmm(6, 0), ctx.constant(0x12345678)),
        (ctx.xmm(6, 1), ctx.constant(0x00000000)),
        (ctx.xmm(6, 2), ctx.constant(0x80808080)),
        (ctx.xmm(6, 3), ctx.constant(0xffffffff)),
    ]);
}

#[test]
fn sse_and_or_xor() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0x0f, 0x54, 0xc1, // andps xmm0, xmm1
        0x66, 0x0f, 0x56, 0xca, // orpd xmm1, xmm2
        0x0f, 0x57, 0xd3, // xorps xmm2, xmm3
        0xc3, // ret
    ], &[
        (ctx.xmm(0, 0), ctx.and(ctx.xmm(0, 0), ctx.xmm(1, 0))),
        (ctx.xmm(0, 1), ctx.and(ctx.xmm(0, 1), ctx.xmm(1, 1))),
        (ctx.xmm(0, 2), ctx.and(ctx.xmm(0, 2), ctx.xmm(1, 2))),
        (ctx.xmm(0, 3), ctx.and(ctx.xmm(0, 3), ctx.xmm(1, 3))),
        (ctx.xmm(1, 0), ctx.or(ctx.xmm(1, 0), ctx.xmm(2, 0))),
        (ctx.xmm(1, 1), ctx.or(ctx.xmm(1, 1), ctx.xmm(2, 1))),
        (ctx.xmm(1, 2), ctx.or(ctx.xmm(1, 2), ctx.xmm(2, 2))),
        (ctx.xmm(1, 3), ctx.or(ctx.xmm(1, 3), ctx.xmm(2, 3))),
        (ctx.xmm(2, 0), ctx.xor(ctx.xmm(2, 0), ctx.xmm(3, 0))),
        (ctx.xmm(2, 1), ctx.xor(ctx.xmm(2, 1), ctx.xmm(3, 1))),
        (ctx.xmm(2, 2), ctx.xor(ctx.xmm(2, 2), ctx.xmm(3, 2))),
        (ctx.xmm(2, 3), ctx.xor(ctx.xmm(2, 3), ctx.xmm(3, 3))),
    ]);
}

#[test]
fn minps() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0xc7, 0x44, 0xe4, 0x04, 0x80, 0x80, 0x80, 0x80, // mov [esp + 4], 80808080
        0xc7, 0x44, 0xe4, 0x08, 0x22, 0x11, 0x99, 0x88, // mov [esp + 8], 88991122
        0xc7, 0x44, 0xe4, 0x0c, 0x08, 0xef, 0xcd, 0xab, // mov [esp + c], ABCDEF08
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x0f, 0x57, 0xc9, // xorps xmm1, xmm1
        0x0f, 0x57, 0xd2, // xorps xmm2, xmm2
        0x0f, 0x5d, 0xc8, // minps xmm1, xmm0
        0x66, 0x0f, 0x5d, 0xc2, // minpd xmm0, xmm2
        0xc3, // ret
    ], &[
        (ctx.xmm(0, 0), ctx.constant(0x12345678)),
        (ctx.xmm(0, 1), ctx.constant(0x80808080)),
        (ctx.xmm(0, 2), ctx.constant(0x88991122)),
        (ctx.xmm(0, 3), ctx.constant(0xabcdef08)),
        (ctx.xmm(1, 0), ctx.constant(0)),
        (ctx.xmm(1, 1), ctx.constant(0x80808080)),
        (ctx.xmm(1, 2), ctx.constant(0x88991122)),
        (ctx.xmm(1, 3), ctx.constant(0xabcdef08)),
        (ctx.xmm(2, 0), ctx.constant(0)),
        (ctx.xmm(2, 1), ctx.constant(0)),
        (ctx.xmm(2, 2), ctx.constant(0)),
        (ctx.xmm(2, 3), ctx.constant(0)),
    ]);
}

#[test]
fn maxps() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0xc7, 0x44, 0xe4, 0x04, 0x80, 0x80, 0x80, 0x80, // mov [esp + 4], 80808080
        0xc7, 0x44, 0xe4, 0x08, 0x22, 0x11, 0x99, 0x88, // mov [esp + 8], 88991122
        0xc7, 0x44, 0xe4, 0x0c, 0x08, 0xef, 0xcd, 0xab, // mov [esp + c], ABCDEF08
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x0f, 0x57, 0xc9, // xorps xmm1, xmm1
        0x0f, 0x57, 0xd2, // xorps xmm2, xmm2
        0x0f, 0x5f, 0xc8, // maxps xmm1, xmm0
        0x66, 0x0f, 0x5f, 0xc2, // maxpd xmm0, xmm2
        0xc3, // ret
    ], &[
        (ctx.xmm(0, 0), ctx.constant(0)),
        (ctx.xmm(0, 1), ctx.constant(0)),
        (ctx.xmm(0, 2), ctx.constant(0)),
        (ctx.xmm(0, 3), ctx.constant(0)),
        (ctx.xmm(1, 0), ctx.constant(0x12345678)),
        (ctx.xmm(1, 1), ctx.constant(0)),
        (ctx.xmm(1, 2), ctx.constant(0)),
        (ctx.xmm(1, 3), ctx.constant(0)),
        (ctx.xmm(2, 0), ctx.constant(0)),
        (ctx.xmm(2, 1), ctx.constant(0)),
        (ctx.xmm(2, 2), ctx.constant(0)),
        (ctx.xmm(2, 3), ctx.constant(0)),
    ]);
}

#[test]
fn pextrw() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0x66, 0x0f, 0xc5, 0xc0, 0x04, // pextrw eax, xmm0, 4
        0x66, 0x0f, 0xc5, 0xc9, 0x57, // pextrw ecx, xmm1, 57
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.and_const(ctx.xmm(0, 2), 0xffff)),
        (ctx.register(1), ctx.and_const(ctx.rsh_const(ctx.xmm(1, 3), 0x10), 0xffff)),
    ]);
}

#[test]
fn simple_loop() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        // loop:
        0xff, 0xc0, // inc eax
        0x83, 0xf8, 0x03, // cmp eax, 3
        0x7c, 0xf9, // jl loop
        0x31, 0xc0, // xor eax, eax
        0xff, 0xc0, // inc eax
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.constant(1)),
    ]);
}

#[test]
fn jae_jg_jg() {
    test_inline(&[
        0x85, 0xce, // test esi, ecx
        0x73, 0x01, // jae step0
        0xcc, // int3
        // step0:
        0x73, 0x01, // jae step1
        0xcc, // int3
        // step1:
        0x7f, 0x03, // jg step2
        0x7e, 0x04, // jle end
        0xcc, // int3
        // step2:
        0x7f, 0x01, // jg end
        0xcc, // int3
        // end:
        0xc3, // ret
    ], &[
    ]);
}

#[test]
fn xmm_shifts() {
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0xc7, 0x44, 0xe4, 0x04, 0x80, 0x80, 0x80, 0x80, // mov [esp + 4], 80808080
        0xc7, 0x44, 0xe4, 0x08, 0x22, 0x11, 0x99, 0x88, // mov [esp + 8], 88991122
        0xc7, 0x44, 0xe4, 0x0c, 0x08, 0xef, 0xcd, 0xab, // mov [esp + c], ABCDEF08
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x0f, 0x10, 0xc8, // movups xmm1, xmm0
        0x0f, 0x10, 0xd0, // movups xmm2, xmm0
        0x0f, 0x10, 0xd8, // movups xmm3, xmm0
        0x0f, 0x10, 0xe0, // movups xmm4, xmm0
        0x0f, 0x10, 0xe8, // movups xmm5, xmm0
        0x66, 0x0f, 0x71, 0xd0, 0x05, // psrlw xmm0, 5
        0x66, 0x0f, 0x71, 0xe1, 0x05, // psraw xmm1, 5
        0x66, 0x0f, 0x71, 0xf2, 0x05, // psllw xmm2, 5
        0x66, 0x0f, 0x72, 0xd3, 0x05, // psrld xmm3, 5
        0x66, 0x0f, 0x72, 0xe4, 0x05, // psrad xmm4, 5
        0x66, 0x0f, 0x72, 0xf5, 0x05, // pslld xmm5, 5
        0xc3, // ret
    ], &[
        (ctx.xmm(0, 0), ctx.constant(0x009102b3)),
        (ctx.xmm(0, 1), ctx.constant(0x04040404)),
        (ctx.xmm(0, 2), ctx.constant(0x04440089)),
        (ctx.xmm(0, 3), ctx.constant(0x055e0778)),
        (ctx.xmm(1, 0), ctx.constant(0x009102b3)),
        (ctx.xmm(1, 1), ctx.constant(0xfc04fc04)),
        (ctx.xmm(1, 2), ctx.constant(0xfc440089)),
        (ctx.xmm(1, 3), ctx.constant(0xfd5eff78)),
        (ctx.xmm(2, 0), ctx.constant(0x4680cf00)),
        (ctx.xmm(2, 1), ctx.constant(0x10001000)),
        (ctx.xmm(2, 2), ctx.constant(0x13202440)),
        (ctx.xmm(2, 3), ctx.constant(0x79a0e100)),

        (ctx.xmm(3, 0), ctx.constant(0x0091a2b3)),
        (ctx.xmm(3, 1), ctx.constant(0x04040404)),
        (ctx.xmm(3, 2), ctx.constant(0x0444c889)),
        (ctx.xmm(3, 3), ctx.constant(0x055e6f78)),
        (ctx.xmm(4, 0), ctx.constant(0x0091a2b3)),
        (ctx.xmm(4, 1), ctx.constant(0xfc040404)),
        (ctx.xmm(4, 2), ctx.constant(0xfc44c889)),
        (ctx.xmm(4, 3), ctx.constant(0xfd5e6f78)),
        (ctx.xmm(5, 0), ctx.constant(0x468acf00)),
        (ctx.xmm(5, 1), ctx.constant(0x10101000)),
        (ctx.xmm(5, 2), ctx.constant(0x13222440)),
        (ctx.xmm(5, 3), ctx.constant(0x79bde100)),
    ]);
}

#[test]
fn xmm_shifts_over() {
    // All of these set result to zeroes (or ones for arithmetic right with negative)
    // if shift count is greater than elemnt bit size
    let ctx = &OperandContext::new();
    test_inline_xmm(&[
        0xc7, 0x04, 0xe4, 0x78, 0x56, 0x34, 0x12, // mov [esp], 12345678
        0xc7, 0x44, 0xe4, 0x04, 0x80, 0x80, 0x80, 0x80, // mov [esp + 4], 80808080
        0xc7, 0x44, 0xe4, 0x08, 0x22, 0x11, 0x99, 0x88, // mov [esp + 8], 88991122
        0xc7, 0x44, 0xe4, 0x0c, 0x08, 0xef, 0xcd, 0xab, // mov [esp + c], ABCDEF08
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x0f, 0x10, 0xc8, // movups xmm1, xmm0
        0x0f, 0x10, 0xd0, // movups xmm2, xmm0
        0x0f, 0x10, 0xd8, // movups xmm3, xmm0
        0x0f, 0x10, 0xe0, // movups xmm4, xmm0
        0x0f, 0x10, 0xe8, // movups xmm5, xmm0
        0x66, 0x0f, 0x71, 0xd0, 0x45, // psrlw xmm0, 45
        0x66, 0x0f, 0x71, 0xe1, 0x45, // psraw xmm1, 45
        0x66, 0x0f, 0x71, 0xf2, 0x45, // psllw xmm2, 45
        0x66, 0x0f, 0x72, 0xd3, 0x45, // psrld xmm3, 45
        0x66, 0x0f, 0x72, 0xe4, 0x45, // psrad xmm4, 45
        0x66, 0x0f, 0x72, 0xf5, 0x45, // pslld xmm5, 45
        0xc3, // ret
    ], &[
        (ctx.xmm(0, 0), ctx.constant(0)),
        (ctx.xmm(0, 1), ctx.constant(0)),
        (ctx.xmm(0, 2), ctx.constant(0)),
        (ctx.xmm(0, 3), ctx.constant(0)),
        (ctx.xmm(1, 0), ctx.constant(0)),
        (ctx.xmm(1, 1), ctx.constant(0xffffffff)),
        (ctx.xmm(1, 2), ctx.constant(0xffff0000)),
        (ctx.xmm(1, 3), ctx.constant(0xffffffff)),
        (ctx.xmm(2, 0), ctx.constant(0)),
        (ctx.xmm(2, 1), ctx.constant(0)),
        (ctx.xmm(2, 2), ctx.constant(0)),
        (ctx.xmm(2, 3), ctx.constant(0)),

        (ctx.xmm(3, 0), ctx.constant(0)),
        (ctx.xmm(3, 1), ctx.constant(0)),
        (ctx.xmm(3, 2), ctx.constant(0)),
        (ctx.xmm(3, 3), ctx.constant(0)),
        (ctx.xmm(4, 0), ctx.constant(0)),
        (ctx.xmm(4, 1), ctx.constant(0xffffffff)),
        (ctx.xmm(4, 2), ctx.constant(0xffffffff)),
        (ctx.xmm(4, 3), ctx.constant(0xffffffff)),
        (ctx.xmm(5, 0), ctx.constant(0)),
        (ctx.xmm(5, 1), ctx.constant(0)),
        (ctx.xmm(5, 2), ctx.constant(0)),
        (ctx.xmm(5, 3), ctx.constant(0)),
    ]);
}

#[test]
fn flag_constraint_merge_bug() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x85, 0xc0, // test eax, eax
        0x74, 0x03, // je eax_zero
        0x83, 0xc1, 0x03, // add ecx, 3
        // eax_zero:
        0x89, 0xc1, // mov ecx, eax
        0x85, 0xc9, // test ecx, ecx
        0x7a, 0x02, // jp step1
        0x7b, 0x03, // jnp end
        // step1:
        0x7a, 0x01, // jp end
        0xcc, // int3
        // end:
        0x33, 0xc9, // xor ecx, ecx
        0xc3, // ret
    ], &[
         (ctx.register(1), ctx.constant(0)),
    ]);
}

#[test]
fn inc_dec_ax() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0xff, 0x7f, 0x00, 0x00, // mov eax, 7fff
        0x66, 0xff, 0xc0, // inc ax
        0x70, 0x01, // jo more
        0xcc, // int3
        0x66, 0xff, 0xc8, // dec ax
        0x70, 0x01, // jo more
        0xcc, // int3
        0x31, 0xc0, // xor eax, eax
        0x66, 0xff, 0xc8, // dec ax
        0x78, 0x01, // js more
        0xcc, // int3
        0x73, 0x01, // jnc more (inc/dec won't change carry, it is zero from xor)
        0xcc, // int3
        0x66, 0xff, 0xc0, // inc ax
        0x79, 0x01, // jns more
        0xcc, // int3
        0x73, 0x01, // jnc more (inc/dec won't change carry, it is zero from xor)
        0xcc, // int3
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn lahf() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0xff, 0x7f, 0x00, 0x00, // mov eax, 7fff
        0xb9, 0x00, 0x80, 0x00, 0x00, // mov ecx, 8000
        0x39, 0xc8, // cmp eax, ecx
        0x9f, // lahf (SF:ZF:0:AF:0:PF:1:CF) (100a0p11)
        0x25, 0xff, 0xeb, 0xff, 0xff, // and eax, ffff_ebff (Ignore a/p)
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x83ff)),
         (ctx.register(1), ctx.constant(0x8000)),
    ]);
}

#[test]
fn parity() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb0, 0x4f, // mov al, 4f
        0x83, 0xf8, 0x00, // cmp eax, 0 (p = 0)
        0x7b, 0x01, // jpo skip
        0xcc, // int3
        0xb0, 0xff, // mov al, ff
        0x83, 0xf8, 0x00, // cmp eax, 0 (p = 1)
        0x7a, 0x01, // jpe skip
        0xcc, // int3
        0xb0, 0x00, // mov al, 00
        0x83, 0xf8, 0x00, // cmp eax, 0 (p = 1)
        0x7a, 0x01, // jpe skip
        0xcc, // int3
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn stc_jbe_on_undef_zero() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x3b, 0xc8, // cmp ecx, eax
        0xf9, // stc
        0x76, 0x01, // jbe skip
        0xcc, // int3
        0x74, 0x02, // je skip2
        0x31, 0xc0, // xor eax, eax
        // skip2:
        0xeb, 0x00, // jmp end
        // end:
        0xf9, // stc
        0x76, 0x01, // jbe skip
        0xcc, // int3
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.new_undef()),
    ]);
}

#[test]
fn carry_known_zero_after_loop() {
    let ctx = &OperandContext::new();
    test_inline(&[
        // loop:
        0x46, // inc esi
        0x83, 0xfe, 0x10, // cmp esi, 10
        0x72, 0xfa, // jb loop
        0x83, 0xfe, 0x10, // cmp esi, 10
        0x73, 0x01, // jae end
        0xcc, // int3
        0x31, 0xf6, // xor esi, esi
        0xc3, // ret
    ], &[
         (ctx.register(6), ctx.const_0()),
    ]);
}

#[test]
fn ja_jge_jl() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x3c, 0x19, // cmp al, 19
        0x77, 0x03, // ja skip
        0x80, 0xc1, 0x20,  // add cl, 20
        // skip:
        0x7d, 0x03, // jge end
        0x7c, 0x01, // jl end
        0xcc, // int3
        0xc3, // ret
    ], &[
         (ctx.register(1), ctx.new_undef()),
    ]);
}

#[test]
fn sete_undef() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x3c, 0x19, // cmp al, 19
        0x77, 0x02, // ja skip
        0x3c, 0x12, // cmp al, 18
        // skip:
        0xeb, 0x00, // jmp next
        // next:
        0x0f, 0x94, 0xc0, // sete al
        0x3c, 0x02, // cmp al, 2
        0x72, 0x01, // jb end
        0xcc, // int3
        0x33, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.const_0()),
    ]);
}

#[test]
fn mem_move_regs() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0x41, 0x04, // mov [ecx + 4], eax
        0x8b, 0x41, 0x07, // mov eax, [ecx + 7]
        0x89, 0x49, 0x08, // mov [ecx + 8], ecx
        0x8b, 0x49, 0x06, // mov eax, [ecx + 6]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.or(
            ctx.rsh_const(
                ctx.and_const(
                    ctx.register(0),
                    0xff00_0000,
                ),
                0x18,
            ),
            ctx.lsh_const(
                ctx.and_const(
                    ctx.mem32(ctx.register(1), 8),
                    0xffff_ff,
                ),
                8,
            ),
        )),
        (ctx.register(1), ctx.or(
            ctx.rsh_const(
                ctx.and_const(
                    ctx.register(0),
                    0xffff_0000,
                ),
                0x10,
            ),
            ctx.lsh_const(
                ctx.and_const(
                    ctx.register(1),
                    0xffff,
                ),
                0x10,
            ),
        ))
    ]);
}

#[test]
fn jl_jge() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x3b, 0x05, 0x23, 0x01, 0x00, 0x00, // cmp eax, [123]
        0x75, 0x04, // jne skip
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0xc3, // ret
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x7c, 0x03, // jl end
        0x7d, 0x01, // jge end
        0xcc, // int3
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem32(ctx.register(4), 0)),
    ]);
}

#[test]
fn parity_crash() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x00, 0x00, // add [eax], al
        0x00, 0x00, // add [eax], al
        0x25, 0x40, 0x09, 0x00, 0x09, // add eax, 0900_0940
        0x04, 0x7a, // add al, 7a
        0x09, 0x00, // or [eax], eax
        0x7a, 0x00, // jpe
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn or_simplify_crash() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x0d, 0x0d, 0x0d, 0x09, 0x00, // or eax, 09_0d0d
        0x00, 0x00, // add [eax], al
        0x0b, 0x00, // or eax, [eax]
        0x00, 0x20, // add [eax], ah
        0x0b, 0x00, // or eax, [eax]
        0x00, 0x20, // add [eax], ah
        0x20, 0x00, // and [eax], al
        0x0b, 0x00, // or eax, [eax]
        0x00, 0x00, // add [eax], al
        0x0b, 0x00, // or eax, [eax]
        0x20, 0x00, // and [eax], al
        0x00, 0x00, // add [eax], al
        0x00, 0x00, // add [eax], al
        0x0b, 0x00, // or eax, [eax]
        0x31, 0xc0, // xor eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn memory_writes_every_second_byte() {
    // Write (1, _, 2, _, 6, _, cl, _)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0xc6, 0x00, 0x01, // mov byte [eax], 1
            0xc6, 0x40, 0x02, 0x02, // mov byte [eax + 2], 2
            0xc6, 0x40, 0x04, 0x06, // mov byte [eax + 4], 6
            0x88, 0x48, 0x06, // mov byte [eax + 6], cl
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), ctx.or(
                ctx.and_const(ctx.mem32(base, 0), 0xff00_ff00),
                ctx.constant(0x0002_0001)
            )),
            (ctx.register(2), ctx.or(
                ctx.and_const(ctx.mem32(base, 4), 0xff00_ff00),
                ctx.or(
                    ctx.lsh_const(
                        ctx.and_const(ctx.register(1), 0xff),
                        0x10,
                    ),
                    ctx.constant(0x0000_0006)
                ),
            )),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn memory_writes_every_second_byte_2() {
    // Write (_, 1, _, 2, _, 6, _, cl)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0xc6, 0x40, 0x01, 0x01, // mov byte [eax + 1], 1
            0xc6, 0x40, 0x03, 0x02, // mov byte [eax + 3], 2
            0xc6, 0x40, 0x05, 0x06, // mov byte [eax + 5], 6
            0x88, 0x48, 0x07, // mov byte [eax + 7], cl
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), ctx.or(
                ctx.and_const(ctx.mem32(base, 0), 0x00ff_00ff),
                ctx.constant(0x0200_0100)
            )),
            (ctx.register(2), ctx.or(
                ctx.and_const(ctx.mem32(base, 4), 0x00ff_00ff),
                ctx.or(
                    ctx.lsh_const(
                        ctx.and_const(ctx.register(1), 0xff),
                        0x18,
                    ),
                    ctx.constant(0x0000_0600)
                ),
            )),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn memory_writes_every_second_word() {
    // Write u16 (2001, _, cx, _)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0x66, 0xc7, 0x00, 0x01, 0x20, // mov word [eax], 2001
            0x66, 0x89, 0x48, 0x04, // mov word [eax + 4], cx
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), ctx.or(
                ctx.and_const(ctx.mem32(base, 0), 0xffff_0000),
                ctx.constant(0x0000_2001)
            )),
            (ctx.register(2), ctx.or(
                ctx.and_const(ctx.mem32(base, 4), 0xffff_0000),
                ctx.and_const(
                    ctx.register(1),
                    0xffff,
                ),
            )),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn memory_writes_every_second_word_2() {
    // Write u16 (_, 2001, _, cx)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0x66, 0xc7, 0x40, 0x02, 0x01, 0x20, // mov word [eax + 2], 2001
            0x66, 0x89, 0x48, 0x06, // mov word [eax + 6], cx
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), ctx.or(
                ctx.and_const(ctx.mem32(base, 0), 0x0000_ffff),
                ctx.constant(0x2001_0000)
            )),
            (ctx.register(2), ctx.or(
                ctx.and_const(ctx.mem32(base, 4), 0x0000_ffff),
                ctx.lsh_const(
                    ctx.and_const(
                        ctx.register(1),
                        0xffff,
                    ),
                    0x10,
                ),
            )),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn memory_writes_dword() {
    // Write u32 (ecx, _)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0x89, 0x08, // mov [eax], ecx
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), mask_if_64bit(ctx, ctx.register(1), 0xffff_ffff)),
            (ctx.register(2), ctx.mem32(base, 4)),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn memory_writes_dword_2() {
    // Write u32 (u16 _, ecx, u16 _)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0x89, 0x48, 0x02, // mov [eax + 2], ecx
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), ctx.or(
                ctx.and_const(ctx.mem32(base, 0), 0x0000_ffff),
                ctx.lsh_const(
                    ctx.and_const(
                        ctx.register(1),
                        0xffff,
                    ),
                    0x10,
                ),
            )),
            (ctx.register(2), ctx.or(
                ctx.and_const(ctx.mem32(base, 4), 0xffff_0000),
                ctx.rsh_const(
                    ctx.and_const(
                        ctx.register(1),
                        0xffff_0000,
                    ),
                    0x10,
                ),
            )),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn memory_writes_dword_3() {
    // Write u32 (ecx, edx)
    // with 8 different offsets.
    let ctx = &OperandContext::new();
    for i in 0..8 {
        let base = ctx.add_const(ctx.register(5), 0x100 + i);
        test_inline_with_init(&[
            0x89, 0x48, 0x00, // mov [eax], ecx
            0x89, 0x50, 0x04, // mov [eax + 4], edx
            0x8b, 0x08, // mov ecx, [eax]
            0x8b, 0x50, 0x04, // mov edx, [eax + 4]
            0xc3, // ret
        ], &[
            (ctx.register(0), base),
            (ctx.register(1), mask_if_64bit(ctx, ctx.register(1), 0xffff_ffff)),
            (ctx.register(2), mask_if_64bit(ctx, ctx.register(2), 0xffff_ffff)),
        ], &[
            (ctx.register(0), base),
        ]);
    }
}

#[test]
fn merge_unwritten_mem() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0xb8, 0x04, 0x00, 0x05, 0x00, // mov eax, 0005_0004
        0x31, 0xc9, // xor ecx, ecx
        0x85, 0xd2, // test edx, edx
        0x74, 0x04, // je skip_store
        0xc6, 0x40, 0x02, 0x04, // mov byte [eax + 2], 4
        // skip_store:
        0xeb, 0x00, // jmp next
        // next:
        0x8a, 0x08, // mov cl, byte [eax]
        0x8b, 0x50, 0x03, // mov edx, [eax + 3]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x5_0004)),
         (ctx.register(1), ctx.mem8(ctx.constant(0x5_0004), 0)),
         (ctx.register(2), ctx.mem32(ctx.constant(0x5_0007), 0)),
    ]);
}

#[test]
fn merge_unwritten_mem_correct_masks_when_read_back_unaligned() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0xb8, 0x04, 0x00, 0x05, 0x00, // mov eax, 0005_0004
        0x31, 0xc9, // xor ecx, ecx
        0x31, 0xd2, // xor edx, edx
        0x85, 0xf6, // test esi, esi
        0x74, 0x11, // je skip_store
        0xc6, 0x40, 0x02, 0x04, // mov [eax + 2], 4
        0x66, 0xc7, 0x40, 0x06, 0x04, 0x00, // mov word [eax + 6], 4
        0xc7, 0x40, 0x0a, 0x04, 0x00, 0x00, 0x00, // mov dword [eax + a], 4
        // skip_store:
        0xeb, 0x00, // jmp next
        // next:
        0x8a, 0x48, 0x02, // mov cl, byte [eax + 2]
        0x0f, 0xb6, 0x58, 0x02, // movzx ebx, byte [eax + 2]
        0x66, 0x8b, 0x50, 0x06, // mov dx, [eax + 6]
        0x0f, 0xb7, 0x70, 0x06, // movzx esi, word [eax + 6]
        0x8b, 0x40, 0x0a, // mov eax, [eax + a]
        0xc3, // ret
    ], &[
         (ctx.register(0), if !is_64bit() {
             // Idk these can be many things, though the two undefs shouldn't overlap
             let ud = ctx.new_undef();
             ctx.and_const(
                 ctx.or(
                     ctx.and_const(ud, 0xffff),
                     ctx.lsh_const(ud, 0x10),
                 ),
                 0xffff_ffff,
             )
         } else {
             let ud = ctx.new_undef();
             ctx.and_const(
                 ctx.or(
                     ctx.and_const(ctx.rsh_const(ud, 0x20), 0xffff),
                     ctx.lsh_const(ud, 0x10),
                 ),
                 0xffff_ffff,
             )
         }),
         (ctx.register(1), ctx.and_const(ctx.new_undef(), 0xff)),
         (ctx.register(2), ctx.and_const(ctx.new_undef(), 0xffff)),
         (ctx.register(3), ctx.and_const(ctx.new_undef(), 0xff)),
         (ctx.register(6), ctx.and_const(ctx.new_undef(), 0xffff)),
    ]);
}

#[test]
fn merge_unwritten_mem_correct_masks_when_read_back_aligned() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0xb8, 0x00, 0x00, 0x05, 0x00, // mov eax, 0005_0000
        0x31, 0xc9, // xor ecx, ecx
        0x31, 0xd2, // xor edx, edx
        0x85, 0xf6, // test esi, esi
        0x74, 0x11, // je skip_store
        0xc6, 0x40, 0x00, 0x04, // mov [eax + 0], 4
        0x66, 0xc7, 0x40, 0x08, 0x04, 0x00, // mov word [eax + 8], 4
        0xc7, 0x40, 0x10, 0x04, 0x00, 0x00, 0x00, // mov dword [eax + 10], 4
        // skip_store:
        0xeb, 0x00, // jmp next
        // next:
        0x8a, 0x48, 0x00, // mov cl, byte [eax + 0]
        0x0f, 0xb6, 0x58, 0x00, // movzx ebx, byte [eax + 0]
        0x66, 0x8b, 0x50, 0x08, // mov dx, [eax + 8]
        0x0f, 0xb7, 0x70, 0x08, // movzx esi, word [eax + 8]
        0x8b, 0x40, 0x10, // mov eax, [eax + 10]
        0xc3, // ret
    ], &[
         (ctx.register(0), mask_if_64bit(ctx, ctx.new_undef(), 0xffff_ffff)),
         (ctx.register(1), ctx.and_const(ctx.new_undef(), 0xff)),
         (ctx.register(2), ctx.and_const(ctx.new_undef(), 0xffff)),
         (ctx.register(3), ctx.and_const(ctx.new_undef(), 0xff)),
         (ctx.register(6), ctx.and_const(ctx.new_undef(), 0xffff)),
    ]);
}

#[test]
fn crc32() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x66, 0xb9, 0xff, 0x05, // mov cx, 5ff
        0xb2, 0xaa, // mov dl, aa
        0x31, 0xdb, // xor ebx, ebx
        0xbe, 0xff, 0xaa, 0x00, 0x05, // mov esi, 0500aaff
        0xf2, 0x0f, 0x38, 0xf0, 0xc1, // crc32 eax, cl
        0x89, 0xc1, // mov ecx, eax
        0xf2, 0x0f, 0x38, 0xf0, 0xc2, // crc32 eax, dl
        0x89, 0xc2, // mov edx, eax
        0xf2, 0x0f, 0x38, 0xf0, 0xc3, // crc32 eax, bl
        0x89, 0xc3, // mov ebx, eax
        0xf2, 0x0f, 0x38, 0xf0, 0xc0, // crc32 eax, al
        0x89, 0xc5, // mov ebp, eax
        0xb8, 0xf0, 0x00, 0x00, 0x00, // mov eax, 0xf0
        0xf2, 0x0f, 0x38, 0xf1, 0xc6, // crc32 eax, esi
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xc7e82faf)),
         (ctx.register(1), ctx.constant(0xad7d5351)),
         (ctx.register(2), ctx.constant(0x6a4ab91d)),
         (ctx.register(3), ctx.constant(0xaf1cc105)),
         (ctx.register(5), ctx.constant(0x00af1cc1)),
         (ctx.register(6), ctx.constant(0x0500aaff)),
    ]);
}

#[test]
fn crc32_u16() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0xbe, 0xff, 0xaa, 0x00, 0x05, // mov esi, 0500aaff
        0x66, 0xf2, 0x0f, 0x38, 0xf1, 0xc6, // crc32 eax, si
        0x89, 0xc1, // mov ecx, eax
        0x66, 0xf2, 0x0f, 0x38, 0xf1, 0xc0, // crc32 eax, ax
        0x89, 0xc2, // mov edx, eax
        0xf2, 0x0f, 0x38, 0xf1, 0xc0, // crc32 eax, eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(0x6A4AB91D)),
         (ctx.register(2), ctx.constant(0x00006A4A)),
         (ctx.register(6), ctx.constant(0x0500aaff)),
    ]);
}

#[test]
fn crc32_eax_ah() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0xb8, 0xff, 0xaa, 0x00, 0x05, // mov eax, 0500aaff
        0xf2, 0x0f, 0x38, 0xf0, 0xc4, // crc32, eax, ah
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x64D1CE65)),
    ]);
}

#[test]
fn cached_low_register_merge_bug() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0x83, 0xf8, 0x01, // cmp eax, 1
        0x74, 0x18, // je branch_c
        0x85, 0xc0, // test eax, eax
        0x75, 0x09, // jne branch_b
        0x89, 0xcb, // mov ebx, ecx
        0xb3, 0x01, // mov bl, 1
        0x84, 0xdb, // test bl, bl
        0x75, 0x0a, // jne merge_a_b
        0xcc, // int3
        // branch_b:
        0x89, 0xd3, // mov ebx, edx
        0xb3, 0x01, // mov bl, 1
        0x84, 0xc0, // test al, al (Do something different to have flags be merged to undef)
        0xeb, 0x01, // jmp merge_a_b
        0xcc, // int3
        // merge_a_b:
        // Here ebx should be undef, but bl could be known to be 1.
        // But keeping bl as 1 made branch_c -> merge_c jump not consider
        // state to be changed, resulting final value of bl always be 1, which would be wrong.
        0xeb, 0x09, // jmp merge_c
        // branch_c:
        0x89, 0xd3, // mov ebx, edx
        0xb3, 0x00, // mov bl, 0
        0x84, 0xdb, // test bl, bl
        0x74, 0x01, // je merge_c
        0xcc, // int3
        // merge_c:
        0x0f, 0xb6, 0xc3, // movzx eax, bl
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.and_const(ctx.new_undef(), 0xff)),
         (ctx.register(3), ctx.new_undef()),
    ]);
}
