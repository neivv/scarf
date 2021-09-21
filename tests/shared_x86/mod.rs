//! 32/64-bit x86 tests

use super::{test_inline, test_inline_xmm};

use scarf::OperandContext;

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
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem32(ctx.register(4), 4)),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
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
        (ctx.register(1), ctx.and_const(ctx.register(0), 0xffff_ffff)),
        (ctx.register(2), ctx.and_const(ctx.register(1), 0xffff_ffff)),
        (ctx.register(3), ctx.and_const(ctx.register(2), 0xffff_ffff)),
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
        (ctx.register(0), ctx.and_const(ctx.register(1), 0xffff_ffff)),
        (ctx.register(1), ctx.and_const(ctx.register(2), 0xffff_ffff)),
        (ctx.register(2), ctx.and_const(ctx.register(3), 0xffff_ffff)),
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
