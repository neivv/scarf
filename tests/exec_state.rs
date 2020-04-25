extern crate byteorder;
extern crate scarf;

mod helpers;

use std::ffi::OsStr;

use byteorder::{ReadBytesExt, LittleEndian};

use scarf::{
    BinaryFile, BinarySection, DestOperand, Operand, Operation, OperandContext, VirtualAddress,
};
use scarf::analysis::{self, Control};
use scarf::ExecutionStateX86 as ExecutionState;

#[test]
fn movzx() {
    let ctx = &OperandContext::new();
    test(0, &[
         (ctx.register(0), ctx.constant(1)),
         (ctx.register(1), ctx.constant(0xfffd)),
         (ctx.register(2), ctx.constant(0xfffd)),
    ]);
}

#[test]
fn movsx() {
    let ctx = &OperandContext::new();
    test(1, &[
         (ctx.register(0), ctx.constant(1)),
         (ctx.register(1), ctx.constant(0xfffd)),
         (ctx.register(2), ctx.constant(0xfffffffd)),
    ]);
}

#[test]
fn movzx_mem() {
    let ctx = &OperandContext::new();
    test(2, &[
         (ctx.register(0), ctx.constant(0x90)),
    ]);
}

#[test]
fn movsx_mem() {
    let ctx = &OperandContext::new();
    test(3, &[
         (ctx.register(0), ctx.constant(0xffffff90)),
    ]);
}

#[test]
fn f6_f7_and_const() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.constant(1)),
    ]);
}

#[test]
fn movsx_16_self() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(0xff94)),
         (ctx.register(1), ctx.constant(0xff94)),
         (ctx.register(2), ctx.constant(0xff94)),
         (ctx.register(6), ctx.constant(0xff94)),
    ]);
}

#[test]
fn cmp_const_16() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x66, 0xc7, 0x03, 0x05, 0x05, // mov word [ebx], 0505
        0x31, 0xc0, // xor eax, eax
        0x66, 0x81, 0x3b, 0xef, 0xbb, // cmp word [ebx], bbef
        0x0f, 0x92, 0xc0, // setb al
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(1)),
    ]);
}

#[test]
fn add_i8() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x83, 0xc0, 0xfb, // add eax, 0xfffffffb
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0xfffffffb)),
    ]);
}

#[test]
fn shld() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb9, 0x23, 0x01, 0x00, 0xff, // mov ecx, 0xff000123
        0xb8, 0x00, 0x00, 0x00, 0x40, // mov eax, 0x40000000
        0x0f, 0xa4, 0xc1, 0x04, // shld ecx, eax, 4
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0x40000000)),
         (ctx.register(1), ctx.constant(0xf0001234)),
    ]);
}

#[test]
fn double_jbe() {
    let ctx = &OperandContext::new();
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
         (ctx.register(0), ctx.constant(1)),
    ]);
}

#[test]
fn shr_mem_10() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x45, 0xf4, 0x78, 0x56, 0x34, 0x12, // mov [ebp - 0xc], 0x12345678
        0xc1, 0x6d, 0xf4, 0x10, // shr [ebp - 0xc], 0x10
        0x8b, 0x45, 0xf4, // mov eax, [ebp - 0xc]
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0x1234)),
    ]);
}

#[test]
fn shr_mem_10_b() {
    let ctx = &OperandContext::new();
    // Memory address (base + ffffffff)
    test_inline(&[
        0xc7, 0x45, 0xff, 0x78, 0x56, 0x34, 0x12, // mov [ebp - 0x1], 0x12345678
        0xc1, 0x6d, 0xff, 0x10, // shr [ebp - 0x1], 0x10
        0x8b, 0x45, 0xff, // mov eax, [ebp - 0x1]
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0x1234)),
    ]);
}

#[test]
fn read_ffffffff() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xa1, 0xff, 0xff, 0xff, 0xff, // mov eax, [ffffffff]
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.mem32(ctx.constant(0xffff_ffff))),
    ]);
}

#[test]
fn read_this() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xa1, 0x00, 0x10, 0x40, 0x00, // mov eax, [401000]
        0x8b, 0x0d, 0x0e, 0x10, 0x40, 0x00, // mov ecx, [40100e]
        0x8b, 0x15, 0x0f, 0x10, 0x40, 0x00, // mov edx, [40100f]
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0x4010_00a1)),
         (ctx.register(1), ctx.constant(0xc300_4010)),
         (ctx.register(2), ctx.mem32(ctx.constant(0x0040_100f))),
    ]);
}

#[test]
fn je_jne_with_memory_write() {
    let ctx = &OperandContext::new();
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
        (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn not_is_xor() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xf7, 0xd0, // not eax
        0x83, 0xf0, 0xff, // xor eax, ffff_ffff
        0x66, 0xf7, 0xd1, // not cx
        0x81, 0xf1, 0xff, 0xff, 0x00, 0x00, // xor ecx, ffff
        0xc3, //ret
    ], &[
        (ctx.register(0), ctx.register(0)),
        (ctx.register(1), ctx.register(1)),
    ]);
}

#[test]
fn jge_jge() {
    let ctx = &OperandContext::new();
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
        (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn inc_dec_flags() {
    let ctx = &OperandContext::new();
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
        (ctx.register(0), ctx.constant(0xffff_ffff)),
    ]);
}

#[test]
fn jo_jno_sometimes_undef() {
    let ctx = &OperandContext::new();
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
        (ctx.register(0), ctx.constant(0)),
    ]);
}

#[test]
fn call_removes_constraints() {
    let ctx = &OperandContext::new();
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
        (ctx.register(0), ctx.new_undef()),
        (ctx.register(1), ctx.new_undef()),
        (ctx.register(2), ctx.new_undef()),
        (ctx.register(3), ctx.new_undef()),
        (ctx.register(4), ctx.new_undef()),
    ]);
}

#[test]
fn div_mod() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x33, 0xd2, // xor edx, edx
        0xb9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0xf7, 0xf1, // div ecx
        0xc3, //ret
    ], &[
        (ctx.register(1), ctx.constant(7)),
        (ctx.register(2), ctx.modulo(
            ctx.and(ctx.register(0), ctx.constant(0xffff_ffff)),
            ctx.constant(7),
        )),
        (ctx.register(0), ctx.div(
            ctx.and(ctx.register(0), ctx.constant(0xffff_ffff)),
            ctx.constant(7),
        )),
    ]);
}

#[test]
fn cmp_mem8() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0x04, 0x95, // add al, 95
        0x89, 0x01, // mov [ecx], eax
        0x80, 0x39, 0x94, // cmp byte [ecx], 94
        0x77, 0x01, // ja ok
        0xcc,
        0xc3, //ret
    ], &[
         (ctx.register(0), ctx.constant(0x95)),
    ]);
}

#[test]
fn movzx_mem8() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x0f, 0xb6, 0x56, 0x4d, // movzx edx, byte [esi + 4d]
        0xc3, //ret
    ], &[
         (ctx.register(2), ctx.mem8(ctx.add(ctx.register(6), ctx.constant(0x4d)))),
    ]);
}

#[test]
fn resolved_constraints_differ() {
    // Check when the jg/jle pair is reached twice with different constraints
    test_inline(&[
        0x39, 0xda, // cmp edx, ebx
        0x74, 0x04, // je undef_flags
        0x39, 0xc8, // cmp eax, ecx
        0xeb, 0x00, // jmp undef_flags (Force next instruction to have undef flags)
        // undef_flags
        0x74, 0x09, // je end
        0x70, 0x07, // jo end
        0xeb, 0x00, // jmp jg_jle
        // jg_jle
        0x7f, 0x03, // jg ret
        0x7e, 0x01, // jle ret
        0xcc, // int3
        // end
        0x39, 0xc8, // cmp eax, ecx
        0x7a, 0xf7, // jpe jg_jle
        0xc3, //ret
    ], &[
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
        0x76, 0x05, // jbe ret_
        0x39, 0xca, // cmp edx, ecx
        0x77, 0x01, // ja ret_
        0x46, // inc esi
        0xeb, 0x00, // jmp ret
        0xc3, // ret
    ], &[
         (ctx.register(6), ctx.new_undef()),
    ]);
}

#[test]
fn psllq() {
    let ctx = &OperandContext::new();
    // ecx:eax = 1212_1212_4545_4545 << 20
    // ebx:edx = 8800_8800_9999_9999 << 20
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x45, 0x45, 0x45, 0x45, // mov [esp], 4545_4545
        0xc7, 0x44, 0xe4, 0x04, 0x12, 0x12, 0x12, 0x12, // mov [esp + 4], 1212_1212
        0xc7, 0x44, 0xe4, 0x08, 0x99, 0x99, 0x99, 0x99, // mov [esp + 8], 9999_9999
        0xc7, 0x44, 0xe4, 0x0c, 0x00, 0x88, 0x00, 0x88, // mov [esp + c], 8800_8800
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0xc7, 0x04, 0xe4, 0x10, 0x00, 0x00, 0x00, // mov [esp], 10
        0xc7, 0x44, 0xe4, 0x04, 0x00, 0x00, 0x00, 0x00, // mov [esp + 4], 0
        0x66, 0x0f, 0xf3, 0x04, 0xe4, // psllq xmm0, [esp]
        0x0f, 0x11, 0x04, 0xe4, // movups [esp], xmm0
        0x8b, 0x04, 0xe4, // mov eax, [esp]
        0x8b, 0x4c, 0xe4, 0x04, // mov ecx, [esp + 4]
        0x8b, 0x54, 0xe4, 0x08, // mov edx, [esp + 8]
        0x8b, 0x5c, 0xe4, 0x0c, // mov ebx, [esp + c]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0x4545_0000)),
         (ctx.register(1), ctx.constant(0x1212_4545)),
         (ctx.register(2), ctx.constant(0x9999_0000)),
         (ctx.register(3), ctx.constant(0x8800_9999)),
    ]);
}

#[test]
fn negative_offset() {
    let ctx = &OperandContext::new();
    // Had a bug that lazy flag updates didn't invalidate extra constraints
    // (So the unrelated cmp -> ja would always be taken)
    test_inline(&[
        0x31, 0xc0, // xor eax, eax
        0xb0, 0x09, // mov al, 9
        0x8b, 0x0d, 0x00, 0x34, 0x12, 0x00, // mov ecx [123400]
        0x89, 0xca, // mov edx, ecx
        0x8d, 0x49, 0x01, // lea ecx, [ecx + 1]
        0x88, 0x41, 0xff, // mov [ecx - 1], al
        0x8a, 0x02, // mov al, [edx]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(9)),
         (ctx.register(1), ctx.add(ctx.mem32(ctx.constant(0x123400)), ctx.constant(1))),
         (ctx.register(2), ctx.mem32(ctx.constant(0x123400))),
    ]);
}

#[test]
fn push_pop() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0xe0, // mov eax, esp
        0x50, // push eax
        0xff, 0x34, 0xe4, // push dword [esp]
        0x50, // push eax
        0x68, 0x90, 0x00, 0x00, 0x00, // push 90
        0x50, // push eax
        0x50, // push eax
        0xc7, 0x00, 0x80, 0x00, 0x00, 0x00, // mov dword [eax], 80
        0x8b, 0x4c, 0xe4, 0x18, // mov ecx, [esp + 18]
        0x8b, 0x50, 0xf0, // mov edx, [eax - 10]
        0x5b, // pop ebx
        0x8f, 0x45, 0x00, // pop dword [ebp]
        0x5b, // pop ebx
        0x8f, 0x45, 0x00, // pop dword [ebp]
        0x5b, // pop ebx
        0x8f, 0x45, 0x00, // pop dword [ebp]
        // esp = esp_at_start = eax
        0x5b, // pop ebx
        0x8f, 0x45, 0x00, // pop dword [ebp]
        0x5b, // pop ebx
        0x8f, 0x45, 0x00, // pop dword [ebp]
        // esp = eax + 10
        0xc7, 0x00, 0x22, 0x00, 0x00, 0x00, // mov dword [eax], 22
        0x8b, 0x5c, 0xe4, 0xf0, // mov ebx, [esp - 10]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.register(4)),
         (ctx.register(1), ctx.constant(0x80)),
         (ctx.register(2), ctx.constant(0x90)),
         (ctx.register(3), ctx.constant(0x22)),
         (ctx.register(4), ctx.add(ctx.register(4), ctx.constant(0x10))),
    ]);
}

#[test]
fn stack_sub() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x89, 0xe0, // mov eax, esp
        0x50, // push eax
        0x83, 0xec, 0x10, // sub esp, 10
        0x50, // push eax
        0x83, 0xec, 0x10, // sub esp, 10
        0xc7, 0x00, 0x80, 0x00, 0x00, 0x00, // mov dword [eax], 80
        0x8b, 0x4c, 0xe4, 0x28, // mov ecx, [esp + 28]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.register(4)),
         (ctx.register(1), ctx.constant(0x80)),
         (ctx.register(4), ctx.sub(ctx.register(4), ctx.constant(0x28))),
    ]);
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
fn mov_al() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb0, 0x55, // mov al, 55
        0xa2, 0xdd, 0xcc, 0xbb, 0xaa, // mov [aabbccdd], al
        0x0f, 0xb6, 0x0d, 0xdd, 0xcc, 0xbb, 0xaa, // movzx ecx, byte [aabbccdd]
        0xa0, 0x78, 0x56, 0x34, 0x12, // mov al, [12345678]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem8(ctx.constant(0x12345678))),
         (ctx.register(1), ctx.constant(0x55)),
    ]);
}

#[test]
fn test_eax_after_call() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x31, 0xdb, // xor ebx, ebx
        0x85, 0xc0, // test eax, eax
        0x75, 0x0c, // jne end
        0xe8, 0x00, 0x00, 0x00, 0x00, // call x
        0x66, 0x85, 0xc0, // test ax, ax
        0x74, 0x02, // je end
        0xb3, 0x01, // mov ebx, 1
        // end
        0xeb, 0x00,
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.new_undef()),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.new_undef()),
         (ctx.register(3), ctx.new_undef()),
         (ctx.register(4), ctx.new_undef()),
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
fn movd_to_reg() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xc7, 0x04, 0xe4, 0x02, 0x00, 0x00, 0x00, // mov [esp], 2
        0x0f, 0x10, 0x04, 0xe4, // movups xmm0, [esp]
        0x66, 0x0f, 0x7e, 0xc0, // movd eax, xmm0
        0x66, 0x0f, 0x7e, 0xc1, // movd ecx, xmm0
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(2)),
         (ctx.register(1), ctx.constant(2)),
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

struct CollectEndState<'e> {
    end_state: Option<ExecutionState<'e>>,
}

impl<'e> analysis::Analyzer<'e> for CollectEndState<'e> {
    type State = analysis::DefaultState;
    type Exec = ExecutionState<'e>;
    fn operation(&mut self, control: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        println!("@ {:x} {:#?}", control.address(), op);
        if let Operation::Move(_, val, _) = *op {
            println!("Resolved is {}", control.resolve(val));
        }
        if let Operation::Jump { condition, .. } = *op {
            println!("Resolved cond is {}", control.resolve(condition));
            println!(
                "Resolved cond is {} (Constraints applied)",
                control.resolve_apply_constraints(condition),
            );
        }
        if let Operation::Return(_) = *op {
            let state = control.exec_state();
            self.end_state = Some(state.clone());
        }
    }
}

fn test_inner<'e, 'b>(
    file: &'e BinaryFile<VirtualAddress>,
    func: VirtualAddress,
    changes: &[(Operand<'b>, Operand<'b>)],
) {
    let ctx = &OperandContext::new();
    let changes = changes.iter().map(|&(a, b)| {
        (ctx.copy_operand(a), ctx.copy_operand(b))
    }).collect::<Vec<_>>();

    let state = ExecutionState::with_binary(file, ctx);
    let mut expected_state = state.clone();
    for &(op, val) in &changes {
        let op = Operation::Move(DestOperand::from_oper(op), val, None);
        expected_state.update(&op);
    }
    let mut analysis = analysis::FuncAnalysis::with_state(file, ctx, func, state);
    let mut collect_end_state = CollectEndState {
        end_state: None,
    };
    analysis.analyze(&mut collect_end_state);

    println!("{:?}", analysis.errors);
    assert!(analysis.errors.is_empty());
    let mut end_state = collect_end_state.end_state.unwrap();
    for i in 0..8 {
        let expected = expected_state.resolve(ctx.register(i));
        let end = end_state.resolve(ctx.register(i));
        if end.iter().any(|x| x.is_undefined()) {
            let expected_is_ud = expected.is_undefined();
            assert!(expected_is_ud, "Register {}: got undef {} expected {}", i, end, expected);
        } else {
            assert_eq!(expected, end, "Register {}: got {} expected {}", i, end, expected);
        }
    }
}

fn test_inline<'e>(code: &[u8], changes: &[(Operand<'e>, Operand<'e>)]) {
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

fn test<'b>(idx: usize, changes: &[(Operand<'b>, Operand<'b>)]) {
    let binary = helpers::raw_bin(OsStr::new("test_inputs/exec_state.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = binary.code_section().virtual_address + offset;
    test_inner(&binary, func, changes);
}
