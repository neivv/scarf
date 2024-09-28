extern crate byteorder;
extern crate scarf;

mod helpers;
mod shared_x86;

use std::ffi::OsStr;

use byteorder::{ReadBytesExt, LittleEndian};

use scarf::{
    BinaryFile, BinarySection, DestOperand, Operand, Operation, OperandContext, VirtualAddress32,
    MemAccessSize,
};
use scarf::analysis::{self, Control};
use scarf::ExecutionStateX86 as ExecutionState;
use scarf::exec_state::ExecutionState as _;
use scarf::exec_state::{OperandCtxExtX86};

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
        // Not even sure if this should be Mem32[ffffffff] or
        // Mem8[ffffffff] | ((Mem32[0] << 8) & ffffff00)
        // I think this test was here to just verify things don't panic,
        // not so much that the result stays stable.
        (ctx.register(0), ctx.mem32c(0xffff_ffff)),
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
        (ctx.register(2), ctx.or(
            ctx.constant(0xc3_0040),
            ctx.and_const(ctx.lsh_const(ctx.mem32(ctx.const_0(), 0x401010), 0x8), 0xff00_0000),
        )),
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
         (ctx.register(2), ctx.mem8(ctx.register(6), 0x4d)),
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
         (ctx.register(1), ctx.add(ctx.mem32(ctx.const_0(), 0x123400), ctx.constant(1))),
         (ctx.register(2), ctx.mem32(ctx.const_0(), 0x123400)),
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
fn mov_al() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb0, 0x55, // mov al, 55
        0xa2, 0xdd, 0xcc, 0xbb, 0xaa, // mov [aabbccdd], al
        0x0f, 0xb6, 0x0d, 0xdd, 0xcc, 0xbb, 0xaa, // movzx ecx, byte [aabbccdd]
        0x31, 0xc0, // xor eax, eax
        0xa0, 0x78, 0x56, 0x34, 0x12, // mov al, [12345678]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem8(ctx.const_0(), 0x12345678)),
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
fn move_mem_in_parts() {
    let ctx = &OperandContext::new();
    // Issue was that eax was being assigned esi,
    // and esi was zeroed afterwards, but the small stores
    // kept assuming that unmodified value of `Mem32[eax]` == `Mem32[0]`
    test_inline(&[
        0x89, 0xf0, // mov eax, esi
        0x31, 0xf6, // xor esi, esi
        0x31, 0xc9, // xor ecx, ecx
        0x88, 0x08, // mov [eax], cl
        0x66, 0x8b, 0x0d, 0x30, 0x12, 0x00, 0x00, // mov cx, [1230]
        0x66, 0x89, 0x08, // mov [eax], cx
        0x8a, 0x0d, 0x32, 0x12, 0x00, 0x00, // mov cl, [1232]
        0x88, 0x48, 0x02, // mov [eax + 2], cl
        0x8a, 0x0d, 0x33, 0x12, 0x00, 0x00, // mov cl, [1233]
        0x88, 0x48, 0x03, // mov [eax + 3], cl
        0x31, 0xc9, // xor ecx, ecx
        0x8b, 0x00, // mov eax, [eax]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem32(ctx.const_0(), 0x1230)),
         (ctx.register(1), ctx.constant(0)),
         (ctx.register(6), ctx.constant(0)),
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
        0x40, // inc eax (eax = (eax > ecx) == 0)
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
        (ctx.register(2), ctx.lsh_const(ctx.register(0), 0x10)),
        (ctx.register(3), ctx.and_const(
            ctx.or(
                ctx.and_const(ctx.rsh_const(ctx.register(0), 0x10), 0x0000_ffff),
                ctx.and_const(ctx.lsh_const(ctx.register(1), 0x10), 0xffff_0000),
            ),
            0xffff_ffff,
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
        (ctx.register(0), ctx.and_const(
            ctx.or(
                ctx.and_const(ctx.rsh_const(ctx.register(2), 0x10), 0x0000_ffff),
                ctx.and_const(ctx.lsh_const(ctx.register(3), 0x10), 0xffff_0000),
            ),
            0xffff_ffff,
        )),
        (ctx.register(1), ctx.and_const(
            ctx.rsh_const(ctx.register(3), 0x10),
            0xffff,
        )),
        (ctx.register(2), ctx.constant(0)),
        (ctx.register(3), ctx.constant(0)),
    ]);
}

#[test]
fn test_switch_cases_in_memory() {
    let ctx = &OperandContext::new();
    // 2 cases to ok, 3rd fake
    // rcx is undef if the cases are run
    test(4, &[
         (ctx.register(0), ctx.constant(0)),
         (ctx.register(1), ctx.new_undef()),
         (ctx.register(2), ctx.constant(0)),
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
         (ctx.register(4), ctx.new_undef()),
    ]);
}

#[test]
fn misc_coverage() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x0f, 0xb6, 0xcc, // movzx ecx, ah
        0x00, 0xe1, // add cl, ah
        0x00, 0xf1, // add cl, dh
        0x8b, 0x04, 0xbd, 0x00, 0x50, 0x00, 0x00, // mov eax, [edi * 4 + 5000]
        0xc3, // ret
    ], &[
         (
            ctx.register(0),
            ctx.mem32(
                ctx.mul_const(ctx.register(7), 4),
                0x5000,
            ),
         ),
         (
            ctx.register(1),
            ctx.and_const(
                ctx.add(
                    ctx.add(
                        ctx.rsh_const(
                            ctx.and_const(ctx.register(0), 0xff00),
                            0x8,
                        ),
                        ctx.rsh_const(
                            ctx.and_const(ctx.register(0), 0xff00),
                            0x8,
                        ),
                    ),
                    ctx.rsh_const(
                        ctx.and_const(ctx.register(2), 0xff00),
                        0x8,
                    ),
                ),
                0xff,
            ),
         ),
    ]);
}

#[test]
fn mov_al_constmem() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0xb8, 0x34, 0x12, 0x00, 0x00, // mov eax, 1234
        0xa0, 0x34, 0x12, 0x00, 0x00, // mov al, [1234]
        0x89, 0xc1, // mov ecx, eax
        0xb8, 0x78, 0x56, 0x34, 0x12, // mov eax, 12345678
        0x66, 0xa1, 0x11, 0x11, 0x00, 0x00, // mov ax, [1111]
        0xc3, // ret
    ], &[
         (
            ctx.register(0),
            ctx.or(
                ctx.mem16c(0x1111),
                ctx.constant(0x1234_0000),
            ),
         ),
         (
            ctx.register(1),
            ctx.or(
                ctx.mem8c(0x1234),
                ctx.constant(0x1200),
            ),
         ),
    ]);
}

#[test]
fn mov_mem32_consistency_1() {
    let ctx = &OperandContext::new();
    // Single register to memory and back does not get & ffff_ffff masks
    // It could, but currently not done to not break too much user code
    // (Mainly since ecx is used as 'this' argument in functions and
    // so far the correct way to check it in scarf has been not assuming
    // ffff_ffff mask there)
    test_inline(&[
        0xa3, 0x10, 0x10, 0x00, 0x00, // mov [1010], eax
        0xa1, 0x10, 0x10, 0x00, 0x00, // mov eax, [1010]
        0x89, 0x0d, 0x10, 0x10, 0x00, 0x00, // mov [1010], ecx
        0x8b, 0x0d, 0x10, 0x10, 0x00, 0x00, // mov ecx, [1010]
        0x89, 0x16, // mov [esi], edx
        0x8b, 0x16, // mov edx, [esi]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.register(0)),
        (ctx.register(1), ctx.register(1)),
        (ctx.register(2), ctx.register(2)),
        (ctx.mem32c(0x1010), ctx.register(1)),
    ]);
}

#[test]
fn mov_mem32_consistency_2() {
    let ctx = &OperandContext::new();
    // Should not get & ffff_ffff masks due to ctx.normalize_32bit
    test_inline_with_init(&[
        0xa3, 0x10, 0x10, 0x00, 0x00, // mov [1010], eax
        0xa1, 0x10, 0x10, 0x00, 0x00, // mov eax, [1010]
        0x89, 0x0d, 0x10, 0x10, 0x00, 0x00, // mov [1010], ecx
        0x8b, 0x0d, 0x10, 0x10, 0x00, 0x00, // mov ecx, [1010]
        0x89, 0x16, // mov [esi], edx
        0x8b, 0x16, // mov edx, [esi]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.custom(0)),
        (ctx.register(1), ctx.custom(1)),
        (ctx.register(2), ctx.custom(2)),
    ], &[
        (ctx.register(0), ctx.custom(0)),
        (ctx.register(1), ctx.custom(1)),
        (ctx.register(2), ctx.custom(2)),
    ]);
}

#[test]
fn mov_mem32_consistency_3() {
    let ctx = &OperandContext::new();
    // Register move but misaligned memory address
    test_inline(&[
        0xa3, 0x1e, 0x10, 0x00, 0x00, // mov [101e], eax
        0xa1, 0x1e, 0x10, 0x00, 0x00, // mov eax, [101e]
        0x89, 0x0d, 0x1e, 0x10, 0x00, 0x00, // mov [101e], ecx
        0x8b, 0x0d, 0x1e, 0x10, 0x00, 0x00, // mov ecx, [101e]
        0x89, 0x56, 0x0e, // mov [esi + e], edx
        0x8b, 0x56, 0x0e, // mov edx, [esi + e]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.register(0)),
        (ctx.register(1), ctx.register(1)),
        (ctx.register(2), ctx.register(2)),
        (ctx.mem32c(0x101e), ctx.register(1)),
    ]);
}

#[test]
fn mov_mem32_consistency_4() {
    let ctx = &OperandContext::new();
    // Custom move with misaligned memory address
    // Should not get & ffff_ffff masks due to ctx.normalize_32bit
    test_inline_with_init(&[
        0xa3, 0x1e, 0x10, 0x00, 0x00, // mov [101e], eax
        0xa1, 0x1e, 0x10, 0x00, 0x00, // mov eax, [101e]
        0x89, 0x0d, 0x1e, 0x10, 0x00, 0x00, // mov [101e], ecx
        0x8b, 0x0d, 0x1e, 0x10, 0x00, 0x00, // mov ecx, [101e]
        0x89, 0x56, 0x0e, // mov [esi + e], edx
        0x8b, 0x56, 0x0e, // mov edx, [esi + e]
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.custom(0)),
        (ctx.register(1), ctx.custom(1)),
        (ctx.register(2), ctx.custom(2)),
    ], &[
        (ctx.register(0), ctx.custom(0)),
        (ctx.register(1), ctx.custom(1)),
        (ctx.register(2), ctx.custom(2)),
    ]);
}

#[test]
fn or_xor_simplify_bug() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x09, 0x09, // or [ecx], ecx ; [ecx] | ecx
        0x41, // inc ecx ; ecx + 1
        // Let [ecx1] = read_memory(ecx + 1, Mem32) =
        //      (((Mem32[ecx] | ecx) >> 8) & ff_ffff) | (Mem8 [ecx + 4] << 18)
        0x31, 0x31, // xor [ecx], esi ; [ecx1] ^ esi
        0x00, 0x31, // add [ecx], dh ; ([ecx1] ^ esi) + ((edx >> 8) & ff)
        0x31, 0x31, // xor [ecx], esi ; (([ecx1] ^ esi) + ((edx >> 8) & ff)) ^ esi
        0x0c, 0xff, // or al, ff ; eax | ff
        0x31, 0x01, // xor [ecx], eax ; (([ecx1] ^ esi) + ((edx >> 8) & ff)) ^ esi ^ (eax | ff)
        0x33, 0x31, // xor esi, [ecx] ; (([ecx1] ^ esi) + ((edx >> 8) & ff)) ^ (eax | ff)
        0x31, 0x31, // xor [ecx], esi ; esi
        0x8b, 0x01, // mov eax, [ecx] ; esi
        0x31, 0xf6, // xor esi, esi ; 0
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.register(6)),
        (ctx.register(1), ctx.add_const(ctx.register(1), 1)),
        (ctx.register(6), ctx.constant(0)),
    ]);
}

#[test]
fn or_xor_simplify_complex() {
    let ctx = &OperandContext::new();
    test_inline(&[
        0x41, // 00 inc ecx
        0x33, 0x33, // 01 xor esi, [ebx] ; esi ^ [ebx]
        0x33, 0x32, // 03 xor esi, [edx] ; esi ^ [ebx] ^ [edx]
        0x31, 0x31, // 05 xor [ecx], esi ; esi ^ [ebx] ^ [edx] ^ [ecx] = E4
        0x33, 0x33, // 07 xor esi, [ebx] ; esi ^ [edx]
        0x0b, 0x31, // 09 or esi, [ecx] ; (esi ^ [edx]) | E4
        0x29, 0x33, // 0b sub [ebx], esi ; [ebx] - ((esi ^ [edx]) | E4)
        0x31, 0x31, // 0d xor [ecx], esi ; ((esi ^ [edx]) | E4) ^ E4
        0x33, 0x31, // 0f xor esi, [ecx] ; E4
        0x31, 0x31, // 11 xor [ecx], esi ; (esi ^ [edx]) | E4
        0x0b, 0x31, // 13 or esi, [ecx] ; E4 | (esi ^ [edx])
        // Complicates [ebx] to hit a bug; won't change result otherwise
        0x29, 0x33, // 15 sub [ebx], esi
        0x31, 0x31, // 17 xor [ecx], esi ; 0
        // Xor [ecx] twice with same value won't matter
        // (As long as simplification doesn't give up)
        0x33, 0x33, // 19 xor esi, [ebx]
        0x31, 0x31, // 1b xor [ecx], esi
        0x31, 0x31, // 1d xor [ecx], esi
        0x8b, 0x01, // 1f mov eax, [ecx]
        0x31, 0xf6, // 21 xor esi, esi ; 0
        0xc3, // 22 ret
    ], &[
        (ctx.register(0), ctx.constant(0)),
        (ctx.register(1), ctx.add_const(ctx.register(1), 1)),
        (ctx.register(6), ctx.constant(0)),
    ]);
}

#[test]
fn many_xor_bug() {
    // Caused simplification debug assert due to constants not being properly folded in xor
    let ctx = &OperandContext::new();
    test_inline(&[
        0x20, 0x22, // and byte [edx], ah
        0x09, 0x09, // or [ecx], ecx
        0x09, 0x31, // or [ecx], esi
        0x35, 0x36, 0x37, 0x32, 0x37, // xor eax, 37323736
        0x31, 0x41, 0xff, // xor [ecx - 1], eax
        0x31, 0x31, // xor [ecx], esi
        0x41, // inc ecx
        0x00, 0x31, // add byte [ecx], dh
        0x31, 0x39, // xor [ecx], edi
        0x10, 0x29, // adc [ecx], ch
        0x33, 0x34, 0x03, // xor esi, [eax + ebx]
        0x33, 0x32, // xor esi, [edx]
        0x32, 0x32, // xor dh, byte [edx]
        0x33, 0x33, // xor esi, [ebx]
        0x33, 0x34, 0x33, // xor esi, [esi + ebx]
        0x33, 0x31, // xor esi, [ecx]
        0x41, // inc ecx
        0x33, 0x32, // xor esi, [edx]
        0x33, 0x31, // xor esi, [ecx]
        0x31, 0x31, // xor [ecx], esi
        0x21, 0x31, // and [ecx], esi
        0x31, 0x31, // xor [ecx], esi
        0x31, 0x31, // xor [ecx], esi
        0x09, 0x09, // or [ecx], ecx
        0x33, 0xd2, // xor edx, edx
        0x31, 0xf6, // xor esi, esi
        0xc3, // ret
    ], &[
        (ctx.register(0), ctx.xor_const(ctx.register(0), 0x37323736)),
        (ctx.register(1), ctx.add_const(ctx.register(1), 2)),
        (ctx.register(2), ctx.constant(0)),
        (ctx.register(6), ctx.constant(0)),
    ]);
}

#[test]
fn u16_push() {
    let ctx = &OperandContext::new();

    test_inline(&[
        // Pushes only u16 (Probs illegal in any OS ABI stack pointer requirements)
        0x66, 0x6a, 0xfd, // push -3
        0x66, 0x68, 0x34, 0x12, // push 1234
        0x58, // pop eax
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.constant(0xfffd_1234)),
    ]);
}

#[test]
fn nop_xor_on_register_1() {
    // xor bl, 0 should not affect mov ebx, [ebx]
    test_inline(&[
        0x53, // push ebx
        0x89, 0xe3, // mov ebx, esp
        0x85, 0xc0, // test eax, eax
        0x74, 0x06, // je end
        0x80, 0xf3, 0x00, // xor bl, 0
        0x73, 0x01, // jae end
        0xcc, // int3
        // end:
        0x8b, 0x1b, // mov ebx, [ebx] (Old ebx)
        0x83, 0xc4, 0x04, // add esp, 4
        0xc3, // ret
    ], &[
    ]);
}

#[test]
fn nop_xor_on_register_2() {
    // xor bl, 88 twice should not affect mov ebx, [ebx]
    test_inline(&[
        0x53, // push ebx
        0x89, 0xe3, // mov ebx, esp
        0x85, 0xc0, // test eax, eax
        0x74, 0x09, // je end
        0x80, 0xf3, 0x88, // xor bl, 88
        0x80, 0xf3, 0x88, // xor bl, 88
        0x73, 0x01, // jae end
        0xcc, // int3
        // end:
        0x8b, 0x1b, // mov ebx, [ebx] (Old ebx)
        0x83, 0xc4, 0x04, // add esp, 4
        0xc3, // ret
    ], &[
    ]);
}

#[test]
fn jump_conditions() {
    let ctx = &OperandContext::new();
    test(5, &[
         (ctx.register(0), ctx.new_undef()),
    ]);
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

#[test]
fn stack_arr_write_read() {
    let file = &make_binary(&[
        // Make esp undefined
        0x85, 0xc9, // test ecx, ecx
        0x74, 0x03, // je skip_sub
        0x83, 0xec, 0x04, // sub esp, 4
        // skip_sub:
        0xeb, 0x00, // jmp next (Force state merge at next)
        // next:
        0x8b, 0xec, // mov ebp, esp
        0x8d, 0x45, 0xff, // lea eax, [ebp - 1]
        0x50, // push eax
        0xc6, 0x45, 0xff, 0x11, // mov byte [ebp - 1], 0x11
        0x6a, 0x01, // push 1
        0x50, // push eax
        0xc3, // ret
    ]);
    let func = file.code_section().virtual_address;
    let ctx = &OperandContext::new();

    let state = ExecutionState::with_binary(file, ctx);
    let mut analysis = analysis::FuncAnalysis::with_state(file, ctx, func, state);
    let mut collect_end_state = CollectEndState {
        end_state: None,
    };
    analysis.analyze(&mut collect_end_state);

    let mut end_state = collect_end_state.end_state.unwrap();
    // Read value at [esp], it should be pointer to stack memory
    // read u8 at that value, it should be 0x11
    let value = end_state.resolve(ctx.mem32(ctx.register(4), 0));
    println!("Value at [esp] = {value}");
    let inner = end_state.read_memory(&ctx.mem_access(value, 0, MemAccessSize::Mem8));
    assert_eq!(inner, ctx.constant(0x11));
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
        ctx.normalize_32bit(
            ctx.lsh_const(
                ctx.and_const(
                    ctx.custom(2),
                    0xffff_ffff,
                ),
                8,
            ),
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
fn not_binary_mem_read() {
    let ctx = &OperandContext::new();

    test_inline(&[
        0x8b, 0x80, 0x00, 0x10, 0x40, 0x00, // mov eax, [eax + 401000]
        0xc6, 0x81, 0x00, 0x10, 0x40, 0x00, 0x05, // mov byte [ecx + 401000], 5
        0x8b, 0x89, 0x00, 0x10, 0x40, 0x00, // mov rcx, [ecx + 401000]
        0xc3, // ret
    ], &[
         (ctx.register(0), ctx.mem32(ctx.register(0), 0x401000)),
         (
            ctx.register(1),
            ctx.or(
                ctx.and_const(
                    ctx.mem32(ctx.register(1), 0x401000),
                    0xffff_ff00,
                ),
                ctx.constant(5),
            ),
         ),
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

fn test_inner<'e, 'b>(
    file: &'e BinaryFile<VirtualAddress32>,
    func: VirtualAddress32,
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
    for i in 0..8 {
        let reg = ctx.register(i);
        if expected.iter().any(|x| x.0 == reg) {
            continue;
        }
        let end = end_state.resolve(reg);
        assert_eq!(end, reg, "Register {}: got {} expected {}", i, end, reg);
    }
    if xmm {
        for i in 0..8 {
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

fn make_binary_with_virtual_size(code: &[u8], virtual_size: u32) -> BinaryFile<VirtualAddress32> {
    scarf::raw_bin(VirtualAddress32(0x00400000), vec![BinarySection {
        name: *b".text\0\0\0",
        virtual_address: VirtualAddress32(0x401000),
        virtual_size,
        data: code.into(),
    }])
}

fn make_binary(code: &[u8]) -> BinaryFile<VirtualAddress32> {
    make_binary_with_virtual_size(code, code.len() as u32)
}

fn test_inline_xmm<'e>(code: &[u8], changes: &[(Operand<'e>, Operand<'e>)]) {
    let binary = make_binary(code);
    test_inner(&binary, binary.code_section().virtual_address, changes, &[], true);
}

fn test_inline<'e>(code: &[u8], changes: &[(Operand<'e>, Operand<'e>)]) {
    test_inline_with_init(code, changes, &[])
}

fn test_inline_with_init<'e>(
    code: &[u8],
    changes: &[(Operand<'e>, Operand<'e>)],
    init: &[(Operand<'e>, Operand<'e>)],
) {
    let binary = scarf::raw_bin(VirtualAddress32(0x00400000), vec![BinarySection {
        name: {
            // ugh
            let mut x = [0; 8];
            for (out, &val) in x.iter_mut().zip(b".text\0\0\0".iter()) {
                *out = val;
            }
            x
        },
        virtual_address: VirtualAddress32(0x401000),
        virtual_size: code.len() as u32,
        data: code.into(),
    }]);
    test_inner(&binary, binary.code_section().virtual_address, changes, init, false);
}

fn test<'b>(idx: usize, changes: &[(Operand<'b>, Operand<'b>)]) {
    let binary = helpers::raw_bin(OsStr::new("test_inputs/exec_state.bin")).unwrap();
    let offset = (&binary.code_section().data[idx * 4..]).read_u32::<LittleEndian>().unwrap();
    let func = VirtualAddress32(offset);
    test_inner(&binary, func, changes, &[], false);
}
