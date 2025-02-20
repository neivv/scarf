; nasm -f bin test_inputs/exec_state_x86_64.asm -o test_inputs/exec_state_x86_64.bin
bits 64
default rel
org 0x401000

dq switch_cases_in_memory
dq switch_different_resolved_constraints_on_branch_end
dq switch_u32_with_sub
dq switch_negative_cases

switch_cases_in_memory:
xor edx, edx
mov DWORD [rsp + 0x10], eax
.loop_back:
xor ecx, ecx
cmp DWORD [rsp + 0x10], 2
jae .end
lea rcx, [.switch_table]
mov eax, [rsp + 0x10]
movzx rcx, WORD [rcx + rax * 2]
lea rax, [.fail]
add rax, rcx
xor ecx, ecx
jmp rax
.switch_table:
dw .case1 - .fail
dw .case2 - .fail
dw .fail - .fail
.fail:
int3
.case1:
add ecx, 6
.case2:
add ecx, 6
jmp .end
.end:
inc edx
inc eax
mov [rsp + 0x10], eax
cmp edx, 2
jl .loop_back
xor eax, eax
xor edx, edx
ret

switch_different_resolved_constraints_on_branch_end:
.base:
cmp qword [rax], 0
je .skip
mov rax, [rcx]
mov r8d, [rax + 8]
.skip:
mov r13d, r8d
cmp r13d, 2
ja .end
.switch_start:
lea r9, [.base]
mov ecx, [r9 + r13 * 4 + .switch_table - .base]
add rcx, r9
jmp rcx
.switch_table:
dd .case0 - .base
dd .case0 - .base
dd .case2 - .base
dd .fake - .base
.case0:
xor esi, esi
jmp .end
.case2:
xor edi, edi
.end:
jmp .end2
.end2:
mov rax, [rax]
mov r13d, [rax + 8]
cmp r13d, 2
jbe .switch_start
ret
.fake:
int3

switch_u32_with_sub:
.base:
lea eax, [rcx - 0xd]
cmp eax, 0xd
ja .end
lea r9, [.base]
mov ecx, [r9 + rax * 4 + .switch_table - .base]
add rcx, r9
jmp rcx
.switch_table:
dd .zero_eax - .base
dd .zero_ecx - .base
dd .zero_edx - .base
dd .zero_esi - .base
dd .zero_edi - .base
dd .zero_r8 - .base
dd .zero_r8 - .base
dd .zero_ecx - .base
dd .zero_ecx - .base
dd .zero_edx - .base
dd .zero_esi - .base
dd .zero_edi - .base
dd .zero_r8 - .base
dd .zero_r9 - .base
dd .fake - .base
.zero_eax:
xor eax, eax
jmp .end
.zero_ecx:
xor ecx, ecx
jmp .end
.zero_edx:
xor edx, edx
jmp .end
.zero_edi:
xor edi, edi
jmp .end
.zero_esi:
xor esi, esi
jmp .end
.zero_r8:
xor r8d, r8d
jmp .end
.zero_r9:
xor r9d, r9d
.end:
jmp .end2
.end2:
ret
.fake:
int3

switch_negative_cases:
xor ecx, ecx
add rax, 3
cmp rax, 3
jae .end
; rax < 3
lea rcx, [.switch_table]
mov ecx, dword [rcx + rax * 4]
lea rax, [.fail]
add rax, rcx
xor ecx, ecx
jmp rax
.switch_table:
dd .case1 - .fail
dd .case2 - .fail
dd .case2 - .fail
dd .fail - .fail
.fail:
int3
.case1:
add ecx, 6
add ebx, 6
.case2:
add ecx, 6
add edx, 6
jmp .end
.end:
xor eax, eax
ret
