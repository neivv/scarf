; nasm -f bin exec_state_x86_64.asm -o exec_state_x86_64.bin
bits 64
default rel
org 0x401000

dq switch_cases_in_memory
dq switch_different_resolved_constraints_on_branch_end

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
