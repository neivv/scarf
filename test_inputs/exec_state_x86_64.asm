; nasm -f bin exec_state_x86_64.asm -o exec_state_x86_64.bin
bits 64
default rel
org 0x401000

dq switch_cases_in_memory

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
