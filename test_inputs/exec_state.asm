; nasm -f bin exec_state.asm -o exec_state.bin
bits 32
org 0x401000

dd movzx_test
dd movsx_test
dd movzx_mem
dd movsx_mem
dd switch_cases_in_memory

movzx_test:
mov eax, 0x88081001
movzx eax, al
movzx ecx, ax
sub cx, 4
movzx edx, cx
ret

movsx_test:
mov eax, 0x88081001
movsx eax, al
movsx ecx, ax
sub cx, 4
movsx edx, cx
ret

movzx_mem:
mov byte [0x112233], 0x90
mov eax, 0x112233
movzx eax, byte [eax]
ret

movsx_mem:
mov byte [0x112233], 0x90
mov eax, 0x112233
movsx eax, byte [eax]
ret

switch_cases_in_memory:
xor edx, edx
mov DWORD [esp + 0x10], eax
.loop_back:
xor ecx, ecx
cmp DWORD [esp + 0x10], 2
jae .end
lea ecx, [.switch_table]
mov eax, [esp + 0x10]
movzx ecx, WORD [ecx + eax * 2]
lea eax, [.fail]
add eax, ecx
xor ecx, ecx
jmp eax
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
mov [esp + 0x10], eax
cmp edx, 2
jl .loop_back
xor eax, eax
xor edx, edx
ret
