; nasm -f bin test_inputs/exec_state.asm -o test_inputs/exec_state.bin
bits 32
org 0x401000

dd movzx_test
dd movsx_test
dd movzx_mem
dd movsx_mem
dd switch_cases_in_memory
dd jump_conditions
dd switch_negative_cases

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

jump_conditions:
mov eax, [eax]
cmp esi, eax
jg .end
.loop:
jbe .ok
xchg bh, bh
ja .ok
int3
.ok:
add eax, 4
cmp esi, eax
jle .loop
.end:
ret

switch_negative_cases:
xor ecx, ecx
add eax, 3
cmp eax, 3
jae .end
; eax < 3
lea ecx, [.switch_table]
mov ecx, dword [ecx + eax * 4]
lea eax, [.fail]
add eax, ecx
xor ecx, ecx
jmp eax
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
