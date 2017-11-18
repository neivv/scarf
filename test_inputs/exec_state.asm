; nasm -f bin exec_state.asm -o exec_state.bin
bits 32

dd movzx_test
dd movsx_test
dd movzx_mem
dd movsx_mem

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
