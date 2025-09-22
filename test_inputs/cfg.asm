; nasm -f bin cfg.asm -o cfg.bin
bits 32

dd switch_func
dd undecideable_jump
dd constant_switch

constant_switch:
mov dword [esi + 0x3c], 1

switch_func:
mov eax, [esi + 0x38]
mov ecx, [esi + 0x3c]
cmp ecx, 4
ja .default
jmp [0x401000 + switch_table + ecx * 4]
.default:
ret

_switch0:
add eax, 4
ret

_switch1:
shl eax, 13
ret

_switch2:
mov eax, [0x1234]
ret

_switch3:
sub eax, edx
ret

_switch4:
add eax, edx
ret

undecideable_jump:
jmp dword [eax + 4]

; Leave this at the end
switch_table:
dd 0x401000 + _switch0
dd 0x401000 + _switch1
dd 0x401000 + _switch2
dd 0x401000 + _switch3
dd 0x401000 + _switch4
