; nasm -f bin slow.asm -o slow.bin
bits 32

dd hash 
dd hash2
dd hash3
dd slow4
dd slow5
dd slow6

; Interestingly this was slow since it analyzed the loop twice, creating two distinct Operand
; trees which it then compared.
hash:
mov [ebp - 0x50], esi
add eax, 2
mov [ebp - 0x58], ecx
lea esi, [ebp - 0x44]
mov edi, 0x10
nop dword [eax + eax]
.loop:
movzx edx, byte [eax + 1]
lea esi, [esi + 4]
movzx ecx, byte [eax]
lea eax, [eax + 4]
shl edx, 8
or edx, ecx
movzx ecx, byte [eax - 5]
shl edx, 8
or edx, ecx
movzx ecx, byte [eax - 6]
shl edx, 8
or edx, ecx
mov [esi - 4], edx
sub edi, 1
jne .loop
.hash:

mov edi, [ebp - 0x50]
mov ecx, ebx
not ecx
mov esi, [ebp - 0x4c]
and ecx, esi
mov edx, [ebp - 0x58]
mov eax, edi
add edx, 0xd76aa478
and eax, ebx
add esi, 0xe8c7b756
or ecx, eax
mov eax, ebx
add ecx, [ebp - 0x44]
add edx, ecx
rol edx, 0x7
add edx, ebx
and eax, edx
mov ecx, edx
not ecx
and ecx, edi
add edi, 0x242070db
or ecx, eax
add ecx, [ebp - 0x40]
add esi, ecx
rol esi, 0x0c
add esi, edx
mov ecx, esi
mov eax, esi
not ecx
and eax, edx
and ecx, ebx
add ebx, 0xc1bdceee
or ecx, eax
mov eax, esi
add ecx, [ebp - 0x3c]
add edi, ecx
ror edi, 0x0f
add edi, esi
and eax, edi
mov ecx, edi
not ecx
and ecx, edx
or ecx, eax
mov eax, edi
add ecx, [ebp - 0x38]
add ebx, ecx
ror ebx, 0x0a
add ebx, edi
and eax, ebx
mov ecx, ebx
not ecx
and ecx, esi
or ecx, eax
mov eax, ebx
add ecx, 0xf57c0faf
add ecx, [ebp - 0x34]
add edx, ecx
rol edx, 0x7
add edx, ebx
mov ecx, edx
and eax, edx
not ecx
and ecx, edi
or ecx, eax
add ecx, 0x4787c62a
add ecx, [ebp - 0x30]
add esi, ecx
rol esi, 0x0c
add esi, edx
mov ecx, esi
mov eax, esi
not ecx
and eax, edx
and ecx, ebx
or ecx, eax
add ecx, 0xa8304613
add ecx, [ebp - 0x2c]
add edi, ecx
ror edi, 0x0f
add edi, esi
mov ecx, edi
not ecx
mov eax, esi
and ecx, edx
and eax, edi
or ecx, eax
mov eax, edi
add ecx, 0xfd469501
add ecx, [ebp - 0x28]
add ebx, ecx
ror ebx, 0x0a
add ebx, edi
and eax, ebx
mov ecx, ebx
not ecx
and ecx, esi
or ecx, eax
mov eax, ebx
add ecx, 0x698098d8
add ecx, [ebp - 0x24]
add edx, ecx
rol edx, 0x7
add edx, ebx
and eax, edx
mov ecx, edx
not ecx
and ecx, edi
or ecx, eax
add ecx, 0x8b44f7af
add ecx, [ebp - 0x20]
add esi, ecx
rol esi, 0x0c
add esi, edx
mov ecx, esi
mov eax, esi
not ecx
and eax, edx
and ecx, ebx
or ecx, eax
mov eax, esi
add ecx, 0xffff5bb1
add ecx, [ebp - 0x1c]
add edi, ecx
ror edi, 0x0f
add edi, esi
mov ecx, edi
mov [ebp - 0x50], edi
not ecx
and eax, edi
and ecx, edx
or ecx, eax
mov eax, edi
add ecx, 0x895cd7be
add ecx, [ebp - 0x18]
add ebx, ecx
ror ebx, 0x0a
add ebx, edi
mov ecx, ebx
mov [ebp - 0x54], ebx
not ecx
and eax, ebx
and ecx, esi
or ecx, eax
mov eax, ebx
add ecx, 0x6b901122
add ecx, [ebp - 0x14]
add edx, ecx
rol edx, 0x7
add edx, ebx
mov ecx, edx
mov [ebp - 0x4c], edx
not ecx
and eax, edx
and ecx, edi
lea edi, [esi + 0xfd987193]
or ecx, eax
add ecx, [ebp - 0x10]
add edi, ecx
rol edi, 0x0c
add edi, edx
mov [ebp - 0x48], edi
mov esi, edi
not esi
mov eax, edi
and eax, edx
mov ecx, esi
and ecx, ebx
mov ebx, [ebp - 0x50]
or ecx, eax
add ebx, 0xa679438e
add ecx, [ebp - 0x0c]
mov eax, edi
add ebx, ecx
ror ebx, 0x0f
add ebx, edi
mov edi, [ebp - 0x54]
and eax, ebx
mov edx, ebx
not edx
add edi, 0x49b40821
mov ecx, edx
and esi, ebx
and ecx, [ebp - 0x4c]
or ecx, eax
add ecx, [ebp - 0x8]
add edi, ecx
mov ecx, [ebp - 0x48]
mov eax, ecx
ror edi, 0x0a
add edi, ebx
add ecx, 0xc040b340
and eax, edi
and edx, edi
or esi, eax
mov eax, [ebp - 0x4c]
add esi, [ebp - 0x40]
add eax, 0xf61e2562
add esi, eax
mov eax, ebx
rol esi, 0x5
add esi, edi
and eax, esi
or edx, eax
add edx, [ebp - 0x2c]
add edx, ecx
mov ecx, edi
not ecx
rol edx, 0x9
and ecx, esi
add edx, esi
mov eax, edx
and eax, edi
or ecx, eax
add ecx, 0x265e5a51
add ecx, [ebp - 0x18]
add ebx, ecx
mov ecx, esi
not ecx
rol ebx, 0x0e
and ecx, edx
add ebx, edx
mov eax, ebx
and eax, esi
or ecx, eax
mov eax, edx
add ecx, 0xe9b6c7aa
add ecx, [ebp - 0x44]
add edi, ecx
mov ecx, edx
not ecx
ror edi, 0x0c
and ecx, ebx
add edi, ebx
and eax, edi
or ecx, eax
add ecx, 0xd62f105d
add ecx, [ebp - 0x30]
add esi, ecx
mov ecx, ebx
rol esi, 0x5
not ecx
add esi, edi
and ecx, edi
mov eax, ebx
and eax, esi
or ecx, eax
add ecx, 0x2441453
add ecx, [ebp - 0x1c]
add edx, ecx
mov ecx, edi
not ecx
rol edx, 0x9
and ecx, esi
add edx, esi
mov eax, edx
and eax, edi
or ecx, eax
add ecx, 0xd8a1e681
add ecx, [ebp - 0x8]
add ebx, ecx
mov ecx, esi
not ecx
rol ebx, 0x0e
and ecx, edx
add ebx, edx
mov eax, ebx
and eax, esi
or ecx, eax
mov eax, edx
add ecx, 0xe7d3fbc8
add ecx, [ebp - 0x34]
add edi, ecx
mov ecx, edx
not ecx
ror edi, 0x0c
and ecx, ebx
add edi, ebx
and eax, edi
or ecx, eax
mov eax, ebx
add ecx, 0x21e1cde6
add ecx, [ebp - 0x20]
add esi, ecx
mov ecx, ebx
not ecx
rol esi, 0x5
and ecx, edi
add esi, edi
and eax, esi
or ecx, eax
add ecx, 0xc33707d6
add ecx, [ebp - 0x0c]
add edx, ecx
mov ecx, edi
not ecx
rol edx, 0x9
and ecx, esi
add edx, esi
mov eax, edx
mov [ebp - 0x48], edx
and eax, edi
or ecx, eax
add ecx, 0xf4d50d87
add ecx, [ebp - 0x38]
add ebx, ecx
mov ecx, esi
not ecx
rol ebx, 0x0e
add ebx, edx
and ecx, edx
mov eax, ebx
and eax, esi
or ecx, eax
mov eax, edx
add ecx, 0x455a14ed
add ecx, [ebp - 0x24]
add edi, ecx
mov ecx, edx
ror edi, 0x0c
not ecx
add edi, ebx
and ecx, ebx
mov [ebp - 0x54], edi
and eax, edi
or ecx, eax
lea edx, [esi + 0xa9e3e905]
add ecx, [ebp - 0x10]
mov eax, ebx
add edx, ecx
mov esi, [ebp - 0x48]
mov ecx, ebx
rol edx, 0x5
add edx, edi
not ecx
and eax, edx
and ecx, edi
or ecx, eax
add esi, 0xfcefa3f8
add ecx, [ebp - 0x3c]
add esi, ecx
mov ecx, edi
not ecx
rol esi, 0x9
and ecx, edx
add esi, edx
mov eax, esi
and eax, edi
lea edi, [ebx + 0x676f02d9]
or ecx, eax
mov ebx, [ebp - 0x54]
add ecx, [ebp - 0x28]
add ebx, 0x8d2a4c8a
add edi, ecx
mov ecx, edx
rol edi, 0x0e
not ecx
add edi, esi
and ecx, esi
mov eax, edi
and eax, edx
or ecx, eax
mov eax, esi
add ecx, [ebp - 0x14]
xor eax, edi
add ebx, ecx
ror ebx, 0x0c
add ebx, edi
xor eax, ebx
add eax, 0xfffa3942
add eax, [ebp - 0x30]
add edx, eax
mov eax, edi
xor eax, ebx
rol edx, 0x4
add edx, ebx
xor eax, edx
add eax, 0x8771f681
add eax, [ebp - 0x24]
add esi, eax
rol esi, 0x0b
add esi, edx
mov eax, esi
mov ecx, esi
xor eax, ebx
xor eax, edx
add eax, 0x6d9d6122
add eax, [ebp - 0x18]
add edi, eax
rol edi, 0x10
add edi, esi
xor ecx, edi
mov eax, ecx
xor eax, edx
add eax, 0xfde5380c
add eax, [ebp - 0x0c]
add ebx, eax
mov eax, edi
ror ebx, 0x9
add ebx, edi
xor ecx, ebx
add ecx, 0xa4beea44
add ecx, [ebp - 0x40]
add edx, ecx
rol edx, 0x4
add edx, ebx
xor eax, ebx
xor eax, edx
add eax, 0x4bdecfa9
add eax, [ebp - 0x34]
add esi, eax
rol esi, 0x0b
add esi, edx
mov eax, esi
mov ecx, esi
xor eax, ebx
xor eax, edx
add eax, 0xf6bb4b60
add eax, [ebp - 0x28]
add edi, eax
rol edi, 0x10
add edi, esi
xor ecx, edi
mov eax, ecx
xor eax, edx
add eax, 0xbebfbc70
add eax, [ebp - 0x1c]
add ebx, eax
mov eax, edi
ror ebx, 0x9
add ebx, edi
xor eax, ebx
xor ecx, ebx
add ecx, 0x289b7ec6
add ecx, [ebp - 0x10]
add edx, ecx
rol edx, 0x4
add edx, ebx
xor eax, edx
mov [ebp - 0x48], edx
add eax, 0xeaa127fa
add eax, [ebp - 0x44]
add esi, eax
rol esi, 0x0b
add esi, edx
mov eax, esi
mov ecx, esi
xor eax, ebx
xor eax, edx
add eax, 0xd4ef3085
add eax, [ebp - 0x38]
add edi, eax
mov eax, [ebp - 0x48]
rol edi, 0x10
add eax, 0xd9d4d039
add edi, esi
xor ecx, edi
xor edx, ecx
add edx, 0x4881d05
add edx, [ebp - 0x2c]
add edx, ebx
ror edx, 0x9
add edx, edi
xor ecx, edx
add ecx, [ebp - 0x20]
add ecx, eax
mov eax, edi
xor eax, edx
rol ecx, 0x4
add ecx, edx
xor eax, ecx
add eax, 0xe6db99e5
add eax, [ebp - 0x14]
add esi, eax
rol esi, 0x0b
add esi, ecx
mov eax, esi
xor eax, edx
xor eax, ecx
add eax, 0x1fa27cf8
add eax, [ebp - 0x8]
add edi, eax
mov eax, esi
rol edi, 0x10
add edi, esi
xor eax, edi
xor eax, ecx
add eax, 0xc4ac5665
add eax, [ebp - 0x3c]
add edx, eax
mov eax, esi
not eax
ror edx, 0x9
add edx, edi
or eax, edx
xor eax, edi
add eax, 0xf4292244
add eax, [ebp - 0x44]
add ecx, eax
mov eax, edi
not eax
rol ecx, 0x6
add ecx, edx
or eax, ecx
xor eax, edx
add eax, 0x432aff97
add eax, [ebp - 0x28]
add esi, eax
mov eax, edx
not eax
rol esi, 0x0a
add esi, ecx
or eax, esi
xor eax, ecx
add eax, 0xab9423a7
add eax, [ebp - 0x0c]
add edi, eax
mov eax, ecx
not eax
rol edi, 0x0f
add edi, esi
or eax, edi
xor eax, esi
add eax, 0xfc93a039
add eax, [ebp - 0x30]
add edx, eax
mov eax, esi
not eax
ror edx, 0x0b
add edx, edi
or eax, edx
xor eax, edi
add eax, 0x655b59c3
add eax, [ebp - 0x14]
add ecx, eax
mov eax, edi
not eax
rol ecx, 0x6
add ecx, edx
or eax, ecx
xor eax, edx
add eax, 0x8f0ccc92
add eax, [ebp - 0x38]
add esi, eax
mov eax, edx
not eax
rol esi, 0x0a
add esi, ecx
or eax, esi
xor eax, ecx
add eax, 0xffeff47d
add eax, [ebp - 0x1c]
add edi, eax
mov eax, ecx
not eax
rol edi, 0x0f
add edi, esi
or eax, edi
xor eax, esi
add eax, 0x85845dd1
add eax, [ebp - 0x40]
add edx, eax
mov eax, esi
ror edx, 0x0b
not eax
add edx, edi
or eax, edx
xor eax, edi
add eax, 0x6fa87e4f
lea ebx, [esi + 0xfe2ce6e0]
add eax, [ebp - 0x24]
add ecx, eax
mov eax, edi
not eax
rol ecx, 0x6
add ecx, edx
or eax, ecx
xor eax, edx
add eax, [ebp - 0x8]
add ebx, eax
lea esi, [ecx + 0xf7537e82]
mov eax, edx
rol ebx, 0x0a
not eax
add ebx, ecx
or eax, ebx
xor eax, ecx
add eax, 0xa3014314
add eax, [ebp - 0x2c]
add edi, eax
mov eax, ecx
not eax
rol edi, 0x0f
add edi, ebx
or eax, edi
xor eax, ebx
add eax, 0x4e0811a1
add eax, [ebp - 0x10]
lea ecx, [edi + 0x2ad7d2bb]
add edx, eax
mov eax, ebx
not eax
ror edx, 0x0b
add edx, edi
or eax, edx
mov [ebp - 0x48], edx
xor eax, edi
add eax, [ebp - 0x34]
add esi, eax
mov eax, edi
not eax
mov edi, [ebp - 0x5c]
rol esi, 0x6
add esi, edx
or eax, esi
xor eax, edx
lea edx, [ebx + 0xbd3af235]
add eax, [ebp - 0x18]
mov ebx, [ebp - 0x48]
add edx, eax
rol edx, 0x0a
mov eax, ebx
not eax
add edx, esi
or eax, edx
xor eax, esi
add eax, [ebp - 0x3c]
add ecx, eax
mov eax, [ebp - 0x58]
add eax, esi
rol ecx, 0x0f
add ecx, edx
mov [edi], eax
add [edi + 0x8], ecx
lea eax, [ebx + 0xeb86d391]
not esi
or esi, ecx
xor esi, edx
jne .hash
ret

hash2:
push ebp
mov ebp, esp
sub esp, 0x70
push esi
mov esi, ecx
mov [ebp - 0x8], esi
mov ecx, [ebp + 0x8]
push ebx
push edi
nop dword [eax + eax]
.loop:
mov eax, [esi + 0x2c]
sub eax, [esi + 0x28]
sar eax, 0x2
test eax, eax
je .skip
shl eax, 0x2
push eax
push ecx
push dword [esi + 0x28]
call [0x1122]
.skip:
add esp, 0x0c
mov eax, [esi + 0x28]
mov edx, [esi + 0x34]
mov [ebp - 0x38], edx
mov edi, [eax]
mov ebx, [edx + 0x0c]
mov esi, ebx
mov ecx, [edx]
mov edx, [edx + 0x10]
mov [ebp - 0x44], edi
mov edi, [ebp - 0x38]
xor esi, [edi + 0x8]
xor esi, [edi + 0x4]
add esi, [eax]
mov eax, [eax + 0x14]
add esi, ecx
mov [ebp - 0x10], eax
add ecx, 0x50a28be6
rol esi, 0x0b
mov eax, ebx
not eax
mov edi, [edi + 0x8]
add esi, edx
rol edi, 0x0a
mov [ebp - 0x50], esi
mov esi, [ebp - 0x38]
or eax, [esi + 0x8]
xor eax, [esi + 0x4]
add eax, [ebp - 0x10]
add eax, ecx
rol eax, 0x8
add eax, edx
mov [ebp - 0x5c], eax
mov eax, esi
mov esi, [ebp - 0x8]
mov ecx, [eax + 0x8]
mov eax, [esi + 0x28]
mov esi, [ebp - 0x38]
rol ecx, 0x0a
mov [ebp - 0x70], ecx
mov eax, [eax + 0x4]
mov [ebp - 0x1c], eax
mov eax, edi
xor eax, [esi + 0x4]
xor eax, [ebp - 0x50]
add eax, [ebp - 0x1c]
mov ecx, [ebp - 0x8]
add eax, edx
rol eax, 0x0e
add eax, ebx
mov [ebp - 0x4c], eax
mov eax, esi
mov esi, [eax + 0x4]
mov eax, [ecx + 0x28]
mov ecx, [ebp - 0x38]
rol esi, 0x0a
mov eax, [eax + 0x38]
mov [ebp - 0x28], eax
mov eax, [ebp - 0x70]
not eax
or eax, [ecx + 0x4]
xor eax, [ebp - 0x5c]
add eax, 0x50a28be6
add eax, [ebp - 0x28]
add eax, edx
rol eax, 0x9
add eax, ebx
mov [ebp - 0x58], eax
mov eax, ecx
mov ecx, [ebp - 0x8]
mov edx, [eax + 0x4]
mov eax, [ecx + 0x28]
rol edx, 0x0a
mov [ebp - 0x6c], edx
mov eax, [eax + 0x8]
mov [ebp - 0x20], eax
mov eax, [ebp - 0x4c]
xor eax, esi
xor eax, [ebp - 0x50]
add eax, [ebp - 0x20]
add eax, ebx
rol eax, 0x0f
add eax, edi
rol dword [ebp - 0x50], 0x0a
mov [ebp - 0x54], eax
not edx
mov eax, [ecx + 0x28]
or edx, [ebp - 0x5c]
xor edx, [ebp - 0x58]
add edx, 0x50a28be6
mov ecx, [ebp - 0x70]
mov eax, [eax + 0x1c]
add edx, eax
mov [ebp - 0x18], eax
add ebx, edx
mov eax, [ebp - 0x5c]
mov edx, [ebp - 0x8]
rol eax, 0x0a
mov [ebp - 0x5c], eax
rol ebx, 0x9
mov eax, [edx + 0x28]
add ebx, ecx
mov edx, [eax + 0x0c]
mov eax, [ebp - 0x4c]
mov [ebp - 0x4], eax
mov eax, [ebp - 0x54]
xor [ebp - 0x4], eax
mov eax, [ebp - 0x50]
xor [ebp - 0x4], eax
mov eax, edx
add [ebp - 0x4], eax
add [ebp - 0x4], edi
mov edi, [ebp - 0x4]
rol dword [ebp - 0x4c], 0x0a
mov eax, [ebp - 0x5c]
rol edi, 0x0c
not eax
or eax, [ebp - 0x58]
add edi, esi
mov [ebp - 0x30], edx
xor eax, ebx
mov edx, [ebp - 0x8]
add eax, [ebp - 0x44]
mov [ebp - 0x4], edi
lea edi, [ecx + 0x50a28be6]
add edi, eax
mov eax, [ebp - 0x58]
mov ecx, [edx + 0x28]
rol eax, 0x0a
mov [ebp - 0x58], eax
not eax
or eax, ebx
rol edi, 0x0b
mov ecx, [ecx + 0x10]
add edi, [ebp - 0x6c]
mov [ebp - 0x0c], ecx
xor eax, edi
mov ecx, [ebp - 0x4c]
xor ecx, [ebp - 0x54]
xor ecx, [ebp - 0x4]
add ecx, [ebp - 0x0c]
add ecx, esi
rol ebx, 0x0a
mov esi, [ebp - 0x54]
rol ecx, 0x5
add ecx, [ebp - 0x50]
mov [ebp - 0x34], ecx
mov ecx, [edx + 0x28]
mov edx, [ebp - 0x5c]
rol esi, 0x0a
mov [ebp - 0x54], esi
mov ecx, [ecx + 0x24]
add eax, ecx
mov [ebp - 0x48], ecx
mov ecx, [ebp - 0x6c]
add ecx, 0x50a28be6
add ecx, eax
mov eax, [ebp - 0x4]
xor esi, eax
rol ecx, 0x0d
xor esi, [ebp - 0x34]
add ecx, edx
add esi, [ebp - 0x10]
add esi, [ebp - 0x50]
rol eax, 0x0a
mov [ebp - 0x4], eax
mov eax, ebx
rol esi, 0x8
not eax
add esi, [ebp - 0x4c]
or eax, edi
mov [ebp - 0x5c], esi
xor eax, ecx
add eax, [ebp - 0x20]
lea esi, [edx + 0x50a28be6]
mov edx, [ebp - 0x8]
add esi, eax
rol esi, 0x0f
add esi, [ebp - 0x58]
rol edi, 0x0a
mov eax, [edx + 0x28]
mov [ebp - 0x60], esi
mov eax, [eax + 0x18]
mov [ebp - 0x2c], eax
mov eax, [ebp - 0x4]
mov [ebp - 0x3c], eax
mov eax, [ebp - 0x34]
mov edx, [ebp - 0x3c]
xor edx, eax
rol eax, 0x0a
xor edx, [ebp - 0x5c]
add edx, [ebp - 0x2c]
add edx, [ebp - 0x4c]
rol edx, 0x7
add edx, [ebp - 0x54]
mov [ebp - 0x3c], edx
mov edx, [ebp - 0x8]
mov [ebp - 0x34], eax
mov eax, [edx + 0x28]
mov edx, [eax + 0x2c]
mov eax, edi
not eax
mov [ebp - 0x50], edx
or eax, ecx
rol ecx, 0x0a
xor eax, esi
add eax, edx
mov edx, [ebp - 0x58]
add edx, 0x50a28be6
add edx, eax
mov eax, [ebp - 0x3c]
xor eax, [ebp - 0x34]
mov [ebp - 0x40], eax
mov eax, [ebp - 0x5c]
mov esi, [ebp - 0x40]
xor esi, eax
rol edx, 0x0f
add esi, [ebp - 0x18]
add edx, ebx
add esi, [ebp - 0x54]
rol esi, 0x9
add esi, [ebp - 0x4]
mov [ebp - 0x40], esi
mov esi, ecx
rol eax, 0x0a
not esi
or esi, [ebp - 0x60]
mov [ebp - 0x5c], eax
xor esi, edx
mov eax, [ebp - 0x60]
add esi, 0x50a28be6
add esi, [ebp - 0x0c]
rol eax, 0x0a
add esi, ebx
mov [ebp - 0x60], eax
mov eax, [ebp - 0x8]
rol esi, 0x5
add esi, edi
mov ebx, [eax + 0x28]
mov ebx, [ebx + 0x20]
mov [ebp - 0x24], ebx
mov ebx, [ebp - 0x3c]
xor ebx, [ebp - 0x40]
xor ebx, [ebp - 0x5c]
add ebx, [ebp - 0x24]
add ebx, [ebp - 0x4]
rol dword [ebp - 0x3c], 0x0a
rol ebx, 0x0b
add ebx, [ebp - 0x34]
mov [ebp - 0x54], ebx
mov ebx, [eax + 0x28]
mov eax, [ebp - 0x60]
not eax
or eax, edx
rol edx, 0x0a
mov ebx, [ebx + 0x34]
xor eax, esi
mov [ebp - 0x4c], ebx
add ebx, 0x50a28be6
add eax, ebx
lea ebx, [ecx + 0x50a28be6]
add edi, eax
mov eax, [ebp - 0x3c]
xor eax, [ebp - 0x40]
xor eax, [ebp - 0x54]
add eax, [ebp - 0x48]
add eax, [ebp - 0x34]
rol eax, 0x0d
add eax, [ebp - 0x5c]
mov [ebp - 0x58], eax
mov eax, edx
not eax
rol dword [ebp - 0x40], 0x0a
or eax, esi
rol edi, 0x7
add edi, ecx
rol esi, 0x0a
mov ecx, [ebp - 0x8]
xor eax, edi
add eax, [ebp - 0x2c]
add ebx, eax
rol ebx, 0x7
mov eax, [ecx + 0x28]
add ebx, [ebp - 0x60]
mov eax, [eax + 0x28]
mov [ebp - 0x14], eax
mov eax, [ebp - 0x40]
xor eax, [ebp - 0x54]
xor eax, [ebp - 0x58]
add eax, [ebp - 0x14]
add eax, [ebp - 0x5c]
rol eax, 0x0e
add eax, [ebp - 0x3c]
rol dword [ebp - 0x54], 0x0a
mov [ebp - 0x5c], eax
mov eax, [ecx + 0x28]
mov ecx, [ebp - 0x60]
add ecx, 0x50a28be6
mov eax, [eax + 0x3c]
mov [ebp - 0x34], eax
mov eax, esi
not eax
or eax, edi
rol edi, 0x0a
xor eax, ebx
mov [ebp - 0x70], edi
add eax, [ebp - 0x34]
not edi
add ecx, eax
or edi, ebx
mov eax, [ebp - 0x54]
xor eax, [ebp - 0x58]
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x50]
add eax, [ebp - 0x3c]
rol dword [ebp - 0x58], 0x0a
rol ecx, 0x8
add ecx, edx
rol eax, 0x0f
add eax, [ebp - 0x40]
xor edi, ecx
add edi, 0x50a28be6
mov [ebp - 0x60], eax
add edi, [ebp - 0x24]
add edx, edi
mov edi, [ebp - 0x8]
rol ebx, 0x0a
rol edx, 0x0b
add edx, esi
mov eax, [edi + 0x28]
mov eax, [eax + 0x30]
mov [ebp - 0x4], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x58]
mov [ebp - 0x3c], eax
mov edi, [ebp - 0x3c]
mov eax, [ebp - 0x5c]
xor edi, eax
add edi, [ebp - 0x4]
add edi, [ebp - 0x40]
rol eax, 0x0a
mov [ebp - 0x5c], eax
mov eax, ebx
not eax
rol edi, 0x6
add edi, [ebp - 0x54]
or eax, ecx
xor eax, edx
mov [ebp - 0x3c], edi
add eax, 0x50a28be6
rol ecx, 0x0a
add eax, [ebp - 0x1c]
add esi, eax
mov [ebp - 0x6c], ecx
mov eax, [ebp - 0x60]
not ecx
mov edi, eax
rol esi, 0x0e
xor edi, [ebp - 0x3c]
or ecx, edx
xor edi, [ebp - 0x5c]
add edi, [ebp - 0x4c]
add edi, [ebp - 0x54]
add esi, [ebp - 0x70]
xor ecx, esi
rol edi, 0x7
add ecx, [ebp - 0x14]
add edi, [ebp - 0x58]
rol eax, 0x0a
mov [ebp - 0x54], edi
mov edi, [ebp - 0x70]
mov [ebp - 0x60], eax
add edi, 0x50a28be6
mov eax, [ebp - 0x3c]
add edi, ecx
mov ecx, [ebp - 0x60]
xor ecx, eax
rol edx, 0x0a
xor ecx, [ebp - 0x54]
add ecx, [ebp - 0x28]
add ecx, [ebp - 0x58]
rol eax, 0x0a
mov [ebp - 0x3c], eax
mov eax, edx
not eax
rol edi, 0x0e
or eax, esi
rol ecx, 0x9
add ecx, [ebp - 0x5c]
add edi, ebx
xor eax, edi
mov [ebp - 0x58], ecx
mov ecx, [ebp - 0x3c]
add eax, 0x50a28be6
add eax, [ebp - 0x30]
add ebx, eax
rol esi, 0x0a
mov eax, [ebp - 0x54]
xor ecx, eax
xor ecx, [ebp - 0x58]
add ecx, [ebp - 0x34]
add ecx, [ebp - 0x5c]
rol ebx, 0x0c
add ebx, [ebp - 0x6c]
mov [ebp - 0x70], esi
rol ecx, 0x8
rol eax, 0x0a
not esi
mov [ebp - 0x54], eax
or esi, edi
xor eax, [ebp - 0x58]
xor esi, ebx
add ecx, [ebp - 0x60]
add esi, [ebp - 0x4]
mov [ebp - 0x40], ecx
and eax, [ebp - 0x40]
xor eax, [ebp - 0x54]
add eax, [ebp - 0x18]
mov ecx, [ebp - 0x6c]
rol dword [ebp - 0x58], 0x0a
add ecx, 0x50a28be6
add ecx, esi
rol edi, 0x0a
mov esi, [ebp - 0x60]
add esi, 0x5a827999
rol ecx, 0x6
add eax, esi
add ecx, edx
rol eax, 0x7
add edx, 0x5c4dd124
add eax, [ebp - 0x3c]
mov [ebp - 0x68], eax
mov eax, ebx
xor eax, ecx
mov esi, [ebp - 0x3c]
and eax, edi
add esi, 0x5a827999
xor eax, ebx
rol ebx, 0x0a
add eax, [ebp - 0x2c]
add eax, edx
mov edx, [ebp - 0x68]
rol eax, 0x9
add eax, [ebp - 0x70]
mov [ebp - 0x6c], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x40]
and eax, edx
rol dword [ebp - 0x40], 0x0a
xor eax, [ebp - 0x58]
add eax, [ebp - 0x0c]
add eax, esi
mov esi, [ebp - 0x70]
rol eax, 0x6
add esi, 0x5c4dd124
add eax, [ebp - 0x54]
mov [ebp - 0x68], eax
mov eax, [ebp - 0x6c]
xor eax, ecx
and eax, ebx
xor eax, ecx
rol ecx, 0x0a
add eax, [ebp - 0x50]
add eax, esi
mov [ebp - 0x3c], ecx
mov esi, [ebp - 0x54]
rol eax, 0x0d
add esi, 0x5a827999
add eax, edi
mov [ebp - 0x70], eax
mov eax, edx
xor eax, [ebp - 0x40]
and eax, [ebp - 0x68]
xor eax, [ebp - 0x40]
add eax, [ebp - 0x4c]
add eax, esi
rol edx, 0x0a
mov esi, [ebp - 0x6c]
rol eax, 0x8
add eax, [ebp - 0x58]
mov [ebp - 0x64], eax
mov eax, esi
xor eax, [ebp - 0x70]
and eax, ecx
xor eax, esi
add eax, [ebp - 0x30]
add edi, 0x5c4dd124
add eax, edi
mov ecx, [ebp - 0x58]
rol eax, 0x0f
add ecx, 0x5a827999
add eax, ebx
mov edi, [ebp - 0x68]
mov [ebp - 0x60], eax
mov eax, edx
xor eax, edi
rol esi, 0x0a
and eax, [ebp - 0x64]
xor eax, edx
rol edi, 0x0a
add eax, [ebp - 0x1c]
add eax, ecx
mov ecx, [ebp - 0x40]
rol eax, 0x0d
add ecx, 0x5a827999
add eax, [ebp - 0x40]
mov [ebp - 0x54], eax
mov eax, [ebp - 0x70]
xor eax, [ebp - 0x60]
and eax, esi
xor eax, [ebp - 0x70]
add eax, 0x5c4dd124
add eax, [ebp - 0x18]
add eax, ebx
mov ebx, [ebp - 0x70]
rol eax, 0x7
add eax, [ebp - 0x3c]
mov [ebp - 0x5c], eax
mov eax, edi
xor eax, [ebp - 0x64]
and eax, [ebp - 0x54]
xor eax, edi
rol dword [ebp - 0x64], 0x0a
add eax, [ebp - 0x14]
add eax, ecx
rol ebx, 0x0a
rol eax, 0x0b
add eax, edx
mov ecx, [ebp - 0x3c]
mov [ebp - 0x40], eax
add ecx, 0x5c4dd124
mov eax, [ebp - 0x60]
add edx, 0x5a827999
xor eax, [ebp - 0x5c]
and eax, ebx
xor eax, [ebp - 0x60]
add eax, [ebp - 0x44]
add eax, ecx
rol dword [ebp - 0x60], 0x0a
mov ecx, [ebp - 0x64]
rol eax, 0x0c
add eax, esi
add esi, 0x5c4dd124
mov [ebp - 0x58], eax
mov eax, ecx
xor eax, [ebp - 0x54]
and eax, [ebp - 0x40]
xor eax, ecx
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x2c]
add eax, edx
mov edx, [ebp - 0x60]
rol eax, 0x9
add eax, edi
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
xor eax, [ebp - 0x58]
and eax, edx
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x4c]
rol dword [ebp - 0x5c], 0x0a
add eax, esi
mov esi, [ebp - 0x54]
rol eax, 0x8
add eax, ebx
mov [ebp - 0x60], eax
mov eax, esi
xor eax, [ebp - 0x40]
and eax, [ebp - 0x64]
add edi, 0x5a827999
xor eax, esi
rol dword [ebp - 0x40], 0x0a
add eax, [ebp - 0x34]
add ebx, 0x5c4dd124
add eax, edi
mov edi, [ebp - 0x5c]
rol eax, 0x7
add eax, ecx
add ecx, 0x5a827999
mov [ebp - 0x54], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x58]
and eax, edi
xor eax, [ebp - 0x58]
add eax, [ebp - 0x10]
add eax, ebx
rol dword [ebp - 0x58], 0x0a
rol eax, 0x9
add eax, edx
mov ebx, [ebp - 0x40]
mov [ebp - 0x5c], eax
add edx, 0x5c4dd124
mov eax, [ebp - 0x64]
xor eax, ebx
rol dword [ebp - 0x64], 0x0a
and eax, [ebp - 0x54]
xor eax, ebx
add eax, [ebp - 0x30]
add eax, ecx
mov ecx, [ebp - 0x58]
rol eax, 0x0f
add eax, esi
add esi, 0x5a827999
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x5c]
and eax, ecx
xor eax, [ebp - 0x60]
add eax, [ebp - 0x14]
add eax, edx
rol dword [ebp - 0x60], 0x0a
mov edx, [ebp - 0x64]
rol eax, 0x0b
add eax, edi
add edi, 0x5c4dd124
mov [ebp - 0x58], eax
mov eax, edx
xor eax, [ebp - 0x54]
and eax, [ebp - 0x40]
xor eax, edx
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x4]
add eax, esi
mov esi, [ebp - 0x60]
rol eax, 0x7
add eax, ebx
add ebx, 0x5a827999
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
xor eax, [ebp - 0x58]
and eax, esi
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x28]
add eax, edi
rol dword [ebp - 0x5c], 0x0a
mov edi, [ebp - 0x54]
rol eax, 0x7
add eax, ecx
mov [ebp - 0x60], eax
mov eax, edi
xor eax, [ebp - 0x40]
and eax, [ebp - 0x64]
xor eax, edi
add eax, [ebp - 0x44]
add eax, ebx
rol eax, 0x0c
add eax, edx
rol dword [ebp - 0x40], 0x0a
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
add ecx, 0x5c4dd124
xor eax, [ebp - 0x60]
add edx, 0x5a827999
mov ebx, [ebp - 0x5c]
and eax, ebx
xor eax, [ebp - 0x58]
add eax, [ebp - 0x34]
add eax, ecx
rol dword [ebp - 0x58], 0x0a
rol eax, 0x7
add eax, esi
mov ecx, [ebp - 0x40]
mov [ebp - 0x5c], eax
add esi, 0x5c4dd124
mov eax, ecx
xor eax, [ebp - 0x64]
and eax, [ebp - 0x54]
xor eax, ecx
rol dword [ebp - 0x64], 0x0a
add eax, [ebp - 0x48]
add eax, edx
mov edx, [ebp - 0x58]
rol eax, 0x0f
add eax, edi
add edi, 0x5a827999
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x5c]
and eax, edx
xor eax, [ebp - 0x60]
add eax, [ebp - 0x24]
add eax, esi
rol dword [ebp - 0x60], 0x0a
rol eax, 0x0c
add eax, ebx
mov esi, [ebp - 0x64]
mov [ebp - 0x58], eax
add ebx, 0x5c4dd124
mov eax, esi
xor eax, [ebp - 0x54]
and eax, [ebp - 0x40]
xor eax, esi
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x10]
add eax, edi
mov edi, [ebp - 0x60]
rol eax, 0x9
add eax, ecx
add ecx, 0x5a827999
mov [ebp - 0x64], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x5c]
and eax, edi
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x4]
add eax, ebx
rol dword [ebp - 0x5c], 0x0a
mov ebx, [ebp - 0x54]
rol eax, 0x7
add eax, edx
add edx, 0x5c4dd124
mov [ebp - 0x60], eax
mov eax, [ebp - 0x40]
xor eax, ebx
rol dword [ebp - 0x40], 0x0a
and eax, [ebp - 0x64]
xor eax, ebx
add eax, [ebp - 0x20]
add eax, ecx
mov ecx, [ebp - 0x5c]
rol eax, 0x0b
add eax, esi
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x60]
and eax, ecx
xor eax, [ebp - 0x58]
add eax, [ebp - 0x0c]
add eax, edx
rol eax, 0x6
add eax, edi
mov edx, [ebp - 0x40]
mov [ebp - 0x5c], eax
add esi, 0x5a827999
rol dword [ebp - 0x58], 0x0a
mov eax, edx
xor eax, [ebp - 0x64]
add edi, 0x5c4dd124
and eax, [ebp - 0x54]
xor eax, edx
rol dword [ebp - 0x64], 0x0a
add eax, [ebp - 0x28]
add eax, esi
mov esi, [ebp - 0x58]
rol eax, 0x7
add eax, ebx
add ebx, 0x5a827999
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x5c]
and eax, esi
xor eax, [ebp - 0x60]
add eax, [ebp - 0x48]
add eax, edi
rol dword [ebp - 0x60], 0x0a
rol eax, 0x0f
add eax, ecx
mov edi, [ebp - 0x64]
mov [ebp - 0x58], eax
add ecx, 0x5c4dd124
mov eax, edi
xor eax, [ebp - 0x54]
and eax, [ebp - 0x40]
xor eax, edi
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x50]
add eax, ebx
mov ebx, [ebp - 0x60]
rol eax, 0x0d
add eax, edx
add edx, 0x5a827999
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
xor eax, [ebp - 0x58]
and eax, ebx
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x1c]
add eax, ecx
rol dword [ebp - 0x5c], 0x0a
mov ecx, [ebp - 0x54]
rol eax, 0x0d
add eax, esi
add esi, 0x5c4dd124
mov [ebp - 0x60], eax
mov eax, ecx
xor eax, [ebp - 0x40]
and eax, [ebp - 0x64]
xor eax, ecx
rol dword [ebp - 0x40], 0x0a
add eax, [ebp - 0x24]
add eax, edx
mov edx, [ebp - 0x5c]
rol eax, 0x0c
add eax, edi
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x60]
and eax, edx
xor eax, [ebp - 0x58]
add eax, [ebp - 0x20]
rol dword [ebp - 0x58], 0x0a
add eax, esi
mov esi, [ebp - 0x40]
rol eax, 0x0b
add eax, ebx
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x64]
not eax
or eax, [ebp - 0x54]
xor eax, esi
add eax, [ebp - 0x30]
rol dword [ebp - 0x64], 0x0a
add edi, 0x6ed9eba1
add eax, edi
add ebx, 0x6d703ef3
rol eax, 0x0b
add eax, ecx
mov edi, [ebp - 0x58]
mov [ebp - 0x40], eax
add ecx, 0x6ed9eba1
mov eax, [ebp - 0x60]
not eax
rol dword [ebp - 0x60], 0x0a
or eax, [ebp - 0x5c]
xor eax, edi
add eax, [ebp - 0x34]
add eax, ebx
mov ebx, [ebp - 0x64]
rol eax, 0x9
add eax, edx
add edx, 0x6d703ef3
mov [ebp - 0x58], eax
mov eax, [ebp - 0x54]
not eax
rol dword [ebp - 0x54], 0x0a
or eax, [ebp - 0x40]
xor eax, ebx
add eax, [ebp - 0x14]
add eax, ecx
mov ecx, [ebp - 0x60]
rol eax, 0x0d
add eax, esi
add esi, 0x6ed9eba1
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
not eax
rol dword [ebp - 0x5c], 0x0a
or eax, [ebp - 0x58]
xor eax, ecx
add eax, [ebp - 0x10]
add eax, edx
mov edx, [ebp - 0x54]
rol eax, 0x7
add eax, edi
add edi, 0x6d703ef3
mov [ebp - 0x60], eax
mov eax, [ebp - 0x40]
not eax
rol dword [ebp - 0x40], 0x0a
or eax, [ebp - 0x64]
xor eax, edx
add eax, [ebp - 0x28]
add eax, esi
mov esi, [ebp - 0x5c]
rol eax, 0x6
add eax, ebx
add ebx, 0x6ed9eba1
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
not eax
rol dword [ebp - 0x58], 0x0a
or eax, [ebp - 0x60]
xor eax, esi
add eax, [ebp - 0x1c]
add eax, edi
mov edi, [ebp - 0x40]
rol eax, 0x0f
add eax, ecx
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x64]
rol dword [ebp - 0x64], 0x0a
not eax
or eax, [ebp - 0x54]
xor eax, edi
add eax, [ebp - 0x0c]
add eax, ebx
rol eax, 0x7
add eax, edx
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
not eax
or eax, [ebp - 0x5c]
mov ebx, [ebp - 0x58]
add ecx, 0x6d703ef3
xor eax, ebx
rol dword [ebp - 0x60], 0x0a
add eax, [ebp - 0x30]
add edx, 0x6ed9eba1
add eax, ecx
mov ecx, [ebp - 0x64]
rol eax, 0x0b
add eax, esi
add esi, 0x6d703ef3
mov [ebp - 0x58], eax
mov eax, [ebp - 0x54]
not eax
rol dword [ebp - 0x54], 0x0a
or eax, [ebp - 0x40]
xor eax, ecx
add eax, [ebp - 0x48]
add eax, edx
mov edx, [ebp - 0x60]
rol eax, 0x0e
add eax, edi
add edi, 0x6ed9eba1
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
not eax
rol dword [ebp - 0x5c], 0x0a
or eax, [ebp - 0x58]
xor eax, edx
add eax, [ebp - 0x18]
add eax, esi
mov esi, [ebp - 0x54]
rol eax, 0x8
add eax, ebx
add ebx, 0x6d703ef3
mov [ebp - 0x60], eax
mov eax, [ebp - 0x40]
not eax
rol dword [ebp - 0x40], 0x0a
or eax, [ebp - 0x64]
xor eax, esi
add eax, [ebp - 0x34]
add eax, edi
mov edi, [ebp - 0x5c]
rol eax, 0x9
add eax, ecx
add ecx, 0x6ed9eba1
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
not eax
rol dword [ebp - 0x58], 0x0a
or eax, [ebp - 0x60]
xor eax, edi
add eax, [ebp - 0x28]
add eax, ebx
mov ebx, [ebp - 0x40]
rol eax, 0x6
add eax, edx
add edx, 0x6d703ef3
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x64]
not eax
rol dword [ebp - 0x64], 0x0a
or eax, [ebp - 0x54]
xor eax, ebx
add eax, [ebp - 0x24]
add eax, ecx
mov ecx, [ebp - 0x58]
rol eax, 0x0d
add eax, esi
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
not eax
or eax, [ebp - 0x5c]
xor eax, ecx
add eax, [ebp - 0x2c]
add eax, edx
rol eax, 0x6
add eax, edi
rol dword [ebp - 0x60], 0x0a
mov [ebp - 0x58], eax
mov eax, [ebp - 0x54]
add esi, 0x6ed9eba1
not eax
mov edx, [ebp - 0x64]
or eax, [ebp - 0x40]
add edi, 0x6d703ef3
xor eax, edx
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x1c]
add eax, esi
mov esi, [ebp - 0x60]
rol eax, 0x0f
add eax, ebx
add ebx, 0x6ed9eba1
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
not eax
rol dword [ebp - 0x5c], 0x0a
or eax, [ebp - 0x58]
xor eax, esi
add eax, [ebp - 0x48]
add eax, edi
mov edi, [ebp - 0x54]
rol eax, 0x0e
add eax, ecx
add ecx, 0x6d703ef3
mov [ebp - 0x60], eax
mov eax, [ebp - 0x40]
not eax
rol dword [ebp - 0x40], 0x0a
or eax, [ebp - 0x64]
xor eax, edi
add eax, [ebp - 0x20]
add eax, ebx
mov ebx, [ebp - 0x5c]
rol eax, 0x0e
add eax, edx
add edx, 0x6ed9eba1
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
not eax
rol dword [ebp - 0x58], 0x0a
or eax, [ebp - 0x60]
xor eax, ebx
add eax, [ebp - 0x50]
add eax, ecx
mov ecx, [ebp - 0x40]
rol eax, 0x0c
add eax, esi
add esi, 0x6d703ef3
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x64]
not eax
rol dword [ebp - 0x64], 0x0a
or eax, [ebp - 0x54]
xor eax, ecx
add eax, [ebp - 0x18]
add eax, edx
mov edx, [ebp - 0x58]
rol eax, 0x8
add eax, edi
add edi, 0x6ed9eba1
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
not eax
rol dword [ebp - 0x60], 0x0a
or eax, [ebp - 0x5c]
xor eax, edx
add eax, [ebp - 0x24]
add eax, esi
mov esi, [ebp - 0x64]
rol eax, 0x0d
add eax, ebx
mov [ebp - 0x58], eax
mov eax, [ebp - 0x54]
not eax
or eax, [ebp - 0x40]
xor eax, esi
add eax, [ebp - 0x44]
add eax, edi
rol eax, 0x0d
add eax, ecx
mov edi, [ebp - 0x60]
mov [ebp - 0x64], eax
add ebx, 0x6d703ef3
mov eax, [ebp - 0x5c]
add ecx, 0x6ed9eba1
not eax
rol dword [ebp - 0x54], 0x0a
or eax, [ebp - 0x58]
xor eax, edi
rol dword [ebp - 0x5c], 0x0a
add eax, [ebp - 0x4]
add eax, ebx
mov ebx, [ebp - 0x54]
rol eax, 0x5
add eax, edx
add edx, 0x6d703ef3
mov [ebp - 0x60], eax
mov eax, [ebp - 0x40]
not eax
rol dword [ebp - 0x40], 0x0a
or eax, [ebp - 0x64]
xor eax, ebx
add eax, [ebp - 0x2c]
add eax, ecx
mov ecx, [ebp - 0x5c]
rol eax, 0x6
add eax, esi
add esi, 0x6ed9eba1
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
not eax
rol dword [ebp - 0x58], 0x0a
or eax, [ebp - 0x60]
xor eax, ecx
add eax, [ebp - 0x20]
add eax, edx
mov edx, [ebp - 0x40]
rol eax, 0x0e
add eax, edi
add edi, 0x6d703ef3
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x64]
not eax
rol dword [ebp - 0x64], 0x0a
or eax, [ebp - 0x54]
xor eax, edx
add eax, [ebp - 0x4c]
add eax, esi
mov esi, [ebp - 0x58]
rol eax, 0x5
add eax, ebx
add ebx, 0x6ed9eba1
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
not eax
rol dword [ebp - 0x60], 0x0a
or eax, [ebp - 0x5c]
xor eax, esi
add eax, [ebp - 0x14]
add eax, edi
mov edi, [ebp - 0x64]
rol eax, 0x0d
add eax, ecx
mov [ebp - 0x58], eax
mov eax, [ebp - 0x54]
not eax
rol dword [ebp - 0x54], 0x0a
or eax, [ebp - 0x40]
xor eax, edi
add eax, [ebp - 0x50]
add eax, ebx
mov ebx, [ebp - 0x60]
rol eax, 0x0c
add eax, edx
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
not eax
or eax, [ebp - 0x58]
xor eax, ebx
add eax, [ebp - 0x44]
rol dword [ebp - 0x5c], 0x0a
add ecx, 0x6d703ef3
add eax, ecx
add edx, 0x6ed9eba1
rol eax, 0x0d
add eax, esi
mov ecx, [ebp - 0x54]
mov [ebp - 0x3c], eax
add esi, 0x6d703ef3
mov eax, [ebp - 0x40]
not eax
rol dword [ebp - 0x40], 0x0a
or eax, [ebp - 0x64]
xor eax, ecx
add eax, [ebp - 0x10]
add eax, edx
mov edx, [ebp - 0x5c]
rol eax, 0x7
add eax, edi
add edi, 0x6ed9eba1
mov [ebp - 0x60], eax
mov eax, [ebp - 0x58]
not eax
rol dword [ebp - 0x58], 0x0a
or eax, [ebp - 0x3c]
xor eax, edx
add eax, [ebp - 0x0c]
add eax, esi
mov esi, [ebp - 0x40]
rol eax, 0x7
add eax, ebx
add ebx, 0x6d703ef3
mov [ebp - 0x54], eax
mov eax, [ebp - 0x64]
not eax
rol dword [ebp - 0x64], 0x0a
or eax, [ebp - 0x60]
xor eax, esi
add eax, [ebp - 0x4]
add eax, edi
mov edi, [ebp - 0x58]
rol eax, 0x5
add eax, ecx
add ecx, 0x8f1bbcdc
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x3c]
not eax
rol dword [ebp - 0x3c], 0x0a
or eax, [ebp - 0x54]
xor eax, edi
add eax, [ebp - 0x4c]
add eax, ebx
mov ebx, [ebp - 0x64]
rol eax, 0x5
add eax, edx
add edx, 0x7a6d76e9
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x5c]
and eax, ebx
xor eax, [ebp - 0x60]
add eax, [ebp - 0x1c]
add eax, ecx
rol dword [ebp - 0x60], 0x0a
mov ecx, [ebp - 0x3c]
rol eax, 0x0b
add eax, esi
mov [ebp - 0x58], eax
mov eax, [ebp - 0x54]
rol dword [ebp - 0x54], 0x0a
xor eax, ecx
and eax, [ebp - 0x40]
xor eax, ecx
add eax, [ebp - 0x24]
add eax, edx
mov edx, [ebp - 0x60]
rol eax, 0x0f
add eax, edi
mov [ebp - 0x64], eax
mov eax, [ebp - 0x5c]
xor eax, [ebp - 0x58]
and eax, edx
add esi, 0x8f1bbcdc
xor eax, [ebp - 0x5c]
add edi, 0x7a6d76e9
add eax, [ebp - 0x48]
add eax, esi
rol dword [ebp - 0x5c], 0x0a
rol eax, 0x0c
add eax, ebx
mov esi, [ebp - 0x54]
mov [ebp - 0x60], eax
add ebx, 0x8f1bbcdc
mov eax, esi
xor eax, [ebp - 0x40]
and eax, [ebp - 0x64]
xor eax, esi
rol dword [ebp - 0x40], 0x0a
add eax, [ebp - 0x2c]
add eax, edi
mov edi, [ebp - 0x5c]
rol eax, 0x5
add eax, ecx
add ecx, 0x7a6d76e9
mov [ebp - 0x54], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x60]
and eax, edi
xor eax, [ebp - 0x58]
add eax, [ebp - 0x50]
add eax, ebx
rol dword [ebp - 0x58], 0x0a
rol eax, 0x0e
add eax, edx
mov ebx, [ebp - 0x40]
mov [ebp - 0x5c], eax
add edx, 0x8f1bbcdc
mov eax, ebx
xor eax, [ebp - 0x64]
and eax, [ebp - 0x54]
xor eax, ebx
rol dword [ebp - 0x64], 0x0a
add eax, [ebp - 0x0c]
add eax, ecx
mov ecx, [ebp - 0x58]
rol eax, 0x8
add eax, esi
add esi, 0x7a6d76e9
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x5c]
and eax, ecx
xor eax, [ebp - 0x60]
add eax, [ebp - 0x14]
add eax, edx
rol dword [ebp - 0x60], 0x0a
mov edx, [ebp - 0x64]
rol eax, 0x0f
add eax, edi
add edi, 0x8f1bbcdc
mov [ebp - 0x58], eax
mov eax, edx
xor eax, [ebp - 0x54]
and eax, [ebp - 0x40]
xor eax, edx
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x1c]
add eax, esi
mov esi, [ebp - 0x60]
rol eax, 0x0b
add eax, ebx
mov [ebp - 0x64], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x5c]
and eax, esi
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x44]
add eax, edi
rol eax, 0x0e
add eax, ecx
rol dword [ebp - 0x5c], 0x0a
mov [ebp - 0x60], eax
mov edi, [ebp - 0x54]
add ebx, 0x7a6d76e9
mov eax, edi
add ecx, 0x8f1bbcdc
xor eax, [ebp - 0x40]
and eax, [ebp - 0x64]
xor eax, edi
rol dword [ebp - 0x40], 0x0a
add eax, [ebp - 0x30]
add eax, ebx
mov ebx, [ebp - 0x5c]
rol eax, 0x0e
add eax, edx
add edx, 0x7a6d76e9
mov [ebp - 0x3c], eax
mov eax, [ebp - 0x58]
xor eax, [ebp - 0x60]
and eax, ebx
xor eax, [ebp - 0x58]
add eax, [ebp - 0x24]
add eax, ecx
rol dword [ebp - 0x58], 0x0a
rol eax, 0x0f
add eax, esi
mov ecx, [ebp - 0x40]
mov [ebp - 0x5c], eax
add esi, 0x8f1bbcdc
mov eax, [ebp - 0x64]
xor eax, ecx
rol dword [ebp - 0x64], 0x0a
and eax, [ebp - 0x3c]
xor eax, ecx
add eax, [ebp - 0x50]
add eax, edx
mov edx, [ebp - 0x58]
rol eax, 0x0e
add eax, edi
add edi, 0x7a6d76e9
mov [ebp - 0x40], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x5c]
and eax, edx
xor eax, [ebp - 0x60]
add eax, [ebp - 0x4]
add eax, esi
rol dword [ebp - 0x60], 0x0a
mov esi, [ebp - 0x64]
rol eax, 0x9
add eax, ebx
add ebx, 0x8f1bbcdc
mov [ebp - 0x54], eax
mov eax, esi
xor eax, [ebp - 0x3c]
and eax, [ebp - 0x40]
xor eax, esi
rol dword [ebp - 0x3c], 0x0a
add eax, [ebp - 0x34]
add eax, edi
mov edi, [ebp - 0x60]
rol eax, 0x6
add eax, ecx
add ecx, 0x7a6d76e9
mov [ebp - 0x68], eax
mov eax, [ebp - 0x5c]
xor eax, [ebp - 0x54]
and eax, edi
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x0c]
rol dword [ebp - 0x5c], 0x0a
add eax, ebx
mov ebx, [ebp - 0x3c]
rol eax, 0x8
add eax, edx
mov [ebp - 0x58], eax
mov eax, ebx
xor eax, [ebp - 0x40]
and eax, [ebp - 0x68]
xor eax, ebx
add eax, [ebp - 0x44]
add eax, ecx
rol eax, 0x0e
add eax, esi
mov ecx, [ebp - 0x5c]
mov [ebp - 0x6c], eax
add edx, 0x8f1bbcdc
mov eax, [ebp - 0x54]
add esi, 0x7a6d76e9
xor eax, [ebp - 0x58]
and eax, ecx
rol dword [ebp - 0x40], 0x0a
xor eax, [ebp - 0x54]
add eax, [ebp - 0x4c]
add eax, edx
rol dword [ebp - 0x54], 0x0a
rol eax, 0x9
add eax, edi
mov edx, [ebp - 0x40]
mov [ebp - 0x60], eax
add edi, 0x8f1bbcdc
mov eax, edx
xor eax, [ebp - 0x68]
and eax, [ebp - 0x6c]
xor eax, edx
rol dword [ebp - 0x68], 0x0a
add eax, [ebp - 0x10]
add eax, esi
mov esi, [ebp - 0x54]
rol eax, 0x6
add eax, ebx
add ebx, 0x7a6d76e9
mov [ebp - 0x70], eax
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x58]
and eax, esi
xor eax, [ebp - 0x58]
add eax, [ebp - 0x30]
add eax, edi
rol dword [ebp - 0x58], 0x0a
mov edi, [ebp - 0x68]
rol eax, 0x0e
add eax, ecx
add ecx, 0x8f1bbcdc
mov [ebp - 0x64], eax
mov eax, edi
xor eax, [ebp - 0x6c]
and eax, [ebp - 0x70]
xor eax, edi
add eax, [ebp - 0x4]
add eax, ebx
mov ebx, [ebp - 0x6c]
rol eax, 0x9
add eax, edx
rol ebx, 0x0a
mov [ebp - 0x54], eax
add edx, 0x7a6d76e9
mov eax, [ebp - 0x60]
xor eax, [ebp - 0x64]
and eax, [ebp - 0x58]
xor eax, [ebp - 0x60]
add eax, [ebp - 0x18]
add eax, ecx
mov [ebp - 0x6c], ebx
mov ecx, [ebp - 0x60]
rol eax, 0x5
add eax, esi
rol ecx, 0x0a
mov [ebp - 0x5c], eax
mov eax, [ebp - 0x70]
xor eax, ebx
mov [ebp - 0x60], ecx
and eax, [ebp - 0x54]
xor eax, ebx
mov ebx, [ebp - 0x64]
add eax, [ebp - 0x20]
add eax, edx
mov edx, [ebp - 0x70]
rol eax, 0x0c
add eax, edi
rol edx, 0x0a
mov [ebp - 0x40], eax
mov eax, ebx
mov [ebp - 0x70], edx
xor eax, [ebp - 0x5c]
add esi, 0x8f1bbcdc
and eax, ecx
mov ecx, [ebp - 0x58]
xor eax, ebx
rol ebx, 0x0a
add eax, [ebp - 0x34]
add eax, esi
mov [ebp - 0x64], ebx
rol eax, 0x6
lea ebx, [edi + 0x7a6d76e9]
add eax, ecx
mov esi, [ebp - 0x6c]
mov [ebp - 0x3c], eax
mov eax, edx
xor eax, [ebp - 0x54]
and eax, [ebp - 0x40]
xor eax, edx
rol dword [ebp - 0x54], 0x0a
add eax, [ebp - 0x4c]
lea edx, [ecx + 0x8f1bbcdc]
add ebx, eax
mov edi, [ebp - 0x60]
mov eax, [ebp - 0x5c]
lea ecx, [esi + 0x7a6d76e9]
xor eax, [ebp - 0x3c]
and eax, [ebp - 0x64]
xor eax, [ebp - 0x5c]
add eax, [ebp - 0x28]
add edx, eax
rol dword [ebp - 0x5c], 0x0a
mov eax, [ebp - 0x54]
xor eax, [ebp - 0x40]
rol dword [ebp - 0x40], 0x0a
rol ebx, 0x9
add ebx, esi
rol edx, 0x8
and eax, ebx
mov esi, [ebp - 0x70]
xor eax, [ebp - 0x54]
add edx, edi
add eax, [ebp - 0x48]
add edi, 0x8f1bbcdc
add ecx, eax
mov [ebp - 0x6c], edx
mov eax, [ebp - 0x3c]
xor eax, edx
rol ecx, 0x0c
and eax, [ebp - 0x5c]
add ecx, esi
xor eax, [ebp - 0x3c]
add esi, 0x7a6d76e9
add eax, [ebp - 0x10]
rol dword [ebp - 0x3c], 0x0a
add edi, eax
mov eax, [ebp - 0x40]
xor eax, ebx
rol edi, 0x6
add edi, [ebp - 0x64]
and eax, ecx
xor eax, [ebp - 0x40]
add eax, [ebp - 0x18]
add esi, eax
rol ebx, 0x0a
mov eax, edi
rol esi, 0x5
add esi, [ebp - 0x54]
xor eax, edx
and eax, [ebp - 0x3c]
xor eax, edx
mov [ebp - 0x70], edi
add eax, [ebp - 0x2c]
mov edx, [ebp - 0x64]
add edx, 0x8f1bbcdc
mov [ebp - 0x60], ebx
add edx, eax
mov eax, ebx
rol edx, 0x5
add edx, [ebp - 0x5c]
rol dword [ebp - 0x6c], 0x0a
mov [ebp - 0x68], edx
xor eax, ecx
rol dword [ebp - 0x70], 0x0a
and eax, esi
rol ecx, 0x0a
xor eax, ebx
mov [ebp - 0x58], ecx
add eax, [ebp - 0x14]
mov ebx, [ebp - 0x54]
add ebx, 0x7a6d76e9
add ebx, eax
mov eax, edi
xor eax, edx
rol ebx, 0x0f
and eax, [ebp - 0x6c]
xor eax, edi
add ebx, [ebp - 0x40]
add eax, [ebp - 0x20]
mov edi, [ebp - 0x5c]
add edi, 0x8f1bbcdc
add edi, eax
mov eax, esi
xor eax, ecx
rol esi, 0x0a
and eax, ebx
mov [ebp - 0x5c], esi
xor eax, ecx
rol edi, 0x0c
add eax, [ebp - 0x28]
xor esi, ebx
mov ecx, [ebp - 0x40]
add edi, [ebp - 0x3c]
add ecx, 0x7a6d76e9
add ecx, eax
rol ebx, 0x0a
mov eax, [ebp - 0x70]
not eax
rol ecx, 0x8
add ecx, [ebp - 0x60]
or eax, edx
mov edx, [ebp - 0x3c]
xor eax, edi
add eax, [ebp - 0x0c]
xor esi, ecx
add esi, [ebp - 0x4]
add edx, 0xa953fd4e
add esi, [ebp - 0x60]
add edx, eax
mov eax, [ebp - 0x68]
rol eax, 0x0a
mov [ebp - 0x68], eax
not eax
or eax, edi
rol esi, 0x8
add esi, [ebp - 0x58]
mov [ebp - 0x64], esi
mov esi, [ebp - 0x6c]
rol edx, 0x9
add esi, 0xa953fd4e
add edx, [ebp - 0x6c]
xor eax, edx
rol edi, 0x0a
add eax, [ebp - 0x44]
add esi, eax
mov [ebp - 0x40], ebx
mov eax, ebx
rol esi, 0x0f
add esi, [ebp - 0x70]
xor eax, ecx
xor eax, [ebp - 0x64]
add eax, [ebp - 0x34]
add eax, [ebp - 0x58]
rol eax, 0x5
add eax, [ebp - 0x5c]
mov [ebp - 0x60], eax
mov eax, edi
not eax
rol ecx, 0x0a
or eax, edx
mov [ebp - 0x54], edi
xor eax, esi
add eax, [ebp - 0x10]
mov edi, [ebp - 0x70]
rol edx, 0x0a
add edi, 0xa953fd4e
add edi, eax
mov [ebp - 0x6c], edx
mov eax, [ebp - 0x64]
mov edx, ecx
xor edx, eax
rol edi, 0x5
xor edx, [ebp - 0x60]
add edx, [ebp - 0x14]
add edx, [ebp - 0x5c]
add edi, [ebp - 0x68]
rol eax, 0x0a
mov [ebp - 0x64], eax
mov eax, [ebp - 0x6c]
rol edx, 0x0c
not eax
add edx, ebx
or eax, esi
mov ebx, [ebp - 0x68]
xor eax, edi
add eax, [ebp - 0x48]
add ebx, 0xa953fd4e
mov [ebp - 0x5c], edx
add ebx, eax
xor edx, [ebp - 0x64]
mov eax, [ebp - 0x60]
xor edx, eax
add edx, [ebp - 0x0c]
add edx, [ebp - 0x40]
rol eax, 0x0a
rol edx, 0x9
add edx, ecx
mov [ebp - 0x60], eax
mov [ebp - 0x58], edx
mov edx, [ebp - 0x54]
rol esi, 0x0a
add edx, 0xa953fd4e
rol ebx, 0x0b
mov eax, esi
add ebx, [ebp - 0x54]
not eax
or eax, edi
rol edi, 0x0a
xor eax, ebx
mov [ebp - 0x68], edi
add eax, [ebp - 0x18]
not edi
add eax, edx
or edi, ebx
rol eax, 0x6
add eax, [ebp - 0x6c]
mov [ebp - 0x70], eax
mov eax, [ebp - 0x5c]
mov edx, eax
xor edx, [ebp - 0x58]
xor edx, [ebp - 0x60]
add edx, [ebp - 0x1c]
add edx, ecx
rol eax, 0x0a
mov ecx, edx
mov [ebp - 0x40], edx
mov edx, [ebp - 0x6c]
rol ecx, 0x0c
add edx, 0xa953fd4e
add ecx, [ebp - 0x64]
mov [ebp - 0x40], ecx
mov ecx, [ebp - 0x70]
xor edi, ecx
add edi, [ebp - 0x4]
add edx, edi
mov [ebp - 0x5c], eax
mov edi, [ebp - 0x5c]
mov eax, [ebp - 0x58]
xor edi, eax
xor edi, [ebp - 0x40]
rol edx, 0x8
add edx, esi
rol ebx, 0x0a
add edi, [ebp - 0x10]
add edi, [ebp - 0x64]
rol eax, 0x0a
mov [ebp - 0x58], eax
mov eax, ebx
not eax
rol edi, 0x5
add edi, [ebp - 0x60]
or eax, ecx
xor eax, edx
mov [ebp - 0x64], edi
mov edi, [ebp - 0x58]
add eax, 0xa953fd4e
add eax, [ebp - 0x20]
add esi, eax
rol ecx, 0x0a
mov eax, [ebp - 0x40]
xor edi, eax
xor edi, [ebp - 0x64]
add edi, [ebp - 0x24]
add edi, [ebp - 0x60]
mov [ebp - 0x70], ecx
not ecx
or ecx, edx
rol eax, 0x0a
rol edi, 0x0e
add edi, [ebp - 0x5c]
rol esi, 0x0d
add esi, [ebp - 0x68]
xor ecx, esi
mov [ebp - 0x40], eax
add ecx, [ebp - 0x14]
mov eax, [ebp - 0x64]
mov [ebp - 0x60], edi
mov edi, [ebp - 0x68]
add edi, 0xa953fd4e
rol edx, 0x0a
add edi, ecx
mov ecx, [ebp - 0x40]
xor ecx, eax
rol edi, 0x0c
xor ecx, [ebp - 0x60]
add edi, ebx
add ecx, [ebp - 0x18]
add ecx, [ebp - 0x5c]
rol eax, 0x0a
mov [ebp - 0x64], eax
mov eax, edx
not eax
rol ecx, 0x6
add ecx, [ebp - 0x58]
or eax, esi
xor eax, edi
mov [ebp - 0x5c], ecx
add eax, 0xa953fd4e
rol esi, 0x0a
add eax, [ebp - 0x28]
add ebx, eax
mov [ebp - 0x6c], esi
mov eax, ecx
rol ebx, 0x5
xor eax, [ebp - 0x64]
not esi
add ebx, [ebp - 0x70]
or esi, edi
mov [ebp - 0x3c], eax
xor esi, ebx
mov eax, [ebp - 0x60]
mov ecx, [ebp - 0x3c]
add esi, [ebp - 0x1c]
xor ecx, eax
add ecx, [ebp - 0x2c]
add ecx, [ebp - 0x58]
rol ecx, 0x8
add ecx, [ebp - 0x40]
mov [ebp - 0x3c], ecx
mov ecx, [ebp - 0x70]
add ecx, 0xa953fd4e
rol eax, 0x0a
add ecx, esi
mov [ebp - 0x60], eax
rol ecx, 0x0c
mov eax, [ebp - 0x5c]
add ecx, edx
mov esi, eax
rol edi, 0x0a
xor esi, [ebp - 0x3c]
xor esi, [ebp - 0x60]
add esi, [ebp - 0x20]
add esi, [ebp - 0x40]
rol eax, 0x0a
mov [ebp - 0x5c], eax
mov eax, edi
not eax
rol esi, 0x0d
add esi, [ebp - 0x64]
or eax, ebx
xor eax, ecx
mov [ebp - 0x70], esi
mov esi, [ebp - 0x5c]
add eax, 0xa953fd4e
add eax, [ebp - 0x30]
add edx, eax
rol ebx, 0x0a
mov eax, [ebp - 0x3c]
xor esi, eax
xor esi, [ebp - 0x70]
add esi, [ebp - 0x4c]
add esi, [ebp - 0x64]
rol eax, 0x0a
mov [ebp - 0x3c], eax
mov eax, ebx
rol esi, 0x6
not eax
add esi, [ebp - 0x60]
or eax, ecx
mov [ebp - 0x64], esi
mov esi, [ebp - 0x6c]
rol edx, 0x0d
add esi, 0xa953fd4e
add edx, [ebp - 0x6c]
xor eax, edx
mov [ebp - 0x54], ebx
add eax, [ebp - 0x24]
mov ebx, [ebp - 0x3c]
add eax, esi
mov esi, [ebp - 0x70]
xor ebx, esi
xor ebx, [ebp - 0x64]
add ebx, [ebp - 0x28]
add ebx, [ebp - 0x60]
rol ecx, 0x0a
mov [ebp - 0x58], ecx
not ecx
rol esi, 0x0a
or ecx, edx
mov [ebp - 0x68], edx
lea edx, [edi + 0xa953fd4e]
rol eax, 0x0e
add eax, edi
mov [ebp - 0x70], esi
mov edi, [ebp - 0x64]
xor ecx, eax
add ecx, [ebp - 0x50]
xor esi, edi
rol ebx, 0x5
add edx, ecx
add ebx, [ebp - 0x5c]
mov ecx, [ebp - 0x54]
xor esi, ebx
add esi, [ebp - 0x44]
add esi, [ebp - 0x5c]
mov [ebp - 0x6c], eax
mov eax, [ebp - 0x68]
rol edx, 0x0b
rol esi, 0x0f
add edx, ecx
add esi, [ebp - 0x3c]
rol eax, 0x0a
rol edi, 0x0a
mov [ebp - 0x68], eax
mov [ebp - 0x60], esi
mov [ebp - 0x64], edi
not eax
rol dword [ebp - 0x60], 0x0a
or eax, [ebp - 0x6c]
add ecx, 0xa953fd4e
xor eax, edx
mov edi, esi
add eax, [ebp - 0x2c]
add ecx, eax
xor edi, [ebp - 0x64]
mov eax, [ebp - 0x6c]
xor edi, ebx
add edi, [ebp - 0x30]
add edi, [ebp - 0x3c]
rol eax, 0x0a
mov [ebp - 0x6c], eax
not eax
or eax, edx
rol ecx, 0x8
add ecx, [ebp - 0x58]
xor eax, ecx
rol ebx, 0x0a
add eax, [ebp - 0x34]
mov [ebp - 0x5c], ebx
mov ebx, [ebp - 0x58]
rol edi, 0x0d
add ebx, 0xa953fd4e
add edi, [ebp - 0x70]
add ebx, eax
xor esi, edi
rol edx, 0x0a
xor esi, [ebp - 0x5c]
mov eax, edx
add esi, [ebp - 0x48]
not eax
add esi, [ebp - 0x70]
or eax, ecx
rol ebx, 0x5
add ebx, [ebp - 0x68]
xor eax, ebx
mov [ebp - 0x58], edx
add eax, [ebp - 0x4c]
mov edx, [ebp - 0x68]
rol esi, 0x0b
add edx, 0xa953fd4e
add esi, [ebp - 0x64]
add edx, eax
mov [ebp - 0x70], esi
mov eax, edi
mov esi, [ebp - 0x38]
rol eax, 0x0a
rol ecx, 0x0a
rol edx, 0x6
add eax, [esi + 0x4]
add ebx, eax
mov eax, esi
mov esi, [ebp - 0x8]
add ecx, [eax + 0x8]
add ecx, [ebp - 0x60]
mov [eax + 0x4], ecx
mov ecx, [esi + 0x34]
mov eax, [ecx + 0x0c]
add eax, [ebp - 0x58]
add eax, [ebp - 0x5c]
mov [ecx + 0x8], eax
mov ecx, [ebp - 0x60]
mov eax, [esi + 0x34]
xor ecx, edi
xor ecx, [ebp - 0x70]
add ecx, [ebp - 0x50]
add ecx, [ebp - 0x64]
rol ecx, 0x0b
add ecx, [eax + 0x10]
add ecx, [ebp - 0x5c]
add ecx, [ebp - 0x6c]
mov [eax + 0x0c], ecx
mov eax, esi
mov ecx, [eax + 0x34]
mov eax, [ecx]
add eax, edx
add eax, [ebp - 0x70]
add eax, [ebp - 0x6c]
mov [ecx + 0x10], eax
mov eax, [esi + 0x34]
mov ecx, [ebp + 0x8]
mov [eax], ebx
mov eax, [esi + 0x8]
sub eax, [esi + 0x4]
add ecx, eax
sub dword [ebp + 0x0c], 0x1
mov [ebp + 0x8], ecx
jne .loop
pop edi
pop ebx
pop esi
mov esp, ebp
pop ebp
ret 0x8

hash3:
push ebp
mov ebp, esp
sub esp, 0x14
push ebx
push esi
push edi
push 0x0
push 0x1
mov [ebp - 0x4], ecx
call [0x2123]
mov ebx, [ebp + 0x10]
add esp, 0x8
mov esi, [ebp + 0x0c]
mov edi, [ebp + 0x8]
mov [ebp + 0x10], ebx
test al, al
je .skip
cmp ebx, 0x4
jb .skip
mov eax, ebx
shr eax, 0x2
mov [ebp + 0x10], eax
.back:
mov ecx, [ebp - 0x4]
push esi
push edi
call [0x12345]
add edi, 0x40
add esi, 0x40
sub ebx, 0x4
sub dword [ebp + 0x10], 0x1
jne .back
mov [ebp + 0x10], ebx
.skip:
test ebx, ebx
je .end
lea eax, [esi + 0x0c]
sub esi, edi
lea ebx, [edi + 0x8]
mov [ebp - 0x10], eax
mov [ebp - 0x0c], ebx
mov [ebp - 0x14], esi
nop word [eax + eax]
.loop:
mov eax, [ebp - 0x4]
mov edi, [ebx - 0x4]
mov esi, [ebp - 0x4]
mov edx, [eax + 0x4]
mov eax, [ebx - 0x8]
xor eax, [edx + 0x200]
xor edi, [edx + 0x204]
mov ecx, [edx + 0x208]
xor ecx, [ebx]
mov ebx, [ebx + 0x4]
xor ebx, [edx + 0x20c]
mov edx, ecx
xor ecx, eax
or edx, ebx
and eax, ebx
not ecx
xor ebx, edi
or edi, eax
xor eax, ecx
and ebx, edx
and ecx, edx
xor edi, ecx
xor ecx, eax
or eax, ecx
xor edx, edi
xor eax, ebx
mov [ebp + 0x8], eax
mov eax, edx
xor eax, ecx
xor ebx, eax
mov eax, [ebp + 0x8]
or edx, eax
xor edx, ecx
mov ecx, [esi + 0x4]
xor eax, [ecx + 0x1f4]
mov ecx, [ecx + 0x1fc]
xor ecx, edx
mov [ebp + 0x8], eax
mov edx, [esi + 0x4]
shl eax, 0x7
mov edx, [edx + 0x1f8]
xor edx, edi
mov edi, esi
rol edx, 0x0a
xor edx, eax
xor edx, ecx
mov eax, [edi + 0x4]
mov edi, [eax + 0x1f0]
xor edi, ebx
mov ebx, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
ror ebx, 0x1
xor edi, ecx
ror ecx, 0x7
xor ebx, edi
xor ebx, edx
lea eax, [edi*8]
ror edi, 0x0d
xor ecx, eax
xor ecx, edx
ror edx, 0x3
mov [ebp + 0x0c], ecx
xor edi, edx
mov eax, [ebp + 0x0c]
mov ecx, edx
xor ecx, eax
xor eax, ebx
mov [ebp + 0x0c], eax
or ecx, edi
mov eax, edi
and eax, edx
mov edx, [ebp + 0x0c]
not eax
xor eax, edx
xor edx, ecx
xor ecx, ebx
xor edi, eax
and ebx, edx
xor ebx, edi
mov [ebp + 0x8], ebx
xor edi, edx
xor edx, ebx
or edi, eax
mov [ebp + 0x0c], edx
xor ecx, edi
mov edx, esi
mov ebx, [edx + 0x4]
mov edx, [ebx + 0x1e4]
mov edi, [ebx + 0x1e8]
xor edx, eax
mov eax, [ebp + 0x0c]
xor edi, ecx
xor eax, [ebx + 0x1ec]
mov ebx, [ebx + 0x1e0]
xor ebx, [ebp + 0x8]
mov [ebp + 0x0c], eax
mov eax, edx
mov ecx, [ebp + 0x0c]
shl eax, 0x7
rol edi, 0x0a
xor edi, eax
ror ebx, 0x5
xor edi, ecx
mov [ebp - 0x8], edx
xor ebx, edx
xor ebx, ecx
ror ecx, 0x7
lea eax, [ebx*8]
xor ecx, eax
mov eax, edx
xor ecx, edi
ror eax, 0x1
xor eax, ebx
ror ebx, 0x0d
xor eax, edi
mov edx, ebx
or edx, ecx
ror edi, 0x3
not eax
xor edi, eax
xor edx, edi
xor ecx, edx
mov [ebp + 0x0c], ecx
mov ecx, eax
or ecx, edi
mov edi, [ebp + 0x0c]
and ecx, ebx
xor ecx, edi
or edi, ebx
xor edi, eax
and eax, ecx
xor edi, ecx
xor eax, edx
and edx, edi
mov [ebp - 0x8], eax
xor edi, eax
mov eax, edi
not edi
xor eax, ebx
mov ebx, esi
xor edx, eax
mov eax, [ebx + 0x4]
xor edi, [eax + 0x1d4]
mov ebx, [eax + 0x1d8]
mov [ebp + 0x0c], edi
xor ebx, edx
mov edi, [eax + 0x1dc]
mov eax, [ebp + 0x0c]
xor edi, ecx
mov ecx, esi
rol ebx, 0x0a
shl eax, 0x7
xor ebx, eax
xor ebx, edi
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x1d0]
xor ecx, [ebp - 0x8]
ror ecx, 0x5
xor ecx, [ebp + 0x0c]
xor ecx, edi
ror edi, 0x7
lea eax, [ecx*8]
xor edi, eax
mov eax, [ebp + 0x0c]
xor edi, ebx
ror eax, 0x1
xor eax, ecx
ror ecx, 0x0d
xor eax, ebx
ror ebx, 0x3
mov edx, ebx
and edx, edi
xor edx, eax
or eax, edi
and eax, ecx
not ecx
xor ebx, eax
and eax, edx
xor ebx, edx
xor edi, ebx
mov [ebp - 0x8], ebx
xor eax, edi
mov [ebp + 0x0c], eax
mov eax, ecx
xor ecx, [ebp + 0x0c]
and eax, edi
xor eax, edx
mov [ebp + 0x8], ecx
xor eax, ecx
and ecx, edx
xor ecx, ebx
mov edx, esi
or ecx, eax
xor ecx, [ebp + 0x0c]
mov edi, [edx + 0x4]
mov ebx, [edi + 0x1c4]
mov edx, [edi + 0x1cc]
mov edi, [edi + 0x1c8]
xor ebx, [ebp + 0x8]
xor edi, ecx
xor edx, [ebp - 0x8]
xor ebx, eax
rol edi, 0x0a
mov eax, ebx
shl eax, 0x7
mov ecx, esi
xor edi, eax
xor edi, edx
mov [ebp + 0x0c], edi
mov eax, [ecx + 0x4]
mov edi, [eax + 0x1c0]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
ror ebx, 0x1
xor edi, edx
ror edx, 0x7
xor ebx, edi
lea eax, [edi*8]
ror edi, 0x0d
xor edx, eax
mov eax, [ebp + 0x0c]
xor edx, eax
xor ebx, eax
ror eax, 0x3
mov ecx, eax
xor eax, ebx
xor edi, eax
and ecx, eax
xor ecx, edi
and edi, ebx
xor ebx, edx
or edx, ecx
xor eax, edx
xor edi, edx
mov [ebp + 0x0c], eax
xor ebx, ecx
and eax, edx
mov edx, edi
xor edx, ebx
xor eax, ebx
or edx, [ebp + 0x0c]
xor edx, ecx
mov ebx, esi
xor edi, edx
xor edi, eax
mov ecx, [ebx + 0x4]
xor edx, [ecx + 0x1b4]
mov ebx, [ecx + 0x1bc]
xor ebx, edi
mov [ebp + 0x8], edx
mov edx, [ecx + 0x1b8]
mov ecx, [ecx + 0x1b0]
xor edx, eax
xor ecx, [ebp + 0x0c]
mov edi, [ebp + 0x8]
mov eax, edi
ror ecx, 0x5
xor ecx, edi
shl eax, 0x7
xor ecx, ebx
rol edx, 0x0a
xor edx, eax
ror edi, 0x1
xor edx, ebx
xor edi, ecx
xor edi, edx
ror ebx, 0x7
lea eax, [ecx*8]
ror ecx, 0x0d
xor ebx, eax
xor ebx, edx
ror edx, 0x3
xor edx, ebx
xor ebx, ecx
mov eax, edx
and eax, ebx
xor eax, edi
or edi, edx
xor edi, ebx
xor edx, eax
and ebx, ecx
and ebx, eax
not eax
xor ebx, edx
mov [ebp + 0x8], eax
mov eax, edi
and eax, edx
mov edx, [ebp + 0x8]
or eax, ecx
xor ecx, edx
and ecx, edi
xor eax, edx
xor ecx, ebx
xor edx, ecx
mov ecx, [esi + 0x4]
mov ecx, [ecx + 0x1a0]
xor ecx, edi
mov edi, [esi + 0x4]
ror ecx, 0x5
mov edi, [edi + 0x1a4]
xor edi, ebx
mov ebx, [esi + 0x4]
xor ecx, edi
xor edx, [ebx + 0x1ac]
mov ebx, [ebx + 0x1a8]
xor ecx, edx
xor ebx, eax
mov eax, edi
rol ebx, 0x0a
shl eax, 0x7
xor ebx, eax
ror edi, 0x1
xor ebx, edx
lea eax, [ecx*8]
ror edx, 0x7
xor edi, ecx
xor edx, eax
xor edi, ebx
xor edx, ebx
ror ebx, 0x3
mov [ebp + 0x8], edx
ror ecx, 0x0d
mov [ebp + 0x0c], ecx
mov edx, edi
xor edi, [ebp + 0x8]
xor edx, ebx
mov eax, [ebp + 0x0c]
mov ecx, edi
and ecx, [ebp + 0x8]
xor ecx, eax
or eax, edi
xor ebx, ecx
xor eax, edx
or eax, ebx
xor edi, ecx
xor eax, edi
not edx
or edi, ecx
mov [ebp + 0x0c], eax
xor edi, eax
or eax, edi
xor edx, edi
mov edi, [ebp + 0x0c]
xor eax, edi
or eax, edx
xor ecx, eax
mov eax, [esi + 0x4]
xor edi, [eax + 0x194]
mov eax, [eax + 0x19c]
xor eax, ebx
mov ebx, esi
mov [ebp + 0x8], eax
mov eax, [ebx + 0x4]
mov ebx, [eax + 0x198]
mov eax, edi
xor ebx, ecx
shl eax, 0x7
rol ebx, 0x0a
mov ecx, esi
xor ebx, eax
xor ebx, [ebp + 0x8]
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x190]
xor ecx, edx
mov edx, [ebp + 0x8]
ror ecx, 0x5
xor ecx, edi
ror edi, 0x1
xor ecx, edx
ror edx, 0x7
xor edi, ecx
xor edi, ebx
mov [ebp + 0x0c], edi
lea eax, [ecx*8]
ror ecx, 0x0d
xor edx, eax
xor edx, ebx
ror ebx, 0x3
mov [ebp + 0x8], edx
not ebx
mov edx, edi
mov edi, ecx
or edi, [ebp + 0x0c]
not edx
xor ecx, edx
xor edi, ebx
xor edi, [ebp + 0x8]
mov [ebp + 0x0c], ecx
mov ecx, edx
or ecx, ebx
mov ebx, [ebp + 0x0c]
xor ecx, ebx
and ebx, [ebp + 0x8]
xor edx, ebx
or ebx, edi
xor ebx, ecx
mov eax, edx
xor eax, ebx
xor eax, edi
xor [ebp + 0x8], eax
mov eax, edi
xor eax, ecx
mov ecx, [ebp + 0x8]
and eax, ecx
xor edx, eax
mov eax, [esi + 0x4]
mov eax, [eax + 0x184]
xor eax, edx
mov edx, esi
mov [ebp + 0x0c], eax
mov eax, [edx + 0x4]
xor ecx, [eax + 0x18c]
mov edx, [eax + 0x188]
xor edx, edi
mov edi, [ebp + 0x0c]
rol edx, 0x0a
mov eax, edi
shl eax, 0x7
xor edx, eax
mov eax, [esi + 0x4]
xor edx, ecx
mov eax, [eax + 0x180]
xor eax, ebx
mov ebx, eax
mov [ebp + 0x8], eax
ror ebx, 0x5
xor ebx, edi
ror edi, 0x1
xor ebx, ecx
ror ecx, 0x7
xor edi, ebx
xor edi, edx
lea eax, [ebx*8]
ror ebx, 0x0d
xor ecx, eax
xor ecx, edx
ror edx, 0x3
mov [ebp + 0x8], ecx
mov ecx, edx
mov eax, [ebp + 0x8]
xor edx, ebx
or ecx, eax
and ebx, eax
xor eax, edi
not edx
or edi, ebx
and eax, ecx
xor ebx, edx
mov [ebp + 0x8], eax
and edx, ecx
xor edi, edx
xor edx, ebx
xor ecx, edi
or ebx, edx
xor ebx, eax
mov eax, ecx
xor eax, edx
or ecx, ebx
xor [ebp + 0x8], eax
xor ecx, edx
mov eax, esi
mov edx, [eax + 0x4]
xor ebx, [edx + 0x174]
mov eax, [edx + 0x17c]
mov edx, [edx + 0x178]
xor eax, ecx
xor edx, edi
mov [ebp + 0x0c], eax
mov ecx, [ebp + 0x0c]
mov edi, esi
rol edx, 0x0a
mov eax, ebx
shl eax, 0x7
xor edx, eax
mov eax, [edi + 0x4]
xor edx, ecx
mov edi, [eax + 0x170]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
xor edi, ecx
ror ecx, 0x7
lea eax, [edi*8]
xor ecx, eax
xor ecx, edx
ror ebx, 0x1
xor ebx, edi
mov [ebp + 0x0c], ecx
mov eax, [ebp + 0x0c]
xor ebx, edx
ror edx, 0x3
ror edi, 0x0d
mov ecx, edx
xor ecx, eax
xor edi, edx
xor eax, ebx
or ecx, edi
mov [ebp + 0x0c], eax
mov eax, edi
and eax, edx
mov edx, [ebp + 0x0c]
not eax
xor eax, edx
xor edx, ecx
xor edi, eax
xor ecx, ebx
and ebx, edx
xor ebx, edi
xor edi, edx
xor edx, ebx
mov [ebp + 0x8], ebx
or edi, eax
mov [ebp + 0x0c], edx
xor ecx, edi
mov edx, esi
mov ebx, [edx + 0x4]
mov edx, [ebx + 0x164]
mov edi, [ebx + 0x168]
xor edx, eax
mov eax, [ebp + 0x0c]
xor edi, ecx
xor eax, [ebx + 0x16c]
mov ebx, [ebx + 0x160]
xor ebx, [ebp + 0x8]
mov [ebp + 0x0c], eax
mov eax, edx
mov ecx, [ebp + 0x0c]
shl eax, 0x7
rol edi, 0x0a
xor edi, eax
ror ebx, 0x5
xor edi, ecx
mov [ebp - 0x8], edx
xor ebx, edx
xor ebx, ecx
ror ecx, 0x7
lea eax, [ebx*8]
xor ecx, eax
mov eax, edx
xor ecx, edi
ror eax, 0x1
xor eax, ebx
ror ebx, 0x0d
xor eax, edi
mov edx, ebx
ror edi, 0x3
or edx, ecx
not eax
xor edi, eax
xor edx, edi
xor ecx, edx
mov [ebp + 0x0c], ecx
mov ecx, eax
or ecx, edi
mov edi, [ebp + 0x0c]
and ecx, ebx
xor ecx, edi
or edi, ebx
xor edi, eax
and eax, ecx
xor eax, edx
xor edi, ecx
and edx, edi
mov [ebp - 0x8], eax
xor edi, eax
mov eax, edi
not edi
xor eax, ebx
mov ebx, esi
xor edx, eax
mov eax, [ebx + 0x4]
xor edi, [eax + 0x154]
mov ebx, [eax + 0x158]
mov [ebp + 0x0c], edi
xor ebx, edx
mov edi, [eax + 0x15c]
mov eax, [ebp + 0x0c]
xor edi, ecx
shl eax, 0x7
mov ecx, esi
rol ebx, 0x0a
xor ebx, eax
xor ebx, edi
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x150]
xor ecx, [ebp - 0x8]
ror ecx, 0x5
xor ecx, [ebp + 0x0c]
xor ecx, edi
ror edi, 0x7
lea eax, [ecx*8]
xor edi, eax
mov eax, [ebp + 0x0c]
xor edi, ebx
ror eax, 0x1
xor eax, ecx
ror ecx, 0x0d
xor eax, ebx
ror ebx, 0x3
mov edx, ebx
and edx, edi
xor edx, eax
or eax, edi
and eax, ecx
not ecx
xor ebx, eax
and eax, edx
xor ebx, edx
xor edi, ebx
mov [ebp - 0x8], ebx
xor eax, edi
mov [ebp + 0x0c], eax
mov eax, ecx
xor ecx, [ebp + 0x0c]
and eax, edi
xor eax, edx
mov [ebp + 0x8], ecx
xor eax, ecx
and ecx, edx
mov edx, esi
xor ecx, ebx
or ecx, eax
xor ecx, [ebp + 0x0c]
mov edi, [edx + 0x4]
mov ebx, [edi + 0x144]
mov edx, [edi + 0x14c]
mov edi, [edi + 0x148]
xor ebx, [ebp + 0x8]
xor edi, ecx
xor edx, [ebp - 0x8]
xor ebx, eax
rol edi, 0x0a
mov eax, ebx
shl eax, 0x7
mov ecx, esi
xor edi, eax
xor edi, edx
mov [ebp + 0x0c], edi
mov eax, [ecx + 0x4]
mov edi, [eax + 0x140]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
xor edi, edx
ror edx, 0x7
lea eax, [edi*8]
xor edx, eax
ror ebx, 0x1
mov eax, [ebp + 0x0c]
xor ebx, edi
xor edx, eax
ror edi, 0x0d
xor ebx, eax
ror eax, 0x3
mov ecx, eax
xor eax, ebx
and ecx, eax
xor edi, eax
xor ecx, edi
and edi, ebx
xor ebx, edx
or edx, ecx
xor eax, edx
xor edi, edx
xor ebx, ecx
mov [ebp + 0x0c], eax
and eax, edx
mov edx, edi
xor eax, ebx
xor edx, ebx
or edx, [ebp + 0x0c]
mov ebx, esi
xor edx, ecx
xor edi, edx
xor edi, eax
mov ecx, [ebx + 0x4]
xor edx, [ecx + 0x134]
mov ebx, [ecx + 0x13c]
mov [ebp + 0x8], edx
xor ebx, edi
mov edx, [ecx + 0x138]
mov ecx, [ecx + 0x130]
xor edx, eax
xor ecx, [ebp + 0x0c]
mov edi, [ebp + 0x8]
mov eax, edi
ror ecx, 0x5
xor ecx, edi
shl eax, 0x7
xor ecx, ebx
rol edx, 0x0a
xor edx, eax
ror edi, 0x1
xor edx, ebx
xor edi, ecx
ror ebx, 0x7
xor edi, edx
lea eax, [ecx*8]
ror ecx, 0x0d
xor ebx, eax
xor ebx, edx
ror edx, 0x3
xor edx, ebx
xor ebx, ecx
mov eax, edx
and eax, ebx
xor eax, edi
or edi, edx
xor edi, ebx
xor edx, eax
and ebx, ecx
and ebx, eax
not eax
mov [ebp + 0x8], eax
xor ebx, edx
mov eax, edi
and eax, edx
mov edx, [ebp + 0x8]
or eax, ecx
xor ecx, edx
and ecx, edi
xor eax, edx
xor ecx, ebx
xor edx, ecx
mov ecx, [esi + 0x4]
mov ecx, [ecx + 0x120]
xor ecx, edi
mov edi, [esi + 0x4]
ror ecx, 0x5
mov edi, [edi + 0x124]
xor edi, ebx
mov ebx, [esi + 0x4]
xor ecx, edi
xor edx, [ebx + 0x12c]
mov ebx, [ebx + 0x128]
xor ecx, edx
xor ebx, eax
mov eax, edi
shl eax, 0x7
rol ebx, 0x0a
xor ebx, eax
ror edi, 0x1
xor ebx, edx
lea eax, [ecx*8]
xor edi, ecx
ror edx, 0x7
xor edx, eax
ror ecx, 0x0d
xor edx, ebx
mov [ebp + 0x0c], ecx
mov eax, [ebp + 0x0c]
xor edi, ebx
mov [ebp + 0x8], edx
mov edx, edi
xor edi, [ebp + 0x8]
mov ecx, edi
ror ebx, 0x3
and ecx, [ebp + 0x8]
xor edx, ebx
xor ecx, eax
or eax, edi
xor eax, edx
xor ebx, ecx
or eax, ebx
xor edi, ecx
xor eax, edi
not edx
or edi, ecx
mov [ebp + 0x0c], eax
xor edi, eax
or eax, edi
xor edx, edi
mov edi, [ebp + 0x0c]
xor eax, edi
or eax, edx
xor ecx, eax
mov eax, [esi + 0x4]
xor edi, [eax + 0x114]
mov eax, [eax + 0x11c]
xor eax, ebx
mov ebx, esi
mov [ebp + 0x8], eax
mov eax, [ebx + 0x4]
mov ebx, [eax + 0x118]
mov eax, edi
xor ebx, ecx
shl eax, 0x7
mov ecx, esi
rol ebx, 0x0a
xor ebx, eax
xor ebx, [ebp + 0x8]
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x110]
xor ecx, edx
mov edx, [ebp + 0x8]
ror ecx, 0x5
xor ecx, edi
ror edi, 0x1
xor ecx, edx
ror edx, 0x7
xor edi, ecx
xor edi, ebx
mov [ebp + 0x0c], edi
lea eax, [ecx*8]
xor edx, eax
xor edx, ebx
ror ecx, 0x0d
mov [ebp + 0x8], edx
mov edx, edi
ror ebx, 0x3
not edx
mov edi, ecx
or edi, [ebp + 0x0c]
xor ecx, edx
not ebx
mov [ebp + 0x0c], ecx
xor edi, ebx
mov ecx, edx
xor edi, [ebp + 0x8]
or ecx, ebx
mov ebx, [ebp + 0x0c]
xor ecx, ebx
and ebx, [ebp + 0x8]
xor edx, ebx
or ebx, edi
xor ebx, ecx
mov eax, edx
xor eax, ebx
xor eax, edi
xor [ebp + 0x8], eax
mov eax, edi
xor eax, ecx
mov ecx, [ebp + 0x8]
and eax, ecx
xor edx, eax
mov eax, [esi + 0x4]
mov eax, [eax + 0x104]
xor eax, edx
mov edx, esi
mov [ebp + 0x0c], eax
mov eax, [edx + 0x4]
xor ecx, [eax + 0x10c]
mov edx, [eax + 0x108]
xor edx, edi
mov edi, [ebp + 0x0c]
rol edx, 0x0a
mov eax, edi
shl eax, 0x7
xor edx, eax
mov eax, [esi + 0x4]
xor edx, ecx
mov eax, [eax + 0x100]
xor eax, ebx
mov ebx, eax
mov [ebp + 0x8], eax
ror ebx, 0x5
xor ebx, edi
ror edi, 0x1
xor ebx, ecx
ror ecx, 0x7
xor edi, ebx
xor edi, edx
lea eax, [ebx*8]
ror ebx, 0x0d
xor ecx, eax
xor ecx, edx
ror edx, 0x3
mov [ebp + 0x8], ecx
mov ecx, edx
mov eax, [ebp + 0x8]
xor edx, ebx
and ebx, eax
or ecx, eax
xor eax, edi
not edx
or edi, ebx
and eax, ecx
xor ebx, edx
mov [ebp + 0x8], eax
and edx, ecx
xor edi, edx
xor edx, ebx
or ebx, edx
xor ecx, edi
xor ebx, eax
mov eax, ecx
xor eax, edx
xor [ebp + 0x8], eax
or ecx, ebx
xor ecx, edx
mov eax, esi
mov edx, [eax + 0x4]
xor ebx, [edx + 0x0f4]
mov eax, [edx + 0x0fc]
mov edx, [edx + 0x0f8]
xor eax, ecx
xor edx, edi
mov [ebp + 0x0c], eax
mov ecx, [ebp + 0x0c]
mov eax, ebx
shl eax, 0x7
mov edi, esi
rol edx, 0x0a
xor edx, eax
xor edx, ecx
mov eax, [edi + 0x4]
mov edi, [eax + 0x0f0]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
ror ebx, 0x1
xor edi, ecx
ror ecx, 0x7
xor ebx, edi
xor ebx, edx
lea eax, [edi*8]
ror edi, 0x0d
xor ecx, eax
xor ecx, edx
ror edx, 0x3
xor edi, edx
mov [ebp + 0x0c], ecx
mov eax, [ebp + 0x0c]
mov ecx, edx
xor ecx, eax
xor eax, ebx
mov [ebp + 0x0c], eax
or ecx, edi
mov eax, edi
and eax, edx
mov edx, [ebp + 0x0c]
not eax
xor eax, edx
xor edx, ecx
xor edi, eax
xor ecx, ebx
and ebx, edx
xor ebx, edi
xor edi, edx
xor edx, ebx
mov [ebp + 0x8], ebx
mov [ebp + 0x0c], edx
or edi, eax
mov edx, esi
xor ecx, edi
mov ebx, [edx + 0x4]
mov edx, [ebx + 0x0e4]
mov edi, [ebx + 0x0e8]
xor edx, eax
mov eax, [ebp + 0x0c]
xor edi, ecx
xor eax, [ebx + 0x0ec]
mov ebx, [ebx + 0x0e0]
xor ebx, [ebp + 0x8]
mov [ebp + 0x0c], eax
mov eax, edx
mov ecx, [ebp + 0x0c]
ror ebx, 0x5
shl eax, 0x7
xor ebx, edx
xor ebx, ecx
rol edi, 0x0a
xor edi, eax
mov [ebp - 0x8], edx
xor edi, ecx
ror ecx, 0x7
lea eax, [ebx*8]
xor ecx, eax
mov eax, edx
xor ecx, edi
ror eax, 0x1
xor eax, ebx
ror ebx, 0x0d
xor eax, edi
mov edx, ebx
not eax
ror edi, 0x3
xor edi, eax
or edx, ecx
xor edx, edi
xor ecx, edx
mov [ebp + 0x0c], ecx
mov ecx, eax
or ecx, edi
mov edi, [ebp + 0x0c]
and ecx, ebx
xor ecx, edi
or edi, ebx
xor edi, eax
and eax, ecx
xor eax, edx
xor edi, ecx
mov [ebp - 0x8], eax
and edx, edi
xor edi, eax
mov eax, edi
not edi
xor eax, ebx
mov ebx, esi
xor edx, eax
mov eax, [ebx + 0x4]
xor edi, [eax + 0x0d4]
mov ebx, [eax + 0x0d8]
mov [ebp + 0x0c], edi
xor ebx, edx
mov edi, [eax + 0x0dc]
mov eax, [ebp + 0x0c]
xor edi, ecx
shl eax, 0x7
mov ecx, esi
rol ebx, 0x0a
xor ebx, eax
xor ebx, edi
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x0d0]
xor ecx, [ebp - 0x8]
ror ecx, 0x5
xor ecx, [ebp + 0x0c]
xor ecx, edi
ror edi, 0x7
lea eax, [ecx*8]
xor edi, eax
mov eax, [ebp + 0x0c]
ror eax, 0x1
xor edi, ebx
xor eax, ecx
ror ecx, 0x0d
xor eax, ebx
ror ebx, 0x3
mov edx, ebx
and edx, edi
xor edx, eax
or eax, edi
and eax, ecx
not ecx
xor ebx, eax
and eax, edx
xor ebx, edx
xor edi, ebx
mov [ebp - 0x8], ebx
xor eax, edi
mov [ebp + 0x0c], eax
mov eax, ecx
xor ecx, [ebp + 0x0c]
and eax, edi
xor eax, edx
mov [ebp + 0x8], ecx
xor eax, ecx
and ecx, edx
xor ecx, ebx
or ecx, eax
xor ecx, [ebp + 0x0c]
mov edx, esi
mov edi, [edx + 0x4]
mov ebx, [edi + 0x0c4]
mov edx, [edi + 0x0cc]
mov edi, [edi + 0x0c8]
xor ebx, [ebp + 0x8]
xor edi, ecx
xor edx, [ebp - 0x8]
xor ebx, eax
rol edi, 0x0a
mov eax, ebx
shl eax, 0x7
mov ecx, esi
xor edi, eax
xor edi, edx
mov [ebp + 0x0c], edi
mov eax, [ecx + 0x4]
mov edi, [eax + 0x0c0]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
ror ebx, 0x1
xor edi, edx
ror edx, 0x7
xor ebx, edi
lea eax, [edi*8]
ror edi, 0x0d
xor edx, eax
mov eax, [ebp + 0x0c]
xor edx, eax
xor ebx, eax
ror eax, 0x3
mov ecx, eax
xor eax, ebx
xor edi, eax
and ecx, eax
xor ecx, edi
and edi, ebx
xor ebx, edx
or edx, ecx
xor eax, edx
xor edi, edx
xor ebx, ecx
mov [ebp + 0x0c], eax
and eax, edx
mov edx, edi
xor edx, ebx
xor eax, ebx
or edx, [ebp + 0x0c]
mov ebx, esi
xor edx, ecx
xor edi, edx
xor edi, eax
mov ecx, [ebx + 0x4]
xor edx, [ecx + 0x0b4]
mov ebx, [ecx + 0x0bc]
mov [ebp + 0x8], edx
xor ebx, edi
mov edx, [ecx + 0x0b8]
mov ecx, [ecx + 0x0b0]
xor edx, eax
xor ecx, [ebp + 0x0c]
mov edi, [ebp + 0x8]
mov eax, edi
ror ecx, 0x5
xor ecx, edi
rol edx, 0x0a
xor ecx, ebx
shl eax, 0x7
xor edx, eax
ror edi, 0x1
xor edx, ebx
xor edi, ecx
ror ebx, 0x7
xor edi, edx
lea eax, [ecx*8]
ror ecx, 0x0d
xor ebx, eax
xor ebx, edx
ror edx, 0x3
xor edx, ebx
xor ebx, ecx
mov eax, edx
and eax, ebx
xor eax, edi
or edi, edx
xor edi, ebx
xor edx, eax
and ebx, ecx
and ebx, eax
not eax
mov [ebp + 0x8], eax
xor ebx, edx
mov eax, edi
and eax, edx
mov edx, [ebp + 0x8]
or eax, ecx
xor ecx, edx
and ecx, edi
xor eax, edx
xor ecx, ebx
xor edx, ecx
mov ecx, [esi + 0x4]
mov ecx, [ecx + 0x0a0]
xor ecx, edi
mov edi, [esi + 0x4]
ror ecx, 0x5
mov edi, [edi + 0x0a4]
xor edi, ebx
mov ebx, [esi + 0x4]
xor ecx, edi
xor edx, [ebx + 0x0ac]
mov ebx, [ebx + 0x0a8]
xor ecx, edx
xor ebx, eax
mov eax, edi
shl eax, 0x7
rol ebx, 0x0a
xor ebx, eax
ror edi, 0x1
xor ebx, edx
lea eax, [ecx*8]
xor edi, ecx
ror edx, 0x7
xor edx, eax
ror ecx, 0x0d
xor edi, ebx
mov [ebp + 0x0c], ecx
mov eax, [ebp + 0x0c]
xor edx, ebx
mov [ebp + 0x8], edx
mov edx, edi
xor edi, [ebp + 0x8]
mov ecx, edi
ror ebx, 0x3
and ecx, [ebp + 0x8]
xor edx, ebx
xor ecx, eax
or eax, edi
xor eax, edx
xor edi, ecx
xor ebx, ecx
not edx
or eax, ebx
xor eax, edi
or edi, ecx
xor edi, eax
mov [ebp + 0x0c], eax
or eax, edi
xor edx, edi
mov edi, [ebp + 0x0c]
xor eax, edi
or eax, edx
xor ecx, eax
mov eax, [esi + 0x4]
xor edi, [eax + 0x94]
mov eax, [eax + 0x9c]
xor eax, ebx
mov ebx, esi
mov [ebp + 0x8], eax
mov eax, [ebx + 0x4]
mov ebx, [eax + 0x98]
mov eax, edi
xor ebx, ecx
shl eax, 0x7
rol ebx, 0x0a
mov ecx, esi
xor ebx, eax
xor ebx, [ebp + 0x8]
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x90]
xor ecx, edx
mov edx, [ebp + 0x8]
ror ecx, 0x5
xor ecx, edi
ror edi, 0x1
xor ecx, edx
ror edx, 0x7
xor edi, ecx
xor edi, ebx
mov [ebp + 0x0c], edi
lea eax, [ecx*8]
ror ecx, 0x0d
xor edx, eax
xor edx, ebx
ror ebx, 0x3
mov [ebp + 0x8], edx
not ebx
mov edx, edi
mov edi, ecx
or edi, [ebp + 0x0c]
not edx
xor ecx, edx
xor edi, ebx
xor edi, [ebp + 0x8]
mov [ebp + 0x0c], ecx
mov ecx, edx
or ecx, ebx
mov ebx, [ebp + 0x0c]
xor ecx, ebx
and ebx, [ebp + 0x8]
xor edx, ebx
or ebx, edi
xor ebx, ecx
mov eax, edx
xor eax, ebx
xor eax, edi
xor [ebp + 0x8], eax
mov eax, edi
xor eax, ecx
mov ecx, [ebp + 0x8]
and eax, ecx
xor edx, eax
mov eax, [esi + 0x4]
mov eax, [eax + 0x84]
xor eax, edx
mov edx, esi
mov [ebp + 0x0c], eax
mov eax, [edx + 0x4]
xor ecx, [eax + 0x8c]
mov edx, [eax + 0x88]
xor edx, edi
mov edi, [ebp + 0x0c]
mov eax, edi
rol edx, 0x0a
shl eax, 0x7
xor edx, eax
mov eax, [esi + 0x4]
xor edx, ecx
mov eax, [eax + 0x80]
xor eax, ebx
mov ebx, eax
mov [ebp + 0x8], eax
ror ebx, 0x5
xor ebx, edi
xor ebx, ecx
ror ecx, 0x7
lea eax, [ebx*8]
xor ecx, eax
xor ecx, edx
ror edi, 0x1
mov [ebp + 0x8], ecx
mov eax, [ebp + 0x8]
xor edi, ebx
xor edi, edx
ror ebx, 0x0d
ror edx, 0x3
mov ecx, edx
xor edx, ebx
or ecx, eax
and ebx, eax
xor eax, edi
not edx
or edi, ebx
and eax, ecx
xor ebx, edx
mov [ebp + 0x8], eax
and edx, ecx
xor edi, edx
xor edx, ebx
xor ecx, edi
or ebx, edx
xor ebx, eax
mov eax, ecx
xor eax, edx
or ecx, ebx
xor [ebp + 0x8], eax
xor ecx, edx
mov eax, esi
mov edx, [eax + 0x4]
xor ebx, [edx + 0x74]
mov eax, [edx + 0x7c]
mov edx, [edx + 0x78]
xor eax, ecx
xor edx, edi
mov [ebp + 0x0c], eax
mov ecx, [ebp + 0x0c]
mov eax, ebx
shl eax, 0x7
mov edi, esi
rol edx, 0x0a
xor edx, eax
xor edx, ecx
mov eax, [edi + 0x4]
mov edi, [eax + 0x70]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
ror ebx, 0x1
xor edi, ecx
ror ecx, 0x7
xor ebx, edi
xor ebx, edx
lea eax, [edi*8]
ror edi, 0x0d
xor ecx, eax
xor ecx, edx
ror edx, 0x3
mov [ebp + 0x0c], ecx
xor edi, edx
mov eax, [ebp + 0x0c]
mov ecx, edx
xor ecx, eax
xor eax, ebx
mov [ebp + 0x0c], eax
or ecx, edi
mov eax, edi
and eax, edx
mov edx, [ebp + 0x0c]
not eax
xor eax, edx
xor edx, ecx
xor edi, eax
xor ecx, ebx
and ebx, edx
xor ebx, edi
xor edi, edx
xor edx, ebx
mov [ebp + 0x8], ebx
or edi, eax
mov [ebp + 0x0c], edx
xor ecx, edi
mov edx, esi
mov ebx, [edx + 0x4]
mov edx, [ebx + 0x64]
mov edi, [ebx + 0x68]
xor edx, eax
mov eax, [ebp + 0x0c]
xor edi, ecx
xor eax, [ebx + 0x6c]
mov ebx, [ebx + 0x60]
xor ebx, [ebp + 0x8]
mov [ebp + 0x0c], eax
mov eax, edx
mov ecx, [ebp + 0x0c]
shl eax, 0x7
rol edi, 0x0a
xor edi, eax
ror ebx, 0x5
xor edi, ecx
mov [ebp - 0x8], edx
xor ebx, edx
xor ebx, ecx
ror ecx, 0x7
lea eax, [ebx*8]
xor ecx, eax
mov eax, edx
xor ecx, edi
ror eax, 0x1
xor eax, ebx
ror ebx, 0x0d
xor eax, edi
mov edx, ebx
not eax
ror edi, 0x3
xor edi, eax
or edx, ecx
xor edx, edi
xor ecx, edx
mov [ebp + 0x0c], ecx
mov ecx, eax
or ecx, edi
mov edi, [ebp + 0x0c]
and ecx, ebx
xor ecx, edi
or edi, ebx
xor edi, eax
and eax, ecx
xor edi, ecx
xor eax, edx
and edx, edi
mov [ebp - 0x8], eax
xor edi, eax
mov eax, edi
not edi
xor eax, ebx
mov ebx, esi
xor edx, eax
mov eax, [ebx + 0x4]
xor edi, [eax + 0x54]
mov ebx, [eax + 0x58]
mov [ebp + 0x0c], edi
xor ebx, edx
mov edi, [eax + 0x5c]
mov eax, [ebp + 0x0c]
xor edi, ecx
shl eax, 0x7
mov ecx, esi
rol ebx, 0x0a
xor ebx, eax
xor ebx, edi
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x50]
xor ecx, [ebp - 0x8]
ror ecx, 0x5
xor ecx, [ebp + 0x0c]
xor ecx, edi
ror edi, 0x7
lea eax, [ecx*8]
xor edi, eax
mov eax, [ebp + 0x0c]
ror eax, 0x1
xor edi, ebx
xor eax, ecx
xor eax, ebx
ror ecx, 0x0d
ror ebx, 0x3
mov edx, ebx
and edx, edi
xor edx, eax
or eax, edi
and eax, ecx
not ecx
xor ebx, eax
and eax, edx
xor ebx, edx
xor edi, ebx
mov [ebp - 0x8], ebx
xor eax, edi
mov [ebp + 0x0c], eax
mov eax, ecx
xor ecx, [ebp + 0x0c]
and eax, edi
xor eax, edx
mov [ebp + 0x8], ecx
xor eax, ecx
and ecx, edx
xor ecx, ebx
mov edx, esi
or ecx, eax
xor ecx, [ebp + 0x0c]
mov edi, [edx + 0x4]
mov ebx, [edi + 0x44]
mov edx, [edi + 0x4c]
mov edi, [edi + 0x48]
xor ebx, [ebp + 0x8]
xor edi, ecx
xor edx, [ebp - 0x8]
xor ebx, eax
rol edi, 0x0a
mov eax, ebx
shl eax, 0x7
mov ecx, esi
xor edi, eax
xor edi, edx
mov [ebp + 0x0c], edi
mov eax, [ecx + 0x4]
mov edi, [eax + 0x40]
xor edi, [ebp + 0x8]
ror edi, 0x5
xor edi, ebx
ror ebx, 0x1
xor edi, edx
ror edx, 0x7
xor ebx, edi
lea eax, [edi*8]
ror edi, 0x0d
xor edx, eax
mov eax, [ebp + 0x0c]
xor edx, eax
xor ebx, eax
ror eax, 0x3
mov ecx, eax
xor eax, ebx
xor edi, eax
and ecx, eax
xor ecx, edi
and edi, ebx
xor ebx, edx
or edx, ecx
xor eax, edx
xor edi, edx
mov [ebp + 0x0c], eax
xor ebx, ecx
and eax, edx
mov edx, edi
xor edx, ebx
xor eax, ebx
or edx, [ebp + 0x0c]
mov ebx, esi
xor edx, ecx
xor edi, edx
xor edi, eax
mov ecx, [ebx + 0x4]
xor edx, [ecx + 0x34]
mov ebx, [ecx + 0x3c]
xor ebx, edi
mov [ebp + 0x8], edx
mov edx, [ecx + 0x38]
mov ecx, [ecx + 0x30]
xor edx, eax
xor ecx, [ebp + 0x0c]
mov eax, [ebp + 0x8]
ror ecx, 0x5
xor ecx, [ebp + 0x8]
xor ecx, ebx
shl eax, 0x7
rol edx, 0x0a
xor edx, eax
xor edx, ebx
ror ebx, 0x7
lea eax, [ecx*8]
xor ebx, eax
mov eax, [ebp + 0x8]
xor ebx, edx
ror eax, 0x1
xor eax, ecx
ror ecx, 0x0d
xor eax, edx
ror edx, 0x3
xor edx, ebx
xor ebx, ecx
mov edi, edx
and edi, ebx
xor edi, eax
or eax, edx
xor eax, ebx
xor edx, edi
and ebx, ecx
mov [ebp + 0x8], eax
and ebx, edi
and eax, edx
or eax, ecx
xor ebx, edx
mov edx, [esi + 0x4]
not edi
xor ecx, edi
xor eax, edi
and ecx, [ebp + 0x8]
xor ecx, ebx
mov edx, [edx + 0x24]
xor edi, ecx
mov ecx, [esi + 0x4]
xor edx, ebx
mov ebx, [esi + 0x4]
mov ecx, [ecx + 0x20]
xor edi, [ebx + 0x2c]
xor ecx, [ebp + 0x8]
mov ebx, [ebx + 0x28]
xor ebx, eax
ror ecx, 0x5
xor ecx, edx
rol ebx, 0x0a
xor ecx, edi
mov eax, edx
shl eax, 0x7
xor ebx, eax
ror edx, 0x1
xor ebx, edi
xor edx, ecx
lea eax, [ecx*8]
ror edi, 0x7
xor edi, eax
ror ecx, 0x0d
xor edi, ebx
mov [ebp + 0x8], ecx
mov eax, [ebp + 0x8]
xor edx, ebx
mov [ebp + 0x0c], edi
mov edi, edx
xor edx, [ebp + 0x0c]
ror ebx, 0x3
mov ecx, edx
and ecx, [ebp + 0x0c]
xor edi, ebx
xor ecx, eax
or eax, edx
xor ebx, ecx
xor eax, edi
xor edx, ecx
or eax, ebx
not edi
xor eax, edx
or edx, ecx
xor edx, eax
mov [ebp + 0x8], eax
or eax, edx
xor edi, edx
mov edx, [ebp + 0x8]
xor eax, edx
or eax, edi
xor ecx, eax
mov eax, [esi + 0x4]
xor edx, [eax + 0x14]
mov eax, [eax + 0x1c]
xor eax, ebx
mov [ebp + 0x8], edx
mov [ebp + 0x0c], eax
mov edx, esi
mov ebx, [ebp + 0x0c]
mov eax, [edx + 0x4]
mov edx, [eax + 0x18]
mov eax, [ebp + 0x8]
xor edx, ecx
shl eax, 0x7
mov ecx, esi
rol edx, 0x0a
xor edx, eax
xor edx, ebx
mov eax, [ecx + 0x4]
mov ecx, [eax + 0x10]
xor ecx, edi
mov edi, [ebp + 0x8]
ror ecx, 0x5
xor ecx, edi
ror edi, 0x1
xor ecx, ebx
ror ebx, 0x7
xor edi, ecx
xor edi, edx
lea eax, [ecx*8]
ror ecx, 0x0d
xor ebx, eax
mov [ebp + 0x8], ecx
mov eax, [ebp + 0x8]
xor ebx, edx
mov [ebp + 0x0c], ebx
or ecx, edi
mov ebx, edi
ror edx, 0x3
not edx
not ebx
xor eax, ebx
xor ecx, edx
xor ecx, [ebp + 0x0c]
mov edi, ebx
or edi, edx
mov [ebp - 0x8], ecx
xor edi, eax
mov edx, esi
and eax, [ebp + 0x0c]
xor ebx, eax
or eax, ecx
xor eax, edi
mov [ebp + 0x8], eax
mov eax, ebx
xor eax, [ebp + 0x8]
xor eax, ecx
mov ecx, [ebp + 0x0c]
xor ecx, eax
mov eax, [edx + 0x4]
mov [ebp + 0x0c], ecx
mov edx, [eax + 0x0c]
xor edx, ecx
mov ecx, [eax + 0x8]
mov eax, [ebp - 0x8]
xor ecx, eax
xor eax, edi
mov edi, [esi + 0x4]
and eax, [ebp + 0x0c]
mov esi, [ebp - 0x14]
xor eax, [edi + 0x4]
xor eax, ebx
mov ebx, [ebp - 0x0c]
mov [ebp - 0x8], eax
mov eax, [edi]
mov edi, [ebp - 0x10]
xor eax, [ebp + 0x8]
mov [ebx + esi], ecx
add ebx, 0x10
mov [ebp - 0x0c], ebx
mov [edi - 0x0c], eax
mov eax, [ebp - 0x8]
mov [edi - 0x8], eax
mov [edi], edx
add edi, 0x10
sub dword [ebp + 0x10], 0x1
mov [ebp - 0x10], edi
jne .loop
.end:
pop edi
pop esi
pop ebx
mov esp, ebp
pop ebp
ret 0x0c

slow4:
push ebp
mov ebp,esp
sub esp, 0x0c4
cmp dword [ebp + 0x10],0
mov eax,dword [ebp + 0x8]
push ebx
push esi
mov eax,dword [eax]
mov ecx,dword [eax]
mov dword [ebp - 0x24],ecx
mov ecx,dword [eax + 0x4]
mov esi,dword [eax + 0x20]
mov ebx,dword [eax + 0x24]
mov dword [ebp - 0x1c],ecx
mov ecx,dword [eax + 0x8]
mov dword [ebp - 0x28],ecx
mov ecx,dword [eax + 0x0c]
mov dword [ebp - 0x0c],ecx
mov ecx,dword [eax + 0x10]
mov dword [ebp - 0x2c],ecx
mov ecx,dword [eax + 0x14]
mov dword [ebp - 0x10],ecx
mov ecx,dword [eax + 0x18]
mov dword [ebp - 0x30],ecx
mov ecx,dword [eax + 0x1c]
mov dword [ebp - 0x14],ecx
mov ecx,dword [eax + 0x28]
mov dword [ebp - 0x34],ecx
mov ecx,dword [eax + 0x2c]
mov dword [ebp - 0x4],ecx
mov ecx,dword [eax + 0x30]
mov dword [ebp - 0x38],ecx
mov ecx,dword [eax + 0x34]
mov dword [ebp - 0x8],ecx
mov ecx,dword [eax + 0x38]
mov eax,dword [eax + 0x3c]
mov dword [ebp - 0x40],esi
mov dword [ebp - 0x18],ebx
mov dword [ebp - 0x3c],ecx
mov dword [ebp - 0x20],eax
je slow4_2
push edi
mov edi,dword [ebp + 0x0c]
add edi, 0x10
mov dword [ebp - 0x0c4],edi
slow4_1:
mov ecx,dword [edi - 0x10]
mov eax,dword [edi - 0x0c]
mov dword [ebp - 0x44],eax
mov dword [ebp - 0x48],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
xor edx,edx
mov dword [ebp - 0x0a4],eax
or edx,ecx
mov ecx,dword [edi - 0x4]
mov dword [ebp - 0x0a8],edx
mov edx,dword [edi - 0x8]
mov dword [ebp - 0x44],ecx
mov dword [ebp - 0x48],edx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi]
mov dword [ebp - 0x0b8],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x4]
mov dword [ebp - 0x0b4],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x8]
mov dword [ebp - 0x50],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x0c]
mov dword [ebp - 0x4c],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x10]
mov dword [ebp - 0x58],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x14]
mov dword [ebp - 0x54],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x18]
mov dword [ebp - 0x60],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x1c]
mov dword [ebp - 0x5c],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x20]
mov dword [ebp - 0x68],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x24]
mov dword [ebp - 0x64],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x28]
mov dword [ebp - 0x70],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x2c]
mov dword [ebp - 0x6c],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x30]
mov dword [ebp - 0x98],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x34]
mov dword [ebp - 0x94],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x38]
mov dword [ebp - 0x0b0],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x3c]
mov dword [ebp - 0x0ac],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x40]
mov dword [ebp - 0x78],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x44]
mov dword [ebp - 0x74],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x48]
mov dword [ebp - 0x80],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x4c]
mov dword [ebp - 0x7c],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov edx,dword [edi + 0x50]
mov dword [ebp - 0x88],eax
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x54]
mov dword [ebp - 0x84],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
xor edx,edx
mov dword [ebp - 0x8c],eax
or edx,ecx
mov ecx,dword [edi + 0x5c]
mov dword [ebp - 0x90],edx
mov edx,dword [edi + 0x58]
mov dword [ebp - 0x44],ecx
mov dword [ebp - 0x48],edx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
xor edx,edx
mov dword [ebp - 0x9c],eax
or edx,ecx
mov ecx,dword [edi + 0x64]
mov dword [ebp - 0x0a0],edx
mov edx,dword [edi + 0x60]
mov dword [ebp - 0x44],ecx
mov dword [ebp - 0x48],edx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
mov dword [ebp - 0x0c0],eax
mov edx,dword [edi + 0x68]
xor eax,eax
or eax,ecx
mov dword [ebp - 0x48],edx
mov ecx,dword [edi + 0x6c]
mov dword [ebp - 0x0bc],eax
mov dword [ebp - 0x44],ecx
mov eax,dword [ebp - 0x44]
bswap eax
mov ecx,eax
mov eax,dword [ebp - 0x48]
bswap eax
push 0x29
push 0x12
mov dword [ebp - 0x44],eax
xor eax,eax
push 0x0e
or eax,ecx
push ebx
push esi
mov dword [ebp - 0x48],eax
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov edi,esi
and ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
not edi
and edi,dword [ebp - 0x38]
not esi
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0a8]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0a4]
add ebx, 0xd728ae22
push 0x22
adc edx, 0x428a2f98
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x2c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0c0]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x0bc]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
mov esi,eax
mov edi,edx
push 0x8
push 0x1
push dword [ebp - 0x0b8]
push dword [ebp - 0x0b4]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x74]
adc edi,dword [ebp - 0x78]
add dword [ebp - 0x0a8],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0a4],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0b4]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x0b8]
add ebx, 0x23ef65cd
push 0x22
adc edx, 0x71374491
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x44]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x48]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x50]
mov esi,eax
mov edi,edx
push dword [ebp - 0x4c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x7c]
adc edi,dword [ebp - 0x80]
add dword [ebp - 0x0b4],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x0b8],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x4c]
push 0x22
adc edx,dword [ebp - 0x50]
add ebx, 0xec4d3b2f
push 0x1c
push dword [ebp - 0x8]
adc edx, 0xb5c0fbcf
add eax,ebx
push dword [ebp - 0x38]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0a4]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0a8]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x58]
mov esi,eax
mov edi,edx
push dword [ebp - 0x54]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x84]
adc edi,dword [ebp - 0x88]
add dword [ebp - 0x4c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x50],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x54]
push 0x22
adc edx,dword [ebp - 0x58]
add ebx, 0x8189dbbc
push 0x1c
push dword [ebp - 0x4]
adc edx, 0xe9b5dba5
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b8]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x0b4]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x60]
mov esi,eax
mov edi,edx
push dword [ebp - 0x5c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x90]
adc edi,dword [ebp - 0x8c]
add dword [ebp - 0x54],esi
push 0x29
adc dword [ebp - 0x58],edi
mov esi,dword [ebp - 0x1c]
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x5c]
push 0x22
adc edx,dword [ebp - 0x60]
add ebx, 0xf348b538
push 0x1c
push dword [ebp - 0x18]
adc edx, 0x3956c25b
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x50]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x4c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x68]
mov esi,eax
mov edi,edx
push dword [ebp - 0x64]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a0]
adc edi,dword [ebp - 0x9c]
add dword [ebp - 0x5c],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x60],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x64]
push 0x22
adc edx,dword [ebp - 0x68]
add ebx, 0xb605d019
push 0x1c
push dword [ebp - 0x14]
adc edx, 0x59f111f1
add eax,ebx
push dword [ebp - 0x30]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x58]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x54]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x70]
mov esi,eax
mov edi,edx
push dword [ebp - 0x6c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0bc]
adc edi,dword [ebp - 0x0c0]
add dword [ebp - 0x64],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x68],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x6c]
push 0x22
adc edx,dword [ebp - 0x70]
add ebx, 0xaf194f9b
push 0x1c
push dword [ebp - 0x10]
adc edx, 0x923f82a4
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x60]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x5c]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x98]
mov esi,eax
mov edi,edx
push dword [ebp - 0x94]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x48]
adc edi,dword [ebp - 0x44]
add dword [ebp - 0x6c],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x70],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
not esi
not edi
mov ebx,eax
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
mov ecx,dword [ebp - 0x38]
and ecx,dword [ebp - 0x34]
mov eax,dword [ebp - 0x8]
xor edi,ecx
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x94]
push 0x22
adc edx,dword [ebp - 0x98]
add ebx, 0xda6d8118
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0xab1c5ed5
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x68]
adc edx,esi
add dword [ebp - 0x24],ebx
push dword [ebp - 0x64]
adc dword [ebp - 0x1c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0ac]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a8]
adc edi,dword [ebp - 0x0a4]
add dword [ebp - 0x94],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x98],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
mov eax,dword [ebp - 0x4]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x40]
and eax,dword [ebp - 0x18]
xor edi,ecx
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
mov ecx,dword [ebp - 0x20]
adc edx,esi
add ebx,dword [ebp - 0x0ac]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0b0]
add ebx, 0xa3030242
push 0x27
adc edx, 0xd807aa98
add eax,ebx
push 0x22
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push 0x1c
push dword [ebp - 0x1c]
adc dword [ebp - 0x14],ecx
push esi
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x24]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x70]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x6c]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x78]
mov esi,eax
mov edi,edx
push dword [ebp - 0x74]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0b4]
adc edi,dword [ebp - 0x0b8]
add dword [ebp - 0x0ac],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0b0],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and edi,dword [ebp - 0x34]
not esi
and ecx,dword [ebp - 0x30]
mov eax,dword [ebp - 0x18]
xor edi,ecx
and esi,dword [ebp - 0x4]
and eax,dword [ebp - 0x14]
xor esi,eax
add ebx,edi
mov eax,dword [ebp - 0x38]
adc edx,esi
add ebx,dword [ebp - 0x74]
mov ecx,dword [ebp - 0x8]
adc edx,dword [ebp - 0x78]
add ebx, 0x45706fbe
mov esi,dword [ebp - 0x20]
push 0x27
adc edx, 0x12835b01
add eax,ebx
push 0x22
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push 0x1c
adc dword [ebp - 0x10],ecx
push esi
push dword [ebp - 0x3c]
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x98]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x94]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x80]
mov esi,eax
mov edi,edx
push dword [ebp - 0x7c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x4c]
adc edi,dword [ebp - 0x50]
add dword [ebp - 0x74],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x78],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and edi,dword [ebp - 0x40]
not esi
and ecx,dword [ebp - 0x2c]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x7c]
adc edx,dword [ebp - 0x80]
add ebx, 0x4ee4b28c
mov eax,dword [ebp - 0x34]
adc edx, 0x243185be
mov ecx,dword [ebp - 0x4]
add eax,ebx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x8]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
push dword [ebp - 0x38]
adc dword [ebp - 0x0c],ecx
mov dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b0]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0ac]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x88]
mov esi,eax
mov edi,edx
push dword [ebp - 0x84]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x54]
adc edi,dword [ebp - 0x58]
add dword [ebp - 0x7c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x80],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and edi,dword [ebp - 0x30]
not esi
and ecx,dword [ebp - 0x28]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x84]
adc edx,dword [ebp - 0x88]
add ebx, 0xd5ffb4e2
adc edx, 0x550c7dc3
add eax,ebx
mov dword [ebp - 0x40],eax
mov ecx,dword [ebp - 0x18]
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x4]
adc ecx,edx
add dword [ebp - 0x24],eax
push dword [ebp - 0x34]
adc dword [ebp - 0x1c],ecx
mov dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x78]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x74]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x8c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x90]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x5c]
adc edi,dword [ebp - 0x60]
add dword [ebp - 0x84],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x88],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x90]
adc edx,dword [ebp - 0x8c]
add ebx, 0xf27b896f
adc edx, 0x72be5d74
add eax,ebx
mov dword [ebp - 0x30],eax
adc ecx,edx
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x18]
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x80]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x7c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x9c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a0]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x64]
adc edi,dword [ebp - 0x68]
add dword [ebp - 0x90],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x8c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0a0]
push 0x22
adc edx,dword [ebp - 0x9c]
add ebx, 0x3b1696b1
push 0x1c
adc edx, 0x80deb1fe
add eax,ebx
mov dword [ebp - 0x2c],eax
adc ecx,edx
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
push dword [ebp - 0x14]
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x88]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x84]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0c0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0bc]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x6c]
adc edi,dword [ebp - 0x70]
add dword [ebp - 0x0a0],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x9c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0bc]
push 0x22
adc edx,dword [ebp - 0x0c0]
add ebx, 0x25c71235
push 0x1c
push dword [ebp - 0x10]
adc edx, 0x9bdc06a7
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
mov edi,ecx
or edi,dword [ebp - 0x2c]
mov esi,dword [ebp - 0x14]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x8c]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x90]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x44]
mov esi,eax
mov edi,edx
push dword [ebp - 0x48]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x94]
adc edi,dword [ebp - 0x98]
add dword [ebp - 0x0bc],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x0c0],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and ecx,dword [ebp - 0x34]
not esi
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x48]
push 0x22
adc edx,dword [ebp - 0x44]
add ebx, 0xcf692694
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0xc19bf174
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x9c]
adc edx,esi
add dword [ebp - 0x24],ebx
push dword [ebp - 0x0a0]
adc dword [ebp - 0x1c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0a4]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a8]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0ac]
adc edi,dword [ebp - 0x0b0]
add dword [ebp - 0x48],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x44],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0a8]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0a4]
add ebx, 0x9ef14ad2
push 0x22
adc edx, 0xe49b69c1
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x24]
mov eax,dword [ebp - 0x0c]
or edi,ecx
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0c0]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x0bc]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b8]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0b4]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x74]
adc edi,dword [ebp - 0x78]
add dword [ebp - 0x0a8],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0a4],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0b4]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x0b8]
add ebx, 0x384f25e3
push 0x22
adc edx, 0xefbe4786
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x3c]
mov eax,dword [ebp - 0x20]
and edi,dword [ebp - 0x28]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
or edi,ecx
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x44]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x48]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x50]
mov esi,eax
mov edi,edx
push dword [ebp - 0x4c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x7c]
adc edi,dword [ebp - 0x80]
add dword [ebp - 0x0b4],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x0b8],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x4c]
push 0x22
adc edx,dword [ebp - 0x50]
add ebx, 0x8b8cd5b5
push 0x1c
push dword [ebp - 0x8]
adc edx, 0x0fc19dc6
add eax,ebx
push dword [ebp - 0x38]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and edi,dword [ebp - 0x24]
and ecx,dword [ebp - 0x38]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
or esi,eax
add ebx,edi
adc edx,esi
add dword [ebp - 0x34],ebx
adc dword [ebp - 0x4],edx
push 0x6
push 0x3d
push 0x13
push dword [ebp - 0x0a4]
push dword [ebp - 0x0a8]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x58]
mov esi,eax
mov edi,edx
push dword [ebp - 0x54]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x84]
adc edi,dword [ebp - 0x88]
add dword [ebp - 0x4c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x50],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x54]
push 0x22
adc edx,dword [ebp - 0x58]
add ebx, 0x77ac9c65
push 0x1c
push dword [ebp - 0x4]
adc edx, 0x240ca1cc
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and edi,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x34]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
or esi,eax
add ebx,edi
push 0x6
adc edx,esi
add dword [ebp - 0x40],ebx
push 0x3d
adc dword [ebp - 0x18],edx
push 0x13
push dword [ebp - 0x0b8]
push dword [ebp - 0x0b4]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x60]
mov esi,eax
mov edi,edx
push dword [ebp - 0x5c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x90]
adc edi,dword [ebp - 0x8c]
add dword [ebp - 0x54],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x58],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x5c]
push 0x22
adc edx,dword [ebp - 0x60]
add ebx, 0x592b0275
push 0x1c
push dword [ebp - 0x18]
adc edx, 0x2de92c6f
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and edi,dword [ebp - 0x38]
and ecx,dword [ebp - 0x40]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x50]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x4c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
mov esi,eax
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x68]
mov edi,edx
push dword [ebp - 0x64]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a0]
adc edi,dword [ebp - 0x9c]
add dword [ebp - 0x5c],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x60],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x64]
push 0x22
adc edx,dword [ebp - 0x68]
add ebx, 0x6ea6e483
push 0x1c
push dword [ebp - 0x14]
adc edx, 0x4a7484aa
add eax,ebx
push dword [ebp - 0x30]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and edi,dword [ebp - 0x34]
and ecx,dword [ebp - 0x30]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x58]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x54]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
mov esi,eax
mov edi,edx
push 0x1
push dword [ebp - 0x70]
push dword [ebp - 0x6c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0bc]
adc edi,dword [ebp - 0x0c0]
add dword [ebp - 0x64],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x68],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x6c]
push 0x22
adc edx,dword [ebp - 0x70]
add ebx, 0xbd41fbd4
push 0x1c
push dword [ebp - 0x10]
adc edx, 0x5cb0a9dc
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x60]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x5c]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x98]
mov esi,eax
mov edi,edx
push dword [ebp - 0x94]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
push 0x29
adc edi,edx
add esi,dword [ebp - 0x48]
push 0x12
adc edi,dword [ebp - 0x44]
add dword [ebp - 0x6c],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x70],edi
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and ecx,dword [ebp - 0x34]
not esi
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x94]
push 0x22
adc edx,dword [ebp - 0x98]
add ebx, 0x831153b5
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0x76f988da
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x68]
adc edx,esi
add dword [ebp - 0x24],ebx
push dword [ebp - 0x64]
adc dword [ebp - 0x1c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0ac]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a8]
adc edi,dword [ebp - 0x0a4]
add dword [ebp - 0x94],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x98],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0ac]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0b0]
add ebx, 0xee66dfab
push 0x22
adc edx, 0x983e5152
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x24]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x70]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x6c]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x78]
mov esi,eax
mov edi,edx
push dword [ebp - 0x74]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0b4]
adc edi,dword [ebp - 0x0b8]
add dword [ebp - 0x0ac],esi
push 0x29
adc dword [ebp - 0x0b0],edi
mov esi,dword [ebp - 0x14]
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x74]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x78]
add ebx, 0x2db43210
push 0x22
adc edx, 0xa831c66d
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x98]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x94]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x80]
mov esi,eax
mov edi,edx
push dword [ebp - 0x7c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x4c]
adc edi,dword [ebp - 0x50]
add dword [ebp - 0x74],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x78],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x7c]
push 0x22
adc edx,dword [ebp - 0x80]
add ebx, 0x98fb213f
push 0x1c
push dword [ebp - 0x8]
adc edx, 0xb00327c8
add eax,ebx
push dword [ebp - 0x38]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b0]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0ac]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x88]
mov esi,eax
mov edi,edx
push dword [ebp - 0x84]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x54]
adc edi,dword [ebp - 0x58]
add dword [ebp - 0x7c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x80],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x84]
push 0x22
adc edx,dword [ebp - 0x88]
add ebx, 0xbeef0ee4
push 0x1c
push dword [ebp - 0x4]
adc edx, 0xbf597fc7
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x78]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x74]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x8c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x90]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x5c]
adc edi,dword [ebp - 0x60]
add dword [ebp - 0x84],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x88],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
not esi
not edi
mov ebx,eax
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x28]
and ecx,dword [ebp - 0x24]
mov eax,dword [ebp - 0x0c]
xor edi,ecx
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x90]
push 0x22
adc edx,dword [ebp - 0x8c]
add ebx, 0x3da88fc2
push 0x1c
push dword [ebp - 0x18]
adc edx, 0xc6e00bf3
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x80]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x7c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x9c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a0]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x64]
adc edi,dword [ebp - 0x68]
add dword [ebp - 0x90],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x8c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
mov eax,dword [ebp - 0x20]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x24]
and eax,dword [ebp - 0x1c]
xor edi,ecx
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
mov ecx,dword [ebp - 0x10]
adc edx,esi
add ebx,dword [ebp - 0x0a0]
push 0x27
adc edx,dword [ebp - 0x9c]
add ebx, 0x930aa725
push 0x22
adc edx, 0xd5a79147
add eax,ebx
push 0x1c
push dword [ebp - 0x14]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
push dword [ebp - 0x30]
adc dword [ebp - 0x8],ecx
mov dword [ebp - 0x10],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x88]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x84]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0c0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0bc]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x6c]
adc edi,dword [ebp - 0x70]
add dword [ebp - 0x0a0],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x9c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and edi,dword [ebp - 0x24]
not esi
and ecx,dword [ebp - 0x38]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x0bc]
mov eax,dword [ebp - 0x28]
adc edx,dword [ebp - 0x0c0]
add ebx, 0xe003826f
mov ecx,dword [ebp - 0x0c]
push 0x27
adc edx, 0x6ca6351
add eax,ebx
push 0x22
push 0x1c
push dword [ebp - 0x10]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
push dword [ebp - 0x2c]
adc dword [ebp - 0x4],ecx
mov dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x8c]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x90]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x44]
mov esi,eax
mov edi,edx
push dword [ebp - 0x48]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x94]
adc edi,dword [ebp - 0x98]
add dword [ebp - 0x0bc],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x0c0],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and edi,dword [ebp - 0x3c]
not esi
and ecx,dword [ebp - 0x34]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x48]
adc edx,dword [ebp - 0x44]
add ebx, 0x0a0e6e70
adc edx, 0x14292967
mov eax,dword [ebp - 0x24]
mov ecx,dword [ebp - 0x1c]
add eax,ebx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x0c]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
push dword [ebp - 0x28]
adc dword [ebp - 0x18],ecx
mov dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x9c]
adc edx,esi
add dword [ebp - 0x24],ebx
push dword [ebp - 0x0a0]
adc dword [ebp - 0x1c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0a4]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a8]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0ac]
adc edi,dword [ebp - 0x0b0]
add dword [ebp - 0x48],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x44],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x0a8]
adc edx,dword [ebp - 0x0a4]
add ebx, 0x46d22ffc
adc edx, 0x27b70a85
add eax,ebx
mov dword [ebp - 0x3c],eax
mov esi,dword [ebp - 0x24]
adc ecx,edx
add dword [ebp - 0x30],eax
push 0x27
adc dword [ebp - 0x14],ecx
push 0x22
push 0x1c
push dword [ebp - 0x1c]
mov dword [ebp - 0x20],ecx
push esi
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x24]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0c0]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x0bc]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b8]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0b4]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x74]
adc edi,dword [ebp - 0x78]
add dword [ebp - 0x0a8],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0a4],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x0b4]
adc edx,dword [ebp - 0x0b8]
add ebx, 0x5c26c926
adc edx, 0x2e1b2138
add eax,ebx
mov dword [ebp - 0x38],eax
adc ecx,edx
add dword [ebp - 0x2c],eax
mov dword [ebp - 0x8],ecx
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x10],ecx
push 0x27
push 0x22
push 0x1c
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x44]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x48]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x50]
mov esi,eax
mov edi,edx
push dword [ebp - 0x4c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x7c]
adc edi,dword [ebp - 0x80]
add dword [ebp - 0x0b4],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x0b8],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x4c]
push 0x22
adc edx,dword [ebp - 0x50]
add ebx, 0x5ac42aed
push 0x1c
adc edx, 0x4d2c6dfc
add eax,ebx
mov dword [ebp - 0x34],eax
adc ecx,edx
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
push dword [ebp - 0x8]
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0a4]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0a8]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x58]
mov esi,eax
mov edi,edx
push dword [ebp - 0x54]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x84]
adc edi,dword [ebp - 0x88]
add dword [ebp - 0x4c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x50],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x54]
push 0x22
adc edx,dword [ebp - 0x58]
add ebx, 0x9d95b3df
push 0x1c
push dword [ebp - 0x4]
adc edx, 0x53380d13
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [eax]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
mov edi,ecx
or edi,dword [ebp - 0x34]
mov esi,dword [ebp - 0x8]
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b8]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x0b4]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x60]
mov esi,eax
mov edi,edx
push dword [ebp - 0x5c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x90]
adc edi,dword [ebp - 0x8c]
add dword [ebp - 0x54],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x58],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x5c]
push 0x22
adc edx,dword [ebp - 0x60]
add ebx, 0x8baf63de
push 0x1c
push dword [ebp - 0x18]
adc edx, 0x650a7354
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x50]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x4c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x68]
mov esi,eax
mov edi,edx
push dword [ebp - 0x64]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a0]
adc edi,dword [ebp - 0x9c]
add dword [ebp - 0x5c],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x60],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x64]
push 0x22
adc edx,dword [ebp - 0x68]
add ebx, 0x3c77b2a8
push 0x1c
push dword [ebp - 0x14]
adc edx, 0x766a0abb
add eax,ebx
push dword [ebp - 0x30]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x30]
mov eax,dword [ebp - 0x18]
or edi,ecx
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x58]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x54]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x70]
mov esi,eax
mov edi,edx
push dword [ebp - 0x6c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0bc]
adc edi,dword [ebp - 0x0c0]
add dword [ebp - 0x64],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x68],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x6c]
push 0x22
adc edx,dword [ebp - 0x70]
add ebx, 0x47edaee6
push 0x1c
push dword [ebp - 0x10]
adc edx, 0x81c2c92e
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and edi,dword [ebp - 0x40]
and ecx,dword [ebp - 0x2c]
mov eax,dword [ebp - 0x14]
or edi,ecx
and esi,dword [ebp - 0x18]
and eax,dword [ebp - 0x10]
or esi,eax
push 0x6
push 0x3d
add ebx,edi
push 0x13
push dword [ebp - 0x60]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x5c]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x98]
mov esi,eax
mov edi,edx
push dword [ebp - 0x94]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x48]
adc edi,dword [ebp - 0x44]
add dword [ebp - 0x6c],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x70],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and ecx,dword [ebp - 0x34]
not esi
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x94]
push 0x22
adc edx,dword [ebp - 0x98]
add ebx, 0x1482353b
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0x92722c85
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and edi,dword [ebp - 0x30]
and ecx,dword [ebp - 0x28]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
or esi,eax
add ebx,edi
adc edx,esi
add dword [ebp - 0x24],ebx
adc dword [ebp - 0x1c],edx
push 0x6
push 0x3d
push 0x13
push dword [ebp - 0x68]
push dword [ebp - 0x64]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0ac]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a8]
adc edi,dword [ebp - 0x0a4]
add dword [ebp - 0x94],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x98],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0ac]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0b0]
add ebx, 0x4cf10364
push 0x22
adc edx, 0xa2bfe8a1
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x2c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
or esi,eax
add ebx,edi
push 0x6
adc edx,esi
add dword [ebp - 0x3c],ebx
push 0x3d
adc dword [ebp - 0x20],edx
push 0x13
push dword [ebp - 0x70]
push dword [ebp - 0x6c]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x78]
mov esi,eax
mov edi,edx
push dword [ebp - 0x74]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0b4]
adc edi,dword [ebp - 0x0b8]
add dword [ebp - 0x0ac],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0b0],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x74]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x78]
add ebx, 0xbc423001
push 0x22
adc edx, 0xa81a664b
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
or esi,dword [ebp - 0x1c]
mov ebx,eax
mov edi,dword [ebp - 0x3c]
or edi,dword [ebp - 0x24]
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
mov eax,dword [ebp - 0x20]
or edi,ecx
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x98]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x94]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
mov esi,eax
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x80]
mov edi,edx
push dword [ebp - 0x7c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x4c]
adc edi,dword [ebp - 0x50]
add dword [ebp - 0x74],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x78],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x7c]
push 0x22
adc edx,dword [ebp - 0x80]
add ebx, 0xd0f89791
push 0x1c
push dword [ebp - 0x8]
adc edx, 0xc24b8b70
add eax,ebx
push dword [ebp - 0x38]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and edi,dword [ebp - 0x24]
and ecx,dword [ebp - 0x38]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b0]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0ac]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
mov esi,eax
mov edi,edx
push 0x1
push dword [ebp - 0x88]
push dword [ebp - 0x84]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x54]
adc edi,dword [ebp - 0x58]
add dword [ebp - 0x7c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x80],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x84]
push 0x22
adc edx,dword [ebp - 0x88]
add ebx, 0x654be30
push 0x1c
push dword [ebp - 0x4]
adc edx, 0xc76c51a3
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x78]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x74]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x8c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x90]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
push 0x29
adc edi,edx
add esi,dword [ebp - 0x5c]
push 0x12
adc edi,dword [ebp - 0x60]
add dword [ebp - 0x84],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x88],edi
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x90]
push 0x22
adc edx,dword [ebp - 0x8c]
add ebx, 0xd6ef5218
push 0x1c
push dword [ebp - 0x18]
adc edx, 0xd192e819
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x80]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x7c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x9c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a0]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x64]
adc edi,dword [ebp - 0x68]
add dword [ebp - 0x90],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x8c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0a0]
push 0x22
adc edx,dword [ebp - 0x9c]
add ebx, 0x5565a910
push 0x1c
push dword [ebp - 0x14]
adc edx, 0xd6990624
add eax,ebx
push dword [ebp - 0x30]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x88]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x84]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0c0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0bc]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x6c]
adc edi,dword [ebp - 0x70]
add dword [ebp - 0x0a0],esi
push 0x29
adc dword [ebp - 0x9c],edi
push 0x12
mov esi,dword [ebp - 0x8]
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0bc]
push 0x22
adc edx,dword [ebp - 0x0c0]
add ebx, 0x5771202a
push 0x1c
push dword [ebp - 0x10]
adc edx, 0xf40e3585
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x8c]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x90]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x44]
mov esi,eax
mov edi,edx
push dword [ebp - 0x48]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x94]
adc edi,dword [ebp - 0x98]
add dword [ebp - 0x0bc],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x0c0],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and ecx,dword [ebp - 0x34]
not esi
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x48]
push 0x22
adc edx,dword [ebp - 0x44]
add ebx, 0x32bbd1b8
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0x106aa070
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x9c]
adc edx,esi
add dword [ebp - 0x24],ebx
push dword [ebp - 0x0a0]
adc dword [ebp - 0x1c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0a4]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a8]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0ac]
adc edi,dword [ebp - 0x0b0]
add dword [ebp - 0x48],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x44],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
not edi
and edi,dword [ebp - 0x38]
not esi
and esi,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x34]
and ecx,dword [ebp - 0x40]
mov eax,dword [ebp - 0x4]
xor edi,ecx
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0a8]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0a4]
add ebx, 0xb8d2d0c8
push 0x22
adc edx, 0x19a4c116
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x24]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0c0]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x0bc]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b8]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0b4]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x74]
adc edi,dword [ebp - 0x78]
add dword [ebp - 0x0a8],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0a4],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
not esi
not edi
mov ebx,eax
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x40]
and ecx,dword [ebp - 0x30]
mov eax,dword [ebp - 0x18]
xor edi,ecx
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0b4]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x0b8]
add ebx, 0x5141ab53
push 0x22
adc edx, 0x1e376c08
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x44]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x48]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x50]
mov esi,eax
mov edi,edx
push dword [ebp - 0x4c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x7c]
adc edi,dword [ebp - 0x80]
add dword [ebp - 0x0b4],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x0b8],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
mov eax,dword [ebp - 0x14]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x2c]
and eax,dword [ebp - 0x10]
xor edi,ecx
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
mov ecx,dword [ebp - 0x4]
adc edx,esi
add ebx,dword [ebp - 0x4c]
push 0x27
adc edx,dword [ebp - 0x50]
add ebx, 0xdf8eeb99
push 0x22
adc edx, 0x2748774c
add eax,ebx
push 0x1c
push dword [ebp - 0x8]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
push dword [ebp - 0x38]
adc dword [ebp - 0x0c],ecx
mov dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0a4]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0a8]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x58]
mov esi,eax
mov edi,edx
push dword [ebp - 0x54]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x84]
adc edi,dword [ebp - 0x88]
add dword [ebp - 0x4c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x50],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and edi,dword [ebp - 0x30]
not esi
and ecx,dword [ebp - 0x28]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x54]
mov eax,dword [ebp - 0x40]
adc edx,dword [ebp - 0x58]
add ebx, 0xe19b48a8
mov ecx,dword [ebp - 0x18]
push 0x27
adc edx, 0x34b0bcb5
add eax,ebx
push 0x22
push 0x1c
push dword [ebp - 0x4]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
push dword [ebp - 0x34]
adc dword [ebp - 0x1c],ecx
mov dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b8]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x0b4]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x60]
mov esi,eax
mov edi,edx
push dword [ebp - 0x5c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x90]
adc edi,dword [ebp - 0x8c]
add dword [ebp - 0x54],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x58],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and edi,dword [ebp - 0x2c]
not esi
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x5c]
adc edx,dword [ebp - 0x60]
add ebx, 0xc5c95a63
adc edx, 0x391c0cb3
mov eax,dword [ebp - 0x30]
mov ecx,dword [ebp - 0x14]
add eax,ebx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x18]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
push dword [ebp - 0x40]
adc dword [ebp - 0x20],ecx
mov dword [ebp - 0x14],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x50]
adc edx,esi
add dword [ebp - 0x30],ebx
push dword [ebp - 0x4c]
adc dword [ebp - 0x14],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x68]
mov esi,eax
mov edi,edx
push dword [ebp - 0x64]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a0]
adc edi,dword [ebp - 0x9c]
add dword [ebp - 0x5c],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x60],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x64]
adc edx,dword [ebp - 0x68]
add ebx, 0xe3418acb
adc edx, 0x4ed8aa4a
add eax,ebx
mov dword [ebp - 0x2c],eax
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x14]
adc ecx,edx
add dword [ebp - 0x38],eax
push dword [ebp - 0x30]
adc dword [ebp - 0x8],ecx
mov dword [ebp - 0x10],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x58]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x54]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x70]
mov esi,eax
mov edi,edx
push dword [ebp - 0x6c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0bc]
adc edi,dword [ebp - 0x0c0]
add dword [ebp - 0x64],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x68],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x6c]
adc edx,dword [ebp - 0x70]
add ebx, 0x7763e373
adc edx, 0x5b9cca4f
add eax,ebx
mov dword [ebp - 0x28],eax
adc ecx,edx
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x10]
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x60]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x5c]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x98]
mov esi,eax
mov edi,edx
push dword [ebp - 0x94]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x48]
adc edi,dword [ebp - 0x44]
add dword [ebp - 0x6c],esi
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x70],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and ecx,dword [ebp - 0x34]
not esi
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x94]
push 0x22
adc edx,dword [ebp - 0x98]
add ebx, 0xd6b2b8a3
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0x682e6ff3
add eax,ebx
mov dword [ebp - 0x24],eax
adc ecx,edx
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x68]
adc edx,esi
add dword [ebp - 0x24],ebx
push dword [ebp - 0x64]
adc dword [ebp - 0x1c],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0b0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0ac]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0a8]
adc edi,dword [ebp - 0x0a4]
add dword [ebp - 0x94],esi
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x98],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0ac]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0b0]
add ebx, 0x5defb2fc
push 0x22
adc edx, 0x748f82ee
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ebx,eax
mov edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x24]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x70]
adc edx,esi
add dword [ebp - 0x3c],ebx
push dword [ebp - 0x6c]
adc dword [ebp - 0x20],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x78]
mov esi,eax
mov edi,edx
push dword [ebp - 0x74]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x0b4]
adc edi,dword [ebp - 0x0b8]
add dword [ebp - 0x0ac],esi
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x0b0],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x74]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x78]
add ebx, 0x43172f60
push 0x22
adc edx, 0x78a5636f
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x98]
adc edx,esi
add dword [ebp - 0x38],ebx
push dword [ebp - 0x94]
adc dword [ebp - 0x8],edx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x80]
mov esi,eax
mov edi,edx
push dword [ebp - 0x7c]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x4c]
adc edi,dword [ebp - 0x50]
add dword [ebp - 0x74],esi
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x78],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x7c]
push 0x22
adc edx,dword [ebp - 0x80]
add ebx, 0xa1f0ab72
push 0x1c
push dword [ebp - 0x8]
adc edx, 0x84c87814
add eax,ebx
push dword [ebp - 0x38]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x38]
mov eax,dword [ebp - 0x20]
or edi,ecx
and eax,dword [ebp - 0x8]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x0b0]
adc edx,esi
add dword [ebp - 0x34],ebx
push dword [ebp - 0x0ac]
adc dword [ebp - 0x4],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x88]
mov esi,eax
mov edi,edx
push dword [ebp - 0x84]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x54]
adc edi,dword [ebp - 0x58]
add dword [ebp - 0x7c],esi
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x80],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x84]
push 0x22
adc edx,dword [ebp - 0x88]
add ebx, 0x1a6439ec
push 0x1c
push dword [ebp - 0x4]
adc edx, 0x8cc70208
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
or esi,dword [ebp - 0x4]
and edi,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x34]
mov eax,dword [ebp - 0x8]
or edi,ecx
and esi,dword [ebp - 0x20]
and eax,dword [ebp - 0x4]
or esi,eax
push 0x6
push 0x3d
add ebx,edi
push 0x13
push dword [ebp - 0x78]
adc edx,esi
add dword [ebp - 0x40],ebx
push dword [ebp - 0x74]
adc dword [ebp - 0x18],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x8c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x90]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x5c]
adc edi,dword [ebp - 0x60]
add dword [ebp - 0x84],esi
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x88],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x90]
push 0x22
adc edx,dword [ebp - 0x8c]
add ebx, 0x23631e28
push 0x1c
push dword [ebp - 0x18]
adc edx, 0x90befffa
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and edi,dword [ebp - 0x38]
and ecx,dword [ebp - 0x40]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
or esi,eax
add ebx,edi
adc edx,esi
add dword [ebp - 0x30],ebx
adc dword [ebp - 0x14],edx
push 0x6
push 0x3d
push 0x13
push dword [ebp - 0x80]
push dword [ebp - 0x7c]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x9c]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0a0]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x64]
adc edi,dword [ebp - 0x68]
add dword [ebp - 0x90],esi
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x8c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0a0]
push 0x22
adc edx,dword [ebp - 0x9c]
add ebx, 0xde82bde9
push 0x1c
push dword [ebp - 0x14]
adc edx, 0xa4506ceb
add eax,ebx
push dword [ebp - 0x30]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
mov ebx,eax
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
or esi,dword [ebp - 0x14]
and edi,dword [ebp - 0x34]
and ecx,dword [ebp - 0x30]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
or esi,eax
add ebx,edi
push 0x6
adc edx,esi
add dword [ebp - 0x2c],ebx
push 0x3d
adc dword [ebp - 0x10],edx
push 0x13
push dword [ebp - 0x88]
push dword [ebp - 0x84]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0c0]
mov esi,eax
mov edi,edx
push dword [ebp - 0x0bc]
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
add esi,eax
adc edi,edx
add esi,dword [ebp - 0x6c]
adc edi,dword [ebp - 0x70]
add dword [ebp - 0x0a0],esi
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x9c],edi
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x38]
not esi
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0bc]
push 0x22
adc edx,dword [ebp - 0x0c0]
add ebx, 0xb2c67915
push 0x1c
push dword [ebp - 0x10]
adc edx, 0xbef9a3f7
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x2c]
mov eax,dword [ebp - 0x14]
or edi,ecx
and eax,dword [ebp - 0x10]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x28],ebx
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x0c],edx
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
mov ebx,eax
mov edi,dword [ebp - 0x34]
not esi
and esi,dword [ebp - 0x20]
not edi
and edi,dword [ebp - 0x3c]
mov ecx,dword [ebp - 0x38]
and ecx,dword [ebp - 0x34]
mov eax,dword [ebp - 0x8]
xor edi,ecx
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x48]
push 0x22
adc edx,dword [ebp - 0x44]
add ebx, 0xe372532b
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0xc67178f2
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
add esp, 0x50
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
mov ebx,eax
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x24],ebx
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x1c],edx
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x0a8]
adc edx,dword [ebp - 0x0a4]
add ebx, 0xea26619c
adc edx, 0xca273ece
add eax,ebx
mov dword [ebp - 0x3c],eax
adc ecx,edx
add dword [ebp - 0x30],eax
mov dword [ebp - 0x20],ecx
mov esi,dword [ebp - 0x24]
adc dword [ebp - 0x14],ecx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x1c]
push esi
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x24]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
or edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x3c],ebx
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x20],edx
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x30]
mov ebx,eax
mov ecx,dword [ebp - 0x40]
not edi
and ecx,dword [ebp - 0x30]
not esi
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
xor edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0b4]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x0b8]
add ebx, 0x21c0c207
push 0x22
adc edx, 0xd186b8c7
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
add esp, 0x50
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x28]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
or esi,eax
add ebx,edi
adc edx,esi
add dword [ebp - 0x38],ebx
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x8],edx
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x4c]
push 0x22
adc edx,dword [ebp - 0x50]
add ebx, 0xcde0eb1e
push 0x1c
push dword [ebp - 0x8]
adc edx, 0xeada7dd6
add eax,ebx
push dword [ebp - 0x38]
adc ecx,edx
mov dword [ebp - 0x34],eax
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x34],ebx
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x4],edx
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and edi,dword [ebp - 0x30]
not esi
and ecx,dword [ebp - 0x28]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x54]
adc edx,dword [ebp - 0x58]
add ebx, 0xee6ed178
mov eax,dword [ebp - 0x40]
adc edx, 0xf57d4f7f
mov ecx,dword [ebp - 0x18]
add eax,ebx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x4]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
push dword [ebp - 0x34]
adc dword [ebp - 0x1c],ecx
mov dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
add esp, 0x50
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
mov ebx,eax
or esi,dword [ebp - 0x4]
and ecx,dword [ebp - 0x34]
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x40],ebx
mov esi,dword [ebp - 0x1c]
adc dword [ebp - 0x18],edx
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x5c]
push 0x22
adc edx,dword [ebp - 0x60]
add ebx, 0x72176fba
push 0x1c
push dword [ebp - 0x18]
adc edx, 0x6f067aa
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x40]
mov eax,dword [ebp - 0x4]
or edi,ecx
and eax,dword [ebp - 0x18]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x30],ebx
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x14],edx
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x10]
xor esi,eax
mov eax,dword [ebp - 0x2c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x64]
push 0x22
adc edx,dword [ebp - 0x68]
add ebx, 0xa2c898a6
push 0x1c
push dword [ebp - 0x14]
adc edx, 0x0a637dc5
add eax,ebx
push dword [ebp - 0x30]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
mov dword [ebp - 0x10],ecx
adc dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
add esp, 0x50
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
mov ebx,eax
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x2c],ebx
mov esi,dword [ebp - 0x8]
adc dword [ebp - 0x10],edx
push 0x0e
push esi
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x38]
not esi
mov ecx,dword [ebp - 0x3c]
not edi
and edi,dword [ebp - 0x24]
mov ebx,eax
and esi,dword [ebp - 0x1c]
and ecx,dword [ebp - 0x38]
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
xor edi,ecx
mov ecx,dword [ebp - 0x0c]
xor esi,eax
mov eax,dword [ebp - 0x28]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x6c]
push 0x22
adc edx,dword [ebp - 0x70]
add ebx, 0xbef90dae
push 0x1c
push dword [ebp - 0x10]
adc edx, 0x113f9804
add eax,ebx
push dword [ebp - 0x2c]
adc ecx,edx
mov dword [ebp - 0x28],eax
add dword [ebp - 0x34],eax
mov dword [ebp - 0x0c],ecx
adc dword [ebp - 0x4],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
mov ebx,eax
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x28],ebx
mov esi,dword [ebp - 0x4]
adc dword [ebp - 0x0c],edx
push 0x0e
push esi
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x34]
mov ebx,eax
mov ecx,dword [ebp - 0x38]
not edi
and ecx,dword [ebp - 0x34]
not esi
and edi,dword [ebp - 0x3c]
and esi,dword [ebp - 0x20]
xor edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
mov ecx,dword [ebp - 0x1c]
xor esi,eax
mov eax,dword [ebp - 0x24]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x94]
push 0x22
adc edx,dword [ebp - 0x98]
add ebx, 0x131c471b
push 0x1c
push dword [ebp - 0x0c]
adc edx, 0x1b710b35
add eax,ebx
push dword [ebp - 0x28]
adc ecx,edx
mov dword [ebp - 0x24],eax
add dword [ebp - 0x40],eax
mov dword [ebp - 0x1c],ecx
adc dword [ebp - 0x18],ecx
mov eax, [esp]
mov eax, [eax]
add esp, 0x50
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
mov edi,ecx
or edi,dword [ebp - 0x28]
mov esi,dword [ebp - 0x10]
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x24],ebx
mov esi,dword [ebp - 0x18]
adc dword [ebp - 0x1c],edx
push 0x0e
push esi
push dword [ebp - 0x40]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x40]
mov ebx,eax
mov ecx,dword [ebp - 0x34]
not edi
and ecx,dword [ebp - 0x40]
not esi
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
xor edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
mov ecx,dword [ebp - 0x20]
xor esi,eax
mov eax,dword [ebp - 0x3c]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x0ac]
mov esi,dword [ebp - 0x24]
adc edx,dword [ebp - 0x0b0]
add ebx, 0x23047d84
push 0x22
adc edx, 0x28db77f5
add eax,ebx
push 0x1c
push dword [ebp - 0x1c]
adc ecx,edx
mov dword [ebp - 0x3c],eax
add dword [ebp - 0x30],eax
push esi
adc dword [ebp - 0x14],ecx
mov dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
or edi,esi
mov esi,dword [ebp - 0x0c]
or esi,dword [ebp - 0x1c]
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x24]
mov eax,dword [ebp - 0x0c]
or edi,ecx
and eax,dword [ebp - 0x1c]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x3c],ebx
mov esi,dword [ebp - 0x14]
adc dword [ebp - 0x20],edx
push 0x0e
push esi
push dword [ebp - 0x30]
mov eax, [esp]
mov eax, [eax]
mov ebx,eax
mov edi,dword [ebp - 0x30]
not esi
and esi,dword [ebp - 0x4]
not edi
and edi,dword [ebp - 0x34]
mov ecx,dword [ebp - 0x40]
and ecx,dword [ebp - 0x30]
mov eax,dword [ebp - 0x18]
xor edi,ecx
and eax,dword [ebp - 0x14]
mov ecx,dword [ebp - 0x8]
xor esi,eax
mov eax,dword [ebp - 0x38]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x74]
mov esi,dword [ebp - 0x20]
adc edx,dword [ebp - 0x78]
add ebx, 0x40c72493
push 0x22
adc edx, 0x32caab7b
add eax,ebx
push 0x1c
adc ecx,edx
mov dword [ebp - 0x38],eax
add dword [ebp - 0x2c],eax
push esi
push dword [ebp - 0x3c]
adc dword [ebp - 0x10],ecx
mov dword [ebp - 0x8],ecx
mov eax, [esp]
mov eax, [eax]
or esi,dword [ebp - 0x1c]
add esp, 0x50
mov edi,dword [ebp - 0x3c]
mov ebx,eax
or edi,dword [ebp - 0x24]
and edi,dword [ebp - 0x28]
and esi,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x24]
mov eax,dword [ebp - 0x20]
or edi,ecx
and eax,dword [ebp - 0x1c]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x38],ebx
mov esi,dword [ebp - 0x10]
adc dword [ebp - 0x8],edx
push 0x0e
push esi
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x2c]
mov ebx,eax
mov ecx,dword [ebp - 0x30]
not edi
and ecx,dword [ebp - 0x2c]
not esi
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
xor edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
mov ecx,dword [ebp - 0x4]
xor esi,eax
mov eax,dword [ebp - 0x34]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x7c]
adc edx,dword [ebp - 0x80]
add ebx, 0x15c9bebc
adc edx, 0x3c9ebe0a
add eax,ebx
mov dword [ebp - 0x34],eax
adc ecx,edx
add dword [ebp - 0x28],eax
mov dword [ebp - 0x4],ecx
adc dword [ebp - 0x0c],ecx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x8]
push dword [ebp - 0x38]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x3c]
mov ebx,eax
mov esi,dword [ebp - 0x20]
mov edi,ecx
or edi,dword [ebp - 0x38]
or esi,dword [ebp - 0x8]
and ecx,dword [ebp - 0x38]
and edi,dword [ebp - 0x24]
and esi,dword [ebp - 0x1c]
or edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x8]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x34],ebx
mov esi,dword [ebp - 0x0c]
adc dword [ebp - 0x4],edx
push 0x0e
push esi
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x28]
mov ebx,eax
mov ecx,dword [ebp - 0x2c]
not edi
and ecx,dword [ebp - 0x28]
not esi
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
xor edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
mov ecx,dword [ebp - 0x18]
xor esi,eax
mov eax,dword [ebp - 0x40]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x84]
push 0x22
adc edx,dword [ebp - 0x88]
add ebx, 0x9c100d4c
push 0x1c
push dword [ebp - 0x4]
adc edx, 0x431d67c4
add eax,ebx
push dword [ebp - 0x34]
adc ecx,edx
mov dword [ebp - 0x40],eax
add dword [ebp - 0x24],eax
mov dword [ebp - 0x18],ecx
adc dword [ebp - 0x1c],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x38]
mov ebx,eax
mov esi,dword [ebp - 0x8]
mov edi,ecx
or edi,dword [ebp - 0x34]
add esp, 0x50
or esi,dword [ebp - 0x4]
and edi,dword [ebp - 0x3c]
and ecx,dword [ebp - 0x34]
and esi,dword [ebp - 0x20]
or edi,ecx
mov eax,dword [ebp - 0x8]
and eax,dword [ebp - 0x4]
or esi,eax
add ebx,edi
adc edx,esi
add dword [ebp - 0x40],ebx
adc dword [ebp - 0x18],edx
mov esi,dword [ebp - 0x1c]
push 0x29
push 0x12
push 0x0e
push esi
push dword [ebp - 0x24]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x24]
mov ebx,eax
mov ecx,dword [ebp - 0x28]
not edi
and ecx,dword [ebp - 0x24]
not esi
and edi,dword [ebp - 0x2c]
and esi,dword [ebp - 0x10]
xor edi,ecx
mov eax,dword [ebp - 0x0c]
and eax,dword [ebp - 0x1c]
mov ecx,dword [ebp - 0x14]
xor esi,eax
mov eax,dword [ebp - 0x30]
add ebx,edi
push 0x27
adc edx,esi
add ebx,dword [ebp - 0x90]
push 0x22
adc edx,dword [ebp - 0x8c]
add ebx, 0xcb3e42b6
push 0x1c
push dword [ebp - 0x18]
adc edx, 0x4cc5d4be
add eax,ebx
push dword [ebp - 0x40]
adc ecx,edx
mov dword [ebp - 0x30],eax
add dword [ebp - 0x3c],eax
mov dword [ebp - 0x14],ecx
adc dword [ebp - 0x20],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x34]
mov ebx,eax
mov esi,dword [ebp - 0x4]
mov edi,ecx
or edi,dword [ebp - 0x40]
or esi,dword [ebp - 0x18]
and ecx,dword [ebp - 0x40]
and edi,dword [ebp - 0x38]
and esi,dword [ebp - 0x8]
or edi,ecx
mov eax,dword [ebp - 0x4]
and eax,dword [ebp - 0x18]
or esi,eax
add ebx,edi
push 0x29
push 0x12
adc edx,esi
add dword [ebp - 0x30],ebx
mov esi,dword [ebp - 0x20]
adc dword [ebp - 0x14],edx
push 0x0e
push esi
push dword [ebp - 0x3c]
mov eax, [esp]
mov eax, [eax]
mov edi,dword [ebp - 0x3c]
mov ebx,eax
mov ecx,dword [ebp - 0x3c]
not edi
and edi,dword [ebp - 0x28]
not esi
and ecx,dword [ebp - 0x24]
and esi,dword [ebp - 0x0c]
xor edi,ecx
mov eax,dword [ebp - 0x20]
and eax,dword [ebp - 0x1c]
xor esi,eax
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x0a0]
adc edx,dword [ebp - 0x9c]
add ebx, 0xfc657e2a
adc edx, 0x597f299c
mov eax,dword [ebp - 0x2c]
mov ecx,dword [ebp - 0x10]
add eax,ebx
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x14]
adc ecx,edx
mov dword [ebp - 0x2c],eax
add dword [ebp - 0x38],eax
push dword [ebp - 0x30]
adc dword [ebp - 0x8],ecx
mov dword [ebp - 0x10],ecx
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x40]
add esp, 0x50
mov esi,dword [ebp - 0x18]
mov edi,ecx
or edi,dword [ebp - 0x30]
mov ebx,eax
or esi,dword [ebp - 0x14]
and ecx,dword [ebp - 0x30]
and edi,dword [ebp - 0x34]
and esi,dword [ebp - 0x4]
or edi,ecx
mov eax,dword [ebp - 0x18]
and eax,dword [ebp - 0x14]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x8c]
adc edx,esi
add dword [ebp - 0x2c],ebx
push dword [ebp - 0x90]
adc dword [ebp - 0x10],edx
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x44]
mov edi,eax
mov ebx,edx
push dword [ebp - 0x48]
mov eax, [esp]
mov eax, [eax]
push 0x29
push 0x12
push 0x0e
push dword [ebp - 0x8]
add edi,eax
push dword [ebp - 0x38]
adc ebx,edx
mov eax, [esp]
mov eax, [eax]
mov esi,dword [ebp - 0x38]
add edi,eax
mov ecx,dword [ebp - 0x3c]
not esi
mov eax,dword [ebp - 0x20]
adc ebx,edx
and ecx,dword [ebp - 0x38]
and esi,dword [ebp - 0x24]
and eax,dword [ebp - 0x8]
xor esi,ecx
mov edx,dword [ebp - 0x8]
mov ecx,dword [ebp - 0x0c]
not edx
and edx,dword [ebp - 0x1c]
xor edx,eax
mov eax,dword [ebp - 0x28]
add edi,esi
adc ebx,edx
add edi,dword [ebp - 0x0bc]
adc ebx,dword [ebp - 0x0c0]
add edi,dword [ebp - 0x94]
adc ebx,dword [ebp - 0x98]
add edi, 0x3ad6faec
adc ebx, 0x5fcb6fab
add eax,edi
mov dword [ebp - 0x28],eax
adc ecx,ebx
add dword [ebp - 0x34],eax
push 0x27
adc dword [ebp - 0x4],ecx
push 0x22
push 0x1c
push dword [ebp - 0x10]
mov dword [ebp - 0x0c],ecx
push dword [ebp - 0x2c]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x30]
add esp, 0x50
mov esi,dword [ebp - 0x14]
mov edi,ecx
or edi,dword [ebp - 0x2c]
mov ebx,eax
or esi,dword [ebp - 0x10]
and ecx,dword [ebp - 0x2c]
and edi,dword [ebp - 0x40]
and esi,dword [ebp - 0x18]
or edi,ecx
mov eax,dword [ebp - 0x14]
and eax,dword [ebp - 0x10]
push 0x6
or esi,eax
add ebx,edi
push 0x3d
push 0x13
push dword [ebp - 0x9c]
adc edx,esi
add dword [ebp - 0x28],ebx
push dword [ebp - 0x0a0]
adc dword [ebp - 0x0c],edx
mov eax, [esp]
mov eax, [eax]
push 0x29
push 0x12
push 0x0e
push dword [ebp - 0x4]
mov edi,eax
mov ebx,edx
push dword [ebp - 0x34]
mov eax, [esp]
mov eax, [eax]
push 0x7
push 0x8
push 0x1
push dword [ebp - 0x0a4]
add edi,eax
push dword [ebp - 0x0a8]
adc ebx,edx
mov eax, [esp]
mov eax, [eax]
mov esi,dword [ebp - 0x34]
add edi,eax
mov ecx,dword [ebp - 0x38]
not esi
mov eax,dword [ebp - 0x8]
adc ebx,edx
and ecx,dword [ebp - 0x34]
and eax,dword [ebp - 0x4]
and esi,dword [ebp - 0x3c]
mov edx,dword [ebp - 0x4]
xor esi,ecx
mov ecx,dword [ebp - 0x1c]
not edx
and edx,dword [ebp - 0x20]
xor edx,eax
mov eax,dword [ebp - 0x24]
add edi,esi
adc ebx,edx
add edi,dword [ebp - 0x48]
adc ebx,dword [ebp - 0x44]
add edi,dword [ebp - 0x0ac]
adc ebx,dword [ebp - 0x0b0]
add edi, 0x4a475817
adc ebx, 0x6c44198c
add eax,edi
mov dword [ebp - 0x24],eax
adc ecx,ebx
add dword [ebp - 0x40],eax
mov eax,dword [ebp + 0x8]
adc dword [ebp - 0x18],ecx
mov dword [ebp - 0x1c],ecx
mov eax,dword [eax]
push 0x27
push 0x22
push 0x1c
push dword [ebp - 0x0c]
mov dword [ebp - 0x44],eax
push dword [ebp - 0x28]
mov eax, [esp]
mov eax, [eax]
mov ecx,dword [ebp - 0x2c]
mov ebx,eax
mov esi,dword [ebp - 0x10]
mov edi,ecx
or edi,dword [ebp - 0x28]
add esp, 0x50
or esi,dword [ebp - 0x0c]
and ecx,dword [ebp - 0x28]
and edi,dword [ebp - 0x30]
and esi,dword [ebp - 0x14]
or edi,ecx
mov eax,dword [ebp - 0x10]
and eax,dword [ebp - 0x0c]
or esi,eax
mov eax,dword [ebp - 0x44]
add ebx,edi
adc edx,esi
add ebx,dword [ebp - 0x24]
adc edx,dword [ebp - 0x1c]
add dword [eax],ebx
mov ecx,dword [eax]
adc dword [eax + 0x4],edx
mov eax,dword [eax + 0x4]
mov edx,dword [ebp - 0x28]
mov dword [ebp - 0x1c],eax
mov dword [ebp - 0x24],ecx
mov ecx,dword [ebp + 0x8]
mov eax,dword [ecx]
add dword [eax + 0x8],edx
mov edx,dword [ebp - 0x0c]
adc dword [eax + 0x0c],edx
mov edx,dword [eax + 0x8]
mov eax,dword [eax + 0x0c]
mov dword [ebp - 0x28],edx
mov edx,dword [ebp - 0x2c]
mov dword [ebp - 0x0c],eax
mov eax,dword [ecx]
add dword [eax + 0x10],edx
mov edx,dword [ebp - 0x10]
adc dword [eax + 0x14],edx
mov edx,dword [eax + 0x10]
mov eax,dword [eax + 0x14]
mov dword [ebp - 0x10],eax
mov eax,dword [ecx]
mov dword [ebp - 0x2c],edx
mov edx,dword [ebp - 0x30]
add dword [eax + 0x18],edx
mov edx,dword [ebp - 0x14]
adc dword [eax + 0x1c],edx
mov edx,dword [eax + 0x18]
mov eax,dword [eax + 0x1c]
mov dword [ebp - 0x14],eax
mov eax,dword [ecx]
mov dword [ebp - 0x30],edx
mov edx,dword [ebp - 0x40]
add dword [eax + 0x20],edx
mov esi,dword [eax + 0x20]
mov edx,dword [ebp - 0x18]
adc dword [eax + 0x24],edx
mov ebx,dword [eax + 0x24]
mov eax,dword [ecx]
mov edx,dword [ebp - 0x34]
mov dword [ebp - 0x40],esi
mov dword [ebp - 0x18],ebx
add dword [eax + 0x28],edx
mov edx,dword [ebp - 0x4]
adc dword [eax + 0x2c],edx
mov edx,dword [eax + 0x28]
mov eax,dword [eax + 0x2c]
mov dword [ebp - 0x4],eax
mov eax,dword [ecx]
mov dword [ebp - 0x34],edx
mov edx,dword [ebp - 0x38]
add dword [eax + 0x30],edx
mov edx,dword [ebp - 0x8]
adc dword [eax + 0x34],edx
mov edx,dword [eax + 0x30]
mov eax,dword [eax + 0x34]
mov edi,dword [ebp - 0x0c4]
mov dword [ebp - 0x8],eax
mov eax,dword [ecx]
mov ecx,dword [ebp - 0x3c]
mov dword [ebp - 0x38],edx
add dword [eax + 0x38],ecx
mov ecx,dword [ebp - 0x20]
adc dword [eax + 0x3c],ecx
sub edi,-80
sub dword [ebp + 0x10],1
mov ecx,dword [eax + 0x38]
mov eax,dword [eax + 0x3c]
mov dword [ebp - 0x3c],ecx
mov dword [ebp - 0x20],eax
mov dword [ebp - 0x0c4],edi
jne slow4_1
pop edi
slow4_2:
pop esi
pop ebx
mov esp,ebp
pop ebp
ret

; Fuzzer-generated semi-slow input
slow5:
and byte [eax], al
and byte [ebx], dl
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc [ebx], edx
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
add byte [eax], al
add byte [eax], al
adc byte [eax], al
add bh, bh
dec eax
dec eax
cwde
cwde
cwde
add byte [eax], al
add byte [eax], al
add byte [eax], al
add byte [ecx], bl
cwde
js .end1
js .end2
js .end3
js .end4
js .end5
js .end6
js .end7
sub bh, byte [eax+78]
js .end8
js .end9
js .end10
js .end11
js .end12
js .end13
js .end14
adc edx, [ebx]
adc ebp, [eax]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc edx, [ebx]
adc eax, [eax]
and byte [eax],al
add byte [eax],al
add byte [eax],al
add byte [eax],al
sbb al, byte [edx]
stosb ; mov byte [edi], al (No REP)
add byte [edi+0x6a000000], bl
jne slow5
ret
.end1:
ret
.end2:
ret
.end3:
ret
.end4:
ret
.end5:
ret
.end6:
ret
.end7:
ret
.end8:
ret
.end9:
ret
.end10:
ret
.end11:
ret
.end12:
ret
.end13:
ret
.end14:
ret

align 0x100
; Fuzzer-generated case that made scarf get stuck
slow6:
inc eax
and byte [ebp + 0x2d], dh
sub eax, 0x2d2d2d2d
sub eax, 0x7575752d
sub eax, 0x752d2d2d
xor [eax], eax
add byte [eax + 0x20], al
add byte [eax], al
inc eax
sub byte [eax], al
add [ecx], ecx
add bh, bh
add byte [eax + 0x20], al
sub eax, 0x2d2d2d2d
sub eax, 0x7575752d
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
div edi
jne $ + 0x800000 ; Jumps that go out of bounds
jne $ + 0x800000
jne $ + 0x800000
sub eax, 0x752d2d2d
jne $ + 0x800000
jne $ + 0x800000
xor byte [0x2d2d2d2d], ch
sub eax, 0x2d00ff2d
and byte [ebp + 0x40], dh
or byte [eax], cl
or byte [eax], cl
add byte [ecx], cl
add byte [ecx], al
add byte [eax], al
ret
