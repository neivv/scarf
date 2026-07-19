; nasm -f bin slow_64.asm -o slow_64.bin
bits 64

dd hash 

hash:
mov qword [rsp + 0x10], rdx
mov qword [rsp + 8], rcx
push rbx
push rbp
push rsi
push rdi
push r12
push r13
push r14
push r15
sub rsp, 0x48
mov ebx, dword [rcx]
mov r15, rdx
mov r11d, dword [rcx + 4]
mov r10d, dword [rcx + 8]
mov r9d, dword [rcx + 0xc]
mov r8d, dword [rcx + 0x10]
mov ecx, r9d
mov eax, dword [rdx]
xor ecx, r10d
mov edx, eax
and ecx, r11d
xor ecx, r9d
rol eax, 8
and eax, 0xff00ff
rol edx, 0x18
add r8d, 0x5a827999
rol r11d, 0x1e
add r9d, 0x5a827999
and edx, 0xff00ff00
or edx, eax
mov eax, ebx
rol eax, 5
mov dword [rsp + 0x18], edx
add edx, eax
mov eax, dword [r15 + 4]
add edx, ecx
add edx, r8d
mov ecx, r10d
mov r8d, eax
xor ecx, r11d
rol eax, 8
and ecx, ebx
xor ecx, r10d
rol r8d, 0x18
and eax, 0xff00ff
rol ebx, 0x1e
add r10d, 0x5a827999
and r8d, 0xff00ff00
or r8d, eax
mov eax, edx
rol eax, 5
mov dword [rsp + 0x10], r8d
add r8d, eax
mov eax, dword [r15 + 8]
add r8d, ecx
add r8d, r9d
mov ecx, r11d
mov r9d, eax
xor ecx, ebx
rol eax, 8
and ecx, edx
and eax, 0xff00ff
rol r9d, 0x18
xor ecx, r11d
rol edx, 0x1e
and r9d, 0xff00ff00
add r11d, 0x5a827999
or r9d, eax
mov eax, r8d
rol eax, 5
mov dword [rsp + 0x30], r9d
add r9d, eax
mov eax, dword [r15 + 0xc]
add r9d, ecx
add r9d, r10d
mov ecx, edx
mov r10d, eax
xor ecx, ebx
rol r10d, 0x18
and ecx, r8d
rol eax, 8
and r10d, 0xff00ff00
and eax, 0xff00ff
xor ecx, ebx
or r10d, eax
mov eax, r9d
mov dword [rsp + 4], r10d
rol eax, 5
add r10d, eax
add r10d, ecx
add r10d, r11d
mov eax, dword [r15 + 0x10]
add ebx, 0x5a827999
mov r11d, eax
rol r8d, 0x1e
rol eax, 8
mov ecx, edx
and eax, 0xff00ff
rol r11d, 0x18
xor ecx, r8d
and r11d, 0xff00ff00
or r11d, eax
and ecx, r9d
xor ecx, edx
rol r9d, 0x1e
mov dword [rsp + 0x38], r11d
mov eax, r10d
rol eax, 5
add r11d, eax
mov eax, dword [r15 + 0x14]
add r11d, ecx
mov ecx, r8d
add r11d, ebx
xor ecx, r9d
mov ebx, eax
and ecx, r10d
rol eax, 8
xor ecx, r8d
and eax, 0xff00ff
rol ebx, 0x18
and ebx, 0xff00ff00
rol r10d, 0x1e
or ebx, eax
mov eax, r11d
rol eax, 5
mov dword [rsp + 0x34], ebx
add ebx, 0x5a827999
add eax, ebx
add ecx, eax
mov eax, dword [r15 + 0x18]
mov ebx, eax
add edx, ecx
rol ebx, 0x18
mov ecx, r9d
rol eax, 8
and ebx, 0xff00ff00
and eax, 0xff00ff
xor ecx, r10d
or ebx, eax
and ecx, r11d
mov dword [rsp], ebx
xor ecx, r9d
add ebx, 0x5a827999
rol r11d, 0x1e
mov eax, edx
add r9d, 0x5a827999
rol eax, 5
add eax, ebx
add ecx, eax
mov eax, dword [r15 + 0x1c]
mov ebx, eax
add r8d, ecx
rol ebx, 0x18
mov ecx, r10d
rol eax, 8
and ebx, 0xff00ff00
and eax, 0xff00ff
xor ecx, r11d
or ebx, eax
and ecx, edx
mov dword [rsp + 0xc], ebx
xor ecx, r10d
mov eax, r8d
rol edx, 0x1e
rol eax, 5
add ebx, eax
mov eax, dword [r15 + 0x20]
add ebx, ecx
mov r12d, eax
add ebx, r9d
rol r12d, 0x18
and r12d, 0xff00ff00
rol eax, 8
mov edi, edx
and eax, 0xff00ff
mov esi, edx
or r12d, eax
xor edi, r11d
and edi, r8d
mov dword [rsp + 8], r12d
rol r8d, 0x1e
xor edi, r11d
xor esi, r8d
mov eax, ebx
rol eax, 5
and esi, ebx
add eax, 0x5a827999
rol ebx, 0x1e
add eax, r12d
xor esi, edx
add edi, eax
add edx, 0x5a827999
mov eax, dword [r15 + 0x24]
add edi, r10d
mov r9d, eax
mov ebp, r8d
rol eax, 8
xor ebp, ebx
and eax, 0xff00ff
rol r9d, 0x18
and r9d, 0xff00ff00
and ebp, edi
or r9d, eax
xor ebp, r8d
mov dword [rsp + 0xa0], r9d
mov eax, edi
rol eax, 5
add r9d, 0x5a827999
add eax, r9d
rol edi, 0x1e
add esi, eax
mov eax, dword [r15 + 0x28]
mov r9d, eax
add esi, r11d
rol eax, 8
and eax, 0xff00ff
rol r9d, 0x18
and r9d, 0xff00ff00
or r9d, eax
mov eax, esi
rol eax, 5
add eax, r9d
mov dword [rsp + 0xa8], r9d
add ebp, eax
mov r9d, ebx
mov eax, dword [r15 + 0x2c]
add ebp, edx
mov edx, eax
xor r9d, edi
rol eax, 8
and r9d, esi
and eax, 0xff00ff
rol edx, 0x18
and edx, 0xff00ff00
rol esi, 0x1e
or edx, eax
xor r9d, ebx
mov dword [rsp + 0x2c], edx
mov eax, ebp
rol eax, 5
add edx, 0x5a827999
add eax, edx
add r9d, eax
mov eax, dword [r15 + 0x30]
mov edx, eax
add r9d, r8d
rol eax, 8
rol edx, 0x18
and eax, 0xff00ff
and edx, 0xff00ff00
or edx, eax
mov eax, r9d
mov dword [rsp + 0x28], edx
rol eax, 5
add ebx, 0x5a827999
add eax, edx
mov r10d, edi
xor r10d, esi
and r10d, ebp
rol ebp, 0x1e
xor r10d, edi
mov r11d, ebp
add r10d, eax
xor r11d, esi
mov eax, dword [r15 + 0x34]
add r10d, ebx
mov r14d, eax
and r11d, r9d
rol eax, 8
xor r11d, esi
and eax, 0xff00ff
rol r9d, 0x1e
rol r14d, 0x18
mov edx, ebp
xor edx, r9d
and r14d, 0xff00ff00
or r14d, eax
and edx, r10d
mov eax, r10d
mov dword [rsp + 0x20], r14d
rol eax, 5
xor edx, ebp
add eax, 0x5a827999
rol r10d, 0x1e
add eax, r14d
mov r8d, r9d
add r11d, eax
xor r8d, r10d
mov eax, dword [r15 + 0x38]
add r11d, edi
mov ebx, eax
and r8d, r11d
rol eax, 8
xor r8d, r9d
and eax, 0xff00ff
rol ebx, 0x18
and ebx, 0xff00ff00
mov r13d, r14d
xor r13d, dword [rsp + 0x30]
or ebx, eax
xor r13d, dword [rsp + 0x18]
mov eax, r11d
rol eax, 5
xor r13d, r12d
add eax, 0x5a827999
rol r13d, 1
add eax, ebx
rol r11d, 0x1e
add edx, eax
mov dword [rsp + 0x24], ebx
mov eax, dword [r15 + 0x3c]
add edx, esi
mov r15d, eax
mov dword [rsp + 0x14], r13d
rol eax, 8
mov ecx, r10d
and eax, 0xff00ff
rol r15d, 0x18
and r15d, 0xff00ff00
xor ecx, r11d
or r15d, eax
and ecx, edx
mov eax, edx
mov dword [rsp + 0x1c], r15d
rol eax, 5
add eax, 0x5a827999
add eax, r15d
add r8d, eax
mov rax, qword [rsp + 0x98]
add r8d, ebp
mov dword [rax], r13d
mov eax, r8d
rol eax, 5
xor ecx, r10d
mov edi, dword [rsp]
add eax, 0x5a827999
xor edi, dword [rsp + 0x38]
add eax, r13d
add ecx, eax
xor edi, dword [rsp + 0x28]
mov esi, dword [rsp + 0xc]
add r9d, ecx
xor esi, dword [rsp + 0x34]
mov r12d, ebx
xor r12d, dword [rsp + 4]
mov eax, r9d
xor r12d, dword [rsp + 0x10]
mov ecx, r11d
mov rbx, qword [rsp + 0x98]
mov r14d, r15d
xor r14d, dword [rsp + 0x38]
xor r14d, dword [rsp + 0x30]
xor r12d, dword [rsp + 0xa0]
xor r14d, dword [rsp + 0xa8]
xor esi, dword [rsp + 0x20]
rol eax, 5
add eax, 0x5a827999
rol edx, 0x1e
xor ecx, edx
rol r12d, 1
and ecx, r8d
mov dword [rbx + 4], r12d
xor ecx, r11d
rol r8d, 0x1e
add eax, r12d
rol r14d, 1
add ecx, eax
mov dword [rbx + 8], r14d
mov ebx, dword [rsp + 0x34]
add r10d, ecx
xor ebx, dword [rsp + 4]
mov eax, r10d
xor ebx, dword [rsp + 0x2c]
mov ecx, r8d
rol eax, 5
xor ecx, edx
add eax, 0x5a827999
and ecx, r9d
add eax, r14d
rol r9d, 0x1e
xor ecx, edx
xor ebx, r13d
add ecx, eax
rol ebx, 1
mov rax, qword [rsp + 0x98]
add r11d, ecx
mov ecx, r8d
xor edi, r12d
xor ecx, r9d
rol edi, 1
and ecx, r10d
xor esi, r14d
mov dword [rax + 0xc], ebx
xor ecx, r8d
mov eax, r11d
rol r10d, 0x1e
rol eax, 5
add eax, 0x5a827999
add eax, ebx
add ecx, eax
mov rax, qword [rsp + 0x98]
add edx, ecx
mov ecx, r9d
xor ecx, r10d
xor ecx, r11d
rol r11d, 0x1e
mov dword [rax + 0x10], edi
mov eax, edx
rol eax, 5
add eax, 0x6ed9eba1
add eax, edi
add ecx, eax
mov rax, qword [rsp + 0x98]
add r8d, ecx
rol esi, 1
mov ebp, dword [rsp]
mov ecx, r10d
xor ebp, dword [rsp + 0x24]
xor ecx, r11d
mov dword [rax + 0x14], esi
xor ecx, edx
xor ebp, ebx
rol edx, 0x1e
xor ebp, dword [rsp + 8]
mov eax, r8d
rol eax, 5
add eax, 0x6ed9eba1
rol ebp, 1
add eax, esi
mov dword [rsp], ebp
add ecx, eax
mov rax, qword [rsp + 0x98]
add r9d, ecx
mov ecx, r8d
xor ecx, r11d
rol r8d, 0x1e
xor ecx, edx
mov dword [rax + 0x18], ebp
add ebp, 0x6ed9eba1
mov eax, r9d
rol eax, 5
add eax, ebp
mov ebp, r15d
xor ebp, dword [rsp + 0xc]
add ecx, eax
mov rax, qword [rsp + 0x98]
add r10d, ecx
xor ebp, edi
mov ecx, r8d
xor ebp, dword [rsp + 0xa0]
xor ecx, r9d
xor ecx, edx
rol ebp, 1
mov dword [rax + 0x1c], ebp
mov r15d, esi
xor r15d, dword [rsp + 0xa8]
mov eax, r10d
rol eax, 5
xor r15d, r13d
mov r13, qword [rsp + 0x98]
add eax, 0x6ed9eba1
xor r15d, dword [rsp + 8]
add eax, ebp
add ecx, eax
rol r9d, 0x1e
add r11d, ecx
rol r15d, 1
mov eax, r11d
mov dword [r13 + 0x20], r15d
rol eax, 5
mov ecx, r8d
add eax, 0x6ed9eba1
xor ecx, r9d
add eax, r15d
xor ecx, r10d
add ecx, eax
rol r10d, 0x1e
mov eax, dword [rsp]
add edx, ecx
xor eax, dword [rsp + 0x2c]
mov ecx, r9d
xor eax, r12d
xor ecx, r10d
xor eax, dword [rsp + 0xa0]
xor ecx, r11d
rol eax, 1
mov dword [r13 + 0x24], eax
mov dword [rsp + 4], eax
mov eax, edx
mov r13d, dword [rsp + 4]
add r13d, 0x6ed9eba1
rol eax, 5
add eax, r13d
add ecx, eax
add r8d, ecx
rol r11d, 0x1e
mov rax, qword [rsp + 0x98]
mov ecx, r10d
xor ecx, r11d
mov r13d, ebp
xor r13d, dword [rsp + 0x28]
xor ecx, edx
xor r13d, r14d
rol edx, 0x1e
xor r13d, dword [rsp + 0xa8]
rol r13d, 1
mov dword [rax + 0x28], r13d
mov eax, r8d
rol eax, 5
mov dword [rsp + 0xc], r13d
add r13d, 0x6ed9eba1
add eax, r13d
mov r13d, r15d
xor r13d, dword [rsp + 0x20]
add ecx, eax
mov rax, qword [rsp + 0x98]
add r9d, ecx
xor r13d, ebx
mov ecx, r8d
xor r13d, dword [rsp + 0x2c]
xor ecx, r11d
xor ecx, edx
rol r13d, 1
mov dword [rax + 0x2c], r13d
mov eax, r9d
rol eax, 5
mov dword [rsp + 8], r13d
add r13d, 0x6ed9eba1
add eax, r13d
rol r8d, 0x1e
add ecx, eax
mov eax, dword [rsp + 4]
xor eax, dword [rsp + 0x24]
add r10d, ecx
mov rcx, qword [rsp + 0x98]
xor eax, edi
xor eax, dword [rsp + 0x28]
rol eax, 1
mov dword [rsp + 0x10], eax
mov r13d, dword [rsp + 0x10]
mov dword [rcx + 0x30], eax
add r13d, 0x6ed9eba1
mov eax, r10d
mov ecx, r8d
rol eax, 5
xor ecx, r9d
add eax, r13d
rol r9d, 0x1e
mov r13d, dword [rsp + 0xc]
xor ecx, edx
xor r13d, dword [rsp + 0x1c]
add ecx, eax
mov rax, qword [rsp + 0x98]
add r11d, ecx
xor r13d, esi
mov ecx, r8d
xor r13d, dword [rsp + 0x20]
xor ecx, r9d
rol r13d, 1
xor ecx, r10d
mov dword [rax + 0x34], r13d
mov eax, r11d
rol eax, 5
mov dword [rsp + 0x18], r13d
add r13d, 0x6ed9eba1
add eax, r13d
rol r10d, 0x1e
add ecx, eax
mov eax, dword [rsp + 8]
xor eax, dword [rsp]
add edx, ecx
xor eax, dword [rsp + 0x24]
mov rcx, qword [rsp + 0x98]
xor eax, dword [rsp + 0x14]
rol eax, 1
mov dword [rsp + 0xa0], eax
mov dword [rcx + 0x38], eax
mov r13d, dword [rsp + 0xa0]
mov ecx, r9d
xor ecx, r10d
add r13d, 0x6ed9eba1
xor ecx, r11d
mov eax, edx
rol eax, 5
add eax, r13d
rol r11d, 0x1e
add ecx, eax
mov r13d, dword [rsp + 0x10]
mov rax, qword [rsp + 0x98]
add r8d, ecx
xor r13d, ebp
mov ecx, r10d
xor r13d, dword [rsp + 0x1c]
xor ecx, r11d
xor ecx, edx
xor r13d, r12d
rol r13d, 1
mov dword [rax + 0x3c], r13d
mov eax, r8d
rol eax, 5
mov dword [rsp + 0xa8], r13d
add r13d, 0x6ed9eba1
add eax, r13d
rol edx, 0x1e
add ecx, eax
mov eax, dword [rsp + 0x18]
xor eax, r15d
add r9d, ecx
mov rcx, qword [rsp + 0x98]
xor eax, r14d
xor eax, dword [rsp + 0x14]
rol eax, 1
mov dword [rsp + 0x24], eax
mov r13d, dword [rsp + 0x24]
mov dword [rcx], eax
add r13d, 0x6ed9eba1
mov ecx, r8d
mov eax, r9d
rol eax, 5
xor ecx, r11d
add eax, r13d
rol r8d, 0x1e
mov r13d, dword [rsp + 4]
xor ecx, edx
add ecx, eax
mov eax, r13d
xor eax, dword [rsp + 0xa0]
add r10d, ecx
mov rcx, qword [rsp + 0x98]
xor eax, ebx
xor eax, r12d
rol eax, 1
mov dword [rsp + 4], eax
mov r12d, dword [rsp + 4]
mov dword [rcx + 4], eax
add r12d, 0x6ed9eba1
mov ecx, r8d
mov eax, r10d
rol eax, 5
xor ecx, r9d
add eax, r12d
rol r9d, 0x1e
mov r12d, dword [rsp + 0xc]
xor ecx, edx
add ecx, eax
mov eax, r12d
xor eax, dword [rsp + 0xa8]
add r11d, ecx
mov rcx, qword [rsp + 0x98]
xor eax, edi
xor eax, r14d
rol eax, 1
mov dword [rsp + 0x14], eax
mov dword [rcx + 8], eax
mov eax, r11d
rol eax, 5
mov ecx, r8d
xor ecx, r9d
mov r14d, dword [rsp + 0x14]
xor ecx, r10d
rol r10d, 0x1e
add r14d, 0x6ed9eba1
add eax, r14d
mov r14d, dword [rsp + 8]
add ecx, eax
mov eax, r14d
xor eax, esi
add edx, ecx
xor eax, ebx
mov rcx, qword [rsp + 0x98]
xor eax, dword [rsp + 0x24]
rol eax, 1
mov dword [rsp + 0xc], eax
mov dword [rcx + 0xc], eax
mov ecx, r9d
xor ecx, r10d
mov eax, edx
rol eax, 5
xor ecx, r11d
add eax, 0x6ed9eba1
rol r11d, 0x1e
add eax, dword [rsp + 0xc]
add ecx, eax
mov eax, dword [rsp + 0x10]
xor eax, dword [rsp]
add r8d, ecx
mov rcx, qword [rsp + 0x98]
xor eax, edi
xor eax, dword [rsp + 4]
rol eax, 1
mov dword [rsp + 8], eax
mov dword [rcx + 0x10], eax
mov ecx, r10d
xor ecx, r11d
mov eax, r8d
rol eax, 5
xor ecx, edx
add eax, 0x6ed9eba1
rol edx, 0x1e
add eax, dword [rsp + 8]
add ecx, eax
mov eax, dword [rsp + 0x18]
xor eax, ebp
add r9d, ecx
mov rcx, qword [rsp + 0x98]
xor eax, esi
xor eax, dword [rsp + 0x14]
rol eax, 1
mov dword [rsp + 0x1c], eax
mov dword [rcx + 0x14], eax
mov ecx, r8d
xor ecx, r11d
rol r8d, 0x1e
xor ecx, edx
mov eax, r9d
rol eax, 5
add eax, 0x6ed9eba1
add eax, dword [rsp + 0x1c]
add ecx, eax
mov eax, dword [rsp + 0xa0]
xor eax, r15d
add r10d, ecx
xor eax, dword [rsp]
xor eax, dword [rsp + 0xc]
mov rcx, qword [rsp + 0x98]
rol eax, 1
mov dword [rsp], eax
mov ebx, dword [rsp]
mov dword [rcx + 0x18], eax
add ebx, 0x6ed9eba1
mov eax, r10d
mov ecx, r8d
xor ecx, r9d
rol eax, 5
add eax, ebx
xor ecx, edx
add ecx, eax
add r11d, ecx
rol r9d, 0x1e
mov rax, qword [rsp + 0x98]
add edx, 0x6ed9eba1
mov rsi, qword [rsp + 0x98]
mov ecx, r8d
mov ebx, dword [rsp + 0xa8]
xor ecx, r9d
xor ecx, r10d
xor ebx, r13d
xor ebx, ebp
rol r10d, 0x1e
xor ebx, dword [rsp + 8]
mov edi, r12d
mov ebp, dword [rsp + 4]
xor edi, r15d
xor edi, dword [rsp + 0x1c]
mov r15d, dword [rsp + 0x24]
xor edi, r15d
rol edi, 1
mov dword [rsi + 0x20], edi
rol ebx, 1
mov dword [rax + 0x1c], ebx
mov eax, r11d
rol eax, 5
mov dword [rsp + 0x28], edi
mov dword [rsp + 0x20], ebx
add ebx, eax
add ebx, ecx
mov eax, r10d
add ebx, edx
and eax, r11d
mov edx, ebx
mov ecx, r10d
or ecx, r11d
rol edx, 5
and ecx, r9d
rol r11d, 0x1e
or ecx, eax
mov eax, r11d
add ecx, edi
and eax, ebx
add ecx, r8d
mov edi, r14d
xor edi, r13d
lea r8d, [rdx - 0x70e44324]
xor edi, dword [rsp]
add r8d, ecx
mov r13d, dword [rsp + 0x14]
xor edi, ebp
rol edi, 1
mov ecx, r11d
or ecx, ebx
mov dword [rsi + 0x24], edi
and ecx, r10d
rol ebx, 0x1e
or ecx, eax
mov dword [rsp + 0x2c], edi
mov rax, qword [rsp + 0x98]
add ecx, edi
mov edi, dword [rsp + 0x10]
add ecx, r9d
mov edx, r8d
mov esi, edi
rol edx, 5
xor esi, r12d
xor esi, dword [rsp + 0x20]
xor esi, r13d
rol esi, 1
lea r9d, [rdx - 0x70e44324]
mov dword [rax + 0x28], esi
add r9d, ecx
mov dword [rsp + 0x14], esi
mov ecx, r8d
mov eax, r8d
or ecx, ebx
and eax, ebx
and ecx, r11d
mov edx, r9d
or ecx, eax
rol edx, 5
add ecx, esi
add ecx, r10d
mov esi, dword [rsp + 0x18]
lea r10d, [rdx - 0x70e44324]
mov rax, qword [rsp + 0x98]
add r10d, ecx
rol r8d, 0x1e
mov r12d, esi
xor r12d, r14d
mov ecx, r8d
xor r12d, dword [rsp + 0x28]
or ecx, r9d
mov r14d, dword [rsp + 0xc]
and ecx, ebx
xor r12d, r14d
mov edx, r10d
rol r12d, 1
mov dword [rax + 0x2c], r12d
mov eax, r8d
mov dword [rsp + 0x10], r12d
and eax, r9d
or ecx, eax
rol edx, 5
mov rax, qword [rsp + 0x98]
add ecx, r12d
mov r12d, dword [rsp + 0xa0]
add ecx, r11d
xor r12d, edi
rol r9d, 0x1e
xor r12d, dword [rsp + 0x2c]
lea r11d, [rdx - 0x70e44324]
xor r12d, dword [rsp + 8]
add r11d, ecx
rol r12d, 1
mov edx, r11d
mov dword [rax + 0x30], r12d
mov ecx, r9d
or ecx, r10d
rol edx, 5
and ecx, r8d
mov dword [rsp + 0x18], r12d
mov eax, r9d
and eax, r10d
rol r10d, 0x1e
or ecx, eax
mov rax, qword [rsp + 0x98]
add ecx, r12d
mov r12d, dword [rsp + 0xa8]
add ecx, ebx
xor r12d, esi
xor r12d, dword [rsp + 0x14]
lea ebx, [rdx - 0x70e44324]
xor r12d, dword [rsp + 0x1c]
add ebx, ecx
mov esi, dword [rsp + 0xa0]
mov ecx, r10d
xor esi, dword [rsp + 0x10]
or ecx, r11d
xor esi, dword [rsp]
and ecx, r9d
rol r12d, 1
xor esi, r15d
mov dword [rax + 0x34], r12d
mov edx, ebx
rol edx, 5
mov eax, r10d
and eax, r11d
rol esi, 1
or ecx, eax
rol r11d, 0x1e
mov rax, qword [rsp + 0x98]
add ecx, r12d
add ecx, r8d
mov dword [rsp + 4], r12d
lea r8d, [rdx - 0x70e44324]
mov dword [rsp + 0xa0], esi
add r8d, ecx
mov ecx, r11d
mov edx, r8d
mov dword [rax + 0x38], esi
rol edx, 5
or ecx, ebx
and ecx, r10d
mov eax, r11d
and eax, ebx
rol ebx, 0x1e
or ecx, eax
mov rax, qword [rsp + 0x98]
add ecx, esi
mov esi, dword [rsp + 0xa8]
xor esi, dword [rsp + 0x18]
add ecx, r9d
xor esi, dword [rsp + 0x20]
lea r9d, [rdx - 0x70e44324]
add r9d, ecx
xor esi, ebp
rol esi, 1
mov ecx, r8d
or ecx, ebx
mov dword [rax + 0x3c], esi
and ecx, r11d
mov dword [rsp + 0xa8], esi
mov eax, r8d
mov edx, r9d
and eax, ebx
rol edx, 5
or ecx, eax
rol r8d, 0x1e
mov rax, qword [rsp + 0x98]
add ecx, esi
add ecx, r10d
mov esi, r12d
xor esi, dword [rsp + 0x28]
lea r10d, [rdx - 0x70e44324]
mov r12d, dword [rsp + 0xa8]
add r10d, ecx
xor r12d, dword [rsp + 0x14]
xor esi, r13d
xor r12d, dword [rsp + 8]
xor esi, r15d
rol esi, 1
mov edx, r10d
mov dword [rax], esi
xor r12d, r13d
rol edx, 5
mov ecx, r8d
or ecx, r9d
mov dword [rsp + 0x24], esi
and ecx, ebx
rol r12d, 1
mov eax, r8d
mov dword [rsp + 0x34], r12d
and eax, r9d
rol r9d, 0x1e
or ecx, eax
mov eax, r9d
add ecx, esi
and eax, r10d
mov esi, dword [rsp + 0xa0]
add ecx, r11d
xor esi, dword [rsp + 0x2c]
lea r11d, [rdx - 0x70e44324]
add r11d, ecx
xor esi, r14d
xor esi, ebp
mov ecx, r9d
mov rbp, qword [rsp + 0x98]
or ecx, r10d
and ecx, r8d
rol esi, 1
or ecx, eax
rol r10d, 0x1e
add ecx, esi
mov dword [rsp + 0xc], esi
add ecx, ebx
mov dword [rbp + 4], esi
mov edx, r11d
mov dword [rbp + 8], r12d
rol edx, 5
lea ebx, [rdx - 0x70e44324]
add ebx, ecx
mov ecx, r10d
mov edx, ebx
rol edx, 5
mov r15d, dword [rsp + 0x1c]
or ecx, r11d
mov edi, dword [rsp + 0x10]
and ecx, r9d
xor edi, r15d
mov eax, r10d
and eax, r11d
xor edi, r14d
xor edi, dword [rsp + 0x24]
or ecx, eax
mov r14d, dword [rsp + 0x18]
add ecx, r12d
xor r14d, dword [rsp]
add ecx, r8d
xor r14d, dword [rsp + 8]
lea r8d, [rdx - 0x70e44324]
add r8d, ecx
rol r11d, 0x1e
rol edi, 1
mov ecx, r11d
or ecx, ebx
mov dword [rbp + 0xc], edi
and ecx, r10d
mov dword [rsp + 0x1c], edi
mov eax, r11d
xor r14d, esi
mov rsi, qword [rsp + 0x98]
and eax, ebx
or ecx, eax
rol ebx, 0x1e
add ecx, edi
rol r14d, 1
add ecx, r9d
mov dword [rbp + 0x10], r14d
mov ebp, dword [rsp + 4]
mov edx, r8d
xor ebp, dword [rsp + 0x20]
mov eax, r8d
xor ebp, r15d
rol edx, 5
mov r15d, dword [rsp + 0xa0]
and eax, ebx
xor r15d, dword [rsp + 0x28]
xor ebp, r12d
xor r15d, dword [rsp]
lea r9d, [rdx - 0x70e44324]
rol ebp, 1
add r9d, ecx
mov dword [rsp + 8], r14d
xor r15d, edi
mov dword [rsp + 0x30], ebp
rol r15d, 1
mov ecx, r8d
or ecx, ebx
rol r8d, 0x1e
and ecx, r11d
mov dword [rsi + 0x14], ebp
or ecx, eax
mov dword [rsp], r15d
add ecx, r14d
mov dword [rsi + 0x18], r15d
add ecx, r10d
mov edx, r9d
rol edx, 5
mov eax, r8d
and eax, r9d
lea r10d, [rdx - 0x70e44324]
add r10d, ecx
mov ecx, r8d
or ecx, r9d
mov edx, r10d
and ecx, ebx
rol edx, 5
or ecx, eax
rol r9d, 0x1e
add ecx, 0x8f1bbcdc
add ecx, ebp
add r11d, ecx
add r11d, edx
mov edi, r11d
rol edi, 5
mov r13, qword [rsp + 0x98]
add ebx, 0x8f1bbcdc
mov esi, dword [rsp + 0xa8]
mov eax, r9d
xor esi, dword [rsp + 0x2c]
and eax, r10d
xor esi, dword [rsp + 0x20]
mov ecx, r9d
or ecx, r10d
xor esi, r14d
mov r14d, dword [rsp + 0x14]
and ecx, r8d
xor r14d, dword [rsp + 0x28]
or ecx, eax
mov rax, qword [rsp + 0x98]
add ecx, r15d
add ecx, ebx
rol r10d, 0x1e
add edi, ecx
rol esi, 1
xor r14d, ebp
mov dword [rsp + 0x20], esi
xor r14d, dword [rsp + 0x24]
mov ecx, r10d
mov ebp, dword [rsp + 0x10]
or ecx, r11d
xor ebp, dword [rsp + 0x2c]
and ecx, r9d
mov dword [rax + 0x1c], esi
xor ebp, r15d
xor ebp, dword [rsp + 0xc]
mov eax, r10d
mov r15d, dword [rsp + 0x18]
and eax, r11d
or ecx, eax
rol r11d, 0x1e
add ecx, 0x8f1bbcdc
rol r14d, 1
add ecx, esi
mov dword [r13 + 0x20], r14d
add ecx, r8d
rol ebp, 1
mov r8, qword [rsp + 0x98]
mov ebx, edi
mov dword [r13 + 0x24], ebp
mov eax, r11d
and eax, edi
rol ebx, 5
add ebx, ecx
mov dword [rsp + 0x38], r14d
mov edx, ebx
mov dword [rsp + 0x2c], ebp
rol edx, 5
mov ecx, r11d
or ecx, edi
mov r13d, r15d
xor r13d, dword [rsp + 0x14]
and ecx, r10d
or ecx, eax
rol edi, 0x1e
add ecx, r14d
mov eax, ebx
add ecx, r9d
and eax, edi
lea r9d, [rdx - 0x70e44324]
xor r13d, esi
add r9d, ecx
xor r13d, r12d
mov ecx, ebx
mov edx, r9d
or ecx, edi
rol edx, 5
and ecx, r11d
rol ebx, 0x1e
or ecx, eax
add ecx, ebp
add ecx, r10d
lea r10d, [rdx - 0x70e44324]
add r10d, ecx
rol r13d, 1
mov dword [rsp + 0x18], r13d
mov r12d, dword [rsp + 4]
mov eax, ebx
xor r12d, dword [rsp + 0x10]
and eax, r9d
mov ecx, ebx
mov dword [r8 + 0x28], r13d
or ecx, r9d
xor r12d, r14d
xor r12d, dword [rsp + 0x1c]
and ecx, edi
or ecx, eax
rol r9d, 0x1e
add ecx, r13d
rol r12d, 1
add ecx, r11d
mov dword [r8 + 0x2c], r12d
add edi, 0x8f1bbcdc
mov dword [rsp + 0x28], r12d
mov eax, r9d
mov edx, r10d
and eax, r10d
rol edx, 5
lea r11d, [rdx - 0x70e44324]
add r11d, ecx
mov ecx, r9d
or ecx, r10d
mov r8d, r11d
and ecx, ebx
rol r10d, 0x1e
or ecx, eax
rol r8d, 5
mov rax, qword [rsp + 0x98]
add ecx, r12d
add ecx, edi
add ebx, 0xca62c1d6
mov edi, dword [rsp + 0xa0]
add r8d, ecx
mov esi, edi
mov edx, r8d
xor esi, r15d
rol edx, 5
xor esi, ebp
xor edi, r12d
xor esi, dword [rsp + 8]
xor edi, dword [rsp]
xor edi, dword [rsp + 0x24]
rol esi, 1
mov dword [rax + 0x30], esi
mov eax, r9d
xor eax, r10d
rol edi, 1
xor eax, r11d
mov dword [rsp + 0x14], esi
add eax, esi
rol r11d, 0x1e
add eax, ebx
mov dword [rsp + 0xa0], edi
mov ebx, dword [rsp + 0xa8]
add edx, eax
mov rax, qword [rsp + 0x98]
mov r15d, ebx
xor r15d, dword [rsp + 4]
mov ecx, edx
xor r15d, r13d
rol ecx, 5
xor r15d, dword [rsp + 0x30]
rol r15d, 1
mov dword [rax + 0x34], r15d
mov eax, r10d
xor eax, r11d
mov dword [rsp + 0x10], r15d
xor eax, r8d
rol r8d, 0x1e
add eax, r15d
add eax, r9d
lea r9d, [rcx - 0x359d3e2a]
add r9d, eax
mov rax, qword [rsp + 0x98]
mov ecx, r9d
rol ecx, 5
mov dword [rax + 0x38], edi
xor ebx, esi
mov eax, edx
xor ebx, dword [rsp + 0x20]
xor eax, r11d
xor ebx, dword [rsp + 0xc]
xor eax, r8d
add eax, edi
rol ebx, 1
add eax, r10d
rol edx, 0x1e
lea r10d, [rcx - 0x359d3e2a]
mov dword [rsp + 0xa8], ebx
add r10d, eax
mov edi, r15d
mov rax, qword [rsp + 0x98]
xor edi, r14d
xor edi, dword [rsp + 0x34]
mov ecx, r10d
xor edi, dword [rsp + 0x24]
mov r14d, dword [rsp + 0xa0]
mov dword [rax + 0x3c], ebx
xor r14d, ebp
xor r14d, dword [rsp + 0x1c]
mov eax, edx
xor r14d, dword [rsp + 0xc]
xor eax, r9d
xor eax, r8d
rol ecx, 5
add eax, ebx
rol r9d, 0x1e
add eax, r11d
rol edi, 1
rol r14d, 1
mov ebp, ebx
lea r11d, [rcx - 0x359d3e2a]
mov dword [rsp + 4], edi
add r11d, eax
mov dword [rsp + 0xc], r14d
mov rax, qword [rsp + 0x98]
xor ebp, r13d
xor ebp, dword [rsp + 8]
mov ecx, r11d
xor ebp, dword [rsp + 0x34]
mov r13d, dword [rsp + 0x30]
mov dword [rax], edi
mov eax, edx
xor eax, r9d
rol ecx, 5
xor eax, r10d
rol ebp, 1
add eax, edi
rol r10d, 0x1e
add eax, r8d
mov dword [rsp + 0x34], ebp
lea r8d, [rcx - 0x359d3e2a]
add r8d, eax
mov rax, qword [rsp + 0x98]
mov ecx, r8d
rol ecx, 5
mov dword [rax + 4], r14d
mov eax, r9d
xor eax, r10d
add r9d, 0xca62c1d6
xor eax, r11d
rol r11d, 0x1e
add eax, r14d
add eax, edx
lea edx, [rcx - 0x359d3e2a]
add edx, eax
mov rax, qword [rsp + 0x98]
mov ebx, edx
rol ebx, 5
mov dword [rax + 8], ebp
mov eax, r10d
xor eax, r11d
xor eax, r8d
add eax, ebp
add eax, r9d
mov r9d, r12d
add ebx, eax
rol r8d, 0x1e
xor esi, dword [rsp]
xor r9d, r13d
xor r9d, dword [rsp + 0x1c]
mov eax, edx
xor esi, dword [rsp + 8]
xor eax, r11d
mov r12d, dword [rsp + 0xa8]
xor eax, r8d
xor r12d, dword [rsp + 0x2c]
add eax, 0xca62c1d6
xor r12d, dword [rsp + 0x20]
xor r9d, edi
mov rdi, qword [rsp + 0x98]
add r11d, 0xca62c1d6
rol r9d, 1
xor esi, r14d
add eax, r9d
rol esi, 1
add r10d, eax
rol edx, 0x1e
mov dword [rdi + 0xc], r9d
mov eax, edx
xor eax, ebx
mov dword [rdi + 0x10], esi
xor eax, r8d
mov dword [rsp + 0x30], r9d
add eax, esi
mov dword [rsp + 8], esi
add eax, r11d
mov ecx, ebx
mov r11d, r15d
rol ebx, 0x1e
xor r11d, dword [rsp + 0x20]
xor r12d, esi
mov rsi, qword [rsp + 0x98]
xor r11d, r13d
mov r13d, dword [rsp + 0xa0]
xor r11d, ebp
xor r13d, dword [rsp + 0x38]
xor r13d, dword [rsp]
xor r13d, r9d
rol r11d, 1
rol r13d, 1
rol ecx, 5
add r10d, ecx
rol r12d, 1
mov edi, r10d
mov dword [rsp + 0x1c], r11d
rol edi, 5
add edi, eax
mov dword [rsi + 0x1c], r12d
mov rax, qword [rsp + 0x98]
mov r15d, edi
rol r15d, 5
mov dword [rax + 0x14], r11d
mov eax, edx
xor eax, ebx
add edx, 0xca62c1d6
xor eax, r10d
rol r10d, 0x1e
add eax, 0xca62c1d6
add eax, r11d
add eax, r8d
add r15d, eax
mov rax, qword [rsp + 0x98]
mov r9d, r15d
rol r9d, 5
mov dword [rax + 0x18], r13d
mov eax, ebx
xor eax, r10d
xor eax, edi
rol edi, 0x1e
add eax, r13d
add eax, edx
add r9d, eax
mov eax, r10d
xor eax, edi
mov r14d, r9d
xor eax, r15d
rol r14d, 5
add eax, 0xca62c1d6
mov edx, dword [rsp + 0x18]
add eax, r12d
xor edx, dword [rsp + 0x38]
add eax, ebx
mov r8d, dword [rsp + 0x14]
add r14d, eax
xor r8d, dword [rsp + 0x18]
xor edx, r11d
xor edx, dword [rsp + 4]
mov eax, r9d
mov r11d, dword [rsp + 0x28]
xor eax, edi
xor r11d, dword [rsp + 0x2c]
add edi, 0xca62c1d6
rol r9d, 0x1e
xor r11d, r13d
xor r11d, dword [rsp + 0xc]
xor r8d, r12d
xor r8d, dword [rsp + 0x34]
mov ebp, r14d
rol r11d, 1
mov dword [rsi + 0x24], r11d
rol edx, 1
mov dword [rsi + 0x20], edx
rol r15d, 0x1e
xor eax, r15d
rol ebp, 5
add eax, 0xca62c1d6
rol r8d, 1
add eax, edx
add eax, r10d
mov r10d, dword [rsp + 0x10]
xor r10d, dword [rsp + 0x28]
add ebp, eax
mov eax, r9d
xor r10d, edx
xor r10d, dword [rsp + 0x30]
xor eax, r14d
mov rdx, qword [rsp + 0x98]
xor eax, r15d
add eax, r11d
rol r14d, 0x1e
add eax, edi
rol r10d, 1
mov esi, ebp
rol esi, 5
add esi, eax
mov dword [rdx + 0x2c], r10d
mov rax, qword [rsp + 0x98]
mov edi, esi
rol edi, 5
mov dword [rax + 0x28], r8d
mov eax, r9d
xor eax, r14d
add r9d, 0xca62c1d6
xor eax, ebp
rol ebp, 0x1e
add eax, 0xca62c1d6
add eax, r8d
add eax, r15d
add edi, eax
mov eax, r14d
xor eax, ebp
mov ebx, edi
xor eax, esi
rol ebx, 5
add eax, r10d
rol esi, 0x1e
add eax, r9d
mov r9d, dword [rsp + 0xa0]
xor r9d, dword [rsp + 0x14]
add ebx, eax
xor r9d, dword [rsp + 8]
mov eax, ebp
xor r9d, r11d
xor eax, esi
rol r9d, 1
mov r11d, ebx
rol r11d, 5
xor eax, edi
mov dword [rdx + 0x30], r9d
mov r15, qword [rsp + 0x98]
add eax, r9d
add r14d, 0xca62c1d6
rol edi, 0x1e
add eax, r14d
mov r14d, dword [rsp + 0xa8]
add r11d, eax
mov edx, r14d
xor edx, dword [rsp + 0x10]
xor r14d, r12d
xor edx, dword [rsp + 0x1c]
xor r14d, r9d
xor r14d, dword [rsp + 0xc]
xor edx, r8d
rol edx, 1
mov eax, ebx
xor eax, esi
mov dword [r15 + 0x34], edx
xor eax, edi
rol ebx, 0x1e
add edx, 0xca62c1d6
rol r14d, 1
add eax, edx
mov dword [r15 + 0x3c], r14d
mov edx, dword [rsp + 0xa0]
add eax, ebp
xor edx, r13d
mov r8d, r11d
xor edx, dword [rsp + 4]
xor edx, r10d
rol r8d, 5
add r8d, eax
rol edx, 1
mov dword [r15 + 0x38], edx
mov ecx, r8d
rol ecx, 5
mov eax, ebx
xor eax, r11d
rol r11d, 0x1e
xor eax, edi
add edi, 0xca62c1d6
add eax, edx
add eax, esi
lea edx, [rcx - 0x359d3e2a]
add edx, eax
mov eax, ebx
xor eax, r11d
mov ecx, edx
xor eax, r8d
rol ecx, 5
add ecx, edi
rol r8d, 0x1e
add eax, r14d
add eax, ecx
mov rcx, qword [rsp + 0x90]
add dword [rcx], eax
add dword [rcx + 4], edx
add dword [rcx + 8], r8d
add dword [rcx + 0xc], r11d
add dword [rcx + 0x10], ebx
add rsp, 0x48
pop r15
pop r14
pop r13
pop r12
pop rdi
pop rsi
pop rbp
pop rbx
ret 
