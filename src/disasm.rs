use lde::Isa;
use quick_error::quick_error;

use crate::disasm_cache::{DisasmArch};
use crate::exec_state::{VirtualAddress};
use crate::operand::{
    self, ArithOpType, Flag, MemAccess, Operand, OperandCtx, OperandType, MemAccessSize,
};
use crate::VirtualAddress as VirtualAddress32;
use crate::{BinaryFile, VirtualAddress64};

quick_error! {
    // NOTE: Try avoid making this have a destructor
    /// Errors from disassembly ([`Operation`] generation)
    #[derive(Debug, Copy, Clone)]
    pub enum Error {
        /// Unknown opcode. The tuple is `(opcode_bytes, opcode_len)`.
        ///
        /// To keep this `Error`, and by extension any `Result` using `Error`
        /// as lightweight as possible, at most 8 bytes of the opcode are
        /// stored in this error.
        UnknownOpcode(op: [u8; 8], len: u8) {
            display("Unknown opcode {:02x?}", &op[..*len as usize])
        }
        /// Reached end of code section.
        End {
            display("End of file")
        }
        /// Internal error, there's a bug in scarf :)
        InternalDecodeError {
            display("Internal decode error")
        }
    }
}

/// Used by InstructionOpsState to signal that something had failed and return with ?
/// without making return value heavier.
/// Error should be stored in &mut self
#[derive(Debug)]
struct Failed;

pub type OperationVec<'e> = Vec<Operation<'e>>;

pub struct Disassembler32<'e> {
    buf: &'e [u8],
    pos: usize,
    register_cache: RegisterCache<'e>,
    ops_buffer: Vec<Operation<'e>>,
    ctx: OperandCtx<'e>,
    binary: &'e BinaryFile<VirtualAddress32>,
    current_section_start: VirtualAddress32,
}

fn instruction_length_32(buf: &[u8]) -> usize {
    let length = lde::X86::ld(buf) as usize;
    if length == 0 {
        // Hackfix for lde bug with bswap
        let first = buf.get(0).cloned().unwrap_or(0);
        let second = buf.get(1).cloned().unwrap_or(0);
        let actually_ok = first == 0x0f && second >= 0xc8 && second < 0xd0;
        if actually_ok {
            2
        } else {
            0
        }
    } else {
        if buf.len() > 4 && &buf[..2] == &[0x66, 0x0f] {
            if buf[2] >= 0x71 && buf[2] <= 0x73 {
                // Another lde bug
                5
            } else {
                length
            }
        } else {
            length
        }
    }
}

fn instruction_length_64(buf: &[u8]) -> usize {
    let length = lde::X64::ld(buf) as usize;
    if length == 0 {
        // Hackfix for lde bug with bswap
        let first = buf.get(0).cloned().unwrap_or(0);
        let second = buf.get(1).cloned().unwrap_or(0);
        let third = buf.get(2).cloned().unwrap_or(0);
        match first {
            0x0f if second >= 0xc8 && second < 0xd0 => 2,
            0x40 ..= 0x4f if second == 0x0f && third >= 0xc8 && third < 0xd0 => 3,
            _ => 0,
        }
    } else {
        if buf.len() > 4 && &buf[..2] == &[0x66, 0x0f] {
            if buf[2] >= 0x71 && buf[2] <= 0x73 {
                // Another lde bug
                5
            } else {
                length
            }
        } else {
            length
        }
    }
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler32<'a> {
    type VirtualAddress = VirtualAddress32;

    // Inline(never) seems to help binary size *enough* and this function
    // is only called once per function-to-be-analyzed
    #[inline(never)]
    fn new(
        ctx: OperandCtx<'a>,
        binary: &'a BinaryFile<VirtualAddress32>,
        start_address: VirtualAddress32,
    ) -> Disassembler32<'a> {
        let current_section = binary.section_by_addr(start_address);
        let (buf, pos, current_section_start) = match current_section {
            Some(s) => {
                let relative = (start_address.0 - s.virtual_address.0) as usize;
                (&s.data[..], relative, s.virtual_address)
            }
            None => (&[][..], 0, VirtualAddress32(0)),
        };
        Disassembler32 {
            buf,
            pos,
            register_cache: RegisterCache::new(ctx, false),
            ops_buffer: Vec::with_capacity(16),
            ctx,
            binary,
            current_section_start,
        }
    }

    fn set_pos(&mut self, address: VirtualAddress32) -> Result<(), ()> {
        if address >= self.current_section_start {
            let relative = (address.0 - self.current_section_start.0) as usize;
            if relative < self.buf.len() {
                self.pos = relative;
                return Ok(());
            }
        }
        self.set_pos_cold(address)
    }

    fn next<'s>(&'s mut self) -> Instruction<'s, 'a, VirtualAddress32> {
        let instruction_bytes = &self.buf[self.pos..];
        let length = instruction_length_32(instruction_bytes);
        let address = self.address();
        self.ops_buffer.clear();
        if length == 0 {
            if self.pos == self.buf.len() {
                self.ops_buffer.push(Operation::Error(Error::End));
            } else {
                let mut bytes = [0u8; 8];
                bytes[0] = self.buf[self.pos];
                self.ops_buffer.push(Operation::Error(Error::UnknownOpcode(bytes, 1)));
            }
        } else {
            instruction_operations32(
                address,
                instruction_bytes,
                length,
                self.ctx,
                &mut self.ops_buffer,
                &mut self.register_cache,
            );
        }
        self.pos += length;
        Instruction {
            address,
            ops: &self.ops_buffer,
            length: length as u32,
        }
    }

    #[inline]
    fn address(&self) -> VirtualAddress32 {
        self.current_section_start + self.pos as u32
    }
}

impl<'a> Disassembler32<'a> {
    #[cold]
    fn set_pos_cold(&mut self, address: VirtualAddress32) -> Result<(), ()> {
        let section = self.binary.section_by_addr(address).ok_or(())?;
        let relative = (address.0 - self.current_section_start.0) as usize;
        self.buf = &section.data;
        self.pos = relative;
        self.current_section_start = section.virtual_address;
        Ok(())
    }
}

pub struct Disassembler64<'e> {
    buf: &'e [u8],
    pos: usize,
    register_cache: RegisterCache<'e>,
    ops_buffer: Vec<Operation<'e>>,
    ctx: OperandCtx<'e>,
    binary: &'e BinaryFile<VirtualAddress64>,
    current_section_start: VirtualAddress64,
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler64<'a> {
    type VirtualAddress = VirtualAddress64;

    // Inline(never) seems to help binary size *enough* and this function
    // is only called once per function-to-be-analyzed
    #[inline(never)]
    fn new(
        ctx: OperandCtx<'a>,
        binary: &'a BinaryFile<VirtualAddress64>,
        start_address: VirtualAddress64,
    ) -> Disassembler64<'a> {
        let current_section = binary.section_by_addr(start_address);
        let (buf, pos, current_section_start) = match current_section {
            Some(s) => {
                let relative = (start_address.0 - s.virtual_address.0) as usize;
                (&s.data[..], relative, s.virtual_address)
            }
            None => (&[][..], 0, VirtualAddress64(0)),
        };
        Disassembler64 {
            buf,
            pos,
            register_cache: RegisterCache::new(ctx, true),
            ops_buffer: Vec::with_capacity(16),
            ctx,
            binary,
            current_section_start,
        }
    }

    fn set_pos(&mut self, address: VirtualAddress64) -> Result<(), ()> {
        if address >= self.current_section_start {
            let relative = (address.0 - self.current_section_start.0) as usize;
            if relative < self.buf.len() {
                self.pos = relative;
                return Ok(());
            }
        }
        self.set_pos_cold(address)
    }

    fn next<'s>(&'s mut self) -> Instruction<'s, 'a, VirtualAddress64> {
        let instruction_bytes = &self.buf[self.pos..];
        let length = instruction_length_64(instruction_bytes);
        let address = self.address();
        self.ops_buffer.clear();
        if length == 0 {
            if self.pos == self.buf.len() {
                self.ops_buffer.push(Operation::Error(Error::End));
            } else {
                let mut bytes = [0u8; 8];
                bytes[0] = self.buf[self.pos];
                self.ops_buffer.push(Operation::Error(Error::UnknownOpcode(bytes, 1)));
            }
        } else {
            instruction_operations64(
                address,
                instruction_bytes,
                length,
                self.ctx,
                &mut self.ops_buffer,
                &mut self.register_cache,
            );
        }
        self.pos += length;
        Instruction {
            address,
            ops: &self.ops_buffer,
            length: length as u32,
        }
    }

    fn address(&self) -> VirtualAddress64 {
        self.current_section_start + self.pos as u32
    }
}

impl<'a> Disassembler64<'a> {
    #[cold]
    fn set_pos_cold(&mut self, address: VirtualAddress64) -> Result<(), ()> {
        let section = self.binary.section_by_addr(address).ok_or(())?;
        let relative = (address.0 - self.current_section_start.0) as usize;
        self.buf = &section.data;
        self.pos = relative;
        self.current_section_start = section.virtual_address;
        Ok(())
    }
}

pub struct Instruction<'a, 'e, Va: VirtualAddress> {
    address: Va,
    ops: &'a [Operation<'e>],
    length: u32,
}

impl<'a, 'e, Va: VirtualAddress> Instruction<'a, 'e, Va> {
    pub fn ops(&self) -> &'a [Operation<'e>] {
        &self.ops
    }

    pub fn address(&self) -> Va {
        self.address
    }

    pub fn len(&self) -> u32 {
        self.length
    }
}

#[derive(Copy, Clone)]
struct InstructionPrefixes {
    rex_prefix: u8,
    prefix_66: bool,
    prefix_67: bool,
    prefix_f2: bool,
    prefix_f3: bool,
}

struct RegisterCache<'e> {
    // Ordered as 16 unmasked registers, 16 32bit, 16 16bit, 4 8bit high, 16 8bit
    registers: [Option<Operand<'e>>; 16 * 4 + 4],
    conditions: [Option<Operand<'e>>; 16],
    esp_mem_word: Operand<'e>,
    esp_pos_word_offset: Operand<'e>,
    esp_neg_word_offset: Operand<'e>,
    esp_mem: MemAccess<'e>,
    ctx: OperandCtx<'e>,
}

impl<'e> RegisterCache<'e> {
    fn new(ctx: OperandCtx<'e>, is_64: bool) -> RegisterCache<'e> {
        let size = if is_64 { MemAccessSize::Mem64 } else { MemAccessSize::Mem32 };
        let esp_mem = ctx.mem_access(ctx.register(4), 0, size);
        let esp_pos_word_offset = ctx.add_const(ctx.register(4), size.bits() as u64 >> 3);
        let esp_neg_word_offset = ctx.sub_const(ctx.register(4), size.bits() as u64 >> 3);
        let esp_mem_word = ctx.memory(&esp_mem);
        let mut registers = [None; 16 * 4 + 4];
        for i in 0..16 {
            registers[i] = Some(ctx.register(i as u8));
        }
        RegisterCache {
            ctx,
            registers,
            conditions: [None; 16],
            esp_mem_word,
            esp_mem,
            esp_pos_word_offset,
            esp_neg_word_offset,
        }
    }

    fn register(&mut self, i: u8, size: RegisterSize) -> Operand<'e> {
        let start_index = match size {
            RegisterSize::Low8 => 52u8,
            RegisterSize::High8 => 48,
            RegisterSize::R16 => 32,
            RegisterSize::R32 => 16,
            RegisterSize::R64 => 0,
        };
        let mask = match size {
            RegisterSize::Low8 => 15u8,
            RegisterSize::High8 => 3,
            RegisterSize::R16 => 15,
            RegisterSize::R32 => 15,
            RegisterSize::R64 => 15,
        };
        let index = start_index as usize + (i & mask) as usize;
        if let Some(op) = self.registers[index] {
            op
        } else {
            self.gen_register_cold(i, index, size)
        }
    }

    #[cold]
    fn gen_register_cold(&mut self, i: u8, index: usize, size: RegisterSize) -> Operand<'e> {
        let mask = match size {
            RegisterSize::Low8 => 0xffu32,
            RegisterSize::High8 => 0xff00,
            RegisterSize::R16 => 0xffff,
            RegisterSize::R32 => 0xffff_ffff,
            // Unreachable
            RegisterSize::R64 => 0,
        };
        let ctx = self.ctx;
        let mut op = ctx.and_const(ctx.register(i as u8), mask as u64);
        if size == RegisterSize::High8 {
            op = ctx.rsh_const(op, 8);
        }
        self.registers[index] = Some(op);
        op
    }

    fn condition(&mut self, i: u8) -> Operand<'e> {
        let ctx = self.ctx;
        let cond_id = i & 0xf;
        if let Some(cond) = self.conditions[cond_id as usize] {
            cond
        } else {
            let zero = ctx.const_0();
            let cond = if cond_id & 1 == 0 {
                let cond = self.condition(cond_id + 1);
                ctx.eq(cond, ctx.const_0())
            } else {
                match (cond_id >> 1) & 7 {
                    // jo, jno
                    0x0 => ctx.eq(ctx.flag_o(), zero),
                    // jb, jnb (jae) (jump if carry)
                    0x1 => ctx.eq(ctx.flag_c(), zero),
                    // je, jne
                    0x2 => ctx.eq(ctx.flag_z(), zero),
                    // jbe, jnbe (ja)
                    0x3 => ctx.and(
                        self.condition(3),
                        self.condition(5),
                    ),
                    // js, jns
                    0x4 => ctx.eq(ctx.flag_s(), zero),
                    // jpe, jpo
                    0x5 => ctx.eq(ctx.flag_p(), zero),
                    // jl, jnl (jge)
                    0x6 => ctx.eq(ctx.flag_s(), ctx.flag_o()),
                    // jle, jnle (jg)
                    0x7 => ctx.and(
                        self.condition(5),
                        ctx.eq(ctx.flag_s(), ctx.flag_o()),
                    ),
                    _ => unreachable!(),
                }
            };
            self.conditions[cond_id as usize] = Some(cond);
            cond
        }
    }

    fn esp_mem_word(&mut self) -> Operand<'e> {
        self.esp_mem_word
    }

    fn esp_mem(&mut self) -> MemAccess<'e> {
        self.esp_mem
    }

    fn esp_pos_word_offset(&mut self) -> Operand<'e> {
        self.esp_pos_word_offset
    }

    fn esp_neg_word_offset(&mut self) -> Operand<'e> {
        self.esp_neg_word_offset
    }
}

struct InstructionOpsState<'a, 'e: 'a, Va: VirtualAddress> {
    address: Va,
    data: [u8; 16],
    /// Equal to full_data and any following bytes (Or cut off)
    cache_input: &'a [u8; 8],
    full_data: &'a [u8],
    prefixes: InstructionPrefixes,
    len: u8,
    ctx: OperandCtx<'e>,
    register_cache: &'a mut RegisterCache<'e>,
    out: &'a mut OperationVec<'e>,
    /// Initialize to false.
    /// If the decoding function returns Err(Failed) with this set to false,
    /// generates UnknownOpcode (possible), if true, InternalDecodeError
    /// (Ideally never)
    error_is_decode_error: bool,
    is_ext: bool,
}

fn instruction_operations32<'e>(
    address: VirtualAddress32,
    data: &[u8],
    instruction_len: usize,
    ctx: OperandCtx<'e>,
    out: &mut OperationVec<'e>,
    register_cache: &mut RegisterCache<'e>,
) {
    let is_prefix_byte = |byte| match byte {
        0x64 => true, // TODO fs segment is not handled
        0x65 => true, // TODO gs segment is not handled
        0x66 => true,
        0x67 => true,
        0xf0 => true, // TODO lock prefix not handled
        0xf2 => true,
        0xf3 => true,
        _ => false,
    };
    let mut prefixes = InstructionPrefixes {
        rex_prefix: 0,
        prefix_66: false,
        prefix_67: false,
        prefix_f2: false,
        prefix_f3: false,
    };

    let full_data = &data[..instruction_len];
    let mut hash_buffer_for_end_of_section;
    let cache_input: &[u8; 8] = match data.get(..8).and_then(|x| x.try_into().ok()) {
        Some(s) => s,
        None => {
            hash_buffer_for_end_of_section = [0u8; 8];
            hash_buffer_for_end_of_section[..full_data.len()].copy_from_slice(&full_data);
            &hash_buffer_for_end_of_section
        }
    };
    let prefix_count = data.iter().take_while(|&&x| is_prefix_byte(x)).count();
    for &prefix in data.iter().take_while(|&&x| is_prefix_byte(x)) {
        match prefix {
            0x66 => prefixes.prefix_66 = true,
            0x67 => prefixes.prefix_67 = true,
            0xf2 => prefixes.prefix_f2 = true,
            0xf3 => prefixes.prefix_f3 = true,
            _ => (),
        }
    }
    let data = &data[prefix_count..];
    let is_ext = data[0] == 0xf;
    let data_in = match is_ext {
        true => &data[1..],
        false => data,
    };
    // Try to write this in a way that LLVM doesn't convert the 99.99% common
    // case of 16-byte copy to a memcpy call.
    let mut data;
    if data_in.len() >= 16 {
        data = [0u8; 16];
        (&mut data[..16]).copy_from_slice(&data_in[..16]);
    } else {
        data = [0u8; 16];
        // This seems to still become a memcpy call, oh well.
        for i in 0..data_in.len() {
            data[i] = data_in[i];
        }
    }
    let mut s = InstructionOpsState {
        address,
        data,
        cache_input,
        full_data,
        is_ext,
        prefixes,
        len: instruction_len as u8,
        ctx,
        register_cache,
        out,
        error_is_decode_error: false,
    };
    let result = instruction_operations32_main(&mut s);
    if let Err(Failed) = result {
        let error = if s.error_is_decode_error {
            Operation::Error(Error::InternalDecodeError)
        } else {
            let mut bytes = [0u8; 8];
            let len = full_data.len().min(bytes.len());
            for (out, &val) in bytes.iter_mut().zip(full_data.iter()) {
                *out = val;
            }
            Operation::Error(Error::UnknownOpcode(bytes, len as u8))
        };
        out.push(error);
    }
}

static ARITH_MAPPING: [ArithOperation; 0x20] = {
    let mut ret = [ArithOperation::Move; 0x20];
    ret[0] = ArithOperation::Add;
    ret[1] = ArithOperation::Or;
    ret[2] = ArithOperation::Adc;
    ret[3] = ArithOperation::Sbb;
    ret[4] = ArithOperation::And;
    ret[5] = ArithOperation::Sub;
    ret[6] = ArithOperation::Xor;
    ret[7] = ArithOperation::Cmp;
    ret
};

static BITWISE_ARITH_OPS: [Option<ArithOperation>; 8] = [
    Some(ArithOperation::RotateLeft),
    Some(ArithOperation::RotateRight),
    None, // Unimpl rotate left with carry
    None, // Unimpl rotate right with carry
    Some(ArithOperation::LeftShift),
    Some(ArithOperation::RightShift),
    Some(ArithOperation::LeftShift),
    Some(ArithOperation::RightShiftArithmetic),
];

fn instruction_operations32_main(
    s: &mut InstructionOpsState<VirtualAddress32>,
) -> Result<(), Failed> {
    let ctx = s.ctx;
    // Rustc falls over at patterns containing ranges, manually type out all
    // cases for a first byte to make sure this compiles to a single switch.
    // (Or very least leave it to LLVM to decide)
    // Also represent extended commands as 0x100 ..= 0x1ff to make it even "nicer" switch.
    let first_byte = s.data[0] as u32 | ((s.is_ext as u32) << 8);
    let (can_cache, nop) = s.can_cache(first_byte);
    if can_cache {
        if nop {
            return Ok(());
        }
        if ctx.disasm_cache_read(DisasmArch::X86, s.cache_input, s.len as usize, s.out) {
            return Ok(());
        }
    }
    let result = match first_byte {
        0x00 | 0x01 | 0x02 | 0x03 | 0x04 | 0x05 |
            0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x0d |
            0x10 | 0x11 | 0x12 | 0x13 | 0x14 | 0x15 |
            0x18 | 0x19 | 0x1a | 0x1b | 0x1c | 0x1d |
            0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 |
            0x28 | 0x29 | 0x2a | 0x2b | 0x2c | 0x2d |
            0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35 |
            0x38 | 0x39 | 0x3a | 0x3b | 0x3c | 0x3d |
            0x88 | 0x89 | 0x8a | 0x8b =>
        {
            let arith = ARITH_MAPPING[first_byte as usize / 8];
            let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
            if eax_imm_arith {
                s.eax_imm_arith(arith)
            } else {
                s.generic_arith_op(arith)
            }
        }
        0x40 | 0x41 | 0x42 | 0x43 | 0x44 | 0x45 | 0x46 | 0x47 |
            0x48 | 0x49 | 0x4a | 0x4b | 0x4c | 0x4d | 0x4e | 0x4f => s.inc_dec_op(),
        0x50 | 0x51 | 0x52 | 0x53 | 0x54 | 0x55 | 0x56 | 0x57 |
            0x58 | 0x59 | 0x5a | 0x5b | 0x5c | 0x5d | 0x5e | 0x5f => s.pushpop_reg_op(),
        0x68 | 0x6a => s.push_imm(),
        0x69 | 0x6b => s.signed_multiply_rm_imm(),
        0x70 | 0x71 | 0x72 | 0x73 | 0x74 | 0x75 | 0x76 | 0x77 |
            0x78 | 0x79 | 0x7a | 0x7b | 0x7c | 0x7d | 0x7e | 0x7f =>
        {
            s.conditional_jmp(MemAccessSize::Mem8)
        }
        0x80 | 0x81 | 0x82 | 0x83 => s.arith_with_imm_op(),
        // Test
        0x84 | 0x85 => s.generic_arith_op(ArithOperation::Test),
        0x86 | 0x87 => s.xchg(),
        0x8d => s.lea(),
        0x8f => s.pop_rm(),
        0x90 => Ok(()),
        // Cwde
        0x98 | 0x99 => s.sign_extend(),
        0x9f => s.lahf(),
        0xa0 | 0xa1 | 0xa2 | 0xa3 => s.move_mem_eax(),
        // rep mov, rep stos
        0xa4 | 0xa5 | 0xaa | 0xab => {
            s.output(special_op(&s.full_data)?);
            Ok(())
        }
        0xa8 | 0xa9 => s.eax_imm_arith(ArithOperation::Test),
        0xb0 | 0xb1 | 0xb2 | 0xb3 | 0xb4 | 0xb5 | 0xb6 | 0xb7 |
            0xb8 | 0xb9 | 0xba | 0xbb | 0xbc | 0xbd | 0xbe | 0xbf => s.move_const_to_reg(),
        0xc0 | 0xc1 => s.bitwise_with_imm_op(),
        0xc2 | 0xc3 => {
            let stack_pop_size = match first_byte {
                0xc2 => u32::from(s.read_u16(1)?),
                _ => 0,
            };
            s.output(Operation::Return(stack_pop_size));
            Ok(())
        }
        0xc6 | 0xc7 => {
            s.generic_arith_with_imm_op(ArithOperation::Move, match first_byte {
                0xc6 => MemAccessSize::Mem8,
                _ => s.mem16_32(),
            })
        }
        0xd0 | 0xd1 | 0xd2 | 0xd3 => s.bitwise_compact_op(),
        0xd8 => s.various_d8(),
        0xd9 => s.various_d9(),
        0xdd => s.various_dd(),
        0xe8 => s.call_op(),
        0xe9 => s.jump_op(),
        0xeb => s.short_jmp(),
        0xf6 | 0xf7 => s.various_f7(),
        0xf8 | 0xf9 | 0xfc | 0xfd => {
            let flag = match first_byte {
                0xf8 | 0xf9 => Flag::Carry,
                _ => Flag::Direction,
            };
            let state = first_byte & 0x1 == 1;
            s.flag_set(flag, state)
        }
        0xfe | 0xff => s.various_fe_ff(),
        // --------------------------------
        // --- Extended commands
        // --------------------------------
        0x112 => s.mov_sse_12(),
        // Prefetch/nop
        0x118 | 0x119 | 0x11a | 0x11b | 0x11c | 0x11d | 0x11e | 0x11f => Ok(()),
        0x110 | 0x111| 0x113 | 0x128 | 0x129 | 0x12b | 0x16f | 0x17e | 0x17f => s.sse_move(),
        0x12a => s.sse_int_to_float(),
        0x12c | 0x12d => s.sse_float_to_i32(),
        // ucomiss, comiss, comiss signals exceptions but that isn't simulated
        0x12e | 0x12f => s.sse_compare(),
        // rdtsc
        0x131 => {
            for &reg in &[0, 2] {
                s.output_mov_to_reg(reg, s.ctx.new_undef());
            }
            Ok(())
        }
        0x138 => s.opcode_0f38(),
        0x140 | 0x141 | 0x142 | 0x143 | 0x144 | 0x145 | 0x146 | 0x147 |
            0x148 | 0x149 | 0x14a | 0x14b | 0x14c | 0x14d | 0x14e | 0x14f => s.cmov(),
        0x154 => s.sse_bit_arith(ArithOpType::And),
        0x156 => s.sse_bit_arith(ArithOpType::Or),
        0x157 => s.sse_bit_arith(ArithOpType::Xor),
        0x158 => s.sse_float_arith(ArithOpType::Add),
        0x159 => s.sse_float_arith(ArithOpType::Mul),
        0x15a => s.sse_f32_f64_conversion(),
        0x15b => s.cvtdq2ps(),
        0x15c => s.sse_float_arith(ArithOpType::Sub),
        0x15d | 0x15f => s.sse_float_min_max(),
        0x15e => s.sse_float_arith(ArithOpType::Div),
        0x160 => s.sse_unpack(),
        0x16e => s.mov_sse_6e(),
        0x171 | 0x172 | 0x173 => s.packed_shift_imm(),
        0x180 | 0x181 | 0x182 | 0x183 | 0x184 | 0x185 | 0x186 | 0x187 |
            0x188 | 0x189 | 0x18a | 0x18b | 0x18c | 0x18d | 0x18e | 0x18f =>
        {
            s.conditional_jmp(s.mem16_32())
        }
        0x190 | 0x191 | 0x192 | 0x193 | 0x194 | 0x195 | 0x196 | 0x197 |
            0x198 | 0x199 | 0x19a | 0x19b | 0x19c | 0x19d | 0x19e | 0x19f => s.conditional_set(),
        0x1a3 => s.bit_test(BitTest::NoChange, false),
        0x1a4 => s.shld_imm(),
        0x1ab => s.bit_test(BitTest::Set, false),
        0x1ac => s.shrd_imm(),
        0x1ae => {
            match (s.read_u8(1)? >> 3) & 0x7 {
                // Memory fences
                // (5 is also xrstor though)
                5 | 6 | 7 => Ok(()),
                _ => Err(s.unknown_opcode()),
            }
        }
        0x1af => s.imul_normal(),
        0x1b0 | 0x1b1 => s.cmpxchg(),
        0x1b3 => s.bit_test(BitTest::Reset, false),
        0x1b6 | 0x1b7 => s.movzx(),
        0x1ba => s.various_0f_ba(),
        0x1bb => s.bit_test(BitTest::Complement, false),
        0x1bc | 0x1bd => {
            // bsf, bsr, just set dest as undef.
            // Could maybe emit Special?
            let (_rm, r) = s.parse_modrm(s.mem16_32());
            s.output_mov(r.dest_operand(), ctx.new_undef());
            Ok(())
        }
        0x1be => s.movsx(MemAccessSize::Mem8),
        0x1bf => s.movsx(MemAccessSize::Mem16),
        0x1c0 | 0x1c1 => s.xadd(),
        0x1c5 => s.pextrw(),
        0x1c8 | 0x1c9 | 0x1ca | 0x1cb | 0x1cc | 0x1cd | 0x1ce | 0x1cf => s.bswap(),
        0x1d3 => s.packed_shift_right(),
        0x1d5 => s.pmullw(),
        0x1d6 => s.mov_sse_d6(),
        0x1d7 => s.pmovmskb(),
        0x1e6 => s.sse_int_double_conversion(),
        0x1ef => {
            if s.has_prefix(0x66) {
                // pxor
                s.sse_bit_arith(ArithOpType::Xor)
            } else {
                // MMX xor
                Err(s.unknown_opcode())
            }
        }
        0x1f3 => s.packed_shift_left(),
        _ => Err(s.unknown_opcode()),
    };
    if can_cache && result.is_ok() {
        ctx.disasm_cache_write(DisasmArch::X86, s.cache_input, s.len as usize, s.out);
    }
    result
}

fn instruction_operations64<'e>(
    address: VirtualAddress64,
    data: &[u8],
    instruction_len: usize,
    ctx: OperandCtx<'e>,
    out: &mut OperationVec<'e>,
    register_cache: &mut RegisterCache<'e>,
) {
    let is_prefix_byte = |byte| match byte {
        0x40 ..= 0x4f => true,
        0x64 => true, // TODO fs segment is not handled
        0x65 => true, // TODO gs segment is not handled
        0x66 => true,
        0x67 => true,
        0xf0 => true, // TODO lock prefix not handled
        0xf2 => true,
        0xf3 => true,
        _ => false,
    };
    let mut prefixes = InstructionPrefixes {
        rex_prefix: 0,
        prefix_66: false,
        prefix_67: false,
        prefix_f2: false,
        prefix_f3: false,
    };

    let full_data = &data[..instruction_len];
    let mut hash_buffer_for_end_of_section;
    let cache_input: &[u8; 8] = match data.get(..8).and_then(|x| x.try_into().ok()) {
        Some(s) => s,
        None => {
            hash_buffer_for_end_of_section = [0u8; 8];
            hash_buffer_for_end_of_section[..full_data.len()].copy_from_slice(&full_data);
            &hash_buffer_for_end_of_section
        }
    };
    let prefix_count = data.iter().take_while(|&&x| is_prefix_byte(x)).count();
    for &prefix in data.iter().take_while(|&&x| is_prefix_byte(x)) {
        match prefix {
            0x40 ..= 0x4f => prefixes.rex_prefix = prefix,
            0x66 => prefixes.prefix_66 = true,
            0x67 => prefixes.prefix_67 = true,
            0xf2 => prefixes.prefix_f2 = true,
            0xf3 => prefixes.prefix_f3 = true,
            _ => (),
        }
    }
    let data = &data[prefix_count..];
    let is_ext = data[0] == 0xf;
    let data_in = match is_ext {
        true => &data[1..],
        false => data,
    };
    // Try to write this in a way that LLVM doesn't convert the 99.99% common
    // case of 16-byte copy to a memcpy call.
    let mut data;
    if data_in.len() >= 16 {
        data = [0u8; 16];
        (&mut data[..16]).copy_from_slice(&data_in[..16]);
    } else {
        data = [0u8; 16];
        // This seems to still become a memcpy call, oh well.
        for i in 0..data_in.len() {
            data[i] = data_in[i];
        }
    }
    let mut s = InstructionOpsState {
        address,
        data,
        cache_input,
        full_data,
        prefixes,
        len: instruction_len as u8,
        ctx,
        register_cache,
        out,
        error_is_decode_error: false,
        is_ext,
    };
    let result = instruction_operations64_main(&mut s);
    if let Err(Failed) = result {
        let error = if s.error_is_decode_error {
            Operation::Error(Error::InternalDecodeError)
        } else {
            let mut bytes = [0u8; 8];
            let len = full_data.len().min(bytes.len());
            for (out, &val) in bytes.iter_mut().zip(full_data.iter()) {
                *out = val;
            }
            Operation::Error(Error::UnknownOpcode(bytes, len as u8))
        };
        out.push(error);
    }
}

fn instruction_operations64_main(
    s: &mut InstructionOpsState<VirtualAddress64>,
) -> Result<(), Failed> {
    let ctx = s.ctx;
    // Rustc falls over at patterns containing ranges, manually type out all
    // cases for a first byte to make sure this compiles to a single switch.
    // (Or very least leave it to LLVM to decide)
    // Also represent extended commands as 0x100 ..= 0x1ff to make it even "nicer" switch.
    let first_byte = s.data[0] as u32 | ((s.is_ext as u32) << 8);
    let (can_cache, nop) = s.can_cache(first_byte);
    if can_cache {
        if nop {
            return Ok(());
        }
        if ctx.disasm_cache_read(DisasmArch::X86_64, s.cache_input, s.len as usize, s.out) {
            return Ok(());
        }
    }
    let result = match first_byte {
        0x00 | 0x01 | 0x02 | 0x03 | 0x04 | 0x05 |
            0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x0d |
            0x10 | 0x11 | 0x12 | 0x13 | 0x14 | 0x15 |
            0x18 | 0x19 | 0x1a | 0x1b | 0x1c | 0x1d |
            0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 |
            0x28 | 0x29 | 0x2a | 0x2b | 0x2c | 0x2d |
            0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35 |
            0x38 | 0x39 | 0x3a | 0x3b | 0x3c | 0x3d |
            0x88 | 0x89 | 0x8a | 0x8b =>
        {
            let arith = ARITH_MAPPING[first_byte as usize / 8];
            let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
            if eax_imm_arith {
                s.eax_imm_arith(arith)
            } else {
                s.generic_arith_op(arith)
            }
        }
        0x40 | 0x41 | 0x42 | 0x43 | 0x44 | 0x45 | 0x46 | 0x47 |
            0x48 | 0x49 | 0x4a | 0x4b | 0x4c | 0x4d | 0x4e | 0x4f => Ok(()),
        0x50 | 0x51 | 0x52 | 0x53 | 0x54 | 0x55 | 0x56 | 0x57 |
            0x58 | 0x59 | 0x5a | 0x5b | 0x5c | 0x5d | 0x5e | 0x5f => s.pushpop_reg_op(),
        0x63 => s.movsx(MemAccessSize::Mem32),
        0x68 | 0x6a => s.push_imm(),
        0x69 | 0x6b => s.signed_multiply_rm_imm(),
        0x70 | 0x71 | 0x72 | 0x73 | 0x74 | 0x75 | 0x76 | 0x77 |
            0x78 | 0x79 | 0x7a | 0x7b | 0x7c | 0x7d | 0x7e | 0x7f =>
        {
            s.conditional_jmp(MemAccessSize::Mem8)
        }
        0x80 | 0x81 | 0x82 | 0x83 => s.arith_with_imm_op(),
        // Test
        0x84 | 0x85 => s.generic_arith_op(ArithOperation::Test),
        0x86 | 0x87 => s.xchg(),
        0x8d => s.lea(),
        0x8f => s.pop_rm(),
        0x98 | 0x99 => s.sign_extend(),
        0x9f => s.lahf(),
        0xa0 | 0xa1 | 0xa2 | 0xa3 => s.move_mem_eax(),
        // rep mov, rep stos
        0xa4 | 0xa5 | 0xaa | 0xab => {
            s.output(special_op(&s.full_data)?);
            Ok(())
        }
        0xa8 | 0xa9 => s.eax_imm_arith(ArithOperation::Test),
        0xb0 | 0xb1 | 0xb2 | 0xb3 | 0xb4 | 0xb5 | 0xb6 | 0xb7 |
            0xb8 | 0xb9 | 0xba | 0xbb | 0xbc | 0xbd | 0xbe | 0xbf => s.move_const_to_reg(),
        0xc0 | 0xc1 => s.bitwise_with_imm_op(),
        0xc2 | 0xc3 => {
            let stack_pop_size = match first_byte {
                0xc2 => u32::from(s.read_u16(1)?),
                _ => 0,
            };
            s.output(Operation::Return(stack_pop_size));
            Ok(())
        }
        0xc6 | 0xc7 => {
            s.generic_arith_with_imm_op(ArithOperation::Move, match first_byte {
                0xc6 => MemAccessSize::Mem8,
                _ => s.mem16_32(),
            })
        }
        0xd0 | 0xd1 | 0xd2 | 0xd3 => s.bitwise_compact_op(),
        0xe8 => s.call_op(),
        0xe9 => s.jump_op(),
        0xeb => s.short_jmp(),
        0xf6 | 0xf7 => s.various_f7(),
        0xf8 | 0xf9 | 0xfc | 0xfd => {
            let flag = match first_byte {
                0xf8 | 0xf9 => Flag::Carry,
                _ => Flag::Direction,
            };
            let state = first_byte & 0x1 == 1;
            s.flag_set(flag, state)
        }
        0xfe | 0xff => s.various_fe_ff(),
        // --------------------------------
        // --- Extended commands
        // --------------------------------
        0x112 => s.mov_sse_12(),
        // Prefetch/nop
        0x118 | 0x119 | 0x11a | 0x11b | 0x11c | 0x11d | 0x11e | 0x11f => Ok(()),
        0x110 | 0x111| 0x113 | 0x128 | 0x129 | 0x12b | 0x16f | 0x17e | 0x17f => s.sse_move(),
        0x114 | 0x115 | 0x160 | 0x161 | 0x162 | 0x168 | 0x169 | 0x16a | 0x16c | 0x16d =>
        {
            s.sse_unpack()
        }
        0x12a => s.sse_int_to_float(),
        0x12c | 0x12d => s.sse_float_to_i32(),
        // ucomiss, comiss, comiss signals exceptions but that isn't simulated
        0x12e | 0x12f => s.sse_compare(),
        // rdtsc
        0x131 => {
            for &reg in &[0, 2] {
                s.output_mov(DestOperand::Register32(reg), s.ctx.new_undef());
            }
            Ok(())
        }
        0x138 => s.opcode_0f38(),
        0x140 | 0x141 | 0x142 | 0x143 | 0x144 | 0x145 | 0x146 | 0x147 |
            0x148 | 0x149 | 0x14a | 0x14b | 0x14c | 0x14d | 0x14e | 0x14f => s.cmov(),
        0x154 => s.sse_bit_arith(ArithOpType::And),
        0x156 => s.sse_bit_arith(ArithOpType::Or),
        0x157 => s.sse_bit_arith(ArithOpType::Xor),
        0x158 => s.sse_float_arith(ArithOpType::Add),
        0x159 => s.sse_float_arith(ArithOpType::Mul),
        0x15a => s.sse_f32_f64_conversion(),
        0x15b => s.cvtdq2ps(),
        0x15c => s.sse_float_arith(ArithOpType::Sub),
        0x15d | 0x15f => s.sse_float_min_max(),
        0x15e => s.sse_float_arith(ArithOpType::Div),
        0x16e => s.mov_sse_6e(),
        0x171 | 0x172 | 0x173 => s.packed_shift_imm(),
        0x180 | 0x181 | 0x182 | 0x183 | 0x184 | 0x185 | 0x186 | 0x187 |
            0x188 | 0x189 | 0x18a | 0x18b | 0x18c | 0x18d | 0x18e | 0x18f =>
        {
            s.conditional_jmp(s.mem16_32())
        }
        0x190 | 0x191 | 0x192 | 0x193 | 0x194 | 0x195 | 0x196 | 0x197 |
            0x198 | 0x199 | 0x19a | 0x19b | 0x19c | 0x19d | 0x19e | 0x19f => s.conditional_set(),
        0x1a3 => s.bit_test(BitTest::NoChange, false),
        0x1a4 => s.shld_imm(),
        0x1ab => s.bit_test(BitTest::Set, false),
        0x1ac => s.shrd_imm(),
        0x1ae => {
            match (s.read_u8(1)? >> 3) & 0x7 {
                // Memory fences
                // (5 is also xrstor though)
                5 | 6 | 7 => Ok(()),
                _ => Err(s.unknown_opcode()),
            }
        }
        0x1af => s.imul_normal(),
        0x1b0 | 0x1b1 => s.cmpxchg(),
        0x1b3 => s.bit_test(BitTest::Reset, false),
        0x1b6 | 0x1b7 => s.movzx(),
        0x1ba => s.various_0f_ba(),
        0x1bb => s.bit_test(BitTest::Complement, false),
        0x1bc | 0x1bd => {
            // bsf, bsr, just set dest as undef.
            // Could maybe emit Special?
            let (_rm, r) = s.parse_modrm(s.mem16_32());
            s.output_mov(r.dest_operand(), ctx.new_undef());
            Ok(())
        }
        0x1be => s.movsx(MemAccessSize::Mem8),
        0x1bf => s.movsx(MemAccessSize::Mem16),
        0x1c0 | 0x1c1 => s.xadd(),
        0x1c5 => s.pextrw(),
        0x1c8 | 0x1c9 | 0x1ca | 0x1cb | 0x1cc | 0x1cd | 0x1ce | 0x1cf => s.bswap(),
        0x1d3 => s.packed_shift_right(),
        0x1d5 => s.pmullw(),
        0x1d6 => s.mov_sse_d6(),
        0x1d7 => s.pmovmskb(),
        0x1e6 => s.sse_int_double_conversion(),
        0x1ef => {
            if s.has_prefix(0x66) {
                // pxor
                s.sse_bit_arith(ArithOpType::Xor)
            } else {
                // MMX xor
                Err(s.unknown_opcode())
            }
        }
        0x1f3 => s.packed_shift_left(),
        _ => Err(s.unknown_opcode()),
    };
    if can_cache && result.is_ok() {
        ctx.disasm_cache_write(DisasmArch::X86_64, s.cache_input, s.len as usize, s.out);
    }
    result
}

/// 00 = Cache, 01 = Cache unless RIP-relative R/M, 10 = Never cache, 11 = Nop
/// Generally instructions with constants are marked as not cached.
/// This table should work fine for both 32 and 64-bit instructions, as the changes
/// are prefix bytes that are unused or some that won't change anyway.
static INSTRUCTION_CACHABILITY: [u8; 0x80] = [
    //            03 02 01 00    07 06 05 04    0b 0a 09 08    0f 0e 0d 0c
    /* 00 */    0b01_01_01_01, 0b00_00_00_00, 0b01_01_01_01, 0b00_00_00_00,
    /* 10 */    0b01_01_01_01, 0b00_00_10_00, 0b01_01_01_01, 0b00_00_10_00,
    /* 20 */    0b01_01_01_01, 0b00_00_00_00, 0b01_01_01_01, 0b00_00_00_00,
    /* 30 */    0b01_01_01_01, 0b00_00_00_00, 0b01_01_01_01, 0b00_00_00_00,
    /* 40 */    0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00,
    /* 50 */    0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00,
    /* 60 */    0b01_00_00_00, 0b00_00_00_00, 0b01_00_01_00, 0b00_00_00_00,
    /* 70 */    0b10_10_10_10, 0b10_10_10_10, 0b10_10_10_10, 0b10_10_10_10,
    /* 80 */    0b01_01_10_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 90 */    0b00_00_00_11, 0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00,
    /* a0 */    0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00,
    /* b0 */    0b00_00_00_00, 0b00_00_00_00, 0b10_10_10_10, 0b10_10_10_10,
    /* c0 */    0b00_00_01_01, 0b10_01_00_00, 0b00_00_00_00, 0b00_00_00_00,
    /* d0 */    0b01_01_01_01, 0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00,
    /* e0 */    0b10_10_10_10, 0b00_00_00_00, 0b10_10_10_10, 0b00_00_00_00,
    /* f0 */    0b00_00_00_00, 0b01_01_00_00, 0b00_00_00_00, 0b01_01_00_00,
    //            03 02 01 00    07 06 05 04    0b 0a 09 08    0f 0e 0d 0c
    /* 0f 00 */ 0b00_00_00_00, 0b00_00_00_00, 0b00_00_00_00, 0b00_00_11_00,
    /* 0f 10 */ 0b01_01_01_01, 0b01_01_01_01, 0b11_11_11_11, 0b11_11_11_11,
    /* 0f 20 */ 0b00_00_00_00, 0b00_00_00_00, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f 30 */ 0b00_00_00_00, 0b00_00_00_00, 0b01_01_01_01, 0b00_00_00_00,
    /* 0f 40 */ 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f 50 */ 0b01_01_01_00, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f 60 */ 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f 70 */ 0b00_00_00_01, 0b00_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f 80 */ 0b10_10_10_10, 0b10_10_10_10, 0b10_10_10_10, 0b10_10_10_10,
    /* 0f 90 */ 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f a0 */ 0b01_00_00_00, 0b00_00_01_01, 0b01_00_00_00, 0b01_00_01_01,
    /* 0f b0 */ 0b01_00_01_01, 0b01_01_00_00, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f c0 */ 0b01_01_01_01, 0b01_01_01_01, 0b00_00_00_00, 0b00_00_00_00,
    /* 0f d0 */ 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f e0 */ 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
    /* 0f f0 */ 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01, 0b01_01_01_01,
];

/// opcode is in range 0x100 .. 0x200 for 0f xx instructions
fn can_cache_instruction<Va: VirtualAddress>(
    s: &mut InstructionOpsState<Va>,
    opcode: u32,
) -> (bool, bool) {
    let index = (opcode >> 2) as usize;
    let shift = (opcode & 3) << 1;
    let cachability = (INSTRUCTION_CACHABILITY[index & 0x7f] >> shift) & 3;
    if cachability == 0 || cachability == 3 {
        (true, cachability == 3)
    } else if cachability == 2 {
        (false, false)
    } else {
        let modrm = s.read_u8(1).unwrap_or(0);
        // Skip rip-relative and 32-bit offsets, 32-bit offsets likely
        // have too few cache hits.
        let is_rip_relative = Va::SIZE == 8 && modrm & 0xc7 == 5;
        if !is_rip_relative && modrm & 0xc0 != 0x80 {
            (true, false)
        } else {
            (false, false)
        }
    }
}

#[derive(Copy, Clone)]
enum BitTest {
    Set,
    Reset,
    NoChange,
    Complement,
}

fn x87_variant<'e>(ctx: OperandCtx<'e>, op: Operand<'e>, offset: i8) -> Operand<'e> {
    match *op.ty() {
        OperandType::Register(r) => ctx.register_fpu((r as i8 + offset) as u8 & 7),
        _ => op,
    }
}

impl<'a, 'e: 'a, Va: VirtualAddress> InstructionOpsState<'a, 'e, Va> {
    pub fn len(&self) -> usize {
        self.len as usize
    }

    fn can_cache(&mut self, first_byte: u32) -> (bool, bool) {
        if self.len < 8 {
            can_cache_instruction(self, first_byte)
        } else {
            (false, false)
        }
    }

    /// This is a separate function mainly since SmallVec::push and SmallVec::reserve
    /// are marked as #[inline], and it hurted binary size a lot.
    ///
    /// Now SmallVec isn't used anymore since Disassembler is being kept alive for
    /// entire analysis, so this is more of a legacy function. Probably still nicer
    /// to keep as is.
    #[inline(never)]
    fn output(&mut self, op: Operation<'e>) {
        self.out.push(op);
    }

    #[inline(never)]
    fn output_mov(&mut self, dest: DestOperand<'e>, value: Operand<'e>) {
        self.out.push(Operation::Move(dest, value, None));
    }

    #[inline(never)]
    fn output_mov_to_reg(&mut self, dest: u8, value: Operand<'e>) {
        self.out.push(Operation::Move(DestOperand::Register64(dest), value, None));
    }

    fn output_flag_set(&mut self, flag: Flag, value: Operand<'e>) {
        self.output_mov(DestOperand::Flag(flag), value)
    }

    fn output_arith(
        &mut self,
        dest: DestOperand<'e>,
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
    ) {
        let op = self.ctx.arithmetic(ty, left, right);
        self.output_mov(dest, op)
    }

    fn output_add(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.add(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_add_const(&mut self, dest: DestAndOperand<'e>, rhs: u64) {
        let op = self.ctx.add_const(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_sub(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.sub(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_sub_const(&mut self, dest: DestAndOperand<'e>, rhs: u64) {
        let op = self.ctx.sub_const(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_mul(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.mul(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_signed_mul(
        &mut self,
        dest: DestAndOperand<'e>,
        rhs: Operand<'e>,
        size: MemAccessSize,
    ) {
        let op = self.ctx.signed_mul(dest.op, rhs, size);
        self.output_mov(dest.dest, op)
    }

    fn output_xor(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.xor(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_lsh(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.lsh(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_rsh(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.rsh(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_or(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.or(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_and(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.and(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    fn output_and_const(&mut self, dest: DestAndOperand<'e>, rhs: u64) {
        let op = self.ctx.and_const(dest.op, rhs);
        self.output_mov(dest.dest, op)
    }

    #[inline]
    fn rex_prefix(&self) -> u8 {
        // Ideally eliminates branches checking REX on 32-bit
        if Va::SIZE == 4 {
            0
        } else {
            self.prefixes.rex_prefix
        }
    }

    #[inline]
    fn has_prefix(&self, val: u8) -> bool {
        match val {
            0x66 => self.prefixes.prefix_66,
            0x67 => self.prefixes.prefix_67,
            0xf2 => self.prefixes.prefix_f2,
            0xf3 => self.prefixes.prefix_f3,
            _ => {
                error!("Tried to check prefix {:x}", val);
                false
            }
        }
    }

    fn mem16_32(&self) -> MemAccessSize {
        if Va::SIZE == 4 {
            match self.has_prefix(0x66) {
                true => MemAccessSize::Mem16,
                false => MemAccessSize::Mem32,
            }
        } else {
            match self.rex_prefix() & 0x8 != 0 {
                true => MemAccessSize::Mem64,
                false => match self.has_prefix(0x66) {
                    true => MemAccessSize::Mem16,
                    false => MemAccessSize::Mem32,
                },
            }
        }
    }

    #[inline]
    fn mem32_64(&self) -> MemAccessSize {
        match self.rex_prefix() & 0x8 != 0 {
            true => MemAccessSize::Mem64,
            false => MemAccessSize::Mem32,
        }
    }

    fn reg_variable_size(&mut self, register: u8, op_size: MemAccessSize) -> Operand<'e> {
        let mut register = register;
        let size;
        if register >= 4 && self.rex_prefix() == 0 && op_size == MemAccessSize::Mem8 {
            register -= 4;
            size = RegisterSize::High8;
        } else {
            size = RegisterSize::from_mem_access_size(op_size);
        }
        self.register_cache.register(register, size)
    }

    fn r_to_operand(&mut self, r: ModRm_R) -> Operand<'e> {
        self.register_cache.register(r.0, r.1)
    }

    fn r_to_operand_xmm(&self, r: ModRm_R, i: u8) -> Operand<'e> {
        self.ctx.xmm(r.0, i)
    }

    fn r_to_operand_xmm_64(&mut self, r: ModRm_R, i: u8) -> Operand<'e> {
        let low = self.ctx.xmm(r.0, i * 2);
        let high = self.ctx.xmm(r.0, i * 2 + 1);
        self.ctx.or(
            low,
            self.ctx.lsh_const(
                high,
                0x20,
            ),
        )
    }

    /// Returns a structure containing both DestOperand and Operand
    /// variations, not useful with ModRm_R, but ModRm_Rm avoids
    /// recalculating address twice with this.
    fn r_to_dest_and_operand(&mut self, r: ModRm_R) -> DestAndOperand<'e> {
        let dest = match r.1 {
            RegisterSize::R64 => DestOperand::Register64(r.0),
            RegisterSize::R32 => DestOperand::Register32(r.0),
            RegisterSize::R16 => DestOperand::Register16(r.0),
            RegisterSize::Low8 => DestOperand::Register8Low(r.0),
            RegisterSize::High8 => DestOperand::Register8High(r.0),
        };
        let op = self.register_cache.register(r.0, r.1);
        DestAndOperand {
            op,
            dest,
        }
    }

    fn r_to_dest_and_operand_xmm(&self, r: ModRm_R, i: u8) -> DestAndOperand<'e> {
        DestAndOperand {
            op: self.r_to_operand_xmm(r, i),
            dest: r.dest_operand_xmm(i),
        }
    }

    /// Assumes rm.is_memory() == true.
    /// Returns simplified values.
    fn rm_address_operand(&mut self, rm: &ModRm_Rm) -> (Operand<'e>, u64) {
        let ctx = self.ctx;
        // Optimization: avoid having to go through simplify for x + x * 4 -type accesses
        let constant = rm.constant as i32 as i64 as u64;
        let (base, offset) = if rm.constant_base() {
            let zero = ctx.const_0();
            if Va::SIZE == 4 || !rm.rip_relative() {
                (zero, constant as u32 as u64)
            } else {
                let addr = self.address.as_u64()
                    .wrapping_add(self.len() as u64)
                    .wrapping_add(constant);
                (zero, addr)
            }
        } else {
            let base = ctx.register(rm.base & 0xf);
            let base_index_same = rm.base == rm.index && rm.index_mul != 0;
            if base_index_same {
                let base = ctx.mul_const(base, rm.index_mul as u64 + 1);
                return (base, constant);
            } else {
                (base, constant)
            }
        };
        match rm.index_mul {
            0 => (base, offset),
            1 => (ctx.add(base, ctx.register(rm.index & 0xf)), offset),
            x => {
                let with_index = ctx.add(
                    base,
                    ctx.mul_const(ctx.register(rm.index & 0xf), x as u64),
                );
                (with_index, offset)
            }
        }
    }

    fn rm_to_dest_and_operand(&mut self, rm: &ModRm_Rm) -> DestAndOperand<'e> {
        if rm.is_memory() {
            let (base, offset) = self.rm_address_operand(rm);
            let size = rm.size.to_mem_access_size();
            let mem = self.ctx.mem_access(base, offset, size);
            DestAndOperand {
                op: self.ctx.memory(&mem),
                dest: DestOperand::Memory(mem),
            }
        } else {
            self.r_to_dest_and_operand(ModRm_R(rm.base, rm.size))
        }
    }

    fn rm_to_dest_operand(&mut self, rm: &ModRm_Rm) -> DestOperand<'e> {
        if rm.is_memory() {
            let (base, offset) = self.rm_address_operand(&rm);
            DestOperand::Memory(self.ctx.mem_access(base, offset, rm.size.to_mem_access_size()))
        } else {
            ModRm_R(rm.base, rm.size).dest_operand()
        }
    }

    fn rm_to_dest_operand_xmm(&mut self, rm: &ModRm_Rm, i: u8) -> DestOperand<'e> {
        if rm.is_memory() {
            // Would be nice to just add the i * 4 offset on rm_address_operand,
            // but `rm.constant += i * 4` has issues if the constant overflows
            let (base, mut offset) = self.rm_address_operand(&rm);
            offset = offset.wrapping_add(i as u64 * 4);
            DestOperand::Memory(self.ctx.mem_access(base, offset, MemAccessSize::Mem32))
        } else {
            DestOperand::Xmm(rm.base, i)
        }
    }

    fn rm_to_operand_xmm_size(
        &mut self,
        rm: &ModRm_Rm,
        i: u8,
        size: MemAccessSize,
    ) -> Operand<'e> {
        let bytes = size.bytes();
        let ctx = self.ctx;
        if rm.is_memory() {
            let (base, mut offset) = self.rm_address_operand(&rm);
            offset = offset.wrapping_add(i as u64 * bytes as u64);
            ctx.mem_any(size, base, offset)
        } else {
            if size == MemAccessSize::Mem64 {
                self.rm_to_operand_xmm_64(rm, i)
            } else {
                let shift = match size {
                    MemAccessSize::Mem8 => 2,
                    MemAccessSize::Mem16 => 1,
                    MemAccessSize::Mem32 => 0,
                    MemAccessSize::Mem64 => 0,
                };
                let xmm_word = ctx.xmm(rm.base, i >> shift);
                if shift != 0 {
                    let shift = if size == MemAccessSize::Mem8 {
                        (u64::from(i) & 3) << 3
                    } else {
                        // Mem16
                        (u64::from(i) & 1) << 4
                    };
                    ctx.and_const(ctx.rsh_const(xmm_word, shift), size.mask())
                } else {
                    xmm_word
                }
            }
        }
    }

    fn rm_to_operand_xmm(&mut self, rm: &ModRm_Rm, i: u8) -> Operand<'e> {
        if rm.is_memory() {
            // Would be nice to just add the i * 4 offset on rm_address_operand,
            // but `rm.constant += i * 4` has issues if the constant overflows
            let (base, mut offset) = self.rm_address_operand(&rm);
            offset = offset.wrapping_add(i as u64 * 4);
            self.ctx.mem_any(MemAccessSize::Mem32, base, offset)
        } else {
            self.ctx.xmm(rm.base, i)
        }
    }

    fn rm_to_operand_xmm_64(&mut self, rm: &ModRm_Rm, i: u8) -> Operand<'e> {
        let low = self.rm_to_operand_xmm(rm, i * 2);
        let high = self.rm_to_operand_xmm(rm, i * 2 + 1);
        self.ctx.or(
            low,
            self.ctx.lsh_const(
                high,
                0x20,
            ),
        )
    }

    fn rm_to_operand(&mut self, rm: &ModRm_Rm) -> Operand<'e> {
        if rm.is_memory() {
            let (base, offset) = self.rm_address_operand(rm);
            self.ctx.mem_any(rm.size.to_mem_access_size(), base, offset)
        } else {
            self.r_to_operand(ModRm_R(rm.base, rm.size))
        }
    }


    #[must_use]
    #[cold]
    fn unknown_opcode(&mut self) -> Failed {
        // Does nothing as self.error_is_decode_error is supposed to be inited to false
        debug_assert!(self.error_is_decode_error == false);
        Failed
    }

    #[must_use]
    #[cold]
    fn internal_decode_error(&mut self) -> Failed {
        debug_assert!(self.error_is_decode_error == false);
        self.error_is_decode_error = true;
        Failed
    }

    /// Most opcodes in 64-bit mode only take sign-extended 32-bit immediates.
    fn read_variable_size_32(&mut self, offset: usize, size: MemAccessSize) -> Result<u64, Failed> {
        match size {
            MemAccessSize::Mem64 => self.read_u32(offset).map(|x| x as i32 as i64 as u64),
            MemAccessSize::Mem32 => self.read_u32(offset).map(|x| u64::from(x)),
            MemAccessSize::Mem16 => self.read_u16(offset).map(|x| u64::from(x)),
            MemAccessSize::Mem8 => self.read_u8(offset).map(|x| u64::from(x)),
        }
    }

    fn read_variable_size_64(&mut self, offset: usize, size: MemAccessSize) -> Result<u64, Failed> {
        match size {
            MemAccessSize::Mem64 => self.read_u64(offset),
            MemAccessSize::Mem32 => self.read_u32(offset).map(|x| u64::from(x)),
            MemAccessSize::Mem16 => self.read_u16(offset).map(|x| u64::from(x)),
            MemAccessSize::Mem8 => self.read_u8(offset).map(|x| u64::from(x)),
        }
    }

    fn read_variable_size_signed(&mut self, offset: usize, size: MemAccessSize) -> Result<u32, Failed> {
        match size {
            MemAccessSize::Mem32 | MemAccessSize::Mem64 => self.read_u32(offset),
            MemAccessSize::Mem16 => self.read_u16(offset).map(|x| x as i16 as u32),
            MemAccessSize::Mem8 => self.read_u8(offset).map(|x| x as i8 as u32),
        }
    }

    fn read_u64(&mut self, offset: usize) -> Result<u64, Failed> {
        use crate::light_byteorder::ReadLittleEndian;
        match self.data.get(offset..).and_then(|mut x| x.read_u64().ok()) {
            Some(s) => Ok(s),
            None => Err(self.internal_decode_error()),
        }
    }

    #[inline]
    fn read_u32(&mut self, offset: usize) -> Result<u32, Failed> {
        use crate::light_byteorder::ReadLittleEndian;
        match self.data.get(offset..).and_then(|mut x| x.read_u32().ok()) {
            Some(s) => Ok(s),
            None => Err(self.internal_decode_error()),
        }
    }

    #[inline]
    fn read_u16(&mut self, offset: usize) -> Result<u16, Failed> {
        use crate::light_byteorder::ReadLittleEndian;
        match self.data.get(offset..).and_then(|mut x| x.read_u16().ok()) {
            Some(s) => Ok(s),
            None => Err(self.internal_decode_error()),
        }
    }

    #[inline]
    fn read_u8(&mut self, offset: usize) -> Result<u8, Failed> {
        use crate::light_byteorder::ReadLittleEndian;
        match self.data.get(offset..).and_then(|mut x| x.read_u8().ok()) {
            Some(s) => Ok(s),
            None => Err(self.internal_decode_error()),
        }
    }

    /// Returns (rm, r, modrm_size)
    fn parse_modrm_inner(
        &mut self,
        op_size: MemAccessSize
    ) -> (ModRm_Rm, ModRm_R, usize) {
        let modrm = self.read_u8(1).unwrap_or(0);
        let rm_val = modrm & 0x7;
        let rex = self.rex_prefix();
        let register_id = ((modrm >> 3) & 0x7) | ((rex & 0x4) << 1);
        let high8_available = (rex == 0) & (op_size == MemAccessSize::Mem8);

        let size = RegisterSize::from_mem_access_size(op_size);
        let register = register_id & ((high8_available as u8).wrapping_sub(1) | 3);
        let r_size = [size, RegisterSize::High8][(register != register_id) as usize];
        let r = ModRm_R(register, r_size);

        let rm_variant = (modrm >> 6) & 0x3;
        let is_rm_variant_3 = (rm_variant + 1) >> 2;
        let rm = ModRm_Rm {
            size,
            base: 0,
            index: 0,
            index_mul: 0,
            constant: 0,
        };
        let mut retval = (rm, r, 0);
        let rm = &mut retval.0;
        retval.2 = if is_rm_variant_3.wrapping_sub(1) & rm_val == 4 {
            self.parse_sib(rm_variant, rm)
        } else {
            // variant 0, rm_val 5 is constant (or rip relative) offset
            let rm_val5_mask = if rm_val == 5 {
                u32::MAX
            } else {
                0
            };
            let imm = self.read_u32(2).unwrap_or(0);
            rm.constant = [
                imm & rm_val5_mask, // Only if const offset
                imm as i8 as u32, // i8 offset
                imm, // i32/u32 offset
                0, // No offset
            ][rm_variant as usize];
            rm.index = if is_rm_variant_3 != 0 { u8::MAX } else { rm.index };
            // base to 0xff if const offset
            rm.base = rm_val | ((rex & 0x1) << 3);
            rm.base |= ((0xffu32 >> (rm_variant * 8)) & rm_val5_mask) as u8;

            // Only 1 if high8_available && rm_val >= 4 && variant == 3
            let high8_base = (rm_val >> 2) & // 1 only if rm_val >= 4
                is_rm_variant_3 & // 1 only if variant == 3
                high8_available as u8;
            rm.base &= 0x3 | high8_base.wrapping_sub(1);
            rm.size = [rm.size, RegisterSize::High8][(high8_base as usize) & 1];
            // Bytes read depends on immediate size (so rm_variant) (2/6, 3, 6, 2)
            let size =
                ((0x02060302u32 | (rm_val5_mask & 0x4)) >> (rm_variant * 8)) as u8;
            size as usize
        };
        retval
    }

    fn parse_modrm(&mut self, op_size: MemAccessSize) -> (ModRm_Rm, ModRm_R) {
        let (rm, r, _) = self.parse_modrm_inner(op_size);
        (rm, r)
    }

    fn parse_modrm_imm(
        &mut self,
        op_size: MemAccessSize,
        imm_size: MemAccessSize,
    ) -> Result<(ModRm_Rm, ModRm_R, Operand<'e>), Failed> {
        let (rm, r, offset) = self.parse_modrm_inner(op_size);
        let imm = self.read_variable_size_32(offset, imm_size)?;
        let imm = if imm_size == op_size || imm_size == MemAccessSize::Mem64 {
            imm
        } else {
            // Set any bit above sign bit to 1 if sign bit is 1
            // (So sign extend without switch on size)
            // Also avoid u64 shifts as x86 micro-optimization
            let bits_minus_one = imm_size.bits() - 1;
            let sign_bit = 1 << bits_minus_one;
            if (imm as u32) & sign_bit != 0 {
                (imm |
                    (!0u32 >> bits_minus_one << bits_minus_one) as u64 |
                    0xffff_ffff_0000_0000) & op_size.mask()
            } else {
                imm
            }
        };
        Ok((rm, r, self.ctx.constant(imm)))
    }

    fn parse_sib(
        &mut self,
        variation: u8,
        result: &mut ModRm_Rm,
    ) -> usize {
        let sib = self.read_u8(2).unwrap_or(0);
        let rex = self.rex_prefix();
        let mul = 1 << ((sib >> 6) & 0x3);
        let constant = self.read_u32(3).unwrap_or(0);
        let reg = sib & 7;
        // reg == 5 && variation == 0
        let is_constant_base = reg & variation.wrapping_sub(1) == 5;
        result.base = reg | ((rex & 1) << 3);
        if is_constant_base {
            result.base = 0xfe;
        }
        result.index = ((sib >> 3) & 0x7) | ((rex & 0x2) << 2);
        // Index reg 4 = None
        result.index_mul = mul & ((result.index == 0x4) as u8).wrapping_sub(1);
        // [3, 4, 7] depending on variation, 3 + 4 if constant base
        let size = ((0x070403u32 >> (variation * 8)) | ((is_constant_base as u32) << 2)) as u8;

        result.constant = [
            // size == 3 or 7 (variation 0 or 2), 3 gets masked to zero after
            constant,
            // size == 4 (variation 1)
            constant as u8 as i8 as u32,
        ][(variation & 1) as usize] & ((!(size >> 2) as u32) & 1).wrapping_sub(1);
        size as usize
    }

    fn inc_dec_op(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(0)?;
        let is_inc = byte < 0x48;
        let reg_id = byte & 0x7;
        let op_size = self.mem16_32();
        let reg = self.reg_variable_size(reg_id, op_size);
        let dest = DestAndOperand {
            op: reg,
            dest: DestOperand::reg_variable_size(reg_id, op_size),
        };
        match is_inc {
            true => self.output_add_const(dest, 1),
            false => self.output_sub_const(dest, 1),
        }
        self.inc_dec_flags(is_inc, reg, op_size);
        Ok(())
    }

    fn inc_dec_flags(&mut self, is_inc: bool, reg: Operand<'e>, op_size: MemAccessSize) {
        let sign_bit = op_size.sign_bit();
        let max_positive = sign_bit.wrapping_sub(1);
        let ctx = self.ctx;
        self.output_flag_set(Flag::Zero, ctx.eq_const(reg, 0));
        self.output_flag_set(Flag::Sign, ctx.gt_const(reg, max_positive));
        // Overflow if result is 0x8000...000 for inc and 0x7fff..fff for dec
        let eq_value = max_positive.wrapping_add(is_inc as u64);
        self.output_flag_set(Flag::Overflow, ctx.eq_const(reg, eq_value));
    }

    fn flag_set(&mut self, flag: Flag, value: bool) -> Result<(), Failed> {
        self.output_flag_set(flag, self.ctx.constant(value as u64));
        Ok(())
    }

    fn condition(&mut self) -> Result<Operand<'e>, Failed> {
        let cond_id = self.read_u8(0)?;
        Ok(self.register_cache.condition(cond_id))
    }

    fn cmov(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(self.mem16_32());
        let condition = Some(self.condition()?);
        let rm = self.rm_to_operand(&rm);
        self.output(Operation::Move(r.dest_operand(), rm, condition));
        Ok(())
    }

    fn conditional_set(&mut self) -> Result<(), Failed> {
        let condition = self.condition()?;
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem8);
        let rm = self.rm_to_dest_and_operand(&rm);
        self.output_mov(rm.dest, condition);
        // Reads from flags may contain undefined which is not masked
        // to 1bit automatically. The masking has to be done in a separate
        // operation as `ctx.and_const(flag, 1)` simplifies to `flag`.
        self.output_mov(rm.dest, self.ctx.and_const(rm.op, 1));
        Ok(())
    }

    fn xchg(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r) = self.parse_modrm(op_size);
        if !r.equal_to_rm(&rm) {
            let r = self.r_to_dest_and_operand(r);
            let rm = self.rm_to_dest_and_operand(&rm);
            self.output(Operation::Freeze);
            self.output_mov(r.dest, rm.op);
            self.output_mov(rm.dest, r.op);
            self.output(Operation::Unfreeze);
        }
        Ok(())
    }

    fn xadd(&mut self) -> Result<(), Failed> {
        self.xchg()?;
        self.generic_arith_op(ArithOperation::Add)?;
        Ok(())
    }

    fn bswap(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;

        let register = if self.rex_prefix() & 0x1 != 0 {
            8 + (self.read_u8(0)? & 0x7)
        } else {
            self.read_u8(0)? & 0x7
        };
        let ctx = self.ctx;
        let reg_op = ctx.register(register);
        let mut shift;
        let halfway;
        let size = self.mem16_32();
        let repeats;
        if size == MemAccessSize::Mem64 {
            // ((reg >> 38) & ff) | ((reg >> 28) & ff00) |
            // ((reg >> 18) & ff_0000) | ((reg >> 8) & ff00_0000) |
            // ((reg << 8) & 0000_00ff_0000_0000) | ((reg << 18) & 0000_ff00_0000_0000) |
            // ((reg << 28) & 00ff_0000_0000_0000) | ((reg << 38) & ff00_0000_0000_0000)
            shift = 0x38;
            halfway = 0x1_0000_0000;
            repeats = 8;
        } else {
            // ((reg >> 18) & ff) | ((reg >> 8) & ff00) | ((reg << 8) & ff0000) |
            // ((reg << 18) & ff000_0000)
            shift = 0x18;
            halfway = 0x1_0000;
            repeats = 4;
        }
        let mut mask = 0xff;
        let mut value = None;
        for _ in 0..repeats {
            let shifted = if mask < halfway {
                ctx.rsh_const(reg_op, shift)
            } else {
                ctx.lsh_const(reg_op, shift)
            };
            let masked = ctx.and_const(shifted, mask);
            if mask < halfway {
                if shift > 0x10 {
                    shift -= 0x10;
                }
            } else {
                shift += 0x10;
            }
            mask = mask << 8;

            if let Some(old) = value.take() {
                value = Some(ctx.or(old, masked));
            } else {
                value = Some(masked);
            }
        }
        let value = value.unwrap();
        self.output(mov_to_reg_variable_size(size, register, value));
        Ok(())
    }

    fn sign_extend(&mut self) -> Result<(), Failed> {
        // Handles Cwde (eax = sext32(ax)), Cdq (edx = high(sext64(eax))),
        // Cdqe (rax = sext64(eax)), Cqo (rdx = high(sext128(rax)))
        let ctx = self.ctx;
        let (mut mem_size, dest) = match self.read_u8(0)? {
            0x98 => (MemAccessSize::Mem16, 0),
            _ => (MemAccessSize::Mem32, 2),
        };
        if self.rex_prefix() & 0x8 != 0 {
            if dest == 0 {
                mem_size = MemAccessSize::Mem32;
            } else {
                mem_size = MemAccessSize::Mem64;
            }
        }
        let dest_size = match mem_size {
            MemAccessSize::Mem8 => MemAccessSize::Mem16,
            MemAccessSize::Mem16 => MemAccessSize::Mem32,
            MemAccessSize::Mem32 => MemAccessSize::Mem64,
            MemAccessSize::Mem64 => MemAccessSize::Mem64,
        };
        let result = if mem_size == MemAccessSize::Mem64 {
            // rdx = high(sext128(rax))
            // Have to use awkward (x & sign_bit) move
            ctx.sub_const(
                ctx.eq_const(
                    ctx.and_const(
                        ctx.register(0),
                        0x8000_0000_0000_0000,
                    ),
                    0,
                ),
                1,
            )
        } else {
            // Can use simple sign_extend operand
            let result = ctx.sign_extend(ctx.register(0), mem_size, dest_size);
            if dest == 2 {
                ctx.rsh_const(result, 32)
            } else {
                result
            }
        };
        let size = match self.rex_prefix() & 0x8 == 0 {
            true => MemAccessSize::Mem32,
            false => MemAccessSize::Mem64,
        };
        self.output(operation_helpers::mov_to_reg_variable_size(size, dest, result));
        Ok(())
    }

    fn lahf(&mut self) -> Result<(), Failed> {
        // Should also have (AuxCarry, 4) but scarf doesn't include that
        static FLAGS: [(Flag, u8); 4] = [
            (Flag::Carry, 0),
            (Flag::Parity, 2),
            (Flag::Zero, 6),
            (Flag::Sign, 7),
        ];
        let ctx = self.ctx;
        let mut value = ctx.constant(0x2);
        for &(flag, shift) in &FLAGS {
            value = ctx.or(value, ctx.lsh_const(ctx.flag(flag), shift.into()));
        }
        self.output_mov(
            DestOperand::Register8High(0),
            value,
        );
        Ok(())
    }

    fn move_mem_eax(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let const_size = match Va::SIZE == 4 {
            true => MemAccessSize::Mem32,
            false => MemAccessSize::Mem64,
        };
        let constant = self.read_variable_size_64(1, const_size)?;
        let eax_left = self.read_u8(0)? & 0x2 == 0;
        let ctx = self.ctx;
        let mem = self.ctx.mem_access(ctx.const_0(), constant, op_size);
        match eax_left {
            true => {
                let dest = match op_size {
                    MemAccessSize::Mem8 => DestOperand::Register8Low(0),
                    MemAccessSize::Mem16 => DestOperand::Register16(0),
                    MemAccessSize::Mem32 => DestOperand::Register32(0),
                    MemAccessSize::Mem64 => DestOperand::Register64(0),
                };
                let value = ctx.memory(&mem);
                self.out.push(Operation::Move(dest, value, None));
            }
            false => {
                let mut size = RegisterSize::from_mem_access_size(op_size);
                if Va::SIZE == 4 && size == RegisterSize::R32 {
                    // Not going to add and masks for r32 moves on 32-bit
                    size = RegisterSize::R64;
                }
                let value = self.register_cache.register(0, size);
                self.output_mov(DestOperand::Memory(mem), value);
            }
        }
        Ok(())
    }

    fn move_const_to_reg(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let op_size = match self.read_u8(0)? & 0x8 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let register = if self.rex_prefix() & 0x1 != 0 {
            8 + (self.read_u8(0)? & 0x7)
        } else {
            self.read_u8(0)? & 0x7
        };
        let constant = self.read_variable_size_64(1, op_size)?;
        self.output(mov_to_reg_variable_size(op_size, register, self.ctx.constant(constant)));
        Ok(())
    }

    fn do_arith_operation(
        &mut self,
        arith: ArithOperation,
        lhs: &mut ModRm_Rm,
        rhs: Operand<'e>,
    ) {
        // Operation size must be correct, but the lhs operand size may be
        // extended to 64 after size has been taken.
        let size = lhs.size.to_mem_access_size();
        self.skip_unnecessary_32bit_operand_masking(lhs, arith);
        let dest = self.rm_to_dest_and_operand(lhs);
        let ctx = self.ctx;
        let flags = match arith {
            ArithOperation::Add => Some(FlagArith::Add),
            ArithOperation::Sub | ArithOperation::Cmp => Some(FlagArith::Sub),
            ArithOperation::And | ArithOperation::Test => Some(FlagArith::And),
            ArithOperation::Or => Some(FlagArith::Or),
            ArithOperation::Xor => Some(FlagArith::Xor),
            ArithOperation::Adc => Some(FlagArith::Adc),
            ArithOperation::Sbb => Some(FlagArith::Sbb),
            ArithOperation::LeftShift => Some(FlagArith::LeftShift),
            ArithOperation::RightShift => Some(FlagArith::RightShift),
            ArithOperation::RotateLeft => Some(FlagArith::RotateLeft),
            ArithOperation::RotateRight => Some(FlagArith::RotateRight),
            ArithOperation::RightShiftArithmetic => Some(FlagArith::RightShiftArithmetic),
            ArithOperation::Move => None,
        };
        if let Some(ty) = flags {
            if ty == FlagArith::Sbb || ty == FlagArith::Adc {
                // Will have to freeze this since SetFlags writes to carry,
                // but the old carry is used as input too.
                self.output(Operation::Freeze);
            }
            self.output(Operation::SetFlags(FlagUpdate {
                left: dest.op,
                right: rhs,
                ty,
                size,
            }));
        }
        match arith {
            ArithOperation::Cmp | ArithOperation::Test => (),
            ArithOperation::Add => {
                self.output_add(dest, rhs);
            }
            ArithOperation::Sub => {
                self.output_sub(dest, rhs);
            }
            ArithOperation::Adc | ArithOperation::Sbb => {
                let op = if arith == ArithOperation::Adc {
                    ArithOpType::Add
                } else {
                    ArithOpType::Sub
                };
                // dest + rhs + c or dest - rhs - c
                let result = ctx.arithmetic(
                    op,
                    ctx.arithmetic(
                        op,
                        dest.op,
                        rhs,
                    ),
                    ctx.flag_c(),
                );
                self.output_mov(dest.dest, result);
                self.output(Operation::Unfreeze);
            }
            ArithOperation::And => {
                self.output_and(dest, rhs);
            }
            ArithOperation::Or => {
                // While usually in 32-bit mode the 64-bit registers are used,
                // for operations, e.g. (rax | 1) to make users not having to
                // consider unnecessary 32-bit masks everywhere, special case
                // bitwise or with rhs 0xffff_ffff to mov.
                if Va::SIZE == 4 && rhs.if_constant() == Some(0xffff_ffff) {
                    self.output_mov(dest.dest, rhs);
                } else {
                    self.output_or(dest, rhs);
                }
            }
            ArithOperation::Xor => {
                if dest.op == rhs {
                    // Zeroing xor is not that common, usually only done few times
                    // per function at most, but skip its simplification anyway.
                    self.output_mov(dest.dest, self.ctx.const_0());
                } else {
                    self.output_xor(dest, rhs);
                }
            }
            ArithOperation::Move => {
                if dest.op != rhs {
                    self.output_mov(dest.dest, rhs);
                }
            }
            ArithOperation::LeftShift => {
                self.output_lsh(dest, rhs);
            }
            ArithOperation::RightShift => {
                self.output_rsh(dest, rhs);
            }
            ArithOperation::RightShiftArithmetic => {
                // Arithmetic shift shifts in the value of sign bit,
                // that can be represented as bitwise or of
                // `not(ffff...ffff << rhs >> rhs) & ((sign_bit == 0) - 1)`
                // with logical right shift
                // (sign_bit == 0) - 1 is 0 if sign_bit is clear, ffff...ffff if sign_bit is set
                let sign_bit = 1u64 << (size.bits() - 1);
                let logical_rsh = ctx.rsh(dest.op, rhs);
                let mask = (sign_bit << 1).wrapping_sub(1);
                let negative_shift_in_bits = if let Some(rhs) = rhs.if_constant() {
                    let c = if rhs >= 64 {
                        u64::MAX
                    } else {
                        (((mask << rhs) & mask) >> rhs) ^ mask
                    };
                    ctx.constant(c)
                } else {
                    ctx.xor_const(
                        ctx.rsh(
                            ctx.and_const(
                                ctx.lsh(
                                    ctx.constant(mask),
                                    rhs,
                                ),
                                mask,
                            ),
                            rhs,
                        ),
                        mask,
                    )
                };
                let sign_bit_set_mask = ctx.sub_const(
                    ctx.eq_const(
                        ctx.and_const(
                            dest.op,
                            sign_bit,
                        ),
                        0,
                    ),
                    1,
                );
                let result = ctx.or(
                    logical_rsh,
                    ctx.and(
                        negative_shift_in_bits,
                        sign_bit_set_mask,
                    ),
                );
                self.output_mov(dest.dest, result);
            }
            ArithOperation::RotateLeft | ArithOperation::RotateRight => {
                // rol(x, y) == (x << y) | (x >> (0x20 - y))
                // ror(x, y) == (x >> y) | (x << (0x20 - y))
                let bits = size.bits();
                let left;
                let right;
                let bits_minus_rsh = ctx.sub_const_left(bits as u64, rhs);
                if arith == ArithOperation::RotateLeft {
                    left = ctx.lsh(dest.op, rhs);
                    right = ctx.rsh(dest.op, bits_minus_rsh);
                } else {
                    left = ctx.rsh(dest.op, rhs);
                    right = ctx.lsh(dest.op, bits_minus_rsh);
                }
                let full = ctx.or(left, right);
                if size == MemAccessSize::Mem64 {
                    self.output_mov(dest.dest, full);
                } else {
                    let mask = ctx.constant(!(!0u64 >> bits << bits));
                    self.output_arith(dest.dest, ArithOpType::And, full, mask);
                }
            }
        }
    }

    fn eax_imm_arith(&mut self, arith: ArithOperation) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let imm = self.read_variable_size_32(1, op_size)?;
        let val = self.ctx.constant(imm);
        let mut rm = ModRm_Rm::reg_variable_size(0, op_size);
        self.do_arith_operation(arith, &mut rm, val);
        Ok(())
    }

    /// Also mov even though I'm not sure if I should count it as no-op arith or a separate
    /// thing.
    fn generic_arith_op(&mut self, arith: ArithOperation) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (mut rm, r) = self.parse_modrm(op_size);
        let mut r = r.to_rm();
        let rm_left = self.read_u8(0)? & 0x3 < 2;
        let (left, right) = match rm_left {
            true => (&mut rm, &mut r),
            false => (&mut r, &mut rm),
        };
        self.skip_unnecessary_32bit_operand_masking(right, arith);
        let right = self.rm_to_operand(right);
        self.do_arith_operation(arith, left, right);
        Ok(())
    }

    fn lea(&mut self) -> Result<(), Failed> {
        let op_size = self.mem16_32();
        let (rm, r) = self.parse_modrm(op_size);
        if rm.is_memory() {
            let (base, offset) = self.rm_address_operand(&rm);
            let mut addr = self.ctx.add_const(base, offset);
            if Va::SIZE != 4 {
                // 64-bit lea only writes the low 32 bits if either address size is
                // overridden to 32-bit with 0x67 or dest size is not extended to 64-bit
                // with 0x48.
                // Combination of 0x67 + 0x48 (lea rax, [eax]) is also possible.
                // Even if the address calculation overflows there (lea rax, [eax + eax]),
                // the high dword of rax stays zero, so it won't need special casing.
                if self.has_prefix(0x67) || self.rex_prefix() & 0x8 == 0 {
                    addr = self.ctx.and_const(addr, 0xffff_ffff);
                }
            }
            self.output_mov(r.dest_operand(), addr);
        }
        Ok(())
    }

    fn opcode_0f38(&mut self) -> Result<(), Failed> {
        self.data.rotate_left(1);
        let opcode = self.read_u8(0)?;
        let ctx = self.ctx;
        if self.has_prefix(0xf2) {
            // f0 crc32 u8, f1 crc32 u16/u32
            let in_size = match opcode {
                0xf0 => MemAccessSize::Mem8,
                0xf1 => self.mem16_32(),
                _ => return Err(self.unknown_opcode()),
            };
            let (src, dest) = self.parse_modrm(in_size);
            let src_eq_dest = dest.equal_to_rm(&src);
            let dest_op = ctx.register(dest.0);
            let dest = DestOperand::Register32(dest.0);

            let src = self.rm_to_operand(&src);
            let mut input = dest_op;
            for i in 0..in_size.bits() {
                let result = ctx.xor(
                    ctx.rsh_const(
                        input,
                        1,
                    ),
                    ctx.and_const(
                        ctx.sub_const(
                            ctx.eq_const(
                                ctx.and_const(
                                    ctx.xor(
                                        ctx.rsh_const(
                                            src,
                                            i as u64,
                                        ),
                                        input,
                                    ),
                                    1,
                                ),
                                0,
                            ),
                            1,
                        ),
                        0x82f63b78,
                    ),
                );
                // Ideally split the crc calucation to one move per bit so that the megaoperand
                // doesn't slow things down too much, but that obviously won't work when src
                // equals dest. Luckily scarf does realize that this is equivalent to just
                // right shift (Unless doing crc32 eax, ah), so that won't be too unreasonable
                // either. Not worth special casing crc32 with self explicitly to right
                // shift here though.
                if !src_eq_dest {
                    self.output_mov(dest, result);
                } else {
                    input = result;
                }
            }
            if src_eq_dest {
                self.output_mov(dest, input);
            }
        } else if self.has_prefix(0x66) {
            // Only pmovsx 0f, 38, 20 ..= 25 is implemented
            let (in_size, out_size) = match opcode {
                0x20 => (MemAccessSize::Mem8, MemAccessSize::Mem16),
                0x21 => (MemAccessSize::Mem8, MemAccessSize::Mem32),
                0x22 => (MemAccessSize::Mem8, MemAccessSize::Mem64),
                0x23 => (MemAccessSize::Mem16, MemAccessSize::Mem32),
                0x24 => (MemAccessSize::Mem16, MemAccessSize::Mem64),
                0x25 => (MemAccessSize::Mem32, MemAccessSize::Mem64),
                _ => return Err(self.unknown_opcode()),
            };
            let (src, dest) = self.parse_modrm(MemAccessSize::Mem32);
            let mut in_pos = 0;
            let mut out_xmm_word = 0;
            while out_xmm_word < 4 {
                let dest_op = dest.dest_operand_xmm(out_xmm_word);
                let in_value = self.rm_to_operand_xmm_size(&src, in_pos, in_size);
                in_pos += 1;
                let mut value = ctx.sign_extend(in_value, in_size, out_size);
                if out_size != MemAccessSize::Mem64 {
                    for i in 0..((32 / out_size.bits()) - 1) {
                        let in_value = self.rm_to_operand_xmm_size(&src, in_pos, in_size);
                        let new_value = ctx.sign_extend(in_value, in_size, out_size);
                        let shifted = ctx.lsh_const(
                            new_value,
                            u64::from((i + 1) * out_size.bits()),
                        );
                        value = ctx.or(value, shifted);
                        in_pos += 1;
                    }
                    self.output_mov(dest_op, value);
                    out_xmm_word += 1;
                } else {
                    self.output_mov(dest_op, ctx.and_const(value, 0xffff_ffff));
                    self.output_mov(
                        dest.dest_operand_xmm(out_xmm_word + 1),
                        ctx.rsh_const(value, 32),
                    );
                    out_xmm_word += 2;
                }
            }
        } else {
            return Err(self.unknown_opcode());
        }
        Ok(())
    }

    fn movsx(&mut self, op_size: MemAccessSize) -> Result<(), Failed> {
        let dest_size = self.mem16_32();
        let (mut rm, r) = self.parse_modrm(dest_size);
        let reg8_high = op_size == MemAccessSize::Mem8 &&
            !rm.is_memory() &&
            rm.base >= 4 &&
            self.rex_prefix() == 0;
        if reg8_high {
            rm.base -= 4;
            rm.size = RegisterSize::High8;
        } else {
            rm.size = RegisterSize::from_mem_access_size(op_size);
        }

        let rm = self.rm_to_operand(&rm);
        self.output_mov(
            r.dest_operand(),
            self.ctx.sign_extend(rm, op_size, dest_size),
        );
        Ok(())
    }

    fn movzx(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (mut rm, r) = self.parse_modrm(self.mem16_32());
        let reg8_high = op_size == MemAccessSize::Mem8 &&
            !rm.is_memory() &&
            rm.base >= 4 &&
            self.rex_prefix() == 0;
        if reg8_high {
            rm.base -= 4;
            rm.size = RegisterSize::High8;
        } else {
            rm.size = RegisterSize::from_mem_access_size(op_size);
        }
        if is_rm_short_r_register(&rm, r) {
            let r = self.r_to_dest_and_operand(r);
            let size = match op_size {
                MemAccessSize::Mem8 => 0xff,
                MemAccessSize::Mem16 | _ => 0xffff,
            };
            self.output_and_const(r, size);
        } else {
            let rm_oper = self.rm_to_operand(&rm);
            self.output_mov(r.dest_operand(), rm_oper);
        }
        Ok(())
    }

    fn various_f7(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        let (rm, _) = self.parse_modrm(op_size);
        let rm = self.rm_to_dest_and_operand(&rm);
        let ctx = self.ctx;
        match variant {
            0 | 1 => return self.generic_arith_with_imm_op(ArithOperation::Test, op_size),
            2 => {
                // Not
                let constant = self.ctx.constant(!0u64 & op_size.mask());
                self.output_xor(rm, constant);
            }
            3 => {
                // Neg
                self.output(Operation::SetFlags(FlagUpdate {
                    left: ctx.const_0(),
                    right: rm.op,
                    ty: FlagArith::Sub,
                    size: op_size,
                }));
                self.output_arith(
                    rm.dest,
                    ArithOpType::Sub,
                    ctx.const_0(),
                    rm.op,
                );
            }
            4 | 5 => {
                // TODO signed mul
                let eax = self.reg_variable_size(0, op_size);
                let edx = self.reg_variable_size(2, op_size);
                let multiply = ctx.mul(eax, rm.op);
                self.output(Operation::Freeze);
                if op_size == MemAccessSize::Mem64 {
                    self.output_mov(
                        DestOperand::from_oper(edx),
                        ctx.mul_high(eax, rm.op),
                    );
                    self.output_mov(
                        DestOperand::from_oper(eax),
                        multiply,
                    );
                } else {
                    let size = op_size.bits() as u64;
                    self.output_mov(
                        DestOperand::from_oper(edx),
                        ctx.rsh_const(multiply, size),
                    );
                    self.output_mov(
                        DestOperand::from_oper(eax),
                        multiply,
                    );
                }
                self.output(Operation::Unfreeze);
            },
            // Div, idiv
            6 | 7 => {
                // edx = edx:eax % rm, eax = edx:eax / rm
                let eax = self.reg_variable_size(0, op_size);
                let edx = self.reg_variable_size(2, op_size);
                let div;
                let modulo;
                if op_size == MemAccessSize::Mem64 || variant == 7 {
                    // Difficult to do unless rdx is known to be 0
                    // Also idiv is not done
                    div = self.ctx.new_undef();
                    modulo = self.ctx.new_undef();
                } else {
                    let size = op_size.bits() as u64;
                    let pair = ctx.or(
                        ctx.lsh_const(edx, size),
                        eax,
                    );
                    div = ctx.div(pair, rm.op);
                    modulo = ctx.modulo(pair, rm.op);
                }
                self.output(Operation::Freeze);
                self.output_mov(DestOperand::from_oper(edx), modulo);
                self.output_mov(DestOperand::from_oper(eax), div);
                self.output(Operation::Unfreeze);
            }
            _ => return Err(self.unknown_opcode()),
        }
        Ok(())
    }

    fn various_0f_ba(&mut self) -> Result<(), Failed> {
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        static TYPES: [Option<BitTest>; 8] = [
            None, None, None, None,
            Some(BitTest::NoChange), Some(BitTest::Set), Some(BitTest::Reset),
            Some(BitTest::Complement),
        ];
        let ty = TYPES[variant as usize]
            .ok_or_else(|| self.unknown_opcode())?;
        self.bit_test(ty, true)
    }

    fn bit_test(&mut self, test: BitTest, imm8: bool) -> Result<(), Failed> {
        let op_size = self.mem16_32();
        let (dest, index) = if imm8 {
            let (rm, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
            (rm, imm)
        } else {
            let (rm, r) = self.parse_modrm(op_size);
            (rm, self.r_to_operand(r))
        };
        let ctx = self.ctx;
        // Move bit at index to carry, clear it
        // c = (dest >> index) & 1; dest &= !(1 << index)
        let dest = self.rm_to_dest_and_operand(&dest);
        let new_carry = ctx.and_const(ctx.rsh(dest.op, index), 1);
        self.output_flag_set(Flag::Carry, new_carry);
        let bit_mask = ctx.lsh_const_left(1, index);
        match test {
            BitTest::Set => {
                self.output_or(dest, bit_mask);
            }
            BitTest::Reset => {
                self.output_and(
                    dest,
                    ctx.xor_const(
                        bit_mask,
                        0xffff_ffff_ffff_ffff,
                    ),
                );
            }
            BitTest::Complement => {
                self.output_xor(dest, bit_mask);
            }
            BitTest::NoChange => (),
        }
        Ok(())
    }

    fn sse_f32_f64_conversion(&mut self) -> Result<(), Failed> {
        let (src_size, amt) = if self.has_prefix(0xf2) {
            (MemAccessSize::Mem64, 1)
        } else if self.has_prefix(0x66) {
            (MemAccessSize::Mem64, 2)
        } else if self.has_prefix(0xf3) {
            (MemAccessSize::Mem32, 1)
        } else {
            (MemAccessSize::Mem32, 2)
        };
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        let zero = ctx.const_0();
        if src_size == MemAccessSize::Mem32 {
            for i in (0..amt).rev() {
                let dest1 = dest.dest_operand_xmm(i * 2);
                let dest2 = dest.dest_operand_xmm(i * 2 + 1);
                let val = self.rm_to_operand_xmm(&rm, i);
                let arith = ctx.float_arithmetic(ArithOpType::ToDouble, val, zero, src_size);
                let op = ctx.rsh_const(arith, 0x20);
                self.output_mov(dest2, op);
                let op = ctx.and_const(arith, 0xffff_ffff);
                self.output_mov(dest1, op);
            }
        } else {
            for i in 0..amt {
                let dest = dest.dest_operand_xmm(i);
                let val = self.rm_to_operand_xmm_64(&rm, i);
                let arith = ctx.float_arithmetic(ArithOpType::ToDouble, val, zero, src_size);
                self.output_mov(dest, arith);
            }
            for i in amt..4 {
                let dest = dest.dest_operand_xmm(i);
                self.output_mov(dest, zero);
            }
        }
        Ok(())
    }

    fn sse_float_min_max(&mut self) -> Result<(), Failed> {
        let is_min = self.read_u8(0)? == 0x5d;
        let (size, amt) = if self.has_prefix(0xf2) {
            (MemAccessSize::Mem64, 1)
        } else if self.has_prefix(0x66) {
            (MemAccessSize::Mem64, 2)
        } else if self.has_prefix(0xf3) {
            (MemAccessSize::Mem32, 1)
        } else {
            (MemAccessSize::Mem32, 4)
        };
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        if size == MemAccessSize::Mem32 {
            for i in 0..amt {
                let dest = self.r_to_dest_and_operand_xmm(dest, i);
                let rhs = self.rm_to_operand_xmm(&rm, i);
                let (gt_l, gt_r) = if is_min { (dest.op, rhs) } else { (rhs, dest.op) };
                let cmp = ctx.float_arithmetic(ArithOpType::GreaterThan, gt_l, gt_r, size);
                let op = Operation::Move(dest.dest, rhs, Some(cmp));
                self.output(op);
            }
        } else {
            self.output(Operation::Freeze);
            for i in 0..amt {
                let dest1 = self.r_to_dest_and_operand_xmm(dest, i * 2);
                let dest2 = self.r_to_dest_and_operand_xmm(dest, i * 2 + 1);
                let rhs1 = self.rm_to_operand_xmm(&rm, i * 2);
                let rhs2 = self.rm_to_operand_xmm(&rm, i * 2 + 1);
                let dest_op = ctx.or(
                    dest1.op,
                    ctx.lsh_const(dest2.op, 0x20),
                );
                let rhs = self.rm_to_operand_xmm_64(&rm, i);
                let (gt_l, gt_r) = if is_min { (dest_op, rhs) } else { (rhs, dest_op) };
                let cmp = ctx.float_arithmetic(ArithOpType::GreaterThan, gt_l, gt_r, size);
                self.output(Operation::Move(dest1.dest, rhs1, Some(cmp)));
                self.output(Operation::Move(dest2.dest, rhs2, Some(cmp)));
            }
            self.output(Operation::Unfreeze);
        }
        Ok(())
    }

    fn sse_float_arith(&mut self, ty: ArithOpType) -> Result<(), Failed> {
        let (size, amt) = if self.has_prefix(0xf2) {
            (MemAccessSize::Mem64, 1)
        } else if self.has_prefix(0x66) {
            (MemAccessSize::Mem64, 2)
        } else if self.has_prefix(0xf3) {
            (MemAccessSize::Mem32, 1)
        } else {
            (MemAccessSize::Mem32, 4)
        };
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        if size == MemAccessSize::Mem32 {
            for i in 0..amt {
                let dest = self.r_to_dest_and_operand_xmm(dest, i);
                let rhs = self.rm_to_operand_xmm(&rm, i);
                let op = ctx.float_arithmetic(ty, dest.op, rhs, size);
                self.output_mov(dest.dest, op);
            }
        } else {
            self.output(Operation::Freeze);
            for i in 0..amt {
                let dest1 = self.r_to_dest_and_operand_xmm(dest, i << 1);
                let dest2 = self.r_to_dest_and_operand_xmm(dest, (i << 1) + 1);
                let dest_op = ctx.or(
                    dest1.op,
                    ctx.lsh_const(dest2.op, 0x20),
                );
                let rhs = self.rm_to_operand_xmm_64(&rm, i);
                let arith = ctx.float_arithmetic(ty, dest_op, rhs, size);
                let op = ctx.and_const(arith, 0xffff_ffff);
                self.output_mov(dest1.dest, op);
                let op = ctx.rsh_const(arith, 0x20);
                self.output_mov(dest2.dest, op);
            }
            self.output(Operation::Unfreeze);
        }
        Ok(())
    }

    fn sse_bit_arith(&mut self, arith_type: ArithOpType) -> Result<(), Failed> {
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        for i in 0..4 {
            let dest = self.r_to_dest_and_operand_xmm(dest, i);
            let rhs = self.rm_to_operand_xmm(&rm, i);
            let op = ctx.arithmetic(arith_type, dest.op, rhs);
            self.output_mov(dest.dest, op);
        }
        Ok(())
    }

    fn pmullw(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        // Mul 16-bit packed
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        for i in 0..4 {
            let dest = self.r_to_dest_and_operand_xmm(dest, i);
            let rhs = self.rm_to_operand_xmm(&rm, i);
            let low_word = ctx.and_const(
                ctx.mul(
                    dest.op,
                    rhs,
                ),
                0xffff,
            );
            let high_word = ctx.and_const(
                ctx.mul(
                    ctx.rsh_const(dest.op, 0x10),
                    ctx.rsh_const(rhs, 0x10),
                ),
                0xffff,
            );
            self.output_mov(
                dest.dest,
                ctx.or(
                    low_word,
                    ctx.lsh_const(
                        high_word,
                        0x10,
                    ),
                ),
            );
        }
        Ok(())
    }

    fn sse_compare(&mut self) -> Result<(), Failed> {
        let ctx = self.ctx;
        // zpc = 111 if unordered, 000 if greater, 001 if less, 100 if equal
        // or alternatively
        // z = equal or unordererd
        // p = unordered
        // c = less than or unordered
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32);
        let rm = self.rm_to_operand_xmm(&rm, 0);
        let r = self.r_to_operand_xmm(r, 0);
        // isnan(r) | isnan(rm)
        let unordered = ctx.or(
            ctx.gt_const(
                ctx.and_const(r, 0x7fff_ffff),
                0x7f80_0000,
            ),
            ctx.gt_const(
                ctx.and_const(rm, 0x7fff_ffff),
                0x7f80_0000,
            ),
        );
        let zero = ctx.or(
            unordered,
            ctx.float_arithmetic(ArithOpType::Equal, r, rm, MemAccessSize::Mem32),
        );
        let carry = ctx.or(
            unordered,
            ctx.float_arithmetic(ArithOpType::GreaterThan, rm, r, MemAccessSize::Mem32),
        );

        self.output_mov(DestOperand::Flag(Flag::Zero), zero);
        self.output_mov(DestOperand::Flag(Flag::Carry), carry);
        self.output_mov(DestOperand::Flag(Flag::Parity), unordered);
        for &flag in &[Flag::Overflow, Flag::Sign] {
            self.output_mov(DestOperand::Flag(flag), zero);
        }
        Ok(())
    }

    fn sse_int_to_float(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0xf3) && !self.has_prefix(0xf2) {
            return Err(self.unknown_opcode());
        }
        let arith_type = if self.has_prefix(0xf3) {
            ArithOpType::ToFloat
        } else {
            ArithOpType::ToDouble
        };
        let op_size = self.mem32_64();
        let (rm, r) = self.parse_modrm(op_size);
        let ctx = self.ctx;

        let mut rm = self.rm_to_operand(&rm);
        if op_size == MemAccessSize::Mem32 {
            rm = ctx.sign_extend(rm, MemAccessSize::Mem32, MemAccessSize::Mem64);
        }
        let arith = ctx.arithmetic(arith_type, rm, ctx.const_0());
        self.output_mov(
            r.dest_operand_xmm(0),
            ctx.and_const(
                arith,
                0xffff_ffff,
            ),
        );
        if arith_type == ArithOpType::ToDouble {
            self.output_mov(
                r.dest_operand_xmm(1),
                ctx.rsh_const(
                    arith,
                    0x20,
                ),
            );
        }
        Ok(())
    }

    fn sse_int_double_conversion(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        let zero = ctx.const_0();
        if self.has_prefix(0xf3) {
            // 2 i32 to 2 f64.
            // High one first to account for rm == r
            for i in (0..2).rev() {
                let result_f64 = ctx.arithmetic(
                    ArithOpType::ToDouble,
                    self.rm_to_operand_xmm(&rm, i),
                    zero,
                );
                let high = ctx.rsh_const(result_f64, 0x20);
                self.output_mov(r.dest_operand_xmm(i * 2 + 1), high);
                let low = ctx.and_const(result_f64, 0xffff_ffff);
                self.output_mov(r.dest_operand_xmm(i * 2), low);
            }
        } else if self.has_prefix(0xf2) {
            // 2 f64 to 2 i32
            for i in 0..2 {
                let op = self.rm_to_operand_xmm_64(&rm, i);
                self.output_mov(
                    r.dest_operand_xmm(i),
                    ctx.float_arithmetic(ArithOpType::ToInt, op, zero, MemAccessSize::Mem64),
                );
            }
            for i in 2..4 {
                self.output_mov(r.dest_operand_xmm(i), zero);
            }
        } else {
            return Err(self.unknown_opcode());
        }
        Ok(())
    }

    fn sse_float_to_i32(&mut self) -> Result<(), Failed> {
        // TODO Doesn't actually truncate overflows
        // (Opcode 0f, 2c)
        let src_size = if self.has_prefix(0xf3) {
            MemAccessSize::Mem32
        } else if self.has_prefix(0xf2) {
            MemAccessSize::Mem64
        } else {
            return Err(self.unknown_opcode());
        };
        let (rm, r) = self.parse_modrm(self.mem32_64());
        let op = if src_size == MemAccessSize::Mem64 {
            self.rm_to_operand_xmm_64(&rm, 0)
        } else {
            self.rm_to_operand_xmm(&rm, 0)
        };
        self.output_mov(
            r.dest_operand(),
            self.ctx.float_arithmetic(
                ArithOpType::ToInt,
                op,
                self.ctx.const_0(),
                src_size,
            ),
        );
        Ok(())
    }

    fn cvtdq2ps(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32);
        let ctx = self.ctx;
        let zero = ctx.const_0();
        for i in 0..4 {
            let dest = r.dest_operand_xmm(i);
            let rm = ctx.sign_extend(
                self.rm_to_operand_xmm(&rm, i),
                MemAccessSize::Mem32,
                MemAccessSize::Mem64,
            );
            self.output_arith(dest, ArithOpType::ToFloat, rm, zero);
        }
        Ok(())
    }

    fn sse_unpack(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(0)?;
        // 0x14, 0x66 0x14 = low f32/f64
        // 0x15, 0x66 0x15 = high f32/f64
        // 0x60, 0x61, 0x62 = low u8/u16/u32
        // 0x68, 0x69, 0x6a = high u8/u16/u32
        // 0x6c = low u64
        // 0x6d = high u64
        static SIZES: [(MemAccessSize, bool); 0x10] = [
            (MemAccessSize::Mem8, false), // 0x60
            (MemAccessSize::Mem16, false), // 0x61
            (MemAccessSize::Mem32, false), // 0x62
            (MemAccessSize::Mem8, false), // 3
            (MemAccessSize::Mem64, false), // 0x14
            (MemAccessSize::Mem64, true), // 0x15
            (MemAccessSize::Mem8, false), // 6
            (MemAccessSize::Mem8, false), // 7
            (MemAccessSize::Mem8, true), // 0x68
            (MemAccessSize::Mem16, true), // 0x69
            (MemAccessSize::Mem32, true), // 0x6a
            (MemAccessSize::Mem8, false), // b
            (MemAccessSize::Mem64, false), // 0x6c
            (MemAccessSize::Mem64, true), // 0x6d
            (MemAccessSize::Mem8, false), // e
            (MemAccessSize::Mem8, false), // f
        ];
        let (mut size, high) = SIZES[byte as usize & 0xf];
        if !self.has_prefix(0x66) {
            // The non-float non-66 instructions are MMX
            if byte > 0x15 {
                return Err(self.unknown_opcode());
            } else {
                size = MemAccessSize::Mem32;
            }
        }

        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32);
        // Example of operation with Mem8
        // r.0 = (r.0 & ff) | (rm.0 & ff) << 8 | (r.0 & ff00) << 8 | (rm.0 & ff00) << 10
        // r.1 = (r.0 & ff_0000) >> 10 | (rm.0 & ff_0000) >> 8 |
        //      (r.0 & ff00_0000) >> 8 | (rm.0 & ff00_0000)
        // r.2 = (r.1 & ff) | (rm.1 & ff) << 8 | (r.1 & ff00) << 8 | (rm.1 & ff00) << 10
        // r.3 = (r.1 & ff_0000) >> 10 | (rm.1 & ff_0000) >> 8 |
        //      (r.1 & ff00_0000) >> 8 | (rm.1 & ff00_0000)
        let ctx = self.ctx;
        let mut first = self.r_to_operand_xmm_64(r, high as u8);
        let mut second = self.rm_to_operand_xmm_64(&rm, high as u8);
        self.output(Operation::Freeze);
        let mask = size.mask() as u32;
        let count_per_xmm_op = match size {
            MemAccessSize::Mem8 => 4,
            MemAccessSize::Mem16 => 2,
            MemAccessSize::Mem32 => 1,
            MemAccessSize::Mem64 => 0,
        };
        let shift = size.bits().into();
        for xmm_word in 0..4 {
            let mut val = ctx.const_0();
            let mut current_shift = 0;
            for _ in 0..(count_per_xmm_op / 2) {
                // 8 / 16 bit values
                val = ctx.or(
                    ctx.or(
                        ctx.lsh_const(
                            ctx.and_const(
                                first,
                                mask as u64,
                            ),
                            current_shift,
                        ),
                        ctx.lsh_const(
                            ctx.and_const(
                                second,
                                mask as u64,
                            ),
                            current_shift.wrapping_add(shift),
                        ),
                    ),
                    val,
                );
                first = ctx.rsh_const(first, shift);
                second = ctx.rsh_const(second, shift);
                current_shift = shift.wrapping_mul(2);
            }
            if count_per_xmm_op <= 1 {
                // 32 / 64 bit values
                val = ctx.and_const(first, 0xffff_ffff);
                first = ctx.rsh_const(first, 32);
                if xmm_word == 1 || count_per_xmm_op == 1 {
                    std::mem::swap(&mut first, &mut second);
                }
            }
            self.output_mov(r.dest_operand_xmm(xmm_word), val);
        }
        self.output(Operation::Unfreeze);
        Ok(())
    }

    fn mov_sse_6e(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let op_size = match self.rex_prefix() & 0x8 != 0 {
            true => MemAccessSize::Mem64,
            false => MemAccessSize::Mem32,
        };
        let (rm, r) = self.parse_modrm(op_size);
        let rm_op = self.rm_to_operand(&rm);
        let ctx = self.ctx;
        self.output_mov(r.dest_operand_xmm(0), ctx.and_const(rm_op, 0xffff_ffff));
        let zero = ctx.const_0();
        if op_size == MemAccessSize::Mem64 {
            let rm_high = ctx.rsh_const(rm_op, 0x20);
            self.output_mov(r.dest_operand_xmm(1), rm_high);
        } else {
            self.output_mov(r.dest_operand_xmm(1), zero);
        }
        self.output_mov(r.dest_operand_xmm(2), zero);
        self.output_mov(r.dest_operand_xmm(3), zero);
        Ok(())
    }

    fn mov_sse_12(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            // f2 0f 12 = movddup (duplicate)
            // f3 0f 12 = movsldup (duplicate)
            // 0f 12 with rm = xmm, xmm = movhlps (high to low)
            Err(self.unknown_opcode())
        } else {
            // 66 0f 12 = movlpd (non-special move)
            self.sse_move()
        }
    }

    fn sse_move(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32);
        let r = r.to_rm();
        let byte = self.read_u8(0)?;
        let (src, dest) = match byte {
            0x10 | 0x28 | 0x6f | 0x7e | 0x13 => (&rm, &r),
            _ => (&r, &rm),
        };
        let len = match byte {
            0x10 | 0x11 => match (self.has_prefix(0xf3), self.has_prefix(0xf2)) {
                // movss
                (true, false) => 1,
                // movsd
                (false, true) => 2,
                // movups, movupd
                (false, false) => 4,
                (true, true) => return Err(self.unknown_opcode()),
            },
            0x12 | 0x13 => 2,
            0x28 | 0x29 | 0x2b => 4,
            0x6f => match self.has_prefix(0xf3) || self.has_prefix(0x66) {
                true => 4,
                false => return Err(self.unknown_opcode()),
            }
            0x7e => match (self.has_prefix(0xf3), Va::SIZE) {
                (true, 4) | (true, 8) => 2,
                (false, 4) => {
                    // This is special, moves to r32/mem from xmm
                    // dest/src are intentionally swapped here since the other 0x7e
                    // has them in this order
                    let dest_op = self.rm_to_dest_operand(src);
                    let src = self.rm_to_operand_xmm(dest, 0);
                    self.output_mov(dest_op, src);
                    return Ok(());
                }
                (false, 8) => {
                    let mut src = src.clone();
                    if self.rex_prefix() & 0x8 != 0 {
                        src.size = RegisterSize::R64;
                    }
                    let dest_op = self.rm_to_dest_operand(&src);
                    let src = self.rm_to_operand_xmm_64(dest, 0);
                    self.output_mov(dest_op, src);
                    return Ok(());
                }
                _ => unreachable!(),
            },
            0x7f => match self.has_prefix(0xf3) || self.has_prefix(0x66) {
                true => 4,
                false => 2,
            },
            _ => return Err(self.unknown_opcode()),
        };
        for i in 0..len {
            let dest = self.rm_to_dest_operand_xmm(dest, i);
            let src = self.rm_to_operand_xmm(src, i);
            self.output_mov(dest, src);
        }
        Ok(())
    }

    fn mov_sse_d6(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, src) = self.parse_modrm(MemAccessSize::Mem32);
        let zero = self.ctx.const_0();
        for i in 0..4 {
            let val = if i >= 2 {
                if rm.is_memory() {
                    break;
                } else {
                    zero
                }
            } else {
                self.r_to_operand_xmm(src, i)
            };
            let dest = self.rm_to_dest_operand_xmm(&rm, i);
            self.output_mov(dest, val);
        }
        Ok(())
    }

    fn pmovmskb(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        // Sign bit of each byte in input
        // out 0x1 = in 0x80, 0x2 = in 0x8000, 0x4 = 0x80_0000, ...
        let (rm, src) = self.parse_modrm(MemAccessSize::Mem32);
        if rm.is_memory() {
            return Err(self.unknown_opcode());
        }
        // Going to implement this as a zero rm + chain of ors to rm
        // Could just do single large assignment but maybe this ends up being nicer?
        let ctx = self.ctx;
        let dest = self.rm_to_dest_and_operand(&rm);
        self.output_mov(dest.dest, ctx.const_0());
        let mut out_bit_pos = 0i32;
        for i in 0..4 {
            let xmm = self.r_to_operand_xmm(src, i);
            for byte in 0..4 {
                let value = ctx.and_const(xmm, 0x80 << (byte * 8));
                let in_bit_pos = 7i32 + byte * 8;
                let shift = in_bit_pos - out_bit_pos;
                let value = if shift > 0 {
                    ctx.rsh_const(value, shift as u64)
                } else {
                    ctx.lsh_const(value, (0 - shift) as u64)
                };
                self.output_or(dest.clone(), value);
                out_bit_pos += 1;
            }
        }
        Ok(())
    }

    /// Packed shift helper
    fn zero_xmm_if_rm1_nonzer0(&mut self, rm: &ModRm_Rm, dest: ModRm_R) {
        let ctx = self.ctx;
        let rm_1 = self.rm_to_operand_xmm(&rm, 1);
        let high_u32_set = ctx.neq_const(rm_1, 0);
        for i in 0..4 {
            self.output(Operation::Move(
                dest.dest_operand_xmm(i),
                ctx.const_0(),
                Some(high_u32_set),
            ));
        }
    }

    fn packed_shift_imm(&mut self) -> Result<(), Failed> {
        let opcode = self.read_u8(0)?;
        let byte = self.read_u8(1)?;
        if !self.has_prefix(0x66) || byte & 0xc0 != 0xc0 {
            return Err(self.unknown_opcode());
        }
        let variant = (byte >> 3) & 0x7;
        let register = if self.rex_prefix() & 0x1 == 0 {
            byte & 0x7
        } else {
            8 + (byte & 0x7)
        };
        let dest = ModRm_R(register, RegisterSize::R32);
        let constant = self.read_u8(2)?;
        if opcode == 0x73 {
            match variant {
                2 | 6 => {
                    let constant = self.ctx.constant(constant as u64);
                    if variant == 2 {
                        self.packed_shift_right_xmm_u64(dest, constant)
                    } else {
                        self.packed_shift_left_xmm_u64(dest, constant)
                    }
                }
                3 | 7 => {
                    // Shift value is in bytes
                    let constant = constant << 3;
                    if variant == 3 {
                        self.packed_shift_right_xmm_u128(dest, constant)
                    } else {
                        self.packed_shift_left_xmm_u128(dest, constant)
                    }
                }
                _ => return Err(self.unknown_opcode()),
            }
        } else {
            let size = if opcode == 0x71 { MemAccessSize::Mem16 } else { MemAccessSize::Mem32 };
            let ctx = self.ctx;
            let (is_right, is_arithmetic)  = match variant {
                2 => (true, false),
                4 => (true, true),
                6 => (false, false),
                _ => return Err(self.unknown_opcode()),
            };
            let mut keep_mask = if constant >= 0x20 {
                0
            } else if is_right {
                0xffff_ffffu32 >> constant
            } else {
                0xffff_ffffu32 << constant
            };
            if size == MemAccessSize::Mem16 {
                if is_right {
                    keep_mask &= 0xffff_0000;
                    keep_mask |= keep_mask >> 16;
                } else {
                    keep_mask &= 0xffff;
                    keep_mask |= keep_mask << 16;
                }
            }

            for i in 0..4 {
                let input = ctx.xmm(register, i);
                let mut result = if is_right {
                    ctx.rsh_const(input, constant.into())
                } else {
                    ctx.lsh_const(input, constant.into())
                };
                result = ctx.and_const(result, keep_mask.into());
                if is_arithmetic {
                    if size == MemAccessSize::Mem16 {
                        // was_signed_low = (0 - (input >> 0xf) & 1)
                        // was_signed_high = (0 - (input >> 0x1f))
                        // result |= was_signed_low & (!keep_mask & ffff)
                        // result |= was_signed_high & (!keep_mask & ffff_0000)
                        let was_signed_low =
                            ctx.sub_const_left(0, ctx.and_const(ctx.rsh_const(input, 0xf), 1));
                        let was_signed_high = ctx.sub_const_left(0, ctx.rsh_const(input, 0x1f));
                        result = ctx.or(
                            result,
                            ctx.and_const(was_signed_low, u64::from(!keep_mask & 0xffff)),
                        );
                        result = ctx.or(
                            result,
                            ctx.and_const(was_signed_high, u64::from(!keep_mask & 0xffff_0000)),
                        );
                    } else {
                        // was_signed = (0 - (input >> 0x1f))
                        // result |= was_signed & !keep_mask
                        let was_signed = ctx.sub_const_left(0, ctx.rsh_const(input, 0x1f));
                        result = ctx.or(result, ctx.and_const(was_signed, u64::from(!keep_mask)));
                    }
                }
                self.output_mov(DestOperand::Xmm(register, i), result);
            }
        }
        Ok(())
    }

    fn packed_shift_left_xmm_u64(&mut self, dest: ModRm_R, with: Operand<'e>) {
        // let x = with & 0x1f
        // dest.1 = (dest.1 << x) | (dest.0 >> (32 - x))
        // dest.0 = (dest.0 << x)
        // dest.3 = (dest.3 << x) | (dest.2 >> (32 - x))
        // dest.2 = (dest.2 << x)
        // let x = with & 0x20
        // dest.1 = (dest.1 << x) | (dest.0 >> (32 - x))
        // dest.0 = (dest.0 << x)
        // dest.3 = (dest.3 << x) | (dest.2 >> (32 - x))
        // dest.2 = (dest.2 << x)
        let ctx = self.ctx;
        let x_arr = [
            ctx.and_const(with, 0x1f),
            ctx.and_const(with, !0x1f),
        ];
        for &x in &x_arr {
            for i in 0..2 {
                let low_id = i * 2;
                let high_id = low_id + 1;
                let low = self.r_to_operand_xmm(dest, low_id);
                let high = self.r_to_operand_xmm(dest, high_id);
                self.output_arith(
                    dest.dest_operand_xmm(high_id),
                    ArithOpType::Or,
                    ctx.lsh(high, x),
                    ctx.rsh(low, ctx.sub_const_left(0x20, x)),
                );
                self.output_lsh(self.r_to_dest_and_operand_xmm(dest, low_id), x);
            }
        }
    }

    fn packed_shift_left_xmm_u128(&mut self, dest: ModRm_R, with: u8) {
        // let x = with & 0x1f
        // dest.3 = (dest.3 << x) | (dest.2 >> (32 - x))
        // dest.2 = (dest.2 << x) | (dest.1 >> (32 - x))
        // dest.1 = (dest.1 << x) | (dest.0 >> (32 - x))
        // dest.0 = (dest.0 << x)
        // let x = with & 0x20
        // dest.3 = (dest.3 << x) | (dest.2 >> (32 - x))
        // dest.2 = (dest.2 << x) | (dest.1 >> (32 - x))
        // dest.1 = (dest.1 << x) | (dest.0 >> (32 - x))
        // dest.0 = (dest.0 << x)
        // (The following is done twice)
        // let x = (with & 0xffff_ffc0) >> 1
        // dest.3 = (dest.3 << x) | (dest.2 >> (32 - x))
        // dest.2 = (dest.2 << x) | (dest.1 >> (32 - x))
        // dest.1 = (dest.1 << x) | (dest.0 >> (32 - x))
        // dest.0 = (dest.0 << x)
        let ctx = self.ctx;
        let x_arr = [
            with & 0x1f,
            with & 0x20,
            (with & !0x3f) >> 1,
        ];
        let dest_zero = self.r_to_dest_and_operand_xmm(dest, 0);
        let dests: [_; 3] = array_init::array_init(|i| {
            dest.dest_operand_xmm((i as u8).wrapping_add(1))
        });
        let ops: [_; 4] = array_init::array_init(|i| self.r_to_operand_xmm(dest, i as u8));
        for &x in &x_arr {
            for i in (1..4).rev() {
                self.output_mov(
                    dests[i - 1],
                    ctx.and_const(
                        ctx.or(
                            ctx.lsh_const(ops[i], x as u64),
                            ctx.rsh_const(ops[i - 1], 0x20u8.wrapping_sub(x) as u64),
                        ),
                        0xffff_ffff,
                    ),
                );
            }
            self.output_mov(
                dest_zero.dest,
                ctx.and_const(
                    ctx.lsh_const(
                        dest_zero.op,
                        x as u64,
                    ),
                    0xffff_ffff,
                ),
            );
        }
        for _ in 0..4 {
            let val = self.out[self.out.len() - 4].clone();
            self.output(val);
        }
    }

    fn packed_shift_right_xmm_u64(&mut self, dest: ModRm_R, with: Operand<'e>) {
        // let x = with & 0x1f
        // dest.0 = (dest.0 >> x) | (dest.1 << (32 - x))
        // dest.1 = (dest.1 >> x)
        // dest.2 = (dest.2 >> x) | (dest.3 << (32 - x))
        // dest.3 = (dest.3 >> x)
        // ...
        let ctx = self.ctx;
        let x_arr = [
            ctx.and_const(with, 0x1f),
            ctx.and_const(with, !0x1f),
        ];
        for &x in &x_arr {
            for i in 0..2 {
                let low_id = i * 2;
                let high_id = low_id + 1;
                let low = self.r_to_operand_xmm(dest, low_id);
                let high = self.r_to_operand_xmm(dest, high_id);
                self.output_arith(
                    dest.dest_operand_xmm(low_id),
                    ArithOpType::Or,
                    ctx.rsh(low, x),
                    ctx.lsh(high, ctx.sub_const_left(0x20, x)),
                );
                self.output_rsh(self.r_to_dest_and_operand_xmm(dest, high_id), x);
            }
        }
    }

    fn packed_shift_right_xmm_u128(&mut self, dest: ModRm_R, with: u8) {
        // let x = with & 0x1f
        // dest.0 = (dest.0 >> x) | (dest.1 << (32 - x))
        // dest.1 = (dest.1 >> x) | (dest.2 << (32 - x))
        // dest.2 = (dest.2 >> x) | (dest.3 << (32 - x))
        // dest.3 = (dest.3 >> x)
        // let x = with & 0x20
        // ...
        let ctx = self.ctx;
        let x_arr = [
            with & 0x1f,
            with & 0x20,
            (with & !0x3f) >> 1,
        ];
        let dest_three = self.r_to_dest_and_operand_xmm(dest, 3);
        let dests: [_; 3] = array_init::array_init(|i| dest.dest_operand_xmm(i as u8));
        let ops: [_; 4] = array_init::array_init(|i| self.r_to_operand_xmm(dest, i as u8));
        for &x in &x_arr {
            for i in 0..3 {
                self.output_mov(
                    dests[i],
                    ctx.and_const(
                        ctx.or(
                            ctx.rsh_const(ops[i], x as u64),
                            ctx.lsh_const(ops[i + 1], 0x20u8.wrapping_sub(x) as u64),
                        ),
                        0xffff_ffff,
                    ),
                );
            }
            let op = self.ctx.rsh_const(dest_three.op, x as u64);
            self.output_mov(dest_three.dest, op);
        }
        for _ in 0..4 {
            let val = self.out[self.out.len() - 4].clone();
            self.output(val);
        }
    }

    fn packed_shift_left(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        // Zero everything if rm.1 is set
        let rm_0 = self.rm_to_operand_xmm(&rm, 0);
        self.packed_shift_left_xmm_u64(dest, rm_0);
        self.zero_xmm_if_rm1_nonzer0(&rm, dest);
        Ok(())
    }

    fn packed_shift_right(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32);
        let rm_0 = self.rm_to_operand_xmm(&rm, 0);
        // Zero everything if rm.1 is set
        self.packed_shift_right_xmm_u64(dest, rm_0);
        self.zero_xmm_if_rm1_nonzer0(&rm, dest);
        Ok(())
    }

    fn pextrw(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, dest, imm) = self.parse_modrm_imm(MemAccessSize::Mem32, MemAccessSize::Mem8)?;
        if rm.is_memory() {
            return Err(self.unknown_opcode());
        }
        if let Some(word) = imm.if_constant() {
            let op = self.rm_to_operand_xmm_size(&rm, word as u8 & 7, MemAccessSize::Mem16);
            self.output_mov(dest.dest_operand(), op);
        }
        Ok(())
    }

    fn fpu_push(&mut self) {
        // fdecstp
        self.output(special_op(&[0xd9, 0xf6]).unwrap());
    }

    fn fpu_pop(&mut self) {
        // fincstp
        self.output(special_op(&[0xd9, 0xf7]).unwrap());
    }

    fn various_d8(&mut self) -> Result<(), Failed> {
        static TYPES: [Option<ArithOpType>; 8] = [
            Some(ArithOpType::Add), // Fadd
            Some(ArithOpType::Mul), // Fmul
            None,
            None,
            Some(ArithOpType::Sub), // Fsub
            Some(ArithOpType::Sub), // Fsubr
            Some(ArithOpType::Div), // Fdiv
            Some(ArithOpType::Div), // Fdivr
        ];

        let byte = self.read_u8(1)?;
        let variant = (byte >> 3) & 0x7;
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem32);
        let op_ty = TYPES[variant as usize]
            .ok_or_else(|| self.unknown_opcode())?;
        let rm = self.rm_to_operand(&rm);
        let st0 = self.ctx.register_fpu(0);
        let dest = DestOperand::Fpu(0);
        let (lhs, rhs) = if variant == 5 || variant == 7 {
            (rm, st0)
        } else {
            (st0, rm)
        };
        let op = self.ctx.float_arithmetic(op_ty, lhs, rhs, MemAccessSize::Mem32);
        self.output_mov(dest, op);
        Ok(())
    }

    fn various_d9(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(1)?;
        let variant = (byte >> 3) & 0x7;
        if variant == 6 && byte == 0xf6 || byte == 0xf7 {
            // Fincstp, fdecstp
            self.output(special_op(&self.data)?);
            return Ok(());
        }
        let (rm_parsed, _) = self.parse_modrm(MemAccessSize::Mem32);
        let rm = self.rm_to_dest_and_operand(&rm_parsed);
        let ctx = self.ctx;
        match variant {
            // Fld
            0 => {
                self.fpu_push();
                self.output_mov(DestOperand::Fpu(0), x87_variant(ctx, rm.op, 1));
                Ok(())
            }
            // Fst/Fstp, as long as rm is mem
            2 | 3 => {
                self.output_mov(rm.dest, ctx.register_fpu(0));
                if variant == 3 {
                    self.fpu_pop();
                }
                Ok(())
            }
            // Fstenv
            6 => {
                let mem_size = self.mem16_32();
                let mem_bytes = match mem_size {
                    MemAccessSize::Mem16 => 2,
                    _ => 4,
                };
                if let Some(mem) = rm.op.if_memory() {
                    let ctx = self.ctx;
                    let (base, offset) = mem.address();
                    for i in 0..10 {
                        let offset = offset.wrapping_add(i * mem_bytes);
                        let access = ctx.mem_access(base, offset, mem_size);
                        self.output_mov(DestOperand::Memory(access), ctx.new_undef());
                    }
                }
                Ok(())
            }
            // Fstcw
            7 => {
                if rm_parsed.is_memory() {
                    self.output_mov(rm.dest, ctx.new_undef());
                }
                Ok(())
            }
            _ => return Err(self.unknown_opcode()),
        }
    }

    fn various_dd(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(1)?;
        let variant = (byte >> 3) & 0x7;
        let (rm_parsed, _) = self.parse_modrm(MemAccessSize::Mem64);
        let rm = self.rm_to_dest_and_operand(&rm_parsed);
        let ctx = self.ctx;
        match variant {
            // Fld f64, as long as rm is mem
            // Fpu is defined to hold f32?? I guess?? fpu is quite underdefined
            // and i don't want to add f80 which would be the most correct, so
            // going with f32 since it was assumed first.
            // Means that this has to do ToFloat()
            0 => {
                self.fpu_push();
                let op = ctx.float_arithmetic(
                    ArithOpType::ToFloat,
                    rm.op,
                    ctx.const_0(),
                    MemAccessSize::Mem64,
                );
                self.output_mov(DestOperand::Fpu(0), op);
                Ok(())
            }
            // Fst/Fstp f64, as long as rm is mem
            2 | 3 => {
                let op = ctx.float_arithmetic(
                    ArithOpType::ToDouble,
                    ctx.register_fpu(0),
                    ctx.const_0(),
                    MemAccessSize::Mem32,
                );
                self.output_mov(rm.dest, op);
                if variant == 3 {
                    self.fpu_pop();
                }
                Ok(())
            }
            _ => return Err(self.unknown_opcode()),
        }
    }

    fn various_fe_ff(&mut self) -> Result<(), Failed> {
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        let is_64 = Va::SIZE == 8;
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => match is_64 {
                true => match variant {
                    // Call / jump are always 64
                    2 | 3 | 4 | 5 => MemAccessSize::Mem64,
                    _ => self.mem16_32(),
                },
                false => self.mem16_32(),
            },
        };
        let (rm, _) = self.parse_modrm(op_size);
        let rm = self.rm_to_dest_and_operand(&rm);
        match variant {
            0 | 1 => {
                let is_inc = variant == 0;
                let op = rm.op;
                match is_inc {
                    true => self.output_add_const(rm, 1),
                    false => self.output_sub_const(rm, 1),
                }
                self.inc_dec_flags(is_inc, op, op_size);
            }
            2 | 3 => self.output(Operation::Call(rm.op)),
            4 | 5 => self.output(Operation::Jump { condition: self.ctx.const_1(), to: rm.op }),
            6 => {
                let new_esp = self.register_cache.esp_neg_word_offset();
                self.output_mov_to_reg(4, new_esp);
                let dest = DestOperand::Memory(self.register_cache.esp_mem());
                self.output_mov(dest, rm.op);
            }
            _ => return Err(self.unknown_opcode()),
        }
        Ok(())
    }

    fn pop_rm(&mut self) -> Result<(), Failed> {
        let (rm, _) = self.parse_modrm(self.mem16_32());
        let rm_dest = self.rm_to_dest_operand(&rm);
        let esp_mem = self.register_cache.esp_mem_word();
        self.output_mov(rm_dest, esp_mem);
        let new_esp = self.register_cache.esp_pos_word_offset();
        self.output_mov_to_reg(4, new_esp);
        Ok(())
    }

    fn bitwise_with_imm_op(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (_, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
        if imm == self.ctx.const_0() {
            return Ok(());
        }

        let arith = BITWISE_ARITH_OPS[((self.read_u8(1)? >> 3) & 0x7) as usize]
            .ok_or_else(|| self.unknown_opcode())?;
        self.generic_arith_with_imm_op(arith, MemAccessSize::Mem8)
    }

    fn bitwise_compact_op(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (mut rm, _) = self.parse_modrm(op_size);
        let shift_count = match self.read_u8(0)? & 2 {
            0 => self.ctx.const_1(),
            _ => self.reg_variable_size(1, operand::MemAccessSize::Mem8).clone(),
        };
        let arith = BITWISE_ARITH_OPS[((self.read_u8(1)? >> 3) & 0x7) as usize]
            .ok_or_else(|| self.unknown_opcode())?;
        self.do_arith_operation(arith, &mut rm, shift_count);
        Ok(())
    }

    fn signed_multiply_rm_imm(&mut self) -> Result<(), Failed> {
        let imm_size = match self.read_u8(0)? & 0x2 {
            2 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let ctx = self.ctx;
        let op_size = self.mem16_32();
        let (rm, r, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        let rm = self.rm_to_operand(&rm);
        // TODO flags, imul only sets c and o on overflow
        // TODO Signed mul isn't sensible if it doesn't contain operand size
        // I don't think any of these sizes are correct but w/e
        let dest = r.dest_operand();
        if Va::SIZE == 4 {
            if op_size != MemAccessSize::Mem32 {
                self.output_mov(dest, ctx.signed_mul(rm, imm, op_size));
            } else {
                self.output_mov(dest, ctx.mul(rm, imm));
            }
        } else {
            if op_size != MemAccessSize::Mem64 {
                self.output_mov(dest, ctx.signed_mul(rm, imm, op_size));
            } else {
                self.output_mov(dest, ctx.mul(rm, imm));
            }
        }
        Ok(())
    }

    fn shld_imm(&mut self) -> Result<(), Failed> {
        let (hi, low, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let ctx = self.ctx;
        let imm = ctx.and_const(imm, 0x1f);
        let hi = self.rm_to_dest_and_operand(&hi);
        if imm != ctx.const_0() {
            // TODO flags
            let low = self.r_to_operand(low);
            self.output_mov(
                hi.dest,
                ctx.or(
                    ctx.lsh(
                        hi.op,
                        imm,
                    ),
                    ctx.rsh(
                        low,
                        ctx.sub_const_left(0x20, imm)
                    ),
                ),
            );
        }
        Ok(())
    }

    fn shrd_imm(&mut self) -> Result<(), Failed> {
        let (low, hi, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let ctx = self.ctx;
        let imm = ctx.and_const(imm, 0x1f);
        let low = self.rm_to_dest_and_operand(&low);
        if imm != ctx.const_0() {
            // TODO flags
            let hi = self.r_to_operand(hi);
            self.output_mov(
                low.dest,
                ctx.or(
                    ctx.rsh(
                        low.op,
                        imm,
                    ),
                    ctx.lsh(
                        hi,
                        ctx.sub_const_left(0x20, imm)
                    ),
                ),
            );
        }
        Ok(())
    }

    fn imul_normal(&mut self) -> Result<(), Failed> {
        let size = self.mem16_32();
        let (rm, r) = self.parse_modrm(size);
        let rm = self.rm_to_operand(&rm);
        // TODO flags, imul only sets c and o on overflow
        // Signed multiplication should be different only when result is being sign extended.
        let dest = self.r_to_dest_and_operand(r);
        if size.bits() != Va::SIZE * 8 {
            self.output_signed_mul(dest, rm, size);
        } else {
            self.output_mul(dest, rm);
        }
        Ok(())
    }

    fn cmpxchg(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _r) = self.parse_modrm(op_size);
        let dest = self.rm_to_dest_operand(&rm);
        let ctx = self.ctx;
        self.output_mov(dest, ctx.new_undef());
        self.output_mov(DestOperand::reg_variable_size(0, op_size), ctx.new_undef());
        Ok(())
    }

    fn arith_with_imm_op(&mut self) -> Result<(), Failed> {
        let arith = ARITH_MAPPING[((self.read_u8(1)? >> 3) & 0x7) as usize];
        let imm_size = match self.read_u8(0)? & 0x3 {
            0 | 2 | 3 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        self.generic_arith_with_imm_op(arith, imm_size)
    }

    /// Skips and masking arith input on x86 by converting register
    /// to full size if the high bits of arithmetic cannot affect
    /// low 32 bits of the result.
    fn skip_unnecessary_32bit_operand_masking(
        &mut self,
        rm: &mut ModRm_Rm,
        arith: ArithOperation,
    ) {
        use self::ArithOperation::*;
        if Va::SIZE == 4 && !rm.is_memory() && rm.size == RegisterSize::R32 {
            match arith {
                Add | Or | Adc | Sbb | And | Sub | Xor | Cmp | Move | Test | LeftShift => {
                    rm.size = RegisterSize::R64;
                }
                RightShift | RightShiftArithmetic | RotateLeft | RotateRight => (),
            }
        }
    }

    fn generic_arith_with_imm_op(
        &mut self,
        arith: ArithOperation,
        imm_size: MemAccessSize,
    ) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (mut rm, _, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        self.do_arith_operation(arith, &mut rm, imm);
        Ok(())
    }

    fn pushpop_reg_op(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(0)?;
        let is_push = byte < 0x58;
        let reg = if self.rex_prefix() & 0x1 == 0 {
            byte & 0x7
        } else {
            8 + (byte & 0x7)
        };
        match is_push {
            true => {
                let new_esp = self.register_cache.esp_neg_word_offset();
                self.output_mov_to_reg(4, new_esp);
                let reg = self.ctx.register(reg);
                let dest = DestOperand::Memory(self.register_cache.esp_mem());
                self.output_mov(dest, reg);
            }
            false => {
                let esp_mem = self.register_cache.esp_mem_word();
                self.output_mov_to_reg(reg, esp_mem);
                let new_esp = self.register_cache.esp_pos_word_offset();
                self.output_mov_to_reg(4, new_esp);
            }
        }
        Ok(())
    }

    fn push_imm(&mut self) -> Result<(), Failed> {
        let imm_size = match self.read_u8(0)? {
            0x68 => self.mem16_32(),
            _ => MemAccessSize::Mem8,
        };
        let constant = self.read_variable_size_32(1, imm_size)? as u32;
        let new_esp = self.register_cache.esp_neg_word_offset();
        self.output_mov_to_reg(4, new_esp);
        let dest = DestOperand::Memory(self.register_cache.esp_mem());
        self.output_mov(dest, self.ctx.constant(constant as u64));
        Ok(())
    }
}

impl<'a, 'e: 'a> InstructionOpsState<'a, 'e, VirtualAddress32> {
    fn conditional_jmp(&mut self, op_size: MemAccessSize) -> Result<(), Failed> {
        let offset = self.read_variable_size_signed(1, op_size)?;
        let from = self.address.0.wrapping_add(self.len() as u32);
        let to = self.ctx.constant(from.wrapping_add(offset) as u64);
        let condition = self.condition()?;
        self.output(Operation::Jump { condition, to });
        Ok(())
    }

    fn short_jmp(&mut self) -> Result<(), Failed> {
        let offset = self.read_variable_size_signed(1, MemAccessSize::Mem8)?;
        let from = self.address.0.wrapping_add(self.len() as u32);
        let to = self.ctx.constant(from.wrapping_add(offset) as u64);
        self.output(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(())
    }

    fn call_op(&mut self) -> Result<(), Failed> {
        let offset = self.read_u32(1)?;
        let from = self.address.0.wrapping_add(self.len() as u32);
        let to = self.ctx.constant(from.wrapping_add(offset) as u64);
        self.output(Operation::Call(to));
        Ok(())
    }

    fn jump_op(&mut self) -> Result<(), Failed> {
        let offset = self.read_u32(1)?;
        let from = self.address.0.wrapping_add(self.len() as u32);
        let to = self.ctx.constant(from.wrapping_add(offset) as u64);
        self.output(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(())
    }
}

impl<'a, 'e: 'a> InstructionOpsState<'a, 'e, VirtualAddress64> {
    fn conditional_jmp(&mut self, op_size: MemAccessSize) -> Result<(), Failed> {
        let offset = self.read_variable_size_signed(1, op_size)? as i32 as i64 as u64;
        let to = self.ctx.constant((self.address.0 + self.len() as u64).wrapping_add(offset));
        let condition = self.condition()?;
        self.output(Operation::Jump { condition, to });
        Ok(())
    }

    fn short_jmp(&mut self) -> Result<(), Failed> {
        let offset = self.read_variable_size_signed(1, MemAccessSize::Mem8)?
            as i32 as i64 as u64;
        let to = self.ctx.constant((self.address.0 + self.len() as u64).wrapping_add(offset));
        self.output(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(())
    }

    fn call_op(&mut self) -> Result<(), Failed> {
        let offset = self.read_u32(1)? as i32 as i64 as u64;
        let to = self.ctx.constant((self.address.0 + self.len() as u64).wrapping_add(offset));
        self.output(Operation::Call(to));
        Ok(())
    }

    fn jump_op(&mut self) -> Result<(), Failed> {
        let offset = self.read_u32(1)? as i32 as i64 as u64;
        let to = self.ctx.constant((self.address.0 + self.len() as u64).wrapping_add(offset));
        self.output(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(())
    }
}

/// Checks if r is a register, and rm is the equivalent short register
fn is_rm_short_r_register(rm: &ModRm_Rm, r: ModRm_R) -> bool {
    !rm.is_memory() && rm.base == r.0 && rm.size.bits() < r.1.bits()
}

pub mod operation_helpers {
    use crate::operand::{Operand, MemAccessSize};
    use super::{DestOperand, Operation};

    pub fn mov_to_reg<'e>(dest: u8, from: Operand<'e>) -> Operation<'e> {
        Operation::Move(DestOperand::Register64(dest), from, None)
    }

    pub fn mov_to_reg_variable_size<'e>(
        size: MemAccessSize,
        dest: u8,
        from: Operand<'e>,
    ) -> Operation<'e> {
        let dest = match size {
            MemAccessSize::Mem8 => DestOperand::Register8Low(dest),
            MemAccessSize::Mem16 => DestOperand::Register16(dest),
            MemAccessSize::Mem32 => DestOperand::Register32(dest),
            MemAccessSize::Mem64 => DestOperand::Register64(dest),
        };
        Operation::Move(dest, from, None)
    }
}

/// A sub-operation of simulated instruction.
///
/// As the analysis walks through code, it translates instructions to
/// zero or more `Operation`s, which are used as input for analysis
/// to decide which branches are to be analyzed, and to update
/// [`ExecutionState`](crate::exec_state::ExecutionState).
///
/// User-side code gets gets to intercept any `Operation` before it
/// is otherwise processed in [`Analyzer::operation`](crate::analysis::Analyzer::operation)
/// callback, which is the usually where most of the logic for user's code is.
///
/// In `Analyzer::operation`, [`skip_operation`](crate::analysis::Control::skip_operation)
/// can be used to prevent the default `Operation` from passsing through(*).
/// Whereas [`update`](crate::exec_state::ExecutionState::update), as well as
/// [`move_resolved`](crate::analysis::Control::move_resolved), and
/// [`move_unresolved`](crate::analysis::Control::move_unresolved) can be used to add
/// additional `Operation`s to be processed by scarf.
///
/// (*) `skip_operation` and `update` do not do anything for control flow operations
/// `Operation::Jump` and `Operation::Return`. This probably should be fixed, but
/// [`end_branch`](crate::analysis::Control::end_branch) and
/// [`add_branch_with_current_state`](crate::analysis::Control::add_branch_with_current_state)
/// can be used to work around this limitation.
///
/// As `Operation` is representing CPU instructions without any external state,
/// all [`Operand`]s and `DestOperand`s in `Operation` are always
/// [unresolved](../exec_state/trait.ExecutionState.html#resolved-and-unresolved-operands).
#[derive(Clone, Copy, Debug)]
pub enum Operation<'e> {
    /// Set `DestOperand` to `Operand`.
    ///
    /// If `Option<Operand>` exists, it is taken as a condition, and `DestOperand` will only
    /// be updated if the condition is nonzero.
    ///
    /// The conditional moves are generated quite rarely, and in practice they usually
    /// cause move of [`Undefined`](../analysis/struct.FuncAnalysis.html#state-merging-and-loops)
    /// to the `DestOperand`.
    Move(DestOperand<'e>, Operand<'e>, Option<Operand<'e>>),

    /// Calls the function at `Operand`.
    ///
    /// `ExecutionState` just implements this as a "Clear all non-preserved registers"
    /// operation, but the user [`Analyzer`](crate::analysis::Analyzer) code can often
    /// have reasons to handle calls.
    ///
    /// Note that for performance and usability reasons, `ExecutionState` only clears registers
    /// on functions calls. Any memory is assumed to stay unchanged, which sometimes will
    /// cause analysis to be confused when a function was called to write something to a
    /// pointer that will be read later. For now, these cases have to be handled by the user
    /// code if they pop up.
    Call(Operand<'e>),

    /// Jumps to `to` if `condition` is nonzero.
    ///
    /// If the analysis is able to determine condition to be constant, it will only
    /// analyze the branch that is taken. Otherwise both branches will be queued to
    /// be analyzed.
    ///
    /// Note that the analysis calls
    /// [`resolve_apply_constraints`](crate::exec_state::ExecutionState::resolve_apply_constraints)
    /// to resolve the condition, which sometimes gives better results, especially with jump
    /// conditions, than the regular [`resolve`](crate::exec_state::ExecutionState::resolve)
    /// would.
    Jump { condition: Operand<'e>, to: Operand<'e> },
    /// Returns to caller.
    ///
    /// The `u32` parameter is additional stack bytes popped after the return
    /// address has been popped. (E.g. x86 stdcall with 3 arguments would pop 12 bytes)
    ///
    /// Analysis will consider this operation a branch end.
    Return(u32),
    /// Special instructions that are not representable by other `Operation`s.
    /// For example, `rep mov` is represented with `Special`. The bytes ideally
    /// will represent the instruction bytes, but in the end their interpretation
    /// is very much up to ExecutionState. `rep mov` for example, is completely
    /// ignored, but the user [`Analyzer`](crate::analysis::Analyzer) code can recognize
    /// the instruction bytes and do its own handling if deemed necessary.
    Special(SpecialBytes),
    /// Set flags based on operation type. While Move(..) could handle this
    /// (And it does for odd cases like inc), it would mean generating 5
    /// additional operations for each instruction, so special-case flags.
    SetFlags(FlagUpdate<'e>),
    /// Makes the following `Operation`s until `Unfreeze` be buffered, resolving
    /// input `Operand`s, without mutating the state.
    ///
    /// Allows implementing operations that write to several outputs, which are
    /// simultaneously used as inputs. For example, long mul, div (Result + modulo), swap,
    /// carry add/sub.
    ///
    /// Cloning state which is in middle of freeze will not clone any buffered operations;
    /// the new copy can be updated but any buffered writes are lost.
    Freeze,
    /// Commits any buffered operations since `Freeze` was used to buffer them.
    Unfreeze,
    /// Error - Should assume that no more operations can be decoded from current position.
    Error(Error),
}

/// Part of [`Operation`], representing an update to
/// [`ExecutionState`s](crate::exec_state::ExecutionState) flags.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FlagUpdate<'e> {
    pub left: Operand<'e>,
    pub right: Operand<'e>,
    pub ty: FlagArith,
    pub size: MemAccessSize,
}

/// Operations which operate on 2 operands, (almost always) writing to lhs.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
enum ArithOperation {
    // First 8 are same order as x86 internally, maybe compiler can optimize
    // some matches.
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    Cmp,
    Move,
    Test,
    RotateLeft,
    RotateRight,
    LeftShift,
    RightShift,
    RightShiftArithmetic,
}

/// Operations used by `[FlagUpdate]`.
///
/// Contains some operations that the 'primary' arithmetic operation
/// enum [`ArithOpType`] implements in terms of other operations.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum FlagArith {
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    RotateLeft,
    RotateRight,
    LeftShift,
    RightShift,
    RightShiftArithmetic,
}

// Maybe it would be better to pass RegisterSize as an argument to functions
// which use it? It goes unused with xmm. Though maybe better to keep
// its calculation at one point.
#[allow(bad_style)]
#[derive(Copy, Clone)]
struct ModRm_R(u8, RegisterSize);

#[allow(bad_style)]
#[derive(Clone)]
struct ModRm_Rm {
    size: RegisterSize,
    /// 0xff or 0xfe signifies that this is constant base.
    base: u8,
    /// u8::max_value signifies that this is register instead of memory operand.
    /// (Register is in `base`)
    index: u8,
    index_mul: u8,
    /// Rip-relative if 64-bit and base == 0xff
    constant: u32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum RegisterSize {
    // NOTE: Variant amount / ordering being like this is used by ModRm_R::dest_operand
    Low8,
    High8,
    R16,
    R32,
    R64,
}

// LLVM seems to be unable to convert matching on RegisterSize / MemAccessSize
// to array read, so this does it explicitly.
impl RegisterSize {
    fn bits(self) -> u32 {
        // Seems that while static works better when output is enum,
        // this is better for integers.
        [8, 8, 16, 32, 64][self as usize] as u32
    }

    fn to_mem_access_size(self) -> MemAccessSize {
        static MAPPING: [MemAccessSize; 5] = [
            MemAccessSize::Mem8,
            MemAccessSize::Mem8,
            MemAccessSize::Mem16,
            MemAccessSize::Mem32,
            MemAccessSize::Mem64,
        ];
        MAPPING[self as usize]
    }

    fn from_mem_access_size(size: MemAccessSize) -> RegisterSize {
        static MAPPING: [RegisterSize; 4] = {
            let mut map = [RegisterSize::Low8; 4];
            map[MemAccessSize::Mem8 as usize] = RegisterSize::Low8;
            map[MemAccessSize::Mem16 as usize] = RegisterSize::R16;
            map[MemAccessSize::Mem32 as usize] = RegisterSize::R32;
            map[MemAccessSize::Mem64 as usize] = RegisterSize::R64;
            map
        };
        MAPPING[size as usize]
    }
}

impl ModRm_R {
    #[inline(never)]
    fn dest_operand<'e>(self) -> DestOperand<'e> {
        let reg = self.0;
        [
            DestOperand::Register8Low(reg),
            DestOperand::Register8High(reg),
            DestOperand::Register16(reg),
            DestOperand::Register32(reg),
            DestOperand::Register64(reg),
        ][self.1 as usize]
    }

    fn dest_operand_xmm<'e>(self, i: u8) -> DestOperand<'e> {
        DestOperand::Xmm(self.0, i)
    }

    fn equal_to_rm(self, rm: &ModRm_Rm) -> bool {
        self.0 == rm.base && !rm.is_memory()
    }

    fn to_rm(self) -> ModRm_Rm {
        ModRm_Rm {
            size: self.1,
            base: self.0,
            index: u8::MAX,
            index_mul: 0,
            constant: 0,
        }
    }
}

impl ModRm_Rm {
    fn is_memory(&self) -> bool {
        self.index != u8::MAX
    }

    fn constant_base(&self) -> bool {
        self.base >= 0x80
    }

    fn rip_relative(&self) -> bool {
        self.base == 0xff
    }

    fn reg_variable_size(reg: u8, size: MemAccessSize) -> ModRm_Rm {
        ModRm_Rm {
            size: RegisterSize::from_mem_access_size(size),
            base: reg,
            index: u8::MAX,
            index_mul: 0,
            constant: 0,
        }
    }
}

pub struct DestAndOperand<'e> {
    pub dest: DestOperand<'e>,
    pub op: Operand<'e>,
}

// #[derive(Clone)] would have inline hint
impl<'e> Clone for DestAndOperand<'e> {
    fn clone(&self) -> DestAndOperand<'e> {
        DestAndOperand {
            dest: self.dest.clone(),
            op: self.op,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DestOperand<'e> {
    Register64(u8),
    Register32(u8),
    Register16(u8),
    Register8High(u8),
    Register8Low(u8),
    Xmm(u8, u8),
    Fpu(u8),
    Flag(Flag),
    Memory(MemAccess<'e>),
}

impl<'e> DestOperand<'e> {
    pub fn reg_variable_size(reg: u8, size: MemAccessSize) -> DestOperand<'e> {
        match size {
            MemAccessSize::Mem8 => DestOperand::Register8Low(reg),
            MemAccessSize::Mem16 => DestOperand::Register16(reg),
            MemAccessSize::Mem32 => DestOperand::Register32(reg),
            MemAccessSize::Mem64 => DestOperand::Register64(reg),
        }
    }

    pub fn from_oper(val: Operand<'e>) -> DestOperand<'e> {
        dest_operand(val)
    }

    /// Creates `DestOperand` referring to memory specified by `mem`.
    #[inline]
    pub fn memory(mem: &MemAccess<'e>) -> DestOperand<'e> {
        DestOperand::Memory(*mem)
    }

    /// Creates `DestOperand` by building `MemAccess` from the arguments.
    #[inline]
    pub fn make_memory(
        ctx: OperandCtx<'e>,
        base: Operand<'e>,
        offset: u64,
        size: MemAccessSize,
    ) -> DestOperand<'e> {
        Self::memory(&ctx.mem_access(base, offset, size))
    }

    pub fn as_operand(&self, ctx: OperandCtx<'e>) -> Operand<'e> {
        match *self {
            DestOperand::Register32(x) => ctx.and_const(ctx.register(x), 0xffff_ffff),
            DestOperand::Register16(x) => ctx.and_const(ctx.register(x), 0xffff),
            DestOperand::Register8High(x) => ctx.rsh_const(
                ctx.and_const(ctx.register(x), 0xffff),
                8,
            ),
            DestOperand::Register8Low(x) => ctx.and_const(ctx.register(x), 0xff),
            DestOperand::Register64(x) => ctx.register(x),
            DestOperand::Xmm(x, y) => ctx.xmm(x, y),
            DestOperand::Fpu(x) => ctx.register_fpu(x),
            DestOperand::Flag(x) => ctx.flag(x).clone(),
            DestOperand::Memory(ref x) => ctx.memory(x),
        }
    }

    /// Returns MemAccessSize that any assignments to this DestOperand will be masked to.
    pub(crate) fn size(&self) -> MemAccessSize {
        if let DestOperand::Memory(ref mem) = *self {
            mem.size
        } else {
            match *self {
                DestOperand::Register32(..) | DestOperand::Xmm(..) |
                    DestOperand::Fpu(..) => MemAccessSize::Mem32,
                DestOperand::Register16(..) => MemAccessSize::Mem16,
                DestOperand::Register8High(..) |
                    DestOperand::Register8Low(..) => MemAccessSize::Mem8,
                // Flag maybe could be Mem8? But currently assignments to flags aren't masked so
                DestOperand::Register64(..) | DestOperand::Flag(..) |
                    DestOperand::Memory(..) => MemAccessSize::Mem64,
            }
        }
    }
}

fn dest_operand<'e>(val: Operand<'e>) -> DestOperand<'e> {
    use crate::operand::OperandType::*;
    match *val.ty() {
        Register(x) => DestOperand::Register64(x),
        Xmm(x, y) => DestOperand::Xmm(x, y),
        Fpu(x) => DestOperand::Fpu(x),
        Flag(x) => DestOperand::Flag(x),
        Memory(ref x) => DestOperand::Memory(x.clone()),
        Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let result = arith.right.if_constant()
                        .and_then(|c| {
                            let reg = arith.left.if_register()?;
                            match c {
                                0xff => Some(DestOperand::Register8Low(reg)),
                                0xffff => Some(DestOperand::Register16(reg)),
                                0xffff_ffff => Some(DestOperand::Register32(reg)),
                                _ => None,
                            }
                        });
                    if let Some(result) = result {
                        return result;
                    }
                }
                ArithOpType::Rsh => {
                    if arith.right.if_constant() == Some(8) {
                        let reg = arith.left.if_arithmetic_and()
                            .and_then(|(l, r)| {
                                r.if_constant()
                                    .filter(|&c| c == 0xff00)?;
                                l.if_register()
                            });
                        if let Some(reg) = reg {
                            return DestOperand::Register8High(reg);
                        }
                    }
                }
                _ => (),
            }
            // Avoid adding operand formatting code in binary for this if it isn't needed
            // elsewhere.
            #[cfg(not(debug_assertions))]
            panic!("Invalid value for converting Operand -> DestOperand");
            #[cfg(debug_assertions)]
            panic!("Invalid value for converting Operand -> DestOperand {}", val);
        }
        #[cfg(not(debug_assertions))]
        _ => panic!("Invalid value for converting Operand -> DestOperand"),
        #[cfg(debug_assertions)]
        _ => panic!("Invalid value for converting Operand -> DestOperand {}", val),
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpecialBytes {
    data: [u8; 8],
    length: u8,
}

impl std::ops::Deref for SpecialBytes {
    type Target = [u8];
    fn deref(&self) -> &Self::Target {
        &self.data[..(self.length as usize & 7)]
    }
}

#[inline]
fn special_op<'e>(bytes: &[u8]) -> Result<Operation<'e>, Failed> {
    if bytes.len() >= 8 {
        Err(Failed)
    } else {
        let mut data = [0u8; 8];
        for i in 0..bytes.len() {
            data[i] = bytes[i];
        }
        Ok(Operation::Special(SpecialBytes { data, length: bytes.len() as u8 }))
    }
}
