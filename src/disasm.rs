use lde::Isa;
use quick_error::quick_error;

use crate::exec_state::{VirtualAddress};
use crate::operand::{
    self, ArithOpType, Flag, MemAccess, Operand, OperandCtx, OperandType, Register,
    MemAccessSize, ArithOperand,
};
use crate::VirtualAddress as VirtualAddress32;
use crate::VirtualAddress64;

quick_error! {
    // NOTE: Try avoid making this have a destructor
    #[derive(Debug, Copy, Clone)]
    pub enum Error {
        // First 8 bytes of the instruction should be enough information
        UnknownOpcode(op: [u8; 8], len: u8) {
            display("Unknown opcode {:02x?}", &op[..*len as usize])
        }
        End {
            display("End of file")
        }
        InternalDecodeError {
            display("Internal decode error")
        }
    }
}

/// Used by InstructionOpsState to signal that something had failed and return with ?
/// without making return value heavier.
/// Error should be stored in &mut self
struct Failed;

pub type OperationVec<'e> = Vec<Operation<'e>>;

pub struct Disassembler32<'e> {
    buf: &'e [u8],
    pos: usize,
    virtual_address: VirtualAddress32,
    register_cache: RegisterCache<'e>,
    ops_buffer: Vec<Operation<'e>>,
    ctx: OperandCtx<'e>,
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
        if buf.len() > 4 && &buf[..3] == &[0x66, 0x0f, 0x73] {
            // Another lde bug
            5
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
        if buf.len() > 4 && &buf[..3] == &[0x66, 0x0f, 0x73] {
            // Another lde bug
            5
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
    fn new(ctx: OperandCtx<'a>) -> Disassembler32<'a> {
        Disassembler32 {
            buf: &[],
            pos: 0,
            virtual_address: VirtualAddress32(0),
            register_cache: RegisterCache::new(ctx, false),
            ops_buffer: Vec::with_capacity(16),
            ctx,
        }
    }

    fn set_pos(&mut self, buf: &'a [u8], pos: usize, address: VirtualAddress32) {
        assert!(pos < buf.len());
        self.buf = buf;
        self.pos = pos;
        self.virtual_address = address;
    }

    fn next<'s>(&'s mut self) -> Instruction<'s, 'a, VirtualAddress32> {
        let length = instruction_length_32(&self.buf[self.pos..]);
        let address = self.virtual_address + self.pos as u32;
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
            let data = &self.buf[self.pos..self.pos + length];
            instruction_operations32(
                address,
                data,
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
        self.virtual_address + self.pos as u32
    }
}

pub struct Disassembler64<'e> {
    buf: &'e [u8],
    pos: usize,
    virtual_address: VirtualAddress64,
    register_cache: RegisterCache<'e>,
    ops_buffer: Vec<Operation<'e>>,
    ctx: OperandCtx<'e>,
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler64<'a> {
    type VirtualAddress = VirtualAddress64;

    // Inline(never) seems to help binary size *enough* and this function
    // is only called once per function-to-be-analyzed
    #[inline(never)]
    fn new(ctx: OperandCtx<'a>) -> Disassembler64<'a> {
        Disassembler64 {
            buf: &[],
            pos: 0,
            virtual_address: VirtualAddress64(0),
            register_cache: RegisterCache::new(ctx, true),
            ops_buffer: Vec::with_capacity(16),
            ctx,
        }
    }

    fn set_pos(&mut self, buf: &'a [u8], pos: usize, address: VirtualAddress64) {
        assert!(pos < buf.len());
        self.buf = buf;
        self.pos = pos;
        self.virtual_address = address;
    }

    fn next<'s>(&'s mut self) -> Instruction<'s, 'a, VirtualAddress64> {
        let length = instruction_length_64(&self.buf[self.pos..]);
        let address = self.virtual_address + self.pos as u32;
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
            let data = &self.buf[self.pos..self.pos + length];
            instruction_operations64(
                address,
                data,
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
        self.virtual_address + self.pos as u32
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
    register8_low: [Option<Operand<'e>>; 16],
    register8_high: [Option<Operand<'e>>; 4],
    register16: [Option<Operand<'e>>; 16],
    register32: [Option<Operand<'e>>; 16],
    ctx: OperandCtx<'e>,
}

impl<'e> RegisterCache<'e> {
    fn new(ctx: OperandCtx<'e>, is_64: bool) -> RegisterCache<'e> {
        if is_64 {
            ctx.resize_offset_cache(16);
        } else {
            ctx.resize_offset_cache(8);
        }
        RegisterCache {
            ctx,
            register8_low: [None; 16],
            register8_high: [None; 4],
            register16: [None; 16],
            register32: [None; 16],
        }
    }

    fn register8_low(&mut self, i: u8) -> Operand<'e> {
        let ctx = self.ctx;
        *self.register8_low[i as usize & 15].get_or_insert_with(|| {
            ctx.and_const(ctx.register(i as u8), 0xff)
        })
    }

    fn register8_high(&mut self, i: u8) -> Operand<'e> {
        let ctx = self.ctx;
        *self.register8_high[i as usize & 15].get_or_insert_with(|| {
            ctx.rsh_const(
                ctx.and_const(ctx.register(i as u8), 0xff00),
                8,
            )
        })
    }

    fn register16(&mut self, i: u8) -> Operand<'e> {
        let ctx = self.ctx;
        *self.register16[i as usize & 15].get_or_insert_with(|| {
            ctx.and_const(ctx.register(i as u8), 0xffff)
        })
    }

    fn register32(&mut self, i: u8) -> Operand<'e> {
        let ctx = self.ctx;
        *self.register32[i as usize & 15].get_or_insert_with(|| {
            ctx.and_const(ctx.register(i as u8), 0xffff_ffff)
        })
    }

    fn register_offset_const(&mut self, register: u8, offset: i32) -> Operand<'e> {
        self.ctx.register_offset_const(register, offset)
    }
}

struct InstructionOpsState<'a, 'e: 'a, Va: VirtualAddress> {
    address: Va,
    data: &'a [u8],
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

    let full_data = data;
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
    let instruction_len = data.len();
    let data = &data[prefix_count..];
    let is_ext = data[0] == 0xf;
    let data = match is_ext {
        true => &data[1..],
        false => data,
    };
    let mut s = InstructionOpsState {
        address,
        data,
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

fn instruction_operations32_main(
    s: &mut InstructionOpsState<VirtualAddress32>,
) -> Result<(), Failed> {
    use self::operation_helpers::*;

    let ctx = s.ctx;
    // Rustc falls over at patterns containing ranges, manually type out all
    // cases for a first byte to make sure this compiles to a single switch.
    // (Or very least leave it to LLVM to decide)
    // Also represent extended commands as 0x100 ..= 0x1ff to make it even "nicer" switch.
    let first_byte = s.data[0] as u32 | ((s.is_ext as u32) << 8);
    match first_byte {
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
            let arith = match first_byte / 8 {
                0 => ArithOperation::Add,
                1 => ArithOperation::Or,
                2 => ArithOperation::Adc,
                3 => ArithOperation::Sbb,
                4 => ArithOperation::And,
                5 => ArithOperation::Sub,
                6 => ArithOperation::Xor,
                7 => ArithOperation::Cmp,
                _ => ArithOperation::Move,
            };
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
        0x98 => {
            let extended_sign = ctx.and_const(
                ctx.sub_const(
                    ctx.eq_const(
                        ctx.and_const(
                            ctx.register(0),
                            0x8000,
                        ),
                        0,
                    ),
                    1,
                ),
                0xffff_0000,
            );
            let merged_eax = ctx.or(
                extended_sign,
                s.register_cache.register16(0),
            );
            s.output(mov_to_reg(0, merged_eax));
            Ok(())
        }
        // Cdq
        0x99 => {
            let extended_sign = ctx.sub_const(
                ctx.eq_const(
                    ctx.and_const(
                        ctx.register(0),
                        0x8000_0000,
                    ),
                    0,
                ),
                1,
            );
            s.output(mov_to_reg(2, extended_sign));
            Ok(())
        },
        0x9f => s.lahf(),
        0xa0 | 0xa1 | 0xa2 | 0xa3 => s.move_mem_eax(),
        // rep mov, rep stos
        0xa4 | 0xa5 | 0xaa | 0xab => {
            s.output(Operation::Special(s.full_data.into()));
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
        0x12c | 0x12d => s.cvttss2si(),
        // ucomiss, comiss, comiss signals exceptions but that isn't simulated
        0x12e | 0x12f => s.sse_compare(),
        // rdtsc
        0x131 => {
            s.output(mov_to_reg(0, s.ctx.new_undef()));
            s.output(mov_to_reg(2, s.ctx.new_undef()));
            Ok(())
        }
        0x140 | 0x141 | 0x142 | 0x143 | 0x144 | 0x145 | 0x146 | 0x147 |
            0x148 | 0x149 | 0x14a | 0x14b | 0x14c | 0x14d | 0x14e | 0x14f => s.cmov(),
        0x157 => s.xorps(),
        0x158 => s.sse_float_arith(ArithOpType::Add),
        0x159 => s.sse_float_arith(ArithOpType::Mul),
        0x15a => s.sse_f32_f64_conversion(),
        0x15b => s.cvtdq2ps(),
        0x15c => s.sse_float_arith(ArithOpType::Sub),
        0x15e => s.sse_float_arith(ArithOpType::Div),
        0x15f => s.sse_float_max(),
        0x160 => s.punpcklbw(),
        0x16e => s.mov_sse_6e(),
        0x173 => s.packed_shift_imm(),
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
        0x1b1 => {
            // Cmpxchg
            let (rm, _r) = s.parse_modrm(s.mem16_32())?;
            let dest = s.rm_to_dest_operand(&rm);
            s.output(mov(dest, ctx.new_undef()));
            s.output(mov_to_reg(0, ctx.new_undef()));
            Ok(())
        }
        0x1b3 => s.bit_test(BitTest::Reset, false),
        0x1b6 | 0x1b7 => s.movzx(),
        0x1ba => s.various_0f_ba(),
        0x1bb => s.bit_test(BitTest::Complement, false),
        0x1bc | 0x1bd => {
            // bsf, bsr, just set dest as undef.
            // Could maybe emit Special?
            let (_rm, r) = s.parse_modrm(s.mem16_32())?;
            s.output(mov(r.dest_operand(), ctx.new_undef()));
            Ok(())
        }
        0x1be => s.movsx(MemAccessSize::Mem8),
        0x1bf => s.movsx(MemAccessSize::Mem16),
        0x1c0 | 0x1c1 => s.xadd(),
        0x1c8 | 0x1c9 | 0x1ca | 0x1cb | 0x1cc | 0x1cd | 0x1ce | 0x1cf => s.bswap(),
        0x1d3 => s.packed_shift_right(),
        0x1d5 => s.pmullw(),
        0x1d6 => s.mov_sse_d6(),
        0x1e6 => s.sse_int_double_conversion(),
        0x1ef => {
            if s.has_prefix(0x66) {
                // pxor
                s.xorps()
            } else {
                // MMX xor
                Err(s.unknown_opcode())
            }
        }
        0x1f3 => s.packed_shift_left(),
        _ => Err(s.unknown_opcode()),
    }
}

fn instruction_operations64<'e>(
    address: VirtualAddress64,
    data: &[u8],
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

    let full_data = data;
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
    let instruction_len = data.len();
    let data = &data[prefix_count..];
    let is_ext = data[0] == 0xf;
    let data = match is_ext {
        true => &data[1..],
        false => data,
    };
    let mut s = InstructionOpsState {
        address,
        data,
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
    use self::operation_helpers::*;

    let ctx = s.ctx;
    // Rustc falls over at patterns containing ranges, manually type out all
    // cases for a first byte to make sure this compiles to a single switch.
    // (Or very least leave it to LLVM to decide)
    // Also represent extended commands as 0x100 ..= 0x1ff to make it even "nicer" switch.
    let first_byte = s.data[0] as u32 | ((s.is_ext as u32) << 8);
    match first_byte {
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
            let arith = match first_byte / 8 {
                0 => ArithOperation::Add,
                1 => ArithOperation::Or,
                2 => ArithOperation::Adc,
                3 => ArithOperation::Sbb,
                4 => ArithOperation::And,
                5 => ArithOperation::Sub,
                6 => ArithOperation::Xor,
                7 => ArithOperation::Cmp,
                _ => ArithOperation::Move,
            };
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
        0x90 => Ok(()),
        0x98 => {
            let sign_bit;
            let high_half_mask;
            if s.rex_prefix() & 0x8 == 0 {
                sign_bit = 0x8000;
                high_half_mask = 0xffff_0000;
            } else {
                sign_bit = 0x8000_0000;
                high_half_mask = 0xffff_ffff_0000_0000;
            }
            let extended_sign = ctx.and_const(
                ctx.sub_const(
                    ctx.eq_const(
                        ctx.and_const(
                            ctx.register(0),
                            sign_bit,
                        ),
                        0,
                    ),
                    1,
                ),
                high_half_mask,
            );
            let new_eax = ctx.or(
                extended_sign,
                s.register_cache.register16(0),
            );
            s.output(mov_to_reg(0, new_eax));
            Ok(())
        }
        // Cdq
        0x99 => {
            let sign_bit;
            let mask;
            if s.rex_prefix() & 0x8 == 0 {
                sign_bit = 0x8000_0000;
                mask = 0xffff_ffff;
            } else {
                sign_bit = 0x8000_0000_0000_0000;
                mask = 0xffff_ffff_ffff_ffff;
            }
            let extended_sign = ctx.and_const(
                ctx.sub_const(
                    ctx.eq_const(
                        ctx.and_const(
                            ctx.register(0),
                            sign_bit,
                        ),
                        0,
                    ),
                    1,
                ),
                mask,
            );
            s.output(mov_to_reg(2, extended_sign));
            Ok(())
        },
        0x9f => s.lahf(),
        0xa0 | 0xa1 | 0xa2 | 0xa3 => s.move_mem_eax(),
        // rep mov, rep stos
        0xa4 | 0xa5 | 0xaa | 0xab => {
            s.output(Operation::Special(s.full_data.into()));
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
        0x12a => s.sse_int_to_float(),
        0x12c | 0x12d => s.cvttss2si(),
        // ucomiss, comiss, comiss signals exceptions but that isn't simulated
        0x12e | 0x12f => s.sse_compare(),
        // rdtsc
        0x131 => {
            s.output(mov_to_reg(0, s.ctx.new_undef()));
            s.output(mov_to_reg(2, s.ctx.new_undef()));
            Ok(())
        }
        0x140 | 0x141 | 0x142 | 0x143 | 0x144 | 0x145 | 0x146 | 0x147 |
            0x148 | 0x149 | 0x14a | 0x14b | 0x14c | 0x14d | 0x14e | 0x14f => s.cmov(),
        0x157 => s.xorps(),
        0x158 => s.sse_float_arith(ArithOpType::Add),
        0x159 => s.sse_float_arith(ArithOpType::Mul),
        0x15a => s.sse_f32_f64_conversion(),
        0x15b => s.cvtdq2ps(),
        0x15c => s.sse_float_arith(ArithOpType::Sub),
        0x15e => s.sse_float_arith(ArithOpType::Div),
        0x15f => s.sse_float_max(),
        0x160 => s.punpcklbw(),
        0x16e => s.mov_sse_6e(),
        0x173 => s.packed_shift_imm(),
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
        0x1b1 => {
            // Cmpxchg
            let (rm, _r) = s.parse_modrm(s.mem16_32())?;
            let dest = s.rm_to_dest_operand(&rm);
            s.output(mov(dest, ctx.new_undef()));
            s.output(mov_to_reg(0, ctx.new_undef()));
            Ok(())
        }
        0x1b3 => s.bit_test(BitTest::Reset, false),
        0x1b6 | 0x1b7 => s.movzx(),
        0x1ba => s.various_0f_ba(),
        0x1bb => s.bit_test(BitTest::Complement, false),
        0x1bc | 0x1bd => {
            // bsf, bsr, just set dest as undef.
            // Could maybe emit Special?
            let (_rm, r) = s.parse_modrm(s.mem16_32())?;
            s.output(mov(r.dest_operand(), ctx.new_undef()));
            Ok(())
        }
        0x1be => s.movsx(MemAccessSize::Mem8),
        0x1bf => s.movsx(MemAccessSize::Mem16),
        0x1c0 | 0x1c1 => s.xadd(),
        0x1c8 | 0x1c9 | 0x1ca | 0x1cb | 0x1cc | 0x1cd | 0x1ce | 0x1cf => s.bswap(),
        0x1d3 => s.packed_shift_right(),
        0x1d5 => s.pmullw(),
        0x1d6 => s.mov_sse_d6(),
        0x1e6 => s.sse_int_double_conversion(),
        0x1ef => {
            if s.has_prefix(0x66) {
                // pxor
                s.xorps()
            } else {
                // MMX xor
                Err(s.unknown_opcode())
            }
        }
        0x1f3 => s.packed_shift_left(),
        _ => Err(s.unknown_opcode()),
    }
}

enum BitTest {
    Set,
    Reset,
    NoChange,
    Complement,
}

fn x87_variant<'e>(ctx: OperandCtx<'e>, op: Operand<'e>, offset: i8) -> Operand<'e> {
    match *op.ty() {
        OperandType::Register(Register(r)) => ctx.register_fpu((r as i8 + offset) as u8 & 7),
        _ => op,
    }
}

impl<'a, 'e: 'a, Va: VirtualAddress> InstructionOpsState<'a, 'e, Va> {
    pub fn len(&self) -> usize {
        self.len as usize
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

    fn output_flag_set(&mut self, flag: Flag, value: Operand<'e>) {
        self.output(Operation::Move(DestOperand::Flag(flag), value, None))
    }

    fn output_arith(
        &mut self,
        dest: DestOperand<'e>,
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
    ) {
        let op = self.ctx.arithmetic(ty, left, right);
        self.output(Operation::Move(dest, op, None));
    }

    fn output_add(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.add(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_add_const(&mut self, dest: DestAndOperand<'e>, rhs: u64) {
        let op = self.ctx.add_const(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_sub(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.sub(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_sub_const(&mut self, dest: DestAndOperand<'e>, rhs: u64) {
        let op = self.ctx.sub_const(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_mul(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.mul(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_signed_mul(
        &mut self,
        dest: DestAndOperand<'e>,
        rhs: Operand<'e>,
        size: MemAccessSize,
    ) {
        let op = self.ctx.signed_mul(dest.op, rhs, size);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_xor(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.xor(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_lsh(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.lsh(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_rsh(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.rsh(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_or(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.or(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_and(&mut self, dest: DestAndOperand<'e>, rhs: Operand<'e>) {
        let op = self.ctx.and(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
    }

    fn output_and_const(&mut self, dest: DestAndOperand<'e>, rhs: u64) {
        let op = self.ctx.and_const(dest.op, rhs);
        self.output(Operation::Move(dest.dest, op, None))
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

    fn reg_variable_size(&mut self, register: Register, op_size: MemAccessSize) -> Operand<'e> {
        if register.0 >= 4 && self.rex_prefix() == 0 && op_size == MemAccessSize::Mem8 {
            self.register_cache.register8_high(register.0 - 4)
        } else {
            match op_size {
                MemAccessSize::Mem8 => self.register_cache.register8_low(register.0),
                MemAccessSize::Mem16 => self.register_cache.register16(register.0),
                MemAccessSize::Mem32 => self.register_cache.register32(register.0),
                MemAccessSize::Mem64 => self.ctx.register_ref(register.0),
            }
        }
    }

    fn r_to_operand(&mut self, r: ModRm_R) -> Operand<'e> {
        match r.1 {
            RegisterSize::Low8 => self.register_cache.register8_low(r.0),
            RegisterSize::High8 => self.register_cache.register8_high(r.0),
            RegisterSize::R16 => self.register_cache.register16(r.0),
            RegisterSize::R32 => self.register_cache.register32(r.0),
            RegisterSize::R64 => self.ctx.register_ref(r.0),
        }
    }

    fn r_to_operand_xmm(&self, r: ModRm_R, i: u8) -> Operand<'e> {
        self.ctx.xmm(r.0, i)
    }

    /// Returns a structure containing both DestOperand and Operand
    /// variations, not useful with ModRm_R, but ModRm_Rm avoids
    /// recalculating address twice with this.
    fn r_to_dest_and_operand(&mut self, r: ModRm_R) -> DestAndOperand<'e> {
        let op;
        let dest;
        match r.1 {
            RegisterSize::R64 => {
                op = self.ctx.register(r.0);
                dest = DestOperand::Register64(Register(r.0));
            }
            RegisterSize::R32 => {
                op = self.register_cache.register32(r.0);
                dest = DestOperand::Register32(Register(r.0));
            }
            RegisterSize::R16 => {
                op = self.register_cache.register16(r.0);
                dest = DestOperand::Register16(Register(r.0));
            }
            RegisterSize::Low8 => {
                op = self.register_cache.register8_low(r.0);
                dest = DestOperand::Register8Low(Register(r.0));
            }
            RegisterSize::High8 => {
                op = self.register_cache.register8_high(r.0);
                dest = DestOperand::Register8High(Register(r.0));
            }
        }
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
    fn rm_address_operand(&mut self, rm: &ModRm_Rm) -> Operand<'e> {
        let ctx = self.ctx;
        // Optimization: avoid having to go through simplify for x + x * 4 -type accesses
        let base_index_same = rm.base == rm.index && rm.index_mul != 0;
        let base_offset = if rm.constant_base() {
            if Va::SIZE == 4 || !rm.rip_relative() {
                self.ctx.constant(rm.constant as u64)
            } else {
                let addr = self.address.as_u64()
                    .wrapping_add(self.len() as u64)
                    .wrapping_add(rm.constant as i32 as i64 as u64);
                self.ctx.constant(addr)
            }
        } else {
            if base_index_same {
                let base = self.ctx.register(rm.base);
                let base = ctx.mul_const(base, rm.index_mul as u64 + 1);
                if rm.constant == 0 {
                    base
                } else {
                    ctx.add_const(base, rm.constant as i32 as i64 as u64)
                }
            } else {
                if rm.constant == 0 {
                    self.ctx.register(rm.base)
                } else {
                    self.register_cache.register_offset_const(rm.base, rm.constant as i32)
                }
            }
        };
        let with_index = if base_index_same {
            base_offset
        } else {
            match rm.index_mul {
                0 => base_offset,
                1 => ctx.add(base_offset, self.ctx.register(rm.index)),
                x => {
                    ctx.add(
                        base_offset,
                        ctx.mul_const(self.ctx.register(rm.index), x as u64),
                    )
                }
            }
        };
        with_index
    }

    fn rm_to_dest_and_operand(&mut self, rm: &ModRm_Rm) -> DestAndOperand<'e> {
        if rm.is_memory() {
            let address = self.rm_address_operand(rm);
            let size = rm.size.to_mem_access_size();
            DestAndOperand {
                op: self.ctx.mem_variable_rc(size, address),
                dest: DestOperand::Memory(MemAccess {
                    size,
                    address: address,
                })
            }
        } else {
            self.r_to_dest_and_operand(ModRm_R(rm.base, rm.size))
        }
    }

    fn rm_to_dest_operand(&mut self, rm: &ModRm_Rm) -> DestOperand<'e> {
        if rm.is_memory() {
            let address = self.rm_address_operand(&rm);
            DestOperand::Memory(MemAccess {
                size: rm.size.to_mem_access_size(),
                address,
            })
        } else {
            ModRm_R(rm.base, rm.size).dest_operand()
        }
    }

    fn rm_to_dest_operand_xmm(&mut self, rm: &ModRm_Rm, i: u8) -> DestOperand<'e> {
        if rm.is_memory() {
            // Would be nice to just add the i * 4 offset on rm_address_operand,
            // but `rm.constant += i * 4` has issues if the constant overflows
            let mut address = self.rm_address_operand(&rm);
            if i != 0 {
                address = self.ctx.add_const(address, i as u64 * 4);
            }
            DestOperand::Memory(MemAccess {
                size: MemAccessSize::Mem32,
                address,
            })
        } else {
            DestOperand::Xmm(rm.base, i)
        }
    }

    fn rm_to_operand_xmm(&mut self, rm: &ModRm_Rm, i: u8) -> Operand<'e> {
        if rm.is_memory() {
            // Would be nice to just add the i * 4 offset on rm_address_operand,
            // but `rm.constant += i * 4` has issues if the constant overflows
            let mut address = self.rm_address_operand(&rm);
            if i != 0 {
                address = self.ctx.add_const(address, i as u64 * 4);
            }
            self.ctx.mem_variable_rc(MemAccessSize::Mem32, address)
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
            let address = self.rm_address_operand(rm);
            self.ctx.mem_variable_rc(rm.size.to_mem_access_size(), address)
        } else {
            self.r_to_operand(ModRm_R(rm.base, rm.size)).clone()
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

    fn read_u32(&mut self, offset: usize) -> Result<u32, Failed> {
        use crate::light_byteorder::ReadLittleEndian;
        match self.data.get(offset..).and_then(|mut x| x.read_u32().ok()) {
            Some(s) => Ok(s),
            None => Err(self.internal_decode_error()),
        }
    }

    fn read_u16(&mut self, offset: usize) -> Result<u16, Failed> {
        use crate::light_byteorder::ReadLittleEndian;
        match self.data.get(offset..).and_then(|mut x| x.read_u16().ok()) {
            Some(s) => Ok(s),
            None => Err(self.internal_decode_error()),
        }
    }

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
    ) -> Result<(ModRm_Rm, ModRm_R, usize), Failed> {
        let modrm = self.read_u8(1)?;
        let rm_val = modrm & 0x7;
        let register = if self.rex_prefix() & 0x4 == 0 {
            (modrm >> 3) & 0x7
        } else {
            8 + ((modrm >> 3) & 0x7)
        };
        let rm_ext = self.rex_prefix() & 0x1 != 0;
        let r_high = register >= 4 &&
            self.rex_prefix() == 0 &&
            op_size == MemAccessSize::Mem8;
        let register_size = if r_high {
            RegisterSize::High8
        } else {
            RegisterSize::from_mem_access_size(op_size)
        };
        let r = if r_high {
            ModRm_R(register - 4, register_size)
        } else {
            ModRm_R(register, register_size)
        };

        let rm_variant = (modrm >> 6) & 0x3;
        let (rm, size) = if rm_variant < 3 && rm_val == 4 {
            self.parse_sib(rm_variant, op_size)?
        } else {
            let mut rm = ModRm_Rm {
                size: RegisterSize::from_mem_access_size(op_size),
                base: match rm_ext {
                    false => rm_val,
                    true => rm_val + 8,
                },
                index: 0,
                index_mul: 0,
                constant: 0,
            };
            match rm_variant {
                0 => match rm_val {
                    5 => {
                        // 32-bit has the immediate as mem[imm],
                        // 64-bit has mem[rip + imm]
                        let imm = self.read_u32(2)?;
                        rm.base = u8::max_value();
                        rm.constant = imm;
                        (rm, 6)
                    }
                    _ => {
                        (rm, 2)
                    }
                },
                1 => {
                    let offset = self.read_u8(2)? as i8 as u32;
                    rm.constant = offset;
                    (rm, 3)
                }
                2 => {
                    let offset = self.read_u32(2)?;
                    rm.constant = offset;
                    (rm, 6)
                }
                3 => {
                    let rm_high = rm_val >= 4 &&
                        self.rex_prefix() == 0 &&
                        op_size == MemAccessSize::Mem8;

                    if rm_high {
                        rm.size = RegisterSize::High8;
                        rm.base = rm_val - 4;
                    }
                    rm.index = u8::max_value();
                    (rm, 2)
                }
                _ => unreachable!(),
            }
        };
        Ok((rm, r, size))
    }

    fn parse_modrm(&mut self, op_size: MemAccessSize) -> Result<(ModRm_Rm, ModRm_R), Failed> {
        let (rm, r, _) = self.parse_modrm_inner(op_size)?;
        Ok((rm, r))
    }

    fn parse_modrm_imm(
        &mut self,
        op_size: MemAccessSize,
        imm_size: MemAccessSize,
    ) -> Result<(ModRm_Rm, ModRm_R, Operand<'e>), Failed> {
        let (rm, r, offset) = self.parse_modrm_inner(op_size)?;
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
                imm | (!0u32 >> bits_minus_one << bits_minus_one) as u64 | 0xffff_ffff_0000_0000
            } else {
                imm
            }
        };
        Ok((rm, r, self.ctx.constant(imm)))
    }

    fn parse_sib(
        &mut self,
        variation: u8,
        op_size: MemAccessSize,
    ) -> Result<(ModRm_Rm, usize), Failed> {
        let sib = self.read_u8(2)?;
        let mul = 1 << ((sib >> 6) & 0x3);
        let base_ext = self.rex_prefix() & 0x1 != 0;
        let mut result = ModRm_Rm {
            size: RegisterSize::from_mem_access_size(op_size),
            base: 0,
            index: 0,
            index_mul: 0,
            constant: 0,
        };
        let size = match (sib & 0x7, variation) {
            (5, 0) => {
                // Constant base
                let constant = self.read_u32(3)?;
                result.base = 0xfe;
                result.constant = constant;
                7
            }
            (reg, _) => {
                result.base = match base_ext {
                    false => reg,
                    true => reg + 8,
                };
                3
            }
        };
        let index_reg = if self.rex_prefix() & 0x2 == 0 {
            (sib >> 3) & 0x7
        } else {
            8 + ((sib >> 3) & 0x7)
        };
        // Index reg 4 = None
        if index_reg != 4 {
            result.index = index_reg;
            result.index_mul = mul;
        }
        match variation {
            0 => Ok((result, size)),
            1 => {
                result.constant = self.read_u8(size)? as i8 as u32;
                Ok((result, size + 1))
            }
            2 | _ => {
                result.constant = self.read_u32(size)?;
                Ok((result, size + 4))
            }
        }
    }

    fn inc_dec_op(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(0)?;
        let is_inc = byte < 0x48;
        let reg_id = byte & 0x7;
        let op_size = self.mem16_32();
        let reg = self.reg_variable_size(Register(reg_id), op_size);
        let dest = DestAndOperand {
            op: reg.clone(),
            dest: DestOperand::reg_variable_size(reg_id, op_size),
        };
        match is_inc {
            true => self.output_add_const(dest, 1),
            false => self.output_sub_const(dest, 1),
        }
        self.inc_dec_flags(is_inc, reg);
        Ok(())
    }

    fn inc_dec_flags(&mut self, is_inc: bool, reg: Operand<'e>) {
        let is_64 = Va::SIZE == 8;
        let ctx = self.ctx;
        if is_64 {
            self.output_flag_set(Flag::Zero, ctx.eq_const(reg, 0));
            self.output_flag_set(Flag::Sign, ctx.gt_const(reg, 0x7fff_ffff_ffff_ffff));
        } else {
            self.output_flag_set(Flag::Zero, ctx.eq_const(ctx.and_const(reg, 0xffff_ffff), 0));
            self.output_flag_set(Flag::Sign, ctx.gt_const(reg, 0x7fff_ffff));
        }
        let eq_value = match (is_inc, is_64) {
            (true, false) => 0x8000_0000,
            (false, false) => 0x7fff_ffff,
            (true, true) => 0x8000_0000_0000_0000,
            (false, true) => 0x7fff_ffff_ffff_ffff,
        };
        self.output_flag_set(Flag::Overflow, ctx.eq_const(reg, eq_value));
    }

    fn flag_set(&mut self, flag: Flag, value: bool) -> Result<(), Failed> {
        self.output_flag_set(flag, self.ctx.constant(value as u64));
        Ok(())
    }

    fn condition(&mut self) -> Result<Operand<'e>, Failed> {
        let ctx = self.ctx;
        let cond_id = self.read_u8(0)? & 0xf;
        let zero = ctx.const_0();
        let cond = match cond_id >> 1 {
            // jo, jno
            0x0 => ctx.eq(ctx.flag_o(), zero),
            // jb, jnb (jae) (jump if carry)
            0x1 => ctx.eq(ctx.flag_c(), zero),
            // je, jne
            0x2 => ctx.eq(ctx.flag_z(), zero),
            // jbe, jnbe (ja)
            0x3 => ctx.and(
                ctx.eq(ctx.flag_z(), zero),
                ctx.eq(ctx.flag_c(), zero),
            ),
            // js, jns
            0x4 => ctx.eq(ctx.flag_s(), zero),
            // jpe, jpo
            0x5 => ctx.eq(ctx.flag_p(), zero),
            // jl, jnl (jge)
            0x6 => ctx.eq(ctx.flag_s(), ctx.flag_o()),
            // jle, jnle (jg)
            0x7 => ctx.and(
                ctx.eq(ctx.flag_z(), zero),
                ctx.eq(ctx.flag_s(), ctx.flag_o()),
            ),
            _ => unreachable!(),
        };
        if cond_id & 1 == 0 {
            Ok(ctx.eq(cond, zero))
        } else {
            Ok(cond)
        }
    }

    fn cmov(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let condition = Some(self.condition()?);
        let rm = self.rm_to_operand(&rm);
        self.output(Operation::Move(r.dest_operand(), rm, condition));
        Ok(())
    }

    fn conditional_set(&mut self) -> Result<(), Failed> {
        let condition = self.condition()?;
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem8)?;
        let dest = self.rm_to_dest_operand(&rm);
        self.output(Operation::Move(dest, condition, None));
        Ok(())
    }

    fn xchg(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r) = self.parse_modrm(op_size)?;
        if !r.equal_to_rm(&rm) {
            let r = self.r_to_dest_and_operand(r);
            let rm = self.rm_to_dest_and_operand(&rm);
            self.output(Operation::MoveSet(vec![
                (r.dest, rm.op),
                (rm.dest, r.op),
            ]));
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
        let reg_op = ctx.register_ref(register);
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

    fn lahf(&mut self) -> Result<(), Failed> {
        // TODO implement
        self.output(Operation::Move(
            DestOperand::Register8High(Register(0)), self.ctx.new_undef(), None
        ));
        Ok(())
    }

    fn move_mem_eax(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let const_size = match Va::SIZE == 4 {
            true => MemAccessSize::Mem32,
            false => MemAccessSize::Mem64,
        };
        let constant = self.read_variable_size_64(1, const_size)?;
        let constant = self.ctx.constant(constant);
        let eax_left = self.read_u8(0)? & 0x2 == 0;
        self.output(match eax_left {
            true => mov_to_reg(0, self.ctx.mem_variable_rc(op_size, constant)),
            false => mov_to_mem(op_size, constant, self.ctx.register(0)),
        });
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
        use self::operation_helpers::*;
        // Operation size must be correct, but the lhs operand size may be
        // extended to 64 after size has been taken.
        let size = lhs.size.to_mem_access_size();
        self.skip_unnecessary_32bit_operand_masking(lhs, arith);
        let dest = self.rm_to_dest_and_operand(lhs);
        let ctx = self.ctx;
        let normal_flags = match arith {
            ArithOperation::Add => Some(ArithOpType::Add),
            ArithOperation::Sub | ArithOperation::Cmp => Some(ArithOpType::Sub),
            ArithOperation::And | ArithOperation::Test => Some(ArithOpType::And),
            ArithOperation::Or => Some(ArithOpType::Or),
            ArithOperation::Xor => Some(ArithOpType::Xor),
            ArithOperation::LeftShift => Some(ArithOpType::Lsh),
            ArithOperation::RightShift => Some(ArithOpType::Rsh),
            _ => None,
        };
        if let Some(ty) = normal_flags {
            self.output(flags(ty, dest.op, rhs, size));
        }
        match arith {
            ArithOperation::Add => {
                self.output_add(dest, rhs);
            }
            ArithOperation::Sub | ArithOperation::Cmp => {
                if arith != ArithOperation::Cmp {
                    self.output_sub(dest, rhs);
                }
            }
            ArithOperation::Sbb | ArithOperation::Adc => {
                // Since sbb wants to do
                //
                // carry = (l - r - carry > l) | (l - r > l)
                // l = l - r - carry
                //
                // which is bad since carry depends on original l
                // and l depends on original carry,
                // order it as
                //
                // l = l - carry
                // carry = (l - r > l + carry) | (l - r + carry > l + carry)
                // overflow = signed_gt(l - r, l + carry) | signed_gt(..)
                // l = l - r
                // (set zero, sign, parity)
                //
                // That works when l != r
                // if l == r just do l = 0 - c
                //
                // With adc, do
                //
                // l = l + carry
                // carry = (l - carry > l - carry + r) | (l - carry > l + r)
                // l = l + r
                //
                // if l == r:
                // l = l + carry
                // carry = (l - carry > l - carry + r - 1) | (l - carry > l + r - 1)
                // l = l + r - 1
                let carry = ctx.flag_c();
                let zero = ctx.const_0();
                let dest_op = dest.op;
                let lhs_eq_rhs = dest_op == rhs;
                let is_sbb = arith == ArithOperation::Sbb;
                let a;
                let b;
                let c;
                let gt_lhs1;
                let gt_lhs2;
                let gt_rhs1;
                let gt_rhs2;
                let mut rhs = rhs;
                if is_sbb {
                    if lhs_eq_rhs {
                        self.output(mov(dest.dest, ctx.sub(zero, carry)));
                        self.output_flag_set(Flag::Overflow, zero);
                        self.output_flag_set(Flag::Parity, ctx.const_1());
                        self.output_flag_set(Flag::Zero, ctx.eq(carry, zero));
                        self.output_flag_set(Flag::Sign, carry);
                        return;
                    }
                    self.output_sub(dest.clone(), carry);
                    // dest is now dest_orig - carry
                    // carry = ((dest - rhs) > (dest + carry)) ||
                    //         ((dest - rhs + carry) > (dest + carry))
                    // (Overflow is same but signed)
                    a = ctx.sub(dest_op, rhs);
                    b = ctx.add(dest_op, carry);
                    c = ctx.add(a, carry);
                    gt_lhs1 = a;
                    gt_lhs2 = c;
                    gt_rhs1 = b;
                    gt_rhs2 = b;
                } else {
                    // Adc
                    if lhs_eq_rhs {
                        rhs = ctx.sub_const(rhs, 1)
                    }
                    self.output_add(dest.clone(), carry);
                    // dest is now dest_orig + carry
                    // carry = ((dest - carry) > (dest + rhs)) ||
                    //         ((dest - carry) > (dest + rhs - carry))
                    // (Overflow is same but signed)
                    a = ctx.sub(dest_op, carry);
                    b = ctx.add(dest_op, rhs);
                    c = ctx.sub(b, carry);
                    gt_lhs1 = a;
                    gt_lhs2 = a;
                    gt_rhs1 = b;
                    gt_rhs2 = c;
                }

                let gt = ctx.or(
                    ctx.gt(gt_lhs1, gt_rhs1),
                    ctx.gt(gt_lhs2, gt_rhs2),
                );
                let signed_gt = ctx.or(
                    ctx.gt_signed(gt_lhs1, gt_rhs1, size),
                    ctx.gt_signed(gt_lhs2, gt_rhs2, size),
                );
                self.output_flag_set(Flag::Carry, gt);
                self.output_flag_set(Flag::Overflow, signed_gt);
                if is_sbb {
                    self.output_sub(dest, rhs);
                } else {
                    self.output_add(dest, rhs);
                }

                let dest_zero = ctx.eq(dest_op, zero);
                let parity = ctx.arithmetic(ArithOpType::Parity, dest_op, zero);
                let sign_bit = 1 << (size.bits() - 1);
                let sign = ctx.neq_const(
                    ctx.and_const(dest_op, sign_bit),
                    0,
                );
                self.output_flag_set(Flag::Zero, dest_zero);
                self.output_flag_set(Flag::Parity, parity);
                self.output_flag_set(Flag::Sign, sign);
            }
            ArithOperation::And | ArithOperation::Test => {
                if arith != ArithOperation::Test {
                    self.output_and(dest, rhs);
                }
            }
            ArithOperation::Or => {
                self.output_or(dest, rhs);
            }
            ArithOperation::Xor => {
                if dest.op == rhs {
                    // Zeroing xor is not that common, usually only done few times
                    // per function at most, but skip its simplification anyway.
                    self.output(mov(dest.dest, self.ctx.const_0()));
                } else {
                    self.output_xor(dest, rhs);
                }
            }
            ArithOperation::Move => {
                if dest.op != rhs {
                    self.output(mov(dest.dest, rhs));
                }
            }
            ArithOperation::LeftShift => {
                self.output_lsh(dest, rhs);
            }
            ArithOperation::RightShift => {
                self.output_rsh(dest, rhs);
            }
            ArithOperation::RightShiftArithmetic => {
                // TODO flags, using ToFloat to cheat
                self.output(flags(ArithOpType::ToFloat, dest.op, rhs, size));
                // Arithmetic shift shifts in the value of sign bit,
                // that can be represented as bitwise or of
                // `not(ffff...ffff << rhs >> rhs) & ((sign_bit == 0) - 1)`
                // with logical right shift
                // (sign_bit == 0) - 1 is 0 if sign_bit is clear, ffff...ffff if sign_bit is set
                let sign_bit = 1u64 << (size.bits() - 1);
                let logical_rsh = ctx.rsh(dest.op, rhs);
                let mask = (sign_bit << 1).wrapping_sub(1);
                let negative_shift_in_bits = ctx.xor_const(
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
                );
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
                self.output(Operation::Move(
                    dest.dest,
                    result,
                    None,
                ));
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
                // TODO set overflow if 1bit??
                if size == MemAccessSize::Mem64 {
                    self.output(Operation::Move(dest.dest, full, None));
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
        let (mut rm, r) = self.parse_modrm(op_size)?;
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
        use self::operation_helpers::*;

        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        if rm.is_memory() {
            let addr = self.rm_address_operand(&rm);
            self.output(mov(r.dest_operand(), addr));
        }
        Ok(())
    }

    fn movsx(&mut self, op_size: MemAccessSize) -> Result<(), Failed> {
        let dest_size = self.mem16_32();
        let (mut rm, r) = self.parse_modrm(dest_size)?;
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
        self.output(Operation::Move(
            r.dest_operand(),
            self.ctx.sign_extend(rm, op_size, dest_size),
            None,
        ));
        Ok(())
    }

    fn movzx(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;

        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (mut rm, r) = self.parse_modrm(self.mem16_32())?;
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
            self.output(mov(r.dest_operand(), rm_oper));
        }
        Ok(())
    }

    fn various_f7(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        let (rm, _) = self.parse_modrm(op_size)?;
        let rm = self.rm_to_dest_and_operand(&rm);
        let ctx = self.ctx;
        match variant {
            0 | 1 => return self.generic_arith_with_imm_op(ArithOperation::Test, op_size),
            2 => {
                // Not
                let constant = self.ctx.constant(!0u64);
                self.output_xor(rm, constant);
            }
            3 => {
                // Neg
                self.output_arith(
                    rm.dest,
                    ArithOpType::Sub,
                    ctx.const_0(),
                    rm.op,
                );
            }
            4 | 5 => {
                // TODO signed mul
                // No way to represent rdx = imul_128(rax, rm) >> 64,
                // Just set to undefined for now.
                // Could alternatively either Special or add Arithmetic64High.
                let eax = self.reg_variable_size(Register(0), op_size);
                let edx = self.reg_variable_size(Register(2), op_size);
                let multiply = ctx.mul(eax, rm.op);
                if op_size == MemAccessSize::Mem64 {
                    self.output(Operation::MoveSet(vec![
                        (DestOperand::from_oper(edx), self.ctx.new_undef()),
                        (DestOperand::from_oper(eax), multiply),
                    ]));
                } else {
                    let size = op_size.bits() as u64;
                    self.output(Operation::MoveSet(vec![
                        (DestOperand::from_oper(edx), ctx.rsh_const(multiply, size)),
                        (DestOperand::from_oper(eax), multiply),
                    ]));
                }
            },
            // Div, idiv
            6 | 7 => {
                // edx = edx:eax % rm, eax = edx:eax / rm
                let eax = self.reg_variable_size(Register(0), op_size);
                let edx = self.reg_variable_size(Register(2), op_size);
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
                self.output(Operation::MoveSet(vec![
                    (DestOperand::from_oper(edx), modulo),
                    (DestOperand::from_oper(eax), div),
                ]));
            }
            _ => return Err(self.unknown_opcode()),
        }
        Ok(())
    }

    fn various_0f_ba(&mut self) -> Result<(), Failed> {
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        match variant {
            4 => self.bit_test(BitTest::NoChange, true),
            5 => self.bit_test(BitTest::Set, true),
            6 => self.bit_test(BitTest::Reset, true),
            7 => self.bit_test(BitTest::Complement, true),
            _ => Err(self.unknown_opcode()),
        }
    }

    fn bit_test(&mut self, test: BitTest, imm8: bool) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let op_size = self.mem16_32();
        let (dest, index) = if imm8 {
            let (rm, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
            (rm, imm)
        } else {
            let (rm, r) = self.parse_modrm(op_size)?;
            (rm, self.r_to_operand(r).clone())
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
                self.output(mov(
                    dest.dest,
                    ctx.or(
                        dest.op,
                        bit_mask,
                    ),
                ));
            }
            BitTest::Reset => {
                self.output(mov(
                    dest.dest,
                    ctx.and(
                        dest.op,
                        ctx.xor_const(
                            bit_mask,
                            0xffff_ffff_ffff_ffff,
                        ),
                    ),
                ));
            }
            BitTest::Complement => {
                self.output(mov(
                    dest.dest,
                    ctx.xor(
                        dest.op,
                        bit_mask,
                    ),
                ));
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
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let ctx = self.ctx;
        let zero = ctx.const_0();
        if src_size == MemAccessSize::Mem32 {
            for i in (0..amt).rev() {
                let dest1 = dest.dest_operand_xmm(i * 2);
                let dest2 = dest.dest_operand_xmm(i * 2 + 1);
                let val = self.rm_to_operand_xmm(&rm, i);
                let arith = ctx.float_arithmetic(ArithOpType::ToDouble, val, zero, src_size);
                let op = ctx.rsh_const(arith, 0x20);
                self.output(Operation::Move(dest2, op, None));
                let op = ctx.and_const(arith, 0xffff_ffff);
                self.output(Operation::Move(dest1, op, None));
            }
        } else {
            for i in 0..amt {
                let dest = dest.dest_operand_xmm(i);
                let val = self.rm_to_operand_xmm_64(&rm, i);
                let arith = ctx.float_arithmetic(ArithOpType::ToDouble, val, zero, src_size);
                self.output(Operation::Move(dest, arith, None));
            }
            for i in amt..4 {
                let dest = dest.dest_operand_xmm(i);
                self.output(Operation::Move(dest, zero, None));
            }
        }
        Ok(())
    }

    fn sse_float_max(&mut self) -> Result<(), Failed> {
        let (size, amt) = if self.has_prefix(0xf2) {
            (MemAccessSize::Mem64, 1)
        } else if self.has_prefix(0x66) {
            (MemAccessSize::Mem64, 2)
        } else if self.has_prefix(0xf3) {
            (MemAccessSize::Mem32, 1)
        } else {
            (MemAccessSize::Mem32, 4)
        };
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let ctx = self.ctx;
        if size == MemAccessSize::Mem32 {
            for i in 0..amt {
                let dest = self.r_to_dest_and_operand_xmm(dest, i);
                let rhs = self.rm_to_operand_xmm(&rm, i);
                let cmp = ctx.float_arithmetic(ArithOpType::GreaterThan, rhs, dest.op, size);
                let op = Operation::Move(dest.dest, rhs, Some(cmp));
                self.output(op);
            }
        } else {
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
                let cmp = ctx.float_arithmetic(ArithOpType::GreaterThan, rhs, dest_op, size);
                self.output(Operation::Move(dest1.dest, rhs1, Some(cmp)));
                self.output(Operation::Move(dest2.dest, rhs2, Some(cmp)));
            }
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
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let ctx = self.ctx;
        if size == MemAccessSize::Mem32 {
            for i in 0..amt {
                let dest = self.r_to_dest_and_operand_xmm(dest, i);
                let rhs = self.rm_to_operand_xmm(&rm, i);
                let op = make_float_operation(self.ctx, dest.dest, ty, dest.op, rhs, size);
                self.output(op);
            }
        } else {
            for i in 0..amt {
                let dest1 = self.r_to_dest_and_operand_xmm(dest, i * 2);
                let dest2 = self.r_to_dest_and_operand_xmm(dest, i * 2 + 1);
                let dest_op = ctx.or(
                    dest1.op,
                    ctx.lsh_const(dest2.op, 0x20),
                );
                let rhs = self.rm_to_operand_xmm_64(&rm, i);
                let arith = ctx.float_arithmetic(ty, dest_op, rhs, size);
                let op = ctx.and_const(arith, 0xffff_ffff);
                self.output(Operation::Move(dest1.dest, op, None));
                let op = ctx.rsh_const(arith, 0x20);
                self.output(Operation::Move(dest2.dest, op, None));
            }
        }
        Ok(())
    }

    fn xorps(&mut self) -> Result<(), Failed> {
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        for i in 0..4 {
            let dest = self.r_to_dest_and_operand_xmm(dest, i);
            let rhs = self.rm_to_operand_xmm(&rm, i);
            self.output_xor(dest, rhs);
        }
        Ok(())
    }

    fn pmullw(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        // Mul 16-bit packed
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
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
            self.output(Operation::Move(
                dest.dest,
                ctx.or(
                    low_word,
                    ctx.lsh_const(
                        high_word,
                        0x10,
                    ),
                ),
                None,
            ));
        }
        Ok(())
    }

    fn sse_compare(&mut self) -> Result<(), Failed> {
        // TODO
        // Needs to specify how to check for unordered. Does not(y > x) mean
        // that `y == x || x > y || nan(y) || nan(x)` or just `y == x || x > y`.
        let ctx = self.ctx;
        let zero = ctx.const_0();
        // zpc = 111 if unordered, 000 if greater, 001 if less, 100 if equal
        // or alternatively
        // z = equal or unordererd
        // p = unordered
        // c = less than or unordered
        self.output(Operation::Move(
            DestOperand::Flag(Flag::Zero), ctx.and_const(ctx.new_undef(), 1), None,
        ));
        self.output(Operation::Move(
            DestOperand::Flag(Flag::Carry), ctx.and_const(ctx.new_undef(), 1), None,
        ));
        self.output(Operation::Move(
            DestOperand::Flag(Flag::Parity), ctx.and_const(ctx.new_undef(), 1), None,
        ));
        self.output(Operation::Move(DestOperand::Flag(Flag::Overflow), zero, None));
        self.output(Operation::Move(DestOperand::Flag(Flag::Sign), zero, None));
        Ok(())
    }

    fn sse_int_to_float(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0xf3) {
            return Err(self.unknown_opcode());
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let rm = self.rm_to_operand(&rm);
        self.output(Operation::Move(
            r.dest_operand_xmm(0),
            self.ctx.float_arithmetic(
                ArithOpType::ToFloat,
                rm,
                self.ctx.const_0(),
                MemAccessSize::Mem32
            ),
            None,
        ));
        Ok(())
    }

    fn sse_int_double_conversion(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;

        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
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
                self.output(mov(r.dest_operand_xmm(i * 2 + 1), high));
                let low = ctx.and_const(result_f64, 0xffff_ffff);
                self.output(mov(r.dest_operand_xmm(i * 2), low));
            }
        } else if self.has_prefix(0xf2) {
            // 2 f64 to 2 i32
            for i in 0..2 {
                let op = self.rm_to_operand_xmm_64(&rm, i);
                self.output(Operation::Move(
                    r.dest_operand_xmm(i),
                    ctx.float_arithmetic(ArithOpType::ToInt, op, zero, MemAccessSize::Mem64),
                    None,
                ));
            }
            for i in 2..4 {
                self.output(mov(r.dest_operand_xmm(i), zero));
            }
        } else {
            return Err(self.unknown_opcode());
        }
        Ok(())
    }

    fn cvttss2si(&mut self) -> Result<(), Failed> {
        // TODO Doesn't actually truncate overflows
        if !self.has_prefix(0xf3) {
            return Err(self.unknown_opcode());
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let op = self.rm_to_operand_xmm(&rm, 0);
        self.output(Operation::Move(
            r.dest_operand(),
            self.ctx.float_arithmetic(
                ArithOpType::ToInt,
                op,
                self.ctx.const_0(),
                MemAccessSize::Mem32
            ),
            None,
        ));
        Ok(())
    }

    fn cvtdq2ps(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let zero = self.ctx.const_0();
        for i in 0..4 {
            let dest = r.dest_operand_xmm(i);
            let rm = self.rm_to_operand_xmm(&rm, i);
            self.output_arith(dest, ArithOpType::ToFloat, rm, zero);
        }
        Ok(())
    }

    fn punpcklbw(&mut self) -> Result<(), Failed> {
        fn rsh_and_const<'e>(
            ctx: OperandCtx<'e>,
            val: Operand<'e>,
            and_mask: u32,
            shift: u32,
        ) -> Operand<'e> {
            ctx.rsh_const(
                ctx.and_const(
                    val,
                    and_mask as u64,
                ),
                shift as u64,
            )
        }

        fn lsh_and_const<'e>(
            ctx: OperandCtx<'e>,
            val: Operand<'e>,
            and_mask: u32,
            shift: u32,
        ) -> Operand<'e> {
            ctx.lsh_const(
                ctx.and_const(
                    val,
                    and_mask as u64,
                ),
                shift as u64,
            )
        }

        use self::operation_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        // r.0 = (r.0 & ff) | (rm.0 & ff) << 8 | (r.0 & ff00) << 8 | (rm.0 & ff00) << 10
        // r.1 = (r.0 & ff_0000) >> 10 | (rm.0 & ff_0000) >> 8 |
        //      (r.0 & ff00_0000) >> 8 | (rm.0 & ff00_0000)
        // r.2 = (r.1 & ff) | (rm.1 & ff) << 8 | (r.1 & ff00) << 8 | (rm.1 & ff00) << 10
        // r.3 = (r.1 & ff_0000) >> 10 | (rm.1 & ff_0000) >> 8 |
        //      (r.1 & ff00_0000) >> 8 | (rm.1 & ff00_0000)
        // Do things in reverse to avoid overwriting r0/r1
        let ctx = self.ctx;
        for &i in &[1, 0] {
            let out0 = r.dest_operand_xmm(i * 2);
            let out1 = r.dest_operand_xmm(i * 2 + 1);
            let in_r = self.r_to_operand_xmm(r, i);
            let in_rm = self.rm_to_operand_xmm(&rm, i);
            self.output(mov(out1, ctx.or(
                ctx.or(
                    rsh_and_const(ctx, in_r, 0xff_0000, 0x10),
                    rsh_and_const(ctx, in_rm, 0xff_0000, 0x8),
                ),
                ctx.or(
                    rsh_and_const(ctx, in_r, 0xff00_0000, 0x8),
                    rsh_and_const(ctx, in_rm, 0xff00_0000, 0),
                ),
            )));
            self.output(mov(out0, ctx.or(
                ctx.or(
                    lsh_and_const(ctx, in_r, 0xff, 0),
                    lsh_and_const(ctx, in_rm, 0xff, 0x8),
                ),
                ctx.or(
                    lsh_and_const(ctx, in_r, 0xff00, 0x8),
                    lsh_and_const(ctx, in_rm, 0xff00, 0x10),
                ),
            )));
        }
        Ok(())
    }

    fn mov_sse_6e(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let op_size = match self.rex_prefix() & 0x8 != 0 {
            true => MemAccessSize::Mem64,
            false => MemAccessSize::Mem32,
        };
        let (rm, r) = self.parse_modrm(op_size)?;
        let rm_op = self.rm_to_operand(&rm);
        self.output(mov(r.dest_operand_xmm(0), rm_op));
        let ctx = self.ctx;
        let zero = ctx.const_0();
        if op_size == MemAccessSize::Mem64 {
            let rm_high = ctx.rsh_const(rm_op, 0x20);
            self.output(mov(r.dest_operand_xmm(1), rm_high));
        } else {
            self.output(mov(r.dest_operand_xmm(1), zero));
        }
        self.output(mov(r.dest_operand_xmm(2), zero));
        self.output(mov(r.dest_operand_xmm(3), zero));
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
        use self::operation_helpers::*;
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
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
                    self.output(mov(dest_op, src));
                    return Ok(());
                }
                (false, 8) => {
                    let mut src = src.clone();
                    if self.rex_prefix() & 0x8 != 0 {
                        src.size = RegisterSize::R64;
                    }
                    let dest_op = self.rm_to_dest_operand(&src);
                    let src = self.rm_to_operand_xmm_64(dest, 0);
                    self.output(mov(dest_op, src));
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
            self.output(mov(dest, src));
        }
        Ok(())
    }

    fn mov_sse_d6(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, src) = self.parse_modrm(MemAccessSize::Mem32)?;
        let dest = self.rm_to_dest_operand_xmm(&rm, 0);
        self.output(mov(dest, self.r_to_operand_xmm(src, 0)));
        let dest = self.rm_to_dest_operand_xmm(&rm, 1);
        self.output(mov(dest, self.r_to_operand_xmm(src, 1)));
        if !rm.is_memory() {
            let dest = self.rm_to_dest_operand_xmm(&rm, 3);
            self.output(mov(dest, self.ctx.const_0()));
            let dest = self.rm_to_dest_operand_xmm(&rm, 3);
            self.output(mov(dest, self.ctx.const_0()));
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
                Some(high_u32_set.clone()),
            ));
        }
    }

    fn packed_shift_imm(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(1)?;
        if !self.has_prefix(0x66) || byte & 0xc0 != 0xc0 {
            return Err(self.unknown_opcode());
        }
        let variant = (byte >> 3) & 0x7;
        let dest = ModRm_R(byte & 0x7, RegisterSize::R32);
        let mut constant = self.read_u8(2)? as u64;
        // variants 3/7 shift in bytes
        if variant == 3 || variant == 7 {
            constant = constant << 3;
        }
        let constant = self.ctx.constant(constant);
        match variant {
            2 => self.packed_shift_right_xmm_u64(dest, constant),
            3 => self.packed_shift_right_xmm_u128(dest, constant),
            6 => self.packed_shift_left_xmm_u64(dest, constant),
            7 => self.packed_shift_left_xmm_u128(dest, constant),
            _ => return Err(self.unknown_opcode()),
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

    fn packed_shift_left_xmm_u128(&mut self, dest: ModRm_R, with: Operand<'e>) {
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
        let high_bits = ctx.rsh_const(
            ctx.and_const(
                with,
                !0x3f,
            ),
            0x1,
        );
        let x_arr = [
            ctx.and_const(with, 0x1f),
            ctx.and_const(with, 0x20),
            high_bits,
        ];
        let dest_zero = self.r_to_dest_and_operand_xmm(dest, 0);
        let dests = [
            dest.dest_operand_xmm(1),
            dest.dest_operand_xmm(2),
            dest.dest_operand_xmm(3),
        ];
        let ops = [
            self.r_to_operand_xmm(dest, 0),
            self.r_to_operand_xmm(dest, 1),
            self.r_to_operand_xmm(dest, 2),
            self.r_to_operand_xmm(dest, 3),
        ];
        for &x in &x_arr {
            for i in (1..4).rev() {
                self.output_arith(
                    dests[i - 1].clone(),
                    ArithOpType::Or,
                    ctx.lsh(ops[i], x),
                    ctx.rsh(ops[i - 1], ctx.sub_const_left(0x20, x)),
                );
            }
            self.output_lsh(dest_zero.clone(), x);
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

    fn packed_shift_right_xmm_u128(&mut self, dest: ModRm_R, with: Operand<'e>) {
        // let x = with & 0x1f
        // dest.0 = (dest.0 >> x) | (dest.1 << (32 - x))
        // dest.1 = (dest.1 >> x) | (dest.2 << (32 - x))
        // dest.2 = (dest.2 >> x) | (dest.3 << (32 - x))
        // dest.3 = (dest.3 >> x)
        // let x = with & 0x20
        // ...
        let ctx = self.ctx;
        let high_bits = ctx.rsh_const(
            ctx.and_const(
                with,
                !0x3f,
            ),
            0x1,
        );
        let x_arr = [
            ctx.and_const(with, 0x1f),
            ctx.and_const(with, 0x20),
            high_bits,
        ];
        let dest_three = self.r_to_dest_and_operand_xmm(dest, 3);
        let dests = [
            dest.dest_operand_xmm(0),
            dest.dest_operand_xmm(1),
            dest.dest_operand_xmm(2),
        ];
        let ops = [
            self.r_to_operand_xmm(dest, 0),
            self.r_to_operand_xmm(dest, 1),
            self.r_to_operand_xmm(dest, 2),
            self.r_to_operand_xmm(dest, 3),
        ];
        for &x in &x_arr {
            for i in 0..3 {
                self.output_arith(
                    dests[i].clone(),
                    ArithOpType::Or,
                    ctx.rsh(ops[i], x),
                    ctx.lsh(ops[i + 1], ctx.sub_const_left(0x20, x)),
                );
            }
            self.output_rsh(dest_three.clone(), x);
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
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
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
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let rm_0 = self.rm_to_operand_xmm(&rm, 0);
        // Zero everything if rm.1 is set
        self.packed_shift_right_xmm_u64(dest, rm_0);
        self.zero_xmm_if_rm1_nonzer0(&rm, dest);
        Ok(())
    }

    fn fpu_push(&mut self) {
        // fdecstp
        self.output(Operation::Special(vec![0xd9, 0xf6]));
    }

    fn fpu_pop(&mut self) {
        // fincstp
        self.output(Operation::Special(vec![0xd9, 0xf7]));
    }

    fn various_d8(&mut self) -> Result<(), Failed> {
        let byte = self.read_u8(1)?;
        let variant = (byte >> 3) & 0x7;
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem32)?;
        let op_ty = match variant {
            // Fadd
            0 => ArithOpType::Add,
            // Fmul
            1 => ArithOpType::Mul,
            // Fsub, fsubr
            4 | 5 => ArithOpType::Sub,
            // Fdiv, fdivr
            6 | 7 => ArithOpType::Div,
            _ => return Err(self.unknown_opcode()),
        };
        let rm = self.rm_to_operand(&rm);
        let st0 = self.ctx.register_fpu(0);
        let dest = DestOperand::Fpu(0);
        let (lhs, rhs) = if variant == 5 || variant == 7 {
            (rm, st0)
        } else {
            (st0, rm)
        };
        let op = make_float_operation(self.ctx, dest, op_ty, lhs, rhs, MemAccessSize::Mem32);
        self.output(op);
        Ok(())
    }

    fn various_d9(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let byte = self.read_u8(1)?;
        let variant = (byte >> 3) & 0x7;
        if variant == 6 && byte == 0xf6 || byte == 0xf7 {
            // Fincstp, fdecstp
            self.output(Operation::Special(self.data.into()));
            return Ok(());
        }
        let (rm_parsed, _) = self.parse_modrm(MemAccessSize::Mem32)?;
        let rm = self.rm_to_dest_and_operand(&rm_parsed);
        let ctx = self.ctx;
        match variant {
            // Fld
            0 => {
                self.fpu_push();
                self.output(mov_to_fpu(0, x87_variant(ctx, rm.op, 1)));
                Ok(())
            }
            // Fst/Fstp, as long as rm is mem
            2 | 3 => {
                self.output(mov(rm.dest, ctx.register_fpu(0)));
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
                    self.out.extend((0..10).map(|i| {
                        let address = ctx.add_const(
                            mem.address,
                            i * mem_bytes,
                        );
                        mov_to_mem(mem_size, address, ctx.new_undef())
                    }));
                }
                Ok(())
            }
            // Fstcw
            7 => {
                if rm_parsed.is_memory() {
                    self.output(mov(rm.dest, ctx.new_undef()));
                }
                Ok(())
            }
            _ => return Err(self.unknown_opcode()),
        }
    }

    fn various_fe_ff(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
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
        let (rm, _) = self.parse_modrm(op_size)?;
        let rm = self.rm_to_dest_and_operand(&rm);
        match variant {
            0 | 1 => {
                let is_inc = variant == 0;
                let op = rm.op;
                match is_inc {
                    true => self.output_add_const(rm, 1),
                    false => self.output_sub_const(rm, 1),
                }
                self.inc_dec_flags(is_inc, op);
            }
            2 | 3 => self.output(Operation::Call(rm.op)),
            4 | 5 => self.output(Operation::Jump { condition: self.ctx.const_1(), to: rm.op }),
            6 => {
                let esp = self.ctx.register(4);
                if Va::SIZE == 4 {
                    let new_esp = self.register_cache.register_offset_const(4, -4);
                    self.output(mov_to_reg(4, new_esp));
                    self.output(mov_to_mem(MemAccessSize::Mem32, esp, rm.op));
                } else {
                    let new_esp = self.register_cache.register_offset_const(4, -8);
                    self.output(mov_to_reg(4, new_esp));
                    self.output(mov_to_mem(MemAccessSize::Mem64, esp, rm.op));
                }
            }
            _ => return Err(self.unknown_opcode()),
        }
        Ok(())
    }

    fn pop_rm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let (rm, _) = self.parse_modrm(self.mem16_32())?;
        let esp = self.ctx.register(4);
        let rm_dest = self.rm_to_dest_operand(&rm);
        if Va::SIZE == 4 {
            self.output(mov(rm_dest, self.ctx.mem32(esp)));
            let new_esp = self.register_cache.register_offset_const(4, 4);
            self.output(mov_to_reg(4, new_esp));
        } else {
            self.output(mov(rm_dest, self.ctx.mem64(esp)));
            let new_esp = self.register_cache.register_offset_const(4, 8);
            self.output(mov_to_reg(4, new_esp));
        }
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
        let arith = match (self.read_u8(1)? >> 3) & 0x7 {
            0 => ArithOperation::RotateLeft,
            1 => ArithOperation::RotateRight,
            4 | 6 => ArithOperation::LeftShift,
            5 => ArithOperation::RightShift,
            7 => ArithOperation::RightShiftArithmetic,
            _ => return Err(self.unknown_opcode()),
        };
        self.generic_arith_with_imm_op(arith, MemAccessSize::Mem8)
    }

    fn bitwise_compact_op(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (mut rm, _) = self.parse_modrm(op_size)?;
        let shift_count = match self.read_u8(0)? & 2 {
            0 => self.ctx.const_1(),
            _ => self.reg_variable_size(Register(1), operand::MemAccessSize::Mem8).clone(),
        };
        let arith = match (self.read_u8(1)? >> 3) & 0x7 {
            0 => ArithOperation::RotateLeft,
            1 => ArithOperation::RotateRight,
            4 | 6 => ArithOperation::LeftShift,
            5 => ArithOperation::RightShift,
            7 => ArithOperation::RightShiftArithmetic,
            _ => return Err(self.unknown_opcode()),
        };
        self.do_arith_operation(arith, &mut rm, shift_count);
        Ok(())
    }

    fn signed_multiply_rm_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
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
        if Va::SIZE == 4 {
            if op_size != MemAccessSize::Mem32 {
                self.output(mov(r.dest_operand(), ctx.signed_mul(rm, imm, op_size)));
            } else {
                self.output(mov(r.dest_operand(), ctx.mul(rm, imm)));
            }
        } else {
            if op_size != MemAccessSize::Mem64 {
                self.output(mov(r.dest_operand(), ctx.signed_mul(rm, imm, op_size)));
            } else {
                self.output(mov(r.dest_operand(), ctx.mul(rm, imm)));
            }
        }
        Ok(())
    }

    fn shld_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let (hi, low, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let ctx = self.ctx;
        let imm = ctx.and_const(imm, 0x1f);
        let hi = self.rm_to_dest_and_operand(&hi);
        if imm != ctx.const_0() {
            // TODO flags
            let low = self.r_to_operand(low);
            self.output(
                mov(
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
                ),
            );
        }
        Ok(())
    }

    fn shrd_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let (low, hi, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let ctx = self.ctx;
        let imm = ctx.and_const(imm, 0x1f);
        let low = self.rm_to_dest_and_operand(&low);
        if imm != ctx.const_0() {
            // TODO flags
            let hi = self.r_to_operand(hi);
            self.output(
                mov(
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
                ),
            );
        }
        Ok(())
    }

    fn imul_normal(&mut self) -> Result<(), Failed> {
        let size = self.mem16_32();
        let (rm, r) = self.parse_modrm(size)?;
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

    fn arith_with_imm_op(&mut self) -> Result<(), Failed> {
        let arith = match (self.read_u8(1)? >> 3) & 0x7 {
            0 => ArithOperation::Add,
            1 => ArithOperation::Or,
            2 => ArithOperation::Adc,
            3 => ArithOperation::Sbb,
            4 => ArithOperation::And,
            5 => ArithOperation::Sub,
            6 => ArithOperation::Xor,
            7 => ArithOperation::Cmp,
            _ => unreachable!(),
        };
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
        use self::operation_helpers::*;
        let byte = self.read_u8(0)?;
        let is_push = byte < 0x58;
        let reg = if self.rex_prefix() & 0x1 == 0 {
            byte & 0x7
        } else {
            8 + (byte & 0x7)
        };
        let esp = self.ctx.register(4);
        match is_push {
            true => {
                if Va::SIZE == 4 {
                    let new_esp = self.register_cache.register_offset_const(4, -4);
                    self.output(mov_to_reg(4, new_esp));
                    self.output(mov_to_mem(MemAccessSize::Mem32, esp, self.ctx.register(reg)));
                } else {
                    let new_esp = self.register_cache.register_offset_const(4, -8);
                    self.output(mov_to_reg(4, new_esp));
                    self.output(mov_to_mem(MemAccessSize::Mem64, esp, self.ctx.register(reg)));
                }
            }
            false => {
                if Va::SIZE == 4 {
                    self.output(mov_to_reg(reg, self.ctx.mem32(esp)));
                    let new_esp = self.register_cache.register_offset_const(4, 4);
                    self.output(mov_to_reg(4, new_esp));
                } else {
                    self.output(mov_to_reg(reg, self.ctx.mem64(esp)));
                    let new_esp = self.register_cache.register_offset_const(4, 8);
                    self.output(mov_to_reg(4, new_esp));
                }
            }
        }
        Ok(())
    }

    fn push_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let imm_size = match self.read_u8(0)? {
            0x68 => self.mem16_32(),
            _ => MemAccessSize::Mem8,
        };
        let constant = self.read_variable_size_32(1, imm_size)? as u32;
        let esp = self.ctx.register_ref(4);
        // TODO is this right on 64bit? Probably not
        let new_esp = self.register_cache.register_offset_const(4, -4);
        self.output(mov_to_reg(4, new_esp));
        self.output(mov_to_mem(MemAccessSize::Mem32, esp, self.ctx.constant(constant as u64)));
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
    use crate::operand::{Operand, MemAccessSize, ArithOpType, ArithOperand, Register};
    use super::{DestOperand, Operation};

    pub fn mov_to_reg<'e>(dest: u8, from: Operand<'e>) -> Operation<'e> {
        Operation::Move(DestOperand::Register64(Register(dest)), from, None)
    }

    pub fn mov_to_fpu<'e>(dest: u8, from: Operand<'e>) -> Operation<'e> {
        Operation::Move(DestOperand::Fpu(dest), from, None)
    }

    pub fn mov_to_reg_variable_size<'e>(
        size: MemAccessSize,
        dest: u8,
        from: Operand<'e>,
    ) -> Operation<'e> {
        let dest = match size {
            MemAccessSize::Mem8 => DestOperand::Register8Low(Register(dest)),
            MemAccessSize::Mem16 => DestOperand::Register16(Register(dest)),
            MemAccessSize::Mem32 => DestOperand::Register32(Register(dest)),
            MemAccessSize::Mem64 => DestOperand::Register64(Register(dest)),
        };
        Operation::Move(dest, from, None)
    }

    pub fn mov_to_mem<'e>(
        size: MemAccessSize,
        address: Operand<'e>,
        from: Operand<'e>,
    ) -> Operation<'e> {
        let access = crate::operand::MemAccess {
            size,
            address: address.clone(),
        };
        Operation::Move(DestOperand::Memory(access), from, None)
    }

    pub fn mov<'e>(dest: DestOperand<'e>, from: Operand<'e>) -> Operation<'e> {
        Operation::Move(dest, from, None)
    }

    pub fn flags<'e>(
        ty: ArithOpType,
        left: Operand<'e>,
        right: Operand<'e>,
        size: MemAccessSize,
    ) -> Operation<'e> {
        let arith = ArithOperand {
            ty,
            left,
            right,
        };
        Operation::SetFlags(arith, size)
    }
}

#[derive(Clone, Debug)]
pub enum Operation<'e> {
    Move(DestOperand<'e>, Operand<'e>, Option<Operand<'e>>),
    Call(Operand<'e>),
    Jump { condition: Operand<'e>, to: Operand<'e> },
    Return(u32),
    /// Special cases like interrupts etc that scarf doesn't want to touch.
    /// Also rep mov for now
    Special(Vec<u8>),
    /// Set flags based on operation type. While Move(..) could handle this
    /// (And it does for odd cases like inc), it would mean generating 5
    /// additional operations for each instruction, so special-case flags.
    SetFlags(ArithOperand<'e>, MemAccessSize),
    /// Like Move, but evaluate all operands before assigning over any.
    /// Used for mul/div/swap.
    MoveSet(Vec<(DestOperand<'e>, Operand<'e>)>),
    /// Error - Should assume that no more operations can be decoded from current position.
    Error(Error),
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

fn make_float_operation<'e>(
    ctx: OperandCtx<'e>,
    dest: DestOperand<'e>,
    ty: ArithOpType,
    left: Operand<'e>,
    right: Operand<'e>,
    size: MemAccessSize,
) -> Operation<'e> {
    let op = ctx.float_arithmetic(ty, left, right, size);
    Operation::Move(dest, op, None)
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
    Low8,
    High8,
    R16,
    R32,
    R64,
}

impl RegisterSize {
    fn bits(self) -> u32 {
        match self {
            RegisterSize::Low8 | RegisterSize::High8 => 8,
            RegisterSize::R16 => 16,
            RegisterSize::R32 => 32,
            RegisterSize::R64 => 64,
        }
    }

    fn to_mem_access_size(self) -> MemAccessSize {
        match self {
            RegisterSize::Low8 | RegisterSize::High8 => MemAccessSize::Mem8,
            RegisterSize::R16 => MemAccessSize::Mem16,
            RegisterSize::R32 => MemAccessSize::Mem32,
            RegisterSize::R64 => MemAccessSize::Mem64,
        }
    }

    fn from_mem_access_size(size: MemAccessSize) -> RegisterSize {
        match size {
            MemAccessSize::Mem8 => RegisterSize::Low8,
            MemAccessSize::Mem16 => RegisterSize::R16,
            MemAccessSize::Mem32 => RegisterSize::R32,
            MemAccessSize::Mem64 => RegisterSize::R64,
        }
    }
}

impl ModRm_R {
    fn dest_operand<'e>(self) -> DestOperand<'e> {
        match self.1 {
            RegisterSize::R64 => DestOperand::Register64(Register(self.0)),
            RegisterSize::R32 => DestOperand::Register32(Register(self.0)),
            RegisterSize::R16 => DestOperand::Register16(Register(self.0)),
            RegisterSize::Low8 => DestOperand::Register8Low(Register(self.0)),
            RegisterSize::High8 => DestOperand::Register8High(Register(self.0)),
        }
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
            index: u8::max_value(),
            index_mul: 0,
            constant: 0,
        }
    }
}

impl ModRm_Rm {
    fn is_memory(&self) -> bool {
        self.index != u8::max_value()
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
            index: u8::max_value(),
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
    Register64(Register),
    Register32(Register),
    Register16(Register),
    Register8High(Register),
    Register8Low(Register),
    Xmm(u8, u8),
    Fpu(u8),
    Flag(Flag),
    Memory(MemAccess<'e>),
}

impl<'e> DestOperand<'e> {
    pub fn reg_variable_size(reg: u8, size: MemAccessSize) -> DestOperand<'e> {
        match size {
            MemAccessSize::Mem8 => DestOperand::Register8Low(Register(reg)),
            MemAccessSize::Mem16 => DestOperand::Register16(Register(reg)),
            MemAccessSize::Mem32 => DestOperand::Register32(Register(reg)),
            MemAccessSize::Mem64 => DestOperand::Register64(Register(reg)),
        }
    }

    pub fn from_oper(val: Operand<'e>) -> DestOperand<'e> {
        dest_operand(val)
    }

    pub fn as_operand(&self, ctx: OperandCtx<'e>) -> Operand<'e> {
        match *self {
            DestOperand::Register32(x) => ctx.and_const(ctx.register_ref(x.0), 0xffff_ffff),
            DestOperand::Register16(x) => ctx.and_const(ctx.register_ref(x.0), 0xffff),
            DestOperand::Register8High(x) => ctx.rsh_const(
                ctx.and_const(ctx.register(x.0), 0xffff),
                8,
            ),
            DestOperand::Register8Low(x) => ctx.and_const(ctx.register_ref(x.0), 0xff),
            DestOperand::Register64(x) => ctx.register(x.0),
            DestOperand::Xmm(x, y) => ctx.xmm(x, y),
            DestOperand::Fpu(x) => ctx.register_fpu(x),
            DestOperand::Flag(x) => ctx.flag(x).clone(),
            DestOperand::Memory(ref x) => ctx.mem_variable_rc(x.size, x.address),
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

#[cfg(test)]
mod test {
    use super::*;
    use crate::VirtualAddress;
    use crate::exec_state::Disassembler;
    #[test]
    fn test_operations_mov16() {
        use crate::operand::OperandContext;

        let ctx = &OperandContext::new();
        let buf = [0x66, 0xc7, 0x47, 0x62, 0x00, 0x20];
        let mut disasm = Disassembler32::new(ctx);
        disasm.set_pos(&buf[..], 0, VirtualAddress(0));
        let ins = disasm.next();
        assert_eq!(ins.ops().len(), 1);
        let op = &ins.ops()[0];
        let dest = ctx.mem16(ctx.add_const(ctx.register(0x7), 0x62));

        assert!(matches!(*op, Operation::Move(a, b, None) if
                a == dest_operand(dest) && b == ctx.constant(0x2000)));
    }

    #[test]
    fn test_sib() {
        use crate::operand::OperandContext;

        let ctx = &OperandContext::new();
        let buf = [0x89, 0x84, 0xb5, 0x18, 0xeb, 0xff, 0xff];
        let mut disasm = Disassembler32::new(ctx);
        disasm.set_pos(&buf[..], 0, VirtualAddress(0));
        let ins = disasm.next();
        assert_eq!(ins.ops().len(), 1);
        let op = &ins.ops()[0];
        let dest = ctx.mem32(
            ctx.add(
                ctx.mul(
                    ctx.constant(4),
                    ctx.register(6),
                ),
                ctx.sub(
                    ctx.register(5),
                    ctx.constant(0x14e8),
                ),
            ),
        );

        match op.clone() {
            Operation::Move(d, f, cond) => {
                let d = d.as_operand(ctx);
                assert_eq!(d, dest);
                assert_eq!(f, ctx.register(0));
                assert_eq!(cond, None);
            }
            _ => panic!("Unexpected op {:?}", op),
        }
    }
}
