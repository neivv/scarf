use std::cell::RefCell;
use std::rc::Rc;

use lde::Isa;
use quick_error::quick_error;
use smallvec::SmallVec;

use crate::exec_state::{VirtualAddress};
use crate::operand::{
    self, ArithOpType, Flag, MemAccess, Operand, OperandContext, OperandType, Register,
    MemAccessSize, ArithOperand,
};
use crate::VirtualAddress as VirtualAddress32;
use crate::VirtualAddress64;

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        UnknownOpcode(op: Vec<u8>) {
            description("Unknown opcode")
            display("Unknown opcode {:02x?}", op)
        }
        End {
            description("End of file")
        }
        // The preceding instruction's operations will be given before this error.
        Branch {
            description("Reached a branch")
        }
        InternalDecodeError {
            description("Internal decode error")
        }
    }
}

/// Used by InstructionOpsState to signal that something had failed and return with ?
/// without making return value heavier.
/// Error should be stored in &mut self
struct Failed;

pub type OperationVec = SmallVec<[Operation; 8]>;

pub struct Disassembler32<'a> {
    buf: &'a [u8],
    pos: usize,
    virtual_address: VirtualAddress32,
    is_branching: bool,
    register_cache: Rc<RegisterCache>,
    ops_buffer: SmallVec<[Operation; 8]>,
    ctx: &'a OperandContext,
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler32<'a> {
    type VirtualAddress = VirtualAddress32;

    fn new(
        buf: &'a [u8],
        pos: usize,
        address: VirtualAddress32,
        ctx: &'a OperandContext,
    ) -> Disassembler32<'a> {
        assert!(pos < buf.len());
        Disassembler32 {
            buf,
            pos,
            virtual_address: address,
            is_branching: false,
            register_cache: RegisterCache::get(&ctx),
            ops_buffer: SmallVec::new(),
            ctx,
        }
    }

    fn next<'s>(&'s mut self) -> Result<Instruction<'s, VirtualAddress32>, Error> {
        if self.is_branching {
            return Err(Error::Branch);
        }
        let length = lde::X86::ld(&self.buf[self.pos..]) as usize;
        if length == 0 {
            if self.pos == self.buf.len() {
                return Err(Error::End);
            } else {
                return Err(Error::UnknownOpcode(vec![self.buf[self.pos]]));
            }
        }
        let address = self.virtual_address + self.pos as u32;
        let data = &self.buf[self.pos..self.pos + length];
        self.ops_buffer.clear();
        instruction_operations32(
            address,
            data,
            self.ctx,
            &mut self.ops_buffer,
            &self.register_cache,
        )?;
        let ins = Instruction {
            address,
            ops: &self.ops_buffer,
            length: length as u32,
        };
        if ins.is_finishing() {
            self.is_branching = true;
            self.buf = &self.buf[..self.pos + length];
            self.pos = self.buf.len();
        } else {
            self.pos += length;
        }
        Ok(ins)
    }

    fn address(&self) -> VirtualAddress32 {
        self.virtual_address + self.pos as u32
    }
}

pub struct Disassembler64<'a> {
    buf: &'a [u8],
    pos: usize,
    virtual_address: VirtualAddress64,
    is_branching: bool,
    register_cache: Rc<RegisterCache>,
    ops_buffer: SmallVec<[Operation; 8]>,
    ctx: &'a OperandContext,
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler64<'a> {
    type VirtualAddress = VirtualAddress64;

    fn new(
        buf: &'a [u8],
        pos: usize,
        address: VirtualAddress64,
        ctx: &'a OperandContext,
    ) -> Disassembler64<'a> {
        assert!(pos < buf.len());
        Disassembler64 {
            buf,
            pos,
            virtual_address: address,
            is_branching: false,
            register_cache: RegisterCache::get(&ctx),
            ops_buffer: SmallVec::new(),
            ctx,
        }
    }

    fn next<'s>(&'s mut self) -> Result<Instruction<'s, VirtualAddress64>, Error> {
        if self.is_branching {
            return Err(Error::Branch);
        }
        let length = lde::X64::ld(&self.buf[self.pos..]) as usize;
        if length == 0 {
            if self.pos == self.buf.len() {
                return Err(Error::End);
            } else {
                return Err(Error::UnknownOpcode(vec![self.buf[self.pos]]));
            }
        }
        let address = self.virtual_address + self.pos as u32;
        let data = &self.buf[self.pos..self.pos + length];
        self.ops_buffer.clear();
        instruction_operations64(
            address,
            data,
            self.ctx,
            &mut self.ops_buffer,
            &self.register_cache,
        )?;
        let ins = Instruction {
            address,
            ops: &self.ops_buffer,
            length: length as u32,
        };
        if ins.is_finishing() {
            self.is_branching = true;
            self.buf = &self.buf[..self.pos + length];
            self.pos = self.buf.len();
        } else {
            self.pos += length;
        }
        Ok(ins)
    }

    fn address(&self) -> VirtualAddress64 {
        self.virtual_address + self.pos as u32
    }
}

pub struct Instruction<'a, Va: VirtualAddress> {
    address: Va,
    ops: &'a SmallVec<[Operation; 8]>,
    length: u32,
}

impl<'a, Va: VirtualAddress> Instruction<'a, Va> {
    pub fn ops(&self) -> &[Operation] {
        &self.ops
    }

    pub fn address(&self) -> Va {
        self.address
    }

    pub fn len(&self) -> u32 {
        self.length
    }

    fn is_finishing(&self) -> bool {
        self.ops().iter().any(|op| match *op {
            Operation::Jump { .. } => true,
            Operation::Return(..) => true,
            _ => false,
        })
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

struct RegisterCache {
    register8_low: [Rc<Operand>; 16],
    register8_high: [Rc<Operand>; 4],
    register16: [Rc<Operand>; 16],
    register32: [Rc<Operand>; 16],
}

thread_local! {
    static REGISTER_CACHE: RefCell<Option<Rc<RegisterCache>>> = RefCell::new(None);
}

impl RegisterCache {
    fn get(ctx: &OperandContext) -> Rc<RegisterCache> {
        REGISTER_CACHE.with(|x| {
            let mut x = x.borrow_mut();
            x.get_or_insert_with(|| Rc::new(RegisterCache::new(ctx))).clone()
        })
    }

    fn new(ctx: &OperandContext) -> RegisterCache {
        use crate::operand_helpers::*;
        let reg8_low = |i: u8| -> Rc<Operand> {
            Operand::simplified(operand_and(
                ctx.const_ff(),
                ctx.register(i),
            ))
        };
        let reg8_high = |i: u8| -> Rc<Operand> {
            Operand::simplified(operand_rsh(
                operand_and(
                    ctx.const_ff00(),
                    ctx.register(i),
                ),
                ctx.const_8(),
            ))
        };
        let reg16 = |i: u8| -> Rc<Operand> {
            Operand::simplified(operand_and(
                ctx.const_ffff(),
                ctx.register(i),
            ))
        };
        let reg32 = |i: u8| -> Rc<Operand> {
            Operand::simplified(operand_and(
                ctx.const_ffffffff(),
                ctx.register(i),
            ))
        };

        RegisterCache {
            register8_low: [
                reg8_low(0), reg8_low(1), reg8_low(2), reg8_low(3),
                reg8_low(4), reg8_low(5), reg8_low(6), reg8_low(7),
                reg8_low(8), reg8_low(9), reg8_low(10), reg8_low(11),
                reg8_low(12), reg8_low(13), reg8_low(14), reg8_low(15),
            ],
            register8_high: [
                reg8_high(0),
                reg8_high(1),
                reg8_high(2),
                reg8_high(3),
            ],
            register16: [
                reg16(0), reg16(1), reg16(2), reg16(3),
                reg16(4), reg16(5), reg16(6), reg16(7),
                reg16(8), reg16(9), reg16(10), reg16(11),
                reg16(12), reg16(13), reg16(14), reg16(15),
            ],
            register32: [
                reg32(0), reg32(1), reg32(2), reg32(3),
                reg32(4), reg32(5), reg32(6), reg32(7),
                reg32(8), reg32(9), reg32(10), reg32(11),
                reg32(12), reg32(13), reg32(14), reg32(15),
            ],
        }
    }

    fn register8_low(&self, i: u8) -> Rc<Operand> {
        self.register8_low[i as usize & 15].clone()
    }

    fn register8_high(&self, i: u8) -> Rc<Operand> {
        self.register8_high[i as usize & 3].clone()
    }

    fn register16(&self, i: u8) -> Rc<Operand> {
        self.register16[i as usize & 15].clone()
    }

    fn register32(&self, i: u8) -> Rc<Operand> {
        self.register32[i as usize & 15].clone()
    }
}

struct InstructionOpsState<'a, 'exec: 'a, Va: VirtualAddress> {
    address: Va,
    data: &'a [u8],
    prefixes: InstructionPrefixes,
    len: u8,
    ctx: &'exec OperandContext,
    register_cache: &'a RegisterCache,
    out: &'a mut OperationVec,
    /// Initialize to false.
    /// If the decoding function returns Err(Failed) with this set to false,
    /// generates UnknownOpcode (possible), if true, InternalDecodeError
    /// (Ideally never)
    error_is_decode_error: bool,
    is_ext: bool,
}

fn instruction_operations32(
    address: VirtualAddress32,
    data: &[u8],
    ctx: &OperandContext,
    out: &mut OperationVec,
    register_cache: &RegisterCache,
) -> Result<(), Error> {
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
        if s.error_is_decode_error {
            Err(Error::InternalDecodeError)
        } else {
            Err(Error::UnknownOpcode(full_data.into()))
        }
    } else {
        Ok(())
    }
}

fn instruction_operations32_main(
    s: &mut InstructionOpsState<VirtualAddress32>,
) -> Result<(), Failed> {
    use self::operation_helpers::*;
    use crate::operand::operand_helpers::*;

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
            0x88 | 0x89 | 0x8a | 0x8b =>
        {
            // Avoid ridiculous generic binary bloat
            let ops: for<'x> fn(_, _, &'x mut _, bool) = match first_byte {
                0x00 | 0x01 | 0x02 | 0x03 | 0x04 | 0x05 => add_ops,
                0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x0d => or_ops,
                0x10 | 0x11 | 0x12 | 0x13 | 0x14 | 0x15 => adc_ops,
                0x18 | 0x19 | 0x1a | 0x1b | 0x1c | 0x1d => sbb_ops,
                0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 => and_ops,
                0x28 | 0x29 | 0x2a | 0x2b | 0x2c | 0x2d => sub_ops,
                0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35 => xor_ops,
                0x88 | 0x89 | 0x8a | 0x8b | _ => mov_ops,
            };
            let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
            if eax_imm_arith {
                s.eax_imm_arith(ops)
            } else {
                s.generic_arith_op(ops)
            }
        }

        0x38 | 0x39 | 0x3a | 0x3b => s.generic_cmp_op(cmp_ops),
        0x3c | 0x3d => s.eax_imm_cmp(cmp_ops),
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
        0x84 | 0x85 => s.generic_cmp_op(test_ops),
        0x86 | 0x87 => s.xchg(),
        0x8d => s.lea(),
        0x90 => Ok(()),
        // Cwde
        0x98 => {
            let eax = s.dest_and_op_register(0);
            let signed_max = ctx.const_7fff();
            let cond = operand_gt(eax.op.clone(), signed_max);
            let neg_sign_extend = operand_or(eax.op.clone(), ctx.const_ffff0000());
            let neg_sign_extend_op = Operation::Move(
                dest_operand_reg64(0),
                neg_sign_extend,
                Some(cond),
            );
            s.output(and(eax, ctx.const_ffff(), false));
            s.output(neg_sign_extend_op);
            Ok(())
        }
        // Cdq
        0x99 => {
            let eax = ctx.register(0);
            let signed_max = ctx.const_7fffffff();
            let cond = operand_gt(eax, signed_max);
            let neg_sign_extend_op = Operation::Move(
                dest_operand_reg64(2),
                ctx.const_ffffffff(),
                Some(cond),
            );
            s.output(mov_to_reg(2, ctx.const_0()));
            s.output(neg_sign_extend_op);
            Ok(())
        },
        0xa0 | 0xa1 | 0xa2 | 0xa3 => s.move_mem_eax(),
        // rep mov
        0xa4 | 0xa5 => {
            s.output(Operation::Special(s.data.into()));
            Ok(())
        }
        0xa8 | 0xa9 => s.eax_imm_cmp(test_ops),
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
            s.generic_arith_with_imm_op(&MOV_OPS, match first_byte {
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
        0x12c => s.cvttss2si(),
        // rdtsc
        0x131 => {
            s.output(mov_to_reg(0, s.ctx.undefined_rc()));
            s.output(mov_to_reg(2, s.ctx.undefined_rc()));
            Ok(())
        }
        0x140 | 0x141 | 0x142 | 0x143 | 0x144 | 0x145 | 0x146 | 0x147 |
            0x148 | 0x149 | 0x14a | 0x14b | 0x14c | 0x14d | 0x14e | 0x14f => s.cmov(),
        0x157 => s.xorps(),
        0x15b => s.cvtdq2ps(),
        0x160 => s.punpcklbw(),
        0x16e => s.mov_sse_6e(),
        0x180 | 0x181 | 0x182 | 0x183 | 0x184 | 0x185 | 0x186 | 0x187 |
            0x188 | 0x189 | 0x18a | 0x18b | 0x18c | 0x18d | 0x18e | 0x18f =>
        {
            s.conditional_jmp(s.mem16_32())
        }
        0x190 | 0x191 | 0x192 | 0x193 | 0x194 | 0x195 | 0x196 | 0x197 |
            0x198 | 0x199 | 0x19a | 0x19b | 0x19c | 0x19d | 0x19e | 0x19f => s.conditional_set(),
        0x1a3 => s.bit_test(false),
        0x1a4 => s.shld_imm(),
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
            s.output(mov(s.rm_to_dest_operand(&rm), ctx.undefined_rc()));
            s.output(mov_to_reg(0, ctx.undefined_rc()));
            Ok(())
        }
        0x1b3 => s.btr(false),
        0x1b6 | 0x1b7 => s.movzx(),
        0x1ba => s.various_0f_ba(),
        0x1be => s.movsx(MemAccessSize::Mem8),
        0x1bf => s.movsx(MemAccessSize::Mem16),
        0x1c0 | 0x1c1 => s.xadd(),
        0x1d3 => s.packed_shift_right(),
        0x1d6 => s.mov_sse_d6(),
        0x1f3 => s.packed_shift_left(),
        _ => Err(s.unknown_opcode()),
    }
}

fn instruction_operations64(
    address: VirtualAddress64,
    data: &[u8],
    ctx: &OperandContext,
    out: &mut OperationVec,
    register_cache: &RegisterCache,
) -> Result<(), Error> {
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
        if s.error_is_decode_error {
            Err(Error::InternalDecodeError)
        } else {
            Err(Error::UnknownOpcode(full_data.into()))
        }
    } else {
        Ok(())
    }
}

fn instruction_operations64_main(
    s: &mut InstructionOpsState<VirtualAddress64>,
) -> Result<(), Failed> {
    use self::operation_helpers::*;
    use crate::operand::operand_helpers::*;

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
            0x88 | 0x89 | 0x8a | 0x8b =>
        {
            // Avoid ridiculous generic binary bloat
            let ops: for<'x> fn(_, _, &'x mut _, bool) = match first_byte {
                0x00 | 0x01 | 0x02 | 0x03 | 0x04 | 0x05 => add_ops,
                0x08 | 0x09 | 0x0a | 0x0b | 0x0c | 0x0d => or_ops,
                0x10 | 0x11 | 0x12 | 0x13 | 0x14 | 0x15 => adc_ops,
                0x18 | 0x19 | 0x1a | 0x1b | 0x1c | 0x1d => sbb_ops,
                0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 => and_ops,
                0x28 | 0x29 | 0x2a | 0x2b | 0x2c | 0x2d => sub_ops,
                0x30 | 0x31 | 0x32 | 0x33 | 0x34 | 0x35 => xor_ops,
                0x88 | 0x89 | 0x8a | 0x8b | _ => mov_ops,
            };
                let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
                if eax_imm_arith {
                    s.eax_imm_arith(ops)
                } else {
                    s.generic_arith_op(ops)
                }
            }

        0x38 | 0x39 | 0x3a | 0x3b => s.generic_cmp_op(cmp_ops),
        0x3c | 0x3d => s.eax_imm_cmp(cmp_ops),
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
        0x84 | 0x85 => s.generic_cmp_op(test_ops),
        0x86 | 0x87 => s.xchg(),
        0x8d => s.lea(),
        0x90 => Ok(()),
        0x98 => {
            let eax = s.dest_and_op_register(0);
            if s.prefixes.rex_prefix & 0x8 == 0 {
                let signed_max = ctx.const_7fff();
                let cond = operand_gt(eax.op.clone(), signed_max);
                let neg_sign_extend = operand_or(eax.op.clone(), ctx.const_ffff0000());
                let neg_sign_extend_op =
                    Operation::Move(dest_operand_reg64(0), neg_sign_extend, Some(cond));
                s.output(and(eax, ctx.const_ffff(), false));
                s.output(neg_sign_extend_op);
            } else {
                let signed_max = ctx.const_7fffffff();
                let cond = operand_gt(eax.op.clone(), signed_max);
                let neg_sign_extend =
                    operand_or64(eax.op.clone(), ctx.constant(0xffff_ffff_0000_0000));
                let neg_sign_extend_op =
                    Operation::Move(dest_operand_reg64(0), neg_sign_extend, Some(cond));
                s.output(and(eax, ctx.const_ffffffff(), false));
                s.output(neg_sign_extend_op);
            }
            Ok(())
        }
        // Cdq
        0x99 => {
            if s.prefixes.rex_prefix & 0x8 == 0 {
                let eax = ctx.register(0);
                let signed_max = ctx.const_7fffffff();
                let cond = operand_gt(eax, signed_max);
                let neg_sign_extend_op = Operation::Move(
                    dest_operand_reg64(2),
                    ctx.const_ffffffff(),
                    Some(cond),
                );
                s.output(mov_to_reg(2, ctx.const_0()));
                s.output(neg_sign_extend_op);
            } else {
                let rax = ctx.register(0);
                let signed_max = ctx.constant(0x7fff_ffff_ffff_ffff);
                let cond = operand_gt64(rax, signed_max);
                let neg_sign_extend_op = Operation::Move(
                    dest_operand_reg64(2),
                    ctx.constant(!0),
                    Some(cond),
                );
                s.output(mov_to_reg(2, ctx.const_0()));
                s.output(neg_sign_extend_op);
            }
            Ok(())
        },
        0xa0 | 0xa1 | 0xa2 | 0xa3 => s.move_mem_eax(),
        // rep mov
        0xa4 | 0xa5 => {
            s.output(Operation::Special(s.data.into()));
            Ok(())
        }
        0xa8 | 0xa9 => s.eax_imm_cmp(test_ops),
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
            s.generic_arith_with_imm_op(&MOV_OPS, match first_byte {
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
        0x12c => s.cvttss2si(),
        // rdtsc
        0x131 => {
            s.output(mov_to_reg(0, s.ctx.undefined_rc()));
            s.output(mov_to_reg(2, s.ctx.undefined_rc()));
            Ok(())
        }
        0x140 | 0x141 | 0x142 | 0x143 | 0x144 | 0x145 | 0x146 | 0x147 |
            0x148 | 0x149 | 0x14a | 0x14b | 0x14c | 0x14d | 0x14e | 0x14f => s.cmov(),
        0x157 => s.xorps(),
        0x15b => s.cvtdq2ps(),
        0x160 => s.punpcklbw(),
        0x16e => s.mov_sse_6e(),
        0x180 | 0x181 | 0x182 | 0x183 | 0x184 | 0x185 | 0x186 | 0x187 |
            0x188 | 0x189 | 0x18a | 0x18b | 0x18c | 0x18d | 0x18e | 0x18f =>
        {
            s.conditional_jmp(s.mem16_32())
        }
        0x190 | 0x191 | 0x192 | 0x193 | 0x194 | 0x195 | 0x196 | 0x197 |
            0x198 | 0x199 | 0x19a | 0x19b | 0x19c | 0x19d | 0x19e | 0x19f => s.conditional_set(),
        0x1a3 => s.bit_test(false),
        0x1a4 => s.shld_imm(),
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
            s.output(mov(s.rm_to_dest_operand(&rm), ctx.undefined_rc()));
            s.output(mov_to_reg(0, ctx.undefined_rc()));
            Ok(())
        }
        0x1b3 => s.btr(false),
        0x1b6 | 0x1b7 => s.movzx(),
        0x1ba => s.various_0f_ba(),
        0x1be => s.movsx(MemAccessSize::Mem8),
        0x1bf => s.movsx(MemAccessSize::Mem16),
        0x1c0 | 0x1c1 => s.xadd(),
        0x1d3 => s.packed_shift_right(),
        0x1d6 => s.mov_sse_d6(),
        0x1f3 => s.packed_shift_left(),
        _ => Err(s.unknown_opcode()),
    }
}

fn x87_variant(ctx: &OperandContext, op: Rc<Operand>, offset: i8) -> Rc<Operand> {
    match op.ty {
        OperandType::Register(Register(r)) => ctx.register_fpu((r as i8 + offset) as u8 & 7),
        _ => op,
    }
}

impl<'a, 'exec: 'a, Va: VirtualAddress> InstructionOpsState<'a, 'exec, Va> {
    pub fn len(&self) -> usize {
        self.len as usize
    }

    /// Separate function mainly since SmallVec::push and SmallVec::reserve
    /// are marked as #[inline], so calling them directly at 100+ different
    /// places, often also used by rare instruction ends up being really
    /// wasteful wrt. binary size.
    #[inline(never)]
    fn output(&mut self, op: Operation) {
        self.out.push(op);
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
            match self.prefixes.rex_prefix & 0x8 != 0 {
                true => MemAccessSize::Mem64,
                false => match self.has_prefix(0x66) {
                    true => MemAccessSize::Mem16,
                    false => MemAccessSize::Mem32,
                },
            }
        }
    }

    fn word_add(&self, left: Rc<Operand>, right: Rc<Operand>) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;
        if Va::SIZE == 4 {
            operand_add(left, right)
        } else {
            operand_add64(left, right)
        }
    }

    fn dest_and_op_register(&self, register: u8) -> DestAndOperand {
        DestAndOperand {
            op: self.ctx.register(register),
            dest: DestOperand::Register64(Register(register)),
        }
    }

    fn reg_variable_size(&self, register: Register, op_size: MemAccessSize) -> Rc<Operand> {
        if register.0 >= 4 && self.prefixes.rex_prefix == 0 && op_size == MemAccessSize::Mem8 {
            self.register_cache.register8_high(register.0 - 4)
        } else {
            match op_size {
                MemAccessSize::Mem8 => self.register_cache.register8_low(register.0),
                MemAccessSize::Mem16 => self.register_cache.register16(register.0),
                MemAccessSize::Mem32 => self.register_cache.register32(register.0),
                MemAccessSize::Mem64 => self.ctx.register(register.0),
            }
        }
    }

    fn r_to_operand(&self, r: ModRm_R) -> Rc<Operand> {
        match r.1 {
            RegisterSize::Low8 => self.register_cache.register8_low(r.0),
            RegisterSize::High8 => self.register_cache.register8_high(r.0),
            RegisterSize::R16 => self.register_cache.register16(r.0),
            RegisterSize::R32 => self.register_cache.register32(r.0),
            RegisterSize::R64 => self.ctx.register(r.0),
        }
    }

    fn r_to_operand_xmm(&self, r: ModRm_R, i: u8) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;
        operand_xmm(r.0, i)
    }

    /// Returns a structure containing both DestOperand and Rc<Operand>
    /// variations, not useful with ModRm_R, but ModRm_Rm avoids
    /// recalculating address twice with this.
    fn r_to_dest_and_operand(&self, r: ModRm_R) -> DestAndOperand {
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

    fn r_to_dest_and_operand_xmm(&self, r: ModRm_R, i: u8) -> DestAndOperand {
        DestAndOperand {
            op: self.r_to_operand_xmm(r, i),
            dest: r.dest_operand_xmm(i),
        }
    }

    /// Assumes rm.is_memory() == true.
    /// Returns simplified values.
    fn rm_address_operand(&self, rm: &ModRm_Rm) -> Rc<Operand> {
        use crate::operand_helpers::*;
        let base_offset = if rm.constant_base() {
            if Va::SIZE == 4 {
                self.ctx.constant(rm.constant as u64)
            } else {
                let addr = self.address.as_u64()
                    .wrapping_add(self.len() as u64)
                    .wrapping_add(rm.constant as i32 as i64 as u64);
                self.ctx.constant(addr)
            }
        } else {
            let base = self.ctx.register(rm.base);
            if rm.constant == 0 {
                base
            } else {
                self.word_add(base, self.ctx.constant(rm.constant as i32 as i64 as u64))
            }
        };
        let with_index = match rm.index_mul {
            0 => base_offset,
            1 => self.word_add(base_offset, self.ctx.register(rm.index)),
            x => {
                let multiplier = self.ctx.constant(x as u64);
                self.word_add(
                    base_offset,
                    operand_mul64(self.ctx.register(rm.index), multiplier),
                )
            }
        };
        Operand::simplified(with_index)
    }

    fn rm_to_dest_and_operand(&self, rm: &ModRm_Rm) -> DestAndOperand {
        use crate::operand_helpers::*;
        if rm.is_memory() {
            let address = self.rm_address_operand(rm);
            let size = rm.size.to_mem_access_size();
            DestAndOperand {
                op: mem_variable_rc(size, address.clone()),
                dest: DestOperand::Memory(MemAccess {
                    size,
                    address: address,
                })
            }
        } else {
            self.r_to_dest_and_operand(ModRm_R(rm.base, rm.size))
        }
    }

    fn rm_to_dest_operand(&self, rm: &ModRm_Rm) -> DestOperand {
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

    fn rm_to_dest_operand_xmm(&self, rm: &ModRm_Rm, i: u8) -> DestOperand {
        if rm.is_memory() {
            // Would be nice to just add the i * 4 offset on rm_address_operand,
            // but `rm.constant += i * 4` has issues if the constant overflows
            let mut address = self.rm_address_operand(&rm);
            if i != 0 {
                address = Operand::simplified(
                    self.word_add(address, self.ctx.constant(i as u64 * 4))
                );
            }
            DestOperand::Memory(MemAccess {
                size: MemAccessSize::Mem32,
                address,
            })
        } else {
            DestOperand::Xmm(rm.base, i)
        }
    }

    fn rm_to_operand_xmm(&self, rm: &ModRm_Rm, i: u8) -> Rc<Operand> {
        use crate::operand_helpers::*;
        if rm.is_memory() {
            // Would be nice to just add the i * 4 offset on rm_address_operand,
            // but `rm.constant += i * 4` has issues if the constant overflows
            let mut address = self.rm_address_operand(&rm);
            if i != 0 {
                address = Operand::simplified(
                    self.word_add(address, self.ctx.constant(i as u64 * 4))
                );
            }
            mem_variable_rc(MemAccessSize::Mem32, address)
        } else {
            operand_xmm(rm.base, i)
        }
    }

    fn rm_to_operand(&self, rm: &ModRm_Rm) -> Rc<Operand> {
        use crate::operand_helpers::*;
        if rm.is_memory() {
            let address = self.rm_address_operand(rm);
            mem_variable_rc(rm.size.to_mem_access_size(), address)
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
        let register = if self.prefixes.rex_prefix & 0x4 == 0 {
            (modrm >> 3) & 0x7
        } else {
            8 + ((modrm >> 3) & 0x7)
        };
        let rm_ext = self.prefixes.rex_prefix & 0x1 != 0;
        let r_high = register >= 4 &&
            self.prefixes.rex_prefix == 0 &&
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
                        self.prefixes.rex_prefix == 0 &&
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
    ) -> Result<(ModRm_Rm, ModRm_R, Rc<Operand>), Failed> {
        let (rm, r, offset) = self.parse_modrm_inner(op_size)?;
        let imm = self.read_variable_size_32(offset, imm_size)?;
        let imm = match imm_size {
            x if x == op_size => imm,
            MemAccessSize::Mem8 => imm as i8 as u64,
            MemAccessSize::Mem16 => imm as i16 as u64,
            MemAccessSize::Mem32 => imm as i32 as u64,
            MemAccessSize::Mem64 => imm,
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
        let base_ext = self.prefixes.rex_prefix & 0x1 != 0;
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
                result.base = u8::max_value();
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
        let index_reg = if self.prefixes.rex_prefix & 0x2 == 0 {
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
        use self::operation_helpers::*;
        let byte = self.read_u8(0)?;
        let is_inc = byte < 0x48;
        let reg_id = byte & 0x7;
        let op_size = self.mem16_32();
        let reg = self.reg_variable_size(Register(reg_id), op_size);
        let dest = DestAndOperand {
            op: reg.clone(),
            dest: DestOperand::reg_variable_size(reg_id, op_size),
        };
        let is_64 = Va::SIZE == 8;
        self.output(match is_inc {
            true => add(dest, self.ctx.const_1(), is_64),
            false => sub(dest, self.ctx.const_1(), is_64),
        });
        self.output(
            make_arith_operation(
                DestOperand::Flag(Flag::Zero),
                ArithOpType::Equal,
                reg.clone(),
                self.ctx.const_0(),
                is_64,
            )
        );
        if is_64 {
            self.output(make_arith_operation(
                DestOperand::Flag(Flag::Sign),
                ArithOpType::GreaterThan,
                reg.clone(),
                self.ctx.constant(0x7fff_ffff_ffff_ffff),
                true,
            ));
        } else {
            self.output(make_arith_operation(
                DestOperand::Flag(Flag::Sign),
                ArithOpType::GreaterThan,
                reg.clone(),
                self.ctx.const_7fffffff(),
                false,
            ));
        }
        let eq_value = match (is_inc, is_64) {
            (true, false) => self.ctx.constant(0x8000_0000),
            (false, false) => self.ctx.const_7fffffff(),
            (true, true) => self.ctx.constant(0x8000_0000_0000_0000),
            (false, true) => self.ctx.constant(0x7fff_ffff_ffff_ffff),
        };
        self.output(make_arith_operation(
            DestOperand::Flag(Flag::Overflow),
            ArithOpType::Equal,
            reg,
            eq_value,
            is_64,
        ));
        Ok(())
    }

    fn flag_set(&mut self, flag: Flag, value: bool) -> Result<(), Failed> {
        self.output(Operation::Move(DestOperand::Flag(flag), self.ctx.constant(value as u64), None));
        Ok(())
    }

    fn condition(&mut self) -> Result<Rc<Operand>, Failed> {
        use crate::operand::operand_helpers::*;
        let ctx = self.ctx;
        let cond_id = self.read_u8(0)? & 0xf;
        let zero = ctx.const_0();
        let zero2 = zero.clone();
        let cond = match cond_id >> 1 {
            // jo, jno
            0x0 => operand_eq(ctx.flag_o(), zero),
            // jb, jnb (jae) (jump if carry)
            0x1 => operand_eq(ctx.flag_c(), zero),
            // je, jne
            0x2 => operand_eq(ctx.flag_z(), zero),
            // jbe, jnbe (ja)
            0x3 => operand_and(
                operand_eq(ctx.flag_z(), zero.clone()),
                operand_eq(ctx.flag_c(), zero),
            ),
            // js, jns
            0x4 => operand_eq(ctx.flag_s(), zero),
            // jpe, jpo
            0x5 => operand_eq(ctx.flag_p(), zero),
            // jl, jnl (jge)
            0x6 => operand_eq(ctx.flag_s(), ctx.flag_o()),
            // jle, jnle (jg)
            0x7 => operand_and(
                operand_eq(ctx.flag_z(), zero),
                operand_eq(ctx.flag_s(), ctx.flag_o()),
            ),
            _ => unreachable!(),
        };
        if cond_id & 1 == 0 {
            Ok(operand_eq(cond, zero2))
        } else {
            Ok(cond)
        }
    }

    fn cmov(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let condition = Some(self.condition()?);
        self.output(Operation::Move(
            r.dest_operand(),
            self.rm_to_operand(&rm),
            condition,
        ));
        Ok(())
    }

    fn conditional_set(&mut self) -> Result<(), Failed> {
        let condition = self.condition()?;
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem8)?;
        self.output(Operation::Move(self.rm_to_dest_operand(&rm), condition, None));
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
        use self::operation_helpers::*;
        self.xchg()?;
        self.generic_arith_op(add_ops)?;
        Ok(())
    }

    fn move_mem_eax(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let constant = self.read_variable_size_64(1, op_size)?;
        let constant = self.ctx.constant(constant);
        let eax_left = self.read_u8(0)? & 0x2 == 0;
        self.output(match eax_left {
            true => mov_to_reg(0, mem_variable_rc(op_size, constant)),
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
        let register = if self.prefixes.rex_prefix & 0x1 != 0 {
            8 + (self.read_u8(0)? & 0x7)
        } else {
            self.read_u8(0)? & 0x7
        };
        let constant = self.read_variable_size_64(1, op_size)?;
        self.output(mov_to_reg_variable_size(op_size, register, self.ctx.constant(constant)));
        Ok(())
    }

    fn eax_imm_arith(
        &mut self,
        make_arith: fn(DestAndOperand, Rc<Operand>, &mut OperationVec, bool),
    ) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let dest = DestAndOperand {
            op: self.reg_variable_size(Register(0), op_size),
            dest: DestOperand::reg_variable_size(0, op_size),
        };
        let imm = self.read_variable_size_32(1, op_size)?;
        let val = self.ctx.constant(imm);
        let is_64 = Va::SIZE == 8;
        make_arith(dest, val, &mut self.out, is_64);
        Ok(())
    }

    /// Also mov even though I'm not sure if I should count it as no-op arith or a separate
    /// thing.
    fn generic_arith_op(
        &mut self,
        make_arith: fn(DestAndOperand, Rc<Operand>, &mut OperationVec, bool),
    ) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r) = self.parse_modrm(op_size)?;
        let r = r.to_rm();
        let rm_left = self.read_u8(0)? & 0x3 < 2;
        let (left, right) = match rm_left {
            true => (&rm, &r),
            false => (&r, &rm),
        };
        let left = self.rm_to_dest_and_operand(left);
        let right = self.rm_to_operand(right);
        let is_64 = Va::SIZE == 8;
        make_arith(left, right, &mut self.out, is_64);
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
        if op_size == MemAccessSize::Mem8 && rm.index_mul == 255 && rm.base > 4 {
            rm.base -= 4;
            rm.size = RegisterSize::High8;
        } else {
            rm.size = RegisterSize::from_mem_access_size(op_size);
        }

        self.output(Operation::Move(
            r.dest_operand(),
            Operand::new_not_simplified_rc(
                OperandType::SignExtend(self.rm_to_operand(&rm), op_size, dest_size),
            ),
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
        if op_size == MemAccessSize::Mem8 && rm.index_mul == 255 && rm.base > 4 {
            rm.base -= 4;
            rm.size = RegisterSize::High8;
        } else {
            rm.size = RegisterSize::from_mem_access_size(op_size);
        }
        if is_rm_short_r_register(&rm, r) {
            let r = self.r_to_dest_and_operand(r);
            self.output(match op_size {
                MemAccessSize::Mem8 => and(r, self.ctx.const_ff(), false),
                MemAccessSize::Mem16 => and(r, self.ctx.const_ffff(), false),
                _ => unreachable!(),
            });
        } else {
            let rm_oper = self.rm_to_operand(&rm);
            if rm.is_memory() {
                self.output(mov(r.dest_operand(), rm_oper));
            } else {
                self.output(mov(r.dest_operand(), self.ctx.const_0()));
                self.output(mov(r.dest_operand(), rm_oper));
            }
        }
        Ok(())
    }

    fn various_f7(&mut self) -> Result<(), Failed> {
        use crate::operand::operand_helpers::*;
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        let (rm, _) = self.parse_modrm(op_size)?;
        let rm = self.rm_to_dest_and_operand(&rm);
        let is_64 = Va::SIZE == 8;
        match variant {
            0 | 1 => return self.generic_arith_with_imm_op(&TEST_OPS, op_size),
            2 => {
                // Not
                let constant = self.ctx.constant(!0u64);
                self.output(
                    make_arith_operation(rm.dest, ArithOpType::Xor, rm.op, constant, is_64)
                );
            }
            3 => {
                // Neg
                self.output(make_arith_operation(
                    rm.dest,
                    ArithOpType::Sub,
                    self.ctx.const_0(),
                    rm.op,
                    is_64,
                ));
            }
            4 | 5 => {
                // TODO signed mul
                // No way to represent rdx = imul_128(rax, rm) >> 64,
                // Just set to undefined for now.
                // Could alternatively either Special or add Arithmetic64High.
                let eax = self.reg_variable_size(Register(0), op_size);
                let edx = self.reg_variable_size(Register(2), op_size);
                let multiply = operand_mul(eax.clone(), rm.op);
                if op_size == MemAccessSize::Mem64 {
                    self.output(Operation::MoveSet(vec![
                        (DestOperand::from_oper(&edx), self.ctx.undefined_rc()),
                        (DestOperand::from_oper(&eax), multiply),
                    ]));
                } else {
                    let size = self.ctx.constant(op_size.bits() as u64);
                    self.output(Operation::MoveSet(vec![
                        (DestOperand::from_oper(&edx), operand_rsh64(multiply.clone(), size)),
                        (DestOperand::from_oper(&eax), multiply),
                    ]));
                }
            },
            6 => {
                // edx = edx:eax % rm, eax = edx:eax / rm
                let eax = self.reg_variable_size(Register(0), op_size);
                let edx = self.reg_variable_size(Register(2), op_size);
                let size = self.ctx.constant(op_size.bits() as u64);
                let div;
                let modulo;
                if op_size == MemAccessSize::Mem64 {
                    // Difficult to do unless rdx is known to be 0
                    div = self.ctx.undefined_rc();
                    modulo = self.ctx.undefined_rc();
                } else {
                    let pair = operand_or64(
                        operand_lsh64(
                            edx.clone(),
                            size,
                        ),
                        eax.clone(),
                    );
                    div = operand_div64(pair.clone(), rm.op.clone());
                    modulo = operand_mod64(pair, rm.op);
                }
                self.output(Operation::MoveSet(vec![
                    (DestOperand::from_oper(&edx), modulo),
                    (DestOperand::from_oper(&eax), div),
                ]));
            }
            _ => return Err(self.unknown_opcode()),
        }
        Ok(())
    }

    fn various_0f_ba(&mut self) -> Result<(), Failed> {
        let variant = (self.read_u8(1)? >> 3) & 0x7;
        match variant {
            4 => self.bit_test(true),
            6 => self.btr(true),
            _ => Err(self.unknown_opcode()),
        }
    }

    fn bit_test(&mut self, imm8: bool) -> Result<(), Failed> {
        use crate::operand::operand_helpers::*;
        let op_size = self.mem16_32();
        let (dest, index) = if imm8 {
            let (rm, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
            (rm, imm)
        } else {
            let (rm, r) = self.parse_modrm(op_size)?;
            (rm, self.r_to_operand(r))
        };
        // Move bit at index to carry
        // c = (dest >> index) & 1
        self.output(Operation::Move(
            DestOperand::Flag(Flag::Carry),
            operand_and64(
                operand_rsh64(
                    self.rm_to_operand(&dest),
                    index,
                ),
                self.ctx.const_1(),
            ),
            None,
        ));
        Ok(())
    }

    fn btr(&mut self, imm8: bool) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let op_size = self.mem16_32();
        let (dest, index) = if imm8 {
            let (rm, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
            (rm, imm)
        } else {
            let (rm, r) = self.parse_modrm(op_size)?;
            (rm, self.r_to_operand(r))
        };
        // Move bit at index to carry, clear it
        // c = (dest >> index) & 1; dest &= !(1 << index)
        let dest = self.rm_to_dest_and_operand(&dest);
        self.output(Operation::Move(
            DestOperand::Flag(Flag::Carry),
            operand_and64(
                operand_rsh64(
                    dest.op.clone(),
                    index.clone(),
                ),
                self.ctx.const_1(),
            ),
            None,
        ));
        self.output(mov(
            dest.dest,
            operand_and64(
                dest.op,
                operand_not64(
                    operand_lsh64(
                        self.ctx.const_1(),
                        index,
                    ),
                ),
            ),
        ));
        Ok(())
    }

    fn xorps(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        for i in 0..4 {
            let dest = self.r_to_dest_and_operand_xmm(dest, i);
            let rhs = self.rm_to_operand_xmm(&rm, i);
            self.output(xor(dest, rhs, false));
        }
        Ok(())
    }

    fn cvttss2si(&mut self) -> Result<(), Failed> {
        if !self.has_prefix(0xf3) {
            return Err(self.unknown_opcode());
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let op = make_arith_operation(
            r.dest_operand(),
            ArithOpType::FloatToInt,
            self.rm_to_operand_xmm(&rm, 0),
            self.ctx.const_0(),
            false,
        );
        self.output(op);
        Ok(())
    }

    fn cvtdq2ps(&mut self) -> Result<(), Failed> {
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        for i in 0..4 {
            let op = make_arith_operation(
                r.dest_operand_xmm(i),
                ArithOpType::IntToFloat,
                self.rm_to_operand_xmm(&rm, i),
                self.ctx.const_0(),
                false,
            );
            self.output(op);
        }
        Ok(())
    }

    fn punpcklbw(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand_helpers::*;
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
        let c_8 = self.ctx.const_8();
        let c_10 = self.ctx.constant(0x10);
        let c_ff = self.ctx.const_ff();
        let c_ff00 = self.ctx.constant(0xff00);
        let c_ff_0000 = self.ctx.constant(0xff_0000);
        let c_ff00_0000 = self.ctx.constant(0xff00_0000);
        for &i in &[1, 0] {
            let out0 = r.dest_operand_xmm(i * 2);
            let out1 = r.dest_operand_xmm(i * 2 + 1);
            let in_r = self.r_to_operand_xmm(r, i);
            let in_rm = self.rm_to_operand_xmm(&rm, i);
            self.output(mov(out1, operand_or(
                operand_or(
                    operand_rsh(
                        operand_and(
                            in_r.clone(),
                            c_ff_0000.clone(),
                        ),
                        c_10.clone(),
                    ),
                    operand_rsh(
                        operand_and(
                            in_rm.clone(),
                            c_ff_0000.clone(),
                        ),
                        c_8.clone(),
                    ),
                ),
                operand_or(
                    operand_rsh(
                        operand_and(
                            in_r.clone(),
                            c_ff00_0000.clone(),
                        ),
                        c_8.clone(),
                    ),
                    operand_and(
                        in_rm.clone(),
                        c_ff00_0000.clone(),
                    ),
                ),
            )));
            self.output(mov(out0, operand_or(
                operand_or(
                    operand_and(
                        in_r.clone(),
                        c_ff.clone(),
                    ),
                    operand_lsh(
                        operand_and(
                            in_rm.clone(),
                            c_ff.clone(),
                        ),
                        c_8.clone(),
                    ),
                ),
                operand_or(
                    operand_lsh(
                        operand_and(
                            in_r,
                            c_ff00.clone(),
                        ),
                        c_8.clone(),
                    ),
                    operand_lsh(
                        operand_and(
                            in_rm,
                            c_ff00.clone(),
                        ),
                        c_10.clone(),
                    ),
                ),
            )));
        }
        Ok(())
    }

    fn mov_sse_6e(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let op_size = self.mem16_32();
        let (rm, r) = self.parse_modrm(op_size)?;
        let rm_op = self.rm_to_operand(&rm);
        self.output(mov(r.dest_operand_xmm(0), rm_op.clone()));
        if op_size == MemAccessSize::Mem64 {
            let rm_high = operand_rsh64(rm_op, self.ctx.const_20());
            self.output(mov(r.dest_operand_xmm(1), rm_high));
        } else {
            self.output(mov(r.dest_operand_xmm(1), self.ctx.const_0()));
        }
        self.output(mov(r.dest_operand_xmm(2), self.ctx.const_0()));
        self.output(mov(r.dest_operand_xmm(3), self.ctx.const_0()));
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
        let (src, dest) = match self.read_u8(0)? {
            0x10 | 0x28 | 0x7e | 0x13 => (&rm, &r),
            _ => (&r, &rm),
        };
        let len = match self.read_u8(0)? {
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
                (true, 4) | (_, 8) => 2,
                (false, 4) => 1,
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
        self.output(mov(self.rm_to_dest_operand_xmm(&rm, 0), self.r_to_operand_xmm(src, 0)));
        self.output(mov(self.rm_to_dest_operand_xmm(&rm, 1), self.r_to_operand_xmm(src, 1)));
        if !rm.is_memory() {
            self.output(mov(self.rm_to_dest_operand_xmm(&rm, 2), self.ctx.const_0()));
            self.output(mov(self.rm_to_dest_operand_xmm(&rm, 3), self.ctx.const_0()));
        }
        Ok(())
    }

    fn packed_shift_left(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let dest_op = self.r_to_operand(dest);
        // dest.1 = (dest.1 << rm.0) | (dest.0 >> (32 - rm.0))
        // shl dest.0, rm.0
        // dest.3 = (dest.3 << rm.0) | (dest.2 >> (32 - rm.0))
        // shl dest.2, rm.0
        // Zero everything if rm.1 is set
        let rm_0 = self.rm_to_operand_xmm(&rm, 0);
        self.output({
            let (low, high) = Operand::to_xmm_64(&dest_op, 0);
            make_arith_operation(
                dest.dest_operand_xmm(1),
                ArithOpType::Or,
                operand_lsh(high, rm_0.clone()),
                operand_rsh(low, operand_sub(self.ctx.const_20(), rm_0.clone())),
                false,
            )
        });
        self.output({
            let low = self.r_to_dest_and_operand_xmm(dest, 0);
            lsh(low, rm_0.clone(), false)
        });
        self.output({
            let (low, high) = Operand::to_xmm_64(&dest_op, 1);
            make_arith_operation(
                dest.dest_operand_xmm(3),
                ArithOpType::Or,
                operand_lsh(high, rm_0.clone()),
                operand_rsh(low, operand_sub(self.ctx.const_20(), rm_0.clone())),
                false,
            )
        });
        self.output({
            let low = self.r_to_dest_and_operand_xmm(dest, 2);
            lsh(low, rm_0, false)
        });
        let rm_1 = self.rm_to_operand_xmm(&rm, 1);
        let high_u32_set = operand_logical_not(operand_eq(rm_1, self.ctx.const_0()));
        for i in 0..4 {
            self.output(Operation::Move(
                dest.dest_operand_xmm(i),
                self.ctx.const_0(),
                Some(high_u32_set.clone()),
            ));
        }
        Ok(())
    }

    fn packed_shift_right(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(self.unknown_opcode());
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let dest_op = self.r_to_operand(dest);
        let rm_0 = self.rm_to_operand_xmm(&rm, 0);
        // dest.0 = (dest.0 >> rm.0) | (dest.1 << (32 - rm.0))
        // shr dest.1, rm.0
        // dest.2 = (dest.2 >> rm.0) | (dest.3 << (32 - rm.0))
        // shr dest.3, rm.0
        // Zero everything if rm.1 is set
        self.output({
            let (low, high) = Operand::to_xmm_64(&dest_op, 0);
            make_arith_operation(
                dest.dest_operand_xmm(0),
                ArithOpType::Or,
                operand_rsh(low, rm_0.clone()),
                operand_lsh(high, operand_sub(self.ctx.const_20(), rm_0.clone())),
                false,
            )
        });
        self.output({
            let high = self.r_to_dest_and_operand_xmm(dest, 1);
            rsh(high, rm_0.clone(), false)
        });
        self.output({
            let (low, high) = Operand::to_xmm_64(&dest_op, 1);
            make_arith_operation(
                dest.dest_operand_xmm(2),
                ArithOpType::Or,
                operand_rsh(low, rm_0.clone()),
                operand_lsh(high, operand_sub(self.ctx.const_20(), rm_0.clone())),
                false,
            )
        });
        self.output({
            let high = self.r_to_dest_and_operand_xmm(dest, 3);
            rsh(high, rm_0, false)
        });
        let rm_1 = self.rm_to_operand_xmm(&rm, 1);
        let high_u32_set = operand_logical_not(operand_eq(rm_1, self.ctx.const_0()));
        for i in 0..4 {
            self.output(Operation::Move(
                dest.dest_operand_xmm(i),
                self.ctx.const_0(),
                Some(high_u32_set.clone()),
            ));
        }
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
        let op = if variant == 5 || variant == 7 {
            make_f32_operation(dest, op_ty, rm, st0)
        } else {
            make_f32_operation(dest, op_ty, st0, rm)
        };
        self.output(op);
        Ok(())
    }

    fn various_d9(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let byte = self.read_u8(1)?;
        let variant = (byte >> 3) & 0x7;
        if variant == 6 && byte == 0xf6 || byte == 0xf7 {
            // Fincstp, fdecstp
            self.output(Operation::Special(self.data.into()));
            return Ok(());
        }
        let (rm_parsed, _) = self.parse_modrm(MemAccessSize::Mem32)?;
        let rm = self.rm_to_dest_and_operand(&rm_parsed);
        match variant {
            // Fld
            0 => {
                self.fpu_push();
                self.output(mov_to_fpu(0, x87_variant(self.ctx, rm.op, 1)));
                Ok(())
            }
            // Fst/Fstp, as long as rm is mem
            2 | 3 => {
                self.output(mov(rm.dest, self.ctx.register_fpu(0)));
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
                        let address = operand_add64(
                            mem.address.clone(),
                            ctx.constant(i * mem_bytes),
                        );
                        mov_to_mem(mem_size, address, ctx.undefined_rc())
                    }));
                }
                Ok(())
            }
            // Fstcw
            7 => {
                if rm_parsed.is_memory() {
                    self.output(mov(rm.dest, self.ctx.undefined_rc()));
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
                self.output(match is_inc {
                    true => add(rm, self.ctx.const_1(), is_64),
                    false => sub(rm, self.ctx.const_1(), is_64),
                });
            }
            2 | 3 => self.output(Operation::Call(rm.op)),
            4 | 5 => self.output(Operation::Jump { condition: self.ctx.const_1(), to: rm.op }),
            6 => {
                let dest = self.dest_and_op_register(4);
                let esp = dest.op.clone();
                if Va::SIZE == 4 {
                    self.output(sub(dest, self.ctx.const_4(), is_64));
                    self.output(mov_to_mem(MemAccessSize::Mem32, esp, rm.op));
                } else {
                    self.output(sub(dest, self.ctx.const_8(), is_64));
                    self.output(mov_to_mem(MemAccessSize::Mem64, esp, rm.op));
                }
            }
            _ => return Err(self.unknown_opcode()),
        }
        Ok(())
    }

    fn bitwise_with_imm_op(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (_, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
        if imm.ty == OperandType::Constant(0) {
            return Ok(());
        }
        let op_gen: &dyn ArithOperationGenerator = match (self.read_u8(1)? >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            7 => &SAR_OPS,
            _ => return Err(self.unknown_opcode()),
        };
        self.generic_arith_with_imm_op(op_gen, MemAccessSize::Mem8)
    }

    fn bitwise_compact_op(&mut self) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _) = self.parse_modrm(op_size)?;
        let shift_count = match self.read_u8(0)? & 2 {
            0 => self.ctx.const_1(),
            _ => self.reg_variable_size(Register(1), operand::MemAccessSize::Mem8),
        };
        let op_gen: &dyn ArithOperationGenerator = match (self.read_u8(1)? >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            7 => &SAR_OPS,
            _ => return Err(self.unknown_opcode()),
        };
        let is_64 = Va::SIZE == 8;
        let rm = self.rm_to_dest_and_operand(&rm);
        op_gen.operation(rm, shift_count, self.out, is_64);
        Ok(())
    }

    fn signed_multiply_rm_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let imm_size = match self.read_u8(0)? & 0x2 {
            2 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let op_size = self.mem16_32();
        let (rm, r, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        let rm = self.rm_to_operand(&rm);
        // TODO flags, imul only sets c and o on overflow
        // TODO Signed mul isn't sensible if it doesn't contain operand size
        // I don't think any of these sizes are correct but w/e
        if Va::SIZE == 4 {
            if op_size != MemAccessSize::Mem32 {
                self.output(mov(r.dest_operand(), operand_signed_mul(rm, imm)));
            } else {
                self.output(mov(r.dest_operand(), operand_mul(rm, imm)));
            }
        } else {
            if op_size != MemAccessSize::Mem64 {
                self.output(mov(r.dest_operand(), operand_signed_mul64(rm, imm)));
            } else {
                self.output(mov(r.dest_operand(), operand_mul64(rm, imm)));
            }
        }
        Ok(())
    }

    fn shld_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let (hi, low, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let imm = Operand::simplified(operand_and(imm, self.ctx.const_1f()));
        let hi = self.rm_to_dest_and_operand(&hi);
        if imm.ty != OperandType::Constant(0) {
            // TODO flags
            self.output(
                mov(
                    hi.dest,
                    operand_or(
                        operand_lsh(
                            hi.op,
                            imm.clone(),
                        ),
                        operand_rsh(
                            self.r_to_operand(low),
                            operand_sub(
                                self.ctx.const_20(),
                                imm,
                            )
                        ),
                    ),
                ),
            );
        }
        Ok(())
    }

    fn shrd_imm(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let (low, hi, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let imm = Operand::simplified(operand_and(imm, self.ctx.const_1f()));
        let low = self.rm_to_dest_and_operand(&low);
        if imm.ty != OperandType::Constant(0) {
            // TODO flags
            self.output(
                mov(
                    low.dest,
                    operand_or(
                        operand_rsh(
                            low.op,
                            imm.clone(),
                        ),
                        operand_lsh(
                            self.r_to_operand(hi),
                            operand_sub(
                                self.ctx.const_20(),
                                imm,
                            )
                        ),
                    ),
                ),
            );
        }
        Ok(())
    }

    fn imul_normal(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        let size = self.mem16_32();
        let (rm, r) = self.parse_modrm(size)?;
        let rm = self.rm_to_operand(&rm);
        // TODO flags, imul only sets c and o on overflow
        // Signed multiplication should be different only when result is being sign extended.
        if size.bits() != Va::SIZE * 8 {
            // TODO Signed mul should actually specify bit size
            self.output(signed_mul(self.r_to_dest_and_operand(r), rm, Va::SIZE == 8));
        } else {
            self.output(mul(self.r_to_dest_and_operand(r), rm, Va::SIZE == 8));
        }
        Ok(())
    }

    fn arith_with_imm_op(&mut self) -> Result<(), Failed> {
        let op_gen: &dyn ArithOperationGenerator = match (self.read_u8(1)? >> 3) & 0x7 {
            0 => &ADD_OPS,
            1 => &OR_OPS,
            2 => &ADC_OPS,
            3 => &SBB_OPS,
            4 => &AND_OPS,
            5 => &SUB_OPS,
            6 => &XOR_OPS,
            7 => &CMP_OPS,
            _ => unreachable!(),
        };
        let imm_size = match self.read_u8(0)? & 0x3 {
            0 | 2 | 3 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        self.generic_arith_with_imm_op(op_gen, imm_size)
    }

    fn generic_arith_with_imm_op(
        &mut self,
        op_gen: &dyn ArithOperationGenerator,
        imm_size: MemAccessSize,
    ) -> Result<(), Failed> {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        let rm = self.rm_to_dest_and_operand(&rm);
        let is_64 = Va::SIZE == 8;
        op_gen.operation(rm, imm, self.out, is_64);
        Ok(())
    }

    fn eax_imm_cmp<F>(&mut self, ops: F) -> Result<(), Failed>
    where F: FnOnce(DestAndOperand, Rc<Operand>, &mut OperationVec, bool),
    {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let eax = DestAndOperand {
            dest: DestOperand::reg_variable_size(0, op_size),
            op: self.reg_variable_size(Register(0), op_size),
        };
        let imm = self.read_variable_size_32(1, op_size)?;
        let val = self.ctx.constant(imm);
        let is_64 = Va::SIZE == 8;
        ops(eax, val, self.out, is_64);
        Ok(())
    }


    fn generic_cmp_op<F>(&mut self, ops: F) -> Result<(), Failed>
    where F: FnOnce(DestAndOperand, Rc<Operand>, &mut OperationVec, bool),
    {
        let op_size = match self.read_u8(0)? & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let rm_left = self.read_u8(0)? & 0x3 < 2;
        let (rm, r) = self.parse_modrm(op_size)?;
        let r = r.to_rm();
        let (left, right) = match rm_left {
            true => (&rm, &r),
            false => (&r, &rm),
        };
        let left = self.rm_to_dest_and_operand(left);
        let right = self.rm_to_operand(right);
        let is_64 = Va::SIZE == 8;
        ops(left, right, self.out, is_64);
        Ok(())
    }

    fn pushpop_reg_op(&mut self) -> Result<(), Failed> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let byte = self.read_u8(0)?;
        let is_push = byte < 0x58;
        let reg = if self.prefixes.rex_prefix & 0x1 == 0 {
            byte & 0x7
        } else {
            8 + (byte & 0x7)
        };
        let is_64 = Va::SIZE != 4;
        let dest = self.dest_and_op_register(4);
        let esp = dest.op.clone();
        match is_push {
            true => {
                if Va::SIZE == 4 {
                    self.output(sub(dest, self.ctx.const_4(), is_64));
                    self.output(mov_to_mem(MemAccessSize::Mem32, esp, self.ctx.register(reg)));
                } else {
                    self.output(sub(dest, self.ctx.const_8(), is_64));
                    self.output(mov_to_mem(MemAccessSize::Mem64, esp, self.ctx.register(reg)));
                }
            }
            false => {
                if Va::SIZE == 4 {
                    self.output(mov_to_reg(reg, mem32(esp)));
                    self.output(add(dest, self.ctx.const_4(), is_64));
                } else {
                    self.output(mov_to_reg(reg, mem64(esp)));
                    self.output(add(dest, self.ctx.const_8(), is_64));
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
        let dest = self.dest_and_op_register(4);
        let esp = dest.op.clone();
        self.output(sub(dest, self.ctx.const_4(), Va::SIZE == 8));
        self.output(mov_to_mem(MemAccessSize::Mem32, esp, self.ctx.constant(constant as u64)));
        Ok(())
    }
}

impl<'a, 'exec: 'a> InstructionOpsState<'a, 'exec, VirtualAddress32> {
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

impl<'a, 'exec: 'a> InstructionOpsState<'a, 'exec, VirtualAddress64> {
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

trait ArithOperationGenerator {
    fn operation(&self, _: DestAndOperand, _: Rc<Operand>, _: &mut OperationVec, _64bit: bool);
}

macro_rules! arith_op_generator {
    ($stname: ident, $name:ident, $op:ident) => {
        struct $name;
        impl ArithOperationGenerator for $name {
            fn operation(&self, a: DestAndOperand, b: Rc<Operand>, c: &mut OperationVec, d: bool) {
                self::operation_helpers::$op(a, b, c, d)
            }
        }
        static $stname: $name = $name;
    }
}
arith_op_generator!(ADD_OPS, AddOps, add_ops);
arith_op_generator!(ADC_OPS, AdcOps, adc_ops);
arith_op_generator!(OR_OPS, OrOps, or_ops);
arith_op_generator!(AND_OPS, AndOps, and_ops);
arith_op_generator!(SUB_OPS, SubOps, sub_ops);
arith_op_generator!(SBB_OPS, SbbOps, sbb_ops);
arith_op_generator!(XOR_OPS, XorOps, xor_ops);
arith_op_generator!(CMP_OPS, CmpOps, cmp_ops);
arith_op_generator!(TEST_OPS, TestOps, test_ops);
arith_op_generator!(MOV_OPS, MovOps, mov_ops);
arith_op_generator!(ROL_OPS, RolOps, rol_ops);
arith_op_generator!(ROR_OPS, RorOps, ror_ops);
arith_op_generator!(LSH_OPS, LshOps, lsh_ops);
arith_op_generator!(RSH_OPS, RshOps, rsh_ops);
arith_op_generator!(SAR_OPS, SarOps, sar_ops);

pub mod operation_helpers {
    use std::rc::Rc;

    use crate::operand::ArithOpType::*;
    use crate::operand::{
        Flag, Operand, OperandType, MemAccessSize, ArithOpType, ArithOperand, Register,
    };
    use crate::operand::operand_helpers::*;
    use super::{
        DestOperand, make_arith_operation, Operation, OperationVec,
        DestAndOperand,
    };

    pub fn mov_to_reg(dest: u8, from: Rc<Operand>) -> Operation {
        Operation::Move(DestOperand::Register64(Register(dest)), from, None)
    }

    pub fn mov_to_fpu(dest: u8, from: Rc<Operand>) -> Operation {
        Operation::Move(DestOperand::Fpu(dest), from, None)
    }

    pub fn mov_to_reg_variable_size(
        size: MemAccessSize,
        dest: u8,
        from: Rc<Operand>,
    ) -> Operation {
        let dest = match size {
            MemAccessSize::Mem8 => DestOperand::Register8Low(Register(dest)),
            MemAccessSize::Mem16 => DestOperand::Register16(Register(dest)),
            MemAccessSize::Mem32 => DestOperand::Register32(Register(dest)),
            MemAccessSize::Mem64 => DestOperand::Register64(Register(dest)),
        };
        Operation::Move(dest, from, None)
    }

    pub fn mov_to_mem(size: MemAccessSize, address: Rc<Operand>, from: Rc<Operand>) -> Operation {
        let access = crate::operand::MemAccess {
            size,
            address,
        };
        Operation::Move(DestOperand::Memory(access), from, None)
    }

    pub fn mov(dest: DestOperand, from: Rc<Operand>) -> Operation {
        Operation::Move(dest, from, None)
    }

    pub fn mov_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, _64: bool) {
        if dest.op != rhs {
            out.push(mov(dest.dest, rhs));
        }
    }

    pub fn add(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Add, dest.op, rhs, is_64)
    }

    pub fn add_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Add, dest.op.clone(), rhs.clone(), is_64));
        out.push(add(dest, rhs, is_64));
    }

    pub fn adc_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        let rhs = operand_add64(rhs, flag_c());
        out.push(flags(Add, dest.op.clone(), rhs.clone(), is_64));
        out.push(add(dest, rhs, is_64));
    }

    pub fn sub(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Sub, dest.op, rhs, is_64)
    }

    pub fn sub_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Sub, dest.op.clone(), rhs.clone(), is_64));
        out.push(sub(dest, rhs, is_64));
    }

    pub fn cmp_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Sub, dest.op, rhs, is_64));
    }

    pub fn sbb_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        let rhs = operand_add64(rhs, flag_c());
        out.push(flags(Sub, dest.op.clone(), rhs.clone(), is_64));
        out.push(sub(dest, rhs, is_64));
    }

    pub fn mul(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Mul, dest.op, rhs, is_64)
    }

    pub fn signed_mul(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, SignedMul, dest.op, rhs, is_64)
    }

    pub fn xor(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Xor, dest.op, rhs, is_64)
    }

    pub fn xor_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Xor, dest.op.clone(), rhs.clone(), is_64));
        out.push(xor(dest, rhs, is_64));
    }

    pub fn rol_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        // rol(x, y) == (x << y) | (x >> (32 - y))
        let left;
        let right;
        if is_64 {
            left = operand_lsh64(dest.op.clone(), rhs.clone());
            right = operand_rsh64(dest.op, operand_sub64(constval(64), rhs));
        } else {
            left = operand_lsh(dest.op.clone(), rhs.clone());
            right = operand_rsh(dest.op, operand_sub(constval(32), rhs));
        }
        // TODO set overflow if 1bit??
        out.push(make_arith_operation(dest.dest, Or, left, right, is_64));
    }

    pub fn ror_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        // ror(x, y) == (x >> y) | (x << (32 - y))
        let left;
        let right;
        if is_64 {
            left = operand_rsh64(dest.op.clone(), rhs.clone());
            right = operand_lsh64(dest.op, operand_sub64(constval(64), rhs));
        } else {
            left = operand_rsh(dest.op.clone(), rhs.clone());
            right = operand_lsh(dest.op, operand_sub(constval(32), rhs));
        }
        // TODO set overflow if 1bit??
        out.push(make_arith_operation(dest.dest, Or, left, right, is_64));
    }

    pub fn lsh(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Lsh, dest.op, rhs, is_64)
    }

    pub fn lsh_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Lsh, dest.op.clone(), rhs.clone(), is_64));
        out.push(make_arith_operation(dest.dest, Lsh, dest.op, rhs, is_64));
    }

    pub fn rsh(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Rsh, dest.op, rhs, is_64)
    }

    pub fn rsh_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Rsh, dest.op.clone(), rhs.clone(), is_64));
        out.push(rsh(dest, rhs, is_64));
    }

    pub fn sar_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        // TODO flags, using IntToFloat to cheat
        out.push(flags(IntToFloat, dest.op.clone(), rhs.clone(), is_64));
        if is_64 {
            let is_positive = operand_eq64(
                operand_and(constval(0x8000_0000_0000_0000), dest.op.clone()),
                constval(0),
            );
            out.push(Operation::Move(
                dest.dest.clone(),
                operand_or(operand_rsh64(dest.op.clone(), rhs.clone()), constval(0x8000_0000_0000_0000)),
                Some(operand_logical_not(is_positive.clone())),
            ));
            out.push(Operation::Move(
                dest.dest,
                operand_rsh64(dest.op, rhs),
                Some(is_positive),
            ));
        } else {
            let is_positive = operand_eq(
                operand_and(constval(0x8000_0000), dest.op.clone()),
                constval(0),
            );
            out.push(Operation::Move(
                dest.dest.clone(),
                operand_or(operand_rsh(dest.op.clone(), rhs.clone()), constval(0x8000_0000)),
                Some(operand_logical_not(is_positive.clone())),
            ));
            out.push(Operation::Move(
                dest.dest,
                operand_rsh(dest.op, rhs),
                Some(is_positive),
            ));
        }
    }

    pub fn or(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, Or, dest.op, rhs, is_64)
    }

    pub fn or_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Or, dest.op.clone(), rhs.clone(), is_64));
        out.push(or(dest, rhs, is_64));
    }

    pub fn and(dest: DestAndOperand, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest.dest, And, dest.op, rhs, is_64)
    }

    pub fn and_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(And, dest.op.clone(), rhs.clone(), is_64));
        out.push(and(dest, rhs, is_64));
    }

    pub fn test_ops(dest: DestAndOperand, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(And, dest.op, rhs, is_64));
    }

    fn flags(ty: ArithOpType, left: Rc<Operand>, right: Rc<Operand>, is_64: bool) -> Operation {
        let arith = ArithOperand {
            ty,
            left,
            right,
        };
        if is_64 {
            Operation::SetFlags(arith, MemAccessSize::Mem64)
        } else {
            Operation::SetFlags(arith, MemAccessSize::Mem32)
        }
    }

    thread_local! {
        static FLAG_C: Rc<Operand> = Operand::new_simplified_rc(OperandType::Flag(Flag::Carry));
    }

    pub(crate) fn flag_c() -> Rc<Operand> {
        FLAG_C.with(|x| x.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Operation {
    Move(DestOperand, Rc<Operand>, Option<Rc<Operand>>),
    Call(Rc<Operand>),
    Jump { condition: Rc<Operand>, to: Rc<Operand> },
    Return(u32),
    /// Special cases like interrupts etc that scarf doesn't want to touch.
    /// Also rep mov for now
    Special(Vec<u8>),
    /// Set flags based on operation type. While Move(..) could handle this
    /// (And it does for odd cases like inc), it would mean generating 5
    /// additional operations for each instruction, so special-case flags.
    SetFlags(ArithOperand, MemAccessSize),
    /// Like Move, but evaluate all operands before assigning over any.
    /// Used for mul/div/swap.
    MoveSet(Vec<(DestOperand, Rc<Operand>)>),
}

fn make_f32_operation(
    dest: DestOperand,
    ty: ArithOpType,
    left: Rc<Operand>,
    right: Rc<Operand>,
) -> Operation {
    use crate::operand_helpers::*;
    let op = operand_arith_f32(ty, left, right);
    Operation::Move(dest, Operand::simplified(op), None)
}

fn make_arith_operation(
    dest: DestOperand,
    ty: ArithOpType,
    left: Rc<Operand>,
    right: Rc<Operand>,
    is_64: bool,
) -> Operation {
    use crate::operand_helpers::*;
    let op = if is_64 {
        operand_arith64(ty, left, right)
    } else {
        operand_arith(ty, left, right)
    };
    Operation::Move(dest, Operand::simplified(op), None)
}

// Maybe it would be better to pass RegisterSize as an argument to functions
// which use it? It goes unused with xmm. Though maybe better to keep
// its calculation at one point.
#[allow(bad_style)]
#[derive(Copy, Clone)]
struct ModRm_R(u8, RegisterSize);

#[allow(bad_style)]
struct ModRm_Rm {
    size: RegisterSize,
    /// u8::max_value signifies that this is constant base.
    base: u8,
    /// u8::max_value signifies that this is register instead of memory operand.
    /// (Register is in `base`)
    index: u8,
    index_mul: u8,
    /// Rip-relative if 64-bit and base == u8::max_value.
    constant: u32,
}

#[derive(Copy, Clone)]
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
    fn dest_operand(self) -> DestOperand {
        match self.1 {
            RegisterSize::R64 => DestOperand::Register64(Register(self.0)),
            RegisterSize::R32 => DestOperand::Register32(Register(self.0)),
            RegisterSize::R16 => DestOperand::Register16(Register(self.0)),
            RegisterSize::Low8 => DestOperand::Register8Low(Register(self.0)),
            RegisterSize::High8 => DestOperand::Register8High(Register(self.0)),
        }
    }

    fn dest_operand_xmm(self, i: u8) -> DestOperand {
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
        self.base == u8::max_value()
    }
}

pub struct DestAndOperand {
    pub dest: DestOperand,
    pub op: Rc<Operand>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DestOperand {
    Register64(Register),
    Register32(Register),
    Register16(Register),
    Register8High(Register),
    Register8Low(Register),
    Xmm(u8, u8),
    Fpu(u8),
    Flag(Flag),
    Memory(MemAccess),
}

impl DestOperand {
    pub fn reg_variable_size(reg: u8, size: MemAccessSize) -> DestOperand {
        match size {
            MemAccessSize::Mem8 => DestOperand::Register8Low(Register(reg)),
            MemAccessSize::Mem16 => DestOperand::Register16(Register(reg)),
            MemAccessSize::Mem32 => DestOperand::Register32(Register(reg)),
            MemAccessSize::Mem64 => DestOperand::Register64(Register(reg)),
        }
    }

    pub fn from_oper(val: &Operand) -> DestOperand {
        dest_operand(val)
    }

    pub fn as_operand(&self, ctx: &OperandContext) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;
        match *self {
            DestOperand::Register32(x) => Operand::simplified(
                operand_and(ctx.register(x.0), ctx.const_ffffffff())
            ),
            DestOperand::Register16(x) => Operand::simplified(
                operand_and(ctx.register(x.0), ctx.const_ffff())
            ),
            DestOperand::Register8High(x) => Operand::simplified(
                operand_rsh(
                    operand_and(ctx.register(x.0), ctx.const_ffff()),
                    ctx.const_8(),
                )
            ),
            DestOperand::Register8Low(x) => Operand::simplified(
                operand_and(ctx.register(x.0), ctx.const_ff())
            ),
            DestOperand::Register64(x) => ctx.register(x.0),
            DestOperand::Xmm(x, y) => Rc::new(Operand::new_xmm(x, y)),
            DestOperand::Fpu(x) => ctx.register_fpu(x),
            DestOperand::Flag(x) => ctx.flag(x),
            DestOperand::Memory(ref x) => mem_variable_rc(x.size, x.address.clone()),
        }
    }
}

fn dest_operand(val: &Operand) -> DestOperand {
    use crate::operand::OperandType::*;
    match val.ty {
        Register(x) => DestOperand::Register64(x),
        Xmm(x, y) => DestOperand::Xmm(x, y),
        Fpu(x) => DestOperand::Fpu(x),
        Flag(x) => DestOperand::Flag(x),
        Memory(ref x) => DestOperand::Memory(x.clone()),
        Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let result = Operand::either(&arith.left, &arith.right, |x| x.if_constant())
                        .and_then(|(c, other)| {
                            let reg = other.if_register()?;
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
                            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                            .filter(|&(c, _)| c == 0xff00)
                            .and_then(|(_, other)| other.if_register());
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

fn dest_operand_reg64(reg: u8) -> DestOperand {
    DestOperand::Register64(Register(reg))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::VirtualAddress;
    use crate::exec_state::Disassembler;
    #[test]
    fn test_operations_mov16() {
        use crate::operand::operand_helpers::*;
        use crate::operand::OperandContext;

        let ctx = OperandContext::new();
        let buf = [0x66, 0xc7, 0x47, 0x62, 0x00, 0x20];
        let mut disasm = Disassembler32::new(&buf[..], 0, VirtualAddress(0), &ctx);
        let ins = disasm.next().unwrap();
        assert_eq!(ins.ops().len(), 1);
        let op = &ins.ops()[0];
        let dest = mem_variable(
            operand::MemAccessSize::Mem16,
            operand_add(operand_register(0x7), constval(0x62))
        );

        assert_eq!(*op, Operation::Move(dest_operand(&dest), constval(0x2000), None));
    }

    #[test]
    fn test_sib() {
        use crate::operand::operand_helpers::*;
        use crate::operand::OperandContext;

        let ctx = OperandContext::new();
        let buf = [0x89, 0x84, 0xb5, 0x18, 0xeb, 0xff, 0xff];
        let mut disasm = Disassembler32::new(&buf[..], 0, VirtualAddress(0), &ctx);
        let ins = disasm.next().unwrap();
        assert_eq!(ins.ops().len(), 1);
        let op = &ins.ops()[0];
        let dest = mem32(
            operand_add(
                operand_mul(
                    constval(4),
                    operand_register(6),
                ),
                operand_sub(
                    operand_register(5),
                    constval(0x14e8),
                ),
            ),
        );

        match op.clone() {
            Operation::Move(d, f, cond) => {
                let d = d.as_operand(&ctx);
                assert_eq!(Operand::simplified(d), Operand::simplified(dest));
                assert_eq!(
                    f,
                    Operand::simplified(operand_and(operand_register(0), constval(0xffff_ffff))),
                );
                assert_eq!(cond, None);
            }
            _ => panic!("Unexpected op {:?}", op),
        }
    }
}

