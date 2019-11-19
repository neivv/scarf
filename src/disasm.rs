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

pub type OperationVec = SmallVec<[Operation; 8]>;

pub struct Disassembler32<'a> {
    buf: &'a [u8],
    pos: usize,
    virtual_address: VirtualAddress32,
    is_branching: bool,
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler32<'a> {
    type VirtualAddress = VirtualAddress32;

    fn new(buf: &'a [u8], pos: usize, address: VirtualAddress32) -> Disassembler32<'a> {
        assert!(pos < buf.len());
        Disassembler32 {
            buf,
            pos,
            virtual_address: address,
            is_branching: false,
        }
    }

    fn next(&mut self, ctx: &OperandContext) -> Result<Instruction<VirtualAddress32>, Error> {
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
        let ops = instruction_operations32(address, data, ctx)?;
        let ins = Instruction {
            address,
            ops,
            length,
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
}

impl<'a> crate::exec_state::Disassembler<'a> for Disassembler64<'a> {
    type VirtualAddress = VirtualAddress64;

    fn new(buf: &'a [u8], pos: usize, address: VirtualAddress64) -> Disassembler64<'a> {
        assert!(pos < buf.len());
        Disassembler64 {
            buf,
            pos,
            virtual_address: address,
            is_branching: false,
        }
    }

    fn next(&mut self, ctx: &OperandContext) -> Result<Instruction<VirtualAddress64>, Error> {
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
        let ops = instruction_operations64(address, data, ctx)?;
        let ins = Instruction {
            address,
            ops,
            length,
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

pub struct Instruction<Va: VirtualAddress> {
    address: Va,
    ops: SmallVec<[Operation; 8]>,
    length: usize,
}

impl<Va: VirtualAddress> Instruction<Va> {
    pub fn ops(&self) -> &[Operation] {
        &self.ops
    }

    pub fn address(&self) -> Va {
        self.address
    }

    pub fn len(&self) -> usize {
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

struct InstructionOpsState<'a, 'exec: 'a, Va: VirtualAddress> {
    address: Va,
    data: &'a [u8],
    prefixes: InstructionPrefixes,
    len: u8,
    ctx: &'exec OperandContext,
}

fn instruction_operations32(
    address: VirtualAddress32,
    data: &[u8],
    ctx: &OperandContext,
) -> Result<SmallVec<[Operation; 8]>, Error> {
    use self::Error::*;
    use self::operation_helpers::*;
    use crate::operand::operand_helpers::*;

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
    let s = InstructionOpsState {
        address,
        data,
        prefixes,
        len: instruction_len as u8,
        ctx,
    };
    let first_byte = data[0];
    if !is_ext {
        match first_byte {
            0x00 ..= 0x05 | 0x08 ..= 0x0b | 0x0c ..= 0x0d | 0x10 ..= 0x15 | 0x18 ..= 0x1d |
            0x20 ..= 0x25 | 0x28 ..= 0x2d | 0x30 ..= 0x35 | 0x88 ..= 0x8b | 0x8d =>
            {
                // Avoid ridiculous generic binary bloat
                let ops: for<'x> fn(_, _, &'x mut _, bool) = match first_byte {
                    0x00 ..= 0x05 => add_ops,
                    0x08 ..= 0x0d => or_ops,
                    0x10 ..= 0x15 => adc_ops,
                    0x18 ..= 0x1d => sbb_ops,
                    0x20 ..= 0x25 => and_ops,
                    0x28 ..= 0x2d => sub_ops,
                    0x30 ..= 0x35 => xor_ops,
                    0x88 ..= 0x8b => mov_ops,
                    0x8d | _ => lea_ops,
                };
                let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
                if eax_imm_arith {
                    s.eax_imm_arith(ops)
                } else {
                    s.generic_arith_op(ops)
                }
            }

            0x38 ..= 0x3b => s.generic_cmp_op(cmp_ops),
            0x3c ..= 0x3d => s.eax_imm_cmp(cmp_ops),
            0x40 ..= 0x4f => s.inc_dec_op(),
            0x50 ..= 0x5f => s.pushpop_reg_op(),
            0x68 | 0x6a => s.push_imm(),
            0x69 | 0x6b => s.signed_multiply_rm_imm(),
            0x70 ..= 0x7f => s.conditional_jmp(MemAccessSize::Mem8),
            0x80 ..= 0x83 => s.arith_with_imm_op(),
            // Test
            0x84 ..= 0x85 => s.generic_cmp_op(test_ops),
            0x86 ..= 0x87 => s.xchg(),
            0x90 => Ok(SmallVec::new()),
            // Cwde
            0x98 => {
                let mut out = SmallVec::new();
                let eax = ctx.register(0);
                let signed_max = ctx.const_7fff();
                let cond = operand_gt(eax.clone(), signed_max);
                let neg_sign_extend = operand_or(eax.clone(), ctx.const_ffff0000());
                let neg_sign_extend_op =
                    Operation::Move(dest_operand(&eax), neg_sign_extend, Some(cond));
                out.push(and(eax, ctx.const_ffff(), false));
                out.push(neg_sign_extend_op);
                Ok(out)
            }
            // Cdq
            0x99 => {
                let mut out = SmallVec::new();
                let eax = ctx.register(0);
                let edx = ctx.register(2);
                let signed_max = ctx.const_7fffffff();
                let cond = operand_gt(eax, signed_max);
                let neg_sign_extend_op =
                    Operation::Move(dest_operand(&edx), ctx.const_ffffffff(), Some(cond));
                out.push(mov(edx, ctx.const_0()));
                out.push(neg_sign_extend_op);
                Ok(out)
            },
            0xa0 ..= 0xa3 => s.move_mem_eax(),
            // rep mov
            0xa4 ..= 0xa5 => {
                let mut out = SmallVec::new();
                out.push(Operation::Special(s.data.into()));
                Ok(out)
            }
            0xa8 ..= 0xa9 => s.eax_imm_cmp(test_ops),
            0xb0 ..= 0xbf => s.move_const_to_reg(),
            0xc0 ..= 0xc1 => s.bitwise_with_imm_op(),
            0xc2 ..= 0xc3 => {
                let stack_pop_size = match data[0] {
                    0xc2 => match read_u16(&data[1..]) {
                        Err(_) => 0,
                        Ok(o) => u32::from(o),
                    },
                    _ => 0,
                };
                Ok(Some(Operation::Return(stack_pop_size)).into_iter().collect())
            }
            0xc6 ..= 0xc7 => {
                s.generic_arith_with_imm_op(&MOV_OPS, match s.get(0) {
                    0xc6 => MemAccessSize::Mem8,
                    _ => s.mem16_32(),
                })
            }
            0xd0 ..= 0xd3 => s.bitwise_compact_op(),
            0xd8 => s.various_d8(),
            0xd9 => s.various_d9(),
            0xe8 => s.call_op(),
            0xe9 => s.jump_op(),
            0xeb => s.short_jmp(),
            0xf6 | 0xf7 => s.various_f7(),
            0xf8 | 0xf9 | 0xfc | 0xfd => {
                let flag = match first_byte {
                    0xf8 ..= 0xf9 => Flag::Carry,
                    _ => Flag::Direction,
                };
                let state = first_byte & 0x1 == 1;
                s.flag_set(flag, state)
            }
            0xfe ..= 0xff => s.various_fe_ff(),
            _ => Err(UnknownOpcode(s.data.into()))
        }
    } else {
        match first_byte {
            0x12 | 0x13 => s.mov_sse_12_13(),
            // Prefetch/nop
            0x18 ..= 0x1f => Ok(SmallVec::new()),
            0x10 | 0x11| 0x28 | 0x29 | 0x2b | 0x7e | 0x7f => s.sse_move(),
            0x2c => s.cvttss2si(),
            // rdtsc
            0x31 => {
                let mut out = SmallVec::new();
                out.push(mov(ctx.register(0), s.ctx.undefined_rc()));
                out.push(mov(ctx.register(2), s.ctx.undefined_rc()));
                Ok(out)
            }
            0x40 ..= 0x4f => s.cmov(),
            0x57 => s.xorps(),
            0x5b => s.cvtdq2ps(),
            0x6e => s.mov_sse_6e(),
            0x6f => {
                if s.has_prefix(0xf3) || s.has_prefix(0x66) {
                    // movdqa/u
                    s.mov_rm_to_xmm_128()
                } else {
                    Err(UnknownOpcode(s.data.into()))
                }
            }
            0x80 ..= 0x8f => s.conditional_jmp(s.mem16_32()),
            0x90 ..= 0x9f => s.conditional_set(),
            0xa3 => s.bit_test(false),
            0xa4 => s.shld_imm(),
            0xac => s.shrd_imm(),
            0xae => {
                match (s.get(1) >> 3) & 0x7 {
                    // Memory fences
                    // (5 is also xrstor though)
                    5 | 6 | 7 => Ok(SmallVec::new()),
                    _ => Err(Error::UnknownOpcode(s.data.into())),
                }
            }
            0xaf => s.imul_normal(),
            0xb1 => {
                // Cmpxchg
                let (rm, _r) = s.parse_modrm(s.mem16_32())?;
                let mut out = SmallVec::new();
                out.push(mov(rm, ctx.undefined_rc()));
                out.push(mov(operand_register(0), ctx.undefined_rc()));
                Ok(out)
            }
            0xb3 => s.btr(false),
            0xb6 ..= 0xb7 => s.movzx(),
            0xba => s.various_0f_ba(),
            0xbe => s.movsx(MemAccessSize::Mem8),
            0xbf => s.movsx(MemAccessSize::Mem16),
            0xc0 ..= 0xc1 => s.xadd(),
            0xd3 => s.packed_shift_right(),
            0xd6 => s.mov_sse_d6(),
            0xf3 => s.packed_shift_left(),
            _ => {
                let mut bytes = vec![0xf];
                bytes.extend(s.data);
                Err(UnknownOpcode(bytes))
            }
        }
    }
}

fn instruction_operations64(
    address: VirtualAddress64,
    data: &[u8],
    ctx: &OperandContext,
) -> Result<SmallVec<[Operation; 8]>, Error> {
    use self::Error::*;
    use self::operation_helpers::*;
    use crate::operand::operand_helpers::*;

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
    let s = InstructionOpsState {
        address,
        data,
        prefixes,
        len: instruction_len as u8,
        ctx,
    };
    let first_byte = data[0];
    if !is_ext {
        match first_byte {
            0x00 ..= 0x05 | 0x08 ..= 0x0b | 0x0c ..= 0x0d | 0x10 ..= 0x15 | 0x18 ..= 0x1d |
            0x20 ..= 0x25 | 0x28 ..= 0x2d | 0x30 ..= 0x35 | 0x88 ..= 0x8b | 0x8d =>
            {
                // Avoid ridiculous generic binary bloat
                let ops: for<'x> fn(_, _, &'x mut _, bool) = match first_byte {
                    0x00 ..= 0x05 => add_ops,
                    0x08 ..= 0x0d => or_ops,
                    0x10 ..= 0x15 => adc_ops,
                    0x18 ..= 0x1d => sbb_ops,
                    0x20 ..= 0x25 => and_ops,
                    0x28 ..= 0x2d => sub_ops,
                    0x30 ..= 0x35 => xor_ops,
                    0x88 ..= 0x8b => mov_ops,
                    0x8d | _ => lea_ops,
                };
                let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
                if eax_imm_arith {
                    s.eax_imm_arith(ops)
                } else {
                    s.generic_arith_op(ops)
                }
            }

            0x38 ..= 0x3b => s.generic_cmp_op(cmp_ops),
            0x3c ..= 0x3d => s.eax_imm_cmp(cmp_ops),
            0x50 ..= 0x5f => s.pushpop_reg_op(),
            0x63 => s.movsx(MemAccessSize::Mem32),
            0x68 | 0x6a => s.push_imm(),
            0x69 | 0x6b => s.signed_multiply_rm_imm(),
            0x70 ..= 0x7f => s.conditional_jmp(MemAccessSize::Mem8),
            0x80 ..= 0x83 => s.arith_with_imm_op(),
            // Test
            0x84 ..= 0x85 => s.generic_cmp_op(test_ops),
            0x86 ..= 0x87 => s.xchg(),
            0x90 => Ok(SmallVec::new()),
            // Cwde
            0x98 => {
                let mut out = SmallVec::new();
                if s.prefixes.rex_prefix & 0x8 == 0 {
                    let eax = ctx.register(0);
                    let signed_max = ctx.const_7fff();
                    let cond = operand_gt(eax.clone(), signed_max);
                    let neg_sign_extend = operand_or(eax.clone(), ctx.const_ffff0000());
                    let neg_sign_extend_op =
                        Operation::Move(dest_operand(&eax), neg_sign_extend, Some(cond));
                    out.push(and(eax, ctx.const_ffff(), false));
                    out.push(neg_sign_extend_op);
                } else {
                    let rax = ctx.register64(0);
                    let signed_max = ctx.const_7fffffff();
                    let cond = operand_gt(rax.clone(), signed_max);
                    let neg_sign_extend =
                        operand_or64(rax.clone(), ctx.constant64(0xffff_ffff_0000_0000));
                    let neg_sign_extend_op =
                        Operation::Move(dest_operand(&rax), neg_sign_extend, Some(cond));
                    out.push(and(rax, ctx.const_ffffffff(), false));
                    out.push(neg_sign_extend_op);
                }
                Ok(out)
            }
            // Cdq
            0x99 => {
                let mut out = SmallVec::new();
                if s.prefixes.rex_prefix & 0x8 == 0 {
                    let eax = ctx.register(0);
                    let edx = ctx.register(2);
                    let signed_max = ctx.const_7fffffff();
                    let cond = operand_gt(eax, signed_max);
                    let neg_sign_extend_op =
                        Operation::Move(dest_operand(&edx), ctx.const_ffffffff(), Some(cond));
                    out.push(mov(edx, ctx.const_0()));
                    out.push(neg_sign_extend_op);
                } else {
                    let rax = ctx.register64(0);
                    let rdx = ctx.register64(2);
                    let signed_max = ctx.constant64(0x7fff_ffff_ffff_ffff);
                    let cond = operand_gt64(rax, signed_max);
                    let neg_sign_extend_op =
                        Operation::Move(dest_operand(&rdx), ctx.constant64(!0), Some(cond));
                    out.push(mov(rdx, ctx.const_0()));
                    out.push(neg_sign_extend_op);
                }
                Ok(out)
            },
            0xa0 ..= 0xa3 => s.move_mem_eax(),
            // rep mov
            0xa4 ..= 0xa5 => {
                let mut out = SmallVec::new();
                out.push(Operation::Special(s.data.into()));
                Ok(out)
            }
            0xa8 ..= 0xa9 => s.eax_imm_cmp(test_ops),
            0xb0 ..= 0xbf => s.move_const_to_reg(),
            0xc0 ..= 0xc1 => s.bitwise_with_imm_op(),
            0xc2 ..= 0xc3 => {
                let stack_pop_size = match data[0] {
                    0xc2 => match read_u16(&data[1..]) {
                        Err(_) => 0,
                        Ok(o) => u32::from(o),
                    },
                    _ => 0,
                };
                Ok(Some(Operation::Return(stack_pop_size)).into_iter().collect())
            }
            0xc6 ..= 0xc7 => {
                s.generic_arith_with_imm_op(&MOV_OPS, match s.get(0) {
                    0xc6 => MemAccessSize::Mem8,
                    _ => s.mem16_32(),
                })
            }
            0xd0 ..= 0xd3 => s.bitwise_compact_op(),
            0xe8 => s.call_op(),
            0xe9 => s.jump_op(),
            0xeb => s.short_jmp(),
            0xf6 | 0xf7 => s.various_f7(),
            0xf8 | 0xf9 | 0xfc | 0xfd => {
                let flag = match first_byte {
                    0xf8 ..= 0xf9 => Flag::Carry,
                    _ => Flag::Direction,
                };
                let state = first_byte & 0x1 == 1;
                s.flag_set(flag, state)
            }
            0xfe ..= 0xff => s.various_fe_ff(),
            _ => Err(UnknownOpcode(s.data.into()))
        }
    } else {
        match first_byte {
            0x12 | 0x13 => s.mov_sse_12_13(),
            // Prefetch/nop
            0x18 ..= 0x1f => Ok(SmallVec::new()),
            0x10 | 0x11| 0x28 | 0x29 | 0x2b | 0x7e | 0x7f => s.sse_move(),
            0x2c => s.cvttss2si(),
            // rdtsc
            0x31 => {
                let mut out = SmallVec::new();
                out.push(mov(ctx.register(0), s.ctx.undefined_rc()));
                out.push(mov(ctx.register(2), s.ctx.undefined_rc()));
                Ok(out)
            }
            0x40 ..= 0x4f => s.cmov(),
            0x57 => s.xorps(),
            0x5b => s.cvtdq2ps(),
            0x6e => s.mov_sse_6e(),
            0x6f => {
                if s.has_prefix(0xf3) || s.has_prefix(0x66) {
                    // movdqa/u
                    s.mov_rm_to_xmm_128()
                } else {
                    Err(UnknownOpcode(s.data.into()))
                }
            }
            0x80 ..= 0x8f => s.conditional_jmp(s.mem16_32()),
            0x90 ..= 0x9f => s.conditional_set(),
            0xa3 => s.bit_test(false),
            0xa4 => s.shld_imm(),
            0xac => s.shrd_imm(),
            0xae => {
                match (s.get(1) >> 3) & 0x7 {
                    // Memory fences
                    // (5 is also xrstor though)
                    5 | 6 | 7 => Ok(SmallVec::new()),
                    _ => Err(Error::UnknownOpcode(s.data.into())),
                }
            }
            0xaf => s.imul_normal(),
            0xb1 => {
                // Cmpxchg
                let (rm, _r) = s.parse_modrm(s.mem16_32())?;
                let mut out = SmallVec::new();
                out.push(mov(rm, ctx.undefined_rc()));
                out.push(mov(operand_register(0), ctx.undefined_rc()));
                Ok(out)
            }
            0xb3 => s.btr(false),
            0xb6 ..= 0xb7 => s.movzx(),
            0xba => s.various_0f_ba(),
            0xbe => s.movsx(MemAccessSize::Mem8),
            0xbf => s.movsx(MemAccessSize::Mem16),
            0xc0 ..= 0xc1 => s.xadd(),
            0xd3 => s.packed_shift_right(),
            0xd6 => s.mov_sse_d6(),
            0xf3 => s.packed_shift_left(),
            _ => {
                let mut bytes = vec![0xf];
                bytes.extend(s.data);
                Err(UnknownOpcode(bytes))
            }
        }
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

    fn get(&self, idx: usize) -> u8 {
        self.data[idx]
    }

    fn slice(&self, idx: usize) -> &[u8] {
        &self.data[idx..]
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

    fn word_register(&self, register: u8) -> Rc<Operand> {
        if Va::SIZE == 4 || self.has_prefix(0x67) {
            self.ctx.register(register)
        } else {
            self.ctx.register64(register)
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

    fn xmm_variant(&self, op: &Rc<Operand>, i: u8) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;
        assert!(i < 4);
        match op.ty {
            OperandType::Register(Register(r)) | OperandType::Xmm(r, _) => operand_xmm(r, i),
            OperandType::Memory(ref mem) => {
                let bytes = match mem.size {
                    MemAccessSize::Mem8 => 1,
                    MemAccessSize::Mem16 => 2,
                    MemAccessSize::Mem32 => 4,
                    MemAccessSize::Mem64 => unreachable!(),
                };
                mem_variable_rc(
                    mem.size,
                    self.word_add(mem.address.clone(), constval(bytes * u32::from(i))),
                )
            }
            _ => panic!("Cannot xmm {:?}", op),
        }
    }

    fn reg_variable_size(&self, register: Register, op_size: MemAccessSize) -> Rc<Operand> {
        if register.0 >= 4 && self.prefixes.rex_prefix == 0 && op_size == MemAccessSize::Mem8 {
            self.ctx.register8_high(register.0 - 4)
        } else {
            self.ctx.reg_variable_size(register, op_size)
        }
    }

    /// Returns (rm, r, modrm_size)
    fn parse_modrm_inner(
        &self,
        op_size: MemAccessSize
    ) -> Result<(Rc<Operand>, Rc<Operand>, usize), Error> {
        use crate::operand::operand_helpers::*;

        let modrm = self.get(1);
        let rm_val = modrm & 0x7;
        let register = if self.prefixes.rex_prefix & 0x4 == 0 {
            (modrm >> 3) & 0x7
        } else {
            8 + ((modrm >> 3) & 0x7)
        };
        let rm_ext = self.prefixes.rex_prefix & 0x1 != 0;
        let r = self.reg_variable_size(Register(register), op_size);
        let (rm, size) = match (modrm >> 6) & 0x3 {
            0 => match rm_val {
                4 => self.parse_sib(0, op_size)?,
                5 => {
                    // 32-bit has the immediate as mem[imm],
                    // 64-bit has mem[rip + imm]
                    let imm = read_u32(self.slice(2))?;
                    if Va::SIZE == 4 {
                        (mem_variable_rc(op_size, self.ctx.constant(imm)), 6)
                    } else {
                        let addr = self.address.as_u64()
                            .wrapping_add(self.len() as u64)
                            .wrapping_add(imm as i32 as i64 as u64);
                        (mem_variable_rc(op_size, self.ctx.constant64(addr)), 6)
                    }
                }
                reg => {
                    let reg = match rm_ext {
                        false => reg,
                        true => reg + 8,
                    };
                    (mem_variable_rc(op_size, self.word_register(reg)), 2)
                }
            },
            1 => match rm_val {
                4 => self.parse_sib(1, op_size)?,
                reg => {
                    let reg = match rm_ext {
                        false => reg,
                        true => reg + 8,
                    };
                    let offset = if Va::SIZE == 4 {
                        self.ctx.constant(read_u8(self.slice(2))? as i8 as u32)
                    } else {
                        self.ctx.constant64(read_u8(self.slice(2))? as i8 as u64)
                    };
                    let addition = self.word_add(self.word_register(reg), offset);
                    (mem_variable_rc(op_size, addition), 3)
                }
            },
            2 => match rm_val {
                4 => self.parse_sib(2, op_size)?,
                reg => {
                    let reg = match rm_ext {
                        false => reg,
                        true => reg + 8,
                    };
                    let offset = if Va::SIZE == 4 {
                        self.ctx.constant(read_u32(self.slice(2))? as i32 as u32)
                    } else {
                        self.ctx.constant64(read_u32(self.slice(2))? as i32 as u64)
                    };
                    let addition = self.word_add(self.word_register(reg), offset);
                    (mem_variable_rc(op_size, addition), 6)
                }
            },
            3 => {
                if rm_ext {
                    (self.reg_variable_size(Register(rm_val + 8), op_size), 2)
                } else {
                    (self.reg_variable_size(Register(rm_val), op_size), 2)
                }
            }
            _ => unreachable!(),
        };
        Ok((rm, r, size))
    }

    fn parse_modrm(&self, op_size: MemAccessSize) -> Result<(Rc<Operand>, Rc<Operand>), Error> {
        let (rm, r, _) = self.parse_modrm_inner(op_size)?;
        Ok((rm, r))
    }

    fn parse_modrm_imm(
        &self,
        op_size: MemAccessSize,
        imm_size: MemAccessSize,
    ) -> Result<(Rc<Operand>, Rc<Operand>, Rc<Operand>), Error> {
        let (rm, r, offset) = self.parse_modrm_inner(op_size)?;
        let imm = read_variable_size_32(self.slice(offset), imm_size)?;
        let imm = match imm_size {
            x if x == op_size => imm,
            MemAccessSize::Mem8 => imm as i8 as u64,
            MemAccessSize::Mem16 => imm as i16 as u64,
            MemAccessSize::Mem32 => imm as i32 as u64,
            MemAccessSize::Mem64 => imm,
        };
        Ok((rm, r, self.ctx.constant64(imm)))
    }

    fn parse_sib(
        &self,
        variation: u8,
        op_size: operand::MemAccessSize,
    ) -> Result<(Rc<Operand>, usize), Error> {
        use crate::operand::operand_helpers::*;
        let sib = self.get(2);
        let mul = 1 << ((sib >> 6) & 0x3);
        let base_ext = self.prefixes.rex_prefix & 0x1 != 0;
        let (base_reg, size) = match (sib & 0x7, variation) {
            (5, 0) => {
                if Va::SIZE == 4 {
                    (self.ctx.constant(read_u32(self.slice(3))?), 7)
                } else {
                    (self.ctx.constant64(read_u32(self.slice(3))? as i32 as u64), 7)
                }
            }
            (reg, _) => {
                match base_ext {
                    false => (self.word_register(reg), 3),
                    true => (self.word_register(8 + reg), 3),
                }
            }
        };
        let reg = if self.prefixes.rex_prefix & 0x2 == 0 {
            (sib >> 3) & 0x7
        } else {
            8 + ((sib >> 3) & 0x7)
        };
        let full_mem_op = if reg != 4 {
            let scale_reg = if mul != 1 {
                operand_mul(self.word_register(reg), self.ctx.constant(mul))
            } else {
                self.word_register(reg)
            };
            self.word_add(scale_reg, base_reg)
        } else {
            base_reg
        };
        match variation {
            0 => Ok((mem_variable_rc(op_size, full_mem_op), size)),
            1 => {
                let constant = if Va::SIZE == 4 {
                    self.ctx.constant(read_u8(self.slice(size))? as i8 as u32)
                } else {
                    self.ctx.constant64(read_u8(self.slice(size))? as i8 as u64)
                };
                Ok((mem_variable_rc(op_size, self.word_add(full_mem_op, constant)), size + 1))
            }
            2 => {
                let constant = if Va::SIZE == 4 {
                    self.ctx.constant(read_u32(self.slice(size))?)
                } else {
                    self.ctx.constant64(read_u32(self.slice(size))? as i32 as u64)
                };
                Ok((mem_variable_rc(op_size, self.word_add(full_mem_op, constant)), size + 4))
            }
            _ => unreachable!(),
        }
    }

    fn inc_dec_op(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let byte = self.get(0);
        let is_inc = byte < 0x48;
        let reg = byte & 0x7;
        let reg = self.ctx.reg_variable_size(Register(reg), self.mem16_32());
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        out.push(match is_inc {
            true => add(reg.clone(), self.ctx.const_1(), is_64),
            false => sub(reg.clone(), self.ctx.const_1(), is_64),
        });
        out.push(
            make_arith_operation(
                DestOperand::Flag(Flag::Zero),
                ArithOpType::Equal,
                reg.clone(),
                self.ctx.const_0(),
                is_64,
            )
        );
        if is_64 {
            out.push(make_arith_operation(
                DestOperand::Flag(Flag::Sign),
                ArithOpType::GreaterThan,
                reg.clone(),
                self.ctx.constant64(0x7fff_ffff_ffff_ffff),
                true,
            ));
        } else {
            out.push(make_arith_operation(
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
            (true, true) => self.ctx.constant64(0x8000_0000_0000_0000),
            (false, true) => self.ctx.constant64(0x7fff_ffff_ffff_ffff),
        };
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Overflow),
            ArithOpType::Equal,
            reg,
            eq_value,
            is_64,
        ));
        Ok(out)
    }

    fn flag_set(&self, flag: Flag, value: bool) -> Result<OperationVec, Error> {
        let mut out = SmallVec::new();
        out.push(Operation::Move(DestOperand::Flag(flag), self.ctx.constant(value as u32), None));
        Ok(out)
    }

    fn condition(&self) -> Rc<Operand> {
        use crate::operand::operand_helpers::*;
        let ctx = self.ctx;
        match self.get(0) & 0xf {
            // jo, jno
            0x0 => operand_logical_not(operand_eq(ctx.flag_o(), ctx.const_0())),
            0x1 => operand_eq(ctx.flag_o(), ctx.const_0()),
            // jb, jnb (jae) (jump if carry)
            0x2 => operand_logical_not(operand_eq(ctx.flag_c(), ctx.const_0())),
            0x3 => operand_eq(ctx.flag_c(), ctx.const_0()),
            // je, jne
            0x4 => operand_logical_not(operand_eq(ctx.flag_z(), ctx.const_0())),
            0x5 => operand_eq(ctx.flag_z(), ctx.const_0()),
            // jbe, jnbe (ja)
            0x6 => operand_or(
                operand_logical_not(operand_eq(ctx.flag_z(), ctx.const_0())),
                operand_logical_not(operand_eq(ctx.flag_c(), ctx.const_0())),
            ),
            0x7 => operand_and(
                operand_eq(ctx.flag_z(), ctx.const_0()),
                operand_eq(ctx.flag_c(), ctx.const_0()),
            ),
            // js, jns
            0x8 => operand_logical_not(operand_eq(ctx.flag_s(), ctx.const_0())),
            0x9 => operand_eq(ctx.flag_s(), ctx.const_0()),
            // jpe, jpo
            0xa => operand_logical_not(operand_eq(ctx.flag_p(), ctx.const_0())),
            0xb => operand_eq(ctx.flag_p(), ctx.const_0()),
            // jl, jnl (jge)
            0xc => operand_logical_not(operand_eq(ctx.flag_s(), ctx.flag_o())),
            0xd => operand_eq(ctx.flag_s(), ctx.flag_o()),
            // jle, jnle (jg)
            0xe => operand_or(
                operand_logical_not(operand_eq(ctx.flag_z(), ctx.const_0())),
                operand_logical_not(operand_eq(ctx.flag_s(), ctx.flag_o())),
            ),
            0xf => operand_and(
                operand_eq(ctx.flag_z(), ctx.const_0()),
                operand_eq(ctx.flag_s(), ctx.flag_o()),
            ),
            _ => unreachable!(),
        }
    }

    fn cmov(&self) -> Result<OperationVec, Error> {
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let mut out = SmallVec::new();
        out.push(Operation::Move(dest_operand(&r), rm, Some(self.condition())));
        Ok(out)
    }

    fn conditional_set(&self) -> Result<OperationVec, Error> {
        let condition = self.condition();
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem8)?;
        let mut out = SmallVec::new();
        out.push(Operation::Move(dest_operand(&rm), condition, None));
        Ok(out)
    }

    fn xchg(&self) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r) = self.parse_modrm(op_size)?;
        let mut out = SmallVec::new();
        if rm != r {
            out.push(Operation::Swap(dest_operand(&r), dest_operand(&rm)));
        }
        Ok(out)
    }

    fn xadd(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let mut result = self.xchg()?;
        result.extend(self.generic_arith_op(add_ops)?);
        Ok(result)
    }

    fn move_mem_eax(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let constant = read_variable_size_64(self.slice(1), op_size)?;
        let mem = mem_variable(op_size, self.ctx.constant64(constant)).into();
        let eax_left = self.get(0) & 0x2 == 0;
        let eax = self.ctx.register(0);
        let mut out = SmallVec::new();
        out.push(match eax_left {
            true => mov(eax, mem),
            false => mov(mem, eax),
        });
        Ok(out)
    }

    fn move_const_to_reg(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let op_size = match self.get(0) & 0x8 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let register = if self.prefixes.rex_prefix & 0x1 != 0 {
            8 + (self.get(0) & 0x7)
        } else {
            self.get(0) & 0x7
        };
        let constant = read_variable_size_64(self.slice(1), op_size)?;
        let mut out = SmallVec::new();
        let register = self.ctx.reg_variable_size(Register(register), op_size);
        out.push(mov(register, self.ctx.constant64(constant)));
        Ok(out)
    }

    fn eax_imm_arith(
        &self,
        make_arith: fn(Rc<Operand>, Rc<Operand>, &mut OperationVec, bool),
    ) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let dest = self.reg_variable_size(Register(0), op_size);
        let imm = read_variable_size_32(self.slice(1), op_size)?;
        let val = self.ctx.constant64(imm);
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        make_arith(dest.clone(), val.clone(), &mut out, is_64);
        Ok(out)
    }

    /// Also mov even though I'm not sure if I should count it as no-op arith or a separate
    /// thing.
    fn generic_arith_op(
        &self,
        make_arith: fn(Rc<Operand>, Rc<Operand>, &mut OperationVec, bool),
    ) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r) = self.parse_modrm(op_size)?;
        let rm_left = self.get(0) & 0x3 < 2;
        let mut out = SmallVec::new();
        let (left, right) = match rm_left {
            true => (rm, r),
            false => (r, rm),
        };
        let is_64 = Va::SIZE == 8;
        make_arith(left.clone(), right.clone(), &mut out, is_64);
        Ok(out)
    }

    fn movsx(&self, op_size: MemAccessSize) -> Result<OperationVec, Error> {
        use crate::operand::operand_helpers::*;
        let dest_size = self.mem16_32();
        let (rm, r) = self.parse_modrm(dest_size)?;
        let rm = match rm.ty {
            OperandType::Memory(ref mem) => {
                mem_variable_rc(op_size, mem.address.clone())
            }
            OperandType::Register(r) | OperandType::Register16(r) => {
                self.reg_variable_size(r, op_size)
            }
            _ => rm.clone(),
        };

        let mut out = SmallVec::new();
        out.push(Operation::Move(
            dest_operand(&r),
            Operand::new_not_simplified_rc(
                OperandType::SignExtend(rm, op_size, dest_size),
            ),
            None,
        ));
        Ok(out)
    }

    fn movzx(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;

        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let rm = match rm.ty {
            OperandType::Memory(ref mem) => mem_variable_rc(op_size, mem.address.clone()),
            OperandType::Register(r) | OperandType::Register16(r) => {
                self.reg_variable_size(r, op_size)
            }
            _ => rm.clone(),
        };
        let mut out = SmallVec::new();
        if is_rm_short_r_register(&rm, &r) {
            out.push(match op_size {
                MemAccessSize::Mem8 => and(r, self.ctx.const_ff(), false),
                MemAccessSize::Mem16 => and(r, self.ctx.const_ffff(), false),
                _ => unreachable!(),
            });
        } else {
            let is_mem = match rm.ty {
                OperandType::Memory(_) => true,
                _ => false,
            };
            if is_mem {
                out.push(mov(r, rm));
            } else {
                out.push(mov(r.clone(), self.ctx.const_0()));
                out.push(mov(r, rm));
            }
        }
        Ok(out)
    }

    fn various_f7(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _) = self.parse_modrm(op_size)?;
        let variant = (self.get(1) >> 3) & 0x7;
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        match variant {
            0 | 1 => return self.generic_arith_with_imm_op(&TEST_OPS, op_size),
            2 => {
                // Not
                let dest = dest_operand(&rm);
                let constant = self.ctx.constant64(!0u64);
                out.push(
                    make_arith_operation(dest, ArithOpType::Xor, rm, constant, is_64)
                );
            }
            3 => {
                // Neg
                let dest = dest_operand(&rm);
                out.push(make_arith_operation(dest, ArithOpType::Sub, self.ctx.const_0(), rm, is_64));
            }
            4 => {
                out.push(mov(pair_edx_eax(), operand_mul(self.ctx.register(0), rm)));
            },
            5 => {
                out.push(mov(pair_edx_eax(), operand_signed_mul(self.ctx.register(0), rm)));
            },
            6 => {
                let div = operand_div(pair_edx_eax(), rm.clone());
                let modulo = operand_mod(pair_edx_eax(), rm);
                out.push(mov(pair_edx_eax(), pair(modulo, div)));
            }
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        }
        Ok(out)
    }

    fn various_0f_ba(&self) -> Result<OperationVec, Error> {
        let variant = (self.get(1) >> 3) & 0x7;
        match variant {
            4 => self.bit_test(true),
            6 => self.btr(true),
            _ => Err(Error::UnknownOpcode(self.data.into())),
        }
    }

    fn bit_test(&self, imm8: bool) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let op_size = self.mem16_32();
        let (dest, index) = if imm8 {
            let (rm, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
            (rm, imm)
        } else {
            self.parse_modrm(op_size)?
        };
        // Move bit at index to carry
        // c = (dest >> index) & 1
        let mut result = SmallVec::new();
        result.push(mov(
            self.ctx.flag_c(),
            operand_and64(
                operand_rsh64(
                    dest.clone(),
                    index.clone(),
                ),
                self.ctx.const_1(),
            ),
        ));
        Ok(result)
    }

    fn btr(&self, imm8: bool) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let op_size = self.mem16_32();
        let (dest, index) = if imm8 {
            let (rm, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
            (rm, imm)
        } else {
            self.parse_modrm(op_size)?
        };
        // Move bit at index to carry, clear it
        // c = (dest >> index) & 1; dest &= !(1 << index)
        let mut result = SmallVec::new();
        result.push(mov(
            self.ctx.flag_c(),
            operand_and64(
                operand_rsh64(
                    dest.clone(),
                    index.clone(),
                ),
                self.ctx.const_1(),
            ),
        ));
        result.push(mov(
            dest.clone(),
            operand_and64(
                dest,
                operand_not64(
                    operand_lsh64(
                        self.ctx.const_1(),
                        index,
                    ),
                ),
            ),
        ));
        Ok(result)
    }

    fn xorps(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        Ok((0..4).map(|i| xor(self.xmm_variant(&dest, i), self.xmm_variant(&rm, i), false)).collect())
    }

    fn cvttss2si(&self) -> Result<OperationVec, Error> {
        if !self.has_prefix(0xf3) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        let op = make_arith_operation(
            dest_operand(&r),
            ArithOpType::FloatToInt,
            self.xmm_variant(&rm, 0),
            self.ctx.const_0(),
            false,
        );
        out.push(op);
        Ok(out)
    }

    fn cvtdq2ps(&self) -> Result<OperationVec, Error> {
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        for i in 0..4 {
            let op = make_arith_operation(
                dest_operand(&self.xmm_variant(&r, i)),
                ArithOpType::IntToFloat,
                self.xmm_variant(&rm, i),
                self.ctx.const_0(),
                false,
            );
            out.push(op);
        }
        Ok(out)
    }

    fn mov_sse_6e(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        out.push(mov(self.xmm_variant(&r, 0), rm));
        out.push(mov(self.xmm_variant(&r, 1), self.ctx.const_0()));
        out.push(mov(self.xmm_variant(&r, 2), self.ctx.const_0()));
        out.push(mov(self.xmm_variant(&r, 3), self.ctx.const_0()));
        Ok(out)
    }

    fn mov_sse_12_13(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        // movlpd
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let (src, dest) = match self.get(0) == 0x12 {
            true => (rm, r),
            false => (r, rm),
        };
        Ok((0..2).map(|i| mov(self.xmm_variant(&dest, i), self.xmm_variant(&src, i))).collect())
    }

    fn sse_move(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let (src, dest) = match self.get(0) {
            0x10 | 0x28 | 0x7e => (&rm, &r),
            _ => (&r, &rm),
        };
        let len = match self.get(0) {
            0x10 | 0x11 => match (self.has_prefix(0xf3), self.has_prefix(0xf2)) {
                // movss
                (true, false) => 1,
                // movsd
                (false, true) => 2,
                // movups, movupd
                (false, false) => 4,
                (true, true) => return Err(Error::UnknownOpcode(self.data.into())),
            },
            0x28 | 0x29 | 0x2b => 4,
            0x7e => match (self.has_prefix(0xf3), Va::SIZE) {
                (true, 4) | (_, 8) => 2,
                (false, 4) => 1,
                _ => unreachable!(),
            },
            0x7f => match self.has_prefix(0xf3) || self.has_prefix(0x66) {
                true => 4,
                false => 2,
            },
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        };
        Ok((0..len).map(|i| mov(self.xmm_variant(dest, i), self.xmm_variant(src, i))).collect())
    }

    fn mov_rm_to_xmm_128(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let (src, dest) = (rm, r);
        Ok((0..4).map(|i| mov(self.xmm_variant(&dest, i), self.xmm_variant(&src, i))).collect())
    }

    fn mov_sse_d6(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, src) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        out.push(mov(self.xmm_variant(&rm, 0), self.xmm_variant(&src, 0)));
        out.push(mov(self.xmm_variant(&rm, 1), self.xmm_variant(&src, 1)));
        if let OperandType::Xmm(_, _) = rm.ty {
            out.push(mov(self.xmm_variant(&rm, 2), self.ctx.const_0()));
            out.push(mov(self.xmm_variant(&rm, 3), self.ctx.const_0()));
        }
        Ok(out)
    }

    fn packed_shift_left(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        // dest.1 = (dest.1 << rm.0) | (dest.0 >> (32 - rm.0))
        // shl dest.0, rm.0
        // dest.3 = (dest.3 << rm.0) | (dest.2 >> (32 - rm.0))
        // shl dest.2, rm.0
        // Zero everything if rm.1 is set
        let mut out = SmallVec::new();
        out.push({
            let (low, high) = Operand::to_xmm_64(&dest, 0);
            let rm = Operand::to_xmm_32(&rm, 0);
            make_arith_operation(
                dest_operand(&low),
                ArithOpType::Or,
                operand_lsh(high, rm.clone()),
                operand_rsh(low, operand_sub(self.ctx.const_20(), rm)),
                false,
            )
        });
        out.push({
            let (low, _) = Operand::to_xmm_64(&dest, 0);
            let rm = Operand::to_xmm_32(&rm, 0);
            lsh(low, rm, false)
        });
        out.push({
            let (low, high) = Operand::to_xmm_64(&dest, 1);
            let rm = Operand::to_xmm_32(&rm, 0);
            make_arith_operation(
                dest_operand(&low),
                ArithOpType::Or,
                operand_lsh(high, rm.clone()),
                operand_rsh(low, operand_sub(self.ctx.const_20(), rm)),
                false,
            )
        });
        out.push({
            let (low, _) = Operand::to_xmm_64(&dest, 1);
            let rm = Operand::to_xmm_32(&rm, 0);
            lsh(low, rm, false)
        });
        let (_, high) = Operand::to_xmm_64(&rm, 0);
        let high_u32_set = operand_logical_not(operand_eq(high, self.ctx.const_0()));
        for i in 0..4 {
            let dest = Operand::to_xmm_32(&dest, i);
            out.push(Operation::Move(
                dest_operand(&dest),
                self.ctx.const_0(),
                Some(high_u32_set.clone()),
            ));
        }
        Ok(out)
    }

    fn packed_shift_right(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        // dest.0 = (dest.0 >> rm.0) | (dest.1 << (32 - rm.0))
        // shr dest.1, rm.0
        // dest.2 = (dest.2 >> rm.0) | (dest.3 << (32 - rm.0))
        // shr dest.3, rm.0
        // Zero everything if rm.1 is set
        let mut out = SmallVec::new();
        out.push({
            let (low, high) = Operand::to_xmm_64(&dest, 0);
            let rm = Operand::to_xmm_32(&rm, 0);
            make_arith_operation(
                dest_operand(&low),
                ArithOpType::Or,
                operand_rsh(low, rm.clone()),
                operand_lsh(high, operand_sub(self.ctx.const_20(), rm)),
                false,
            )
        });
        out.push({
            let (_, high) = Operand::to_xmm_64(&dest, 0);
            let rm = Operand::to_xmm_32(&rm, 0);
            rsh(high, rm, false)
        });
        out.push({
            let (low, high) = Operand::to_xmm_64(&dest, 1);
            let rm = Operand::to_xmm_32(&rm, 0);
            make_arith_operation(
                dest_operand(&low),
                ArithOpType::Or,
                operand_rsh(low, rm.clone()),
                operand_lsh(high, operand_sub(self.ctx.const_20(), rm)),
                false,
            )
        });
        out.push({
            let (_, high) = Operand::to_xmm_64(&dest, 1);
            let rm = Operand::to_xmm_32(&rm, 0);
            rsh(high, rm, false)
        });
        let (_, high) = Operand::to_xmm_64(&rm, 0);
        let high_u32_set = operand_logical_not(operand_eq(high, self.ctx.const_0()));
        for i in 0..4 {
            let dest = Operand::to_xmm_32(&dest, i);
            out.push(Operation::Move(
                dest_operand(&dest),
                self.ctx.const_0(),
                Some(high_u32_set.clone()),
            ));
        }
        Ok(out)
    }

    fn fpu_push(&self, out: &mut OperationVec) {
        // fdecstp
        out.push(Operation::Special(vec![0xd9, 0xf6]));
    }

    fn fpu_pop(&self, out: &mut OperationVec) {
        // fincstp
        out.push(Operation::Special(vec![0xd9, 0xf7]));
    }

    fn various_d8(&self) -> Result<OperationVec, Error> {
        let byte = self.get(1);
        let variant = (byte >> 3) & 0x7;
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        let st0 = self.ctx.register_fpu(0);
        match variant {
            // Fadd
            0 => {
                out.push(make_f32_operation(
                    dest_operand(&st0),
                    ArithOpType::Add,
                    st0,
                    rm,
                ));
            }
            // Fmul
            1 => {
                out.push(make_f32_operation(
                    dest_operand(&st0),
                    ArithOpType::Mul,
                    st0,
                    rm,
                ));
            }
            // Fsub
            4 => {
                out.push(make_f32_operation(
                    dest_operand(&st0),
                    ArithOpType::Sub,
                    st0,
                    rm,
                ));
            }
            // Fsubr
            5 => {
                out.push(make_f32_operation(
                    dest_operand(&st0),
                    ArithOpType::Sub,
                    rm,
                    st0,
                ));
            }
            // Fdiv
            6 => {
                out.push(make_f32_operation(
                    dest_operand(&st0),
                    ArithOpType::Div,
                    st0,
                    rm,
                ));
            }
            // Fdivr
            7 => {
                out.push(make_f32_operation(
                    dest_operand(&st0),
                    ArithOpType::Div,
                    rm,
                    st0,
                ));
            }
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        }
        Ok(out)
    }

    fn various_d9(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let byte = self.get(1);
        let variant = (byte >> 3) & 0x7;
        match variant {
            // Fld
            0 => {
                let (rm, _) = self.parse_modrm(MemAccessSize::Mem32)?;
                let mut out = SmallVec::new();
                self.fpu_push(&mut out);
                out.push(mov(self.ctx.register_fpu(0), x87_variant(self.ctx, rm, 1)));
                Ok(out)
            }
            // Fst/Fstp, as long as rm is mem
            2 | 3 => {
                let (rm, _) = self.parse_modrm(MemAccessSize::Mem32)?;
                let mut out = SmallVec::new();
                out.push(mov(rm, self.ctx.register_fpu(0)));
                if variant == 3 {
                    self.fpu_pop(&mut out);
                }
                Ok(out)
            }
            // Fincstp, fdecstp
            6 if byte == 0xf6 || byte == 0xf7 => {
                let mut out = SmallVec::new();
                out.push(Operation::Special(self.data.into()));
                Ok(out)
            }
            // Fstenv
            6 => {
                let mem_size = self.mem16_32();
                let mem_bytes = match mem_size {
                    MemAccessSize::Mem16 => 2,
                    _ => 4,
                };
                let (rm, _) = self.parse_modrm(mem_size)?;
                let mut out = SmallVec::new();
                if let Some(mem) = rm.if_memory() {
                    out.extend((0..10).map(|i| {
                        let address = operand_add(
                            mem.address.clone(),
                            self.ctx.constant(i * mem_bytes),
                        );
                        let dest = mem_variable_rc(mem_size, address);
                        mov(dest, self.ctx.undefined_rc())
                    }));
                }
                Ok(out)
            }
            // Fstcw
            7 => {
                let (rm, _) = self.parse_modrm(MemAccessSize::Mem16)?;
                let mut out = SmallVec::new();
                if rm.if_memory().is_some() {
                    out.push(mov(rm.clone(), self.ctx.undefined_rc()));
                }
                Ok(out)
            }
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        }
    }

    fn various_fe_ff(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand_helpers::*;
        let variant = (self.get(1) >> 3) & 0x7;
        let is_64 = Va::SIZE == 8;
        let op_size = match self.get(0) & 0x1 {
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
        let mut out = SmallVec::new();
        match variant {
            0 | 1 => {
                let is_inc = variant == 0;
                out.push(match is_inc {
                    true => add(rm, self.ctx.const_1(), is_64),
                    false => sub(rm, self.ctx.const_1(), is_64),
                });
            }
            2 | 3 => out.push(Operation::Call(rm)),
            4 | 5 => out.push(Operation::Jump { condition: self.ctx.const_1(), to: rm }),
            6 => {
                if Va::SIZE == 4 {
                    let esp = self.ctx.register(4);
                    out.push(sub(esp.clone(), self.ctx.const_4(), is_64));
                    out.push(mov(mem32(esp), rm));
                } else {
                    let rsp = self.ctx.register64(4);
                    out.push(sub(rsp.clone(), self.ctx.const_8(), is_64));
                    out.push(mov(mem64(rsp), rm));
                }
            }
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        }
        Ok(out)
    }

    fn bitwise_with_imm_op(&self) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (_, _, imm) = self.parse_modrm_imm(op_size, MemAccessSize::Mem8)?;
        if imm.ty == OperandType::Constant(0) {
            return Ok(SmallVec::new());
        }
        let op_gen: &dyn ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            7 => &SAR_OPS,
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        };
        self.generic_arith_with_imm_op(op_gen, MemAccessSize::Mem8)
    }

    fn bitwise_compact_op(&self) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _) = self.parse_modrm(op_size)?;
        let shift_count = match self.get(0) & 2 {
            0 => self.ctx.const_1(),
            _ => self.ctx.reg_variable_size(Register(1), operand::MemAccessSize::Mem8),
        };
        let op_gen: &dyn ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            7 => &SAR_OPS,
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        };
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        op_gen.operation(rm.clone(), shift_count.clone(), &mut out, is_64);
        Ok(out)
    }

    fn signed_multiply_rm_imm(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let imm_size = match self.get(0) & 0x2 {
            2 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let op_size = self.mem16_32();
        let (rm, r, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        // TODO flags, imul only sets c and o on overflow
        // TODO Signed mul isn't sensible if it doesn't contain operand size
        // I don't think any of these sizes are correct but w/e
        let mut out = SmallVec::new();
        if Va::SIZE == 4 {
            if op_size != MemAccessSize::Mem32 {
                out.push(mov(r, operand_signed_mul(rm, imm)));
            } else {
                out.push(mov(r, operand_mul(rm, imm)));
            }
        } else {
            if op_size != MemAccessSize::Mem64 {
                out.push(mov(r, operand_signed_mul64(rm, imm)));
            } else {
                out.push(mov(r, operand_mul64(rm, imm)));
            }
        }
        Ok(out)
    }

    fn shld_imm(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let (hi, low, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let mut out = SmallVec::new();
        let imm = Operand::simplified(operand_and(imm, self.ctx.const_1f()));
        if imm.ty != OperandType::Constant(0) {
            // TODO flags
            out.push(
                mov(
                    hi.clone(),
                    operand_or(
                        operand_lsh(
                            hi,
                            imm.clone(),
                        ),
                        operand_rsh(
                            low,
                            operand_sub(
                                self.ctx.const_20(),
                                imm.clone(),
                            )
                        ),
                    ),
                ),
            );
        }
        Ok(out)
    }

    fn shrd_imm(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let (low, hi, imm) = self.parse_modrm_imm(self.mem16_32(), MemAccessSize::Mem8)?;
        let mut out = SmallVec::new();
        let imm = Operand::simplified(operand_and(imm, self.ctx.const_1f()));
        if imm.ty != OperandType::Constant(0) {
            // TODO flags
            out.push(
                mov(
                    low.clone(),
                    operand_or(
                        operand_rsh(
                            low,
                            imm.clone(),
                        ),
                        operand_lsh(
                            hi,
                            operand_sub(
                                self.ctx.const_20(),
                                imm.clone(),
                            )
                        ),
                    ),
                ),
            );
        }
        Ok(out)
    }

    fn imul_normal(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let size = self.mem16_32();
        let (rm, r) = self.parse_modrm(size)?;
        // TODO flags, imul only sets c and o on overflow
        let mut out = SmallVec::new();
        // Signed multiplication should be different only when result is being sign extended.
        if size.bits() != Va::SIZE * 8 {
            // TODO Signed mul should actually specify bit size
            out.push(signed_mul(r, rm, Va::SIZE == 8));
        } else {
            out.push(mul(r, rm, Va::SIZE == 8));
        }
        Ok(out)
    }

    fn arith_with_imm_op(&self) -> Result<OperationVec, Error> {
        let op_gen: &dyn ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
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
        let imm_size = match self.get(0) & 0x3 {
            0 | 2 | 3 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        self.generic_arith_with_imm_op(op_gen, imm_size)
    }

    fn generic_arith_with_imm_op(
        &self,
        op_gen: &dyn ArithOperationGenerator,
        imm_size: MemAccessSize,
    ) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        op_gen.operation(rm.clone(), imm.clone(), &mut out, is_64);
        Ok(out)
    }

    fn eax_imm_cmp<F>(
        &self,
        ops: F,
    ) -> Result<OperationVec, Error>
    where F: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec, bool),
    {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let eax = self.reg_variable_size(Register(0), op_size);
        let imm = read_variable_size_32(self.slice(1), op_size)?;
        let val = self.ctx.constant64(imm);
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        ops(eax, val, &mut out, is_64);
        Ok(out)
    }


    fn generic_cmp_op<F>(
        &self,
        ops: F,
    ) -> Result<OperationVec, Error>
    where F: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec, bool),
    {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let rm_left = self.get(0) & 0x3 < 2;
        let (rm, r) = self.parse_modrm(op_size)?;
        let (left, right) = match rm_left {
            true => (rm, r),
            false =>  (r, rm),
        };
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE == 8;
        ops(left, right, &mut out, is_64);
        Ok(out)
    }

    fn pushpop_reg_op(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let byte = self.get(0);
        let is_push = byte < 0x58;
        let reg = if self.prefixes.rex_prefix & 0x1 == 0 {
            byte & 0x7
        } else {
            8 + (byte & 0x7)
        };
        let mut out = SmallVec::new();
        let is_64 = Va::SIZE != 4;
        let esp = if Va::SIZE == 4 {
            self.ctx.register(4)
        } else {
            self.ctx.register64(4)
        };
        match is_push {
            true => {
                if Va::SIZE == 4 {
                    out.push(sub(esp.clone(), self.ctx.const_4(), is_64));
                    out.push(mov(mem32(esp), self.ctx.register(reg)));
                } else {
                    out.push(sub(esp.clone(), self.ctx.const_8(), is_64));
                    out.push(mov(mem64(esp), self.ctx.register64(reg)));
                }
            }
            false => {
                if Va::SIZE == 4 {
                    out.push(mov(self.ctx.register(reg), mem32(esp.clone())));
                    out.push(add(esp, self.ctx.const_4(), is_64));
                } else {
                    out.push(mov(self.ctx.register64(reg), mem64(esp.clone())));
                    out.push(add(esp, self.ctx.const_8(), is_64));
                }
            }
        }
        Ok(out)
    }

    fn push_imm(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use crate::operand::operand_helpers::*;
        let imm_size = match self.get(0) {
            0x68 => self.mem16_32(),
            _ => MemAccessSize::Mem8,
        };
        let constant = read_variable_size_32(self.slice(1), imm_size)? as u32;
        let mut out = SmallVec::new();
        let esp = if Va::SIZE == 4 {
            self.ctx.register(4)
        } else {
            self.ctx.register64(4)
        };
        out.push(sub(esp.clone(), self.ctx.const_4(), Va::SIZE == 8));
        out.push(mov(mem32(esp), self.ctx.constant(constant)));
        Ok(out)
    }
}

impl<'a, 'exec: 'a> InstructionOpsState<'a, 'exec, VirtualAddress32> {
    fn conditional_jmp(&self, op_size: MemAccessSize) -> Result<OperationVec, Error> {
        let offset = read_variable_size_signed(self.slice(1), op_size)?;
        let to = self.ctx.constant((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.condition(), to });
        Ok(out)
    }

    fn short_jmp(&self) -> Result<OperationVec, Error> {
        let offset = read_variable_size_signed(self.slice(1), MemAccessSize::Mem8)?;
        let to = self.ctx.constant((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(out)
    }

    fn call_op(&self) -> Result<OperationVec, Error> {
        let offset = read_u32(self.slice(1))?;
        let to = self.ctx.constant((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Call(to));
        Ok(out)
    }

    fn jump_op(&self) -> Result<OperationVec, Error> {
        let offset = read_u32(self.slice(1))?;
        let to = self.ctx.constant((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(out)
    }
}

impl<'a, 'exec: 'a> InstructionOpsState<'a, 'exec, VirtualAddress64> {
    fn conditional_jmp(&self, op_size: MemAccessSize) -> Result<OperationVec, Error> {
        let offset = read_variable_size_signed(self.slice(1), op_size)? as i32 as i64 as u64;
        let to = self.ctx.constant64((self.address.0 + self.len() as u64).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.condition(), to });
        Ok(out)
    }

    fn short_jmp(&self) -> Result<OperationVec, Error> {
        let offset = read_variable_size_signed(self.slice(1), MemAccessSize::Mem8)?
            as i32 as i64 as u64;
        let to = self.ctx.constant64((self.address.0 + self.len() as u64).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(out)
    }

    fn call_op(&self) -> Result<OperationVec, Error> {
        let offset = read_u32(self.slice(1))? as i32 as i64 as u64;
        let to = self.ctx.constant64((self.address.0 + self.len() as u64).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Call(to));
        Ok(out)
    }

    fn jump_op(&self) -> Result<OperationVec, Error> {
        let offset = read_u32(self.slice(1))? as i32 as i64 as u64;
        let to = self.ctx.constant64((self.address.0 + self.len() as u64).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.ctx.const_1(), to });
        Ok(out)
    }
}

/// Checks if r is a register, and rm is the equivalent short register
fn is_rm_short_r_register(rm: &Rc<Operand>, r: &Rc<Operand>) -> bool {
    use crate::operand::OperandType::*;
    match (&rm.ty, &r.ty) {
        (&Register8Low(s), &Register(l)) | (&Register8Low(s), &Register16(l)) => l == s,
        (&Register16(s), &Register(l)) => l == s,
        _ => false,
    }
}

trait ArithOperationGenerator {
    fn operation(&self, _: Rc<Operand>, _: Rc<Operand>, _: &mut OperationVec, _64bit: bool);
}

macro_rules! arith_op_generator {
    ($stname: ident, $name:ident, $op:ident) => {
        struct $name;
        impl ArithOperationGenerator for $name {
            fn operation(&self, a: Rc<Operand>, b: Rc<Operand>, c: &mut OperationVec, d: bool) {
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

/// Most opcodes in 64-bit mode only take sign-extended 32-bit immediates.
fn read_variable_size_32(val: &[u8], size: MemAccessSize) -> Result<u64, Error> {
    match size {
        MemAccessSize::Mem64 => read_u32(val).map(|x| x as i32 as i64 as u64),
        MemAccessSize::Mem32 => read_u32(val).map(|x| u64::from(x)),
        MemAccessSize::Mem16 => read_u16(val).map(|x| u64::from(x)),
        MemAccessSize::Mem8 => read_u8(val).map(|x| u64::from(x)),
    }
}

fn read_variable_size_64(val: &[u8], size: MemAccessSize) -> Result<u64, Error> {
    match size {
        MemAccessSize::Mem64 => read_u64(val),
        MemAccessSize::Mem32 => read_u32(val).map(|x| u64::from(x)),
        MemAccessSize::Mem16 => read_u16(val).map(|x| u64::from(x)),
        MemAccessSize::Mem8 => read_u8(val).map(|x| u64::from(x)),
    }
}

fn read_variable_size_signed(val: &[u8], size: MemAccessSize) -> Result<u32, Error> {
    match size {
        MemAccessSize::Mem32 | MemAccessSize::Mem64 => read_u32(val),
        MemAccessSize::Mem16 => read_u16(val).map(|x| x as i16 as u32),
        MemAccessSize::Mem8 => read_u8(val).map(|x| x as i8 as u32),
    }
}

fn read_u64(mut val: &[u8]) -> Result<u64, Error> {
    use byteorder::{LE, ReadBytesExt};
    val.read_u64::<LE>().map_err(|_| Error::InternalDecodeError)
}

fn read_u32(mut val: &[u8]) -> Result<u32, Error> {
    use byteorder::{LE, ReadBytesExt};
    val.read_u32::<LE>().map_err(|_| Error::InternalDecodeError)
}

fn read_u16(mut val: &[u8]) -> Result<u16, Error> {
    use byteorder::{LE, ReadBytesExt};
    val.read_u16::<LE>().map_err(|_| Error::InternalDecodeError)
}

fn read_u8(mut val: &[u8]) -> Result<u8, Error> {
    use byteorder::{ReadBytesExt};
    val.read_u8().map_err(|_| Error::InternalDecodeError)
}

pub mod operation_helpers {
    use std::rc::Rc;

    use crate::operand::ArithOpType::*;
    use crate::operand::{
        Flag, Operand, OperandType, MemAccessSize, ArithOpType, ArithOperand,
    };
    use crate::operand::operand_helpers::*;
    use super::{
        dest_operand, make_arith_operation, Operation, OperationVec,
    };

    pub fn mov(dest: Rc<Operand>, from: Rc<Operand>) -> Operation {
        Operation::Move(dest_operand(&dest), from, None)
    }

    pub fn mov_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, _64: bool) {
        if dest != rhs {
            out.push(mov(dest, rhs));
        }
    }

    pub fn lea_ops(rhs: Rc<Operand>, dest: Rc<Operand>, out: &mut OperationVec, _64: bool) {
        if let OperandType::Memory(ref mem) = rhs.ty {
            out.push(mov(dest, mem.address.clone()));
        }
    }

    pub fn add(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Add, dest, rhs, is_64)
    }

    pub fn add_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Add, dest.clone(), rhs.clone(), is_64));
        out.push(add(dest, rhs, is_64));
    }

    pub fn adc_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        let rhs = operand_add64(rhs, flag_c());
        out.push(flags(Add, dest.clone(), rhs.clone(), is_64));
        out.push(add(dest, rhs, is_64));
    }

    pub fn sub(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Sub, dest, rhs, is_64)
    }

    pub fn sub_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Sub, dest.clone(), rhs.clone(), is_64));
        out.push(sub(dest, rhs, is_64));
    }

    pub fn cmp_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Sub, dest.clone(), rhs.clone(), is_64));
    }

    pub fn sbb_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        let rhs = operand_add64(rhs, flag_c());
        out.push(flags(Sub, dest.clone(), rhs.clone(), is_64));
        out.push(sub(dest, rhs, is_64));
    }

    pub fn mul(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Mul, dest, rhs, is_64)
    }

    pub fn signed_mul(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), SignedMul, dest, rhs, is_64)
    }

    pub fn xor(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Xor, dest, rhs, is_64)
    }

    pub fn xor_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Xor, dest.clone(), rhs.clone(), is_64));
        out.push(xor(dest, rhs, is_64));
    }

    pub fn rol_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        // rol(x, y) == (x << y) | (x >> (32 - y))
        let dest_operand = dest_operand(&dest);
        let left;
        let right;
        if is_64 {
            left = operand_lsh64(dest.clone(), rhs.clone());
            right = operand_rsh64(dest, operand_sub64(constval(64), rhs));
        } else {
            left = operand_lsh(dest.clone(), rhs.clone());
            right = operand_rsh(dest, operand_sub(constval(32), rhs));
        }
        // TODO set overflow if 1bit??
        out.push(make_arith_operation(dest_operand, Or, left, right, is_64));
    }

    pub fn ror_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        // ror(x, y) == (x >> y) | (x << (32 - y))
        let dest_operand = dest_operand(&dest);
        let left;
        let right;
        if is_64 {
            left = operand_rsh64(dest.clone(), rhs.clone());
            right = operand_lsh64(dest, operand_sub64(constval(64), rhs));
        } else {
            left = operand_rsh(dest.clone(), rhs.clone());
            right = operand_lsh(dest, operand_sub(constval(32), rhs));
        }
        // TODO set overflow if 1bit??
        out.push(make_arith_operation(dest_operand, Or, left, right, is_64));
    }

    pub fn lsh(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Lsh, dest, rhs, is_64)
    }

    pub fn lsh_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Lsh, dest.clone(), rhs.clone(), is_64));
        out.push(make_arith_operation(dest_operand(&dest), Lsh, dest, rhs, is_64));
    }

    pub fn rsh(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Rsh, dest, rhs, is_64)
    }

    pub fn rsh_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Rsh, dest.clone(), rhs.clone(), is_64));
        out.push(rsh(dest, rhs, is_64));
    }

    pub fn sar_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        // TODO flags, using IntToFloat to cheat
        out.push(flags(IntToFloat, dest.clone(), rhs.clone(), is_64));
        if is_64 {
            let is_positive = operand_eq64(
                operand_and(constval64(0x8000_0000_0000_0000), dest.clone()),
                constval(0),
            );
            out.push(Operation::Move(
                dest_operand(&dest),
                operand_or(operand_rsh64(dest.clone(), rhs.clone()), constval64(0x8000_0000_0000_0000)),
                Some(operand_logical_not(is_positive.clone())),
            ));
            out.push(Operation::Move(
                dest_operand(&dest),
                operand_rsh64(dest, rhs),
                Some(is_positive),
            ));
        } else {
            let is_positive = operand_eq(
                operand_and(constval(0x8000_0000), dest.clone()),
                constval(0),
            );
            out.push(Operation::Move(
                dest_operand(&dest),
                operand_or(operand_rsh(dest.clone(), rhs.clone()), constval(0x8000_0000)),
                Some(operand_logical_not(is_positive.clone())),
            ));
            out.push(Operation::Move(
                dest_operand(&dest),
                operand_rsh(dest, rhs),
                Some(is_positive),
            ));
        }
    }

    pub fn or(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), Or, dest, rhs, is_64)
    }

    pub fn or_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(Or, dest.clone(), rhs.clone(), is_64));
        out.push(or(dest, rhs, is_64));
    }

    pub fn and(dest: Rc<Operand>, rhs: Rc<Operand>, is_64: bool) -> Operation {
        make_arith_operation(dest_operand(&dest), And, dest, rhs, is_64)
    }

    pub fn and_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(And, dest.clone(), rhs.clone(), is_64));
        out.push(and(dest, rhs, is_64));
    }

    pub fn test_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec, is_64: bool) {
        out.push(flags(And, dest.clone(), rhs.clone(), is_64));
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
    Swap(DestOperand, DestOperand),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DestOperand {
    Register(Register),
    Register16(Register),
    Register8High(Register),
    Register8Low(Register),
    Register64(Register),
    PairEdxEax,
    Xmm(u8, u8),
    Fpu(u8),
    Flag(Flag),
    Memory(MemAccess),
}

impl DestOperand {
    pub fn from_oper(val: &Operand) -> DestOperand {
        dest_operand(val)
    }
}

fn dest_operand(val: &Operand) -> DestOperand {
    use crate::operand::OperandType::*;
    match val.ty {
        Register(x) => DestOperand::Register(x),
        Register16(x) => DestOperand::Register16(x),
        Register8High(x) => DestOperand::Register8High(x),
        Register8Low(x) => DestOperand::Register8Low(x),
        Register64(x) => DestOperand::Register64(x),
        Pair(ref hi, ref low) => {
            assert_eq!(hi.ty, Register(crate::operand::Register(2)));
            assert_eq!(low.ty, Register(crate::operand::Register(0)));
            DestOperand::PairEdxEax
        }
        Xmm(x, y) => DestOperand::Xmm(x, y),
        Fpu(x) => DestOperand::Fpu(x),
        Flag(x) => DestOperand::Flag(x),
        Memory(ref x) => DestOperand::Memory(x.clone()),
        ref x => panic!("Invalid value for converting Operand -> DestOperand: {:?}", x),
    }
}

impl From<DestOperand> for Operand {
    fn from(val: DestOperand) -> Operand {
        use crate::operand::operand_helpers::*;
        use crate::operand::OperandType::*;
        let ty = match val {
            DestOperand::Register(x) => Register(x),
            DestOperand::Register16(x) => Register16(x),
            DestOperand::Register8High(x) => Register8High(x),
            DestOperand::Register8Low(x) => Register8Low(x),
            DestOperand::Register64(x) => Register64(x),
            DestOperand::PairEdxEax => Pair(operand_register(2), operand_register(0)),
            DestOperand::Xmm(x, y) => Xmm(x, y),
            DestOperand::Fpu(x) => Fpu(x),
            DestOperand::Flag(x) => Flag(x),
            DestOperand::Memory(x) => Memory(x),
        };
        Operand::new_not_simplified(ty)
    }
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
        let mut disasm = Disassembler32::new(&buf[..], 0, VirtualAddress(0));
        let ins = disasm.next(&ctx).unwrap();
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
        let mut disasm = Disassembler32::new(&buf[..], 0, VirtualAddress(0));
        let ins = disasm.next(&ctx).unwrap();
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
                let d: Operand = d.into();
                assert_eq!(Operand::simplified(d.into()), Operand::simplified(dest));
                assert_eq!(f, operand_register(0));
                assert_eq!(cond, None);
            }
            _ => panic!("Unexpected op {:?}", op),
        }
    }
}

