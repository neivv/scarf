use std::rc::Rc;

use hex_slice::AsHex;
use lde::{self, InsnSet};

use ::{VirtualAddress};
use operand::{
    ArithOpType, Flag, MemAccess, MemAccessSize, Operand, OperandContext, OperandType, Register
};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        UnknownOpcode(op: Vec<u8>) {
            description("Unknown opcode")
            display("Unknown opcode {:02x}", op.as_hex())
        }
        End(addr: VirtualAddress) {
            description("End of file")
        }
        // The preceding instruction's operations will be given before this error.
        Branch(addr: VirtualAddress) {
            description("Reached a branch")
        }
        InternalDecodeError {
            description("Internal decode error")
        }
    }
}

pub struct Disassembler<'a> {
    buf: &'a [u8],
    pos: usize,
    virtual_address: VirtualAddress,
    finishing_instruction_pos: Option<usize>,
}

impl<'a> Disassembler<'a> {
    pub fn new(buf: &[u8], pos: usize, address: VirtualAddress) -> Disassembler {
        assert!(pos < buf.len());
        Disassembler {
            buf,
            pos,
            virtual_address: address,
            finishing_instruction_pos: None,
        }
    }

    pub fn next<'b, 'c>(&'b mut self, ctx: &'c OperandContext) -> Result<Instruction<'a, 'c>, Error> {
        if let Some(finishing_instruction_pos) = self.finishing_instruction_pos {
            return Err(Error::Branch(self.virtual_address + finishing_instruction_pos as u32));
        }
        let length = lde::x86::ld(&self.buf[self.pos..]) as usize;
        if length == 0 {
            if self.pos == self.buf.len() {
                return Err(Error::End(self.virtual_address + self.pos as u32))
            } else {
                return Err(Error::UnknownOpcode(self.buf[self.pos..self.pos + 1].into()));
            }
        }
        let ins = Instruction {
            address: VirtualAddress(self.virtual_address.0 + self.pos as u32),
            data: &self.buf[self.pos..self.pos + length],
            ctx: ctx,
        };
        if ins.is_finishing() {
            self.finishing_instruction_pos = Some(self.pos);
            self.buf = &self.buf[..self.pos + length];
            self.pos = self.buf.len();
        } else {
            self.pos += length;
        }
        Ok(ins)
    }

    pub fn address(&self) -> VirtualAddress {
        return self.virtual_address + self.pos as u32
    }
}

pub struct Instruction<'a, 'b> {
    address: VirtualAddress,
    data: &'a [u8],
    ctx: &'b OperandContext,
}

impl<'a, 'b> Instruction<'a, 'b> {
    pub fn ops<'c>(&'c self) -> InstructionOps<'a, 'b> {
        InstructionOps::new(self.address, self.data, self.ctx)
    }

    fn is_finishing(&self) -> bool {
        self.ops().take_while(|x| !x.is_err()).flat_map(|op| op.ok()).any(|op| match op {
            Operation::Jump { .. } => true,
            Operation::Return(..) => true,
            _ => false,
        })
    }
}

#[derive(Copy, Clone)]
struct InstructionPrefixes {
    prefix_66: bool,
    prefix_67: bool,
    prefix_f2: bool,
    prefix_f3: bool,
}

pub struct InstructionOps<'a, 'exec: 'a> {
    inner: InstructionOpsState<'a, 'exec>,
    next_fn: Box<
        FnMut(&mut InstructionOpsState) -> Option<Result<Operation, Error>> + 'static
    >,
}

struct InstructionOpsState<'a, 'exec: 'a> {
    address: VirtualAddress,
    data: &'a [u8],
    prefixes: InstructionPrefixes,
    pos: u8,
    len: u8,
    ctx: &'exec OperandContext,
}

impl<'a, 'exec: 'a> Iterator for InstructionOps<'a, 'exec> {
    type Item = Result<Operation, Error>;

    fn next(&mut self) -> Option<Result<Operation, Error>> {
        let result = (self.next_fn)(&mut self.inner);
        self.inner.pos = self.inner.pos.checked_add(1).unwrap();
        result
    }
}

// Type hints the closure
fn ins_next<V>(val: V) ->
    Box<FnMut(&mut InstructionOpsState) -> Option<Result<Operation, Error>> + 'static>
where V: FnMut(&mut InstructionOpsState) -> Option<Result<Operation, Error>> + 'static,
{
    Box::new(val)
}

impl<'a, 'exec: 'a> InstructionOps<'a, 'exec> {
    fn new<'b, 'e>(
        address: VirtualAddress,
        data: &'b [u8],
        ctx: &'e OperandContext,
    ) -> InstructionOps<'b, 'e> {
        use self::Error::*;
        use self::operation_helpers::*;
        use operand::operand_helpers::*;

        let is_prefix_byte = |byte| match byte {
            0x64 => true, // TODO fs segment is not handled
            0x65 => true, // TODO gs segment is not handled
            0x66 => true,
            0x67 => true,
            0xf2 => true,
            0xf3 => true,
            _ => false,
        };
        let mut prefixes = InstructionPrefixes {
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
        let first_byte = data[0];
        let next_fn = if !is_ext {
            match first_byte {
                0x00 ... 0x03 | 0x04 ... 0x05 | 0x08 ... 0x0b | 0x0c ... 0x0d | 0x20 ... 0x23 |
                0x24 ... 0x25 | 0x28 ... 0x2b | 0x2c ... 0x2d | 0x30 ... 0x33 | 0x34 ... 0x35 |
                0x88 ... 0x8b | 0x8d => {
                    // Avoid ridiculous generic binary bloat
                    let (ops, flags, flags_post):
                        (fn(_, _, _) -> _, fn(_, _, _) -> _, fn(_, _, _) -> _) = match first_byte {
                        0x00 ... 0x05 => (add_ops, add_flags, result_flags),
                        0x08 ... 0x0d => (or_ops, zero_carry_oflow, result_flags),
                        0x20 ... 0x25 => (and_ops, zero_carry_oflow, result_flags),
                        0x28 ... 0x2d => (sub_ops, zero_carry_oflow, result_flags),
                        0x30 ... 0x35 => (xor_ops, zero_carry_oflow, result_flags),
                        0x88 ... 0x8b => (mov_ops, |_, _, s| Err(s), |_, _, _| None),
                        0x8d | _ => (lea_ops, |_, _, s| Err(s), |_, _, _| None),
                    };
                    let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
                    ins_next(move |s| if eax_imm_arith {
                        s.eax_imm_arith(ops, flags, flags_post)
                    } else {
                        s.generic_arith_op(ops, flags, flags_post)
                    })
                }

                0x38 ... 0x3b => ins_next(|s| s.generic_cmp_op(operand_sub, sub_flags, result_flags)),
                0x3c ... 0x3d => ins_next(|s| s.eax_imm_cmp(operand_sub, sub_flags, result_flags)),
                0x40 ... 0x4f => ins_next(|s| s.inc_dec_op()),
                0x50 ... 0x5f => ins_next(|s| s.pushpop_reg_op()),
                0x68 | 0x6a => ins_next(|s| s.push_imm()),
                0x69 | 0x6b => ins_next(|s| s.signed_multiply_rm_imm()),
                0x70 ... 0x7f => ins_next(|s| s.conditional_jmp(MemAccessSize::Mem8)),
                0x80 ... 0x83 => ins_next(|s| s.arith_with_imm_op()),
                // Test
                0x84 ... 0x85 => ins_next(|s| s.generic_cmp_op(operand_and, zero_carry_oflow, result_flags)),
                0x86 ... 0x87 => ins_next(|s| s.xchg()),
                0x90 => ins_next(|_| None),
                // Cwde
                0x98 => ins_next(|s| match s.pos {
                    0 => Some(Ok(and(operand_register(0), constval(0xffff)))),
                    1 => {
                        let eax = operand_register(0);
                        let signed_max = constval(0x7fff);
                        let compare = ArithOpType::GreaterThan(eax.clone(), signed_max);
                        let cond = Operand::new_not_simplified_rc(OperandType::Arithmetic(compare));
                        let negative_sign_extend = operand_or(eax.clone(), constval(0xffff0000));
                        Some(Ok(Operation::Move((*eax).clone().into(), negative_sign_extend, Some(cond))))
                    }
                    _ => None,
                }),
                // Cdq
                0x99 => ins_next(|s| match s.pos {
                    0 => Some(Ok(mov(operand_register(2), constval(0)))),
                    1 => {
                        let eax = operand_register(0);
                        let edx = operand_register(2);
                        let signed_max = constval(0x7fffffff);
                        let compare = ArithOpType::GreaterThan(eax, signed_max);
                        let cond = Operand::new_not_simplified_rc(OperandType::Arithmetic(compare));
                        Some(Ok(Operation::Move((*edx).clone().into(), constval(!0), Some(cond))))
                    }
                    _ => None,
                }),
                0xa0 ... 0xa3 => ins_next(|s| s.move_mem_eax()),
                0xa8 ... 0xa9 => ins_next(|s| s.eax_imm_cmp(operand_and, zero_carry_oflow, result_flags)),
                0xb0 ... 0xbf => ins_next(|s| s.move_const_to_reg()),
                0xc0 ... 0xc1 => ins_next(|s| s.bitwise_with_imm_op()),
                0xc2 ... 0xc3 => {
                    let stack_pop_size = match data[0] {
                        0xc2 => match read_u16(&data[1..]) {
                            Err(_) => 0,
                            Ok(o) => o as u32,
                        },
                        _ => 0,
                    };
                    ins_next(move |s| match s.pos {
                        0 => Some(Ok(Operation::Return(stack_pop_size))),
                        _ => None,
                    })
                }
                0xc6 ... 0xc7 => ins_next(|s| s.generic_arith_with_imm_op(&MOV_OPS, match s.get(0) {
                    0xc6 => MemAccessSize::Mem8,
                    _ => s.mem16_32(),
                })),
                0xd0 ... 0xd3 => ins_next(|s| s.bitwise_compact_op()),
                0xe8 => ins_next(|s| s.call_op()),
                0xe9 => ins_next(|s| s.jump_op()),
                0xeb => ins_next(|s| s.short_jmp()),
                0xf6 | 0xf7 => ins_next(|s| s.various_f7()),
                0xf8 ... 0xfd => {
                    let flag = match first_byte {
                        0xf8 ... 0xf9 => Flag::Carry,
                        _ => Flag::Direction,
                    };
                    let state = first_byte & 0x1 == 1;
                    ins_next(move |s| s.flag_set(flag, state))
                }
                0xfe ... 0xff => ins_next(|s| s.various_fe_ff()),
                _ => ins_next(|s| Some(Err(UnknownOpcode(s.data.into()))))
            }
        } else {
            match first_byte {
                0x11 => ins_next(|s| s.mov_sse_11()),
                // nop
                0x1f => ins_next(|_| None),
                // rdtsc
                0x31 => ins_next(|s| match s.pos {
                    0 => Some(Ok(mov(operand_register(0), s.ctx.undefined_rc()))),
                    1 => Some(Ok(mov(operand_register(2), s.ctx.undefined_rc()))),
                    _ => None,
                }),
                0x40 ... 0x4f => ins_next(|s| s.cmov()),
                0x57 => ins_next(|s| s.xorps()),
                0x6e => ins_next(|s| s.mov_sse_6e()),
                0x7e => ins_next(|s| s.mov_sse_7e()),
                0x80 ... 0x8f => ins_next(|s| s.conditional_jmp(s.mem16_32())),
                0x90 ... 0x9f => ins_next(|s| s.conditional_set()),
                0xaf => ins_next(|s| s.imul_normal()),
                0xb6 ... 0xb7 => ins_next(|s| s.movzx()),
                0xbe ... 0xbf => ins_next(|s| s.movsx()),
                0xd3 => ins_next(|s| s.packed_shift_right()),
                0xd6 => ins_next(|s| s.mov_sse_d6()),
                _ => ins_next(|s| {
                    let mut bytes = vec![0xf];
                    bytes.extend(s.data);
                    Some(Err(UnknownOpcode(bytes)))
                })
            }
        };
        InstructionOps {
            inner: InstructionOpsState {
                address,
                data,
                prefixes,
                pos: 0,
                len: instruction_len as u8,
                ctx,
            },
            next_fn,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn address(&self) -> VirtualAddress {
        self.inner.address
    }
}

impl<'a, 'exec: 'a> InstructionOpsState<'a, 'exec> {
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
        match self.has_prefix(0x66) {
            true => MemAccessSize::Mem16,
            false => MemAccessSize::Mem32,
        }
    }

    fn imm16_32(&self) -> MemAccessSize {
        // TODO 16-bit override
        MemAccessSize::Mem32
    }

    fn parse_modrm_xmm(&self, word_id: u8) -> Result<(Operand, Operand, usize), Error> {
        use operand::operand_helpers::*;

        let (mut rm, _, bytes) = self.parse_modrm(MemAccessSize::Mem32)?;
        let modrm = self.get(1);
        let register = (modrm >> 3) & 0x7;
        let r = Operand::new_xmm(register, word_id);
        rm.ty = match rm.ty {
            OperandType::Register(r) => OperandType::Xmm(r.0, word_id),
            OperandType::Memory(mem) => match word_id {
                0 => OperandType::Memory(mem),
                1 => OperandType::Memory(MemAccess {
                    address: operand_add(mem.address, constval(4)),
                    size: MemAccessSize::Mem32,
                }),
                2 => OperandType::Memory(MemAccess {
                    address: operand_add(mem.address, constval(8)),
                    size: MemAccessSize::Mem32,
                }),
                _ => OperandType::Memory(MemAccess {
                    address: operand_add(mem.address, constval(12)),
                    size: MemAccessSize::Mem32,
                }),
            },
            x => x,
        };
        Ok((rm, r, bytes))
    }

    /// Returns (rm, r, modrm_size)
    fn parse_modrm(
        &self,
        op_size: MemAccessSize
    ) -> Result<(Operand, Operand, usize), Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;

        let modrm = self.get(1);
        let register = (modrm >> 3) & 0x7;
        let rm_val = modrm & 0x7;
        let r = Operand::reg_variable_size(Register(register), op_size);
        let (rm, size) = match (modrm >> 6) & 0x3 {
            0 => match rm_val {
                4 => self.parse_sib(0, op_size)?,
                5 => (mem32_norc(constval(read_u32(self.slice(2))?)), 6),
                reg => (mem_variable(op_size, operand_register(reg)), 2),
            },
            1 => match rm_val {
                4 => self.parse_sib(1, op_size)?,
                reg => {
                    let offset = read_u8(self.slice(2))? as i8 as u32;
                    let addition =
                        operand_add(operand_register(reg), constval(offset));
                    (mem_variable(op_size, addition), 3)
                }
            },
            2 => match rm_val {
                4 => self.parse_sib(2, op_size)?,
                reg => {
                    let offset = read_u32(self.slice(2))?;
                    let addition =
                        operand_add(operand_register(reg), constval(offset));
                    (mem_variable(op_size, addition), 6)
                }
            },
            3 => (Operand::reg_variable_size(Register(rm_val), op_size), 2),
            _ => unreachable!(),
        };
        Ok((rm, r, size))
    }

    fn parse_sib(
        &self,
        variation: u8,
        op_size: MemAccessSize
    ) -> Result<(Operand, usize), Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let sib = self.get(2);
        let mul = 1 << ((sib >> 6) & 0x3);
        let (base_reg, size) = match (sib & 0x7, variation) {
            (5, 0) => (constval(read_u32(self.slice(3))?), 7),
            (reg, _) => (operand_register(reg), 3),
        };
        let reg = (sib >> 3) & 0x7;
        let full_mem_op = if reg != 4 {
            let scale_reg = if mul != 1 {
                operand_mul(operand_register(reg), constval(mul))
            } else {
                operand_register(reg)
            };
            operand_add(scale_reg, base_reg)
        } else {
            base_reg
        };
        match variation {
            0 => Ok((mem_variable(op_size, full_mem_op), size)),
            1 => {
                let constant = constval(read_u8(self.slice(size))? as i8 as u32);
                Ok((mem_variable(op_size, operand_add(full_mem_op, constant)), size + 1))
            }
            2 => {
                let constant = constval(read_u32(self.slice(size))?);
                Ok((mem_variable(op_size, operand_add(full_mem_op, constant)), size + 4))
            }
            _ => unreachable!(),
        }
    }

    fn push_imm(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let imm_size = match self.get(0) {
            0x68 => self.mem16_32(),
            _ => MemAccessSize::Mem8,
        };
        let constant = match read_variable_size(self.slice(1), imm_size) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(match self.pos {
            0 => sub(esp(), constval(4)),
            1 => mov(mem32(esp()), constval(constant)),
            _ => return None,
        }))
    }

    fn inc_dec_op(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let byte = self.get(0);
        let is_inc = byte < 0x48;
        let reg = byte & 0x7;
        let reg = Rc::new(Operand::reg_variable_size(Register(reg), self.mem16_32()));
        Some(Ok(match (self.pos, is_inc) {
            (0, true) => add(reg, constval(1)),
            (0, false) => sub(reg, constval(1)),
            _ => return None,
        }))
    }

    fn pushpop_reg_op(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let byte = self.get(0);
        let is_push = byte < 0x58;
        let reg = byte & 0x7;
        Some(Ok(match (self.pos, is_push) {
            (0, true) => sub(esp(), constval(4)),
            (1, true) => mov(mem32(esp()), operand_register(reg)),
            (0, false) => mov(operand_register(reg), mem32(esp())),
            (1, false) => add(esp(), constval(4)),
            _ => return None,
        }))
    }

    fn flag_set(&self, flag: Flag, value: bool) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        Some(Ok(match self.pos {
            0 => mov(Operand::new_simplified_rc(OperandType::Flag(flag)), constval(value as u32)),
            _ => return None,
        }))
    }

    fn condition(&self) -> Rc<Operand> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        match self.get(0) & 0xf {
            // jo, jno
            0x0 => operand_logical_not(operand_eq(flag_o(), constval(0))),
            0x1 => operand_eq(flag_o(), constval(0)),
            // jb, jnb (jae) (jump if carry)
            0x2 => operand_logical_not(operand_eq(flag_c(), constval(0))),
            0x3 => operand_eq(flag_c(), constval(0)),
            // je, jne
            0x4 => operand_logical_not(operand_eq(flag_z(), constval(0))),
            0x5 => operand_eq(flag_z(), constval(0)),
            // jbe, jnbe (ja)
            0x6 => operand_or(
                operand_logical_not(operand_eq(flag_z(), constval(0))),
                operand_logical_not(operand_eq(flag_c(), constval(0))),
            ),
            0x7 => operand_and(
                operand_eq(flag_z(), constval(0)),
                operand_eq(flag_c(), constval(0)),
            ),
            // js, jns
            0x8 => operand_logical_not(operand_eq(flag_s(), constval(0))),
            0x9 => operand_eq(flag_s(), constval(0)),
            // jpe, jpo
            0xa => operand_logical_not(operand_eq(flag_p(), constval(0))),
            0xb => operand_eq(flag_p(), constval(0)),
            // jl, jnl (jge)
            0xc => operand_logical_not(operand_eq(flag_s(), flag_o())),
            0xd => operand_eq(flag_s(), flag_o()),
            // jle, jnle (jg)
            0xe => operand_or(
                operand_logical_not(operand_eq(flag_z(), constval(0))),
                operand_logical_not(operand_eq(flag_s(), flag_o())),
            ),
            0xf => operand_and(
                operand_eq(flag_z(), constval(0)),
                operand_eq(flag_s(), flag_o()),
            ),
            _ => unreachable!(),
        }
    }

    fn cmov(&self) -> Option<Result<Operation, Error>> {
        let (rm, r, _) = match self.parse_modrm(self.mem16_32()) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        let r = Rc::new(r);
        return Some(Ok(match self.pos {
            0 => Operation::Move((*r).clone().into(), rm, Some(self.condition())),
            _ => return None,
        }))
    }

    fn conditional_jmp(&self, op_size: MemAccessSize) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = match read_variable_size_signed(self.slice(1), op_size) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        match self.pos {
            0 => Some(Ok(Operation::Jump { condition: self.condition(), to })),
            _ => None,
        }
    }

    fn conditional_set(&self) -> Option<Result<Operation, Error>> {
        let condition = self.condition();
        let (rm, _, _) = match self.parse_modrm(MemAccessSize::Mem8) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        return Some(Ok(match self.pos {
            0 => Operation::Move((*rm).clone().into(), condition, None),
            _ => return None,
        }))
    }

    fn short_jmp(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = match read_variable_size_signed(self.slice(1), MemAccessSize::Mem8) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        match self.pos {
            0 => Some(Ok(Operation::Jump { condition: constval(1), to })),
            _ => None,
        }
    }

    fn xchg(&self) -> Option<Result<Operation, Error>> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r, _) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        let r = Rc::new(r);
        match self.pos {
            0 => Some(Ok(Operation::Swap((*r).clone().into(), (*rm).clone().into()))),
            _ => None,
        }
    }

    fn move_mem_eax(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let constant = match read_variable_size(self.slice(1), op_size) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let mem = mem_variable(op_size, constval(constant)).into();
        let eax_left = self.get(0) & 0x2 == 0;
        Some(Ok(match (eax_left, self.pos) {
            (true, 0) => mov(operand_register(0), mem),
            (false, 0) => mov(mem, operand_register(0)),
            _ => return None,
        }))
    }

    fn move_const_to_reg(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x8 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let register = self.get(0) & 0x7;
        let constant = match read_variable_size(self.slice(1), op_size) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(match self.pos {
            0 => mov(operand_register(register), constval(constant)),
            _ => return None,
        }))
    }

    fn eax_imm_arith<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Option<Result<Operation, Error>>
    where F: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>,
          G: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>,
          H: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Option<Result<Operation, Error>>,
    {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let dest = Rc::new(Operand::reg_variable_size(Register(0), op_size));
        let imm = match read_variable_size(self.slice(1), op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let val = constval(imm);
        pre_flags(dest.clone(), val.clone(), self.pos)
            .or_else(|state| make_arith(dest.clone(), val.clone(), state))
            .or_else(|state| post_flags(dest, val, state).ok_or(!0))
            .ok()
    }

    /// Also mov even though I'm not sure if I should count it as no-op arith or a separate
    /// thing.
    fn generic_arith_op<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Option<Result<Operation, Error>>
    where F: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>,
          G: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>,
          H: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Option<Result<Operation, Error>>,
    {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r, _) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        let r = Rc::new(r);
        let rm_left = self.get(0) & 0x3 < 2;
        match rm_left {
            true => pre_flags(rm.clone(), r.clone(), self.pos),
            false => pre_flags(r.clone(), rm.clone(), self.pos),
        }.or_else(|state| match rm_left {
            true => make_arith(rm.clone(), r.clone(), state),
            false => make_arith(r.clone(), rm.clone(), state),
        }).or_else(|state| match rm_left {
            true => return post_flags(rm, r, state).ok_or(!0),
            false => return post_flags(r, rm, state).ok_or(!0),
        }).ok()
    }

    fn movsx(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (mut rm, r, _) = match self.parse_modrm(self.mem16_32()) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        rm.ty = match rm.ty {
            OperandType::Memory(mem) => {
                OperandType::Memory(MemAccess {
                    address: mem.address,
                    size: op_size,
                })
            }
            OperandType::Register(r) => match op_size {
                MemAccessSize::Mem8 => OperandType::Register8Low(r),
                MemAccessSize::Mem16 => OperandType::Register16(r),
                _ => unreachable!(),
            },
            x => x,
        };
        let rm = Rc::new(rm);
        let r = Rc::new(r);

        if is_rm_short_r_register(&rm, &r) {
            Some(Ok(match self.pos {
                0 => match op_size {
                    MemAccessSize::Mem8 => and(r, constval(0xff)),
                    MemAccessSize::Mem16 => and(r, constval(0xffff)),
                    _ => unreachable!(),
                },
                1 => {
                    use self::ArithOpType::*;
                    let signed_max = match op_size {
                        MemAccessSize::Mem8 => constval(0x7f),
                        MemAccessSize::Mem16 => constval(0x7fff),
                        _ => unreachable!(),
                    };
                    let high_const = match op_size {
                        MemAccessSize::Mem8 => operand_or(constval(0xffffff00), r.clone()),
                        MemAccessSize::Mem16 => operand_or(constval(0xffff0000), r.clone()),
                        _ => unreachable!(),
                    };

                    let compare = OperandType::Arithmetic(GreaterThan(rm, signed_max));
                    let rm_cond = Operand::new_not_simplified_rc(compare);
                    Operation::Move((*r).clone().into(), high_const, Some(rm_cond))
                }
                _ => return None,
            }))
        } else {
            let mem_size = match rm.ty {
                OperandType::Memory(ref mem) => Some(mem.size),
                _ => None,
            };
            if let Some(mem_size) = mem_size {
                Some(Ok(match self.pos {
                    0 => mov(r, rm),
                    x => {
                        // sigh
                        let reg = match r.ty {
                            OperandType::Register(r) => r.0,
                            _ => panic!("Movsx r, [mem] r is not register? {:?}", r),
                        };
                        let dat = match mem_size == MemAccessSize::Mem8 {
                            true => [0xbe, 0xc0 + reg * 9],
                            false => [0xbf, 0xc0 + reg * 9],
                        };
                        let state = InstructionOpsState {
                            address: self.address,
                            data: &dat[..],
                            prefixes: self.prefixes,
                            pos: x - 1,
                            len: 3,
                            ctx: self.ctx,
                        };
                        return state.movsx();
                    }
                }))
            } else {
                Some(Ok(match self.pos {
                    0 => mov(r, constval(0)),
                    1 => {
                        use self::ArithOpType::*;
                        let signed_max = match op_size {
                            MemAccessSize::Mem8 => constval(0x7f),
                            MemAccessSize::Mem16 => constval(0x7fff),
                            _ => unreachable!(),
                        };
                        let compare = OperandType::Arithmetic(GreaterThan(rm, signed_max));
                        let rm_cond = Operand::new_not_simplified_rc(compare);
                        Operation::Move((*r).clone().into(), constval(!0), Some(rm_cond))
                    }
                    2 => mov(r, rm),
                    _ => return None,
                }))
            }
        }
    }

    fn movzx(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;

        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (mut rm, r, _) = match self.parse_modrm(self.mem16_32()) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        rm.ty = match rm.ty {
            OperandType::Memory(mem) => {
                OperandType::Memory(MemAccess {
                    address: mem.address,
                    size: op_size,
                })
            }
            OperandType::Register(r) => match op_size {
                MemAccessSize::Mem8 => OperandType::Register8Low(r),
                MemAccessSize::Mem16 => OperandType::Register16(r),
                _ => unreachable!(),
            },
            x => x,
        };
        let rm = Rc::new(rm);
        let r = Rc::new(r);
        if is_rm_short_r_register(&rm, &r) {
            Some(Ok(match self.pos {
                0 => match op_size {
                    MemAccessSize::Mem8 => and(r, constval(0xff)),
                    MemAccessSize::Mem16 => and(r, constval(0xffff)),
                    _ => unreachable!(),
                },
                _ => return None,
            }))
        } else {
            let is_mem = match rm.ty {
                OperandType::Memory(_) => true,
                _ => false,
            };
            if is_mem {
                Some(Ok(match self.pos {
                    0 => mov(r, rm),
                    _ => return None,
                }))
            } else {
                Some(Ok(match self.pos {
                    0 => mov(r, constval(0)),
                    1 => mov(r, rm),
                    _ => return None,
                }))
            }
        }
    }

    fn various_f7(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, _) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let variant = (self.get(1) >> 3) & 0x7;
        match variant {
            0 | 1 => self.generic_arith_with_imm_op(&TEST_OPS, MemAccessSize::Mem8),
            2 => match self.pos {
                0 => Some(Ok(make_arith_operation(
                    rm.clone().into(),
                    ArithOpType::Not(rm.into()),
                ))),
                _ => None,
            },
            5 => match self.pos {
                0 => Some(Ok(signed_mul(pair_edx_eax(), rm.clone().into()))),
                // TODO flags, imul only sets c and o on overflow
                _ => None,
            },
            6 => match self.pos {
                0 => {
                    let div = operand_div(pair_edx_eax(), rm.clone().into());
                    let modulo = operand_mod(pair_edx_eax(), rm.clone().into());
                    Some(Ok(Operation::Move(
                        (*pair_edx_eax()).clone().into(),
                        pair(modulo, div),
                        None,
                    )))
                }
                _ => None,
            },
            _ => return Some(Err(Error::UnknownOpcode(self.data.into()))),
        }
    }

    fn xorps(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        let (rm, dest, _) = match self.parse_modrm_xmm(self.pos) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        Some(Ok(match self.pos {
            0 | 1 | 2 | 3 => xor(dest.into(), rm.into()),
            _ => return None,
        }))
    }

    fn mov_sse_6e(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Some(Err(Error::UnknownOpcode(self.data.into())));
        }
        let (rm, _, _) = match self.parse_modrm(MemAccessSize::Mem32) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let dest = (self.get(1) >> 3) & 0x7;
        Some(Ok(match self.pos {
            0 => mov(operand_xmm(dest, 0).into(), rm.into()),
            1 | 2 | 3 => mov(operand_xmm(dest, self.pos).into(), constval(0)),
            _ => return None,
        }))
    }

    fn mov_sse_11(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        let length = match (self.has_prefix(0xf3), self.has_prefix(0xf2)) {
            // movss
            (true, false) => 1,
            // movsd
            (false, true) => 2,
            // movups, movupd
            (false, false) => 4,
            (true, true) => return Some(Err(Error::UnknownOpcode(self.data.into()))),
        };
        let (rm, src, _) = match self.parse_modrm_xmm(self.pos) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        if self.pos < length {
            Some(Ok(mov(rm.into(), src.into())))
        } else {
            None
        }
    }

    fn mov_sse_7e(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        if !self.has_prefix(0xf3) {
            return Some(Err(Error::UnknownOpcode(self.data.into())));
        }
        let (rm, dest, _) = match self.parse_modrm_xmm(self.pos) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        Some(Ok(match self.pos {
            0 | 1 | 2 | 3 => mov(dest.into(), rm.into()),
            _ => return None,
        }))
    }

    fn mov_sse_d6(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Some(Err(Error::UnknownOpcode(self.data.into())));
        }
        let (rm, src, _) = match self.parse_modrm_xmm(self.pos) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        Some(Ok(match self.pos {
            0 | 1 => mov(rm.into(), src.into()),
            2 | 3 => match rm.ty {
                OperandType::Xmm(_, _) => mov(rm.into(), constval(0)),
                _ => return None,
            },
            _ => return None,
        }))
    }

    fn packed_shift_right(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Some(Err(Error::UnknownOpcode(self.data.into())));
        }
        let (rm, dest, _) = match self.parse_modrm(MemAccessSize::Mem32) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        // dest.0 = (dest.0 >> rm.0) | (dest.1 << (32 - rm.0))
        // shl dest.1, rm.0
        // dest.2 = (dest.2 >> rm.0) | (dest.3 << (32 - rm.0))
        // shl dest.3, rm.0
        // Zero everything if rm.1 is set
        Some(Ok(match self.pos {
            0 | 2 => {
                let (low, high) = dest.to_xmm_64(self.pos >> 1);
                let rm = rm.to_xmm_32(0);
                make_arith_operation(
                    (*low).clone().into(),
                    ArithOpType::Or(
                        operand_rsh(low, rm.clone()),
                        operand_lsh(high, operand_sub(constval(32), rm)),
                    ),
                )
            }
            1 | 3 => {
                let (_, high) = dest.to_xmm_64(self.pos >> 1);
                let rm = rm.to_xmm_32(0);
                rsh(high, rm)
            }
            4 | 5 | 6 | 7 => {
                let dest = dest.to_xmm_32(self.pos & 3);
                let (_, high) = rm.to_xmm_64(0);
                let high_u32_set = operand_logical_not(operand_eq(high, constval(0)));
                Operation::Move((*dest).clone().into(), constval(0), Some(high_u32_set))
            }
            _ => return None,
        }))
    }

    fn various_fe_ff(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let variant = (self.get(1) >> 3) & 0x7;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, _) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        match variant {
            0 | 1 => {
                let is_inc = variant == 0;
                Some(Ok(match (is_inc, self.pos) {
                    (true, 0) => add(rm.into(), constval(1)),
                    (false, 0) => sub(rm.into(), constval(1)),
                    _ => return None,
                }))
            }
            2 | 3 => match self.pos {
                0 => Some(Ok(Operation::Call(rm.into()))),
                _ => None,
            },
            4 | 5 => match self.pos {
                0 => Some(Ok(Operation::Jump { condition: constval(1), to: rm.into() })),
                _ => None,
            },
            6 => {
                PUSH_OPS.operation(rm.into(), constval(!0), self.pos).ok()
            }
            _ => return Some(Err(Error::UnknownOpcode(self.data.into()))),
        }
    }

    fn bitwise_with_imm_op(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (_, _, modrm_size) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let imm = match read_variable_size(self.slice(modrm_size), MemAccessSize::Mem8) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        if imm == 0 {
            return None;
        }
        let op_gen: &ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            7 => &SAR_OPS,
            _ => return Some(Err(Error::UnknownOpcode(self.data.into()))),
        };
        self.generic_arith_with_imm_op(op_gen, MemAccessSize::Mem8)
    }

    fn bitwise_compact_op(&self) -> Option<Result<Operation, Error>> {
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, _) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        let shift_count = match self.get(0) & 2 {
            0 => constval(1),
            _ => Rc::new(Operand::reg_variable_size(Register(1), MemAccessSize::Mem8)),
        };
        let op_gen: &ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            _ => return Some(Err(Error::UnknownOpcode(self.data.into()))),
        };
        op_gen.pre_flags(rm.clone(), shift_count.clone(), self.pos).or_else(|state| {
            op_gen.operation(rm.clone(), shift_count.clone(), state)
        }).or_else(|state| {
            op_gen.post_flags(rm, shift_count, state).ok_or(!0)
        }).ok()
    }

    fn signed_multiply_rm_imm(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let imm_size = match self.get(0) & 0x2 {
            2 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r, modrm_size) = match self.parse_modrm(self.mem16_32()) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let imm = constval(match read_variable_size(self.slice(modrm_size), imm_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        });
        Some(Ok(match self.pos {
            0 => mov(r.into(), operand_signed_mul(rm.into(), imm)),
            // TODO flags, imul only sets c and o on overflow
            _ => return None,
        }))
    }

    fn imul_normal(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        let (rm, r, _) = match self.parse_modrm(self.mem16_32()) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        Some(Ok(match self.pos {
            0 => signed_mul(r.into(), rm.into()),
            // TODO flags, imul only sets c and o on overflow
            _ => return None,
        }))
    }

    fn arith_with_imm_op(&self) -> Option<Result<Operation, Error>> {
        let op_gen: &ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
            0 => &ADD_OPS,
            1 => &OR_OPS,
            2 => &ADC_OPS,
            3 => &SBB_OPS,
            4 => &AND_OPS,
            5 => &SUB_OPS,
            6 => &XOR_OPS,
            7 => &CMP_OPS,
            _ => return Some(Err(Error::UnknownOpcode(self.data.into()))),
        };
        let imm_size = match self.get(0) & 0x3 {
            0 | 2 | 3 => MemAccessSize::Mem8,
            _ => self.imm16_32(),
        };
        self.generic_arith_with_imm_op(op_gen, imm_size)
    }

    fn generic_arith_with_imm_op(
        &self,
        op_gen: &ArithOperationGenerator,
        imm_size: MemAccessSize,
    ) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, modrm_size) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        let imm = constval(match read_variable_size(self.slice(modrm_size), imm_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        });
        op_gen.pre_flags(rm.clone(), imm.clone(), self.pos).or_else(|state| {
            op_gen.operation(rm.clone(), imm.clone(), state)
        }).or_else(|state| {
            op_gen.post_flags(rm, imm, state).ok_or(!0)
        }).ok()
    }

    fn eax_imm_cmp<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Option<Result<Operation, Error>>
    where F: FnOnce(Rc<Operand>, Rc<Operand>) -> Rc<Operand>,
          G: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>,
          H: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Option<Result<Operation, Error>>,
    {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let eax = Rc::new(Operand::reg_variable_size(Register(0), op_size));
        let imm = match read_variable_size(self.slice(1), op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let val = constval(imm);
        pre_flags(eax.clone(), val.clone(), self.pos)
            .or_else(|state| {
                let operand = make_arith(eax, val);
                post_flags(operand, constval(!0), state).ok_or(!0)
            })
            .ok()
    }


    fn generic_cmp_op<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Option<Result<Operation, Error>>
    where F: FnOnce(Rc<Operand>, Rc<Operand>) -> Rc<Operand>,
          G: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>,
          H: FnOnce(Rc<Operand>, Rc<Operand>, u8) -> Option<Result<Operation, Error>>,
    {
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let rm_left = self.get(0) & 0x3 < 2;
        let (rm, r, _) = match self.parse_modrm(op_size) {
            Err(e) => return Some(Err(e)),
            Ok(x) => x,
        };
        let rm = Rc::new(rm);
        let r = Rc::new(r);
        match rm_left {
            true => pre_flags(rm.clone(), r.clone(), self.pos),
            false => pre_flags(r.clone(), rm.clone(), self.pos),
        }.or_else(|state| {
            let operand = match rm_left {
                true => make_arith(rm, r),
                false => make_arith(r, rm),
            };
            post_flags(operand, constval(!0), state).ok_or(!0)
        }).ok()
    }

    fn call_op(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = match read_u32(self.slice(1)) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        Some(Ok(match self.pos {
            0 => Operation::Call(to),
            _ => return None,
        }))
    }

    fn jump_op(&self) -> Option<Result<Operation, Error>> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = match read_u32(self.slice(1)) {
            Ok(o) => o,
            Err(e) => return Some(Err(e)),
        };
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        Some(Ok(match self.pos {
            0 => Operation::Jump { condition: constval(1), to },
            _ => return None,
        }))
    }
}

/// Checks if r is a register, and rm is the equivalent short register
fn is_rm_short_r_register(rm: &Rc<Operand>, r: &Rc<Operand>) -> bool {
    use operand::OperandType::*;
    match (&rm.ty, &r.ty) {
        (&Register8Low(s), &Register(l)) => l == s,
        (&Register16(s), &Register(l)) => l == s,
        _ => false,
    }
}

trait ArithOperationGenerator {
    fn operation(&self, Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>;
    fn pre_flags(&self, Rc<Operand>, Rc<Operand>, u8) -> Result<Result<Operation, Error>, u8>;
    fn post_flags(&self, Rc<Operand>, Rc<Operand>, u8) -> Option<Result<Operation, Error>>;
}

macro_rules! arith_op_generator {
    ($stname: ident, $name:ident, $pre:ident, $op:ident, $post:ident) => {
        struct $name;
        impl ArithOperationGenerator for $name {
            fn operation(&self, a: Rc<Operand>, b: Rc<Operand>, c: u8) -> Result<Result<Operation, Error>, u8> {
                self::operation_helpers::$op(a, b, c)
            }
            fn pre_flags(&self, a: Rc<Operand>, b: Rc<Operand>, c: u8) -> Result<Result<Operation, Error>, u8> {
                self::operation_helpers::$pre(a, b, c)
            }
            fn post_flags(&self, a: Rc<Operand>, b: Rc<Operand>, c :u8) -> Option<Result<Operation, Error>> {
                self::operation_helpers::$post(a, b, c)
            }
        }
        static $stname: $name = $name;
    }
}
arith_op_generator!(ADD_OPS, AddOps, add_flags, add_ops, result_flags);
arith_op_generator!(ADC_OPS, AdcOps, adc_flags, adc_ops, result_flags);
arith_op_generator!(OR_OPS, OrOps, zero_carry_oflow, or_ops, result_flags);
arith_op_generator!(AND_OPS, AndOps, zero_carry_oflow, and_ops, result_flags);
arith_op_generator!(SUB_OPS, SubOps, sub_flags, sub_ops, result_flags);
arith_op_generator!(SBB_OPS, SbbOps, sbb_flags, sbb_ops, result_flags);
arith_op_generator!(XOR_OPS, XorOps, zero_carry_oflow, xor_ops, result_flags);
arith_op_generator!(CMP_OPS, CmpOps, sub_flags, nop_ops, cmp_result_flags);
arith_op_generator!(TEST_OPS, TestOps, zero_carry_oflow, nop_ops, cmp_result_flags);
arith_op_generator!(MOV_OPS, MovOps, nop_ops, mov_ops, nop_ops_post);
arith_op_generator!(PUSH_OPS, PushOps, nop_ops, push_ops, nop_ops_post);
// zero_carry_oflow is wrong but lazy
arith_op_generator!(ROL_OPS, RolOps, zero_carry_oflow, rol_ops, result_flags);
arith_op_generator!(ROR_OPS, RorOps, zero_carry_oflow, ror_ops, result_flags);
arith_op_generator!(LSH_OPS, LshOps, zero_carry_oflow, lsh_ops, result_flags);
arith_op_generator!(RSH_OPS, RshOps, zero_carry_oflow, rsh_ops, result_flags);
arith_op_generator!(SAR_OPS, SarOps, zero_carry_oflow, sar_ops, result_flags);

pub mod operation_helpers {
    use std::rc::Rc;

    use byteorder::{LittleEndian, ReadBytesExt};

    use operand::ArithOpType::*;
    use operand::MemAccessSize::*;
    use operand::{Flag, MemAccessSize, Operand, OperandType};
    use operand::operand_helpers::*;
    use super::{make_arith_operation, DestOperand, Error, Operation};

    pub fn read_u32(mut val: &[u8]) -> Result<u32, Error> {
        val.read_u32::<LittleEndian>().map_err(|_| Error::InternalDecodeError)
    }

    pub fn read_u16(mut val: &[u8]) -> Result<u16, Error> {
        val.read_u16::<LittleEndian>().map_err(|_| Error::InternalDecodeError)
    }

    pub fn read_u8(mut val: &[u8]) -> Result<u8, Error> {
        val.read_u8().map_err(|_| Error::InternalDecodeError)
    }

    pub fn read_variable_size(val: &[u8], size: MemAccessSize) -> Result<u32, Error> {
        match size {
            Mem32 => read_u32(val),
            Mem16 => read_u16(val).map(|x| x as u32),
            Mem8 => read_u8(val).map(|x| x as u32),
        }
    }

    pub fn read_variable_size_signed(val: &[u8], size: MemAccessSize) -> Result<u32, Error> {
        match size {
            Mem32 => read_u32(val),
            Mem16 => read_u16(val).map(|x| x as i16 as u32),
            Mem8 => read_u8(val).map(|x| x as i8 as u32),
        }
    }

    pub fn mov(dest: Rc<Operand>, from: Rc<Operand>) -> Operation {
        Operation::Move((*dest).clone().into(), from, None)
    }

    pub fn mov_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(mov(dest, rhs))),
            x => Err(x - 1),
        }
    }

    pub fn lea_ops(rhs: Rc<Operand>, dest: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => match rhs.ty {
                OperandType::Memory(ref mem) => Ok(Ok(mov(dest, mem.address.clone()))),
                _ => Ok(Err(Error::UnknownOpcode(vec![])))
            },
            x => Err(x - 1),
        }
    }

    pub fn push_ops(val: Rc<Operand>, _unused: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(sub(esp(), constval(4)))),
            1 => Ok(Ok(mov(mem32(esp()), val))),
            x => Err(x - 2),
        }
    }

    pub fn add(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            Add(dest, rhs),
        )
    }

    pub fn add_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(add(dest, rhs))),
            x => Err(x - 1),
        }
    }

    pub fn adc_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(add(dest, operand_add(rhs, flag_c())))),
            x => Err(x - 1),
        }
    }

    pub fn sub(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            Sub(dest, rhs),
        )
    }

    pub fn sub_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(sub(dest, rhs))),
            x => Err(x - 1),
        }
    }

    pub fn sbb_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(sub(dest, operand_add(rhs, flag_c())))),
            x => Err(x - 1),
        }
    }

    pub fn signed_mul(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            SignedMul(dest, rhs),
        )
    }

    pub fn xor(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            Xor(dest, rhs),
        )
    }

    pub fn xor_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(xor(dest, rhs))),
            x => Err(x - 1),
        }
    }

    pub fn rol_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(make_arith_operation(
                (*dest).clone().into(),
                RotateLeft(dest, rhs),
            ))),
            x => Err(x - 1),
        }
    }

    pub fn ror_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(make_arith_operation(
                (*dest).clone().into(),
                RotateLeft(dest, operand_sub(constval(32), rhs)),
            ))),
            x => Err(x - 1),
        }
    }

    pub fn lsh_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(make_arith_operation(
                (*dest).clone().into(),
                Lsh(dest, rhs),
            ))),
            x => Err(x - 1),
        }
    }

    pub fn rsh(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            Rsh(dest, rhs),
        )
    }

    pub fn rsh_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(make_arith_operation(
                (*dest).clone().into(),
                Rsh(dest, rhs),
            ))),
            x => Err(x - 1),
        }
    }

    pub fn sar_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 | 1 => {
                let is_positive = operand_eq(
                    operand_and(constval(0x80000000), dest.clone()),
                    constval(0)
                );
                Ok(Ok(match state {
                    0 => Operation::Move(
                        (*dest).clone().into(),
                        operand_or(operand_rsh(dest, rhs), constval(0x80000000)),
                        Some(operand_logical_not(is_positive)),
                    ),
                    _ => Operation::Move(
                        (*dest).clone().into(),
                        operand_rsh(dest, rhs),
                        Some(is_positive),
                    ),
                }))
            }
            x => Err(x - 2),
        }
    }

    pub fn or(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            Or(dest, rhs),
        )
    }

    pub fn or_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(or(dest, rhs))),
            x => Err(x - 1),
        }
    }

    pub fn and(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            (*dest).clone().into(),
            And(dest, rhs),
        )
    }

    pub fn and_ops(dest: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Result<Result<Operation, Error>, u8> {
        match state {
            0 => Ok(Ok(and(dest, rhs))),
            x => Err(x - 1),
        }
    }

    pub fn nop_ops(
        _dest: Rc<Operand>,
        _rhs: Rc<Operand>,
        state: u8
    ) -> Result<Result<Operation, Error>, u8> {
        Err(state)
    }

    pub fn nop_ops_post(
        _dest: Rc<Operand>,
        _rhs: Rc<Operand>,
        _state: u8
    ) -> Option<Result<Operation, Error>> {
        None
    }

    pub fn add_flags(
        lhs: Rc<Operand>,
        rhs: Rc<Operand>,
        state: u8
    ) -> Result<Result<Operation, Error>, u8> {
        Ok(Ok(match state {
            0 => make_arith_operation(
                DestOperand::Flag(Flag::Carry),
                GreaterThan(lhs.clone(), operand_add(lhs, rhs)),
            ),
            1 => make_arith_operation(
                DestOperand::Flag(Flag::Overflow),
                GreaterThanSigned(lhs.clone(), operand_add(lhs, rhs)),
            ),
            x => return Err(x - 2),
        }))
    }

    pub fn adc_flags(
        lhs: Rc<Operand>,
        rhs: Rc<Operand>,
        state: u8
    ) -> Result<Result<Operation, Error>, u8> {
        Ok(Ok(match state {
            0 => make_arith_operation(
                DestOperand::Flag(Flag::Carry),
                GreaterThan(lhs.clone(), operand_add(operand_add(lhs, rhs), flag_c())),
            ),
            1 => make_arith_operation(
                DestOperand::Flag(Flag::Overflow),
                GreaterThanSigned(lhs.clone(), operand_add(operand_add(lhs, rhs), flag_c())),
            ),
            x => return Err(x - 2),
        }))
    }

    pub fn sub_flags(
        lhs: Rc<Operand>,
        rhs: Rc<Operand>,
        state: u8
    ) -> Result<Result<Operation, Error>, u8> {
        Ok(Ok(match state {
            0 => make_arith_operation(
                DestOperand::Flag(Flag::Carry),
                GreaterThan(operand_sub(lhs.clone(), rhs), lhs),
            ),
            1 => make_arith_operation(
                DestOperand::Flag(Flag::Overflow),
                GreaterThanSigned(operand_sub(lhs.clone(), rhs), lhs),
            ),
            x => return Err(x - 2),
        }))
    }

    pub fn sbb_flags(
        lhs: Rc<Operand>,
        rhs: Rc<Operand>,
        state: u8
    ) -> Result<Result<Operation, Error>, u8> {
        Ok(Ok(match state {
            0 => make_arith_operation(
                DestOperand::Flag(Flag::Carry),
                GreaterThan(operand_sub(operand_sub(lhs.clone(), rhs), flag_c()), lhs),
            ),
            1 => make_arith_operation(
                DestOperand::Flag(Flag::Overflow),
                GreaterThanSigned(operand_sub(operand_sub(lhs.clone(), rhs), flag_c()), lhs),
            ),
            x => return Err(x - 2),
        }))
    }

    pub fn zero_carry_oflow(
        _lhs: Rc<Operand>,
        _rhs: Rc<Operand>,
        state: u8
    ) -> Result<Result<Operation, Error>, u8> {
        Ok(Ok(match state {
            0 => mov(flag_c(), constval(0)),
            1 => mov(flag_o(), constval(0)),
            x => return Err(x - 2),
        }))
    }

    pub fn result_flags(lhs: Rc<Operand>, _: Rc<Operand>, state: u8) -> Option<Result<Operation, Error>> {
        Some(Ok(match state {
            0 => make_arith_operation(
                DestOperand::Flag(Flag::Zero),
                Equal(lhs, constval(0)),
            ),
            1 => make_arith_operation(
                DestOperand::Flag(Flag::Sign),
                GreaterThan(lhs, constval(0x7fffffff)),
            ),
            2 => make_arith_operation(
                DestOperand::Flag(Flag::Parity),
                Parity(lhs),
            ),
            _ => return None,
        }))
    }

    pub fn cmp_result_flags(lhs: Rc<Operand>, rhs: Rc<Operand>, state: u8) -> Option<Result<Operation, Error>> {
        result_flags(operand_sub(lhs, rhs), constval(!0), state)
    }

    pub fn esp() -> Rc<Operand> {
        operand_register(4)
    }

    thread_local! {
        static FLAG_Z: Rc<Operand> = Operand::new_simplified_rc(OperandType::Flag(Flag::Zero));
        static FLAG_C: Rc<Operand> = Operand::new_simplified_rc(OperandType::Flag(Flag::Carry));
        static FLAG_O: Rc<Operand> =
            Operand::new_simplified_rc(OperandType::Flag(Flag::Overflow));
        static FLAG_S: Rc<Operand> = Operand::new_simplified_rc(OperandType::Flag(Flag::Sign));
        static FLAG_P: Rc<Operand> = Operand::new_simplified_rc(OperandType::Flag(Flag::Parity));
    }

    pub fn flag_z() -> Rc<Operand> {
        FLAG_Z.with(|x| x.clone())
    }

    pub fn flag_c() -> Rc<Operand> {
        FLAG_C.with(|x| x.clone())
    }

    pub fn flag_o() -> Rc<Operand> {
        FLAG_O.with(|x| x.clone())
    }

    pub fn flag_s() -> Rc<Operand> {
        FLAG_S.with(|x| x.clone())
    }

    pub fn flag_p() -> Rc<Operand> {
        FLAG_P.with(|x| x.clone())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Operation {
    Move(DestOperand, Rc<Operand>, Option<Rc<Operand>>),
    Swap(DestOperand, DestOperand),
    Call(Rc<Operand>),
    Jump { condition: Rc<Operand>, to: Rc<Operand> },
    Return(u32),
}

fn make_arith_operation(dest: DestOperand, arith: ArithOpType) -> Operation {
    Operation::Move(dest, Operand::new_not_simplified_rc(OperandType::Arithmetic(arith)), None)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DestOperand {
    Register(Register),
    Register16(Register),
    Register8High(Register),
    Register8Low(Register),
    PairEdxEax,
    Xmm(u8, u8),
    Flag(Flag),
    Memory(MemAccess),
}

impl From<Operand> for DestOperand {
    fn from(val: Operand) -> DestOperand {
        use operand::operand_helpers::*;
        use operand::OperandType::*;
        match val.ty {
            Register(x) => DestOperand::Register(x),
            Register16(x) => DestOperand::Register16(x),
            Register8High(x) => DestOperand::Register8High(x),
            Register8Low(x) => DestOperand::Register8Low(x),
            Pair(hi, low) => {
                assert_eq!(hi, operand_register(1));
                assert_eq!(low, operand_register(0));
                DestOperand::PairEdxEax
            }
            Xmm(x, y) => DestOperand::Xmm(x, y),
            Flag(x) => DestOperand::Flag(x),
            Memory(x) => DestOperand::Memory(x),
            x => panic!("Invalid value for converting Operand -> DestOperand: {:?}", x),
        }
    }
}

impl From<DestOperand> for Operand {
    fn from(val: DestOperand) -> Operand {
        use operand::operand_helpers::*;
        use operand::OperandType::*;
        let ty = match val {
            DestOperand::Register(x) => Register(x),
            DestOperand::Register16(x) => Register16(x),
            DestOperand::Register8High(x) => Register8High(x),
            DestOperand::Register8Low(x) => Register8Low(x),
            DestOperand::PairEdxEax => Pair(operand_register(1), operand_register(0)),
            DestOperand::Xmm(x, y) => Xmm(x, y),
            DestOperand::Flag(x) => Flag(x),
            DestOperand::Memory(x) => Memory(x),
        };
        Operand::new_not_simplified(ty)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_operations_mov16() {
        use operand::operand_helpers::*;
        use operand::OperandContext;

        let ctx = OperandContext::new();
        let buf = [0x66, 0xc7, 0x47, 0x62, 0x00, 0x20];
        let mut disasm = Disassembler::new(&buf[..], 0, VirtualAddress(0));
        let ins = disasm.next(&ctx).unwrap();
        assert_eq!(ins.ops().count(), 1);
        let op = ins.ops().next().unwrap().unwrap();
        let dest = mem_variable(
            MemAccessSize::Mem16,
            operand_add(operand_register(0x7), constval(0x62))
        );

        assert_eq!(op, Operation::Move(dest.into(), constval(0x2000), None));
    }

    #[test]
    fn test_sib() {
        use operand::operand_helpers::*;
        use operand::OperandContext;

        let ctx = OperandContext::new();
        let buf = [0x89, 0x84, 0xb5, 0x18, 0xeb, 0xff, 0xff];
        let mut disasm = Disassembler::new(&buf[..], 0, VirtualAddress(0));
        let ins = disasm.next(&ctx).unwrap();
        assert_eq!(ins.ops().count(), 1);
        let op = ins.ops().next().unwrap().unwrap();
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

        match op {
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

