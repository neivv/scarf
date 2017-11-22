use std::rc::Rc;

use hex_slice::AsHex;
use lde::{self, InsnSet};
use smallvec::SmallVec;

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

pub type OperationVec = SmallVec<[Operation; 8]>;

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

    pub fn next<'b, 'c>(&'b mut self, ctx: &'c OperandContext) -> Result<Instruction, Error> {
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
        let address = VirtualAddress(self.virtual_address.0 + self.pos as u32);
        let data = &self.buf[self.pos..self.pos + length];
        let ops = instruction_operations(address, data, ctx)?;
        let ins = Instruction {
            address,
            ops,
            length,
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

pub struct Instruction {
    address: VirtualAddress,
    ops: SmallVec<[Operation; 8]>,
    length: usize,
}

impl Instruction {
    pub fn ops(&self) -> &[Operation] {
        &self.ops
    }

    pub fn address(&self) -> VirtualAddress {
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
    prefix_66: bool,
    prefix_67: bool,
    prefix_f2: bool,
    prefix_f3: bool,
}

struct InstructionOpsState<'a, 'exec: 'a> {
    address: VirtualAddress,
    data: &'a [u8],
    prefixes: InstructionPrefixes,
    len: u8,
    ctx: &'exec OperandContext,
}

fn instruction_operations(
    address: VirtualAddress,
    data: &[u8],
    ctx: &OperandContext,
) -> Result<SmallVec<[Operation; 8]>, Error> {
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
            0x00 ... 0x03 | 0x04 ... 0x05 | 0x08 ... 0x0b | 0x0c ... 0x0d | 0x20 ... 0x23 |
            0x24 ... 0x25 | 0x28 ... 0x2b | 0x2c ... 0x2d | 0x30 ... 0x33 | 0x34 ... 0x35 |
            0x88 ... 0x8b | 0x8d => {
                // Avoid ridiculous generic binary bloat
                let (ops, flags, flags_post):
                (
                    for<'x> fn(_, _, &'x mut _),
                    for<'x> fn(_, _, &'x mut _),
                    for<'x> fn(_, _, &'x mut _),
                ) = match first_byte {
                    0x00 ... 0x05 => (add_ops, add_flags, result_flags),
                    0x08 ... 0x0d => (or_ops, zero_carry_oflow, result_flags),
                    0x20 ... 0x25 => (and_ops, zero_carry_oflow, result_flags),
                    0x28 ... 0x2d => (sub_ops, zero_carry_oflow, result_flags),
                    0x30 ... 0x35 => (xor_ops, zero_carry_oflow, result_flags),
                    0x88 ... 0x8b => (mov_ops, |_, _, _| {}, |_, _, _| {}),
                    0x8d | _ => (lea_ops, |_, _, _| {}, |_, _, _| {}),
                };
                let eax_imm_arith = first_byte < 0x80 && (first_byte & 7) >= 4;
                if eax_imm_arith {
                    s.eax_imm_arith(ops, flags, flags_post)
                } else {
                    s.generic_arith_op(ops, flags, flags_post)
                }
            }

            0x38 ... 0x3b => s.generic_cmp_op(operand_sub, sub_flags, result_flags),
            0x3c ... 0x3d => s.eax_imm_cmp(operand_sub, sub_flags, result_flags),
            0x40 ... 0x4f => s.inc_dec_op(),
            0x50 ... 0x5f => s.pushpop_reg_op(),
            0x68 | 0x6a => s.push_imm(),
            0x69 | 0x6b => s.signed_multiply_rm_imm(),
            0x70 ... 0x7f => s.conditional_jmp(MemAccessSize::Mem8),
            0x80 ... 0x83 => s.arith_with_imm_op(),
            // Test
            0x84 ... 0x85 => s.generic_cmp_op(operand_and, zero_carry_oflow, result_flags),
            0x86 ... 0x87 => s.xchg(),
            0x90 => Ok(SmallVec::new()),
            // Cwde
            0x98 => {
                let mut out = SmallVec::new();
                let eax = operand_register(0);
                let signed_max = constval(0x7fff);
                let compare = ArithOpType::GreaterThan(eax.clone(), signed_max);
                let cond = Operand::new_not_simplified_rc(OperandType::Arithmetic(compare));
                let neg_sign_extend = operand_or(eax.clone(), constval(0xffff0000));
                let neg_sign_extend_op =
                    Operation::Move(dest_operand(&eax), neg_sign_extend, Some(cond));
                out.push(and(eax, constval(0xffff)));
                out.push(neg_sign_extend_op);
                Ok(out)
            }
            // Cdq
            0x99 => {
                let mut out = SmallVec::new();
                let eax = operand_register(0);
                let edx = operand_register(2);
                let signed_max = constval(0x7fffffff);
                let compare = ArithOpType::GreaterThan(eax, signed_max);
                let cond = Operand::new_not_simplified_rc(OperandType::Arithmetic(compare));
                let neg_sign_extend_op =
                    Operation::Move(dest_operand(&edx), constval(!0), Some(cond));
                out.push(mov(edx, constval(0)));
                out.push(neg_sign_extend_op);
                Ok(out)
            },
            0xa0 ... 0xa3 => s.move_mem_eax(),
            0xa8 ... 0xa9 => s.eax_imm_cmp(operand_and, zero_carry_oflow, result_flags),
            0xb0 ... 0xbf => s.move_const_to_reg(),
            0xc0 ... 0xc1 => s.bitwise_with_imm_op(),
            0xc2 ... 0xc3 => {
                let stack_pop_size = match data[0] {
                    0xc2 => match read_u16(&data[1..]) {
                        Err(_) => 0,
                        Ok(o) => o as u32,
                    },
                    _ => 0,
                };
                Ok(Some(Operation::Return(stack_pop_size)).into_iter().collect())
            }
            0xc6 ... 0xc7 => {
                s.generic_arith_with_imm_op(&MOV_OPS, match s.get(0) {
                    0xc6 => MemAccessSize::Mem8,
                    _ => s.mem16_32(),
                })
            }
            0xd0 ... 0xd3 => s.bitwise_compact_op(),
            0xd9 => s.various_d9(),
            0xe8 => s.call_op(),
            0xe9 => s.jump_op(),
            0xeb => s.short_jmp(),
            0xf6 | 0xf7 => s.various_f7(),
            0xf8 ... 0xfd => {
                let flag = match first_byte {
                    0xf8 ... 0xf9 => Flag::Carry,
                    _ => Flag::Direction,
                };
                let state = first_byte & 0x1 == 1;
                s.flag_set(flag, state)
            }
            0xfe ... 0xff => s.various_fe_ff(),
            _ => Err(UnknownOpcode(s.data.into()))
        }
    } else {
        match first_byte {
            0x11 => s.mov_sse_11(),
            // nop
            0x1f => Ok(SmallVec::new()),
            // rdtsc
            0x31 => {
                let mut out = SmallVec::new();
                out.push(mov(operand_register(0), s.ctx.undefined_rc()));
                out.push(mov(operand_register(2), s.ctx.undefined_rc()));
                Ok(out)
            }
            0x40 ... 0x4f => s.cmov(),
            0x57 => s.xorps(),
            0x6e => s.mov_sse_6e(),
            0x7e => s.mov_sse_7e(),
            0x80 ... 0x8f => s.conditional_jmp(s.mem16_32()),
            0x90 ... 0x9f => s.conditional_set(),
            0xaf => s.imul_normal(),
            0xb6 ... 0xb7 => s.movzx(),
            0xbe ... 0xbf => s.movsx(),
            0xd3 => s.packed_shift_right(),
            0xd6 => s.mov_sse_d6(),
            _ => {
                let mut bytes = vec![0xf];
                bytes.extend(s.data);
                Err(UnknownOpcode(bytes))
            }
        }
    }
}

fn xmm_variant(op: &Rc<Operand>, i: u8) -> Rc<Operand> {
    use operand::operand_helpers::*;
    assert!(i < 4);
    match op.ty {
        OperandType::Register(Register(r)) | OperandType::Xmm(r, _) => operand_xmm(r, i),
        OperandType::Memory(ref mem) => {
            let bytes = match mem.size {
                MemAccessSize::Mem8 => 1,
                MemAccessSize::Mem16 => 2,
                MemAccessSize::Mem32 => 4,
            };
            mem_variable_rc(
                mem.size,
                operand_add(mem.address.clone(), constval(bytes * i as u32))
            )
        }
        _ => panic!("Cannot xmm {:?}", op),
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

    /// Returns (rm, r, modrm_size)
    fn parse_modrm_inner(
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

    fn parse_modrm(&self, op_size: MemAccessSize) -> Result<(Rc<Operand>, Rc<Operand>), Error> {
        let (rm, r, _) = self.parse_modrm_inner(op_size)?;
        Ok((Rc::new(rm), Rc::new(r)))
    }

    fn parse_modrm_imm(
        &self,
        op_size: MemAccessSize,
        imm_size: MemAccessSize,
    ) -> Result<(Rc<Operand>, Rc<Operand>, Rc<Operand>), Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let (rm, r, offset) = self.parse_modrm_inner(op_size)?;
        let imm = read_variable_size(self.slice(offset), imm_size)?;
        Ok((Rc::new(rm), Rc::new(r), constval(imm)))
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

    fn push_imm(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let imm_size = match self.get(0) {
            0x68 => self.mem16_32(),
            _ => MemAccessSize::Mem8,
        };
        let constant = read_variable_size(self.slice(1), imm_size)?;
        let mut out = SmallVec::new();
        out.push(sub(esp(), constval(4)));
        out.push(mov(mem32(esp()), constval(constant)));
        Ok(out)
    }

    fn inc_dec_op(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let byte = self.get(0);
        let is_inc = byte < 0x48;
        let reg = byte & 0x7;
        let reg = Rc::new(Operand::reg_variable_size(Register(reg), self.mem16_32()));
        let mut out = SmallVec::new();
        out.push(match is_inc {
            true => add(reg, constval(1)),
            false => sub(reg, constval(1)),
        });
        Ok(out)
    }

    fn pushpop_reg_op(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let byte = self.get(0);
        let is_push = byte < 0x58;
        let reg = byte & 0x7;
        let mut out = SmallVec::new();
        match is_push {
            true => {
                out.push(sub(esp(), constval(4)));
                out.push(mov(mem32(esp()), operand_register(reg)));
            }
            false => {
                out.push(mov(operand_register(reg), mem32(esp())));
                out.push(add(esp(), constval(4)));
            }
        }
        Ok(out)
    }

    fn flag_set(&self, flag: Flag, value: bool) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let mut out = SmallVec::new();
        out.push(
            mov(Operand::new_simplified_rc(OperandType::Flag(flag)), constval(value as u32))
        );
        Ok(out)
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

    fn cmov(&self) -> Result<OperationVec, Error> {
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let mut out = SmallVec::new();
        out.push(Operation::Move(dest_operand(&r), rm, Some(self.condition())));
        Ok(out)
    }

    fn conditional_jmp(&self, op_size: MemAccessSize) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = read_variable_size_signed(self.slice(1), op_size)?;
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: self.condition(), to });
        Ok(out)
    }

    fn conditional_set(&self) -> Result<OperationVec, Error> {
        let condition = self.condition();
        let (rm, _) = self.parse_modrm(MemAccessSize::Mem8)?;
        let mut out = SmallVec::new();
        out.push(Operation::Move(dest_operand(&rm), condition, None));
        Ok(out)
    }

    fn short_jmp(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = read_variable_size_signed(self.slice(1), MemAccessSize::Mem8)?;
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: constval(1), to });
        Ok(out)
    }

    fn xchg(&self) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r) = self.parse_modrm(op_size)?;
        let mut out = SmallVec::new();
        out.push(Operation::Swap(dest_operand(&r), dest_operand(&rm)));
        Ok(out)
    }

    fn move_mem_eax(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let constant = read_variable_size(self.slice(1), op_size)?;
        let mem = mem_variable(op_size, constval(constant)).into();
        let eax_left = self.get(0) & 0x2 == 0;
        let eax = operand_register(0);
        let mut out = SmallVec::new();
        out.push(match eax_left {
            true => mov(eax, mem),
            false => mov(mem, eax),
        });
        Ok(out)
    }

    fn move_const_to_reg(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x8 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let register = self.get(0) & 0x7;
        let constant = read_variable_size(self.slice(1), op_size)?;
        let mut out = SmallVec::new();
        out.push(mov(operand_register(register), constval(constant)));
        Ok(out)
    }

    fn eax_imm_arith<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Result<OperationVec, Error>
    where F: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
          G: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
          H: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
    {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let dest = Rc::new(Operand::reg_variable_size(Register(0), op_size));
        let imm = read_variable_size(self.slice(1), op_size)?;
        let val = constval(imm);
        let mut out = SmallVec::new();
        pre_flags(dest.clone(), val.clone(), &mut out);
        make_arith(dest.clone(), val.clone(), &mut out);
        post_flags(dest.clone(), val.clone(), &mut out);
        Ok(out)
    }

    /// Also mov even though I'm not sure if I should count it as no-op arith or a separate
    /// thing.
    fn generic_arith_op<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Result<OperationVec, Error>
    where F: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
          G: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
          H: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
    {
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
        pre_flags(left.clone(), right.clone(), &mut out);
        make_arith(left.clone(), right.clone(), &mut out);
        post_flags(left, right, &mut out);
        Ok(out)
    }

    fn movsx(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let actual_rm_ty = match rm.ty {
            OperandType::Memory(ref mem) => {
                OperandType::Memory(MemAccess {
                    address: mem.address.clone(),
                    size: op_size,
                })
            }
            OperandType::Register(r) => match op_size {
                MemAccessSize::Mem8 => OperandType::Register8Low(r),
                MemAccessSize::Mem16 => OperandType::Register16(r),
                _ => unreachable!(),
            },
            ref x => x.clone(),
        };
        let rm = Operand::new_not_simplified_rc(actual_rm_ty);

        let mut out = SmallVec::new();
        if is_rm_short_r_register(&rm, &r) {
            let keep_mask = match op_size {
                MemAccessSize::Mem8 => and(r.clone(), constval(0xff)),
                MemAccessSize::Mem16 => and(r.clone(), constval(0xffff)),
                _ => unreachable!(),
            };
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

            let compare = OperandType::Arithmetic(ArithOpType::GreaterThan(rm, signed_max));
            let rm_cond = Operand::new_not_simplified_rc(compare);
            out.push(keep_mask);
            out.push(Operation::Move(dest_operand(&r), high_const, Some(rm_cond)));
        } else {
            let mem_size = match rm.ty {
                OperandType::Memory(ref mem) => Some(mem.size),
                _ => None,
            };
            if let Some(mem_size) = mem_size {
                out.push(mov(r.clone(), rm));
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
                    len: 3,
                    ctx: self.ctx,
                };
                out.extend(state.movsx()?.into_iter());
            } else {
                let signed_max = match op_size {
                    MemAccessSize::Mem8 => constval(0x7f),
                    MemAccessSize::Mem16 => constval(0x7fff),
                    _ => unreachable!(),
                };
                let compare =
                    OperandType::Arithmetic(ArithOpType::GreaterThan(rm.clone(), signed_max));
                let rm_cond = Operand::new_not_simplified_rc(compare);
                let reg = match r.ty {
                    OperandType::Register(r) => r,
                    _ => unreachable!(),
                };
                let short_r = match op_size {
                    MemAccessSize::Mem8 => OperandType::Register8Low(reg),
                    MemAccessSize::Mem16 => OperandType::Register16(reg),
                    _ => unreachable!(),
                };
                out.push(mov(r.clone(), constval(0)));
                out.push(Operation::Move(dest_operand(&r), constval(!0), Some(rm_cond)));
                out.push(mov(Operand::new_simplified_rc(short_r), rm));
            }
        }
        Ok(out)
    }

    fn movzx(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;

        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => MemAccessSize::Mem16,
        };
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        let actual_rm_ty = match rm.ty {
            OperandType::Memory(ref mem) => {
                OperandType::Memory(MemAccess {
                    address: mem.address.clone(),
                    size: op_size,
                })
            }
            OperandType::Register(r) => match op_size {
                MemAccessSize::Mem8 => OperandType::Register8Low(r),
                MemAccessSize::Mem16 => OperandType::Register16(r),
                _ => unreachable!(),
            },
            ref x => x.clone(),
        };
        let rm = Operand::new_not_simplified_rc(actual_rm_ty);
        let mut out = SmallVec::new();
        if is_rm_short_r_register(&rm, &r) {
            out.push(match op_size {
                MemAccessSize::Mem8 => and(r, constval(0xff)),
                MemAccessSize::Mem16 => and(r, constval(0xffff)),
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
                out.push(mov(r.clone(), constval(0)));
                out.push(mov(r, rm));
            }
        }
        Ok(out)
    }

    fn various_f7(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _) = self.parse_modrm(op_size)?;
        let variant = (self.get(1) >> 3) & 0x7;
        let mut out = SmallVec::new();
        match variant {
            0 | 1 => return self.generic_arith_with_imm_op(&TEST_OPS, MemAccessSize::Mem8),
            2 => {
                out.push(make_arith_operation(dest_operand(&rm), ArithOpType::Not(rm.into())));
            }
            4 => {
                out.push(mov(pair_edx_eax(), operand_mul(operand_register(0), rm)));
            },
            5 => {
                out.push(mov(pair_edx_eax(), operand_signed_mul(operand_register(0), rm)));
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

    fn xorps(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        Ok((0..4).map(|i| xor(xmm_variant(&dest, i), xmm_variant(&rm, i))).collect())
    }

    fn mov_sse_6e(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, r) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        out.push(mov(xmm_variant(&r, 0), rm));
        out.push(mov(xmm_variant(&r, 1), constval(0)));
        out.push(mov(xmm_variant(&r, 2), constval(0)));
        out.push(mov(xmm_variant(&r, 3), constval(0)));
        Ok(out)
    }

    fn mov_sse_11(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let length = match (self.has_prefix(0xf3), self.has_prefix(0xf2)) {
            // movss
            (true, false) => 1,
            // movsd
            (false, true) => 2,
            // movups, movupd
            (false, false) => 4,
            (true, true) => return Err(Error::UnknownOpcode(self.data.into())),
        };
        let (rm, src) = self.parse_modrm(MemAccessSize::Mem32)?;
        Ok((0..length).map(|i| mov(xmm_variant(&rm, i), xmm_variant(&src, i))).collect())
    }

    fn mov_sse_7e(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        if !self.has_prefix(0xf3) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        out.push(mov(xmm_variant(&dest, 0), xmm_variant(&rm, 0)));
        out.push(mov(xmm_variant(&dest, 1), xmm_variant(&rm, 1)));
        out.push(mov(xmm_variant(&dest, 2), xmm_variant(&rm, 2)));
        out.push(mov(xmm_variant(&dest, 3), xmm_variant(&rm, 3)));
        Ok(out)
    }

    fn mov_sse_d6(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, src) = self.parse_modrm(MemAccessSize::Mem32)?;
        let mut out = SmallVec::new();
        out.push(mov(xmm_variant(&rm, 0), xmm_variant(&src, 0)));
        out.push(mov(xmm_variant(&rm, 1), xmm_variant(&src, 1)));
        if let OperandType::Xmm(_, _) = rm.ty {
            out.push(mov(xmm_variant(&rm, 2), constval(0)));
            out.push(mov(xmm_variant(&rm, 3), constval(0)));
        }
        Ok(out)
    }

    fn packed_shift_right(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        if !self.has_prefix(0x66) {
            return Err(Error::UnknownOpcode(self.data.into()));
        }
        let (rm, dest) = self.parse_modrm(MemAccessSize::Mem32)?;
        // dest.0 = (dest.0 >> rm.0) | (dest.1 << (32 - rm.0))
        // shl dest.1, rm.0
        // dest.2 = (dest.2 >> rm.0) | (dest.3 << (32 - rm.0))
        // shl dest.3, rm.0
        // Zero everything if rm.1 is set
        let mut out = SmallVec::new();
        out.push({
            let (low, high) = Operand::to_xmm_64(&dest, 0);
            let rm = Operand::to_xmm_32(&rm, 0);
            make_arith_operation(
                dest_operand(&low),
                ArithOpType::Or(
                    operand_rsh(low, rm.clone()),
                    operand_lsh(high, operand_sub(constval(32), rm)),
                ),
            )
        });
        out.push({
            let (_, high) = Operand::to_xmm_64(&dest, 0);
            let rm = Operand::to_xmm_32(&rm, 0);
            rsh(high, rm)
        });
        out.push({
            let (low, high) = Operand::to_xmm_64(&dest, 1);
            let rm = Operand::to_xmm_32(&rm, 0);
            make_arith_operation(
                dest_operand(&low),
                ArithOpType::Or(
                    operand_rsh(low, rm.clone()),
                    operand_lsh(high, operand_sub(constval(32), rm)),
                ),
            )
        });
        out.push({
            let (_, high) = Operand::to_xmm_64(&dest, 1);
            let rm = Operand::to_xmm_32(&rm, 0);
            rsh(high, rm)
        });
        for i in 0..4 {
            let dest = Operand::to_xmm_32(&dest, i);
            let (_, high) = Operand::to_xmm_64(&rm, 0);
            let high_u32_set = operand_logical_not(operand_eq(high, constval(0)));
            out.push(Operation::Move(dest_operand(&dest), constval(0), Some(high_u32_set)));
        }
        Ok(out)
    }

    fn various_d9(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let variant = (self.get(1) >> 3) & 0x7;
        match variant {
            // Fst/Fstp, as long as rm is mem
            2 | 3 => {
                let (rm, _) = self.parse_modrm(MemAccessSize::Mem32)?;
                let mut out = SmallVec::new();
                match rm.ty {
                    OperandType::Memory(_) => {
                        out.push(mov(rm.clone(), self.ctx.undefined_rc()));
                    }
                    _ => (),
                }
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
                match rm.ty {
                    OperandType::Memory(ref mem) => {
                        out.extend((0..10).map(|i| {
                            let address =
                                operand_add(mem.address.clone(), constval(i * mem_bytes));
                            let dest = mem_variable_rc(mem_size, address);
                            mov(dest, self.ctx.undefined_rc())
                        }));
                    }
                    _ => (),
                }
                Ok(out)
            }
            // Fstcw
            7 => {
                let (rm, _) = self.parse_modrm(MemAccessSize::Mem16)?;
                let mut out = SmallVec::new();
                match rm.ty {
                    OperandType::Memory(_) => {
                        out.push(mov(rm.clone(), self.ctx.undefined_rc()));
                    }
                    _ => (),
                }
                Ok(out)
            }
            // Others just touch FPU registers
            _ => Ok(SmallVec::new()),
        }
    }

    fn various_fe_ff(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let variant = (self.get(1) >> 3) & 0x7;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _) = self.parse_modrm(op_size)?;
        let mut out = SmallVec::new();
        match variant {
            0 | 1 => {
                let is_inc = variant == 0;
                out.push(match is_inc {
                    true => add(rm, constval(1)),
                    false => sub(rm, constval(1)),
                });
            }
            2 | 3 => out.push(Operation::Call(rm.into())),
            4 | 5 => out.push(Operation::Jump { condition: constval(1), to: rm.into() }),
            6 => {
                PUSH_OPS.operation(rm.into(), constval(!0), &mut out);
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
        let op_gen: &ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
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
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _) = self.parse_modrm(op_size)?;
        let shift_count = match self.get(0) & 2 {
            0 => constval(1),
            _ => Rc::new(Operand::reg_variable_size(Register(1), MemAccessSize::Mem8)),
        };
        let op_gen: &ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
            0 => &ROL_OPS,
            1 => &ROR_OPS,
            4 | 6 => &LSH_OPS,
            5 => &RSH_OPS,
            7 => &SAR_OPS,
            _ => return Err(Error::UnknownOpcode(self.data.into())),
        };
        let mut out = SmallVec::new();
        op_gen.pre_flags(rm.clone(), shift_count.clone(), &mut out);
        op_gen.operation(rm.clone(), shift_count.clone(), &mut out);
        op_gen.post_flags(rm, shift_count, &mut out);
        Ok(out)
    }

    fn signed_multiply_rm_imm(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let imm_size = match self.get(0) & 0x2 {
            2 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, r, imm) = self.parse_modrm_imm(self.mem16_32(), imm_size)?;
        // TODO flags, imul only sets c and o on overflow
        let mut out = SmallVec::new();
        out.push(mov(r, operand_signed_mul(rm, imm)));
        Ok(out)
    }

    fn imul_normal(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        let (rm, r) = self.parse_modrm(self.mem16_32())?;
        // TODO flags, imul only sets c and o on overflow
        let mut out = SmallVec::new();
        out.push(signed_mul(r, rm));
        Ok(out)
    }

    fn arith_with_imm_op(&self) -> Result<OperationVec, Error> {
        let op_gen: &ArithOperationGenerator = match (self.get(1) >> 3) & 0x7 {
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
            _ => self.imm16_32(),
        };
        self.generic_arith_with_imm_op(op_gen, imm_size)
    }

    fn generic_arith_with_imm_op(
        &self,
        op_gen: &ArithOperationGenerator,
        imm_size: MemAccessSize,
    ) -> Result<OperationVec, Error> {
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let (rm, _, imm) = self.parse_modrm_imm(op_size, imm_size)?;
        let mut out = SmallVec::new();
        op_gen.pre_flags(rm.clone(), imm.clone(), &mut out);
        op_gen.operation(rm.clone(), imm.clone(), &mut out);
        op_gen.post_flags(rm, imm, &mut out);
        Ok(out)
    }

    fn eax_imm_cmp<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Result<OperationVec, Error>
    where F: FnOnce(Rc<Operand>, Rc<Operand>) -> Rc<Operand>,
          G: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
          H: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
    {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let op_size = match self.get(0) & 0x1 {
            0 => MemAccessSize::Mem8,
            _ => self.mem16_32(),
        };
        let eax = Rc::new(Operand::reg_variable_size(Register(0), op_size));
        let imm = read_variable_size(self.slice(1), op_size)?;
        let val = constval(imm);
        let mut out = SmallVec::new();
        pre_flags(eax.clone(), val.clone(), &mut out);
        let operand = make_arith(eax, val);
        post_flags(operand, constval(!0), &mut out);
        Ok(out)
    }


    fn generic_cmp_op<F, G, H>(
        &self,
        make_arith: F,
        pre_flags: G,
        post_flags: H,
    ) -> Result<OperationVec, Error>
    where F: FnOnce(Rc<Operand>, Rc<Operand>) -> Rc<Operand>,
          G: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
          H: FnOnce(Rc<Operand>, Rc<Operand>, &mut OperationVec),
    {
        use operand::operand_helpers::*;
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
        pre_flags(left.clone(), right.clone(), &mut out);
        let operand = make_arith(left, right);
        post_flags(operand, constval(!0), &mut out);
        Ok(out)
    }

    fn call_op(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = read_u32(self.slice(1))?;
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Call(to));
        Ok(out)
    }

    fn jump_op(&self) -> Result<OperationVec, Error> {
        use self::operation_helpers::*;
        use operand::operand_helpers::*;
        let offset = read_u32(self.slice(1))?;
        let to = constval((self.address.0 + self.len() as u32).wrapping_add(offset));
        let mut out = SmallVec::new();
        out.push(Operation::Jump { condition: constval(1), to });
        Ok(out)
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
    fn operation(&self, Rc<Operand>, Rc<Operand>, &mut OperationVec);
    fn pre_flags(&self, Rc<Operand>, Rc<Operand>, &mut OperationVec);
    fn post_flags(&self, Rc<Operand>, Rc<Operand>, &mut OperationVec);
}

macro_rules! arith_op_generator {
    ($stname: ident, $name:ident, $pre:ident, $op:ident, $post:ident) => {
        struct $name;
        impl ArithOperationGenerator for $name {
            fn operation(&self, a: Rc<Operand>, b: Rc<Operand>, c: &mut OperationVec) {
                self::operation_helpers::$op(a, b, c)
            }
            fn pre_flags(&self, a: Rc<Operand>, b: Rc<Operand>, c: &mut OperationVec) {
                self::operation_helpers::$pre(a, b, c)
            }
            fn post_flags(&self, a: Rc<Operand>, b: Rc<Operand>, c :&mut OperationVec) {
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
arith_op_generator!(MOV_OPS, MovOps, nop_ops, mov_ops, nop_ops);
arith_op_generator!(PUSH_OPS, PushOps, nop_ops, push_ops, nop_ops);
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
    use super::{dest_operand, make_arith_operation, DestOperand, Error, Operation, OperationVec};

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
        Operation::Move(dest_operand(&dest), from, None)
    }

    pub fn mov_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(mov(dest, rhs));
    }

    pub fn lea_ops(rhs: Rc<Operand>, dest: Rc<Operand>, out: &mut OperationVec) {
        if let OperandType::Memory(ref mem) = rhs.ty {
            out.push(mov(dest, mem.address.clone()));
        }
    }

    pub fn push_ops(val: Rc<Operand>, _unused: Rc<Operand>, out: &mut OperationVec) {
        out.push(sub(esp(), constval(4)));
        out.push(mov(mem32(esp()), val));
    }

    pub fn add(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            Add(dest, rhs),
        )
    }

    pub fn add_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(add(dest, rhs));
    }

    pub fn adc_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(add(dest, operand_add(rhs, flag_c())));
    }

    pub fn sub(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            Sub(dest, rhs),
        )
    }

    pub fn sub_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(sub(dest, rhs));
    }

    pub fn sbb_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(sub(dest, operand_add(rhs, flag_c())));
    }

    pub fn signed_mul(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            SignedMul(dest, rhs),
        )
    }

    pub fn xor(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            Xor(dest, rhs),
        )
    }

    pub fn xor_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(xor(dest, rhs));
    }

    pub fn rol_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(make_arith_operation(dest_operand(&dest), RotateLeft(dest, rhs)));
    }

    pub fn ror_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(make_arith_operation(
            dest_operand(&dest),
            RotateLeft(dest, operand_sub(constval(32), rhs))
        ));
    }

    pub fn lsh_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(make_arith_operation(dest_operand(&dest), Lsh(dest, rhs)));
    }

    pub fn rsh(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            Rsh(dest, rhs),
        )
    }

    pub fn rsh_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(rsh(dest, rhs));
    }

    pub fn sar_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        let is_positive = operand_eq(
            operand_and(constval(0x80000000), dest.clone()),
            constval(0),
        );
        out.push(Operation::Move(
            dest_operand(&dest),
            operand_or(operand_rsh(dest.clone(), rhs.clone()), constval(0x80000000)),
            Some(operand_logical_not(is_positive.clone())),
        ));
        out.push(Operation::Move(
            dest_operand(&dest),
            operand_rsh(dest, rhs),
            Some(is_positive),
        ));
    }

    pub fn or(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            Or(dest, rhs),
        )
    }

    pub fn or_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(or(dest, rhs));
    }

    pub fn and(dest: Rc<Operand>, rhs: Rc<Operand>) -> Operation {
        make_arith_operation(
            dest_operand(&dest),
            And(dest, rhs),
        )
    }

    pub fn and_ops(dest: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(and(dest, rhs));
    }

    pub fn nop_ops(_dest: Rc<Operand>, _rhs: Rc<Operand>, _out: &mut OperationVec) {
    }

    pub fn add_flags(lhs: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        let add = operand_add(lhs.clone(), rhs.clone());
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Carry),
            GreaterThan(lhs.clone(), add.clone())
        ));
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Overflow),
            GreaterThanSigned(lhs, add)
        ));
    }

    pub fn adc_flags(lhs: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        let add = operand_add(operand_add(lhs.clone(), rhs), flag_c());
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Carry),
            GreaterThan(lhs.clone(), add.clone()),
        ));
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Overflow),
            GreaterThanSigned(lhs, add)
        ));
    }

    pub fn sub_flags(lhs: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        let sub = operand_sub(lhs.clone(), rhs);
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Carry),
            GreaterThan(sub.clone(), lhs.clone()),
        ));
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Overflow),
            GreaterThanSigned(sub, lhs),
        ));
    }

    pub fn sbb_flags(lhs: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        let sub = operand_sub(operand_sub(lhs.clone(), rhs), flag_c());
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Carry),
            GreaterThan(sub.clone(), lhs.clone()),
        ));
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Overflow),
            GreaterThanSigned(sub, lhs),
        ));
    }

    pub fn zero_carry_oflow(_lhs: Rc<Operand>, _rhs: Rc<Operand>, out: &mut OperationVec) {
        out.push(mov(flag_c(), constval(0)));
        out.push(mov(flag_o(), constval(0)));
    }

    pub fn result_flags(lhs: Rc<Operand>, _: Rc<Operand>, out: &mut OperationVec) {
        out.push(
            make_arith_operation(DestOperand::Flag(Flag::Zero), Equal(lhs.clone(), constval(0)))
        );
        out.push(make_arith_operation(
            DestOperand::Flag(Flag::Sign),
            GreaterThan(lhs.clone(), constval(0x7fffffff)),
        ));
        out.push(make_arith_operation(DestOperand::Flag(Flag::Parity), Parity(lhs)));
    }

    pub fn cmp_result_flags(lhs: Rc<Operand>, rhs: Rc<Operand>, out: &mut OperationVec) {
        result_flags(operand_sub(lhs, rhs), constval(!0), out)
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

impl DestOperand {
    pub fn from_oper(val: &Operand) -> DestOperand {
        dest_operand(val)
    }
}

fn dest_operand(val: &Operand) -> DestOperand {
    use operand::operand_helpers::*;
    use operand::OperandType::*;
    match val.ty {
        Register(x) => DestOperand::Register(x),
        Register16(x) => DestOperand::Register16(x),
        Register8High(x) => DestOperand::Register8High(x),
        Register8Low(x) => DestOperand::Register8Low(x),
        Pair(ref hi, ref low) => {
            assert_eq!(*hi, operand_register(1));
            assert_eq!(*low, operand_register(0));
            DestOperand::PairEdxEax
        }
        Xmm(x, y) => DestOperand::Xmm(x, y),
        Flag(x) => DestOperand::Flag(x),
        Memory(ref x) => DestOperand::Memory(x.clone()),
        ref x => panic!("Invalid value for converting Operand -> DestOperand: {:?}", x),
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
        assert_eq!(ins.ops().len(), 1);
        let op = &ins.ops()[0];
        let dest = mem_variable(
            MemAccessSize::Mem16,
            operand_add(operand_register(0x7), constval(0x62))
        );

        assert_eq!(*op, Operation::Move(dest_operand(&dest), constval(0x2000), None));
    }

    #[test]
    fn test_sib() {
        use operand::operand_helpers::*;
        use operand::OperandContext;

        let ctx = OperandContext::new();
        let buf = [0x89, 0x84, 0xb5, 0x18, 0xeb, 0xff, 0xff];
        let mut disasm = Disassembler::new(&buf[..], 0, VirtualAddress(0));
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

