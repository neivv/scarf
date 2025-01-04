use crate::analysis::{Analyzer, Control};
use crate::exec_state::{ExecutionState, VirtualAddress};
use crate::{ArithOpType, BinaryFile, DestOperand, FlagArith, Operation, Operand, Rva};

/// Component to be included in a `Analyzer`
pub struct ConstantCommentTracker<ArchDef> {
    last_constant_comment_address: Rva,
    arch: ArchDef,
}

pub trait ArchDef {
    fn is_at_compare_instruction<Va: VirtualAddress>(
        &mut self,
        binary: &BinaryFile<Va>,
        address: Va,
    ) -> bool;
}

impl ArchDef for () {
    fn is_at_compare_instruction<Va: VirtualAddress>(
        &mut self,
        _binary: &BinaryFile<Va>,
        _address: Va,
    ) -> bool {
        false
    }
}

#[derive(Copy, Clone)]
pub struct ArchX86;

impl ArchDef for ArchX86 {
    fn is_at_compare_instruction<Va: VirtualAddress>(
        &mut self,
        binary: &BinaryFile<Va>,
        address: Va,
    ) -> bool {
        let ins_byte = Self::get_ins_byte(binary, address).unwrap_or(0);
        (ins_byte >= 0x38 && ins_byte <= 0x3d) || ins_byte == 0x83
    }
}

impl ArchX86 {
    fn get_ins_byte<Va: VirtualAddress>(binary: &BinaryFile<Va>, address: Va) -> Option<u8> {
        let bytes = binary.slice_from_address(address, 0x20).ok()?;
        let mut pos = 0;
        if Va::SIZE == 8 {
            while bytes.get(pos).copied().unwrap_or(0) & 0xf0 == 0x40 {
                pos += 1;
            }
        }
        bytes.get(pos).copied()
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ConstantComment {
    /// Instruction resolves to a constant
    Value(u64),
    /// One of inputs (registers) in the instruction is a constant
    Input(u8, u64),
}

impl ConstantCommentTracker<()> {
    /// Returns ConstantCommentTracker without arch-specific code.
    pub fn new() -> ConstantCommentTracker<()> {
        Self::new_arch(())
    }
}

impl ConstantCommentTracker<ArchX86> {
    /// Returns ConstantCommentTracker for x86/x86-64.
    pub fn new_x86() -> ConstantCommentTracker<ArchX86> {
        Self::new_arch(ArchX86)
    }
}

impl<Arch: ArchDef> ConstantCommentTracker<Arch> {
    pub fn new_arch(arch: Arch) -> ConstantCommentTracker<Arch> {
        ConstantCommentTracker {
            last_constant_comment_address: Rva(!0),
            arch,
        }
    }

    /// If a one of registers used in the instruction holds a constant, adds comment
    /// saying `rax = 6` etc.
    /// Call from `Analyzer::operation`, returns a value if something was found.
    pub fn check_constant_comment<'e, A, E>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
        op: &Operation<'e>,
    ) -> Option<ConstantComment>
    where A: Analyzer<'e, Exec = E>,
          E: ExecutionState<'e>,
    {
        let address = ctrl.address();
        let binary = ctrl.binary();
        let rva = Rva(binary.rva_32(address));
        if rva == self.last_constant_comment_address {
            return None;
        }
        self.last_constant_comment_address.0 = !0;
        // If the instruction is arithmetic, and resolves to constant, then comment that
        match *op {
            Operation::Move(_, value) | Operation::ConditionalMove(_, value, _) => {
                if let Some(arith) = value.if_arithmetic_any() {
                    // xor reg, reg is not worth commenting
                    let care = match arith.ty {
                        ArithOpType::Xor => arith.left != arith.right,
                        _ => true,
                    };
                    if !care {
                        return None;
                    }
                    if let Some(c) = ctrl.resolve(value).if_constant() {
                        self.last_constant_comment_address = rva;
                        return Some(ConstantComment::Value(c));
                    }
                }
            }
            _ => (),
        }
        // Comment one register used in the instruction if it is constant
        let regs = self.interesting_registers(ctrl, op);
        for reg in regs.into_iter().filter_map(|x| x) {
            if let Some(c) = ctrl.resolve_register(reg).if_constant() {
                // Assuming that any constant is interesting, even when moving 0
                // somewhere it may have been initialized far from current instruction.
                self.last_constant_comment_address = rva;
                return Some(ConstantComment::Input(reg, c));
            }
        }
        None
    }

    fn interesting_registers<'e, A, E>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
        op: &Operation<'e>,
    ) -> [Option<u8>; 2]
    where A: Analyzer<'e, Exec = E>,
          E: ExecutionState<'e>,
    {
        let mut out = [None; 2];
        match *op {
            Operation::Call(dest) => {
                out[0] = dest.if_register();
            }
            Operation::Jump { to, .. } => {
                out[0] = to.if_register();
            }
            Operation::SetFlags(ref flags) => {
                // FlagArith::Sub filters out most of the non-cmp instructions
                // without having to parse bytes
                let binary = ctrl.binary();
                let address = ctrl.address();
                if flags.ty == FlagArith::Sub &&
                    self.arch.is_at_compare_instruction(binary, address)
                {
                    out[0] = Operand::and_masked(flags.left).0.if_register();
                    out[1] = Operand::and_masked(flags.right).0.if_register();
                }
            }
            Operation::Move(ref dest, value) | Operation::ConditionalMove(ref dest, value, _) => {
                if let Some(reg) = Operand::and_masked(value).0.if_register() {
                    // Not sure if register-to-register moves are too spammy
                    // but allowing them now
                    let care = match dest {
                        DestOperand::Memory(..) => true,
                        DestOperand::Arch(x) => x.if_register().is_some(),
                    };
                    if care {
                        out[0] = Some(reg);
                    }
                } else if let Some(arith) = value.if_arithmetic_any() {
                    out[0] = Operand::and_masked(arith.left).0.if_register();
                    out[1] = Operand::and_masked(arith.right).0.if_register();
                }
            }
            _ => return out,
        };
        out
    }
}
