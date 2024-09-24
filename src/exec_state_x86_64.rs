//! 64-bit x86 architechture state. Rexported as `scarf::ExecutionStateX86_64`.

use std::fmt;
use std::rc::Rc;

use copyless::BoxHelper;

use crate::analysis;
use crate::disasm::{self, FlagArith, Disassembler64, DestOperand, FlagUpdate, Operation};
use crate::exec_state::{
    self, Constraint, FreezeOperation, Memory, MergeStateCache, PendingFlags, OperandExtX86,
    OperandCtxExtX86,
};
use crate::exec_state::ExecutionState as ExecutionStateTrait;
use crate::light_byteorder::ReadLittleEndian;
use crate::operand::{
    Flag, MemAccess, MemAccessSize, Operand, OperandCtx, OperandType, ArithOpType,
};
use crate::{BinaryFile, VirtualAddress64};

const FLAGS_INDEX: usize = 0x10;
const STATE_OPERANDS: usize = FLAGS_INDEX + 6;

/// ExecutionState for 64-bit x86 architecture.
/// See [`trait ExecutionState`](ExecutionStateTrait) for documentation
/// on most of the functionality of this type.
pub struct ExecutionState<'e> {
    inner: Box<State<'e>>,
}

struct State<'e> {
    // 16 registers, 10 xmm registers with 4 parts each, 6 flags
    state: [Operand<'e>; 0x10 + 0x6],
    xmm: Option<Rc<[Operand<'e>; 0x10 * 4]>>,
    cached_low_registers: CachedLowRegisters<'e>,
    memory: Memory<'e>,
    unresolved_constraint: Option<Constraint<'e>>,
    resolved_constraint: Option<Constraint<'e>>,
    /// A constraint where the address is resolved already
    memory_constraint: Option<Constraint<'e>>,
    ctx: OperandCtx<'e>,
    binary: Option<&'e BinaryFile<VirtualAddress64>>,
    /// Lazily update flags since a lot of instructions set them and
    /// they get discarded later.
    /// The separate Operand is for resolved carry for adc/sbb, otherwise dummy value.
    /// (Operands here are resolved)
    pending_flags: PendingFlags<'e>,
    freeze_buffer: Vec<FreezeOperation<'e>>,
    frozen: bool,
}


// Manual impl since derive adds an unwanted inline hint
impl<'e> Clone for ExecutionState<'e> {
    fn clone(&self) -> Self {
        // Codegen optimization: memory cloning isn't a memcpy,
        // doing all memcpys at the end, after other function calls
        // or branching code generally avoids temporaries.
        let s = &*self.inner;
        let memory = s.memory.clone();
        let xmm = s.xmm.clone();
        // Not cloning freeze_buffer, it is always expected to be empty anyway.
        // Operation::Freeze now also documents that freezes will be discarded on clone.
        let freeze_buffer = Vec::new();
        ExecutionState {
            inner: Box::alloc().init(State {
                memory,
                xmm,
                freeze_buffer,
                cached_low_registers: s.cached_low_registers,
                state: s.state,
                resolved_constraint: s.resolved_constraint,
                unresolved_constraint: s.unresolved_constraint,
                memory_constraint: s.memory_constraint,
                ctx: s.ctx,
                binary: s.binary,
                pending_flags: s.pending_flags,
                frozen: false,
            }),
        }
    }
}

impl<'e> fmt::Debug for ExecutionState<'e> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ExecutionStateX86_64")
    }
}

/// Caches eax/ax/al/ah resolving.
#[derive(Debug, Copy, Clone)]
struct CachedLowRegisters<'e> {
    registers: [[Option<Operand<'e>>; 3]; 0x10],
}

static PARTIAL_REGISTER_MASKS: [u32; 3] = [0xff, 0xffff, 0xffff_ffff];

impl<'e> CachedLowRegisters<'e> {
    fn new() -> CachedLowRegisters<'e> {
        CachedLowRegisters {
            registers: [[None; 3]; 0x10],
        }
    }

    #[inline]
    fn invalidate(&mut self, register: u8) {
        self.registers[register as usize] = [None; 3];
    }

    #[inline]
    fn set32(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][2] = Some(value);
    }

    #[inline]
    fn set16(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][1] = Some(value);
    }

    #[inline]
    fn invalidate_for_8high(&mut self, register: u8) {
        // Won't change low u8 if it was cached
        self.registers[register as usize][1] = None;
        self.registers[register as usize][2] = None;
    }

    #[inline]
    fn set8(&mut self, register: u8, value: Operand<'e>) {
        self.registers[register as usize][0] = Some(value);
    }
}

impl<'e> ExecutionStateTrait<'e> for ExecutionState<'e> {
    type VirtualAddress = VirtualAddress64;
    type Disassembler = Disassembler64<'e>;
    const WORD_SIZE: MemAccessSize = MemAccessSize::Mem64;

    fn maybe_convert_memory_immutable(&mut self, limit: usize) {
        self.inner.memory.maybe_convert_immutable(limit);
    }

    fn add_resolved_constraint(&mut self, constraint: Constraint<'e>) {
        self.inner.add_resolved_constraint(constraint);
    }

    fn add_unresolved_constraint(&mut self, constraint: Constraint<'e>) {
        self.inner.add_to_unresolved_constraint(constraint.0);
    }

    fn add_resolved_constraint_from_unresolved(&mut self) {
        if let Some(c) = self.inner.unresolved_constraint {
            let res = self.resolve(c.0);
            self.add_resolved_constraint(Constraint(res));
        }
    }

    #[inline]
    fn move_to(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        self.inner.move_to(dest, value);
    }

    #[inline]
    fn move_resolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        self.inner.move_resolved(dest, value);
    }

    #[inline]
    fn set_register(&mut self, register: u8, value: Operand<'e>) {
        self.inner.set_register(register, value);
    }

    #[inline]
    fn write_memory(&mut self, mem: &MemAccess<'e>, value: Operand<'e>) {
        let (base, offset) = mem.address();
        self.inner.write_memory(base, offset, mem.size, value);
    }

    #[inline]
    fn set_flag(&mut self, flag: Flag, value: Operand<'e>) {
        self.inner.set_flag(flag, value);
    }

    #[inline]
    fn set_flags_resolved(&mut self, arith: &FlagUpdate<'e>, carry: Option<Operand<'e>>) {
        self.inner.set_flags_resolved(arith, carry);
    }

    #[inline]
    fn ctx(&self) -> OperandCtx<'e> {
        self.inner.ctx
    }

    #[inline]
    fn resolve(&mut self, operand: Operand<'e>) -> Operand<'e> {
        self.inner.resolve(operand)
    }

    #[inline]
    fn resolve_mem(&mut self, mem: &MemAccess<'e>) -> MemAccess<'e> {
        self.inner.resolve_mem(mem)
    }

    #[inline]
    fn update(&mut self, operation: &Operation<'e>) {
        self.update(operation)
    }

    #[inline]
    fn resolve_apply_constraints(&mut self, op: Operand<'e>) -> Operand<'e> {
        self.inner.resolve_apply_constraints(op)
    }

    #[inline]
    fn resolve_register(&mut self, register: u8) -> Operand<'e> {
        self.inner.resolve_register(register)
    }

    #[inline]
    fn resolve_flag(&mut self, flag: Flag) -> Operand<'e> {
        self.inner.resolve_flag(flag)
    }

    fn read_memory(&mut self, mem: &MemAccess<'e>) -> Operand<'e> {
        let (base, offset) = mem.address();
        self.inner.read_memory_impl(base, offset, mem.size)
            .unwrap_or_else(|| {
                let binary = self.inner.binary;
                let ctx = self.ctx();
                State::default_mem_value_size(binary, ctx, base, offset, mem.size)
            })
    }

    fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.unresolve(val)
    }

    fn merge_states(
        old: &mut Self,
        new: &mut Self,
        cache: &mut MergeStateCache<'e>,
    ) -> Option<Self> {
        merge_states(&mut old.inner, &mut new.inner, cache)
    }

    fn apply_call(&mut self, ret: VirtualAddress64) {
        let ctx = self.ctx();
        let new_rsp = ctx.sub_const(self.resolve_register(4), 8);
        self.set_register(4, new_rsp);
        self.write_memory(
            &ctx.mem_access(new_rsp, 0, MemAccessSize::Mem64),
            ctx.constant(ret.0),
        );
    }

    fn initial_state(
        ctx: OperandCtx<'e>,
        binary: &'e BinaryFile<VirtualAddress64>,
    ) -> ExecutionState<'e> {
        ExecutionState::with_binary(binary, ctx)
    }

    fn find_functions_with_callers(file: &crate::BinaryFile<Self::VirtualAddress>)
        -> Vec<analysis::FuncCallPair<Self::VirtualAddress>>
    {
        crate::analysis::find_functions_with_callers_x86_64(file)
    }

    fn find_functions_from_calls(
        code: &[u8],
        section_base: Self::VirtualAddress,
        out: &mut Vec<Self::VirtualAddress>
    ) {
        crate::analysis::find_functions_from_calls_x86_64(code, section_base, out)
    }

    fn find_relocs(
        file: &BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<Self::VirtualAddress>, crate::OutOfBounds> {
        crate::analysis::find_relocs_x86_64(file)
    }

    fn function_ranges_from_exception_info(
        file: &crate::BinaryFile<Self::VirtualAddress>,
    ) -> Result<Vec<(u32, u32)>, crate::OutOfBounds> {
        crate::analysis::function_ranges_from_exception_info_x86_64(file)
    }

    fn value_limits(&mut self, value: Operand<'e>) -> (u64, u64) {
        let mut result = (0, u64::MAX);
        if let Some(constraint) = self.inner.resolved_constraint {
            let new = crate::exec_state::value_limits(constraint.0, value);
            result.0 = result.0.max(new.0);
            result.1 = result.1.min(new.1);
        }
        if let Some(constraint) = self.inner.memory_constraint {
            let ctx = self.ctx();
            let transformed = ctx.transform(constraint.0, 8, |op| match *op.ty() {
                OperandType::Memory(ref mem) => Some(self.read_memory(mem)),
                _ => None,
            });
            let new = crate::exec_state::value_limits(transformed, value);
            result.0 = result.0.max(new.0);
            result.1 = result.1.min(new.1);
        }
        result
    }

    unsafe fn clone_to(&self, out: *mut Self) {
        use std::ptr::write;
        write(out, (*self).clone());
    }
}

impl<'e> ExecutionState<'e> {
    pub fn new<'b>(
        ctx: OperandCtx<'b>,
    ) -> ExecutionState<'b> {
        let dummy = ctx.const_0();
        let mut state = [dummy; STATE_OPERANDS];
        for i in 0..16 {
            state[i] = ctx.register(i as u8);
        }
        for i in 0..5 {
            state[FLAGS_INDEX + i] = ctx.flag_by_index(i).clone();
        }
        // Direction
        state[FLAGS_INDEX + 5] = ctx.const_0();
        let inner = Box::alloc().init(State {
            state,
            xmm: None,
            cached_low_registers: CachedLowRegisters::new(),
            memory: Memory::new(),
            unresolved_constraint: None,
            resolved_constraint: None,
            memory_constraint: None,
            ctx,
            binary: None,
            pending_flags: PendingFlags::new(),
            freeze_buffer: Vec::new(),
            frozen: false,
        });
        ExecutionState {
            inner,
        }
    }

    pub fn with_binary<'b>(
        binary: &'b crate::BinaryFile<VirtualAddress64>,
        ctx: OperandCtx<'b>,
    ) -> ExecutionState<'b> {
        let mut result = ExecutionState::new(ctx);
        result.inner.binary = Some(binary);
        result
    }

    pub fn update(&mut self, operation: &Operation<'e>) {
        self.inner.update(operation);
    }

    /// Makes all of memory undefined
    pub fn clear_memory(&mut self) {
        self.inner.memory = Memory::new();
    }

    /// Tries to find an register/memory address corresponding to a resolved value.
    pub fn unresolve(&self, val: Operand<'e>) -> Option<Operand<'e>> {
        // TODO: Could also check xmm but who honestly uses it for unique storage
        for (reg, &val2) in self.inner.state.iter().enumerate().take(0x10) {
            if val == val2 {
                return Some(self.inner.ctx.register(reg as u8));
            }
        }
        None
    }
}

impl<'e> State<'e> {
    #[inline]
    fn ctx(&self) -> OperandCtx<'e> {
        self.ctx
    }

    fn registers(&self) -> &[Operand<'e>; 0x10] {
        (&self.state[..0x10]).try_into().unwrap()
    }

    fn xmm_mut(&mut self) -> &mut [Operand<'e>; 0x10 * 4] {
        let rc = self.xmm.get_or_insert_with(|| {
            let ctx = self.ctx;
            let zero = ctx.const_0();
            let mut rc = Rc::new([zero; 0x10 * 4]);
            let out = Rc::make_mut(&mut rc);
            for i in 0..0x10 {
                for j in 0..4 {
                    out[i * 4 + j] = ctx.xmm(i as u8, j as u8);
                }
            }
            rc
        });
        Rc::make_mut(rc)
    }

    fn flag_state<'a>(&'a mut self) -> exec_state::FlagState<'a, 'e> {
        exec_state::FlagState {
            flags: (&mut self.state[FLAGS_INDEX..][..6]).try_into().unwrap(),
            pending_flags: &mut self.pending_flags,
        }
    }

    fn realize_pending_flag(&mut self, flag: Flag) {
        use crate::disasm::FlagArith::*;

        self.pending_flags.clear_pending(flag);

        let ctx = self.ctx;

        if let Some(result) = self.pending_flags.sub_fast_result(ctx, flag) {
            self.state[FLAGS_INDEX + flag as usize] = result;
            return;
        }

        let result_pair = match self.pending_flags.get_result(ctx) {
            Some(s) => s,
            None => {
                self.state[FLAGS_INDEX + flag as usize] = ctx.new_undef();
                return;
            }
        };

        let result = result_pair.result;
        let base_result = result_pair.base_result;
        let arith = match self.pending_flags.update {
            Some(ref a) => a,
            None => return,
        };
        let size = arith.size;
        match flag {
            Flag::Carry | Flag::Overflow => match arith.ty {
                Add | Sub | Adc | Sbb => {
                    let val = if flag == Flag::Carry {
                        exec_state::carry_for_add_sub(ctx, arith, base_result, result)
                    } else {
                        exec_state::overflow_for_add_sub(ctx, arith, base_result, result)
                    };
                    self.state[FLAGS_INDEX + flag as usize] = val;
                }
                Xor | And | Or => {
                    self.state[FLAGS_INDEX + flag as usize] = ctx.const_0();
                }
                _ => {
                    self.state[FLAGS_INDEX + flag as usize] = ctx.new_undef();
                }
            }
            Flag::Zero => {
                let val = ctx.eq_const(result, 0);
                self.state[FLAGS_INDEX + Flag::Zero as usize] = val;
            }
            Flag::Parity => {
                let val = exec_state::calculate_parity(ctx, result);
                self.state[FLAGS_INDEX + Flag::Parity as usize] = val;
            }
            Flag::Sign => {
                let val = ctx.neq_const(
                    ctx.and_const(
                        result,
                        size.sign_bit(),
                    ),
                    0,
                );
                self.state[FLAGS_INDEX + Flag::Sign as usize] = val;
            }
            Flag::Direction => (),
        }
    }

    fn move_to_dest_invalidate_constraints<'s>(
        &'s mut self,
        dest: &DestOperand<'e>,
        value: Operand<'e>,
    ) {
        let ctx = self.ctx();
        self.unresolved_constraint = match self.unresolved_constraint {
            Some(ref s) => s.x86_invalidate_dest_operand(ctx, dest),
            None => None,
        };
        if let DestOperand::Memory(_) = dest {
            if let Some(ref mut s) = self.resolved_constraint {
                if s.invalidate_memory(ctx) == crate::exec_state::ConstraintFullyInvalid::Yes {
                    self.resolved_constraint = None;
                }
            }
            self.memory_constraint = None;
        }
        self.move_to_dest(dest, value, false)
    }

    fn move_to_dest<'s>(
        &'s mut self,
        dest: &DestOperand<'e>,
        value: Operand<'e>,
        dest_is_resolved: bool,
    ) {
        match *dest {
            DestOperand::Arch(arch_) => {
                let arch = arch_.value();
                match arch_.x86_tag() {
                    0 => {
                        let reg = (arch & 0xf) as u8;
                        self.cached_low_registers.invalidate(reg);
                        self.state[reg as usize] = value;
                    }
                    disasm::X86_REGISTER32_TAG => {
                        let reg = (arch & 0xf) as u8;
                        // 32-bit register dest clears high bits (16- or 8-bit dests don't)
                        let masked = if value.relevant_bits().end > 32 {
                            self.ctx.and_const(value, 0xffff_ffff)
                        } else {
                            value
                        };
                        self.state[reg as usize] = masked;
                        self.cached_low_registers.invalidate(reg);
                        self.cached_low_registers.set32(reg, masked);
                    }
                    disasm::X86_REGISTER16_TAG => {
                        let reg = (arch & 0xf) as u8;
                        let index = reg as usize;
                        let old = self.state[index];
                        let ctx = self.ctx;
                        let masked = if value.relevant_bits().end > 16 {
                            ctx.and_const(value, 0xffff)
                        } else {
                            value
                        };
                        let old_bits = old.relevant_bits();
                        let new = if old_bits.end <= 16 {
                            masked
                        } else {
                            ctx.or(
                                ctx.and_const(old, 0xffff_ffff_ffff_0000),
                                masked,
                            )
                        };
                        self.state[index] = new;
                        self.cached_low_registers.invalidate(reg);
                        self.cached_low_registers.set16(reg, masked);
                    }
                    disasm::X86_REGISTER8_LOW_TAG => {
                        let reg = (arch & 0xf) as u8;
                        let index = reg as usize;
                        let old = self.state[index];
                        let ctx = self.ctx;
                        let masked = if value.relevant_bits().end > 8 {
                            ctx.and_const(value, 0xff)
                        } else {
                            value
                        };
                        let old_bits = old.relevant_bits();
                        let new = if old_bits.end <= 8 {
                            masked
                        } else {
                            ctx.or(
                                ctx.and_const(old, 0xffff_ffff_ffff_ff00),
                                masked,
                            )
                        };
                        self.state[index] = new;
                        self.cached_low_registers.invalidate(reg);
                        self.cached_low_registers.set8(reg, masked);
                    }
                    disasm::X86_REGISTER8_HIGH_TAG => {
                        let reg = (arch & 0xf) as u8;
                        let index = reg as usize;
                        let old = self.state[index];
                        let ctx = self.ctx;
                        let masked = if value.relevant_bits().end > 8 {
                            ctx.and_const(value, 0xff)
                        } else {
                            value
                        };
                        let old_bits = old.relevant_bits();
                        let new = if old_bits.start >= 8 && old_bits.end <= 16 {
                            ctx.lsh_const(masked, 8)
                        } else {
                            ctx.or(
                                ctx.and_const(old, 0xffff_ffff_ffff_00ff),
                                ctx.lsh_const(masked, 8),
                            )
                        };
                        self.state[index] = new;
                        self.cached_low_registers.invalidate_for_8high(reg);
                    }
                    disasm::X86_FLAG_TAG => {
                        let flag = Flag::x86_from_arch(arch as u8);
                        self.pending_flags.make_non_pending(flag);
                        self.state[FLAGS_INDEX + flag as usize] = value;
                    }
                    disasm::X86_XMM_FPU_TAG => {
                        let xmm = self.xmm_mut();
                        let id = arch as u8;
                        if id < 128 {
                            // Xmm
                            xmm[(id & 0x3f) as usize] = value;
                        } else {
                            // Fpu, not supported by this
                        }
                    }
                    _ => (),
                }
            }
            DestOperand::Memory(ref mem) => {
                let (base, offset) = if dest_is_resolved {
                    mem.address()
                } else {
                    self.resolve_mem(mem).address()
                };
                self.write_memory(base, offset, mem.size, value)
            }
        }
    }

    fn resolve_dest(
        &mut self,
        dest: &DestOperand<'e>,
    ) -> DestOperand<'e> {
        match *dest {
            DestOperand::Memory(ref mem) => DestOperand::Memory(self.resolve_mem(mem)),
            x => x,
        }
    }

    fn resolve_mem(&mut self, mem: &MemAccess<'e>) -> MemAccess<'e> {
        let (base, offset) = mem.address();
        let (base, offset2) = self.resolve_address(base);
        self.ctx.mem_access(base, offset.wrapping_add(offset2), mem.size)
    }

    /// Resolves to (base, offset) though base itself may contain offset as well,
    /// just tries to avoid interning new constant additions
    fn resolve_address(&mut self, base: Operand<'e>) -> (Operand<'e>, u64) {
        let ctx = self.ctx;
        exec_state::resolve_address(base, ctx, &mut |x| self.resolve(x))
    }

    /// `value` is required to be `ctx.memory(mem)` that can be used if resolving won't change
    /// anything.
    fn resolve_mem_internal(
        &mut self,
        mem: &MemAccess<'e>,
        value: Operand<'e>,
    ) -> Operand<'e> {
        let resolved = self.resolve_mem(mem);
        let (base, offset) = resolved.address();
        self.read_memory_impl(base, offset, mem.size)
            .unwrap_or_else(|| {
                let binary = self.binary;
                let ctx = self.ctx;
                if (base, offset) == mem.address() {
                    // May be able to use `value` to avoid interning, but check binary section
                    // constant first.
                    if base == ctx.const_0() {
                        if let Some(binary_const) =
                            Self::resolve_binary_constant_mem_merge( binary, ctx, offset, mem.size)
                        {
                            return binary_const;
                        }
                    }
                    value
                } else {
                    Self::default_mem_value_size(binary, ctx, base, offset, mem.size)
                }
            })
    }

    /// Resolves memory operand for which the address is already resolved.
    ///
    /// Returns `None` if the memory at `address` hasn't changed.
    fn read_memory_impl(
        &mut self,
        base: Operand<'e>,
        offset: u64,
        size: MemAccessSize,
    ) -> Option<Operand<'e>> {
        let ctx = self.ctx;
        let binary = self.binary;
        self.memory.read(
            ctx,
            base,
            offset,
            size,
            MemAccessSize::Mem64,
            move |base, offset| Self::default_mem_value(binary, ctx, base, offset),
        )
    }

    fn write_memory(
        &mut self,
        base: Operand<'e>,
        offset: u64,
        size: MemAccessSize,
        value: Operand<'e>,
    ) {
        let ctx = self.ctx;
        // No guarantee that value is actually small enough to fit here.
        // Though assuming that masking isn't necessary if value is Mem32
        // (It can still gets maske if offset isn't 4-aligned, but
        // normalize_32bit should make the end result same).
        let size_bits = size.bits();
        let value = if value.relevant_bits().end <= size_bits as u8 {
            value
        } else {
            ctx.and_const(value, size.mask())
        };
        self.memory.write(ctx, value, base, offset, size, MemAccessSize::Mem64);
    }

    /// Returns value that represents unmodified memory at `[base + offset]`.
    ///
    /// Effectively just `Memx[base + offset]` unless the address is constant in binary
    /// code section. (Wouldn't probably hurt to also accept rdata at some point, but
    /// need to have a way for users to specify what `BinarySection`s should be used here)
    fn default_mem_value_size(
        binary: Option<&BinaryFile<VirtualAddress64>>,
        ctx: OperandCtx<'e>,
        base: Operand<'e>,
        offset: u64,
        size: MemAccessSize,
    ) -> Operand<'e> {
        if base == ctx.const_0() {
            if let Some(val) = Self::resolve_binary_constant_mem_merge(binary, ctx, offset, size) {
                return val;
            }
        }
        ctx.mem_any(size, base, offset)
    }

    fn default_mem_value(
        binary: Option<&BinaryFile<VirtualAddress64>>,
        ctx: OperandCtx<'e>,
        base: Operand<'e>,
        offset: u64,
    ) -> Operand<'e> {
        Self::default_mem_value_size(binary, ctx, base, offset, MemAccessSize::Mem64)
    }

    /// If `address` is in `binary`, reads that constant, handling also the
    /// case where part of the address is outside binary data and merges
    /// the constant with `Mem[address]` for the unknown data.
    fn resolve_binary_constant_mem_merge(
        binary: Option<&BinaryFile<VirtualAddress64>>,
        ctx: OperandCtx<'e>,
        address: u64,
        size: MemAccessSize,
    ) -> Option<Operand<'e>> {
        let (value, mask) = Self::resolve_binary_constant_mem(binary, address)?;
        let size_mask = size.mask();
        if mask >= size_mask {
            // Common case, entire value was in binary section
            Some(ctx.constant(value & size_mask))
        } else {
            // Rare case, read partially past section end
            Some(ctx.or_const(
                ctx.and_const(
                    ctx.mem_any(size, ctx.const_0(), address),
                    !mask,
                ),
                value,
            ))
        }
    }

    /// Returns u64 value at `address` and mask.
    /// Mask is usually u64::MAX, but if address is so close to section end that only
    /// part of the u64 could be read, returns what bytes are valid (0xff, 0xffff, 0xff_ffff, ...).
    fn resolve_binary_constant_mem(
        binary: Option<&BinaryFile<VirtualAddress64>>,
        address: u64,
    ) -> Option<(u64, u64)> {
        let section = binary.and_then(|b| {
            b.code_sections().find(|s| {
                // This checks that at least one byte at address is readable
                s.virtual_address.0 <= address && (s.virtual_address + s.virtual_size).0 > address
            })
        })?;
        let offset = address.wrapping_sub(section.virtual_address.0) as usize;
        let mut bytes = section.data.get(offset..)?;
        let val = if bytes.len() < 8 {
            // Can know that bytes.len() != 0 as the section check above will verify
            // that at least first byte of address is in section bounds
            debug_assert!(bytes.len() > 0);
            let mut buf = [0u8; 8];
            let mut mask = 0;
            for i in 0..bytes.len() {
                buf[i] = bytes[i];
                mask = (mask << 8) | 0xff;
            }
            let value = u64::from_le_bytes(buf);
            (value, mask)
        } else {
            (bytes.read_u64().unwrap_or(0), u64::MAX)
        };
        Some(val)
    }

    fn resolve(&mut self, value: Operand<'e>) -> Operand<'e> {
        if !value.needs_resolve() {
            return value;
        }
        if let Some(reg) = value.if_register() {
            return self.state[(reg & 0xf) as usize];
        }
        match *value.ty() {
            OperandType::Arch(arch) => {
                let ctx = self.ctx;
                let arch = arch.value();
                let tag = disasm::x86_arch_tag(arch);
                match tag {
                    disasm::X86_REGISTER32_TAG | disasm::X86_REGISTER16_TAG |
                        disasm::X86_REGISTER8_LOW_TAG =>
                    {
                        self.fast_resolve_partial_register(arch as u8, tag)
                    }
                    disasm::X86_REGISTER8_HIGH_TAG => {
                        let reg = (arch & 0xf) as u8;
                        let value = ctx.and_const(self.resolve_register(reg), 0xff00);
                        ctx.rsh_const(value, 8)
                    }
                    disasm::X86_FLAG_TAG => {
                        let flag = Flag::x86_from_arch(arch as u8);
                        self.resolve_flag(flag)
                    }
                    disasm::X86_XMM_FPU_TAG => {
                        match self.xmm {
                            Some(ref s) => {
                                let id = arch as u8;
                                if id < 128 {
                                    s[(id & 0x3f) as usize]
                                } else {
                                    // Fpu, not supported
                                    value
                                }
                            }
                            None => value,
                        }
                    }
                    _ => value,
                }
            }
            OperandType::Arithmetic(ref op) => {
                let left = op.left;
                let right = op.right;
                let ctx = self.ctx;
                if op.ty == ArithOpType::And {
                    if let Some(c) = right.if_constant() {
                        return self.resolve_masked(left, c);
                    }
                } else if op.ty == ArithOpType::Equal {
                    // If the value is `x == 0 == 0`, resolve x and call neq_const(x, 0),
                    // as that can give slightly better results.
                    let zero = ctx.const_0();
                    if right == zero {
                        if let Some((l, r)) = left.if_arithmetic_eq() {
                            if r == zero {
                                let l = self.resolve(l);
                                return ctx.neq_const(l, 0);
                            }
                        }
                    }
                }
                // Right is often a constant so predict that case before calling resolve
                let mut resolved_left = self.resolve(left);
                if op.left.x86_if_flag().is_some() && resolved_left.relevant_bits().end > 1 {
                    // Flag operands may contain larger than 1bit values
                    // (Especially undefined), but treat the high bits as unimportant.
                    // Though leave x == 0 untouched.
                    if right != ctx.const_0() {
                        resolved_left = ctx.and_const(resolved_left, 1);
                    }
                }
                let resolved_right = if right.needs_resolve() {
                    let mut right = self.resolve(right);
                    if op.right.x86_if_flag().is_some() && right.relevant_bits().end > 1 {
                        right = ctx.and_const(right, 1);
                    }
                    right
                } else {
                    right
                };
                ctx.arithmetic(op.ty, resolved_left, resolved_right)
            }
            OperandType::ArithmeticFloat(ref op, size) => {
                let left = self.resolve(op.left);
                let right = self.resolve(op.right);
                self.ctx.float_arithmetic(op.ty, left, right, size)
            }
            OperandType::Memory(ref mem) => {
                self.resolve_mem_internal(mem, value)
            }
            OperandType::SignExtend(val, from, to) => {
                let val = self.resolve(val);
                self.ctx.sign_extend(val, from, to)
            }
            OperandType::Select(condition, b, c) => {
                let condition = self.resolve(condition);
                match condition.is_known_bool() {
                    Some(true) => self.resolve(b),
                    Some(false) => self.resolve(c),
                    None => {
                        let b = self.resolve(b);
                        let c = self.resolve(c);
                        self.ctx.select(condition, b, c)
                    }
                }
            }
            OperandType::Undefined(_) | OperandType::Constant(_) | OperandType::Custom(_) => {
                debug_assert!(false, "Should be unreachable due to needs_resolve check");
                value
            }
        }
    }

    /// Variant of resolve which masks the returned value with `ctx.and_const(x, size.mask())`.
    ///
    /// Can avoid some simplification / interning due to using `ctx.arithmetic_masked`
    fn resolve_masked(&mut self, value: Operand<'e>, mask: u64) -> Operand<'e> {
        if !value.needs_resolve() {
            let relbit_mask = value.relevant_bits_mask();
            if mask & relbit_mask != relbit_mask {
                return self.ctx.and_const(value, mask);
            } else {
                return value;
            }
        }
        if let Some(reg) = value.if_register() {
            let r = self.try_resolve_partial_register(reg, mask);
            if let Some(r) = r {
                r
            } else {
                self.ctx.and_const(self.resolve_register(reg), mask)
            }
        } else if let OperandType::Arithmetic(ref arith) = *value.ty() {
            let left = arith.left;
            let right = arith.right;
            let ty = arith.ty;
            if ty == ArithOpType::And {
                if let Some(c) = right.if_constant() {
                    return self.resolve_masked(left, c & mask);
                }
            }
            // Right is often a constant so predict that case before calling resolve
            let resolved_left = self.resolve(left);
            let resolved_right = if right.needs_resolve() {
                self.resolve(right)
            } else {
                right
            };
            self.ctx.arithmetic_masked(ty, resolved_left, resolved_right, mask)
        } else {
            let value = self.resolve(value);
            let relbit_mask = value.relevant_bits_mask();
            if mask & relbit_mask != relbit_mask {
                self.ctx.and_const(value, mask)
            } else {
                value
            }
        }
    }

    /// Checks cached/caches `reg & ff` masks.
    fn try_resolve_partial_register(
        &mut self,
        left: u8,
        right: u64,
    ) -> Option<Operand<'e>> {
        let c = right;
        let reg = left;
        // Effectively `PARTIAL_REGISTER_MASKS.iter().position(|mask| c <= mask)`
        // but in reverse so that large masks (more common) are checked first.
        // Quite silly way to unroll it but gives nicer codegen than otherwise.
        let i = if c <= PARTIAL_REGISTER_MASKS[2] as u64 {
            if c <= PARTIAL_REGISTER_MASKS[1] as u64 {
                if c <= PARTIAL_REGISTER_MASKS[0] as u64 {
                    0
                } else {
                    1
                }
            } else {
                2
            }
        } else {
            3
        };
        if i < 3 {
            let reg = reg & 0xf;
            let reg_cache = &mut self.cached_low_registers.registers[reg as usize];
            let mask = PARTIAL_REGISTER_MASKS[i] as u64;
            let op = match reg_cache[i] {
                None => {
                    let op = self.ctx.and_const(self.state[reg as usize], mask);
                    reg_cache[i] = Some(op);
                    op
                }
                Some(x) => x,
            };
            if c == mask {
                return Some(op);
            } else {
                return Some(self.ctx.and_const(op, c));
            }
        } else {
            None
        }
    }

    /// Similar to try_resolve_partial_register but when the register type is known
    /// (ArchId tag).
    /// Always returns a value and caches what was resolved if it wasn't already.
    fn fast_resolve_partial_register(
        &mut self,
        reg: u8,
        tag: u32,
    ) -> Operand<'e> {
        // 0 = Reg8, 1 = Reg16, 2 = Reg32
        // So map 3 => 0, 2 => 1, 1 => 2,
        let i = (0b00_01_10_00usize >> (2 * (tag & 3))) & 3;

        let reg = reg & 0xf;
        let reg_cache = &mut self.cached_low_registers.registers[reg as usize];

        debug_assert!(i < 3);
        if i >= 3 {
            // Unreachable
            return self.state[0];
        }

        match reg_cache[i] {
            None => {
                let mask = PARTIAL_REGISTER_MASKS[i] as u64;
                let op = self.ctx.and_const(self.state[reg as usize], mask);
                reg_cache[i] = Some(op);
                op
            }
            Some(x) => x,
        }
    }

    fn resolve_apply_constraints(&mut self, mut op: Operand<'e>) -> Operand<'e> {
        if let Some(ref constraint) = self.unresolved_constraint {
            op = constraint.apply_to(self.ctx, op);
        }
        let val = self.resolve(op);
        if let Some(ref constraint) = self.resolved_constraint {
            constraint.apply_to(self.ctx, val)
        } else {
            val
        }
    }

    #[inline]
    fn resolve_register(&mut self, register: u8) -> Operand<'e> {
        self.state[register as usize & 0xf]
    }

    #[inline]
    fn resolve_flag(&mut self, flag: Flag) -> Operand<'e> {
        if self.pending_flags.is_pending(flag) {
            self.realize_pending_flag(flag);
        }
        self.state[FLAGS_INDEX + flag as usize]
    }

    fn move_to(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        let size = dest.x86_size();
        let resolved = if size == MemAccessSize::Mem64 {
            self.resolve(value)
        } else {
            self.resolve_masked(value, size.mask())
        };
        if self.frozen {
            let dest = self.resolve_dest(dest);
            self.freeze_buffer.push(FreezeOperation::Move(dest, resolved));
        } else {
            self.move_to_dest_invalidate_constraints(dest, resolved);
        }
    }

    fn move_resolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        if self.frozen {
            self.freeze_buffer.push(FreezeOperation::Move(*dest, value));
        } else {
            self.unresolved_constraint = None;
            self.move_to_dest(dest, value, true);
        }
    }

    fn set_register(&mut self, register: u8, value: Operand<'e>) {
        self.unresolved_constraint = None;
        let reg = register & 0xf;
        self.cached_low_registers.invalidate(reg);
        self.state[reg as usize] = value;
    }

    #[inline]
    fn set_flag(&mut self, flag: Flag, value: Operand<'e>) {
        self.unresolved_constraint = None;
        self.pending_flags.make_non_pending(flag);
        self.state[FLAGS_INDEX + flag as usize] = value;
    }

    fn set_flags_resolved(&mut self, arith: &FlagUpdate<'e>, carry: Option<Operand<'e>>) {
        if self.frozen {
            self.freeze_buffer.push(FreezeOperation::SetFlags(*arith, carry));
        } else {
            self.pending_flags.reset(arith, carry);
            // Could try to do smarter invalidation, but since in practice unresolved
            // constraints always are bunch of flags, invalidate it completely.
            self.unresolved_constraint = None;
        }
    }

    fn update(&mut self, operation: &Operation<'e>) {
        let ctx = self.ctx;
        match *operation {
            Operation::Move(ref dest, value) => {
                self.move_to(dest, value);
            }
            Operation::ConditionalMove(ref dest, value, cond) => {
                match self.resolve(cond).if_constant() {
                    Some(0) => (),
                    Some(_) => {
                        self.move_to(dest, value);
                    }
                    None => {
                        self.move_to(dest, ctx.new_undef());
                    }
                }
            }
            Operation::Freeze => {
                self.frozen = true;
                ctx.swap_freeze_buffer(&mut self.freeze_buffer);
            }
            Operation::Unfreeze => {
                self.frozen = false;
                let mut i = 0;
                while i < self.freeze_buffer.len() {
                    let op = self.freeze_buffer[i];
                    match op {
                        FreezeOperation::Move(ref dest, val) => {
                            self.move_resolved(dest, val);
                        }
                        FreezeOperation::SetFlags(ref arith, carry) => {
                            self.set_flags_resolved(arith, carry);
                        }
                    }
                    i = i.wrapping_add(1);
                }
                self.freeze_buffer.clear();
                ctx.swap_freeze_buffer(&mut self.freeze_buffer);
            }
            Operation::Call(_) => {
                self.unresolved_constraint = None;
                if let Some(ref mut c) = self.resolved_constraint {
                    if c.invalidate_memory(ctx) == crate::exec_state::ConstraintFullyInvalid::Yes {
                        self.resolved_constraint = None
                    }
                }
                self.memory_constraint = None;
                static UNDEF_REGISTERS: &[u8] = &[0, 1, 2, 8, 9, 10, 11];
                for &i in UNDEF_REGISTERS.iter() {
                    self.state[i as usize] = ctx.new_undef();
                    self.cached_low_registers.invalidate(i);
                }
                for i in 0..5 {
                    self.state[FLAGS_INDEX + i] = ctx.new_undef();
                }
                self.state[FLAGS_INDEX + Flag::Direction as usize] = ctx.const_0();
                self.pending_flags = PendingFlags::new();
            }
            Operation::Special(_) => {
            }
            Operation::SetFlags(ref arith) => {
                let left = self.resolve(arith.left);
                let right = self.resolve(arith.right);
                let arith = FlagUpdate {
                    left,
                    right,
                    ty: arith.ty,
                    size: arith.size,
                };
                let carry = match arith.ty {
                    FlagArith::Adc | FlagArith::Sbb => Some(self.resolve_flag(Flag::Carry)),
                    _ => None,
                };
                self.set_flags_resolved(&arith, carry);
            }
            Operation::Jump { .. } | Operation::Return(_) | Operation::Error(..) => (),
        }
    }

    fn add_memory_constraint(&mut self, constraint: Operand<'e>) {
        if let Some(old) = self.memory_constraint {
            let ctx = self.ctx();
            self.memory_constraint = Some(Constraint(ctx.and(old.0, constraint)));
        } else {
            self.memory_constraint = Some(Constraint(constraint));
        }
    }

    fn add_to_unresolved_constraint(&mut self, constraint: Operand<'e>) {
        let ctx = self.ctx();
        if let Some((flag, value)) = exec_state::is_flag_const_constraint(ctx, constraint) {
            // If we have just constraint of flag == 0 or flag != 0, assign const 0/1 there
            // instead of adding it to constraint.
            // Not doing for non-flags as it'll end up making values such as func
            // args too eagerly undef once they get compared once.
            self.set_flag(flag, value);
            return;
        }
        if let Some(old) = self.unresolved_constraint {
            self.unresolved_constraint = Some(Constraint(ctx.and(old.0, constraint)));
        } else {
            self.unresolved_constraint = Some(Constraint(constraint));
        }
    }

    fn add_resolved_constraint(&mut self, constraint: Constraint<'e>) {
        // If recently accessed memory location also has the same resolved value as this
        // constraint, add it to the constraint as well.
        // As it only works on recently accessed memory, it is very much a good-enough
        // heuristic and not a foolproof solution. A complete, but slower way would
        // be having a state merge function that is merges constraints depending on
        // what the values inside constraints were merged to.
        self.resolved_constraint = Some(constraint);
        let ctx = self.ctx();
        if let OperandType::Arithmetic(ref arith) = *constraint.0.ty() {
            let const_other = if arith.left.if_constant().is_some() {
                Some((arith.left, arith.right, false))
            } else if arith.right.if_constant().is_some() {
                Some((arith.right, arith.left, true))
            } else {
                None
            };
            if let Some((const_op, other_op, const_right)) = const_other {
                if let Some((base, offset, size)) =
                    self.memory.fast_reverse_lookup(ctx, other_op, u64::MAX, 3)
                {
                    let val = ctx.mem_any(size, base, offset);
                    let (l, r) = match const_right {
                        true => (val, const_op),
                        false => (const_op, val),
                    };
                    self.add_memory_constraint(ctx.arithmetic(arith.ty, l, r));
                }
                for i in 0..0x10 {
                    if self.state[i] == other_op {
                        let val = ctx.register(i as u8);
                        let (l, r) = match const_right {
                            true => (val, const_op),
                            false => (const_op, val),
                        };
                        self.add_to_unresolved_constraint(
                            ctx.arithmetic(arith.ty, l, r)
                        );
                    }
                }
            }
            let maybe_wanted_flags = (1 << Flag::Zero as u8) |
                (1 << Flag::Carry as u8) |
                (1 << Flag::Sign as u8);
            if arith.ty == ArithOpType::Equal &&
                self.pending_flags.pending_bits & maybe_wanted_flags != 0
            {
                if let Some(ref flag_update) = self.pending_flags.update {
                    let flags = exec_state::flags_for_resolved_constraint_eq_check(
                        flag_update,
                        arith,
                        ctx,
                    );
                    for &flag in flags {
                        if self.pending_flags.is_pending(flag) {
                            self.realize_pending_flag(flag);
                        }
                    }
                }
            }
        }
        // Check if the constraint ends up making a flag always true
        // (Could do more extensive checks in state but this is cheap-ish,
        // costing flag realization if it is decided to be needed above,
        // and has uses for control flow tautologies)
        for i in 0..6 {
            if self.state[FLAGS_INDEX + i] == constraint.0 {
                if self.pending_flags.pending_bits & (1 << i) == 0 {
                    self.state[FLAGS_INDEX + i] = ctx.const_1();
                }
            }
        }
    }
}

#[inline]
fn check_merge_eq<'e>(a: Operand<'e>, b: Operand<'e>) -> bool {
    a == b || a.is_undefined()
}

fn merge_op<'e>(ctx: OperandCtx<'e>, a: Operand<'e>, b: Operand<'e>) -> Operand<'e> {
    match a == b || a.is_undefined() {
        true => a,
        false => ctx.new_undef(),
    }
}

fn xmm_merge_changed<'e>(
    old: Option<&Rc<[Operand<'e>; 0x10 * 4]>>,
    new: Option<&Rc<[Operand<'e>; 0x10 * 4]>>,
) -> bool {
    if let Some(old) = old {
        if let Some(new) = new {
            if Rc::ptr_eq(old, new) {
                return false;
            } else {
                return old.iter().zip(new.iter())
                    .any(|(&a, &b)| !check_merge_eq(a, b));
            }
        } else {
            // Merge changed if not all (old are initial values or undef)
            for i in 0..(0x10 * 4) {
                let val = old[i as usize];
                if !val.is_undefined() {
                    if val.if_xmm() != Some((i >> 2, i & 3)) {
                        return true;
                    }
                }
            }
        }
    } else if let Some(new) = new {
        // Merge changed if not all new are initial values
        for i in 0..(0x10 * 4) {
            let val = new[i as usize];
            if val.if_xmm() != Some((i >> 2, i & 3)) {
                return true;
            }
        }
    } else {
        // Both none, never merge changed
    }
    false
}

fn merge_xmm_to_old<'e>(
    // Out is assumed to be also "old" values
    out: &mut Option<Rc<[Operand<'e>; 0x10 * 4]>>,
    new: Option<&Rc<[Operand<'e>; 0x10 * 4]>>,
    ctx: OperandCtx<'e>,
) {
    if let Some(out) = out {
        let result = Rc::make_mut(out);
        if let Some(new) = new {
            for i in 0..result.len() {
                result[i] = merge_op(ctx, result[i], new[i]);
            }
        } else {
            // Anything in out that is not undef and differs from default value becomes undef
            for i in 0..(0x10 * 4) {
                let val = &mut result[i as usize];
                if !val.is_undefined() && val.if_xmm() != Some((i >> 2, i & 3)) {
                    *val = ctx.new_undef();
                }
            }
        }
    } else {
        if let Some(new) = new {
            let result_rc = out.insert(new.clone());
            let result = Rc::make_mut(result_rc);
            // Anything differing from default value becomes undef
            for i in 0..(0x10 * 4) {
                let val = &mut result[i as usize];
                if val.if_xmm() != Some((i >> 2, i & 3)) {
                    *val = ctx.new_undef();
                }
            }
        }
    }
}

/// If `old` and `new` have different fields, and the old field is not undefined,
/// return `ExecutionState` which has the differing fields replaced with (a separate) undefined.
fn merge_states<'a: 'r, 'r>(
    old: &'r mut State<'a>,
    new: &'r mut State<'a>,
    cache: &'r mut MergeStateCache<'a>,
) -> Option<ExecutionState<'a>> {

    let ctx = old.ctx;
    let xmm_changed = xmm_merge_changed(old.xmm.as_ref(), new.xmm.as_ref());
    let unresolved_constraint =
        exec_state::merge_constraint(ctx, old.unresolved_constraint, new.unresolved_constraint);
    let resolved_constraint =
        exec_state::merge_constraint(ctx, old.resolved_constraint, new.resolved_constraint);
    let memory_constraint =
        exec_state::merge_constraint(ctx, old.memory_constraint, new.memory_constraint);
    let changed = (
            old.registers().iter().zip(new.registers().iter())
                .any(|(&a, &b)| !check_merge_eq(a, b))
        ) || (
            if old.memory.is_same(&new.memory) {
                false
            } else {
                match cache.get_compare_result(&old.memory, &new.memory) {
                    Some(s) => s,
                    None => {
                        let result = old.memory.has_merge_changed(&new.memory);
                        cache.set_compare_result(&old.memory, &new.memory, result);
                        result
                    }
                }
            }
        ) ||
        xmm_changed ||
        unresolved_constraint != old.unresolved_constraint ||
        resolved_constraint != old.resolved_constraint ||
        memory_constraint != old.memory_constraint ||
        exec_state::flags_merge_changed(&mut old.flag_state(), &mut new.flag_state(), ctx);
    if changed {
        let zero = ctx.const_0();
        let mut state = array_init::array_init(|_| zero);
        for i in 0..16 {
            state[i] = merge_op(ctx, old.state[i], new.state[i]);
        }
        let pending_flags = exec_state::merge_flags(
            &mut old.flag_state(),
            &mut new.flag_state(),
            (&mut state[FLAGS_INDEX..][..6]).try_into().unwrap(),
            ctx,
        );
        let mut cached_low_registers = CachedLowRegisters::new();
        for i in 0..16 {
            // If register didn't change, can take any set cached registers
            // from either state to result.
            if old.state[i] == new.state[i] {
                let old_reg = &old.cached_low_registers.registers[i];
                let new_reg = &new.cached_low_registers.registers[i];
                for j in 0..old_reg.len() {
                    if old_reg[j].is_some() {
                        cached_low_registers.registers[i][j] = old_reg[j];
                    } else if new_reg[j].is_some() {
                        cached_low_registers.registers[i][j] = new_reg[j];
                    }
                }
            }
        }
        let mut xmm = old.xmm.clone();
        if xmm_changed {
            merge_xmm_to_old(&mut xmm, new.xmm.as_ref(), ctx);
        }
        let memory = cache.merge_memory(&old.memory, &new.memory, ctx);
        let inner = Box::alloc().init(State {
            state,
            xmm,
            cached_low_registers,
            memory,
            unresolved_constraint,
            resolved_constraint,
            memory_constraint,
            pending_flags,
            ctx,
            binary: old.binary,
            // Freeze buffer is intended to be empty at merge points,
            // not going to support merging it
            freeze_buffer: Vec::new(),
            frozen: false,
        });
        Some(ExecutionState {
            inner,
        })
    } else {
        None
    }
}
