//! Analyzer component to keep track of memory accessed, and make all of them
//! undefined when requested (Usually on calls).

use std::hash::BuildHasherDefault;
use std::mem;
use std::ptr;

use bumpalo::collections::Vec as BumpVec;
use bumpalo::Bump;
use fxhash::FxHasher;
use hashbrown::HashSet;

use crate::analysis::{Analyzer, Control};
use crate::exec_state::{ExecutionState, VirtualAddress};
use crate::operand::{OperandHashByAddress};
use crate::{
    BinaryFile, DestOperand, MemAccess, MemAccessSize,
    Operation, Operand, OperandCtx, OperandType,
};

use super::{ProjectState};

/// State that has to be included in Analyzer::State for MemoryAccessTracker to work.
/// `Analyzer::State` has to implement `ProjectState<MemoryAccessState>` too.
#[derive(Clone)]
pub struct MemoryAccessState<'bump, 'e> {
    /// More accesses if the vector grows too large without a call
    immutable: Option<&'bump MemoryAccessState<'bump, 'e>>,
    /// Even more accesses if the states being merged have different immutables
    immutable_merge: Option<&'bump MemoryAccessState<'bump, 'e>>,
    accesses: BumpVec<'bump, (Operand<'e>, u64)>,
}

impl<'bump, 'e> MemoryAccessState<'bump, 'e> {
    pub fn new(bump: &'bump Bump) -> MemoryAccessState<'bump, 'e> {
        MemoryAccessState {
            immutable: None,
            immutable_merge: None,
            accesses: BumpVec::with_capacity_in(8, bump),
        }
    }

    pub fn merge(&mut self, newer: Self) {
        // While duplicates in `accesses` are fine, merge may end up having edge cases
        // with switches where the access count gets out of hand, so have some logic
        // to keep it smaller.
        let do_dedup = (self.accesses.len() > 50 && newer.accesses.len() > 50) ||
            self.accesses.len() + newer.accesses.len() > 400;
        self.accesses.extend(newer.accesses.into_iter());
        if do_dedup {
            self.accesses.sort_unstable();
            self.accesses.dedup();
        }
        if let Some(old_imm) = self.immutable {
            if let Some(new_imm) = newer.immutable {
                if !ptr::eq(old_imm, new_imm) {
                    self.immutable_merge = Some(new_imm);
                }
            }
        } else {
            self.immutable = newer.immutable;
        }
    }
}

pub struct MemoryAccessTracker<'bump, 'e, Va: VirtualAddress> {
    bump: &'bump Bump,
    part_buffer: BumpVec<'bump, Operand<'e>>,
    immutable_buffer: BumpVec<'bump, &'bump MemoryAccessState<'bump, 'e>>,
    checked_immutable_buffer: BumpVec<'bump, &'bump MemoryAccessState<'bump, 'e>>,
    /// Memory that won't become undef on make_accessed_memory_undef calls.
    immutable_variables: HashSet<(OperandHashByAddress<'e>, u64), BuildHasherDefault<FxHasher>>,
    data_start: Va,
    data_end: Va,
    custom: Operand<'e>,
}

impl<'bump, 'e, Va: VirtualAddress> MemoryAccessTracker<'bump, 'e, Va>
{
    pub fn new(
        bump: &'bump Bump,
        binary: &BinaryFile<Va>,
        ctx: OperandCtx<'e>,
        custom: u32,
        // These variables will not be made undef on calls
        immutable_variables: &[Operand<'e>],
    ) -> MemoryAccessTracker<'bump, 'e, Va> {
        let data = binary.section(b".data\0\0\0").unwrap();
        let mut immutable_set = HashSet::with_capacity_and_hasher(16, Default::default());
        let mut part_buffer = BumpVec::with_capacity_in(16, bump);
        part_buffer.clear();
        for &op in immutable_variables {
            part_buffer.push(op);
            while let Some(part) = op_parts_next(&mut part_buffer)  {
                if let Some(mem) = part.if_memory() {
                    let (base, offset) = mem.address();
                    immutable_set.insert((OperandHashByAddress(base), offset));
                }
            }
        }
        MemoryAccessTracker {
            bump,
            part_buffer,
            immutable_buffer: BumpVec::new_in(bump),
            checked_immutable_buffer: BumpVec::new_in(bump),
            immutable_variables: immutable_set,
            data_start: data.virtual_address,
            data_end: data.virtual_address + data.virtual_size,
            custom: ctx.custom(custom),
        }
    }

    fn new_undef(&self, ctx: OperandCtx<'e>) -> Operand<'e> {
        ctx.xor(self.custom, ctx.new_undef())
    }
}

impl<'bump, 'e, Va: VirtualAddress> MemoryAccessTracker<'bump, 'e, Va> {
    pub fn operation<A, E, S>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
        op: &Operation<'e>,
    )
    where A: Analyzer<'e, Exec = E, State = S>,
          E: ExecutionState<'e, VirtualAddress = Va>,
          S: ProjectState<MemoryAccessState<'bump, 'e>>,
    {
        let ctx = ctrl.ctx();
        let mut buf = [ctx.const_0(); 2];
        match *op {
            Operation::SetFlags(ref arith) => {
                buf[0] = arith.left;
                buf[1] = arith.right;
            }
            Operation::Move(ref dest, value) => {
                if let DestOperand::Memory(ref mem) = *dest {
                    self.add_address(ctrl, mem);
                }
                buf[0] = value;
            }
            Operation::ConditionalMove(ref dest, value, cond) => {
                if let DestOperand::Memory(ref mem) = *dest {
                    self.add_address(ctrl, mem);
                }
                buf[0] = value;
                buf[1] = cond;
            }
            _ => return,
        }
        self.part_buffer.clear();
        for op in buf {
            self.part_buffer.push(op);
            while let Some(part) = op_parts_next(&mut self.part_buffer)  {
                if let Some(mem) = part.if_memory() {
                    self.add_address(ctrl, mem);
                }
            }
        }
    }

    pub fn branch_end<A, E, S>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
    )
    where A: Analyzer<'e, Exec = E, State = S>,
          E: ExecutionState<'e, VirtualAddress = Va>,
          S: ProjectState<MemoryAccessState<'bump, 'e>>,
    {
        let state = project(ctrl.user_state());
        if state.accesses.len() > 60 {
            let old_state = mem::replace(state, MemoryAccessState {
                immutable: None,
                immutable_merge: None,
                accesses: BumpVec::new_in(self.bump),
            });
            *state = MemoryAccessState {
                immutable: Some(self.bump.alloc(old_state)),
                immutable_merge: None,
                accesses: BumpVec::new_in(self.bump),
            };
        }
    }

    fn add_address<A, E, S>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
        unresolved: &MemAccess<'e>,
    )
    where A: Analyzer<'e, Exec = E, State = S>,
          E: ExecutionState<'e, VirtualAddress = Va>,
          S: ProjectState<MemoryAccessState<'bump, 'e>>,
    {
        let mem = ctrl.resolve_mem(&unresolved);
        let (base, offset) = mem.address();
        if let Some(addr) = mem.if_constant_address().map(|x| Va::from_u64(x)) {
            // Only track stores/reads from .data, assume everything else with a constant
            // address will not change.
            if addr < self.data_start || addr >= self.data_end {
                return;
            }
        }
        if self.immutable_variables.contains(&(OperandHashByAddress(base), offset)) {
            return;
        }
        let end_offset = offset.wrapping_add(mem.size.bytes().into());
        let offset = offset & !7;
        let end_offset = end_offset & !7;
        let state = project(ctrl.user_state());
        state.accesses.push((base, offset));
        if end_offset != offset {
            state.accesses.push((base, end_offset));
        }
    }

    pub fn make_accessed_memory_undef<A, E, S>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
    )
    where A: Analyzer<'e, Exec = E, State = S>,
          E: ExecutionState<'e, VirtualAddress = Va>,
          S: ProjectState<MemoryAccessState<'bump, 'e>>,
    {
        let state = project(ctrl.user_state());
        state.accesses.sort_unstable();
        state.accesses.dedup();
        let mut accesses = mem::replace(&mut state.accesses, BumpVec::new_in(self.bump));
        let ctx = ctrl.ctx();
        for &(base, offset) in accesses.iter() {
            let mem = ctx.mem_access(base, offset, MemAccessSize::Mem64);
            ctrl.move_resolved(&DestOperand::Memory(mem), self.new_undef(ctx));
        }
        accesses.clear();
        let state = project(ctrl.user_state());
        if let Some(imm) = state.immutable {
            self.immutable_buffer.clear();
            self.checked_immutable_buffer.clear();
            self.immutable_buffer.push(imm);
            if let Some(imm) = state.immutable_merge {
                self.immutable_buffer.push(imm);
            }
            self.make_immutable_buffer_mem_undef(ctrl);
        }
        let state = project(ctrl.user_state());
        state.accesses = accesses;
        state.immutable = None;
        state.immutable_merge = None;
    }

    fn make_immutable_buffer_mem_undef<A, E, S>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
    )
    where A: Analyzer<'e, Exec = E, State = S>,
          E: ExecutionState<'e, VirtualAddress = Va>,
          S: ProjectState<MemoryAccessState<'bump, 'e>>,
    {
        while let Some(state) = self.immutable_buffer.pop() {
            if self.checked_immutable_buffer.iter().any(|&x| ptr::eq(x, state)) {
                continue;
            }
            self.checked_immutable_buffer.push(state);
            if let Some(imm) = state.immutable {
                self.immutable_buffer.push(imm);
                if let Some(imm) = state.immutable_merge {
                    self.immutable_buffer.push(imm);
                }
            }
            let ctx = ctrl.ctx();
            for &(base, offset) in state.accesses.iter() {
                let mem = ctx.mem_access(base, offset, MemAccessSize::Mem64);
                ctrl.move_resolved(&DestOperand::Memory(mem), self.new_undef(ctx));
            }
        }
    }
}

fn project<'b, 'e, S: ProjectState<MemoryAccessState<'b, 'e>>>(
    s: &mut S,
) -> &mut MemoryAccessState<'b, 'e> {
    s.project()
}

fn op_parts_next<'e>(buffer: &mut BumpVec<'_, Operand<'e>>) -> Option<Operand<'e>> {
    let mut op = buffer.pop()?;
    loop {
        if let OperandType::Arithmetic(ref a) = *op.ty() {
            buffer.push(a.left);
            op = a.right;
        } else if let OperandType::ArithmeticFloat(ref a, _) = *op.ty() {
            buffer.push(a.left);
            op = a.right;
        } else {
            return Some(op);
        }
    }
}
