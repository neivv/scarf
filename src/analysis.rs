use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::mem;

use crate::cfg::{self, CfgNode, CfgOutEdges, NodeLink, OutEdgeCondition};
use crate::disasm::{Operation};
use crate::exec_state::{self, Constraint, Disassembler, ExecutionState, MergeStateCache};
use crate::exec_state::VirtualAddress as VaTrait;
use crate::light_byteorder::ReadLittleEndian;
use crate::operand::{MemAccessSize, Operand, OperandCtx};
use crate::{BinaryFile, BinarySection, VirtualAddress, VirtualAddress64};

pub use crate::disasm::Error;

pub type Cfg<'a, E, S> = cfg::Cfg<'a, CfgState<'a, E, S>>;

#[derive(Debug)]
pub struct CfgState<'a, E: ExecutionState<'a>, S: AnalysisState> {
    data: (E, S),
    phantom: std::marker::PhantomData<&'a ()>,
}

impl<'a, E: ExecutionState<'a>, S: AnalysisState> cfg::CfgState for CfgState<'a, E, S> {
    type VirtualAddress = E::VirtualAddress;
}

impl<'a, E: ExecutionState<'a>, S: AnalysisState> CfgState<'a, E, S> {
    pub fn exec_state(&self) -> &E {
        &self.data.0
    }

    pub fn user_state(&self) -> &S {
        &self.data.1
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncCallPair<Va: VaTrait> {
    pub caller: Va,
    pub callee: Va,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncPtrPair<Va: VaTrait> {
    pub address: Va,
    pub callee: Va,
}

/// Sorted by callee, so looking up callers can be done with bsearch
pub fn find_functions_with_callers<'a, E: ExecutionState<'a>>(
    file: &BinaryFile<E::VirtualAddress>,
) -> Vec<FuncCallPair<E::VirtualAddress>> {
    E::find_functions_with_callers(file)
}

pub fn find_functions_with_callers_x86(
    file: &BinaryFile<VirtualAddress>,
) -> Vec<FuncCallPair<VirtualAddress>> {
    let mut out = Vec::with_capacity(800);
    for code in file.code_sections() {
        let data = &code.data;
        out.extend(
            data.iter().enumerate()
                .filter(|&(_, &x)| x == 0xe8 || x == 0xe9)
                .flat_map(|(idx, _)| {
                    data.get(idx + 1..).and_then(|mut x| x.read_u32().ok())
                        .map(|relative| FuncCallPair {
                            caller: code.virtual_address + idx as u32,
                            callee: VirtualAddress(
                                code.virtual_address.0
                                    .wrapping_add((idx as u32 + 5).wrapping_add(relative))
                            ),
                        })
                })
                .filter(|pair| {
                    pair.callee >= code.virtual_address &&
                        pair.callee < code.virtual_address + data.len() as u32
                })
        );
    }
    out.sort_unstable_by_key(|x| (x.callee, x.caller));
    out
}

pub fn find_functions_with_callers_x86_64(
    file: &BinaryFile<VirtualAddress64>,
) -> Vec<FuncCallPair<VirtualAddress64>> {
    let mut out = Vec::with_capacity(800);
    for code in file.code_sections() {
        let data = &code.data;
        // Ignoring long jumps here, unlike in 32-bit.
        // Should probably ignore them there as well but I don't want to change it right now.
        out.extend(
            data.iter().enumerate()
                .filter(|&(_, &x)| x == 0xe8)
                .flat_map(|(idx, _)| {
                    data.get(idx + 1..).and_then(|mut x| x.read_i32().ok())
                        .map(|relative| FuncCallPair {
                            caller: code.virtual_address + idx as u32,
                            callee: VirtualAddress64(
                                code.virtual_address.0
                                    .wrapping_add(
                                        (idx as i64 + 5).wrapping_add(relative as i64) as u64
                                    )
                            ),
                        })
                })
                .filter(|pair| {
                    pair.callee >= code.virtual_address &&
                        pair.callee < code.virtual_address + data.len() as u32
                })
        );
    }
    out.sort_unstable_by_key(|x| (x.callee, x.caller));
    out
}

fn find_functions_from_calls<'a, E: ExecutionState<'a>>(
    code: &[u8],
    section_base: E::VirtualAddress,
    out: &mut Vec<E::VirtualAddress>,
) {
    E::find_functions_from_calls(code, section_base, out)
}

pub(crate) fn find_functions_from_calls_x86(
    code: &[u8],
    section_base: VirtualAddress,
    out: &mut Vec<VirtualAddress>,
) {
    out.extend({
        code.iter().enumerate()
            .filter(|&(_, &x)| x == 0xe8)
            .flat_map(|(idx, _)| {
                code.get(idx + 1..).and_then(|mut x| x.read_u32().ok())
                    .map(|relative| (idx as u32 + 5).wrapping_add(relative))
                    .filter(|&target| {
                        (target as usize) < code.len() - 5
                    })
                    .map(|x| section_base + x)
            })
    });
}

pub(crate) fn find_functions_from_calls_x86_64(
    code: &[u8],
    section_base: VirtualAddress64,
    out: &mut Vec<VirtualAddress64>,
) {
    out.extend({
        code.iter().enumerate()
            .filter(|&(_, &x)| x == 0xe8)
            .flat_map(|(idx, _)| {
                code.get(idx + 1..).and_then(|mut x| x.read_i32().ok())
                    .map(|relative| (idx as i64 + 5).wrapping_add(relative as i64) as u64)
                    .filter(|&target| {
                        (target as usize) < code.len() - 5
                    })
                    .map(|x| VirtualAddress64(section_base.0 + x))
            })
    });
}

pub(crate) fn function_ranges_from_exception_info_x86_64(
    file: &BinaryFile<VirtualAddress64>,
) -> Result<Vec<(u32, u32)>, crate::OutOfBounds> {
    let pe_header = file.base + file.read_u32(file.base + 0x3c)?;
    let exception_offset = file.read_u32(pe_header + 0xa0)?;
    let exception_len = file.read_u32(pe_header + 0xa4)?;
    let exceptions = file.slice_from(exception_offset..exception_offset + exception_len)?;
    Ok(exceptions.chunks_exact(0xc).map(|chunk| {
        let start = ReadLittleEndian::read_u32(&mut &chunk[..]).unwrap();
        let end = ReadLittleEndian::read_u32(&mut &chunk[4..]).unwrap();
        (start, end)
    }).collect())
}

/// Attempts to find functions of a binary, accepting ones that are called/long jumped to
/// from somewhere.
/// Returns addresses relative to start of code section.
pub fn find_functions<'a, E: ExecutionState<'a>>(
    file: &BinaryFile<E::VirtualAddress>,
    relocs: &[E::VirtualAddress],
) -> Vec<E::VirtualAddress> {
    let confirmed_ranges = E::function_ranges_from_exception_info(file)
        .unwrap_or_else(|_| Vec::new());
    let mut called_functions = Vec::with_capacity(confirmed_ranges.len());
    for section in file.code_sections() {
        find_functions_from_calls::<E>(
            &section.data,
            section.virtual_address,
            &mut called_functions,
        );
    }
    // Filter out functions that are inside other function's ranges
    if !confirmed_ranges.is_empty() {
        let mut confirmed_pos = confirmed_ranges.iter().cloned();
        let first = confirmed_pos.next().unwrap();
        let mut current_start = file.base + first.0;
        let mut current_end = file.base + first.1;
        called_functions.sort_unstable();
        called_functions.dedup();
        called_functions.retain(|&addr| {
            loop {
                if addr <= current_start {
                    // Either equal to current (Valid function with exception info)
                    // or less than (Likely a (leaf) function which won't ever unwind)
                    return true;
                }
                if addr < current_end {
                    // In middle of another function, so it's not a real address
                    return false;
                }
                let next = match confirmed_pos.next() {
                    Some(s) => s,
                    None => (u32::max_value(), u32::max_value()),
                };
                current_start = file.base + next.0;
                current_end = file.base + next.1;
            }
        })
    }
    called_functions.extend(confirmed_ranges.iter().map(|x| file.base + x.0));
    called_functions.extend(find_funcptrs(file, relocs).iter().map(|x| x.callee));
    called_functions.sort_unstable();
    called_functions.dedup();
    called_functions
}

// Sorted by address
pub fn find_switch_tables<Va: VaTrait>(
    file: &BinaryFile<Va>,
    relocs: &[Va],
) -> Vec<FuncPtrPair<Va>> {
    let mut out = Vec::with_capacity(4096);
    for sect in file.code_sections() {
        collect_relocs_pointing_to_code(sect, relocs, sect, &mut out);
    }
    out
}

fn find_funcptrs<Va: VaTrait>(file: &BinaryFile<Va>, relocs: &[Va]) -> Vec<FuncPtrPair<Va>> {
    let mut out = Vec::with_capacity(4096);
    let code = file.code_section();
    for sect in file.sections.iter().filter(|sect| {
        &sect.name[..] == b".data\0\0\0" ||
            &sect.name[..] == b".rdata\0\0" ||
            &sect.name[..] == b".text\0\0\0"
    }) {
        collect_relocs_pointing_to_code(code, relocs, sect, &mut out);
    }
    out
}

fn collect_relocs_pointing_to_code<Va: VaTrait>(
    code: &BinarySection<Va>,
    relocs: &[Va],
    sect: &BinarySection<Va>,
    out: &mut Vec<FuncPtrPair<Va>>,
) {
    let start = match relocs.binary_search(&sect.virtual_address) {
        Ok(o) => o,
        Err(e) => e,
    };
    let end = match relocs.binary_search(&(sect.virtual_address + sect.data.len() as u32)) {
        Ok(o) => o,
        Err(e) => e,
    };
    let funcs = relocs[start..end].iter()
        .flat_map(|&addr| {
            // TODO broken on x86 for addresses right at end of section as it can't be
            // read as u64
            let offset = (addr.as_u64() - sect.virtual_address.as_u64()) as usize;
            (&sect.data[offset..]).read_u64().ok()
                .map(|x| (addr, Va::from_u64(x)))
        })
        .filter(|&(_src_addr, func_addr)| {
            let code_size = code.data.len() as u32;
            func_addr >= code.virtual_address && func_addr < code.virtual_address + code_size
        })
        .filter(|&(_src_addr, func_addr)| {
            // Skip relocs pointing to other relocs, as they are either switch table
            // or vtable refs in code, which aren't executable code.
            // Ideally find_funcptrs could also detect switch cases and not check them,
            // but not sure how to discern that from vtables/fnptr arrays.
            let is_switch_jump = relocs.binary_search(&func_addr).is_ok();
            !is_switch_jump
        })
        .map(|(src_addr, func_addr)| FuncPtrPair {
            address: src_addr,
            callee: func_addr,
        });
    out.extend(funcs);
}

macro_rules! try_get {
    ($slice:expr, $range:expr) => {
        match $slice.get($range) {
            Some(s) => s,
            None => return Err(crate::OutOfBounds),
        }
    }
}

pub fn find_relocs<'a, E: ExecutionState<'a>>(
    file: &BinaryFile<E::VirtualAddress>,
) -> Result<Vec<E::VirtualAddress>, crate::OutOfBounds> {
    E::find_relocs(file)
}

pub fn find_relocs_x86(file: &BinaryFile<VirtualAddress>) -> Result<Vec<VirtualAddress>, crate::OutOfBounds> {
    let pe_header = file.base + file.read_u32(file.base + 0x3c)?;
    let reloc_offset = file.read_u32(pe_header + 0xa0)?;
    let reloc_len = file.read_u32(pe_header + 0xa4)?;
    let relocs = file.slice_from(reloc_offset..reloc_offset + reloc_len)?;
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < relocs.len() {
        let rva = try_get!(relocs, offset..).read_u32()?;
        let base = file.base + rva;
        let size = try_get!(relocs, offset + 4..).read_u32()? as usize;
        if size < 8 {
            break;
        }
        let block_relocs = try_get!(relocs, offset + 8..offset + size);
        for mut reloc in block_relocs.chunks_exact(2) {
            if let Ok(c) = reloc.read_u16() {
                if c & 0xf000 == 0x3000 {
                    result.push(base + u32::from(c & 0xfff));
                }
            }
        }
        offset += size;
    }
    result.sort_unstable();
    Ok(result)
}

pub fn find_relocs_x86_64(
    file: &BinaryFile<VirtualAddress64>,
) -> Result<Vec<VirtualAddress64>, crate::OutOfBounds> {
    let pe_header = file.base + file.read_u32(file.base + 0x3c)?;
    let reloc_offset = file.read_u32(pe_header + 0xb0)?;
    let reloc_len = file.read_u32(pe_header + 0xb4)?;
    let relocs = file.slice_from(reloc_offset..reloc_offset + reloc_len)?;
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < relocs.len() {
        let rva = try_get!(relocs, offset..).read_u32()?;
        let base = file.base + rva;
        let size = try_get!(relocs, offset + 4..).read_u32()? as usize;
        if size < 8 {
            break;
        }
        let block_relocs = try_get!(relocs, offset + 8..offset + size);
        for mut reloc in block_relocs.chunks_exact(2) {
            if let Ok(c) = reloc.read_u16() {
                if c & 0xf000 == 0xa000 {
                    result.push(base + u32::from(c & 0xfff));
                }
            }
        }
        offset += size;
    }
    result.sort_unstable();
    Ok(result)
}

pub struct RelocValues<Va: VaTrait> {
    pub address: Va,
    pub value: Va,
}

/// The returned array is sorted by value
pub fn relocs_with_values<Va: VaTrait>(
    file: &BinaryFile<Va>,
    mut relocs: &[Va],
) -> Result<Vec<RelocValues<Va>>, crate::OutOfBounds> {
    let mut result = Vec::with_capacity(relocs.len());
    'outer: while !relocs.is_empty() {
        let (section, reloc_count, start_address) = {
            if let Some(section) = file.section_by_addr(relocs[0]) {
                let end = section.virtual_address + section.data.len() as u32;
                let count = relocs.binary_search(&end).unwrap_or_else(|x| x);
                (&section.data, count, section.virtual_address)
            } else {
                relocs = &relocs[1..];
                continue 'outer;
            }
        };
        let values = (&relocs[..reloc_count]).iter().map(|&address| {
            let relative = (address.as_u64().wrapping_sub(start_address.as_u64())) as usize;
            let value = if Va::SIZE == 4 {
                Va::from_u64(
                    section.get(relative..relative.wrapping_add(4))
                        .and_then(|mut x| x.read_u32().ok())
                        .unwrap_or(0) as u64
                )
            } else {
                Va::from_u64(
                    section.get(relative..relative.wrapping_add(8))
                        .and_then(|mut x| x.read_u64().ok())
                        .unwrap_or(0)
                )
            };
            RelocValues {
                address,
                value,
            }
        });
        result.extend(values);
        relocs = &relocs[reloc_count..];
    }
    result.sort_unstable_by_key(|x| x.value);
    Ok(result)
}

pub trait AnalysisState: Clone {
    fn merge(&mut self, newer: Self);
}

#[derive(Default, Clone, Debug)]
pub struct DefaultState;

impl AnalysisState for DefaultState {
    fn merge(&mut self, _newer: Self) {
    }
}

pub struct FuncAnalysis<'a, Exec: ExecutionState<'a>, State: AnalysisState> {
    binary: &'a BinaryFile<Exec::VirtualAddress>,
    cfg: Cfg<'a, Exec, State>,
    unchecked_branches: BTreeMap<Exec::VirtualAddress, (Exec, State)>,
    // Branches which are before current address.
    // The goal is to reduce amount of work when a switch jumps backwards
    // at end of the case, doing all unchecked_branches before promoting
    // more_unchecked_branches to new unchecked_branches.
    // Though this means that some other function layouts end
    // up doing more work. Not sure if there's a good solution.
    //
    // This current solution is a major (I saw over 50% less work) improvement over
    // the previous solution which always just picked branch with lowest address as next.
    // Another thing I tried is to enable this only when a function contained a switch
    // jump; result was slightly worse than now. (In other words, this is a major
    // improvement for functions with switches, minor for others)
    //
    // E.g. this is good for functions looking like
    //
    //  switch:
    //      jmp [switch_table + case * 4]
    //  case0:
    //      xx
    //      jmp switch
    //  case1:
    //      xx
    //      je case1_b:
    //      yy
    //  case1_b:
    //      jmp switch
    //  etc..
    // Avoiding reanalyzing from start of the switch after every time a jump reaches it
    //
    // But this is worse for a function looking like
    //
    //  loop1:
    //      xx
    //      dec eax
    //      jne loop1
    //      mov eax, 6
    //      (lot of code)
    //  loop2:
    //      xx
    //      dec eax
    //      jne loop2
    // as it'll run to the end once, then starts from loop1 again
    more_unchecked_branches: BTreeMap<Exec::VirtualAddress, (Exec, State)>,
    current_branch: Exec::VirtualAddress,
    operand_ctx: OperandCtx<'a>,
    merge_state_cache: MergeStateCache<'a>,
}

pub struct Control<'e: 'b, 'b, 'c, A: Analyzer<'e> + 'b> {
    inner: &'c mut ControlInner<'e, 'b, A::Exec, A::State>,
}

struct ControlInner<'e: 'b, 'b, Exec: ExecutionState<'e> + 'b, State: AnalysisState> {
    state: (Exec, State),
    analysis: &'b mut FuncAnalysis<'e, Exec, State>,
    // Set by Analyzer callback if it wants an early exit
    end: Option<End>,
    address: Exec::VirtualAddress,
    branch_start: Exec::VirtualAddress,
    instruction_length: u8,
    skip_operation: bool,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum End {
    Function,
    Branch,
}

impl<'e: 'b, 'b, 'c, A: Analyzer<'e> + 'b> Control<'e, 'b, 'c, A> {
    pub fn end_analysis(&mut self) {
        self.inner.end = Some(End::Function);
    }

    /// Ends the branch without continuing through anything it leads to
    pub fn end_branch(&mut self) {
        if self.inner.end.is_none() {
            self.inner.end = Some(End::Branch);
        }
    }

    /// Skips the current operation.
    ///
    /// When used with branches (hopefully?) behaves as if branch is never taken, use
    /// `end_branch()` to also stop analyzing the fallthrough case.
    pub fn skip_operation(&mut self) {
        self.inner.skip_operation = true;
    }

    pub fn user_state(&mut self) -> &mut A::State {
        &mut self.inner.state.1
    }

    pub fn exec_state(&mut self) -> &mut A::Exec {
        &mut self.inner.state.0
    }

    pub fn resolve(&mut self, val: Operand<'e>) -> Operand<'e> {
        self.inner.state.0.resolve(val)
    }

    pub fn resolve_apply_constraints(&mut self, val: Operand<'e>) -> Operand<'e> {
        self.inner.state.0.resolve_apply_constraints(val)
    }

    pub fn read_memory(&mut self, address: Operand<'e>, size: MemAccessSize) -> Operand<'e> {
        self.inner.state.0.read_memory(address, size)
    }

    pub fn unresolve(&mut self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.inner.state.0.unresolve(val)
    }

    pub fn unresolve_memory(&mut self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.inner.state.0.unresolve_memory(val)
    }

    /// Takes current analysis' state as starting state for a function.
    /// ("Calls the function")
    /// However, this does not update the state to what this function changes it to.
    pub fn analyze_with_current_state<A2: Analyzer<'e, State = A::State, Exec = A::Exec>>(
        &mut self,
        analyzer: &mut A2,
        entry: <A::Exec as ExecutionState<'e>>::VirtualAddress,
    ) {
        let current_instruction_end = self.current_instruction_end();
        let inner = &mut *self.inner;
        let mut state = inner.state.clone();
        let ctx = inner.analysis.operand_ctx;
        let binary = inner.analysis.binary;
        state.0.apply_call(current_instruction_end);
        let mut analysis = FuncAnalysis::custom_state_boxed(binary, ctx, entry, state);
        analysis.analyze(analyzer);
    }

    /// Calls the function and updates own state (Both ExecutionState and custom) to
    /// a merge of states at this child function's return points.
    ///
    /// NOTE: User is expected to call `ctrl.skip_operation()` if this is during hook
    /// for `Operation::Call`; this is not done automatically as inlining during other
    /// operations is also allowed, even if less useful. (Maybe this is a bit of poor API?)
    pub fn inline<A2: Analyzer<'e, State = A::State, Exec = A::Exec>>(
        &mut self,
        analyzer: &mut A2,
        entry: <A::Exec as ExecutionState<'e>>::VirtualAddress,
    ) {
        let current_instruction_end = self.current_instruction_end();
        let inner = &mut *self.inner;
        let mut state = inner.state.clone();
        let ctx = inner.analysis.operand_ctx;
        let binary = inner.analysis.binary;
        state.0.apply_call(current_instruction_end);
        let mut analysis = FuncAnalysis::custom_state_boxed(binary, ctx, entry, state);
        let mut analyzer = CollectReturnsAnalyzer::new(analyzer);
        analysis.analyze(&mut analyzer);
        if let Some(state) = analyzer.state {
            inner.state = state;
        }
    }

    pub fn ctx(&self) -> OperandCtx<'e> {
        self.inner.analysis.operand_ctx
    }

    pub fn binary(&self) -> &'e BinaryFile<<A::Exec as ExecutionState<'e>>::VirtualAddress> {
        self.inner.analysis.binary
    }

    pub fn address(&self) -> <A::Exec as ExecutionState<'e>>::VirtualAddress {
        self.inner.address
    }

    /// Can be used for determining address for a branch when jump isn't followed and such.
    pub fn current_instruction_end(&self) -> <A::Exec as ExecutionState<'e>>::VirtualAddress {
        self.inner.address + self.inner.instruction_length as u32
    }

    /// Casts to Control<B: Analyzer> with compatible states.
    ///
    /// This is fine as the control doesn't use Analyzer for anything, the type is just defined
    /// to take Analyzer as Control<Self> is cleaner than Control<Self::Exec, Self::State> when
    /// implementing trait methods.
    pub fn cast<'d, B>(&'d mut self) -> Control<'e, 'b, 'd, B>
    where B: Analyzer<'e, Exec = A::Exec, State = A::State>,
    {
        Control {
            inner: self.inner,
        }
    }

    /// Adds a branch to be analyzed using current state.
    ///
    /// Can be useful when combined with end_branch() on jump to only force one branch to
    /// be taken. As with branches in general, if the branch has been already analyzed and the
    /// new state doesn't differ from old, the branch doesn't get analyzed.
    pub fn add_branch_with_current_state(
        &mut self,
        address: <A::Exec as ExecutionState<'e>>::VirtualAddress,
    ) {
        let state = self.inner.state.clone();
        self.inner.analysis.add_unchecked_branch(address, state);
    }

    /// Clears all branches, both analyzed and currently pending.
    ///
    /// Similar to clear_unchecked_branches, but also clears all branches that had been
    /// seen. Effectively same as if a new FuncAnalysis had been started from start of
    /// current branch, with the state that the branch was reached at.
    pub fn clear_all_branches(&mut self) {
        self.clear_unchecked_branches();
        self.inner.analysis.cfg.clear();
    }

    /// Clears any pending (unchecked) branches.
    ///
    /// Effectively guarantees that any branches that were seen from `Operation::Jump`
    /// but have not been checked, will not be checked, and only code reachable from
    /// the currently executing branch will be checked.
    pub fn clear_unchecked_branches(&mut self) {
        let analysis = &mut self.inner.analysis;
        analysis.unchecked_branches.clear();
        analysis.more_unchecked_branches.clear();
    }

    /// Convenience for cases where `address + CONST * REG_SIZE` is needed
    pub fn const_word_offset(&self, left: Operand<'e>, right: u32) -> Operand<'e> {
        let size = <A::Exec as ExecutionState<'e>>::VirtualAddress::SIZE;
        let ctx = self.ctx();
        ctx.add_const(left, right as u64 * size as u64)
    }

    /// Convenience for cases where `Mem[address]` is needed
    pub fn mem_word(&self, addr: Operand<'e>) -> Operand<'e> {
        A::Exec::operand_mem_word(self.ctx(), addr)
    }

    /// Either `op.if_mem32()` or `op.if_mem64()`, depending on word size
    pub fn if_mem_word<'a>(&self, op: Operand<'e>) -> Option<Operand<'e>> {
        if <A::Exec as ExecutionState<'e>>::VirtualAddress::SIZE == 4 {
            op.if_mem32()
        } else {
            op.if_mem64()
        }
    }
}

pub trait Analyzer<'exec> : Sized {
    type State: AnalysisState;
    type Exec: ExecutionState<'exec>;
    fn branch_start(&mut self, _control: &mut Control<'exec, '_, '_, Self>) {}
    fn branch_end(&mut self, _control: &mut Control<'exec, '_, '_, Self>) {}
    fn operation(&mut self, _control: &mut Control<'exec, '_, '_, Self>, _op: &Operation<'exec>) {}
}

impl<'a, Exec: ExecutionState<'a>> FuncAnalysis<'a, Exec, DefaultState> {
    pub fn new(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
    ) -> FuncAnalysis<'a, Exec, DefaultState> {
        FuncAnalysis {
            binary,
            cfg: Cfg::new(),
            unchecked_branches: {
                let user_state = DefaultState::default();
                let init_state = Exec::initial_state(operand_ctx, binary);
                let state = (init_state, user_state);
                let mut map = BTreeMap::new();
                map.insert(start_address, state);
                map
            },
            more_unchecked_branches: BTreeMap::new(),
            current_branch: start_address,
            operand_ctx,
            merge_state_cache: MergeStateCache::new(),
        }
    }

    pub fn with_state(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
        state: Exec,
    ) -> FuncAnalysis<'a, Exec, DefaultState> {
        FuncAnalysis {
            binary,
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                let user_state = DefaultState::default();
                let state = (state, user_state);
                map.insert(start_address, state);
                map
            },
            more_unchecked_branches: BTreeMap::new(),
            current_branch: start_address,
            operand_ctx,
            merge_state_cache: MergeStateCache::new(),
        }
    }
}

impl<'a, Exec: ExecutionState<'a>, State: AnalysisState> FuncAnalysis<'a, Exec, State> {
    pub fn custom_state(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
        exec_state: Exec,
        analysis_state: State,
    ) -> FuncAnalysis<'a, Exec, State> {
        let state = (exec_state, analysis_state);
        FuncAnalysis::custom_state_boxed(binary, operand_ctx, start_address, state)
    }

    fn custom_state_boxed(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
        state: (Exec, State),
    ) -> FuncAnalysis<'a, Exec, State> {
        FuncAnalysis {
            binary,
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                map.insert(start_address, state);
                map
            },
            more_unchecked_branches: BTreeMap::new(),
            current_branch: start_address,
            operand_ctx,
            merge_state_cache: MergeStateCache::new(),
        }
    }

    fn pop_next_branch_merge_with_cfg(
        &mut self,
    ) -> Option<(Exec::VirtualAddress, (Exec, State))> {
        while let Some((addr, mut branch_state)) = self.pop_next_branch() {
            match self.cfg.get_state(addr) {
                Some(state) => {
                    state.data.0.maybe_convert_memory_immutable(16);
                    let merged = Exec::merge_states(
                        &mut state.data.0,
                        &mut branch_state.0,
                        &mut self.merge_state_cache,
                    );
                    match merged {
                        Some(s) => {
                            let mut user_state = state.data.1.clone();
                            user_state.merge(branch_state.1);
                            return Some((addr, (s, user_state)));
                        }
                        // No change, take another branch
                        None => (),
                    }
                }
                None => return Some((addr, branch_state)),
            }
        }
        None
    }

    fn pop_next_branch_and_set_disasm(
        &mut self,
        disasm: &mut Exec::Disassembler,
    ) -> Option<(Exec::VirtualAddress, (Exec, State))> {
        while let Some((addr, state)) = self.pop_next_branch_merge_with_cfg() {
            if self.disasm_set_pos(disasm, addr).is_ok() {
                return Some((addr, state));
            }
        }
        None
    }

    pub fn analyze<A: Analyzer<'a, State = State, Exec = Exec>>(&mut self, analyzer: &mut A) {
        let mut disasm = Exec::Disassembler::new(self.operand_ctx);

        while let Some((addr, state)) = self.pop_next_branch_and_set_disasm(&mut disasm) {
            self.current_branch = addr;
            let end = self.analyze_branch(analyzer, &mut disasm, addr, state);
            if end == Some(End::Function) {
                break;
            }
        }
    }

    fn disasm_set_pos(
        &self,
        disasm: &mut Exec::Disassembler,
        address: Exec::VirtualAddress,
    ) -> Result<(), ()> {
        let section = self.binary.section_by_addr(address).ok_or(())?;
        let rva = (address.as_u64() - section.virtual_address.as_u64()) as usize;
        disasm.set_pos(&section.data, rva, section.virtual_address);
        Ok(())
    }

    /// Disasm must have been set to `addr`
    fn analyze_branch<'b, A: Analyzer<'a, State = State, Exec = Exec>>(
        &mut self,
        analyzer: &mut A,
        disasm: &mut Exec::Disassembler,
        addr: Exec::VirtualAddress,
        state: (Exec, State),
    ) -> Option<End> {
        // update_analysis_for_operation is a small function, which would be cleaner
        // to inline here if it keeping it separate wasn't good for binary size
        // (It is not generic over A).
        // It mainly takes state (ControlInner) by ref, but on branches it will move
        // exec state out and leave AnalysisUpdateResult::End there, which this function
        // uses as a signal to break out of instruction loop.
        let mut inner_wrap = AnalysisUpdateResult::Continue(ControlInner {
            analysis: self,
            address: addr,
            branch_start: addr,
            instruction_length: 0,
            skip_operation: false,
            state,
            end: None,
        });
        let mut control = match inner_wrap {
            AnalysisUpdateResult::Continue(ref mut inner) => Control {
                inner,
            },
            // Unreachable
            AnalysisUpdateResult::End(end) => return end,
        };
        analyzer.branch_start(&mut control);
        if control.inner.end.is_some() {
            return control.inner.end;
        }

        // Create CfgNode in advance to avoid some small moves.
        // state doesn't get changed after creation, other fields do.
        let mut node = Some(CfgNode {
            state: CfgState {
                phantom: Default::default(),
                data: control.inner.state.clone(),
            },
            out_edges: CfgOutEdges::None,
            end_address: Exec::VirtualAddress::from_u64(0),
            distance: 0,
        });
        loop {
            let address = disasm.address();
            control.inner.address = address;
            let instruction = disasm.next();
            control.inner.instruction_length = instruction.len() as u8;
            for op in instruction.ops() {
                control.inner.skip_operation = false;
                analyzer.operation(&mut control, op);
                if control.inner.end.is_some() {
                    return control.inner.end;
                }
                if control.inner.skip_operation {
                    continue;
                }
                match *op {
                    Operation::Jump { .. } | Operation::Return(..) | Operation::Error(..) => {
                        analyzer.branch_end(&mut control);
                    }
                    _ => (),
                }
                update_analysis_for_operation(&mut inner_wrap, op, &mut node);
                control = match inner_wrap {
                    AnalysisUpdateResult::End(end) => {
                        return end;
                    }
                    AnalysisUpdateResult::Continue(ref mut inner) => Control {
                        inner,
                    },
                };
            }
        }
    }

    fn add_unchecked_branch(
        &mut self,
        addr: Exec::VirtualAddress,
        mut state: (Exec, State),
    ) {
        use std::collections::btree_map::Entry;
        let queue = match addr < self.current_branch {
            true => &mut self.more_unchecked_branches,
            false => &mut self.unchecked_branches,
        };

        match queue.entry(addr) {
            Entry::Vacant(e) => {
                e.insert(state);
            }
            Entry::Occupied(mut e) => {
                let val = e.get_mut();
                let result =
                    Exec::merge_states(&mut val.0, &mut state.0, &mut self.merge_state_cache);
                if let Some(new) = result {
                    val.0 = new;
                    val.1.merge(state.1);
                }
            }
        }
    }

    fn pop_next_branch(&mut self) ->
        Option<(Exec::VirtualAddress, (Exec, State))>
    {
        if self.unchecked_branches.is_empty() {
            std::mem::swap(&mut self.unchecked_branches, &mut self.more_unchecked_branches);
        }
        let addr = self.unchecked_branches.keys().next().cloned()?;
        let state = self.unchecked_branches.remove(&addr).unwrap();
        Some((addr, state))
    }

    pub fn finish(self) -> Cfg<'a, Exec, State> {
        self.finish_with_changes(|_, _, _| {})
    }

    /// As this will run analysis, allows user to manipulate the state during it
    pub fn finish_with_changes<F>(
        mut self,
        mut hook: F
    ) -> Cfg<'a, Exec, State>
    where F: FnMut(
        &Operation<'a>,
        &mut Exec,
        Exec::VirtualAddress,
    ) {
        let mut analyzer = RunHookAnalyzer {
            phantom: Default::default(),
            hook: &mut hook,
        };
        self.analyze(&mut analyzer);

        let mut cfg = self.cfg;
        cfg.merge_overlapping_blocks();
        let binary = self.binary;
        let ctx = self.operand_ctx;
        cfg.resolve_cond_jump_operands(|condition, address, end_address| {
            let mut analysis = FuncAnalysis::new(binary, ctx, address);
            let mut analyzer = FinishAnalyzer {
                hook: &mut hook,
                end_address,
                result: condition.clone(),
                condition,
            };
            analysis.analyze(&mut analyzer);
            analyzer.result
        });
        cfg
    }
}

struct RunHookAnalyzer<'e, F, Exec: ExecutionState<'e>, S: AnalysisState> {
    phantom: std::marker::PhantomData<(&'e (), *const Exec, *const S)>,
    hook: F,
}

impl<'e, F, Exec, S> Analyzer<'e> for RunHookAnalyzer<'e, F, Exec, S>
where F: FnMut(&Operation<'e>, &mut Exec, Exec::VirtualAddress),
      Exec: ExecutionState<'e>,
      S: AnalysisState,
{
    type State = S;
    type Exec = Exec;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        let address = ctrl.address();
        let state = ctrl.exec_state();
        (self.hook)(op, state, address);
    }
}

struct FinishAnalyzer<'e, F, Exec: ExecutionState<'e>> {
    hook: F,
    result: Operand<'e>,
    end_address: Exec::VirtualAddress,
    condition: Operand<'e>,
}

impl<'e, F, Exec: ExecutionState<'e>> Analyzer<'e> for FinishAnalyzer<'e, F, Exec>
where F: FnMut(&Operation<'e>, &mut Exec, Exec::VirtualAddress)
{
    type State = DefaultState;
    type Exec = Exec;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation<'e>) {
        let address = ctrl.address();
        let state = ctrl.exec_state();
        (self.hook)(op, state, address);
        let final_op = if address == self.end_address {
            true
        } else {
            match *op {
                Operation::Jump { .. } | Operation::Return(_) | Operation::Error(_) => true,
                _ => false,
            }
        };
        if final_op {
            self.result = ctrl.resolve(self.condition);
            ctrl.end_analysis();
        }
    }
}

/// Merges states at return operations, used for following calls.
struct CollectReturnsAnalyzer<'a, 'e: 'a, A: Analyzer<'e>> {
    inner: &'a mut A,
    state: Option<(A::Exec, A::State)>,
}

impl<'a, 'e: 'a, A: Analyzer<'e>> CollectReturnsAnalyzer<'a, 'e, A> {
    fn new(inner: &'a mut A) -> CollectReturnsAnalyzer<'a, 'e, A> {
        CollectReturnsAnalyzer {
            inner,
            state: None,
        }
    }
}

impl<'a, 'exec: 'a, A: Analyzer<'exec>> Analyzer<'exec> for CollectReturnsAnalyzer<'a, 'exec, A> {
    type State = A::State;
    type Exec = A::Exec;
    fn branch_start(&mut self, control: &mut Control<'exec, '_, '_, Self>) {
        self.inner.branch_start(&mut control.cast())
    }

    fn branch_end(&mut self, control: &mut Control<'exec, '_, '_, Self>) {
        self.inner.branch_end(&mut control.cast())
    }

    fn operation(&mut self, control: &mut Control<'exec, '_, '_, Self>, op: &Operation<'exec>) {
        self.inner.operation(&mut control.cast(), op);
        if let Operation::Return(_) = op {
            if !control.inner.skip_operation || control.inner.end.is_some() {
                let ctx = control.ctx();
                let state = control.exec_state();
                state.move_to(
                    &crate::DestOperand::Register64(4),
                    ctx.add_const(
                        ctx.register(4),
                        <A::Exec as ExecutionState<'exec>>::VirtualAddress::SIZE.into(),
                    ),
                );
                match self.state {
                    Some(ref mut state) => {
                        let new = control.exec_state();
                        let new_exec = Self::Exec::merge_states(
                            &mut state.0,
                            new,
                            &mut MergeStateCache::new(),
                        );
                        if let Some(new_exec) = new_exec {
                            state.0 = new_exec;
                            state.1.merge(control.user_state().clone());
                        }
                    }
                    None => {
                        let exec = &control.inner.state.0;
                        let user = &control.inner.state.1;
                        self.state = Some((exec.clone(), user.clone()));
                    }
                }
            }
        }
    }
}

fn try_add_branch<'e, Exec: ExecutionState<'e>, S: AnalysisState>(
    analysis: &mut FuncAnalysis<'e, Exec, S>,
    state: (Exec, S),
    to: Operand<'e>,
    address: Exec::VirtualAddress,
) -> Option<Exec::VirtualAddress> {
    match to.if_constant() {
        Some(s) => {
            let address = Exec::VirtualAddress::from_u64(s);
            let code_offset = analysis.binary.code_section().virtual_address;
            let code_len = analysis.binary.code_section().data.len() as u32;
            let invalid_dest = address < code_offset || address >= code_offset + code_len;
            if !invalid_dest {
                analysis.add_unchecked_branch(address, state);
            } else {
                trace!("Destination {:x} is out of binary bounds", address);
                // Add cfg node to keep cfg sensible
                // (Adding all branches and checking for binary bounds after states have
                // been merged could be better)
                analysis.cfg.add_node(address, CfgNode {
                    out_edges: CfgOutEdges::None,
                    state: CfgState {
                        data: state,
                        phantom: Default::default(),
                    },
                    end_address: address + 1,
                    distance: 0,
                });
            }
            Some(address)
        }
        None => {
            trace!("Couldnt resolve jump dest @ {:x}: {:?}", address, to);
            None
        }
    }
}

enum AnalysisUpdateResult<'e: 'b, 'b, E, S>
where E: ExecutionState<'e> + 'b,
      S: AnalysisState,
{
    Continue(ControlInner<'e, 'b, E, S>),
    End(Option<End>),
}

// A separate function to minimize binary size
fn update_analysis_for_operation<'e: 'b, 'b, E, S>(
    // Expected to always be AnalysisUpdateResult::Continue
    // On end gets overwritten to AnalysisUpdateResult::End
    control_ref: &mut AnalysisUpdateResult<'e, 'b, E, S>,
    op: &Operation<'e>,
    // Option so that it can be moved out of. Expected to always be Some
    cfg_node_opt: &mut Option<CfgNode<'e, CfgState<'e, E, S>>>,
)
where E: ExecutionState<'e> + 'b,
      S: AnalysisState,
{
    let control = match control_ref {
        AnalysisUpdateResult::Continue(ref mut c) => c,
        AnalysisUpdateResult::End(_) => return,
    };
    match op {
        Operation::Jump { .. } | Operation::Return(..) | Operation::Error(..) => {
            let cfg_node = match cfg_node_opt {
                Some(ref mut s) => s,
                None => return,
            };
            let end = control.end;
            let control = match mem::replace(control_ref, AnalysisUpdateResult::End(end)) {
                AnalysisUpdateResult::Continue(c) => c,
                // Unreachable
                AnalysisUpdateResult::End(_) => return,
            };
            let address = control.address;
            if let Operation::Jump { condition, to } = *op {
                let state = control.state;
                update_analysis_for_jump(
                    control.analysis,
                    state,
                    condition,
                    to,
                    address,
                    u32::from(control.instruction_length),
                    &mut cfg_node.out_edges,
                );
            } else {
                drop(control.state);
            }
            cfg_node.end_address = address;
            if let Some(cfg_node) = cfg_node_opt.take() {
                control.analysis.cfg.add_node(control.branch_start, cfg_node);
            }
        }
        o => {
            control.state.0.update(o);
        }
    }
}

fn update_analysis_for_jump<'e, Exec: ExecutionState<'e>, S: AnalysisState>(
    analysis: &mut FuncAnalysis<'e, Exec, S>,
    mut state: (Exec, S),
    condition: Operand<'e>,
    to: Operand<'e>,
    address: Exec::VirtualAddress,
    instruction_len: u32,
    cfg_out_edge: &mut CfgOutEdges<'e, Exec::VirtualAddress>,
) {
    /// Returns address of the table,
    /// operand that is being used to index the table,
    /// constant that is being added to any values read from the table,
    /// and size in bytes of one value in table.
    /// E.g. dest = ret.2 + read_'ret.3'_bytes(ret.0 + ret.1 * ret.3)
    fn is_switch_jump<'e, VirtualAddress: exec_state::VirtualAddress>(
        to: Operand<'e>,
    ) -> Option<(VirtualAddress, Operand<'e>, u64, MemAccessSize)> {
        let (base, mem) = match to.if_arithmetic_add() {
            Some((l, r)) => (r.if_constant()?, l),
            None => (0, to),
        };
        mem.if_memory()
            .and_then(|mem| mem.address.if_arithmetic_add())
            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_arithmetic_mul()))
            .and_then(|((l, r), switch_table)| {
                let switch_table = switch_table.if_constant()?;
                let (c, index) = Operand::either(l, r, {
                    |x| x.if_constant().and_then(|c| u32::try_from(c).ok())
                })?;
                if c == VirtualAddress::SIZE || base != 0 {
                    let mem_size = match c {
                        1 => MemAccessSize::Mem8,
                        2 => MemAccessSize::Mem16,
                        4 => MemAccessSize::Mem32,
                        8 => MemAccessSize::Mem64,
                        _ => return None,
                    };
                    Some((VirtualAddress::from_u64(switch_table), index, base, mem_size))
                } else {
                    None
                }
            })
    }

    fn switch_cases<'e, Va: VaTrait>(
        ctx: OperandCtx<'e>,
        binary: &'e BinaryFile<Va>,
        mem_size: MemAccessSize,
        switch_table_addr: Va,
        limits: (u64, u64),
        base_addr: u64,
    ) -> impl Iterator<Item = Operand<'e>> {
        let case_size = mem_size.bits() / 8;
        let base = ctx.constant(base_addr);

        let start = limits.0.min(u32::max_value() as u64) as u32;
        let end = limits.1.min(u32::max_value() as u64) as u32;
        (start..=end)
            .take_while(move |index| {
                if !binary.relocs.is_empty() {
                    let addr = switch_table_addr + index * case_size;
                    if binary.relocs.binary_search(&addr).is_err() {
                        return false;
                    }
                }
                true
            })
            .map(move |index| {
                ctx.add(
                    base,
                    ctx.mem_variable_rc(
                        mem_size,
                        ctx.constant(
                            switch_table_addr.as_u64() + (index as u64 * case_size as u64)
                        ),
                    ),
                )
            })
    }

    state.0.maybe_convert_memory_immutable(16);
    match state.0.resolve_apply_constraints(condition).if_constant() {
        Some(0) => {
            let address = address + instruction_len;
            *cfg_out_edge = CfgOutEdges::Single(NodeLink::new(address));
            let constraint = analysis.operand_ctx.eq_const(condition, 0);
            state.0.add_unresolved_constraint(Constraint::new(constraint));
            analysis.add_unchecked_branch(address, state);
        }
        Some(_) => {
            let to = state.0.resolve(to);
            let is_switch = is_switch_jump::<Exec::VirtualAddress>(to);
            if let Some((switch_table_addr, index, base_addr, mem_size)) = is_switch {
                let mut cases = Vec::new();
                let ctx = analysis.operand_ctx;
                let binary = analysis.binary;
                let code_section = binary.code_section();
                let code_offset = code_section.virtual_address;
                let code_len = code_section.data.len() as u32;
                let limits = state.0.value_limits(index);
                let case_iter =
                    switch_cases(ctx, binary, mem_size, switch_table_addr, limits, base_addr);
                for case in case_iter {
                    let addr = state.0.resolve(case).if_constant()
                        .map(Exec::VirtualAddress::from_u64)
                        .filter(|&x| x >= code_offset && x < code_offset + code_len);
                    if let Some(case) = addr {
                        analysis.add_unchecked_branch(case, state.clone());
                        cases.push(NodeLink::new(case));
                    } else {
                        break;
                    }
                }

                if !cases.is_empty() {
                    *cfg_out_edge = CfgOutEdges::Switch(cases, index);
                }
            } else {
                state.0.add_unresolved_constraint(Constraint::new(condition));
                let dest = try_add_branch(analysis, state, to, address);
                *cfg_out_edge = CfgOutEdges::Single(
                    dest.map(NodeLink::new).unwrap_or_else(NodeLink::unknown)
                );
            }
        }
        None => {
            let no_jump_addr = address + instruction_len;
            let mut jump_state = state.clone();
            jump_state.0.assume_jump_flag(condition, true);
            state.0.assume_jump_flag(condition, false);
            let to = jump_state.0.resolve(to);
            analysis.add_unchecked_branch(
                no_jump_addr,
                state,
            );
            let dest = try_add_branch(analysis, jump_state, to, address);
            *cfg_out_edge = CfgOutEdges::Branch(
                NodeLink::new(no_jump_addr),
                OutEdgeCondition {
                    node: dest.map(NodeLink::new).unwrap_or_else(NodeLink::unknown),
                    condition,
                },
            );
        }
    }
}

#[cfg(test)]
mod test {
    static X86_CALL_TEST_DATA: &[u8] = &[
        // 0x0 => Call to 0x10, 0x8 => call to 0x18
        0xe8, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0xe8, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,

        // Invalid, 0x16 => call to 0x10
        0x00, 0x00, 0x00, 0xe8, 0x00, 0x00, 0xe8, 0xf5,
        0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00,

        // 0x24 => Call to 0x5, 0x29 => call to 0x33
        0x00, 0x00, 0x00, 0x00, 0xe8, 0xdc, 0xff, 0xff,
        0xff, 0xe8, 0x05, 0x00, 0x00, 0x00, 0x00, 0x00,

        // Invalid
        0x00, 0x00, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xff, 0x00, 0x00, 0xe8, 0x00, 0xe8, 0x00,
    ];

    #[test]
    fn test_x86_calls() {
        use crate::VirtualAddress;
        use crate::exec_state_x86::ExecutionState;

        let mut result = Vec::new();
        super::find_functions_from_calls::<ExecutionState>(
            X86_CALL_TEST_DATA,
            VirtualAddress(0x1000),
            &mut result,
        );
        assert_eq!(result.len(), 5);
        result.sort();
        result.dedup();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], VirtualAddress(0x1005));
        assert_eq!(result[1], VirtualAddress(0x1010));
        assert_eq!(result[2], VirtualAddress(0x1018));
        assert_eq!(result[3], VirtualAddress(0x1033));
    }

    #[test]
    fn test_x86_calls_callers() {
        use crate::VirtualAddress;
        use crate::exec_state_x86::ExecutionState;

        let file = crate::raw_bin(
            VirtualAddress(0x1000),
            vec![
                crate::BinarySection {
                    name: *b".text\0\0\0",
                    virtual_address: VirtualAddress(0x1000),
                    virtual_size: 0x100,
                    data: X86_CALL_TEST_DATA.into(),
                },
            ],
        );
        let result = super::find_functions_with_callers::<ExecutionState>(&file);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].callee, VirtualAddress(0x1005));
        assert_eq!(result[0].caller, VirtualAddress(0x1024));
        assert_eq!(result[1].callee, VirtualAddress(0x1010));
        assert_eq!(result[2].callee, VirtualAddress(0x1010));
        assert_eq!(result[3].callee, VirtualAddress(0x1018));
        assert_eq!(result[3].caller, VirtualAddress(0x1008));
        assert_eq!(result[4].callee, VirtualAddress(0x1033));
        assert_eq!(result[4].caller, VirtualAddress(0x1029));
    }

    #[test]
    fn test_x86_64_calls() {
        use crate::VirtualAddress64;
        use crate::exec_state_x86_64::ExecutionState;

        let mut result = Vec::new();
        super::find_functions_from_calls::<ExecutionState>(
            X86_CALL_TEST_DATA,
            VirtualAddress64(0x1000),
            &mut result,
        );
        assert_eq!(result.len(), 5);
        result.sort();
        result.dedup();
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], VirtualAddress64(0x1005));
        assert_eq!(result[1], VirtualAddress64(0x1010));
        assert_eq!(result[2], VirtualAddress64(0x1018));
        assert_eq!(result[3], VirtualAddress64(0x1033));
    }

    #[test]
    fn test_x86_64_calls_callers() {
        use crate::VirtualAddress64;
        use crate::exec_state_x86_64::ExecutionState;

        let file = crate::raw_bin(
            VirtualAddress64(0x1000),
            vec![
                crate::BinarySection {
                    name: *b".text\0\0\0",
                    virtual_address: VirtualAddress64(0x1000),
                    virtual_size: 0x100,
                    data: X86_CALL_TEST_DATA.into(),
                },
            ],
        );
        let result = super::find_functions_with_callers::<ExecutionState>(&file);
        assert_eq!(result.len(), 5);
        assert_eq!(result[0].callee, VirtualAddress64(0x1005));
        assert_eq!(result[0].caller, VirtualAddress64(0x1024));
        assert_eq!(result[1].callee, VirtualAddress64(0x1010));
        assert_eq!(result[2].callee, VirtualAddress64(0x1010));
        assert_eq!(result[3].callee, VirtualAddress64(0x1018));
        assert_eq!(result[3].caller, VirtualAddress64(0x1008));
        assert_eq!(result[4].callee, VirtualAddress64(0x1033));
        assert_eq!(result[4].caller, VirtualAddress64(0x1029));
    }
}
