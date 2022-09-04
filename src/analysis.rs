//! Contains [`FuncAnalysis`] and related types and traits.
//!
//! Additionally miscellaneous **heuristic** functions to find function entries and
//! relocations of [`BinaryFile`] are included.

use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::mem;

use byteorder::{ByteOrder, LittleEndian};

use crate::cfg::{self, CfgNode, CfgOutEdges, NodeLink, OutEdgeCondition};
use crate::disasm::{DestOperand, Operation};
use crate::exec_state::{self, Constraint, Disassembler, ExecutionState, MergeStateCache};
use crate::exec_state::VirtualAddress as VaTrait;
use crate::light_byteorder::ReadLittleEndian;
use crate::operand::{MemAccess, MemAccessSize, Operand, OperandCtx};
use crate::{BinaryFile, BinarySection, VirtualAddress, VirtualAddress64};

pub use crate::disasm::Error;

pub type Cfg<'a, E, S> = cfg::Cfg<'a, CfgState<'a, E, S>>;

pub struct CfgState<'a, E: ExecutionState<'a>, S: AnalysisState> {
    data: (E, S),
    preceding_branch: PrecedingBranchAddr,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct RelocValues<Va: VaTrait> {
    pub address: Va,
    pub value: Va,
}

/// The returned array is sorted by value.
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

/// A trait for additional branch-specific analysis state that will be merged by the
/// analysis when two different branches join.
///
/// See [state merging](../analysis/struct.FuncAnalysis.html#state-merging-and-loops)
/// for more details on how analysis merges states.
///
/// At the moment, only differences in `ExecutionState` will cause analysis to merge
/// and recheck a branch; there is no way yet for `AnalysisState` to declare two states
/// different to require a recheck when there hasn't been changes in `ExecutionState`.
///
/// You can access and modify the `AnalysisState` for current branch in [`Analyzer`] callbacks
/// by calling [`Control::user_state`].
pub trait AnalysisState: Clone {
    /// Called when analysis merges states of two different branches.
    ///
    /// The implementation is expected to update `self` with values that are a merge of
    /// `self` and `newer`, but the implementation may do anything it wants, as scarf does not
    /// touch `AnalysisState` at all.
    fn merge(&mut self, newer: Self);
}

/// A no-op zero-sized type implementing [`AnalysisState`] that can be used if there
/// is no need for user-side mergeable state.
#[derive(Default, Clone, Debug)]
pub struct DefaultState;

impl AnalysisState for DefaultState {
    fn merge(&mut self, _newer: Self) {
    }
}

/// The analysis runner of scarf.
///
/// `FuncAnalysis` executes through all paths ('branches') of a function, calling
/// user-defined callbacks in [`Analyzer`] for every state-modifying operation, while
/// keeping track of what branches have been already executed.
///
/// # State merging and loops
///
/// When two separate branches of execution converge, `FuncAnalysis` will merge the
/// [`ExecutionState`] of branches: Any [`Operand`] in registers or in memory that differs
/// will be replaced with `OperandType::Undefined` if it wasn't `Undefined` already. If this
/// caused anything in `ExecutionState` to change, the following branch will be analyzed
/// again, even if analysis had already walked through its instruction once.
///
/// Scarf does not have any special understanding of loops, a jump backwards is just another
/// merge point. This effectively often makes scarf to provide user code an `ExecutionState`
/// that reports registers to contain constants and jumps to be always/never taken for first
/// iteration of a loop, and later on run the branch 'correctly' with `Undefined` when the
/// jump backwards has been seen.
///
/// ## Example
///
/// The following code is a `strlen` implementation in x86-64 assembly. Register `rcx` has
/// the null-terminated input string, and `rax` will contain the string length on return.
///
/// ```asm
/// entry:
///     mov rax, 0
/// loop:
///     movzx rdx, byte [rcx]
///     add rax, 1
///     add rcx, 1
///     cmp rdx, 0
///     jne loop
/// exit:
///     sub rax, 1 ; Account for null byte having been counted above
///     ret
/// ```
///
/// Scarf considers there to 3 branches:
///
/// 1) Branch from `entry` to `exit`
/// 2) Branch from `loop` to `exit`
/// 3) Branch from `exit` that returns
///
/// Note that both branch 1 and 2 include the loop instructions. In this case, the backwards
/// jump can only be known to be possibly taken after the first branch has already been
/// simulated, so there would have been no way for scarf to know that `movzx rdx, byte [rcx]`
/// will be a start of another branch. For consistency, even if a jump to current instruction
/// has been seen, scarf will walk each branch until it sees a jump or return instruction.
///
/// On analysis start, scarf will execute branch 1, seeing that on jump the execution splits
/// to two new branches. [Resolving](ExecutionState::resolve) `rax` during this will return
/// constant 0, later on 1, and resolving the jump condition at the end of the branch
/// gives `(Mem8[rcx] == 0) == 0` - "Jump if first byte of argument `rcx` is nonzero".
///
/// After the first branch, both branches 2 and 3 are queued; there is no guarantee
/// which will be analyzed first. If branch 3 is first, scarf will report the function to
/// return (resolving `rax`) constant 0. Once branch 2 is analyzed, it will have similar
/// 'known' results what branch 1 shows (`rax` changes from 1 to 2, jump condition is
/// `(Mem8[rcx + 1] == 0) == 0`), as this is the first time branch has started from there.
///
/// As branch 2 analysis reaches the jump, scarf queues branches 2 and 3, and notices that the
/// values in registers `rax`, `rcx`, and `rdx` differ from what the older branches 2 and 3 had.
/// Each of those registers will be assigned an `Undefined` value for the new queued branches.
/// If branch 3 wasn't analyzed yet, the state with `Undefined` overwrites the earlier queued
/// 'will-return-zero' state.
///
/// Finally, scarf walks through branches 2 and 3 with the undefined state (in either order).
/// At end of this second run of branch 2, `rax` will have value `Undefined_a + 1`, rcx has
/// `Undefined_b + 1`, and rdx has `Mem8[Undefined_b]`. These do not add anything over the
/// states where these registers were `Undefined`, so no more branches will be queued.
///
/// Since branch 3 contains `sub rax, 1` instruction, the final return value reported by
/// scarf is the not-very-helpful `Undefined - 1`.
///
/// ## Other merge details
///
/// In general, [`Operand`] merging is defined as "If old value is undefined or differs from new,
/// merge to unique undefined value". However, there are few details that are worth noting:
///
/// - Undefined values allocated during `FuncAnalysis::analyze` call will get reused after
///     the call returns. This is mainly worth noting if user code returns results as `Operand`,
///     and compares them to values from other analysis runs.
/// - If old or new value is undefined, the merged value may recycle either of those values, or
///     allocate a new unique undefined.
/// - When values in memory are merged, a single unit of `Operand` to be merged is not a byte,
///     but a word (4 or 8 bytes, depending on the `ExecutionState`). This means that if code
///     has byte-sized globals or structure fields, they may be assigned `Undefined` because
///     values neighbour address differed. So far this has rarely ended up causing issues, but
///     it may sometimes need special handling from user code.
/// - If the address operand of memory contains `Undefined` at all, the merge operation may
///     drop the value entirely, reverting it back to 'original'.
///     - That is, while a write of `5` to `Mem32[Undefined + 4]` is guaranteed to resolve to `5`
///     in same branch as the write, later branches may resolve just to `Mem32[Undefined + 4]`
///     again.
pub struct FuncAnalysis<'a, Exec: ExecutionState<'a>, State: AnalysisState> {
    binary: &'a BinaryFile<Exec::VirtualAddress>,
    cfg: Cfg<'a, Exec, State>,
    unchecked_branches: BTreeMap<Exec::VirtualAddress, (Exec, State, PrecedingBranchAddr)>,
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
    more_unchecked_branches: BTreeMap<Exec::VirtualAddress, (Exec, State, PrecedingBranchAddr)>,
    current_branch: Exec::VirtualAddress,
    operand_ctx: OperandCtx<'a>,
    merge_state_cache: MergeStateCache<'a>,
}

/// The analysis will assume that if only a single branch has lead to a branch,
/// then merging of execution states can be skipped and it will just be assumed
/// that `state = merge(old, new)` can be replaced with just `state = new`, e.g.
/// only thing that should have changed when the single preceding branch is ran a
/// second time is that more values are now undefined.
///
/// If value of this newtype is u32::MAX, then there are multiple preceding branches
/// and states to branch using this PrecedingBranchAddr have to be merged.
/// Otherwise this is an address of the preceding branch relative to binary.base().
#[derive(Copy, Clone)]
struct PrecedingBranchAddr(u32);

impl PrecedingBranchAddr {
    fn multiple() -> PrecedingBranchAddr {
        PrecedingBranchAddr(u32::MAX)
    }

    fn is_multiple(self) -> bool {
        self.0 == u32::MAX
    }

    fn are_same(self, other: PrecedingBranchAddr) -> bool {
        self.0 == other.0 && self.0 != u32::MAX
    }
}

/// Provides access to analysis state when in a [`Analyzer`] callback.
///
/// The most common use cases are:
/// - Access to the current [`ExecutionState`] through [`exec_state`](Control::exec_state),
///     as well as convenience shortcuts:
///     - [`resolve`](Control::resolve)
///     - [`resolve_mem`](Control::resolve_mem)
///     - [`resolve_apply_constraints`](Control::resolve_apply_constraints)
///     - [`read_memory`](Control::read_memory)
///     - [`move_resolved`](Control::move_resolved)
///     - [`move_unresolved`](Control::move_unresolved)
/// - Ability to control what branches are analyzed with:
///     - [`end_branch`](Control::end_branch)
///     - [`end_analysis`](Control::end_analysis)
///     - [`add_branch_with_current_state`](Control::add_branch_with_current_state)
///     - [`clear_unchecked_branches`](Control::clear_unchecked_branches)
///     - [`clear_all_branches`](Control::clear_all_branches)
/// - Easy way to create analysis for child function with:
///     - [`analyze_with_current_state`](Control::analyze_with_current_state)
///     - [`inline`](Control::inline)
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
    continue_at_address: Exec::VirtualAddress,
    instruction_length: u8,
    skip_operation: bool,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum End {
    Function,
    Branch,
    ContinueBranch,
}

impl<'e: 'b, 'b, 'c, A: Analyzer<'e> + 'b> Control<'e, 'b, 'c, A> {
    /// Retreives the `OperandCtx` that was used to construct `FuncAnalysis`.
    pub fn ctx(&self) -> OperandCtx<'e> {
        self.inner.analysis.operand_ctx
    }

    /// Retreives the `BinaryFile` that was used to construct `FuncAnalysis`.
    pub fn binary(&self) -> &'e BinaryFile<<A::Exec as ExecutionState<'e>>::VirtualAddress> {
        self.inner.analysis.binary
    }

    /// For `operation()` callback, eeturns address of the current instruction that is
    /// used as a source for the `Operation`.
    ///
    /// For `branch_start()` callback, returns the start address of the branch.
    ///
    /// For `branch_end()` callback, returns the address of final instruction of branch.
    pub fn address(&self) -> <A::Exec as ExecutionState<'e>>::VirtualAddress {
        self.inner.address
    }

    /// Returns start address of the current branch.
    pub fn branch_start(&self) -> <A::Exec as ExecutionState<'e>>::VirtualAddress {
        self.inner.branch_start
    }

    /// Returns address of the next instruction that will be executed.
    ///
    /// Does not return a valid value in `branch_start()` callback.
    ///
    /// For `branch_end()` callback, returns the address of instruction after final instruction
    /// of the branch.
    pub fn current_instruction_end(&self) -> <A::Exec as ExecutionState<'e>>::VirtualAddress {
        self.inner.address + self.inner.instruction_length as u32
    }

    /// Causes the `FuncAnalysis` to end analysis immediately after this hook is returned from.
    ///
    /// Currently no-op in `branch_end()` callback.
    pub fn end_analysis(&mut self) {
        self.inner.end = Some(End::Function);
    }

    /// Causes the `FuncAnalysis` to end the currently simulated branch without executing
    /// any remaining instructions or adding branches from `Operation::Jump`.
    ///
    /// Using `end_branch` together with `add_branch_with_current_state` for each
    /// jump destination when seeing `Operation::Jump` can be used to customize state
    /// which will be used for branches after the jump. It is a bit awkward way to do it
    /// and could use a better designed API though :)
    ///
    /// Currently no-op in `branch_end()` callback.
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

    /// Returns mutable refence to the current branch's `ExecutionState`.
    ///
    /// `Control` includes convenience functions for the most commonly used `ExecutionState`
    /// functions, which can be used instead of calling this.
    pub fn exec_state(&mut self) -> &mut A::Exec {
        &mut self.inner.state.0
    }

    /// Returns mutable reference to user-defined analyzer state [`Analyzer::State`].
    pub fn user_state(&mut self) -> &mut A::State {
        &mut self.inner.state.1
    }

    /// Convenience function for [`exec_state().resolve()`](ExecutionState::resolve)
    pub fn resolve(&mut self, val: Operand<'e>) -> Operand<'e> {
        self.inner.state.0.resolve(val)
    }

    /// Convenience function for [`exec_state().resolve_mem()`](ExecutionState::resolve_mem)
    pub fn resolve_mem(&mut self, val: &MemAccess<'e>) -> MemAccess<'e> {
        self.inner.state.0.resolve_mem(val)
    }

    /// Convenience function for
    /// [`exec_state().resolve_apply_constraints()`](ExecutionState::resolve_apply_constraints)
    pub fn resolve_apply_constraints(&mut self, val: Operand<'e>) -> Operand<'e> {
        self.inner.state.0.resolve_apply_constraints(val)
    }

    /// Convenience function for [`exec_state().read_memory()`](ExecutionState::read_memory)
    pub fn read_memory(&mut self, mem: &MemAccess<'e>) -> Operand<'e> {
        self.inner.state.0.read_memory(mem)
    }

    /// Don't use this, will be deprecated at some point. See [`ExecutionState::unresolve`].
    pub fn unresolve(&mut self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.inner.state.0.unresolve(val)
    }

    /// Don't use this, will be deprecated at some point.
    /// See [`ExecutionState::unresolve_memory`].
    pub fn unresolve_memory(&mut self, val: Operand<'e>) -> Option<Operand<'e>> {
        self.inner.state.0.unresolve_memory(val)
    }

    /// Convenience function for [`exec_state().move_to()`](ExecutionState::move_to),
    /// with a better name that reminds that this function uses
    /// [unresolved operands](../exec_state/trait.ExecutionState.html#resolved-and-unresolved-operands).
    pub fn move_unresolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        self.inner.state.0.move_to(dest, value);
    }

    /// Convenience function for [`exec_state().move_resolved()`](ExecutionState::move_resolved),
    pub fn move_resolved(&mut self, dest: &DestOperand<'e>, value: Operand<'e>) {
        self.inner.state.0.move_resolved(dest, value);
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
    ///
    /// If the child analyzer uses `end_analysis` or `end_branch` to prevent the inlined analyzer
    /// reaching `Operation::Return`, this will end up merging only branches that do return.
    ///
    /// If any of the branches reaches a return, the state will be updated even if
    /// `end_analysis` is used later.
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
    pub fn add_branch_with_current_state(
        &mut self,
        address: <A::Exec as ExecutionState<'e>>::VirtualAddress,
    ) {
        let state = self.inner.state.clone();
        self.inner.analysis.add_unchecked_branch(address, state);
    }

    /// Adds a branch to be analyzed using given `ExecutionState` and user-defined state.
    pub fn add_branch_with_state(
        &mut self,
        address: <A::Exec as ExecutionState<'e>>::VirtualAddress,
        exec_state: A::Exec,
        user_state: A::State,
    ) {
        self.inner.analysis.add_unchecked_branch(address, (exec_state, user_state));
    }

    /// Causes the current branch jump to an address.
    ///
    /// Internally ends the current branch, and queues a new one at the given
    /// address using current state.
    ///
    /// Slightly differs from what `ctrl.end_branch()` does in that if used on middle of
    /// non-jumping instruction, all `Operation`s in the instruction will be executed
    /// before branch ends and execution will be continued at `address`.
    ///
    /// Can be used on jump to only force one branch to be taken.
    /// As with branches in general, if the branch has been already analyzed and the
    /// new state doesn't differ from old, the branch doesn't get analyzed.
    pub fn continue_at_address(
        &mut self,
        address: <A::Exec as ExecutionState<'e>>::VirtualAddress,
    ) {
        if self.inner.end.is_none() {
            self.inner.continue_at_address = address;
            self.inner.end = Some(End::ContinueBranch);
        }
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

    /// Either `ctx.mem32()` or `ctx.mem64()`, depending on ExecutionState word size.
    pub fn mem_word(&self, addr: Operand<'e>, offset: u64) -> Operand<'e> {
        A::Exec::operand_mem_word(self.ctx(), addr, offset)
    }

    /// Either `op.if_mem32()` or `op.if_mem64()`, depending on ExecutionState word size.
    pub fn if_mem_word<'a>(&self, op: Operand<'e>) -> Option<&'e MemAccess<'e>> {
        if <A::Exec as ExecutionState<'e>>::VirtualAddress::SIZE == 4 {
            op.if_mem32()
        } else {
            op.if_mem64()
        }
    }

    pub fn if_mem_word_offset<'a>(&self, op: Operand<'e>, offset: u64) -> Option<Operand<'e>> {
        self.if_mem_word(op)?.if_offset(offset)
    }
}

/// The trait that will be implemented by user-side code to hook into [`FuncAnalysis`] to
/// retrieve whichever results are wanted.
///
/// Most commonly the `Analyzer` will implement at least the `operation()` hook,
/// which will be called for each sub-step of each instruction `FuncAnalysis` is walking through.
///
/// During the hooks the `Analyzer` has access to [`Control`], which is used to read and modify
/// any scarf-side analysis state available to user code, as well as a `&mut self` reference
/// to the `Analyzer` structure, which can contain any state the user code needs to be shared
/// across entire analysis. If the user code needs branch-specific state with custom
/// [state merging](../analysis/struct.FuncAnalysis.html#state-merging-and-loops) behaviour,
/// [`Analyzer::State`] can be used to easily implement the state.
pub trait Analyzer<'e> : Sized {
    /// The `ExecutionState`, CPU architecture that will be used by this analyzer.
    type Exec: ExecutionState<'e>;

    /// Additional user-defined state that will be
    /// [merged by analysis](../analysis/struct.FuncAnalysis.html#state-merging-and-loops) when
    /// two branches of execution join.
    ///
    /// If not needed, [`DefaultState`] can be used to get a state which carries no information.
    ///
    /// When not using `DefaultState`, call [`FuncAnalysis::custom_state`] to create the
    /// `FuncAnalysis` struct. The state can be read and modified during all Analyzer hooks with
    /// [`Control::user_state`].
    type State: AnalysisState;

    /// Called for each `Operation` of disassembled instruction.
    ///
    /// This is the main function for user code's analysis inspection and manipulation.
    fn operation(&mut self, _control: &mut Control<'e, '_, '_, Self>, _op: &Operation<'e>) {}

    /// Called each time the analysis starts executing instructions from a new branch.
    fn branch_start(&mut self, _control: &mut Control<'e, '_, '_, Self>) {}

    /// Called each time the analysis has finished a branch, either due to `Operation::Jump`
    /// or `Operation::Return`.
    ///
    /// Not called if user has requested the branch to be stopped through [`Control::end_branch`].
    fn branch_end(&mut self, _control: &mut Control<'e, '_, '_, Self>) {}
}

impl<'a, Exec: ExecutionState<'a>> FuncAnalysis<'a, Exec, DefaultState> {
    /// Creates a new `FuncAnalysis` with default `ExecutionState` and no custom
    /// user-side state.
    pub fn new(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
    ) -> FuncAnalysis<'a, Exec, DefaultState> {
        let state = Exec::initial_state(ctx, binary);
        FuncAnalysis::custom_state_boxed(
            binary,
            ctx,
            start_address,
            (state, DefaultState::default()),
        )
    }

    /// Creates a new `FuncAnalysis` with user-given `ExecutionState` and no custom
    /// user-side state.
    pub fn with_state(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
        state: Exec,
    ) -> FuncAnalysis<'a, Exec, DefaultState> {
        FuncAnalysis::custom_state_boxed(
            binary,
            ctx,
            start_address,
            (state, DefaultState::default()),
        )
    }
}

impl<'a, Exec: ExecutionState<'a>, State: AnalysisState> FuncAnalysis<'a, Exec, State> {
    /// Creates a new `FuncAnalysis` with user-given `ExecutionState` and custom
    /// user-side state.
    pub fn custom_state(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
        exec_state: Exec,
        analysis_state: State,
    ) -> FuncAnalysis<'a, Exec, State> {
        let state = (exec_state, analysis_state);
        FuncAnalysis::custom_state_boxed(binary, ctx, start_address, state)
    }

    fn custom_state_boxed(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        ctx: OperandCtx<'a>,
        start_address: Exec::VirtualAddress,
        state: (Exec, State),
    ) -> FuncAnalysis<'a, Exec, State> {
        FuncAnalysis {
            binary,
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                map.insert(start_address, (state.0, state.1, PrecedingBranchAddr::multiple()));
                map
            },
            more_unchecked_branches: BTreeMap::new(),
            current_branch: start_address,
            operand_ctx: ctx,
            merge_state_cache: MergeStateCache::new(),
        }
    }

    fn pop_next_branch_merge_with_cfg(
        &mut self,
    ) -> Option<(Exec::VirtualAddress, (Exec, State), PrecedingBranchAddr)> {
        while let Some((addr, mut branch_state, preceding)) = self.pop_next_branch() {
            match self.cfg.get_state(addr) {
                Some(state) => {
                    if state.preceding_branch.are_same(preceding) {
                        let mut user_state = state.data.1.clone();
                        user_state.merge(branch_state.1);
                        return Some((addr, (branch_state.0, user_state), preceding));
                    } else {
                        state.preceding_branch = PrecedingBranchAddr::multiple();
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
                                return Some((
                                    addr,
                                    (s, user_state),
                                    PrecedingBranchAddr::multiple(),
                                ));
                            }
                            // No change, take another branch
                            None => (),
                        }
                    }
                }
                None => {
                    return Some((addr, branch_state, preceding));
                }
            }
        }
        None
    }

    fn pop_next_branch_and_set_disasm(
        &mut self,
        disasm: &mut Exec::Disassembler,
    ) -> Option<(Exec::VirtualAddress, (Exec, State), PrecedingBranchAddr)> {
        while let Some((addr, state, preceding)) = self.pop_next_branch_merge_with_cfg() {
            if self.disasm_set_pos(disasm, addr).is_ok() {
                return Some((addr, state, preceding));
            }
        }
        None
    }

    /// Runs the analysis, calling the user-defined callbacks defined on [`Analyzer`] trait
    /// as the code is being stepped thorugh.
    pub fn analyze<A: Analyzer<'a, State = State, Exec = Exec>>(&mut self, analyzer: &mut A) {
        let old_undef_pos = self.operand_ctx.get_undef_pos();
        let mut disasm = Exec::Disassembler::new(
            self.operand_ctx,
            self.binary,
            self.current_branch,
        );

        while let Some((addr, state, preceding)) =
            self.pop_next_branch_and_set_disasm(&mut disasm)
        {
            self.current_branch = addr;
            let end = self.analyze_branch(analyzer, &mut disasm, addr, state, preceding);
            if end == Some(End::Function) {
                break;
            }
        }
        self.operand_ctx.set_undef_pos(old_undef_pos);
    }

    fn disasm_set_pos(
        &self,
        disasm: &mut Exec::Disassembler,
        address: Exec::VirtualAddress,
    ) -> Result<(), ()> {
        disasm.set_pos(address)
    }

    /// Disasm must have been set to `addr`
    fn analyze_branch<'b, A: Analyzer<'a, State = State, Exec = Exec>>(
        &mut self,
        analyzer: &mut A,
        disasm: &mut Exec::Disassembler,
        addr: Exec::VirtualAddress,
        state: (Exec, State),
        preceding_branch: PrecedingBranchAddr,
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
            continue_at_address: Exec::VirtualAddress::from_u64(0),
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
                preceding_branch,
            },
            out_edges: CfgOutEdges::None,
            end_address: Exec::VirtualAddress::from_u64(0),
            distance: 0,
        });
        // add_resolved_constraint_from_unresolved is not expected to be useful
        // when only one branch leads to this branch, but multiple branches merging
        // may end up having same unresolved constraint with different resolved
        // constraints.
        if preceding_branch.is_multiple() {
            control.inner.state.0.add_resolved_constraint_from_unresolved();
        }
        loop {
            let address = disasm.address();
            control.inner.address = address;
            let instruction = disasm.next();
            control.inner.instruction_length = instruction.len() as u8;
            for op in instruction.ops() {
                control.inner.skip_operation = false;
                analyzer.operation(&mut control, op);
                if let Some(end) = control.inner.end {
                    let end_now = end != End::ContinueBranch || is_branch_end_op(op);
                    if end_now {
                        update_analysis_for_user_requested_end(&mut inner_wrap, &mut node);
                        return Some(end);
                    }
                }
                if control.inner.skip_operation {
                    continue;
                }
                if is_branch_end_op(op) {
                    analyzer.branch_end(&mut control);
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
            if let Some(end) = control.inner.end {
                update_analysis_for_user_requested_end(&mut inner_wrap, &mut node);
                return Some(end);
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

        let preceding_addr = self.current_branch.as_u64()
            .checked_sub(self.binary.base().as_u64())
            .and_then(|x| Some(PrecedingBranchAddr(u32::try_from(x).ok()?)))
            .unwrap_or(PrecedingBranchAddr::multiple());
        match queue.entry(addr) {
            Entry::Vacant(e) => {
                e.insert((state.0, state.1, preceding_addr));
            }
            Entry::Occupied(mut e) => {
                let val = e.get_mut();
                if preceding_addr.are_same(val.2) {
                    val.0 = state.0;
                    val.1.merge(state.1);
                } else {
                    val.2 = PrecedingBranchAddr::multiple();
                    let result =
                        Exec::merge_states(&mut val.0, &mut state.0, &mut self.merge_state_cache);
                    if let Some(new) = result {
                        val.0 = new;
                        val.1.merge(state.1);
                    }
                }
            }
        }
    }

    fn pop_next_branch(&mut self) ->
        Option<(Exec::VirtualAddress, (Exec, State), PrecedingBranchAddr)>
    {
        if self.unchecked_branches.is_empty() {
            std::mem::swap(&mut self.unchecked_branches, &mut self.more_unchecked_branches);
        }
        let addr = self.unchecked_branches.keys().next().cloned()?;
        let state = self.unchecked_branches.remove(&addr).unwrap();
        Some((addr, (state.0, state.1), state.2))
    }

    /// Runs analyzer to end without any user interaction, producing `Cfg`
    ///
    /// Calling this is not necessary if you do not need a `Cfg`.
    ///
    /// Currently if [`Control::end_analysis`] has been used to stop the analysis,
    /// any remaining branches (but not the one that was being analyzed on `end_analysis` call!)
    /// will be walked through to gain a 'better' idea of the control flow graph. This is somewhat
    /// inconsistent behaviour and should be changed to not analyze any pending branches once
    /// all users relying on this have migrated.
    pub fn finish(self) -> Cfg<'a, Exec, State> {
        self.finish_with_changes(|_, _, _| {})
    }

    /// Runs analyzer to end with any user interaction, producing `Cfg`.
    ///
    /// Maybe to be removed later? User code should ideally not use `end_analysis` and
    /// expect that any pending branches get processed by finish().
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

#[inline]
fn is_branch_end_op<'e>(op: &Operation<'e>) -> bool {
    matches!(op, Operation::Jump { .. } | Operation::Return(..) | Operation::Error(..))
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
                let new_esp = ctx.add_const(
                    state.resolve_register(4),
                    <A::Exec as ExecutionState<'exec>>::VirtualAddress::SIZE.into(),
                );
                state.set_register(4, new_esp);
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
                        preceding_branch: PrecedingBranchAddr::multiple(),
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

fn update_analysis_for_user_requested_end<'e: 'b, 'b, E, S>(
    // Expected to always be AnalysisUpdateResult::Continue
    control_ref: &mut AnalysisUpdateResult<'e, 'b, E, S>,
    // Option so that it can be moved out of. Expected to always be Some
    cfg_node_opt: &mut Option<CfgNode<'e, CfgState<'e, E, S>>>,
)
where E: ExecutionState<'e> + 'b,
      S: AnalysisState,
{
    let control = match mem::replace(control_ref, AnalysisUpdateResult::End(None)) {
        AnalysisUpdateResult::Continue(c) => c,
        // Unreachable
        AnalysisUpdateResult::End(_) => return,
    };
    if let Some(mut cfg_node) = cfg_node_opt.take() {
        let address = control.address;
        cfg_node.end_address = address;
        control.analysis.cfg.add_node(control.branch_start, cfg_node);
    }
    if control.end == Some(End::ContinueBranch) {
        let analysis = control.analysis;
        let state = control.state;
        analysis.add_unchecked_branch(control.continue_at_address, state);
    }
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
        ctx: OperandCtx<'e>,
    ) -> Option<(VirtualAddress, Operand<'e>, u64, MemAccessSize)> {
        let (base, mem) = match to.if_arithmetic_add() {
            Some((l, r)) => (r.if_constant()?, l),
            None => (0, to),
        };
        mem.if_memory()
            .and_then(|mem| {
                let (mut index, mut switch_table) = mem.address();
                let mut and_mask = None;
                if let Some((l, r)) = index.if_arithmetic_and() {
                    // Index such as (((rcx * 4) - 34) & 3fffffffc) for Mem32
                    // should be understood as index = rcx, table offset -= 0xd
                    let mask = r.if_constant()?;
                    if mask as u32 & mem.size.bytes() - 1 != 0 {
                        return None;
                    }
                    let (inner, offset) = l.if_arithmetic_sub()
                        .and_then(|(l, r)| {
                            let c = r.if_constant()?;
                            Some((l, 0u64.wrapping_sub(c)))
                        })
                        .or_else(|| {
                            let (l, r) = l.if_arithmetic_add()?;
                            let c = r.if_constant()?;
                            Some((l, c))
                        })?;
                    index = inner;
                    and_mask = Some(mask);
                    switch_table = switch_table.wrapping_add(offset);
                }
                let (index, r) = index.if_arithmetic_mul()?;
                let c = u32::try_from(r.if_constant()?).ok()?;
                if c == VirtualAddress::SIZE || base != 0 {
                    let (mem_size, shift) = match c {
                        1 => (MemAccessSize::Mem8, 0),
                        2 => (MemAccessSize::Mem16, 1),
                        4 => (MemAccessSize::Mem32, 2),
                        8 => (MemAccessSize::Mem64, 3),
                        _ => return None,
                    };
                    let index = match and_mask {
                        Some(s) => ctx.and_const(index, s >> shift),
                        None => index,
                    };
                    Some((VirtualAddress::from_u64(switch_table), index, base, mem_size))
                } else {
                    None
                }
            })
    }

    fn switch_cases<'e, Va: VaTrait>(
        binary: &'e BinaryFile<Va>,
        mem_size: MemAccessSize,
        switch_table_addr: Va,
        limits: (u64, u64),
        base_addr: u64,
    ) -> Option<impl Iterator<Item = Va> + 'e> {
        let size_shift = match mem_size {
            MemAccessSize::Mem8 => 0u8,
            MemAccessSize::Mem16 => 1,
            MemAccessSize::Mem32 => 2,
            MemAccessSize::Mem64 => 3,
        };
        let case_size = 1u32 << size_shift;
        let size_mask = mem_size.mask();

        let code_section = binary.code_section();
        let code_offset = code_section.virtual_address;
        let code_len = code_section.data.len() as u32;
        let code_end = code_offset + code_len;

        let start = limits.0.min(u32::MAX as u64) as u32;
        let end = limits.1.min(u32::MAX as u64) as u32;
        let mut pos = start << size_shift;
        let end = end << size_shift;

        // Handle reads of switch table values as `(u64(data - shift) >> shift) & mask`,
        // shift is 0 if switch table is right at section start to make `data - shift` not
        // overflow past start, otherwise it is set so that `u64(data - shift)` won't have
        // to read past section end in case switch table goes that far.
        let switch_table_shift;
        let mut switch_table_data;
        let switch_table_section = binary.section_by_addr(switch_table_addr)?;
        let section_relative = (switch_table_addr.as_u64() as usize)
            .wrapping_sub(switch_table_section.virtual_address.as_u64() as usize)
            .wrapping_add(pos as usize);
        if section_relative < 8 {
            switch_table_shift = 0u32;
            switch_table_data = switch_table_section.data.get(section_relative..)?;
        } else {
            let shift_bytes = 8u32.wrapping_sub(case_size);
            switch_table_shift = shift_bytes << 3;
            switch_table_data = switch_table_section.data
                .get((section_relative - shift_bytes as usize)..)?;
        }

        Some(std::iter::from_fn(move || {
            if pos > end {
                return None;
            }

            if !binary.relocs.is_empty() {
                let addr = switch_table_addr + pos;
                if binary.relocs.binary_search(&addr).is_err() {
                    return None;
                }
            }
            pos = pos.wrapping_add(case_size);
            if switch_table_data.len() < 8 {
                return None;
            }
            let switch_table_value =
                (LittleEndian::read_u64(&switch_table_data) >> switch_table_shift) & size_mask;
            switch_table_data = switch_table_data.get((case_size as usize)..)?;
            let jump_addr = Va::from_u64(base_addr.wrapping_add(switch_table_value));
            if jump_addr >= code_offset && jump_addr < code_end {
                Some(jump_addr)
            } else {
                None
            }
        }))
    }

    state.0.maybe_convert_memory_immutable(16);
    let condition_resolved = if condition.needs_resolve() {
        state.0.resolve_apply_constraints(condition)
    } else {
        condition
    };

    let ctx = analysis.operand_ctx;
    if condition_resolved == ctx.const_0() {
        // Never jump
        let address = address + instruction_len;
        *cfg_out_edge = CfgOutEdges::Single(NodeLink::new(address));
        if condition != ctx.const_0() {
            let constraint = analysis.operand_ctx.eq_const(condition, 0);
            state.0.add_unresolved_constraint(Constraint::new(constraint));
        }
        analysis.add_unchecked_branch(address, state);
    } else if condition_resolved == ctx.const_1() {
        // Always jump
        let to = state.0.resolve(to);
        let is_switch = is_switch_jump::<Exec::VirtualAddress>(to, ctx);
        if let Some((switch_table_addr, index, base_addr, mem_size)) = is_switch {
            let binary = analysis.binary;
            let limits = state.0.value_limits(index);
            let case_guess = (limits.1.wrapping_sub(limits.0) as usize)
                .wrapping_add(1)
                .min(128);
            let mut cases = Vec::with_capacity(case_guess);

            let mut case_iter =
                switch_cases(binary, mem_size, switch_table_addr, limits, base_addr);
            if let Some(ref mut case_iter) = case_iter {
                for addr in case_iter {
                    analysis.add_unchecked_branch(addr, state.clone());
                    cases.push(NodeLink::new(addr));
                }

                if !cases.is_empty() {
                    *cfg_out_edge = CfgOutEdges::Switch(cases, index);
                }
            }
        } else {
            if condition != ctx.const_1() {
                state.0.add_unresolved_constraint(Constraint::new(condition));
            }
            let dest = try_add_branch(analysis, state, to, address);
            *cfg_out_edge = CfgOutEdges::Single(
                dest.map(NodeLink::new).unwrap_or_else(NodeLink::unknown)
            );
        }
    } else {
        let no_jump_addr = address + instruction_len;
        let to = state.0.resolve(to);
        let (jump, no_jump) =
            exec_state::assume_jump_flag(state.0, condition, condition_resolved);
        let jump_state = (jump, state.1.clone());
        let no_jump_state = (no_jump, state.1);
        analysis.add_unchecked_branch(
            no_jump_addr,
            no_jump_state,
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
