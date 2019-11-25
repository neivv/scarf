use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt};
use quick_error::quick_error;

use crate::cfg::{self, CfgNode, CfgOutEdges, NodeLink, OutEdgeCondition};
use crate::disasm::{self, Instruction, Operation};
use crate::exec_state::{self, Disassembler, ExecutionState, InternMap};
use crate::exec_state::VirtualAddress as VaTrait;
use crate::operand::{MemAccessSize, Operand, OperandContext};
use crate::{BinaryFile, BinarySection, VirtualAddress, VirtualAddress64};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Disasm(e: disasm::Error) {
            display("Disassembly error: {}", e)
            from()
        }
    }
}

pub type Cfg<'a, E, S> = cfg::Cfg<CfgState<'a, E, S>>;

#[derive(Debug)]
pub struct CfgState<'a, E: ExecutionState<'a>, S: AnalysisState> {
    data: Box<(E, S)>,
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
                    data.get(idx + 1..).and_then(|mut x| x.read_u32::<LittleEndian>().ok())
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
    out.sort_by_key(|x| (x.callee, x.caller));
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
                    data.get(idx + 1..).and_then(|mut x| x.read_i32::<LittleEndian>().ok())
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
    out.sort_by_key(|x| (x.callee, x.caller));
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
                code.get(idx + 1..).and_then(|mut x| x.read_u32::<LittleEndian>().ok())
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
                code.get(idx + 1..).and_then(|mut x| x.read_i32::<LittleEndian>().ok())
                    .map(|relative| (idx as i64 + 5).wrapping_add(relative as i64) as u64)
                    .filter(|&target| {
                        (target as usize) < code.len() - 5
                    })
                    .map(|x| VirtualAddress64(section_base.0 + x))
            })
    });
}

/// Attempts to find functions of a binary, accepting ones that are called/long jumped to
/// from somewhere.
/// Returns addresses relative to start of code section.
pub fn find_functions<'a, E: ExecutionState<'a>>(
    file: &BinaryFile<E::VirtualAddress>,
    relocs: &[E::VirtualAddress],
) -> Vec<E::VirtualAddress> {
    let mut called_functions = Vec::new();
    for section in file.code_sections() {
        find_functions_from_calls::<E>(
            &section.data,
            section.virtual_address,
            &mut called_functions,
        );
    }
    called_functions.extend(find_funcptrs(file, relocs).iter().map(|x| x.callee));
    called_functions.sort();
    called_functions.dedup();
    called_functions
}

// Sorted by address
pub fn find_switch_tables(file: &BinaryFile<VirtualAddress>, relocs: &[VirtualAddress]) -> Vec<FuncPtrPair<VirtualAddress>> {
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
            (&sect.data[offset..]).read_u64::<LittleEndian>().ok()
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
            None => return Err(crate::Error::OutOfBounds),
        }
    }
}

pub fn find_relocs<'a, E: ExecutionState<'a>>(
    file: &BinaryFile<E::VirtualAddress>,
) -> Result<Vec<E::VirtualAddress>, crate::Error> {
    E::find_relocs(file)
}

pub fn find_relocs_x86(file: &BinaryFile<VirtualAddress>) -> Result<Vec<VirtualAddress>, crate::Error> {
    let pe_header = file.base + file.read_u32(file.base + 0x3c)?;
    let reloc_offset = file.read_u32(pe_header + 0xa0)?;
    let reloc_len = file.read_u32(pe_header + 0xa4)?;
    let relocs = file.slice_from(reloc_offset..reloc_offset + reloc_len)?;
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < relocs.len() {
        let rva = try_get!(relocs, offset..).read_u32::<LittleEndian>()?;
        let base = file.base + rva;
        let size = try_get!(relocs, offset + 4..).read_u32::<LittleEndian>()? as usize;
        if size < 8 {
            break;
        }
        let block_relocs = try_get!(relocs, offset + 8..offset + size);
        for mut reloc in block_relocs.chunks_exact(2) {
            if let Ok(c) = reloc.read_u16::<LittleEndian>() {
                if c & 0xf000 == 0x3000 {
                    result.push(base + u32::from(c & 0xfff));
                }
            }
        }
        offset += size;
    }
    result.sort();
    Ok(result)
}

pub fn find_relocs_x86_64(
    file: &BinaryFile<VirtualAddress64>,
) -> Result<Vec<VirtualAddress64>, crate::Error> {
    let pe_header = file.base + file.read_u32(file.base + 0x3c)?;
    let reloc_offset = file.read_u32(pe_header + 0xb0)?;
    let reloc_len = file.read_u32(pe_header + 0xb4)?;
    let relocs = file.slice_from(reloc_offset..reloc_offset + reloc_len)?;
    let mut result = Vec::new();
    let mut offset = 0;
    while offset < relocs.len() {
        let rva = try_get!(relocs, offset..).read_u32::<LittleEndian>()?;
        let base = file.base + rva;
        let size = try_get!(relocs, offset + 4..).read_u32::<LittleEndian>()? as usize;
        if size < 8 {
            break;
        }
        let block_relocs = try_get!(relocs, offset + 8..offset + size);
        for mut reloc in block_relocs.chunks_exact(2) {
            if let Ok(c) = reloc.read_u16::<LittleEndian>() {
                if c & 0xf000 == 0xa000 {
                    result.push(base + u32::from(c & 0xfff));
                }
            }
        }
        offset += size;
    }
    result.sort();
    Ok(result)
}

pub struct RelocValues {
    pub address: VirtualAddress,
    pub value: VirtualAddress,
}

/// The returned array is sorted by value
pub fn relocs_with_values(
    file: &BinaryFile<VirtualAddress>,
    mut relocs: &[VirtualAddress],
) -> Result<Vec<RelocValues>, crate::Error> {
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
        for &address in &relocs[..reloc_count] {
            let relative = (address.0 - start_address.0) as usize;
            let value = (&section[relative..]).read_u32::<LittleEndian>().unwrap_or(0);
            result.push(RelocValues {
                address,
                value: VirtualAddress(value),
            });
        }
        relocs = &relocs[reloc_count..];
    }
    result.sort_by_key(|x| x.value);
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
    unchecked_branches: BTreeMap<Exec::VirtualAddress, Box<(Exec, State)>>,
    operand_ctx: &'a OperandContext,
    pub interner: InternMap,
    /// (Func, arg1, arg2)
    pub errors: Vec<(Exec::VirtualAddress, Error)>,
}

pub struct Control<'exec: 'b, 'b, 'c, A: Analyzer<'exec> + 'b> {
    inner: &'c mut ControlInner<'exec, 'b, A::Exec, A::State>,
}

struct ControlInner<'exec: 'b, 'b, Exec: ExecutionState<'exec> + 'b, State: AnalysisState> {
    state: &'b mut (Exec, State),
    analysis: &'b mut FuncAnalysis<'exec, Exec, State>,
    // Set by Analyzer callback if it wants an early exit
    end: Option<End>,
    address: Exec::VirtualAddress,
    instruction_length: u8,
    skip_operation: bool,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum End {
    Function,
    Branch,
}

impl<'exec: 'b, 'b, 'c, A: Analyzer<'exec> + 'b> Control<'exec, 'b, 'c, A> {
    pub fn end_analysis(&mut self) {
        if self.inner.end.is_none() {
            self.inner.end = Some(End::Function);
        }
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

    pub fn exec_state(&mut self) -> (&mut A::Exec, &mut InternMap) {
        (&mut self.inner.state.0, &mut self.inner.analysis.interner)
    }

    pub fn resolve(&mut self, val: &Rc<Operand>) -> Rc<Operand> {
        self.inner.state.0.resolve(val, &mut self.inner.analysis.interner)
    }

    pub fn resolve_apply_constraints(&mut self, val: &Rc<Operand>) -> Rc<Operand> {
        self.inner.state.0.resolve_apply_constraints(&val, &mut self.inner.analysis.interner)
    }

    pub fn unresolve(&mut self, val: &Rc<Operand>) -> Option<Rc<Operand>> {
        self.inner.state.0.unresolve(val, &mut self.inner.analysis.interner)
    }

    pub fn unresolve_memory(&mut self, val: &Rc<Operand>) -> Option<Rc<Operand>> {
        self.inner.state.0.unresolve_memory(val, &mut self.inner.analysis.interner)
    }

    /// Takes current analysis' state as starting state for a function.
    /// ("Calls the function")
    /// However, this does not update the state to what this function changes it to.
    pub fn analyze_with_current_state<A2: Analyzer<'exec, State = A::State, Exec = A::Exec>>(
        &mut self,
        analyzer: &mut A2,
        entry: <A::Exec as ExecutionState<'exec>>::VirtualAddress,
    ) {
        let current_instruction_end = self.current_instruction_end();
        let inner = &mut *self.inner;
        let mut state = inner.state.clone();
        let ctx = inner.analysis.operand_ctx;
        let mut i = inner.analysis.interner.clone();
        state.0.apply_call(current_instruction_end, &mut i);
        let mut analysis =
            FuncAnalysis::custom_state(inner.analysis.binary, ctx, entry, state.0, state.1, i);
        analysis.analyze(analyzer);
    }

    /// Calls the function and updates own state (Both ExecutionState and custom) to
    /// a merge of states at this child function's return points.
    pub fn inline<A2: Analyzer<'exec, State = A::State, Exec = A::Exec>>(
        &mut self,
        analyzer: &mut A2,
        entry: <A::Exec as ExecutionState<'exec>>::VirtualAddress,
    ) {
        let current_instruction_end = self.current_instruction_end();
        let inner = &mut *self.inner;
        let mut state = inner.state.clone();
        let ctx = inner.analysis.operand_ctx;
        let mut i = inner.analysis.interner.clone();
        state.0.apply_call(current_instruction_end, &mut i);
        let mut analysis =
            FuncAnalysis::custom_state(inner.analysis.binary, ctx, entry, state.0, state.1, i);
        let mut analyzer = CollectReturnsAnalyzer::new(analyzer);
        analysis.analyze(&mut analyzer);
        if let Some(state) = analyzer.state {
            *inner.state = *state;
            inner.analysis.interner = analysis.interner;
        }
    }

    pub fn address(&self) -> <A::Exec as ExecutionState<'exec>>::VirtualAddress {
        self.inner.address
    }

    /// Can be used for determining address for a branch when jump isn't followed and such.
    pub fn current_instruction_end(&self) -> <A::Exec as ExecutionState<'exec>>::VirtualAddress {
        self.inner.address + self.inner.instruction_length as u32
    }

    /// Casts to Control<B: Analyzer> with compatible states.
    ///
    /// This is fine as the control doesn't use Analyzer for anything, the type is just defined
    /// to take Analyzer as Control<Self> is cleaner than Control<Self::Exec, Self::State> when
    /// implementing trait methods.
    pub fn cast<'d, B>(&'d mut self) -> Control<'exec, 'b, 'd, B>
    where B: Analyzer<'exec, Exec = A::Exec, State = A::State>,
    {
        Control {
            inner: self.inner,
        }
    }

    /// Create an arithmetic add with size of VirtualAddress.
    pub fn operand_add(&self, left: Rc<Operand>, right: Rc<Operand>) -> Rc<Operand> {
        A::Exec::operand_arith_word(crate::ArithOpType::Add, left, right)
    }

    /// Convenience for cases where `address + CONST * REG_SIZE` is needed
    pub fn const_word_offset(&self, left: Rc<Operand>, right: u32) -> Rc<Operand> {
        let size = <A::Exec as ExecutionState<'exec>>::VirtualAddress::SIZE;
        let constant = self.inner.analysis.operand_ctx.constant(right as u64 * size as u64);
        A::Exec::operand_arith_word(crate::ArithOpType::Add, left, constant)
    }

    /// Convenience for cases where `Mem[address]` is needed
    pub fn mem_word(&self, addr: Rc<Operand>) -> Rc<Operand> {
        A::Exec::operand_mem_word(addr)
    }

    /// Either `op.if_mem32()` or `op.if_mem64()`, depending on word size
    pub fn if_mem_word<'a>(&self, op: &'a Rc<Operand>) -> Option<&'a Rc<Operand>> {
        if <A::Exec as ExecutionState<'exec>>::VirtualAddress::SIZE == 4 {
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
    fn operation(&mut self, _control: &mut Control<'exec, '_, '_, Self>, _op: &Operation) {}
}

impl<'a, Exec: ExecutionState<'a>> FuncAnalysis<'a, Exec, DefaultState> {
    pub fn new(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: &'a OperandContext,
        start_address: Exec::VirtualAddress,
    ) -> FuncAnalysis<'a, Exec, DefaultState> {
        let mut interner = InternMap::new();
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let init_state = Exec::initial_state(&operand_ctx, binary, &mut interner);
                let mut map = BTreeMap::new();
                map.insert(start_address, Box::new((init_state, DefaultState::default())));
                map
            },
            operand_ctx,
            interner,
        }
    }

    pub fn with_state(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: &'a OperandContext,
        start_address: Exec::VirtualAddress,
        state: Exec,
        interner: InternMap,
    ) -> FuncAnalysis<'a, Exec, DefaultState> {
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                map.insert(start_address, Box::new((state, DefaultState::default())));
                map
            },
            operand_ctx,
            interner,
        }
    }
}

impl<'a, Exec: ExecutionState<'a>, State: AnalysisState> FuncAnalysis<'a, Exec, State> {
    pub fn custom_state(
        binary: &'a BinaryFile<Exec::VirtualAddress>,
        operand_ctx: &'a OperandContext,
        start_address: Exec::VirtualAddress,
        exec_state: Exec,
        analysis_state: State,
        interner: InternMap,
    ) -> FuncAnalysis<'a, Exec, State> {
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                map.insert(start_address, Box::new((exec_state, analysis_state)));
                map
            },
            operand_ctx,
            interner,
        }
    }

    pub fn analyze<A: Analyzer<'a, State = State, Exec = Exec>>(&mut self, analyzer: &mut A) {
        while let Some((addr, mut branch_state)) = self.pop_next_branch() {
            let mut state = match self.cfg.get_state(addr) {
                Some(state) => {
                    Exec::merge_states(&mut state.data.0, &mut branch_state.0, &mut self.interner)
                        .map(|e| {
                            let mut state = state.data.1.clone();
                            state.merge(branch_state.1);
                            (e, state)
                        })
                }
                None => Some(*branch_state),
            };
            if let Some(ref mut state) = state {
                let end = self.analyze_branch(analyzer, addr, state);
                if end == Some(End::Function) {
                    break;
                }
            }
        }
    }

    fn analyze_branch<'b, A: Analyzer<'a, State = State, Exec = Exec>>(
        &mut self,
        analyzer: &mut A,
        addr: Exec::VirtualAddress,
        state: &'b mut (Exec, State),
    ) -> Option<End> {
        let binary = self.binary;
        let section = match binary.section_by_addr(addr) {
            Some(s) => s,
            None => return None,
        };

        let operand_ctx = self.operand_ctx;
        let mut inner = ControlInner {
            analysis: self,
            address: addr,
            instruction_length: 0,
            skip_operation: false,
            state,
            end: None,
        };
        let mut control = Control {
            inner: &mut inner,
        };
        analyzer.branch_start(&mut control);
        if control.inner.end.is_some() {
            return control.inner.end;
        }

        let rva = (addr.as_u64() - section.virtual_address.as_u64()) as usize;
        let mut disasm = Exec::Disassembler::new(
            &section.data,
            rva,
            section.virtual_address,
            operand_ctx,
        );

        let init_state = control.inner.state.clone();
        let mut cfg_out_edge = CfgOutEdges::None;
        // skip_operation from branch_start is no-op, clear the flag here
        // if it was set by user code
        control.inner.skip_operation = false;
        'branch_loop: loop {
            let instruction = loop {
                let address = disasm.address();
                control.inner.address = address;
                let ins = match disasm.next(operand_ctx) {
                    Ok(o) => o,
                    Err(disasm::Error::Branch) => {
                        break 'branch_loop;
                    }
                    Err(e) => {
                        control.inner.analysis.errors.push((address, e.into()));
                        break 'branch_loop;
                    }
                };
                control.inner.instruction_length = ins.len() as u8;
                if !ins.ops().is_empty() {
                    break ins;
                }
            };
            for op in instruction.ops() {
                analyzer.operation(&mut control, op);
                if control.inner.skip_operation {
                    control.inner.skip_operation = false;
                    continue;
                }
                if control.inner.end.is_some() {
                    return control.inner.end;
                }
                match op {
                    disasm::Operation::Jump { condition, to } => {
                        update_analysis_for_jump(
                            control.inner.analysis,
                            &mut control.inner.state,
                            condition.clone(),
                            to.clone(),
                            &instruction,
                            &mut cfg_out_edge,
                        );
                    }
                    disasm::Operation::Return(_) => (),
                    o => {
                        control.inner.state.0.update(o, &mut control.inner.analysis.interner);
                    }
                }
            }
        }
        control.inner.analysis.cfg.add_node(addr, CfgNode {
            out_edges: cfg_out_edge,
            state: CfgState {
                data: Box::new(init_state),
                phantom: Default::default(),
            },
            end_address: control.inner.address, // Is this correct?
            distance: 0,
        });

        analyzer.branch_end(&mut control);
        control.inner.end
    }

    fn add_unchecked_branch(
        &mut self,
        addr: Exec::VirtualAddress,
        mut state: Box<(Exec, State)>,
    ) {
        use std::collections::btree_map::Entry;

        match self.unchecked_branches.entry(addr) {
            Entry::Vacant(e) => {
                e.insert(state);
            }
            Entry::Occupied(mut e) => {
                let interner = &mut self.interner;
                let val = e.get_mut();
                if let Some(new) = Exec::merge_states(&mut val.0, &mut state.0, interner) {
                    val.0 = new;
                    val.1.merge(state.1);
                }
            }
        }
    }

    pub fn next_branch<'b>(&'b mut self) -> Option<Branch<'b, 'a, Exec, State>> {
        while let Some((addr, mut branch_state)) = self.pop_next_branch() {
            let state = match self.cfg.get_state(addr) {
                Some(state) => {
                    let old_exec = &mut state.data.0;
                    let old_state = &state.data.1;
                    Exec::merge_states(old_exec, &mut branch_state.0, &mut self.interner)
                        .map(|exec| {
                            let mut new_custom_state = old_state.clone();
                            new_custom_state.merge(branch_state.1);
                            Box::new((exec, new_custom_state))
                        })
                }
                None => Some(branch_state),
            };

            if let Some(state) = state {
                return Some(Branch {
                    analysis: self,
                    addr,
                    init_state: Some(state.clone()),
                    state: state,
                    cfg_out_edge: CfgOutEdges::None,
                })
            }
        }
        None
    }

    fn pop_next_branch(&mut self) ->
        Option<(Exec::VirtualAddress, Box<(Exec, State)>)>
    {
        let addr = self.unchecked_branches.keys().next().cloned()?;
        let state = self.unchecked_branches.remove(&addr).unwrap();
        Some((addr, state))
    }

    pub fn finish(self) -> (Cfg<'a, Exec, State>, Vec<(Exec::VirtualAddress, Error)>) {
        self.finish_with_changes(|_, _, _, _| {})
    }

    /// As this will run analysis, allows user to manipulate the state during it
    pub fn finish_with_changes<F>(
        mut self,
        mut hook: F
    ) -> (Cfg<'a, Exec, State>, Vec<(Exec::VirtualAddress, Error)>)
    where F: FnMut(
        &Operation,
        &mut Exec,
        Exec::VirtualAddress,
        &mut InternMap,
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
        cfg.interner = self.interner;
        (cfg, self.errors)
    }
}

struct RunHookAnalyzer<'e, F, Exec: ExecutionState<'e>, S: AnalysisState> {
    phantom: std::marker::PhantomData<(&'e (), *const Exec, *const S)>,
    hook: F,
}

impl<'e, F, Exec, S> Analyzer<'e> for RunHookAnalyzer<'e, F, Exec, S>
where F: FnMut(&Operation, &mut Exec, Exec::VirtualAddress, &mut InternMap),
      Exec: ExecutionState<'e>,
      S: AnalysisState,
{
    type State = S;
    type Exec = Exec;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation) {
        let address = ctrl.address();
        let (state, i) = ctrl.exec_state();
        (self.hook)(op, state, address, i);
    }
}

struct FinishAnalyzer<'a, 'e, F, Exec: ExecutionState<'e>> {
    hook: F,
    result: Rc<Operand>,
    end_address: Exec::VirtualAddress,
    condition: &'a Rc<Operand>,
}

impl<'a, 'e, F, Exec: ExecutionState<'e>> Analyzer<'e> for FinishAnalyzer<'a, 'e, F, Exec>
where F: FnMut(&Operation, &mut Exec, Exec::VirtualAddress, &mut InternMap)
{
    type State = DefaultState;
    type Exec = Exec;
    fn operation(&mut self, ctrl: &mut Control<'e, '_, '_, Self>, op: &Operation) {
        let address = ctrl.address();
        let (state, i) = ctrl.exec_state();
        (self.hook)(op, state, address, i);
        let final_op = if address == self.end_address {
            true
        } else {
            match *op {
                Operation::Jump { .. } | Operation::Return(_) => true,
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
struct CollectReturnsAnalyzer<'a, 'exec: 'a, A: Analyzer<'exec>> {
    inner: &'a mut A,
    state: Option<Box<(A::Exec, A::State)>>,
}

impl<'a, 'exec: 'a, A: Analyzer<'exec>> CollectReturnsAnalyzer<'a, 'exec, A> {
    fn new(inner: &'a mut A) -> CollectReturnsAnalyzer<'a, 'exec, A> {
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

    fn operation(&mut self, control: &mut Control<'exec, '_, '_, Self>, op: &Operation) {
        self.inner.operation(&mut control.cast(), op);
        if let disasm::Operation::Return(_) = op {
            match self.state {
                Some(ref mut state) => {
                    let (new, interner) = control.exec_state();
                    let new_exec = Self::Exec::merge_states(&mut state.0, new, interner);
                    if let Some(new_exec) = new_exec {
                        state.0 = new_exec;
                        state.1.merge(control.user_state().clone());
                    }
                }
                None => {
                    self.state = Some(Box::new(
                        (control.exec_state().0.clone(), control.user_state().clone())
                    ));
                }
            }
        }
    }
}

/// NOTE: You have to iterate through branch.operations for the analysis to add new branches
/// correctly.
pub struct Branch<'a, 'exec: 'a, Exec: ExecutionState<'exec>, State: AnalysisState> {
    analysis: &'a mut FuncAnalysis<'exec, Exec, State>,
    addr: Exec::VirtualAddress,
    state: Box<(Exec, State)>,
    init_state: Option<Box<(Exec, State)>>,
    cfg_out_edge: CfgOutEdges<Exec::VirtualAddress>,
}

pub struct Operations<'a, 'branch: 'a, 'exec: 'branch, Exec: ExecutionState<'exec>, S: AnalysisState> {
    branch: &'a mut Branch<'branch, 'exec, Exec, S>,
    disasm: Exec::Disassembler,
    current_ins: Option<Instruction<Exec::VirtualAddress>>,
    ins_pos: usize,
}

impl<'a, 'exec: 'a, Exec: ExecutionState<'exec>, S: AnalysisState> Branch<'a, 'exec, Exec, S> {
    pub fn state<'b>(&'b mut self) -> (&'b Exec, &'b mut InternMap) {
        (&self.state.0, &mut self.analysis.interner)
    }

    pub fn operations<'b>(&'b mut self) -> Operations<'b, 'a, 'exec, Exec, S> {
        let pos = (
            self.addr.as_u64() -
            self.analysis.binary.code_section().virtual_address.as_u64()
        ) as usize;
        Operations {
            disasm: Exec::Disassembler::new(
                &self.analysis.binary.code_section().data,
                pos,
                self.analysis.binary.code_section().virtual_address,
                self.analysis.operand_ctx,
            ),
            current_ins: None,
            ins_pos: 0,
            branch: self,
        }
    }

    fn end_block(&mut self, end_address: Exec::VirtualAddress) {
        self.analysis.cfg.add_node(self.addr, CfgNode {
            out_edges: self.cfg_out_edge.clone(),
            state: CfgState {
                data: self.init_state.take().expect("end_block called twice"),
                phantom: Default::default(),
            },
            end_address,
            distance: 0,
        });
    }

    fn process_operation(&mut self, op: Operation, ins: &Instruction<Exec::VirtualAddress>) -> Result<(), Error> {
        match op {
            disasm::Operation::Jump { condition, to } => {
                update_analysis_for_jump(
                    &mut self.analysis,
                    &mut self.state,
                    condition,
                    to,
                    ins,
                    &mut self.cfg_out_edge,
                );
            }
            disasm::Operation::Return(_) => (),
            o => {
                self.state.0.update(&o, &mut self.analysis.interner);
            }
        }
        Ok(())
    }

    pub fn address(&self) -> Exec::VirtualAddress {
        self.addr
    }
}

impl<'a, 'branch, 'exec: 'a, Exec: ExecutionState<'exec>, S: AnalysisState>
    Operations<'a, 'branch, 'exec, Exec, S>
{
    pub fn next(&mut self) -> Option<(
        &Operation,
        &mut Exec,
        Exec::VirtualAddress,
        &mut InternMap,
    )> {
        // Already finished
        if self.branch.init_state.is_none() {
            return None;
        }
        let mut yield_ins = false;
        if let Some(ref ins) = self.current_ins  {
            let op = &ins.ops()[self.ins_pos];
            if let Err(e) = self.branch.process_operation(op.clone(), ins) {
                self.branch.analysis.errors.push((ins.address(), e));
                self.branch.end_block(ins.address());
                return None;
            }
            self.ins_pos += 1;
            yield_ins = self.ins_pos < ins.ops().len();
        }
        if yield_ins {
            if let Some(ref ins) = self.current_ins {
                let state = &mut self.branch.state.0;
                let interner = &mut self.branch.analysis.interner;
                let op = &ins.ops()[self.ins_pos];
                return Some((op, state, ins.address(), interner));
            }
        }

        let instruction;
        loop {
            let address = self.disasm.address();
            let ins = match self.disasm.next(&self.branch.analysis.operand_ctx) {
                Ok(o) => o,
                Err(disasm::Error::Branch) => {
                    self.branch.end_block(address);
                    return None;
                }
                Err(e) => {
                    self.branch.analysis.errors.push((address, e.into()));
                    self.branch.end_block(address);
                    return None;
                }
            };
            if !ins.ops().is_empty() {
                instruction = ins;
                break;
            }
        }
        self.current_ins = Some(instruction);
        self.ins_pos = 0;
        let ins = self.current_ins.as_ref().unwrap();
        let state = &mut self.branch.state.0;
        let interner = &mut self.branch.analysis.interner;
        let op = &ins.ops()[0];
        Some((op, state, ins.address(), interner))
    }

    pub fn current_address(&self) -> Exec::VirtualAddress {
        self.disasm.address()
    }
}

impl<'a, 'branch: 'a, 'exec: 'branch, Exec: ExecutionState<'exec>, S: AnalysisState> Drop for
    Operations<'a, 'branch, 'exec, Exec, S>
{
    fn drop(&mut self) {
        while let Some(_) = self.next() {
        }
    }
}

fn try_add_branch<'exec, Exec: ExecutionState<'exec>, S: AnalysisState>(
    analysis: &mut FuncAnalysis<'exec, Exec, S>,
    state: Box<(Exec, S)>,
    to: Rc<Operand>,
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
            let simplified = Operand::simplified(to);
            trace!("Couldnt resolve jump dest @ {:x}: {:?}", address, simplified);
            None
        }
    }
}

// TODO this should probs take state as Box<(Exec, S)> to avoid copies, and have it be
// error if a branch has something after the jump.
fn update_analysis_for_jump<'exec, Exec: ExecutionState<'exec>, S: AnalysisState>(
    analysis: &mut FuncAnalysis<'exec, Exec, S>,
    state: &mut (Exec, S),
    condition: Rc<Operand>,
    to: Rc<Operand>,
    ins: &Instruction<Exec::VirtualAddress>,
    cfg_out_edge: &mut CfgOutEdges<Exec::VirtualAddress>,
) {
    /// Returns address of the table,
    /// operand that is being used to index the table,
    /// constant that is being added to any values read from the table,
    /// and size in bytes of one value in table.
    /// E.g. dest = ret.2 + read_'ret.3'_bytes(ret.0 + ret.1 * ret.3)
    fn is_switch_jump<VirtualAddress: exec_state::VirtualAddress>(
        to: &Rc<Operand>,
    ) -> Option<(VirtualAddress, &Rc<Operand>, u64, u32)> {
        let (base, mem) = match to.if_arithmetic_add() {
            Some((l, r)) => Operand::either(l, r, |x| x.if_constant())?,
            None => (0, to),
        };
        mem.if_memory()
            .and_then(|mem| mem.address.if_arithmetic_add())
            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_arithmetic_mul64()))
            .and_then(|((l, r), switch_table)| {
                let switch_table = switch_table.if_constant()?;
                let (c, index) = Operand::either(l, r, {
                    |x| x.if_constant().and_then(|c| u32::try_from(c).ok())
                })?;
                if c == VirtualAddress::SIZE || base != 0 {
                    Some((VirtualAddress::from_u64(switch_table), index, base, c))
                } else {
                    None
                }
            })
    }

    state.0.maybe_convert_memory_immutable();
    match state.0.resolve_apply_constraints(&condition, &mut analysis.interner).if_constant() {
        Some(0) => {
            let address = ins.address() + ins.len() as u32;
            *cfg_out_edge = CfgOutEdges::Single(NodeLink::new(address));
            analysis.add_unchecked_branch(address, Box::new(state.clone()));
        }
        Some(_) => {
            let mut state = Box::new(state.clone());
            let to = state.0.resolve(&to, &mut analysis.interner);
            let is_switch = is_switch_jump::<Exec::VirtualAddress>(&to);
            if let Some((switch_table_addr, index, base_addr, case_size)) = is_switch {
                let mut cases = Vec::new();
                let code_offset = analysis.binary.code_section().virtual_address;
                let code_len = analysis.binary.code_section().data.len() as u32;
                let mem_size = match case_size {
                    1 => Some(MemAccessSize::Mem8),
                    2 => Some(MemAccessSize::Mem16),
                    4 => Some(MemAccessSize::Mem32),
                    8 => Some(MemAccessSize::Mem64),
                    _ => None,
                };
                let ctx = analysis.operand_ctx;
                let switch_table = ctx.constant(switch_table_addr.as_u64());
                let base = ctx.constant(base_addr);
                let binary = analysis.binary;
                if let Some(mem_size) = mem_size {
                    use crate::operand_helpers::*;
                    let limits = state.0.value_limits(&index);
                    let start = limits.0.min(u32::max_value() as u64) as u32;
                    let end = limits.1.min(u32::max_value() as u64) as u32;

                    for index in start..=end {
                        let case = operand_add64(
                            base.clone(),
                            mem_variable_rc(
                                mem_size,
                                operand_add64(
                                    switch_table.clone(),
                                    ctx.constant(index as u64 * case_size as u64),
                                ),
                            ),
                        );
                        if !binary.relocs.is_empty() {
                            let addr = switch_table_addr + index * case_size;
                            if binary.relocs.binary_search(&addr).is_err() {
                                break;
                            }
                        }
                        let addr = state.0.resolve(&case, &mut analysis.interner).if_constant()
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
                        *cfg_out_edge = CfgOutEdges::Switch(cases, index.clone());
                    }
                }
            } else {
                let dest = try_add_branch(analysis, state.clone(), to.clone(), ins.address());
                *cfg_out_edge = CfgOutEdges::Single(
                    dest.map(NodeLink::new).unwrap_or_else(NodeLink::unknown)
                );
            }
        }
        None => {
            let no_jump_addr = ins.address() + ins.len() as u32;
            let mut jump_state = state.0.assume_jump_flag(&condition, true, &mut analysis.interner);
            let no_jump_state = state.0.assume_jump_flag(&condition, false, &mut analysis.interner);
            let to = jump_state.resolve(&to, &mut analysis.interner);
            analysis.add_unchecked_branch(
                no_jump_addr,
                Box::new((no_jump_state, state.1.clone())),
            );
            let s = Box::new((jump_state, state.1.clone()));
            let dest = try_add_branch(analysis, s, to, ins.address());
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
