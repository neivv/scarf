use std::collections::BTreeMap;
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt};

use cfg::{CfgOutEdges, NodeLink, OutEdgeCondition};
use disasm::{self, DestOperand, Instruction, Operation};
use exec_state::{self, ExecutionState};
use operand::{Operand, OperandContext};
use ::{BinaryFile, BinarySection, VirtualAddress};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Disasm(e: disasm::Error) {
            display("Disassembly error: {}", e)
            from()
        }
    }
}

pub type Cfg<'a, S> = crate::cfg::Cfg<Box<(ExecutionState<'a>, S)>>;
pub type CfgNode<'a, S> = crate::cfg::CfgNode<Box<(ExecutionState<'a>, S)>>;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncCallPair {
    pub caller: VirtualAddress,
    pub callee: VirtualAddress,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncPtrPair {
    pub address: VirtualAddress,
    pub callee: VirtualAddress,
}

/// Sorted by callee, so looking up callers can be done with bsearch
pub fn find_functions_with_callers(file: &BinaryFile) -> Vec<FuncCallPair> {
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

fn find_functions_from_calls(
    code: &[u8],
    section_base: VirtualAddress,
    out: &mut Vec<VirtualAddress>,
) {
    out.extend({
        code.iter().enumerate()
            .filter(|&(_, &x)| x == 0xe8 || x == 0xe9)
            .flat_map(|(idx, _)| {
                code.get(idx + 1..).and_then(|mut x| x.read_u32::<LittleEndian>().ok())
                    .map(|relative| (idx as u32 + 5).wrapping_add(relative))
            })
            .filter(|&target| {
                (target as usize) < code.len() - 5
            })
            .map(|x| section_base + x)
    });
}

/// Attempts to find functions of a binary, accepting ones that are called/long jumped to
/// from somewhere.
/// Returns addresses relative to start of code section.
pub fn find_functions(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<VirtualAddress> {
    let mut called_functions = Vec::new();
    for section in file.code_sections() {
        find_functions_from_calls(&section.data, section.virtual_address, &mut called_functions);
    }
    called_functions.extend(find_funcptrs(file, relocs).iter().map(|x| x.callee));
    called_functions.sort();
    called_functions.dedup();
    called_functions
}

// Sorted by address
pub fn find_switch_tables(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<FuncPtrPair> {
    let mut out = Vec::with_capacity(4096);
    for sect in file.code_sections() {
        collect_relocs_pointing_to_code(sect, relocs, sect, &mut out);
    }
    out
}


fn find_funcptrs(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<FuncPtrPair> {
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

fn collect_relocs_pointing_to_code(
    code: &BinarySection,
    relocs: &[VirtualAddress],
    sect: &BinarySection,
    out: &mut Vec<FuncPtrPair>,
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
            let offset = (addr - sect.virtual_address).0 as usize;
            (&sect.data[offset..]).read_u32::<LittleEndian>().ok()
                .map(|x| (addr, VirtualAddress(x)))
        })
        .filter(|&(_src_addr, func_addr)| {
            let code_size = code.data.len() as u32;
            func_addr >= code.virtual_address && func_addr < code.virtual_address + code_size
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
            None => return Err(::Error::OutOfBounds),
        }
    }
}

pub fn find_relocs(file: &BinaryFile) -> Result<Vec<VirtualAddress>, ::Error> {
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
        for mut reloc in block_relocs.chunks(2) {
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

pub struct RelocValues {
    pub address: VirtualAddress,
    pub value: VirtualAddress,
}

/// The returned array is sorted by value
pub fn relocs_with_values(
    file: &BinaryFile,
    mut relocs: &[VirtualAddress],
) -> Result<Vec<RelocValues>, ::Error> {
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

pub struct FuncAnalysis<'a, State: AnalysisState> {
    binary: &'a BinaryFile,
    cfg: Cfg<'a, State>,
    unchecked_branches: BTreeMap<VirtualAddress, (ExecutionState<'a>, State)>,
    operand_ctx: &'a OperandContext,
    pub interner: exec_state::InternMap,
    /// (Func, arg1, arg2)
    pub errors: Vec<(VirtualAddress, Error)>,
}

pub struct Control<'exec: 'b, 'b, State: AnalysisState> {
    state: &'b mut (ExecutionState<'exec>, State),
    analysis: &'b mut FuncAnalysis<'exec, State>,
    // Set by Analyzer callback if it wants an early exit
    end: Option<End>,
    address: VirtualAddress,
    instruction_length: u32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum End {
    Function,
    Branch,
}

impl<'exec: 'b, 'b, S: AnalysisState> Control<'exec, 'b, S> {
    pub fn end_analysis(&mut self) {
        if self.end.is_none() {
            self.end = Some(End::Function);
        }
    }

    /// Ends the branch without continuing through anything it leads to
    pub fn end_branch(&mut self) {
        if self.end.is_none() {
            self.end = Some(End::Branch);
        }
    }

    pub fn user_state(&mut self) -> &mut S {
        &mut self.state.1
    }

    pub fn exec_state(&mut self) -> (&mut ExecutionState<'exec>, &mut exec_state::InternMap) {
        (&mut self.state.0, &mut self.analysis.interner)
    }

    pub fn resolve(&mut self, val: &Rc<Operand>) -> Rc<Operand> {
        self.state.0.resolve(val, &mut self.analysis.interner)
    }

    pub fn try_resolve_const(&mut self, val: &Rc<Operand>) -> Option<u32> {
        self.state.0.try_resolve_const(val, &mut self.analysis.interner)
    }

    /// Takes current analysis' state as starting state for a function.
    /// ("Calls the function")
    /// However, this does not update the state to what this function changes it to.
    pub fn analyze_with_current_state<A: Analyzer<State = S>>(
        &mut self,
        analyzer: &mut A,
        entry: VirtualAddress,
    ) {
        use crate::operand_helpers::*;

        let mut state = self.state.clone();
        let ctx = self.analysis.operand_ctx;
        let esp = ctx.register(4);
        let mut i = self.analysis.interner.clone();
        state.0.move_to(
            DestOperand::from_oper(&esp),
            operand_sub(esp.clone(), ctx.const_4()),
            &mut i,
        );
        state.0.move_to(
            DestOperand::from_oper(&mem32(esp)),
            ctx.constant(self.current_instruction_end().0),
            &mut i,
        );
        let mut analysis =
            FuncAnalysis::custom_state(self.analysis.binary, ctx, entry, state.0, state.1, i);
        analysis.analyze(analyzer);
    }

    pub fn address(&self) -> VirtualAddress {
        self.address
    }

    /// Can be used for determining address for a branch when jump isn't followed and such.
    pub fn current_instruction_end(&self) -> VirtualAddress {
        VirtualAddress(self.address.0.wrapping_add(self.instruction_length))
    }
}

pub trait Analyzer {
    type State: AnalysisState;
    fn branch_start(&mut self, _control: &mut Control<Self::State>) { }
    fn branch_end(&mut self, _control: &mut Control<Self::State>) { }
    fn operation(&mut self, _control: &mut Control<Self::State>, _op: &Operation) { }
}

impl<'a> FuncAnalysis<'a, DefaultState> {
    pub fn new<'b>(
        binary: &'b BinaryFile,
        operand_ctx: &'b OperandContext,
        start_address: VirtualAddress,
    ) -> FuncAnalysis<'b, DefaultState> {
        let mut interner = exec_state::InternMap::new();
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let init_state = initial_exec_state(&operand_ctx, binary, &mut interner);
                let mut map = BTreeMap::new();
                map.insert(start_address, (init_state, DefaultState::default()));
                map
            },
            operand_ctx,
            interner,
        }
    }

    pub fn with_state<'b>(
        binary: &'b BinaryFile,
        operand_ctx: &'b OperandContext,
        start_address: VirtualAddress,
        state: ExecutionState<'b>,
        interner: exec_state::InternMap,
    ) -> FuncAnalysis<'b, DefaultState> {
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                map.insert(start_address, (state, DefaultState::default()));
                map
            },
            operand_ctx,
            interner,
        }
    }
}

impl<'a, State: AnalysisState> FuncAnalysis<'a, State> {
    pub fn custom_state<'b>(
        binary: &'b BinaryFile,
        operand_ctx: &'b OperandContext,
        start_address: VirtualAddress,
        exec_state: ExecutionState<'b>,
        analysis_state: State,
        interner: exec_state::InternMap,
    ) -> FuncAnalysis<'b, State> {
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let mut map = BTreeMap::new();
                map.insert(start_address, (exec_state, analysis_state));
                map
            },
            operand_ctx,
            interner,
        }
    }

    pub fn analyze<A: Analyzer<State = State>>(&mut self, analyzer: &mut A) {
        while let Some((addr, (exec_state, analysis_state))) = self.pop_next_branch() {
            let mut state = match self.cfg.get(addr) {
                Some(old_node) => {
                    exec_state::merge_states(&old_node.state.0, &exec_state, &mut self.interner)
                        .map(|e| {
                            let mut state = old_node.state.1.clone();
                            state.merge(analysis_state);
                            (e, state)
                        })
                }
                None => Some((exec_state, analysis_state)),
            };
            if let Some(ref mut state) = state {
                let end = self.analyze_branch(analyzer, addr, state);
                if end == Some(End::Function) {
                    break;
                }
            }
        }
    }

    fn analyze_branch<'b, A: Analyzer<State = State>>(
        &mut self,
        analyzer: &mut A,
        addr: VirtualAddress,
        state: &'b mut (ExecutionState<'a>, State),
    ) -> Option<End> {
        let binary = self.binary;
        let section = match binary.section_by_addr(addr) {
            Some(s) => s,
            None => return None,
        };

        let operand_ctx = self.operand_ctx;
        let mut control = Control {
            analysis: self,
            address: addr,
            instruction_length: 0,
            state,
            end: None,
        };
        analyzer.branch_start(&mut control);
        if control.end.is_some() {
            return control.end;
        }

        let rva = (addr - section.virtual_address).0 as usize;
        let mut disasm = disasm::Disassembler::new(&section.data, rva, section.virtual_address);

        let init_state = control.state.clone();
        let mut cfg_out_edge = CfgOutEdges::None;
        'branch_loop: loop {
            let instruction = loop {
                let address = disasm.address();
                control.address = address;
                let ins = match disasm.next(operand_ctx) {
                    Ok(o) => o,
                    Err(disasm::Error::Branch(_)) => {
                        break 'branch_loop;
                    }
                    Err(_) => {
                        // TODO ERROR?
                        //self.branch.analysis.errors.push((address, e.into()));
                        break 'branch_loop;
                    }
                };
                control.instruction_length = ins.len() as u32;
                if !ins.ops().is_empty() {
                    break ins;
                }
            };
            for op in instruction.ops() {
                analyzer.operation(&mut control, op);
                if control.end.is_some() {
                    return control.end;
                }
                match op {
                    disasm::Operation::Jump { condition, to } => {
                        update_analysis_for_jump(
                            control.analysis,
                            &mut control.state.0,
                            &mut control.state.1,
                            binary,
                            condition.clone(),
                            to.clone(),
                            &instruction,
                            &mut cfg_out_edge,
                        );
                    }
                    disasm::Operation::Return(_) => (),
                    o => {
                        control.state.0.update(o.clone(), &mut control.analysis.interner);
                    }
                }
            }
        }
        control.analysis.cfg.add_node(addr, CfgNode {
            out_edges: cfg_out_edge,
            state: Box::new(init_state),
            end_address: control.address, // Is this correct?
            distance: 0,
        });

        analyzer.branch_end(&mut control);
        control.end
    }

    fn add_unchecked_branch(
        &mut self,
        addr: VirtualAddress,
        exec_state: ExecutionState<'a>,
        analysis_state: State,
    ) {
        use std::collections::btree_map::Entry;

        match self.unchecked_branches.entry(addr) {
            Entry::Vacant(e) => {
                e.insert((exec_state, analysis_state));
            }
            Entry::Occupied(mut e) => {
                let interner = &mut self.interner;
                let val = e.get_mut();
                if let Some(new) = exec_state::merge_states(&exec_state, &val.0, interner) {
                    val.0 = new;
                    val.1.merge(analysis_state);
                }
            }
        }
    }

    pub fn next_branch<'b>(&'b mut self) -> Option<Branch<'b, 'a, State>> {
        while let Some((addr, (exec_state, analysis_state))) = self.pop_next_branch() {
            let state = match self.cfg.get(addr) {
                Some(old_node) => {
                    exec_state::merge_states(&old_node.state.0, &exec_state, &mut self.interner)
                        .map(|e| {
                            let state = old_node.state.1.clone();
                            (e, state)
                        })
                }
                None => Some((exec_state, analysis_state)),
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

    fn pop_next_branch(&mut self) -> Option<(VirtualAddress, (ExecutionState<'a>, State))> {
        let addr = self.unchecked_branches.keys().next().cloned()?;
        let state = self.unchecked_branches.remove(&addr).unwrap();
        Some((addr, state))
    }

    pub fn finish(self) -> (Cfg<'a, State>, Vec<(VirtualAddress, Error)>) {
        self.finish_with_changes(|_, _, _, _| {})
    }

    /// As this will run analysis, allows user to manipulate the state during it
    pub fn finish_with_changes<F>(
        mut self,
        mut hook: F
    ) -> (Cfg<'a, State>, Vec<(VirtualAddress, Error)>)
    where F: FnMut(&Operation, &mut ExecutionState, VirtualAddress, &mut exec_state::InternMap)
    {
        while let Some(mut branch) = self.next_branch() {
            let mut ops = branch.operations();
            while let Some((a, b, c, d)) = ops.next() {
                hook(a, b, c, d);
            }
        }
        let mut cfg = self.cfg;
        cfg.merge_overlapping_blocks();
        cfg.resolve_cond_jump_operands(self.binary, hook);
        cfg.interner = self.interner;
        (cfg, self.errors)
    }
}

fn initial_exec_state<'e>(
    operand_ctx: &'e OperandContext,
    binary: &'e BinaryFile,
    interner: &mut exec_state::InternMap,
) -> ExecutionState<'e> {
    use operand::MemAccessSize;
    use operand::operand_helpers::*;
    let mut state = ExecutionState::with_binary(binary, operand_ctx, interner);

    // Set the return address to somewhere in 0x400000 range
    let return_address = mem32(operand_ctx.register(4));
    state.move_to(
        DestOperand::from_oper(&return_address),
        operand_ctx.constant(binary.code_section().virtual_address.0 + 0x4230),
        interner
    );

    // Set the bytes above return address to 'call eax' to make it look like a legitmate call.
    state.move_to(
        DestOperand::from_oper(&mem_variable(MemAccessSize::Mem8,
            operand_sub(
                return_address.clone(),
                operand_ctx.const_1(),
            ),
        )),
        operand_ctx.constant(0xd0),
        interner
    );
    state.move_to(
        DestOperand::from_oper(&mem_variable(MemAccessSize::Mem8,
            operand_sub(
                return_address,
                operand_ctx.const_2(),
            ),
        )),
        operand_ctx.const_ff(),
        interner
    );
    state
}

/// NOTE: You have to iterate through branch.operations for the analysis to add new branches
/// correctly.
pub struct Branch<'a, 'exec: 'a, State: AnalysisState> {
    analysis: &'a mut FuncAnalysis<'exec, State>,
    addr: VirtualAddress,
    state: (ExecutionState<'exec>, State),
    init_state: Option<(ExecutionState<'exec>, State)>,
    cfg_out_edge: CfgOutEdges,
}

pub struct Operations<'a, 'branch: 'a, 'exec: 'branch, S: AnalysisState> {
    branch: &'a mut Branch<'branch, 'exec, S>,
    disasm: disasm::Disassembler<'a>,
    current_ins: Option<Instruction>,
    ins_pos: usize,
}

impl<'a, 'exec: 'a, S: AnalysisState> Branch<'a, 'exec, S> {
    pub fn state<'b>(&'b mut self) -> (&'b ExecutionState<'exec>, &'b mut exec_state::InternMap) {
        (&self.state.0, &mut self.analysis.interner)
    }

    pub fn operations<'b>(&'b mut self) -> Operations<'b, 'a, 'exec, S> {
        Operations {
            disasm: disasm::Disassembler::new(
                &self.analysis.binary.code_section().data,
                (self.addr - self.analysis.binary.code_section().virtual_address).0 as usize,
                self.analysis.binary.code_section().virtual_address,
            ),
            current_ins: None,
            ins_pos: 0,
            branch: self,
        }
    }

    fn end_block(&mut self, end_address: VirtualAddress) {
        self.analysis.cfg.add_node(self.addr, CfgNode {
            out_edges: self.cfg_out_edge.clone(),
            state: Box::new(self.init_state.take().expect("end_block called twice")),
            end_address,
            distance: 0,
        });
    }

    fn process_operation(&mut self, op: Operation, ins: &Instruction) -> Result<(), Error> {
        let state = &mut self.state.0;
        let analysis_state = &self.state.1;
        let binary = self.analysis.binary;

        match op {
            disasm::Operation::Jump { condition, to } => {
                update_analysis_for_jump(
                    &mut self.analysis,
                    state,
                    analysis_state,
                    binary,
                    condition,
                    to,
                    ins,
                    &mut self.cfg_out_edge,
                );
            }
            disasm::Operation::Return(_) => (),
            o => {
                state.update(o, &mut self.analysis.interner);
            }
        }
        Ok(())
    }

    pub fn address(&self) -> VirtualAddress {
        self.addr
    }
}

impl<'a, 'branch, 'exec: 'a, S: AnalysisState> Operations<'a, 'branch, 'exec, S> {
    pub fn next(&mut self) -> Option<(
        &Operation,
        &mut ExecutionState<'exec>,
        VirtualAddress,
        &mut exec_state::InternMap,
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
                Err(disasm::Error::Branch(_)) => {
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

    pub fn current_address(&self) -> VirtualAddress {
        self.disasm.address()
    }
}

impl<'a, 'branch: 'a, 'exec: 'branch, S: AnalysisState> Drop for Operations<'a, 'branch, 'exec, S> {
    fn drop(&mut self) {
        while let Some(_) = self.next() {
        }
    }
}

fn try_add_branch<'exec, S: AnalysisState>(
    analysis: &mut FuncAnalysis<'exec, S>,
    state: (ExecutionState<'exec>, S),
    to: Rc<Operand>,
    address: VirtualAddress,
) -> Option<VirtualAddress> {
    match to.if_constant() {
        Some(s) => {
            let address = VirtualAddress(s);
            let code_offset = analysis.binary.code_section().virtual_address;
            let code_len = analysis.binary.code_section().data.len() as u32;
            let invalid_dest = address < code_offset || address >= code_offset + code_len;
            if !invalid_dest {
                analysis.add_unchecked_branch(address, state.0, state.1);
            } else {
                trace!("Destination {:x} is out of binary bounds", address.0);
            }
            Some(address)
        }
        None => {
            let simplified = Operand::simplified(to);
            trace!("Couldnt resolve jump dest @ {:x}: {:?}", address.0, simplified);
            None
        }
    }
}

fn update_analysis_for_jump<'exec, S: AnalysisState>(
    analysis: &mut FuncAnalysis<'exec, S>,
    state: &mut ExecutionState<'exec>,
    analysis_state: &S,
    binary: &BinaryFile,
    condition: Rc<Operand>,
    to: Rc<Operand>,
    ins: &Instruction,
    cfg_out_edge: &mut CfgOutEdges,
) {
    fn is_switch_jump(to: &Operand) -> Option<(VirtualAddress, &Rc<Operand>)> {
        to.if_memory()
            .and_then(|mem| mem.address.if_arithmetic_add())
            .and_then(|(l, r)| Operand::either(l, r, |x| x.if_arithmetic_mul()))
            .and_then(|((l, r), switch_table)| {
                let switch_table = switch_table.if_constant()?;
                let (c, index) = Operand::either(l, r, |x| x.if_constant())?;
                if c == 4 {
                    Some((VirtualAddress(switch_table), index))
                } else {
                    None
                }
            })
    }

    // TODO Move simplify to disasm::next
    let condition = Operand::simplified(condition);
    state.memory.maybe_convert_immutable();
    match state.try_resolve_const(&condition, &mut analysis.interner) {
        Some(0) => {
            let address = ins.address() + ins.len() as u32;
            *cfg_out_edge = CfgOutEdges::Single(NodeLink::new(address));
            analysis.add_unchecked_branch(address, state.clone(), analysis_state.clone());
        }
        Some(_) => {
            let state = state.clone();
            let to = state.resolve(&to, &mut analysis.interner);
            if let Some((switch_table, index)) = is_switch_jump(&to) {
                let mut cases = Vec::new();
                let code_section = binary.code_section();
                let code_section_start = code_section.virtual_address;
                let code_section_end =
                    code_section.virtual_address + code_section.virtual_size;
                let case_iter = (0u32..)
                    .map(|x| binary.read_u32(switch_table + x * 4))
                    .take_while(|x| x.is_ok())
                    .flat_map(|x| x.ok())
                    .map(|x| VirtualAddress(x))
                    .take_while(|&addr| {
                        // TODO: Could use relocs instead
                        addr > code_section_start && addr < code_section_end
                    });
                for case in case_iter {
                    analysis.add_unchecked_branch(case, state.clone(), analysis_state.clone());
                    cases.push(NodeLink::new(case));
                }
                if !cases.is_empty() {
                    *cfg_out_edge = CfgOutEdges::Switch(cases, index.clone());
                }
            } else {
                let s = (state, analysis_state.clone());
                let dest = try_add_branch(analysis, s, to.clone(), ins.address());
                *cfg_out_edge = CfgOutEdges::Single(
                    dest.map(NodeLink::new).unwrap_or_else(NodeLink::unknown)
                );
            }
        }
        None => {
            let no_jump_addr = ins.address() + ins.len() as u32;
            let jump_state = state.assume_jump_flag(&condition, true, &mut analysis.interner);
            let no_jump_state = state.assume_jump_flag(&condition, false, &mut analysis.interner);
            let to = jump_state.resolve(&to, &mut analysis.interner);
            analysis.add_unchecked_branch(no_jump_addr, no_jump_state, analysis_state.clone());
            let s = (jump_state, analysis_state.clone());
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
