use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt};

use disasm::{self, DestOperand, Instruction, Operation};
use exec_state::{self, ExecutionState};
use operand::{Operand, OperandContext};
use ::{BinaryFile, BinarySection, Rva, VirtualAddress};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Disasm(e: disasm::Error) {
            display("Disassembly error: {}", e)
            from()
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncCallPair {
    pub caller: Rva,
    pub callee: Rva,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct FuncPtrPair {
    pub address: VirtualAddress,
    pub callee: Rva,
}

/// Sorted by callee, so looking up callers can be done with bsearch
pub fn find_functions_with_callers(file: &BinaryFile) -> Vec<FuncCallPair> {
    let code = &file.code_section().data;
    let mut called_functions = code.iter().enumerate()
        .filter(|&(_, &x)| x == 0xe8 || x == 0xe9)
        .flat_map(|(idx, _)| {
            code.get(idx + 1..).and_then(|mut x| x.read_u32::<LittleEndian>().ok())
                .map(|relative| FuncCallPair {
                    caller: Rva(idx as u32),
                    callee: Rva((idx as u32 + 5).wrapping_add(relative)),
                })
        })
        .filter(|pair| {
            (pair.callee.0 as usize) < code.len() - 5
        })
        .collect::<Vec<_>>();
    called_functions.sort_by_key(|x| (x.callee, x.caller));
    called_functions
}

fn find_functions_from_calls(file: &BinaryFile) -> Vec<Rva> {
    let code = &file.code_section().data;
    let mut called_functions = code.iter().enumerate()
        .filter(|&(_, &x)| x == 0xe8 || x == 0xe9)
        .flat_map(|(idx, _)| {
            code.get(idx + 1..).and_then(|mut x| x.read_u32::<LittleEndian>().ok())
                .map(|relative| (idx as u32 + 5).wrapping_add(relative))
        })
        .filter(|&target| {
            (target as usize) < code.len() - 5
        })
        .map(|x| Rva(x))
        .collect::<Vec<_>>();
    called_functions.sort();
    called_functions.dedup();
    called_functions
}

/// Attempts to find functions of a binary, accepting ones that are called/long jumped to
/// from somewhere.
/// Returns addresses relative to start of code section.
pub fn find_functions(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<Rva> {
    let mut called_functions = find_functions_from_calls(file);
    called_functions.extend(find_funcptrs(file, relocs).iter().map(|x| x.callee));
    called_functions.sort();
    called_functions.dedup();
    called_functions
}

fn conservative_function_filter(code: &[u8], func: Rva) -> bool {
    match code.get(func.0 as usize) {
        Some(&first_byte) => first_byte >= 0x50 && first_byte < 0x58,
        _ => false,
    }
}

/// Like find_functions, but requires the function to start with "push"
pub fn find_functions_conservative(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<Rva> {
    let code = &file.code_section().data;
    let mut funcs = find_functions(file, relocs);
    funcs.retain(|&rva| conservative_function_filter(code, rva));
    funcs
}

pub fn find_switch_tables(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<FuncPtrPair> {
    let mut out = Vec::with_capacity(4096);
    let code = file.code_section();
    for sect in file.sections.iter().filter(|sect| &sect.name[..] == b".text\0\0\0") {
        collect_relocs_pointing_to_code(code, relocs, sect, &mut out);
    }
    out
}


fn find_funcptrs(file: &BinaryFile, relocs: &[VirtualAddress]) -> Vec<FuncPtrPair> {
    let mut out = Vec::with_capacity(4096);
    let code = file.code_section();
    for sect in file.sections.iter().filter(|sect| {
        &sect.name[..] == b".data\0\0\0" || &sect.name[..] == b".rdata\0\0"
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
    let funcs = sect.data.chunks(4)
        .enumerate()
        .filter(|&(i, _)| {
            let addr = sect.virtual_address + i as u32 * 4;
            relocs.binary_search(&addr).is_ok()
        })
        .flat_map(|(i, mut c)| {
            let addr = sect.virtual_address + i as u32 * 4;
            c.read_u32::<LittleEndian>().ok().map(|x| (addr, VirtualAddress(x)))
        })
        .filter(|&(_src_addr, func_addr)| {
            let code_size = code.data.len() as u32;
            func_addr >= code.virtual_address && func_addr < code.virtual_address + code_size
        })
        .map(|(src_addr, func_addr)| FuncPtrPair {
            address: src_addr,
            callee: func_addr - code.virtual_address,
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
                    result.push(base + (c & 0xfff) as u32);
                }
            }
        }
        offset += size;
    }
    result.sort();
    Ok(result)
}

pub struct Cfg<'exec> {
    // Sorted
    nodes: Vec<(VirtualAddress, CfgNode<'exec>)>,
}

impl<'exec> Cfg<'exec> {
    fn new() -> Cfg<'exec> {
        Cfg {
            nodes: Vec::with_capacity(16),
        }
    }

    fn get(&self, address: VirtualAddress) -> Option<&CfgNode<'exec>> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(&self.nodes[idx].1),
            _ => None,
        }
    }

    fn add_node(&mut self, address: VirtualAddress, node: CfgNode<'exec>) {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => self.nodes[idx] = (address, node),
            Err(idx) => self.nodes.insert(idx, (address, node)),
        }
    }

    pub fn nodes(&self) -> CfgNodeIter {
        CfgNodeIter(self.nodes.iter())
    }

    /// Tries to replace a block with a jump to another one which starts in middle of current.
    fn merge_overlapping_blocks(&mut self) {
        // TODO: Handle instructions inside instructions
        for i in 0..self.nodes.len() - 1 {
            let (left, rest) = self.nodes.split_at_mut(i + 1);
            let &mut (start_addr, ref mut node) = &mut left[i];
            let mut current_addr = start_addr + 1;
            'merge_loop: loop {
                let &mut (next_addr, ref mut next_node) =
                    match rest.binary_search_by_key(&current_addr, |x| x.0) {
                    Ok(o) => &mut rest[o],
                    Err(e) => {
                        if e >= rest.len() {
                            break 'merge_loop;
                        }
                        &mut rest[e]
                    }
                };
                if next_addr >= node.end_address {
                    break 'merge_loop;
                }
                match (&mut node.out_edges, &mut next_node.out_edges) {
                    (&mut Some(ref mut node_out), &mut Some(ref mut next_out)) => {
                        if node_out.default == next_out.default {
                            node.end_address = next_addr;
                            if node_out.cond.is_some() && next_out.cond.is_none() {
                                next_out.cond = node_out.cond.take();
                                next_out.default = node_out.default;
                            }
                            *node_out = CfgOutEdges {
                                default: next_addr,
                                cond: None,
                            };
                            break 'merge_loop;
                        } else {
                            // Keep the nodes separate
                        }
                    }
                    (out @ &mut None, &mut None) => {
                        node.end_address = next_addr;
                        *out = Some(CfgOutEdges {
                            default: next_addr,
                            cond: None,
                        });
                        break 'merge_loop;
                    }
                    _ => {
                        error!("Logic error: One of {:x}, {:x} has no out edges, \
                               other one does", start_addr.0, next_addr.0);
                    }
                }
                current_addr = next_addr + 1;
            }
        }
    }

    fn resolve_cond_jump_operands(&mut self, binary: &BinaryFile) {
        let ctx = OperandContext::new();
        for &mut (address, ref mut node) in &mut self.nodes {
            if let Some(cond) = node.out_edges.as_mut().and_then(|x| x.cond.as_mut()) {
                let mut analysis = FuncAnalysis::new(binary, &ctx, address);
                let mut branch = analysis.next_branch()
                    .expect("New analysis should always have a branch.");
                let mut ops = branch.operations();
                while let Some((op, state, address, i)) = ops.next() {
                    let final_op = if address == node.end_address {
                        true
                    } else {
                        match *op {
                            Operation::Jump { .. } | Operation::Return(_) => true,
                            _ => false,
                        }
                    };
                    if final_op {
                        cond.1 = state.resolve(&cond.1, i);
                        break;
                    }
                }
            }
        }
    }
}

pub struct CfgNodeIter<'a>(::std::slice::Iter<'a, (VirtualAddress, CfgNode<'a>)>);
impl<'a> Iterator for CfgNodeIter<'a> {
    type Item = (&'a VirtualAddress, &'a CfgNode<'a>);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|x| (&x.0, &x.1))
    }
}

pub struct CfgNode<'exec> {
    pub out_edges: Option<CfgOutEdges>,
    // The address is to the first instruction of in edge nodes instead of the jump-here address,
    // obviously so it can be looked up with Cfg::nodes.
    pub state: ExecutionState<'exec>,
    pub end_address: VirtualAddress,
}

pub struct CfgOutEdges {
    pub default: VirtualAddress,
    // The operand is unresolved at the state of jump.
    // Could reanalyse the block and show it resolved (effectively unresolved to start of the
    // block) when Cfg is publicly available.
    pub cond: Option<(VirtualAddress, Rc<Operand>)>,
}

pub struct FuncAnalysis<'a> {
    binary: &'a BinaryFile,
    cfg: Cfg<'a>,
    unchecked_branches: Vec<(VirtualAddress, ExecutionState<'a>)>,
    operand_ctx: &'a OperandContext,
    pub interner: exec_state::InternMap,
    /// (Func, arg1, arg2)
    pub errors: Vec<(VirtualAddress, Error)>,
}

impl<'a> FuncAnalysis<'a> {
    pub fn new<'b>(
        binary: &'b BinaryFile,
        operand_ctx: &'b OperandContext,
        start_address: VirtualAddress,
    ) -> FuncAnalysis<'b> {
        let mut interner = exec_state::InternMap::new();
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                let init_state = initial_exec_state(&operand_ctx, binary, &mut interner);
                vec![(start_address, init_state)]
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
    ) -> FuncAnalysis<'b> {
        FuncAnalysis {
            binary,
            errors: Vec::new(),
            cfg: Cfg::new(),
            unchecked_branches: {
                vec![(start_address, state)]
            },
            operand_ctx,
            interner,
        }
    }

    pub fn next_branch<'b>(&'b mut self) -> Option<Branch<'b, 'a>> {
        while let Some((addr, state)) = self.next_branch_merge_queued() {
            let state = match self.cfg.get(addr) {
                Some(old_node) => {
                    exec_state::merge_states(&old_node.state, &state, &mut self.interner)
                }
                None => Some(state),
            };
            if let Some(state) = state {
                return Some(Branch {
                    analysis: self,
                    addr,
                    init_state: Some(state.clone()),
                    state,
                    cfg_out_edge: None,
                })
            }
        }
        None
    }

    fn next_branch_merge_queued(&mut self) -> Option<(VirtualAddress, ExecutionState<'a>)> {
        let (addr, mut state) = match self.unchecked_branches.pop() {
            Some(s) => s,
            None => return None,
        };
        let interner = &mut self.interner;
        self.unchecked_branches.retain(|&(a, ref s)| if a == addr {
            if let Some(new) = exec_state::merge_states(&state, s, interner) {
                state = new;
            }
            false
        } else {
            true
        });
        Some((addr, state))
    }

    pub fn finish(mut self) -> (Cfg<'a>, Vec<(VirtualAddress, Error)>) {
        while let Some(mut branch) = self.next_branch() {
            let mut ops = branch.operations();
            while let Some(_) = ops.next() {
            }
        }
        let mut cfg = self.cfg;
        cfg.merge_overlapping_blocks();
        cfg.resolve_cond_jump_operands(self.binary);
        (cfg, self.errors)
    }
}

fn initial_exec_state<'e>(
    operand_ctx: &'e OperandContext,
    binary: &BinaryFile,
    interner: &mut exec_state::InternMap,
) -> ExecutionState<'e> {
    use operand::MemAccessSize;
    use operand::operand_helpers::*;
    let mut state = ExecutionState::new(operand_ctx, interner);

    // Set the return address to somewhere in 0x400000 range
    let return_address = mem32(operand_register(4));
    state.update(disasm::Operation::Move(
        DestOperand::from_oper(&return_address),
        constval(binary.dump_code_offset.0 + 0x4230),
        None
    ), interner);

    // Set the bytes above return address to 'call eax' to make it look like a legitmate call.
    state.update(disasm::Operation::Move(
        DestOperand::from_oper(&mem_variable(MemAccessSize::Mem8,
            operand_sub(
                return_address.clone(),
                constval(1),
            ),
        )),
        constval(0xd0),
        None,
    ), interner);
    state.update(disasm::Operation::Move(
        DestOperand::from_oper(&mem_variable(MemAccessSize::Mem8,
            operand_sub(
                return_address,
                constval(2),
            ),
        )),
        constval(0xff),
        None,
    ), interner);
    state
}

/// NOTE: You have to iterate through branch.operations for the analysis to add new branches
/// correctly.
pub struct Branch<'a, 'exec: 'a> {
    analysis: &'a mut FuncAnalysis<'exec>,
    addr: VirtualAddress,
    state: ExecutionState<'exec>,
    init_state: Option<ExecutionState<'exec>>,
    cfg_out_edge: Option<CfgOutEdges>,
}

pub struct Operations<'a, 'branch: 'a, 'exec: 'branch> {
    branch: &'a mut Branch<'branch, 'exec>,
    disasm: disasm::Disassembler<'a>,
    current_ins: Option<Instruction>,
    ins_pos: usize,
}

impl<'a, 'exec: 'a> Branch<'a, 'exec> {
    pub fn state<'b>(&'b mut self) -> (&'b ExecutionState<'exec>, &'b mut exec_state::InternMap) {
        (&self.state, &mut self.analysis.interner)
    }

    pub fn operations<'b>(&'b mut self) -> Operations<'b, 'a, 'exec> {
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

    fn try_add_branch<'b>(
        &'b mut self,
        state: ExecutionState<'exec>,
        to: Rc<Operand>,
        address: VirtualAddress
    ) -> Option<VirtualAddress> {
        match state.try_resolve_const(&to, &mut self.analysis.interner) {
            Some(s) => {
                let address = VirtualAddress(s);
                let code_offset = self.analysis.binary.code_section().virtual_address;
                let code_len = self.analysis.binary.code_section().data.len() as u32;
                let invalid_dest = address < code_offset || address >= code_offset + code_len;
                if !invalid_dest {
                    self.analysis.unchecked_branches.push((address, state));
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

    fn end_block(&mut self, end_address: VirtualAddress) {
        self.analysis.cfg.add_node(self.addr, CfgNode {
            out_edges: self.cfg_out_edge.take(),
            state: self.init_state.take().expect("end_block called twice"),
            end_address,
        });
    }

    fn process_operation(&mut self, op: Operation, ins: &Instruction) -> Result<(), Error> {
        match op {
            disasm::Operation::Jump { condition, to } => {
                // TODO Move simplify to disasm::next
                let condition = Operand::simplified(condition);
                self.state.memory.maybe_convert_immutable();
                match self.state.try_resolve_const(&condition, &mut self.analysis.interner) {
                    Some(0) => {
                        let address = ins.address() + ins.len() as u32;
                        self.cfg_out_edge = Some(CfgOutEdges {
                            default: address,
                            cond: None,
                        });
                        self.analysis.unchecked_branches.push((address, self.state.clone()));
                    }
                    Some(_) => {
                        let state = self.state.clone();
                        let dest = self.try_add_branch(state, to.clone(), ins.address());
                        self.cfg_out_edge = Some(CfgOutEdges {
                            default: dest.unwrap_or(VirtualAddress(!0)),
                            cond: None,
                        });
                    }
                    None => {
                        let no_jump_addr = ins.address() + ins.len() as u32;
                        let jump_state = self.state.assume_jump_flag(
                            &condition,
                            true,
                            &mut self.analysis.interner
                        );
                        let no_jump_state = self.state.assume_jump_flag(
                            &condition,
                            false,
                            &mut self.analysis.interner
                        );
                        // The branch pushed last is analyzed first; prefer loops first
                        let jump_dest =
                            jump_state.try_resolve_const(&to, &mut self.analysis.interner);
                        let jumps_backwards = match jump_dest {
                            Some(x) => x < ins.address().0,
                            None => false,
                        };
                        let dest;
                        if jumps_backwards {
                            self.analysis.unchecked_branches.push((no_jump_addr, no_jump_state));
                            dest = self.try_add_branch(jump_state, to, ins.address());
                        } else {
                            dest = self.try_add_branch(jump_state, to, ins.address());
                            self.analysis.unchecked_branches.push((no_jump_addr, no_jump_state));
                        }
                        self.cfg_out_edge = Some(CfgOutEdges {
                            default: no_jump_addr,
                            cond: Some((dest.unwrap_or(VirtualAddress(!0)), condition)),
                        });
                    }
                }
            }
            disasm::Operation::Return(_) => {
            }
            o => {
                self.state.update(o, &mut self.analysis.interner);
            }
        }
        Ok(())
    }

    pub fn address(&self) -> VirtualAddress {
        self.addr
    }
}

impl<'a, 'branch, 'exec: 'a> Operations<'a, 'branch, 'exec> {
    pub fn next(&mut self) -> Option<(
        &Operation,
        &mut ExecutionState<'exec>,
        VirtualAddress,
        &mut exec_state::InternMap,
    )> {
        let mut yield_ins = false;
        if let Some(ref ins) = self.current_ins  {
            let op = &ins.ops()[self.ins_pos];
            if let Err(e) = self.branch.process_operation(op.clone(), ins) {
                self.branch.analysis.errors.push((ins.address(), e.into()));
                self.branch.end_block(ins.address());
                return None;
            }
            self.ins_pos += 1;
            yield_ins = self.ins_pos < ins.ops().len();
        }
        if yield_ins {
            if let Some(ref ins) = self.current_ins {
                let state = &mut self.branch.state;
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
            if ins.ops().len() != 0 {
                instruction = ins;
                break;
            }
        }
        self.current_ins = Some(instruction);
        self.ins_pos = 0;
        let ins = self.current_ins.as_ref().unwrap();
        let state = &mut self.branch.state;
        let interner = &mut self.branch.analysis.interner;
        let op = &ins.ops()[0];
        Some((op, state, ins.address(), interner))
    }

    pub fn current_address(&self) -> VirtualAddress {
        self.disasm.address()
    }
}
