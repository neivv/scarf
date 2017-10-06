use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt};
use ordermap::{self, OrderMap};

use disasm::{self, InstructionOps, Operation};
use exec_state::{self, ExecutionState};
use operand::{Operand, OperandContext};
use ::{BinaryFile, Rva, VirtualAddress};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        ExecState(e: exec_state::Error) {
            display("Execution state error: {}", e)
            from()
        }
        Disasm(e: disasm::Error) {
            display("Disassembly error: {}", e)
            from()
        }
    }
}

pub struct FuncCallPair {
    pub caller: Rva,
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

/// Attempts to find functions of a binary, accepting ones that are called/long jumped to
/// from somewhere.
/// Returns addresses relative to start of code section.
pub fn find_functions(file: &BinaryFile) -> Vec<Rva> {
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
    find_vtable_calls(file, &mut called_functions);
    called_functions.sort();
    called_functions.dedup();
    called_functions
}

/// Like find_functions, but requires the function to start with "push"
pub fn find_functions_conservative(file: &BinaryFile) -> Vec<Rva> {
    let code = &file.code_section().data;
    let mut funcs = find_functions(file);
    funcs.retain(|rva| {
        match code.get(rva.0 as usize) {
            Some(&first_byte) => first_byte >= 0x50 && first_byte < 0x58,
            _ => false,
        }
    });
    funcs
}

fn find_vtable_calls(file: &BinaryFile, out: &mut Vec<Rva>) {
    // TODO Could only check relocs
    let code = file.code_section();
    for sect in file.sections.iter().filter(|sect| {
        &sect.name[..] == b".data\0\0\0" || &sect.name[..] == b".rdata\0\0"
    }) {
        let funcs = sect.data.chunks(4)
            .flat_map(|mut c| c.read_u32::<LittleEndian>().ok().map(|x| VirtualAddress(x)))
            .filter(|addr| {
                let code_size = code.data.len() as u32;
                addr >= &code.virtual_address && addr < &(code.virtual_address + code_size)
            })
            .map(|addr| addr - code.virtual_address);
        out.extend(funcs);
    }
}

pub struct FuncAnalysis<'a> {
    binary: &'a BinaryFile,
    /// The value is first byte of each instruction, so instructions
    /// inside others can be checked.
    checked_branches: OrderMap<VirtualAddress, ExecutionState<'a>>,
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
            checked_branches: OrderMap::new(),
            unchecked_branches: {
                let init_state = initial_exec_state(&operand_ctx, binary, &mut interner);
                vec![(start_address, init_state)]
            },
            operand_ctx,
            interner,
        }
    }

    pub fn next_branch<'b>(&'b mut self) -> Option<Branch<'b, 'a>> {
        while let Some((addr, state)) = self.unchecked_branches.pop() {
            let state = match self.checked_branches.entry(addr) {
                ordermap::Entry::Occupied(old_state) => {
                    let merged = exec_state::merge_states(
                        old_state.get(),
                        &state,
                        &mut self.interner,
                    );
                    if let Some(merged) = merged {
                        old_state.insert(merged.clone());
                        Some(merged)
                    } else {
                        None
                    }
                }
                ordermap::Entry::Vacant(entry) => {
                    entry.insert(state.clone());
                    Some(state)
                }
            };
            if let Some(state) = state {
                return Some(Branch {
                    analysis: self,
                    addr,
                    state,
                })
            }
        }
        None
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
        (*return_address).clone().into(),
        constval(binary.dump_code_offset.0 + 0x4230),
        None
    ), interner).unwrap();

    // Set the bytes above return address to 'call eax' to make it look like a legitmate call.
    state.update(disasm::Operation::Move(
        mem_variable(MemAccessSize::Mem8,
            operand_sub(
                return_address.clone(),
                constval(1),
            ),
        ).into(),
        constval(0xd0),
        None,
    ), interner).unwrap();
    state.update(disasm::Operation::Move(
        mem_variable(MemAccessSize::Mem8,
            operand_sub(
                return_address,
                constval(2),
            ),
        ).into(),
        constval(0xff),
        None,
    ), interner).unwrap();
    state
}

/// NOTE: You have to iterate through branch.operations for the analysis to add new branches
/// correctly.
pub struct Branch<'a, 'exec: 'a> {
    analysis: &'a mut FuncAnalysis<'exec>,
    addr: VirtualAddress,
    state: ExecutionState<'exec>,
}

pub struct Operations<'a, 'branch: 'a, 'exec: 'branch> {
    branch: &'a mut Branch<'branch, 'exec>,
    disasm: disasm::Disassembler<'a>,
    start_addresses: Vec<VirtualAddress>,
    current_ins: Option<InstructionOps<'a, 'exec>>,
    current_operation: Option<Operation>,
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
            start_addresses: Vec::with_capacity(32),
            current_ins: None,
            current_operation: None,
            branch: self,
        }
    }

    fn try_add_branch<'b>(
        &'b mut self,
        state: ExecutionState<'exec>,
        to: Rc<Operand>,
        address: VirtualAddress
    ) {
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
            }
            None => {
                let simplified = Operand::simplified(to);
                trace!("Couldnt resolve jump dest @ {:x}: {:?}", address.0, simplified);
            }
        }
    }

    fn process_operation(&mut self, op: Operation, ins: &InstructionOps) -> Result<(), Error> {
        match op {
            disasm::Operation::Jump { condition, to } => {
                // TODO Move simplify to disasm::next
                let condition = Operand::simplified(condition);
                self.state.memory.maybe_convert_immutable();
                match self.state.try_resolve_const(&condition, &mut self.analysis.interner) {
                    Some(0) => {
                        trace!("Branch never jumps to {:?}", to);
                        let address = ins.address() + ins.len() as u32;
                        self.analysis.unchecked_branches.push((address, self.state.clone()));
                    }
                    Some(_) => {
                        trace!("Guaranteed jump to {:?}", to);
                        let state = self.state.clone();
                        self.try_add_branch(state, to, ins.address());
                    }
                    None => {
                        let no_jump_addr = ins.address() + ins.len() as u32;
                        let jump_state = self.state.assume_jump_flag(
                            &condition,
                            true,
                            &mut self.analysis.interner
                        ).unwrap_or_else(|_| self.state.clone());
                        let no_jump_state = self.state.assume_jump_flag(
                            &condition,
                            false,
                            &mut self.analysis.interner
                        ).unwrap_or_else(|_| self.state.clone());
                        // The branch pushed last is analyzed first; prefer loops first
                        let jump_dest =
                            jump_state.try_resolve_const(&to, &mut self.analysis.interner);
                        let jumps_backwards = match jump_dest {
                            Some(x) => x < ins.address().0,
                            None => false,
                        };
                        if jumps_backwards {
                            self.analysis.unchecked_branches.push((no_jump_addr, no_jump_state));
                            self.try_add_branch(jump_state, to, ins.address());
                        } else {
                            self.try_add_branch(jump_state, to, ins.address());
                            self.analysis.unchecked_branches.push((no_jump_addr, no_jump_state));
                        }
                    }
                }
            }
            disasm::Operation::Return(_) => {}
            o => {
                self.state.update(o, &mut self.analysis.interner)?;
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
        if let Some(operation) = self.current_operation.take() {
            if let Some(instruction) = self.current_ins.as_ref() {
                if let Err(e) = self.branch.process_operation(operation, instruction) {
                    self.branch.analysis.errors.push((instruction.address(), e.into()));
                    return None;
                }
            }
        }
        loop {
            if let Some(ref mut ins) = self.current_ins.as_mut() {
                match ins.next() {
                    Some(Ok(op)) => {
                        self.current_operation = Some(op);
                        let state = &mut self.branch.state;
                        let interner = &mut self.branch.analysis.interner;
                        return self.current_operation.as_ref()
                            .map(|x| (x, state, ins.address(), interner));
                    }
                    Some(Err(e)) => {
                        self.branch.analysis.errors.push((ins.address(), e.into()));
                        return None;
                    }
                    None => (),
                }
            }
            let address = self.disasm.address();
            self.start_addresses.push(address);
            let ins = match self.disasm.next(&self.branch.analysis.operand_ctx) {
                Ok(o) => o,
                Err(disasm::Error::Branch(_)) => {
                    return None;
                }
                Err(e) => {
                    self.branch.analysis.errors.push((address, e.into()));
                    return None;
                }
            };
            self.current_ins = Some(ins.ops());
        }
    }
}
