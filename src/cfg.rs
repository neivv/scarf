use std::rc::Rc;

use analysis::FuncAnalysis;
use disasm::Operation;
use exec_state::{ExecutionState, InternMap};
use operand::{Operand, OperandContext};
use ::{BinaryFile, VirtualAddress};

#[derive(Debug, Clone)]
pub struct Cfg<'exec> {
    // Sorted
    nodes: Vec<(VirtualAddress, CfgNode<'exec>)>,
}

impl<'exec> Cfg<'exec> {
    pub fn new() -> Cfg<'exec> {
        Cfg {
            nodes: Vec::with_capacity(16),
        }
    }

    pub fn get(&self, address: VirtualAddress) -> Option<&CfgNode<'exec>> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(&self.nodes[idx].1),
            _ => None,
        }
    }

    pub fn add_node(&mut self, address: VirtualAddress, node: CfgNode<'exec>) {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => self.nodes[idx] = (address, node),
            Err(idx) => self.nodes.insert(idx, (address, node)),
        }
    }

    pub fn nodes(&self) -> CfgNodeIter {
        CfgNodeIter(self.nodes.iter())
    }

    /// Tries to replace a block with a jump to another one which starts in middle of current.
    pub fn merge_overlapping_blocks(&mut self) {
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

    pub fn resolve_cond_jump_operands<F>(&mut self, binary: &BinaryFile, mut hook: F)
    where F: FnMut(&Operation, &mut ExecutionState, VirtualAddress, &mut InternMap)
    {
        let ctx = OperandContext::new();
        for &mut (address, ref mut node) in &mut self.nodes {
            if let Some(cond) = node.out_edges.as_mut().and_then(|x| x.cond.as_mut()) {
                let mut analysis = FuncAnalysis::new(binary, &ctx, address);
                let mut branch = analysis.next_branch()
                    .expect("New analysis should always have a branch.");
                let mut ops = branch.operations();
                while let Some((op, state, address, i)) = ops.next() {
                    hook(op, state, address, i);
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

#[derive(Debug, Clone)]
pub struct CfgNode<'exec> {
    pub out_edges: Option<CfgOutEdges>,
    // The address is to the first instruction of in edge nodes instead of the jump-here address,
    // obviously so it can be looked up with Cfg::nodes.
    pub state: ExecutionState<'exec>,
    pub end_address: VirtualAddress,
}

#[derive(Debug, Clone)]
pub struct CfgOutEdges {
    pub default: VirtualAddress,
    // The operand is unresolved at the state of jump.
    // Could reanalyse the block and show it resolved (effectively unresolved to start of the
    // block) when Cfg is publicly available.
    pub cond: Option<(VirtualAddress, Rc<Operand>)>,
}

