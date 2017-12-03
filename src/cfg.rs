use std::cmp::Ordering;
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
    entry: NodeLink,
    node_indices_dirty: bool,
}

impl<'exec> Cfg<'exec> {
    pub fn new() -> Cfg<'exec> {
        Cfg {
            nodes: Vec::with_capacity(16),
            entry: NodeLink::new(VirtualAddress(!0)),
            node_indices_dirty: false,
        }
    }

    fn mark_dirty(&mut self) {
        self.node_indices_dirty = true;
    }

    pub fn get(&self, address: VirtualAddress) -> Option<&CfgNode<'exec>> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(&self.nodes[idx].1),
            _ => None,
        }
    }

    pub fn add_node(&mut self, address: VirtualAddress, node: CfgNode<'exec>) {
        if self.nodes.is_empty() {
            self.entry = NodeLink::new(address);
        }
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => self.nodes[idx] = (address, node),
            Err(idx) => self.nodes.insert(idx, (address, node)),
        }
        self.mark_dirty();
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
                        if node_out.default.address == next_out.default.address {
                            node.end_address = next_addr;
                            if node_out.cond.is_some() && next_out.cond.is_none() {
                                next_out.cond = node_out.cond.take();
                                next_out.default.address = node_out.default.address;
                            }
                            *node_out = CfgOutEdges {
                                default: NodeLink::new(next_addr),
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
                            default: NodeLink::new(next_addr),
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
        self.mark_dirty();
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
                        cond.condition = state.resolve(&cond.condition, i);
                        break;
                    }
                }
            }
        }
    }

    /// Can only be called on a graph with all node links being addresses of one of
    /// the nodes in graph.
    pub fn calculate_distances(&mut self) {
        self.calculate_node_indices();
        let nodes = &mut self.nodes[..];
        if nodes[self.entry.index()].1.distance != 0 {
            for node in &mut *nodes {
                node.1.distance = 0;
            }
        }
        calculate_node_distance(nodes, self.entry.index(), 1);
    }

    /// Can only be called on a graph with all node links being addresses of one of
    /// the nodes in graph.
    pub fn calculate_node_indices(&mut self) {
        if !self.node_indices_dirty {
            return;
        }
        self.node_indices_dirty = false;
        for index in 0..self.nodes.len() {
            let (before, node_pair, after) = {
                let (before, rest) = self.nodes.split_at_mut(index);
                let (node, after) = rest.split_first_mut().unwrap();
                (before, node, after)
            };
            let address = node_pair.0;
            let node = &mut node_pair.1;
            if let Some(ref mut out_edges) = node.out_edges {
                let default_address = out_edges.default.address;
                let default_index = if default_address > address {
                    // Common case
                    if after[0].0 == default_address {
                        index + 1
                    } else {
                        index + 1 + after.binary_search_by_key(&default_address, |x| x.0)
                            .expect("Broken graph: linking to invalid node")
                    }
                } else if default_address == address {
                    index
                } else {
                    before.binary_search_by_key(&default_address, |x| x.0)
                        .expect("Broken graph: linking to invalid node")
                };
                out_edges.default.index = default_index as u32;
                if let Some(ref mut cond) = out_edges.cond {
                    cond.node.index = if cond.node.address > address {
                        index + 1 + after.binary_search_by_key(&cond.node.address, |x| x.0)
                            .expect("Broken graph: linking to invalid node")
                    } else if cond.node.address == address {
                        index
                    } else {
                        before.binary_search_by_key(&cond.node.address, |x| x.0)
                            .expect("Broken graph: linking to invalid node")
                    } as u32;
                }
            }
        }
        self.entry.index = self.nodes.binary_search_by_key(&self.entry.address, |x| x.0)
            .expect("Broken graph: Entry was deleted?") as u32;
    }

    /// Can only be called on a graph with all node links being addresses of one of
    /// the nodes in graph.
    pub fn cycles(&mut self) -> Vec<Vec<NodeLink>> {
        #[derive(Copy, Clone, Eq, PartialEq)]
        enum CheckState {
            Unchecked,
            Checking(u16),
            Checked,
        }

        self.calculate_node_indices();
        self.calculate_distances();
        let mut nodes = vec![CheckState::Unchecked; self.nodes.len()];
        let mut chain: Vec<usize> = Vec::with_capacity(self.nodes.len());
        let mut result = Vec::new();
        let mut result_first_pos = Vec::new();
        let mut forks = Vec::new();
        let mut pos = self.entry.index();
        loop {
            let rewind = match nodes[pos] {
                CheckState::Checked => {
                    // Nothing to find, rewind
                    true
                }
                CheckState::Checking(chain_pos) => {
                    let chain_pos = chain_pos as usize;
                    // Loop found in chain[chain_pos..]
                    let start = chain[chain_pos..].iter().enumerate().min_by_key(|&(_, &x)| {
                        self.nodes[x].1.distance
                    }).map(|(i, _)| chain_pos + i).unwrap();
                    result.push(
                        chain[start..].iter().chain(chain[chain_pos..start].iter())
                            .map(|&x| NodeLink {
                                address: self.nodes[x].0,
                                index: x as u32,
                            }).collect()
                    );
                    result_first_pos.push(chain_pos);
                    true
                }
                CheckState::Unchecked => {
                    nodes[pos] = CheckState::Checking(chain.len() as u16);
                    chain.push(pos);
                    if chain.len() >= u16::max_value() as usize {
                        return result;
                    }
                    let (out1, out2) = {
                        let node = &self.nodes[pos].1;
                        match node.out_edges {
                            Some(ref out) => (
                                Some(out.default.index()),
                                out.cond.as_ref().map(|x| x.node.index()),
                            ),
                            None => (None, None),
                        }
                    };
                    if let Some(out2) = out2 {
                        forks.push((pos, out2));
                    }
                    match out1 {
                        Some(out1) => {
                            pos = out1;
                            false
                        }
                        None => {
                            // End, rewind chain
                            true
                        }
                    }
                }
            };
            if rewind {
                let mut earliest_pos = result_first_pos.last().cloned().unwrap_or(!0);
                let (rewind_until, other_branch) = match forks.pop() {
                    Some(s) => s,
                    None => return result,
                };
                while let Some(rewind) = chain.last().cloned() {
                    if rewind == rewind_until {
                        break;
                    }
                    chain.pop();
                    while chain.len() < earliest_pos && !result_first_pos.is_empty() {
                        result_first_pos.pop();
                        earliest_pos = result_first_pos.last().cloned().unwrap_or(!0);
                    }
                    if chain.len() < earliest_pos {
                        nodes[rewind] = CheckState::Checked;
                    } else {
                        nodes[rewind] = CheckState::Unchecked;
                    }
                }
                pos = other_branch;
            }
        }
    }
}

/// Link indices must be correct.
fn calculate_node_distance(
    nodes: &mut [(VirtualAddress, CfgNode)],
    node_index: usize,
    distance: u32,
) {
    let old_dist = nodes[node_index].1.distance;
    if old_dist == 0 || old_dist > distance {
        let mut default_index = None;
        let mut cond_index = None;
        {
            let node = &mut nodes[node_index].1;
            node.distance = distance;
            if let Some(ref out_edges) = node.out_edges {
                default_index = Some(out_edges.default.index());
                cond_index = out_edges.cond.as_ref().map(|x| x.node.index());
            }
        }
        if let Some(default_index) = default_index {
            calculate_node_distance(nodes, default_index, distance + 1);
        }
        if let Some(cond_index) = cond_index {
            calculate_node_distance(nodes, cond_index, distance + 1);
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
    /// Distance from entry, first node has 1, its connected nodes have 2, etc.
    pub distance: u32,
}

#[derive(Debug, Clone)]
pub struct CfgOutEdges {
    pub default: NodeLink,
    // The operand is unresolved at the state of jump.
    // Could reanalyse the block and show it resolved (effectively unresolved to start of the
    // block) when Cfg is publicly available.
    pub cond: Option<OutEdgeCondition>,
}

#[derive(Debug, Clone)]
pub struct OutEdgeCondition {
    pub node: NodeLink,
    pub condition: Rc<Operand>,
}

#[derive(Debug, Clone, Eq)]
/// The address is considered a stable link to a node, while index is invalid if the parent
/// graph's node_indices_dirty is true.
pub struct NodeLink {
    address: VirtualAddress,
    index: u32,
}

impl Ord for NodeLink {
    fn cmp(&self, other: &Self) -> Ordering {
        let NodeLink {
            ref address,
            index: _,
        } = *self;
        address.cmp(&other.address)
    }
}

impl PartialOrd for NodeLink {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for NodeLink {
    fn eq(&self, other: &Self) -> bool {
        let NodeLink {
            ref address,
            index: _,
        } = *self;
        *address == other.address
    }
}

impl NodeLink {
    pub fn new(address: VirtualAddress) -> NodeLink {
        NodeLink {
            address,
            index: !0,
        }
    }

    pub fn address(&self) -> VirtualAddress {
        self.address
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use exec_state::InternMap;
    use operand::OperandContext;
    use VirtualAddress;

    fn node0<'a>(addr: u32, ctx: &'a OperandContext, i: &mut InternMap) -> CfgNode<'a> {
        CfgNode {
            out_edges: None,
            state: ::exec_state::ExecutionState::new(ctx, i),
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    fn node1<'a>(addr: u32, ctx: &'a OperandContext, out: u32, i: &mut InternMap) -> CfgNode<'a> {
        CfgNode {
            out_edges: Some(CfgOutEdges {
                default: NodeLink::new(VirtualAddress(out)),
                cond: None,
            }),
            state: ::exec_state::ExecutionState::new(ctx, i),
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    fn node2<'a>(
        addr: u32,
        ctx: &'a OperandContext,
        out: u32,
        out2: u32,
        i: &mut InternMap,
    ) -> CfgNode<'a> {
        CfgNode {
            out_edges: Some(CfgOutEdges {
                default: NodeLink::new(VirtualAddress(out)),
                cond: Some(OutEdgeCondition {
                    node: NodeLink::new(VirtualAddress(out2)),
                    condition: ::operand_helpers::operand_register(0),
                }),
            }),
            state: ::exec_state::ExecutionState::new(ctx, i),
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    #[test]
    fn distances() {
        let ctx = &::operand::OperandContext::new();
        let mut interner = ::exec_state::InternMap::new();
        let i = &mut interner;
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, ctx, 101, 104, i));
        cfg.add_node(VirtualAddress(101), node2(101, ctx, 102, 104, i));
        cfg.add_node(VirtualAddress(102), node2(102, ctx, 103, 104, i));
        cfg.add_node(VirtualAddress(103), node1(103, ctx, 104, i));
        cfg.add_node(VirtualAddress(104), node0(104, ctx, i));
        cfg.calculate_distances();
        let mut iter = cfg.nodes().map(|x| x.1.distance);
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 2);
        assert!(iter.next().is_none());
    }

    #[test]
    fn cycles() {
        let ctx = &::operand::OperandContext::new();
        let mut interner = ::exec_state::InternMap::new();
        let i = &mut interner;
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, ctx, 101, 103, i));
        cfg.add_node(VirtualAddress(101), node1(101, ctx, 102, i));
        cfg.add_node(VirtualAddress(102), node2(102, ctx, 103, 101, i));
        cfg.add_node(VirtualAddress(103), node1(103, ctx, 104, i));
        cfg.add_node(VirtualAddress(104), node2(104, ctx, 105, 108, i));
        cfg.add_node(VirtualAddress(105), node1(105, ctx, 106, i));
        cfg.add_node(VirtualAddress(106), node2(106, ctx, 104, 107, i));
        cfg.add_node(VirtualAddress(107), node2(107, ctx, 104, 108, i));
        cfg.add_node(VirtualAddress(108), node0(108, ctx, i));
        let mut cycles = cfg.cycles().into_iter()
            .map(|x| x.into_iter().map(|y| y.address().0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        cycles.sort();
        assert_eq!(cycles[0], vec![101, 102]);
        assert_eq!(cycles[1], vec![104, 105, 106]);
        assert_eq!(cycles[2], vec![104, 105, 106, 107]);
        assert_eq!(cycles.len(), 3);
    }
}
