use std::cmp::Ordering;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::mem;
use std::rc::Rc;

use crate::exec_state::{InternMap, VirtualAddress};
use crate::operand::{Operand};

#[derive(Debug, Clone)]
pub struct Cfg<State: CfgState> {
    // Sorted
    nodes: Vec<(State::VirtualAddress, CfgNode<State>)>,
    entry: NodeLink<State::VirtualAddress>,
    node_indices_dirty: bool,
    // Just to be set once the cfg is available to users,
    // so they can resolve the states
    pub interner: InternMap,
}

pub trait CfgState {
    type VirtualAddress: VirtualAddress;
}

impl<S: CfgState> Cfg<S> {
    pub fn new() -> Cfg<S> {
        Cfg {
            nodes: Vec::with_capacity(16),
            entry: NodeLink::new(S::VirtualAddress::max_value()),
            node_indices_dirty: false,
            interner: InternMap::new(),
        }
    }

    fn mark_dirty(&mut self) {
        self.node_indices_dirty = true;
    }

    pub fn get(&self, address: S::VirtualAddress) -> Option<&CfgNode<S>> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(&self.nodes[idx].1),
            _ => None,
        }
    }

    pub fn entry_link(&self) -> &NodeLink<S::VirtualAddress> {
        &self.entry
    }

    pub fn entry(&self) -> &CfgNode<S> {
        self.get_by_link(&self.entry)
    }

    pub fn get_by_link(&self, link: &NodeLink<S::VirtualAddress>) -> &CfgNode<S> {
        &self.nodes[link.index as usize].1
    }

    pub fn add_node(&mut self, address: S::VirtualAddress, node: CfgNode<S>) {
        if self.nodes.is_empty() {
            self.entry = NodeLink::new(address);
        }
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => self.nodes[idx] = (address, node),
            Err(idx) => self.nodes.insert(idx, (address, node)),
        }
        self.mark_dirty();
    }

    pub fn nodes<'a>(&'a self) -> CfgNodeIter<'a, S> {
        CfgNodeIter(self.nodes.iter(), 0)
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
                let first_default = node.out_edges.default_address();
                let next_default = next_node.out_edges.default_address();
                if let (Some(def1), Some(def2)) = (first_default, next_default) {
                    if def1 == def2 {
                        node.end_address = next_addr;
                        let node_is_superset = match (&node.out_edges, &next_node.out_edges) {
                            (&CfgOutEdges::Branch(..), &CfgOutEdges::Single(..)) => true,
                            _ => false,
                        };
                        if node_is_superset {
                            next_node.out_edges = mem::replace(
                                &mut node.out_edges,
                                CfgOutEdges::Single(NodeLink::new(next_addr))
                            );
                        } else {
                            node.out_edges = CfgOutEdges::Single(NodeLink::new(next_addr));
                        }
                        break 'merge_loop;
                    } else {
                        // Keep the nodes separate
                    }
                } else {
                    match (&mut node.out_edges, &mut next_node.out_edges) {
                        (out @ &mut CfgOutEdges::None, &mut CfgOutEdges::None) => {
                            node.end_address = next_addr;
                            *out = CfgOutEdges::Single(NodeLink::new(next_addr));
                            break 'merge_loop;
                        }
                        (&mut CfgOutEdges::None, _) | (_, &mut CfgOutEdges::None) => {
                            error!("Logic error: One of {:x}, {:x} has no out edges, \
                                   other one does", start_addr, next_addr);
                        }
                        _ => (),
                    }
                }
                current_addr = next_addr + 1;
            }
        }
        self.mark_dirty();
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
        let mut buf = Vec::with_capacity(0x800);
        calculate_node_distance(nodes, &mut buf, self.entry.index(), 1);
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
            let end_address = node.end_address;
            let update_node = |link: &mut NodeLink<S::VirtualAddress>| {
                if link.is_unknown() {
                    return;
                }
                link.index = if link.address > address {
                    // Common case
                    let next = after.get(0)
                        .unwrap_or_else(|| panic!(
                            "Broken graph: {:?} -> {:?} linking to invalid node",
                            end_address,
                            link.address,
                        ));
                    if next.0 == link.address {
                        index + 1
                    } else {
                        index + 1 + after.binary_search_by_key(&link.address, |x| x.0)
                            .unwrap_or_else(|_| panic!(
                                "Broken graph: {:?} -> {:?} linking to invalid node",
                                end_address,
                                link.address,
                            ))
                    }
                } else if link.address == address {
                    index
                } else {
                    before.binary_search_by_key(&link.address, |x| x.0)
                        .unwrap_or_else(|_| panic!(
                            "Broken graph: {:?} -> {:?} linking to invalid node",
                            end_address,
                            link.address,
                        ))
                } as u32;
            };
            match node.out_edges {
                CfgOutEdges::Single(ref mut n) => update_node(n),
                CfgOutEdges::Branch(ref mut n, ref mut cond) => {
                    update_node(n);
                    update_node(&mut cond.node);
                }
                CfgOutEdges::None => (),
                CfgOutEdges::Switch(ref mut nodes, _) => {
                    for node in nodes.iter_mut() {
                        update_node(node);
                    }
                }
            }
        }
        self.entry.index = self.nodes.binary_search_by_key(&self.entry.address, |x| x.0)
            .expect("Broken graph: Entry was deleted?") as u32;
    }

    /// Can only be called on a graph with all node links being addresses of one of
    /// the nodes in graph.
    pub fn cycles(&mut self) -> Vec<Vec<NodeLink<S::VirtualAddress>>> {
        // This currently generates a bunch of duplicates which are removed at the end.
        // Could be better.
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
        'main_loop: loop {
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
                        break 'main_loop;
                    }
                    let out1 = {
                        let node = &self.nodes[pos].1;
                        match node.out_edges {
                            CfgOutEdges::None => None,
                            CfgOutEdges::Single(ref s) => s.index_if_not_unkown(),
                            CfgOutEdges::Branch(ref s, ref cond) => {
                                if !cond.node.is_unknown() {
                                    forks.push((pos, cond.node.index()));
                                }
                                s.index_if_not_unkown()
                            }
                            CfgOutEdges::Switch(ref cases, _) => {
                                for other in cases.iter().skip(1) {
                                    forks.push((pos, other.index()));
                                }
                                cases.first().map(|x| x.index())
                            }
                        }
                    };
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
                    None => break 'main_loop,
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
        result.sort();
        result.dedup();
        result
    }

    pub fn immediate_postdominator<'a>(
        &'a self,
        node: NodeIndex<'a, S>,
    ) -> Option<NodeBorrow<'a, S>> {
        // Do one run through one branch, mark the nodes 1.. based on their distance.
        // Any branches met store the mark they branched off from.
        // Then run through other branch until mark > 2 is found, keep track of largest
        // mark > 2 node of each mark's sub branches.
        // If a branch doesn't reach any marked node, then the branch's original mark and
        // all marks after it are set !0 "unaccptable"
        //
        // Also mark any visited 0-marked nodes 1, and if another branch reaches 1-marked node,
        // it can just stop that branch.
        // Because we stop on 1-marked nodes, it is important to go through any queued
        // branches from lowest mark to highest, as lower marks override higher ones.
        //
        // Once there are no branches to check, the postdominator is received by the first
        // mark, for which any smaller mark won't reach another mark larger than it through
        // its child branches.
        assert!(!self.node_indices_dirty);

        struct State<'exec, S: CfgState> {
            mark_buf: Vec<u32>,
            forks: VecDeque<(usize, u32)>,
            mark_indices: Vec<usize>,
            nodes: &'exec [(S::VirtualAddress, CfgNode<S>)],
        }
        let mut state = State {
            mark_buf: vec![0u32; self.nodes.len()],
            forks: VecDeque::with_capacity(self.nodes.len()),
            mark_indices: Vec::with_capacity(32),
            nodes: &self.nodes,
        };

        // Return Err on quick exit, Ok for continuing to rest of branches
        fn traverse_first_branch<'a, S: CfgState>(
            state: &mut State<'a, S>,
            node: NodeIndex<'a, S>,
        ) -> Result<Vec<u32>, Option<NodeBorrow<'a, S>>> {
            let mut pos = node.0 as usize;
            let mut mark = 1;
            loop {
                while state.mark_buf[pos] != 0 {
                    // We reached a loop, that's not useful.
                    // Switch marks that are larger than last fork to "useless"
                    let (branch, last_good_mark) = match state.forks.pop_back() {
                        None => return Err(None),
                        Some(s) => s,
                    };
                    while mark > last_good_mark + 1 {
                        let index = state.mark_indices.pop().unwrap();
                        state.mark_buf[index] = !0;
                        mark -= 1;
                    }
                    pos = branch;
                }
                state.mark_buf[pos] = mark;
                state.mark_indices.push(pos);
                let index = match state.nodes[pos].1.out_edges {
                    CfgOutEdges::None => None,
                    CfgOutEdges::Single(ref s) => {
                        let index = s.index_if_not_unkown();
                        if state.forks.is_empty() {
                            if let Some(index) = index.or_else(|| state.mark_indices.get(1).cloned()) {
                                // Uh, ok, the postdominator is just the next node then.
                                // Or if a branch was rewinded to start.
                                return Err(Some(NodeBorrow {
                                    address: state.nodes[index].0,
                                    node: &state.nodes[index].1,
                                    index: NodeIndex(index as u32, PhantomData),
                                }));
                            }
                        }
                        index
                    }
                    CfgOutEdges::Branch(ref s, ref cond) => {
                        if !cond.node.is_unknown() {
                            state.forks.push_back((cond.node.index(), mark));
                        }
                        s.index_if_not_unkown()
                    }
                    CfgOutEdges::Switch(ref cases, _) => {
                        for other in cases.iter().skip(1) {
                            state.forks.push_back((other.index(), mark));
                        }
                        cases.first().map(|x| x.index())
                    }
                };
                mark += 1;
                pos = match index {
                    None => {
                        let mut mark_limits = vec![!0u32; mark as usize - 1];
                        for (i, mark) in mark_limits[..mark as usize - 2].iter_mut().enumerate() {
                            *mark = i as u32 + 2;
                        }
                        return Ok(mark_limits);
                    }
                    Some(s) => s,
                };
            }
        }

        fn traverse_rest<'a, S: CfgState>(
            state: &mut State<'a, S>,
            mark_limits: &mut [u32],
        ) -> Option<NodeBorrow<'a, S>> {
            loop {
                let mut current_mark;
                let mut pos;
                loop {
                    match state.forks.pop_front() {
                        Some((p, m)) => {
                            pos = p;
                            current_mark = m;
                        }
                        None => {
                            let mut end_mark = mark_limits[0];
                            let mut pos = 2;
                            while pos < end_mark {
                                if end_mark == !0 {
                                    return None;
                                }
                                if mark_limits[pos as usize - 1] > end_mark {
                                    end_mark = mark_limits[pos as usize - 1];
                                }
                                pos += 1;
                            }
                            if end_mark == !0 {
                                return None;
                            }
                            let index = state.mark_indices[end_mark as usize - 1];
                            return Some(NodeBorrow {
                                address: state.nodes[index].0,
                                node: &state.nodes[index].1,
                                index: NodeIndex(index as u32, PhantomData),
                            });
                        }
                    };
                    if mark_limits[current_mark as usize - 1] != !0 {
                        break;
                    }
                    // The mark was known to not reach any other mark at one branch,
                    // so don't bother.
                }
                match state.mark_buf[pos] {
                    0 => {
                        state.mark_buf[pos] = 1;
                        let index = match state.nodes[pos].1.out_edges {
                            CfgOutEdges::None => None,
                            CfgOutEdges::Single(ref s) => s.index_if_not_unkown(),
                            CfgOutEdges::Branch(ref s, ref cond) => {
                                if cond.node.is_unknown() {
                                    // Effectively an end where we didn't meet any of the marked nodes
                                    for mark in mark_limits[current_mark as usize - 1..].iter_mut() {
                                        *mark = !0;
                                    }
                                    None
                                } else {
                                    let cond_index = cond.node.index();
                                    if state.mark_buf[cond_index] != 1 {
                                        state.forks.push_back((cond_index, current_mark));
                                    }
                                    s.index_if_not_unkown()
                                }
                            }
                            CfgOutEdges::Switch(ref cases, _) => {
                                for other in cases.iter().skip(1) {
                                    if other.index() != 1 {
                                        state.forks.push_back((other.index(), current_mark));
                                    }
                                }
                                cases.first().map(|x| x.index())
                            }
                        };
                        match index {
                            // If we reach an end without seeing a > 2-node, there is no postdominator
                            None => {
                                for mark in mark_limits[current_mark as usize - 1..].iter_mut() {
                                    *mark = !0;
                                }
                            }
                            Some(1) => (),
                            Some(s) => state.forks.push_back((s, current_mark)),
                        }
                    }
                    1 => (),
                    other => {
                        if mark_limits[current_mark as usize - 1] < other {
                            mark_limits[current_mark as usize - 1] = other;
                        }
                    }
                }
            }
        }

        let mut mark_limits = match traverse_first_branch(&mut state, node) {
            Ok(o) => o,
            Err(e) => return e,
        };
        traverse_rest(&mut state, &mut mark_limits)
    }

    /// Converts conditions to resolved form (Relative to branch start).
    ///
    /// The actual conversion function
    /// fn x(condition: Rc<Operand>, start: VirtualAddress, end: VirtualAddress)
    /// has to be provided by caller.
    pub fn resolve_cond_jump_operands<F>(&mut self, mut convert: F)
    where F: FnMut(&Rc<Operand>, S::VirtualAddress, S::VirtualAddress) -> Rc<Operand>
    {
        for &mut (address, ref mut node) in &mut self.nodes {
            if let CfgOutEdges::Branch(_, ref mut cond) = node.out_edges {
                cond.condition = convert(&cond.condition, address, node.end_address);
            }
        }
    }
}

/// Link indices must be correct.
fn calculate_node_distance<S: CfgState>(
    nodes: &mut [(S::VirtualAddress, CfgNode<S>)],
    buf: &mut Vec<usize>,
    node_index: usize,
    distance: u32,
) {
    let old_dist = nodes[node_index].1.distance;
    if old_dist == 0 || old_dist > distance {
        let orig_len = buf.len();
        {
            let node = &mut nodes[node_index].1;
            node.distance = distance;
            buf.extend(
                node.out_edges.out_nodes()
                    .filter(|x| !x.is_unknown())
                    .map(|x| x.index()));
        }
        let end = buf.len();
        for pos in orig_len..end {
            let index = buf[pos];
            calculate_node_distance(nodes, buf, index, distance + 1);
        }
        buf.drain(orig_len..);
    }
}

pub struct CfgNodeIter<'a, S: CfgState>(::std::slice::Iter<'a, (S::VirtualAddress, CfgNode<S>)>, u32);
impl<'a, S: CfgState> Iterator for CfgNodeIter<'a, S> {
    type Item = NodeBorrow<'a, S>;
    fn next(&mut self) -> Option<Self::Item> {
        let index = NodeIndex(self.1, PhantomData);
        self.1 += 1;
        self.0.next().map(|x| NodeBorrow {
            address: x.0,
            node: &x.1,
            index,
        })
    }
}

#[derive(Debug, Clone)]
pub struct NodeBorrow<'a, State: CfgState> {
    pub node: &'a CfgNode<State>,
    pub index: NodeIndex<'a, State>,
    pub address: State::VirtualAddress,
}

#[derive(Debug, Clone, Copy)]
pub struct NodeIndex<'a, S: CfgState>(u32, PhantomData<&'a Cfg<S>>);

#[derive(Debug, Clone)]
pub struct CfgNode<State: CfgState> {
    pub out_edges: CfgOutEdges<State::VirtualAddress>,
    // The address is to the first instruction of in edge nodes instead of the jump-here address,
    // obviously so it can be looked up with Cfg::nodes.
    pub state: State,
    pub end_address: State::VirtualAddress,
    /// Distance from entry, first node has 1, its connected nodes have 2, etc.
    pub distance: u32,
}

#[derive(Debug, Clone)]
pub enum CfgOutEdges<Va: VirtualAddress> {
    None,
    Single(NodeLink<Va>),
    // The operand is unresolved at the state of jump.
    // Could reanalyse the block and show it resolved (effectively unresolved to start of the
    // block) when Cfg is publicly available.
    Branch(NodeLink<Va>, OutEdgeCondition<Va>),
    // The operand should be a direct, resolved index to the table
    Switch(Vec<NodeLink<Va>>, Rc<Operand>),
}

impl<Va: VirtualAddress> CfgOutEdges<Va> {
    pub fn default_address(&self) -> Option<Va> {
        match *self {
            CfgOutEdges::None => None,
            CfgOutEdges::Single(ref x) => Some(x.address),
            CfgOutEdges::Branch(ref x, _) => Some(x.address),
            CfgOutEdges::Switch(..) => None,
        }
    }

    pub fn out_nodes(&self) -> CfgOutEdgesNodes<Va> {
        CfgOutEdgesNodes(self, 0)
    }
}

pub struct CfgOutEdgesNodes<'a, Va: VirtualAddress>(&'a CfgOutEdges<Va>, usize);

impl<'a, Va: VirtualAddress> Iterator for CfgOutEdgesNodes<'a, Va> {
    type Item = &'a NodeLink<Va>;
    fn next(&mut self) -> Option<Self::Item> {
        let result = match *self.0 {
            CfgOutEdges::None => None,
            CfgOutEdges::Single(ref s) => match self.1 {
                0 => Some(s),
                _ => None,
            },
            CfgOutEdges::Branch(ref node, ref cond) => match self.1 {
                0 => Some(node),
                1 => Some(&cond.node),
                _ => None,
            },
            CfgOutEdges::Switch(ref cases, _) => cases.get(self.1),
        };
        self.1 += 1;
        result
    }
}

#[derive(Debug, Clone)]
pub struct OutEdgeCondition<Va: VirtualAddress> {
    pub node: NodeLink<Va>,
    pub condition: Rc<Operand>,
}

#[derive(Debug, Clone, Eq)]
/// The address is considered a stable link to a node, while index is invalid if the parent
/// graph's node_indices_dirty is true.
pub struct NodeLink<Va: VirtualAddress> {
    address: Va,
    index: u32,
}

impl<Va: VirtualAddress> Ord for NodeLink<Va> {
    fn cmp(&self, other: &Self) -> Ordering {
        let NodeLink {
            ref address,
            index: _,
        } = *self;
        address.cmp(&other.address)
    }
}

impl<Va: VirtualAddress> PartialOrd for NodeLink<Va> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<Va: VirtualAddress> PartialEq for NodeLink<Va> {
    fn eq(&self, other: &Self) -> bool {
        let NodeLink {
            ref address,
            index: _,
        } = *self;
        *address == other.address
    }
}

impl<Va: VirtualAddress> NodeLink<Va> {
    pub fn new(address: Va) -> NodeLink<Va> {
        NodeLink {
            address,
            index: !0,
        }
    }

    pub fn unknown() -> NodeLink<Va> {
        NodeLink {
            address: Va::max_value(),
            index: !0,
        }
    }

    pub fn is_unknown(&self) -> bool {
        *self == NodeLink::unknown()
    }

    pub fn address(&self) -> Va {
        self.address
    }

    pub fn index(&self) -> usize {
        self.index as usize
    }

    pub fn index_if_not_unkown(&self) -> Option<usize> {
        if self.is_unknown() {
            None
        } else {
            Some(self.index())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::VirtualAddress;

    struct EmptyState;
    impl CfgState for EmptyState {
        type VirtualAddress = VirtualAddress;
    }

    fn node0(addr: u32) -> CfgNode<EmptyState> {
        CfgNode {
            out_edges: CfgOutEdges::None,
            state: EmptyState,
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    fn node1(addr: u32, out: u32) -> CfgNode<EmptyState> {
        CfgNode {
            out_edges: CfgOutEdges::Single(NodeLink::new(VirtualAddress(out))),
            state: EmptyState,
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    fn node2(addr: u32, out: u32, out2: u32) -> CfgNode<EmptyState> {
        CfgNode {
            out_edges: CfgOutEdges::Branch(
                NodeLink::new(VirtualAddress(out)),
                OutEdgeCondition {
                    node: NodeLink::new(VirtualAddress(out2)),
                    condition: crate::operand_helpers::operand_register(0),
                },
            ),
            state: EmptyState,
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    #[test]
    fn distances() {
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 104));
        cfg.add_node(VirtualAddress(101), node2(101, 102, 104));
        cfg.add_node(VirtualAddress(102), node2(102, 103, 104));
        cfg.add_node(VirtualAddress(103), node1(103, 104));
        cfg.add_node(VirtualAddress(104), node0(104));
        cfg.calculate_distances();
        let mut iter = cfg.nodes().map(|x| x.node.distance);
        assert_eq!(iter.next().unwrap(), 1);
        assert_eq!(iter.next().unwrap(), 2);
        assert_eq!(iter.next().unwrap(), 3);
        assert_eq!(iter.next().unwrap(), 4);
        assert_eq!(iter.next().unwrap(), 2);
        assert!(iter.next().is_none());
    }

    #[test]
    fn cycles() {
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 103));
        cfg.add_node(VirtualAddress(101), node1(101, 102));
        cfg.add_node(VirtualAddress(102), node2(102, 103, 101));
        cfg.add_node(VirtualAddress(103), node1(103, 104));
        cfg.add_node(VirtualAddress(104), node2(104, 105, 108));
        cfg.add_node(VirtualAddress(105), node1(105, 106));
        cfg.add_node(VirtualAddress(106), node2(106, 104, 107));
        cfg.add_node(VirtualAddress(107), node2(107, 104, 108));
        cfg.add_node(VirtualAddress(108), node0(108));
        let mut cycles = cfg.cycles().into_iter()
            .map(|x| x.into_iter().map(|y| y.address().0).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        cycles.sort();
        assert_eq!(cycles[0], vec![101, 102]);
        assert_eq!(cycles[1], vec![104, 105, 106]);
        assert_eq!(cycles[2], vec![104, 105, 106, 107]);
        assert_eq!(cycles.len(), 3);
    }

    #[test]
    fn immediate_postdominator() {
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 104));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node2(103, 110, 106));
        cfg.add_node(VirtualAddress(104), node1(104, 107));
        cfg.add_node(VirtualAddress(105), node1(105, 107));
        cfg.add_node(VirtualAddress(106), node1(106, 102));
        cfg.add_node(VirtualAddress(107), node2(107, 108, 109));
        cfg.add_node(VirtualAddress(108), node0(108));
        cfg.add_node(VirtualAddress(109), node2(109, 111, 112));
        cfg.add_node(VirtualAddress(110), node2(110, 104, 105));
        cfg.add_node(VirtualAddress(111), node0(111));
        cfg.add_node(VirtualAddress(112), node1(112, 109));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 104);
        assert_eq!(cfg.immediate_postdominator(node(102)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 107);
        assert_eq!(cfg.immediate_postdominator(node(105)).unwrap().address.0, 107);
        assert!(cfg.immediate_postdominator(node(107)).is_none());
        assert!(cfg.immediate_postdominator(node(108)).is_none());
        assert_eq!(cfg.immediate_postdominator(node(109)).unwrap().address.0, 111);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 103));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node2(103, 104, 105));
        cfg.add_node(VirtualAddress(104), node1(104, 105));
        cfg.add_node(VirtualAddress(105), node0(105));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(102)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(103)).unwrap().address.0, 105);
        assert_eq!(cfg.immediate_postdominator(node(104)).unwrap().address.0, 105);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 103));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node2(103, 104, 101));
        cfg.add_node(VirtualAddress(104), node0(104));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(103)).unwrap().address.0, 104);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 100));
        cfg.add_node(VirtualAddress(102), node0(102));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 102);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 100);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 104));
        cfg.add_node(VirtualAddress(101), node2(101, 100, 103));
        cfg.add_node(VirtualAddress(102), node1(102, 101));
        cfg.add_node(VirtualAddress(103), node0(103));
        cfg.add_node(VirtualAddress(104), node1(104, 103));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 103);
    }

    #[test]
    fn immediate_postdominator2() {
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(100, 101, 102));
        cfg.add_node(VirtualAddress(101), node2(101, 103, 102));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node0(103));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 103);
    }
}
