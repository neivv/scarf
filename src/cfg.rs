use std::cmp::Ordering;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::mem;

use crate::exec_state::{VirtualAddress};
use crate::operand::{Operand};

#[derive(Debug, Clone)]
pub struct Cfg<'e, State: CfgState> {
    // Sorted
    nodes: Vec<(State::VirtualAddress, CfgNode<'e, State>)>,
    entry: NodeLink<State::VirtualAddress>,
    node_indices_dirty: bool,
}

pub trait CfgState {
    type VirtualAddress: VirtualAddress;
}

impl<'e, S: CfgState> Cfg<'e, S> {
    pub fn new() -> Cfg<'e, S> {
        Cfg {
            nodes: Vec::with_capacity(16),
            entry: NodeLink::new(S::VirtualAddress::max_value()),
            node_indices_dirty: false,
        }
    }

    fn mark_dirty(&mut self) {
        self.node_indices_dirty = true;
    }

    pub fn get(&self, address: S::VirtualAddress) -> Option<&CfgNode<'e, S>> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(&self.nodes[idx].1),
            _ => None,
        }
    }

    pub fn get_link(&self, address: S::VirtualAddress) -> Option<NodeLink<S::VirtualAddress>> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(NodeLink {
                address,
                index: idx as u32,
            }),
            _ => None,
        }
    }

    /// Gets mutable access to state - which is fine as none of the
    /// CFG functionality uses it for anything; it's purely user data.
    ///
    /// Shared access can use `cfg.get().map(|x| &x.state)`
    pub fn get_state(&mut self, address: S::VirtualAddress) -> Option<&mut S> {
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => Some(&mut self.nodes[idx].1.state),
            _ => None,
        }
    }

    pub fn entry_link(&self) -> &NodeLink<S::VirtualAddress> {
        &self.entry
    }

    pub fn entry(&self) -> &CfgNode<'e, S> {
        self.get_by_link(&self.entry)
    }

    pub fn get_by_link(&self, link: &NodeLink<S::VirtualAddress>) -> &CfgNode<'e, S> {
        &self.nodes[link.index as usize].1
    }

    pub fn add_node(&mut self, address: S::VirtualAddress, node: CfgNode<'e, S>) {
        if self.nodes.is_empty() {
            self.entry = NodeLink::new(address);
        }
        match self.nodes.binary_search_by_key(&address, |x| x.0) {
            Ok(idx) => self.nodes[idx] = (address, node),
            Err(idx) => self.nodes.insert(idx, (address, node)),
        }
        self.mark_dirty();
    }

    pub fn nodes<'a>(&'a self) -> CfgNodeIter<'a, 'e, S> {
        CfgNodeIter(self.nodes.iter(), 0)
    }

    /// Tries to replace a block with a jump to another one which starts in middle of current.
    pub fn merge_overlapping_blocks(&mut self) {
        if self.nodes.is_empty() {
            return;
        }
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

    /// Retreives a structure which can be used to lookup what nodes branch to current
    /// node.
    pub fn predecessors(&mut self) -> Predecessors {
        self.calculate_node_indices();
        let mut result = Predecessors {
            lookup: vec![(!0, !0); self.nodes.len()],
            long_lists: Vec::new(),
        };
        for (idx, node) in self.nodes.iter().enumerate().map(|x| (x.0 as u32, &(x.1).1)) {
            match node.out_edges {
                CfgOutEdges::None => (),
                CfgOutEdges::Single(ref link) => result.add_predecessors(idx, &[*link]),
                CfgOutEdges::Branch(ref default, ref cond) => {
                    result.add_predecessors(idx, &[*default, cond.node])
                }
                CfgOutEdges::Switch(ref links, _) => result.add_predecessors(idx, &links),
            }
        }

        result
    }

    pub fn immediate_postdominator<'a>(
        &'a self,
        node: NodeIndex<'a, 'e, S>,
    ) -> Option<NodeBorrow<'a, 'e, S>> {
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

        struct State<'a, 'e, S: CfgState> {
            mark_buf: Vec<u32>,
            forks: VecDeque<(usize, u32)>,
            mark_indices: Vec<usize>,
            nodes: &'a [(S::VirtualAddress, CfgNode<'e, S>)],
        }
        let mut state = State {
            mark_buf: vec![0u32; self.nodes.len()],
            forks: VecDeque::with_capacity(self.nodes.len()),
            mark_indices: Vec::with_capacity(32),
            nodes: &self.nodes,
        };

        // Return Err on quick exit, Ok for continuing to rest of branches
        fn traverse_first_branch<'a, 'e, S: CfgState>(
            state: &mut State<'a, 'e, S>,
            node: NodeIndex<'a, 'e, S>,
        ) -> Result<Vec<u32>, Option<NodeBorrow<'a, 'e, S>>> {
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

        fn traverse_rest<'a, 'e, S: CfgState>(
            state: &mut State<'a, 'e, S>,
            mark_limits: &mut [u32],
        ) -> Option<NodeBorrow<'a, 'e, S>> {
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
    /// fn x(condition: Operand, start: VirtualAddress, end: VirtualAddress)
    /// has to be provided by caller.
    pub fn resolve_cond_jump_operands<F>(&mut self, mut convert: F)
    where F: FnMut(Operand<'e>, S::VirtualAddress, S::VirtualAddress) -> Operand<'e>
    {
        for &mut (address, ref mut node) in &mut self.nodes {
            if let CfgOutEdges::Branch(_, ref mut cond) = node.out_edges {
                cond.condition = convert(cond.condition, address, node.end_address);
            }
        }
    }
}

/// Link indices must be correct.
fn calculate_node_distance<'e, S: CfgState>(
    nodes: &mut [(S::VirtualAddress, CfgNode<'e, S>)],
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

pub struct CfgNodeIter<'a, 'e, S: CfgState>(std::slice::Iter<'a, (S::VirtualAddress, CfgNode<'e, S>)>, u32);
impl<'a, 'e, S: CfgState> Iterator for CfgNodeIter<'a, 'e, S> {
    type Item = NodeBorrow<'a, 'e, S>;
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
pub struct NodeBorrow<'a, 'e, State: CfgState> {
    pub node: &'a CfgNode<'e, State>,
    pub index: NodeIndex<'a, 'e, State>,
    pub address: State::VirtualAddress,
}

#[derive(Debug, Clone, Copy)]
pub struct NodeIndex<'a, 'e, S: CfgState>(u32, PhantomData<&'a Cfg<'e, S>>);

#[derive(Debug, Clone)]
pub struct CfgNode<'e, State: CfgState> {
    pub out_edges: CfgOutEdges<'e, State::VirtualAddress>,
    // The address is to the first instruction of in edge nodes instead of the jump-here address,
    // obviously so it can be looked up with Cfg::nodes.
    pub state: State,
    pub end_address: State::VirtualAddress,
    /// Distance from entry, first node has 1, its connected nodes have 2, etc.
    pub distance: u32,
}

#[derive(Debug, Clone)]
pub enum CfgOutEdges<'e, Va: VirtualAddress> {
    None,
    Single(NodeLink<Va>),
    // The operand is unresolved at the state of jump.
    // Could reanalyse the block and show it resolved (effectively unresolved to start of the
    // block) when Cfg is publicly available.
    Branch(NodeLink<Va>, OutEdgeCondition<'e, Va>),
    // The operand should be a direct, resolved index to the table
    Switch(Vec<NodeLink<Va>>, Operand<'e>),
}

impl<'e, Va: VirtualAddress> CfgOutEdges<'e, Va> {
    pub fn default_address(&self) -> Option<Va> {
        match *self {
            CfgOutEdges::None => None,
            CfgOutEdges::Single(ref x) => Some(x.address),
            CfgOutEdges::Branch(ref x, _) => Some(x.address),
            CfgOutEdges::Switch(..) => None,
        }
    }

    pub fn out_nodes<'a>(&'a self) -> CfgOutEdgesNodes<'a, 'e, Va> {
        CfgOutEdgesNodes(self, 0)
    }
}

pub struct CfgOutEdgesNodes<'a, 'e, Va: VirtualAddress>(&'a CfgOutEdges<'e, Va>, usize);

impl<'a, 'e, Va: VirtualAddress> Iterator for CfgOutEdgesNodes<'a, 'e, Va> {
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
pub struct OutEdgeCondition<'e, Va: VirtualAddress> {
    pub node: NodeLink<Va>,
    pub condition: Operand<'e>,
}

#[derive(Debug, Clone, Copy, Eq)]
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

pub struct Predecessors {
    // Node index to predecessors.
    //   No predecessors => (!0, !0),
    //   1 => (idx, !0),
    //   2 => (idx, idx),
    //   3+ => (!0, idx to long_lists)
    lookup: Vec<(u32, u32)>,
    // Storage when there are more than 2 predecessors.
    // First idx is current sublist length, followed by predecessors of that length,
    // followed by !0 to signify end of list or another index to next sublist.
    long_lists: Vec<u32>,
}

impl Predecessors {
    fn add_predecessors<Va: VirtualAddress>(
        &mut self,
        predecessor: u32,
        successors: &[NodeLink<Va>],
    ) {
        for succ in successors {
            let entry = &mut self.lookup[succ.index as usize];
            if entry.1 == u32::max_value() {
                // 0 or 1 entries, can just add to the current list
                if entry.0 == u32::max_value() {
                    entry.0 = predecessor;
                } else {
                    entry.1 = predecessor;
                }
            } else if entry.0 != u32::max_value() {
                // 2 entries, relocating to long_lists
                let long_list_idx = self.long_lists.len() as u32;
                self.long_lists.extend([
                    3,
                    entry.0,
                    entry.1,
                    predecessor,
                    !0,
                ].iter().copied());
                entry.0 = u32::max_value();
                entry.1 = long_list_idx;
            } else {
                // Already a long list
                // First seek to last sublist index
                let mut tail_index = entry.1 as usize;
                let mut next_location;
                loop {
                    let sublist_len = self.long_lists[tail_index];
                    next_location = tail_index + sublist_len as usize + 1;
                    let next = self.long_lists[next_location];
                    if next == u32::max_value() {
                        break;
                    }
                    tail_index = next as usize;
                }
                if next_location == self.long_lists.len() - 1 {
                    // Can just expand the list in place
                    self.long_lists[tail_index] += 1;
                    self.long_lists[next_location] = predecessor;
                    self.long_lists.push(u32::max_value());
                } else {
                    // Start a new sublist
                    let new_sublist_idx = self.long_lists.len();
                    self.long_lists[next_location] = new_sublist_idx as u32;
                    self.long_lists.extend([
                        1,
                        predecessor,
                        !0,
                    ].iter().copied());
                }
            }

        }
    }

    /// Iterates through predecessors of a single node.
    /// `cfg` must be the same Cfg that was used to create this struct.
    pub fn predecessors<'a, 'e, S: CfgState>(
        &'a self,
        cfg: &'a Cfg<'e, S>,
        link: &NodeLink<S::VirtualAddress>,
    ) -> NodePredecessors<'a, 'e, S> {
        let index = link.index as usize;
        let predecessors = self.lookup[index];
        let mode = if predecessors.0 == u32::max_value() && predecessors.1 != u32::max_value() {
            PredecessorIterMode::Long(predecessors.1, 0)
        } else {
            PredecessorIterMode::Short(predecessors.0, predecessors.1)
        };

        NodePredecessors {
            cfg,
            parent: self,
            mode,
        }
    }
}

pub struct NodePredecessors<'a, 'e, S: CfgState> {
    cfg: &'a Cfg<'e, S>,
    parent: &'a Predecessors,
    mode: PredecessorIterMode,
}

#[derive(Copy, Clone)]
enum PredecessorIterMode {
    Long(u32, u32),
    Short(u32, u32),
}

impl<'a, 'e, S: CfgState> Iterator for NodePredecessors<'a, 'e, S> {
    type Item = NodeLink<S::VirtualAddress>;
    fn next(&mut self) -> Option<Self::Item> {
        let next_index = match self.mode {
            PredecessorIterMode::Short(ref mut a, ref mut b) => {
                if *a == u32::max_value() {
                    return None;
                }
                let next_index = *a;
                *a = *b;
                *b = u32::max_value();
                next_index
            }
            PredecessorIterMode::Long(ref mut sublist_index, ref mut pos) => {
                if *sublist_index == u32::max_value() {
                    return None;
                }
                let sublist_len = self.parent.long_lists[*sublist_index as usize] as usize;
                if *pos as usize == sublist_len {
                    *sublist_index =
                        self.parent.long_lists[*sublist_index as usize + sublist_len + 1];
                    *pos = 0;
                }
                if *sublist_index == u32::max_value() {
                    return None;
                }
                *pos += 1;
                self.parent.long_lists[*sublist_index as usize + *pos as usize]
            }
        };
        let address = self.cfg.nodes[next_index as usize].0;
        Some(NodeLink {
            address,
            index: next_index,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{OperandContext, VirtualAddress};

    struct EmptyState;
    impl CfgState for EmptyState {
        type VirtualAddress = VirtualAddress;
    }

    fn node0<'e>(addr: u32) -> CfgNode<'e, EmptyState> {
        CfgNode {
            out_edges: CfgOutEdges::None,
            state: EmptyState,
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    fn node1<'e>(addr: u32, out: u32) -> CfgNode<'e, EmptyState> {
        CfgNode {
            out_edges: CfgOutEdges::Single(NodeLink::new(VirtualAddress(out))),
            state: EmptyState,
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    fn node2<'e>(
        ctx: &'e OperandContext<'e>,
        addr: u32,
        out: u32,
        out2: u32,
    ) -> CfgNode<'e, EmptyState> {
        CfgNode {
            out_edges: CfgOutEdges::Branch(
                NodeLink::new(VirtualAddress(out)),
                OutEdgeCondition {
                    node: NodeLink::new(VirtualAddress(out2)),
                    condition: ctx.register(0),
                },
            ),
            state: EmptyState,
            end_address: VirtualAddress(addr + 1),
            distance: 0,
        }
    }

    #[test]
    fn distances() {
        let ctx = &OperandContext::new();
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 104));
        cfg.add_node(VirtualAddress(101), node2(ctx, 101, 102, 104));
        cfg.add_node(VirtualAddress(102), node2(ctx, 102, 103, 104));
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
        let ctx = &OperandContext::new();
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 103));
        cfg.add_node(VirtualAddress(101), node1(101, 102));
        cfg.add_node(VirtualAddress(102), node2(ctx, 102, 103, 101));
        cfg.add_node(VirtualAddress(103), node1(103, 104));
        cfg.add_node(VirtualAddress(104), node2(ctx, 104, 105, 108));
        cfg.add_node(VirtualAddress(105), node1(105, 106));
        cfg.add_node(VirtualAddress(106), node2(ctx, 106, 104, 107));
        cfg.add_node(VirtualAddress(107), node2(ctx, 107, 104, 108));
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
        let ctx = &OperandContext::new();
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 104));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node2(ctx, 103, 110, 106));
        cfg.add_node(VirtualAddress(104), node1(104, 107));
        cfg.add_node(VirtualAddress(105), node1(105, 107));
        cfg.add_node(VirtualAddress(106), node1(106, 102));
        cfg.add_node(VirtualAddress(107), node2(ctx, 107, 108, 109));
        cfg.add_node(VirtualAddress(108), node0(108));
        cfg.add_node(VirtualAddress(109), node2(ctx, 109, 111, 112));
        cfg.add_node(VirtualAddress(110), node2(ctx, 110, 104, 105));
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
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 103));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node2(ctx, 103, 104, 105));
        cfg.add_node(VirtualAddress(104), node1(104, 105));
        cfg.add_node(VirtualAddress(105), node0(105));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(102)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(103)).unwrap().address.0, 105);
        assert_eq!(cfg.immediate_postdominator(node(104)).unwrap().address.0, 105);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 103));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node2(ctx, 103, 104, 101));
        cfg.add_node(VirtualAddress(104), node0(104));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(103)).unwrap().address.0, 104);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 102));
        cfg.add_node(VirtualAddress(101), node1(101, 100));
        cfg.add_node(VirtualAddress(102), node0(102));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 102);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 100);

        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 104));
        cfg.add_node(VirtualAddress(101), node2(ctx, 101, 100, 103));
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
        let ctx = &OperandContext::new();
        let mut cfg = Cfg::new();
        cfg.add_node(VirtualAddress(100), node2(ctx, 100, 101, 102));
        cfg.add_node(VirtualAddress(101), node2(ctx, 101, 103, 102));
        cfg.add_node(VirtualAddress(102), node1(102, 103));
        cfg.add_node(VirtualAddress(103), node0(103));
        cfg.calculate_node_indices();
        let node = |i| cfg.nodes().find(|x| x.address.0 == i).unwrap().index;
        assert_eq!(cfg.immediate_postdominator(node(100)).unwrap().address.0, 103);
        assert_eq!(cfg.immediate_postdominator(node(101)).unwrap().address.0, 103);
    }
}
