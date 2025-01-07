use std::cmp::Ordering;

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;
use crate::analysis::{Analyzer, Control};
use crate::exec_state::{ExecutionState, VirtualAddress};

/// Allows adding "comments", which are strings assigned to address.
/// Comments will be reset if scarf reruns the branch,
/// (Scarf operands on later run should be more accurate)
/// and in case two different branch starts execute the same instructions, requires
/// both branches to add same comments.
///
/// Call branch_start() at Analyzer::branch_start,
/// branch_end() at Analyzer::branch_end, and
/// add() from Analyzer::operation when actually adding comments.
pub struct QueuedComments<'b, Va: VirtualAddress> {
    /// (branch_start, branch_end, comments)
    /// Sorted by start address, comments are also sorted by address.
    branch_comments: BumpVec<'b, (Va, Va, BumpVec<'b, Comment<Va>>)>,
    current_index: usize,
    bump: &'b Bump,
}

struct Comment<Va: VirtualAddress> {
    address: Va,
    text: String,
}

impl<'b, Va: VirtualAddress> QueuedComments<'b, Va> {
    pub fn new(bump: &'b Bump) -> QueuedComments<'b, Va> {
        QueuedComments {
            branch_comments: BumpVec::with_capacity_in(0x20, bump),
            current_index: usize::MAX,
            bump,
        }
    }

    pub fn branch_start<'e, A: Analyzer<'e, Exec = E>, E: ExecutionState<'e, VirtualAddress = Va>>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
    ) {
        self.branch_start_(ctrl.address());
    }

    fn branch_start_(&mut self, address: Va) {
        let index = self.branch_comments.binary_search_by_key(&address, |x| x.0);
        match index {
            Ok(index) => {
                self.branch_comments[index].2.clear();
                self.current_index = index;
            }
            Err(index) => {
                let vec = BumpVec::new_in(self.bump);
                self.branch_comments.insert(index, (address, address, vec));
                self.current_index = index;
            }
        }
    }

    pub fn branch_end<'e, A: Analyzer<'e, Exec = E>, E: ExecutionState<'e, VirtualAddress = Va>>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
    ) {
        self.branch_end_(ctrl.address());
    }

    fn branch_end_(&mut self, address: Va) {
        self.branch_comments[self.current_index].1 = address;
    }

    pub fn add<'e, A: Analyzer<'e, Exec = E>, E: ExecutionState<'e, VirtualAddress = Va>>(
        &mut self,
        ctrl: &mut Control<'e, '_, '_, A>,
        comment: String,
    ) {
        self.add_to_address(ctrl.address(), comment);
    }

    pub fn add_to_address(&mut self, address: Va, comment: String) {
        let comments = &mut self.branch_comments[self.current_index].2;
        if let Some(last) = comments.last_mut() {
            if last.address == address {
                last.text.push_str(", ");
                last.text.push_str(&comment);
                return;
            } else {
                if last.address < address {
                    comments.push(Comment {
                        address,
                        text: comment,
                    });
                } else {
                    match comments.binary_search_by_key(&address, |x| x.address) {
                        Ok(index) => {
                            let text = &mut comments[index].text;
                            text.push_str(", ");
                            text.push_str(&comment);
                        }
                        Err(index) => {
                            comments.insert(index, Comment {
                                address,
                                text: comment,
                            });
                        }
                    }
                }
                return;
            }
        } else {
            comments.reserve(8);
            comments.push(Comment {
                address,
                text: comment,
            });
        }
    }

    pub fn finish<F: FnMut(Va, String)>(mut self, mut add_comment: F) {
        let mut i = 0;
        let len = self.branch_comments.len();
        // Remove non-matching comments for branches which overlap
        // Specifically since the assumption is that branches are
        // sorted by start address, and can only overlap as
        // (start, end), (start + n, end) with end being same and
        // start different, after verifying that the comments match,
        // the shorter branch's comments are all cleared to skip
        // the duplicates.
        loop {
            if i + 1 > len {
                break;
            }
            let (a, rest) = self.branch_comments.split_at_mut(i + 1);
            let &mut (addr, end, ref mut comments1) = match a.last_mut() {
                Some(s) => s,
                None => break,
            };
            let mut j = 0;
            loop {
                let &mut (addr2, end2, ref mut comments2) = match rest.get_mut(j) {
                    Some(s) => s,
                    None => break,
                };
                if end != end2 {
                    break;
                }
                assert!(addr != addr2);
                // Set ic to first index of `comments1` which is part of `addr2` `end2` range
                let mut ic = 0;
                while ic < comments1.len() {
                    if comments1[ic].address >= addr2 {
                        break;
                    }
                    ic += 1;
                }
                // Walk through comments1[ic..] and comments2 and remove values which don't match
                let mut jc = 0;
                loop {
                    let comment1 = match comments1.get_mut(ic) {
                        Some(s) => s,
                        None => {
                            // comments2 may have more comments that won't be included,
                            // but it is going to be fully cleared anyway after breaking
                            // out of the loop.
                            break;
                        }
                    };
                    let comment2 = match comments2.get_mut(jc) {
                        Some(s) => s,
                        None => {
                            // If comments1 has more comments left, they didn't match
                            // comments2 and will be removed
                            comments1.truncate(ic);
                            break;
                        }
                    };
                    match comment1.address.cmp(&comment2.address) {
                        Ordering::Less => {
                            comments1.remove(ic);
                        }
                        Ordering::Greater => {
                            // Can leave things on comments2 since it'll be cleared in the end.
                            jc += 1;
                        }
                        Ordering::Equal => {
                            if comment1.text != comment2.text {
                                comments1.remove(ic);
                            } else {
                                ic += 1;
                            }
                            jc += 1;
                        }
                    }
                }
                comments2.clear();

                j += 1;
            }
            for comment in comments1.drain(..) {
                add_comment(comment.address, comment.text);
            }
            i += j + 1;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::VirtualAddress64;

    #[test]
    fn simple() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(100));
        q.branch_end_(VirtualAddress64(108));
        q.branch_start_(VirtualAddress64(108));
        q.add_to_address(VirtualAddress64(108), "Test1".into());
        q.add_to_address(VirtualAddress64(109), "Test2".into());
        q.branch_end_(VirtualAddress64(110));
        q.branch_start_(VirtualAddress64(110));
        q.branch_end_(VirtualAddress64(120));
        q.branch_start_(VirtualAddress64(120));
        q.add_to_address(VirtualAddress64(120), "Test3".into());
        q.branch_end_(VirtualAddress64(130));
        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
                (108, "Test1".into()),
                (109, "Test2".into()),
                (120, "Test3".into()),
            ],
        );
    }

    #[test]
    fn multiple_runs() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(100));
        q.branch_end_(VirtualAddress64(108));
        q.branch_start_(VirtualAddress64(108));
        q.add_to_address(VirtualAddress64(108), "Test1".into());
        q.add_to_address(VirtualAddress64(109), "Test2".into());
        q.branch_end_(VirtualAddress64(110));
        q.branch_start_(VirtualAddress64(110));
        q.branch_end_(VirtualAddress64(120));
        q.branch_start_(VirtualAddress64(110));
        q.branch_end_(VirtualAddress64(120));
        q.branch_start_(VirtualAddress64(120));
        q.add_to_address(VirtualAddress64(120), "Test3".into());
        q.branch_end_(VirtualAddress64(130));
        q.branch_start_(VirtualAddress64(120));
        q.add_to_address(VirtualAddress64(120), "Test7".into());
        q.branch_end_(VirtualAddress64(130));
        q.branch_start_(VirtualAddress64(108));
        q.add_to_address(VirtualAddress64(108), "Test8".into());
        q.branch_end_(VirtualAddress64(110));
        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
                (108, "Test8".into()),
                (120, "Test7".into()),
            ],
        );
    }

    #[test]
    fn overlapping_branches1() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(100));
        q.branch_end_(VirtualAddress64(108));
        q.branch_start_(VirtualAddress64(108));
        q.add_to_address(VirtualAddress64(108), "Test1".into());
        q.add_to_address(VirtualAddress64(110), "Test2".into());
        q.add_to_address(VirtualAddress64(114), "Test3".into());
        q.add_to_address(VirtualAddress64(115), "Test4".into());
        q.branch_end_(VirtualAddress64(120));
        q.branch_start_(VirtualAddress64(111));
        q.add_to_address(VirtualAddress64(114), "Test3".into());
        q.add_to_address(VirtualAddress64(115), "Test4".into());
        q.branch_end_(VirtualAddress64(120));

        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
                (108, "Test1".into()),
                (110, "Test2".into()),
                (114, "Test3".into()),
                (115, "Test4".into()),
            ],
        );
    }

    #[test]
    fn overlapping_branches2() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(100));
        q.branch_end_(VirtualAddress64(108));
        q.branch_start_(VirtualAddress64(108));
        q.add_to_address(VirtualAddress64(108), "Test1".into());
        q.add_to_address(VirtualAddress64(110), "Test2".into());
        q.add_to_address(VirtualAddress64(114), "Test3".into());
        q.add_to_address(VirtualAddress64(115), "Test4".into());
        q.branch_end_(VirtualAddress64(120));
        q.branch_start_(VirtualAddress64(110));
        q.add_to_address(VirtualAddress64(114), "Test3".into());
        q.add_to_address(VirtualAddress64(115), "Test4".into());
        q.branch_end_(VirtualAddress64(120));

        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
                (108, "Test1".into()),
                (114, "Test3".into()),
                (115, "Test4".into()),
            ],
        );
    }

    #[test]
    fn overlapping_branches3() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(208));
        q.add_to_address(VirtualAddress64(208), "A1".into());
        q.add_to_address(VirtualAddress64(210), "A2".into());
        q.add_to_address(VirtualAddress64(214), "A3".into());
        q.add_to_address(VirtualAddress64(215), "A4".into());
        q.branch_end_(VirtualAddress64(220));
        q.branch_start_(VirtualAddress64(211));
        q.add_to_address(VirtualAddress64(214), "A3".into());
        q.add_to_address(VirtualAddress64(215), "A4".into());
        q.branch_end_(VirtualAddress64(220));
        q.branch_start_(VirtualAddress64(214));
        q.add_to_address(VirtualAddress64(214), "A3".into());
        q.branch_end_(VirtualAddress64(220));

        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
                (208, "A1".into()),
                (210, "A2".into()),
                (214, "A3".into()),
            ],
        );
    }

    #[test]
    fn overlapping_branches4() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(308));
        q.add_to_address(VirtualAddress64(308), "B1".into());
        q.add_to_address(VirtualAddress64(310), "B2".into());
        q.add_to_address(VirtualAddress64(314), "B3".into());
        q.add_to_address(VirtualAddress64(315), "B4".into());
        q.branch_end_(VirtualAddress64(320));
        q.branch_start_(VirtualAddress64(309));
        q.add_to_address(VirtualAddress64(314), "B_3".into());
        q.branch_end_(VirtualAddress64(320));
        q.branch_start_(VirtualAddress64(308));
        q.add_to_address(VirtualAddress64(314), "B3".into());
        q.branch_end_(VirtualAddress64(320));

        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
            ],
        );
    }

    #[test]
    fn add_to_earlier_address() {
        let bump = &Bump::new();
        let mut q = QueuedComments::new(bump);
        let mut out = Vec::new();
        q.branch_start_(VirtualAddress64(108));
        q.add_to_address(VirtualAddress64(108), "Test1".into());
        q.add_to_address(VirtualAddress64(116), "Test3".into());
        q.add_to_address(VirtualAddress64(110), "Test2".into());
        q.finish(|a, text| out.push((a.as_u64(), text)));
        out.sort_unstable();
        assert_eq!(
            out,
            vec![
                (108, "Test1".into()),
                (110, "Test2".into()),
                (116, "Test3".into()),
            ],
        );
    }
}
