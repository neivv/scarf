use super::{ArithOperand, ArithOpType, Operand, OperandCtx, OperandType};
use super::slice_stack;

type Slice<'e> = slice_stack::Slice<'e, Operand<'e>>;

/// Iterates through parts of a arithmetic op tree where it is guaranteed
/// that right has never subtrees, only left (E.g. for and, or, xor).
///
/// Order is outermost right first, then next inner right, etc.
#[derive(Copy, Clone)]
pub struct IterArithOps<'e> {
    pub ty: ArithOpType,
    pub next: Option<Operand<'e>>,
    pub next_inner: Option<Operand<'e>>,
}

impl<'e> IterArithOps<'e> {
    pub fn new(op: Operand<'e>, ty: ArithOpType) -> IterArithOps<'e> {
        match op.if_arithmetic(ty) {
            Some((l, r)) => IterArithOps {
                ty,
                next: Some(r),
                next_inner: Some(l),
            },
            None => IterArithOps {
                ty,
                next: Some(op),
                next_inner: None,
            },
        }
    }

    #[inline]
    pub fn new_pair(left: Operand<'e>, right: Operand<'e>, ty: ArithOpType) -> IterArithOps<'e> {
        IterArithOps {
            ty,
            next: Some(right),
            next_inner: Some(left),
        }
    }

    #[inline]
    pub fn new_arith(arith: &ArithOperand<'e>) -> IterArithOps<'e> {
        IterArithOps {
            ty: arith.ty,
            next: Some(arith.right),
            next_inner: Some(arith.left),
        }
    }
}

impl<'e> Iterator for IterArithOps<'e> {
    type Item = Operand<'e>;
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next?;
        if let Some(x) = self.next_inner {
            if let Some((inner_a, inner_b)) = x.if_arithmetic(self.ty) {
                self.next_inner = Some(inner_a);
                self.next = Some(inner_b);
            } else {
                self.next = Some(x);
                self.next_inner = None;
            }
        } else {
            self.next = None;
        }
        Some(next)
    }
}

#[derive(Copy, Clone)]
pub struct IterAddSubArithOps<'e> {
    next: Option<Operand<'e>>,
}

impl<'e> IterAddSubArithOps<'e> {
    pub fn new(operand: Operand<'e>) -> IterAddSubArithOps {
        IterAddSubArithOps {
            next: Some(operand),
        }
    }
}

impl<'e> Iterator for IterAddSubArithOps<'e> {
    /// (op, bool negate)
    type Item = (Operand<'e>, bool);
    fn next(&mut self) -> Option<Self::Item> {
        let next = self.next?;
        if let OperandType::Arithmetic(a) = next.ty() {
            if a.ty == ArithOpType::Add {
                self.next = Some(a.left);
                return Some((a.right, false));
            } else if a.ty == ArithOpType::Sub {
                self.next = Some(a.left);
                return Some((a.right, true));
            }
        }
        self.next = None;
        Some((next, false))
    }
}

/// Does ctx.simplify_temp_stack() allocation and
/// fills it with operand chain (l, r, ty)
pub fn arith_parts_to_new_slice<'e, F, T>(
    ctx: OperandCtx<'e>,
    l: Operand<'e>,
    r: Operand<'e>,
    ty: ArithOpType,
    cb: F,
) -> Option<T>
where F: FnOnce(&mut Slice<'e>) -> Option<T>
{
    ctx.simplify_temp_stack().alloc(|slice| {
        for op in IterArithOps::new_pair(l, r, ty) {
            slice.push(op).ok()?;
        }
        cb(slice)
    })
}

/// If `(l, r, ty)` operand chain is fully included in `slice`,
/// returns rejoined `Operand` of the parts not in `(l, r, ty)`.
/// If the two are equal, returns ctx.const_0()
///     ! 0 may not be identity value for the operation (e.g. and, mul)
///     caller should check for 0 and replace it with id.
///
/// Assumes `slice` to be sorted and contains already simplified arith chain
///     (So that the operation is O(n)).
/// `slice` is not modified.
pub fn remove_eq_arith_parts_sorted<'e>(
    ctx: OperandCtx<'e>,
    slice: &Slice<'e>,
    l: Operand<'e>,
    r: Operand<'e>,
    ty: ArithOpType,
) -> Option<Operand<'e>> {
    let mut iter = IterArithOps::new_pair(l, r, ty);
    let mut slice_pos = &slice[..];
    ctx.simplify_temp_stack().alloc(|result| {
        'outer: while let Some(next) = iter.next() {
            loop {
                match slice_pos.split_first() {
                    Some((&s, rest)) => {
                        slice_pos = rest;
                        if s == next {
                            continue 'outer;
                        } else {
                            result.push(s).ok()?;
                        }
                    }
                    None => return None,
                }
            }
        }
        for &op in slice_pos {
            result.push(op).ok()?;
        }

        // All ops were in slice, join result
        // assumes that the func input was sorted / simplified arith.
        let result = intern_arith_ops_to_tree(ctx, result.iter().copied().rev(), ty)
            .unwrap_or_else(|| ctx.const_0());
        Some(result)
    })
}

/// Removes one operand from slice and joins it back.
/// `orig` is the Operand which was read to `slice`;
/// it will be used to avoid some reinterning as the
/// parts after remove_pos don't need to be changed.
///
/// Assumes `slice` to be sorted and contains already simplified arith chain of `orig`.
/// Does not actually modify the slice.
pub fn sorted_arith_chain_remove_one_and_join<'e>(
    ctx: OperandCtx<'e>,
    slice: &Slice<'e>,
    idx: usize,
    orig: Operand<'e>,
    ty: ArithOpType,
) -> Operand<'e> {
    assert!(idx < slice.len());
    if idx == 0 || (idx == 1 && slice.len() == 2) {
        if let Some(arith) = orig.if_arithmetic_any() {
            if idx == 0 {
                return arith.left;
            } else {
                return arith.right;
            }
        }
    }
    // If removing idx 0, keep orig.left
    // idx 1, orig.left.left
    // idx 2, orig.left.left.left
    // etc
    // So (idx + 1) ends up also being how many times x.left has to be followed.
    let mut pos = orig;
    for _ in 0..(idx + 1) {
        pos = match pos.if_arithmetic_any() {
            Some(arith) => {
                debug_assert_eq!(arith.ty, ty);
                arith.left
            }
            None => {
                debug_assert!(false);
                continue;
            }
        }
    }
    for &part in slice[..idx].iter().rev() {
        let arith = ArithOperand {
            ty,
            left: pos,
            right: part,
        };
        pos = ctx.intern(OperandType::Arithmetic(arith));
    }
    pos
}

/// Use only if the iterator produces items in sorted order.
/// (Meaning slice.rev() of a sorted slice)
pub fn intern_arith_ops_to_tree<'e, I: Iterator<Item = Operand<'e>>>(
    ctx: OperandCtx<'e>,
    mut iter: I,
    ty: ArithOpType,
) -> Option<Operand<'e>> {
    let mut tree = iter.next()?;
    for op in iter {
        let arith = ArithOperand {
            ty,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    Some(tree)
}
