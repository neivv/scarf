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

#[derive(Copy, Clone)]
pub struct IterArithOps1<'e> {
    pub ty: ArithOpType,
    pub rest: Option<Operand<'e>>,
}

impl<'e> IterArithOps<'e> {
    #[inline]
    pub fn new(op: Operand<'e>, ty: ArithOpType) -> IterArithOps1<'e> {
        IterArithOps1 {
            ty,
            rest: Some(op),
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

    #[inline]
    pub fn peek_next(&self) -> Option<Operand<'e>> {
        self.next
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

impl<'e> IterArithOps1<'e> {
    #[inline]
    pub fn rest(&self) -> Option<Operand<'e>> {
        self.rest
    }

    pub fn peek_next(&self) -> Option<Operand<'e>> {
        let rest = self.rest?;
        if let Some((_, inner_b)) = rest.if_arithmetic(self.ty) {
            Some(inner_b)
        } else {
            Some(rest)
        }
    }
}

impl<'e> Iterator for IterArithOps1<'e> {
    type Item = Operand<'e>;
    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.rest?;
        if let Some((inner_a, inner_b)) = rest.if_arithmetic(self.ty) {
            self.rest = Some(inner_a);
            Some(inner_b)
        } else {
            self.rest = None;
            Some(rest)
        }
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

pub fn arith_op_parts_to_new_slice<'e, F, T>(
    ctx: OperandCtx<'e>,
    op: Operand<'e>,
    ty: ArithOpType,
    cb: F,
) -> Option<T>
where F: FnOnce(&mut Slice<'e>) -> Option<T>
{
    ctx.simplify_temp_stack().alloc(|slice| {
        for op in IterArithOps::new(op, ty) {
            slice.push(op).ok()?;
        }
        cb(slice)
    })
}

/// If `(l, r, ty)` operand chain is fully included in `slice`,
/// calls `callback` with parts not in `(l, r, ty)`, and returns
/// `Some(callback_return_value)`.
/// If `(l, r, ty)` is not included in `slice`, returns `None`.
///
/// Assumes `slice` to be sorted and contains already simplified arith chain
///     (So that the operation is O(n)).
/// `slice` is not modified.
pub fn remove_eq_arith_parts_sorted<'e, F: FnOnce(&mut Slice<'e>) -> T, T>(
    ctx: OperandCtx<'e>,
    slice: &Slice<'e>,
    (l, r, ty): (Operand<'e>, Operand<'e>, ArithOpType),
    callback: F,
) -> Option<T> {
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

        // All ops were in slice, call callback
        Some(callback(result))
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
pub fn remove_eq_arith_parts_sorted_and_rejoin<'e>(
    ctx: OperandCtx<'e>,
    slice: &Slice<'e>,
    arith: (Operand<'e>, Operand<'e>, ArithOpType),
) -> Option<Operand<'e>> {
    remove_eq_arith_parts_sorted(ctx, slice, arith, |result| {
        // assumes that the func input was sorted / simplified arith.
        let result = intern_arith_ops_to_tree(ctx, result, arith.2)
            .unwrap_or_else(|| ctx.const_0());
        result
    })
}

/// If `(l, r, ty)` operand chain has any parts in `slice`,
/// returns rejoined `Operand` of the parts not in `slice`.
/// If all parts of `(l, r, ty)` are in `slice` returns `ctx.const_0()`.
///     ! 0 may not be identity value for the operation (e.g. and, mul)
///     caller should check for 0 and replace it with id.
///
/// That is, differs from remove_eq_arith_parts_sorted which requires entirety or `(l, r, ty)`
/// be in `slice`, and rejoins rest of `slice` instead of `(l, r, ty)`.
///
/// Assumes `slice` to be sorted and contains already simplified arith chain,
/// though the first operand of slice may be a constant. Constants are also
/// not removed by this function.
///     (So that the operation is O(n)).
pub fn remove_matching_arith_parts_sorted_and_rejoin<'e>(
    ctx: OperandCtx<'e>,
    slice: &Slice<'e>,
    (l, r, ty): (Operand<'e>, Operand<'e>, ArithOpType),
) -> Option<Operand<'e>> {
    let mut iter = IterArithOps::new_pair(l, r, ty);
    let mut slice_pos = &slice[..];
    // Skip constant if any
    if slice_pos.get(0).and_then(|x| x.if_constant()).is_some() {
        slice_pos = &slice_pos[1..];
    }
    if iter.next.and_then(|x| x.if_constant()).is_some() {
        iter.next();
    }
    let first_matching = 'outer: loop {
        let part = iter.next()?;
        'inner: loop {
            let (&slice_part, rest) = slice_pos.split_first()?;
            if slice_part < part {
                slice_pos = rest;
                continue 'inner;
            } else if slice_part == part {
                slice_pos = rest;
                break 'outer part;
            } else {
                continue 'outer;
            }
        }
    };

    ctx.simplify_temp_stack().alloc(|result| {
        let mut iter = IterArithOps::new_pair(l, r, ty);
        while let Some(part) = iter.next() {
            if part == first_matching {
                break;
            }
            result.push(part).ok()?;
        }
        'outer: loop {
            let part = match iter.next() {
                Some(s) => s,
                None => break 'outer,
            };
            'inner: loop {
                let (&slice_part, rest) = match slice_pos.split_first() {
                    Some(s) => s,
                    None => break 'outer,
                };
                if slice_part < part {
                    slice_pos = rest;
                    continue 'inner;
                } else if slice_part == part {
                    slice_pos = rest;
                    continue 'outer;
                } else {
                    result.push(part).ok()?;
                    continue 'outer;
                }
            }
        }
        let result = intern_arith_ops_to_tree(ctx, result, ty)
            .unwrap_or_else(|| ctx.const_0());
        Some(result)
    })
}

/// Assuming that (l, r, ty) and (l2, r2, ty)
/// make up two arithmetic op chains, which contain at least one
/// operand that is equal, moves the equal parts to a slice,
/// while rejoining rest of the two chains back, and calling
/// `callback(first_joined, second_joined, equal_parts)`.
pub fn split_off_matching_ops_rejoin_rest<'e, F, R>(
    ctx: OperandCtx<'e>,
    (l, r): (Operand<'e>, Operand<'e>),
    (l2, r2): (Operand<'e>, Operand<'e>),
    ty: ArithOpType,
    callback: F,
) -> Option<R>
where F: FnOnce(Option<Operand<'e>>, Option<Operand<'e>>, &mut Slice<'e>) -> Option<R>,
{
    let mut iter1 = IterArithOps::new_pair(l, r, ty);
    let mut iter2 = IterArithOps::new_pair(l2, r2, ty);

    let mut first_matching = None;
    // Handle possibility of one constant being first before rest
    // are correctly sorted.
    if let Some(next) = iter1.peek_next() {
        if next.if_constant().is_some() {
            iter1.next();
            if iter2.peek_next() == Some(next) {
                first_matching = Some(next);
            }
        }
    }
    // Skip iter2 const too
    if let Some(next) = iter2.peek_next() {
        if next.if_constant().is_some() {
            iter2.next();
        }
    }
    let first_matching = match first_matching {
        Some(s) => s,
        None => {
            'outer: loop {
                let part = iter1.next()?;
                'inner: loop {
                    let part2 = iter2.peek_next()?;
                    if part2 < part {
                        iter2.next();
                        continue 'inner;
                    } else if part2 == part {
                        iter2.next();
                        break 'outer part;
                    } else {
                        continue 'outer;
                    }
                }
            }
        }
    };

    ctx.simplify_temp_stack().alloc(|result| {
        result.push(first_matching).ok()?;

        'outer: loop {
            let part = match iter1.next() {
                Some(s) => s,
                None => break 'outer,
            };
            'inner: loop {
                let part2 = match iter2.peek_next() {
                    Some(s) => s,
                    None => break 'outer,
                };
                if part2 < part {
                    iter2.next();
                    continue 'inner;
                } else if part2 == part {
                    iter2.next();
                    result.push(part).ok()?;
                    continue 'outer;
                } else {
                    continue 'outer;
                }
            }
        }
        let first = rejoin_op_without_parts(ctx, (l, r, ty), result);
        let second = rejoin_op_without_parts(ctx, (l2, r2, ty), result);
        callback(first, second, result)
    })
}

/// Moves operands for which `check` returns true from `(l, r, ty)` chain
/// to a slice, and rejoins remaining operands back, after which
/// `callback(rejoined, removed_ops)` is called.
///
/// If no operands return true for `check`, returns `None` without
/// calling callback.
pub fn split_off_by_condition_rejoin_rest<'e, C, F, R>(
    ctx: OperandCtx<'e>,
    (l, r, ty): (Operand<'e>, Operand<'e>, ArithOpType),
    mut check: C,
    callback: F,
) -> Option<R>
where F: FnOnce(Option<Operand<'e>>, &mut Slice<'e>) -> Option<R>,
    C: FnMut(Operand<'e>) -> bool,
{
    let mut iter = IterArithOps::new_pair(l, r, ty);

    let first_matching = loop {
        let part = iter.next()?;
        if check(part) {
            break part;
        }
    };

    ctx.simplify_temp_stack().alloc(|result| {
        result.push(first_matching).ok()?;

        'outer: loop {
            let part = match iter.next() {
                Some(s) => s,
                None => break 'outer,
            };
            if check(part) {
                result.push(part).ok()?;
            }
        }
        let rejoined = rejoin_op_without_parts(ctx, (l, r, ty), result);
        callback(rejoined, result)
    })
}

/// Assuming that `(l, r, ty)` contains all operands in `parts`,
/// and that `parts` is sorted,
/// rejoins remaining parts of `(l, r, ty)`.
fn rejoin_op_without_parts<'e>(
    ctx: OperandCtx<'e>,
    (l, r, ty): (Operand<'e>, Operand<'e>, ArithOpType),
    parts: &Slice<'e>,
) -> Option<Operand<'e>> {
    let mut iter = IterArithOps::new_pair(l, r, ty);

    ctx.simplify_temp_stack().alloc(|result| {
        let mut rest = iter.next_inner;
        for &next in parts.iter() {
            loop {
                rest = iter.next_inner;
                let part = iter.next()?;
                if part == next {
                    break;
                }
                result.push(part).ok()?;
            }
        }
        let (first, parts) = if let Some(rest) = rest {
            (rest, &result[..])
        } else {
            let (&last, rest) = result.split_last()?;
            (last, rest)
        };
        Some(intern_arith_ops_to_tree_with_base(ctx, first, parts, ty))
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
    assert!(idx < slice.len() && slice.len() >= 2);
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
    // If removing last idx, keep nothing / set initial value to second last idx from slice
    // If removing second last idx, set initial value to last idx from slice
    let mut pos;
    let last_index;
    if idx < slice.len() - 2 {
        pos = orig;
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
        last_index = idx;
    } else {
        if idx == slice.len() - 1 {
            // Remove last idx
            pos = slice[slice.len() - 2];
            last_index = idx - 1;
        } else {
            // Remove second last idx
            pos = slice[slice.len() - 1];
            last_index = idx;
        }
    }

    for &part in slice[..last_index].iter().rev() {
        let arith = ArithOperand {
            ty,
            left: pos,
            right: part,
        };
        pos = ctx.intern(OperandType::Arithmetic(arith));
    }
    pos
}

/// Use only if the slice is in sorted order.
pub fn intern_arith_ops_to_tree<'e>(
    ctx: OperandCtx<'e>,
    slice: &[Operand<'e>],
    ty: ArithOpType,
) -> Option<Operand<'e>> {
    let (&first, rest) = slice.split_last()?;
    Some(intern_arith_ops_to_tree_with_base(ctx, first, rest, ty))
}

/// Use only if the slice is in sorted order.
pub fn intern_arith_ops_to_tree_with_base<'e>(
    ctx: OperandCtx<'e>,
    base: Operand<'e>,
    slice: &[Operand<'e>],
    ty: ArithOpType,
) -> Operand<'e> {
    let mut tree = base;
    for &op in slice.iter().rev() {
        let arith = ArithOperand {
            ty,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    tree
}

#[test]
fn test_sorted_arith_chain_remove_one_and_join() {
    let ctx = &super::OperandContext::new();
    let op = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.register(2),
            ctx.register(3),
        )
    );
    let arith = op.if_arithmetic_any().unwrap();
    arith_parts_to_new_slice(ctx, arith.left, arith.right, arith.ty, |slice| {
        assert_eq!(slice.len(), 3);
        let result = sorted_arith_chain_remove_one_and_join(ctx, slice, 0, op, ArithOpType::Xor);
        assert_eq!(result, ctx.xor(slice[1], slice[2]));
        let result = sorted_arith_chain_remove_one_and_join(ctx, slice, 1, op, ArithOpType::Xor);
        assert_eq!(result, ctx.xor(slice[0], slice[2]));
        let result = sorted_arith_chain_remove_one_and_join(ctx, slice, 2, op, ArithOpType::Xor);
        assert_eq!(result, ctx.xor(slice[0], slice[1]));
        Some(())
    }).unwrap();
}

#[test]
fn test_remove_matching_arith_parts_sorted_and_rejoin() {
    let ctx = &super::OperandContext::new();
    let op = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.xor(
                ctx.register(2),
                ctx.xor(
                    ctx.constant(0x5006),
                    ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
                ),
            ),
            ctx.xor(
                ctx.register(3),
                ctx.mem16(ctx.register(8), 0),
            ),
        ),
    );
    let arith = op.if_arithmetic_any().unwrap();
    arith_parts_to_new_slice(ctx, arith.left, arith.right, arith.ty, |slice| {
        let op = ctx.xor(
            ctx.register(8),
            ctx.xor(
                ctx.register(3),
                ctx.xor(
                    ctx.register(5),
                    ctx.mem16(ctx.register(8), 0),
                ),
            ),
        );
        let a = op.if_arithmetic_any().unwrap();
        let result =
            remove_matching_arith_parts_sorted_and_rejoin(ctx, slice, (a.left, a.right, a.ty))
                .unwrap();
        let eq = ctx.xor(
            ctx.register(8),
            ctx.register(5),
        );
        assert_eq!(result, eq);
        Some(())
    }).unwrap();
}

#[test]
fn test_split_off_matching_ops_rejoin_rest() {
    let ctx = &super::OperandContext::new();
    let op1 = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.xor(
                ctx.register(2),
                ctx.xor(
                    ctx.constant(0x5006),
                    ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
                ),
            ),
            ctx.xor(
                ctx.register(3),
                ctx.mem16(ctx.register(8), 0),
            ),
        ),
    );
    let op2 = ctx.xor(
        ctx.register(6),
        ctx.xor(
            ctx.xor(
                ctx.register(2),
                ctx.xor(
                    ctx.constant(0x5006),
                    ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
                ),
            ),
            ctx.xor(
                ctx.register(9),
                ctx.mem16(ctx.register(8), 0),
            ),
        ),
    );
    let a1 = op1.if_arithmetic_any().unwrap();
    let a2 = op2.if_arithmetic_any().unwrap();
    split_off_matching_ops_rejoin_rest(
        ctx,
        (a1.left, a1.right),
        (a2.left, a2.right),
        ArithOpType::Xor,
        |a, b, slice| {
            let a = a.unwrap();
            let b = b.unwrap();
            let eq1 = ctx.xor(
                ctx.register(1),
                ctx.register(3),
            );
            let eq2 = ctx.xor(
                ctx.register(6),
                ctx.register(9),
            );
            assert_eq!(a, eq1);
            assert_eq!(b, eq2);
            let mut eq_slice = [
                ctx.constant(0x5006),
                ctx.mem16(ctx.register(8), 0),
                ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
                ctx.register(2),
            ];
            eq_slice.sort_by_key(|&x| (x.if_constant().is_none(), x));
            assert_eq!(&eq_slice[..], &slice[..]);
            Some(())
        }
    ).unwrap();
}

#[test]
fn test_split_off_matching_ops_rejoin_rest_const() {
    let ctx = &super::OperandContext::new();
    let op1 = ctx.xor(
        ctx.register(1),
        ctx.xor(
            ctx.xor(
                ctx.register(2),
                ctx.xor(
                    ctx.constant(0x5006),
                    ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
                ),
            ),
            ctx.xor(
                ctx.register(3),
                ctx.mem16(ctx.register(8), 0),
            ),
        ),
    );
    let op2 = ctx.xor(
        ctx.register(6),
        ctx.xor(
            ctx.xor(
                ctx.register(2),
                ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
            ),
            ctx.xor(
                ctx.register(9),
                ctx.mem16(ctx.register(8), 0),
            ),
        ),
    );
    let a1 = op1.if_arithmetic_any().unwrap();
    let a2 = op2.if_arithmetic_any().unwrap();
    split_off_matching_ops_rejoin_rest(
        ctx,
        (a1.left, a1.right),
        (a2.left, a2.right),
        ArithOpType::Xor,
        |a, b, slice| {
            let a = a.unwrap();
            let b = b.unwrap();
            let eq1 = ctx.xor(
                ctx.register(1),
                ctx.xor(
                    ctx.register(3),
                    ctx.constant(0x5006),
                ),
            );
            let eq2 = ctx.xor(
                ctx.register(6),
                ctx.register(9),
            );
            assert_eq!(a, eq1);
            assert_eq!(b, eq2);
            let mut eq_slice = [
                ctx.mem16(ctx.register(8), 0),
                ctx.mem32(ctx.mem32(ctx.register(9), 0), 0x40),
                ctx.register(2),
            ];
            eq_slice.sort_by_key(|&x| (x.if_constant().is_none(), x));
            assert_eq!(&eq_slice[..], &slice[..]);
            Some(())
        }
    ).unwrap();
}
