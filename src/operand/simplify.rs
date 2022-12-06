use std::cmp::{min, max};
use std::ops::Range;

use smallvec::SmallVec;

use crate::bit_misc::{bits_overlap, zero_bit_ranges};
use crate::heapsort;

use super::{
    ArithOperand, ArithOpType, MemAccess, MemAccessSize, Operand, OperandType, OperandCtx,
};
use super::slice_stack::{self, SizeLimitReached};
use super::util::{self, IterArithOps, IterAddSubArithOps};
#[cfg(feature = "fuzz")] use super::tls_simplification_incomplete;

type Slice<'e> = slice_stack::Slice<'e, Operand<'e>>;
type AddSlice<'e> = slice_stack::Slice<'e, (Operand<'e>, bool)>;
type MaskedOpSlice<'e> = slice_stack::Slice<'e, (Operand<'e>, u64)>;

#[derive(Default)]
pub struct SimplifyWithZeroBits {
    simplify_count: u8,
    with_and_mask_count: u8,
    /// simplify_with_zero_bits can cause a lot of recursing in xor
    /// simplification with has functions, stop simplifying if a limit
    /// is hit.
    xor_recurse: u8,
}

impl SimplifyWithZeroBits {
    fn has_reached_limit(&self) -> bool {
        self.zero_bits_simplify_count_at_limit() ||
            self.with_and_mask_count_at_limit()
    }

    #[inline]
    fn zero_bits_simplify_count_at_limit(&self) -> bool {
        self.simplify_count > 40
    }

    #[inline]
    fn with_and_mask_count_at_limit(&self) -> bool {
        self.with_and_mask_count > 120
    }
}

pub fn simplify_arith<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ty: ArithOpType,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    // NOTE OperandContext assumes it can call these arith child functions
    // directly. Don't add anything here that is expected to be ran for
    // arith simplify.
    match ty {
        ArithOpType::Add | ArithOpType::Sub => {
            let is_sub = ty == ArithOpType::Sub;
            simplify_add_sub(left, right, is_sub, ctx)
        }
        ArithOpType::Mul => simplify_mul(left, right, ctx),
        ArithOpType::MulHigh => simplify_mul_high(left, right, ctx),
        ArithOpType::And => simplify_and(left, right, ctx, swzb_ctx),
        ArithOpType::Or => simplify_or(left, right, ctx, swzb_ctx),
        ArithOpType::Xor => simplify_xor(left, right, ctx, swzb_ctx),
        ArithOpType::Lsh => simplify_lsh(left, right, ctx, swzb_ctx),
        ArithOpType::Rsh => simplify_rsh(left, right, ctx, swzb_ctx),
        ArithOpType::Equal => simplify_eq(left, right, ctx),
        ArithOpType::GreaterThan => simplify_gt(left, right, ctx, swzb_ctx),
        ArithOpType::Div | ArithOpType::Modulo => {
            if let Some(r) = right.if_constant() {
                if r == 0 {
                    // Use 0 / 0 for any div by zero
                    let arith = ArithOperand {
                        ty,
                        left: right.clone(),
                        right,
                    };
                    return ctx.intern(OperandType::Arithmetic(arith));
                }
                if ty == ArithOpType::Modulo {
                    // If x % y == x if y > x
                    if r > left.relevant_bits_mask() {
                        return left;
                    }
                    if let Some(l) = left.if_constant() {
                        return ctx.constant(l % r);
                    }
                } else {
                    // Div, x / y == 0 if y > x
                    if r > left.relevant_bits_mask() {
                        return ctx.const_0();
                    }
                    if let Some(l) = left.if_constant() {
                        return ctx.constant(l / r);
                    }
                    // x / 1 == x
                    if r == 1 {
                        return left;
                    }
                }
            }
            let zero = ctx.const_0();
            if left == zero {
                return zero;
            }
            if left == right {
                // x % x == 0, x / x = 1
                if ty == ArithOpType::Modulo {
                    return zero;
                } else {
                    return ctx.const_1();
                }
            }
            let arith = ArithOperand {
                ty,
                left,
                right,
            };
            ctx.intern(OperandType::Arithmetic(arith))
        }
        ArithOpType::ToFloat => {
            let val = left;
            if let Some(c) = val.if_constant() {
                let float = f32::to_bits(c as i64 as f32);
                ctx.constant(float as u64)
            } else {
                let ty = OperandType::Arithmetic(ArithOperand {
                    ty,
                    left: val,
                    right: ctx.const_0(),
                });
                ctx.intern(ty)
            }
        }
        ArithOpType::ToDouble => {
            let val = left;
            if let Some(c) = val.if_constant() {
                let float = f64::to_bits(c as i64 as f64);
                ctx.constant(float)
            } else {
                let ty = OperandType::Arithmetic(ArithOperand {
                    ty,
                    left: val,
                    right: ctx.const_0(),
                });
                ctx.intern(ty)
            }
        }
        _ => {
            let ty = OperandType::Arithmetic(ArithOperand {
                ty,
                left,
                right,
            });
            ctx.intern(ty)
        }
    }
}

/// If arith result is known to be and masked, applying the mask eagerly to inputs
/// can save some work.
/// Mainly useful for signed_gt, but has some other uses as well.
pub fn simplify_arith_masked<'e>(
    mut left: Operand<'e>,
    mut right: Operand<'e>,
    ty: ArithOpType,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let useful_mask = mask != u64::MAX;
    if useful_mask {
        match ty {
            ArithOpType::And | ArithOpType::Or | ArithOpType::Xor => {
                if let Some(c) = right.if_constant() {
                    let c = c & mask;
                    if ty == ArithOpType::And {
                        return simplify_and_const(left, c, ctx, swzb_ctx);
                    } else {
                        left = simplify_with_and_mask(left, mask, ctx, swzb_ctx);
                        let c = ctx.constant(c);
                        let val = if ty == ArithOpType::Or {
                            simplify_or(left, c, ctx, swzb_ctx)
                        } else {
                            simplify_xor(left, c, ctx, swzb_ctx)
                        };
                        return ctx.and_const(val, mask);
                    }
                } else {
                    left = simplify_with_and_mask(left, mask, ctx, swzb_ctx);
                    right = simplify_with_and_mask(right, mask, ctx, swzb_ctx);
                }
            }
            ArithOpType::Lsh | ArithOpType::Rsh => {
                if let Some(c) = right.if_constant() {
                    if c < 64 {
                        if ty == ArithOpType::Lsh {
                            left = simplify_with_and_mask(left, mask >> c, ctx, swzb_ctx);
                        } else {
                            left = simplify_with_and_mask(left, mask << c, ctx, swzb_ctx);
                        }
                    }
                }
            }
            ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul => {
                if mask.wrapping_add(1) & mask == 0 {
                    left = simplify_with_and_mask(left, mask, ctx, swzb_ctx);
                    right = simplify_with_and_mask(right, mask, ctx, swzb_ctx);
                }
            }
            _ => (),
        }
    }
    let val = simplify_arith(left, right, ty, ctx, swzb_ctx);
    if useful_mask {
        simplify_and_const(val, mask, ctx, swzb_ctx)
    } else {
        val
    }
}

pub fn simplify_float_arith<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ty: ArithOpType,
    size: MemAccessSize,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    // NOTE OperandContext assumes it can call these arith child functions
    // directly. Don't add anything here that is expected to be ran for
    // arith simplify.
    match ty {
        ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul | ArithOpType::Div => {
            match (left.if_constant(), right.if_constant()) {
                (Some(l), Some(r)) if size == MemAccessSize::Mem32 => {
                    let l = f32::from_bits(l as u32);
                    let r = f32::from_bits(r as u32);
                    let result = match ty {
                        ArithOpType::Add => l + r,
                        ArithOpType::Sub => l - r,
                        ArithOpType::Mul => l * r,
                        ArithOpType::Div | _ => l / r,
                    };
                    return ctx.constant(f32::to_bits(result) as u64);
                }
                (Some(l), Some(r)) if size == MemAccessSize::Mem64 => {
                    let l = f64::from_bits(l);
                    let r = f64::from_bits(r);
                    let result = match ty {
                        ArithOpType::Add => l + r,
                        ArithOpType::Sub => l - r,
                        ArithOpType::Mul => l * r,
                        ArithOpType::Div | _ => l / r,
                    };
                    return ctx.constant(f64::to_bits(result));
                }
                _ => (),
            }
        }
        ArithOpType::GreaterThan => {
            match (left.if_constant(), right.if_constant()) {
                (Some(l), Some(r)) if size == MemAccessSize::Mem32 => {
                    let l = f32::from_bits(l as u32);
                    let r = f32::from_bits(r as u32);
                    let result = l > r;
                    return ctx.constant(result as u64);
                }
                (Some(l), Some(r)) if size == MemAccessSize::Mem64 => {
                    let l = f64::from_bits(l);
                    let r = f64::from_bits(r);
                    let result = l > r;
                    return ctx.constant(result as u64);
                }
                _ => (),
            }
        }
        ArithOpType::ToInt if size == MemAccessSize::Mem32 => {
            let val = left;
            if let Some(c) = val.if_constant() {
                let float = f32::from_bits(c as u32);
                let overflow = float > i32::MAX as f32 ||
                    float < i32::min_value() as f32;
                let int = if overflow {
                    0x8000_0000
                } else {
                    float as i32 as u32
                };
                return ctx.constant(int as u64);
            }
        }
        ArithOpType::ToDouble if size == MemAccessSize::Mem32 => {
            let val = left;
            if let Some(c) = val.if_constant() {
                let float = f32::from_bits(c as u32);
                let double = float as f64;
                let int = f64::to_bits(double);
                return ctx.constant(int);
            }
        }
        ArithOpType::ToInt if size == MemAccessSize::Mem64 => {
            let val = left;
            if let Some(c) = val.if_constant() {
                let float = f64::from_bits(c);
                let overflow = float > i64::MAX as f64 ||
                    float < i64::min_value() as f64;
                let int = if overflow {
                    0x8000_0000_0000_0000
                } else {
                    float as i64 as u64
                };
                return ctx.constant(int as u64);
            }
        }
        ArithOpType::ToFloat if size == MemAccessSize::Mem64 => {
            let val = left;
            if let Some(c) = val.if_constant() {
                let double = f64::from_bits(c);
                let float = double as f32;
                let int = f32::to_bits(float);
                return ctx.constant(int as u64);
            }
        }
        _ => (),
    }
    let ty = OperandType::ArithmeticFloat(ArithOperand {
        ty,
        left,
        right,
    }, size);
    ctx.intern(ty)
}

pub fn simplify_sign_extend<'e>(
    val: Operand<'e>,
    from: MemAccessSize,
    to: MemAccessSize,
    ctx: OperandCtx<'e>
) -> Operand<'e> {
    if from.bits() >= to.bits() {
        return ctx.const_0();
    }
    if val.relevant_bits().end < from.bits() as u8 {
        return val;
    }
    if let Some((inner, inner_from, inner_to)) = val.if_sign_extend() {
        if inner_to == from {
            return simplify_sign_extend(inner, inner_from, to, ctx);
        }
    }
    // Shouldn't be 64bit constant since then `from` would already be Mem64
    // Obviously such thing could be built, but assuming disasm/users don't..
    if let Some(val) = val.if_constant() {
        let mask = from.mask();
        let sign = (mask >> 1).wrapping_add(1);
        let ext = val & sign != 0;
        let val = val & mask;
        if ext {
            let to_mask = to.mask();
            let new = (to_mask & !mask) | val;
            ctx.constant(new)
        } else {
            ctx.constant(val)
        }
    } else {
        // Check for (x - y) & mask to x - y simplify.
        // Valid if both x and y are less than sign bit.
        if let Some((l, r)) = val.if_arithmetic_and() {
            if let Some((x, y)) = l.if_arithmetic_sub() {
                if r.if_constant() == Some(from.mask()) {
                    let max_relbits = (from.bits() - 1) as u8;
                    if x.relevant_bits().end <= max_relbits &&
                        y.relevant_bits().end <= max_relbits
                    {
                        return l;
                    }
                    if to == MemAccessSize::Mem64 {
                        match *x.ty() {
                            OperandType::SignExtend(x, from2, to2) => {
                                if to2 == from {
                                    // Can simplify `(x - y) & mask`
                                    // to `outer_sext(x) - y` if `x - y` won't do
                                    // overflow from negative x to positive result.
                                    // (Positive x to negative result is ok)
                                    // Probably could also do this when outer_sext isn't to
                                    // Mem64 with correct masking.
                                    let min_signed_val = to2.mask()
                                        .wrapping_sub(from2.mask() >> 1);
                                    let lowest_overflowing = min_signed_val
                                        .wrapping_sub(to2.mask() >> 1);
                                    let y_max = match y.if_constant() {
                                        Some(s) => s,
                                        None => y.relevant_bits_mask(),
                                    };
                                    if y_max < lowest_overflowing {
                                        return ctx.sub(
                                            ctx.sign_extend(x, from2, to),
                                            y,
                                        );
                                    }

                                }
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
        let ty = OperandType::SignExtend(val, from, to);
        ctx.intern(ty)
    }
}

fn simplify_gt_lhs_sub<'e>(
    ctx: OperandCtx<'e>,
    left: Operand<'e>,
    right: Operand<'e>,
) -> Option<Operand<'e>> {
    // Does x - y > x == y > x simplification
    // Returns `y` if it works.
    if let OperandType::Arithmetic(arith) = left.ty() {
        if arith.ty == ArithOpType::Sub || arith.ty == ArithOpType::Add {
            return ctx.simplify_temp_stack().alloc(|right_ops| {
                // right_ops won't be modified after collecting, so allocate it first
                let right_const = collect_add_ops_no_mask(right, right_ops, false).ok()?;
                ctx.simplify_temp_stack().alloc(|left_ops| {
                    let left_const = collect_add_ops_no_mask(left, left_ops, false).ok()?;
                    // x + 5 > x + 9 => (x + 9) - 4 > x + 9
                    // so
                    // x + c1 > x + c2 => (x + c2) - (c2 - c1) > x + c2
                    let constant = right_const.wrapping_sub(left_const);
                    for &right_op in right_ops.iter() {
                        if let Some(idx) = left_ops.iter().position(|&x| x == right_op) {
                            left_ops.swap_remove(idx);
                        } else {
                            return None;
                        }
                    }
                    // The ops are stored as `-y`, so toggle all signs in order to return `+y`
                    //
                    // Additionally, if right side is only a constant, the canonicalize
                    // to minimal negations, unless we got rid of the constant on left
                    let got_rid_of_constant = constant == 0 && left_const != 0;
                    if right_ops.is_empty() && !got_rid_of_constant {
                        // This logic could probably be better
                        let was_already_canonical = if constant != left_const {
                            // Prefer smaller constants
                            let left_abs = (left_const as i64)
                                .checked_abs().map(|x| x as u64).unwrap_or(left_const);
                            let constant_abs = (constant as i64)
                                .checked_abs().map(|x| x as u64).unwrap_or(constant);
                            if left_abs == constant_abs {
                                left_const < constant
                            } else {
                                left_abs < constant_abs
                            }
                        } else {
                            let negation_count = left_ops.iter().filter(|x| x.1 == true).count();
                            if negation_count * 2 < left_ops.len() {
                                // Less than half were negated
                                true
                            } else if negation_count * 2 == left_ops.len() {
                                // Exactly half were negated, have canonical form have first
                                // operand by Ord be positive
                                let first_was_positive = left_ops.iter().min()
                                    .map(|x| x.1 == false)
                                    .unwrap_or(false);
                                first_was_positive
                            } else {
                                false
                            }
                        };
                        if was_already_canonical {
                            return None;
                        }
                    }
                    for op in left_ops.iter_mut() {
                        op.1 = !op.1;
                    }
                    if left_ops.is_empty() {
                        return Some(ctx.constant(constant));
                    }
                    if constant != 0 {
                        if constant > 0x8000_0000_0000_0000 {
                            left_ops.push((ctx.constant(0u64.wrapping_sub(constant)), true))
                                .ok()?;
                        } else {
                            left_ops.push((ctx.constant(constant), false)).ok()?;
                        }
                    }
                    Some(add_sub_ops_to_tree(left_ops, ctx))
                })
            })
        }
    }
    None
}

/// True if all 1 bits in val are placed together (or just val is zero)
fn is_continuous_mask(val: u64) -> bool {
    let low = val.trailing_zeros();
    let low_mask = 1u64.wrapping_shl(low).wrapping_sub(1);
    (val | low_mask).wrapping_add(1) & val == 0
}

/// Also used for or.
/// If one of the ops contains (x ^ y) & c,
/// and some of the inner operands don't need the mask c, extract them out.
fn simplify_xor_unpack_and_masks<'e>(
    ops: &mut Slice<'e>,
    ty: ArithOpType,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<(), SizeLimitReached> {
    let mut i = 0;
    let mut end = ops.len();
    while i < end {
        if let Some((l, r)) = ops[i].if_arithmetic_and() {
            if let Some((inner, inner_r)) = l.if_arithmetic(ty) {
                if let Some(mask) = r.if_constant().filter(|&c| is_continuous_mask(c)) {
                    // Check if any should be moved out
                    let iter = IterArithOps {
                        ty,
                        next_inner: Some(inner),
                        next: Some(inner_r),
                    };
                    let any = iter.clone().any(|x| {
                        let op_bits = x.relevant_bits_mask();
                        op_bits & mask == op_bits
                    });
                    if any {
                        ops.swap_remove(i);
                        // Take ones that don't need the mask and add them to the main slice
                        let mut any_needs_mask = false;
                        for op in iter.clone() {
                            let op_bits = op.relevant_bits_mask();
                            if op_bits & mask == op_bits {
                                ops.push(op)?;
                            } else {
                                any_needs_mask = true;
                            }
                        }
                        if any_needs_mask {
                        // Take ones that need the mask, add them to a new slice and rebuild
                        // them to a xor chain which then gets masked and added as a new op
                            let new = ctx.simplify_temp_stack()
                                .alloc(|slice| {
                                    for op in iter {
                                        let op_bits = op.relevant_bits_mask();
                                        if op_bits & mask != op_bits {
                                            slice.push(op)?;
                                        }
                                    }
                                    // These should all be sorted in the correct tree order
                                    // already (Just reverse)
                                    let mut tree = match slice.last() {
                                        Some(&s) => s,
                                        None => return Ok(None),
                                    };
                                    for &op in slice.iter().rev().skip(1) {
                                        let arith = ArithOperand {
                                            ty,
                                            left: tree,
                                            right: op,
                                        };
                                        tree = ctx.intern(OperandType::Arithmetic(arith));
                                    }
                                    Ok(Some(tree))
                                })?;
                            if let Some(new) = new {
                                let new_masked = simplify_and_const(new, mask, ctx, swzb_ctx);
                                ops.push(new_masked)?;
                            }
                        }
                        end -= 1;
                        continue;
                    }
                }
            }
        }
        i += 1;
    }
    Ok(())
}

/// Merges (x & m), (y & m) to (x ^ y) & m
/// Also used for or
fn simplify_xor_merge_ands_with_same_mask<'e>(
    ops: &mut Slice<'e>,
    is_or: bool,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) {
    let mut i = 0;
    let mut limit = 50u32;
    'outer: while i + 1 < ops.len() && limit != 0 {
        let op = ops[i];
        if let Some((l, r)) = op.if_arithmetic_and() {
            let mut j = i + 1;
            while j < ops.len() && limit != 0 {
                let op2 = ops[j];
                if let Some((l2, r2)) = op2.if_arithmetic_and() {
                    let result = util::split_off_matching_ops_rejoin_rest(
                        ctx,
                        (l, r),
                        (l2, r2),
                        ArithOpType::And,
                        |x, y, mask_parts| {
                            if (x.is_none() || y.is_none()) && is_or {
                                // (u64::MAX | x) simplifies to just u64::MAX,
                                // so u64::MAX & mask_parts => mask_parts
                                // So do nothing.
                            } else {
                                let x = x.unwrap_or_else(|| ctx.constant(u64::MAX));
                                let y = y.unwrap_or_else(|| ctx.constant(u64::MAX));
                                let inner = if is_or {
                                    simplify_or(x, y, ctx, swzb_ctx)
                                } else {
                                    simplify_xor(x, y, ctx, swzb_ctx)
                                };
                                mask_parts.push(inner).ok()?;
                            }
                            limit = limit.saturating_sub(mask_parts.len() as u32);
                            simplify_and_ops(mask_parts, ctx, swzb_ctx).ok()
                        });
                    if let Some(result) = result {
                        ops[i] = result;
                        ops.swap_remove(j);
                        continue 'outer;
                    }
                    limit = limit.saturating_sub(2u32);
                }
                j += 1;
            }
            limit = limit.saturating_sub(2u32);
        }
        i += 1;
    }
}

/// Xor simplify with masks:
/// Simplifies (x ^ y) ^ (x | y) to (x & y)
fn simplify_masked_xor_or_to_and<'e>(
    ops: &mut MaskedOpSlice<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) {
    let mut limit = 20u32;
    let mut i = 0;
    let mut ops_sorted = false;
    while limit != 0 && i < ops.len() {
        let (op, orig_mask) = ops[i];
        let mask = if orig_mask != u64::MAX {
            orig_mask
        } else {
            op.relevant_bits_mask()
        };
        if let Some((l, r)) = op.if_arithmetic_or() {
            let result = ctx.simplify_temp_stack().alloc(|slice| {
                collect_xor_ops(l, slice, 8).ok()?;
                collect_xor_ops(r, slice, 8).ok()?;
                if slice.len() >= ops.len() {
                    return None;
                }
                heapsort::sort(slice);
                if !ops_sorted {
                    heapsort::sort_by(ops, |a, b| a.0 < b.0);
                    ops_sorted = true;
                }
                limit = limit.checked_sub(slice.len() as u32)?;

                let mut ops_pos = &ops[..];
                'outer: for &part in slice.iter() {
                    loop {
                        match ops_pos.split_first() {
                            Some((&(next, next_mask), rest)) => {
                                ops_pos = rest;
                                if next == part && next_mask & mask == mask {
                                    continue 'outer;
                                }
                            }
                            None => return None,
                        }
                    }
                }

                // All parts of `slice` were in `ops`, loop again but now remove them
                // Going to invalidate `i` and `ops_sorted`
                ops[i].0 = simplify_and(l, r, ctx, swzb);
                // Reverse so that swap_remove is fine
                let mut ops_pos = ops.len() - 1;
                'outer: for &part in slice.iter().rev() {
                    loop {
                        let &(next, next_mask) = ops.get(ops_pos)?;
                        if next == part && next_mask & mask == mask {

                            let remaining_mask = (next_mask ^ mask) & next.relevant_bits_mask();
                            if remaining_mask == 0 {
                                ops.swap_remove(ops_pos);
                            } else {
                                ops[ops_pos].1 = remaining_mask;
                            }
                            ops_pos = ops_pos.checked_sub(1)?;
                            continue 'outer;
                        } else {
                            ops_pos = ops_pos.checked_sub(1)?;
                        }
                    }
                }
                Some(())
            });
            if result.is_some() {
                ops_sorted = false;
                i = 0;
                continue;
            }
        }
        i += 1;
    }
}

/// Xor simplify with masks:
/// Simplifies (x & y) ^ (x | y) to (x ^ y)
fn simplify_masked_xor_or_and_to_xor<'e>(
    ops: &mut MaskedOpSlice<'e>,
) {
    let mut i = 0;
    while i < ops.len() {
        let (op, mask) = ops[i];
        if let Some(arith) = op.if_arithmetic_any() {
            let other_type = match arith.ty {
                ArithOpType::And => ArithOpType::Or,
                ArithOpType::Or => ArithOpType::And,
                _ => {
                    i += 1;
                    continue;
                }
            };
            let mut j = i + 1;
            while j < ops.len() {
                let (op2, mask2) = ops[j];
                if let Some(arith2) = op2.if_arithmetic_any() {
                    if arith2.left == arith.left &&
                        arith2.right == arith.right &&
                        arith2.ty == other_type &&
                        mask & mask2 != 0
                    {
                        let mask = match mask == u64::MAX {
                            true => op.relevant_bits_mask(),
                            false => mask,
                        };
                        let mask2 = match mask2 == u64::MAX {
                            true => op2.relevant_bits_mask(),
                            false => mask2,
                        };
                        // Expanding to x ^ y adds 2 ops the very least,
                        // and if the and/or ops have masks
                        // that isn't included in the other,
                        // they'll be kept as well.
                        let shared_mask = mask & mask2;
                        let orig_size = ops.len();
                        let ok = collect_masked_xor_ops(arith.left, ops, usize::MAX, shared_mask)
                            .and_then(|()| {
                                collect_masked_xor_ops(arith.right, ops, usize::MAX, shared_mask)
                            }).is_ok();
                        if !ok {
                            ops.shrink(orig_size);
                            break;
                        }
                        if (mask2 ^ mask) & mask2 == 0 {
                            ops.swap_remove(j);
                        } else {
                            ops[j].1 = (mask2 ^ mask) & mask2;
                        }
                        if (mask2 ^ mask) & mask == 0 {
                            ops.swap_remove(i);
                        } else {
                            ops[i].1 = (mask2 ^ mask) & mask;
                        }
                        break;
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
}

/// Simplifies xor of ors where some terms are same:
/// (x | y) ^ (x | z)
/// to !x & (y ^ z)
/// Represented in scarf as: (x ^ ffff...ffff) & (y ^ z)
///
/// In case the mask don't match, e.g.
/// ((x | y) & c1) ^ ((x | z) & c2)
/// can be written out as
/// ((x | y) & (c1 & !c2)) ^
///      ((x | z) & (!c1 & c2)) ^
///      ((x | y) ^ (x | z)) & (c1 & c2)
/// Where the last term is tried to be simplified.
/// If it simplifies to zero, (or the other two terms
/// are zero,) then the result is kept. Otherwise keeping
/// all three terms is considered too heavy to do.
///
/// Though that is not done now.
///
/// `ops` must be sorted on function entry, but the slice
/// being sorted is not preserved.
fn simplify_masked_xor_merge_or<'e>(
    ops: &mut MaskedOpSlice<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) {
    // We care about ors (and shifts containing or), rely on ops
    // being sorted and get index of first such op and last.
    let first = ops.iter()
        .position(|x| match x.0.if_arithmetic_any() {
            Some(arith) => matches!(arith.ty, ArithOpType::Lsh | ArithOpType::Or),
            None => false,
        });
    let first = match first {
        Some(s) => s,
        None => return,
    };
    // All arithmetic ops are sorted in same set, partitioned by ArithOpType,
    // though ors and left shifts may not be right next to each other.
    let last = ops[first..].iter()
        .position(|x| match x.0.if_arithmetic_any() {
            Some(_) => false,
            None => true,
        })
        .unwrap_or(ops.len());

    let mut pos = first;
    let mut limit = 50u32;
    while pos < last && pos < ops.len() && limit != 0 {
        let (op1, mask1) = ops[pos];
        let (op1, shift1) = op1.if_lsh_with_const()
            .unwrap_or((op1, 0));
        let (l1, r1) = match op1.if_arithmetic_or() {
            Some(s) => s,
            None => {
                pos += 1;
                continue;
            }
        };
        let shift1 = shift1 as u8;
        // Shift in ones, allows the mask compatibility check
        // accept some more valid cases.
        // Should be correct?
        let mask1 = (mask1 >> shift1) | !(u64::MAX >> shift1);
        // Collect or parts to slice; if a match is found in other op
        // remove it from the parts1/2, and add to result.
        let slice_stack = ctx.simplify_temp_stack();
        let result = slice_stack.alloc(|parts1| {
            let iter1 = IterArithOps {
                ty: ArithOpType::Or,
                next_inner: Some(l1),
                next: Some(r1),
            };
            for op in iter1 {
                parts1.push(op)?;
            }

            // x ^ (x | y) can still be switched to (!x & y),
            // so check all other ops for matches
            let mut j = 0;
            while j < ops.len() {
                if j == pos {
                    j += 1;
                    continue;
                }
                let (op2, mask2) = ops[j];
                let (op2, shift2) = op2.if_lsh_with_const()
                    .unwrap_or((op2, 0));
                let shift2 = shift2 as u8;
                let mask2 = (mask2 >> shift2) | !(u64::MAX >> shift2);
                let shared_mask = op1.relevant_bits_mask() | op2.relevant_bits_mask();
                if mask1 & mask2 & shared_mask != shared_mask {
                    // Could support these cases with bit more work,
                    // but skipping them now
                    j += 1;
                    continue;
                }

                // Avoid doing same work with pos/j and j/pos switched around
                if j < pos && op2.is_arithmetic(ArithOpType::Or) {
                    j += 1;
                    continue;
                }

                let result = slice_stack.alloc(|parts2| {
                    let iter2 = IterArithOps::new(op2, ArithOpType::Or);
                    for op in iter2 {
                        parts2.push(op)?;
                    }

                    slice_stack.alloc(|result_parts| {
                        let mut idx1 = 0;
                        'part1_loop: while idx1 < parts1.len() {
                            let part1 = parts1[idx1];
                            idx1 += 1;
                            let mut idx2 = 0;
                            while idx2 < parts2.len() {
                                let part2 = parts2[idx2];
                                idx2 += 1;
                                limit = match limit.checked_sub(1) {
                                    Some(s) => s,
                                    None => return Err(SizeLimitReached),
                                };
                                let is_same = simplify_xor_is_same_shifted(
                                    part1,
                                    (shift1, mask1),
                                    part2,
                                    (shift2, mask2),
                                    &mut limit,
                                );
                                if is_same {
                                    result_parts.push(part1)?;
                                    idx1 -= 1;
                                    idx2 -= 1;
                                    parts1.swap_remove(idx1);
                                    parts2.swap_remove(idx2);
                                    continue 'part1_loop;
                                }
                            }
                        }
                        if !result_parts.is_empty() && parts1.len() < 2 && parts2.len() < 2{
                            let mut result = result_parts[0];
                            for &part in &result_parts[1..] {
                                result = ctx.or(result, part);
                            }
                            if shift1 != 0 {
                                result = simplify_lsh_const(result, shift1, ctx, swzb);
                                if swzb.has_reached_limit() {
                                    return Err(SizeLimitReached);
                                }
                            }
                            let xor_left = match parts1.get(0) {
                                Some(&s) => match shift1 {
                                    0 => Some(s),
                                    shift => {
                                        let val = simplify_lsh_const(s, shift, ctx, swzb);
                                        if swzb.has_reached_limit() {
                                            return Err(SizeLimitReached);
                                        }
                                        Some(val)
                                    }
                                },
                                None => None,
                            };
                            let xor_right = match parts2.get(0) {
                                Some(&s) => match shift2 {
                                    0 => Some(s),
                                    shift => {
                                        let val = simplify_lsh_const(s, shift, ctx, swzb);
                                        if swzb.has_reached_limit() {
                                            return Err(SizeLimitReached);
                                        }
                                        Some(val)
                                    }
                                },
                                None => None,
                            };
                            let xor_result = match (xor_left, xor_right) {
                                (Some(l), Some(r)) => simplify_xor(l, r, ctx, swzb),
                                (Some(x), None) | (None, Some(x)) => x,
                                (None, None) => ctx.const_0(),
                            };
                            if swzb.has_reached_limit() {
                                return Err(SizeLimitReached);
                            }
                            // Currently lacking canonicalization between
                            // x ^ 1 and x == 0 when x is 1bit value, but
                            // x == 0 should be preferred.
                            let inv_result = if result.relevant_bits() == (0..1) &&
                                xor_result.relevant_bits() == (0..1)
                            {
                                ctx.eq_const(result, 0)
                            } else {
                                simplify_xor(result, ctx.constant(u64::MAX), ctx, swzb)
                            };
                            if swzb.has_reached_limit() {
                                return Err(SizeLimitReached);
                            }
                            let result = simplify_and(inv_result, xor_result, ctx, swzb);
                            if swzb.has_reached_limit() {
                                return Err(SizeLimitReached);
                            }
                            let mask = mask1 << shift1;
                            Ok(Some((result, mask)))
                        } else {
                            Ok(None)
                        }
                    })
                })?;
                if let Some((result, mask)) = result {
                    return Ok((result, mask, j));
                }
                j += 1;
            }
            // Just using Err for nop since it won't early exit from there
            Err(SizeLimitReached)
        });
        if let Ok((result, mask, other_pos)) = result {
            ops[pos] = (result, mask);
            ops.swap_remove(other_pos);
        }

        pos += 1;
    }
}

/// Separates m and s from (x & m) << s
/// Returning updated input shift/mask
fn simplify_xor_shifted_unwrap_shifts<'e>(
    op: Operand<'e>,
    shift: u8,
    mask: u64,
) -> (Operand<'e>, i8, u64) {
    let mut result;
    match op.if_arithmetic_any() {
        Some(arith) => {
            if let Some(c_u64) = arith.right.if_constant() {
                let c = c_u64 as u8;
                if arith.ty == ArithOpType::Lsh {
                    result = (arith.left, c as i8, mask >> c);
                } else if arith.ty == ArithOpType::Rsh {
                    // Not trying to keep shift state as negative
                    // right now, but should be possible if needed?
                    if c >= shift {
                        result = (arith.left, 0i8.wrapping_sub(c as i8), mask << c);
                    } else {
                        result = (op, 0, mask);
                    }
                } else if arith.ty == ArithOpType::And {
                    return (op, 0, mask & c_u64);
                } else {
                    return (op, 0, mask);
                }
            } else {
                return (op, 0, mask);
            }
        }
        None => return (op, 0, mask),
    };
    // Reached only if there was lsh/rsh. Unpack inner and too.
    if let Some((l, r)) = result.0.if_and_with_const() {
        result.2 &= r;
        result.0 = l;
    }
    result
}

/// If there is an operand for which the
/// operation, `(op & mask) >> shift` for (shift1, mask1) results `op1`,
/// and same for op2, return `Some(op)`
///
/// Example:
/// `(Mem16[rax] & ffff) << 0`,
/// `(Mem8[rax + 2] & ff) << 10`,
/// `(Mem8[rax + 3] & 34) << 18`,
/// are all equal with each other.
/// `(Mem16[rax] & f_ffff) << 0`,
/// is only equal with the first and third, since the
/// mask requires rax + 2 byte be 0.
fn simplify_xor_base_for_shifted<'e>(
    op1: Operand<'e>,
    s1: (u8, u64),
    op2: Operand<'e>,
    s2: (u8, u64),
    ctx: OperandCtx<'e>,
    limit: &mut u32,
) -> Option<Operand<'e>> {
    // This maybe should also return and mask if it is being limited by
    // inner ands? Things seem line up fine due to accurate relevant_bits
    // and simplification canonicalization, but if simplification
    // gives up there may be incorrect results? Or just make sure to not
    // return anything when simplification gives up.
    // simplify_xor_with_masks7 (and other xor_with_masks tests) do test for
    // correct masking so it's probably fine to not do anything extra right now.
    let (op, shift) = simplify_xor_base_for_shifted_rec(op1, s1, op2, s2, ctx, limit)?;
    if shift != 0 {
        Some(ctx.lsh_const(op, shift as u64))
    } else {
        Some(op)
    }
}

fn simplify_xor_base_for_shifted_rec<'e>(
    op1: Operand<'e>,
    (shift1, mask1): (u8, u64),
    op2: Operand<'e>,
    (shift2, mask2): (u8, u64),
    ctx: OperandCtx<'e>,
    limit: &mut u32,
) -> Option<(Operand<'e>, u8)> {
    let (op1, shift1_add, mask1) = simplify_xor_shifted_unwrap_shifts(op1, shift1, mask1);
    let shift1 = (shift1 as i8).wrapping_add(shift1_add) as u8;
    let (op2, shift2_add, mask2) = simplify_xor_shifted_unwrap_shifts(op2, shift2, mask2);
    let shift2 = (shift2 as i8).wrapping_add(shift2_add) as u8;
    if (shift1 as i8) < 0 || (shift2 as i8) < 0 {
        // Maybe could handle but other parts of this are not ready to do it now
        return None;
    }
    if op1 == op2 && shift1 == shift2 {
        return Some((op1, shift1));
    }
    let mut result = None;
    if let Some(a1) = op1.if_arithmetic_any() {
        if let Some(a2) = op2.if_arithmetic_any() {
            if a1.ty == a2.ty &&
                matches!(a1.ty, ArithOpType::Xor | ArithOpType::Or | ArithOpType::And)
            {
                *limit = limit.checked_sub(1)?;
                let s1 = (shift1, mask1);
                let s2 = (shift2, mask2);
                result = simplify_xor_base_for_shifted_arith_chain(a1, s1, a2, s2, ctx, limit);
            }
        }
    } else {
        if let Some((val, mut off1)) = is_offset_mem(op1) {
            if let Some((other_val, mut off2)) = is_offset_mem(op2) {
                if val == other_val && shift1 & 7 == 0 && shift2 & 7 == 0 {
                    off1.3 = off1.3 & (mask1 >> (off1.2 * 8));
                    off1.2 = off1.2.wrapping_add((shift1 / 8) as u32);
                    off2.3 = off2.3 & (mask2 >> (off2.2 * 8));
                    off2.2 = off2.2.wrapping_add((shift2 / 8) as u32);
                    if (off1.3 << (off1.2 * 8)) & (off2.3 << (off2.2 * 8)) != 0 {
                        return None;
                    }
                    let size_mask1 = 1u64.checked_shl(off1.1 * 8).unwrap_or(0).wrapping_sub(1);
                    let size_mask2 = 1u64.checked_shl(off2.1 * 8).unwrap_or(0).wrapping_sub(1);
                    let mask = ((off1.3 & size_mask1) << (off1.2 * 8)) |
                        ((off2.3 & size_mask2) << (off2.2 * 8));
                    if let Some(merged) = try_merge_memory(val, off1, off2, ctx) {
                        // Return shift == 0 since try_merge_memory always shifts
                        // if needed
                        if mask.wrapping_add(1) & mask != 0 {
                            // Not continuous mask, try_merge_memory won't mask in
                            // such cases so do it here.
                            result = Some((ctx.and_const(merged, mask), 0));
                        } else {
                            result = Some((merged, 0));
                        }
                    }
                }
            }
        }
    }
    result
}

fn simplify_xor_base_for_shifted_arith_chain<'e>(
    arith1: &ArithOperand<'e>,
    s1: (u8, u64),
    arith2: &ArithOperand<'e>,
    s2: (u8, u64),
    ctx: OperandCtx<'e>,
    limit: &mut u32,
) -> Option<(Operand<'e>, u8)> {
    util::arith_parts_to_new_slice(ctx, arith1.left, arith1.right, arith1.ty, |parts1| {
        util::arith_parts_to_new_slice(ctx, arith2.left, arith2.right, arith2.ty, |parts2| {
            if parts1.len() != parts2.len() {
                return None;
            }
            ctx.simplify_temp_stack().alloc(|result_parts| {
                'outer: for &part1 in parts1.iter() {
                    for j in 0..parts2.len() {
                        let part2 = parts2[j];
                        *limit = match limit.checked_sub(1) {
                            Some(s) => s,
                            None => return None,
                        };
                        let result =
                            simplify_xor_base_for_shifted_rec(part1, s1, part2, s2, ctx, limit);
                        if let Some(result) = result {
                            result_parts.push(result).ok()?;
                            parts2.swap_remove(j);
                            continue 'outer;
                        }
                    }
                    // Didn't find match
                    return None;
                }
                // Found match for all, join them back
                let min_shift = result_parts.iter().map(|&(_op, shift)| shift).min().unwrap_or(0);
                result_parts.iter()
                    .fold(None, |result, &(next, shift)| {
                        let next = if shift > min_shift {
                            ctx.lsh_const(next, (shift - min_shift) as u64)
                        } else {
                            next
                        };
                        if let Some(old) = result {
                            Some(ctx.arithmetic(arith1.ty, old, next))
                        } else {
                            Some(next)
                        }
                    })
                    .map(|op| (op, min_shift))
            })
        })
    })
}

/// Returns if `(op1 & mask1) << shift1` and `(op2 & mask2) << shift2`
/// are same/mergeable. I.e. simplify_xor_base_for_shifted returns `Some`.
fn simplify_xor_is_same_shifted<'e>(
    op1: Operand<'e>,
    (shift1, mask1): (u8, u64),
    op2: Operand<'e>,
    (shift2, mask2): (u8, u64),
    limit: &mut u32,
) -> bool {
    let (op1, shift1_add, mask1) = simplify_xor_shifted_unwrap_shifts(op1, shift1, mask1);
    let shift1 = (shift1 as i8).wrapping_add(shift1_add) as u8;
    let (op2, shift2_add, mask2) = simplify_xor_shifted_unwrap_shifts(op2, shift2, mask2);
    let shift2 = (shift2 as i8).wrapping_add(shift2_add) as u8;
    if op1 == op2 && shift1 == shift2 {
        return true;
    }
    if let Some(a1) = op1.if_arithmetic_any() {
        if let Some(a2) = op2.if_arithmetic_any() {
            if a1.ty == a2.ty && matches!(a1.ty, ArithOpType::Xor | ArithOpType::Or) {
                *limit = match limit.checked_sub(1) {
                    Some(s) => s,
                    None => return false,
                };
                let s1 = (shift1, mask1);
                let s2 = (shift2, mask2);
                if simplify_xor_is_same_shifted(a1.left, s1, a2.left, s2, limit) {
                    if simplify_xor_is_same_shifted(a1.right, s1, a2.right, s2, limit) {
                        return true;
                    }
                } else if simplify_xor_is_same_shifted(a1.left, s1, a2.right, s2, limit) {
                    if simplify_xor_is_same_shifted(a1.right, s1, a2.left, s2, limit) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

fn simplify_xor_ops<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Operand<'e>, SizeLimitReached> {
    let mut const_val = 0;
    loop {
        simplify_xor_unpack_and_masks(ops, ArithOpType::Xor, ctx, swzb_ctx)?;
        const_val = ops.iter().flat_map(|x| x.if_constant())
            .fold(const_val, |sum, x| sum ^ x);
        ops.retain(|x| x.if_constant().is_none());
        if ops.len() > 1 {
            heapsort::sort(ops);
            simplify_xor_remove_reverting(ops);
            simplify_or_merge_child_ands(ops, ctx, swzb_ctx, u64::MAX, ArithOpType::Xor)?;
            simplify_xor_merge_ands_with_same_mask(ops, false, ctx, swzb_ctx);

            const_val = ops.iter().flat_map(|x| x.if_constant())
                .fold(const_val, |sum, x| sum ^ x);
            ops.retain(|x| x.if_constant().is_none());
        }
        if ops.is_empty() {
            return Ok(ctx.constant(const_val));
        }

        let mut ops_changed = false;
        for i in 0..ops.len() {
            let op = ops[i];
            // Convert c1 ^ (y | z) to c1 ^ z ^ y if y & z == 0
            if let Some((l, r)) = op.if_arithmetic_or() {
                let l_bits = match l.if_constant() {
                    Some(c) => c,
                    None => Operand::and_masked(l).1 & l.relevant_bits_mask(),
                };
                let r_bits = match r.if_constant() {
                    Some(c) => c,
                    None => Operand::and_masked(r).1 & r.relevant_bits_mask(),
                };
                if l_bits & r_bits == 0 {
                    if let Some(c) = r.if_constant() {
                        const_val ^= c;
                        ops[i] = l;
                    } else {
                        ops[i] = l;
                        ops.push(r)?;
                    }
                    ops_changed = true;
                }
            }
        }
        for i in 0..ops.len() {
            let result = simplify_xor_try_extract_constant(ops[i], ctx, swzb_ctx);
            if let Some((new, constant)) = result {
                ops[i] = new;
                const_val ^= constant;
                ops_changed = true;
            }
        }
        let mut i = 0;
        let mut end = ops.len();
        while i < end {
            if let Some((l, r)) = ops[i].if_arithmetic(ArithOpType::Xor) {
                ops_changed = true;
                ops.swap_remove(i);
                end -= 1;
                collect_xor_ops(l, ops, usize::MAX)?;
                if let Some(c) = r.if_constant() {
                    const_val ^= c;
                } else {
                    ops.push(r)?;
                }
            }
            i += 1;
        }
        if !ops_changed {
            break;
        }
    }

    match ops.len() {
        0 => return Ok(ctx.constant(const_val)),
        1 if const_val == 0 => return Ok(ops[0]),
        _ => (),
    };
    let best_mask = simplify_or_xor_canonicalize_and_masks(
        ops,
        ArithOpType::Xor,
        &mut const_val,
        ctx,
        swzb_ctx,
    )?;
    heapsort::sort(ops);
    let mut tree = match ops.pop() {
        Some(s) => s,
        None => {
            let val = ctx.constant(const_val);
            const_val = 0;
            val
        }
    };
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    // Make constant always be on topmost right branch
    if const_val != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: tree,
            right: ctx.constant(const_val),
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    if let Some(c) = best_mask {
        tree = intern_and_const(tree, c, ctx);
    }
    Ok(tree)
}

/// Helper for simplify_or_xor_canonicalize_and_masks and
/// the reverse operation simplify_and_insert_mask_to_or_xor.
fn or_xor_canonicalize_mask_get_mask<'e>(op: Operand<'e>) -> Option<(u64, u64)> {
    // Second return value is for bits that are known to be zero even
    // without mask in the result.
    // E.g. In addition to (x & ffff),
    // (x & ff) << 8 can be used as ffff mask.
    match op.if_arithmetic_any() {
        Some(arith) if
            matches!(arith.ty, ArithOpType::And | ArithOpType::Lsh | ArithOpType::Mul) =>
        {
            let c = arith.right.if_constant()?;
            if arith.ty == ArithOpType::And {
                Some((c, !arith.left.relevant_bits_mask()))
            } else {
                let (inner, mask) = arith.left.if_and_with_const()?;
                let shift = if arith.ty == ArithOpType::Lsh {
                    c as u32
                } else {
                    if c & c.wrapping_sub(1) == 0 {
                        c.trailing_zeros()
                    } else {
                        return None;
                    }
                };
                let result = mask.wrapping_shl(shift);
                let known_zero = !inner.relevant_bits_mask().wrapping_shl(shift);
                Some((result, known_zero))
            }
        }
        _ => None,
    }
}

/// Canonicalize to (x ^ y) & ffff over x ^ (y & ffff)
/// when the outermost mask doesn't modify x
fn simplify_or_xor_canonicalize_and_masks<'e>(
    ops: &mut Slice<'e>,
    arith_ty: ArithOpType,
    const_val: &mut u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Option<u64>, SizeLimitReached> {
    let best_mask = ops.iter()
        .fold(None, |prev: Option<u64>, &op| {
            let new = or_xor_canonicalize_mask_get_mask(op);
            if let Some(new) = new {
                Some(new.0 | new.1 | prev.unwrap_or(0))
            } else {
                prev
            }
        })
        .and_then(|mask| {
            // Mask may now contain unneeded bits from or_xor_canonicalize_mask_get_mask(_, true),
            // so collect result that uses bits from or_xor_canonicalize_mask_get_mask(_, false)
            // And require that mask doesn't clear any bits from non-masked ops
            // by including x.relevant_bits_mask() there.
            if mask == 0 {
                return None;
            }
            let mut result = if arith_ty == ArithOpType::Xor {
                *const_val
            } else {
                // Or places constant outside mask, so it be checked for mask here
                0
            };
            let mut had_non_masked_op = false;
            for &op in ops.iter() {
                let relbits = match or_xor_canonicalize_mask_get_mask(op) {
                    Some((c, _)) => c,
                    _ => {
                        had_non_masked_op = true;
                        op.relevant_bits_mask()
                    }
                };
                result |= relbits;
            }
            if result & mask != result {
                return None;
            }

            // Require at least one non-masked op or one op where mask
            // can be removed if it is moved outside.
            if !had_non_masked_op {
                for &op in ops.iter() {
                    if let Some((mask, known_zero)) = or_xor_canonicalize_mask_get_mask(op) {
                        if mask == result & !known_zero {
                            return Some(result);
                        }
                    }
                }
                None
            } else {
                Some(result)
            }
        });
    if let Some(mask) = best_mask {
        let mut i = 0;
        let mut end = ops.len();
        while i < end {
            // Remove and mask when not needed and pass to simplify_with_and_mask
            // (Technically simplify_with_and_mask does that too, but do this
            // even if simplify_with_and_mask has reached recursion limit)
            let op = match ops[i].if_and_with_const() {
                Some((l, r)) if l.relevant_bits_mask() & mask == r => l,
                _ => ops[i],
            };
            // Have to call this since we want the result from this function not
            // require additional simplification, and just be able to intern the mask.
            let op = masked_or_xor_split_parts_not_needing_mask(op, mask, arith_ty, ctx, swzb_ctx)
                .unwrap_or(op);
            let op = simplify_with_and_mask(op, mask, ctx, swzb_ctx);
            if let Some((l, r)) = op.if_arithmetic(arith_ty) {
                if let Some(c) = r.if_constant() {
                    if arith_ty == ArithOpType::Xor {
                        *const_val ^= c;
                    } else {
                        *const_val |= c;
                    }
                    ops.swap_remove(i);
                    end -= 1;
                } else {
                    ops[i] = r;
                    i += 1;
                }
                collect_arith_ops(l, ops, arith_ty, usize::MAX)?;
                // Skip the default i increment
                continue;
            } else if let Some(c) = op.if_constant() {
                if arith_ty == ArithOpType::Xor {
                    *const_val ^= c;
                } else {
                    *const_val |= c;
                }
            } else {
                ops[i] = op;
            }
            i += 1;
        }
        Ok(Some(mask))
    } else {
        Ok(None)
    }
}

/// Canonicalize (x | y) & mask (Or xor) to (x | (y & mask)) when expected to.
/// (Inverse of simplify_or_xor_canonicalize_and_masks)
/// Usually and mask is kept outside, but a case such as
/// `((x & 4000) | Mem8) & 400f` should be canonicalized
/// to `(x & 4000) | (Mem8 & f)` instead.
fn simplify_and_insert_mask_to_or_xor<'e>(
    op: Operand<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    let arith = op.if_arithmetic_any()?;
    if matches!(arith.ty, ArithOpType::Or | ArithOpType::Xor) == false {
        return None;
    }
    // If all but one of operands has an AND mask,
    // (Or shifted mask. or_xor_canonicalize_mask_get_mask handles those too)
    // and the remaining operand gets reduced by the argument mask,
    // (relevant_bits & mask != relevant_bits)
    // insert the mask to the remaining operand
    let mut iter = IterArithOps::new_arith(arith);
    let mut parts_mask = 0u64;
    let not_masked = loop {
        let part = match iter.next() {
            Some(s) => s,
            None => {
                if parts_mask & mask == parts_mask {
                    // The mask isn't useful, just return op without changes
                    return Some(op);
                } else {
                    // No insertion, but mask is needed.
                    return None;
                }
            }
        };
        match or_xor_canonicalize_mask_get_mask(part) {
            Some((mask, _known_zero)) => {
                parts_mask |= mask;
            }
            None => {
                break part;
            }
        }
    };
    // Verify that `mask` is useful for `not_masked`
    let relbits = not_masked.relevant_bits_mask();
    if relbits & mask == relbits {
        return None;
    }
    // Verify that rest of the iter chain has an AND mask
    while let Some(part) = iter.next() {
        if or_xor_canonicalize_mask_get_mask(part).is_none() {
            return None;
        }
    }
    let masked = simplify_and_const(not_masked, mask, ctx, swzb_ctx);
    ctx.simplify_temp_stack().alloc(|parts| {
        for part in IterArithOps::new_arith(arith) {
            if part != not_masked {
                parts.push(part).ok()?;
            }
        }
        parts.push(masked).ok()?;
        // Maybe could just rejoin the chain without rechecking simplifications?
        // This code path is probably not executed too often anyway though.
        if arith.ty == ArithOpType::Or {
            simplify_or_ops(parts, ctx, swzb_ctx).ok()
        } else {
            simplify_xor_ops(parts, ctx, swzb_ctx).ok()
        }
    })
}

/// Assumes that `ops` is sorted.
fn simplify_xor_remove_reverting<'e>(ops: &mut Slice<'e>) {
    let mut first_same = ops.len() as isize - 1;
    let mut pos = first_same - 1;
    while pos >= 0 {
        let pos_u = pos as usize;
        let first_u = first_same as usize;
        if ops[pos_u] == ops[first_u] {
            ops.swap_remove(first_u);
            ops.swap_remove(pos_u);
            first_same -= 2;
            pos = first_same;
        } else {
            first_same = pos;
        }
        pos -= 1;
    }
}

/// Assumes that `ops` is sorted by x.0 (Mask ordering not determined), and keeps it sorted.
/// Used by xor and or
fn simplify_masked_xor_or_merge_same_ops<'e>(
    ops: &mut MaskedOpSlice<'e>,
    ctx: OperandCtx<'e>,
    ty: ArithOpType,
) {
    let mut first_same = ops.len() as isize - 1;
    let mut pos = first_same - 1;
    let mut limit = 50;
    while pos >= 0 {
        let pos_u = pos as usize;
        let first_u = first_same as usize;
        let result = simplify_xor_base_for_shifted(
            ops[pos_u].0,
            (0u8, ops[pos_u].1),
            ops[first_u].0,
            (0u8, ops[first_u].1),
            ctx,
            &mut limit,
        );
        if let Some(new) = result {
            let mask1 = ops[pos_u].0.relevant_bits_mask() & ops[pos_u].1;
            let mask2 = ops[first_u].0.relevant_bits_mask() & ops[first_u].1;
            let new_mask = if ty == ArithOpType::Xor {
                mask1 ^ mask2
            } else {
                debug_assert_eq!(ty, ArithOpType::Or);
                mask1 | mask2
            };
            let relbit_mask = new.relevant_bits_mask();
            if new_mask & relbit_mask == 0 {
                ops.remove(first_u);
                ops.remove(pos_u);
                first_same -= 2;
                pos = first_same
            } else {
                ops.remove(first_u);
                ops[pos_u].0 = new;
                ops[pos_u].1 = new_mask;
                first_same -= 1;
            }
        } else {
            first_same = pos;
        }
        pos -= 1;
    }
}

pub fn simplify_lsh<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let constant = match right.if_constant() {
        Some(s) => s,
        None => {
            if left == ctx.const_0() {
                return left;
            }
            let arith = ArithOperand {
                ty: ArithOpType::Lsh,
                left,
                right,
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        }
    };
    if constant >= 256 {
        return ctx.const_0();
    }
    simplify_lsh_const(left, constant as u8, ctx, swzb_ctx)
}

fn intern_and_const<'e>(
    left: Operand<'e>,
    constant: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    // Insert mask inside lsh / mul-by-power-of-two
    if let Some(arith) = left.if_arithmetic_any() {
        if matches!(arith.ty, ArithOpType::Lsh | ArithOpType::Mul) {
            if let Some(c) = arith.right.if_constant() {
                let shift = if arith.ty == ArithOpType::Lsh {
                    Some(c as u8)
                } else {
                    if c & c.wrapping_sub(1) == 0 {
                        Some(c.trailing_zeros() as u8)
                    } else {
                        None
                    }
                };
                if let Some(shift) = shift {
                    let a = ArithOperand {
                        ty: ArithOpType::And,
                        left: arith.left,
                        right: ctx.constant(constant >> shift),
                    };
                    let masked = ctx.intern(OperandType::Arithmetic(a));
                    let a = ArithOperand {
                        ty: arith.ty,
                        left: masked,
                        right: arith.right,
                    };
                    return ctx.intern(OperandType::Arithmetic(a));
                }
            }
        }
    }
    let arith = ArithOperand {
        ty: ArithOpType::And,
        left,
        right: ctx.constant(constant),
    };
    ctx.intern(OperandType::Arithmetic(arith))
}

fn intern_lsh_const<'e>(
    left: Operand<'e>,
    constant: u8,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    // Normalize small shifts to a mul
    if constant < 0x6 {
        let arith = ArithOperand {
            ty: ArithOpType::Mul,
            left: left,
            right: ctx.constant(1 << constant),
        };
        ctx.intern(OperandType::Arithmetic(arith))
    } else {
        let arith = ArithOperand {
            ty: ArithOpType::Lsh,
            left: left,
            right: ctx.constant(constant as u64),
        };
        ctx.intern(OperandType::Arithmetic(arith))
    }
}

pub fn simplify_lsh_const<'e>(
    left: Operand<'e>,
    constant: u8,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let default = move || intern_lsh_const(left, constant, ctx);
    if constant == 0 {
        return left;
    } else if constant >= 64 - left.relevant_bits().start {
        return ctx.const_0();
    }

    let new_left = simplify_with_and_mask(left, u64::MAX >> constant, ctx, swzb_ctx);
    let zero_bits = (64 - constant)..64;
    let new_left = match simplify_with_zero_bits(new_left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => s,
    };
    if new_left != left {
        return simplify_lsh_const(new_left, constant, ctx, swzb_ctx);
    }

    match *left.ty() {
        OperandType::Constant(a) => ctx.constant(a << constant),
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    // Simplify (x & mask) << c to (x << c) & (mask << c)
                    // (If it changes anything.)
                    // Ultimately canonical form is (x & mask) << c,
                    // but some simplifications benefit from shifting
                    // the inner value.
                    if let Some(c) = arith.right.if_constant() {
                        let high = 64 - zero_bits.start;
                        let low = left.relevant_bits().start;
                        let no_op_mask = !0u64 >> low << low << high >> high;
                        let is_useful = match arith.left.ty() {
                            OperandType::Arithmetic(arith) => {
                                matches!(
                                    arith.ty,
                                    ArithOpType::Rsh | ArithOpType::Mul |
                                        ArithOpType::Add | ArithOpType::Sub
                                )
                            }
                            _ => false,
                        };
                        if is_useful {
                            let new = simplify_lsh_const(arith.left, constant, ctx, swzb_ctx);
                            if c == no_op_mask {
                                return new;
                            } else {
                                return simplify_and_const(new, c << constant, ctx, swzb_ctx);
                            }
                        }
                    }
                    default()
                }
                ArithOpType::Xor => {
                    // Try to simplify any parts of the xor separately
                    ctx.simplify_temp_stack()
                        .alloc(|slice| {
                            collect_xor_ops(left, slice, 16).ok()?;
                            if simplify_shift_is_too_long_xor(slice) {
                                // Give up on dumb long xors
                                None
                            } else {
                                for op in slice.iter_mut() {
                                    *op = simplify_lsh_const(*op, constant, ctx, swzb_ctx);
                                }
                                simplify_xor_ops(slice, ctx, swzb_ctx).ok()
                            }
                        })
                        .unwrap_or_else(|| default())
                }
                ArithOpType::Lsh => {
                    if let Some(inner_const) = arith.right.if_constant() {
                        let sum = (inner_const as u8).saturating_add(constant);
                        if sum < 64 {
                            simplify_lsh_const(arith.left, sum, ctx, swzb_ctx)
                        } else {
                            ctx.const_0()
                        }
                    } else {
                        default()
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(rsh_const) = arith.right.if_constant() {
                        let diff = rsh_const as i8 - constant as i8;
                        if rsh_const >= 64 {
                            return ctx.const_0();
                        }
                        let mask = (!0u64 >> rsh_const) << constant;
                        let val = match diff {
                            0 => arith.left,
                            // (x >> rsh) << lsh, rsh > lsh
                            x if x > 0 => {
                                simplify_rsh(
                                    arith.left,
                                    ctx.constant(x as u64),
                                    ctx,
                                    swzb_ctx,
                                )
                            }
                            // (x >> rsh) << lsh, lsh > rsh
                            x => {
                                simplify_lsh_const(
                                    arith.left,
                                    x.abs() as u8,
                                    ctx,
                                    swzb_ctx,
                                )
                            }
                        };
                        let relbit_mask = val.relevant_bits_mask();
                        if relbit_mask & mask != relbit_mask {
                            simplify_and_const(val, mask, ctx, swzb_ctx)
                        } else {
                            val
                        }
                    } else {
                        default()
                    }
                }
                ArithOpType::Mul => {
                    if let Some(mul_const) = arith.right.if_constant() {
                        let new_const = ctx.constant(mul_const.wrapping_mul(1 << constant));
                        simplify_mul(arith.left, new_const, ctx)
                    } else {
                        // Try to apply lsh to one part of the mul which may simplify,
                        // if it changes then keep it as is.
                        fn is_simple<'e>(op: Operand<'e>) -> Option<Operand<'e>> {
                            use super::OperandType::*;
                            match *op.ty() {
                                Register(..) | Fpu(..) | Xmm(..) | Custom(..) |
                                    Undefined(..) => Some(op),
                                _ => None,
                            }
                        }
                        if let Some((simple, other)) =
                            Operand::either(arith.left, arith.right, |x| is_simple(x))
                        {
                            let inner = simplify_lsh_const(other, constant, ctx, swzb_ctx);
                            let unchanged = match *inner.ty() {
                                OperandType::Arithmetic(ref a) => a.left == other,
                                _ => false,
                            };
                            if unchanged {
                                default()
                            } else {
                                simplify_mul(inner, simple, ctx)
                            }
                        } else {
                            default()
                        }
                    }
                }
                ArithOpType::Add => {
                    if let Some(add_const) = arith.right.if_constant() {
                        let left = simplify_lsh_const(arith.left, constant, ctx, swzb_ctx);
                        simplify_add_const(left, add_const << constant, ctx)
                    } else {
                        default()
                    }
                }
                ArithOpType::Sub => {
                    if let Some(sub_const) = arith.right.if_constant() {
                        let left = simplify_lsh_const(arith.left, constant, ctx, swzb_ctx);
                        let add_const = 0u64.wrapping_sub(sub_const);
                        simplify_add_const(left, add_const << constant, ctx)
                    } else {
                        default()
                    }
                }
                _ => default(),
            }
        }
        _ => default(),
    }
}

pub fn simplify_rsh<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let constant = match right.if_constant() {
        Some(s) => s,
        None => {
            if left == ctx.const_0() {
                return left;
            }
            let arith = ArithOperand {
                ty: ArithOpType::Rsh,
                left,
                right,
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        }
    };
    if constant >= 256 {
        return ctx.const_0();
    }
    simplify_rsh_const(left, constant as u8, ctx, swzb_ctx)
}

pub fn simplify_rsh_const<'e>(
    left: Operand<'e>,
    constant: u8,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let default = || {
        let arith = ArithOperand {
            ty: ArithOpType::Rsh,
            left,
            right: ctx.constant(constant as u64),
        };
        ctx.intern(OperandType::Arithmetic(arith))
    };
    if constant == 0 {
        return left;
    } else if constant >= left.relevant_bits().end {
        return ctx.const_0();
    }

    let new_left = simplify_with_and_mask(left, u64::MAX << constant, ctx, swzb_ctx);
    let zero_bits = 0..constant;
    let new_left = match simplify_with_zero_bits(new_left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => s,
    };
    if new_left != left {
        return simplify_rsh_const(new_left, constant, ctx, swzb_ctx);
    }

    match *left.ty() {
        OperandType::Constant(a) => ctx.constant(a >> constant),
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    if let Some(c) = arith.right.if_constant() {
                        let other = arith.left;
                        let low = zero_bits.end;
                        let high = 64 - other.relevant_bits().end;
                        let no_op_mask = !0u64 >> low << low << high >> high;
                        if c == no_op_mask {
                            let new = simplify_rsh_const(other, constant, ctx, swzb_ctx);
                            return new;
                        }
                        // `(x & c) >> constant` can be simplified to
                        // `(x >> constant) & (c >> constant)
                        // With lsh/rsh it can simplify further,
                        // but do it always for canonicalization
                        let new = simplify_rsh_const(other, constant, ctx, swzb_ctx);
                        let new = simplify_and_const(new, c >> constant, ctx, swzb_ctx);
                        return new;
                    }
                    let arith = ArithOperand {
                        ty: ArithOpType::Rsh,
                        left: left,
                        right: ctx.constant(constant as u64),
                    };
                    ctx.intern(OperandType::Arithmetic(arith))
                }
                ArithOpType::Or => {
                    // Try to simplify any parts of the or separately
                    ctx.simplify_temp_stack()
                        .alloc(|slice| {
                            collect_arith_ops(left, slice, ArithOpType::Or, 8).ok()?;
                            if simplify_shift_is_too_long_xor(slice) {
                                // Give up on dumb long ors
                                None
                            } else {
                                for op in slice.iter_mut() {
                                    *op = simplify_rsh_const(*op, constant, ctx, swzb_ctx);
                                }
                                simplify_or_ops(slice, ctx, swzb_ctx).ok()
                            }
                        })
                        .unwrap_or_else(|| default())
                }
                ArithOpType::Xor => {
                    // Try to simplify any parts of the xor separately
                    ctx.simplify_temp_stack()
                        .alloc(|slice| {
                            collect_xor_ops(left, slice, 16).ok()?;
                            if simplify_shift_is_too_long_xor(slice) {
                                // Give up on dumb long xors
                                None
                            } else {
                                for op in slice.iter_mut() {
                                    *op = simplify_rsh_const(*op, constant, ctx, swzb_ctx);
                                }
                                simplify_xor_ops(slice, ctx, swzb_ctx).ok()
                            }
                        })
                        .unwrap_or_else(|| default())
                }
                ArithOpType::Mul => {
                    if let Some(c) = arith.right.if_constant().filter(|&c| c & 0x1 == 0) {
                        // Convert (x * c) to ((x * (c >> n)) << n),
                        // and then simplify that lsh
                        let lsh_size = (c.trailing_zeros() as u8).min(constant);
                        let inner = ctx.mul_const(arith.left, c >> lsh_size);
                        simplify_lsh_const_inside_rsh(ctx, swzb_ctx, inner, lsh_size, constant)
                    } else {
                        default()
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(lsh_const) = arith.right.if_constant().map(|x| x as u8) {
                        simplify_lsh_const_inside_rsh(
                            ctx,
                            swzb_ctx,
                            arith.left,
                            lsh_const,
                            constant,
                        )
                    } else {
                        default()
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(inner_const) = arith.right.if_constant().map(|x| x as u8) {
                        let sum = inner_const.saturating_add(constant);
                        if sum < 64 {
                            simplify_rsh_const(arith.left, sum, ctx, swzb_ctx)
                        } else {
                            ctx.const_0()
                        }
                    } else {
                        default()
                    }
                }
                ArithOpType::Add => {
                    // Maybe this could be loosened to only require one of the operands to
                    // not have any bits that would be discarded?
                    let ok = arith.left.relevant_bits().start >= constant &&
                        arith.right.relevant_bits().start >= constant;
                    if ok {
                        if let Some(add_const) = arith.right.if_constant() {
                            let left = ctx.rsh_const(arith.left, constant.into());
                            return simplify_add_const(left, add_const >> constant, ctx);
                        }
                    }
                    default()
                }
                ArithOpType::Sub => {
                    if arith.left.relevant_bits().start >= constant as u8 {
                        if let Some(sub_const) = arith.right.if_constant() {
                            let add_const = 0u64.wrapping_sub(sub_const);
                            if add_const >> constant << constant == add_const {
                                let left = ctx.rsh_const(arith.left, constant.into());
                                return simplify_add_const(left, add_const >> constant, ctx);
                            }
                        }
                    }
                    default()
                }
                _ => default(),
            }
        },
        OperandType::Memory(ref mem) => {
            let (address, offset) = mem.address();
            let (size, offset_add) = match mem.size {
                MemAccessSize::Mem64 => {
                    if constant >= 56 {
                        (MemAccessSize::Mem8, 7u8)
                    } else if constant >= 48 {
                        (MemAccessSize::Mem16, 6)
                    } else if constant >= 32 {
                        (MemAccessSize::Mem32, 4)
                    } else {
                        return default();
                    }
                }
                MemAccessSize::Mem32 => {
                    if constant >= 24 {
                        (MemAccessSize::Mem8, 3)
                    } else if constant >= 16 {
                        (MemAccessSize::Mem16, 2)
                    } else {
                        return default();
                    }
                }
                MemAccessSize::Mem16 => {
                    if constant >= 8 {
                        (MemAccessSize::Mem8, 1)
                    } else {
                        return default();
                    }
                }
                _ => return default(),
            };
            let c = constant - offset_add * 8;
            let new = ctx.mem_any(size, address, offset.wrapping_add(u64::from(offset_add)));
            simplify_rsh_const(new, c, ctx, swzb_ctx)
        }
        _ => default(),
    }
}

fn simplify_lsh_const_inside_rsh<'e>(
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
    lsh_left: Operand<'e>,
    lsh_const: u8,
    constant: u8,
) -> Operand<'e> {
    let diff = constant as i8 - lsh_const as i8;
    let mask = (!0u64 << lsh_const) >> constant;
    let val = match diff {
        0 => lsh_left,
        // (x << rsh) >> lsh, rsh > lsh
        x if x > 0 => {
            simplify_rsh(
                lsh_left,
                ctx.constant(x as u64),
                ctx,
                swzb_ctx,
            )
        }
        // (x << rsh) >> lsh, lsh > rsh
        x => {
            simplify_lsh(
                lsh_left,
                ctx.constant(x.abs() as u64),
                ctx,
                swzb_ctx,
            )
        }
    };
    let relbit_mask = val.relevant_bits_mask();
    if relbit_mask & mask != relbit_mask {
        simplify_and_const(val, mask, ctx, swzb_ctx)
    } else {
        val
    }
}

fn simplify_add_merge_masked_reverting<'e>(ops: &mut AddSlice<'e>, ctx: OperandCtx<'e>) -> u64 {
    // Shouldn't need as complex and_const as other places use
    fn and_const<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>)> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
                arith.right.if_constant().map(|c| (c, arith.left))
            }
            _ => None,
        }
    }

    fn check_vals<'e>(a: Operand<'e>, b: Operand<'e>, ctx: OperandCtx<'e>) -> bool {
        if let Some((l, r)) = a.if_arithmetic_sub() {
            if l == ctx.const_0() && r == b {
                return true;
            }
        }
        if let Some((l, r)) = b.if_arithmetic_sub() {
            if l == ctx.const_0() && r == a {
                return true;
            }
        }
        false
    }

    if ops.is_empty() {
        return 0;
    }
    let mut sum = 0u64;
    let mut i = 0;
    'outer: while i + 1 < ops.len() {
        let op = ops[i].0;
        if let Some((constant, val)) = and_const(op) {
            // Only try merging when mask's low bits are all ones and nothing else is
            if constant.wrapping_add(1).count_ones() <= 1 && ops[i].1 == false {
                let mut j = i + 1;
                while j < ops.len() {
                    let other_op = ops[j].0;
                    if let Some((other_constant, other_val)) = and_const(other_op) {
                        let ok = other_constant == constant &&
                            check_vals(val, other_val, ctx) &&
                            ops[j].1 == false;
                        if ok {
                            sum = sum.wrapping_add(constant.wrapping_add(1));
                            // Skips i += 1, removes j first to not move i
                            ops.swap_remove(j);
                            ops.swap_remove(i);
                            continue 'outer;
                        }
                    }
                    j += 1;
                }
            }
        }
        i += 1;
    }
    sum
}

/// Returns a better approximation of relevant bits in addition.
///
/// Eq uses this to avoid unnecessary masking, as relbits
/// for addition aren't completely stable depending on order.
///
/// E.g. since
/// add_relbits(x, y) = min(x.low, y.low) .. max(x.hi, y.hi) + 1
/// (bool + bool) = 0..2
/// (bool + u8) = 0..9
/// (bool + bool) + u8 = 0..9
/// (bool + u8) + bool = 0..10
///
/// First return value is relbit mask for positive terms, second is for negative terms.
fn relevant_bits_for_eq<'e>(ops: &[(Operand<'e>, bool)], ctx: OperandCtx<'e>) -> (u64, u64) {
    // slice_stack expects at least word sized values.
    // Could probably fix it but this'll do for now.
    #[cfg_attr(target_pointer_width = "32", repr(align(4)))]
    #[cfg_attr(target_pointer_width = "64", repr(align(8)))]
    #[derive(Copy, Clone)]
    struct Vals {
        negate: bool,
        bits_start: u8,
        bits_end: u8,
    }

    ctx.simplify_temp_stack().alloc(|sizes| {
        for &x in ops {
            let relbits = x.0.relevant_bits();
            sizes.push(Vals {
                negate: x.1,
                bits_start: relbits.start,
                bits_end: relbits.end,
            }).ok()?;
        }
        let sizes = &mut *sizes;
        heapsort::sort_by(sizes, |a, b| {
            (a.negate, a.bits_end) < (b.negate, b.bits_end)
        });
        let mut iter = sizes.iter();
        let mut pos_bits = 64..0;
        let mut neg_bits = 64..0;
        while let Some(next) = iter.next() {
            let bits = (next.bits_start)..(next.bits_end);
            if next.negate == true {
                neg_bits = bits;
                while let Some(next) = iter.next() {
                    let bits = (next.bits_start)..(next.bits_end);
                    neg_bits.start = min(bits.start, neg_bits.start);
                    neg_bits.end =
                        max(neg_bits.end.wrapping_add(1), bits.end.wrapping_add(1)).min(64);
                }
                break;
            }
            if pos_bits.end == 0 {
                pos_bits = bits;
            } else {
                pos_bits.start = min(bits.start, pos_bits.start);
                pos_bits.end =
                    max(pos_bits.end.wrapping_add(1), bits.end.wrapping_add(1)).min(64);
            }
        }
        let pos_mask = if pos_bits.end == 0 {
            0
        } else {
            let low = pos_bits.start;
            let high = 64 - pos_bits.end;
            !0u64 << high >> high >> low << low
        };
        let neg_mask = if neg_bits.end == 0 {
            0
        } else {
            let low = neg_bits.start;
            let high = 64 - neg_bits.end;
            !0u64 << high >> high >> low << low
        };
        Some((pos_mask, neg_mask))
    }).unwrap_or_else(|| {
        (0, u64::MAX)
    })
}

/// Canonicalize (inner & mask_c) == eq_c
fn canonicalize_masked_eq_const<'e>(
    ctx: OperandCtx<'e>,
    inner: Operand<'e>,
    mask_c: u64,
    eq_c: u64,
) -> Option<Operand<'e>> {
    match inner.ty() {
        OperandType::Arithmetic(arith) => {
            // If inner is (x >> c), remove the shift
            // and shift constants to opposite direction.
            // so (x & mask_c_shifted) == eq_c_shifted
            if arith.ty == ArithOpType::Rsh {
                if let Some(c) = arith.right.if_constant() {
                    let c = c as u32;
                    let left = arith.left;
                    let mask_c_shifted;
                    let eq_c_shifted;
                    mask_c_shifted = mask_c.wrapping_shl(c);
                    eq_c_shifted = eq_c.wrapping_shl(c);
                    let result = ctx.eq_const(
                        ctx.and_const(
                            left,
                            mask_c_shifted,
                        ),
                        eq_c_shifted,
                    );
                    return Some(result);
                }
            }
        }
        _ => (),
    }
    None
}

/// This is called by main simplify_eq, so it is assuming
/// that left doesn't have any constants
fn simplify_eq_1op_const<'e>(
    left: Operand<'e>,
    right: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if right == 1 {
        // Simplify x == 1 to x if x is just the lowest bit
        if left.relevant_bits().end == 1 {
            return left;
        }
    }
    if right != 0 {
        let relbits = left.relevant_bits_mask();
        if relbits & right != right {
            return ctx.const_0();
        }
    }
    if let Some(arith) = left.if_arithmetic_any() {
        // Canonicalize shifts.
        // Left shifts are outside the mask (a & C2) << C1
        // Right shifts are inside the mask (a >> C1) & C2,
        if let Some(r) = arith.right.if_constant() {
            if arith.ty == ArithOpType::And {
                if let Some(op) = canonicalize_masked_eq_const(ctx, arith.left, r, right) {
                    return op;
                }
            } else if arith.ty == ArithOpType::Lsh {
                debug_assert!(
                    1u64.wrapping_shl(r as u32).wrapping_sub(1) & right == 0,
                    "Simplify eq becomes always false, but should've been caught above {left} {right:x}",
                );
                let new_const = right.wrapping_shr(r as u32);
                return ctx.eq_const(arith.left, new_const);
            }
        }
    } else if let Some((val, from, to)) = left.if_sign_extend() {
        let from_mask = from.mask();
        let to_mask = to.mask();
        if right < to_mask - from_mask / 2 && right > from_mask / 2 {
            return ctx.const_0();
        } else {
            return simplify_eq_const(val, right & from_mask, ctx);
        }
    }
    let arith = ArithOperand {
        ty: ArithOpType::Equal,
        left,
        right: ctx.constant(right),
    };
    ctx.intern(OperandType::Arithmetic(arith))
}

pub fn simplify_eq_const<'e>(
    left: Operand<'e>,
    right: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if right == 0 {
        if let OperandType::Arithmetic(ref arith) = left.ty() {
            if arith.ty == ArithOpType::Equal {
                // Check for (x == 0) == 0 => x
                if arith.right == ctx.const_0() {
                    if arith.left.relevant_bits().end == 1 {
                        return arith.left;
                    }
                }
            } else if arith.ty == ArithOpType::Sub {
                // Simplify x - y == 0 as x == y
                return simplify_eq(arith.left, arith.right, ctx);
            } else if arith.ty == ArithOpType::GreaterThan {
                if let Some(result) = simplify_eq_zero_with_gt(arith.left, arith.right, ctx) {
                    return result;
                }
            }
        }
    }
    if let Some((_, or_const)) = Operand::and_masked(left).0.if_or_with_const() {
        // If the or forces too many bits to one, it can't be equal
        if or_const | right != right {
            return ctx.const_0();
        }
    }
    if right == 1 {
        // Simplify x == 1 to x if x is just the lowest bit
        if left.relevant_bits().end == 1 {
            return left;
        }
    }
    // Check if left can never have value that's high enough to be equal to right
    let left_bits = left.relevant_bits();
    if left_bits.end < 64 {
        let max_value = (1 << left_bits.end) - 1;
        if right > max_value {
            return ctx.const_0();
        }
    }

    let mut can_quick_simplify = can_quick_simplify_type(left.ty());
    if !can_quick_simplify {
        // Should be also able to quick simplify most arithmetic.
        // Gt has some extra logic too which needs to be checked first.
        if let OperandType::Arithmetic(ref arith) = left.ty() {
            can_quick_simplify = match arith.ty {
                ArithOpType::Add | ArithOpType::Sub |
                    ArithOpType::Lsh | ArithOpType::Rsh | ArithOpType::Mul => false,
                _ => true,
            };
            if arith.ty == ArithOpType::And {
                // (x << 8) & 800 == 0 gets simplfied to x & 8 == 0
                // (x >> 8) & 8 == 0 gets simplfied to x & 800 == 0
                // (x + ffff) & ffff == 0 becomes x & ffff == 1
                // And similar for sub.
                if let Some(c) = arith.right.if_constant() {
                    if let OperandType::Arithmetic(ref arith) = arith.left.ty() {
                        can_quick_simplify = match arith.ty {
                            ArithOpType::Add | ArithOpType::Sub => {
                                let continous_mask = c.wrapping_add(1) & c == 0;
                                !continous_mask
                            }
                            ArithOpType::Lsh | ArithOpType::Rsh | ArithOpType::Mul => false,
                            _ => true,
                        };
                    }
                }
            }
        } else if let Some(c) = left.if_constant() {
            return match c == right {
                true => ctx.const_1(),
                false => ctx.const_0(),
            }
        } else if left.if_memory().is_some() {
            can_quick_simplify = true;
        }
    }
    let right = ctx.constant(right);
    if can_quick_simplify {
        let arith = ArithOperand {
            ty: ArithOpType::Equal,
            left,
            right,
        };
        return ctx.intern(OperandType::Arithmetic(arith));
    }
    simplify_eq_main(left, right, ctx)
}

/// Special cases for simplifying `(left > right) == 0`
/// Can become `right > (left - 1)` or `(right + 1) > left`
fn simplify_eq_zero_with_gt<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    if let Some(c) = left.if_constant() {
        return Some(simplify_gt(
            right,
            ctx.constant(c.wrapping_sub(1)),
            ctx,
            &mut SimplifyWithZeroBits::default(),
        ));
    }
    if let Some(c) = right.if_constant() {
        return Some(simplify_gt(
            ctx.constant(c.wrapping_add(1)),
            left,
            ctx,
            &mut SimplifyWithZeroBits::default(),
        ));
    }
    None
}

pub fn simplify_eq<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    // Possibly common enough to be worth the early exit
    if left == right {
        return ctx.const_1();
    }
    if let Some((c, other)) = Operand::either(left, right, |x| x.if_constant()) {
        simplify_eq_const(other, c, ctx)
    } else {
        simplify_eq_main(left, right, ctx)
    }
}

/// Returns true if operand is not getting split into add/sub parts
/// and is not constant, so that eq simplification can just skip to 2op
/// simplification.
fn is_simple_eq(op: Operand<'_>, add_sub_mask: u64) -> bool {
    if op.if_constant().is_some() {
        false
    } else if let OperandType::Arithmetic(arith) = op.ty() {
        // x * 3 == x * 9 will simplify to x * 6 == 0 in add simplification, though
        // if there was special case to simplify those before calling simplify_eq_2_ops
        // it would be fine too.
        if matches!(arith.ty, ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul |
            ArithOpType::Lsh)
        {
            false
        } else if arith.ty == ArithOpType::And {
            if let Some(c) = arith.right.if_constant() {
                c & add_sub_mask != add_sub_mask
            } else {
                true
            }
        } else {
            true
        }
    } else {
        true
    }
}

#[inline]
fn eq_sort_order<'e>(a: Operand<'e>, b: Operand<'e>) -> bool {
    Operand::and_masked(a) < Operand::and_masked(b)
}

fn simplify_eq_main<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    // Equality is just bit comparision without overflow semantics, even though
    // this also uses x == y => x - y == 0 property to simplify it.
    let shared_mask = left.relevant_bits_mask() | right.relevant_bits_mask();
    let add_sub_mask = if shared_mask == 0 {
        u64::MAX
    } else {
        u64::MAX >> shared_mask.leading_zeros()
    };
    let simple = is_simple_eq(left, add_sub_mask) &&
        is_simple_eq(right, add_sub_mask);
    if simple {
        let (left, right) = match eq_sort_order(left, right) {
            true => (left, right),
            false => (right, left),
        };
        return simplify_eq_2_ops(left, right, ctx);
    }
    ctx.simplify_temp_stack().alloc(|ops| {
        simplify_add_sub_ops(ops, left, right, true, add_sub_mask, ctx)?;
        Ok(simplify_eq_ops(ops, add_sub_mask, ctx))
    }).unwrap_or_else(|SizeLimitReached| {
        let arith = ArithOperand {
            ty: ArithOpType::Equal,
            left,
            right,
        };
        let ty = OperandType::Arithmetic(arith);
        ctx.intern(ty)
    })
}

fn simplify_eq_ops<'e>(
    ops: &mut AddSlice<'e>,
    add_sub_mask: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if ops.is_empty() {
        return ctx.const_1();
    }
    // Since with eq the negations can be reverted, canonicalize eq
    // as sorting parts, and making the last one positive, swapping all
    // negations if it isn't yet.
    // Last one since the op tree is constructed in reverse in the end.
    //
    // Sorting without the mask, hopefully is valid way to keep
    // ordering stable.
    let mut constant = if let Some((c, neg)) =
        ops.last().and_then(|op| Some((op.0.if_constant()?, op.1)))
    {
        ops.pop();
        // Leaving constant positive if it's negated as it'll go to right hand side
        if neg {
            c
        } else {
            0u64.wrapping_sub(c)
        }
    } else {
        0
    };
    let zero = ctx.const_0();
    if ops.len() == 0 {
        return if constant == 0 {
            ctx.const_1()
        } else {
            zero
        };
    }
    heapsort::sort_by(ops, |a, b| eq_sort_order(a.0, b.0));
    if constant >= 0x8000_0000_0000_0000 ||
        (constant == 0 && ops.last().filter(|x| x.1 == true).is_some())
    {
        for op in ops.iter_mut() {
            op.1 = !op.1;
        }
        constant = 0u64.wrapping_sub(constant);
    }
    constant &= add_sub_mask;
    let (low, high) = sum_valid_range(ops, add_sub_mask);
    if low < high {
        if constant < low || constant > high {
            return zero;
        }
    } else {
        if constant < low && constant > high {
            return zero;
        }
    }
    match ops.len() {
        1 => {
            let swzb_ctx = &mut SimplifyWithZeroBits::default();
            if constant != 0 {
                let (mut op, neg) = ops[0];
                let constant = if neg {
                    0u64.wrapping_sub(constant) & add_sub_mask
                } else {
                    constant
                };
                let relbits = op.relevant_bits_mask();
                if add_sub_mask & relbits != relbits {
                    op = simplify_and_const(op, add_sub_mask, ctx, swzb_ctx);
                }
                simplify_eq_1op_const(op, constant, ctx)
            } else {
                // constant == 0 so ops[0] == 0 comparision
                if let Some((left, right)) = ops[0].0.if_arithmetic_eq() {
                    // Check for (x == 0) == 0
                    if right == zero {
                        if left.relevant_bits().end == 1 {
                            return left;
                        }
                    }
                }
                // (x | c) == 0 => 0
                if let Some((_, r)) = ops[0].0.if_arithmetic_or() {
                    if r.if_constant().is_some() {
                        return zero;
                    }
                }

                // Simplify (c > x) == 0 to x > (c - 1)
                // Wouldn't be valid for (0 > x) but it should never be created.
                // Same for (x > c) == 0 to c + 1 > x
                if let Some((l, r)) = ops[0].0.if_arithmetic_gt() {
                    if let Some(result) = simplify_eq_zero_with_gt(l, r, ctx) {
                        return result;
                    }
                }

                // Simplify (x << c2) == 0 to x if c2 cannot shift any bits out
                // Or ((x << c2) & c3) == 0 to (x & (c3 >> c2)) == 0
                // Or ((x >> c2) & c3) == 0 to (x & (c3 << c2)) == 0
                let (masked, mask) = Operand::and_masked(ops[0].0);
                let mask = mask & add_sub_mask;
                match masked.ty() {
                    OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Lsh => {
                        if let Some(c2) = arith.right.if_constant() {
                            let new = simplify_and_const(
                                arith.left,
                                mask.wrapping_shr(c2 as u32),
                                ctx,
                                swzb_ctx,
                            );
                            return simplify_eq(new, ctx.const_0(), ctx);
                        }
                    }
                    OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Mul => {
                        // Lsh treatment if the multiplier constant is power of 2
                        if let Some(c2) = arith.right.if_constant() {
                            if c2 & c2.wrapping_sub(1) == 0 {
                                let shift = c2.trailing_zeros();
                                let new = simplify_and_const(
                                    arith.left,
                                    mask.wrapping_shr(shift),
                                    ctx,
                                    swzb_ctx,
                                );
                                return simplify_eq(new, ctx.const_0(), ctx);
                            }
                        }
                    }
                    OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Rsh => {
                        if let Some(c2) = arith.right.if_constant() {
                            let new = simplify_and_const(
                                arith.left,
                                mask.wrapping_shl(c2 as u32),
                                ctx,
                                swzb_ctx,
                            );
                            return simplify_eq(new, ctx.const_0(), ctx);
                        }
                    }
                    _ => ()
                }
                let mut op = ops[0].0;
                let relbits = op.relevant_bits_mask();
                if add_sub_mask & relbits != relbits {
                    op = simplify_and_const(op, add_sub_mask, ctx, swzb_ctx);
                }

                let arith = ArithOperand {
                    ty: ArithOpType::Equal,
                    left: op,
                    right: zero,
                };
                ctx.intern(OperandType::Arithmetic(arith))
            }
        }
        2 if constant == 0 => {
            // Make ops[0] not need sub
            let (left, right) = match ops[0].1 {
                false => (&ops[1], &ops[0]),
                true => (&ops[0], &ops[1]),
            };
            let mask = add_sub_mask;
            fn make_op<'e>(
                ctx: OperandCtx<'e>,
                op: Operand<'e>,
                mask: u64,
                negate: bool,
            ) -> Operand<'e> {
                let swzb_ctx = &mut SimplifyWithZeroBits::default();
                match negate {
                    false => {
                        let relbit_mask = op.relevant_bits_mask();
                        if relbit_mask & mask != relbit_mask {
                            simplify_and_const(op, mask, ctx, swzb_ctx)
                        } else {
                            op
                        }
                    }
                    true => {
                        let op = ctx.sub_const_left(0, op);
                        if mask != u64::MAX {
                            simplify_and_const(op, mask, ctx, swzb_ctx)
                        } else {
                            op
                        }
                    }
                }
            }
            let left = make_op(ctx, left.0, mask, !left.1);
            let right = make_op(ctx, right.0, mask, right.1);
            simplify_eq_2_ops(left, right, ctx)
        },
        _ => {
            let mut right_tree;
            let mut left_tree = match ops.iter().position(|x| x.1 == false) {
                Some(i) => ops.swap_remove(i).0,
                None => zero,
            };
            let (left_rel_bits, right_rel_bits) = relevant_bits_for_eq(&ops, ctx);
            if constant == 0 {
                // Construct a + b + c == d + e + f
                // where left side has all non-negated terms,
                // and right side has all negated terms
                // (Negation forgotten as they're on the right)
                right_tree = match ops.iter().position(|x| x.1 == true) {
                    Some(i) => ops.swap_remove(i).0,
                    None => zero,
                };
                while let Some((op, neg)) = ops.pop() {
                    if !neg {
                        let arith = ArithOperand {
                            ty: ArithOpType::Add,
                            left: left_tree,
                            right: op,
                        };
                        left_tree = ctx.intern(OperandType::Arithmetic(arith));
                    } else {
                        let arith = ArithOperand {
                            ty: ArithOpType::Add,
                            left: right_tree,
                            right: op,
                        };
                        right_tree = ctx.intern(OperandType::Arithmetic(arith));
                    }
                }
                if add_sub_mask & left_rel_bits != left_rel_bits {
                    left_tree = simplify_and_const(
                        left_tree,
                        add_sub_mask,
                        ctx,
                        &mut SimplifyWithZeroBits::default(),
                    );
                }
                if add_sub_mask & right_rel_bits != right_rel_bits {
                    right_tree = simplify_and_const(
                        right_tree,
                        add_sub_mask,
                        ctx,
                        &mut SimplifyWithZeroBits::default(),
                    );
                }
            } else {
                // Collect everything to left, negate what needs to be negated,
                // and put constant to right

                while let Some((op, neg)) = ops.pop() {
                    let arith = ArithOperand {
                        ty: if neg { ArithOpType::Sub } else { ArithOpType::Add },
                        left: left_tree,
                        right: op,
                    };
                    left_tree = ctx.intern(OperandType::Arithmetic(arith));
                }
                if right_rel_bits != 0 || add_sub_mask & left_rel_bits != left_rel_bits {
                    left_tree = simplify_and_const(
                        left_tree,
                        add_sub_mask,
                        ctx,
                        &mut SimplifyWithZeroBits::default(),
                    );
                }
                right_tree = ctx.constant(constant);
            }
            let arith = ArithOperand {
                ty: ArithOpType::Equal,
                left: left_tree,
                right: right_tree,
            };
            let ty = OperandType::Arithmetic(arith);
            ctx.intern(ty)
        }
    }
}

fn simplify_eq_2_ops<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    let (left, right) = match left < right {
        true => (left, right),
        false => (right, left),
    };

    if let Some(result) = simplify_eq_2op_check_signed_less(ctx, left, right) {
        return result;
    }
    let arith = ArithOperand {
        ty: ArithOpType::Equal,
        left,
        right,
    };
    ctx.intern(OperandType::Arithmetic(arith))
}

// Check for sign(x - y) == overflow(x - y) => (y sgt x) == 0
fn simplify_eq_2op_check_signed_less<'e>(
    ctx: OperandCtx<'e>,
    left: Operand<'e>,
    right: Operand<'e>,
) -> Option<Operand<'e>> {
    let zero = ctx.const_0();
    let ((cmp_l, cmp_r, sign_bit, size), other) = Operand::either(left, right, |op| {
        // Sign: (((x - y) & sign_bit) == 0) == 0
        let (sub, sign) = op.if_arithmetic_eq()
            .filter(|&(_, r)| r == zero)
            .and_then(|(l, _)| l.if_arithmetic_eq())
            .filter(|&(_, r)| r == zero)
            .and_then(|(l, _)| l.if_arithmetic_and())?;
        let sign_bit = sign.if_constant()?;
        let size = match sign_bit {
            0x80 => MemAccessSize::Mem8,
            0x8000 => MemAccessSize::Mem16,
            0x8000_0000 => MemAccessSize::Mem32,
            0x8000_0000_0000_0000 => MemAccessSize::Mem64,
            _ => return None,
        };
        let (l, r) = sub.if_arithmetic_sub()?;
        Some((l, r, sign_bit, size))
    })?;
    let mask = (sign_bit << 1).wrapping_sub(1);
    // Overflow: (sign_bit > y) == ((x - y) sgt x)
    // Extract `(x - y) sgt x` part first.
    let other = other.if_arithmetic_eq()
        .and_then(|(l, r)| {
            Operand::either(l, r, |op| {
                op.if_arithmetic_gt()
                    .filter(|(l, _)| l.if_constant() == Some(sign_bit))
                    .filter(|&(_, r)| r == cmp_r)
                    .map(|_| ())
            })
            .map(|((), other)| other)
        })
        .or_else(|| {
            // Also accept just `(x - y) sgt x` if y is known to be positive and
            // `((x - y) sgt x) == 0` if negative
            match cmp_r.if_constant() {
                Some(s) if s < sign_bit => Some(other),
                Some(_) => {
                    let (l, r) = other.if_arithmetic_eq()?;
                    if r == zero {
                        Some(l)
                    } else {
                        None
                    }
                }
                None => None,
            }
        })?;

    // Since `a sgt b` is `((a + sign_bit) & mask) > ((b + sign_bit) & mask)`
    // `(x - y) sgt x` will have been simplified to `(y & mask) > ((x + sign_bit) & mask)`
    let (l, r) = other.if_arithmetic_gt()?;
    let (l, l_mask) = Operand::and_masked(l);
    let (r, r_mask) = Operand::and_masked(r);
    if l_mask != mask {
        if l_mask != !0u64 || l.relevant_bits().end > size.bits() as u8 {
            return None;
        }
    }
    let (r, rc) = r.if_arithmetic_add()
        .and_then(|(r, rc)| Some((r, rc.if_constant()?)))
        .or_else(|| {
            let (r, rc) = r.if_arithmetic_sub()?;
            let rc = rc.if_constant()?;
            Some((r, (sign_bit << 1).wrapping_sub(rc)))
        })?;
    let offset = rc.wrapping_sub(sign_bit) & mask;
    let l_ok = match l.if_constant() {
        Some(a) => match cmp_r.if_constant() {
            Some(b) => a.wrapping_sub(offset) & mask == b,
            None => false,
        },
        None => offset == 0 && l == cmp_r,
    };
    if !l_ok || r_mask != mask || r != cmp_l {
        return None;
    }
    let (cmp_l, cmp_r) = if offset == 0 {
        (cmp_l, cmp_r)
    } else {
        (ctx.add_const(cmp_l, offset), ctx.add_const(cmp_r, offset))
    };
    Some(ctx.eq_const(ctx.gt_signed(cmp_r, cmp_l, size), 0))
}

/// Returns (lowest, highest) constant what the sum of all `ops` can have.
/// Both low and high are inclusive.
///
/// Assumes that ops have been simplified already, so cases like x - x won't
/// exist. Or (0..0) ops.
///
/// The returned range can imply that x is in valid range while it is not,
/// but not imply that x is in invalid range when it is not.
///
/// And the returned range can have low > high, meaning that the values
/// between ends aren't possible. E.g. Mem8 - Mem8 has valid range of
/// (0xffff_ffff_ffff_ff01, 0xff)
fn sum_valid_range(ops: &[(Operand<'_>, bool)], mask: u64) -> (u64, u64) {
    let mut low = 0u64;
    let mut high = 0u64;
    let mut sum = 0u64.wrapping_sub(mask);
    for &val in ops {
        let bits = val.0.relevant_bits();
        let max = match 1u64.checked_shl(bits.end as u32) {
            Some(x) => x.wrapping_sub(1),
            None => return (0, mask),
        };
        sum = match sum.checked_add(max) {
            Some(s) => s,
            None => return (0, mask),
        };
        if val.1 {
            low = low.wrapping_sub(max);
        } else {
            high = high.wrapping_add(max);
        }
    }
    (low & mask, high)
}

/// Return Some(larger) if the the operand with smaller mask is same as the
/// larger operand if it were masked with that small mask.
///
/// Aims to be cheaper subset of `smaller == ctx.and_const(larger, small_mask)` check,
/// with the assumption that a & b are add/sub chains
fn try_merge_ands_check_add_merge<'e>(
    a: Operand<'e>,
    b: Operand<'e>,
    a_mask: u64,
    b_mask: u64,
) -> Option<Operand<'e>> {
    fn is_subset<'e>(larger: Operand<'e>, smaller: Operand<'e>, smaller_mask: u64) -> bool {
        if larger == smaller {
            return true;
        }
        // The larger part must include all of smaller part as it will affect results
        // with add/sub even if the mask would mask those bits out.
        match (larger.ty(), smaller.ty()) {
            (&OperandType::Arithmetic(ref a), &OperandType::Arithmetic(ref b)) => {
                if a.ty != b.ty {
                    return false;
                }
                match a.ty {
                    ArithOpType::Lsh => {
                        if a.right != b.right {
                            return false;
                        }
                        if let Some(c) = a.right.if_constant() {
                            is_subset(a.left, b.left, smaller_mask.wrapping_shr(c as u32))
                        } else {
                            false
                        }
                    }
                    ArithOpType::Rsh => {
                        if a.right != b.right {
                            return false;
                        }
                        if let Some(c) = a.right.if_constant() {
                            is_subset(a.left, b.left, smaller_mask.wrapping_shl(c as u32))
                        } else {
                            false
                        }
                    }
                    ArithOpType::And | ArithOpType::Xor | ArithOpType::Or | ArithOpType::Mul => {
                        is_subset(a.left, b.left, smaller_mask) &&
                            is_subset(a.right, b.right, smaller_mask)
                    }
                    ArithOpType::Add | ArithOpType::Sub => {
                        // Uh, this sould be correct, try_merge_ands_check_add_merge
                        // just ends up having a signature that isn't ideal here.
                        // (It always either returns None or Some(larger) and
                        // large mask would get ignored anyway)
                        try_merge_ands_check_add_merge(
                            larger,
                            smaller,
                            !0u64,
                            smaller_mask,
                        ).is_some()
                    }
                    _ => false,
                }
            }
            (&OperandType::Constant(a), &OperandType::Constant(b)) => {
                a & smaller_mask == b & smaller_mask
            }
            (&OperandType::Memory(ref a_mem), &OperandType::Memory(ref b_mem)) => {
                if a_mem.address() != b_mem.address() {
                    return false;
                }
                if a_mem.size.bits() < b_mem.size.bits() {
                    return false;
                }
                // Shouldn't accept (Mem8[x] - y) & ffff as a subset
                // of (Mem32[x] - y) & ffff_ffff for example
                let smaller_mem_mask = 1u64
                    .wrapping_shl(b_mem.size.bits().into())
                    .wrapping_sub(1);
                if smaller_mem_mask & smaller_mask != smaller_mask {
                    return false;
                }
                true
            }
            _ => false,
        }
    }

    let (larger, smaller, smaller_mask) = match a_mask > b_mask {
        true => (a, b, b_mask),
        false => (b, a, a_mask),
    };
    // Would be cleaner to have these be iterators, but this is likely an one-off case so eh
    let mut l_chain = Some(larger);
    let mut s_chain = Some(smaller);
    let mut current_ty;
    loop {
        let (next_l, next_s) = match (l_chain, s_chain) {
            (Some(l), Some(s)) => (l, s),
            (None, None) => return Some(larger),
            _ => return None,
        };
        let larger_part = match *next_l.ty() {
            OperandType::Arithmetic(ref arith) if
                matches!(arith.ty, ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul |
                    ArithOpType::And | ArithOpType::Or | ArithOpType::Xor) =>
            {
                l_chain = Some(arith.left);
                current_ty = Some(arith.ty);
                arith.right
            }
            _ => {
                l_chain = None;
                current_ty = None;
                next_l
            }
        };
        let smaller_part = match *next_s.ty() {
            OperandType::Arithmetic(ref arith) if current_ty.is_some() => {
                s_chain = Some(arith.left);
                if current_ty != Some(arith.ty) {
                    return None;
                }
                arith.right
            }
            _ => {
                s_chain = None;
                next_s
            }
        };
        if !is_subset(larger_part, smaller_part, smaller_mask) {
            return None;
        }
    }
}

/// Tries to merge (a & a_mask) | (b & b_mask) to (a_mask | b_mask) & result
/// (Does not mask the result)
fn try_merge_ands<'e>(
    a: Operand<'e>,
    b: Operand<'e>,
    a_mask: u64,
    b_mask: u64,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    if a == b {
        return Some(a);
    }
    if let Some(a) = a.if_constant() {
        if let Some(b) = b.if_constant() {
            return Some(ctx.constant(a | b));
        }
    }
    if let Some((val, shift)) = is_offset_mem(a) {
        if let Some((other_val, other_shift)) = is_offset_mem(b) {
            if val == other_val {
                let result = try_merge_memory(val, other_shift, shift, ctx);
                if let Some(merged) = result {
                    return Some(merged);
                }
            }
        }
    }
    match (a.ty(), b.ty()) {
        (&OperandType::Arithmetic(ref c), &OperandType::Arithmetic(ref d)) => {
            if c.ty == d.ty && a_mask & b_mask == 0 {
                if c.ty == ArithOpType::Xor {
                    let mut buf = SmallVec::new();
                    simplify_or_merge_ands_try_merge(
                        &mut buf, a, b, a_mask, b_mask, ArithOpType::Xor, ctx,
                    );
                    if !buf.is_empty() {
                        let result = ctx.simplify_temp_stack()
                            .alloc(|slice| {
                                for op in buf {
                                    slice.push(op)?;
                                }
                                simplify_xor_ops(slice, ctx, &mut SimplifyWithZeroBits::default())
                            }).ok();
                        return result;
                    }
                } else if c.ty == ArithOpType::Lsh {
                    if c.right == d.right {
                        if let Some(shift) = c.right.if_constant() {
                            let c_mask = a_mask.wrapping_shr(shift as u32);
                            let d_mask = b_mask.wrapping_shr(shift as u32);
                            if let Some(result) =
                                try_merge_ands(c.left, d.left, c_mask, d_mask, ctx)
                            {
                                return Some(ctx.lsh_const(result, shift));
                            }
                        }
                    }
                } else if c.ty == ArithOpType::And {
                    if let Some(c_mask) = c.right.if_constant() {
                        if let Some(d_mask) = d.right.if_constant() {
                            if let Some(result) =
                                try_merge_ands(c.left, d.left, c_mask, d_mask, ctx)
                            {
                                if c_mask | d_mask != a_mask | b_mask {
                                    return Some(ctx.and_const(result, c_mask | d_mask));
                                } else {
                                    return Some(result);
                                }
                            }
                        }
                    }
                } else if matches!(c.ty, ArithOpType::Add | ArithOpType::Sub) {
                    if let Some(result) = try_merge_ands_check_add_merge(a, b, a_mask, b_mask) {
                        return Some(result);
                    }
                }
            }
            None
        }
        (&OperandType::Arithmetic(ref arith), _) | (_, &OperandType::Arithmetic(ref arith)) => {
            // Handle things such as (a - 0x1700) & 0xff00 with a & 0xff.
            // They can be merged to (a - 0x1700) & 0xffff.
            let (arith_mask, other, other_mask) = match a.ty() {
                OperandType::Arithmetic(..) => (a_mask, b, b_mask),
                _ => (b_mask, a, a_mask),
            };
            if matches!(arith.ty, ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul) {
                if arith_mask & other_mask == 0 && arith_mask > other_mask {
                    if let Some(result) =
                        try_merge_ands(arith.left, other, arith_mask, other_mask, ctx)
                    {
                        return Some(ctx.arithmetic(arith.ty, result, arith.right));
                    }
                }
            }
            None
        }
        (&OperandType::Memory(ref a_mem), &OperandType::Memory(ref b_mem)) => {
            // Can treat Mem16[x], Mem8[x] as Mem16[x], Mem16[x]
            if a_mem.address() == b_mem.address() {
                fn check_mask<'e>(op: Operand<'e>, mask: u64, ok: Operand<'e>) -> Option<Operand<'e>> {
                    if op.relevant_bits().end >= 64 - mask.leading_zeros() as u8 {
                        Some(ok)
                    } else {
                        None
                    }
                }
                match (a_mem.size, b_mem.size) {
                    (MemAccessSize::Mem64, _) => check_mask(b, b_mask, a),
                    (_, MemAccessSize::Mem64) => check_mask(a, a_mask, b),
                    (MemAccessSize::Mem32, _) => check_mask(b, b_mask, a),
                    (_, MemAccessSize::Mem32) => check_mask(a, a_mask, b),
                    (MemAccessSize::Mem16, _) => check_mask(b, b_mask, a),
                    (_, MemAccessSize::Mem16) => check_mask(a, a_mask, b),
                    _ => None,
                }
            } else {
                None
            }
        }
        _ => None,
    }
}

fn can_quick_simplify_type(ty: &OperandType) -> bool {
    match ty {
        OperandType::Register(_) | OperandType::Xmm(_, _) | OperandType::Fpu(_) |
            OperandType::Flag(_) | OperandType::Undefined(_) | OperandType::Custom(_) => true,
        _ => false,
    }
}

/// Fast path for cases where `register_like OP constant` doesn't simplify more than that.
/// Requires that main simplification always places constant on the right for consistency.
/// (Not all do it as of writing this).
///
/// Note also that with and, it has to be first checked that `register_like & constant` != 0
/// before calling this function.
fn check_quick_arith_simplify<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
) -> Option<(Operand<'e>, Operand<'e>)> {
    let (c, other) = if left.if_constant().is_some() {
        (left, right)
    } else if right.if_constant().is_some() {
        (right, left)
    } else {
        return None;
    };
    if can_quick_simplify_type(other.ty()) {
        Some((other, c))
    } else {
        None
    }
}

/// Checks for example
///     - (Undefined - Register) & ffff_ffff
///     - (Undefined - 2343) & ffff_ffff
///     - (Mem16 + Register) & ffff_ffff
/// Which cannot be simplified and should be just returned as is
/// Assumes: right is continuous mask from 0
/// These kind of ands are ~30% of all const ands
fn can_quick_and_simplify_add_sub<'e>(left: Operand<'e>, right: u64) -> bool {
    /// Checks single part of add/sub chain; not add/sub itself.
    /// Caller is able to guarantee this from the fact that add/sub chains
    /// are canonicalized as (((x + y) + z) + c) always
    fn check<'e>(val: Operand<'e>, right: u64) -> bool {
        match *val.ty() {
            OperandType::Memory(ref mem) => {
                let next_mask = match mem.size {
                    MemAccessSize::Mem8 => 0x0u32,
                    MemAccessSize::Mem16 => 0xff,
                    MemAccessSize::Mem32 => 0xffff,
                    MemAccessSize::Mem64 => 0xffff_ffff,
                };
                u64::from(next_mask) & right != right
            }
            ref x => can_quick_simplify_type(x),
        }
    }

    fn recurse<'e>(left: Operand<'e>, right: u64) -> bool {
        if let Some(c) = left.if_constant() {
            // Can happen for LHS of sub
            c & right == c
        } else if let OperandType::Arithmetic(arith) = left.ty() {
            if arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub {
                check(arith.right, right) && recurse(arith.left, right)
            } else {
                false
            }
        } else {
            check(left, right)
        }
    }

    match *left.ty() {
        OperandType::Arithmetic(ref arith) => {
            if arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub {
                if let Some(c) = arith.right.if_constant() {
                    // Check if c gets simplified by mask
                    if c & right != c {
                        return false;
                    }
                    // Require simplification to canonicalize (x + ffff) & ffff as (x - 1) & ffff
                    // etc. x + 8000 or x - 7fff are limits
                    let mut max = right.wrapping_add(1) >> 1;
                    if arith.ty == ArithOpType::Sub {
                        max = max.wrapping_sub(1);
                    }
                    if c > max {
                        return false;
                    }
                } else {
                    if !check(arith.right, right) {
                        return false;
                    }
                }
                recurse(arith.left, right)
            } else {
                false
            }
        }
        _ => false,
    }
}

fn quick_and_simplify<'e>(
    left: Operand<'e>,
    right: u64,
    mut right_op: Option<Operand<'e>>,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    let mut ok = can_quick_simplify_type(left.ty());
    if !ok {
        let continuous_mask_from_zero = right.wrapping_add(1) & right == 0;
        if continuous_mask_from_zero {
            ok = can_quick_and_simplify_add_sub(left, right);
        }
    }
    if ok {
        let left_bits = left.relevant_bits();
        let right = if left_bits.end != 64 {
            let mask = (1 << left_bits.end) - 1;
            let masked = right & mask;
            if masked == mask {
                return Some(left);
            }
            if masked != right {
                right_op = None;
            }
            masked
        } else {
            right
        };
        let right = right_op.unwrap_or_else(|| ctx.constant(right));
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left,
            right,
        };
        return Some(ctx.intern(OperandType::Arithmetic(arith)));
    }
    None
}

/// Mem[x] & mask is pretty common so have a faster path for it.
///
/// Assumes: Mem[x] & nop_mask has been already checked by the caller (simplify_and_const)
fn simplify_and_const_mem<'e>(
    left: Operand<'e>,
    right: u64,
    mut right_op: Option<Operand<'e>>,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    let mem = left.if_memory()?;
    // If the new value has to be shifted, mask has to go inside shift.
    let (new, shift) = simplify_with_and_mask_mem_dont_apply_shift(left, mem, right, ctx);
    let new_mask = new.relevant_bits_mask() << shift;
    let new_common = new_mask & right;
    let masked = if new_common != new_mask {
        let mask_const = new_common >> shift;
        if mask_const != right {
            right_op = None;
        }
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: new,
            right: right_op.unwrap_or_else(|| ctx.constant(mask_const)),
        };
        ctx.intern(OperandType::Arithmetic(arith))
    } else {
        new
    };
    if shift != 0 {
        Some(intern_lsh_const(masked, shift, ctx))
    } else {
        Some(masked)
    }
}

/// Having a bitwise and with a constant is really common.
pub fn simplify_and_const<'e>(
    left: Operand<'e>,
    right: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    simplify_and_const_op(left, right, None, ctx, swzb_ctx)
}

pub fn simplify_and_const_op<'e>(
    mut left: Operand<'e>,
    mut right: u64,
    mut right_op: Option<Operand<'e>>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    if right == u64::MAX {
        return left;
    }
    // Check if left is x & const
    if let Some((l, r)) = left.if_arithmetic_and() {
        if let Some(c) = r.if_constant() {
            let new = c & right;
            if new == c {
                return left;
            }
            if new != right {
                right = new;
                right_op = None;
            }
            left = l;
        }
    }
    let left_mask = match left.if_constant() {
        Some(c) => {
            let common = c & right;
            if common == c {
                return left;
            } else if common == right {
                return right_op.unwrap_or_else(|| ctx.constant(right));
            } else {
                return ctx.constant(common);
            }
        }
        None => left.relevant_bits_mask(),
    };
    let common_mask = left_mask & right;
    if common_mask == 0 {
        return ctx.const_0();
    }
    if common_mask == left_mask {
        // Masking with right doesn't change the result
        return left;
    }
    if let OperandType::Arithmetic(arith) = left.ty() {
        if arith.ty == ArithOpType::Or || arith.ty == ArithOpType::Xor {
            // These checks aren't too common out of all and_const simplifications (~3%)
            // But pretty common to be useful if left is or/xor (~30%)
            // So probably worth it
            let inner_right_bits = match arith.right.if_constant() {
                Some(c) => c,
                None => arith.right.relevant_bits_mask(),
            };
            if inner_right_bits & right == 0 {
                // Right won't affect the result, use just left
                return simplify_and_const_op(arith.left, right, right_op, ctx, swzb_ctx);
            }
            // Note: Left cannot be constant, it is always canonicalized to outermost right
            let inner_left_bits = arith.left.relevant_bits_mask();
            if inner_left_bits & right == 0 {
                // Left won't affect the result, use just right
                return simplify_and_const_op(arith.right, right, right_op, ctx, swzb_ctx);
            }
        } else if arith.ty == ArithOpType::Sub {
            // Convert `(((x & mask) == 0) - 1) & mask2` to
            // `(x & mask) << c` if `(mask << c) == mask2` (Or right shift)
            // If masks are just single bit masks.
            if arith.right == ctx.const_1() {
                if right & right.wrapping_sub(1) == 0 {
                    if let Some((l, r)) = arith.left.if_arithmetic_eq() {
                        if r == ctx.const_0() {
                            if let Some((_, r2)) = l.if_arithmetic_and() {
                                if let Some(c2) = r2.if_constant() {
                                    if c2 == right {
                                        return l;
                                    } else {
                                        let inner_high_bit = 64 - c2.leading_zeros();
                                        let outer_high_bit = 64 - right.leading_zeros();
                                        if inner_high_bit > outer_high_bit {
                                            let shift = inner_high_bit - outer_high_bit;
                                            if right << shift == c2 {
                                                return ctx.rsh_const(l, shift.into());
                                            }
                                        } else {
                                            let shift = outer_high_bit - inner_high_bit;
                                            if c2 << shift == right {
                                                return ctx.lsh_const(l, shift.into());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Some(result) = quick_and_simplify(left, right, right_op, ctx) {
        return result;
    }
    if let Some(result) = simplify_and_const_mem(left, right, right_op, ctx) {
        return result;
    }
    ctx.simplify_temp_stack().alloc(|slice| {
        collect_and_ops(left, slice, 30)
            .and_then(|()| simplify_and_main(slice, right, ctx, swzb_ctx))
            .unwrap_or_else(|_| {
                let arith = ArithOperand {
                    ty: ArithOpType::And,
                    left,
                    right: right_op.unwrap_or_else(|| ctx.constant(right)),
                };
                ctx.intern(OperandType::Arithmetic(arith))
            })
    })
}

pub fn simplify_and<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    if let Some(result) = simplify_and_before_ops_collect_checks(left, right, ctx, swzb_ctx) {
        return result;
    }

    ctx.simplify_temp_stack().alloc(|slice| {
        collect_and_ops(left, slice, 30)
            .and_then(|()| collect_and_ops(right, slice, 30))
            .and_then(|()| simplify_and_main(slice, !0u64, ctx, swzb_ctx))
            .unwrap_or_else(|_| {
                let arith = ArithOperand {
                    ty: ArithOpType::And,
                    left,
                    right,
                };
                return ctx.intern(OperandType::Arithmetic(arith));
            })
    })
}

fn simplify_and_before_ops_collect_checks<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    if !bits_overlap(&left.relevant_bits(), &right.relevant_bits()) {
        return Some(ctx.const_0());
    }
    if left == right {
        // Early exit for left == right, can end up here with code like `test rax, rax`
        return Some(left);
    }
    let const_other = match left.if_constant() {
        Some(c) => Some((c, left, right)),
        None => match right.if_constant() {
            Some(c) => Some((c, right, left)),
            None => None,
        },
    };
    if let Some((c, c_op, other)) = const_other {
        return Some(simplify_and_const_op(other, c, Some(c_op), ctx, swzb_ctx));
    }
    None
}

/// Builds operand of bitwise AND of operands in `ops`.
/// (Currently effectively just simplify_and_main but having this as more
/// explicit (stableish?) interface that other code can use)
fn simplify_and_ops<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Operand<'e>, SizeLimitReached> {
    if ops.len() == 2 {
        // simplify_and can have some early-exit checks,
        // so do it when there are just 2 ops
        if let Some(result) = simplify_and_before_ops_collect_checks(ops[0], ops[1], ctx, swzb_ctx)
        {
            return Ok(result);
        }
    }
    simplify_and_main(ops, u64::MAX, ctx, swzb_ctx)
}

/// Gives same result as relevant_bits_for_and_simplify, but
/// allows the operand to be chain of ands.
fn relevant_bits_for_and_simplify_of_and_chain<'e>(op: Operand<'e>) -> u64 {
    util::IterArithOps::new(op, ArithOpType::And)
        .fold(u64::MAX, |old, op| old & relevant_bits_for_and_simplify(op))
}

fn relevant_bits_for_and_simplify<'e>(op: Operand<'e>) -> u64 {
    match *op.ty() {
        OperandType::Constant(c) => c,
        OperandType::Arithmetic(ref arith) if arith.ty == ArithOpType::Or => {
            // Special case ors to extract more accurate masks than relevant_bits_mask
            // (To not regress tests where older solution was inadequate..)
            // Specifically gets better result for things like
            // (rax & ff66) | ffff_0000 => ffff_ff66 instead of ffff_fffe
            let mut relevant_bits = if let Some(c) = arith.right.if_constant() {
                c
            } else if let Some((_, r)) = arith.right.if_and_with_const() {
                r
            } else {
                arith.right.relevant_bits_mask()
            };
            // Walk through or chain, but have relatively low limit
            // after which rest are just from relevant_bits
            let mut pos = Some(arith.left);
            let mut limit = 6u32;
            while let Some(next) = pos {
                let part = if let Some((l, r)) = next.if_arithmetic_or() {
                    pos = Some(l);
                    r
                } else {
                    pos = None;
                    next
                };
                if let Some((_, r)) = part.if_and_with_const() {
                    relevant_bits |= r;
                } else {
                    relevant_bits |= part.relevant_bits_mask();
                };
                limit = match limit.checked_sub(1) {
                    Some(s) => s,
                    None => break,
                };
            }
            if let Some(rest) = pos {
                relevant_bits |= rest.relevant_bits_mask();
            }
            relevant_bits
        }
        _ => op.relevant_bits_mask(),
    }
}

fn simplify_and_main<'e>(
    ops: &mut Slice<'e>,
    mut const_remain: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Operand<'e>, SizeLimitReached> {
    loop {
        const_remain = ops.iter()
            .map(|&op| relevant_bits_for_and_simplify(op))
            .fold(const_remain, |sum, x| sum & x);
        ops.retain(|x| x.if_constant().is_none());
        if ops.is_empty() || const_remain == 0 {
            return Ok(ctx.constant(const_remain));
        }

        if ops.len() > 1 {
            heapsort::sort(ops);
            ops.dedup();
            let is_zero = simplify_and_remove_unnecessary_ors_xors(ops, ctx, const_remain);
            if is_zero {
                return Ok(ctx.const_0());
            }
            simplify_demorgan(ops, ctx, ArithOpType::Or);
        }

        // Prefer (rax & 0xff) << 1 over (rax << 1) & 0x1fe.
        // Should this be limited to only when all ops are lsh?
        // Or ops.len() == 1.
        // Can't think of any cases now where this early break
        // would hurt though.
        let skip_simplifications = ops.iter().all(|x| {
            x.relevant_bits_mask() == const_remain
        });
        if skip_simplifications {
            break;
        }

        let mut ops_changed = false;
        if const_remain != !0 {
            slice_filter_map(ops, |op| {
                let new = simplify_with_and_mask(op, const_remain, ctx, swzb_ctx);
                if let Some(c) = new.if_constant() {
                    if c & const_remain != const_remain {
                        const_remain &= c;
                        ops_changed = true;
                    }
                    None
                } else {
                    if new != op {
                        ops_changed = true;
                    }
                    Some(new)
                }
            });
        }
        if ops.is_empty() {
            break;
        }
        let zero = ctx.const_0();
        for bits in zero_bit_ranges(const_remain) {
            slice_filter_map(ops, |op| {
                simplify_with_zero_bits(op, &bits, ctx, swzb_ctx)
                    .and_then(|x| match x == zero {
                        true => None,
                        false => Some(x),
                    })
            });
            // Unlike the other is_empty check above this returns 0, since if zero bit filter
            // removes all remaining ops, the result is 0 even with const_remain != 0
            // (simplify_with_zero_bits is defined to return None instead of Some(const(0)),
            // and obviously constant & 0 == 0)
            if ops.is_empty() {
                return Ok(zero);
            }
        }

        if const_remain != u64::MAX && ops.len() == 1 {
            // Canonicalize (x | const) & mask
            // to (x & (!const mask)) | const
            if let Some((l, r)) = ops[0].if_arithmetic_or() {
                if let Some(or_val) = r.if_constant() {
                    let inner = simplify_and_const(l, !or_val & const_remain, ctx, swzb_ctx);
                    return Ok(simplify_or(inner, r, ctx, swzb_ctx));
                }
            }
            if let Some(result) =
                simplify_and_insert_mask_to_or_xor(ops[0], const_remain, ctx, swzb_ctx)
            {
                return Ok(result);
            }
        }

        let mut i = 0;
        let mut end = ops.len();
        while i < end {
            if let Some((l, r)) = ops[i].if_arithmetic_and() {
                ops.swap_remove(i);
                end -= 1;
                collect_and_ops(l, ops, usize::MAX)?;
                if let Some(c) = r.if_constant() {
                    const_remain &= c;
                } else {
                    collect_and_ops(r, ops, usize::MAX)?;
                }
                ops_changed = true;
            } else if let Some(c) = ops[i].if_constant() {
                ops.swap_remove(i);
                end -= 1;
                if c & const_remain != const_remain {
                    ops_changed = true;
                    const_remain &= c;
                }
            } else {
                i += 1;
            }
        }

        for op in ops.iter() {
            let mask = op.relevant_bits_mask();
            if mask & const_remain != const_remain {
                ops_changed = true;
                const_remain &= mask;
            }
        }
        if !ops_changed {
            break;
        }
    }
    simplify_and_merge_child_ors(ops, ctx);
    let new_const_remain = simplify_and_merge_gt_const(ops, ctx);
    const_remain &= new_const_remain;

    // Replace not(x) & not(y) with not(x | y)
    if ops.len() >= 2 {
        let neq_compare_count = ops.iter().filter(|&&x| is_neq_compare(x, ctx)).count();
        if neq_compare_count >= 2 {
            let not: Result<_, SizeLimitReached> = ctx.simplify_temp_stack().alloc(|slice| {
                for &op in ops.iter() {
                    if is_neq_compare(op, ctx) {
                        if let Some((l, _)) = op.if_arithmetic_eq() {
                            slice.push(l)?;
                        }
                    }
                }
                let or = simplify_or_ops(slice, ctx, swzb_ctx)?;
                Ok(simplify_eq(or, ctx.const_0(), ctx))
            });
            if let Ok(not) = not {
                ops.retain(|x| !is_neq_compare(x, ctx));
                ops.push(not)?;
            }
        }
    }

    let relevant_bits = ops.iter().fold(!0, |bits, &op| {
        bits & relevant_bits_for_and_simplify(op)
    });
    // Don't use a const mask which has all 1s for relevant bits.
    let final_const_remain = if const_remain & relevant_bits == relevant_bits {
        0
    } else {
        const_remain & relevant_bits
    };
    match ops.len() {
        0 => return Ok(ctx.constant(final_const_remain)),
        1 => {
            let op = ops[0];
            if final_const_remain == 0 {
                return Ok(op);
            }
        }
        _ => {
            heapsort::sort(ops);
            ops.dedup();
        }
    };
    let is_zero = simplify_and_remove_unnecessary_ors_xors(ops, ctx, const_remain);
    if is_zero {
        return Ok(ctx.const_0());
    }
    if ops.len() == 1 && final_const_remain != 0 {
        // Canonicalize mask to be inside left shift if there is only one operand.
        // Right shifts are canonicalized to mask outside in simplify_rsh_const
        // however.
        let op = ops[0];
        if let Some(arith) = op.if_arithmetic_any() {
            let shift = arith.right.if_constant()
                .and_then(|c| {
                    if arith.ty == ArithOpType::Lsh {
                        Some(c as u8)
                    } else if arith.ty == ArithOpType::Mul && c.wrapping_sub(1) & c == 0 {
                        Some(c.trailing_zeros() as u8)
                    } else {
                        None
                    }
                });
            if let Some(shift) = shift {
                let masked = simplify_and_const(
                    arith.left,
                    final_const_remain >> shift,
                    ctx,
                    swzb_ctx,
                );
                // I think just interning here is fine...
                // simplify_lsh_const would otherwise try to undo
                // what this does here by doing (arith.left << shift)
                // and seeing if it simplified further.
                // But since it was already shifted it shouldn't ever
                // simplify.
                // Though just in case the simplification reduced to
                // a constant, handle that to prevent degenerate `0 << shift`
                // cases from being interned. (I'd expect that the above
                // simplifications should have reduced it to constant already
                // though, but there may be some edge case with swzb_ctx
                // hitting a limit)
                if let Some(c) = masked.if_constant() {
                    return Ok(ctx.constant(c >> shift));
                }
                return Ok(intern_lsh_const(masked, shift, ctx));
            }
        }
    }

    let mut tree = match ops.pop() {
        Some(s) => s,
        None => return Ok(ctx.constant(final_const_remain)),
    };
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    // Make constant always be on right of simplified and
    if final_const_remain != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: tree,
            right: ctx.constant(final_const_remain),
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    Ok(tree)
}

fn is_neq_compare<'e>(op: Operand<'e>, ctx: OperandCtx<'e>) -> bool {
    match op.if_arithmetic_eq() {
        Some((l, r)) if r == ctx.const_0() => match l.ty() {
            OperandType::Arithmetic(a) => a.is_compare_op(),
            _ => false,
        },
        _ => false,
    }
}

/// Transform (x | y | ...) & x => x
/// and (x ^ y ^ ...) & x => (u64::MAX ^ y ^ ..) & x.
/// Simplifying (x | y | (c1 | c2)) & c2 => c2
/// here would make sense, as it's based on the same transformation,
/// but it is done at simplify_with_and_mask since that is called
/// at more places.
///
/// If this returns true, and result should become 0
fn simplify_and_remove_unnecessary_ors_xors<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    const_remain: u64,
) -> bool {
    enum RemoveResult<'e> {
        // For or
        RemoveOp,
        // For xor
        ReplaceOp(Operand<'e>),
    }
    let mut limit = 50u32;
    let mut pos = 0;
    while pos < ops.len() && limit != 0 {
        let op = ops[pos];
        if let Some(arith) = op.if_arithmetic_any() {
            if matches!(arith.ty, ArithOpType::Or | ArithOpType::Xor) {
                let remove_result =
                    util::arith_parts_to_new_slice(ctx, arith.left, arith.right, arith.ty, |parts| {
                        for j in 0..ops.len() {
                            if j == pos {
                                continue;
                            }
                            let op2 = ops[j];
                            if let Some((l, r)) = op2.if_arithmetic(arith.ty) {
                                // Since both are same type of arithmetic,
                                // check if op2 is subset of op by expanding
                                // it to arith parts and removing matches.
                                // E.g. (x | y) is still subset of (x | z) | y
                                // even if sort order makes z be in between.
                                let result = util::remove_eq_arith_parts_sorted(
                                    ctx,
                                    parts,
                                    (l, r, arith.ty),
                                    |result| {
                                        if arith.ty == ArithOpType::Or {
                                            RemoveResult::RemoveOp
                                        } else {
                                            let rest = util::intern_arith_ops_to_tree(
                                                ctx,
                                                result.iter().rev().copied(),
                                                arith.ty,
                                            ).unwrap_or_else(|| ctx.const_0());
                                            let rest_not = ctx.xor_const(rest, u64::MAX);
                                            RemoveResult::ReplaceOp(rest_not)
                                        }
                                    },
                                );
                                if let Some(result) = result {
                                    return Some(result);
                                }
                            } else {
                                // Not same type of arithmetic, can just check if op2
                                // is any of the op1 parts
                                if let Some(pos) = parts.iter().position(|&part| part == op2) {
                                    if arith.ty == ArithOpType::Or {
                                        return Some(RemoveResult::RemoveOp);
                                    } else {
                                        let rest = util::sorted_arith_chain_remove_one_and_join(
                                            ctx,
                                            parts,
                                            pos,
                                            op,
                                            arith.ty,
                                        );
                                        let rest_not = ctx.xor_const(rest, u64::MAX);
                                        return Some(RemoveResult::ReplaceOp(rest_not));
                                    }
                                }
                            }
                            limit = limit.checked_sub(parts.len() as u32)?;
                        }
                        None
                    });
                match remove_result {
                    Some(RemoveResult::RemoveOp) => {
                        ops.swap_remove(pos);
                        // Don't increment pos
                        continue;
                    }
                    Some(RemoveResult::ReplaceOp(new)) => {
                        if new.relevant_bits_mask() & const_remain == 0 {
                            // Can just remove everything since new doesn't have shared
                            // nonzero bits with const_remain
                            return true;
                        }
                        ops[pos] = new;
                    }
                    None => (),
                }
            }
        }
        pos += 1;
    }
    false
}

/// Merges (x | c1) & (x | c2) to (x | (c1 & c2))
fn simplify_and_merge_child_ors<'e>(ops: &mut Slice<'e>, ctx: OperandCtx<'e>) {
    let mut i = 0;
    while i < ops.len() {
        let op = ops[i];
        let mut new = None;
        let mut changed = false;
        if let Some((val, mut constant)) = op.if_or_with_const() {
            for j in ((i + 1)..ops.len()).rev() {
                let second = ops[j];
                if let Some((other_val, other_constant)) = second.if_or_with_const() {
                    if other_val == val {
                        ops.swap_remove(j);
                        constant &= other_constant;
                        changed = true;
                    }
                }
            }
            if changed {
                new = Some(ctx.or_const(val, constant));
            }
        }
        if let Some(new) = new {
            ops[i] = new;
        }
        i += 1;
    }
}

// Merges (x > c) & ((x == c + 1) == 0) to x > (c + 1)
// Or alternatively (x > c) & ((x == c2) == 0) to x > c
// if c2 < c
// Returns true if constant mask becomes 1
#[must_use]
fn simplify_and_merge_gt_const<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
) -> u64 {
    // Range start, range end (inclusive), mask, compare operand, range value
    // E.g. range value for `x in 5..=10` would be true,
    // false for `not x in 5..=10`
    // Signed ranges, e.g. `-1 ..= 2, true` should be represented as `3 ..= -2, false`
    // instead; start <= end always.
    fn check<'e>(op: Operand<'e>) -> Option<(u64, u64, u64, Operand<'e>, bool)> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::GreaterThan => {
                if let Some(c) = arith.left.if_constant() {
                    // c > x - c2 is a (c2 ..= c2 + c - 1) range
                    let (mask, inner) = arith.right.if_arithmetic_and()
                        .and_then(|(l, r)| {
                            let mask = r.if_constant()
                                .filter(|&c| c.wrapping_add(1) & c == 0)?;
                            Some((mask, l))
                        })
                        .unwrap_or_else(|| (u64::MAX, arith.right));
                    let (c2, inner) = inner.if_arithmetic_sub()
                        .and_then(|x| x.1.if_constant().map(|c2| (c2, x.0)))
                        .unwrap_or_else(|| (0, inner));
                    Some((c2, c2.wrapping_add(c).wrapping_sub(1), mask, inner, true))
                } else if let Some(c) = arith.right.if_constant() {
                    // x - c2 > c is a (c2 ..= c2 + c) false range
                    // Alternatively:
                    // c2 - x > c is a (c2 + 1 ..= c - c2 - 1) true range
                    // but it isn't implemented currently.
                    let (mask, inner) = arith.left.if_arithmetic_and()
                        .and_then(|(l, r)| {
                            let mask = r.if_constant()
                                .filter(|&c| c.wrapping_add(1) & c == 0)?;
                            Some((mask, l))
                        })
                        .unwrap_or_else(|| (u64::MAX, arith.left));
                    inner.if_arithmetic_sub()
                        .and_then(|x| {
                            x.1.if_constant()
                                .map(|c2| (c2, c2.wrapping_add(c), mask, x.0, false))
                        })
                        .or_else(|| Some((0, c, mask, inner, false)))
                } else {
                    None
                }
            }
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Equal => {
                let c = arith.right.if_constant()?;
                if c == 0 {
                    if let Some(mut result) = check(arith.left) {
                        result.4 = !result.4;
                        return Some(result);
                    }
                }
                let (left, mask) = Operand::and_masked(arith.left);
                Some((c, c, mask, left, true))
            }
            _ => None,
        }
    }

    fn try_merge<'e>(
        base: &mut (u64, u64, u64, Operand<'e>, bool),
        other: &(u64, u64, u64, Operand<'e>, bool),
    ) -> bool {
        // Using term "positive range" to mean `x in range` (bool == true)
        // and negative range for `x not in range` (bool == false)
        if base.3 != other.3 {
            return false;
        }
        let mut mask = base.2;
        if mask != other.2 {
            // Should still be able to merge a u32 compare with u32 eq
            // ((x - c1) & mask > c2) & (x != c1) for example,
            // so allow reducing u64::max_value mask to a smaller mask
            // if that one has low == high
            let larger_masked = if base.2 < other.2 { other } else { &*base };
            if larger_masked.0 == larger_masked.1 && larger_masked.2 == u64::MAX {
                mask = base.2.min(other.2);
            } else {
                return false;
            }
        }
        if base.4 != other.4 {
            // Positive range and negative range must be true;
            // base = (pos - neg), true
            let (pos, neg) = if base.4 { (&*base, other) } else { (other, &*base) };
            if pos.0 >= neg.0 && pos.1 <= neg.1 {
                // Negative range contains all of the positive - e.g. the result can never
                // be true
                *base = (0, u64::MAX, u64::MAX, base.3, false);
                true
            } else if neg.1 < pos.0 || neg.0 > pos.1 {
                // No overlap
                *base = *pos;
                base.2 = mask;
                true
            } else if neg.0 > pos.0 && neg.1 < pos.1 {
                // Negative range is fully inside positive, leaving positive ranges on both sides
                // cannot merge with this representation (Could split to 2 positive ranges but eh)
                false
            } else if neg.0 > pos.0 {
                // Negative range removes high part of positive
                *base = (pos.0, neg.1.wrapping_sub(1), mask, pos.3, true);
                true
            } else {
                // Negative range removes low part of positive
                *base = (neg.0.wrapping_add(1), pos.1, mask, pos.3, true);
                true
            }
        } else if base.4 == true {
            // Two positive ranges, intersection of them
            let low = base.0.max(other.0);
            let high = base.1.min(other.1);
            if low > high {
                // Empty set
                *base = (0, u64::MAX, u64::MAX, base.3, false);
                true
            } else {
                base.0 = low;
                base.1 = high;
                base.2 = mask;
                true
            }
        } else {
            // Two negative ranges, union of them.
            // Can't merge if they have a space inbetween them, unless they also are
            // a not (0..=x), not (y..=max) sets, in which case it can be merged as
            // (x + 1)..=(y - 1) positive range.
            let low = base.0.min(other.0);
            let other_low = base.0.max(other.0);
            let high = base.1.max(other.1);
            let other_high = base.1.min(other.1);
            if other_low > other_high && other_low > other_high.wrapping_add(1) {
                // No overlap
                if low == 0 && high == mask {
                    base.0 = other_high.wrapping_add(1);
                    base.1 = other_low.wrapping_sub(1);
                    base.2 = mask;
                    base.4 = false;
                    true
                } else {
                    false
                }
            } else {
                base.0 = low;
                base.1 = high;
                base.2 = mask;
                true
            }
        }
    }

    if !ops.iter().any(|x| x.if_arithmetic_gt().is_some()) {
        return u64::MAX;
    }

    let mut new_const_remain = u64::MAX;
    let mut i = 0;
    'outer_loop: while i < ops.len() {
        let op = ops[i];
        if let Some(mut value) = check(op) {
            let mut changed = false;
            for j in ((i + 1)..ops.len()).rev() {
                let second = ops[j];
                if let Some(other) = check(second) {
                    if try_merge(&mut value, &other) {
                        ops.swap_remove(j);
                        changed = true;
                    }
                }
            }
            if changed {
                let (min, max, mask, op, set) = value;
                if min == 0 && max == mask {
                    // Always 0/1
                    if set {
                        // Became const 1
                        new_const_remain = 1;
                        ops.swap_remove(i);
                        continue 'outer_loop;
                    } else {
                        // Became const 0, everything is zeroed out
                        ops.clear();
                        return 0;
                    }
                } else {
                    let new = if set {
                        if min == max {
                            let op = ctx.and_const(op, mask);
                            ctx.eq_const(op, min)
                        } else if min == 0 {
                            // max + 1 > x
                            let op = ctx.and_const(op, mask);
                            ctx.gt_const_left(max.wrapping_add(1), op)
                        } else if max == mask {
                            // x > min - 1
                            let op = ctx.and_const(op, mask);
                            ctx.gt_const(op, min.wrapping_sub(1))
                        } else {
                            // x is in range min ..= max
                            // So max - min + 1 > x - min
                            ctx.gt_const_left(
                                max.wrapping_sub(min).wrapping_add(1),
                                ctx.and_const(
                                    ctx.sub_const(op, min),
                                    mask,
                                ),
                            )
                        }
                    } else {
                        if min == max {
                            let op = ctx.and_const(op, mask);
                            ctx.neq_const(op, min)
                        } else if min == 0 {
                            // x > max
                            let op = ctx.and_const(op, mask);
                            ctx.gt_const(op, max)
                        } else if max == mask {
                            // min > x
                            let op = ctx.and_const(op, mask);
                            ctx.gt_const_left(min, op)
                        } else {
                            // x is not in range min ..= max
                            // So x - min > max - min
                            ctx.gt_const(
                                ctx.and_const(
                                    ctx.sub_const(op, min),
                                    mask,
                                ),
                                max.wrapping_sub(min),
                            )
                        }
                    };
                    if new == ctx.const_0() {
                        // Became const 0, everything is zeroed out
                        ops.clear();
                        return 0;
                    }
                    if new == ctx.const_1() {
                        new_const_remain = 1;
                        ops.swap_remove(i);
                        continue 'outer_loop;
                    }
                    ops[i] = new;
                };
            }
        }
        i += 1;
    }
    new_const_remain
}

/// Does (x ^ y) | x => x | y simplification
fn simplify_or_with_xor_of_op<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
) {
    let mut limit = 50u32;
    let mut i = 0;
    while i < ops.len() && limit != 0 {
        let op = ops[i];
        if let Some((l, r)) = op.if_arithmetic_xor() {
            let result = util::arith_parts_to_new_slice(ctx, l, r, ArithOpType::Xor, |slice| {
                for j in 0..ops.len() {
                    if j == i {
                        continue;
                    }
                    let op2 = ops[j];
                    if let Some((l, r)) = op2.if_arithmetic_xor() {
                        let result = util::remove_eq_arith_parts_sorted_and_rejoin(
                            ctx,
                            slice,
                            (l, r, ArithOpType::Xor),
                        );
                        if let Some(result) = result {
                            return Some(result);
                        }
                    } else {
                        if let Some(pos) = slice.iter().position(|&x| x == op2) {
                            let result = util::sorted_arith_chain_remove_one_and_join(
                                ctx,
                                slice,
                                pos,
                                op,
                                ArithOpType::Xor,
                            );
                            return Some(result);
                        }
                    }
                    limit = match limit.checked_sub(slice.len() as u32) {
                        Some(s) => s,
                        None => return None,
                    };
                }
                None
            });
            if let Some(result) = result {
                // Replaces (x ^ y) with y,
                // the operand and `j` index being x can be left as is.
                ops[i] = result;
            }
        }
        i += 1;
    }
}

// "Simplify bitwise or: xor merge"
// Converts [x, y] to [x ^ y] where x and y don't have overlapping
// relevant bit ranges. Then ideally the xor can simplify further.
// Technically valid for any non-overlapping x and y, but limit transformation
// to cases where x and y are xors.
fn simplify_or_merge_xors<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) {
    fn is_xor(op: Operand<'_>) -> bool {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Xor => true,
            _ => false,
        }
    }

    let mut i = 0;
    while i < ops.len() {
        let op = ops[i];
        if is_xor(op) {
            let mut j = i + 1;
            let mut new = op;
            let bits = op.relevant_bits();
            while j < ops.len() {
                let other_op = ops[i];
                if is_xor(other_op) {
                    let other_bits = other_op.relevant_bits();
                    if !bits_overlap(&bits, &other_bits) {
                        new = simplify_xor(new, other_op, ctx, swzb);
                        ops.swap_remove(j);
                        continue; // Without incrementing j
                    }
                }
                j += 1;
            }
            ops[i] = new;
        }
        i += 1;
    }
}

/// For or simplify:
///     Joins any (x == 0) ops to single (x & y & z) == 0 (merge_ty == And)
/// For and simplify:
///     Joins any (x == 0) ops to single (x | y | z) == 0 (merge_ty == Or)
fn simplify_demorgan<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    merge_ty: ArithOpType,
) {
    let mut i = 0;
    let mut op = None;
    while i < ops.len() {
        if let Some((l, r)) = ops[i].if_arithmetic_eq() {
            if r == ctx.const_0() {
                op = Some(l);
                break;
            }
        }
        i = i.wrapping_add(1);
    }
    if let Some(op) = op {
        let replace_pos = i;
        i = i.wrapping_add(1);
        let mut result = op;
        while i < ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic_eq() {
                if r == ctx.const_0() {
                    result = ctx.arithmetic(
                        merge_ty,
                        result,
                        l,
                    );
                    ops.swap_remove(i);
                    // Don't increment i
                    continue;
                }
            }
            i += 1;
        }
        if result != op {
            ops[replace_pos] = ctx.eq_const(result, 0);
        }
    }
}

fn and_masked_precise<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>)> {
    match op.ty() {
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
            arith.right.if_constant().map(|c| (c, arith.left))
        }
        OperandType::Memory(mem) => Some((mem.size.mask(), op)),
        _ => {
            let bits = op.relevant_bits();
            if bits != (0..64) && bits.start < bits.end {
                let low = bits.start;
                let high = 64 - bits.end;
                Some((!0 >> low << low << high >> high, op))
            } else {
                None
            }
        }
    }
}


/// "Simplify bitwise or: merge child ands"
/// Converts things like [x & const1, x & const2] to [x & (const1 | const2)]
///
/// Also used by xors with only_nonoverlapping true
fn simplify_or_merge_child_ands<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
    and_mask: u64,
    arith_ty: ArithOpType,
) -> Result<(), SizeLimitReached> {
    if ops.len() > 16 {
        // The loop below is quadratic complexity, being especially bad
        // if there are lot of masked xors, so give up if there are more
        // ops than usual code would have.
        #[cfg(feature = "fuzz")]
        tls_simplification_incomplete();
        return Ok(());
    }

    let only_nonoverlapping = arith_ty == ArithOpType::Xor;
    let mut i = 0;
    'outer: while i < ops.len() {
        if let Some((constant, val)) = and_masked_precise(ops[i]) {
            let mut j = i + 1;
            while j < ops.len() {
                if let Some((other_constant, other_val)) = and_masked_precise(ops[j]) {
                    if !only_nonoverlapping || other_constant & constant == 0 {
                        let mut buf = SmallVec::new();
                        simplify_or_merge_ands_try_merge(
                            &mut buf,
                            val,
                            other_val,
                            constant,
                            other_constant,
                            arith_ty,
                            ctx,
                        );
                        if !buf.is_empty() {
                            ops.swap_remove(j);
                            ops.swap_remove(i);
                            for op in buf {
                                let op = if and_mask == u64::MAX {
                                    op
                                } else {
                                    simplify_with_and_mask(op, and_mask, ctx, swzb_ctx)
                                };
                                ops.push(op)?;
                            }
                            continue 'outer;
                        }
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
    Ok(())
}

fn simplify_or_merge_ands_try_merge<'e>(
    out: &mut SmallVec<[Operand<'e>; 8]>,
    first: Operand<'e>,
    second: Operand<'e>,
    first_c: u64,
    second_c: u64,
    arith_ty: ArithOpType,
    ctx: OperandCtx<'e>,
) {
    let _ = ctx.simplify_temp_stack().alloc(|slice| {
        collect_arith_ops(first, slice, arith_ty, 16)?;
        ctx.simplify_temp_stack().alloc(|other_slice| {
            collect_arith_ops(second, other_slice, arith_ty, 16)?;
            let mut i = 0;
            'outer: while i < slice.len() {
                let mut j = 0;
                while j < other_slice.len() {
                    let a = slice[i];
                    let b = other_slice[j];
                    if let Some(result) = try_merge_ands(a, b, first_c, second_c, ctx) {
                        out.push(ctx.and_const(result, first_c | second_c));
                        slice.swap_remove(i);
                        other_slice.swap_remove(j);
                        continue 'outer;
                    }
                    j += 1;
                }
                i += 1;
            }
            // Rejoin remaining operands to a new arith if any were removed.
            // Assume that they can just be rebuilt without simplification
            if !out.is_empty() {
                let iter = slice.iter().rev().copied();
                if let Some(first) = util::intern_arith_ops_to_tree(ctx, iter, arith_ty) {
                    out.push(ctx.and_const(first, first_c));
                }
                let iter = other_slice.iter().rev().copied();
                if let Some(second) = util::intern_arith_ops_to_tree(ctx, iter, arith_ty) {
                    out.push(ctx.and_const(second, second_c));
                }
            }
            Result::<(), SizeLimitReached>::Ok(())
        })
    });
}

/// If a == (b >> c), return Some(c)
fn equal_when_shifted_right<'e>(
    a: Operand<'e>,
    b: Operand<'e>,
) -> Option<u8> {
    use ArithOpType::*;
    match *a.ty() {
        OperandType::Arithmetic(ref a_arith) => {
            if matches!(a_arith.ty, And | Or | Xor) {
                match b.ty() {
                    OperandType::Arithmetic(b_arith) => {
                        if matches!(b_arith.ty, And | Or | Xor) {
                            let left = equal_when_shifted_right(a_arith.left, b_arith.left)?;
                            let right = equal_when_shifted_right(a_arith.right, b_arith.right)?;
                            if left == right {
                                return Some(left);
                            }
                        }
                    }
                    _ => (),
                }
            } else if a_arith.ty == Rsh {
                if a_arith.left == b {
                    if let Some(c) = a_arith.right.if_constant() {
                        return Some(c as u8);
                    }
                }
            }
        }
        OperandType::Constant(c) => {
            if let Some(c2) = b.if_constant() {
                let a_zeros = a.relevant_bits().start;
                let b_zeros = b.relevant_bits().start;
                let shift = b_zeros.checked_sub(a_zeros)?;
                if c2.wrapping_shr(shift as u32) == c {
                    return Some(shift);
                }
            }
        }
        _ => (),
    }
    None
}

// Simplify or: merge comparisions
// Converts
// (c > x) | (c == x) to (c + 1 > x),
//      More general: (c1 > x - c2) | (c1 + c2) == x to (c1 + 1 > x - c2)
// (x > c) | (x == c) to (x > c - 1).
//      (x > y + c) | (x - y == c) to (x > y + (c - 1))
//      as a variation due to == canonicalizing c alone to right.
// (x == 0) | (x == 1) to (2 > x)
// Cannot do for values that can overflow, so just limit it to constants for now.
// (Well, could do (c + 1 > x) | (c == max_value), but that isn't really simpler)
fn simplify_or_merge_comparisions<'e>(ops: &mut Slice<'e>, ctx: OperandCtx<'e>) {
    struct Match<'e> {
        op: Operand<'e>,
        // y in (x > (y + c))
        const_side_op: Option<Operand<'e>>,
        ty: MatchType,
        constant: u64,
    }

    #[derive(Eq, PartialEq, Copy, Clone)]
    enum MatchType {
        ConstantGreater,
        ConstantLess,
        Equal,
    }

    fn check_match<'e>(op: Operand<'e>) -> Option<Match<'e>> {
        match op.ty() {
            OperandType::Arithmetic(arith) => {
                let left = arith.left;
                let right = arith.right;
                match arith.ty {
                    ArithOpType::Equal => {
                        let c = right.if_constant()?;
                        return Some(Match {
                            op: left,
                            const_side_op: None,
                            constant: c,
                            ty: MatchType::Equal,
                        });
                    }
                    ArithOpType::GreaterThan => {
                        if let Some(c) = left.if_constant() {
                            return Some(Match {
                                op: right,
                                const_side_op: None,
                                constant: c,
                                ty: MatchType::ConstantGreater,
                            });
                        }
                        if let Some(c) = right.if_constant() {
                            return Some(Match {
                                op: left,
                                const_side_op: None,
                                constant: c,
                                ty: MatchType::ConstantLess,
                            });
                        } else if let Some((l, r)) = right.if_arithmetic_add() {
                            if let Some(c) = r.if_constant() {
                                return Some(Match {
                                    op: left,
                                    const_side_op: Some(l),
                                    constant: c,
                                    ty: MatchType::ConstantLess,
                                });
                            }
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
        None
    }

    let mut i = 0;
    'outer: while i < ops.len() {
        if let Some(m1) = check_match(ops[i]) {
            let mut j = i + 1;
            'inner: while j < ops.len() {
                if let Some(m2) = check_match(ops[j]) {
                    match (m1.ty, m2.ty) {
                        (MatchType::ConstantGreater, MatchType::Equal) |
                            (MatchType::Equal, MatchType::ConstantGreater) =>
                        {
                            if m1.const_side_op.is_some() || m2.const_side_op.is_some() {
                                // May be able to do something, but not considering it now.
                                continue 'inner;
                            }

                            // (c1 > y - c2) | (c1 + c2) == y to (c1 + 1 > y - c2)
                            let (gt, c1, eq, mut eq_c) =
                                match m1.ty == MatchType::ConstantGreater
                            {
                                true => (m1.op, m1.constant, m2.op, m2.constant),
                                false => (m2.op, m2.constant, m1.op, m1.constant),
                            };
                            let (gt_inner, gt_mask) = Operand::and_masked(gt);
                            let (y, c2) = gt_inner.if_arithmetic_sub()
                                .and_then(|(l, r)| {
                                    Some((l, r.if_constant()?))
                                })
                                .unwrap_or((gt_inner, 0));
                            let mut values_match = if (y, gt_mask) == Operand::and_masked(eq) {
                                true
                            } else {
                                // Can also be that y == eq == (smth & ff),
                                // in which case the previous check would have done
                                // (y, u64::MAX) != (smth, ff)
                                y == eq && gt_mask == u64::MAX
                            };
                            if !values_match {
                                // If values didn't match, check if they match when eq is shifted
                                // E.g. y = ((rcx >> 10) & ff),
                                //      eq = (rcx & ff0000)
                                // If they do, shift eq_c as well
                                if gt_mask == u64::MAX {
                                    if let Some(shift) = equal_when_shifted_right(y, eq) {
                                        eq_c = eq_c.wrapping_shr(shift as u32);
                                        values_match = true;
                                    }
                                }
                            }
                            let constants_match = c1.checked_add(c2)
                                .filter(|&c| c == eq_c)
                                .is_some();
                            if constants_match && values_match {
                                // min/max edge cases can be handled by gt simplification,
                                // don't do them here.
                                if let Some(new_c) = c1.checked_add(1) {
                                    ops[i] = ctx.gt_const_left(new_c, gt);
                                    ops.swap_remove(j);
                                    continue 'outer;
                                }
                            }
                        }
                        (MatchType::ConstantLess, MatchType::Equal) |
                            (MatchType::Equal, MatchType::ConstantLess) =>
                        {
                            let (eq, cl) = match m1.ty == MatchType::Equal {
                                true => (&m1, &m2),
                                false => (&m2, &m1),
                            };
                            if let Some(new_c) = eq.constant.checked_sub(1) {
                                if let Some(const_side) = cl.const_side_op {
                                    if is_eq_sub_for_gt(cl.op, const_side, eq.op) {
                                        ops[i] = ctx.gt(
                                            cl.op,
                                            ctx.add_const(
                                                const_side,
                                                new_c,
                                            ),
                                        );
                                        ops.swap_remove(j);
                                        continue 'outer;
                                    }
                                } else {
                                    if eq.constant == cl.constant && eq.op == cl.op {
                                        ops[i] = ctx.gt_const(eq.op, new_c);
                                        ops.swap_remove(j);
                                        continue 'outer;
                                    }
                                }
                            }
                        }
                        (MatchType::Equal, MatchType::Equal) => {
                            // No need to check const_side_op since it can't be set on equal
                            debug_assert!(m1.const_side_op.is_none());
                            debug_assert!(m2.const_side_op.is_none());
                            if m1.constant.min(m2.constant) == 0 &&
                                m1.constant.max(m2.constant) == 1 &&
                                m1.op == m2.op
                            {
                                ops[i] = ctx.gt_const_left(2, m1.op);
                                ops.swap_remove(j);
                                continue 'outer;
                            }
                        }
                        _ => (),
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
}

/// Assuming that there are two expressions,
/// `lhs > (rhs + C)` and `lhs == rhs + C`,
/// and that the equality one has been transformed to `compare == C`,
/// verifies that `compare` can be reverted to `lhs` and `rhs`.
///
/// Simply put, if (lhs - rhs) == compare, return true.
/// However, there may be and masks here,
/// so more correct implementation would be something like
/// simplify(`lhs == rhs + 5000`).left == compare
///
/// Helper for simplify_or_merge_comparisions.
fn is_eq_sub_for_gt<'e>(
    lhs: Operand<'e>,
    rhs: Operand<'e>,
    compare: Operand<'e>,
) -> bool {
    let (compare, c_mask) = Operand::and_masked(compare);
    let (lhs, l_mask) = Operand::and_masked(lhs);
    let (rhs, r_mask) = Operand::and_masked(rhs);
    // Not super sure if these mask checks are valid or enough..
    if c_mask != u64::MAX {
        if l_mask != u64::MAX && c_mask != l_mask {
            return false;
        }
        if r_mask != u64::MAX && c_mask != r_mask {
            return false;
        }
    } else {
        if l_mask != u64::MAX || r_mask != u64::MAX {
            return false;
        }
    }

    let mut compare_iter = IterAddSubArithOps::new(compare);
    let mut lhs_iter = IterAddSubArithOps::new(lhs);
    let mut rhs_iter = IterAddSubArithOps::new(rhs);
    let mut lhs_next = lhs_iter.next();
    let mut rhs_next = rhs_iter.next();
    while let Some((x, negate)) = compare_iter.next() {
        if Some((x, negate)) == lhs_next {
            lhs_next = lhs_iter.next();
        } else if Some((x, !negate)) == rhs_next {
            rhs_next = rhs_iter.next();
        } else {
            return false;
        }
    }
    lhs_next.is_none() && rhs_next.is_none()
}

/// Does not collect constants into ops, but returns them added together instead.
fn collect_add_ops<'e>(
    s: Operand<'e>,
    ops: &mut AddSlice<'e>,
    out_mask: u64,
    negate: bool,
) -> Result<u64, SizeLimitReached> {
    fn recurse_with_mask<'e>(
        s: Operand<'e>,
        ops: &mut AddSlice<'e>,
        out_mask: u64,
        negate: bool,
    ) -> Result<u64, SizeLimitReached> {
        match s.ty() {
            OperandType::Arithmetic(arith) if {
                arith.ty == ArithOpType::Add || arith.ty== ArithOpType::Sub
            } => {
                let const1 = recurse_with_mask(arith.left, ops, out_mask, negate)?;
                let negate_right = match arith.ty {
                    ArithOpType::Add => negate,
                    _ => !negate,
                };
                let const2 = recurse_with_mask(arith.right, ops, out_mask, negate_right)?;
                Ok(const1.wrapping_add(const2))
            }
            _ => {
                if let Some((l, r)) = s.if_arithmetic_and() {
                    if let Some(c) = r.if_constant() {
                        if c & out_mask == out_mask {
                            return recurse_with_mask(l, ops, out_mask, negate);
                        }
                    }
                }
                if let Some(c) = s.if_constant() {
                    if negate {
                        Ok(0u64.wrapping_sub(c))
                    } else {
                        Ok(c)
                    }
                } else {
                    ops.push((s, negate))?;
                    Ok(0)
                }
            }
        }
    }
    if out_mask == u64::MAX {
        collect_add_ops_no_mask(s, ops, negate)
    } else {
        recurse_with_mask(s, ops, out_mask, negate)
    }
}

fn collect_add_ops_no_mask<'e>(
    s: Operand<'e>,
    ops: &mut AddSlice<'e>,
    negate: bool,
) -> Result<u64, SizeLimitReached> {
    // This will rely on simplification guaranteeing that:
    // 1) Add/sub Operand is formed as (((a + b) + c) - d), rhs of each arithmetic expression
    //   cannot be an additional add/sub term.
    // 2) Constant is either the rightmost operand, or if the expression is just subtractions
    //   and positive constant, it will be the leftmost.
    match s.ty() {
        OperandType::Arithmetic(arith) if {
            arith.ty == ArithOpType::Add || arith.ty== ArithOpType::Sub
        } => {
            let mut constant;
            let mut pos;
            if let Some(c) = arith.right.if_constant() {
                let negate_right = match arith.ty {
                    ArithOpType::Add => negate,
                    _ => !negate,
                };
                if negate_right {
                    constant = 0u64.wrapping_sub(c);
                } else {
                    constant = c;
                }
                pos = arith.left;
            } else {
                constant = 0;
                pos = s;
            };
            loop {
                match pos.ty() {
                    OperandType::Arithmetic(arith) if {
                        arith.ty == ArithOpType::Add || arith.ty== ArithOpType::Sub
                    } => {
                        let negate_right = match arith.ty {
                            ArithOpType::Add => negate,
                            _ => !negate,
                        };
                        ops.push((arith.right, negate_right))?;
                        pos = arith.left;
                    }
                    _ => {
                        if let Some(c) = pos.if_constant() {
                            // Constant as the leftmost term,
                            // there cannot have been constant on the right then.
                            // Expect `(0 - rax) - 1` is valid too ._.
                            // Leftmost constants should probably be removed..
                            if negate {
                                constant = constant.wrapping_sub(c);
                            } else {
                                constant = constant.wrapping_add(c);
                            }
                        } else {
                            ops.push((pos, negate))?;
                        }
                        break;
                    }
                }
            }
            Ok(constant)
        }
        _ => {
            if let Some(c) = s.if_constant() {
                if negate {
                    Ok(0u64.wrapping_sub(c))
                } else {
                    Ok(c)
                }
            } else {
                ops.push((s, negate))?;
                Ok(0)
            }
        }
    }
}

/// Unwraps a tree chaining arith operation to vector of the operands.
///
/// If the limit is set, caller should verify that it was not hit (ops.len() > limit),
/// as not all ops will end up being collected.
fn collect_arith_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e>,
    arith_type: ArithOpType,
    limit: usize,
) -> Result<(), SizeLimitReached> {
    // Assuming that these operands are always in form (((x | y) | z) | w) | c
    // So only recursing on left is enough.
    let mut s = s;
    for _ in ops.len()..limit {
        match s.ty() {
            OperandType::Arithmetic(arith) if arith.ty == arith_type => {
                s = arith.left;
                ops.push(arith.right)?;
            }
            _ => {
                ops.push(s)?;
                return Ok(());
            }
        }
    }
    if ops.len() >= limit {
        if ops.len() == limit {
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
        }
        return Err(SizeLimitReached);
    }
    Ok(())
}

fn collect_mul_ops<'e>(
    ctx: OperandCtx<'e>,
    s: Operand<'e>,
    ops: &mut Slice<'e>,
) -> Result<(), SizeLimitReached> {
    match s.ty() {
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Mul => {
            collect_mul_ops(ctx, arith.left, ops)?;
            ops.push(arith.right)?;
        }
        // Convert x << constant to x * (1 << constant) for simplifications
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Lsh => {
            if let Some(c) = arith.right.if_constant() {
                collect_mul_ops(ctx, arith.left, ops)?;
                ops.push(ctx.constant(1u64.wrapping_shl(c as u32)))?;
            } else {
                ops.push(s)?;
            }
        }
        _ => ops.push(s)?,
    }
    Ok(())
}

fn collect_and_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e>,
    limit: usize,
) -> Result<(), SizeLimitReached> {
    collect_arith_ops(s, ops, ArithOpType::And, limit)
}

fn collect_or_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e>,
) -> Result<(), SizeLimitReached> {
    collect_arith_ops(s, ops, ArithOpType::Or, usize::MAX)
}

fn collect_xor_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e>,
    limit: usize,
) -> Result<(), SizeLimitReached> {
    collect_arith_ops(s, ops, ArithOpType::Xor, limit)
}

fn collect_masked_or_ops<'e>(
    s: Operand<'e>,
    ops: &mut MaskedOpSlice<'e>,
    limit: usize,
    mask: u64,
) -> Result<(), SizeLimitReached> {
    collect_masked_ops(s, ops, limit, mask, ArithOpType::Or)
}

fn collect_masked_xor_ops<'e>(
    s: Operand<'e>,
    ops: &mut MaskedOpSlice<'e>,
    limit: usize,
    mask: u64,
) -> Result<(), SizeLimitReached> {
    collect_masked_ops(s, ops, limit, mask, ArithOpType::Xor)
}

fn collect_masked_ops<'e>(
    s: Operand<'e>,
    ops: &mut MaskedOpSlice<'e>,
    limit: usize,
    mut mask: u64,
    arith_type: ArithOpType,
) -> Result<(), SizeLimitReached> {
    let mut s = s;
    for _ in ops.len()..limit {
        match s.ty() {
            OperandType::Arithmetic(arith) => {
                if arith.ty == arith_type {
                    s = arith.left;
                    if let Some((l, r)) = arith.right.if_and_with_const() {
                        collect_masked_ops(l, ops, limit, mask & r, arith_type)?;
                    } else {
                        ops.push((arith.right, mask))?;
                    }
                    continue;
                } else if arith.ty == ArithOpType::And {
                    if let Some(r) = arith.right.if_constant() {
                        mask &= r;
                        s = arith.left;
                        continue;
                    }
                }
            }
            _ => (),
        }
        ops.push((s, mask))?;
        return Ok(());
    }
    if ops.len() >= limit {
        if ops.len() == limit {
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
        }
        return Err(SizeLimitReached);
    }
    Ok(())
}

/// Return (base, (offset, len, value_offset, mask))
/// Mask is u64::MAX if there is no explicit and mask
/// (Even if the memory op is not Mem64)
///
/// E.g. Mem32[x + 100] => (x, (100, 4, 0, u64::MAX))
///     (Mem32[x + 100]) << 20 => (x, (100, 4, 4, u64::MAX))
///     (Mem32[x + 100] & f0f0f0f0) << 20 => (x, (100, 4, 4, f0f0f0f0))
fn is_offset_mem<'e>(
    mut op: Operand<'e>,
) -> Option<(Operand<'e>, (u64, u32, u32, u64))> {
    // 3 is enough to have shift + and + actual mem
    let mut loop_limit = 3;
    let mut result = (op, (0, 0, 0, u64::MAX));
    let mut rsh_bytes = 0;
    loop {
        if loop_limit == 0 {
            return None;
        }
        match op.ty() {
            OperandType::Memory(mem) => {
                let (base, offset) = mem.address();
                result.0 = base;
                // If and mask existed, length is determined by it, but otherwise
                // set it to mem size
                result.0 = base;
                result.1.0 = offset.wrapping_add(rsh_bytes as u64);
                if result.1.1 == 0 {
                    let len = mem.size.bits() / 8;
                    result.1.1 = len.wrapping_sub(rsh_bytes);
                }
                return Some(result);
            }
            OperandType::Arithmetic(arith) => {
                if matches!(arith.ty, ArithOpType::Lsh | ArithOpType::Rsh | ArithOpType::And) {
                    if let Some(c) = arith.right.if_constant() {
                        if arith.ty == ArithOpType::Lsh {
                            if c & 0x7 == 0 {
                                result.1.2 = (c / 8) as u32;
                                // result.1.1 is only nonzero if and was seen
                                if result.1.1 != 0 {
                                    // Mask is usually be inside the shift.
                                    // (x & mask) << shift,
                                    // but here it wasn't, so shift the mask
                                    result.1.3 = result.1.3.wrapping_shr(c as u32);
                                }
                            } else {
                                return None;
                            }
                        } else if arith.ty == ArithOpType::Rsh {
                            if c & 0x7 == 0 {
                                // Increases offset by this and decreases length by this
                                rsh_bytes = (c / 8) as u32;
                            } else {
                                return None;
                            }
                        } else {
                            // And
                            let relbits = arith.right.relevant_bits();
                            let start = relbits.start / 8;
                            let end = relbits.end.wrapping_add(7) / 8;
                            let len = end.wrapping_sub(start);
                            result.1.1 = (len as u32).wrapping_sub(rsh_bytes);
                            result.1.3 = c;
                        }
                        op = arith.left;
                        loop_limit -= 1;
                        continue;
                    }
                }
                return None;
            }
            _ => return None,
        }
    }
}

/// The returned result may need an additional and mask
/// if there is hole between the input values.
/// (Mem16[x] and (Mem8[x + 3] << 18) are merged to Mem32[x]
/// which the caller has to mask as Mem32[x] & ff00_ffff.)
///
/// The masking is not done here, expecting that the caller
/// will eventually mask it anyways, but not sure if that
/// is too complex to follow and verify correct..
///
/// shift and other_shift are (offset, len, left_shift_in_operand)
/// E.g. `Mem32[x + 6] << 0x18` => (6, 4, 3, u64::MAX)
/// Same as in is_offset_mem return value.
fn try_merge_memory<'e>(
    val: Operand<'e>,
    shift: (u64, u32, u32, u64),
    other_shift: (u64, u32, u32, u64),
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    let (shift, other_shift) = match shift.2 < other_shift.2 {
        true => (shift, other_shift),
        false => (other_shift, shift),
    };
    let (off1, len1, val_off1, mask1) = shift;
    let (off2, len2, val_off2, mask2) = other_shift;
    let base = off1.wrapping_sub(u64::from(val_off1));
    if off2.wrapping_sub(u64::from(val_off2)) != base {
        return None;
    }
    let len1 = if val_off1.wrapping_add(len1) > 8 {
        8u32.wrapping_sub(val_off1)
    } else {
        len1
    };
    let len2 = if val_off2.wrapping_add(len2) > 8 {
        8u32.wrapping_sub(val_off2)
    } else {
        len2
    };
    let len = val_off1.wrapping_add(len1).max(val_off2.wrapping_add(len2))
        .wrapping_sub(val_off1);
    let mut oper = match len {
        1 => ctx.mem8(val, off1),
        2 => ctx.mem16(val, off1),
        3 => ctx.and_const(ctx.mem32(val, off1), 0x00ff_ffff),
        4 => ctx.mem32(val, off1),
        5 | 6 | 7 => ctx.and_const(ctx.mem64(val, off1), u64::MAX >> ((8 - len) << 3)),
        8 => ctx.mem64(val, off1),
        _ => return None,
    };
    if val_off1 != 0 {
        oper = ctx.lsh_const(
            oper,
            (val_off1 << 3).into(),
        );
    }

    if mask1 != u64::MAX || mask2 != u64::MAX {
        // If mask1 is not u64::MAX use that, otherwise 1.checked_shl(len * 8).unwrap_or(0)
        // but write the conditions like this as a micro-optimization
        let mask1 = if mask1 != u64::MAX || len1 >= 8 {
            mask1
        } else {
            (1u64 << (len1 * 8)).wrapping_sub(1)
        };
        let mask2 = if mask2 != u64::MAX || len2 >= 8 {
            mask2
        } else {
            (1u64 << (len2 * 8)).wrapping_sub(1)
        };
        let mask = (mask1 << (val_off1 * 8)) | (mask2 << (val_off2 * 8));
        oper = ctx.and_const(oper, mask);
    }
    Some(oper)
}

fn is_sub_with_lhs_const<'e>(left: Operand<'e>) -> Option<u64> {
    let mut pos = left;
    while let Some((l, _)) = pos.if_arithmetic_sub() {
        pos = l;
    }
    pos.if_constant()
}

fn simplify_add_sub_const_with_lhs_const<'e>(
    left: Operand<'e>,
    right: u64,
    ty: ArithOpType,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    ctx.simplify_temp_stack()
        .alloc(|ops| {
            let c1 = collect_add_ops_no_mask(left, ops, false)?;
            simplify_collected_add_sub_ops(ops, ctx, right.wrapping_add(c1))?;
            Ok(add_sub_ops_to_tree(ops, ctx))
        }).unwrap_or_else(|SizeLimitReached| {
            let arith = ArithOperand {
                ty,
                left: left,
                right: ctx.constant(right),
            };
            ctx.intern(OperandType::Arithmetic(arith))
        })
}

pub fn simplify_add_const<'e>(
    left: Operand<'e>,
    right: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    simplify_add_const_op(left, right, None, ctx)
}

/// Allows providing `right_op` if `ctx.constant(right)` is already known.
/// Seems to be true pretty often due to exec_state.resolve() having to resolve
/// additions.
fn simplify_add_const_op<'e>(
    mut left: Operand<'e>,
    mut right: u64,
    mut right_op: Option<Operand<'e>>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if right == 0 {
        return left;
    }
    let init_right = right;
    // Check if left is x +/- const
    // Useful since push/pop/esp/ebp offsets are really common.
    match left.ty() {
        OperandType::Arithmetic(ref arith) => {
            if let Some(c) = arith.right.if_constant() {
                if arith.ty == ArithOpType::Add {
                    // (x + c1) + c2 => x + (c2 + c1)
                    left = arith.left;
                    right = right.wrapping_add(c);
                    right_op = None;
                } else if arith.ty == ArithOpType::Sub {
                    // (x - c1) + c2 => x + (c2 - c1)
                    left = arith.left;
                    right = right.wrapping_sub(c);
                    right_op = None;
                }
            }
        }
        OperandType::Constant(c) => return ctx.constant(c.wrapping_add(right)),
        _ => (),
    }
    if right == 0 {
        return left;
    }
    // If right != init_right, there was a constant on right already
    let negate_right = right > 0x8000_0000_0000_0000;
    if right == init_right || !negate_right {
        if let Some(lhs_const) = is_sub_with_lhs_const(left) {
            if !negate_right || lhs_const != 0 {
                return simplify_add_sub_const_with_lhs_const(left, right, ArithOpType::Add, ctx);
            }
        }
    }
    let arith = if negate_right {
        ArithOperand {
            ty: ArithOpType::Sub,
            left: left,
            right: ctx.constant(0u64.wrapping_sub(right)),
        }
    } else {
        ArithOperand {
            ty: ArithOpType::Add,
            left: left,
            right: right_op.unwrap_or_else(|| ctx.constant(right)),
        }
    };
    return ctx.intern(OperandType::Arithmetic(arith));
}

pub fn simplify_sub_const<'e>(
    left: Operand<'e>,
    right: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    simplify_sub_const_op(left, right, None, ctx)
}

/// Allows providing `right_op` if `ctx.constant(right)` is already known.
/// Seems to be true pretty often due to exec_state.resolve() having to resolve
/// additions.
fn simplify_sub_const_op<'e>(
    mut left: Operand<'e>,
    mut right: u64,
    mut right_op: Option<Operand<'e>>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if right == 0 {
        return left;
    }
    let init_right = right;
    // Check if left is x +/- const
    // Useful since push/pop/esp/ebp offsets are really common.
    match left.ty() {
        OperandType::Arithmetic(ref arith) => {
            if let Some(c) = arith.right.if_constant() {
                if arith.ty == ArithOpType::Add {
                    // (x + c1) - c2 => x - (c2 - c1)
                    left = arith.left;
                    right = right.wrapping_sub(c);
                    right_op = None;
                } else if arith.ty == ArithOpType::Sub {
                    // (x - c1) - c2 => x - (c1 + c2)
                    left = arith.left;
                    right = c.wrapping_add(right);
                    right_op = None;
                }
            }
        }
        OperandType::Constant(c) => return ctx.constant(c.wrapping_sub(right)),
        _ => (),
    }
    if right == 0 {
        return left;
    }
    // If right != init_right, there was a constant on right already
    let negate_right = right < 0x8000_0000_0000_0000;
    if right == init_right || !negate_right {
        if let Some(lhs_const) = is_sub_with_lhs_const(left) {
            if !negate_right || lhs_const != 0 {
                let right = 0u64.wrapping_sub(right);
                return simplify_add_sub_const_with_lhs_const(left, right, ArithOpType::Sub, ctx);
            }
        }
    }
    let arith = if !negate_right {
        ArithOperand {
            ty: ArithOpType::Add,
            left: left,
            right: ctx.constant(0u64.wrapping_sub(right)),
        }
    } else {
        ArithOperand {
            ty: ArithOpType::Sub,
            left: left,
            right: right_op.unwrap_or_else(|| ctx.constant(right)),
        }
    };
    ctx.intern(OperandType::Arithmetic(arith))
}

pub fn simplify_add_sub<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    is_sub: bool,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if !is_sub {
        if let Some(c) = left.if_constant() {
            return simplify_add_const_op(right, c, Some(left), ctx);
        }
    }
    if let Some(c) = right.if_constant() {
        if is_sub {
            return simplify_sub_const_op(left, c, Some(right), ctx);
        } else {
            return simplify_add_const_op(left, c, Some(right), ctx);
        }
    }
    ctx.simplify_temp_stack().alloc(|ops| {
        simplify_add_sub_ops(ops, left, right, is_sub, u64::MAX, ctx)?;
        Ok(add_sub_ops_to_tree(ops, ctx))
    }).unwrap_or_else(|SizeLimitReached| {
        let arith = ArithOperand {
            ty: if is_sub { ArithOpType::Sub } else { ArithOpType::Add },
            left: left,
            right: right,
        };
        return ctx.intern(OperandType::Arithmetic(arith));
    })
}

fn add_sub_ops_to_tree<'e>(
    ops: &mut AddSlice<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    use self::ArithOpType::*;

    let const_sum = if let Some(c) = ops.last().filter(|x| x.0.if_constant().is_some()) {
        // If the constant is only positive term, don't place it at the end
        if c.1 == false && (&ops[..ops.len() - 1]).iter().all(|x| x.1 == true) {
            None
        } else {
            ops.pop()
        }
    } else {
        None
    };
    // Place non-negated terms last so the simplified result doesn't become
    // (0 - x) + y
    heapsort::sort_by(ops, |&(ref a_val, a_neg), &(ref b_val, b_neg)| {
        (b_neg, b_val) < (a_neg, a_val)
    });
    let mut tree = match ops.pop() {
        Some((s, neg)) => {
            match neg {
                false => s,
                true => {
                    let arith = ArithOperand {
                        ty: Sub,
                        left: ctx.const_0(),
                        right: s,
                    };
                    ctx.intern(OperandType::Arithmetic(arith))
                }
            }
        }
        None => match const_sum {
            Some((op, neg)) => {
                debug_assert!(neg == false);
                return op;
            }
            None => return ctx.const_0(),
        }
    };
    while let Some((op, neg)) = ops.pop() {
        let arith = ArithOperand {
            ty: if neg { Sub } else { Add },
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    if let Some((op, neg)) = const_sum {
        let arith = ArithOperand {
            ty: if neg { Sub } else { Add },
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    tree
}

pub fn simplify_mul<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    let left_rel = left.relevant_bits();
    let right_rel = right.relevant_bits();
    // Guaranteed to overflow to 0
    if left_rel.start.wrapping_add(right_rel.start) >= 64 {
        return ctx.const_0();
    }
    let const_other = Operand::either(left, right, |x| x.if_constant());
    if let Some((c, other)) = const_other {
        // Go through lsh simplification for power of two, it will still intern Mul
        // if the shift is small enough.
        if c.wrapping_sub(1) & c == 0 {
            let shift = c.trailing_zeros() as u8;
            let swzb = &mut SimplifyWithZeroBits::default();
            return simplify_lsh_const(other, shift, ctx, swzb);
        }
        match c {
            0 => return ctx.const_0(),
            1 => return other,
            _ => (),
        }
        if let Some((l, r)) = check_quick_arith_simplify(left, right) {
            let arith = ArithOperand {
                ty: ArithOpType::Mul,
                left: l,
                right: r,
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        }
    }

    ctx.simplify_temp_stack().alloc(|slice| {
        collect_mul_ops(ctx, left, slice)
            .and_then(|()| collect_mul_ops(ctx, right, slice))
            .and_then(|()| simplify_mul_ops(ctx, slice))
            .unwrap_or_else(|_| {
                let arith = ArithOperand {
                    ty: ArithOpType::Mul,
                    left: left,
                    right: right,
                };
                ctx.intern(OperandType::Arithmetic(arith))
            })
    })
}

pub fn simplify_mul_high<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if let Some(l) = left.if_constant() {
        if let Some(r) = right.if_constant() {
            return ctx.constant(((l as u128 * r as u128) >> 64) as u64);
        }
    }
    if left.relevant_bits().end.wrapping_add(right.relevant_bits().end) <= 64 {
        // Multiplication won't overflow to high u64
        return ctx.const_0();
    }
    let (left, right) = if left.if_constant().is_some() {
        (right, left)
    } else if right.if_constant().is_some() {
        (left, right)
    } else if right < left {
        (right, left)
    } else {
        (left, right)
    };
    let ty = OperandType::Arithmetic(ArithOperand {
        ty: ArithOpType::MulHigh,
        left,
        right,
    });
    ctx.intern(ty)
}

fn simplify_mul_ops<'e>(
    ctx: OperandCtx<'e>,
    ops: &mut Slice<'e>,
) -> Result<Operand<'e>, SizeLimitReached> {
    let mut const_product = ops.iter().flat_map(|x| x.if_constant())
        .fold(1u64, |product, x| product.wrapping_mul(x));
    if const_product == 0 {
        return Ok(ctx.const_0());
    }
    ops.retain(|x| x.if_constant().is_none());
    if ops.is_empty() {
        return Ok(ctx.constant(const_product));
    }
    heapsort::sort(ops);
    if const_product != 1 {
        let mut changed;
        // Apply constant c * (x + y) => (c * x + c * y) as much as possible.
        // (This repeats at least if (c * x + c * y) => c * y due to c * x == 0)
        loop {
            changed = false;
            for i in 0..ops.len() {
                if simplify_mul_should_apply_constant(ops[i]) {
                    let new = simplify_mul_apply_constant(ops[i], const_product, ctx);
                    ops.swap_remove(i);
                    collect_mul_ops(ctx, new, ops)?;
                    changed = true;
                    break;
                }
                let new = simplify_mul_try_mul_constants(ops[i], const_product, ctx);
                if let Some(new) = new {
                    ops.swap_remove(i);
                    collect_mul_ops(ctx, new, ops)?;
                    changed = true;
                    break;
                }
            }
            if changed {
                const_product = ops.iter().flat_map(|x| x.if_constant())
                    .fold(1u64, |product, x| product.wrapping_mul(x));
                ops.retain(|x| x.if_constant().is_none());
                heapsort::sort(ops);
                if const_product == 0 {
                    return Ok(ctx.const_0());
                } else if const_product == 1 {
                    break;
                }
            } else {
                break;
            }
        }
        if changed {
            const_product = 1;
        }
    }
    match ops.len() {
        0 => return Ok(ctx.constant(const_product)),
        1 if const_product == 1 => return Ok(ops[0]),
        _ => (),
    };
    let mut tree = ops.pop()
        .unwrap_or_else(|| ctx.const_1());
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::Mul,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    // Make constant always be on right of simplified mul
    if const_product != 1 {
        // Go through lsh simplification for power of two, it will still intern Mul
        // if the shift is small enough.
        if const_product.wrapping_sub(1) & const_product == 0 {
            let shift = const_product.trailing_zeros() as u8;
            let swzb = &mut SimplifyWithZeroBits::default();
            return Ok(simplify_lsh_const(tree, shift, ctx, swzb));
        }
        let arith = ArithOperand {
            ty: ArithOpType::Mul,
            left: tree,
            right: ctx.constant(const_product),
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    Ok(tree)
}

// For converting c * (x + y) to (c * x + c * y)
fn simplify_mul_should_apply_constant(op: Operand<'_>) -> bool {
    fn inner(op: Operand<'_>) -> bool {
        match op.ty() {
            OperandType::Arithmetic(arith) => match arith.ty {
                ArithOpType::Add | ArithOpType::Sub => {
                    inner(arith.left) && inner(arith.right)
                }
                ArithOpType::Mul => arith.right.if_constant().is_some(),
                _ => false,
            },
            OperandType::Constant(_) => true,
            _ => false,
        }
    }
    match op.ty() {
        OperandType::Arithmetic(arith) if {
            arith.ty == ArithOpType::Add || arith.ty == ArithOpType::Sub
        } => {
            inner(arith.left) && inner(arith.right)
        }
        _ => false,
    }
}

fn simplify_mul_apply_constant<'e>(
    op: Operand<'e>,
    val: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    let constant = ctx.constant(val);
    fn inner<'e>(op: Operand<'e>, constant: Operand<'e>, ctx: OperandCtx<'e>) -> Operand<'e> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Add => {
                ctx.add(inner(arith.left, constant, ctx), inner(arith.right, constant, ctx))
            }
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Sub => {
                ctx.sub(inner(arith.left, constant, ctx), inner(arith.right, constant, ctx))
            }
            _ => ctx.mul(constant, op)
        }
    }
    let new = inner(op, constant, ctx);
    new
}

// For converting c * (c2 + y) to (c_mul_c2 + c * y)
fn simplify_mul_try_mul_constants<'e>(
    op: Operand<'e>,
    c: u64,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    match op.ty() {
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Add => {
            arith.right.if_constant()
                .map(|c2| {
                    let multiplied = c2.wrapping_mul(c);
                    ctx.add_const(ctx.mul_const(arith.left, c), multiplied)
                })
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Sub => {
            match (arith.left.ty(), arith.right.ty()) {
                (&OperandType::Constant(c2), _) => {
                    let multiplied = c2.wrapping_mul(c);
                    Some(ctx.sub_const_left(multiplied, ctx.mul_const(arith.right, c)))
                }
                (_, &OperandType::Constant(c2)) => {
                    let multiplied = c2.wrapping_mul(c);
                    Some(ctx.sub_const(ctx.mul_const(arith.left, c), multiplied))
                }
                _ => None
            }
        }
        _ => None,
    }
}

fn simplify_add_sub_ops<'e>(
    ops: &mut AddSlice<'e>,
    left: Operand<'e>,
    right: Operand<'e>,
    is_sub: bool,
    mask: u64,
    ctx: OperandCtx<'e>,
) -> Result<(), SizeLimitReached> {
    let const1 = collect_add_ops(left, ops, mask, false)?;
    let const2 = collect_add_ops(right, ops, mask, is_sub)?;
    let const_sum = const1.wrapping_add(const2);
    simplify_collected_add_sub_ops(ops, ctx, const_sum)?;
    Ok(())
}

fn simplify_collected_add_sub_ops<'e>(
    ops: &mut AddSlice<'e>,
    ctx: OperandCtx<'e>,
    const_sum: u64,
) -> Result<(), SizeLimitReached> {
    heapsort::sort(ops);
    simplify_add_merge_muls(ops, ctx);
    let new_consts = simplify_add_merge_masked_reverting(ops, ctx);
    let const_sum = const_sum.wrapping_add(new_consts);
    if ops.is_empty() {
        if const_sum != 0 {
            ops.push((ctx.constant(const_sum), false))?;
        }
        return Ok(());
    }

    // NOTE add_sub_ops_to_tree assumes that if there's a constant it is last,
    // so don't move this without changing it.
    if const_sum != 0 {
        if const_sum > 0x8000_0000_0000_0000 {
            ops.push((ctx.constant(0u64.wrapping_sub(const_sum)), true))?;
        } else {
            ops.push((ctx.constant(const_sum), false))?;
        }
    }
    Ok(())
}

// For simplifying (x & a) | (y & a) to (x | y) & a
// And same for xor
fn check_shared_and_mask<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
) -> Option<(Operand<'e>, Operand<'e>, Operand<'e>)> {
    let (a1, a2) = left.if_arithmetic_and()?;
    let (b1, b2) = right.if_arithmetic_and()?;
    if a2 == b2 {
        Some((a1, b1, a2))
    } else if a2 == b1 {
        Some((a1, b2, a2))
    } else if a1 == b1 {
        Some((a2, b2, a1))
    } else if a1 == b2 {
        Some((a2, b2, a1))
    } else {
        None
    }
}

pub fn simplify_or<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let left_bits = left.relevant_bits();
    let right_bits = right.relevant_bits();
    // x | 0 early exit
    if left_bits.start >= left_bits.end {
        return right;
    }
    if right_bits.start >= right_bits.end {
        return left;
    }
    let const_other = if let Some(c) = left.if_constant() {
        Some((c, Some(left), right))
    } else if let Some(c) = right.if_constant() {
        Some((c, Some(right), left))
    } else {
        None
    };
    if let Some((mut c, mut c_op, mut other)) = const_other {
        // Both operands being constant seems to happen often enough
        // for this to be worth it.
        if let Some(c2) = other.if_constant() {
            return ctx.constant(c | c2);
        }
        if let Some((l, r)) = other.if_arithmetic_or() {
            if let Some(c2) = r.if_constant() {
                if c | c2 == c2 {
                    return l;
                }
                c_op = None;
                c |= c2;
                other = l;
            }
        }
        if can_quick_simplify_type(other.ty()) {
            let left_bits = other.relevant_bits_mask();
            let c_op = c_op.unwrap_or_else(|| ctx.constant(c));
            if left_bits & c == left_bits {
                return c_op;
            }
            let arith = ArithOperand {
                ty: ArithOpType::Or,
                left: other,
                right: c_op,
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        }
    } else {
        if !bits_overlap(&left_bits, &right_bits) {
            if let Some(result) = simplify_or_2op_no_overlap(left, right, ctx, swzb) {
                return result;
            }
        }
        if can_quick_simplify_type(left.ty()) && can_quick_simplify_type(right.ty()) {
            // Two variable operands without arithmetic/memory won't simplify to anything
            // unless they are the same value.
            if left == right {
                return left;
            }
            let (left, right) = match left > right {
                true => (left, right),
                false => (right, left),
            };
            let arith = ArithOperand {
                ty: ArithOpType::Or,
                left,
                right,
            };

            return ctx.intern(OperandType::Arithmetic(arith));
        }
    }
    // Simplify (x & a) | (y & a) to (x | y) & a
    if let Some((l, r, mask)) = check_shared_and_mask(left, right) {
        let inner = simplify_or(l, r, ctx, swzb);
        return simplify_and(inner, mask, ctx, swzb);
    }

    ctx.simplify_temp_stack()
        .alloc(|masked_ops| {
            collect_masked_or_ops(left, masked_ops, 30, u64::MAX)?;
            collect_masked_or_ops(right, masked_ops, 30, u64::MAX)?;
            simplify_masked_or_ops(masked_ops, ctx);
            match masked_ops.len() {
                0 => return Ok(ctx.const_0()),
                1 => {
                    let (op, mask) = masked_ops[0];
                    return Ok(ctx.and_const(op, mask));
                }
                _ => {
                    ctx.simplify_temp_stack().alloc(|ops| {
                        move_masked_ops_to_operand_slice(ctx, masked_ops, ops)?;
                        simplify_or_ops(ops, ctx, swzb)
                    })
                }
            }
        })
        .unwrap_or_else(|_| {
            let arith = ArithOperand {
                ty: ArithOpType::Or,
                left,
                right,
            };
            ctx.intern(OperandType::Arithmetic(arith))
        })
}

fn simplify_or_2op_no_overlap<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    // This function doesn't try to handle everything that simplify_or_merge_child_ands
    // does; if there are child ors/xors then just go through the main simplification.
    fn check<'e>(op: Operand<'e>) -> bool {
        match op.ty() {
            OperandType::Arithmetic(x) => {
                if matches!(x.ty, ArithOpType::Xor | ArithOpType::Or) {
                    return false;
                }
                if matches!(x.ty, ArithOpType::And) {
                    return x.right.if_constant().is_none() || check(x.left);
                }
                true
            }
            _ => true,
        }
    }
    if !check(left) || !check(right) {
        return None;
    }
    let (l1, c1) = match left.if_and_with_const() {
        Some((l, r)) => (l, r),
        None => (left, left.relevant_bits_mask()),
    };
    let (l2, c2) = match right.if_and_with_const() {
        Some((l, r)) => (l, r),
        None => (right, right.relevant_bits_mask()),
    };
    if let Some(result) = try_merge_ands(l1, l2, c1, c2, ctx) {
        return Some(ctx.and_const(result, c1 | c2));
    }
    let result = simplify_xor_base_for_shifted(
        l1,
        (0u8, c1),
        l2,
        (0u8, c2),
        ctx,
        &mut 50,
    );
    if let Some(result) = result {
        return Some(ctx.and_const(result, c1 | c2));
    }

    ctx.simplify_temp_stack().alloc(|slice| {
        slice.push(left).ok()?;
        slice.push(right).ok()?;
        let mut const_val = 0;
        let best_mask = simplify_or_xor_canonicalize_and_masks(
            slice,
            ArithOpType::Or,
            &mut const_val,
            ctx,
            swzb,
        ).ok()?;

        Some(finish_or_simplify(slice, ctx, const_val, best_mask))
    })
}

fn simplify_or_ops<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Operand<'e>, SizeLimitReached> {
    let mut const_val = 0;
    loop {
        simplify_xor_unpack_and_masks(ops, ArithOpType::Or, ctx, swzb_ctx)?;
        const_val = ops.iter().flat_map(|x| x.if_constant())
            .fold(const_val, |sum, x| sum | x);
        ops.retain(|x| x.if_constant().is_none());
        for i in 0..ops.len() {
            let op = ops[i];
            if let Some((l, mask)) = op.if_and_with_const() {
                if let Some((l2, or_const)) = l.if_or_with_const() {
                    ops[i] = simplify_and_const(l2, !or_const & mask, ctx, swzb_ctx);
                    const_val |= or_const;
                }
            }
        }
        if ops.is_empty() || const_val == u64::MAX {
            return Ok(ctx.constant(const_val));
        }
        heapsort::sort(ops);
        ops.dedup();
        let mut const_val_changed = false;
        if const_val != 0 {
            slice_filter_map(ops, |op| {
                let new = simplify_with_and_mask(op, !const_val, ctx, swzb_ctx);
                if let Some(c) = new.if_constant() {
                    if c | const_val != const_val {
                        const_val |= c;
                        const_val_changed = true;
                    }
                    None
                } else {
                    Some(new)
                }
            });
        }
        if ops.len() > 1 {
            simplify_or_remove_equivalent_inside_mask(ops, ctx, swzb_ctx);
            simplify_or_merge_child_ands(ops, ctx, swzb_ctx, !const_val, ArithOpType::Or)?;
            simplify_or_merge_xors(ops, ctx, swzb_ctx);
            simplify_or_with_xor_of_op(ops, ctx);
            simplify_or_merge_comparisions(ops, ctx);
            simplify_xor_merge_ands_with_same_mask(ops, true, ctx, swzb_ctx);
            simplify_demorgan(ops, ctx, ArithOpType::And);
        }

        let mut i = 0;
        let mut end = ops.len();
        let mut ops_changed = false;
        while i < end {
            if let Some((l, r)) = ops[i].if_arithmetic_or() {
                ops_changed = true;
                ops.swap_remove(i);
                end -= 1;
                collect_or_ops(l, ops)?;
                if let Some(c) = r.if_constant() {
                    const_val |= c;
                } else {
                    collect_or_ops(r, ops)?;
                }
            } else if let Some(c) = ops[i].if_constant() {
                ops.swap_remove(i);
                end -= 1;
                if c | const_val != const_val {
                    const_val |= c;
                    ops_changed = true;
                }
            } else {
                i += 1;
            }
        }
        if !ops_changed {
            break;
        }
    }
    heapsort::sort(ops);
    ops.dedup();
    match ops.len() {
        0 => return Ok(ctx.constant(const_val)),
        1 if const_val == 0 => return Ok(ops[0]),
        _ => (),
    };
    let best_mask = simplify_or_xor_canonicalize_and_masks(
        ops,
        ArithOpType::Or,
        &mut const_val,
        ctx,
        swzb_ctx,
    )?;
    Ok(finish_or_simplify(ops, ctx, const_val, best_mask))
}

fn finish_or_simplify<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    const_val: u64,
    best_mask: Option<u64>,
) -> Operand<'e> {
    heapsort::sort(ops);
    let mut tree = ops.pop()
        .unwrap_or_else(|| ctx.const_0());
    while let Some(op) = ops.pop() {
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    if let Some(c) = best_mask {
        tree = intern_and_const(tree, c & !const_val, ctx);
    }
    if const_val != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: tree,
            right: ctx.constant(const_val),
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    tree
}

/// Converts ((x | y) & m) | x => (y & m) | x
/// and ((x ^ y) & m) | x => (y & m) | x
/// (Even if mask is not constant)
fn simplify_or_remove_equivalent_inside_mask<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) {
    let mut limit = 50u32;
    let mut i = 0;
    while i < ops.len() && limit != 0 {
        let op = ops[i];
        if let Some((l, r)) = op.if_arithmetic_and() {
            let result = util::arith_parts_to_new_slice(ctx, l, r, ArithOpType::And, |parts| {
                let mut part_pos = 0;
                let mut changed = false;
                while part_pos < parts.len() {
                    let part = parts[part_pos];
                    if let Some(part_arith) = part.if_arithmetic_any()
                        .filter(|a| matches!(a.ty, ArithOpType::Or | ArithOpType::Xor))
                    {
                        // ops[i] technically should not be considered here,
                        // but it is not possible ops[i] be in part_arith
                        // since part_arith is in ops[i]
                        let result = util::remove_matching_arith_parts_sorted_and_rejoin(
                            ctx,
                            ops,
                            (part_arith.left, part_arith.right, part_arith.ty),
                        );
                        limit = limit.saturating_sub(parts.len() as u32);
                        if let Some(result) = result {
                            if let Some((l, r)) = result.if_arithmetic_and() {
                                // Don't increment part_pos, check `r` too for possible
                                // extra simplification.
                                parts[part_pos] = r;
                                collect_and_ops(l, parts, usize::MAX).ok()?;
                            } else {
                                parts[part_pos] = result;
                                part_pos += 1;
                            }
                            changed = true;
                            continue;
                        }
                    } else {
                        if ops.iter().any(|&op| op == part) {
                            // (x & m) | x => x (Remove entire and)
                            return Some(ctx.const_0());
                        }
                        limit = limit.saturating_sub(parts.len() as u32);
                    }
                    part_pos += 1;
                }
                limit = limit.saturating_sub(parts.len() as u32);
                if changed {
                    simplify_and_ops(parts, ctx, swzb_ctx).ok()
                } else {
                    None
                }
            });
            if let Some(result) = result {
                if result == ctx.const_0() {
                    ops.swap_remove(i);
                    // Skip i increment
                    continue;
                } else {
                    ops[i] = result;
                }
            }
        }
        i += 1;
    }
}

/// Counts xor ops, descending into x & c masks, as
/// simplify_rsh/lsh do that as well.
/// Too long xors should not be tried to be simplified in shifts.
fn simplify_shift_is_too_long_xor(ops: &[Operand<'_>]) -> bool {
    fn count_children(op: Operand<'_>, left: &mut usize) -> bool {
        let mut op = op;
        loop {
            match op.ty() {
                OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
                    if arith.right.if_constant().is_some() {
                        op = arith.left;
                        continue;
                    }
                }
                OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Xor => {
                    *left = match left.checked_sub(1) {
                        Some(s) => s,
                        None => return true,
                    };
                    if let Some((l, r)) = arith.right.if_arithmetic_and() {
                        if r.if_constant().is_some() {
                            if !count_children(l, left) {
                                return true;
                            }
                        }
                    }
                    op = arith.left;
                    continue;
                }
                _ => (),
            }
            return false;
        }
    }

    let mut left = 8usize;
    left = match left.checked_sub(ops.len()) {
        Some(s) => s,
        None => return true,
    };
    for &op in ops {
        if count_children(op, &mut left) {
            return true;
        }
    }
    false
}

fn slice_filter_map<'e, F>(slice: &mut Slice<'e>, mut fun: F)
where F: FnMut(Operand<'e>) -> Option<Operand<'e>>,
{
    let mut out_pos = 0;
    for in_pos in 0..slice.len() {
        let val = slice[in_pos];
        if let Some(new) = fun(val) {
            slice[out_pos] = new;
            out_pos += 1;
        }
    }
    slice.shrink(out_pos);
}

fn should_stop_with_and_mask(swzb_ctx: &mut SimplifyWithZeroBits) -> bool {
    if swzb_ctx.with_and_mask_count_at_limit() {
        #[cfg(feature = "fuzz")]
        tls_simplification_incomplete();
        true
    } else {
        false
    }
}

/// Simplifies with assumption that `op` is going to be masked by `mask` afterwards.
///
/// This does mean that (x & 8000) with mask 8000 can return just x.
///
/// Implementation must take care to not reduce the mask by relevant_bits of the operand
/// that will then be simplified, as that will lead to incorrect results, especially with
/// the and mask removing mentioned above. However, in arithmetic like (x & y) reducing the
/// mask passed to x by relevant bits of y is (?) valid.
///
/// Similarly addition and subtraction can simplify their operands if the mask is grown
/// in a way that preserves less significant bits affecting more significant bits, as well
/// as preserving certain bits that have to be *zero* in order for the bit propagation
/// possibilities to not change.
fn simplify_with_and_mask<'e>(
    op: Operand<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    if mask == u64::MAX {
        return op;
    }
    let relevant_mask = op.relevant_bits_mask();
    if relevant_mask & mask == 0 {
        return ctx.const_0();
    }
    // quick simplify types won't change by this (other than becoming 0 if mask zeroes them),
    // check that early as it's cheap and ultimately at some point recursing into op they pop up.
    if can_quick_simplify_type(op.ty()) {
        return op;
    }
    if relevant_mask & mask == relevant_mask {
        if op.0.flags & super::FLAG_COULD_REMOVE_CONST_AND == 0 {
            return op;
        }
    }
    if should_stop_with_and_mask(swzb_ctx) {
        return op;
    }
    swzb_ctx.with_and_mask_count += 1;
    let new = simplify_with_and_mask_inner(op, mask, ctx, swzb_ctx);
    new
}

fn simplify_with_and_mask_inner<'e>(
    op: Operand<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    match *op.ty() {
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    let simplified_left;
                    let simplified_right;
                    // Use relevant_bits_for_and_simplify so that
                    // this will properly cancel out masks that and simplify
                    // considers most accurate.
                    let self_mask = mask & relevant_bits_for_and_simplify_of_and_chain(arith.left);
                    if let Some(c) = arith.right.if_constant() {
                        if c == self_mask {
                            // This constant mask was already applied to arith.left
                            // since the mask is same; can just return it as is.
                            return arith.left;
                        } else if c & self_mask == 0 {
                            return ctx.const_0();
                        } else if c & mask == c {
                            // Mask is superset of the already existing mask,
                            // so it won't simplify anything further.
                            return op;
                        } else {
                            let left_simplify_mask = c & mask;
                            simplified_left = simplify_with_and_mask(
                                arith.left,
                                left_simplify_mask,
                                ctx,
                                swzb_ctx,
                            );
                            let new_self_mask = mask & simplified_left.relevant_bits_mask();
                            if c & mask == new_self_mask {
                                // Left became something that won't need constant mask like the
                                // above check `c == self_mask`
                                return simplified_left;
                            }
                            if should_stop_with_and_mask(swzb_ctx) {
                                return op;
                            }
                            // This is just avoid recursing to simplify_with_and_mask
                            // when it's already known to do this.
                            simplified_right = ctx.constant(c & mask);
                        }
                    } else {
                        simplified_right =
                            simplify_with_and_mask(arith.right, self_mask, ctx, swzb_ctx);
                        if should_stop_with_and_mask(swzb_ctx) {
                            return op;
                        }
                        let left_simplify_mask = mask & simplified_right.relevant_bits_mask();
                        simplified_left =
                            simplify_with_and_mask(arith.left, left_simplify_mask, ctx, swzb_ctx);
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op
                    } else {
                        simplify_and(simplified_left, simplified_right, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Or | ArithOpType::Xor => {
                    simplify_with_and_mask_or_xor(op, arith, mask, ctx, swzb_ctx)
                }
                ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let c = c as u8;
                        let left = simplify_with_and_mask(arith.left, mask >> c, ctx, swzb_ctx);
                        if left == arith.left {
                            op
                        } else {
                            simplify_lsh_const(left, c, ctx, swzb_ctx)
                        }
                    } else {
                        op
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let c = c as u8;
                        let left = simplify_with_and_mask(arith.left, mask << c, ctx, swzb_ctx);
                        if left == arith.left {
                            op
                        } else {
                            simplify_rsh_const(left, c, ctx, swzb_ctx)
                        }
                    } else {
                        op
                    }
                }
                ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul => {
                    if let Some(result) = simplify_with_and_mask_add_sub_try_extract_inner_mask(
                        arith,
                        mask,
                        ctx,
                        swzb_ctx,
                    ) {
                        return result;
                    }
                    let orig_mask = mask;
                    let mut left_mask;
                    let right_mask;
                    let add_sub_max;

                    // The mask can be applied separately to left and right if
                    // any of the unmasked bits input don't affect masked bits in result.
                    // For add/sub/mul, a bit can only affect itself and more
                    // significant bits.
                    //
                    // First, check if relevant bits start of either operand >= mask end,
                    // in which case the operand cannot affect result at all and we can
                    // just return the other operand simplified with the mask.
                    //
                    // Otherwise check if mask has has all low bits 1 and all high bits 0,
                    // and apply left/right separately.
                    //
                    // Assuming it is 00001111...
                    // Adding 1 makes a valid mask to overflow to 10000000...
                    // Though the 1 bit can be carried out so count_ones is 1 or 0.
                    let mask_end_bit = 64 - mask.leading_zeros() as u8;
                    let other = Operand::either(arith.left, arith.right, |x| {
                        if x.relevant_bits().start >= mask_end_bit { Some(()) } else { None }
                    }).map(|((), other)| other);
                    if let Some(other) = other {
                        return simplify_with_and_mask(other, mask, ctx, swzb_ctx);
                    }
                    if arith.ty == ArithOpType::Add {
                        // With add, carry will propagate left until a bit in both
                        // operands is 0. Everything bit starting from 1,1 match
                        // generating the carry until 0,0 match in inputs can
                        // affect more significant bits until the 0,0 end.
                        // Effectively with state machine of two states: no carry and carry
                        // no carry:
                        // 0,0 => 0
                        // 0,1 => 1
                        // 1,0 => 1
                        // 1,1 => 0 -> to carry
                        // carry:
                        // 0,0 => 1 -> to no carry state
                        // 0,1 => 0
                        // 1,0 => 0
                        // 1,1 => 1
                        //
                        // E.g. (In binary)
                        // l = 0011111100011111000111111100
                        // r = 0000010011011110111001000000
                        //       |A ||B | |C        ||D |
                        //       \--/\--/ \---------/\--/
                        // A and C ranges must have continuous mask up to their
                        // lowest byte, but B and D which don't have any 1,1 pairs
                        // don't need the mask be extended.
                        // Conveniently, addition of `l + r` sets
                        // 1,0/0,1 pairs that have 1,1 on right of them (A, C), and
                        // 1,0/0,1 pairs that have 0,0 on right of them (B, D)
                        // to a different value. A,C become 0 and B,D become (stay) 1.
                        // Then, invert the result and add in 1,1 (l & r) to get A,C to 1,
                        // and clear any bits that were 0,0 originally (As the ones left
                        // of A and C ranges become 1 when carry propagation stops there.
                        let l = arith.left.relevant_bits_mask();
                        let r = arith.right.relevant_bits_mask();
                        let z = (!(l.wrapping_add(r)) | (l & r)) & (l | r);
                        // So now z is marking the A,C ranges, e.g.
                        // z = 0011110000011111111111000000
                        // However, the fact that range such as D doesn't
                        // propagate left will rely on bits of r being 0 there.
                        // Therefore r may only be simplified with mask that doesn't
                        // guarantee zeroing of the following ranges marked with *,
                        // that is, right_mask must have 1 for * bits too.
                        // Similarly left_mask would need to include set-to-one bit chunk
                        // on r that are right of a z chunk.
                        // l = 0011111100011111000111111100
                        // r = 0000010011011110111001000000
                        // z = 0011110000011111111111000000
                        //           **              ****
                        // This can be achieved by keeping 1,0 (l,z) pairs that have 1,1
                        // *left* of them, and clearing 1,0 pairs with 0,0 *left* of them.
                        // (Due to how z was built there is never 1,0 with 0,1 left of the pair
                        // before 1,1 or 0,0; so that case doesn't have to be considered)
                        //
                        // This can be done by reversing bits of z/l, using addition
                        // like above, and inverting bits back (Though inverting back is not
                        // done now since the following reduce-by-mask step will also want
                        // inputs be inversed).
                        // Note that this is not same as keeping 1,0 pairs with 0,0
                        // right of them, as it would include bits that are not part of z
                        // at all: The ~~~ bits would get included unnecessarily with this example
                        // l2 = 0011111100011111000000011100
                        // r2 = 0000010011011110111000000000
                        // z2 = 0011110000011110000000000000
                        //            **       *       ~~~
                        let z_rev = z.reverse_bits();
                        let l_rev = l.reverse_bits();
                        let r_rev = r.reverse_bits();
                        // lz is going to be used with l; it uses r as input and vice versa.
                        let lz = (!(z_rev.wrapping_add(r_rev)) & r_rev) | z_rev;
                        let rz = (!(z_rev.wrapping_add(l_rev)) & l_rev) | z_rev;
                        // Now the 1111 chunks can be cut from left if
                        // any mask bits there aren't set, calculating those won't
                        // be useful.
                        // z = 0011110000011111111111000000
                        // m = 0001000000000110110111110000 (mask)
                        // r = 0001110000000111111111000000 (result)
                        //                    E
                        // This can be achieved by keeping 1,0 pairs that have 1,1
                        // *left* of them, and clearing 1,0 pairs with 0,0 *left* of them.
                        let mask_rev = mask.reverse_bits();
                        let l_result = ((!(lz.wrapping_add(lz & mask_rev)) | mask_rev) & lz)
                            .reverse_bits();
                        let r_result = ((!(rz.wrapping_add(rz & mask_rev)) | mask_rev) & rz)
                            .reverse_bits();
                        left_mask = l_result | mask;
                        right_mask = r_result | mask;
                        if left_mask == u64::MAX && right_mask == u64::MAX {
                            return op;
                        }
                        let add_max_bit = 64 - (l_result | r_result).leading_zeros();
                        add_sub_max = 1u64.checked_shl(add_max_bit)
                            .unwrap_or(0u64)
                            .wrapping_sub(1);
                    } else if arith.ty == ArithOpType::Sub {
                        // For sub the bits propagating left can be figured out similarly,
                        // with the state machine
                        // no borrow:
                        // 1,0 => 1
                        // 1,1 => 0
                        // 0,0 => 0
                        // 0,1 => 1 -> borrow
                        // borrow:
                        // 1,0 => 0 -> no borrow
                        // 1,1 => 1
                        // 0,0 => 1
                        // 0,1 => 0
                        // So left bits don't matter (unless known-to-be-one are considered),
                        // starting first nonzero bit of right any bits can be changed.
                        let right_relbits = arith.right.relevant_bits();
                        let r_low_zero_mask = match 1u64.checked_shl(right_relbits.start as u32) {
                            Some(s) => s.wrapping_sub(1),
                            None => u64::MAX,
                        };
                        let mask_filled_to_lowest = match 1u64.checked_shl(mask_end_bit as u32) {
                            Some(s) => s.wrapping_sub(1),
                            None => u64::MAX,
                        };
                        // Right sipmlify mask needs to be 000..111 so that any low bits that
                        // are currently 0 in there won't be allowed to become 1,
                        // but left can be simplified without filling with r_low_zero_mask
                        left_mask = (mask_filled_to_lowest & !r_low_zero_mask) | mask;
                        right_mask = mask_filled_to_lowest;
                        add_sub_max = mask_filled_to_lowest;
                    } else {
                        // Otherwise fill the mask to 000..111 having any bit after
                        // mask_end_bit set, so that something can possibly be done.
                        let mask = if mask_end_bit >= 64 {
                            // Mask would be u64::MAX, which is pointless
                            return op;
                        } else {
                            (1u64 << mask_end_bit).wrapping_sub(1)
                        };
                        left_mask = mask;
                        right_mask = mask;
                        add_sub_max = mask;
                    }
                    // Normalize (x + c000) & ffff to (x - 4000) & ffff and similar.
                    let mut simplified_right = None;
                    if let Some(orig_c) = arith.right.if_constant() {
                        let c = orig_c & right_mask;
                        let max = add_sub_max.wrapping_add(1);
                        let limit = max >> 1;
                        if arith.ty == ArithOpType::Add && max != 0 {
                            if c > limit {
                                let new = ctx.sub_const(arith.left, max.wrapping_sub(c));
                                let new = if max < mask &&
                                    new.relevant_bits_mask() & !add_sub_max & mask != 0
                                {
                                    simplify_and_const(new, add_sub_max, ctx, swzb_ctx)
                                } else {
                                    new
                                };
                                return simplify_with_and_mask(new, orig_mask, ctx, swzb_ctx);
                            }
                        } else if arith.ty == ArithOpType::Sub && max != 0 {
                            if c >= limit {
                                let new = ctx.add_const(arith.left, max.wrapping_sub(c));
                                return simplify_with_and_mask(new, orig_mask, ctx, swzb_ctx);
                            }
                        } else if arith.ty == ArithOpType::Mul {
                            // Simplify mul with power-of-two as left shift.
                            // or even when not power-of-two, reduce mask
                            // by what will be always shifted out
                            if c == 0 {
                                return ctx.const_0();
                            }
                            let shift = c.trailing_zeros();
                            if c & c.wrapping_sub(1) == 0 {
                                let shift = c.trailing_zeros();
                                // Is power of two, give left shift treatment.
                                let left = simplify_with_and_mask(
                                    arith.left,
                                    orig_mask >> shift,
                                    ctx,
                                    swzb_ctx,
                                );
                                if left == arith.left && c == orig_c {
                                    return op;
                                } else {
                                    return simplify_lsh_const(left, shift as u8, ctx, swzb_ctx);
                                }
                            } else {
                                // High bits of left mask can be known to be useless
                                // shift them out
                                left_mask = left_mask >> shift;
                            }
                        }
                        if c != orig_c {
                            simplified_right = Some(ctx.constant(c));
                        } else {
                            simplified_right = Some(arith.right);
                        }
                    }

                    let simplified_right = simplified_right.unwrap_or_else(|| {
                        simplify_with_and_mask(arith.right, right_mask, ctx, swzb_ctx)
                    });

                    if should_stop_with_and_mask(swzb_ctx) {
                        return op;
                    }
                    let simplified_left =
                        simplify_with_and_mask(arith.left, left_mask, ctx, swzb_ctx);
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op;
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op
                    } else {
                        let op = ctx.arithmetic(arith.ty, simplified_left, simplified_right);
                        // The result may simplify again, for example with mask 0x1
                        // Mem16[x] + Mem32[x] + Mem8[x] => 3 * Mem8[x] => 1 * Mem8[x]
                        // But assuming for now that this only happens when non-mul was
                        // converted to mul to save some time in other cases.
                        if arith.ty != ArithOpType::Mul && op.if_arithmetic_mul().is_some() {
                            simplify_with_and_mask(op, orig_mask, ctx, swzb_ctx)
                        } else {
                            op
                        }
                    }
                }
                _ => op,
            }
        }
        OperandType::Memory(ref mem) => {
            simplify_with_and_mask_mem(op, mem, mask, ctx)
        }
        OperandType::Constant(c) => if c & mask != c {
            ctx.constant(c & mask)
        } else {
            op
        }
        OperandType::SignExtend(val, from, _) => {
            let from_mask = from.mask();
            if from_mask & mask == mask {
                simplify_with_and_mask(val, mask, ctx, swzb_ctx)
            } else {
                op
            }
        }
        _ => op,
    }
}

/// simplify_with_and_mask helper for add/sub.
/// If `((x & C) - y) & mask` is equal to
/// `(x - y) & (C & mask)`, canonicalizes to that
/// as that can help moving the mask more out.
fn simplify_with_and_mask_add_sub_try_extract_inner_mask<'e>(
    arith: &ArithOperand<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    if !matches!(arith.ty, ArithOpType::Add | ArithOpType::Sub) {
        return None;
    }
    if let Some((inner, c)) = arith.left.if_and_with_const() {
        // !c & mask < c to make sure to not simplify in case such as
        // ((x & fff0) - y) & ffff_ffff
        if is_continuous_mask(c) && c & mask == c && !c & mask < c {
            if arith.right.relevant_bits().start >= arith.left.relevant_bits().start {
                let new = ctx.arithmetic(arith.ty, inner, arith.right);
                let masked = simplify_and_const(new, c, ctx, swzb_ctx);
                // Should this do simplify_with_and_mask again?
                // Feels like it's not possible for it to do anything
                // but I could be wrong there.
                return Some(masked);
            }
        }
    }
    if arith.ty == ArithOpType::Add {
        // Addition can (obviously) be done with left/right swapped
        if let Some((inner, c)) = arith.right.if_and_with_const() {
            if is_continuous_mask(c) && c & mask == c && !c & mask < c {
                if arith.left.relevant_bits().start >= arith.right.relevant_bits().start {
                    let new = ctx.arithmetic(ArithOpType::Add, inner, arith.left);
                    let masked = simplify_and_const(new, c, ctx, swzb_ctx);
                    return Some(masked);
                }
            }
        }
    }
    None
}

/// simplify_with_and_mask for or/xor.
/// If there are masked or/xor that can now be moved out of the
/// mask, moves them out.
///
/// E.g.
/// ((rax | (rax << 8)) & f000_0000) | rcx
/// with mask 8000_0008
/// (rax << 8) can be moved out of the inner mask to
/// (((rax) & 8000_0000) | rcx) | (rax << 8)
/// since it is known to have low 8 bit cleared.
///
/// Note that for just
/// (rax | (rax << 8)) & f000_0000
/// with mask 8000_0008
/// The operand is kept as
/// ((rax) | (rax << 8)) & f000_000f
///
/// `op` and `arith` are assumed to be linked as
/// `op = OperandType::Arithmetic(arith)`.
fn simplify_with_and_mask_or_xor<'e>(
    op: Operand<'e>,
    arith: &ArithOperand<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let arith_ty = arith.ty;
    util::arith_parts_to_new_slice(ctx, arith.left, arith.right, arith_ty, |slice| {
        let mut changed = false;
        let mut any_zero = false;
        let mut current_mask = mask;
        let end = slice.len();
        let zero = ctx.const_0();
        for i in 0..end {
            let op = slice[i];
            let extracted_out_of_and =
                masked_or_xor_split_parts_not_needing_mask(op, mask, arith_ty, ctx, swzb_ctx)
                    .unwrap_or(op);
            if should_stop_with_and_mask(swzb_ctx) {
                return None;
            }
            let simplified =
                simplify_with_and_mask(extracted_out_of_and, current_mask, ctx, swzb_ctx);
            if should_stop_with_and_mask(swzb_ctx) {
                return None;
            }
            if arith_ty == ArithOpType::Or {
                // If some bits are known to be one due to a constant, they can be cleared
                // from rest_mask (Is this actually correct??)
                if let Some(c) = simplified.if_constant() {
                    current_mask &= !c;
                }
            }
            if simplified != op {
                if let Some((l, r)) = simplified.if_arithmetic(arith_ty) {
                    slice[i] = r;
                    collect_arith_ops(l, slice, arith_ty, usize::MAX).ok()?;
                } else {
                    any_zero |= simplified == zero;
                    slice[i] = simplified;
                }
                changed = true;
            }
        }
        if !changed {
            return None;
        }

        if any_zero {
            slice.retain(|op| op != zero);
            // Maybe somewhat common? So avoid calling simplify func for these cases.
            if slice.len() == 1 {
                return Some(slice[0]);
            } else if slice.len() == 0 {
                return Some(zero);
            }
        }
        if arith.ty == ArithOpType::Or {
            simplify_or_ops(slice, ctx, swzb_ctx).ok()
        } else {
            simplify_xor_ops(slice, ctx, swzb_ctx).ok()
        }
    }).unwrap_or(op)
}

/// For simplify_with_and_mask; if `op` is `(x ^ y) & inner_mask` (Or or),
/// and parts of the or/xor don't need inner_mask if masked by `mask`, moves
/// those parts out of the or/xor.
fn masked_or_xor_split_parts_not_needing_mask<'e>(
    op: Operand<'e>,
    mask: u64,
    arith_ty: ArithOpType,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    let (inner_orig, inner_mask) = op.if_and_with_const()?;

    let (l, r) = inner_orig.if_arithmetic(arith_ty)?;
    // Bits that, if not cleared by inner mask
    // would affect the result after being masked
    // E.g. inner = ff00, mask = fff0, cleared_bits = 00f0
    let cleared_bits = !inner_mask & mask;
    if cleared_bits == 0 {
        return None;
    }
    util::split_off_by_condition_rejoin_rest(
        ctx,
        (l, r, arith_ty),
        // May have to use relevant_bits_for_and_simplify_of_and_chain
        // or something for consistency, but this works for now.
        |x| x.relevant_bits_mask() & cleared_bits == 0,
        |inner, outside_ops| {
            let inner = match inner {
                Some(s) => s,
                None => {
                    // All ops were moved out of the mask..
                    // So just return inner_orig
                    // Maybe this case could allow skipping
                    // simplify_with_and_mask too but not thinking
                    // about that right now.
                    return Some(inner_orig);
                }
            };
            // TODO: Should just do and mask simplify here
            // to save some work
            let inner = ctx.and_const(inner, inner_mask);
            outside_ops.push(inner).ok()?;
            if arith_ty == ArithOpType::Or {
                simplify_or_ops(outside_ops, ctx, swzb_ctx).ok()
            } else {
                simplify_xor_ops(outside_ops, ctx, swzb_ctx).ok()
            }
        },
    )
}

/// Assumes: `mem` is part of `op`.
fn simplify_with_and_mask_mem<'e>(
    op: Operand<'e>,
    mem: &MemAccess<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    let (base, shift) = simplify_with_and_mask_mem_dont_apply_shift(op, mem, mask, ctx);
    if shift == 0 {
        base
    } else {
        intern_lsh_const(base, shift, ctx)
    }
}

/// Assumes: `mem` is part of `op`.
fn simplify_with_and_mask_mem_dont_apply_shift<'e>(
    op: Operand<'e>,
    mem: &MemAccess<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
) -> (Operand<'e>, u8) {
    let mask = mem.size.mask() & mask;
    // Try to do conversions such as Mem32[x] & 00ff_ff00 => Mem16[x + 1] << 8,
    // but also Mem32[x] & 003f_5900 => (Mem16[x + 1] & 3f59) << 8.

    // Round down to 8 -> convert to bytes
    let mask_low = mask.trailing_zeros() / 8;
    // Round up to 8 -> convert to bytes
    let mask_high = (64 - mask.leading_zeros() + 7) / 8;
    if mask_high <= mask_low {
        return (op, 0);
    }
    let mask_size = mask_high - mask_low;
    let mem_size = mem.size.bits();
    let new_size;
    if mask_size <= 1 && mem_size > 8 {
        new_size = MemAccessSize::Mem8;
    } else if mask_size <= 2 && mem_size > 16 {
        new_size = MemAccessSize::Mem16;
    } else if mask_size <= 4 && mem_size > 32 {
        new_size = MemAccessSize::Mem32;
    } else {
        return (op, 0);
    }
    let (address, offset) = mem.address();
    let mem = ctx.mem_any(new_size, address, offset.wrapping_add(mask_low as u64));
    (mem, (mask_low as u8) << 3)
}

/// If `a` is subset of or equal to `b`
fn range_is_subset(a: &Range<u8>, b: &Range<u8>) -> bool {
    a.start >= b.start && a.end <= b.end
}

fn ranges_overlap(a: &Range<u8>, b: &Range<u8>) -> bool {
    a.start < b.end && a.end > b.start
}

/// Simplifies `op` when the bits in the range `bits` are guaranteed to be zero.
/// Returning `None` is considered same as `Some(constval(0))` (The value gets optimized out in
/// bitwise and).
///
/// Bits are assumed to be in 0..64 range
fn simplify_with_zero_bits<'e>(
    op: Operand<'e>,
    bits: &Range<u8>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    if op.0.min_zero_bit_simplify_size > bits.end - bits.start || bits.start >= bits.end {
        return Some(op);
    }
    let relevant_bits = op.relevant_bits();
    // Check if we're setting all nonzero bits to zero
    if range_is_subset(&relevant_bits, bits) {
        return None;
    }
    // Check if we're zeroing bits that were already zero
    if !ranges_overlap(&relevant_bits, bits) {
        return Some(op);
    }

    let recurse_check = match op.ty() {
        OperandType::Arithmetic(arith) => {
            match arith.ty {
                ArithOpType::And | ArithOpType::Or | ArithOpType::Xor |
                    ArithOpType::Lsh | ArithOpType::Rsh => true,
                _ => false,
            }
        }
        _ => false,
    };

    fn should_stop(swzb: &mut SimplifyWithZeroBits) -> bool {
        if swzb.zero_bits_simplify_count_at_limit() {
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
            true
        } else {
            false
        }
    }

    if recurse_check {
        if swzb.xor_recurse > 4 {
            swzb.simplify_count = u8::MAX;
        }
        if should_stop(swzb) {
            return Some(op);
        } else {
            swzb.simplify_count += 1;
        }
    }

    match op.ty() {
        OperandType::Arithmetic(arith) => {
            let left = arith.left;
            let right = arith.right;
            match arith.ty {
                ArithOpType::And => {
                    // If zeroing bits that either of the operands has already zeroed,
                    // the other operand has to have also been simplified to take those
                    // zero bits into account => can't simplify further.
                    if !ranges_overlap(bits, &left.relevant_bits()) {
                        return Some(op);
                    }
                    if !ranges_overlap(bits, &right.relevant_bits()) {
                        return Some(op);
                    }

                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    if should_stop(swzb) {
                        return Some(op);
                    }
                    return match simplified_left {
                        Some(l) => {
                            let simplified_right =
                                simplify_with_zero_bits(right, bits, ctx, swzb);
                            if should_stop(swzb) {
                                return Some(op);
                            }
                            match simplified_right {
                                Some(r) => {
                                    if l == left && r == right {
                                        Some(op)
                                    } else {
                                        Some(simplify_and(l, r, ctx, swzb))
                                    }
                                }
                                None => None,
                            }
                        }
                        None => None,
                    };
                }
                ArithOpType::Or => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    let simplified_right = simplify_with_zero_bits(right, bits, ctx, swzb);
                    if should_stop(swzb) {
                        return Some(op);
                    }
                    return match (simplified_left, simplified_right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l == left && r == right {
                                Some(op)
                            } else {
                                Some(simplify_or(l, r, ctx, swzb))
                            }
                        }
                    };
                }
                ArithOpType::Xor => {
                    let simplified_left = simplify_with_zero_bits(left, bits, ctx, swzb);
                    let simplified_right = simplify_with_zero_bits(right, bits, ctx, swzb);
                    if should_stop(swzb) {
                        return Some(op);
                    }
                    return match (simplified_left, simplified_right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l == left && r == right {
                                Some(op)
                            } else {
                                swzb.xor_recurse += 1;
                                let result = simplify_xor(l, r, ctx, swzb);
                                swzb.xor_recurse -= 1;
                                Some(result)
                            }
                        }
                    };
                }
                ArithOpType::Lsh => {
                    if let Some(c) = right.if_constant() {
                        if bits.end >= 64 && bits.start <= c as u8 {
                            return None;
                        } else {
                            let low = bits.start.saturating_sub(c as u8);
                            let high = bits.end.saturating_sub(c as u8);
                            if low >= high {
                                return Some(op);
                            }
                            let result = simplify_with_zero_bits(left, &(low..high), ctx, swzb);
                            if let Some(result) =  result {
                                if result != left {
                                    return Some(simplify_lsh(result, right, ctx, swzb));
                                }
                            }
                        }
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = right.if_constant() {
                        if bits.start == 0 && c as u8 >= (64 - bits.end) {
                            return None;
                        } else {
                            let low = bits.start.saturating_add(c as u8).min(64);
                            let high = bits.end.saturating_add(c as u8).min(64);
                            if low >= high {
                                return Some(op);
                            }
                            let result1 = if bits.end == 64 {
                                let mask_high = 64 - low;
                                let mask = !0u64 >> c << c << mask_high >> mask_high;
                                simplify_with_and_mask(left, mask, ctx, swzb)
                            } else {
                                left.clone()
                            };
                            let result2 =
                                simplify_with_zero_bits(result1, &(low..high), ctx, swzb);
                            if let Some(result2) =  result2 {
                                if result2 != left {
                                    return Some(
                                        simplify_rsh(result2, right, ctx, swzb)
                                    );
                                }
                            } else if result1 != left {
                                return Some(simplify_rsh(result1, right, ctx, swzb));
                            }
                        }
                    }
                }
                _ => (),
            }
        }
        OperandType::Constant(c) => {
            let low = bits.start;
            let high = 64 - bits.end;
            let mask = !(!0u64 >> low << low << high >> high);
            let new_val = c & mask;
            return match new_val {
                0 => None,
                c => Some(ctx.constant(c)),
            };
        }
        OperandType::Memory(ref mem) => {
            if bits.start == 0 && bits.end >= relevant_bits.end {
                return None;
            } else if bits.end == 64 {
                let (address, offset) = mem.address();
                if bits.start <= 8 && relevant_bits.end > 8 {
                    return Some(ctx.mem8(address, offset));
                } else if bits.start <= 16 && relevant_bits.end > 16 {
                    return Some(ctx.mem16(address, offset));
                } else if bits.start <= 32 && relevant_bits.end > 32 {
                    return Some(ctx.mem32(address, offset));
                }
            }
        }
        _ => (),
    }
    Some(op)
}

/// Merges things like [2 * b, a, c, b, c] to [a, 3 * b, 2 * c]
fn simplify_add_merge_muls<'e>(
    ops: &mut AddSlice<'e>,
    ctx: OperandCtx<'e>,
) {
    fn count_equivalent_opers<'e>(ops: &[(Operand<'e>, bool)], equiv: Operand<'e>) -> Option<u64> {
        ops.iter().map(|&(o, neg)| {
            let (mul, val) = o.if_arithmetic_mul()
                .and_then(|(l, r)| r.if_constant().map(|c| (c, l)))
                .unwrap_or_else(|| (1, o));
            match equiv == val {
                true => if neg { 0u64.wrapping_sub(mul) } else { mul },
                false => 0,
            }
        }).fold(None, |sum, next| if next != 0 {
            Some(sum.unwrap_or(0).wrapping_add(next))
        } else {
            sum
        })
    }

    let mut pos = 0;
    while pos < ops.len() {
        let merged = {
            let (self_mul, op) = ops[pos].0.if_arithmetic_mul()
                .and_then(|(l, r)| r.if_constant().map(|c| (c, l)))
                .unwrap_or_else(|| (1, ops[pos].0));

            let others = count_equivalent_opers(&ops[pos + 1..], op);
            if let Some(others) = others {
                let self_mul = if ops[pos].1 { 0u64.wrapping_sub(self_mul) } else { self_mul };
                let sum = self_mul.wrapping_add(others);
                if sum == 0 {
                    Some(None)
                } else {
                    Some(Some((sum, op)))
                }
            } else {
                None
            }
        };
        match merged {
            Some(Some((sum, equiv))) => {
                let mut other_pos = pos + 1;
                while other_pos < ops.len() {
                    let is_equiv = ops[other_pos].0
                        .if_arithmetic_mul()
                        .and_then(|(l, r)| r.if_constant().map(|c| (c, l)))
                        .map(|(_, other)| other == equiv)
                        .unwrap_or_else(|| ops[other_pos].0 == equiv);
                    if is_equiv {
                        ops.remove(other_pos);
                    } else {
                        other_pos += 1;
                    }
                }
                let negate = sum > 0x8000_0000_0000_0000;
                let sum = if negate { (!sum).wrapping_add(1) } else { sum };
                ops[pos].0 = simplify_mul(equiv, ctx.constant(sum), ctx);
                ops[pos].1 = negate;
                pos += 1;
            }
            // Remove everything matching
            Some(None) => {
                let (op, _) = ops.remove(pos);
                let equiv = op.if_arithmetic_mul()
                    .and_then(|(l, r)| r.if_constant().map(|c| (c, l)))
                    .map(|(_, other)| other)
                    .unwrap_or_else(|| op);
                let mut other_pos = pos;
                while other_pos < ops.len() {
                    let other = ops[other_pos].0;
                    let other = other.if_arithmetic_mul()
                        .and_then(|(l, r)| r.if_constant().map(|c| (c, l)))
                        .map(|(_, other)| other)
                        .unwrap_or_else(|| other);
                    if other == equiv {
                        ops.remove(other_pos);
                    } else {
                        other_pos += 1;
                    }
                }
            }
            None => {
                pos += 1;
            }
        }
    }
}

pub fn simplify_xor<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let left_bits = left.relevant_bits();
    let right_bits = right.relevant_bits();
    // x ^ 0 early exit
    if left_bits.start >= left_bits.end {
        return right;
    }
    if right_bits.start >= right_bits.end {
        return left;
    }
    if let Some((l, r)) = check_quick_arith_simplify(left, right) {
        let arith = ArithOperand {
            ty: ArithOpType::Xor,
            left: l.clone(),
            right: r.clone(),
        };
        return ctx.intern(OperandType::Arithmetic(arith));
    }
    // Simplify (x & a) ^ (y & a) to (x ^ y) & a
    if let Some((l, r, mask)) = check_shared_and_mask(left, right) {
        let inner = simplify_xor(l, r, ctx, swzb);
        return simplify_and(inner, mask, ctx, swzb);
    }
    let temp_stack = ctx.simplify_temp_stack();
    temp_stack
        .alloc(|masked_ops| {
            collect_masked_xor_ops(left, masked_ops, 30, u64::MAX)?;
            collect_masked_xor_ops(right, masked_ops, 30, u64::MAX)?;
            simplify_masked_xor_ops(masked_ops, ctx, swzb);
            match masked_ops.len() {
                0 => return Ok(ctx.const_0()),
                1 => {
                    let (op, mask) = masked_ops[0];
                    return Ok(ctx.and_const(op, mask));
                }
                _ => {
                    temp_stack.alloc(|ops| {
                        move_masked_ops_to_operand_slice(ctx, masked_ops, ops)?;
                        simplify_xor_ops(ops, ctx, swzb)
                    })
                }
            }
        })
        .unwrap_or_else(|_| {
            // This is likely some hash function being unrolled, give up
            // Also set swzb to stop everything
            swzb.simplify_count = u8::MAX;
            swzb.with_and_mask_count = u8::MAX;
            let arith = ArithOperand {
                ty: ArithOpType::Xor,
                left,
                right,
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        })
}

fn simplify_masked_xor_ops<'e>(
    ops: &mut MaskedOpSlice<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) {
    if ops.len() > 1 {
        heapsort::sort_by(ops, |a, b| a.0 < b.0);
        simplify_masked_xor_or_merge_same_ops(ops, ctx, ArithOpType::Xor);
        simplify_masked_xor_merge_or(ops, ctx, swzb);
        simplify_masked_xor_or_and_to_xor(ops);
        simplify_masked_xor_or_to_and(ops, ctx, swzb);
    }
}

fn simplify_masked_or_ops<'e>(
    ops: &mut MaskedOpSlice<'e>,
    ctx: OperandCtx<'e>,
) {
    if ops.len() > 1 {
        heapsort::sort_by(ops, |a, b| a.0 < b.0);
        simplify_masked_xor_or_merge_same_ops(ops, ctx, ArithOpType::Or);
    }
}

fn move_masked_ops_to_operand_slice<'e>(
    ctx: OperandCtx<'e>,
    ops: &MaskedOpSlice<'e>,
    out: &mut Slice<'e>,
) -> Result<(), SizeLimitReached> {
    for &(op, mask) in ops.iter() {
        let relbit_mask = op.relevant_bits_mask();
        if relbit_mask & mask == relbit_mask {
            out.push(op)?;
        } else {
            out.push(ctx.and_const(op, mask))?;
        }
    }
    Ok(())
}

fn simplify_xor_try_extract_constant<'e>(
    op: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<(Operand<'e>, u64)> {
    let (l, r) = op.if_arithmetic_and()?;
    let and_mask = r.if_constant()?;
    if let Some((l, or_const)) = l.if_or_with_const() {
        let new = simplify_and_const(l, !or_const & and_mask, ctx, swzb);
        Some((new, or_const & and_mask))
    } else {
        None
    }
}

pub fn simplify_gt<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let mut left = left;
    let mut right = right;
    if left == right {
        return ctx.const_0();
    }
    if matches!(left.ty(), OperandType::Arithmetic(..)) ||
        matches!(right.ty(), OperandType::Arithmetic(..))
    {
        if let Some(result) = simplify_gt_arith_checks(&mut left, &mut right, ctx, swzb_ctx) {
            return result;
        }
    }

    match (left.if_constant(), right.if_constant()) {
        (Some(a), Some(b)) => match a > b {
            true => return ctx.const_1(),
            false => return ctx.const_0(),
        },
        (Some(c), None) => {
            if c == 0 {
                return ctx.const_0();
            }
            let relbit_mask = right.relevant_bits_mask();
            if c > relbit_mask {
                return ctx.const_1();
            } else if c == relbit_mask {
                // max > x if x != max
                return ctx.neq(left, right);
            }
            if c == 1 {
                // 1 > x if x == 0
                return ctx.eq_const(right, 0);
            }
            if let Some((inner, from, to)) = right.if_sign_extend() {
                return simplify_gt_sext_const(ctx, c, inner, from, to, true);
            }
            if let Some((l, r)) = right.if_arithmetic_sub() {
                if let Some(c2) = r.if_constant() {
                    if let Some((inner, from, to)) = l.if_sign_extend() {
                        let low = c2;
                        let high = c2.wrapping_add(c);
                        // Not sure if this would also work when low > high,
                        // but not doing that now.
                        if high > low {
                            return simplify_gt_sext_range(ctx, low, high, inner, from, to);
                        }
                    }
                }
            }
        }
        (None, Some(c)) => {
            // x > 0 if x != 0
            if c == 0 {
                return ctx.neq(left, right);
            }
            let relbit_mask = left.relevant_bits_mask();
            if c >= relbit_mask {
                return ctx.const_0();
            }
            if let Some((inner, from, to)) = left.if_sign_extend() {
                return simplify_gt_sext_const(ctx, c, inner, from, to, false);
            }
        }
        _ => (),
    }
    let arith = ArithOperand {
        ty: ArithOpType::GreaterThan,
        left,
        right,
    };
    ctx.intern(OperandType::Arithmetic(arith))
}

fn simplify_gt_arith_checks<'e>(
    left_inout: &mut Operand<'e>,
    right_inout: &mut Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Option<Operand<'e>> {
    let mut left = *left_inout;
    let mut right = *right_inout;
    // Normalize (c1 - x) > c2 to (0 - c2 - 1) > (x - c1 - 1)
    // if c2 > sign_bit
    // Similarly (x - c1) > c2 to (0 - c2 - 1) > (x - c2 - c1 - 1)
    if let Some(c2) = right.if_constant() {
        let (left_inner, mask) = Operand::and_masked(left);
        if let Some((l, r)) = left_inner.if_arithmetic_sub() {
            if let Some(c1) = l.if_constant() {
                if c2 > mask >> 1 {
                    left = ctx.constant(0u64.wrapping_sub(c2).wrapping_sub(1) & mask);
                    right = ctx.and_const(
                        ctx.sub_const(
                            r,
                            c1.wrapping_add(1),
                        ),
                        mask,
                    );
                }
            } else if let Some(c1) = r.if_constant() {
                let new_right_c = c1.wrapping_add(c2).wrapping_add(1);
                if new_right_c >= (mask >> 2) + 1 {
                    left = ctx.constant(0u64.wrapping_sub(c2).wrapping_sub(1) & mask);
                    right = ctx.and_const(
                        ctx.sub_const(
                            l,
                            new_right_c,
                        ),
                        mask,
                    );
                }
            }
        }
    }

    // Remove mask in a > ((x - b) & mask)
    // if x <= mask && a + b <= mask
    if let Some((right_inner, mask_op)) = right.if_arithmetic_and() {
        if let Some(mask) = mask_op.if_constant() {
            if let Some((x, b)) = right_inner.if_arithmetic_sub() {
                let mask_is_continuous_from_0 = mask.wrapping_add(1) & mask == 0;
                if mask_is_continuous_from_0 {
                    let a_val = match left.if_constant() {
                        Some(c) => c,
                        None => left.relevant_bits_mask(),
                    };
                    let b_val = match b.if_constant() {
                        Some(c) => c,
                        None => b.relevant_bits_mask(),
                    };
                    let ok = a_val.checked_add(b_val).filter(|&c| c <= mask).is_some() &&
                        x.relevant_bits().end <= mask_op.relevant_bits().end;
                    if ok {
                        right = right_inner;
                    }
                }
            }
        }
    }

    // x - y > x == y > x
    if let Some(new) = simplify_gt_lhs_sub(ctx, left, right) {
        return Some(simplify_gt(new, right, ctx, swzb_ctx));
    } else {
        let (left_inner, mask) = match Operand::and_masked(left) {
            (inner, x) if x == !0u64 =>
                (inner, (1u64 << (inner.relevant_bits().end & 63)).wrapping_sub(1)),
            x => x,
        };
        let (right_inner, mask2) = match Operand::and_masked(right) {
            (inner, x) if x == !0u64 =>
                (inner, (1u64 << (inner.relevant_bits().end & 63)).wrapping_sub(1)),
            x => x,
        };
        // Can simplify x - y > x to y > x if mask starts from bit 0
        let mask_is_continuous_from_0 = mask2.wrapping_add(1) & mask2 == 0;
        if mask & mask2 == mask2 && mask_is_continuous_from_0 {
            for &cand in &[right_inner, right] {
                if let Some(new) = simplify_gt_lhs_sub(ctx, left_inner, cand) {
                    let new = simplify_and_const(new, mask, ctx, swzb_ctx);
                    return Some(simplify_gt(new, right, ctx, swzb_ctx));
                }
            }
        }
    }
    // c1 > (c2 - x) to c1 > (x + (c1 - c2 - 1))
    // Not exactly sure why this works.. Would be good to prove..
    if let Some(c1) = left.if_constant() {
        let (right_inner, mask) = Operand::and_masked(right);
        if let Some((l, r)) = right_inner.if_arithmetic_sub() {
            if let Some(c2) = l.if_constant() {
                if c1 > c2 {
                    right = ctx.and_const(ctx.add_const(r, c1 - c2 - 1), mask);
                }
            }
        }
    }
    *left_inout = left;
    *right_inout = right;
    None
}

/// Note: low inclusive, high exclusive
fn simplify_gt_sext_range<'e>(
    ctx: OperandCtx<'e>,
    low: u64,
    high: u64,
    value: Operand<'e>,
    from: MemAccessSize,
    to: MemAccessSize,
) -> Operand<'e> {
    debug_assert!(low < high);
    let from_mask = from.mask();
    let from_sign = from_mask / 2 + 1;
    let from_sign_high = (to.mask() - from_sign).wrapping_add(1);
    let low_masked = low & from_mask;
    let high_masked = high & from_mask;
    let (from_low, from_high) = if high <= from_sign {
        // Valid range is positive in from, can just return x > value - y
        // Could just be (low, high) since these are positive in from,
        // and as such, the mask doesn't remove anything.
        // But for consistency just using masked values in all of these
        // following branches.
        (low_masked, high_masked)
    } else if high <= from_sign_high {
        // Valid range doesn't include negative values
        if low >= from_sign {
            // Valid range unreachable
            return ctx.const_0();
        } else {
            // Valid range low..sign_bit
            (low_masked, from_sign)
        }
    } else {
        // Valid range ends in negative values
        if low >= from_sign_high {
            // Valid range only negative values
            (low_masked, high_masked)
        } else if low >= from_sign {
            // Valid range sign_bit..high
            (from_sign, high_masked)
        } else {
            // Valid range low..high
            (low_masked, high_masked)
        }
    };
    ctx.gt_const_left(
        from_high.wrapping_sub(from_low),
        ctx.sub_const(
            value,
            from_low,
        ),
    )
}

fn simplify_gt_sext_const<'e>(
    ctx: OperandCtx<'e>,
    c: u64,
    value: Operand<'e>,
    from: MemAccessSize,
    to: MemAccessSize,
    const_on_left: bool,
) -> Operand<'e> {
    // If const on right:
    //   sext(x) > c => x > c if c is positive for `from`
    //   sext(x) > c => x > (from.mask() / 2) if c is in range that sext cannot produce.
    //   sext(x) > c => x > (c & from.mask()) otherwise
    // If const on left:
    //   c > sext(x) => c > x if c is positive for `from`
    //   c > sext(x) => ((from.mask() / 2) + 1) > x if c is in range that sext cannot produce.
    //   c > sext(x) => (c & from.mask()) > x otherwise
    let from_sign = from.mask() / 2 + 1;
    let to_sign = to.mask() / 2 + 1;
    let new_const = if c < from_sign {
        c
    } else if c < to_sign {
        (from.mask() / 2).wrapping_add(const_on_left as u64)
    } else {
        c & from.mask()
    };
    if const_on_left {
        ctx.gt_const_left(new_const, value)
    } else {
        ctx.gt_const(value, new_const)
    }
}

#[test]
fn test_sum_valid_range() {
    let ctx = &crate::OperandContext::new();
    assert_eq!(
        sum_valid_range(
            &[(ctx.mem8(ctx.constant(4), 0), false), (ctx.mem8(ctx.constant(8), 0), true)],
            u64::MAX,
        ),
        (0xffff_ffff_ffff_ff01, 0xff),
    );
    assert_eq!(
        sum_valid_range(&[(ctx.register(4), false)], u64::MAX),
        (0, u64::MAX),
    );
    assert_eq!(
        sum_valid_range(&[(ctx.register(4), true)], u64::MAX),
        (0, u64::MAX),
    );
}

#[test]
fn test_simplify_xor_base_for_shifted() {
    let ctx = &crate::OperandContext::new();
    let mem16 = ctx.mem16(ctx.register(0), 0);
    let mem8_2 = ctx.mem32(ctx.register(0), 2);
    let mem8_3 = ctx.mem32(ctx.register(0), 3);
    let limit = &mut 500000;
    assert_eq!(
        simplify_xor_base_for_shifted(mem16, (0, 0xffff), mem8_2, (0x10, 0xff), ctx, limit),
        // and_const wouldn't be exactly required by the function's contract
        // but is a valid return value.
        Some(ctx.and_const(ctx.mem32(ctx.register(0), 0), 0xffffff)),
    );
    assert_eq!(
        simplify_xor_base_for_shifted(mem16, (0, 0xffff), mem8_3, (0x18, 0x34), ctx, limit),
        Some(ctx.and_const(ctx.mem32(ctx.register(0), 0), 0x3400ffff)),
    );
    assert_eq!(
        simplify_xor_base_for_shifted(mem16, (0, 0xffff), mem16, (0, 0xf_ffff), ctx, limit),
        Some(ctx.mem16(ctx.register(0), 0)),
    );
    assert_eq!(
        simplify_xor_base_for_shifted(mem16, (0, 0xf_ffff), mem8_3, (0x18, 0x34), ctx, limit),
        Some(ctx.and_const(ctx.mem32(ctx.register(0), 0), 0x3400_ffff)),
    );
    // This must be none, as Mem16 with mask f_ffff requires the lower half of third byte be 0
    assert_eq!(
        simplify_xor_base_for_shifted(mem16, (0, 0xf_ffff), mem8_2, (0x10, 0x34), ctx, limit),
        None,
    );
}

#[test]
fn simplify_with_and_mask_reduce_inner_and_mask() {
    let ctx = &super::OperandContext::new();

    // Check that outer mask gets reduced, and that inner mask gets removed
    // in single simplify_with_and_mask call
    // ctx.and_const(op1, 0xffff) has enough redundancy that it will still simplify
    // things correctly.
    // Note: Previously checked that result would be
    // `(rax & fffe) - 6b3e` instead, which would also be valid.
    // Not going to take stance which is more correct, though having and
    // mask moved outside looks nice.
    let op1 = ctx.and_const(
        ctx.sub_const(
            ctx.and_const(
                ctx.register(0),
                0x1_fffe,
            ),
            0x6b3e,
        ),
        0x1_fffe,
    );
    let op1 = simplify_with_and_mask(op1, 0xffff, ctx, &mut SimplifyWithZeroBits::default());
    let eq1 = ctx.and_const(
        ctx.sub_const(
            ctx.register(0),
            0x6b3e,
        ),
        0xfffe,
    );
    assert_eq!(op1, eq1);
}
