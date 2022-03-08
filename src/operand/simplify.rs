use std::cmp::{min, max};
use std::ops::Range;

use smallvec::SmallVec;

use crate::bit_misc::{bits_overlap, one_bit_ranges, zero_bit_ranges};
use crate::heapsort;

use super::{
    ArithOperand, ArithOpType, MemAccess, MemAccessSize, Operand, OperandType, OperandCtx,
};
use super::slice_stack::{self, SizeLimitReached};

type Slice<'e> = slice_stack::Slice<'e, Operand<'e>>;
type AddSlice<'e> = slice_stack::Slice<'e, (Operand<'e>, bool)>;

#[derive(Default)]
pub struct SimplifyWithZeroBits {
    simplify_count: u8,
    with_and_mask_count: u8,
    /// simplify_with_zero_bits can cause a lot of recursing in xor
    /// simplification with has functions, stop simplifying if a limit
    /// is hit.
    xor_recurse: u8,
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
        ArithOpType::Parity => {
            let val = left;
            if let Some(c) = val.if_constant() {
                return match (c as u8).count_ones() & 1 == 0 {
                    true => ctx.const_1(),
                    false => ctx.const_0(),
                }
            } else {
                let ty = OperandType::Arithmetic(ArithOperand {
                    ty: ArithOpType::Parity,
                    left: val,
                    right: ctx.const_0(),
                });
                ctx.intern(ty)
            }
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
                left = simplify_with_and_mask(left, mask, ctx, swzb_ctx);
                right = simplify_with_and_mask(right, mask, ctx, swzb_ctx);
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
        ctx.and_const(
            val,
            mask,
        )
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
                let overflow = float > i32::max_value() as f32 ||
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
                let overflow = float > i64::max_value() as f64 ||
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

/// Iterates through parts of a arithmetic op tree where it is guaranteed
/// that right has never subtrees, only left (E.g. for and, or, xor).
///
/// Order is outermost right first, then next inner right, etc.
#[derive(Copy, Clone)]
struct IterArithOps<'e> {
    ty: ArithOpType,
    next: Option<Operand<'e>>,
    next_inner: Option<Operand<'e>>,
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
                                            ty: ArithOpType::Xor,
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

/// Merges (x & ff), (y & ff) to (x ^ y) & ff
/// Also used for or
fn simplify_xor_merge_ands_with_same_mask<'e>(
    ops: &mut Slice<'e>,
    is_or: bool,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) {
    let mut i = 0;
    while i < ops.len() {
        if let Some((_, r)) = ops[i].if_arithmetic_and() {
            if let Some(mask) = r.if_constant() {
                let any_matching = (ops[(i + 1)..]).iter().any(|x| {
                    x.if_arithmetic_and()
                        .and_then(|x| x.1.if_constant())
                        .filter(|&c| c == mask)
                        .is_some()
                });
                if any_matching {
                    let result = ctx.simplify_temp_stack
                        .alloc(|slice| {
                            for op in &ops[i..] {
                                if let Some((l, r)) = op.if_arithmetic_and() {
                                    if r.if_constant() == Some(mask) {
                                        slice.push(l)?;
                                    }
                                }
                            }
                            if is_or {
                                simplify_or_ops(slice, ctx, swzb_ctx)
                            } else {
                                simplify_xor_ops(slice, ctx, swzb_ctx)
                            }
                        });
                    if let Ok(result) = result {
                        let masked = ctx.and_const(result, mask);
                        ops[i] = masked;
                        for j in ((i + 1)..ops.len()).rev() {
                            let matched = ops[j].if_arithmetic_and()
                                .and_then(|x| x.1.if_constant())
                                .filter(|&c| c == mask)
                                .is_some();
                            if matched {
                                ops.swap_remove(j);
                            }
                        }
                    }
                }
            }
        }
        i += 1;
    }
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
            simplify_or_merge_mem(ops, ctx); // Yes, this is supposed to stay valid for xors.
            simplify_or_merge_child_ands(ops, ctx, ArithOpType::Xor)?;
            simplify_xor_merge_ands_with_same_mask(ops, false, ctx, swzb_ctx);
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
                collect_xor_ops(l, ops, usize::max_value())?;
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
    // Canonicalize to (x ^ y) & ffff over x ^ (y & ffff)
    // when the outermost mask doesn't modify x
    // Keep op with the mask to avoid reinterning it.
    let best_mask = ops.iter()
        .fold(None, |prev: Option<(u64, Operand<'e>)>, &op| {
            if let Some(new) = op.if_arithmetic_and()
                .and_then(|x| x.1.if_constant().map(|c| (c, x.1)))
            {
                if let Some(prev) = prev {
                    if prev.0 & new.0 == new.0 {
                        Some(prev)
                    } else if prev.0 & new.0 == prev.0 {
                        Some(new)
                    } else {
                        None
                    }
                } else {
                    Some(new)
                }
            } else {
                prev
            }
        })
        .filter(|&(mask, _op)| {
            ops.iter().all(|x| {
                let relbits = x.relevant_bits_mask();
                relbits & mask == relbits
            }) && mask & const_val == const_val
        });
    if let Some((mask, _op)) = best_mask {
        for i in 0..ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic_and() {
                if let Some(c) = r.if_constant() {
                    if l.relevant_bits_mask() & mask == c {
                        if let Some((l, r)) = l.if_arithmetic(ArithOpType::Xor) {
                            ops[i] = r;
                            collect_xor_ops(l, ops, usize::max_value())?;
                        } else {
                            ops[i] = l;
                        }
                    }
                }
            }
        }
    }
    heapsort::sort(ops);
    let mut tree = ops.pop()
        .unwrap_or_else(|| ctx.const_0());
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
    if let Some((_, op)) = best_mask {
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: tree,
            right: op,
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    Ok(tree)
}

/// Assumes that `ops` is sorted.
fn simplify_xor_remove_reverting<'e>(ops: &mut Slice<'e>) {
    let mut first_same = ops.len() as isize - 1;
    let mut pos = first_same - 1;
    while pos >= 0 {
        let pos_u = pos as usize;
        let first_u = first_same as usize;
        if ops[pos_u] == ops[first_u] {
            ops.remove(first_u);
            ops.remove(pos_u);
            first_same -= 2;
            if pos > first_same {
                pos = first_same
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
            let arith = ArithOperand {
                ty: ArithOpType::Lsh,
                left: left.clone(),
                right: right.clone(),
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        }
    };
    if constant >= 256 {
        return ctx.const_0();
    }
    simplify_lsh_const(left, constant as u8, ctx, swzb_ctx)
}

pub fn simplify_lsh_const<'e>(
    left: Operand<'e>,
    constant: u8,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    let default = || {
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
    };
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
                    // (If it changes anything. Otherwise prefer keeping the shift/mul outside)
                    if let Some(c) = arith.right.if_constant() {
                        let high = 64 - zero_bits.start;
                        let low = left.relevant_bits().start;
                        let no_op_mask = !0u64 >> low << low << high >> high;

                        let new = simplify_lsh_const(arith.left, constant, ctx, swzb_ctx);
                        let changed = match *new.ty() {
                            OperandType::Arithmetic(ref a) => a.left != arith.left,
                            _ => true,
                        };
                        if changed {
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
    let default = || {
        let arith = ArithOperand {
            ty: ArithOpType::Rsh,
            left: left,
            right: right,
        };
        ctx.intern(OperandType::Arithmetic(arith))
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left;
    } else if constant >= left.relevant_bits().end.into() {
        return ctx.const_0();
    }
    let constant = constant as u8;

    let new_left = simplify_with_and_mask(left, u64::MAX << constant, ctx, swzb_ctx);
    let zero_bits = 0..constant;
    let new_left = match simplify_with_zero_bits(new_left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => s,
    };
    if new_left != left {
        return simplify_rsh(new_left, right, ctx, swzb_ctx);
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
                            let new = simplify_rsh(other, right, ctx, swzb_ctx);
                            return new;
                        }
                        // `(x & c) >> constant` can be simplified to
                        // `(x >> constant) & (c >> constant)
                        // With lsh/rsh it can simplify further,
                        // but do it always for canonicalization
                        let new = simplify_rsh(other, right, ctx, swzb_ctx);
                        let new = simplify_and_const(new, c >> constant, ctx, swzb_ctx);
                        return new;
                    }
                    let arith = ArithOperand {
                        ty: ArithOpType::Rsh,
                        left: left.clone(),
                        right: right.clone(),
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
                                    *op = simplify_rsh(*op, right, ctx, swzb_ctx);
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
                                    *op = simplify_rsh(*op, right, ctx, swzb_ctx);
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
                            simplify_rsh(arith.left, ctx.constant(sum.into()), ctx, swzb_ctx)
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
            let c = ctx.constant(u64::from(constant - offset_add * 8));
            let new = ctx.mem_any(size, address, offset.wrapping_add(u64::from(offset_add)));
            simplify_rsh(new, c, ctx, swzb_ctx)
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
        (0, u64::max_value())
    })
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
    if let Some((val, from, to)) = left.if_sign_extend() {
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
        if let Some((l, r)) = left.if_arithmetic_eq() {
            // Check for (x == 0) == 0 => x
            if r == ctx.const_0() {
                if l.relevant_bits().end == 1 {
                    return l;
                }
            }
        }
        // (x | c) == 0 => 0
        if let Some((_, r)) = left.if_arithmetic_or() {
            if r.if_constant().is_some() {
                return ctx.const_0();
            }
        }
        // Simplify x - y == 0 as x == y
        if let Some((l, r)) = left.if_arithmetic_sub() {
            return simplify_eq(l, r, ctx);
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
    if left_bits.end != 64 {
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
                ArithOpType::And => {
                    // (x << 8) & 800 == 0 gets simplfied to x & 8 == 0
                    // (x >> 8) & 8 == 0 gets simplfied to x & 800 == 0
                    // (x + ffff) & ffff == 0 becomes x & ffff == 1
                    // And similar for sub.
                    if let Some(c) = arith.right.if_constant() {
                        if let OperandType::Arithmetic(ref arith) = arith.left.ty() {
                            match arith.ty {
                                ArithOpType::Add | ArithOpType::Sub => {
                                    let continous_mask = c.wrapping_add(1) & c == 0;
                                    !continous_mask
                                }
                                ArithOpType::Lsh | ArithOpType::Rsh | ArithOpType::Mul => false,
                                _ => true,
                            }
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                }
                ArithOpType::GreaterThan => {
                    // If right > 1, it gets caught by the max_value check above,
                    // if right == 1, it gets caught by the right == 1 check above,
                    // so right must be 0.
                    debug_assert_eq!(right, 0);
                    if let Some(result) = simplify_eq_zero_with_gt(arith.left, arith.right, ctx) {
                        return result;
                    }
                    true
                }
                _ => true,
            };
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

fn simplify_eq_main<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    // Equality is just bit comparision without overflow semantics, even though
    // this also uses x == y => x - y == 0 property to simplify it.
    let shared_mask = left.relevant_bits_mask() | right.relevant_bits_mask();
    let add_sub_mask = if shared_mask == 0 {
        u64::max_value()
    } else {
        u64::max_value() >> shared_mask.leading_zeros()
    };
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
    heapsort::sort_by(ops, |a, b| Operand::and_masked(a.0) < Operand::and_masked(b.0));
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
                        if mask != u64::max_value() {
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
    fn mask_maskee<'e>(x: Operand<'e>) -> Option<(u64, Operand<'e>)> {
        match x.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
                arith.right.if_constant().map(|c| (c, arith.left))
            }
            OperandType::Memory(mem) => {
                match mem.size {
                    MemAccessSize::Mem8 => Some((0xff, x)),
                    MemAccessSize::Mem16 => Some((0xffff, x)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    let (left, right) = match left < right {
        true => (left, right),
        false => (right, left),
    };

    if let Some(result) = simplify_eq_2op_check_signed_less(ctx, left, right) {
        return result;
    }
    // Try to prove (x & mask) == ((x + c) & mask) true/false.
    // If c & mask == 0, it's true if c & mask2 == 0, otherwise unknown
    //    mask2 is mask, where 0-bits whose next bit is 1 are switched to 1.
    // If c & mask == mask, it's unknown, unless mask contains the bit 0x1, in which
    // case it's false
    // Otherwise it's false.
    //
    // This can be deduced from how binary addition works; for digit to not change, the
    // added digit needs to either be 0, or 1 with another 1 carried from lower digit's
    // addition.
    //
    // TODO is this necessary anymore?
    // Probably could be simpler to do just with relevant_bits?
    {
        let left_const = mask_maskee(left);
        let right_const = mask_maskee(right);
        if let (Some((mask1, l)), Some((mask2, r))) = (left_const, right_const) {
            if mask1 == mask2 {
                let add_const = simplify_eq_masked_add(l).map(|(c, other)| (other, r, c))
                    .or_else(|| {
                        simplify_eq_masked_add(r).map(|(c, other)| (other, l, c))
                    });
                if let Some((a, b, added_const)) = add_const {
                    let a = simplify_with_and_mask(
                        a,
                        mask1,
                        ctx,
                        &mut SimplifyWithZeroBits::default(),
                    );
                    if a == b {
                        match added_const & mask1 {
                            0 => {
                                // TODO
                            }
                            x if x == mask1 => {
                                if mask1 & 1 == 1 {
                                    return ctx.const_0();
                                }
                            }
                            _ => return ctx.const_0(),
                        }
                    }
                }
            }
        }
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

fn simplify_eq_masked_add<'e>(operand: Operand<'e>) -> Option<(u64, Operand<'e>)> {
    match operand.ty() {
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Add => {
            arith.left.if_constant().map(|c| (c, arith.right))
                .or_else(|| arith.left.if_constant().map(|c| (c, arith.left)))
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Sub => {
            arith.left.if_constant().map(|c| (0u64.wrapping_sub(c), arith.right))
                .or_else(|| arith.right.if_constant().map(|c| (0u64.wrapping_sub(c), arith.left)))
        }
        _ => None,
    }
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
        match sum.checked_add(max) {
            Some(s) => sum = s,
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

/// Return Some(smaller) if the the operand with smaller mask is same as the
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
        return Some(a.clone());
    }
    if let Some(a) = a.if_constant() {
        if let Some(b) = b.if_constant() {
            return Some(ctx.constant(a | b));
        }
    }
    if let Some((val, shift)) = is_offset_mem(a, ctx) {
        if let Some((other_val, other_shift)) = is_offset_mem(b, ctx) {
            if val == other_val {
                let shift = (
                    shift.0,
                    shift.1,
                    shift.2,
                );
                let other_shift = (
                    other_shift.0,
                    other_shift.1,
                    other_shift.2,
                );
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
    let new = simplify_with_and_mask_mem(left, mem, right, ctx);
    let new_mask = new.relevant_bits_mask();
    let new_common = new_mask & right;
    if new_common != new_mask {
        if new_common != right {
            right_op = None;
        }
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left: new,
            right: right_op.unwrap_or_else(|| ctx.constant(new_common)),
        };
        Some(ctx.intern(OperandType::Arithmetic(arith)))
    } else {
        Some(new)
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
    if right == u64::max_value() {
        return left;
    }
    // Check if left is x & const
    if let Some((l, r)) = left.if_arithmetic_and() {
        if let Some(c) = r.if_constant() {
            let new = c & right;
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
    if !bits_overlap(&left.relevant_bits(), &right.relevant_bits()) {
        return ctx.const_0();
    }
    let const_other = match left.if_constant() {
        Some(c) => Some((c, left, right)),
        None => match right.if_constant() {
            Some(c) => Some((c, right, left)),
            None => None,
        },
    };
    if let Some((c, c_op, other)) = const_other {
        return simplify_and_const_op(other, c, Some(c_op), ctx, swzb_ctx);
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

fn simplify_and_main<'e>(
    ops: &mut Slice<'e>,
    mut const_remain: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Operand<'e>, SizeLimitReached> {
    // Keep second mask in form 00000011111 (All High bits 0, all low bits 1),
    // as that allows simplifying add/sub/mul a bit more
    let mut low_const_remain = !0u64;
    loop {
        for op in ops.iter() {
            let relevant_bits = op.relevant_bits();
            if relevant_bits.start == 0 {
                let shift = (64 - relevant_bits.end) & 63;
                low_const_remain = low_const_remain << shift >> shift;
            }
        }
        const_remain = ops.iter()
            .map(|op| match op.if_constant() {
                Some(c) => c,
                None => op.relevant_bits_mask(),
            })
            .fold(const_remain, |sum, x| sum & x);
        ops.retain(|x| x.if_constant().is_none());
        if ops.is_empty() || const_remain == 0 {
            return Ok(ctx.constant(const_remain));
        }
        let crem_high_zeros = const_remain.leading_zeros();
        low_const_remain = low_const_remain << crem_high_zeros >> crem_high_zeros;

        if ops.len() > 1 {
            heapsort::sort(ops);
            ops.dedup();
            simplify_and_remove_unnecessary_ors(ops, const_remain);
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
        if low_const_remain != !0 && low_const_remain != const_remain {
            slice_filter_map(ops, |op| {
                let benefits_from_low_mask = match *op.ty() {
                    OperandType::Arithmetic(ref arith) => match arith.ty {
                        ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul => true,
                        _ => false,
                    },
                    _ => false,
                };
                if !benefits_from_low_mask {
                    return Some(op);
                }

                let new = simplify_with_and_mask(op, low_const_remain, ctx, swzb_ctx);
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
        // Simplify (x | y) & mask to (x | (y & mask)) if mask is useless to x
        let mut const_remain_necessary = true;
        for i in 0..ops.len() {
            if let Some((l, r)) = ops[i].if_arithmetic_or() {
                let left_mask = match l.if_constant() {
                    Some(c) => c,
                    None => l.relevant_bits_mask(),
                };
                let right_mask = match r.if_constant() {
                    Some(c) => c,
                    None => r.relevant_bits_mask(),
                };
                let left_needs_mask = left_mask & const_remain != left_mask;
                let right_needs_mask = right_mask & const_remain != right_mask;
                if !left_needs_mask && right_needs_mask && left_mask != const_remain {
                    let constant = const_remain & right_mask;
                    let masked = simplify_and_const(r, constant, ctx, swzb_ctx);
                    let new = simplify_or(l, masked, ctx, swzb_ctx);
                    ops[i] = new;
                    const_remain_necessary = false;
                    ops_changed = true;
                } else if left_needs_mask && !right_needs_mask && right_mask != const_remain {
                    let constant = const_remain & left_mask;
                    let masked = simplify_and_const(l, constant, ctx, swzb_ctx);
                    let new = simplify_or(r, masked, ctx, swzb_ctx);
                    ops[i] = new;
                    const_remain_necessary = false;
                    ops_changed = true;
                }
            }
        }
        if !const_remain_necessary {
            // All ops were masked with const remain, so it should not be useful anymore
            const_remain = u64::max_value();
        }

        let mut i = 0;
        let mut end = ops.len();
        while i < end {
            if let Some((l, r)) = ops[i].if_arithmetic_and() {
                ops.swap_remove(i);
                end -= 1;
                collect_and_ops(l, ops, usize::max_value())?;
                if let Some(c) = r.if_constant() {
                    const_remain &= c;
                } else {
                    collect_and_ops(r, ops, usize::max_value())?;
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
    simplify_and_merge_gt_const(ops, ctx);

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

    let relevant_bits = ops.iter().fold(!0, |bits, op| {
        bits & op.relevant_bits_mask()
    });
    // Don't use a const mask which has all 1s for relevant bits.
    let final_const_remain = if const_remain & relevant_bits == relevant_bits {
        0
    } else {
        const_remain & relevant_bits
    };
    match ops.len() {
        0 => return Ok(ctx.constant(final_const_remain)),
        1 if final_const_remain == 0 => return Ok(ops[0]),
        _ => {
            heapsort::sort(ops);
            ops.dedup();
        }
    };
    simplify_and_remove_unnecessary_ors(ops, const_remain);

    let mut tree = ops.pop()
        .unwrap_or_else(|| ctx.const_0());
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
fn simplify_and_remove_unnecessary_ors<'e>(
    ops: &mut Slice<'e>,
    const_remain: u64,
) {
    fn contains_or<'e>(op: Operand<'e>, check: Operand<'e>) -> bool {
        if let Some((l, r)) = op.if_arithmetic_or() {
            if l == check || r == check {
                true
            } else {
                contains_or(l, check) || contains_or(r, check)
            }
        } else {
            false
        }
    }

    fn contains_or_const(op: Operand<'_>, check: u64) -> bool {
        if let Some((_, r)) = op.if_arithmetic_or() {
            if let Some(c) = r.if_constant() {
                c & check == check
            } else {
                false
            }
        } else {
            false
        }
    }

    let mut pos = 0;
    while pos < ops.len() {
        let mut j = 0;
        while j < ops.len() {
            if j == pos {
                j += 1;
                continue;
            }
            if contains_or(ops[j], ops[pos]) {
                ops.remove(j);
                // `j` can be before or after `pos`,
                // depending on that `pos` may need to be decremented
                if j < pos {
                    pos -= 1;
                }
            } else {
                j += 1;
            }
        }
        pos += 1;
    }
    for j in (0..ops.len()).rev() {
        if contains_or_const(ops[j], const_remain) {
            ops.swap_remove(j);
        }
    }
}

fn simplify_and_merge_child_ors<'e>(ops: &mut Slice<'e>, ctx: OperandCtx<'e>) {
    fn or_const<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>)> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Or => {
                arith.right.if_constant().map(|c| (c, arith.left))
            }
            _ => None,
        }
    }

    let mut i = 0;
    while i < ops.len() {
        let op = ops[i];
        let mut new = None;
        let mut changed = false;
        if let Some((mut constant, val)) = or_const(op) {
            for j in ((i + 1)..ops.len()).rev() {
                let second = ops[j];
                if let Some((other_constant, other_val)) = or_const(second) {
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
fn simplify_and_merge_gt_const<'e>(ops: &mut Slice<'e>, ctx: OperandCtx<'e>) {
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
                        .unwrap_or_else(|| (u64::max_value(), arith.right));
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
                        .unwrap_or_else(|| (u64::max_value(), arith.left));
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
            if larger_masked.0 == larger_masked.1 && larger_masked.2 == u64::max_value() {
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
                *base = (0, u64::max_value(), u64::max_value(), base.3, false);
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
                *base = (0, u64::max_value(), u64::max_value(), base.3, false);
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
        return;
    }

    let mut i = 0;
    while i < ops.len() {
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
                let new = if min == 0 && max == mask {
                    // Always 0/1
                    ctx.constant(if set { 1 } else { 0 })
                } else {
                    if set {
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
                    }
                };
                ops[i] = new;
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

/// "Simplify bitwise or: merge child ands"
/// Converts things like [x & const1, x & const2] to [x & (const1 | const2)]
///
/// Also used by xors with only_nonoverlapping true
fn simplify_or_merge_child_ands<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    arith_ty: ArithOpType,
) -> Result<(), SizeLimitReached> {
    fn and_const<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>)> {
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
        if let Some((constant, val)) = and_const(ops[i]) {
            let mut j = i + 1;
            while j < ops.len() {
                if let Some((other_constant, other_val)) = and_const(ops[j]) {
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
                if let Some(first) = intern_arith_ops_to_tree(ctx, iter, arith_ty) {
                    out.push(ctx.and_const(first, first_c));
                }
                let iter = other_slice.iter().rev().copied();
                if let Some(second) = intern_arith_ops_to_tree(ctx, iter, arith_ty) {
                    out.push(ctx.and_const(second, second_c));
                }
            }
            Result::<(), SizeLimitReached>::Ok(())
        })
    });
}

/// Use only if the iterator produces items in sorted order.
fn intern_arith_ops_to_tree<'e, I: Iterator<Item = Operand<'e>>>(
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

// Simplify or: merge comparisions
// Converts
// (c > x) | (c == x) to (c + 1 > x),
//      More general: (c1 > x - c2) | (c1 + c2) == x to (c1 + 1 > x - c2)
// (x > c) | (x == c) to (x > c + 1).
// (x == 0) | (x == 1) to (2 > x)
// Cannot do for values that can overflow, so just limit it to constants for now.
// (Well, could do (c + 1 > x) | (c == max_value), but that isn't really simpler)
fn simplify_or_merge_comparisions<'e>(ops: &mut Slice<'e>, ctx: OperandCtx<'e>) {
    #[derive(Eq, PartialEq, Copy, Clone)]
    enum MatchType {
        ConstantGreater,
        ConstantLess,
        Equal,
    }

    fn check_match<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>, MatchType)> {
        match op.ty() {
            OperandType::Arithmetic(arith) => {
                let left = arith.left;
                let right = arith.right;
                match arith.ty {
                    ArithOpType::Equal => {
                        let c = right.if_constant()?;
                        return Some((c, left, MatchType::Equal));
                    }
                    ArithOpType::GreaterThan => {
                        if let Some(c) = left.if_constant() {
                            return Some((c, right, MatchType::ConstantGreater));
                        }
                        if let Some(c) = right.if_constant() {
                            return Some((c, left, MatchType::ConstantLess));
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
        if let Some((c, x, ty)) = check_match(ops[i]) {
            let mut j = i + 1;
            while j < ops.len() {
                if let Some((c2, x2, ty2)) = check_match(ops[j]) {
                    match (ty, ty2) {
                        (MatchType::ConstantGreater, MatchType::Equal) |
                            (MatchType::Equal, MatchType::ConstantGreater) =>
                        {
                            // (c1 > y - c2) | (c1 + c2) == y to (c1 + 1 > y - c2)
                            let (gt, c1, eq, eq_c) = match ty == MatchType::ConstantGreater {
                                true => (x, c, x2, c2),
                                false => (x2, c2, x, c),
                            };
                            let (gt_inner, gt_mask) = Operand::and_masked(gt);
                            let (y, c2) = gt_inner.if_arithmetic_sub()
                                .and_then(|(l, r)| {
                                    Some((l, r.if_constant()?))
                                })
                                .unwrap_or((gt_inner, 0));
                            let constants_match = c1.checked_add(c2)
                                .filter(|&c| c == eq_c)
                                .is_some();
                            let values_match = if (y, gt_mask) == Operand::and_masked(eq) {
                                true
                            } else {
                                // Can also be that y == eq == (smth & ff),
                                // in which case the previous check would have done
                                // (y, u64::MAX) != (smth, ff)
                                y == eq && gt_mask == u64::MAX
                            };
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
                            if c == c2 && x == x2 {
                                if let Some(new_c) = c.checked_sub(1) {
                                    ops[i] = ctx.gt_const(x, new_c);
                                    ops.swap_remove(j);
                                    continue 'outer;
                                }
                            }
                        }
                        _ => (),
                    }
                    if c.min(c2) == 0 && c.max(c2) == 1 && x == x2 &&
                        ty == MatchType::Equal && ty2 == MatchType::Equal
                    {
                        ops[i] = ctx.gt_const_left(2, x);
                        ops.swap_remove(j);
                        continue 'outer;
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
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
            ops.push(s)?;
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
    collect_arith_ops(s, ops, ArithOpType::Or, usize::max_value())
}

fn collect_xor_ops<'e>(
    s: Operand<'e>,
    ops: &mut Slice<'e>,
    limit: usize,
) -> Result<(), SizeLimitReached> {
    collect_arith_ops(s, ops, ArithOpType::Xor, limit)
}

/// Return (base, (offset, len, value_offset))
///
/// E.g. Mem32[x + 100] => (x, (100, 4, 0))
///     (Mem32[x + 100]) << 20 => (x, (100, 4, 4))
fn is_offset_mem<'e>(
    op: Operand<'e>,
    ctx: OperandCtx<'e>,
) -> Option<(Operand<'e>, (u64, u32, u32))> {
    match op.ty() {
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Lsh => {
            if let Some(c) = arith.right.if_constant() {
                if c & 0x7 == 0 && c < 0x40 {
                    let bytes = (c / 8) as u32;
                    return is_offset_mem(arith.left, ctx)
                        .map(|(x, (off, len, val_off))| {
                            (x, (off, len, val_off + bytes))
                        });
                }
            }
            None
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Rsh => {
            if let Some(c) = arith.right.if_constant() {
                if c & 0x7 == 0 && c < 0x40 {
                    let bytes = (c / 8) as u32;
                    return is_offset_mem(arith.left, ctx)
                        .and_then(|(x, (off, len, val_off))| {
                            if bytes < len {
                                let off = off.wrapping_add(bytes as u64);
                                Some((x, (off, len - bytes, val_off)))
                            } else {
                                None
                            }
                        });
                }
            }
            None
        }
        OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
            if let Some(c) = arith.right.if_constant() {
                if c.wrapping_add(1) & c == 0 {
                    return is_offset_mem(arith.left, ctx)
                        .map(|(x, (off, _, val_off))| {
                            let len_bits = op.relevant_bits().end;
                            let len = len_bits.wrapping_add(7) / 8;
                            (x, (off, len as u32, val_off))
                        })
                }
            }
            None
        }
        OperandType::Memory(ref mem) => {
            let len = mem.size.bits() / 8;
            let (base, offset) = mem.address();
            Some((base, (offset, len, 0)))
        }
        _ => None,
    }
}

/// Returns simplified operands.
///
/// shift and other_shift are (offset, len, left_shift_in_operand)
/// E.g. `Mem32[x + 6] << 0x18` => (6, 4, 3)
fn try_merge_memory<'e>(
    val: Operand<'e>,
    shift: (u64, u32, u32),
    other_shift: (u64, u32, u32),
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    let (shift, other_shift) = match shift.2 < other_shift.2 {
        true => (shift, other_shift),
        false => (other_shift, shift),
    };
    let (off1, len1, val_off1) = shift;
    let (off2, len2, val_off2) = other_shift;
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
        5 | 6 | 7 => ctx.and_const(ctx.mem64(val, off1), u64::max_value() >> ((8 - len) << 3)),
        8 => ctx.mem64(val, off1),
        _ => return None,
    };
    if val_off1 != 0 {
        oper = ctx.lsh_const(
            oper,
            (val_off1 << 3).into(),
        );
    }
    Some(oper)
}

/// Simplify or: merge memory
/// Converts (Mem32[x] >> 8) | (Mem32[x + 4] << 18) to Mem32[x + 1]
/// Also used for xor since x ^ y == x | y if x and y do not overlap at all.
fn simplify_or_merge_mem<'e>(ops: &mut Slice<'e>, ctx: OperandCtx<'e>) {
    let mut i = 0;
    'outer: while i < ops.len() {
        if let Some((val, shift)) = is_offset_mem(ops[i], ctx) {
            let mut j = i + 1;
            while j < ops.len() {
                if let Some((other_val, other_shift)) = is_offset_mem(ops[j], ctx) {
                    if val == other_val {
                        if let Some(merged) = try_merge_memory(val, other_shift, shift, ctx) {
                            ops[i] = merged;
                            ops.swap_remove(j);
                            continue 'outer;
                        }
                    }
                }
                j += 1;
            }
        }
        i += 1;
    }
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
        simplify_add_sub_ops(ops, left, right, is_sub, u64::max_value(), ctx)?;
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
    if let Some((l, r)) = check_quick_arith_simplify(left, right) {
        let r_const = r.if_constant().unwrap_or(0);
        let left_bits = l.relevant_bits_mask();
        if left_bits & r_const == left_bits {
            return r;
        }
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: l,
            right: r,
        };
        return ctx.intern(OperandType::Arithmetic(arith));
    }
    // Simplify (x & a) | (y & a) to (x | y) & a
    if let Some((l, r, mask)) = check_shared_and_mask(left, right) {
        let inner = simplify_or(l, r, ctx, swzb);
        return simplify_and(inner, mask, ctx, swzb);
    }

    ctx.simplify_temp_stack().alloc(|ops| {
        collect_or_ops(left, ops)
            .and_then(|()| collect_or_ops(right, ops))
            .and_then(|()| simplify_or_ops(ops, ctx, swzb))
            .unwrap_or_else(|_| {
                let arith = ArithOperand {
                    ty: ArithOpType::Or,
                    left,
                    right,
                };
                ctx.intern(OperandType::Arithmetic(arith))
            })
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
        if ops.is_empty() || const_val == u64::max_value() {
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
        for bits in one_bit_ranges(const_val) {
            slice_filter_map(ops, |op| simplify_with_one_bits(op, &bits, ctx));
        }
        if ops.len() > 1 {
            simplify_or_merge_child_ands(ops, ctx, ArithOpType::Or)?;
            simplify_or_merge_xors(ops, ctx, swzb_ctx);
            simplify_or_merge_mem(ops, ctx);
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
    if const_val != 0 {
        let arith = ArithOperand {
            ty: ArithOpType::Or,
            left: tree,
            right: ctx.constant(const_val),
        };
        tree = ctx.intern(OperandType::Arithmetic(arith));
    }
    Ok(tree)
}

/// Counts xor ops, descending into x & c masks, as
/// simplify_rsh/lsh do that as well.
/// Too long xors should not be tried to be simplified in shifts.
fn simplify_shift_is_too_long_xor(ops: &[Operand<'_>]) -> bool {
    fn count(op: Operand<'_>) -> usize {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
                if arith.right.if_constant().is_some() {
                    count(arith.left)
                } else {
                    1
                }
            }
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Xor => {
                count(arith.left) + count(arith.right)
            }
            _ => 1,
        }
    }

    const LIMIT: usize = 16;
    if ops.len() > LIMIT {
        return true;
    }
    let mut sum = 0;
    for &op in ops {
        sum += count(op);
        if sum > LIMIT {
            break;
        }
    }
    sum > LIMIT
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
    if swzb_ctx.with_and_mask_count > 120 {
        #[cfg(feature = "fuzz")]
        tls_simplification_incomplete();
        true
    } else {
        false
    }
}

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
    if relevant_mask & mask == relevant_mask {
        if op.0.flags & super::FLAG_COULD_REMOVE_CONST_AND == 0 {
            return op;
        }
    }
    if should_stop_with_and_mask(swzb_ctx) {
        return op;
    }
    swzb_ctx.with_and_mask_count += 1;
    let op = simplify_with_and_mask_inner(op, mask, ctx, swzb_ctx);
    op
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
                    let simplified_right;
                    if let Some(c) = arith.right.if_constant() {
                        let self_mask = mask & arith.left.relevant_bits_mask();
                        if c == self_mask {
                            return arith.left;
                        } else if c & self_mask == 0 {
                            return ctx.const_0();
                        } else if c & mask == c {
                            // Mask is superset of the already existing mask,
                            // so it won't simplify anything further
                            return op;
                        } else {
                            // This is just avoid recursing to simplify_with_and_mask
                            // when it's already known to do this.
                            simplified_right = ctx.constant(c & mask);
                        }
                    } else {
                        simplified_right =
                            simplify_with_and_mask(arith.right, mask, ctx, swzb_ctx);
                    }
                    let simplified_left =
                        simplify_with_and_mask(arith.left, mask, ctx, swzb_ctx);
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op;
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op
                    } else {
                        let op = simplify_and(simplified_left, simplified_right, ctx, swzb_ctx);
                        simplify_with_and_mask(op, mask, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Or => {
                    let simplified_left = simplify_with_and_mask(arith.left, mask, ctx, swzb_ctx);
                    if let Some(c) = simplified_left.if_constant() {
                        if mask & c == mask & arith.right.relevant_bits_mask() {
                            return simplified_left;
                        }
                    }
                    let simplified_right =
                        simplify_with_and_mask(arith.right, mask, ctx, swzb_ctx);
                    if let Some(c) = simplified_right.if_constant() {
                        if mask & c == mask & arith.left.relevant_bits_mask() {
                            return simplified_right;
                        }
                    }
                    // Possibly common to get zeros here
                    let zero = ctx.const_0();
                    if simplified_left == zero {
                        return simplified_right;
                    }
                    if simplified_right == zero {
                        return simplified_left;
                    }
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op;
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op
                    } else {
                        simplify_or(simplified_left, simplified_right, ctx, swzb_ctx)
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(c) = arith.right.if_constant() {
                        let left = simplify_with_and_mask(arith.left, mask >> c, ctx, swzb_ctx);
                        if left == arith.left {
                            op
                        } else {
                            ctx.lsh(left, arith.right)
                        }
                    } else {
                        op
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(c) = arith.right.if_constant() {
                        // Using mask with all bits-to-be-shifted out as 1 for better
                        // add/sub/etc simplification.
                        // Maybe should do that and the actual mask?
                        let inner_mask = (mask << c) | (1u64 << c).wrapping_sub(1);
                        let left = simplify_with_and_mask(arith.left, inner_mask, ctx, swzb_ctx);
                        if left == arith.left {
                            op
                        } else {
                            ctx.rsh(left, arith.right)
                        }
                    } else {
                        op
                    }
                }
                ArithOpType::Xor | ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul => {
                    if arith.ty != ArithOpType::Xor {
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
                        let ok = mask.wrapping_add(1).count_ones() <= 1;
                        if !ok {
                            return op;
                        }
                    }
                    // Normalize (x + c000) & ffff to (x - 4000) & ffff and similar.
                    if let Some(c) = arith.right.if_constant() {
                        let c = c & mask;
                        let max = mask.wrapping_add(1);
                        debug_assert!(max != 0);
                        let limit = max >> 1;
                        if arith.ty == ArithOpType::Add {
                            if c > limit {
                                let new = ctx.sub_const(arith.left, max.wrapping_sub(c));
                                return simplify_with_and_mask(new, mask, ctx, swzb_ctx);
                            }
                        } else if arith.ty == ArithOpType::Sub {
                            if c >= limit {
                                let new = ctx.add_const(arith.left, max.wrapping_sub(c));
                                return simplify_with_and_mask(new, mask, ctx, swzb_ctx);
                            }
                        }
                    }
                    let simplified_left =
                        simplify_with_and_mask(arith.left, mask, ctx, swzb_ctx);
                    let simplified_right =
                        simplify_with_and_mask(arith.right, mask, ctx, swzb_ctx);
                    if should_stop_with_and_mask(swzb_ctx) {
                        return op;
                    }
                    if simplified_left == arith.left && simplified_right == arith.right {
                        op
                    } else {
                        let op = ctx.arithmetic(arith.ty, simplified_left, simplified_right);
                        // The result may simplify again, for example with mask 0x1
                        // Mem16[x] + Mem32[x] + Mem8[x] => 3 * Mem8[x] => 1 * Mem8[x]
                        simplify_with_and_mask(op, mask, ctx, swzb_ctx)
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
                ctx.and_const(val, mask)
            } else {
                op
            }
        }
        _ => op,
    }
}

/// Assumes: `mem` is part of `op`.
fn simplify_with_and_mask_mem<'e>(
    op: Operand<'e>,
    mem: &MemAccess<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    let mask = mem.size.mask() & mask;
    // Try to do conversions such as Mem32[x] & 00ff_ff00 => Mem16[x + 1] << 8,
    // but also Mem32[x] & 003f_5900 => (Mem16[x + 1] & 3f59) << 8.

    // Round down to 8 -> convert to bytes
    let mask_low = mask.trailing_zeros() / 8;
    // Round up to 8 -> convert to bytes
    let mask_high = (64 - mask.leading_zeros() + 7) / 8;
    if mask_high <= mask_low {
        return op;
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
        return op;
    }
    let (address, offset) = mem.address();
    let mem = ctx.mem_any(new_size, address, offset.wrapping_add(mask_low as u64));
    let shifted = if mask_low == 0 {
        mem
    } else {
        ctx.lsh_const(mem, mask_low as u64 * 8)
    };
    shifted
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
        if swzb.simplify_count > 40 {
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
            true
        } else {
            false
        }
    }

    if recurse_check {
        if swzb.xor_recurse > 4 {
            swzb.simplify_count = u8::max_value();
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

/// Simplifies `op` when the bits in the range `bits` are guaranteed to be one.
/// Returning `None` means that `op | constval(bits) == constval(bits)`
fn simplify_with_one_bits<'e>(
    op: Operand<'e>,
    bits: &Range<u8>,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    fn check_useless_and_mask<'e>(
        left: Operand<'e>,
        right: Operand<'e>,
        bits: &Range<u8>,
    ) -> Option<Operand<'e>> {
        // one_bits | (other & c) can be transformed to other & (c | one_bits)
        // if c | one_bits is all ones for other's relevant bits, const mask
        // can be removed.
        let const_other = Operand::either(left, right, |x| x.if_constant());
        if let Some((c, other)) = const_other {
            let low = bits.start;
            let high = 64 - bits.end;
            let mask = !0u64 >> low << low << high >> high;
            let nop_mask = other.relevant_bits_mask();
            if c | mask == nop_mask {
                return Some(other);
            }
        }
        None
    }

    if bits.start >= bits.end {
        return Some(op);
    }
    let default = || {
        let relevant_bits = op.relevant_bits();
        match relevant_bits.start >= bits.start && relevant_bits.end <= bits.end {
            true => None,
            false => Some(op),
        }
    };
    match *op.ty() {
        OperandType::Arithmetic(ref arith) => {
            let left = arith.left;
            let right = arith.right;
            match arith.ty {
                ArithOpType::And => {
                    if let Some(other) = check_useless_and_mask(left, right, bits) {
                        return simplify_with_one_bits(other, bits, ctx);
                    }
                    let left = simplify_with_one_bits(left, bits, ctx);
                    let right = simplify_with_one_bits(right, bits, ctx);
                    match (left, right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => {
                            let low = bits.start;
                            let high = 64 - bits.end;
                            let mask = !0u64 >> low << low << high >> high;
                            if mask == s.relevant_bits_mask() {
                                Some(s)
                            } else {
                                Some(ctx.and_const(s, mask))
                            }
                        }
                        (Some(l), Some(r)) => {
                            if l != arith.left || r != arith.right {
                                if let Some(other) = check_useless_and_mask(l, r, bits) {
                                    return simplify_with_one_bits(other, bits, ctx);
                                }
                                let new = ctx.and(l, r);
                                if new == op {
                                    Some(new)
                                } else {
                                    simplify_with_one_bits(new, bits, ctx)
                                }
                            } else {
                                Some(op)
                            }
                        }
                    }
                }
                ArithOpType::Or => {
                    let left = simplify_with_one_bits(left, bits, ctx);
                    let right = simplify_with_one_bits(right, bits, ctx);
                    match (left, right) {
                        (None, None) => None,
                        (None, Some(s)) | (Some(s), None) => Some(s),
                        (Some(l), Some(r)) => {
                            if l != arith.left || r != arith.right {
                                let new = ctx.or(l, r);
                                if new == op {
                                    Some(new)
                                } else {
                                    simplify_with_one_bits(new, bits, ctx)
                                }
                            } else {
                                Some(op)
                            }
                        }
                    }
                }
                _ => default(),
            }
        }
        OperandType::Constant(c) => {
            let low = bits.start;
            let high = 64 - bits.end;
            let mask = !0u64 >> low << low << high >> high;
            let new_val = c | mask;
            match new_val & !mask {
                0 => None,
                c => Some(ctx.constant(c)),
            }
        }
        OperandType::Memory(ref mem) => {
            let max_bits = op.relevant_bits();
            if bits.start == 0 && bits.end >= max_bits.end {
                None
            } else if bits.end >= max_bits.end {
                let (address, offset) = mem.address();
                if bits.start <= 8 && max_bits.end > 8 {
                    Some(ctx.mem8(address, offset))
                } else if bits.start <= 16 && max_bits.end > 16 {
                    Some(ctx.mem16(address, offset))
                } else if bits.start <= 32 && max_bits.end > 32 {
                    Some(ctx.mem32(address, offset))
                } else {
                    Some(op)
                }
            } else {
                Some(op)
            }
        }
        _ => default(),
    }
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
    ctx.simplify_temp_stack().alloc(|ops| {
        collect_xor_ops(left, ops, 30)
            .and_then(|()| collect_xor_ops(right, ops, 30))
            .and_then(|()| simplify_xor_ops(ops, ctx, swzb))
            .unwrap_or_else(|_| {
                // This is likely some hash function being unrolled, give up
                // Also set swzb to stop everything
                swzb.simplify_count = u8::max_value();
                swzb.with_and_mask_count = u8::max_value();
                let arith = ArithOperand {
                    ty: ArithOpType::Xor,
                    left,
                    right,
                };
                return ctx.intern(OperandType::Arithmetic(arith));
            })
    })
}

fn simplify_xor_try_extract_constant<'e>(
    op: Operand<'e>,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
) -> Option<(Operand<'e>, u64)> {
    fn recurse<'e>(op: Operand<'e>, ctx: OperandCtx<'e>) -> Option<(Operand<'e>, u64)> {
        match op.ty() {
            OperandType::Arithmetic(arith) => {
                match arith.ty {
                    ArithOpType::And => {
                        let left = recurse(arith.left, ctx);
                        let right = recurse(arith.right, ctx);
                        return match (left, right) {
                            (None, None) => None,
                            (Some(a), None) => {
                                Some((ctx.and(a.0, arith.right), a.1))
                            }
                            (None, Some(a)) => {
                                Some((ctx.and(a.0, arith.left), a.1))
                            }
                            (Some(a), Some(b)) => {
                                Some((ctx.and(a.0, b.0), a.1 ^ b.1))
                            }
                        };
                    }
                    ArithOpType::Xor => {
                        if let Some(c) = arith.right.if_constant() {
                            return Some((arith.left, c));
                        }
                    }
                    _ => (),
                }
            }
            _ => (),
        }
        None
    }

    let (l, r) = op.if_arithmetic_and()?;
    let and_mask = r.if_constant()?;
    let (new, c) = recurse(l, ctx)?;
    let new = simplify_and(new, r, ctx, swzb);
    Some((new, c & and_mask))
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
        return simplify_gt(new, right, ctx, swzb_ctx);
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
                    return simplify_gt(new, right, ctx, swzb_ctx);
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

    match (left.if_constant(), right.if_constant()) {
        (Some(a), Some(b)) => match a > b {
            true => return ctx.const_1(),
            false => return ctx.const_0(),
        },
        (Some(c), None) => {
            if c == 0 {
                return ctx.const_0();
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
            // max > x if x != max
            let relbit_mask = right.relevant_bits_mask();
            if c == relbit_mask {
                return ctx.neq(left, right);
            }
        }
        (None, Some(c)) => {
            // x > 0 if x != 0
            if c == 0 {
                return ctx.neq(left, right);
            }
            if let Some((inner, from, to)) = left.if_sign_extend() {
                return simplify_gt_sext_const(ctx, c, inner, from, to, false);
            }
            let relbit_mask = left.relevant_bits_mask();
            if c == relbit_mask {
                return ctx.const_0();
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
            u64::max_value(),
        ),
        (0xffff_ffff_ffff_ff01, 0xff),
    );
    assert_eq!(
        sum_valid_range(&[(ctx.register(4), false)], u64::max_value()),
        (0, u64::max_value()),
    );
    assert_eq!(
        sum_valid_range(&[(ctx.register(4), true)], u64::max_value()),
        (0, u64::max_value()),
    );
}
