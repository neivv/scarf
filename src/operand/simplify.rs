use std::cmp::{min, max};
use std::ops::Range;

use crate::bit_misc::{bits_overlap, one_bit_ranges, zero_bit_ranges};
use crate::heapsort;

use super::{
    ArithOperand, ArithOpType, MemAccessSize, Operand, OperandType, OperandCtx,
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
            if left.if_constant() == Some(0) {
                return ctx.const_0();
            }
            if left == right {
                // x % x == 0, x / x = 1
                if ty == ArithOpType::Modulo {
                    return ctx.const_0();
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
                let float = f32::to_bits(c as i32 as f32);
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
    // Shouldn't be 64bit constant since then `from` would already be Mem64
    // Obviously such thing could be built, but assuming disasm/users don't..
    if let Some(val) = val.if_constant() {
        let (ext, mask) = match from {
            MemAccessSize::Mem8 => (val & 0x80 != 0, 0xff),
            MemAccessSize::Mem16 => (val & 0x8000 != 0, 0xffff),
            MemAccessSize::Mem32 | _ => (val & 0x8000_0000 != 0, 0xffff_ffff),
        };
        let val = val & mask;
        if ext {
            let new = match to {
                MemAccessSize::Mem16 => (0xffff & !mask) | val as u64,
                MemAccessSize::Mem32 => (0xffff_ffff & !mask) | val as u64,
                MemAccessSize::Mem64 | _ => {
                    (0xffff_ffff_ffff_ffff & !mask) | val as u64
                }
            };
            ctx.constant(new)
        } else {
            ctx.constant(val)
        }
    } else {
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
                let right_const = collect_add_ops(right, right_ops, !0u64, false).ok()?;
                ctx.simplify_temp_stack().alloc(|left_ops| {
                    let left_const = collect_add_ops(left, left_ops, !0u64, false).ok()?;
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
                            left_abs < constant_abs
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

fn simplify_xor_ops<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Result<Operand<'e>, SizeLimitReached> {
    let mut const_val = 0;
    loop {
        const_val = ops.iter().flat_map(|x| x.if_constant())
            .fold(const_val, |sum, x| sum ^ x);
        ops.retain(|x| x.if_constant().is_none());
        heapsort::sort(ops);
        simplify_xor_remove_reverting(ops);
        simplify_or_merge_mem(ops, ctx); // Yes, this is supposed to stay valid for xors.
        simplify_or_merge_child_ands(ops, ctx, swzb_ctx, true);
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
                collect_xor_ops(r, ops, usize::max_value())?;
            }
            i += 1;
        }
        if !ops_changed {
            heapsort::sort(ops);
            break;
        }
    }

    match ops.len() {
        0 => return Ok(ctx.constant(const_val)),
        1 if const_val == 0 => return Ok(ops[0]),
        _ => (),
    };
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
    let default = || {
        let arith = ArithOperand {
            ty: ArithOpType::Lsh,
            left: left.clone(),
            right: right.clone(),
        };
        ctx.intern(OperandType::Arithmetic(arith))
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left.clone();
    } else if constant >= 64 - u64::from(left.relevant_bits().start) {
        return ctx.const_0();
    }
    let zero_bits = (64 - constant as u8)..64;
    match simplify_with_zero_bits(left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => {
            if s != left {
                return simplify_lsh(s, right, ctx, swzb_ctx);
            }
        }
    }
    match *left.ty() {
        OperandType::Constant(a) => ctx.constant(a << constant as u8),
        OperandType::Arithmetic(ref arith) => {
            match arith.ty {
                ArithOpType::And => {
                    // Simplify (x & mask) << c to (x << c) & (mask << c)
                    if let Some(c) = arith.right.if_constant() {
                        let high = 64 - zero_bits.start;
                        let low = left.relevant_bits().start;
                        let no_op_mask = !0u64 >> low << low << high >> high;

                        let new = simplify_lsh(arith.left, right, ctx, swzb_ctx);
                        if c == no_op_mask {
                            return new;
                        } else {
                            return simplify_and_const(new, c << constant, ctx, swzb_ctx);
                        }
                    }
                    let arith = ArithOperand {
                        ty: ArithOpType::Lsh,
                        left: left,
                        right: right,
                    };
                    ctx.intern(OperandType::Arithmetic(arith))
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
                                    *op = simplify_lsh(*op, right, ctx, swzb_ctx);
                                }
                                simplify_xor_ops(slice, ctx, swzb_ctx).ok()
                            }
                        })
                        .unwrap_or_else(|| default())
                }
                ArithOpType::Mul => {
                    if constant < 0x10 {
                        // Prefer (x * y * 4) over ((x * y) << 2),
                        // especially since usually there's already a constant there.
                        let multiply_constant = 1 << constant;
                        simplify_mul(left, ctx.constant(multiply_constant), ctx)
                    } else {
                        default()
                    }
                }
                ArithOpType::Lsh => {
                    if let Some(inner_const) = arith.right.if_constant() {
                        let sum = inner_const.saturating_add(constant);
                        if sum < 64 {
                            simplify_lsh(arith.left, ctx.constant(sum), ctx, swzb_ctx)
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
                                simplify_lsh(
                                    arith.left,
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
            left: left.clone(),
            right: right.clone(),
        };
        ctx.intern(OperandType::Arithmetic(arith))
    };
    let constant = match right.if_constant() {
        Some(s) => s,
        None => return default(),
    };
    if constant == 0 {
        return left.clone();
    } else if constant >= left.relevant_bits().end.into() {
        return ctx.const_0();
    }
    let zero_bits = 0..(constant as u8);
    match simplify_with_zero_bits(left, &zero_bits, ctx, swzb_ctx) {
        None => return ctx.const_0(),
        Some(s) => {
            if s != left {
                return simplify_rsh(s, right, ctx, swzb_ctx);
            }
        }
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
                ArithOpType::Lsh => {
                    if let Some(lsh_const) = arith.right.if_constant() {
                        if lsh_const >= 64 {
                            return ctx.const_0();
                        }
                        let diff = constant as i8 - lsh_const as i8;
                        let mask = (!0u64 << lsh_const) >> constant;
                        let val = match diff {
                            0 => arith.left,
                            // (x << rsh) >> lsh, rsh > lsh
                            x if x > 0 => {
                                simplify_rsh(
                                    arith.left,
                                    ctx.constant(x as u64),
                                    ctx,
                                    swzb_ctx,
                                )
                            }
                            // (x << rsh) >> lsh, lsh > rsh
                            x => {
                                simplify_lsh(
                                    arith.left,
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
                    } else {
                        default()
                    }
                }
                ArithOpType::Rsh => {
                    if let Some(inner_const) = arith.right.if_constant() {
                        let sum = inner_const.saturating_add(constant);
                        if sum < 64 {
                            simplify_rsh(arith.left, ctx.constant(sum), ctx, swzb_ctx)
                        } else {
                            ctx.const_0()
                        }
                    } else {
                        default()
                    }
                }
                _ => default(),
            }
        },
        OperandType::Memory(ref mem) => {
            match mem.size {
                MemAccessSize::Mem64 => {
                    if constant >= 56 {
                        let addr = ctx.add_const(mem.address, 7);
                        let c = ctx.constant(constant - 56);
                        let new = ctx.mem8(addr);
                        return simplify_rsh(new, c, ctx, swzb_ctx);
                    } else if constant >= 48 {
                        let addr = ctx.add_const(mem.address, 6);
                        let c = ctx.constant(constant - 48);
                        let new = ctx.mem16(addr);
                        return simplify_rsh(new, c, ctx, swzb_ctx);
                    } else if constant >= 32 {
                        let addr = ctx.add_const(mem.address, 4);
                        let c = ctx.constant(constant - 32);
                        let new = ctx.mem32(addr);
                        return simplify_rsh(new, c, ctx, swzb_ctx);
                    }
                }
                MemAccessSize::Mem32 => {
                    if constant >= 24 {
                        let addr = ctx.add_const(mem.address, 3);
                        let c = ctx.constant(constant - 24);
                        let new = ctx.mem8(addr);
                        return simplify_rsh(new, c, ctx, swzb_ctx);
                    } else if constant >= 16 {
                        let addr = ctx.add_const(mem.address, 2);
                        let c = ctx.constant(constant - 16);
                        let new = ctx.mem16(addr);
                        return simplify_rsh(new, c, ctx, swzb_ctx);
                    }
                }
                MemAccessSize::Mem16 => {
                    if constant >= 8 {
                        let addr = ctx.add_const(mem.address, 1);
                        let c = ctx.constant(constant - 8);
                        let new = ctx.mem8(addr);
                        return simplify_rsh(new, c, ctx, swzb_ctx);
                    }
                }
                _ => (),
            }
            default()
        }
        _ => default(),
    }
}

fn simplify_add_merge_masked_reverting<'e>(ops: &mut AddSlice<'e>) -> u64 {
    // Shouldn't need as complex and_const as other places use
    fn and_const<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>)> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
                arith.right.if_constant().map(|c| (c, arith.left))
            }
            _ => None,
        }
    }

    fn check_vals<'e>(a: Operand<'e>, b: Operand<'e>) -> bool {
        if let Some((l, r)) = a.if_arithmetic_sub() {
            if l.if_constant() == Some(0) && r == b {
                return true;
            }
        }
        if let Some((l, r)) = b.if_arithmetic_sub() {
            if l.if_constant() == Some(0) && r == a {
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
                            check_vals(val, other_val) &&
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
fn relevant_bits_for_eq<'e>(ops: &[(Operand<'e>, bool)]) -> (u64, u64) {
    let mut sizes = ops.iter().map(|x| (x.1, x.0.relevant_bits())).collect::<Vec<_>>();
    heapsort::sort_by(&mut sizes, |(a_neg, a_bits), (b_neg, b_bits)| {
        (a_neg, a_bits.end) < (b_neg, b_bits.end)
    });
    let mut iter = sizes.iter();
    let mut pos_bits = 64..0;
    let mut neg_bits = 64..0;
    while let Some(next) = iter.next() {
        let bits = next.1.clone();
        if next.0 == true {
            neg_bits = bits;
            while let Some(next) = iter.next() {
                let bits = next.1.clone();
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
            pos_bits.end = max(pos_bits.end.wrapping_add(1), bits.end.wrapping_add(1)).min(64);
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
    (pos_mask, neg_mask)
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
    heapsort::sort_by(ops, |a, b| Operand::and_masked(a.0) < Operand::and_masked(b.0));
    if ops[ops.len() - 1].1 == true {
        for op in ops.iter_mut() {
            op.1 = !op.1;
        }
    }
    match ops.len() {
        0 => ctx.const_1(),
        1 => match *ops[0].0.ty() {
            OperandType::Constant(0) => ctx.const_1(),
            OperandType::Constant(_) => ctx.const_0(),
            _ => {
                if let Some((left, right)) = ops[0].0.if_arithmetic_eq() {
                    // Check for (x == 0) == 0
                    let either_const = Operand::either(left, right, |x| x.if_constant());
                    if let Some((0, other)) = either_const {
                        if other.relevant_bits().end == 1 {
                            return other.clone();
                        }
                    }
                }
                let swzb_ctx = &mut SimplifyWithZeroBits::default();
                // Simplify (c > x) == 0 to x > (c - 1)
                // Wouldn't be valid for (0 > x) but it should never be created.
                // Same for (x > c) == 0 to c + 1 > x
                if let Some((l, r)) = ops[0].0.if_arithmetic_gt() {
                    if let Some(c) = l.if_constant() {
                        return simplify_gt(
                            r,
                            ctx.constant(c.wrapping_sub(1)),
                            ctx,
                            swzb_ctx,
                        );
                    }
                    if let Some(c) = r.if_constant() {
                        return simplify_gt(
                            ctx.constant(c.wrapping_add(1)),
                            l,
                            ctx,
                            swzb_ctx,
                        );
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
                    right: ctx.const_0(),
                };
                ctx.intern(OperandType::Arithmetic(arith))
            }
        },
        2 => {
            let first_const = ops[0].0.if_constant().is_some();

            let (left, right) = match first_const {
                // ops[1] isn't const, so make it to not need sub(0, x)
                true => match ops[1].1 {
                    false => (&ops[0], &ops[1]),
                    true => (&ops[1], &ops[0]),
                },
                // Otherwise just make ops[0] not need sub
                _ => match ops[0].1 {
                    false => (&ops[1], &ops[0]),
                    true => (&ops[0], &ops[1]),
                },
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
            };
            let left = make_op(ctx, left.0, mask, !left.1);
            let right = make_op(ctx, right.0, mask, right.1);
            simplify_eq_2_ops(left, right, ctx)
        },
        _ => {
            let (left_rel_bits, right_rel_bits) = relevant_bits_for_eq(&ops);
            // Construct a + b + c == d + e + f
            // where left side has all non-negated terms,
            // and right side has all negated terms (Negation forgotten as they're on the right)
            let mut left_tree = match ops.iter().position(|x| x.1 == false) {
                Some(i) => ops.swap_remove(i).0,
                None => ctx.const_0(),
            };
            let mut right_tree = match ops.iter().position(|x| x.1 == true) {
                Some(i) => ops.swap_remove(i).0,
                None => ctx.const_0(),
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
    };

    let (left, right) = match left < right {
        true => (left, right),
        false => (right, left),
    };

    if let Some((c, other)) = Operand::either(left, right, |x| x.if_constant()) {
        if c == 1 {
            // Simplify x == 1 to x if x is just the lowest bit
            if other.relevant_bits().end == 1 {
                return other;
            }
        }
    }
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
    let ((cmp_l, cmp_r, sign_bit, size), other) = Operand::either(left, right, |op| {
        // Sign: (((x - y) & sign_bit) == 0) == 0
        let (sub, sign) = op.if_arithmetic_eq()
            .filter(|(_, r)| r.if_constant() == Some(0))
            .and_then(|(l, _)| l.if_arithmetic_eq())
            .filter(|(_, r)| r.if_constant() == Some(0))
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
                    if r.if_constant() == Some(0) {
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
    let (r, rc) = r.if_arithmetic_add()?;
    let rc = rc.if_constant()?;
    let offset = rc.wrapping_sub(sign_bit) & mask;
    let l_ok = match l.if_constant() {
        Some(a) => match cmp_r.if_constant() {
            Some(b) => a.wrapping_sub(offset) == b,
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

// Tries to merge (a & a_mask) | (b & b_mask) to (a_mask | b_mask) & result
fn try_merge_ands<'e>(
    a: Operand<'e>,
    b: Operand<'e>,
    a_mask: u64,
    b_mask: u64,
    ctx: OperandCtx<'e>,
    swzb: &mut SimplifyWithZeroBits,
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
                let result = try_merge_memory(val, other_shift, shift, ctx);
                if let Some(merged) = result {
                    return Some(merged);
                }
            }
        }
    }
    match (a.ty(), b.ty()) {
        (&OperandType::Arithmetic(ref c), &OperandType::Arithmetic(ref d)) => {
            if c.ty == ArithOpType::Xor && d.ty == ArithOpType::Xor {
                try_merge_ands(c.left, d.left, a_mask, b_mask, ctx, swzb).and_then(|left| {
                    try_merge_ands(c.right, d.right, a_mask, b_mask, ctx, swzb).map(|right| (left, right))
                }).or_else(|| try_merge_ands(c.left, d.right, a_mask, b_mask, ctx, swzb).and_then(|first| {
                    try_merge_ands(c.right, d.left, a_mask, b_mask, ctx, swzb).map(|second| (first, second))
                })).map(|(first, second)| {
                    simplify_xor(first, second, ctx, swzb)
                })
            } else {
                None
            }
        }
        (&OperandType::Memory(ref a_mem), &OperandType::Memory(ref b_mem)) => {
            // Can treat Mem16[x], Mem8[x] as Mem16[x], Mem16[x]
            if a_mem.address == b_mem.address {
                fn check_mask<'e>(op: Operand<'e>, mask: u64, ok: Operand<'e>) -> Option<Operand<'e>> {
                    if op.relevant_bits().end >= 64 - mask.leading_zeros() as u8 {
                        Some(ok)
                    } else {
                        None
                    }
                };
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

fn quick_and_simplify<'e>(
    left: Operand<'e>,
    right: u64,
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    let left_relbit_mask = left.relevant_bits_mask();
    if right & left_relbit_mask == left_relbit_mask {
        return Some(left);
    }
    if can_quick_simplify_type(left.ty()) {
        let left_bits = left.relevant_bits();
        let right = if left_bits.end != 64 {
            let mask = (1 << left_bits.end) - 1;
            if right & mask == mask {
                return Some(left);
            }
            right & mask
        } else {
            right
        };
        let right = ctx.constant(right);
        let arith = ArithOperand {
            ty: ArithOpType::And,
            left,
            right,
        };
        return Some(ctx.intern(OperandType::Arithmetic(arith)));
    }
    None
}

/// Having a bitwise and with a constant is really common.
pub fn simplify_and_const<'e>(
    mut left: Operand<'e>,
    mut right: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    // Check if left is x & const
    if let Some((l, r)) = left.if_arithmetic_and() {
        if let Some(c) = r.if_constant() {
            right &= c;
            left = l;
        }
    }
    let left_mask = match left.if_constant() {
        Some(c) => return ctx.constant(c & right),
        None => left.relevant_bits_mask(),
    };
    if left_mask & right == 0 {
        return ctx.const_0();
    }
    if let Some(result) = quick_and_simplify(left, right, ctx) {
        return result;
    }
    ctx.simplify_temp_stack().alloc(|slice| {
        collect_and_ops(left, slice, 30)
            .and_then(|()| simplify_and_main(slice, right, ctx, swzb_ctx))
            .unwrap_or_else(|_| {
                let arith = ArithOperand {
                    ty: ArithOpType::And,
                    left,
                    right: ctx.constant(right),
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
    let const_other = Operand::either(left, right, |x| x.if_constant());
    if let Some((c, other)) = const_other {
        return simplify_and_const(other, c, ctx, swzb_ctx);
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

        heapsort::sort(ops);
        ops.dedup();
        simplify_and_remove_unnecessary_ors(ops, const_remain);

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
        for bits in zero_bit_ranges(const_remain) {
            slice_filter_map(ops, |op| {
                simplify_with_zero_bits(op, &bits, ctx, swzb_ctx)
                    .and_then(|x| match x.if_constant() {
                        Some(0) => None,
                        _ => Some(x),
                    })
            });
            // Unlike the other is_empty check above this returns 0, since if zero bit filter
            // removes all remaining ops, the result is 0 even with const_remain != 0
            // (simplify_with_zero_bits is defined to return None instead of Some(const(0)),
            // and obviously constant & 0 == 0)
            if ops.is_empty() {
                return Ok(ctx.const_0());
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
        let neq_compare_count = ops.iter().filter(|&&x| is_neq_compare(x)).count();
        if neq_compare_count >= 2 {
            let not: Result<_, SizeLimitReached> = ctx.simplify_temp_stack().alloc(|slice| {
                for &op in ops.iter() {
                    if is_neq_compare(op) {
                        if let Some((l, _)) = op.if_arithmetic_eq() {
                            slice.push(l)?;
                        }
                    }
                }
                let or = simplify_or_ops(slice, ctx, swzb_ctx)?;
                Ok(simplify_eq(or, ctx.const_0(), ctx))
            });
            if let Ok(not) = not {
                ops.retain(|x| !is_neq_compare(x));
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

fn is_neq_compare(op: Operand<'_>) -> bool {
    match op.if_arithmetic_eq() {
        Some((l, r)) => match l.ty() {
            OperandType::Arithmetic(a) => a.is_compare_op() && r.if_constant() == Some(0),
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
    #[derive(Copy, Clone)]
    enum Variant {
        GtLeft,
        GtRight,
        Neq,
    }

    fn check<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>, Variant)> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::GreaterThan => {
                arith.left.if_constant().map(|c| (c, arith.right, Variant::GtLeft))
                    .or_else(|| {
                        arith.right.if_constant().map(|c| (c, arith.left, Variant::GtRight))
                    })
            }
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::Equal => {
                Operand::either(arith.left, arith.right, |x| x.if_constant().filter(|&c| c == 0))
                    .and_then(|(_, other)| other.if_arithmetic_eq())
                    .and_then(|(l, r)| Operand::either(l, r, |x| x.if_constant()))
                    .map(|(c, other)| (c, other, Variant::Neq))
            }
            _ => None,
        }
    }

    if !ops.iter().any(|x| x.if_arithmetic_gt().is_some()) {
        return;
    }

    let mut i = 0;
    while i < ops.len() {
        let op = ops[i];
        let mut new = None;
        let mut changed = false;
        if let Some((mut constant, val, mut variant)) = check(op) {
            for j in ((i + 1)..ops.len()).rev() {
                let second = ops[j];
                if let Some((other_constant, other_val, other_variant)) = check(second) {
                    if val == other_val {
                        match (variant, other_variant) {
                            // (c > x) & (c2 > x) allows removing the one with greater c
                            (Variant::GtLeft, Variant::GtLeft) => {
                                if other_constant >= constant {
                                    ops.swap_remove(j);
                                } else {
                                    ops.swap_remove(j);
                                    constant = other_constant;
                                    changed = true;
                                }
                            }
                            // Same as above, but remove smaller c
                            (Variant::GtRight, Variant::GtRight) => {
                                if other_constant <= constant {
                                    ops.swap_remove(j);
                                } else {
                                    ops.swap_remove(j);
                                    constant = other_constant;
                                    changed = true;
                                }
                            }
                            // Could simplify this if it were x != 0 & x != 1,
                            // but not doing it right now
                            (Variant::Neq, Variant::Neq) => {
                            }
                            // (c > x) & (x > c2) is 0 if c >= c2
                            (Variant::GtLeft, Variant::GtRight) => {
                                if constant >= other_constant {
                                    ops.clear();
                                    return;
                                }
                            }
                            (Variant::GtRight, Variant::GtLeft) => {
                                if constant <= other_constant {
                                    ops.clear();
                                    return;
                                }
                            }
                            // (c > x) & (x != c2) is c > x if c2 >= c,
                            // or (c - 1) > x if c2 == c - 1
                            (Variant::GtLeft, Variant::Neq) => {
                                if other_constant >= constant {
                                    ops.swap_remove(j);
                                } else if other_constant == constant - 1 {
                                    ops.swap_remove(j);
                                    constant = constant - 1;
                                    changed = true;
                                }
                            }
                            // Same as above, just swapped c and c2
                            (Variant::Neq, Variant::GtLeft) => {
                                if constant >= other_constant {
                                    variant = Variant::GtLeft;
                                    constant = other_constant;
                                    ops.swap_remove(j);
                                    changed = true;
                                } else if constant == other_constant - 1 {
                                    variant = Variant::GtLeft;
                                    ops.swap_remove(j);
                                    changed = true;
                                }
                            }
                            (Variant::GtRight, Variant::Neq) => {
                                if other_constant <= constant {
                                    ops.swap_remove(j);
                                } else if other_constant == constant + 1 {
                                    ops.swap_remove(j);
                                    constant = constant + 1;
                                    changed = true;
                                }
                            }
                            (Variant::Neq, Variant::GtRight) => {
                                if constant <= other_constant {
                                    variant = Variant::GtRight;
                                    constant = other_constant;
                                    ops.swap_remove(j);
                                    changed = true;
                                } else if constant == other_constant + 1 {
                                    variant = Variant::GtRight;
                                    ops.swap_remove(j);
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
            if changed {
                new = Some(match variant {
                    Variant::GtLeft => ctx.gt_const_left(constant, val),
                    Variant::GtRight => ctx.gt_const(val, constant),
                    Variant::Neq => ctx.neq_const(val, constant),
                });
            }
        }
        if let Some(new) = new {
            ops[i] = new;
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

/// "Simplify bitwise or: merge child ands"
/// Converts things like [x & const1, x & const2] to [x & (const1 | const2)]
///
/// Also used by xors with only_nonoverlapping true
fn simplify_or_merge_child_ands<'e>(
    ops: &mut Slice<'e>,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
    only_nonoverlapping: bool,
) {
    fn and_const<'e>(op: Operand<'e>) -> Option<(u64, Operand<'e>)> {
        match op.ty() {
            OperandType::Arithmetic(arith) if arith.ty == ArithOpType::And => {
                arith.right.if_constant().map(|c| (c, arith.left))
            }
            OperandType::Memory(mem) => match mem.size {
                MemAccessSize::Mem8 => Some((0xff, op)),
                MemAccessSize::Mem16 => Some((0xffff, op)),
                MemAccessSize::Mem32 => Some((0xffff_ffff, op)),
                MemAccessSize::Mem64 => Some((0xffff_ffff_ffff_ffff, op)),
            }
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
        return;
    }

    let mut i = 0;
    while i < ops.len() {
        if let Some((mut constant, val)) = and_const(ops[i]) {
            let mut new_val = val;
            let mut j = i + 1;
            let mut changed = false;
            while j < ops.len() {
                if let Some((other_constant, other_val)) = and_const(ops[j]) {
                    let result = if only_nonoverlapping && other_constant & constant != 0 {
                        None
                    } else {
                        try_merge_ands(other_val, val, other_constant, constant, ctx, swzb_ctx)
                    };
                    if let Some(merged) = result {
                        constant |= other_constant;
                        new_val = merged;
                        ops.swap_remove(j);
                        changed = true;
                        continue; // Without incrementing j
                    }
                }
                j += 1;
            }
            if changed {
                ops[i] = simplify_and_const(new_val, constant, ctx, swzb_ctx);
            }
        }
        i += 1;
    }
}

// Simplify or: merge comparisions
// Converts
// (c > x) | (c == x) to (c + 1 > x),
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
                        let (c, other) = Operand::either(left, right, |x| x.if_constant())?;
                        return Some((c, other, MatchType::Equal));
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
                    if c == c2 && x == x2 {
                        match (ty, ty2) {
                            (MatchType::ConstantGreater, MatchType::Equal) |
                                (MatchType::Equal, MatchType::ConstantGreater) =>
                            {
                                // min/max edge cases can be handled by gt simplification,
                                // don't do them here.
                                if let Some(new_c) = c.checked_add(1) {
                                    ops[i] = ctx.gt_const_left(new_c, x);
                                    ops.swap_remove(j);
                                    continue 'outer;
                                }
                            }
                            (MatchType::ConstantLess, MatchType::Equal) |
                                (MatchType::Equal, MatchType::ConstantLess) =>
                            {
                                if let Some(new_c) = c.checked_sub(1) {
                                    ops[i] = ctx.gt_const(x, new_c);
                                    ops.swap_remove(j);
                                    continue 'outer;
                                }
                            }
                            _ => (),
                        }
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
    fn recurse<'e>(
        s: Operand<'e>,
        ops: &mut AddSlice<'e>,
        out_mask: u64,
        negate: bool,
    ) -> Result<u64, SizeLimitReached> {
        match s.ty() {
            OperandType::Arithmetic(arith) if {
                arith.ty == ArithOpType::Add || arith.ty== ArithOpType::Sub
            } => {
                let const1 = recurse(arith.left, ops, out_mask, negate)?;
                let negate_right = match arith.ty {
                    ArithOpType::Add => negate,
                    _ => !negate,
                };
                let const2 = recurse(arith.right, ops, out_mask, negate_right)?;
                Ok(const1.wrapping_add(const2))
            }
            _ => {
                if let Some((l, r)) = s.if_arithmetic_and() {
                    if let Some(c) = r.if_constant() {
                        if c & out_mask == out_mask {
                            return recurse(l, ops, out_mask, negate);
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
    recurse(s, ops, out_mask, negate)
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
    if ops.len() >= limit {
        if ops.len() == limit {
            ops.push(s)?;
            #[cfg(feature = "fuzz")]
            tls_simplification_incomplete();
        }
        return Err(SizeLimitReached);
    }
    match s.ty() {
        OperandType::Arithmetic(arith) if arith.ty == arith_type => {
            collect_arith_ops(arith.left, ops, arith_type, limit)?;
            collect_arith_ops(arith.right, ops, arith_type, limit)?;
        }
        _ => ops.push(s)?,
    }
    Ok(())
}

fn collect_mul_ops<'e>(s: Operand<'e>, ops: &mut Slice<'e>) -> Result<(), SizeLimitReached> {
    collect_arith_ops(s, ops, ArithOpType::Mul, usize::max_value())
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

/// Return (offset, len, value_offset)
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
            if arith.right.if_constant().is_some() {
                return is_offset_mem(arith.left, ctx)
                    .map(|(x, (off, _, val_off))| {
                        let len_bits = op.relevant_bits().end;
                        let len = len_bits.wrapping_add(7) / 8;
                        (x, (off, len as u32, val_off))
                    })
            }
            None
        }
        OperandType::Memory(ref mem) => {
            let len = match mem.size {
                MemAccessSize::Mem64 => 8,
                MemAccessSize::Mem32 => 4,
                MemAccessSize::Mem16 => 2,
                MemAccessSize::Mem8 => 1,
            };

            Some(
                Operand::const_offset(mem.address, ctx)
                    .map(|(val, off)| (val, (off, len, 0)))
                    .unwrap_or_else(|| (mem.address, (0, len, 0)))
            )
        }
        _ => None,
    }
}

/// Returns simplified operands.
fn try_merge_memory<'e>(
    val: Operand<'e>,
    shift: (u64, u32, u32),
    other_shift: (u64, u32, u32),
    ctx: OperandCtx<'e>,
) -> Option<Operand<'e>> {
    let (shift, other_shift) = match (shift.2, other_shift.2) {
        (0, 0) => return None,
        (0, _) => (shift, other_shift),
        (_, 0) => (other_shift, shift),
        _ => return None,
    };
    let (off1, len1, _) = shift;
    let (off2, len2, val_off2) = other_shift;
    if off1.wrapping_add(len1 as u64) != off2 || len1 != val_off2 {
        return None;
    }
    let addr = ctx.add_const(val, off1);
    let oper = match (len1 + len2).min(4) {
        1 => ctx.mem8(addr),
        2 => ctx.mem16(addr),
        3 => ctx.and_const(ctx.mem32(addr), 0x00ff_ffff),
        4 => ctx.mem32(addr),
        _ => return None,
    };
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

// Only x + y + c, x - y + c, and x * y + c have a chance of
// being simplified at all in simplify_collected_add_sub_ops
fn can_quick_simplify_add_sub(left: Operand<'_>) -> bool {
    match left.ty() {
        OperandType::Arithmetic(ref arith) => match arith.ty {
            ArithOpType::Add | ArithOpType::Sub | ArithOpType::Mul => false,
            _ => true,
        }
        _ => true,
    }
}

pub fn simplify_add_const<'e>(
    mut left: Operand<'e>,
    mut right: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if right == 0 {
        return left;
    }
    // Check if left is x +/- const
    // Useful since push/pop/esp/ebp offsets are really common.
    match left.ty() {
        OperandType::Arithmetic(ref arith) => {
            if let Some(c) = arith.right.if_constant() {
                if arith.ty == ArithOpType::Add {
                    // (x + c1) + c2 => x + (c2 + c1)
                    left = arith.left;
                    right = right.wrapping_add(c);
                } else if arith.ty == ArithOpType::Sub {
                    // (x - c1) + c2 => x + (c2 - c1)
                    left = arith.left;
                    right = right.wrapping_sub(c);
                }
            }
        }
        OperandType::Constant(c) => return ctx.constant(c.wrapping_add(right)),
        _ => (),
    }
    if right == 0 {
        return left;
    }
    if can_quick_simplify_add_sub(left) {
        let arith = if right > 0x8000_0000_0000_0000 {
            ArithOperand {
                ty: ArithOpType::Sub,
                left: left,
                right: ctx.constant(0u64.wrapping_sub(right)),
            }
        } else {
            ArithOperand {
                ty: ArithOpType::Add,
                left: left,
                right: ctx.constant(right),
            }
        };
        return ctx.intern(OperandType::Arithmetic(arith));
    }

    ctx.simplify_temp_stack()
        .alloc(|ops| {
            let const1 = collect_add_ops(left, ops, u64::max_value(), false)?;
            let const_sum = const1.wrapping_add(right);
            simplify_collected_add_sub_ops(ops, ctx, const_sum)?;
            Ok(add_sub_ops_to_tree(ops, ctx))
        }).unwrap_or_else(|SizeLimitReached| {
            let arith = ArithOperand {
                ty: ArithOpType::Add,
                left: left,
                right: ctx.constant(right),
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        })
}

pub fn simplify_sub_const<'e>(
    mut left: Operand<'e>,
    mut right: u64,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if right == 0 {
        return left;
    }
    // Check if left is x +/- const
    // Useful since push/pop/esp/ebp offsets are really common.
    match left.ty() {
        OperandType::Arithmetic(ref arith) => {
            if let Some(c) = arith.right.if_constant() {
                if arith.ty == ArithOpType::Add {
                    // (x + c1) - c2 => x - (c2 - c1)
                    left = arith.left;
                    right = right.wrapping_sub(c);
                } else if arith.ty == ArithOpType::Sub {
                    // (x - c1) - c2 => x - (c1 + c2)
                    left = arith.left;
                    right = c.wrapping_add(right);
                }
            }
        }
        OperandType::Constant(c) => return ctx.constant(c.wrapping_sub(right)),
        _ => (),
    }
    if right == 0 {
        return left;
    }
    if can_quick_simplify_add_sub(left) {
        let arith = if right > 0x8000_0000_0000_0000 {
            ArithOperand {
                ty: ArithOpType::Add,
                left: left,
                right: ctx.constant(0u64.wrapping_sub(right)),
            }
        } else {
            ArithOperand {
                ty: ArithOpType::Sub,
                left: left,
                right: ctx.constant(right),
            }
        };
        return ctx.intern(OperandType::Arithmetic(arith));
    }

    ctx.simplify_temp_stack()
        .alloc(|ops| {
            let const1 = collect_add_ops(left, ops, u64::max_value(), false)?;
            let const_sum = const1.wrapping_sub(right);
            simplify_collected_add_sub_ops(ops, ctx, const_sum)?;
            Ok(add_sub_ops_to_tree(ops, ctx))
        }).unwrap_or_else(|SizeLimitReached| {
            let arith = ArithOperand {
                ty: ArithOpType::Add,
                left: left,
                right: ctx.constant(right),
            };
            return ctx.intern(OperandType::Arithmetic(arith));
        })
}

pub fn simplify_add_sub<'e>(
    left: Operand<'e>,
    right: Operand<'e>,
    is_sub: bool,
    ctx: OperandCtx<'e>,
) -> Operand<'e> {
    if !is_sub {
        if let Some(c) = left.if_constant() {
            return simplify_add_const(right, c, ctx);
        }
    }
    if let Some(c) = right.if_constant() {
        if is_sub {
            return simplify_sub_const(left, c, ctx);
        } else {
            return simplify_add_const(left, c, ctx);
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
        collect_mul_ops(left, slice)
            .and_then(|()| collect_mul_ops(right, slice))
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
                    collect_mul_ops(new, ops)?;
                    changed = true;
                    break;
                }
                let new = simplify_mul_try_mul_constants(ops[i], const_product, ctx);
                if let Some(new) = new {
                    ops.swap_remove(i);
                    collect_mul_ops(new, ops)?;
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
    let new_consts = simplify_add_merge_masked_reverting(ops);
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
        simplify_or_merge_child_ands(ops, ctx, swzb_ctx, false);
        simplify_or_merge_xors(ops, ctx, swzb_ctx);
        simplify_or_merge_mem(ops, ctx);
        simplify_or_merge_comparisions(ops, ctx);

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
    if swzb_ctx.with_and_mask_count > 80 {
        #[cfg(feature = "fuzz")]
        tls_simplification_incomplete();
        true
    } else {
        false
    }
}

/// Convert and(x, mask) to x
fn simplify_with_and_mask<'e>(
    op: Operand<'e>,
    mask: u64,
    ctx: OperandCtx<'e>,
    swzb_ctx: &mut SimplifyWithZeroBits,
) -> Operand<'e> {
    if op.relevant_bits_mask() & mask == 0 {
        return ctx.const_0();
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
                    if simplified_left.if_constant() == Some(0) {
                        return simplified_right;
                    }
                    if simplified_right.if_constant() == Some(0) {
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
            let mask = match mem.size {
                MemAccessSize::Mem8 => mask & 0xff,
                MemAccessSize::Mem16 => mask & 0xffff,
                MemAccessSize::Mem32 => mask & 0xffff_ffff,
                MemAccessSize::Mem64 => mask,
            };
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
            let new_addr = if mask_low == 0 {
                mem.address.clone()
            } else {
                ctx.add_const(mem.address, mask_low as u64)
            };
            let mem = ctx.mem_variable_rc(new_size, new_addr);
            let shifted = if mask_low == 0 {
                mem
            } else {
                ctx.lsh_const(mem, mask_low as u64 * 8)
            };
            shifted
        }
        OperandType::Constant(c) => if c & mask != c {
            ctx.constant(c & mask)
        } else {
            op
        }
        OperandType::SignExtend(val, from, _) => {
            let from_mask = match from {
                MemAccessSize::Mem8 => 0xffu32,
                MemAccessSize::Mem16 => 0xffff,
                MemAccessSize::Mem32 => 0xffff_ffff,
                MemAccessSize::Mem64 => return val,
            };
            if from_mask as u64 & mask == mask {
                ctx.and_const(val, mask)
            } else {
                op
            }
        }
        _ => op,
    }
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
                if bits.start <= 8 && relevant_bits.end > 8 {
                    return Some(ctx.mem8(mem.address));
                } else if bits.start <= 16 && relevant_bits.end > 16 {
                    return Some(ctx.mem16(mem.address));
                } else if bits.start <= 32 && relevant_bits.end > 32 {
                    return Some(ctx.mem32(mem.address));
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
                if bits.start <= 8 && max_bits.end > 8 {
                    Some(ctx.mem8(mem.address))
                } else if bits.start <= 16 && max_bits.end > 16 {
                    Some(ctx.mem16(mem.address))
                } else if bits.start <= 32 && max_bits.end > 32 {
                    Some(ctx.mem32(mem.address))
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
    // x - y > x == y > x
    if let Some(new) = simplify_gt_lhs_sub(ctx, left, right) {
        left = new;
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
            if let Some(new) = simplify_gt_lhs_sub(ctx, left_inner, right_inner) {
                left = simplify_and_const(new, mask, ctx, swzb_ctx);
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
