use super::{ArithOperand, ArithOpType, Operand, OperandType};

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
