use std::cell::RefCell;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

use hashbrown::hash_map::{HashMap, RawEntryMut};
use typed_arena::Arena;

use super::{ArithOperand, MemAccess, Operand, OperandHashByAddress, OperandType, OperandBase};

pub struct Interner {
    // Lookups are done by using raw entry api.
    // To be precise, hash is calculated from OperandType to be interned,
    // after which the hash is looked up, comparing input OperandType
    // with `InternHashOperand.operand`.
    interned_operands: RefCell<HashMap<InternHashOperand, (), BuildHasherDefault<DummyHasher>>>,
    // Using static lifetime as cannot refer to 'self.
    arena: Arena<OperandBase<'static>>,
}

impl Interner {
    pub fn new() -> Interner {
        Interner {
            interned_operands:
                RefCell::new(HashMap::with_capacity_and_hasher(160, Default::default())),
            arena: Arena::new(),
        }
    }

    pub fn intern<'e>(&self, ty: OperandType<'e>) -> Operand<'e> {
        let hash = operand_type_hash(&ty);
        let mut map = self.interned_operands.borrow_mut();
        let raw_entry = map.raw_entry_mut();
        let entry = raw_entry.from_hash(hash as u64, |x| unsafe {
            let op = x.transmute_operand_lifetime();
            x.hash == hash && *op.ty() == ty
        });
        unsafe {
            match entry {
                RawEntryMut::Occupied(e) => e.key().transmute_operand_lifetime(),
                RawEntryMut::Vacant(e) => {
                    let relevant_bits = ty.calculate_relevant_bits();
                    let min_zero_bit_simplify_size = ty.min_zero_bit_simplify_size();
                    let base = OperandBase {
                        ty: mem::transmute::<OperandType<'e>, OperandType<'static>>(ty),
                        min_zero_bit_simplify_size,
                        relevant_bits,
                    };
                    let operand: Operand<'static> =
                        Operand(mem::transmute(self.arena.alloc(base)), PhantomData);
                    e.insert(InternHashOperand {
                        hash,
                        operand,
                    }, ());
                    mem::transmute::<Operand<'static>, Operand<'e>>(operand)
                }
            }
        }
    }

    pub fn interned_count(&self) -> usize {
        self.interned_operands.borrow().len()
    }
}

fn operand_type_hash(ty: &OperandType<'_>) -> usize {
    let mut hasher = fxhash::FxHasher::default();
    std::mem::discriminant(ty).hash(&mut hasher);
    match *ty {
        OperandType::Constant(c) => c.hash(&mut hasher),
        OperandType::Undefined(u) => u.0.hash(&mut hasher),
        OperandType::Custom(c) => c.hash(&mut hasher),
        OperandType::Flag(f) => (f as u8).hash(&mut hasher),
        OperandType::Fpu(f) => f.hash(&mut hasher),
        OperandType::Xmm(a, b) => {
            a.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        OperandType::Register(r) => r.0.hash(&mut hasher),
        OperandType::Memory(ref mem) => {
            match *mem {
                MemAccess {
                    address,
                    size,
                } => {
                    OperandHashByAddress(address).hash(&mut hasher);
                    size.hash(&mut hasher);
                }
            }
        }
        OperandType::Arithmetic(ref arith) | OperandType::ArithmeticF32(ref arith) => {
            match *arith {
                ArithOperand {
                    ty,
                    left,
                    right
                } => {
                    ty.hash(&mut hasher);
                    OperandHashByAddress(left).hash(&mut hasher);
                    OperandHashByAddress(right).hash(&mut hasher);
                }
            }
        }
        OperandType::SignExtend(op, from, to) => {
            OperandHashByAddress(op).hash(&mut hasher);
            from.hash(&mut hasher);
            to.hash(&mut hasher);
        }
    }
    hasher.finish() as usize
}

// Precalculates its hash and uses it to keep rehashing cache friendly.
struct InternHashOperand {
    // Using usize hash and extending that to 64-bit can be problematic
    // if the hash table implementation assumes that all bits of the
    // hash are usable, but apparently at least hashbrown is already
    // discarding high half on 32-bit as fxhash uses usize-hashes anyway.
    hash: usize,
    operand: Operand<'static>,
}

impl InternHashOperand {
    unsafe fn transmute_operand_lifetime<'e>(&self) -> Operand<'e> {
        mem::transmute::<Operand<'static>, Operand<'e>>(self.operand)
    }
}

impl Hash for InternHashOperand {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.hash as u64).hash(state)
    }
}

/// Horrible hasher that relies on data being hashed already being well-spread data
/// (Like Operands with their cached hash)
#[derive(Default)]
struct DummyHasher {
    value: u64,
}

impl Hasher for DummyHasher {
    fn finish(&self) -> u64 {
        self.value
    }

    fn write(&mut self, data: &[u8]) {
        for &x in data.iter().take(8) {
            self.value = (self.value << 8) | u64::from(x);
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.value = value;
    }
}

