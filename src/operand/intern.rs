use std::cell::RefCell;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::mem;

use arrayvec::ArrayVec;
use fxhash::FxHasher;
use hashbrown::hash_map::{HashMap, RawEntryMut};
use typed_arena::Arena;

use super::{ArithOperand, MemAccess, Operand, OperandHashByAddress, OperandType, OperandBase};

pub struct Interner<'e> {
    // Lookups are done by using raw entry api.
    // To be precise, hash is calculated from OperandType to be interned,
    // after which the hash is looked up, comparing input OperandType
    // with `InternHashOperand.operand`.
    interned_operands: RefCell<HashMap<InternHashOperand, (), BuildHasherDefault<DummyHasher>>>,
    // Using static lifetime as cannot refer to 'self.
    arena: Arena<OperandBase<'e>>,
}

pub struct ConstInterner<'e> {
    interned_operands: RefCell<HashMap<u64, Operand<'static>, BuildHasherDefault<ConstFxHasher>>>,
    arena: Arena<OperandBase<'e>>,
}

/// Interner for undefined, which can just do a by-index lookup.
pub struct UndefInterner {
    chunks: RefCell<Vec<Box<ArrayVec<OperandBase<'static>, UNDEF_CHUNK_SIZE>>>>,
}

const UNDEF_CHUNK_SIZE: usize = 1024;

impl<'e> Interner<'e> {
    pub fn new() -> Interner<'e> {
        Interner {
            interned_operands:
                RefCell::new(HashMap::with_capacity_and_hasher(0x4000, Default::default())),
            arena: Arena::new(),
        }
    }

    pub fn intern(&'e self, ty: OperandType<'e>) -> Operand<'e> {
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
                    let operand: Operand<'static> = mem::transmute(self.add_operand(ty));
                    e.insert(InternHashOperand {
                        hash,
                        operand,
                    }, ());
                    mem::transmute::<Operand<'static>, Operand<'e>>(operand)
                }
            }
        }
    }

    /// Adds an Operand to the arena that the caller intends to *never* pass to intern()
    /// (Or it would intern a separate copy of it)
    pub fn add_uninterned(&'e self, ty: OperandType<'e>) -> Operand<'e> {
        self.add_operand(ty)
    }

    pub fn interned_count(&self) -> usize {
        self.interned_operands.borrow().len()
    }

    fn add_operand(&'e self, ty: OperandType<'e>) -> Operand<'e> {
        let relevant_bits = ty.calculate_relevant_bits();
        let min_zero_bit_simplify_size =
            ty.min_zero_bit_simplify_size(relevant_bits.clone());
        let flags = ty.flags();
        let base = OperandBase {
            ty,
            min_zero_bit_simplify_size,
            relevant_bits,
            flags,
        };
        Operand(self.arena.alloc(base), PhantomData)
    }
}

impl<'e> ConstInterner<'e> {
    pub fn new() -> ConstInterner<'e> {
        ConstInterner {
            interned_operands:
                RefCell::new(HashMap::with_capacity_and_hasher(0x2000, Default::default())),
            arena: Arena::new(),
        }
    }

    pub fn intern(&'e self, value: u64) -> Operand<'e> {
        let mut map = self.interned_operands.borrow_mut();
        let entry = map.entry(value);
        let op = *entry.or_insert_with(|| {
            let op = self.add_operand(value);
            unsafe { mem::transmute::<Operand<'e>, Operand<'static>>(op) }
        });
        unsafe { mem::transmute::<Operand<'static>, Operand<'e>>(op) }
    }

    pub fn add_uninterned(&'e self, value: u64) -> Operand<'e> {
        self.add_operand(value)
    }

    pub fn interned_count(&self) -> usize {
        self.interned_operands.borrow().len()
    }

    fn add_operand(&'e self, value: u64) -> Operand<'e> {
        let relevant_bits = OperandType::const_relevant_bits(value);
        let min_zero_bit_simplify_size = 0;
        let flags = OperandType::const_flags();
        let base = OperandBase {
            ty: OperandType::Constant(value),
            min_zero_bit_simplify_size,
            relevant_bits,
            flags,
        };
        Operand(self.arena.alloc(base), PhantomData)
    }
}

fn operand_type_hash(ty: &OperandType<'_>) -> usize {
    let mut hasher = FxHasher::default();
    std::mem::discriminant(ty).hash(&mut hasher);
    match *ty {
        // Should be handled by const interner
        OperandType::Constant(_) => unreachable!(),
        // Note: copy_operand requires being able to copy undef, so it may get
        // interned by this -- even if it doesn't behave too well for comparisions.
        OperandType::Undefined(s) => s.0.hash(&mut hasher),
        OperandType::Custom(c) => c.hash(&mut hasher),
        OperandType::Flag(f) => (f as u8).hash(&mut hasher),
        OperandType::Fpu(f) => f.hash(&mut hasher),
        OperandType::Xmm(a, b) => {
            a.hash(&mut hasher);
            b.hash(&mut hasher);
        }
        OperandType::Register(r) => r.hash(&mut hasher),
        OperandType::Memory(ref mem) => {
            match *mem {
                MemAccess {
                    base,
                    offset,
                    size,
                    const_base: _,
                } => {
                    OperandHashByAddress(base).hash(&mut hasher);
                    offset.hash(&mut hasher);
                    size.hash(&mut hasher);
                }
            }
        }
        OperandType::Arithmetic(ref arith) => {
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
        OperandType::ArithmeticFloat(ref arith, size) => {
            match *arith {
                ArithOperand {
                    ty,
                    left,
                    right
                } => {
                    size.hash(&mut hasher);
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

/// Fxhash seems to be bit better for single integers when the result is xored with
/// itself shifted.
#[derive(Default)]
struct ConstFxHasher {
    hasher: FxHasher,
}

impl Hasher for ConstFxHasher {
    fn finish(&self) -> u64 {
        let val = self.hasher.finish();
        val ^ (val.rotate_left(mem::size_of::<usize>() as u32 * 4))
    }

    fn write(&mut self, data: &[u8]) {
        self.hasher.write(data);
    }

    fn write_u8(&mut self, value: u8) {
        self.hasher.write_u8(value);
    }

    fn write_u16(&mut self, value: u16) {
        self.hasher.write_u16(value);
    }

    fn write_u32(&mut self, value: u32) {
        self.hasher.write_u32(value);
    }

    fn write_u64(&mut self, value: u64) {
        self.hasher.write_u64(value);
    }

    fn write_usize(&mut self, value: usize) {
        self.hasher.write_usize(value);
    }
}

impl UndefInterner {
    pub(crate) fn new() -> UndefInterner{
        UndefInterner {
            chunks: RefCell::new(Vec::new()),
        }
    }

    pub(crate) fn push<'e>(&'e self, base: OperandBase<'e>) -> Operand<'e> {
        let base: OperandBase<'static> = unsafe { mem::transmute(base) };
        let mut chunks = self.chunks.borrow_mut();
        loop {
            if let Some(chunk) = chunks.last_mut() {
                if !chunk.is_full() {
                    chunk.push(base);
                    let base: &OperandBase<'static> = &chunk[chunk.len() - 1];
                    let base: &'e OperandBase<'e> = unsafe { mem::transmute(base) };
                    return Operand(base, PhantomData);
                }
            }

            chunks.push(Box::new(ArrayVec::new()));
        }
    }
}
