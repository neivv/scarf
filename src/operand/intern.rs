use std::cell::{Cell, RefCell};
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::ops::Range;

use copyless::BoxHelper;
use fxhash::FxHasher;
use hashbrown::hash_map::{HashMap, EntryRef};
use typed_arena::Arena;

use super::{ArithOperand, MemAccess, Operand, OperandHashByAddress, OperandType, OperandBase};
use crate::u64_hash::{ConstFxHasher};

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
    chunks: RefCell<Vec<Box<[MaybeUninit<OperandBase<'static>>; UNDEF_CHUNK_SIZE]>>>,
    current_used: Cell<usize>,
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

    pub fn intern(&'e self, ty: &OperandType<'e>) -> Operand<'e> {
        let hash = operand_type_hash(&ty);
        let mut map = self.interned_operands.borrow_mut();
        let key = &InternHashOperandRef {
            hash,
            ty: &ty,
            interner: self,
        };
        let entry = map.entry_ref(key);
        unsafe {
            match entry {
                EntryRef::Occupied(e) => e.key().transmute_operand_lifetime(),
                EntryRef::Vacant(e) => {
                    e.insert_entry(()).key().transmute_operand_lifetime()
                }
            }
        }
    }

    /// Adds an Operand to the arena that the caller intends to *never* pass to intern()
    /// (Or it would intern a separate copy of it)
    pub fn add_uninterned(&'e self, ty: &OperandType<'e>) -> Operand<'e> {
        self.add_operand(ty)
    }

    pub fn interned_count(&self) -> usize {
        self.interned_operands.borrow().len()
    }

    fn add_operand(&'e self, ty: &OperandType<'e>) -> Operand<'e> {
        let relevant_bits = ty.calculate_relevant_bits();
        debug_assert!(relevant_bits.start != relevant_bits.end, "Operand should be zero {:?}", ty);
        let relevant_bits_mask = relevant_bits_mask(relevant_bits.clone());
        let flags = ty.flags(relevant_bits.clone());
        let sort_order = ty.sort_order();
        let base = OperandBase {
            ty: *ty,
            type_alt_tag: ty.alt_tag(),
            relevant_bits,
            flags,
            sort_order,
            relevant_bits_mask,
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
        let relevant_bits_mask = relevant_bits_mask(relevant_bits.clone());
        let flags = OperandType::const_flags(value);
        let base = OperandBase {
            ty: OperandType::Constant(value),
            type_alt_tag: 0,
            relevant_bits,
            flags,
            sort_order: value,
            relevant_bits_mask,
        };
        Operand(self.arena.alloc(base), PhantomData)
    }
}

#[cfg(target_pointer_width = "64")]
#[inline]
fn relevant_bits_mask(bits: Range<u8>) -> u64 {
    relevant_bits_mask_64(bits)
}

#[cfg(target_pointer_width = "32")]
#[inline]
fn relevant_bits_mask(bits: Range<u8>) -> u64 {
    relevant_bits_mask_32(bits)
}

// u64 shifts on 32bit generate really iffy code,
// use lookup tables instead.
#[cfg(any(not(target_pointer_width = "64"), test))]
fn relevant_bits_mask_32(bits: Range<u8>) -> u64 {
    const fn gen_masks() -> [u64; 65] {
        let mut result = [0u64; 65];
        let mut i = 1;
        let mut state = 1;
        while i < 65 {
            // (0,) 1, 3, 7, f, ...
            result[i] = state;
            state = (state.wrapping_add(1) << 1).wrapping_sub(1);
            i += 1;
        }
        result
    }
    static MASKS: [u64; 65] = gen_masks();
    let start = bits.start;
    let end = bits.end;
    if end == 0 {
        0
    } else {
        let mask1 = MASKS[1 + ((end as usize - 1) & 0x3f)];
        let mask2 = MASKS[start as usize & 0x3f];
        mask1 & !mask2
    }
}

#[cfg(any(target_pointer_width = "64", test))]
fn relevant_bits_mask_64(bits: Range<u8>) -> u64 {
    let start = bits.start as u32;
    let end = bits.end as u32;
    if end >= 64 {
        !(1u64.wrapping_shl(start)
            .wrapping_sub(1))
    } else {
        1u64.wrapping_shl(end)
            .wrapping_sub(1)
            .wrapping_shr(start)
            .wrapping_shl(start)
    }
}

fn operand_type_hash(ty: &OperandType<'_>) -> usize {
    let mut hasher = FxHasher::default();
    std::mem::discriminant(ty).hash(&mut hasher);
    match *ty {
        // Should be handled by const interner
        OperandType::Constant(_) => unreachable!(),
        // Note: copy_operand requires being able to copy undef, so it may get
        // interned by this -- even if it doesn't behave too well for comparisons.
        OperandType::Undefined(s) => s.0.hash(&mut hasher),
        OperandType::Custom(c) => c.hash(&mut hasher),
        OperandType::Arch(r) => r.value().hash(&mut hasher),
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
        OperandType::Select(a, b, c) => {
            OperandHashByAddress(a).hash(&mut hasher);
            OperandHashByAddress(b).hash(&mut hasher);
            OperandHashByAddress(c).hash(&mut hasher);
        }
    }
    hasher.finish() as usize
}

// Precalculates its hash and uses it to keep rehashing cache friendly.
#[derive(Eq)]
struct InternHashOperand {
    // Using usize hash and extending that to 64-bit can be problematic
    // if the hash table implementation assumes that all bits of the
    // hash are usable, but apparently at least hashbrown is already
    // discarding high half on 32-bit as fxhash uses usize-hashes anyway.
    hash: usize,
    operand: Operand<'static>,
}

impl PartialEq for InternHashOperand {
    fn eq(&self, other: &Self) -> bool {
        // This function is only implemented since hashbrown required Eq to call entry_ref, pretty
        // sure if this is called there was some unexpected code path.
        debug_assert!(false, "Should not be called");
        self.operand == other.operand
    }
}

impl InternHashOperand {
    unsafe fn transmute_operand_lifetime<'e>(&self) -> Operand<'e> {
        mem::transmute::<Operand<'static>, Operand<'e>>(self.operand)
    }
}

struct InternHashOperandRef<'a, 'e> {
    hash: usize,
    ty: &'a OperandType<'e>,
    interner: &'e Interner<'e>,
}

impl<'a, 'b, 'e> From<&'b InternHashOperandRef<'a, 'e>> for InternHashOperand {
    fn from(value: &'b InternHashOperandRef<'a, 'e>) -> Self {
        let op = value.interner.add_operand(value.ty);
        let op = unsafe { mem::transmute::<Operand<'e>, Operand<'static>>(op) };
        Self {
            hash: value.hash,
            operand: op,
        }
    }
}

impl<'a, 'e> hashbrown::Equivalent<InternHashOperand> for InternHashOperandRef<'a, 'e> {
    fn equivalent(&self, key: &InternHashOperand) -> bool {
        unsafe {
            let op = key.transmute_operand_lifetime();
            key.hash == self.hash && *op.ty() == *self.ty
        }
    }
}

impl Hash for InternHashOperand {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (self.hash as u64).hash(state)
    }
}

impl<'a, 'e> Hash for InternHashOperandRef<'a, 'e> {
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

impl UndefInterner {
    pub(crate) fn new() -> UndefInterner{
        UndefInterner {
            chunks: RefCell::new(Vec::new()),
            current_used: Cell::new(0),
        }
    }

    pub(crate) fn push<'e>(&'e self, base: OperandBase<'e>) -> Operand<'e> {
        let base: OperandBase<'static> = unsafe { mem::transmute(base) };
        let mut chunks = self.chunks.borrow_mut();
        loop {
            if let Some(chunk) = chunks.last_mut() {
                let current_used = self.current_used.get();
                if current_used < UNDEF_CHUNK_SIZE {
                    self.current_used.set(current_used + 1);
                    let ptr = chunk[current_used as usize].as_mut_ptr();
                    unsafe {
                        ptr.write(base);
                        let base: &OperandBase<'static> = &*ptr;
                        let base: &'e OperandBase<'e> = mem::transmute(base);
                        return Operand(base, PhantomData);
                    }
                }
                self.current_used.set(0);
            }

            chunks.push(Box::alloc().init([const { MaybeUninit::uninit() }; UNDEF_CHUNK_SIZE]));
        }
    }

    pub(crate) fn get<'e>(&'e self, index: usize) -> Operand<'e> {
        let chunks = self.chunks.borrow_mut();
        let index1 = index / UNDEF_CHUNK_SIZE;
        let index2 = index % UNDEF_CHUNK_SIZE;
        let chunk: &[MaybeUninit<OperandBase<'_>>; UNDEF_CHUNK_SIZE] = &chunks[index1];
        let index2_ok = if index1 == chunks.len() - 1 {
            index2 < self.current_used.get()
        } else {
            index2 < UNDEF_CHUNK_SIZE
        };
        if !index2_ok {
            panic!("Unallocated UndefinedId");
        }
        unsafe {
            let ptr = chunk.get_unchecked(index2).as_ptr();
            let base: &OperandBase<'static> = &*ptr;
            let base: &'e OperandBase<'e> = mem::transmute(base);
            return Operand(base, PhantomData);
        }
    }
}

#[test]
fn test_relevant_bits_mask() {
    use crate::{OperandContext};

    fn check<'e>(op: Operand<'e>, expected: u64) {
        let bits = op.relevant_bits();
        assert_eq!(
            relevant_bits_mask_32(bits.clone()), expected,
            "32bit impl fail {op} {:x}", relevant_bits_mask_32(bits.clone()),
        );
        assert_eq!(
            relevant_bits_mask_64(bits.clone()), expected,
            "64bit impl fail {op} {:x}", relevant_bits_mask_64(bits.clone()),
        );
    }

    let ctx = OperandContext::new();
    check(ctx.const_0(), 0u64);
    check(ctx.constant(u64::MAX), u64::MAX);
    let op = ctx.or(
        ctx.lsh_const(ctx.mem8(ctx.register(0), 0), 8),
        ctx.lsh_const(ctx.mem8(ctx.register(0), 0), 0x18),
    );
    check(op, 0xffffff00);
}
