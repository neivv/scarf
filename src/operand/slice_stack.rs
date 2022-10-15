//! Reusable vector/arena-like struct for temp space during operand simplification.

use std::alloc;
use std::cell::{Cell, UnsafeCell};
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};
use std::slice;

// These are usize-sized, must allow at least 16-byte structs
// 48 * (16 / sizeof(usize) on 32-bit) = 192 <= CHUNK_SIZE
const CHUNK_SIZE: usize = 256;
const SLICE_SIZE_LIMIT: usize = 48;
const INIT_CAPACITY: usize = 6;
const CAPACITY_INCREASE: usize = 6;

pub struct SliceStack {
    // Chunk start is aligned to 16 bytes
    chunks: UnsafeCell<Vec<*mut [usize; CHUNK_SIZE]>>,
    pos: Cell<usize>,
}

unsafe impl Send for SliceStack {}

pub struct Slice<'e, T: Copy> {
    parent: &'e SliceStack,
    slice_ptr: NonNull<usize>,
    /// Length of slice in T (not usize)
    slice_len: usize,
    /// Index to parent's chunks at which `self.slice` is located
    /// Index in usize, even if T is differently sized
    start_index: usize,
    /// Index to parent's chunks at the slice's allocation end.
    /// This index won't decrease if the slice is shrunk;
    /// when `parent.pos` is equal to this then the slice can be grown
    /// in place (Assuming the backing ArrayVec isn't full)
    ///
    /// End_index must be equal to `parent.pos` when the slice is grown
    /// or dropped (Otherwise the functions panic). In other words,
    /// the slices must be dropped in reverse order of their creation,
    /// and not grown when a newer slice exists.
    /// Index in usize, even if T is differently sized
    end_index: usize,
    index_per_entry: u8,
    phantom: std::marker::PhantomData<&'e T>,
}

impl SliceStack {
    pub fn new() -> SliceStack {
        SliceStack {
            chunks: UnsafeCell::new(Vec::new()),
            pos: Cell::new(0),
        }
    }

    pub fn alloc<'e, T: Copy + 'e, F: FnOnce(&mut Slice<'e, T>) -> R, R>(&'e self, func: F) -> R {
        let required_align = mem::align_of::<T>();
        assert!(mem::size_of::<T>() <= 16);
        assert!(mem::size_of::<T>() >= mem::size_of::<usize>());
        assert!(required_align <= 16);
        let index_per_entry = mem::size_of::<T>() / mem::size_of::<usize>();
        assert!(index_per_entry > 0 && index_per_entry < 0x100);
        let usize_align = required_align / mem::size_of::<usize>();
        let capacity = index_per_entry * INIT_CAPACITY;
        let old_pos = self.pos.get();
        let mut slice = self.alloc_nongeneric(usize_align, capacity);
        let result = unsafe {
            slice.index_per_entry = index_per_entry as u8;
            let slice = mem::transmute::<&mut Slice<usize>, &mut Slice<T>>(&mut slice);
            func(slice)
        };
        self.pos.set(old_pos);
        result
    }

    fn alloc_nongeneric<'e>(&'e self, usize_align: usize, capacity: usize) -> Slice<'e, usize> {
        let pos = self.pos.get();
        let slice_start = if usize_align > 1 {
            // Align start
            (pos | usize_align - 1).wrapping_sub(usize_align - 1)
        } else {
            pos
        };
        // Allocate 6 entries, or to end of current slice if it would be too much
        let mut slice_end = slice_start.wrapping_add(capacity);
        if slice_end & !(CHUNK_SIZE - 1) != slice_start & !(CHUNK_SIZE - 1) {
            slice_end &= !(CHUNK_SIZE - 1);
        }
        let chunk_index = slice_start / CHUNK_SIZE;
        let slice_ptr;
        unsafe {
            let chunks = self.chunks.get();
            if let Some(chunk) = (*chunks).get_mut(chunk_index) {
                slice_ptr = NonNull::new_unchecked((**chunk).as_mut_ptr()
                    .add(slice_start % CHUNK_SIZE));
            } else {
                // Current chunk for slice_start hasn't been allocated,
                // change capacity to 0 so that when slice is grown it'll allocate.
                slice_end = slice_start;
                slice_ptr = NonNull::dangling();
            }
        }
        self.pos.set(slice_end);
        Slice {
            parent: self,
            slice_ptr,
            slice_len: 0,
            start_index: slice_start,
            end_index: slice_end,
            index_per_entry: 1,
            phantom: Default::default(),
        }
    }
}

#[derive(Debug)]
pub struct SizeLimitReached;

impl<'e, T: Copy> Slice<'e, T> {
    /// Returns error if the slice reaches a constant size limit.
    /// The caller (which is expected to be in simplification code) should just
    /// return an imperfectly simplified operand in that case.
    pub fn push(&mut self, value: T) -> Result<(), SizeLimitReached> {
        let index_per_entry = mem::size_of::<T>() / mem::size_of::<usize>();
        assert!(index_per_entry > 0);
        unsafe {
            let len = self.len();
            if len != self.end_index.wrapping_sub(self.start_index) / index_per_entry {
                // Can just grow the slice
                // Pushing to non-topmost slice of stack should not be done, but
                // if it had capacity it is fine to do without corrupting anything,
                // so only check this on debug.
                // grow_for_push checks it separately always.
                debug_assert!(self.parent.pos.get() == self.end_index);
            } else if len < SLICE_SIZE_LIMIT {
                // grow_for_push is not generic, but not going to bother making
                // everything but the PhantomData part of a smaller structure so
                // just transmute this.
                let this = mem::transmute::<&mut Slice<'e, T>, &mut Slice<'e, usize>>(&mut *self);
                this.grow_for_push();
            } else {
                return Err(SizeLimitReached);
            }
            self.slice_len = len.wrapping_add(1);
            let ptr = (self.slice_ptr.as_ptr() as *mut T).add(len);
            ptr.write(value);
            Ok(())
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        // This if branch is faster than just the else branch with `?` instead of `unwrap`
        let len = self.len();
        if len == 0 {
            None
        } else {
            unsafe {
                self.slice_len = self.slice_len.wrapping_sub(1);
                let ptr = (self.slice_ptr.as_ptr() as *mut T).add(self.slice_len);
                Some(*ptr)
            }
        }
    }

    pub fn clear(&mut self) {
        self.slice_len = 0;
    }

    pub fn shrink(&mut self, new_len: usize) {
        assert!(new_len <= self.len());
        self.slice_len = new_len;
    }

    pub fn retain<F: FnMut(T) -> bool>(&mut self, mut func: F) {
        let mut i = 0;
        let mut end = self.len();
        unsafe {
            let ptr = self.slice_ptr.as_ptr() as *mut T;
            while i < end {
                let input = ptr.add(i);
                if !func(*input) {
                    *input = *ptr.add(end - 1);
                    end -= 1;
                } else {
                    i += 1;
                }
            }
            self.slice_len = end;
        }
    }

    pub fn swap_remove(&mut self, i: usize) -> T {
        let len = self.len();
        let slice = self.deref_mut();
        let ret = slice[i];
        slice[i] = slice[len - 1];
        self.pop();
        ret
    }

    pub fn remove(&mut self, i: usize) -> T {
        let len = self.len();
        let slice = self.deref_mut();
        let ret = slice[i];
        unsafe {
            let ptr = slice.as_mut_ptr();
            ptr::copy(ptr.add(i).add(1), ptr.add(i), len.wrapping_sub(i).wrapping_sub(1));
            self.slice_len = self.slice_len.wrapping_sub(1);
        }
        ret
    }
}

impl<'e> Slice<'e, usize> {
    // Assumes that the fast push path was already checked (len == capacity here).
    #[cold]
    unsafe fn grow_for_push(&mut self) {
        if self.parent.pos.get() != self.end_index {
            panic!("Tried pushing to subslice which was not on top of slice stack");
        }
        let len = self.len();
        let index_per_entry = self.index_per_entry as usize;
        debug_assert!(index_per_entry > 0);

        let new_capacity = len.wrapping_add(CAPACITY_INCREASE)
            .min(SLICE_SIZE_LIMIT)
            .wrapping_mul(index_per_entry);
        let last_write_index_for_next_obj =
            self.end_index.wrapping_add(index_per_entry.wrapping_sub(1));
        if last_write_index_for_next_obj % CHUNK_SIZE < index_per_entry {
            // Allocate a new index in parent if it doesn't exist,
            // copy the already existing slice there.
            let chunks = self.parent.chunks.get();
            let insert_chunk_index = last_write_index_for_next_obj / CHUNK_SIZE;
            if (*chunks).len() <= insert_chunk_index {
                let layout = alloc::Layout::new::<[usize; CHUNK_SIZE]>();
                let layout = alloc::Layout::from_size_align(layout.size(), 16).unwrap();
                let ptr = alloc::alloc(layout);
                if ptr.is_null() {
                    alloc::handle_alloc_error(layout);
                }
                (*chunks).push(ptr as *mut [usize; CHUNK_SIZE]);
            }
            let new_chunk = (*chunks)[insert_chunk_index] as *mut usize;
            if len != 0 {
                let len_usize = len.wrapping_mul(index_per_entry);
                ptr::copy_nonoverlapping(self.slice_ptr.as_ptr(), new_chunk, len_usize);
            }
            self.slice_ptr = NonNull::new_unchecked(new_chunk);
            self.start_index = insert_chunk_index * CHUNK_SIZE;
            self.end_index = self.start_index
                .wrapping_add(new_capacity);
        } else {
            // Can just extend capacity in this chunk
            let current_chunk_start = self.start_index & !(CHUNK_SIZE - 1);
            let max_end_index = current_chunk_start.wrapping_add(CHUNK_SIZE);
            self.end_index = self.start_index.wrapping_add(new_capacity)
                .min(max_end_index);
        }
        // Last writable index and first writable index must be on same chunk,
        // just sanity check this.
        debug_assert_ne!(self.start_index, self.end_index);
        debug_assert_eq!(
            self.start_index / CHUNK_SIZE,
            self.end_index.wrapping_sub(1) / CHUNK_SIZE,
        );
        self.parent.pos.set(self.end_index);
    }
}

impl<'e, T: Copy + PartialEq> Slice<'e, T> {
    pub fn dedup(&mut self) {
        let mut in_pos = 0;
        let mut out_pos = 0;
        while in_pos < self.len() {
            let slice = self.deref_mut();
            let compare = slice[in_pos];
            unsafe { *slice.get_unchecked_mut(out_pos) = compare; }
            in_pos = in_pos.wrapping_add(1);
            out_pos = out_pos.wrapping_add(1);
            while in_pos < slice.len() && slice[in_pos] == compare {
                in_pos = in_pos.wrapping_add(1);
            }
        }
        self.shrink(out_pos);
    }
}

impl Drop for SliceStack {
    fn drop(&mut self) {
        unsafe {
            for &chunk in (*self.chunks.get()).iter() {
                let layout = alloc::Layout::new::<[usize; CHUNK_SIZE]>();
                let layout = alloc::Layout::from_size_align(layout.size(), 16).unwrap();
                alloc::dealloc(chunk as *mut u8, layout);
            }
        }
    }
}

impl<'e, T: Copy + fmt::Debug> fmt::Debug for Slice<'e, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.deref().fmt(f)
    }
}

impl<'e, T: Copy> Deref for Slice<'e, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe {
            slice::from_raw_parts(self.slice_ptr.as_ptr() as *mut T, self.slice_len)
        }
    }
}

impl<'e, T: Copy> DerefMut for Slice<'e, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            slice::from_raw_parts_mut(self.slice_ptr.as_ptr() as *mut T, self.slice_len)
        }
    }
}

#[test]
fn test_remove() {
    let ctx = &super::OperandContext::new();
    ctx.simplify_temp_stack().alloc(|slice| {
        for i in 0..32 {
            slice.push(ctx.constant(i)).unwrap();
        }
        for i in 0..32 {
            assert_eq!(slice[i], ctx.constant(i as u64));
        }
        slice.remove(5);
        slice.remove(30);
        for i in 0..5 {
            assert_eq!(slice[i], ctx.constant(i as u64));
        }
        for i in 5..30 {
            assert_eq!(slice[i], ctx.constant(i as u64 + 1));
        }
    });
}

#[test]
fn test_nested() {
    let ctx = &super::OperandContext::new();
    fn recurse<'e>(ctx: super::OperandCtx<'e>, pos: u64) {
        if pos > 1024 {
            return;
        }
        ctx.simplify_temp_stack().alloc(|slice| {
            for i in (pos..).take(31) {
                slice.push(ctx.constant(i)).unwrap();
            }
            recurse(ctx, pos + 31);
            for i in 0..31 {
                assert_eq!(slice[i], ctx.constant(pos + i as u64));
            }
        });
    }
    recurse(ctx, 0);
}

#[test]
fn test_mixed() {
    use super::Operand;
    let ctx = &super::OperandContext::new();
    #[derive(Copy, Clone)]
    struct OtherThing<'e> {
        val: Operand<'e>,
        something: u8,
    }

    fn recurse<'e>(ctx: super::OperandCtx<'e>, pos: u64) {
        if pos > 1024 {
            return;
        }
        if pos & 1 == 0 {
            ctx.simplify_temp_stack().alloc(|slice| {
                for i in (pos..).take(31) {
                    slice.push(ctx.constant(i)).unwrap();
                }
                recurse(ctx, pos + 31);
                for i in 0..31 {
                    assert_eq!(slice[i], ctx.constant(pos + i as u64));
                }
            });
        } else {
            ctx.simplify_temp_stack().alloc(|slice| {
                for i in (pos..).take(31) {
                    slice.push(OtherThing {
                        val: ctx.constant(i),
                        something: 255 - i as u8,
                    }).unwrap();
                }
                recurse(ctx, pos + 31);
                for i in 0..31 {
                    assert_eq!(slice[i].val, ctx.constant(pos + i as u64));
                    assert_eq!(slice[i].something, 255 - (pos + i as u64) as u8);
                }
            });
        }
    }
    recurse(ctx, 0);
}
