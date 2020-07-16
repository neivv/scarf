//! Reusable vector/arena-like struct for temp space during operand simplification.

use std::alloc;
use std::cell::{Cell, UnsafeCell};
use std::fmt;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::ptr;
use std::slice;

// These are usize-sized, must allow at least 16-byte structs
// 48 * (16 / sizeof(usize) on 32-bit) = 192 <= CHUNK_SIZE
const CHUNK_SIZE: usize = 256;
const SLICE_SIZE_LIMIT: usize = 48;

pub struct SliceStack {
    // Chunk start is aligned to 16 bytes
    chunks: UnsafeCell<Vec<*mut [usize; CHUNK_SIZE]>>,
    pos: Cell<usize>,
}

unsafe impl Send for SliceStack {}

pub struct Slice<'e, T: Copy> {
    parent: &'e SliceStack,
    slice: &'e mut [T],
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
        let usize_align = required_align / mem::size_of::<usize>();
        let pos = self.pos.get();
        let slice_start = if usize_align > 1 {
            // Align start
            (pos | usize_align - 1).wrapping_sub(usize_align - 1)
        } else {
            pos
        };
        let mut slice = Slice {
            parent: self,
            slice: &mut [],
            start_index: slice_start,
            end_index: slice_start,
        };
        let result = func(&mut slice);
        self.pos.set(pos);
        result
    }
}

#[derive(Debug)]
pub struct SizeLimitReached;

impl<'e, T: Copy> Slice<'e, T> {
    /// Returns error if the slice reaches a constant size limit.
    /// The caller (which is expected to be in simplification code) should just
    /// return an imperfectly simplified operand in that case.
    pub fn push(&mut self, value: T) -> Result<(), SizeLimitReached> {
        if self.parent.pos.get() != self.end_index {
            panic!("Tried pushing to subslice which was not on top of slice stack");
        }
        let len = self.len();
        if len == SLICE_SIZE_LIMIT {
            return Err(SizeLimitReached);
        }
        let index_per_entry = mem::size_of::<T>() / mem::size_of::<usize>();
        assert!(index_per_entry > 0);
        unsafe {
            if len != self.end_index.wrapping_sub(self.start_index) / index_per_entry {
                // Can just grow the slice
                let ptr = self.slice.as_mut_ptr();
                ptr.add(len).write(value);
                self.slice = &mut [];
                self.slice = slice::from_raw_parts_mut(ptr, len.wrapping_add(1));
            } else if self.end_index % CHUNK_SIZE < index_per_entry {
                // Allocate a new index in parent if it doesn't exist,
                // copy the already existing slice there.
                let chunks = self.parent.chunks.get();
                let chunk_index = self.end_index / CHUNK_SIZE;
                if (*chunks).len() <= chunk_index {
                    let layout = alloc::Layout::new::<[usize; CHUNK_SIZE]>();
                    let layout = alloc::Layout::from_size_align(layout.size(), 16).unwrap();
                    let ptr = alloc::alloc(layout);
                    if ptr.is_null() {
                        alloc::handle_alloc_error(layout);
                    }
                    (*chunks).push(ptr as *mut [usize; CHUNK_SIZE]);
                }
                let new_chunk = (*chunks)[chunk_index] as *mut T;
                if len != 0 {
                    ptr::copy_nonoverlapping(self.slice.as_ptr(), new_chunk, len);
                }
                new_chunk.add(len).write(value);
                self.slice = slice::from_raw_parts_mut(new_chunk, len.wrapping_add(1));
                self.start_index = chunk_index * CHUNK_SIZE;
                self.end_index = self.start_index.wrapping_add(
                    len.wrapping_add(1).wrapping_mul(index_per_entry)
                );
                self.parent.pos.set(self.end_index);
            } else {
                // Cannot use self.slice.as_mut_ptr here, if the slice is empty
                // it won't point to backing store (maybe should be fixed elsewhere?)
                let chunks = self.parent.chunks.get();
                let chunk_index = self.end_index / CHUNK_SIZE;
                let chunk = (*chunks)[chunk_index];
                self.slice = &mut [];
                let ptr = (chunk as *mut usize).add(self.start_index % CHUNK_SIZE) as *mut T;
                ptr.add(len).write(value);
                self.slice = slice::from_raw_parts_mut(ptr, len.wrapping_add(1));
                self.end_index = self.end_index.wrapping_add(index_per_entry);
                self.parent.pos.set(self.end_index);
            }
        }
        Ok(())
    }

    pub fn pop(&mut self) -> Option<T> {
        // This if branch is faster than just the else branch with `?` instead of `unwrap`
        let len = self.len();
        if len == 0 {
            None
        } else {
            unsafe {
                let ptr = self.slice.as_mut_ptr();
                self.slice = slice::from_raw_parts_mut(ptr, len - 1);
                Some(*ptr.add(len - 1))
            }
        }
    }

    pub fn clear(&mut self) {
        self.shrink(0);
    }

    pub fn shrink(&mut self, new_len: usize) {
        assert!(new_len <= self.len());
        let slice = mem::replace(&mut self.slice, &mut []);
        self.slice = &mut slice[..new_len];
    }

    pub fn retain<F: FnMut(T) -> bool>(&mut self, mut func: F) {
        let mut i = 0;
        let mut end = self.len();
        unsafe {
            let ptr = self.slice.as_mut_ptr();
            while i < end {
                let input = ptr.add(i);
                if !func(*input) {
                    *input = *ptr.add(end - 1);
                    end -= 1;
                } else {
                    i += 1;
                }
            }
            self.slice = slice::from_raw_parts_mut(ptr, end);
        }
    }

    pub fn swap_remove(&mut self, i: usize) -> T {
        let len = self.len();
        let ret = self.slice[i];
        self.slice[i] = self.slice[len - 1];
        self.pop();
        ret
    }

    pub fn remove(&mut self, i: usize) -> T {
        let len = self.len();
        let ret = self.slice[i];
        unsafe {
            let ptr = self.slice.as_mut_ptr();
            ptr::copy(ptr.add(i).add(1), ptr.add(i), len.wrapping_sub(i).wrapping_sub(1));
            self.slice = slice::from_raw_parts_mut(ptr, len.wrapping_sub(1));
        }
        ret
    }
}

impl<'e, T: Copy + PartialEq> Slice<'e, T> {
    pub fn dedup(&mut self) {
        let mut in_pos = 0;
        let mut out_pos = 0;
        while in_pos < self.len() {
            let compare = self.slice[in_pos];
            unsafe { *self.slice.get_unchecked_mut(out_pos) = compare; }
            in_pos = in_pos.wrapping_add(1);
            out_pos = out_pos.wrapping_add(1);
            while in_pos < self.len() && self.slice[in_pos] == compare {
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
        self.slice.fmt(f)
    }
}

impl<'e, T: Copy> Deref for Slice<'e, T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl<'e, T: Copy> DerefMut for Slice<'e, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice
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
