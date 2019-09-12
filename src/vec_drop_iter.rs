use std::mem;
use std::ops::{Deref, DerefMut};

/// An "iterator" which allows to drop the current element.
pub struct VecDropIter<'a, T: 'a>(&'a mut Vec<T>, usize);

impl<'a, T: 'a> VecDropIter<'a, T> {
    pub fn new(vec: &mut Vec<T>) -> VecDropIter<T> {
        VecDropIter(vec, 0)
    }

    pub fn duplicate(&mut self) -> VecDropIter<T> {
        VecDropIter(self.0, self.1)
    }

    pub fn next(&mut self) -> Option<Item<'a, T>> {
        if self.1 < self.0.len() {
            let item = &mut self.0[self.1];
            self.1 += 1;
            // The transmute is valid since the item cannot be accessed again,
            // even if the VecDropIter stays usable.
            // On the other hand, next_removable gives an item which can be removed,
            // and removing will cause all items after it to be accessed, so it will
            // borrow VecDropIter.
            Some(Item(unsafe { mem::transmute(item) }))
        } else {
            None
        }
    }

    pub fn next_removable<'b>(&'b mut self) -> Option<ItemRemovable<'b, 'a, T>> {
        if self.1 < self.0.len() {
            self.1 += 1;
            Some(ItemRemovable(self))
        } else {
            None
        }
    }
}

pub struct Item<'a, T: 'a>(&'a mut T);
pub struct ItemRemovable<'a, 'b: 'a, T: 'b>(&'a mut VecDropIter<'b, T>);

impl<'a, T: 'a> Deref for Item<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0
    }
}

impl<'a, T: 'a> DerefMut for Item<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0
    }
}

impl<'a, 'b: 'a, T: 'b> ItemRemovable<'a, 'b, T> {
    pub fn remove(self) {
        (self.0).1 -= 1;
        (self.0).0.remove((self.0).1);
    }
}

impl<'a, 'b: 'a, T: 'b> Deref for ItemRemovable<'a, 'b, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &(self.0).0[(self.0).1 - 1]
    }
}

impl<'a, 'b: 'a, T: 'b> DerefMut for ItemRemovable<'a, 'b, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut (self.0).0[(self.0).1 - 1]
    }
}
