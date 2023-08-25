use std::hash::{Hasher};

use fxhash::FxHasher;

/// Fxhash seems to be bit better for single integers when the result is xored with
/// itself shifted.
#[derive(Default)]
pub struct ConstFxHasher {
    pub hasher: FxHasher,
}

pub fn hash_u64(value: u64) -> u64 {
    let mut hasher = ConstFxHasher::default();
    hasher.write_u64(value);
    hasher.finish()
}

impl Hasher for ConstFxHasher {
    fn finish(&self) -> u64 {
        let val = self.hasher.finish();
        val ^ (val.rotate_right(std::mem::size_of::<usize>() as u32 * 4))
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
