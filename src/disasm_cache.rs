use crate::u64_hash::hash_u64;
use crate::Operation;

pub struct DisasmCache<'e> {
    arch: DisasmArch,
    /// Size is always power of two + 8 to be able to trivially do one u64 read.
    /// `(hash(bytes) & (len_power_of_2 - 1))` to start index, and from there insert
    /// current instruction bytes. May be partially overwritten by
    /// another instruction, so instruction_lengths has to be checked
    /// too for validity.
    instruction_bytes: Vec<u8>,
    /// Same length as instruction_bytes. Start index contains the instruction
    /// length, rest must be 0 to detect partial overwrites.
    instruction_lengths: Vec<u8>,
    /// Same length as instruction_bytes's power of 2 base value.
    /// Only start index is used (Rest undef), contains (index, len) to `operations` vec.
    operation_indices: Vec<(u32, u32)>,
    hash_table_mask: u32,
    /// Ring buffer, read as `operations[index & (len - 1)]` and check
    /// that `index >= next_operations_index - len`.
    /// So the index contains buffer generation and old entries can be easily overwritten.
    operations: Vec<Operation<'e>>,
    next_operations_index: u32,
}

/// Used just to tell what instructions the
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum DisasmArch {
    X86,
    X86_64,
}

pub const MAX_INSTRUCTION_LEN: usize = 8;

impl<'e> DisasmCache<'e> {
    pub fn new(arch: DisasmArch) -> DisasmCache<'e> {
        // These seem to be reasonable values, allowing caches to save ~20% time spent
        // in disasm when analyzing lot of functions.
        // (Though time spent in disasm isn't that much compared to exec state updates..)
        //
        // Most of the memory is spent on operations vec, 512 operations is ~60kB memory,
        // but using just 256 operations would hurt cache quality a lot already.
        let hash_table_size = 4096;
        let operations_limit = 512;
        let hash_table_total_size = hash_table_size + MAX_INSTRUCTION_LEN;
        DisasmCache {
            instruction_bytes: vec![0; hash_table_total_size],
            instruction_lengths: vec![0; hash_table_total_size],
            operation_indices: vec![(0, 0); hash_table_size],
            // Just using Freeze as a trivially initializable dummy value
            operations: vec![Operation::Freeze; operations_limit],
            next_operations_index: 0,
            hash_table_mask: hash_table_size as u32 - 1,
            arch,
        }
    }

    pub fn set_arch(&mut self, arch: DisasmArch) {
        if self.arch != arch {
            self.reset();
            self.arch = arch;
        }
    }

    fn reset(&mut self) {
        self.instruction_lengths.fill(0u8);
        self.next_operations_index = 0;
    }

    /// Instruction must be written to a buffer with MAX_INSTRUCTION_LEN, bytes after `len`
    /// can be anything.
    pub fn get(
        &self,
        instruction: &[u8; MAX_INSTRUCTION_LEN],
        len: usize,
    ) -> Option<&[Operation<'e>]> {
        debug_assert!(len != 0 && len < MAX_INSTRUCTION_LEN);
        let mask_value_hash = instruction_to_u64(instruction, len, self.hash_table_mask);
        self.get_(len, &mask_value_hash)
    }

    fn get_(
        &self,
        len: usize,
        &(mask, value, hash): &(u64, u64, u32),
    ) -> Option<&[Operation<'e>]> {
        let old_bytes = ht_slice(&self.instruction_bytes, hash)?;
        if old_bytes & mask != value {
            return None;
        }
        let old_len = ht_slice(&self.instruction_lengths, hash)?;
        if old_len & mask != len as u64 {
            return None;
        }
        let (index, op_len) = ht_read_index(&self.operation_indices, hash)?;
        let op_buf_size = self.operations.len() as u32;
        if index.wrapping_add(op_buf_size) < self.next_operations_index {
            return None;
        }
        self.operations.get(((index & (op_buf_size - 1)) as usize)..)?.get(..(op_len as usize))
    }

    pub fn set(
        &mut self,
        instruction: &[u8; MAX_INSTRUCTION_LEN],
        len: usize,
        data: &[Operation<'e>],
    ) {
        debug_assert!(len != 0 && len < MAX_INSTRUCTION_LEN);
        let mask_value_hash = instruction_to_u64(instruction, len, self.hash_table_mask);
        self.set_(len, data, &mask_value_hash)
    }

    fn set_(
        &mut self,
        len: usize,
        data: &[Operation<'e>],
        &(mask, value, hash): &(u64, u64, u32),
    ) {
        let old_bytes;
        let old_len;

        let op_buf_size = self.operations.len() as u32;
        if self.next_operations_index > 0u32.wrapping_sub(op_buf_size * 2) {
            // Near to overflowing the index generations, reset to prevent issues.
            self.reset();
        }
        if data.len() > self.operations.len() / 4 {
            // ??? Not letting to fill most of the cache with one operation
            return;
        }
        if len < 8 {
            old_bytes = match ht_slice(&mut self.instruction_bytes, hash) {
                Some(s) => s & !mask,
                None => return,
            };
            old_len = match ht_slice(&mut self.instruction_lengths, hash) {
                Some(s) => s & !mask,
                None => return,
            };
        } else {
            old_bytes = 0;
            old_len = 0;
        }
        ht_set_slice(&mut self.instruction_bytes, value | old_bytes, hash);
        ht_set_slice(&mut self.instruction_lengths, (len as u64) | old_len, hash);

        let op_count = data.len() as u32;
        let op_buf_mask = op_buf_size - 1;
        let mut index = self.next_operations_index;
        let mut next_index = index.wrapping_add(op_count);
        if index & !op_buf_mask != next_index & !op_buf_mask {
            // data doesn't fit at end of op buf, set index to start of op buf
            index = (index | op_buf_mask) + 1;
            next_index = index.wrapping_add(op_count);
        }
        self.next_operations_index = next_index;
        ht_set_index(&mut self.operation_indices, (index, op_count), hash);
        self.operations[((index & op_buf_mask) as usize)..((next_index & op_buf_mask) as usize)]
            .copy_from_slice(data);
    }
}

fn instruction_to_u64(
    instruction: &[u8; MAX_INSTRUCTION_LEN],
    len: usize,
    hash_mask: u32,
) -> (u64, u64, u32) {
    // 1 => 0xff, 2 => 0xff_fff, ... 8 => u64::MAX
    let mask = u64::MAX.wrapping_shr(8u32.wrapping_sub(len as u32) << 3);
    let value = u64::from_le_bytes(*instruction) & mask;
    let hash = (hash_u64(value) as u32) & hash_mask;
    (mask, value, hash)
}

fn ht_slice(table: &[u8], index: u32) -> Option<u64> {
    let index = index as usize;
    let slice: &[u8; 8] = table.get(index..(index.wrapping_add(8)))?.try_into().ok()?;
    Some(u64::from_le_bytes(*slice))
}

fn ht_set_slice(table: &mut [u8], value: u64, index: u32) {
    let index = index as usize;
    let slice: Option<&mut [u8; 8]> = table.get_mut(index..(index.wrapping_add(8)))
        .and_then(|x| x.try_into().ok());
    if let Some(slice) = slice {
        *slice = u64::to_le_bytes(value);
    }
}

fn ht_read_index(table: &[(u32, u32)], index: u32) -> Option<(u32, u32)> {
    let index = index as usize;
    table.get(index).copied()
}

fn ht_set_index(table: &mut [(u32, u32)], value: (u32, u32), index: u32) {
    let index = index as usize;
    if let Some(out) = table.get_mut(index) {
        *out = value;
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{DestOperand, OperandContext, OperandCtx};

    fn test_hash(bytes: &[u8; MAX_INSTRUCTION_LEN], len: usize, hash: u32) -> (u64, u64, u32) {
        let (a, b, _) = instruction_to_u64(bytes, len, u32::MAX);
        (a, b, hash)
    }

    fn gen_test_ops<'e>(ctx: OperandCtx<'e>, i: u32) -> Vec<Operation<'e>> {
        match i {
            0 => {
                vec![
                    Operation::Call(ctx.register(6)),
                    Operation::Jump { to: ctx.constant(9), condition: ctx.constant(1), },
                ]
            }
            1..=10 => {
                vec![
                    Operation::Move(
                        DestOperand::from_oper(ctx.register(1)),
                        ctx.register(2 + i as u8),
                        None,
                    ),
                ]
            }
            _ => unreachable!(),
        }
    }

    fn verify_test_ops<'e>(ctx: OperandCtx<'e>, i: u32, out: &[Operation<'e>]) {
        match i {
            0 => {
                assert_eq!(out.len(), 2);
                match out[0] {
                    Operation::Call(x) => assert_eq!(x, ctx.register(6)),
                    _ => panic!(),
                };
                match out[1] {
                    Operation::Jump { to, condition } => {
                        assert_eq!(to, ctx.constant(9));
                        assert_eq!(condition, ctx.constant(1));
                    }
                    _ => panic!(),
                };
            }
            1..=10 => {
                assert_eq!(out.len(), 1);
                match out[0] {
                    Operation::Move(dest, val, None) => {
                        assert_eq!(dest, DestOperand::from_oper(ctx.register(1)));
                        assert_eq!(val, ctx.register(2 + i as u8));
                    }
                    _ => panic!(),
                };
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn simple() {
        let ctx = &OperandContext::new();
        let mut cache = DisasmCache::new(DisasmArch::X86);
        let test_ins = [6, 7, 8, 9, 0, 0, 0, 0];
        let test_ops = gen_test_ops(ctx, 0);
        cache.set(&test_ins, 4, &test_ops);
        let out = cache.get(&test_ins, 4).unwrap();
        verify_test_ops(ctx, 0, out);
        assert!(cache.get(&test_ins, 3).is_none());
        assert!(cache.get(&test_ins, 5).is_none());
    }

    #[test]
    fn overwrite1() {
        let ctx = &OperandContext::new();
        let mut cache = DisasmCache::new(DisasmArch::X86);
        // Overwrite at same hash with shorter
        let test_ins = [6, 7, 8, 9, 0, 0, 0, 0];
        let test_ins2 = [5, 7, 8, 9, 0, 0, 0, 0];
        let test_ops = gen_test_ops(ctx, 0);
        let test_ops2 = gen_test_ops(ctx, 1);
        cache.set_(4, &test_ops, &test_hash(&test_ins, 4, 79));
        cache.set_(3, &test_ops2, &test_hash(&test_ins2, 3, 79));

        assert!(cache.get_(4, &test_hash(&test_ins, 4, 79)).is_none());
        let out = cache.get_(3, &test_hash(&test_ins2, 3, 79)).unwrap();
        verify_test_ops(ctx, 1, out);
    }

    #[test]
    fn overwrite2() {
        let ctx = &OperandContext::new();
        let mut cache = DisasmCache::new(DisasmArch::X86);
        // Overwrite at same hash with longer
        let test_ins = [6, 7, 8, 9, 0, 0, 0, 0];
        let test_ins2 = [5, 7, 8, 9, 0, 0, 0, 0];
        let test_ops = gen_test_ops(ctx, 0);
        let test_ops2 = gen_test_ops(ctx, 1);
        cache.set_(3, &test_ops2, &test_hash(&test_ins2, 3, 79));
        cache.set_(4, &test_ops, &test_hash(&test_ins, 4, 79));

        assert!(cache.get_(3, &test_hash(&test_ins2, 3, 79)).is_none());
        let out = cache.get_(4, &test_hash(&test_ins, 4, 79)).unwrap();
        verify_test_ops(ctx, 0, out);
    }

    #[test]
    fn overwrite3() {
        let ctx = &OperandContext::new();
        let mut cache = DisasmCache::new(DisasmArch::X86);
        // Overwrite at hash + 1
        let test_ins = [6, 7, 8, 9, 0, 0, 0, 0];
        let test_ins2 = [5, 7, 8, 9, 0, 0, 0, 0];
        let test_ops = gen_test_ops(ctx, 0);
        let test_ops2 = gen_test_ops(ctx, 1);
        cache.set_(4, &test_ops, &test_hash(&test_ins, 4, 79));
        cache.set_(3, &test_ops2, &test_hash(&test_ins2, 3, 80));

        assert!(cache.get_(4, &test_hash(&test_ins, 4, 79)).is_none());
        let out = cache.get_(3, &test_hash(&test_ins2, 3, 80)).unwrap();
        verify_test_ops(ctx, 1, out);
    }

    #[test]
    fn overwrite4() {
        let ctx = &OperandContext::new();
        let mut cache = DisasmCache::new(DisasmArch::X86);
        // Overwrite at hash - 1
        let test_ins = [6, 7, 8, 9, 0, 0, 0, 0];
        let test_ins2 = [5, 7, 8, 9, 0, 0, 0, 0];
        let test_ops = gen_test_ops(ctx, 0);
        let test_ops2 = gen_test_ops(ctx, 1);
        cache.set_(4, &test_ops, &test_hash(&test_ins, 4, 79));
        cache.set_(3, &test_ops2, &test_hash(&test_ins2, 3, 78));

        assert!(cache.get_(4, &test_hash(&test_ins, 4, 79)).is_none());
        let out = cache.get_(3, &test_hash(&test_ins2, 3, 78)).unwrap();
        verify_test_ops(ctx, 1, out);
    }
}
