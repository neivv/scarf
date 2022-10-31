use std::cmp::min;
use std::ops::Range;

pub struct ZeroBitRanges(u64, u8);

impl Iterator for ZeroBitRanges {
    type Item = Range<u8>;
    fn next(&mut self) -> Option<Range<u8>> {
        while self.0 & 1 == 1 {
            self.1 += 1;
            self.0 = self.0 >> 1;
        }
        if self.1 >= 64 {
            None
        } else {
            let amt = min(64 - self.1, self.0.trailing_zeros() as u8);
            let range = self.1..self.1 + amt;
            self.1 += amt;
            self.0 = self.0.checked_shr(u32::from(amt)).unwrap_or(0);
            Some(range)
        }
    }
}

pub fn zero_bit_ranges(val: u64) -> ZeroBitRanges {
    ZeroBitRanges(val, 0)
}

pub fn bits_overlap(a: &Range<u8>, b: &Range<u8>) -> bool {
    a.end > b.start && a.start < b.end
}

#[test]
fn test_zero_bit_range() {
    let mut iter = zero_bit_ranges(0xff00f40f);
    assert_eq!(iter.next().unwrap(), 0x4..0xa);
    assert_eq!(iter.next().unwrap(), 0xb..0xc);
    assert_eq!(iter.next().unwrap(), 0x10..0x18);
    assert_eq!(iter.next().unwrap(), 0x20..0x40);
    assert_eq!(iter.next(), None);
    let mut iter = zero_bit_ranges(0xffff_ffff_ff00f40f);
    assert_eq!(iter.next().unwrap(), 0x4..0xa);
    assert_eq!(iter.next().unwrap(), 0xb..0xc);
    assert_eq!(iter.next().unwrap(), 0x10..0x18);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_zero_bit_range2() {
    let mut iter = zero_bit_ranges(0x0f00f40e);
    assert_eq!(iter.next().unwrap(), 0x0..0x1);
    assert_eq!(iter.next().unwrap(), 0x4..0xa);
    assert_eq!(iter.next().unwrap(), 0xb..0xc);
    assert_eq!(iter.next().unwrap(), 0x10..0x18);
    assert_eq!(iter.next().unwrap(), 0x1c..0x40);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_bits_overlap() {
    assert!(bits_overlap(&(0..16), &(15..16)));
    assert!(!bits_overlap(&(0..16), &(16..16)));
    assert!(bits_overlap(&(0..19), &(5..16)));
    assert!(bits_overlap(&(3..16), &(1..4)));
    assert!(!bits_overlap(&(3..16), &(0..3)));
}

#[test]
fn test_full_ranges() {
    let mut iter = zero_bit_ranges(0);
    assert_eq!(iter.next().unwrap(), 0..64);
    assert_eq!(iter.next(), None);
    let mut iter = zero_bit_ranges(!0);
    assert_eq!(iter.next(), None);
}
