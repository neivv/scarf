use std::ops::Range;

pub fn bits_overlap(a: &Range<u8>, b: &Range<u8>) -> bool {
    a.end > b.start && a.start < b.end
}

#[test]
fn test_bits_overlap() {
    assert!(bits_overlap(&(0..16), &(15..16)));
    assert!(!bits_overlap(&(0..16), &(16..16)));
    assert!(bits_overlap(&(0..19), &(5..16)));
    assert!(bits_overlap(&(3..16), &(1..4)));
    assert!(!bits_overlap(&(3..16), &(0..3)));
}
