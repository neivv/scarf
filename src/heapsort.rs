//! Heapsort implementation from Rust's standard library.
//! Used over slice::sort, as there are several places in this library
//! where mostly small vectors are being sorted, and slice::sort adds
//! unnecessary bloat by pulling in multiple different sorting
//! algorithms, multiplied by the fact that each different sortable
//! type causes a new function to be generated.


pub fn sort_by<T, F>(v: &mut [T], mut is_less: F)
    where F: FnMut(&T, &T) -> bool
{
    // This binary heap respects the invariant `parent >= child`.
    let mut sift_down = |v: &mut [T], mut node: usize| {
        loop {
            // Children of `node`:
            let left = node.wrapping_mul(2).wrapping_add(1);
            let right = node.wrapping_mul(2).wrapping_add(2);

            // Choose the greater child.
            let greater = if right < v.len() && is_less(&v[left], &v[right]) {
                right
            } else {
                left
            };

            // Stop if the invariant holds at `node`.
            if greater >= v.len() || !is_less(&v[node], &v[greater]) {
                break;
            }

            // Swap `node` with the greater child, move one step down, and continue sifting.
            v.swap(node, greater);
            node = greater;
        }
    };


    // Build the heap in linear time.
    for i in (0 .. v.len() / 2).rev() {
        sift_down(v, i);
    }

    // Pop maximal elements from the heap.
    for i in (1 .. v.len()).rev() {
        v.swap(0, i);
        sift_down(&mut v[..i], 0);
    }
}

pub fn sort<T: std::cmp::Ord>(ops: &mut [T]) {
    sort_by(&mut *ops, |a, b| a < b);
}
