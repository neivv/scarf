use std;
use std::ffi::OsStr;
use std::io::Read;

use scarf::{self, BinaryFile, BinarySection, VirtualAddress};

pub fn raw_bin(filename: &OsStr) -> Result<BinaryFile, scarf::Error> {
    let mut file = std::fs::File::open(filename)?;
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    Ok(scarf::raw_bin(VirtualAddress(0x00400000), vec![BinarySection {
        name: {
            // ugh
            let mut x = [0; 8];
            for (out, &val) in x.iter_mut().zip(b".text\0\0\0".iter()) {
                *out = val;
            }
            x
        },
        virtual_address: VirtualAddress(0x401000),
        physical_address: ::scarf::PhysicalAddress(0x1000),
        virtual_size: buf.len() as u32,
        data: buf,
    }]))
}
