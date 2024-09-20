use std::ffi::OsStr;
use std::io::Read;

use scarf::{BinaryFile, BinarySection, VirtualAddress32, VirtualAddress64};

pub fn raw_bin(filename: &OsStr) -> Result<BinaryFile<VirtualAddress32>, scarf::Error> {
    let mut file = std::fs::File::open(filename)?;
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    Ok(scarf::raw_bin(VirtualAddress32(0x00400000), vec![BinarySection {
        name: *b".text\0\0\0",
        virtual_address: VirtualAddress32(0x401000),
        virtual_size: buf.len() as u32,
        data: buf,
    }]))
}

#[allow(dead_code)]
pub fn raw_bin_64(filename: &OsStr) -> Result<BinaryFile<VirtualAddress64>, scarf::Error> {
    let mut file = std::fs::File::open(filename)?;
    let mut buf = vec![];
    file.read_to_end(&mut buf)?;
    Ok(scarf::raw_bin(VirtualAddress64(0x00400000), vec![BinarySection {
        name: *b".text\0\0\0",
        virtual_address: VirtualAddress64(0x401000),
        virtual_size: buf.len() as u32,
        data: buf,
    }]))
}
