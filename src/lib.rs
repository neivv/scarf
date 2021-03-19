#![allow(clippy::style, clippy::bool_comparison, clippy::needless_lifetimes)]

#[macro_use] extern crate log;

pub mod analysis;
mod bit_misc;
pub mod cfg;
pub mod cfg_dot;
mod disasm;
pub mod exec_state;
pub mod exec_state_x86;
pub mod exec_state_x86_64;
mod heapsort;
mod light_byteorder;
pub mod operand;

pub use crate::analysis::{Analyzer};
pub use crate::disasm::{DestOperand, Operation, FlagArith, FlagUpdate, operation_helpers};
pub use crate::operand::{
    ArithOpType, MemAccessSize, Operand, OperandType, OperandContext, OperandCtx,
};

pub use crate::exec_state_x86::ExecutionState as ExecutionStateX86;
pub use crate::exec_state_x86_64::ExecutionState as ExecutionStateX86_64;

use std::ffi::{OsString, OsStr};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};

use quick_error::quick_error;

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rva(pub u32);

impl std::fmt::Debug for Rva {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Rva({:08x})", self.0)
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualAddress(pub u32);

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualAddress64(pub u64);

impl std::fmt::Debug for VirtualAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "VirtualAddress({:08x})", self.0)
    }
}

impl std::fmt::Debug for VirtualAddress64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "VirtualAddress({:08x}_{:08x})", self.0 >> 32, self.0 & 0xffff_ffff)
    }
}

impl std::fmt::LowerHex for VirtualAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:08x}", self.0)
    }
}

impl std::fmt::LowerHex for VirtualAddress64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

impl std::fmt::UpperHex for VirtualAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:08X}", self.0)
    }
}

impl std::fmt::UpperHex for VirtualAddress64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:016X}", self.0)
    }
}

impl std::ops::Add<Rva> for VirtualAddress {
    type Output = VirtualAddress;
    #[inline]
    fn add(self, rhs: Rva) -> VirtualAddress {
        self + rhs.0
    }
}

impl std::ops::Add<Rva> for VirtualAddress64 {
    type Output = VirtualAddress64;
    #[inline]
    fn add(self, rhs: Rva) -> VirtualAddress64 {
        self + rhs.0
    }
}

impl std::ops::Add<u32> for VirtualAddress {
    type Output = VirtualAddress;
    #[inline]
    fn add(self, rhs: u32) -> VirtualAddress {
        VirtualAddress(self.0.wrapping_add(rhs))
    }
}

impl std::ops::Add<u32> for VirtualAddress64 {
    type Output = VirtualAddress64;
    #[inline]
    fn add(self, rhs: u32) -> VirtualAddress64 {
        VirtualAddress64(self.0.wrapping_add(rhs as u64))
    }
}

impl std::ops::Sub<u32> for VirtualAddress {
    type Output = VirtualAddress;
    #[inline]
    fn sub(self, rhs: u32) -> VirtualAddress {
        VirtualAddress(self.0.wrapping_sub(rhs))
    }
}

impl std::ops::Sub<u32> for VirtualAddress64 {
    type Output = VirtualAddress64;
    #[inline]
    fn sub(self, rhs: u32) -> VirtualAddress64 {
        VirtualAddress64(self.0.wrapping_sub(rhs as u64))
    }
}

impl std::ops::Sub<VirtualAddress> for VirtualAddress {
    type Output = Rva;
    #[inline]
    fn sub(self, rhs: VirtualAddress) -> Rva {
        Rva(self.0 - rhs.0)
    }
}

impl std::ops::Sub<VirtualAddress64> for VirtualAddress64 {
    type Output = u64;
    #[inline]
    fn sub(self, rhs: VirtualAddress64) -> u64 {
        self.0 - rhs.0
    }
}

impl std::ops::Add<u32> for Rva {
    type Output = Rva;
    #[inline]
    fn add(self, rhs: u32) -> Rva {
        Rva(self.0 + rhs)
    }
}

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Io(e: io::Error) {
            display("I/O error {}", e)
            from()
        }
        InvalidPeFile(detail: String) {
            display("Invalid PE file ({})", detail)
        }
        InvalidFilename(filename: OsString) {
            display("Invalid filename {:?}", filename)
        }
    }
}

/// Error type that is used when reading from BinaryFile by VirtualAddress
#[derive(Copy, Clone, Debug)]
pub struct OutOfBounds;

impl std::fmt::Display for OutOfBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Out of bounds")
    }
}

impl std::error::Error for OutOfBounds { }

impl From<light_byteorder::UnexpectedEof> for OutOfBounds {
    fn from(_: light_byteorder::UnexpectedEof) -> OutOfBounds {
        OutOfBounds
    }
}

#[derive(Debug, Clone)]
pub struct BinaryFile<Va: exec_state::VirtualAddress> {
    pub base: Va,
    sections: Vec<BinarySection<Va>>,
    relocs: Vec<Va>,
}

#[derive(Debug, Clone)]
pub struct BinarySection<Va: exec_state::VirtualAddress> {
    pub name: [u8; 8],
    pub virtual_address: Va,
    pub virtual_size: u32,
    pub data: Vec<u8>,
}

impl<Va: exec_state::VirtualAddress> BinaryFile<Va> {
    #[inline]
    pub fn base(&self) -> Va {
        self.base
    }

    /// Panics if code section for some reason would not exist.
    /// Also bad since assumes only one
    pub fn code_section(&self) -> &BinarySection<Va> {
        self.section(b".text\0\0\0").unwrap()
    }

    pub fn sections<'a>(&'a self) -> impl Iterator<Item = &'a BinarySection<Va>> + 'a {
        self.sections.iter()
    }

    pub fn code_sections<'a>(&'a self) -> impl Iterator<Item = &'a BinarySection<Va>> + 'a {
        self.sections.iter().filter(|x| &x.name[..] == &b".text\0\0\0"[..])
    }

    pub fn section(&self, name: &[u8; 0x8]) -> Option<&BinarySection<Va>> {
        self.sections.iter().find(|x| x.name[..] == name[..])
    }

    /// Returns a section containing the address, if it exists
    pub fn section_by_addr(&self, address: Va) -> Option<&BinarySection<Va>> {
        self.sections.iter().find(|x| {
            address >= x.virtual_address && address < (x.virtual_address + x.data.len() as u32)
        })
    }

    /// Range is relative from base
    pub fn slice_from(&self, range: std::ops::Range<u32>) -> Result<&[u8], OutOfBounds> {
        self.slice_from_address(self.base + range.start, range.end - range.start)
    }

    pub fn slice_from_address(&self, start: Va, len: u32) -> Result<&[u8], OutOfBounds> {
        self.section_by_addr(start)
            .and_then(|s| {
                let section_relative = start.as_u64() - s.virtual_address.as_u64();
                s.data.get(
                    section_relative as usize ..
                    (section_relative + len as u64) as usize,
                )
            })
            .ok_or_else(|| OutOfBounds)
    }

    pub fn read_u8(&self, addr: Va) -> Result<u8, OutOfBounds> {
        use crate::light_byteorder::ReadLittleEndian;
        self.section_by_addr(addr)
            .and_then(|s| {
                let section_relative = addr.as_u64() - s.virtual_address.as_u64();
                s.data.get(section_relative as usize..)
            })
            .and_then(|mut data| {
                data.read_u8().ok()
            })
            .ok_or_else(|| OutOfBounds)
    }

    pub fn read_u16(&self, addr: Va) -> Result<u16, OutOfBounds> {
        use crate::light_byteorder::ReadLittleEndian;
        self.section_by_addr(addr)
            .and_then(|s| {
                let section_relative = addr.as_u64() - s.virtual_address.as_u64();
                s.data.get(section_relative as usize..)
            })
            .and_then(|mut data| {
                data.read_u16().ok()
            })
            .ok_or_else(|| OutOfBounds)
    }

    pub fn read_u32(&self, addr: Va) -> Result<u32, OutOfBounds> {
        use crate::light_byteorder::ReadLittleEndian;
        self.section_by_addr(addr)
            .and_then(|s| {
                let section_relative = addr.as_u64() - s.virtual_address.as_u64();
                s.data.get(section_relative as usize..)
            })
            .and_then(|mut data| {
                data.read_u32().ok()
            })
            .ok_or_else(|| OutOfBounds)
    }

    pub fn read_u64(&self, addr: Va) -> Result<u64, OutOfBounds> {
        use crate::light_byteorder::ReadLittleEndian;
        self.section_by_addr(addr)
            .and_then(|s| {
                let section_relative = addr.as_u64() - s.virtual_address.as_u64();
                s.data.get(section_relative as usize..)
            })
            .and_then(|mut data| {
                data.read_u64().ok()
            })
            .ok_or_else(|| OutOfBounds)
    }

    pub fn read_address(&self, addr: Va) -> Result<Va, OutOfBounds> {
        match Va::SIZE {
            4 => self.read_u32(addr).map(|x| Va::from_u64(x as u64)),
            8 => self.read_u64(addr).map(|x| Va::from_u64(x)),
            x => panic!("Unsupported VirtualAddress size {}", x),
        }
    }

    pub fn set_relocs(&mut self, relocs: Vec<Va>) {
        self.relocs = relocs;
    }
}

/// Allows loading a BinaryFile from memory buffer(s) representing the binary sections.
pub fn raw_bin<Va: exec_state::VirtualAddress>(
    base: Va,
    sections: Vec<BinarySection<Va>>,
) -> BinaryFile<Va> {
    BinaryFile {
        base,
        sections,
        relocs: Vec::new(),
    }
}

pub fn parse(filename: &OsStr) -> Result<BinaryFile<VirtualAddress>, Error> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use crate::Error::*;
    let mut file = BufReader::new(File::open(filename)?);
    if file.read_u16::<LittleEndian>()? != 0x5a4d {
        return Err(InvalidPeFile("Missing DOS magic".into()));
    }
    let pe_offset = u64::from(read_at_32(&mut file, 0x3c)?);
    if read_at_32(&mut file, pe_offset)? != 0x0000_4550 {
        return Err(InvalidPeFile("Missing PE magic".into()));
    }
    let section_count = read_at_16(&mut file, pe_offset + 6)?;
    let base = VirtualAddress(read_at_32(&mut file, pe_offset + 0x34)?);
    let mut sections = (0..u64::from(section_count)).map(|i| {
        let mut name = [0; 8];
        file.seek(io::SeekFrom::Start(pe_offset + 0xf8 + 0x28 * i))?;
        file.read_exact(&mut name)?;
        file.seek(io::SeekFrom::Start(pe_offset + 0xf8 + 0x28 * i + 0x8))?;
        let virtual_size = file.read_u32::<LittleEndian>()?;
        let rva = Rva(file.read_u32::<LittleEndian>()?);
        let phys_size = file.read_u32::<LittleEndian>()?;
        let phys = file.read_u32::<LittleEndian>()?;

        file.seek(io::SeekFrom::Start(u64::from(phys)))?;
        let mut data = vec![0; phys_size as usize];
        file.read_exact(&mut data)?;
        Ok(BinarySection {
            name,
            virtual_address: base + rva,
            virtual_size,
            data,
        })
    }).collect::<Result<Vec<_>, Error>>()?;
    let header_block_size = read_at_32(&mut file, pe_offset + 0x54)?;
    file.seek(io::SeekFrom::Start(0))?;
    let mut header_data = vec![0; header_block_size as usize];
    file.read_exact(&mut header_data)?;
    sections.push(BinarySection {
        name: {
            let mut name = [0; 8];
            for (&c, out) in b"(header)".iter().zip(name.iter_mut()) {
                *out = c;
            }
            name
        },
        virtual_address: base,
        virtual_size: header_block_size,
        data: header_data,
    });
    Ok(BinaryFile {
        base,
        sections,
        relocs: Vec::new(),
    })
}

pub fn parse_x86_64(filename: &OsStr) -> Result<BinaryFile<VirtualAddress64>, Error> {
    use byteorder::{LittleEndian, ReadBytesExt};
    use crate::Error::*;
    let mut file = BufReader::new(File::open(filename)?);
    if file.read_u16::<LittleEndian>()? != 0x5a4d {
        return Err(InvalidPeFile("Missing DOS magic".into()));
    }
    let pe_offset = u64::from(read_at_32(&mut file, 0x3c)?);
    if read_at_32(&mut file, pe_offset)? != 0x0000_4550 {
        return Err(InvalidPeFile("Missing PE magic".into()));
    }
    let section_count = read_at_16(&mut file, pe_offset + 6)?;
    let base = VirtualAddress64(read_at_64(&mut file, pe_offset + 0x30)?);
    let mut sections = (0..u64::from(section_count)).map(|i| {
        let mut name = [0; 8];
        file.seek(io::SeekFrom::Start(pe_offset + 0x108 + 0x28 * i))?;
        file.read_exact(&mut name)?;
        file.seek(io::SeekFrom::Start(pe_offset + 0x108 + 0x28 * i + 0x8))?;
        let virtual_size = file.read_u32::<LittleEndian>()?;
        let rva = file.read_u32::<LittleEndian>()?;
        let phys_size = file.read_u32::<LittleEndian>()?;
        let phys = file.read_u32::<LittleEndian>()?;

        file.seek(io::SeekFrom::Start(u64::from(phys)))?;
        let mut data = vec![0; phys_size as usize];
        file.read_exact(&mut data)?;
        Ok(BinarySection {
            name,
            virtual_address: base + rva,
            virtual_size,
            data,
        })
    }).collect::<Result<Vec<_>, Error>>()?;
    let header_block_size = read_at_32(&mut file, pe_offset + 0x54)?;
    file.seek(io::SeekFrom::Start(0))?;
    let mut header_data = vec![0; header_block_size as usize];
    file.read_exact(&mut header_data)?;
    sections.push(BinarySection {
        name: {
            let mut name = [0; 8];
            for (&c, out) in b"(header)".iter().zip(name.iter_mut()) {
                *out = c;
            }
            name
        },
        virtual_address: base,
        virtual_size: header_block_size,
        data: header_data,
    });
    Ok(BinaryFile {
        base,
        sections,
        relocs: Vec::new(),
    })
}

fn read_at_16<R: Read + Seek>(f: &mut R, at: u64) -> Result<u16, io::Error> {
    use byteorder::{LittleEndian, ReadBytesExt};
    f.seek(io::SeekFrom::Start(at))?;
    f.read_u16::<LittleEndian>()
}

fn read_at_32<R: Read + Seek>(f: &mut R, at: u64) -> Result<u32, io::Error> {
    use byteorder::{LittleEndian, ReadBytesExt};
    f.seek(io::SeekFrom::Start(at))?;
    f.read_u32::<LittleEndian>()
}

fn read_at_64<R: Read + Seek>(f: &mut R, at: u64) -> Result<u64, io::Error> {
    use byteorder::{LittleEndian, ReadBytesExt};
    f.seek(io::SeekFrom::Start(at))?;
    f.read_u64::<LittleEndian>()
}
