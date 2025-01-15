//! Scarf is a library for analyzing x86 functions.
//!
//! The main concept of scarf is that the user gives [`FuncAnalysis`] a function
//! which will be simulated. `FuncAnalysis` will walk through the function's different
//! execution paths, while calling [user-defined callbacks](Analyzer), letting the user code
//! to examine execution to extract information it needs. The callback is also able to
//! modify state and have some control over execution, allowing scarf to be adapted for cases
//! where the library is not quite able to handle unusual assmebly patterns.
//!
//! Examples of problems that could be answered with scarf:
//! - Find all child function calls of a function, where one of the arguments is
//!     constant integer between 0x600 and 0x700
//! - If the function writes a 64-bit value to `(Base pointer)+0x28`, return the base pointer
//!     and value which was written to. That is, detect writes to a field of a struct when the
//!     field offset is known to be 0x28.
//! - Check if the function reads memory at given constant address, and track non-stack
//!     locations where the read value is passed to.
//! - Determine all constant arguments that are passed to a certain function `f`, by analyzing all
//!     of the functions calling `f`.
//! - Find a point where the function compares some value `x` to be less than constant 0x100, and
//!     return what expression `x` is, as well as the jump address and whether it has to be
//!     changed to always or never to jump in order to always go to `x < 0x100` branch.
//!
//! In general, scarf is still relatively low-level in its execution representation.
//! Good analysis results often require user to handle edge cases, which often requires
//! iterative improvements to analysis code when you come across an executable that the
//! analysis quite does not work on. As ultimately the only input to scarf analysis is often
//! just the executable binary, keeping tests using those binaries to prevent scarf-using code
//! from suddenly regressing is a good idea.
//!
//! Scarf strives to be fast enough to analyze an average function in less than 1 millisecond
//! even on slower machines. This makes it quite feasible to brute force every function of
//! even in larger executable in few minutes, as well as have more targeted analysis be fast
//! enough that it can be ran without anyone noticing. Some of this speed means giving up
//! accuracy, and if an adversary codegen wanted to explicitly break scarf, it would likely
//! at least require user callback to actively help scarf from breaking.
//! The simulation accuracy issues do not seem to cause too much problem in regular
//! compiler-generated code though.
//!
//! The following are main types used by scarf:
//!
//! - [`FuncAnalysis`] - The entry point for scarf analysis. Walks through and keeps track of
//!     all branches the execution may take.
//! - [`trait Analyzer`](Analyzer) - User-implemented trait that `FuncAnalysis` calls back to,
//!     allowing user code to query and manipulate analysis.
//! - [`analysis::Control`] - A type passed to `Analyzer` callbacks, providing various
//!     ways to query and manipulate analysis state.
//! - [`BinaryFile`] - Contains sections of the binary, including code that is to be simulated.
//!     - [`BinarySection`] - A single section, practically just `Vec<u8>` and a base address.
//! - [`VirtualAddress`] / [`VirtualAddress64`] - Integer newtype representing a constant
//!     address, usually in `BinaryFile`
//!     - [`trait exec_state::VirtualAddress`](exec_state::VirtualAddress) - A trait allowing
//!     handling both address sizes generically.
//! - [`Operand`] - The main value / expression type of scarf.
//! - [`OperandContext`] - Allocation arena and interner for `Operand`s. Has to outlive
//!     `FuncAnalysis`, so user code is required to create this and pass to rest of scarf.
//! - [`trait ExecutionState`](exec_state::ExecutionState) - Holds all of the simulated
//!     CPU and memory state of one point in analysis's execution.
//!     Concrete types are [`ExecutionStateX86`] and [`ExecutionStateX86_64`].

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

pub use crate::analysis::{Analyzer, FuncAnalysis};
pub use crate::disasm::{DestOperand, Operation, FlagArith, FlagUpdate, operation_helpers};
pub use crate::operand::{
    ArithOpType, MemAccess, MemAccessSize, Operand, OperandType, OperandContext, OperandCtx,
};
pub use crate::exec_state::ExecutionState;

pub use crate::exec_state_x86::ExecutionState as ExecutionStateX86;
pub use crate::exec_state_x86_64::ExecutionState as ExecutionStateX86_64;

use std::ffi::{OsString, OsStr};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};

use quick_error::quick_error;

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

/// Represents relative virtual address.
///
/// Not really used by scarf. User code working with 64-bit binaries may use this
/// to save a bit of memory storing single 'base' `VirtualAddress64`, and other
/// addresses `Rva`s, calculating the actual address when needed as `base + rva.0`.
///
/// That probably never wnds up being relevant.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rva(pub u32);

/// Represents relative virtual address.
///
/// Not really used by scarf. Was only defined to have some sort of sensible
/// result for `VirtualAddress64 - VirtualAddress64` operation, which probably
/// should just be removed.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rva64(pub u64);

impl std::fmt::Debug for Rva {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Rva({:08x})", self.0)
    }
}

impl std::fmt::Debug for Rva64 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Rva64({:08x})", self.0)
    }
}

/// `VirtualAddress` represents a constant 32-bit memory address.
///
/// See also [`exec_state::VirtualAddress`] trait for the trait
/// is used by generic code to operate on either this type or
/// 64-bit [`VirtualAddress64`]. Due to legacy naming decisions, this type
/// confusingly has same name as the trait, instead of `VirtualAddress32`.
///
/// Most of scarf does not require memory addresses be a integer constant,
/// using [`Operand`s](Operand) instead. `VirtualAddress` is mainly used when
/// reading from [`BinaryFile`], and when creating new [`analysis::FuncAnalysis`].
///
/// The only notable difference from a plain `u32` is that
/// addition and subtraction(*) are defined to wrap on overflow.
///
/// (*) Subtracting `VirtualAddress - VirtualAddress` does panic on overflow,
/// Subtracting `VirtualAddress - u32` wraps.
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualAddress(pub u32);

/// `VirtualAddress64` represents a constant 64-bit memory address.
///
/// See also [`exec_state::VirtualAddress`] trait for the trait
/// is used by generic code to operate on either this type or
/// 32-bit [`VirtualAddress`].
///
/// Most of scarf does not require memory addresses be a integer constant,
/// using [`Operands`](Operand) instead. `VirtualAddress` is mainly used when
/// reading from [`BinaryFile`], and when creating new [`analysis::FuncAnalysis`].
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
        write!(f, "VirtualAddress({:08x})", self.0)
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
    type Output = Rva64;
    #[inline]
    fn sub(self, rhs: VirtualAddress64) -> Rva64 {
        Rva64(self.0 - rhs.0)
    }
}

impl std::ops::Add<u32> for Rva {
    type Output = Rva;
    #[inline]
    fn add(self, rhs: u32) -> Rva {
        Rva(self.0 + rhs)
    }
}

impl std::ops::Add<u32> for Rva64 {
    type Output = Rva64;
    #[inline]
    fn add(self, rhs: u32) -> Rva64 {
        Rva64(self.0 + rhs as u64)
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

/// Zero-sized error type that is returned when reading from BinaryFile by VirtualAddress cannot
/// be done.
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

/// Contains the binary that is to be analyzed, loaded to memory.
///
/// Not much more than a Vec<[BinarySection]>, and functions to read
/// the correct section in different ways, when given
/// [`VirtualAddress`](exec_state::VirtualAddress).
///
/// `BinaryFile` is often passed as by reference `&'e BinaryFile`, using same lifetime as
/// [`OperandContext`]. This would not be required, as `BinaryFile` does not hold any references
/// to `OperandContext`, the lifetime was chosen to be same to keep all code interfacing
/// with scarf simpler, only requiring a single lifetime being passed around. This unfortunately
/// does require `BinaryFile` to be created so that it outlives the `OperandContext` passed
/// to [`FuncAnalysis`], which usually should not be a too big of a problem.
///
/// [`parse`] and [`parse_x86_64`] can be used to load a `BinaryFile` from Windows
/// PE executable, [`raw_bin`] can be used to create a `BinaryFile` from arbitrary
/// sections.
#[derive(Debug, Clone)]
pub struct BinaryFile<Va: exec_state::VirtualAddress> {
    pub base: Va,
    sections: Vec<BinarySection<Va>>,
    relocs: Vec<Va>,
}

/// Single section of a [`BinaryFile`].
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

    /// Returns the section named ".text".
    ///
    /// Panics if this section for some reason does not exist.
    ///
    /// `[code_sections]` should be preferred if the user code wants to handle possibility
    /// of multiple executable sections... But currently its implementation is bad and
    /// just returns section named ".text", if it exists...
    pub fn code_section(&self) -> &BinarySection<Va> {
        self.section(b".text\0\0\0").unwrap()
    }

    /// Returns iterator, which yields all `[BinarySection]`s of this `BinaryFile`.
    pub fn sections<'a>(&'a self) -> impl Iterator<Item = &'a BinarySection<Va>> + 'a {
        self.sections.iter()
    }

    pub fn code_sections<'a>(&'a self) -> impl Iterator<Item = &'a BinarySection<Va>> + 'a {
        self.sections.iter().filter(|x| &x.name[..] == &b".text\0\0\0"[..])
    }

    /// Returns section by name, if it exists.
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

    /// Receives a slice of data from address with length `len`, or Err(OutOfBounds)
    /// if there is no single section containing the entire slice.
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

    /// Receives a slice of data from address to end of the respective section.
    pub fn slice_from_address_to_end(&self, start: Va) -> Result<&[u8], OutOfBounds> {
        self.section_by_addr(start)
            .and_then(|s| {
                let section_relative = start.as_u64() - s.virtual_address.as_u64();
                s.data.get((section_relative as usize)..)
            })
            .ok_or_else(|| OutOfBounds)
    }

    /// Searches for section containing `addr`, and reads a little-endian u8 from there.
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

    /// Searches for section containing `addr`, and reads a little-endian u16 from there.
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

    /// Searches for section containing `addr`, and reads a little-endian u32 from there.
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

    /// Searches for section containing `addr`, and reads a little-endian u64 from there.
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

    /// Searches for section containing `addr`, and reads a `VirtualAddress`
    /// (Same size as the addresses of this `BinaryFile` are) from there.
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

/// Creates a BinaryFile from memory buffer(s) representing the binary sections.
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

/// Creates a `BinaryFile` from 32-bit Windows executable at `filename`.
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

/// Creates a `BinaryFile` from 64-bit Windows executable at `filename`.
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
