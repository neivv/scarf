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
mod disasm_cache;
pub mod exec_state;
pub mod exec_state_x86;
pub mod exec_state_x86_64;
mod heapsort;
mod light_byteorder;
pub mod operand;
mod u64_hash;

pub use crate::analysis::{Analyzer, FuncAnalysis};
pub use crate::disasm::{DestOperand, Operation, FlagArith, FlagUpdate, operation_helpers};
pub use crate::operand::{
    ArithOpType, MemAccess, MemAccessSize, Operand, OperandType, OperandContext, OperandCtx,
};
pub use crate::exec_state::ExecutionState;

pub use crate::exec_state_x86::ExecutionState as ExecutionStateX86;
pub use crate::exec_state_x86_64::ExecutionState as ExecutionStateX86_64;

use std::convert::{TryFrom, TryInto};
use std::ffi::{OsStr};
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
        InvalidPeFile(detail: &'static str) {
            display("Invalid PE file ({})", detail)
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

/// BinaryFile, but caches last section from which data was read from,
/// allowing faster repeated small reads when the addresses are expected,
/// but not required to be in same section.
pub struct BinaryFileWithCachedSection<'e, Va: exec_state::VirtualAddress> {
    pub file: &'e BinaryFile<Va>,
    pub last_section: &'e BinarySection<Va>,
}

impl<Va: exec_state::VirtualAddress> BinaryFile<Va> {
    #[inline]
    pub fn base(&self) -> Va {
        self.base
    }

    /// Returns `(address - self.base()) as u32`.
    ///
    /// Panics on overflow when overflow checks are enabled.
    /// If the address is known to be in one of the sections,
    /// at least on Windows any PE file will have offset that should fit in u32
    /// even if truncated.
    pub fn rva_32(&self, address: Va) -> u32 {
        (address.as_u64() as u32) - (self.base.as_u64() as u32)
    }

    /// Returns `(address - self.base()) as u32`, or `None` if `address`
    /// is less than `self.base()` or the difference does not fit in `u32`.
    pub fn try_rva_32(&self, address: Va) -> Option<u32> {
        u32::try_from((address.as_u64()).checked_sub(self.base.as_u64())?).ok()
    }

    /// Returns `(address - self.base()) as u64`.
    ///
    /// Panics on overflow when overflow checks are enabled.
    pub fn rva_64(&self, address: Va) -> u64 {
        address.as_u64() - self.base.as_u64()
    }

    /// Returns `(address - self.base()) as u64`, or `None` if `address`
    /// is less than `self.base()`.
    pub fn try_rva_64(&self, address: Va) -> Option<u64> {
        address.as_u64().checked_sub(self.base.as_u64())
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

impl<Va: exec_state::VirtualAddress> BinarySection<Va> {
    /// Returns true if `address` is within this section.
    #[inline]
    pub fn contains(&self, address: Va) -> bool {
        address >= self.virtual_address && address < self.end()
    }

    /// Returns end address (First byte not included) of this section.
    #[inline]
    pub fn end(&self) -> Va {
        self.virtual_address + self.virtual_size
    }

    /// Receives a slice of data from address with length of `length, or None,
    /// if the slice is not fully contained in the section.
    pub fn slice_from_address(&self, address: Va, length: u32) -> Option<&[u8]> {
        let offset = usize::try_from(
            address.as_u64().wrapping_sub(self.virtual_address.as_u64())
        ).ok()?;
        self.data.get(offset..)?.get(..(length as usize))
    }

    /// Receives a slice of data from address to end of section, or None,
    /// if the address is not in the section
    pub fn slice_from_address_to_end(&self, address: Va) -> Option<&[u8]> {
        let offset = usize::try_from(
            address.as_u64().wrapping_sub(self.virtual_address.as_u64())
        ).ok()?;
        self.data.get(offset..)
    }

    /// Reads an u8 from `addr` if the address is within this section.
    pub fn read_u8(&self, addr: Va) -> Option<u8> {
        use crate::light_byteorder::ReadLittleEndian;
        self.slice_from_address_to_end(addr)?.read_u8().ok()
    }

    /// Reads an little-endian u16 from `addr` if the address is within this section.
    pub fn read_u16(&self, addr: Va) -> Option<u16> {
        use crate::light_byteorder::ReadLittleEndian;
        self.slice_from_address_to_end(addr)?.read_u16().ok()
    }

    /// Reads an little-endian u32 from `addr` if the address is within this section.
    pub fn read_u32(&self, addr: Va) -> Option<u32> {
        use crate::light_byteorder::ReadLittleEndian;
        self.slice_from_address_to_end(addr)?.read_u32().ok()
    }

    /// Reads an little-endian u32 from `addr` if the address is within this section.
    pub fn read_u64(&self, addr: Va) -> Option<u64> {
        use crate::light_byteorder::ReadLittleEndian;
        self.slice_from_address_to_end(addr)?.read_u64().ok()
    }

    /// Reads a little-endian `VirtualAddress`
    /// (Same size as the addresses of this `BinarySection` are)
    /// from `addr` if the address is within this section.
    #[inline]
    pub fn read_address(&self, addr: Va) -> Option<Va> {
        match Va::SIZE {
            4 => self.read_u32(addr).map(|x| Va::from_u64(x as u64)),
            8 => self.read_u64(addr).map(|x| Va::from_u64(x)),
            x => panic!("Unsupported VirtualAddress size {}", x),
        }
    }
}

impl<'e, Va: exec_state::VirtualAddress> BinaryFileWithCachedSection<'e, Va> {
    /// Panics if given BinaryFile with no sections.
    /// (Would be nice to have it just start referencing a static
    /// dummy section, but it's bit pain to do with generics)
    pub fn new(file: &'e BinaryFile<Va>) -> Self {
        Self::try_new(file).expect("Empty BinaryFile")
    }

    pub fn try_new(file: &'e BinaryFile<Va>) -> Option<Self> {
        Some(Self {
            file,
            last_section: file.sections().next()?,
        })
    }

    #[cold]
    fn update_last_section(&mut self, address: Va) -> Option<()> {
        let new_section = self.file.section_by_addr(address)?;
        if std::ptr::eq(new_section, self.last_section) {
            None
        } else {
            self.last_section = new_section;
            Some(())
        }
    }

    pub fn read_u8(&mut self, address: Va) -> Option<u8> {
        loop {
            match self.last_section.read_u8(address) {
                Some(s) => return Some(s),
                None => self.update_last_section(address)?,
            }
        }
    }

    pub fn read_u16(&mut self, address: Va) -> Option<u16> {
        loop {
            match self.last_section.read_u16(address) {
                Some(s) => return Some(s),
                None => self.update_last_section(address)?,
            }
        }
    }

    pub fn read_u32(&mut self, address: Va) -> Option<u32> {
        loop {
            match self.last_section.read_u32(address) {
                Some(s) => return Some(s),
                None => self.update_last_section(address)?,
            }
        }
    }

    pub fn read_u64(&mut self, address: Va) -> Option<u64> {
        loop {
            match self.last_section.read_u64(address) {
                Some(s) => return Some(s),
                None => self.update_last_section(address)?,
            }
        }
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
    use byteorder::{ByteOrder, LittleEndian};
    use crate::Error::*;
    let mut file = BufReader::new(File::open(filename)?);
    let mut buffer = [0u8; 0x58];
    let mut seek_pos = 0u64;
    file.read_exact(&mut buffer[..0x40])?;
    seek_pos += 0x40;
    if LittleEndian::read_u16(&buffer[0..]) != 0x5a4d {
        return Err(InvalidPeFile("Missing DOS magic"));
    }
    let pe_offset = LittleEndian::read_u32(&buffer[0x3c..]) as u64;
    file.seek_relative((pe_offset as u64).wrapping_sub(seek_pos) as i64)?;
    seek_pos = pe_offset as u64;
    file.read_exact(&mut buffer)?;
    seek_pos = seek_pos.wrapping_add(buffer.len() as u64);

    if LittleEndian::read_u32(&buffer[0..]) != 0x0000_4550 {
        return Err(InvalidPeFile("Missing PE magic"));
    }

    let section_count = LittleEndian::read_u16(&buffer[0x6..]);
    let base = VirtualAddress(LittleEndian::read_u32(&buffer[0x34..]));
    let header_block_size = LittleEndian::read_u32(&buffer[0x54..]);

    let mut sections = Vec::with_capacity(section_count as usize + 1);
    // Read header
    file.seek_relative(0u64.wrapping_sub(seek_pos) as i64)?;
    let mut header_data = Vec::with_capacity(header_block_size as usize);
    (&mut file).take(header_block_size as u64).read_to_end(&mut header_data)?;

    let section_headers = match header_data.get(((pe_offset + 0xf8) as usize)..)
        .and_then(|x| x.get(..(0x28 * section_count as usize)))
    {
        Some(s) => s,
        None => return Err(InvalidPeFile("Sections not in header")),
    };

    // Read sections without extra buffering.
    // This may not even be worth it as BufReader is smart enough to not
    // use buffer for reads larger than the buffer, and small sections
    // could be read with single syscall if using BufReader.
    let mut file = file.into_inner();
    // No guarantee of inner seek pos atm, set to invalid
    seek_pos = u64::MAX;
    for section in section_headers.chunks_exact(0x28) {
        let virtual_size = LittleEndian::read_u32(&section[0x8..]);
        let rva = LittleEndian::read_u32(&section[0xc..]);
        let phys_size = LittleEndian::read_u32(&section[0x10..]);
        let phys = LittleEndian::read_u32(&section[0x14..]);
        if phys as u64 != seek_pos {
            file.seek(io::SeekFrom::Start(phys as u64))?;
        }
        let mut data = Vec::with_capacity(phys_size as usize);
        (&mut file).take(phys_size as u64).read_to_end(&mut data)?;
        seek_pos = phys as u64 + phys_size as u64;
        sections.push(BinarySection {
            name: section[..8].try_into().unwrap(),
            virtual_address: base + rva,
            virtual_size,
            data,
        })
    }

    sections.push(BinarySection {
        name: *b"(header)",
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
    use byteorder::{ByteOrder, LittleEndian};
    use crate::Error::*;
    let mut file = BufReader::new(File::open(filename)?);
    let mut buffer = [0u8; 0x58];
    let mut seek_pos = 0u64;
    file.read_exact(&mut buffer[..0x40])?;
    seek_pos += 0x40;
    if LittleEndian::read_u16(&buffer[0..]) != 0x5a4d {
        return Err(InvalidPeFile("Missing DOS magic"));
    }
    let pe_offset = LittleEndian::read_u32(&buffer[0x3c..]);
    file.seek_relative((pe_offset as u64).wrapping_sub(seek_pos) as i64)?;
    seek_pos = pe_offset as u64;
    file.read_exact(&mut buffer)?;
    seek_pos = seek_pos.wrapping_add(buffer.len() as u64);

    if LittleEndian::read_u32(&buffer[0..]) != 0x0000_4550 {
        return Err(InvalidPeFile("Missing PE magic"));
    }
    let section_count = LittleEndian::read_u16(&buffer[0x6..]);
    let base = VirtualAddress64(LittleEndian::read_u64(&buffer[0x30..]));
    let header_block_size = LittleEndian::read_u32(&buffer[0x54..]);

    let mut sections = Vec::with_capacity(section_count as usize + 1);
    // Read header
    file.seek_relative(0u64.wrapping_sub(seek_pos) as i64)?;
    let mut header_data = Vec::with_capacity(header_block_size as usize);
    (&mut file).take(header_block_size as u64).read_to_end(&mut header_data)?;

    let section_headers = match header_data.get(((pe_offset + 0x108) as usize)..)
        .and_then(|x| x.get(..(0x28 * section_count as usize)))
    {
        Some(s) => s,
        None => return Err(InvalidPeFile("Sections not in header")),
    };

    // Read sections without extra buffering.
    // This may not even be worth it as BufReader is smart enough to not
    // use buffer for reads larger than the buffer, and small sections
    // could be read with single syscall if using BufReader.
    let mut file = file.into_inner();
    // No guarantee of inner seek pos atm, set to invalid
    seek_pos = u64::MAX;
    for section in section_headers.chunks_exact(0x28) {
        let virtual_size = LittleEndian::read_u32(&section[0x8..]);
        let rva = LittleEndian::read_u32(&section[0xc..]);
        let phys_size = LittleEndian::read_u32(&section[0x10..]);
        let phys = LittleEndian::read_u32(&section[0x14..]);
        if phys as u64 != seek_pos {
            file.seek(io::SeekFrom::Start(phys as u64))?;
        }
        let mut data = Vec::with_capacity(phys_size as usize);
        (&mut file).take(phys_size as u64).read_to_end(&mut data)?;
        seek_pos = phys as u64 + phys_size as u64;
        sections.push(BinarySection {
            name: section[..8].try_into().unwrap(),
            virtual_address: base + rva,
            virtual_size,
            data,
        })
    }

    sections.push(BinarySection {
        name: *b"(header)",
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
