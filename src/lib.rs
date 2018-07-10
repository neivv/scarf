#![cfg_attr(feature = "cargo-clippy", allow(
    match_bool, trivially_copy_pass_by_ref, new_without_default, unneeded_field_pattern,
    redundant_closure, new_without_default_derive, len_without_is_empty, many_single_char_names,
    collapsible_if, verbose_bit_mask, wrong_self_convention, type_complexity, bool_comparison,
    assign_op_pattern, question_mark, should_implement_trait,
))]

extern crate byteorder;
extern crate fxhash;
extern crate hex_slice;
extern crate lde;
#[macro_use] extern crate log;
#[macro_use] extern crate quick_error;
extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate smallvec;

pub mod analysis;
mod bit_misc;
pub mod cfg;
pub mod cfg_dot;
mod disasm;
pub mod exec_state;
pub mod operand;
mod vec_drop_iter;

pub use disasm::{DestOperand, Operation, operation_helpers};
pub use exec_state::{ExecutionState};
pub use operand::{Operand, OperandType, operand_helpers};

use std::ffi::{OsString, OsStr};
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt};

use operand::MemAccessSize;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rva(pub u32);

impl std::fmt::Debug for Rva {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Rva({:08x})", self.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualAddress(pub u32);

impl std::fmt::Debug for VirtualAddress {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "VirtualAddress({:08x})", self.0)
    }
}

impl std::ops::Add<Rva> for VirtualAddress {
    type Output = VirtualAddress;
    fn add(self, rhs: Rva) -> VirtualAddress {
        VirtualAddress(self.0 + rhs.0)
    }
}

impl std::ops::Add<u32> for VirtualAddress {
    type Output = VirtualAddress;
    fn add(self, rhs: u32) -> VirtualAddress {
        VirtualAddress(self.0.wrapping_add(rhs))
    }
}

impl std::ops::Sub<u32> for VirtualAddress {
    type Output = VirtualAddress;
    fn sub(self, rhs: u32) -> VirtualAddress {
        VirtualAddress(self.0.wrapping_sub(rhs))
    }
}

impl std::ops::Sub<VirtualAddress> for VirtualAddress {
    type Output = Rva;
    fn sub(self, rhs: VirtualAddress) -> Rva {
        Rva(self.0 - rhs.0)
    }
}

impl std::ops::Add<u32> for Rva {
    type Output = Rva;
    fn add(self, rhs: u32) -> Rva {
        Rva(self.0 + rhs)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PhysicalAddress(pub u32);

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Io(e: io::Error) {
            display("I/O error {}", e)
            from()
        }
        IoWithFilename(e: io::Error, path: PathBuf) {
            display("I/O error {}, file {:?}", e, path)
        }
        InvalidPeFile(detail: String) {
            description("Invalid PE file")
            display("Invalid PE file ({})", detail)
        }
        InvalidFilename(filename: OsString) {
            description("Invalid filename")
            display("Invalid filename {:?}", filename)
        }
        OutOfBounds {
            display("Out of bounds")
        }
    }
}

#[derive(Debug, Clone)]
pub struct BinaryFile {
    pub base: VirtualAddress,
    sections: Vec<BinarySection>,
}

#[derive(Debug, Clone)]
pub struct BinarySection {
    pub name: [u8; 8],
    pub virtual_address: VirtualAddress,
    pub physical_address: PhysicalAddress,
    pub virtual_size: u32,
    pub data: Vec<u8>,
}

impl BinaryFile {
    /// Panics if code section for some reason would not exist
    pub fn code_section(&self) -> &BinarySection {
        self.section(b".text\0\0\0").unwrap()
    }

    pub fn section(&self, name: &[u8; 0x8]) -> Option<&BinarySection> {
        self.sections.iter().find(|x| x.name[..] == name[..])
    }

    /// Returns a section containing the address, if it exists
    pub fn section_by_addr(&self, address: VirtualAddress) -> Option<&BinarySection> {
        self.sections.iter().find(|x| {
            address >= x.virtual_address && address < (x.virtual_address + x.data.len() as u32)
        })
    }

    /// Range is relative from base
    pub fn slice_from(&self, range: std::ops::Range<u32>) -> Result<&[u8], Error> {
        self.section_by_addr(self.base + range.start)
            .and_then(|s| {
                let section_relative = (self.base + range.start) - s.virtual_address;
                s.data.get(
                    section_relative.0 as usize ..
                    (section_relative + (range.end - range.start)).0 as usize,
                )
            })
            .ok_or_else(|| Error::OutOfBounds)
    }

    pub fn read_u32(&self, addr: VirtualAddress) -> Result<u32, Error> {
        self.section_by_addr(addr)
            .and_then(|s| {
                let section_relative = addr - s.virtual_address;
                s.data.get(section_relative.0 as usize..)
            })
            .and_then(|mut data| {
                data.read_u32::<LittleEndian>().ok()
            })
            .ok_or_else(|| Error::OutOfBounds)
    }

    pub fn to_physical(&self, addr: VirtualAddress) -> Option<PhysicalAddress> {
        self.section_by_addr(addr)
            .map(|s| {
                let section_relative = addr - s.virtual_address;
                PhysicalAddress(s.physical_address.0 + section_relative.0)
            })
    }
}

/// Allows loading a BinaryFile from memory buffer(s) representing the binary sections.
pub fn raw_bin(base: VirtualAddress, sections: Vec<BinarySection>) -> BinaryFile {
    BinaryFile {
        base,
        sections,
    }
}

pub fn parse(filename: &OsStr) -> Result<BinaryFile, Error> {
    use Error::*;
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
        let phys = PhysicalAddress(file.read_u32::<LittleEndian>()?);

        file.seek(io::SeekFrom::Start(u64::from(phys.0)))?;
        let mut data = vec![0; phys_size as usize];
        file.read_exact(&mut data)?;
        Ok(BinarySection {
            name,
            virtual_address: base + rva,
            physical_address: phys,
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
        physical_address: PhysicalAddress(0),
        virtual_size: header_block_size,
        data: header_data,
    });
    Ok(BinaryFile {
        base,
        sections,
    })
}

fn read_at_16<R: Read + Seek>(f: &mut R, at: u64) -> Result<u16, io::Error> {
    f.seek(io::SeekFrom::Start(at))?;
    f.read_u16::<LittleEndian>()
}

fn read_at_32<R: Read + Seek>(f: &mut R, at: u64) -> Result<u32, io::Error> {
    f.seek(io::SeekFrom::Start(at))?;
    f.read_u32::<LittleEndian>()
}

pub struct SectionDumps(Vec<SectionDump>);

struct SectionDump {
    address: VirtualAddress,
    data: Vec<u8>,
}

pub fn load_section_dumps(
    dir: &OsStr,
    binary_name: &OsStr,
    binary: &BinaryFile,
) -> Result<SectionDumps, Error> {
    use self::Error::*;

    let filename_root = Path::new(binary_name).file_stem()
        .ok_or_else(|| InvalidFilename(binary_name.into()))?;
    let sections = binary.sections.iter()
        .take_while(|section| !section.name.starts_with(&b".silps"[..]))
        .enumerate().map(|(i, section)| {
        let mut filename = filename_root.to_os_string();
        let section_name = std::str::from_utf8(&section.name[..])
            .unwrap_or("(no name)")
            .trim_matches(|x: char| !x.is_alphanumeric());
        filename.push(format!("_{}_{}", i, section_name));
        let full_path = Path::new(dir).join(filename);
        let mut file = File::open(&full_path)
            .map_err(|e| IoWithFilename(e, full_path))?;
        let mut buf = vec![];
        file.read_to_end(&mut buf)?;
        Ok(SectionDump {
            address: section.virtual_address,
            data: buf,
        })
    }).collect::<Result<Vec<_>, Error>>()?;
    Ok(SectionDumps(sections))
}

impl SectionDumps {
    pub fn resolve_mem_accesses(&self, val: &Rc<Operand>) -> (Rc<Operand>, Vec<VirtualAddress>) {
        use operand::ArithOpType::*;
        let mut addresses = vec![];
        let val = match val.ty {
            OperandType::Memory(ref mem) => {
                if let OperandType::Constant(c) = mem.address.ty {
                    if let Some(val) = self.resolve_mem_value(VirtualAddress(c), mem.size) {
                        addresses.push(VirtualAddress(c));
                        Operand::new_simplified_rc(OperandType::Constant(val))
                    } else {
                        val.clone()
                    }
                } else {
                    val.clone()
                }
            }
            OperandType::Arithmetic(ref arith) => {
                let mut arith = arith.clone();
                match arith {
                    Add(ref mut l, ref mut r) | Sub(ref mut l, ref mut r) |
                        Mul(ref mut l, ref mut r) | And(ref mut l, ref mut r) |
                        Or(ref mut l, ref mut r) | Xor(ref mut l, ref mut r) |
                        Lsh(ref mut l, ref mut r) | Rsh(ref mut l, ref mut r) |
                        RotateLeft(ref mut l, ref mut r) | Equal(ref mut l, ref mut r) |
                        GreaterThan(ref mut l, ref mut r) | SignedMul(ref mut l, ref mut r) |
                        Div(ref mut l, ref mut r) | Modulo(ref mut l, ref mut r) |
                        GreaterThanSigned(ref mut l, ref mut r) =>
                    {
                        let (l_resolved, l_addresses) = self.resolve_mem_accesses(l);
                        let (r_resolved, r_addresses) = self.resolve_mem_accesses(r);
                        *l = l_resolved;
                        *r = r_resolved;
                        addresses.extend(l_addresses);
                        addresses.extend(r_addresses);
                    }
                    Not(ref mut l) | Parity(ref mut l) => {
                        let (l_resolved, l_addresses) = self.resolve_mem_accesses(l);
                        *l = l_resolved;
                        addresses.extend(l_addresses);
                    }
                }
                Operand::new_not_simplified_rc(OperandType::Arithmetic(arith))
            },
            _ => val.clone(),
        };
        (val, addresses)
    }

    fn resolve_mem_value(&self, address: VirtualAddress, size: MemAccessSize) -> Option<u32>{
        self.0.iter().find(|sect| {
            address > sect.address && address < sect.address + sect.data.len() as u32
        }).and_then(|sect| {
            let offset = (address.0 - sect.address.0) as usize;
            let mut sliced = &sect.data[offset..];
            match size {
                MemAccessSize::Mem8 => sliced.read_u8().ok().map(|x| u32::from(x)),
                MemAccessSize::Mem16 => {
                    sliced.read_u16::<LittleEndian>().ok().map(|x| u32::from(x))
                }
                MemAccessSize::Mem32 => sliced.read_u32::<LittleEndian>().ok(),
            }
        })
    }
}

#[derive(Deserialize, Serialize)]
pub struct ScarfResults {
    pub antidebug_keys: Vec<EncryptRuleSerialize>,
    pub encrypt_key_fn: EncryptKeyResult,
    pub antidebug_addr: u32
}

#[derive(Deserialize, Serialize)]
pub struct EncryptKeyResult {
    pub address: String,
    pub dynamic_key_readable: String,
    pub static_key_readable: String,
    pub dynamic_key: Option<Rc<Operand>>,
    pub static_key: Option<Rc<Operand>>,
}

#[derive(Deserialize, Serialize)]
pub struct EncryptRuleSerialize {
    pub offset: String,
    pub rule_human_readable_high: String,
    pub rule_human_readable_low: String,
    pub rule_high: Option<Rc<Operand>>,
    pub rule_low: Option<Rc<Operand>>,
}
