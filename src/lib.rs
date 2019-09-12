extern crate byteorder;
extern crate hex_slice;
extern crate lde;
#[macro_use] extern crate log;
extern crate ordermap;
#[macro_use] extern crate quick_error;
extern crate serde;
#[macro_use] extern crate serde_derive;

pub mod analysis;
mod bit_misc;
mod disasm;
pub mod exec_state;
pub mod operand;
mod vec_drop_iter;

pub use disasm::{ArithmeticOp, Operation, operation_helpers};
pub use exec_state::{ExecutionState};
pub use operand::{Operand, OperandType, operand_helpers};

use std::ffi::{OsString, OsStr};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use byteorder::{LittleEndian, ReadBytesExt};

use operand::MemAccessSize;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Rva(pub u32);

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct VirtualAddress(pub u32);

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
    }
}

pub struct BinaryFile {
    // For "hardcoded" checks that check that return address matches module code section
    pub dump_code_offset: VirtualAddress,
    sections: Vec<BinarySection>,
}

pub struct BinarySection {
    pub name: [u8; 8],
    pub virtual_address: VirtualAddress,
    pub data: Vec<u8>,
}

impl BinaryFile {
    /// Panics if code section for some reason would not exist
    pub fn code_section(&self) -> &BinarySection {
        self.section(b".text\0\0\0").unwrap()
    }

    pub fn section(&self, name: &[u8; 0x8]) -> Option<&BinarySection> {
        self.sections.iter().find(|x| &x.name[..] == &name[..])
    }
}

/// Allows loading a BinaryFile from memory buffer(s) representing the binary sections.
pub fn raw_bin(sections: Vec<BinarySection>) -> BinaryFile {
    BinaryFile {
        dump_code_offset: VirtualAddress(0x401000),
        sections: sections,
    }
}

pub fn parse(filename: &OsStr, silps_path: &OsStr) -> Result<BinaryFile, Error> {
    use Error::*;
    let mut file = BufReader::new(File::open(filename)?);
    if file.read_u16::<LittleEndian>()? != 0x5a4d {
        return Err(InvalidPeFile("Missing DOS magic".into()));
    }
    let pe_offset = read_at_32(&mut file, 0x3c)? as u64;
    if read_at_32(&mut file, pe_offset)? != 0x00004550 {
        return Err(InvalidPeFile("Missing PE magic".into()));
    }
    let section_count = read_at_16(&mut file, pe_offset + 6)?;
    let base = VirtualAddress(read_at_32(&mut file, pe_offset + 0x34)?);
    let sections = (0..section_count).map(|i| {
        let mut name = [0; 8];
        file.seek(io::SeekFrom::Start(pe_offset + 0xf8 + 0x28 * i as u64))?;
        file.read_exact(&mut name)?;
        file.seek(io::SeekFrom::Start(pe_offset + 0xf8 + 0x28 * i as u64 + 0xc))?;
        let rva = Rva(file.read_u32::<LittleEndian>()?);
        let phys_size = file.read_u32::<LittleEndian>()?;
        let phys = PhysicalAddress(file.read_u32::<LittleEndian>()?);

        file.seek(io::SeekFrom::Start(phys.0 as u64))?;
        let mut data = vec![0; phys_size as usize];
        file.read_exact(&mut data)?;
        Ok(BinarySection {
            name: name,
            virtual_address: base + rva,
            data,
        })
    }).collect::<Result<Vec<_>, Error>>()?;
    let dump_code_offset;
    let code_offset;
    {
        let code = sections.iter().find(|s| &s.name[..] == b".text\0\0\0")
            .ok_or_else(|| InvalidPeFile("No .text found".into()))?;

        code_offset = code.virtual_address;
        let silps_module_bases = Path::new(silps_path).join("module_bases.txt");
        dump_code_offset = silps_dump_base(filename, &silps_module_bases)
            .map(|x| x + (code_offset - base));
    }
    Ok(BinaryFile {
        dump_code_offset: dump_code_offset.unwrap_or(code_offset),
        sections,
    })
}

fn silps_dump_base(
    filename: &OsStr,
    module_base_file: &Path
) -> Option<VirtualAddress> {
    use std::ascii::AsciiExt;
    let fun = |filename: &OsStr, module_base_file: &Path| -> Result<_, io::Error> {
        let file = BufReader::new(File::open(module_base_file)?);
        for line in file.lines() {
            let line = line?;
            let mut tokens = line.split_whitespace();
            let ok = tokens.next().and_then(|x| {
                filename.to_str().map(|f| (f, x))
            }).map(|(filename, binary_name)| {
                let filename = filename.to_ascii_lowercase();
                let binary_name = &binary_name[..binary_name.len() - 1].to_ascii_lowercase();
                filename.contains(binary_name)
            }).unwrap_or(false);
            if ok {
                return Ok(
                    tokens.next().and_then(|x| u32::from_str_radix(x, 16).ok())
                        .map(|x| VirtualAddress(x))
                );
            }
        }
        Ok(None)
    };
    match fun(filename, module_base_file) {
        Ok(o) => o,
        Err(e) => {
            error!("Couldn't read silps module base file: {}", e);
            None
        }
    }
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
    pub fn resolve_mem_accesses(&self, val: Rc<Operand>) -> (Rc<Operand>, Vec<VirtualAddress>) {
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
                        let (l_resolved, l_addresses) = self.resolve_mem_accesses(l.clone());
                        let (r_resolved, r_addresses) = self.resolve_mem_accesses(r.clone());
                        *l = l_resolved;
                        *r = r_resolved;
                        addresses.extend(l_addresses);
                        addresses.extend(r_addresses);
                    }
                    Not(ref mut l) | LogicalNot(ref mut l) | Parity(ref mut l) => {
                        let (l_resolved, l_addresses) = self.resolve_mem_accesses(l.clone());
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
                MemAccessSize::Mem8 => sliced.read_u8().ok().map(|x| x as u32),
                MemAccessSize::Mem16 => sliced.read_u16::<LittleEndian>().ok().map(|x| x as u32),
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
