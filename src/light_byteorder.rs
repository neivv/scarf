//! More or less same as `byteorder`, but returns empty struct as an error,
//! instead of heavier io::Error.
//! Only implemented for slices since other readers may have non-eof
//! I/O errors, so blindly discarding them isn't the best idea.

use std::convert::TryInto;

#[derive(Copy, Clone, Debug)]
pub struct UnexpectedEof;

pub trait ReadLittleEndian {
    fn read_u8(&mut self) -> Result<u8, UnexpectedEof>;
    fn read_u16(&mut self) -> Result<u16, UnexpectedEof>;
    fn read_u32(&mut self) -> Result<u32, UnexpectedEof>;
    fn read_u64(&mut self) -> Result<u64, UnexpectedEof>;
}

impl ReadLittleEndian for &[u8] {
    #[inline]
    fn read_u8(&mut self) -> Result<u8, UnexpectedEof> {
        if self.len() < 1 {
            Err(UnexpectedEof)
        } else {
            let result = self[0];
            *self = &self[1..];
            Ok(result)
        }
    }

    #[inline]
    fn read_u16(&mut self) -> Result<u16, UnexpectedEof> {
        if self.len() < 2 {
            Err(UnexpectedEof)
        } else {
            let result = u16::from_le_bytes((&self[..2]).try_into().unwrap());
            *self = &self[2..];
            Ok(result)
        }
    }

    #[inline]
    fn read_u32(&mut self) -> Result<u32, UnexpectedEof> {
        if self.len() < 4 {
            Err(UnexpectedEof)
        } else {
            let result = u32::from_le_bytes((&self[..4]).try_into().unwrap());
            *self = &self[4..];
            Ok(result)
        }
    }

    #[inline]
    fn read_u64(&mut self) -> Result<u64, UnexpectedEof> {
        if self.len() < 8 {
            Err(UnexpectedEof)
        } else {
            let result = u64::from_le_bytes((&self[..8]).try_into().unwrap());
            *self = &self[8..];
            Ok(result)
        }
    }
}
