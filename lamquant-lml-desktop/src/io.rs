//! Host `Read`/`Write` adapters around the buffer-oriented MCU codec.

use std::io::{Read, Write};

use lamquant_lml_mcu::error::{LmlError, LmlResult};
use lamquant_lml_mcu::lml;
use lamquant_lml_mcu::lpc::LpcMode;

/// Compress a signal and write the complete LML packet to a host sink.
pub fn compress_into<W: Write>(
    signal: &[Vec<i64>],
    noise_bits: u8,
    mode: LpcMode,
    sink: &mut W,
) -> LmlResult<usize> {
    let bytes = lml::compress_with_mode(signal, noise_bits, mode)?;
    sink.write_all(&bytes).map_err(LmlError::Io)?;
    Ok(bytes.len())
}

/// Read one complete LML packet from a host source and decompress it.
pub fn decompress_from<R: Read>(source: &mut R) -> LmlResult<Vec<Vec<i64>>> {
    let mut bytes = Vec::new();
    source.read_to_end(&mut bytes).map_err(LmlError::Io)?;
    lml::decompress(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_io_roundtrip_handles_partial_reads() {
        struct ByteAtATime<'a>(&'a [u8]);

        impl Read for ByteAtATime<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.0.is_empty() || buffer.is_empty() {
                    return Ok(0);
                }
                buffer[0] = self.0[0];
                self.0 = &self.0[1..];
                Ok(1)
            }
        }

        let signal = vec![vec![42; 128]];
        let mut packet = Vec::new();
        let written = compress_into(&signal, 0, LpcMode::default(), &mut packet).unwrap();
        assert_eq!(written, packet.len());
        let mut source = ByteAtATime(&packet);
        assert_eq!(decompress_from(&mut source).unwrap(), signal);
    }
}
