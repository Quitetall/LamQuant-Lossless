//! Raw USB CDC output — streams 24-bit ADC samples over USB serial.
//!
//! Used when a wired connection is available from the Pico to the base
//! station. Non-blocking: DMA queues data to the USB endpoint (~2 ms),
//! does not contend with the codec pipeline.
//!
//! Wire format per window:
//!   Header (8 bytes):  'L' 'A' 'M' 'R' + chan_count + reserved + win_id_hi + win_id_lo
//!   Payload:           3 bytes per sample (24-bit packed, big-endian),
//!                      `chan_count × 2500` samples per window.
//!
//! Phase 4: protocol logic only. Phase 5 wires the actual USB CDC writer
//! (rp235x-hal exposes a CDC ACM device via usb-device + usbd-serial).

pub const RAW_SYNC: [u8; 4] = *b"LAMR";
pub const SAMPLES_PER_WINDOW: usize = 2500;

/// Build the 8-byte raw-stream header.
pub fn raw_header(window_id: u32, total_channels: u8) -> [u8; 8] {
    [
        RAW_SYNC[0],
        RAW_SYNC[1],
        RAW_SYNC[2],
        RAW_SYNC[3],
        total_channels,
        0, // reserved
        ((window_id >> 8) & 0xFF) as u8,
        (window_id & 0xFF) as u8,
    ]
}

/// Pack one 24-bit Q31 sample big-endian. Caller does the streaming.
#[inline]
pub fn pack_sample(value: i32, out: &mut [u8; 3]) {
    out[0] = ((value >> 16) & 0xFF) as u8;
    out[1] = ((value >> 8) & 0xFF) as u8;
    out[2] = (value & 0xFF) as u8;
}

/// Pure-data version: emit header + every sample of every channel into
/// a writer (e.g. a USB CDC sink). The writer just needs to accept bytes
/// and drop on full — backpressure is handled by USB OUT buffer.
///
/// This is host-testable. The real device path uses a `embedded_io_async::Write`
/// impl wrapping the rp235x-hal USB CDC class; that wiring is Phase 5.
pub fn stream_window<W: ByteSink>(
    writer: &mut W,
    window_id: u32,
    raw_adc_buffer: &[[i32; SAMPLES_PER_WINDOW]],
    total_channels: u8,
) {
    let header = raw_header(window_id, total_channels);
    writer.write_bytes(&header);

    let mut sample_buf = [0u8; 3];
    for ch in 0..total_channels as usize {
        if ch >= raw_adc_buffer.len() {
            break;
        }
        for s in 0..SAMPLES_PER_WINDOW {
            pack_sample(raw_adc_buffer[ch][s], &mut sample_buf);
            writer.write_bytes(&sample_buf);
        }
    }
}

/// Minimal byte sink. Callers implement this for their USB CDC class.
pub trait ByteSink {
    fn write_bytes(&mut self, bytes: &[u8]);
}

#[cfg(all(test, feature = "host-verify"))]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct VecSink(Vec<u8>);
    impl ByteSink for VecSink {
        fn write_bytes(&mut self, bytes: &[u8]) {
            self.0.extend_from_slice(bytes);
        }
    }

    #[test]
    fn header_layout() {
        let h = raw_header(0x12_3456, 21);
        assert_eq!(&h[..4], b"LAMR");
        assert_eq!(h[4], 21);
        assert_eq!(h[5], 0);
        assert_eq!(h[6], 0x34); // (0x123456 >> 8) & 0xFF
        assert_eq!(h[7], 0x56);
    }

    #[test]
    fn pack_sample_be() {
        let mut buf = [0u8; 3];
        pack_sample(0x12_3456, &mut buf);
        assert_eq!(buf, [0x12, 0x34, 0x56]);
    }

    #[test]
    fn stream_window_emits_header_plus_samples() {
        let mut buf = [[0i32; SAMPLES_PER_WINDOW]; 2];
        for s in 0..SAMPLES_PER_WINDOW {
            buf[0][s] = s as i32;
            buf[1][s] = -(s as i32);
        }
        let mut sink = VecSink(Vec::new());
        stream_window(&mut sink, 7, &buf, 2);
        // 8-byte header + 2 channels × 2500 × 3 bytes = 15008
        assert_eq!(sink.0.len(), 8 + 2 * SAMPLES_PER_WINDOW * 3);
        assert_eq!(&sink.0[..4], b"LAMR");
    }
}
