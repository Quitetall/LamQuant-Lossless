//! ASCII int-per-line detector + parser + re-emitter.
//!
//! Bonn EEG dataset stores each recording as a `.txt` file with one
//! integer per line. This module covers that shape plus the
//! straightforward variations seen in practice:
//!
//!   - LF or CRLF line endings
//!   - Optional leading zeros / fixed field width
//!   - Optional trailing newline
//!   - Whitespace before the integer (Bonn N set has `  -12` style)
//!
//! `detect_ascii_int_lines` reads the first chunk of a byte buffer and
//! returns either `Some(template)` describing the file's format, or
//! `None` if it doesn't look like ASCII ints. `parse_ascii_int_lines`
//! consumes the full buffer + template and returns the `i16` samples.
//! `render_ascii_int_lines` is the inverse — `(samples, template) ->
//! bytes`. The round-trip
//!
//! ```text
//! bytes == render(parse(bytes), detect(bytes).unwrap())
//! ```
//!
//! is the SHA-equivalence contract that makes Bonn `.txt` lossless
//! through the codec.

/// Captures everything needed to re-emit the original ASCII file
/// from the parsed `i16` samples. Stored in the manifest's
/// `synthetic_from.template` field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsciiLinesTemplate {
    /// `"\n"` or `"\r\n"`. The byte sequence between samples.
    pub line_ending: LineEnding,
    /// Bytes of whitespace BEFORE the integer on each line. Bonn's
    /// N set uses 2 spaces before each value; F/O/S/Z use 0.
    pub leading_whitespace: u8,
    /// Optional pad-with-spaces width. 0 = natural width (no padding).
    /// Detected from the first 64 lines; if any line's natural width
    /// exceeds the detected width, the parser refuses (mismatched
    /// format means we'd lose information on re-emit).
    pub field_width: u8,
    /// Whether the file ends with a final newline. Most Bonn files do;
    /// some don't. We preserve whichever was seen.
    pub trailing_newline: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    Lf,
    CrLf,
}

impl LineEnding {
    pub fn as_bytes(self) -> &'static [u8] {
        match self {
            LineEnding::Lf => b"\n",
            LineEnding::CrLf => b"\r\n",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            LineEnding::Lf => "Lf",
            LineEnding::CrLf => "CrLf",
        }
    }
}

impl core::str::FromStr for LineEnding {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "Lf" => Ok(LineEnding::Lf),
            "CrLf" => Ok(LineEnding::CrLf),
            _ => Err(()),
        }
    }
}

impl AsciiLinesTemplate {
    /// Serialise to a JSON object suitable for the LMA manifest's
    /// `synthetic_from.template` slot. Stable shape — older readers
    /// without new fields must still accept the object via opaque
    /// `serde_json::Value` storage and round-trip it on re-pack.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "line_ending": self.line_ending.as_str(),
            "leading_whitespace": self.leading_whitespace,
            "field_width": self.field_width,
            "trailing_newline": self.trailing_newline,
        })
    }

    /// Parse back from JSON. Hard error on any missing or wrong-typed
    /// field — Bible R5, no silent fallback on corrupt template data.
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let line_ending = v
            .get("line_ending")
            .and_then(|x| x.as_str())
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| "ascii_lines template: missing/invalid line_ending".to_string())?;
        let leading_whitespace = v
            .get("leading_whitespace")
            .and_then(|x| x.as_u64())
            .and_then(|n| u8::try_from(n).ok())
            .ok_or_else(|| {
                "ascii_lines template: missing/invalid leading_whitespace".to_string()
            })?;
        let field_width = v
            .get("field_width")
            .and_then(|x| x.as_u64())
            .and_then(|n| u8::try_from(n).ok())
            .ok_or_else(|| "ascii_lines template: missing/invalid field_width".to_string())?;
        let trailing_newline = v
            .get("trailing_newline")
            .and_then(|x| x.as_bool())
            .ok_or_else(|| "ascii_lines template: missing/invalid trailing_newline".to_string())?;
        Ok(AsciiLinesTemplate {
            line_ending,
            leading_whitespace,
            field_width,
            trailing_newline,
        })
    }
}

/// Maximum number of leading characters scanned to fingerprint the
/// format. 4 KiB is enough for ~200 lines on Bonn-class files
/// (average 14 bytes per line), which is well past the point where
/// every variable in the template should have stabilised.
const SNIFF_BYTES: usize = 4096;

/// Minimum tokens that must parse successfully in the sniff window
/// for the file to be considered ASCII int-per-line. Filters out
/// short config-style files (e.g. a 2-line key=value config).
const MIN_TOKENS_FOR_DETECTION: usize = 64;

/// Hard cap on per-sample value range. Detection refuses anything
/// outside i16 — the synthesized EDF stores i16 samples, and
/// outside that range we'd lose precision. The Bonn dataset is
/// 12-bit ADC so this is comfortably wide.
const VALUE_BOUND: i32 = 32767;

/// Examine the leading bytes of `data` and decide if it looks like
/// "one ASCII integer per line". Returns the template that captures
/// the formatting variations. `None` means the format doesn't match —
/// caller should fall through to the next detector (or zstd).
///
/// The function is conservative on purpose: it would rather mis-tag a
/// borderline ASCII numeric file as `None` (caller falls back to
/// zstd, which still compresses fine) than mis-tag a non-numeric
/// file as ascii and fail at parse time.
pub fn detect_ascii_int_lines(data: &[u8]) -> Option<AsciiLinesTemplate> {
    if data.is_empty() {
        return None;
    }
    let sniff = &data[..data.len().min(SNIFF_BYTES)];

    // 1. Determine line ending from the first newline encountered.
    let line_ending = if sniff.windows(2).any(|w| w == b"\r\n") {
        LineEnding::CrLf
    } else if sniff.contains(&b'\n') {
        LineEnding::Lf
    } else {
        return None; // no newline in sniff window — not line-oriented
    };

    // 2. Walk lines, count tokens, observe whitespace + width patterns.
    let mut tokens_seen: usize = 0;
    let mut leading_ws: Option<u8> = None;
    let mut max_width: u8 = 0;
    let field_widths_consistent = true;

    // Cap the sniff scan at the last complete line break. A trailing
    // partial line is undefined (we may have cut mid-CR-LF, or mid-
    // integer), so don't try to fingerprint it.
    let scan_end = match line_ending {
        LineEnding::Lf => sniff.iter().rposition(|&b| b == b'\n').map(|p| p + 1),
        LineEnding::CrLf => sniff.windows(2).rposition(|w| w == b"\r\n").map(|p| p + 2),
    };
    let scan = match scan_end {
        Some(end) => &sniff[..end],
        None => return None, // no complete line in sniff window
    };

    for raw_line in split_lines(scan, line_ending) {
        if raw_line.is_empty() {
            continue;
        }
        // Count leading-whitespace bytes.
        let lw = raw_line.iter().take_while(|&&b| b == b' ').count() as u8;
        match leading_ws {
            None => leading_ws = Some(lw),
            Some(prior) if prior != lw => {
                // Inconsistent leading whitespace — refuse. Files
                // mixing "12\n" and "  12\n" don't fit the template.
                return None;
            }
            _ => {}
        }

        let rest = &raw_line[lw as usize..];
        if !is_ascii_int_token(rest) {
            return None;
        }
        // Track natural width.
        let w = rest.len() as u8;
        if w > max_width {
            max_width = w;
        }
        // Parse + bounds check.
        let parsed = std::str::from_utf8(rest).ok()?.parse::<i32>().ok()?;
        if !(-VALUE_BOUND - 1..=VALUE_BOUND).contains(&parsed) {
            return None;
        }
        tokens_seen += 1;
    }

    if tokens_seen < MIN_TOKENS_FOR_DETECTION {
        return None;
    }

    // For Bonn-style ragged-width files, the template's field_width
    // stays 0 (natural width). We only flag a non-zero width if EVERY
    // sniffed line was the same width.
    let field_width = if field_widths_consistent && max_width > 0 {
        // Re-scan against the same complete-line window to confirm
        // all lines used exactly `max_width`. If any are shorter,
        // leave width=0 (natural).
        let all_same = split_lines(scan, line_ending)
            .filter(|l| !l.is_empty())
            .all(|line| {
                let lw = leading_ws.unwrap_or(0) as usize;
                line.len().saturating_sub(lw) == max_width as usize
            });
        if all_same {
            max_width
        } else {
            0
        }
    } else {
        0
    };

    let trailing_newline = ends_with_line_break(data, line_ending);

    Some(AsciiLinesTemplate {
        line_ending,
        leading_whitespace: leading_ws.unwrap_or(0),
        field_width,
        trailing_newline,
    })
}

/// Parse the complete buffer into `i16` samples per the template.
/// Returns an error on any token that doesn't parse, doesn't match the
/// expected leading-whitespace, or doesn't fit in `i16`.
pub fn parse_ascii_int_lines(
    data: &[u8],
    template: &AsciiLinesTemplate,
) -> Result<Vec<i16>, String> {
    let mut samples = Vec::new();
    for (idx, raw_line) in split_lines(data, template.line_ending).enumerate() {
        if raw_line.is_empty() {
            continue;
        }
        let lw = template.leading_whitespace as usize;
        if raw_line.len() < lw {
            return Err(format!(
                "ascii_lines: line {} shorter than expected leading whitespace ({})",
                idx + 1,
                lw
            ));
        }
        let rest = &raw_line[lw..];
        let s = std::str::from_utf8(rest)
            .map_err(|e| format!("ascii_lines: line {} not UTF-8: {}", idx + 1, e))?;
        let v: i32 = s
            .parse()
            .map_err(|e| format!("ascii_lines: line {} parse: {}", idx + 1, e))?;
        if !(-32768..=32767).contains(&v) {
            return Err(format!(
                "ascii_lines: line {} value {} doesn't fit in i16",
                idx + 1,
                v
            ));
        }
        samples.push(v as i16);
    }
    Ok(samples)
}

/// Inverse of `parse_ascii_int_lines` — given the recovered samples
/// + the stored template, produce the original byte sequence. Together
///
/// with `parse_ascii_int_lines`, this provides the bit-exact roundtrip
/// promised by ADR 0023.
pub fn render_ascii_int_lines(samples: &[i16], template: &AsciiLinesTemplate) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 8);
    let lw = b" ".repeat(template.leading_whitespace as usize);
    for s in samples {
        out.extend_from_slice(&lw);
        let mut s_str = s.to_string();
        if template.field_width > 0 && (s_str.len() as u8) < template.field_width {
            let pad = template.field_width as usize - s_str.len();
            for _ in 0..pad {
                s_str.insert(0, ' ');
            }
        }
        out.extend_from_slice(s_str.as_bytes());
        out.extend_from_slice(template.line_ending.as_bytes());
    }
    if !template.trailing_newline {
        let le_len = template.line_ending.as_bytes().len();
        let new_len = out.len().saturating_sub(le_len);
        out.truncate(new_len);
    }
    out
}

// ─── Helpers ──────────────────────────────────────────────────────

fn split_lines(data: &[u8], le: LineEnding) -> impl Iterator<Item = &[u8]> {
    let sep = le.as_bytes();
    let sep_len = sep.len();
    let mut start = 0usize;
    std::iter::from_fn(move || {
        if start >= data.len() {
            return None;
        }
        let rest = &data[start..];
        if let Some(pos) = find_subslice(rest, sep) {
            let line = &rest[..pos];
            start += pos + sep_len;
            Some(line)
        } else {
            // Trailing fragment with no terminator — emit + done.
            let line = &data[start..];
            start = data.len();
            Some(line)
        }
    })
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn ends_with_line_break(data: &[u8], le: LineEnding) -> bool {
    let sep = le.as_bytes();
    data.ends_with(sep)
}

/// Returns `true` if `rest` is a valid signed-int ASCII token (optional
/// leading `-`, then at least one digit, no other characters). Used by
/// the sniffer to gatekeep before paying for full `i32::from_str_radix`.
fn is_ascii_int_token(rest: &[u8]) -> bool {
    if rest.is_empty() {
        return false;
    }
    let mut i = 0;
    if rest[0] == b'-' || rest[0] == b'+' {
        i += 1;
    }
    if i >= rest.len() {
        return false;
    }
    while i < rest.len() {
        if !rest[i].is_ascii_digit() {
            return false;
        }
        i += 1;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_bonn_z_style() {
        // Bonn Z-set: LF, no leading whitespace, natural width.
        // Need >= 64 tokens.
        let mut bytes = Vec::new();
        for i in 0..100 {
            let v = (i - 50) * 3;
            bytes.extend_from_slice(format!("{}\n", v).as_bytes());
        }
        let t = detect_ascii_int_lines(&bytes).expect("should detect");
        assert_eq!(t.line_ending, LineEnding::Lf);
        assert_eq!(t.leading_whitespace, 0);
        assert_eq!(t.field_width, 0); // natural widths vary (-150..147)
        assert!(t.trailing_newline);
    }

    #[test]
    fn detect_bonn_n_style_crlf_with_leading_ws() {
        let mut bytes = Vec::new();
        for i in 0..100 {
            let v = (i - 50) * 3;
            bytes.extend_from_slice(format!("  {}\r\n", v).as_bytes());
        }
        let t = detect_ascii_int_lines(&bytes).expect("should detect");
        assert_eq!(t.line_ending, LineEnding::CrLf);
        assert_eq!(t.leading_whitespace, 2);
    }

    #[test]
    fn detect_rejects_non_numeric() {
        let bytes = b"hello\nworld\nthis is not numbers\n".repeat(20);
        assert!(detect_ascii_int_lines(&bytes).is_none());
    }

    #[test]
    fn detect_rejects_too_few_tokens() {
        let bytes = b"1\n2\n3\n";
        assert!(detect_ascii_int_lines(bytes).is_none());
    }

    #[test]
    fn detect_rejects_inconsistent_leading_ws() {
        // First line has 2-space lead; second has 0. Should refuse.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"  12\n");
        bytes.extend_from_slice(b"34\n");
        for _ in 0..100 {
            bytes.extend_from_slice(b"  5\n");
        }
        assert!(detect_ascii_int_lines(&bytes).is_none());
    }

    #[test]
    fn parse_then_render_roundtrip_lf() {
        let mut bytes = Vec::new();
        for i in 0..200 {
            let v = (i % 100) - 50;
            bytes.extend_from_slice(format!("{}\n", v).as_bytes());
        }
        let t = detect_ascii_int_lines(&bytes).unwrap();
        let samples = parse_ascii_int_lines(&bytes, &t).unwrap();
        let rendered = render_ascii_int_lines(&samples, &t);
        assert_eq!(bytes, rendered);
    }

    #[test]
    fn parse_then_render_roundtrip_crlf_leading_ws() {
        let mut bytes = Vec::new();
        for i in 0..120 {
            let v = (i % 200) - 100;
            bytes.extend_from_slice(format!("  {}\r\n", v).as_bytes());
        }
        let t = detect_ascii_int_lines(&bytes).unwrap();
        assert_eq!(t.leading_whitespace, 2);
        assert_eq!(t.line_ending, LineEnding::CrLf);
        let samples = parse_ascii_int_lines(&bytes, &t).unwrap();
        let rendered = render_ascii_int_lines(&samples, &t);
        assert_eq!(bytes, rendered);
    }

    #[test]
    fn roundtrip_without_trailing_newline() {
        // Build a file that doesn't end with newline.
        let mut bytes = Vec::new();
        for i in 0..80 {
            bytes.extend_from_slice(format!("{}\n", i).as_bytes());
        }
        bytes.extend_from_slice(b"999"); // no trailing \n
        let t = detect_ascii_int_lines(&bytes).unwrap();
        assert!(!t.trailing_newline);
        let samples = parse_ascii_int_lines(&bytes, &t).unwrap();
        let rendered = render_ascii_int_lines(&samples, &t);
        assert_eq!(bytes, rendered);
    }

    #[test]
    fn parse_rejects_value_outside_i16() {
        let t = AsciiLinesTemplate {
            line_ending: LineEnding::Lf,
            leading_whitespace: 0,
            field_width: 0,
            trailing_newline: true,
        };
        let bytes = b"100\n50000\n";
        let err = parse_ascii_int_lines(bytes, &t).unwrap_err();
        assert!(err.contains("doesn't fit in i16"), "got: {}", err);
    }

    #[test]
    fn empty_buffer_not_detected() {
        assert!(detect_ascii_int_lines(b"").is_none());
    }

    #[test]
    fn template_to_from_json_roundtrip() {
        let t = AsciiLinesTemplate {
            line_ending: LineEnding::CrLf,
            leading_whitespace: 2,
            field_width: 0,
            trailing_newline: true,
        };
        let v = t.to_json();
        let back = AsciiLinesTemplate::from_json(&v).unwrap();
        assert_eq!(t, back);
    }

    #[test]
    fn template_from_json_rejects_missing_field() {
        let v = serde_json::json!({
            "line_ending": "Lf",
            // missing leading_whitespace + field_width + trailing_newline
        });
        let err = AsciiLinesTemplate::from_json(&v).unwrap_err();
        assert!(err.contains("leading_whitespace"), "got: {}", err);
    }

    #[test]
    fn template_from_json_rejects_bad_line_ending() {
        let v = serde_json::json!({
            "line_ending": "Cr",   // invalid
            "leading_whitespace": 0,
            "field_width": 0,
            "trailing_newline": true,
        });
        assert!(AsciiLinesTemplate::from_json(&v).is_err());
    }
}
