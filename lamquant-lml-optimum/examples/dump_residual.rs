//! Dump the SHIPPED MV-RLS residual (config 1) to <bin>.res for offline entropy analysis.
use std::fs;
use lamquant_lml_optimum::mv_rls;
fn main() {
    let path = std::env::args().nth(1).expect("bin");
    let b = fs::read(&path).expect("read");
    let nch = u32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
    let t = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let mut off = 8; let mut sig = Vec::with_capacity(nch);
    for _ in 0..nch { let mut ch = Vec::with_capacity(t);
        for _ in 0..t { ch.push(i32::from_le_bytes(b[off..off+4].try_into().unwrap()) as i64); off += 4; }
        sig.push(ch); }
    let res = mv_rls::residuals(&sig, 1, 0);
    let mut out = Vec::new();
    out.extend_from_slice(&(nch as u32).to_le_bytes());
    out.extend_from_slice(&(t as u32).to_le_bytes());
    for ch in &res { for &v in ch { out.extend_from_slice(&(v as i32).to_le_bytes()); } }
    fs::write(format!("{path}.res"), out).expect("write");
    eprintln!("dumped {nch}ch x {t} residual");
}
