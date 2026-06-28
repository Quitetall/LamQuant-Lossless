//! Floor (MCU/deployed, single 5/3+LPC pass) vs Optimum (host max-ratio keep-best) encode time.
use std::fs; use std::time::Instant;
use lamquant_lml_mcu::codec::{Codec, Mode, LmlCodec};
use lamquant_lml_optimum::LmoCodec;
const W:usize=32768;
fn read_bin(p:&str)->Vec<Vec<i64>>{let b=fs::read(p).unwrap();
  let nch=u32::from_le_bytes(b[0..4].try_into().unwrap())as usize;let t=u32::from_le_bytes(b[4..8].try_into().unwrap())as usize;
  let mut o=8;let mut s=vec![];for _ in 0..nch{let mut c=vec![];for _ in 0..t{c.push(i32::from_le_bytes(b[o..o+4].try_into().unwrap())as i64);o+=4;}s.push(c);}s}
fn main(){let path=std::env::args().nth(1).unwrap();let sig=read_bin(&path);let t=sig[0].len();
  let windows:Vec<Vec<Vec<i64>>>=(0..t).step_by(W).map(|s|{let e=(s+W).min(t);sig.iter().map(|c|c[s..e].to_vec()).collect()}).collect();
  let t0=Instant::now();let mut fb=0;for w in &windows{fb+=LmlCodec.encode(w,Mode::Lossless).unwrap().len();}let ft=t0.elapsed().as_secs_f64();
  let t0=Instant::now();let mut ob=0;for w in &windows{ob+=LmoCodec.encode(w,Mode::Lossless).unwrap().len();}let ot=t0.elapsed().as_secs_f64();
  println!("  {:<28} floor {:.2}s ({} B) | optimum {:.2}s ({} B)",path.rsplit('/').next().unwrap(),ft,fb,ot,ob);}
