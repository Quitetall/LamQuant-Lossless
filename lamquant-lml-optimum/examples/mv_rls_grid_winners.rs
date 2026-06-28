//! Which of MV-RLS's 14 (config × seg) variants ever win the internal keep-best?
//! Per W-window, find the min over the grid; tally winners. Never-winners are prunable
//! (removing a variant that never wins can't regress — keep-best is unchanged).
use std::fs; use std::collections::BTreeMap;
use lamquant_lml_optimum::mv_rls;
const W:usize=32768;
// the shipped CONFIGS grid (λ, reset, m)
const CFG:&[(f64,usize,usize)]=&[(0.999,8192,32),(0.997,4096,32),(0.995,2048,32),
  (0.990,1024,32),(0.985,512,32),(0.990,1024,8),(0.985,512,4)];
fn read_bin(p:&str)->Vec<Vec<i64>>{let b=fs::read(p).unwrap();
  let nch=u32::from_le_bytes(b[0..4].try_into().unwrap())as usize;let t=u32::from_le_bytes(b[4..8].try_into().unwrap())as usize;
  let mut o=8;let mut s=vec![];for _ in 0..nch{let mut c=vec![];for _ in 0..t{c.push(i32::from_le_bytes(b[o..o+4].try_into().unwrap())as i64);o+=4;}s.push(c);}s}
fn main(){let mut wins:BTreeMap<(usize,usize),usize>=BTreeMap::new();
  for path in std::env::args().skip(1){let sig=read_bin(&path);let t=sig[0].len();let mut start=0;
    while start<t{let end=(start+W).min(t);let win:Vec<Vec<i64>>=sig.iter().map(|c|c[start..end].to_vec()).collect();
      let(mut best,mut bi)=(usize::MAX,(0,0));
      for(ci,&(l,r,m))in CFG.iter().enumerate(){for seg in 0..2{
        let len=mv_rls::encode_len_params(&win,l,r,m,seg);
        if len<best{best=len;bi=(ci,seg);}}}
      *wins.entry(bi).or_insert(0)+=1;start=end;}
    println!("  processed {}",path.rsplit('/').next().unwrap());}
  println!("\n  per-config win counts (cfg_idx, seg) → wins:");
  for((ci,seg),n)in &wins{let(l,r,m)=CFG[*ci];println!("    cfg{} (λ={},reset={},m={}) seg={}  ×{}",ci,l,r,m,seg,n);}
  println!("  # configs absent here are candidates to PRUNE (never win ⇒ never-worse preserved).");
}
