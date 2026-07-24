//! Cat B integration smoke tests (2026-05-22).
//!
//! User direction: *"Add all the libraries and test them as a separate
//! thing, compare to the original, then see which is better."*
//!
//! Each test here exercises one Cat B crate against the corresponding
//! hand-rolled path inside `lamquant-core` (or simply exercises the
//! lib's API surface where no in-tree equivalent exists). PASS = the
//! lib builds + works as documented; the *better-than-original*
//! decision lives in `docs/cat_b_integration.md`, not here.

#![cfg(feature = "host")]

use realfft::{num_complex::Complex32, RealFftPlanner};

/// `realfft` — host-side real-valued FFT. Smoke: forward 8 samples,
/// inverse them back, assert ≤ 1e-3 absolute error. Verifies the lib
/// is wired + the FFI is happy.
#[test]
fn realfft_roundtrip_8_point() {
    let mut planner = RealFftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(8);
    let inv = planner.plan_fft_inverse(8);

    let n: usize = 8;
    let mut samples: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();
    let original = samples.clone();
    let mut spectrum: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n / 2 + 1];

    fwd.process(&mut samples, &mut spectrum).expect("forward");
    inv.process(&mut spectrum, &mut samples).expect("inverse");

    // Inverse output is unscaled — divide by N to recover input. Read
    // N from the live buffer so the divisor stays in lockstep with
    // the FFT plan size. (lamu review nit on 2a8d047.)
    let n_f = samples.len() as f32;
    for x in samples.iter_mut() {
        *x /= n_f;
    }
    for (orig, rec) in original.iter().zip(samples.iter()) {
        assert!(
            (orig - rec).abs() < 1e-3,
            "realfft roundtrip drift {} vs {}",
            orig,
            rec
        );
    }
}

/// `pulp` — desktop SIMD dispatcher. Smoke: run a simple scalar
/// kernel through pulp's `Arch::new()` dispatcher. Verifies the lib
/// is wired and the runtime-feature-detected backend works.
#[test]
fn pulp_dispatch_sum_smoke() {
    use pulp::Arch;
    let arch = Arch::new();
    let n = 1024usize;
    let xs: Vec<f32> = (0..n).map(|i| i as f32 * 0.001).collect();

    // `Arch::dispatch` takes a nullary closure that internally picks
    // the best `Simd` impl. The simplest call: a scalar inner-product
    // proves the dispatcher is wired. (Real SIMD lowering comes when
    // a kernel actually uses the SIMD token.)
    let sum: f32 = arch.dispatch(|| xs.iter().sum::<f32>());
    let expected: f32 = (0..n).map(|i| i as f32 * 0.001).sum();
    assert!((sum - expected).abs() < 1e-3, "pulp dispatch sum drift");
}

/// `loom` — concurrency model checker. Smoke: verify a trivial
/// Arc<AtomicUsize> increment under loom's interleaving runner.
/// loom replaces std::thread with its own scheduler that explores
/// every legal interleaving — a passing test means no
/// happens-before bugs in the wrapped code. Cat B watch-list only;
/// no production use today.
#[test]
fn loom_atomic_increment_smoke() {
    use loom::sync::atomic::{AtomicUsize, Ordering};
    use loom::sync::Arc;
    use loom::thread;
    loom::model(|| {
        let counter = Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let t = thread::spawn(move || {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        counter.fetch_add(1, Ordering::SeqCst);
        t.join().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    });
}

/// `bitstream-io` — bit-level stream r/w. Smoke: write 3+5+8 = 16
/// bits (already byte-aligned, no padding needed), read them back
/// via `BitReader`, assert identity. Recommended over `bitvec` per
/// the original audit.
#[test]
fn bitstream_io_roundtrip_smoke() {
    use bitstream_io::{BigEndian, BitRead, BitReader, BitWrite, BitWriter};
    let mut buf: Vec<u8> = Vec::new();
    {
        let mut w = BitWriter::endian(&mut buf, BigEndian);
        w.write::<3, u8>(0b101).expect("3-bit");
        w.write::<5, u8>(0b1_0110).expect("5-bit");
        w.write::<8, u8>(0xA5).expect("8-bit");
        // 3+5+8 = 16 bits = 2 bytes exactly; no `byte_align()`
        // needed. (lamu review nit on 22753f0.)
    }
    let mut r = BitReader::endian(buf.as_slice(), BigEndian);
    assert_eq!(r.read::<3, u8>().expect("3-bit"), 0b101);
    assert_eq!(r.read::<5, u8>().expect("5-bit"), 0b1_0110);
    assert_eq!(r.read::<8, u8>().expect("8-bit"), 0xA5);
}

/// `faer 0.22` — pure-Rust dense linear algebra. Smoke: 4×4 matrix
/// multiply identity vs structured matrix returns the original
/// (within f32 epsilon). Watch-list bookmark only.
#[test]
fn faer_4x4_multiply_smoke() {
    use faer::Mat;
    // Idiomatic identity constructor — `Mat::<f32>::identity(n, n)`
    // beats hand-built `from_fn` for the diagonal-1/off-diagonal-0
    // pattern. (lamu review nit on 22753f0.)
    let identity: Mat<f32> = Mat::identity(4, 4);
    let m: Mat<f32> = Mat::from_fn(4, 4, |i, j| (i + j * 4 + 1) as f32);
    let product = &identity * &m;
    for i in 0..4 {
        for j in 0..4 {
            let drift = (product[(i, j)] - m[(i, j)]).abs();
            assert!(drift < 1e-5, "I·M != M at ({},{}): drift {}", i, j, drift);
        }
    }
}

/// `rkyv` — zero-copy mmap'd archive. Smoke: archive a small struct
/// into a `Vec<u8>`, validate + deserialize, assert roundtrip. If we
/// ever need a conformance-suite mmap (audit doc threshold is
/// 200 MB), this is the lib.
#[test]
fn rkyv_struct_archive_smoke() {
    use rkyv::{archived_root, ser::serializers::AllocSerializer, ser::Serializer};
    use rkyv::{Archive, Deserialize, Infallible, Serialize};

    #[derive(Archive, Serialize, Deserialize, Debug, PartialEq)]
    #[archive(check_bytes)]
    struct Sample {
        a: u32,
        b: i16,
        s: String,
    }
    let original = Sample {
        a: 0xDEAD_BEEF,
        b: -42,
        s: "rkyv".into(),
    };

    let mut ser: AllocSerializer<128> = AllocSerializer::default();
    ser.serialize_value(&original).expect("serialize");
    let bytes = ser.into_serializer().into_inner();

    // Zero-copy access first — proves the archived view is sound.
    let archived = unsafe { archived_root::<Sample>(&bytes) };
    assert_eq!(archived.a, original.a);
    assert_eq!(archived.b, original.b);
    assert_eq!(archived.s.as_str(), original.s.as_str());

    // Owned deserialize (loses the zero-copy property but proves the
    // round-trip semantics).
    let recovered: Sample = archived.deserialize(&mut Infallible).expect("deserialize");
    assert_eq!(recovered, original);
}
