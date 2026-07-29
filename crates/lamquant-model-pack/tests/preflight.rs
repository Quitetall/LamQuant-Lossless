use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use lamquant_model_pack::{ModelPack, ModelTensor, PackErrorKind, TensorDtype, MAX_PACK_BYTES};

struct ObservedAllocator;

static OBSERVE: AtomicBool = AtomicBool::new(false);
static MAX_ALLOCATION: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for ObservedAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if OBSERVE.load(Ordering::Relaxed) {
            MAX_ALLOCATION.fetch_max(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: ObservedAllocator = ObservedAllocator;

#[test]
fn oversized_pack_is_rejected_before_payload_clone_or_output_allocation() {
    let tensor = ModelTensor {
        name: "oversized".into(),
        dtype: TensorDtype::I8,
        shape: vec![u32::try_from(MAX_PACK_BYTES).unwrap()],
        scale_numerator: 1,
        scale_shift: 0,
        data: vec![0; MAX_PACK_BYTES],
    };

    MAX_ALLOCATION.store(0, Ordering::Relaxed);
    OBSERVE.store(true, Ordering::Relaxed);
    let error = ModelPack::encode(&[tensor]).unwrap_err();
    OBSERVE.store(false, Ordering::Relaxed);

    assert_eq!(error.kind(), PackErrorKind::InvalidInput);
    assert!(
        MAX_ALLOCATION.load(Ordering::Relaxed) < 1_048_576,
        "oversized preflight allocated or cloned a large payload"
    );
}
