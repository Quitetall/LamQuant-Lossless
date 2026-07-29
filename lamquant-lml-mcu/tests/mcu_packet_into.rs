use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use lamquant_lml_mcu::{
    lml::{compress_with_mode_views_explicit, EncodeFeatures},
    lpc::LpcMode,
    mcu_packet::{
        compress_fixed_invocation_into, compress_invocation_into, fixed_packet_workspace_len,
        uniform_i64_invocation_len, write_uniform_i64_invocation, McuLpcSchedule,
    },
};

struct ThreadCountingAllocator;

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

#[test]
fn adaptive_packet_invocation_is_byte_equal_and_heap_free() {
    for (channels, samples, max_order) in [(1, 31, 1), (4, 313, 8), (8, 2_500, 16), (32, 2_500, 64)]
    {
        let signal = signal(channels, samples);
        let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut invocation = vec![0_u8; uniform_i64_invocation_len(channels, samples).unwrap()];
        write_uniform_i64_invocation(&views, &mut invocation).unwrap();
        let mut output = vec![0_u8; 8 * 1024 * 1024];
        let mut workspace = vec![0_u8; fixed_packet_workspace_len(samples).unwrap()];
        let expected = compress_with_mode_views_explicit(
            &views,
            0,
            LpcMode::Adaptive { max_order },
            EncodeFeatures {
                max_packet_bytes: Some(output.len()),
                ..EncodeFeatures::default()
            },
        )
        .unwrap();

        ALLOCATION_COUNT.with(|count| count.set(0));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        let written = compress_invocation_into(
            &invocation,
            McuLpcSchedule::Adaptive { max_order },
            &mut workspace,
            &mut output,
        )
        .unwrap();
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        let allocations = ALLOCATION_COUNT.with(Cell::get);

        assert_eq!(allocations, 0, "{channels}ch x {samples} order {max_order}");
        assert_eq!(
            &output[..written],
            expected,
            "{channels}ch x {samples} order {max_order}"
        );
    }
}

unsafe impl GlobalAlloc for ThreadCountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        TRACK_ALLOCATIONS.with(|tracking| {
            if tracking.get() {
                ALLOCATION_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: ThreadCountingAllocator = ThreadCountingAllocator;

fn signal(channels: usize, samples: usize) -> Vec<Vec<i64>> {
    (0..channels)
        .map(|channel| {
            (0..samples)
                .map(|sample| {
                    let base = ((sample * 3 + channel * 7) % 512) as i64 - 256;
                    let wobble = ((sample * sample + channel) % 97) as i64 - 48;
                    base * 40 + wobble
                })
                .collect()
        })
        .collect()
}

#[test]
fn fixed_packet_invocation_is_byte_equal_and_heap_free() {
    for (channels, samples) in [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 31),
        (4, 313),
        (8, 2_500),
        (32, 2_500),
    ] {
        let signal = signal(channels, samples);
        let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let mut invocation = vec![0_u8; uniform_i64_invocation_len(channels, samples).unwrap()];
        let invocation_len = write_uniform_i64_invocation(&views, &mut invocation).unwrap();
        assert_eq!(invocation_len, invocation.len());

        let mut output = vec![0_u8; 8 * 1024 * 1024];
        let mut workspace = vec![0_u8; fixed_packet_workspace_len(samples).unwrap()];
        let expected = compress_with_mode_views_explicit(
            &views,
            0,
            LpcMode::Fixed,
            EncodeFeatures {
                max_packet_bytes: Some(output.len()),
                ..EncodeFeatures::default()
            },
        )
        .unwrap();

        ALLOCATION_COUNT.with(|count| count.set(0));
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
        let written =
            compress_fixed_invocation_into(&invocation, &mut workspace, &mut output).unwrap();
        TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
        let allocations = ALLOCATION_COUNT.with(Cell::get);

        assert_eq!(allocations, 0, "{channels}ch x {samples}");
        assert_eq!(&output[..written], expected, "{channels}ch x {samples}");
    }
}

#[test]
fn invocation_and_resource_bounds_fail_closed() {
    assert!(uniform_i64_invocation_len(0, 1).is_err());
    assert!(uniform_i64_invocation_len(257, 1).is_err());
    assert!(uniform_i64_invocation_len(1, 0).is_err());
    assert!(uniform_i64_invocation_len(1, u16::MAX as usize + 1).is_err());

    let ragged = [vec![1_i64, 2], vec![3]];
    let ragged_views = ragged.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut invocation = [0_u8; 64];
    assert!(write_uniform_i64_invocation(&ragged_views, &mut invocation).is_err());

    let signal = signal(1, 4);
    let views = signal.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let mut invocation = vec![0_u8; uniform_i64_invocation_len(1, 4).unwrap()];
    write_uniform_i64_invocation(&views, &mut invocation).unwrap();
    let mut output = [0_u8; 4096];
    let required_workspace = fixed_packet_workspace_len(4).unwrap();
    assert!(compress_fixed_invocation_into(
        &invocation,
        &mut vec![0_u8; required_workspace - 1],
        &mut output
    )
    .is_err());
    assert!(compress_fixed_invocation_into(
        &invocation,
        &mut vec![0_u8; required_workspace],
        &mut [0_u8; 8]
    )
    .is_err());
    invocation.push(0);
    assert!(compress_fixed_invocation_into(
        &invocation,
        &mut vec![0_u8; required_workspace],
        &mut output
    )
    .is_err());
}
