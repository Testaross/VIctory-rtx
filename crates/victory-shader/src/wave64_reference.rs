//! Synthetic reference model for paired-lane Wave64 lowering on Wave32.
//! This is a test oracle, not a GPU fallback. The emitter must eventually
//! represent each guest VGPR as two values and run both halves under a
//! single logical execution mask. No title data is used here.

/// A logical wave has two 32-bit masks and two values at every host lane.
struct Wave {
    lo: u32,
    hi: u32,
    values: [[u32; 32]; 2],
}

impl Wave {
    fn mask(&self) -> u64 {
        u64::from(self.lo) | (u64::from(self.hi) << 32)
    }

    fn active(&self, lane: usize) -> bool {
        self.mask() & (1u64 << lane) != 0
    }

    /// Explicit lane reads address the VGPR, not just active lanes.
    fn read_lane(&self, lane: usize) -> u32 {
        self.values[lane / 32][lane % 32]
    }

    fn first_active(&self) -> Option<u32> {
        (self.mask() != 0).then(|| self.read_lane(self.mask().trailing_zeros() as usize))
    }

    fn prefix(&self, lane: usize) -> u32 {
        (self.mask() & ((1u64 << lane) - 1)).count_ones()
    }

    fn count(&self) -> u32 {
        self.mask().count_ones()
    }

    fn write_active(&mut self, value: u32) {
        for lane in 0..64 {
            if self.active(lane) {
                self.values[lane / 32][lane % 32] = value;
            }
        }
    }
}

/// Preserve consecutive 64-lane guest waves: each host lane owns two
/// guest threads separated by 32, not two adjacent threads.
fn guest_threads(host_thread: usize) -> [usize; 2] {
    let wave = host_thread / 32;
    let lane = host_thread % 32;
    [wave * 64 + lane, wave * 64 + lane + 32]
}

fn example(lo: u32, hi: u32) -> Wave {
    Wave {
        lo,
        hi,
        values: std::array::from_fn(|half| {
            std::array::from_fn(|lane| (half * 32 + lane) as u32)
        }),
    }
}

#[test]
fn paired_mapping_preserves_two_independent_guest_waves() {
    let mut seen = [false; 128];
    for host in 0..64 {
        for guest in guest_threads(host) {
            assert!(!seen[guest]);
            seen[guest] = true;
            assert_eq!(guest / 64, host / 32);
        }
    }
    assert!(seen.into_iter().all(|v| v));
    assert_eq!(guest_threads(31), [31, 63]);
    assert_eq!(guest_threads(32), [64, 96]);
}

#[test]
fn mask_prefix_crosses_the_half_boundary() {
    let w = example(0x8000_0001, 0x8000_0003);
    assert_eq!(w.count(), 5);
    assert_eq!(w.prefix(0), 0);
    assert_eq!(w.prefix(32), 2);
    assert_eq!(w.prefix(33), 3);
    assert_eq!(w.prefix(63), 4);
    assert_eq!(w.read_lane(63), 63);
}

#[test]
fn sparse_high_half_and_empty_mask() {
    let w = example(0, 0x8000_0000);
    assert_eq!(w.first_active(), Some(63));
    assert_eq!(w.prefix(63), 0);
    assert_eq!(example(0, 0).first_active(), None);
}

#[test]
fn masked_writes_preserve_inactive_lane_registers() {
    let mut w = example(1, 2);
    w.write_active(100);
    assert_eq!(w.read_lane(0), 100);
    assert_eq!(w.read_lane(33), 100);
    assert_eq!(w.read_lane(32), 32);
    assert_eq!(w.read_lane(63), 63);
}

#[test]
fn every_single_lane_and_all_lanes() {
    for lane in 0..64 {
        let mask = 1u64 << lane;
        let w = example(mask as u32, (mask >> 32) as u32);
        assert_eq!(w.count(), 1);
        assert_eq!(w.first_active(), Some(lane as u32));
        for other in 0..64 {
            assert_eq!(w.prefix(other), u32::from(lane < other));
        }
    }
    let w = example(u32::MAX, u32::MAX);
    assert_eq!(w.count(), 64);
    for lane in 0..64 {
        assert_eq!(w.prefix(lane), lane as u32);
    }
}
