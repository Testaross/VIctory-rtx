//! Wave64 arithmetic expressed as two Wave32 register values.
//! The paired operations require a converged physical warp: guest EXEC
//! predicates are data, not host control flow. They do not introduce group
//! barriers. Full paired control-flow and register lowering is separate.

use super::bitcode::{Attr, Bin, V, pred};
use super::shader::{Shader, flags};

/// A mask word's bits strictly below a logical lane index. Saturate at
/// 32 instead of emitting LLVM's undefined shift by the word width.
pub(super) fn prefix_word(s: &mut Shader, at: usize, word: V, below: V) -> V {
    let one = s.m.i32(1);
    let k31 = s.m.i32(31);
    let k32 = s.m.i32(32);
    let ones = s.m.i32(u32::MAX);
    let sh = s.m.bin(at, Bin::And, below, k31);
    let mask = s.m.bin(at, Bin::Shl, one, sh);
    let mask = s.m.bin(at, Bin::Sub, mask, one);
    let full = s.m.cmp(at, pred::IUGE, below, k32);
    let mask = s.m.select(at, full, ones, mask);
    let bits = s.m.bin(at, Bin::And, word, mask);
    s.op(at, "unaryBits.i32", Attr::ReadNone, s.t.i32, 31, &[bits])
}

pub(super) fn mask_bit(s: &mut Shader, at: usize, lo: V, hi: Option<V>, lane: V) -> V {
    let k31 = s.m.i32(31);
    let k32 = s.m.i32(32);
    let one = s.m.i32(1);
    let zero = s.m.i32(0);
    let word = match hi {
        Some(hi) => {
            let low = s.m.cmp(at, pred::IULT, lane, k32);
            s.m.select(at, low, lo, hi)
        }
        None => lo,
    };
    let sh = s.m.bin(at, Bin::And, lane, k31);
    let bit = s.m.bin(at, Bin::LShr, word, sh);
    let bit = s.m.bin(at, Bin::And, bit, one);
    s.m.cmp(at, pred::NE, bit, zero)
}

/// Paired per-lane registers, or a pair of 32-bit mask words. The low and
/// high VGPR values belong to guest lanes n and n+32 respectively.
#[allow(dead_code)] // consumed by the paired control-flow lowering once integrated
#[derive(Clone, Copy)]
pub(super) struct Pair {
    pub lo: V,
    pub hi: V,
}

#[allow(dead_code)]
impl Pair {
    /// Both inputs are per-logical-lane i1 values. Every physical lane
    /// must execute both ballots, even if both guest lanes are inactive.
    pub fn ballot(s: &mut Shader, at: usize, active: Self) -> Self {
        s.flags |= flags::WAVE_OPS;
        let four = s.m.named_struct("dx.types.fouri32", vec![s.t.i32; 4]);
        let low = s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[active.lo]);
        let high = s.op(at, "waveActiveBallot", Attr::NoUnwind, four, 116, &[active.hi]);
        Self {
            lo: s.m.extract(at, low, 0),
            hi: s.m.extract(at, high, 0),
        }
    }

    /// Read guest VGPR lane 0..63. Both shuffles execute before selecting
    /// the correct half; physical lane indices are always 0..31.
    pub fn read_lane(self, s: &mut Shader, at: usize, guest_lane: V) -> V {
        s.flags |= flags::WAVE_OPS;
        let k31 = s.m.i32(31);
        let k32 = s.m.i32(32);
        let physical = s.m.bin(at, Bin::And, guest_lane, k31);
        let low = s.op(at, "waveReadLaneAt.i32", Attr::NoUnwind, s.t.i32, 117, &[self.lo, physical]);
        let high = s.op(at, "waveReadLaneAt.i32", Attr::NoUnwind, s.t.i32, 117, &[self.hi, physical]);
        let lower = s.m.cmp(at, pred::IULT, guest_lane, k32);
        s.m.select(at, lower, low, high)
    }

    /// Exclusive prefix population counts for both logical lanes owned by
    /// a physical lane. The high half includes all active low-half lanes.
    pub fn prefix(self, s: &mut Shader, at: usize, physical: V) -> Self {
        let lo = prefix_word(s, at, self.lo, physical);
        let total_lo = s.op(at, "unaryBits.i32", Attr::ReadNone, s.t.i32, 31, &[self.lo]);
        let hi = prefix_word(s, at, self.hi, physical);
        let hi = s.m.bin(at, Bin::Add, total_lo, hi);
        Self { lo, hi }
    }

    /// Masked register writes do not destroy inactive guest VGPR values.
    pub fn write(self, s: &mut Shader, at: usize, active: Self, next: Self) -> Self {
        Self {
            lo: s.m.select(at, active.lo, next.lo, self.lo),
            hi: s.m.select(at, active.hi, next.hi, self.hi),
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use super::super::bitcode::Term;
    use super::super::{HostShader, host::{Dispatch, Host, View}, validator};

    #[test]
    fn paired_wave32_dxil_ballot_prefix_and_lane_reads() {
        let mut s = Shader::new(false);
        s.threads = Some([32, 1, 1]);
        s.wave_size = Some(32);
        let at = s.entry;
        let lane = s.op(at, "waveGetLaneIndex", Attr::ReadOnly, s.t.i32, 111, &[]);
        let zero = s.m.i32(0);
        let one = s.m.i32(1);
        let k31 = s.m.i32(31);
        let k32 = s.m.i32(32);
        let k63 = s.m.i32(63);
        let lo_active = s.m.cmp(at, pred::EQ, lane, zero);
        let hi_active = s.m.cmp(at, pred::EQ, lane, k31);
        let active = Pair { lo: lo_active, hi: hi_active };
        let mask = Pair::ballot(&mut s, at, active);
        let prefix = mask.prefix(&mut s, at, lane);
        let upper = s.m.bin(at, Bin::Add, lane, k32);
        let values = Pair { lo: lane, hi: upper };
        let kept = values.write(&mut s, at, active, Pair { lo: zero, hi: zero });
        let read = values.read_lane(&mut s, at, k63);
        let h = s.heap_handle(at, zero, false, false, (11 | 0x1000, 0));
        let k16 = s.m.i32(16);
        let byte = s.m.bin(at, Bin::Mul, lane, k16);
        let undef = s.m.undef(s.t.i32);
        let align = s.m.i32(4);
        let channels = s.m.i8(15);
        // Distinct addresses per physical lane, no racing UAV writes.
        s.op(at, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140,
             &[h, byte, undef, prefix.lo, prefix.hi, read, kept.hi, channels, align]);
        s.m.terminate(at, Term::Ret);
        let dxil = s.finish();
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = dxil.clone();
            v.validate(&mut copy).expect("paired Wave32 DXIL validation failed");
        }
        let shader = HostShader {
            stage: crate::HwStage::Cs,
            dxil,
            user_data: Vec::new(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: false,
        };
        let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware device; paired Wave32 execution not tested");
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("paired Wave32 pipeline");
        let out = host.dispatch(&root, &pso, &Dispatch {
            roots: vec![],
            views: &[View::RawUav(128)],
            samplers: &[],
            groups: [1, 1, 1],
        }).expect("paired Wave32 dispatch");
        for n in 0..32usize {
            let expected = [u32::from(n > 0), 1, 63, if n == 31 { 0 } else { n as u32 + 32 }];
            assert_eq!(&out[0][n * 4..n * 4 + 4], &expected);
        }
        // Keep the low/high prefix checks above independent of host lane ordering.
        let _ = one;
    }
}
