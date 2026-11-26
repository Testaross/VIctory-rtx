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

    /// Scalar EXEC test. All physical lanes agree on the result, even
    /// when only the upper logical half is active.
    pub fn any(self, s: &mut Shader, at: usize) -> V {
        let combined = s.m.bin(at, Bin::Or, self.lo, self.hi);
        let zero = s.m.i32(0);
        s.m.cmp(at, pred::NE, combined, zero)
    }

    /// Count both words of a logical ballot.
    pub fn count(self, s: &mut Shader, at: usize) -> V {
        let lo = s.op(at, "unaryBits.i32", Attr::ReadNone, s.t.i32, 31, &[self.lo]);
        let hi = s.op(at, "unaryBits.i32", Attr::ReadNone, s.t.i32, 31, &[self.hi]);
        s.m.bin(at, Bin::Add, lo, hi)
    }

    /// The lowest logical active lane. For an empty EXEC the lane index
    /// is zero and the validity predicate is false, so consumers never
    /// shuffle from an out-of-range source.
    pub fn first(self, s: &mut Shader, at: usize) -> (V, V) {
        let zero = s.m.i32(0);
        let k32 = s.m.i32(32);
        let flo = s.op(at, "unaryBits.i32", Attr::ReadNone, s.t.i32, 32, &[self.lo]);
        let fhi = s.op(at, "unaryBits.i32", Attr::ReadNone, s.t.i32, 32, &[self.hi]);
        let fhi = s.m.bin(at, Bin::Add, fhi, k32);
        let has_lo = s.m.cmp(at, pred::NE, self.lo, zero);
        let any = self.any(s, at);
        let lane = s.m.select(at, has_lo, flo, fhi);
        (s.m.select(at, any, lane, zero), any)
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
    }

    /// Two physical warps are two guest Wave64s. The waves perform
    /// different numbers of iterations, then meet at the *same* group
    /// barrier and consume the LDS results. No barrier in the wave loop.
    #[test]
    fn paired_wave_loops_rejoin_at_group_barrier() {
        use super::super::bitcode::Rmw;
        let mut s = Shader::new(false);
        s.threads = Some([64, 1, 1]);
        s.wave_size = Some(32);
        s.lds_dwords = 3;
        s.flags |= flags::WAVE_OPS;

        let entry = s.entry;
        let init = s.m.new_block();
        let start = s.m.new_block();
        let header = s.m.new_block();
        let body = s.m.new_block();
        let exit = s.m.new_block();
        let write = s.m.new_block();
        let meet = s.m.new_block();

        let zero = s.m.i32(0);
        let one = s.m.i32(1);
        let two = s.m.i32(2);
        let k31 = s.m.i32(31);
        let k32 = s.m.i32(32);
        let k63 = s.m.i32(63);
        let mode = s.m.i32(9); // thread-group sync + TGSM fence
        let lds = s.lds();
        let flat = s.op(entry, "flattenedThreadIdInGroup.i32", Attr::ReadNone, s.t.i32, 96, &[]);
        let lane = s.op(entry, "waveGetLaneIndex", Attr::ReadOnly, s.t.i32, 111, &[]);
        let first_thread = s.m.cmp(entry, pred::EQ, flat, zero);
        let first_lane = s.m.cmp(entry, pred::EQ, lane, zero);
        s.m.terminate(entry, Term::CondBr(first_thread, init, start));
        let counter = s.m.element(init, lds, zero);
        s.m.store(init, counter, zero);
        s.m.terminate(init, Term::Br(start));

        s.op(start, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
        // Allocate unique wave IDs without assuming the API maps flat
        // thread IDs to a particular physical warp.
        let counter = s.m.element(start, lds, zero);
        let add = s.m.select(start, first_lane, one, zero);
        let old = s.m.atomic(start, Rmw::Add, counter, add);
        let wave = s.op(start, "waveReadLaneFirst.i32", Attr::NoUnwind, s.t.i32, 118, &[old]);
        let limit = s.m.bin(start, Bin::Add, wave, one);
        s.m.terminate(start, Term::Br(header));

        let iter = s.m.phi(header, s.t.i32);
        let sum = s.m.phi(header, s.t.i32);
        let running = s.m.cmp(header, pred::IULT, iter, limit);
        let hi_lane = s.m.cmp(header, pred::EQ, lane, k31);
        let hi_active = s.m.bin(header, Bin::And, running, hi_lane);
        let no = s.m.bool(false);
        let mask = Pair::ballot(&mut s, header, Pair {
            lo: no,
            hi: hi_active,
        });
        let any = mask.any(&mut s, header);
        s.m.terminate(header, Term::CondBr(any, body, exit));

        let (first, _) = mask.first(&mut s, body);
        let upper = s.m.bin(body, Bin::Add, lane, k32);
        let value = Pair { lo: lane, hi: upper }.read_lane(&mut s, body, first);
        let count = mask.count(&mut s, body);
        let total = s.m.bin(body, Bin::Add, value, count);
        let next_sum = s.m.bin(body, Bin::Add, sum, total);
        let next_iter = s.m.bin(body, Bin::Add, iter, one);
        s.m.terminate(body, Term::Br(header));
        s.m.set_phi(iter, vec![(zero, start), (next_iter, body)]);
        s.m.set_phi(sum, vec![(zero, start), (next_sum, body)]);

        s.m.terminate(exit, Term::CondBr(first_lane, write, meet));
        let slot = s.m.bin(write, Bin::Add, wave, one);
        let ptr = s.m.element(write, lds, slot);
        s.m.store(write, ptr, sum);
        s.m.terminate(write, Term::Br(meet));

        s.op(meet, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
        let p0 = s.m.element(meet, lds, one);
        let p1 = s.m.element(meet, lds, two);
        let a = s.m.load(meet, p0);
        let b = s.m.load(meet, p1);
        let h = s.heap_handle(meet, zero, false, false, (11 | 0x1000, 0));
        let base = s.m.bin(meet, Bin::Mul, wave, k32);
        let index = s.m.bin(meet, Bin::Add, base, lane);
        let stride = s.m.i32(16);
        let byte = s.m.bin(meet, Bin::Mul, index, stride);
        let undef = s.m.undef(s.t.i32);
        let channels = s.m.i8(15);
        let align = s.m.i32(4);
        s.op(meet, "rawBufferStore.i32", Attr::NoUnwind, s.t.void, 140,
             &[h, byte, undef, a, b, sum, k63, channels, align]);
        s.m.terminate(meet, Term::Ret);
        let dxil = s.finish();
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = dxil.clone();
            v.validate(&mut copy).expect("paired divergent loop DXIL validation failed");
        }
        let shader = HostShader {
            stage: crate::HwStage::Cs, dxil,
            user_data: Vec::new(), samplers: Vec::new(), static_samplers: Vec::new(),
            root_cbvs: Vec::new(), root_srvs: Vec::new(), root_uavs: Vec::new(),
            root_indexed_views: false,
        };
        let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware device; paired divergent loop execution not tested");
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("paired divergent loop pipeline");
        let out = host.dispatch(&root, &pso, &Dispatch {
            roots: vec![], views: &[View::RawUav(256)], samplers: &[], groups: [1, 1, 1],
        }).expect("paired divergent loop dispatch");
        for n in 0..64usize {
            let expected = [64, 128, if n < 32 { 64 } else { 128 }, 63];
            assert_eq!(&out[0][n * 4..n * 4 + 4], &expected);
        }
    }

}
