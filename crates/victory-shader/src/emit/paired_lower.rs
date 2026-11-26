//! Conservative Wave64-on-Wave32 compute fallback.
//!
//! Each physical wave owns a guest wave, with two VGPR values per lane.
//! Guest EXEC is explicit data. The group scheduler advances each wave
//! independently to the next instruction segment, then all physical threads
//! meet outside divergent code. Guest barriers wait until every guest wave
//! reaches that same barrier. This intentionally favors correctness over
//! throughput and accepts only the instruction/resource forms implemented here.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use super::bitcode::{Attr, Bin, Cast, Rmw, Term, V, pred};
use super::paired::{Pair, mask_bit, prefix_word};
use super::shader::{Shader, flags};
use super::{HostShader, UserData};
use crate::gfx10::{Fields, Operand, NULL};
use crate::ir::{self, EXEC, VCC, Reg, Inst, Exit, Program, Value};
use crate::io::{Io, Input};
use crate::native::{NativeProgram, StageIo, RootType, ResourceType};
use crate::resource::{Resources, Role, Source, Site};
use crate::ShaderError;

const SCC: usize = 256;
const VECTOR: usize = 257;
const STATE: usize = VECTOR + 512;
const DONE: u32 = u32::MAX;
type Result<T> = std::result::Result<T, ShaderError>;
fn unsupported(offset: usize, what: &'static str) -> ShaderError {
    ShaderError::Unsupported { offset, what }
}
fn slot(reg: Reg, half: usize) -> usize {
    match reg {
        Reg::S(n) => n as usize,
        Reg::Scc => SCC,
        Reg::V(n) => VECTOR + 2 * n as usize + half,
    }
}
#[derive(Clone)]
struct State { regs: Vec<V> }
impl State {
    fn get(&self, r: Reg, half: usize) -> V { self.regs[slot(r, half)] }
    fn put(&mut self, r: Reg, half: usize, value: V) { self.regs[slot(r, half)] = value; }
    fn pair(&self, reg: u8) -> Pair {
        Pair { lo: self.get(Reg::V(reg), 0), hi: self.get(Reg::V(reg), 1) }
    }
    fn mask(&self, reg: u8) -> Pair {
        Pair { lo: self.get(Reg::S(reg), 0), hi: self.get(Reg::S(reg + 1), 0) }
    }
    fn set_mask(&mut self, reg: u8, x: Pair) {
        self.put(Reg::S(reg), 0, x.lo);
        self.put(Reg::S(reg + 1), 0, x.hi);
    }
}
#[derive(Clone, Copy)]
struct Segment { block: usize, start: usize, end: usize }

/// All reachable exits must see exactly the same ordered static group
/// barriers. A cycle containing a barrier is deliberately not supported by
/// this first scheduler (a loop without a barrier is fine).
fn barrier_order(p: &Program) -> Result<()> {
    let mut before: Vec<Option<Vec<(usize, usize)>>> = vec![None; p.blocks.len()];
    before[0] = Some(Vec::new());
    let mut work = vec![0];
    let mut exit = None;
    while let Some(b) = work.pop() {
        let mut seq = before[b].clone().unwrap();
        let block = &p.blocks[b];
        seq.extend(block.insts.iter().enumerate().filter(|(_, i)| i.name() == "s_barrier").map(|(i, _)| (b, i)));
        match block.exit {
            Exit::End => {
                if exit.as_ref().is_some_and(|x| *x != seq) {
                    return Err(unsupported(block.offset, "paired waves reach different group barriers"));
                }
                exit = Some(seq.clone());
            }
            Exit::Indirect => return Err(unsupported(block.offset, "paired wave indirect control flow")),
            _ => {}
        }
        for next in block.exit.successors() {
            match &before[next] {
                Some(old) if *old != seq => return Err(unsupported(block.offset, "paired wave barrier in a loop or divergent path")),
                Some(_) => {}
                None => { before[next] = Some(seq.clone()); work.push(next); }
            }
        }
    }
    Ok(())
}
fn segments(p: &Program) -> (Vec<Segment>, Vec<usize>) {
    let mut list = Vec::new();
    let mut first = vec![usize::MAX; p.blocks.len()];
    for b in ir::reverse_postorder(&p.blocks) {
        let block = &p.blocks[b];
        first[b] = list.len();
        let mut start = 0;
        for (i, inst) in block.insts.iter().enumerate() {
            let separate = matches!(inst.fields, Fields::Ds { .. } | Fields::Buffer { .. } | Fields::Smem { .. }) || inst.name() == "s_barrier";
            if separate {
                if start < i { list.push(Segment { block: b, start, end: i }); }
                list.push(Segment { block: b, start: i, end: i + 1 });
                start = i + 1;
            }
        }
        if start < block.insts.len() || first[b] == list.len() {
            list.push(Segment { block: b, start, end: block.insts.len() });
        }
    }
    (list, first)
}

pub(super) fn lower(native: &NativeProgram, p: &Program, io: &Io, res: &Resources) -> Result<HostShader> {
    let StageIo::Cs(cs) = &native.io else { return Err(unsupported(0, "paired wave stage")); };
    let [width, y, z] = cs.threads;
    if !native.wave64 || y != 1 || z != 1 || width == 0 || width > 1024 || !width.is_multiple_of(64) || crate::spill::is_spilled(native) {
        return Err(unsupported(0, "paired waves require complete one-dimensional compute waves without spilled user data"));
    }
    if p.blocks.is_empty() { return Err(unsupported(0, "empty paired wave program")); }
    barrier_order(p)?;
    let lds = cs.lds_bytes() / 4;
    let waves = width / 64;
    if lds + waves + 1 > 8192 { return Err(unsupported(0, "paired scheduler exceeds host group shared memory")); }
    let mut s = Shader::new(false);
    s.threads = Some([width / 2, 1, 1]);
    s.wave_size = Some(32);
    s.lds_dwords = lds + waves + 1;
    s.flags |= flags::WAVE_OPS;
    let at = s.entry;
    let zero = s.m.i32(0);
    let lane = s.op(at, "waveGetLaneIndex", Attr::ReadOnly, s.t.i32, 111, &[]);
    let mut c = Compiler {
        s, native, p, res, cur: at, lane, lds,
        data: Vec::new(), user_handle: None, words: BTreeMap::new(),
        unavailable: HashSet::new(),
    };
    // Descriptor loads are represented by the independently traced descriptor
    // source, never by fake descriptor contents. A numerical read is rejected.
    for block in &p.blocks {
        for inst in &block.insts {
            if inst.name().starts_with("s_load_dword") {
                c.unavailable.extend(inst.writes.iter().map(|(_, v)| *v));
            }
        }
    }
    loop {
        let mut changed = false;
        for block in &p.blocks {
            for phi in &block.phis {
                if phi.args.iter().any(|v| c.unavailable.contains(v)) {
                    changed |= c.unavailable.insert(phi.value);
                }
            }
        }
        if !changed { break; }
    }

    let mut initial = State { regs: vec![zero; STATE] };
    let ones = c.k(u32::MAX);
    initial.set_mask(EXEC, Pair { lo: ones, hi: ones });
    let shared = c.s.lds();
    let flat = c.s.op(at, "flattenedThreadIdInGroup.i32", Attr::ReadNone, c.s.t.i32, 96, &[]);
    let first_thread = c.cmp(pred::EQ, flat, zero);
    let first_lane = c.cmp(pred::EQ, lane, zero);
    let counter_index = c.k(lds);
    let counter = c.s.m.element(at, shared, counter_index);
    let (init, join) = c.guard(first_thread);
    c.s.m.store(c.cur, counter, zero);
    c.unguard(init, join, None);
    c.sync();
    let one = c.k(1);
    let add = c.s.m.select(c.cur, first_lane, one, zero);
    let old = c.s.m.atomic(c.cur, Rmw::Add, counter, add);
    let wave = c.s.op(c.cur, "waveReadLaneFirst.i32", Attr::NoUnwind, c.s.t.i32, 118, &[old]);
    let k64 = c.k(64);
    let wave_base = c.bin(Bin::Mul, wave, k64);
    let thread = c.bin(Bin::Add, wave_base, lane);
    let k32 = c.k(32);
    let upper = c.bin(Bin::Add, thread, k32);
    // All input loads are emitted before the scheduler header. The entry
    // block holds the handle and constant loads, while wave IDs are assigned
    // after the initialization rendezvous.
    for &(reg, value) in &p.inputs {
        if matches!(reg, Reg::S(EXEC | 127)) { continue; }
        if let Some(&(_, input)) = io.inputs.iter().find(|(v, _)| *v == value) {
            match input {
                Input::ThreadIdInGroup(0) => {
                    initial.put(reg, 0, thread);
                    initial.put(reg, 1, upper);
                }
                Input::ThreadIdInGroup(1 | 2) => {}
                Input::GroupId(component) => {
                    let k = c.k(component as u32);
                    let v = c.s.op(c.cur, "groupId.i32", Attr::ReadNone, c.s.t.i32, 94, &[k]);
                    initial.put(reg, 0, v);
                }
                _ => return Err(unsupported(0, "paired wave system input")),
            }
        } else if let Some(&(_, param, dword)) = res.params.iter().find(|(v, _, _)| *v == value) {
            let rp = native.layout.get(param).ok_or(unsupported(0, "paired root input"))?;
            let v = match rp.root {
                RootType::Constants | RootType::Table => c.user(rp.root, rp.slot, dword)?,
                RootType::ResourceHeap | RootType::SamplerHeap => zero,
                _ => return Err(unsupported(0, "paired root descriptor input")),
            };
            initial.put(reg, 0, v);
        } else if matches!(reg, Reg::S(n) if n < cs.initialized_sgprs()) || reg == Reg::Scc {
            return Err(unsupported(0, "untracked initialized paired scalar input"));
        }
        // Uninitialized vector/scalar registers have no defined guest
        // content; zero matches the ordinary lowerer's convention.
    }
    let mut slots = BTreeSet::from([EXEC as usize, EXEC as usize + 1, SCC]);
    for block in &p.blocks {
        for inst in &block.insts {
            for &(reg, _) in inst.reads.iter().chain(&inst.writes) {
                slots.insert(slot(reg, 0));
                if matches!(reg, Reg::V(_)) { slots.insert(slot(reg, 1)); }
            }
        }
    }
    let (segments, first) = segments(p);
    let entry_tail = c.cur;
    let header = c.s.m.new_block();
    let published = c.s.m.new_block();
    let finish = c.s.m.new_block();
    let dispatch = c.s.m.new_block();
    let merge = c.s.m.new_block();
    c.s.m.terminate(entry_tail, Term::Br(header));
    c.cur = header;
    let pc = c.s.m.phi(header, c.s.t.i32);
    let mut current = initial.clone();
    for &i in &slots { current.regs[i] = c.s.m.phi(header, c.s.t.i32); }
    let pc_base = c.k(lds + 1);
    let pc_index = c.bin(Bin::Add, pc_base, wave);
    let pc_ptr = c.s.m.element(header, shared, pc_index);
    let publish = c.s.m.new_block();
    c.s.m.terminate(header, Term::CondBr(first_lane, publish, published));
    c.s.m.store(publish, pc_ptr, pc);
    c.s.m.terminate(publish, Term::Br(published));
    c.cur = published;
    c.sync();
    let done = c.k(DONE);
    let mut all_done = c.s.m.bool(true);
    let mut all_here = c.s.m.bool(true);
    for n in 0..waves {
        let index = c.k(lds + 1 + n);
        let ptr = c.s.m.element(c.cur, shared, index);
        let other = c.s.m.load(c.cur, ptr);
        let ended = c.cmp(pred::EQ, other, done);
        let same = c.cmp(pred::EQ, other, pc);
        all_done = c.bin(Bin::And, all_done, ended);
        all_here = c.bin(Bin::And, all_here, same);
    }
    c.s.m.terminate(published, Term::CondBr(all_done, finish, dispatch));
    c.s.m.terminate(finish, Term::Ret);
    let mut next_check = dispatch;
    let mut incoming: Vec<(usize, V, State)> = Vec::with_capacity(segments.len() + 1);
    for (id, seg) in segments.iter().copied().enumerate() {
        c.cur = next_check;
        let idv = c.k(id as u32);
        let is = c.cmp(pred::EQ, pc, idv);
        let body = c.s.m.new_block();
        next_check = c.s.m.new_block();
        c.s.m.terminate(c.cur, Term::CondBr(is, body, next_check));
        c.cur = body;
        let mut st = current.clone();
        let block = &p.blocks[seg.block];
        let mut barrier = false;
        for i in seg.start..seg.end {
            let inst = &block.insts[i];
            if inst.name() == "s_barrier" { barrier = true; continue; }
            c.instruction(&mut st, inst, (seg.block, i))?;
        }
        let mut next = if seg.end < block.insts.len() {
            c.k(id as u32 + 1)
        } else {
            match block.exit {
                Exit::End => done,
                Exit::Jump(b) => c.k(first[b] as u32),
                Exit::Branch { taken, not_taken } => {
                    let inst = block.insts.last().ok_or(unsupported(block.offset, "paired branch without instruction"))?;
                    let cond = c.branch(&st, inst)?;
                    let a = c.k(first[taken] as u32);
                    let b = c.k(first[not_taken] as u32);
                    c.s.m.select(c.cur, cond, a, b)
                }
                Exit::Indirect => return Err(unsupported(block.offset, "paired indirect branch")),
            }
        };
        if barrier { next = c.s.m.select(c.cur, all_here, next, pc); }
        let tail = c.cur;
        c.s.m.terminate(tail, Term::Br(merge));
        incoming.push((tail, next, st));
    }
    // Completed waves participate in all remaining rendezvous without
    // executing guest code or producing memory side effects.
    c.s.m.terminate(next_check, Term::Br(merge));
    incoming.push((next_check, pc, current.clone()));
    c.cur = merge;
    let next_pc = c.s.m.phi(merge, c.s.t.i32);
    c.s.m.set_phi(next_pc, incoming.iter().map(|(b, pc, _)| (*pc, *b)).collect());
    let mut next_regs = current.clone();
    for &i in &slots {
        if incoming.iter().all(|(_, _, st)| st.regs[i] == current.regs[i]) { continue; }
        let phi = c.s.m.phi(merge, c.s.t.i32);
        c.s.m.set_phi(phi, incoming.iter().map(|(b, _, st)| (st.regs[i], *b)).collect());
        next_regs.regs[i] = phi;
    }
    // No group barrier is emitted inside a wave-dependent case.
    // This fence orders all guest shared-memory operations, also across
    // the two halves of a guest wave, before the next instruction segment.
    c.sync();
    c.s.m.terminate(merge, Term::Br(header));
    let initial_pc = c.k(first[0] as u32);
    c.s.m.set_phi(pc, vec![(initial_pc, entry_tail), (next_pc, merge)]);
    for i in slots {
        c.s.m.set_phi(current.regs[i], vec![(initial.regs[i], entry_tail), (next_regs.regs[i], merge)]);
    }
    let dxil = c.s.finish();
    Ok(HostShader {
        stage: native.stage, dxil, user_data: c.data, samplers: Vec::new(),
        static_samplers: Vec::new(), root_cbvs: Vec::new(), root_srvs: Vec::new(),
        root_uavs: Vec::new(), root_indexed_views: res.root_indexed_views.clone(),
    })
}

struct Compiler<'a> {
    s: Shader,
    native: &'a NativeProgram,
    p: &'a Program,
    res: &'a Resources,
    cur: usize,
    lane: V,
    lds: u32,
    data: Vec<UserData>,
    user_handle: Option<V>,
    words: BTreeMap<(u8, u32, bool), V>,
    unavailable: HashSet<Value>,
}
impl Compiler<'_> {
    fn k(&mut self, n: u32) -> V { self.s.m.i32(n) }
    fn bin(&mut self, op: Bin, a: V, b: V) -> V { self.s.m.bin(self.cur, op, a, b) }
    fn cmp(&mut self, op: u8, a: V, b: V) -> V { self.s.m.cmp(self.cur, op, a, b) }
    fn bit(&mut self, x: V) -> V { self.s.m.cast(self.cur, Cast::ZExt, x, self.s.t.i32) }
    fn nonzero(&mut self, x: V) -> V { let zero = self.k(0); self.cmp(pred::NE, x, zero) }
    fn not(&mut self, x: V) -> V { let ones = self.k(u32::MAX); self.bin(Bin::Xor, x, ones) }
    fn sync(&mut self) {
        let mode = self.k(11); // SyncThreadGroup | UAVFenceGlobal | TGSMFence
        self.s.op(self.cur, "barrier", Attr::NoDuplicate, self.s.t.void, 80, &[mode]);
    }
    fn guard(&mut self, cond: V) -> (usize, usize) {
        let pre = self.cur;
        let body = self.s.m.new_block();
        let join = self.s.m.new_block();
        self.s.m.terminate(pre, Term::CondBr(cond, body, join));
        self.cur = body;
        (pre, join)
    }
    fn unguard(&mut self, pre: usize, join: usize, result: Option<(V, V)>) -> Option<V> {
        let body = self.cur;
        self.s.m.terminate(body, Term::Br(join));
        self.cur = join;
        result.map(|(yes, no)| {
            let v = self.s.m.phi(join, self.s.t.i32);
            self.s.m.set_phi(v, vec![(no, pre), (yes, body)]);
            v
        })
    }
    fn logical_lane(&mut self, half: usize) -> V {
        if half == 0 { self.lane } else { let k = self.k(32); self.bin(Bin::Add, self.lane, k) }
    }
    fn active(&mut self, st: &State, half: usize) -> V {
        let exec = st.mask(EXEC);
        let lane = self.lane;
        mask_bit(&mut self.s, self.cur, if half == 0 { exec.lo } else { exec.hi }, None, lane)
    }
    fn read(&self, st: &State, inst: &Inst, reg: Reg, half: usize) -> Result<V> {
        if inst.read(reg).is_some_and(|v| self.unavailable.contains(&v)) {
            return Err(unsupported(inst.machine.offset, "paired descriptor read as numerical data"));
        }
        Ok(st.get(reg, half))
    }
    fn op(&mut self, st: &State, inst: &Inst, op: &Operand, half: usize, word: u8) -> Result<V> {
        let bad = || unsupported(inst.machine.offset, "paired operand");
        Ok(match *op {
            Operand::Scalar { code: NULL, .. } => self.k(0),
            Operand::Scalar { code, dwords } if word < dwords => self.read(st, inst, Reg::S(code.checked_add(word).ok_or_else(bad)?), 0)?,
            Operand::Vector { reg, dwords } if word < dwords => self.read(st, inst, Reg::V(reg.checked_add(word).ok_or_else(bad)?), half)?,
            Operand::Const(n @ 128..=192) => self.k(if word == 0 { (n - 128) as u32 } else { 0 }),
            Operand::Const(n @ 193..=208) => self.k(if word == 0 { (192i32 - n as i32) as u32 } else { u32::MAX }),
            Operand::Literal(n) if word == 0 => self.k(n),
            _ => return Err(bad()),
        })
    }
    fn scalar_mask(&mut self, st: &State, inst: &Inst, op: &Operand) -> Result<Pair> {
        Ok(Pair { lo: self.op(st, inst, op, 0, 0)?, hi: self.op(st, inst, op, 0, 1)? })
    }
    fn vector(&mut self, st: &State, inst: &Inst, op: &Operand) -> Result<Pair> {
        Ok(Pair { lo: self.op(st, inst, op, 0, 0)?, hi: self.op(st, inst, op, 1, 0)? })
    }
    fn write_v(&mut self, st: &mut State, reg: u8, value: Pair) {
        let active = Pair { lo: self.active(st, 0), hi: self.active(st, 1) };
        let old = st.pair(reg);
        let value = old.write(&mut self.s, self.cur, active, value);
        st.put(Reg::V(reg), 0, value.lo);
        st.put(Reg::V(reg), 1, value.hi);
    }
    fn set_scc(&mut self, st: &mut State, cond: V) {
        let v = self.bit(cond);
        st.put(Reg::Scc, 0, v);
    }
    fn branch(&mut self, st: &State, inst: &Inst) -> Result<V> {
        let name = inst.name();
        let test = match name {
            "s_cbranch_scc0" | "s_cbranch_scc1" => self.nonzero(st.get(Reg::Scc, 0)),
            "s_cbranch_vccz" | "s_cbranch_vccnz" => st.mask(VCC).any(&mut self.s, self.cur),
            "s_cbranch_execz" | "s_cbranch_execnz" => st.mask(EXEC).any(&mut self.s, self.cur),
            _ => return Err(unsupported(inst.machine.offset, name)),
        };
        Ok(if matches!(name, "s_cbranch_scc0" | "s_cbranch_vccz" | "s_cbranch_execz") {
            let yes = self.s.m.bool(true); self.bin(Bin::Xor, test, yes)
        } else { test })
    }
    fn instruction(&mut self, st: &mut State, inst: &Inst, site: Site) -> Result<()> {
        match &inst.fields {
            Fields::Sopp { .. } if matches!(inst.name(), "s_branch" | "s_endpgm" | "s_endpgm_saved" | "s_code_end" | "s_cbranch_scc0" | "s_cbranch_scc1" | "s_cbranch_vccz" | "s_cbranch_vccnz" | "s_cbranch_execz" | "s_cbranch_execnz") => Ok(()),
            Fields::Sop { .. } => self.scalar(st, inst),
            Fields::Valu(_) => self.valu(st, inst),
            Fields::Ds { .. } => self.ds(st, inst),
            Fields::Smem { .. } => self.smem(st, inst, site),
            Fields::Buffer { .. } => self.buffer(st, inst, site),
            _ => Err(unsupported(inst.machine.offset, inst.name())),
        }
    }
    fn user(&mut self, root: RootType, slot: u8, dword: u32) -> Result<V> {
        if !matches!(root, RootType::Constants | RootType::Table) { return Err(unsupported(0, "paired root word")); }
        let key = (slot, dword, root == RootType::Table);
        if let Some(&v) = self.words.get(&key) { return Ok(v); }
        let index = self.data.len() as u32;
        self.data.push(UserData { index, slot, root, dword });
        self.s.user_data = index + 1;
        let h = if let Some(h) = self.user_handle { h } else {
            let h = self.s.user_data_handle(); self.user_handle = Some(h); h
        };
        let at = self.s.entry;
        let row = self.k(index / 4);
        let r = self.s.op(at, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.s.t.cbuf_ret, 59, &[h, row]);
        let v = self.s.m.extract(at, r, index % 4);
        self.words.insert(key, v);
        Ok(v)
    }
    // INSTRUCTION_METHODS
}
