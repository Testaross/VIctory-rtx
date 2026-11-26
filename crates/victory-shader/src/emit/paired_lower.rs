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
    fn scalar(&mut self, st: &mut State, inst: &Inst) -> Result<()> {
        let Fields::Sop { dst, src, imm: None } = &inst.fields else {
            return Err(unsupported(inst.machine.offset, inst.name()));
        };
        let name = inst.name();
        let bad = || unsupported(inst.machine.offset, name);
        if let Some(cmp) = name.strip_prefix("s_cmp_") {
            let op = compare(cmp).ok_or_else(bad)?;
            let a = self.op(st, inst, src.first().ok_or_else(bad)?, 0, 0)?;
            let b = self.op(st, inst, src.get(1).ok_or_else(bad)?, 0, 0)?;
            let x = self.cmp(op, a, b);
            self.set_scc(st, x);
            return Ok(());
        }
        let Some(Operand::Scalar { code, dwords }) = *dst else { return Err(bad()); };
        if code == NULL { return Err(bad()); }
        let aop = src.first().ok_or_else(bad)?;
        if name == "s_and_saveexec_b64" {
            let a = self.scalar_mask(st, inst, aop)?;
            let old = st.mask(EXEC);
            let lo = self.bin(Bin::And, old.lo, a.lo);
            let hi = self.bin(Bin::And, old.hi, a.hi);
            let x = Pair { lo, hi };
            st.set_mask(code, old);
            st.set_mask(EXEC, x);
            let any = x.any(&mut self.s, self.cur);
            self.set_scc(st, any);
            return Ok(());
        }
        if dwords == 2 && name.ends_with("_b64") {
            let a = self.scalar_mask(st, inst, aop)?;
            let x = if name == "s_mov_b64" { a } else {
                let b = self.scalar_mask(st, inst, src.get(1).ok_or_else(bad)?)?;
                let op = match name {
                    "s_and_b64" | "s_andn2_b64" => Bin::And,
                    "s_or_b64" => Bin::Or,
                    "s_xor_b64" => Bin::Xor,
                    _ => return Err(bad()),
                };
                let b = if name == "s_andn2_b64" {
                    Pair { lo: self.not(b.lo), hi: self.not(b.hi) }
                } else { b };
                Pair { lo: self.bin(op, a.lo, b.lo), hi: self.bin(op, a.hi, b.hi) }
            };
            st.set_mask(code, x);
            if inst.write(Reg::Scc).is_some() {
                let any = x.any(&mut self.s, self.cur);
                self.set_scc(st, any);
            }
            return Ok(());
        }
        if dwords != 1 { return Err(bad()); }
        if name == "s_bcnt1_i32_b64" {
            let a = self.scalar_mask(st, inst, aop)?;
            let x = a.count(&mut self.s, self.cur);
            st.put(Reg::S(code), 0, x);
            if inst.write(Reg::Scc).is_some() {
                let nz = self.nonzero(x); self.set_scc(st, nz);
            }
            return Ok(());
        }
        let a = self.op(st, inst, aop, 0, 0)?;
        let (x, scc) = if name == "s_mov_b32" { (a, None) } else {
            let b = self.op(st, inst, src.get(1).ok_or_else(bad)?, 0, 0)?;
            let x = match name {
                "s_and_b32" => self.bin(Bin::And, a, b),
                "s_or_b32" => self.bin(Bin::Or, a, b),
                "s_xor_b32" => self.bin(Bin::Xor, a, b),
                "s_add_i32" | "s_add_u32" => self.bin(Bin::Add, a, b),
                "s_lshl_b32" => {
                    let mask = self.k(31);
                    let shift = self.bin(Bin::And, b, mask);
                    self.bin(Bin::Shl, a, shift)
                }
                _ => return Err(bad()),
            };
            let scc = match name {
                "s_add_u32" => self.cmp(pred::IULT, x, a),
                "s_add_i32" => {
                    let ab = self.bin(Bin::Xor, a, b);
                    let ax = self.bin(Bin::Xor, a, x);
                    let ab = self.not(ab);
                    let overflow = self.bin(Bin::And, ab, ax);
                    let zero = self.k(0);
                    self.cmp(pred::SLT, overflow, zero)
                }
                _ => self.nonzero(x),
            };
            (x, Some(scc))
        };
        st.put(Reg::S(code), 0, x);
        if inst.write(Reg::Scc).is_some() {
            self.set_scc(st, scc.ok_or_else(bad)?);
        }
        Ok(())
    }

    fn valu(&mut self, st: &mut State, inst: &Inst) -> Result<()> {
        let Fields::Valu(v) = &inst.fields else { unreachable!() };
        let name = inst.name();
        let bad = || unsupported(inst.machine.offset, name);
        if v.neg != 0 || v.abs != 0 || v.neg_hi != 0 || v.opsel != 0 || v.clamp || v.omod != 0 || v.sdwa.is_some() || v.dpp.is_some() || v.carry.is_some() {
            return Err(bad());
        }
        let src = |n| v.src.get(n).ok_or_else(bad);
        if let Some(rest) = name.strip_prefix("v_cmp_").or_else(|| name.strip_prefix("v_cmpx_")) {
            let op = compare(rest).ok_or_else(bad)?;
            let a = self.vector(st, inst, src(0)?)?;
            let b = self.vector(st, inst, src(1)?)?;
            let lo = self.cmp(op, a.lo, b.lo);
            let hi = self.cmp(op, a.hi, b.hi);
            let active_lo = self.active(st, 0);
            let active_hi = self.active(st, 1);
            let active = Pair {
                lo: self.bin(Bin::And, lo, active_lo),
                hi: self.bin(Bin::And, hi, active_hi),
            };
            let mask = Pair::ballot(&mut self.s, self.cur, active);
            let dst = if name.starts_with("v_cmpx_") { EXEC } else {
                let Some(Operand::Scalar { code, dwords: 2 }) = v.dst else { return Err(bad()); };
                code
            };
            st.set_mask(dst, mask);
            return Ok(());
        }
        if matches!(name, "v_readfirstlane_b32" | "v_readlane_b32") {
            let Some(Operand::Scalar { code, dwords: 1 }) = v.dst else { return Err(bad()); };
            let value = self.vector(st, inst, src(0)?)?;
            let lane = if name == "v_readfirstlane_b32" {
                let mask = st.mask(EXEC);
                mask.first(&mut self.s, self.cur).0
            } else {
                let idx = self.op(st, inst, src(1)?, 0, 0)?;
                let mask = self.k(63);
                self.bin(Bin::And, idx, mask)
            };
            let x = value.read_lane(&mut self.s, self.cur, lane);
            st.put(Reg::S(code), 0, x);
            return Ok(());
        }
        let Some(Operand::Vector { reg, dwords: 1 }) = v.dst else { return Err(bad()); };
        let a = self.vector(st, inst, src(0)?)?;
        let mut out = [a.lo, a.hi];
        for (half, result) in out.iter_mut().enumerate() {
            let a = *result;
            let x = match name {
                "v_mov_b32" => a,
                "v_not_b32" => self.not(a),
                _ => {
                    let b = self.op(st, inst, src(1)?, half, 0)?;
                    match name {
                        "v_add_nc_u32" => self.bin(Bin::Add, a, b),
                        "v_sub_nc_u32" => self.bin(Bin::Sub, a, b),
                        "v_subrev_nc_u32" => self.bin(Bin::Sub, b, a),
                        "v_and_b32" => self.bin(Bin::And, a, b),
                        "v_or_b32" => self.bin(Bin::Or, a, b),
                        "v_xor_b32" => self.bin(Bin::Xor, a, b),
                        "v_min_u32" | "v_max_u32" | "v_min_i32" | "v_max_i32" => {
                            let op = match name {
                                "v_min_u32" => pred::IULT, "v_max_u32" => pred::IUGT,
                                "v_min_i32" => pred::SLT, _ => pred::SGT,
                            };
                            let take = self.cmp(op, a, b);
                            self.s.m.select(self.cur, take, a, b)
                        }
                        "v_lshlrev_b32" | "v_lshrrev_b32" | "v_ashrrev_i32" => {
                            let k31 = self.k(31);
                            let shift = self.bin(Bin::And, a, k31);
                            let op = match name {
                                "v_lshlrev_b32" => Bin::Shl, "v_lshrrev_b32" => Bin::LShr, _ => Bin::AShr,
                            };
                            self.bin(op, b, shift)
                        }
                        "v_add3_u32" | "v_or3_b32" => {
                            let c = self.op(st, inst, src(2)?, half, 0)?;
                            let op = if name == "v_add3_u32" { Bin::Add } else { Bin::Or };
                            let ab = self.bin(op, a, b);
                            self.bin(op, ab, c)
                        }
                        "v_cndmask_b32" => {
                            let m = self.scalar_mask(st, inst, src(2)?)?;
                            let lane = self.lane;
                            let word = if half == 0 { m.lo } else { m.hi };
                            let take = mask_bit(&mut self.s, self.cur, word, None, lane);
                            self.s.m.select(self.cur, take, b, a)
                        }
                        "v_mbcnt_lo_u32_b32" | "v_mbcnt_hi_u32_b32" => {
                            let below = if name == "v_mbcnt_lo_u32_b32" { self.logical_lane(half) }
                                else if half == 0 { self.k(0) } else { self.lane };
                            let n = prefix_word(&mut self.s, self.cur, a, below);
                            self.bin(Bin::Add, n, b)
                        }
                        _ => return Err(bad()),
                    }
                }
            };
            *result = x;
        }
        self.write_v(st, reg, Pair { lo: out[0], hi: out[1] });
        Ok(())
    }

    /// Keep guest LDS bounds separate from scheduler scratch; invalid
    /// guest addresses cannot overwrite program counters.
    fn lds_address(&mut self, base: V, offset: u32, enabled: V) -> (V, V) {
        let off = self.k(offset);
        let byte = self.bin(Bin::Add, base, off);
        let two = self.k(2);
        let index = self.bin(Bin::LShr, byte, two);
        let size = self.k(self.lds);
        let inside = self.cmp(pred::IULT, index, size);
        let guard = self.bin(Bin::And, inside, enabled);
        (index, guard)
    }
    fn ds(&mut self, st: &mut State, inst: &Inst) -> Result<()> {
        let Fields::Ds { dst, addr, data, offset0, offset1, gds } = &inst.fields else { unreachable!() };
        let name = inst.name();
        let bad = || unsupported(inst.machine.offset, name);
        if *gds || self.lds == 0 { return Err(bad()); }
        let addr = self.vector(st, inst, addr.as_ref().ok_or_else(bad)?)?;
        let (read, write, atomic, offsets): (bool, bool, bool, Vec<u32>) = match name {
            "ds_read_b32" => (true, false, false, vec![u32::from(*offset0) | u32::from(*offset1) << 8]),
            "ds_write_b32" => (false, true, false, vec![u32::from(*offset0) | u32::from(*offset1) << 8]),
            "ds_read2_b32" | "ds_read2st64_b32" | "ds_write2_b32" | "ds_write2st64_b32" => {
                let scale = if name.contains("st64") { 256 } else { 4 };
                (name.starts_with("ds_read"), name.starts_with("ds_write"), false,
                 vec![u32::from(*offset0) * scale, u32::from(*offset1) * scale])
            }
            "ds_cmpst_rtn_b32" | "ds_add_rtn_u32" => (false, false, true, vec![u32::from(*offset0) | u32::from(*offset1) << 8]),
            _ => return Err(bad()),
        };
        let dst = match *dst {
            Some(Operand::Vector { reg, .. }) => Some(reg),
            None if write => None,
            _ => return Err(bad()),
        };
        let old_state = st.clone();
        let shared = self.s.lds();
        for half in 0..2 {
            let base = if half == 0 { addr.lo } else { addr.hi };
            let enabled = self.active(&old_state, half);
            for (n, off) in offsets.iter().copied().enumerate() {
                let (index, valid) = self.lds_address(base, off, enabled);
                let reg = dst.map(|r| r.checked_add(n as u8).ok_or_else(bad)).transpose()?;
                let old = reg.map(|r| old_state.get(Reg::V(r), half)).unwrap_or_else(|| self.k(0));
                let zero = self.k(0);
                let outside = self.s.m.select(self.cur, enabled, zero, old);
                // Evaluate operands before entering a per-half conditional.
                let value = if write {
                    Some(self.op(&old_state, inst, data.get(n).ok_or_else(bad)?, half, 0)?)
                } else if atomic {
                    Some(self.op(&old_state, inst, data.first().ok_or_else(bad)?, half, 0)?)
                } else { None };
                let exchange = if name == "ds_cmpst_rtn_b32" {
                    Some(self.op(&old_state, inst, data.get(1).ok_or_else(bad)?, half, 0)?)
                } else { None };
                let (pre, join) = self.guard(valid);
                let ptr = self.s.m.element(self.cur, shared, index);
                let result = if read {
                    Some(self.s.m.load(self.cur, ptr))
                } else if write {
                    self.s.m.store(self.cur, ptr, value.ok_or_else(bad)?);
                    None
                } else if let Some(new) = exchange {
                    Some(self.s.m.cmpxchg(self.cur, ptr, value.ok_or_else(bad)?, new))
                } else {
                    Some(self.s.m.atomic(self.cur, Rmw::Add, ptr, value.ok_or_else(bad)?))
                };
                let result = self.unguard(pre, join, result.map(|v| (v, outside)));
                if let (Some(reg), Some(result)) = (reg, result) {
                    st.put(Reg::V(reg), half, result);
                }
            }
        }
        Ok(())
    }

    /// Only fixed table descriptor loads are accepted. Descriptor bytes
    /// cannot be read as ordinary data in this backend.
    fn descriptor(&mut self, site: Site) -> Result<(V, ResourceType, u32)> {
        let at = self.p.blocks[site.0].insts[site.1].machine.offset;
        let bad = || unsupported(at, "paired descriptor is not a fixed table entry");
        let desc = self.res.descriptors.iter().find(|d| d.site == site && d.role == Role::Resource).ok_or_else(bad)?;
        let Source::Loaded { load, first } = &desc.source else { return Err(bad()); };
        let (_, addr) = self.res.scalar.iter().find(|(s, _)| s == load).ok_or_else(bad)?;
        let rp = self.native.layout.get(addr.param).ok_or_else(bad)?;
        if rp.root != RootType::Table || addr.dynamic.is_some() { return Err(bad()); }
        let byte = addr.offset.checked_add(u32::from(*first) * 4).ok_or_else(bad)?;
        if !byte.is_multiple_of(32) { return Err(bad()); }
        let used = self.native.used.iter().find(|u| u.root == RootType::Table && u.user_data == rp.user_data && u.table_offset_dwords == byte / 4).ok_or_else(bad)?;
        let (kind, stride, slot) = (used.kind, used.stride, rp.slot);
        let base = self.user(RootType::Table, slot, 0)?;
        let offset = self.k(byte);
        let address = self.bin(Bin::Add, base, offset);
        let five = self.k(5);
        let index = self.bin(Bin::LShr, address, five);
        Ok((index, kind, stride))
    }
    fn smem(&mut self, st: &mut State, inst: &Inst, site: Site) -> Result<()> {
        let Fields::Smem { data, offset, soffset, .. } = &inst.fields else { unreachable!() };
        let name = inst.name();
        let bad = || unsupported(inst.machine.offset, name);
        if name.starts_with("s_load_dword") {
            let (_, a) = self.res.scalar.iter().find(|(s, _)| *s == site).ok_or_else(bad)?;
            if a.dynamic.is_some() || !self.native.layout.get(a.param).is_some_and(|p| p.root == RootType::Table) {
                return Err(bad());
            }
            // The resource tracker resolves the descriptor's use. The
            // SSA unavailable set rejects any attempt to read its bits.
            return Ok(());
        }
        let count = dwords(name.strip_prefix("s_buffer_load_dword").ok_or_else(bad)?).ok_or_else(bad)?;
        let Some(Operand::Scalar { code, .. }) = *data else { return Err(bad()); };
        let (index, kind, _) = self.descriptor(site)?;
        if kind != ResourceType::Cbv { return Err(bad()); }
        if !matches!(soffset, Operand::Scalar { code: NULL, .. } | Operand::Const(128)) || *offset < 0 || *offset % 4 != 0 { return Err(bad()); }
        let h = self.s.heap_handle(self.cur, index, false, false, (13, 65536));
        for n in 0..count {
            let dword = *offset as u32 / 4 + n as u32;
            let row = self.k(dword / 4);
            let r = self.s.op(self.cur, "cbufferLoadLegacy.i32", Attr::ReadOnly, self.s.t.cbuf_ret, 59, &[h, row]);
            let value = self.s.m.extract(self.cur, r, dword % 4);
            st.put(Reg::S(code.checked_add(n).ok_or_else(bad)?), 0, value);
        }
        Ok(())
    }
    fn buffer(&mut self, st: &mut State, inst: &Inst, site: Site) -> Result<()> {
        let Fields::Buffer { data, addr, soffset, offset, offen, idxen, format, lds, tfe, .. } = &inst.fields else { unreachable!() };
        let name = inst.name();
        let bad = || unsupported(inst.machine.offset, name);
        if format.is_some() || *lds || *tfe { return Err(bad()); }
        let (store, count) = if let Some(n) = name.strip_prefix("buffer_store_dword") {
            (true, dwords(n).ok_or_else(bad)?)
        } else if let Some(n) = name.strip_prefix("buffer_load_dword") {
            (false, dwords(n).ok_or_else(bad)?)
        } else { return Err(bad()); };
        let Some(Operand::Vector { reg, .. }) = *data else { return Err(bad()); };
        let (index, kind, stride) = self.descriptor(site)?;
        let uav = kind == ResourceType::Uav;
        if !(uav || !store && kind == ResourceType::Srv) { return Err(bad()); }
        let structured = stride > 1;
        if structured && (stride < 4 || !stride.is_multiple_of(4) || (!*idxen && *offen)) {
            return Err(bad());
        }
        let props = (if structured { 12 } else { 11 } | if uav { 0x1000 } else { 0 }, if structured { stride } else { 0 });
        let handle = self.s.heap_handle(self.cur, index, false, false, props);
        let old_state = st.clone();
        for half in 0..2 {
            let so = self.op(&old_state, inst, soffset, 0, 0)?;
            let imm = self.k(u32::from(*offset));
            let mut off = self.bin(Bin::Add, so, imm);
            if *offen {
                let v = self.op(&old_state, inst, addr, half, u8::from(*idxen))?;
                off = self.bin(Bin::Add, off, v);
            }
            let idx = if *idxen { self.op(&old_state, inst, addr, half, 0)? } else { self.k(0) };
            let undef = self.s.m.undef(self.s.t.i32);
            let (a, b) = if structured { (idx, off) } else {
                let byte = self.bin(Bin::Add, idx, off);
                (byte, undef)
            };
            let mask = self.s.m.i8((1u8 << count) - 1);
            let align = self.k(4);
            let mut vals = Vec::new();
            if store {
                for n in 0..count {
                    vals.push(self.read(&old_state, inst, Reg::V(reg.checked_add(n).ok_or_else(bad)?), half)?);
                }
            }
            let active = self.active(&old_state, half);
            let (pre, join) = self.guard(active);
            if store {
                vals.resize(4, undef);
                self.s.op(self.cur, "rawBufferStore.i32", Attr::NoUnwind, self.s.t.void, 140,
                    &[handle, a, b, vals[0], vals[1], vals[2], vals[3], mask, align]);
                self.unguard(pre, join, None);
            } else {
                let ty = self.s.m.named_struct("dx.types.ResRet.i32", vec![self.s.t.i32; 5]);
                let r = self.s.op(self.cur, "rawBufferLoad.i32", Attr::ReadOnly, ty, 139, &[handle, a, b, mask, align]);
                let body = self.cur;
                let mut values = Vec::new();
                for n in 0..count { values.push(self.s.m.extract(body, r, u32::from(n))); }
                self.unguard(pre, join, None);
                for (n, value) in values.into_iter().enumerate() {
                    let r = reg.checked_add(n as u8).ok_or_else(bad)?;
                    let old = old_state.get(Reg::V(r), half);
                    let phi = self.s.m.phi(join, self.s.t.i32);
                    self.s.m.set_phi(phi, vec![(old, pre), (value, body)]);
                    st.put(Reg::V(r), half, phi);
                }
            }
        }
        Ok(())
    }
}

fn dwords(s: &str) -> Option<u8> {
    match s { "" => Some(1), "x2" => Some(2), "x3" => Some(3), "x4" => Some(4), _ => None }
}
fn compare(name: &str) -> Option<u8> {
    let (op, ty) = name.rsplit_once('_')?;
    if !matches!(ty, "u32" | "i32") { return None; }
    Some(match op {
        "eq" => pred::EQ, "ne" | "lg" => pred::NE,
        "lt" => if ty == "i32" { pred::SLT } else { pred::IULT },
        "le" => if ty == "i32" { pred::SLE } else { pred::IULE },
        "gt" => if ty == "i32" { pred::SGT } else { pred::IUGT },
        "ge" => if ty == "i32" { pred::SGE } else { pred::IUGE },
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::{ComputeIo, HwStage, RootParam, UsedResource};

    // Independently authored guest program: one writer in each wave,
    // a wave-dependent scalar loop, a common barrier, then every lane
    // stores the logical wave's result. Uses the real decoder and
    // resource tracker, not hand-authored DXIL.
    fn synthetic() -> NativeProgram {
        let words = [
            0xBE81_0380, // s_mov_b32 s1, 0
            0xF408_0200, 0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0
            0xD765_0001, 0x0001_00C1, // v_mbcnt_lo_u32_b32 v1, -1, 0
            0xD766_0001, 0x0002_02C1, // v_mbcnt_hi_u32_b32 v1, -1, v1
            0xD760_0007, 0x0001_7F00, // v_readlane_b32 s7, v0, 63
            0x7D84_02BF, // v_cmp_eq_u32 vcc, 63, v1
            0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
            0xBF88_0007, // s_cbranch_execz restore
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0xD834_0000, 0x0000_0003, // ds_write_b32 v3, v0
            0xD8D8_0000, 0x0200_0003, // ds_read_b32 v2, v3
            0xD760_0006, 0x0001_7F02, // v_readlane_b32 s6, v2, 63
            0xBEFE_040C, // restore: s_mov_b64 exec, s[12:13]
            0xBF06_BF07, // s_cmp_eq_u32 s7, 63
            0xBF84_0004, // s_cbranch_scc0 meet
            0x8106_8106, // loop: s_add_i32 s6, s6, 1
            0xBF0A_FF06, 66, // s_cmp_lt_u32 s6, 66
            0xBF85_FFFC, // s_cbranch_scc1 loop
            0xBF8A_0000, // meet: s_barrier
            0x7E02_0206, // v_mov_b32 v1, s6
            0x3406_0082, // v_lshlrev_b32 v3, 2, v0
            0xE070_1000, 0x8002_0103, // buffer_store_dword v1, v3, s[8:11], 0 offen
            0xBF81_0000, // s_endpgm
        ];
        NativeProgram {
            stage: HwStage::Cs, wave64: true, vgprs: 4,
            layout: vec![RootParam {
                root: RootType::Table, table_offset_dwords: 0, sgprs: 1,
                base_register: 0, user_data: 0, slot: 0,
            }],
            used: vec![UsedResource {
                kind: ResourceType::Uav, root: RootType::Table,
                table_offset_dwords: 0, lower_bound: 0, stride: 1,
                dwords: 4, user_data: 0, slot: 0,
            }],
            io: StageIo::Cs(ComputeIo { rsrc2: 2 | (1 << 15), threads: [128, 1, 1] }),
            code: words.iter().flat_map(|w: &u32| w.to_le_bytes()).collect(),
        }
    }
    fn translated() -> HostShader {
        let native = synthetic();
        let p = ir::build(&native.code, true, native.vgprs).expect("decode synthetic paired program");
        let st = crate::structure::structure(&p).expect("structure");
        let io = crate::io::raise(&p, &native).expect("IO");
        let res = crate::resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
        lower(&native, &p, &io, &res).expect("paired lowering")
    }

    #[test]
    fn paired_lower_emits_guest_loop_and_lds() {
        let shader = translated();
        assert!(!shader.dxil.is_empty());
        assert_eq!(shader.user_data.len(), 1);
        assert_eq!(shader.user_data[0].root, RootType::Table);
        #[cfg(windows)]
        if let Ok(mut v) = super::super::validator::Validator::load(super::super::validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("paired scheduler DXIL validation failed");
        }
    }

    #[cfg(windows)]
    #[test]
    fn paired_lower_executes_guest_loop_and_lds() {
        use super::super::host::{Dispatch, Host, Root, View};
        let shader = translated();
        let _one = super::super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
        let Ok(host) = Host::new(false) else {
            eprintln!("no hardware device; paired scheduler execution not tested");
            return;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("paired scheduler pipeline");
        let out = host.dispatch(&root, &pso, &Dispatch {
            roots: vec![(0, Root::Constants(vec![0; 4]))],
            views: &[View::RawUav(128)], samplers: &[], groups: [1, 1, 1],
        }).expect("paired scheduler dispatch");
        assert_eq!(out[0].len(), 128);
        for (lane, v) in out[0].iter().enumerate() {
            assert_eq!(*v, if lane < 64 { 66 } else { 127 }, "guest lane {lane}");
        }
    }

    #[test]
    fn paired_lower_refuses_different_barrier_paths() {
        let mut native = synthetic();
        // Replace the scalar branch over the loop with a branch over
        // the barrier as well, so the barrier-order proof must refuse it.
        let word = 0xBF84_0004u32.to_le_bytes();
        let at = native.code.chunks_exact(4).position(|b| b == word).unwrap() * 4;
        native.code[at..at+4].copy_from_slice(&0xBF84_0005u32.to_le_bytes());
        let p = ir::build(&native.code, true, native.vgprs).expect("decode divergent barrier");
        assert!(barrier_order(&p).is_err());
    }
}
