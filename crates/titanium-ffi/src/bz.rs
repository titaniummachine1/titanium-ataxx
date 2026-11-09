//! The Botzone long-session loop INSIDE the engine library (D662, user: "Python just a middleman that forwards"). Python only starts the
//! process, creates the engine and makes ONE call, `ti_botzone_serve`; from then on no Python code runs between a request arriving and
//! the answer leaving: the engine reads stdin itself, stamps the arrival with a monotonic clock, applies the move(s), searches against
//! its own deadline, and writes the answer to stdout itself. A watchdog THREAD (also Rust) plays the best move so far if the search has
//! not answered by the hard deadline. No garbage collector, no GIL, no Python timer can make an answer late.
//!
//! Protocol (identical to the Python driver it replaces): first request "turns" + 4 * (2 * turns - 1) ints (entries with x0 < 0 are the
//! start marker), later requests 4 ints. Answer: "x0 y0 x1 y1\n<debug line>\n\n\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n".
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use titanium::board::Move;

use crate::{apply_raw, search_core, ti_new_game, TiEngine};

const MARKER: &str = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<";

#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(ms: u32) -> u32;
}

struct Turn {
    /// monotonically increasing turn id (wakes the watchdog when a new turn is armed)
    gen: u64,
    /// the hard deadline of the armed turn
    hard: Option<Instant>,
    /// the answer of this turn has been written (by the main loop or by the watchdog)
    answered: bool,
    /// the move the watchdog played instead of the engine's
    fallback: Option<(i32, i32, i32, i32)>,
    quit: bool,
}

struct Shared {
    m: Mutex<Turn>,
    cv: Condvar,
}

/// Write one complete answer with a single write call.
fn emit(mv: (i32, i32, i32, i32), dbg: &str) {
    let text = format!("{} {} {} {}\n{}\n\n\n{}\n", mv.0, mv.1, mv.2, mv.3, dbg, MARKER);
    let so = std::io::stdout();
    let mut w = so.lock();
    let _ = w.write_all(text.as_bytes());
    let _ = w.flush();
}

fn decode(v: u32) -> Option<(i32, i32, i32, i32)> {
    if v == 0 {
        return None;
    }
    let m = Move::from_u32(v - 1);
    Some(((m.from % 7) as i32, (m.from / 7) as i32, (m.to % 7) as i32, (m.to / 7) as i32))
}

fn watchdog(sh: Arc<Shared>, stop: Arc<AtomicBool>, best: Arc<AtomicU32>, t_turn: Arc<Mutex<Instant>>) {
    #[cfg(windows)]
    unsafe {
        timeBeginPeriod(1);
    }
    let mut g = sh.m.lock().unwrap();
    loop {
        if g.quit {
            return;
        }
        match g.hard {
            Some(at) if !g.answered => {
                let now = Instant::now();
                if now >= at {
                    if let Some(mv) = decode(best.load(Ordering::Relaxed)) {
                        stop.store(true, Ordering::Relaxed);
                        g.answered = true;
                        g.fallback = Some(mv);
                        let t0 = *t_turn.lock().unwrap();
                        emit(mv, &format!("FALLBACK best-so-far t{:.0} rust", t0.elapsed().as_secs_f64() * 1000.0));
                    }
                    g.hard = None;
                } else {
                    let (ng, _) = sh.cv.wait_timeout(g, at - now).unwrap();
                    g = ng;
                }
            }
            _ => {
                g = sh.cv.wait(g).unwrap();
            }
        }
    }
}

/// Minimal whitespace-separated integer reader over stdin. `arrival` = the instant the FIRST byte of the current request arrived.
struct Tokens {
    arrival: Option<Instant>,
}

impl Tokens {
    fn byte(&mut self) -> Option<u8> {
        let mut b = [0u8; 1];
        loop {
            match std::io::stdin().lock().read(&mut b) {
                Ok(0) => return None,
                Ok(_) => return Some(b[0]),
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => return None,
            }
        }
    }

    /// Next integer token (non-numeric tokens are skipped); None at end of input.
    fn int(&mut self) -> Option<i32> {
        loop {
            let mut c = self.byte()?;
            while c.is_ascii_whitespace() {
                c = self.byte()?;
            }
            if self.arrival.is_none() {
                self.arrival = Some(Instant::now());
            }
            let mut tok = vec![c];
            loop {
                match self.byte() {
                    Some(d) if !d.is_ascii_whitespace() => tok.push(d),
                    _ => break,
                }
            }
            if let Ok(v) = String::from_utf8_lossy(&tok).parse::<i32>() {
                return Some(v);
            }
        }
    }
}

fn serve_inner(e: &mut TiEngine, first_ms: u32, turn_ms: u32, pad_stop_ms: u32, hard_ms: u32, age_ms: u32) -> i32 {
    e.searcher.set_safety_ms(0.0);
    let start = Instant::now().checked_sub(Duration::from_millis(age_ms as u64)).unwrap_or_else(Instant::now);
    let sh = Arc::new(Shared { m: Mutex::new(Turn { gen: 0, hard: None, answered: false, fallback: None, quit: false }), cv: Condvar::new() });
    let t_turn = Arc::new(Mutex::new(start));
    {
        let (sh2, stop, best, tt) = (sh.clone(), e.stop.clone(), e.best.clone(), t_turn.clone());
        std::thread::spawn(move || watchdog(sh2, stop, best, tt));
    }
    let mut tk = Tokens { arrival: None };
    let mut hist: Vec<(i32, i32, i32, i32)> = Vec::new();
    let mut first = true;
    let rc = loop {
        tk.arrival = None;
        let mut incoming: Vec<(i32, i32, i32, i32)> = Vec::new();
        if first {
            let Some(turns) = tk.int() else { break 0 };
            let n = 4 * (2 * turns.max(1) - 1) as usize;
            let mut flat = Vec::with_capacity(n);
            for _ in 0..n {
                match tk.int() {
                    Some(v) => flat.push(v),
                    None => break,
                }
            }
            if flat.len() < n {
                break 0;
            }
            for c in flat.chunks(4) {
                if c[0] >= 0 {
                    incoming.push((c[0], c[1], c[2], c[3]));
                }
            }
        } else {
            let mut m = [0i32; 4];
            for slot in m.iter_mut() {
                match tk.int() {
                    Some(v) => *slot = v,
                    None => return 0,
                }
            }
            incoming.push((m[0], m[1], m[2], m[3]));
        }
        let t_req = tk.arrival.unwrap_or_else(Instant::now);
        let t0 = if first { start } else { t_req };
        let limit = if first { first_ms } else { turn_ms } as u64;
        *t_turn.lock().unwrap() = t0;
        // arm the watchdog for this turn
        {
            let mut g = sh.m.lock().unwrap();
            g.gen += 1;
            g.answered = false;
            g.fallback = None;
            g.hard = Some(t0 + Duration::from_millis(limit.saturating_sub(hard_ms as u64)));
            sh.cv.notify_all();
        }
        // apply the incoming moves (resync from the full history once if one is rejected)
        let mut ok = true;
        for mv in &incoming {
            if apply_raw(e, mv.0, mv.1, mv.2, mv.3) != 1 {
                ok = false;
                break;
            }
            hist.push(*mv);
        }
        if !ok {
            ti_new_game(e as *mut TiEngine);
            let replay: Vec<_> = hist.clone();
            hist.clear();
            ok = true;
            for mv in replay.iter().chain(incoming.iter()) {
                if apply_raw(e, mv.0, mv.1, mv.2, mv.3) != 1 {
                    ok = false;
                    break;
                }
                hist.push(*mv);
            }
        }
        if !ok {
            let mut g = sh.m.lock().unwrap();
            if !g.answered {
                g.answered = true;
                emit((-1, -1, -1, -1), "ENGINE_FAILED history_move_rejected rust");
            }
            g.hard = None;
            first = false;
            continue;
        }
        // search against the engine's own clock: stop at limit - pad_stop (the watchdog backs it up at limit - hard)
        let deadline = t0 + Duration::from_millis(limit.saturating_sub(pad_stop_ms as u64));
        let budget = deadline.saturating_duration_since(Instant::now()).as_millis().max(1) as u64;
        let res = search_core(e, budget, 0, 255);
        let mut g = sh.m.lock().unwrap();
        g.hard = None;
        let played: (i32, i32, i32, i32);
        match res {
            None => {
                played = (-1, -1, -1, -1);
                if !g.answered {
                    g.answered = true;
                    emit(played, "NO_MOVE rust");
                }
            }
            Some((mv, st)) => {
                let mine = ((mv.from % 7) as i32, (mv.from / 7) as i32, (mv.to % 7) as i32, (mv.to / 7) as i32);
                if !g.answered {
                    g.answered = true;
                    played = mine;
                    emit(
                        mine,
                        &format!(
                            "d{} n{} t{:.0} s{}{} bundled-main/rust",
                            st.depth,
                            st.nodes,
                            t0.elapsed().as_secs_f64() * 1000.0,
                            st.score,
                            if st.repetition_override != 0 { " rep" } else { "" }
                        ),
                    );
                } else {
                    // the watchdog answered first: THAT move is the one played, keep the engine's board consistent with it
                    played = g.fallback.unwrap_or(mine);
                }
            }
        }
        drop(g);
        if played.0 >= 0 {
            // the engine's own move was not pushed by search_core: push the move that was actually played
            if apply_raw(e, played.0, played.1, played.2, played.3) == 1 {
                hist.push(played);
            }
        }
        first = false;
    };
    let mut g = sh.m.lock().unwrap();
    g.quit = true;
    sh.cv.notify_all();
    rc
}

/// Run the whole Botzone long session inside the library. Blocks until stdin closes. Returns 0 = normal end, -1 = bad arguments, -3 = panic.
/// Times in ms: `first_ms` / `turn_ms` = the platform limits (12000 / 4000), `pad_stop_ms` = how long before the limit the engine stops its
/// own search, `hard_ms` = how long before the limit the watchdog plays the best move so far, `age_ms` = age of the process at the call.
#[no_mangle]
pub extern "C" fn ti_botzone_serve(e: *mut TiEngine, first_ms: u32, turn_ms: u32, pad_stop_ms: u32, hard_ms: u32, age_ms: u32) -> i32 {
    if e.is_null() {
        return -1;
    }
    let e = unsafe { &mut *e };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| serve_inner(e, first_ms, turn_ms, pad_stop_ms, hard_ms, age_ms))) {
        Ok(rc) => rc,
        Err(_) => -3,
    }
}
