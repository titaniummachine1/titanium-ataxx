//! Stockfish-rule guardrail, ported from QuoridorBestAI `engine/tests/repo_hygiene.rs`.
//!
//! The workspace root is docs + engine + training sources. What is NEVER
//! tracked: built binaries, weight blobs, databases, match logs, run output,
//! bytecode. Those live in `scripts/` (gate .exes, Moonbird, launchers),
//! `data/` (nets, corpora, caches), `logs/`, `target*/` — all gitignored.
//! `.gitignore` stops the accidental `git add`; this test stops the
//! deliberate one (`git add -f`). It reads `git ls-files`, so it fails on
//! what is TRACKED, not on what sits in the working tree.
//!
//! Root docs (`LEDGER.md`, `HANDOFFS.md`, `BEHAVIOR.md`, `README.md`) ARE
//! tracked on purpose: the ledger is the experiment record, like Quoridor's
//! workspace root. If a path is genuinely part of the build, add it to
//! EXEMPT with a comment saying why.

use std::path::Path;
use std::process::Command;

/// Path prefixes that must never be tracked.
const FORBIDDEN_DIRS: &[&str] = &[
    "data/",    // corpora, nets, caches — local only
    "logs/",    // match logs, gate output
    "scripts/", // snapshot .exes, Moonbird, per-gate launchers — local scratch
];

/// Build-dir prefixes (`target`, `target-*`, `target_*`) — checked
/// separately since they are prefixes, not whole path components.
fn is_target_dir(lower: &str) -> bool {
    lower == "target"
        || ["target/", "target-", "target_"]
            .iter()
            .any(|p| lower.starts_with(p))
}

/// Filename patterns that must never be tracked. `*` matches any run of
/// characters; matching is case-insensitive.
const FORBIDDEN_FILES: &[&str] = &[
    "*.exe",   // built/snapshot binaries
    "*.dll",   //
    "*.pdb",   //
    "*.pyc",   // bytecode
    "*.tup",   // tuple weight blobs
    "*.s1",    // NNUE weight blobs
    "*.pt",    // checkpoints
    "*.npz",   // corpora/caches
    "*.onnx",  //
    "*.db",    // databases
    "*.db-wal",
    "*.db-shm",
    "*.log",   // run output
    "*.jsonl", // match/bench dumps
];

/// Paths allowed through despite matching a rule above.
const EXEMPT: &[&str] = &[];

fn glob_match(pattern: &str, name: &str) -> bool {
    // Only `*` is supported, which is all these patterns need.
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut rest = name;

    if let Some(first) = parts.first() {
        if !rest.starts_with(first) {
            return false;
        }
        rest = &rest[first.len()..];
    }
    if let Some(last) = parts.last() {
        if parts.len() > 1 {
            if rest.len() < last.len() || !rest.ends_with(last) {
                return false;
            }
            rest = &rest[..rest.len() - last.len()];
        }
    }
    for mid in parts.iter().skip(1).take(parts.len().saturating_sub(2)) {
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

fn reason(path: &str) -> Option<String> {
    let lower = path.to_ascii_lowercase();

    if EXEMPT.iter().any(|e| lower.starts_with(e)) {
        return None;
    }
    if is_target_dir(&lower) {
        return Some("lives under a `target*/` build dir".to_string());
    }
    if let Some(dir) = FORBIDDEN_DIRS.iter().find(|d| lower.starts_with(**d)) {
        return Some(format!("lives under `{dir}`"));
    }
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if let Some(pat) = FORBIDDEN_FILES.iter().find(|p| glob_match(p, name)) {
        return Some(format!("matches `{pat}`"));
    }
    None
}

#[test]
fn workspace_is_strength_only() {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = crate_root.join("..").join(".."); // crates/titanium -> workspace

    let out = match Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(&root)
        .output()
    {
        Ok(o) if o.status.success() => o,
        // No git, or a source tarball with no history: nothing to check.
        _ => return,
    };

    let tracked = String::from_utf8_lossy(&out.stdout);
    let offenders: Vec<String> = tracked
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(|p| p.replace('\\', "/"))
        .filter_map(|p| reason(&p).map(|why| format!("  {p}  ({why})")))
        .collect();

    assert!(
        offenders.is_empty(),
        "\n\
         {} file(s) tracked that do not belong in the repo:\n\
         {}\n\n\
         Binaries, weight blobs, databases, logs and run output are never\n\
         versioned: build them, don't track them.\n\
         \x20 snapshot/gate .exes + Moonbird  -> scripts/bench/, scripts/moonbird/ (local)\n\
         \x20 nets / corpora / caches        -> data/ (local)\n\
         \x20 match + gate logs              -> logs/ (local)\n\
         \x20 session state                  -> LEDGER.md / HANDOFFS.md (tracked, append-only)\n\n\
         If a path is genuinely part of the build, add it to EXEMPT in this file\n\
         with a comment saying why.\n",
        offenders.len(),
        offenders.join("\n"),
    );
}

#[test]
fn glob_match_works() {
    assert!(glob_match("*.exe", "titanium_main.exe"));
    assert!(glob_match("*.tup", "tupS25ow20.tup"));
    assert!(glob_match("*.log", "loop.log"));
    assert!(glob_match("*.jsonl", "restart_sweep.jsonl"));

    assert!(!glob_match("*.exe", "search.rs"));
    assert!(!glob_match("*.log", "logic.rs"));
    assert!(!glob_match("*.tup", "tuple.rs"));
}
