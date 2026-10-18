#!/usr/bin/env python3
"""Cloud execution backend interface for the auto-gate system (STUB).

Future Modal integration point. Contract:
  run_match(job) -> (w, l, d)   # plays job['games'] at job['nodes'] on book job['book']
  job = {'a_exe_sha':..., 'b_exe_sha':..., 'a_setup':[...], 'b_setup':[...],
         'book': path, 'games': int, 'nodes': int, 'movetime': int,
         'seed': int, 'tup': path, 'a_dialect': 'uai'|'uci', ...}
Binaries + net + books ship via Modal Volume (see deploy notes below).
The gate runner (auto_gate.py) uses this module ONLY with --cloud
(manual flag); default is always local background execution.

Deploy notes (when wiring Modal):
  1. pip install modal; modal setup.
  2. Image: python:3.10 + repo snapshot + prebuilt Windows->NO;
     Modal runs Linux: rebuild engines for linux-64 (cmake+gcc) and
     store per-platform binaries. UAI protocol is platform-independent,
     so game logs stay comparable; record platform per match.
  3. Volume 'ataxx-artifacts': engines, .tup nets, books. Uploaded once
     per commit (hash-addressed).
  4. Modal function run_match.remote(job) with _concurrency = workers.
  5. This stub raises until step 1-4 are done; auto_gate falls back local.
"""


def run_match(job):
    raise NotImplementedError(
        'cloud backend not configured. See training/cloud_backend.py deploy notes: '
        'modal setup + linux engine builds + artifact volume, then implement run_match(job).')
