#!/usr/bin/env python3
"""Verify a bundled botzone file against repo sources (per-module sha256).
Usage: python training/verify_bundle.py <bundle>   (exit 0 = identical)
"""
import hashlib
import os
import re
import sys

REPO = r'C:\gitProjects\titanium-ataxx'


def main():
    path = sys.argv[1]
    b = open(path, encoding='utf-8').read()
    ok = True
    mods = re.findall(r'// ===BEGIN (\w+) sha256:([0-9a-f]{64})===', b)
    print('embedded modules:', [m[0] for m in mods])
    for name, want in mods:
        with open(os.path.join(REPO, 'crates', 'titanium', 'src', name + '.rs'),
                  encoding='utf-8') as f:
            src = f.read()
        got = hashlib.sha256(src.encode('utf-8')).hexdigest()
        match = (got == want)
        ok = ok and match
        print('%-8s repo=%s bundle=%s %s' %
              (name, got[:12], want[:12], 'MATCH' if match else 'MISMATCH'))
    # main driver present?
    print('has botzone main:', 'Botzone entry' in b or 'MOVETIME_MS' in b)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
