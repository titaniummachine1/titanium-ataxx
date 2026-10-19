#!/usr/bin/env python3
"""Bundle titanium-ataxx engine core + Botzone main into ONE compilable file.
No external crates (pure std), so `rustc -O` builds it directly.
Usage: python training/bundle_botzone.py  (writes C:/git-projects/botzoo/...)
"""
import hashlib
import os

REPO = r'C:\gitProjects\titanium-ataxx'
BASE = REPO


def read(rel):
    with open(os.path.join(BASE, rel), encoding='utf-8') as f:
        return f.read()


parts = []
parts.append('// titanium-ataxx v1 (Botzone entry) -- SINGLE FILE, MIT.\n'
             '// Bundled engine core + Botzone simple-protocol driver.\n'
             '// Each engine module below carries its source sha256; verify with\n'
             '// training/verify_bundle.py. Build: rustc -O titanium-ataxx-v1.rs\n')
order = ['board', 'tuple', 'sancta', 'search']
manifest = []
for name in order:
    src = read('crates/titanium/src/%s.rs' % name)
    h = hashlib.sha256(src.encode('utf-8')).hexdigest()
    manifest.append('%s %s' % (name, h))
    parts.append('// ===BEGIN %s sha256:%s===\nmod %s {\n%s\n}\n// ===END %s===\n' % (name, h, name, src, name))
parts.append(read('training/botzone_main.rs'))
parts.append('\n// BUNDLE-MANIFEST:\n' + '\n'.join('// %s' % m for m in manifest) + '\n')
out = '\n'.join(parts)
fn = r'C:\gitProjects\botzoo\titanium-ataxx-v1.rs'
open(fn, 'w', encoding='utf-8').write(out)
print('wrote %s (%d bytes)' % (fn, len(out)))
print('\n'.join(manifest))
