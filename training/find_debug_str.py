import os

for root, dirs, files in os.walk('C:/gitProjects'):
    if any(x in root for x in ['.git', 'target', 'node_modules', '__pycache__', 'data']):
        continue
    for f in files:
        if f.endswith(('.rs', '.py', '.cpp', '.c', '.sh', '.json')):
            p = os.path.join(root, f)
            try:
                t = open(p, encoding='utf-8', errors='ignore').read()
                if 'bundled-main' in t:
                    print(f"FOUND in {p}")
            except Exception:
                pass
