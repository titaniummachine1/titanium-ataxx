"""D584 (user shared a reference picture: search graph by ply, positions merged across transpositions): one dumped shallow
alpha-beta tree (titanium-cli treediff --dump, PREFIX.treeK.tsv) + the deep PV (PREFIX.pv.tsv) -> PREFIX.treeK.graph.html.
Columns = plies; each dot = one POSITION at that ply (records with the same hash merged: a transposition = reached from
several parents, drawn with a dashed ring); dot area = search nodes spent under it; colour = value for the side to move
at the ROOT (teal = good for us, orange = good for the opponent, grey = only a bound came back); grey links = parent ->
child; dark line = the shallow search's main line; gold line = the deep search's line. Hover a dot for its numbers.
usage: python -I tree_graph.py PREFIX K [--plies 14] [--per-ply 60]"""
import csv
import json
import sys
from collections import defaultdict

prefix, k = sys.argv[1], int(sys.argv[2])
max_ply = int(sys.argv[sys.argv.index('--plies') + 1]) if '--plies' in sys.argv else 14
per_ply = int(sys.argv[sys.argv.index('--per-ply') + 1]) if '--per-ply' in sys.argv else 60
rows = list(csv.DictReader(open(f'{prefix}.tree{k}.tsv'), delimiter='\t'))
pvrows = [r for r in csv.DictReader(open(f'{prefix}.pv.tsv'), delimiter='\t') if int(r['pos']) == k]
pos = [r for r in csv.DictReader(open(f'{prefix}.pos.tsv'), delimiter='\t') if int(r['pos']) == k][0]
I = lambda r, f: int(r[f])
SKIP = ('skip_sn', 'skip_fut', 'skip_rep')
byid = {I(r, 'id'): r for r in rows}
kids = defaultdict(list)
for r in rows:
    if I(r, 'parent') != 4294967295:
        kids[I(r, 'parent')].append(r)

# merge records into positions per ply: key = (ply, hash); null/probcut verification records are not positions of the tree
nodes = {}
for r in rows:
    if r['kind'] in SKIP or r['via'] in ('null', 'probcut'):
        continue
    ply = I(r, 'ply')
    if ply > max_ply:
        continue
    key = (ply, r['hash'] if 'hash' in r else r['id'])
    n = nodes.setdefault(key, {'ply': ply, 'sub': 0, 'recs': 0, 'parents': set(), 'score': None, 'move': r['move'], 'ids': [], 'exact': False})
    n['sub'] += I(r, 'subtree')
    n['recs'] += 1
    n['ids'].append(I(r, 'id'))
    p = I(r, 'parent')
    if p in byid:
        n['parents'].add(p)
    s = I(r, 'score')
    if s != -2147483648:
        root_view = s if ply % 2 == 0 else -s
        # keep the most informative (largest subtree) record's value
        if n['score'] is None or I(r, 'subtree') >= n.get('best_sub', -1):
            n['score'], n['best_sub'] = root_view, I(r, 'subtree')
            n['exact'] = I(r, 'alpha') < s < I(r, 'beta')
id2key = {i: key for key, n in nodes.items() for i in n['ids']}

# keep the biggest positions per ply (plus every main-line / deep-PV position)
def main_line(start_id):
    line, cur = [], start_id
    while True:
        ch = [c for c in kids[cur] if c['kind'] not in SKIP and c['via'] not in ('null', 'probcut') and c['score'] != '-2147483648']
        if not ch:
            return line
        best = max(ch, key=lambda c: -I(c, 'score'))  # child score is from the child's side: best for the parent = lowest
        line.append(I(best, 'id'))
        cur = I(best, 'id')


shallow_ids = main_line(0)
deep_ids, cur = [], 0
for step in pvrows:
    ch = [c for c in kids[cur] if c['move'] == step['move'] and c['via'] not in ('null', 'probcut')]
    if not ch:
        break
    deep_ids.append(I(ch[-1], 'id'))
    if ch[-1]['kind'] in SKIP:
        break
    cur = I(ch[-1], 'id')
must = {id2key[i] for i in shallow_ids + deep_ids if i in id2key}
keep = set()
by_ply = defaultdict(list)
for key, n in nodes.items():
    by_ply[n['ply']].append(key)
for ply, ks in by_ply.items():
    ks.sort(key=lambda kk: -nodes[kk]['sub'])
    keep.update(ks[:per_ply])
keep |= must
root_key = id2key.get(0)
keep.add(root_key)

# layout: column x by ply; rows ordered by the parent's row, then size (keeps families together)
order = {root_key: 0.0}
cols = defaultdict(list)
for ply in sorted(by_ply):
    ks = [kk for kk in by_ply[ply] if kk in keep]
    def pos_of(kk):
        ps = [order[id2key[p]] for p in nodes[kk]['parents'] if p in id2key and id2key[p] in order]
        return (min(ps) if ps else 1e9, -nodes[kk]['sub'])
    ks.sort(key=pos_of)
    for i, kk in enumerate(ks):
        order[kk] = i
    cols[ply] = ks
out_nodes, idx = [], {}
for ply in sorted(cols):
    for i, kk in enumerate(cols[ply]):
        n = nodes[kk]
        idx[kk] = len(out_nodes)
        out_nodes.append({'x': ply, 'y': i, 'n': len(cols[ply]), 'sub': n['sub'], 'v': n['score'], 'ex': n['exact'], 'tr': len(n['parents']) > 1,
                          'm': n['move'], 'np': len(n['parents']), 'recs': n['recs']})
links = set()
for kk in idx:
    for p in nodes[kk]['parents']:
        pk = id2key.get(p)
        if pk in idx:
            links.add((idx[pk], idx[kk]))
line = lambda ids: [idx[id2key[i]] for i in ids if i in id2key and id2key[i] in idx]
data = {'nodes': out_nodes, 'links': sorted(links), 'main': [idx[root_key]] + line(shallow_ids), 'deep': [idx[root_key]] + line(deep_ids),
        'title': f"Titanium alpha-beta tree, position {k}: NNUE bucket {pos['phase']}, {pos['empties']} empties, shallow depth {pos['sdepth']} plays {pos['shallow']}, "
                 f"deep depth {pos['ddepth']} plays {pos['deep']} ({'agree' if pos['agree'] == '1' else 'DISAGREE'})",
        'sub': f"{len(out_nodes)} positions shown (top {per_ply} per ply by nodes spent + both main lines), {sum(1 for n in out_nodes if n['tr'])} transpositions (dashed rings), "
               f"{max_ply} plies. Size = nodes searched under the position, colour = value for the root side (teal good, orange bad, grey = bound only). Dark line = shallow main line, gold = deep line."}
html = '''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Titanium search graph</title><style>
body{margin:0;background:#f4f6f5;font:12px system-ui,sans-serif}h1{font-size:18px;margin:12px 16px 2px}p{margin:0 16px 8px;color:#555}
#tip{position:fixed;pointer-events:none;background:#222;color:#fff;padding:5px 7px;border-radius:4px;font:11px monospace;white-space:pre;display:none}</style></head>
<body><h1 id="t"></h1><p id="s"></p><svg id="g"></svg><div id="tip"></div><script>
const D=__DATA__;document.getElementById('t').textContent=D.title;document.getElementById('s').textContent=D.sub;
const maxX=Math.max(...D.nodes.map(n=>n.x)),maxN=Math.max(...D.nodes.map(n=>n.n));
const W=Math.max(1200,(maxX+1)*95+80),H=Math.max(600,maxN*13+60),cx=x=>50+x*95,cy=n=>H/2+(n.y-(n.n-1)/2)*13;
const maxS=Math.max(...D.nodes.map(n=>n.sub)),rad=n=>2+11*Math.sqrt(n.sub/maxS);
function col(n){if(n.v===null)return'#c8cdd0';const t=Math.max(-1,Math.min(1,n.v/300));const a=n.ex?1:0.45;
 return t>=0?`rgba(${Math.round(160-130*t)},${Math.round(175-30*t)},${Math.round(175-20*t)},${a})`:`rgba(${Math.round(190+40*-t)},${Math.round(150-60*-t)},${Math.round(130-90*-t)},${a})`;}
const svg=document.getElementById('g');svg.setAttribute('width',W);svg.setAttribute('height',H);let s='';
for(let x=0;x<=maxX;x++)s+=`<text x="${cx(x)}" y="14" font-size="10" fill="#888" text-anchor="middle">${x?'+'+x:'root'}</text>`;
const path=(a,b)=>{const A=D.nodes[a],B=D.nodes[b],x1=cx(A.x),y1=cy(A),x2=cx(B.x),y2=cy(B),m=(x1+x2)/2;return`M${x1},${y1}C${m},${y1} ${m},${y2} ${x2},${y2}`};
for(const[a,b]of D.links)s+=`<path d="${path(a,b)}" stroke="#b8c2c6" stroke-width="0.6" fill="none" opacity="0.6"/>`;
for(const[L,c,w]of[[D.deep,'#d9a400',4],[D.main,'#1d2b30',2.2]])for(let i=1;i<L.length;i++)s+=`<path d="${path(L[i-1],L[i])}" stroke="${c}" stroke-width="${w}" fill="none"/>`;
D.nodes.forEach((n,i)=>{s+=`<circle data-i="${i}" cx="${cx(n.x)}" cy="${cy(n)}" r="${rad(n)}" fill="${col(n)}" stroke="${n.tr?'#334':'#fff'}" stroke-width="${n.tr?1:0.5}" ${n.tr?'stroke-dasharray="2,1.5"':''}/>`;
 if(rad(n)>6)s+=`<text x="${cx(n.x)+rad(n)+2}" y="${cy(n)+3}" font-size="9" fill="#333">${n.m} ${n.sub}</text>`;});
svg.innerHTML=s;const tip=document.getElementById('tip');
svg.addEventListener('mousemove',e=>{const i=e.target.dataset&&e.target.dataset.i;if(i===undefined){tip.style.display='none';return}const n=D.nodes[i];
 tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
 tip.textContent=`ply ${n.x}  ${n.m}\\nnodes under it ${n.sub}\\nvalue (root side) ${n.v===null?'-':n.v}${n.ex?' exact':' bound'}\\nreached from ${n.np} parent(s), ${n.recs} visit(s)`;});
</script></body></html>'''
open(f'{prefix}.tree{k}.graph.html', 'w', encoding='utf-8').write(html.replace('__DATA__', json.dumps(data, separators=(',', ':'))))
print(f'wrote {prefix}.tree{k}.graph.html: {len(out_nodes)} positions, {len(links)} links, transpositions {sum(1 for n in out_nodes if n["tr"])}')
