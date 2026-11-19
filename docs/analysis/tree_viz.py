"""D568 (user: 'load a visualization of the search tree and look at what is happening'): one dumped shallow search tree
(titanium-cli treediff --dump, PREFIX.treeK.tsv) + the deep PV (PREFIX.pv.tsv) -> a self-contained HTML icicle viewer
(click a block to zoom, hover for numbers) and a text summary of where the nodes went.
Every searched node is marked by what it contributed to its parent:
  cutoff   (green)  the move refuted the parent's window (fail high at the parent)
  raised   (blue)   the move raised the parent's alpha (new best)
  bound    (light grey) failed low exactly at the bound (fail-hard child: true margin unknown)
  close    (yellow) failed low by 1..50 cp: needed to be searched
  low      (tan)    failed low by 50..200 cp
  wasted   (red)    failed low by > 200 cp: a clearly bad move searched with a full subtree (pruning could have saved it)
  null/pc  (grey/purple) null-move / ProbCut verification searches (cut or failed)
  pruned   (orange) a child skipped without search (small net / futility); OVERPRUNED (magenta) if it is the deep PV move
Deep-PV nodes get a gold outline. Analysis only (no training/tuning).
usage: python -I tree_viz.py PREFIX K [--max-ply 8] [--min-sub 4] -> PREFIX.treeK.html"""
import csv
import json
import sys
from collections import defaultdict

prefix, k = sys.argv[1], int(sys.argv[2])
max_ply = int(sys.argv[sys.argv.index('--max-ply') + 1]) if '--max-ply' in sys.argv else 8
min_sub = int(sys.argv[sys.argv.index('--min-sub') + 1]) if '--min-sub' in sys.argv else 4

rows = list(csv.DictReader(open(f'{prefix}.tree{k}.tsv'), delimiter='\t'))
pv = [r for r in csv.DictReader(open(f'{prefix}.pv.tsv'), delimiter='\t') if int(r['pos']) == k]
pos = [r for r in csv.DictReader(open(f'{prefix}.pos.tsv'), delimiter='\t') if int(r['pos']) == k][0]
I = lambda r, f: int(r[f])
children = defaultdict(list)
for r in rows:
    p = I(r, 'parent')
    if p != 4294967295 and p >= 0:
        children[p].append(r)

SKIP = ('skip_sn', 'skip_fut', 'skip_rep')


def category(r, parent):
    if r['via'] == 'null':
        return 'null'
    if r['via'] == 'probcut':
        return 'probcut'
    if r['kind'] in SKIP:
        return 'pruned'
    if r['score'] in ('-2147483648',):
        return 'open'
    v = -I(r, 'score')  # value of the move for the parent
    pa = -I(r, 'beta')  # the parent's alpha when the child was called
    if parent is not None and v >= I(parent, 'beta'):
        return 'cutoff'
    if v > pa:
        return 'raised'
    m = pa - v
    if m == 0:
        return 'bound'  # failed low exactly at the bound (fail-hard return: the true margin is unknown)
    return 'close' if m <= 50 else ('low' if m <= 200 else 'wasted')


# deep PV path through the tree: list of record ids
deep_ids = set()
over = set()
cur = rows[0]
for step in pv:
    cands = [c for c in children[I(cur, 'id')] if c['move'] == step['move'] and c['via'] not in ('null', 'probcut')]
    if not cands:
        break
    c = cands[-1]
    deep_ids.add(I(c, 'id'))
    if c['kind'] in SKIP:
        over.add(I(c, 'id'))
        break
    cur = c

stats = defaultdict(lambda: [0, 0])  # category -> [count, subtree nodes]
by_ply = defaultdict(lambda: defaultdict(int))


def build(r, parent, ply):
    cat = category(r, parent) if parent is not None else 'root'
    rid = I(r, 'id')
    if rid in over:
        cat = 'OVERPRUNED'
    sub = I(r, 'subtree') if r['kind'] not in SKIP else 0
    stats[cat][0] += 1
    stats[cat][1] += sub
    by_ply[ply][cat] += max(sub, 1)
    node = {
        'n': r['move'], 'c': cat, 'v': max(sub, 1), 'k': r['kind'], 'via': r['via'], 'd': I(r, 'depth'), 'fd': I(r, 'full_depth'),
        'i': I(r, 'idx'), 'a': I(r, 'alpha'), 'b': I(r, 'beta'), 'st': I(r, 'stat'), 's': I(r, 'score'), 'e': I(r, 'empties'),
    }
    if rid in deep_ids:
        node['pv'] = 1
    kids = children.get(rid, [])
    if ply < max_ply and kids:
        out, rest, rest_n, pruned_n = [], 0, 0, 0
        for c in kids:
            cid = I(c, 'id')
            csub = I(c, 'subtree') if c['kind'] not in SKIP else 0
            if c['kind'] in SKIP and cid not in deep_ids:
                pruned_n += 1
                stats['pruned'][0] += 1
                by_ply[ply + 1]['pruned'] += 1
                continue
            if csub >= min_sub or cid in deep_ids or ply < 1:
                out.append(build(c, r, ply + 1))
            else:
                rest += max(csub, 1)
                rest_n += 1
                cc = category(c, r)
                stats[cc][0] += 1
                stats[cc][1] += csub
                by_ply[ply + 1][cc] += max(csub, 1)
        if rest:
            out.append({'n': f'+{rest_n} small', 'c': 'small', 'v': rest})
        if pruned_n:
            out.append({'n': f'{pruned_n} pruned', 'c': 'pruned', 'v': pruned_n})
        node['ch'] = out
        node.pop('v')
    return node


tree = build(rows[0], None, 0)
total = I(rows[0], 'subtree')
summary = [f"position {k}: bucket {pos['phase']} empties {pos['empties']}, shallow depth {pos['sdepth']} ({pos['snodes']} nodes incl. earlier iterations), "
           f"deep depth {pos['ddepth']}; shallow move {pos['shallow']} vs deep move {pos['deep']} ({'agree' if pos['agree'] == '1' else 'DISAGREE'}); "
           f"deep PV {' '.join(p['move'] for p in pv)}; final-iteration tree {total} nodes"]
summary.append('  children by category (count): ' + ', '.join(f'{c} {n}' for c, (n, s) in sorted(stats.items(), key=lambda x: -x[1][0])))
summary.append('  by ply (subtree nodes by category of the node at that ply):')
for ply in sorted(by_ply):
    if ply > 6:
        break
    t = sum(by_ply[ply].values())
    summary.append(f"   ply {ply}: " + ', '.join(f"{c} {100 * v / t:.0f}%" for c, v in sorted(by_ply[ply].items(), key=lambda x: -x[1])))
text = '\n'.join(summary)
print(text)

html = '''<!DOCTYPE html><html><head><meta charset="utf-8"><title>Titanium search tree</title>
<style>body{font:12px system-ui,sans-serif;margin:8px;background:#fafafa}#info{white-space:pre;font:11px monospace;background:#fff;border:1px solid #ddd;padding:6px;margin-bottom:6px}
#tip{position:fixed;pointer-events:none;background:#222;color:#fff;padding:6px 8px;border-radius:4px;font:11px monospace;white-space:pre;display:none}
.leg span{display:inline-block;padding:1px 6px;margin-right:4px;border-radius:3px;color:#000}</style></head><body>
<div id="info"></div><div class="leg" id="leg"></div><svg id="v"></svg><div id="tip"></div>
<script src="https://cdn.jsdelivr.net/npm/d3@7"></script><script>
const DATA=__DATA__, INFO=__INFO__;
const COL={root:'#999',bound:'#cfd8dc',cutoff:'#5cb85c',raised:'#4a90d9',close:'#f0d050',low:'#d8c3a0',wasted:'#d9534f',null:'#bbb',probcut:'#b08fd8',pruned:'#f0a040',OVERPRUNED:'#e020c0',small:'#ddd',open:'#eee'};
document.getElementById('info').textContent=INFO;
document.getElementById('leg').innerHTML=Object.entries(COL).map(([k,v])=>`<span style="background:${v}">${k}</span>`).join('')+' gold outline = deep-search PV. Click a block to zoom, click the top bar to go back.';
const W=Math.max(900,window.innerWidth-30),H=Math.max(500,window.innerHeight-200);
const root=d3.hierarchy(DATA,d=>d.ch).sum(d=>d.ch?0:d.v);
d3.partition().size([H,W*1.0])(root);
const svg=d3.select('#v').attr('width',W).attr('height',H);
const levels=root.height+1, cw=W/Math.min(levels,9);
let focus=root;
const g=svg.selectAll('g').data(root.descendants()).join('g');
const rect=g.append('rect').attr('fill',d=>COL[d.data.c]||'#ccc').attr('stroke',d=>d.data.pv?'#c90':'#fff').attr('stroke-width',d=>d.data.pv?2.5:0.3);
const lab=g.append('text').attr('font-size',10).attr('dy','1em').attr('dx',2).text(d=>d.data.n);
function pos(f){const x0=f.x0,x1=f.x1,dy=f.depth;
 rect.attr('x',d=>(d.depth-dy)*cw).attr('y',d=>(d.x0-x0)/(x1-x0)*H).attr('width',d=>d.depth>=dy?cw-1:0).attr('height',d=>Math.max(0,(d.x1-d.x0)/(x1-x0)*H-0.5));
 lab.attr('x',d=>(d.depth-dy)*cw).attr('y',d=>(d.x0-x0)/(x1-x0)*H).attr('display',d=>d.depth>=dy&&(d.x1-d.x0)/(x1-x0)*H>11?null:'none');}
pos(root);
g.on('click',(e,d)=>{focus=(d===focus&&d.parent)?d.parent:d;pos(focus);});
const tip=document.getElementById('tip');
g.on('mousemove',(e,d)=>{const x=d.data;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';
 tip.textContent=`${x.n}  [${x.c}]  ply ${d.depth}\\nsubtree ${d.value} nodes\\n`+(x.k?`kind ${x.k}  via ${x.via}  move idx ${x.i}\\ndepth ${x.d} (wanted ${x.fd})  empties ${x.e}\\nwindow (${x.a}, ${x.b})  static ${x.st}  score ${x.s}`:'');});
g.on('mouseleave',()=>tip.style.display='none');
</script></body></html>'''
html = html.replace('__DATA__', json.dumps(tree, separators=(',', ':'))).replace('__INFO__', json.dumps(text))
open(f'{prefix}.tree{k}.html', 'w', encoding='utf-8').write(html)
print(f'wrote {prefix}.tree{k}.html')
