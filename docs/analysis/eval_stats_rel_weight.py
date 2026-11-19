import glob,json,math,numpy as np
def sig(x,k=1.0): return 1/(1+np.exp(-k*x/400))
def load(f):
    out=[]
    for l in open(f,errors='ignore'):
        try:g=json.loads(l)
        except: continue
        c=[];b=[]
        for m in g.get('moves') or []:
            if m.get('mate') is not None or m.get('score') is None or abs(m['score'])>1500 or not (20<=m['ply']<=60): continue
            (c if m['mover']=='candidate' else b).append(m['score'] if m['mover']=='candidate' else -m['score'])  # candidate view
        if len(c)<5 or len(b)<5: continue
        out.append((g['book'],g['candidate_points'],np.mean(c),np.mean(b)))
    P={}
    for bk,pt,c,b in out: P.setdefault(bk,[]).append((pt,c,b))
    return [v for v in P.values() if len(v)==2]
def brier(pairs,col,K):   # how well one engine's own view predicts the pair result (candidate view)
    e=0;n=0
    for v in pairs:
        for pt,c,b in v:
            s=c if col==1 else b
            e+=(pt-sig(s,K))**2;n+=1
    return e/n
def agree(pairs,wc,wb):
    ok=0;n=0
    for v in pairs:
        r=sum(x[0] for x in v)/2-.5
        t=sum((wc*x[1]+wb*x[2])/(wc+wb) for x in v)/2
        if r!=0 and t!=0: n+=1; ok+= (r*t>0)
    return ok/n,n
res={'skill^1':[],'skill^2':[],'skill^4':[],'unweighted':[], 'reliability-weighted (fit on other half)':[], 'winner-only (oracle, leaks)':[], 'candidate-only':[], 'baseline-only':[]}
K=2.0
import sys
MINP=int(sys.argv[1]) if len(sys.argv)>1 else 150
ONLY=sys.argv[2] if len(sys.argv)>2 else ""
for f in sorted(glob.glob('logs/gates/*.games.jsonl')+glob.glob('../logs/sbx_gate1/res/*.games.jsonl')):
    if ONLY and ONLY not in f: continue
    P=load(f)
    print(f[-40:], len(P),'pairs')
    print(f[-40:], len(P), 'pairs')
    if len(P)<MINP: continue
    rng=np.random.RandomState(1)
    for rep in range(20):
        idx=rng.permutation(len(P)); h=len(P)//2
        A=[P[i] for i in idx[:h]]; B=[P[i] for i in idx[h:]]
        bc,bb=brier(A,1,K),brier(A,2,K); sc,sb=max(0.25-bc,1e-4),max(0.25-bb,1e-4); wc,wb=1/bc,1/bb
        res['unweighted'].append(agree(B,1,1)[0])
        for pw in (1,2,4): res['skill^%d'%pw].append(agree(B,sc**pw,sb**pw)[0])
        res['reliability-weighted (fit on other half)'].append(agree(B,wc,wb)[0])
        win_c = np.mean([sum(x[0] for x in v)/2 for v in P])>0.5
        res['winner-only (oracle, leaks)'].append(agree(B,1,0)[0] if win_c else agree(B,0,1)[0])
        res['candidate-only'].append(agree(B,1,0)[0]); res['baseline-only'].append(agree(B,0,1)[0])
for k,v in res.items(): print('%-42s mean sign-agreement of tilt with pair result on held-out half: %.3f'%(k,np.mean(v)))
