import glob,os,json,math,numpy as np
from sklearn.linear_model import LogisticRegression
import sys
SPLIT=len(sys.argv)>1
KS=(16,24,32,40,48,56)
def val(m):
    if m.get('mate') is not None: return 1500.0*(1 if m['mate']>0 else -1)
    s=m.get('score'); return None if s is None else max(-1500.0,min(1500.0,float(s)))
def feats(g):
    mv=g.get('moves') or []
    c=[];b=[]  # (ply, candidate-view value) from each engine's own eval
    for i,m in enumerate(mv):
        v=val(m)
        if v is None: continue
        (c if m['mover']=='candidate' else b).append((i, v if m['mover']=='candidate' else -v))
    if len(c)<8 or len(b)<8: return None
    f=[]
    for K in KS:
        for L in (c,b):
            prev=[v for i,v in L if i<=K]
            v=prev[-1]/400.0 if prev else 0.0
            f.extend([max(v,0.0),min(v,0.0)]) if SPLIT else f.append(v)
    return f
X=[];y=[];grp=[];book=[]
files=sorted(glob.glob('logs/gates/*.games.jsonl')+glob.glob('../logs/sbx_gate1/res/*.games.jsonl'))
for fi,f in enumerate(files):
    for l in open(f,errors='ignore'):
        try:g=json.loads(l)
        except: continue
        ft=feats(g)
        if ft is None: continue
        X.append(ft); y.append(g['candidate_points']); grp.append(fi); book.append(g.get('book'))
X=np.array(X);y=np.array(y);grp=np.array(grp)
print('games',len(y),'features',X.shape[1],'gates',len(set(grp)))
yb=(y>0.5).astype(int)   # draws counted as loss for fitting; handled by using y==0.5 weight below
pred=np.zeros(len(y))
for fi in set(grp):
    tr=grp!=fi
    # soft labels via duplication: win=1, draw=0.5 -> two rows weights
    Xt=np.vstack([X[tr],X[tr]]); yt=np.concatenate([np.ones(tr.sum()),np.zeros(tr.sum())]); wt=np.concatenate([y[tr],1-y[tr]])
    m=LogisticRegression(C=1.0,max_iter=500).fit(Xt,yt,sample_weight=wt)
    pred[~tr]=m.predict_proba(X[~tr])[:,1]
res=[]
for fi in sorted(set(grp)):
    s=grp==fi
    if s.sum()<150: continue
    a=y[s].mean(); p=pred[s].mean(); sea=y[s].std(ddof=1)/math.sqrt(s.sum()); sep=pred[s].std(ddof=1)/math.sqrt(s.sum())
    res.append((os.path.basename(files[fi])[:-16],s.sum(),a,p,sea,sep))
print('gate n actual predicted se_actual se_pred   (held-out)')
for r in res: print('%-24s %4d %.3f %.3f  %.4f %.4f'%r)
A=np.array([r[2] for r in res]);Pp=np.array([r[3] for r in res])
print('corr(actual,pred) over gates %.3f; slope %.2f; mean se actual %.4f vs pred %.4f (ratio %.2f)'%(np.corrcoef(A,Pp)[0,1],np.polyfit(Pp-0.5,A-0.5,1)[0],np.mean([r[4] for r in res]),np.mean([r[5] for r in res]),np.mean([r[5] for r in res])/np.mean([r[4] for r in res])))
brier0=((y-y.mean())**2).mean();brier=((y-pred)**2).mean();print('Brier base %.4f model %.4f'%(brier0,brier))

