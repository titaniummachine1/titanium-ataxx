import glob,os,json,math,numpy as np
from sklearn.linear_model import LogisticRegression
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
            f.append(prev[-1]/400.0 if prev else 0.0)
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

# --- combined estimator: T = a*R + (1-a)*Ps, Ps = 0.5 + slope*(pred-0.5); per-game stats within each held-out gate
sl=np.polyfit(pred-0.5,y-0.5,1)[0]
print('global slope',round(sl,2))
rat=[];rho_l=[]; zr=[];zp=[];zc=[]
for fi in sorted(set(grp)):
    s=grp==fi
    if s.sum()<300: continue
    R=y[s]; Ps=0.5+sl*(pred[s]-0.5)
    vR=R.var(ddof=1); vP=Ps.var(ddof=1); c=np.cov(R,Ps)[0,1]
    a=(vP-c)/(vR+vP-2*c)   # min-variance weight on R
    a=min(1,max(0,a))
    vT=a*a*vR+(1-a)**2*vP+2*a*(1-a)*c
    rho=c/math.sqrt(vR*vP); rat.append(vT/vR); rho_l.append(rho)
    n=s.sum(); T=a*R+(1-a)*Ps
    zr.append((R.mean()-.5)/math.sqrt(vR/n)); zp.append((Ps.mean()-.5)/math.sqrt(vP/n)); zc.append((T.mean()-.5)/math.sqrt(T.var(ddof=1)/n))
print('median per-game corr(R,Ps) %.2f; variance ratio of the optimal blend vs result alone: median %.2f (=> %.0f%% fewer games for the same se)'%(np.median(rho_l),np.median(rat),100*(1-np.median(rat))))
print('mean |z| actual %.2f  pred %.2f  blend %.2f'%(np.mean(np.abs(zr)),np.mean(np.abs(zp)),np.mean(np.abs(zc))))

# --- split-half reliability across gates (pairs: even/odd by book), estimator quality = corr(half A, half B) over gates
book=np.array([b if b is not None else -1 for b in book])
def half_means(fn,minn=300):
    A=[];B=[]
    for fi in sorted(set(grp)):
        s=np.where(grp==fi)[0]
        if len(s)<minn: continue
        bk=book[s]; order=np.unique(bk); rng=np.random.RandomState(fi); rng.shuffle(order); h=set(order[:len(order)//2])
        ia=[i for i in s if book[i] in h]; ib=[i for i in s if book[i] not in h]
        A.append(fn(ia));B.append(fn(ib))
    return np.array(A),np.array(B)
sl2=1.83
for name,fn in (('result R',lambda ix:(y[ix]-.5).mean()),
                ('pred scaled x1.83',lambda ix:sl2*(pred[ix]-.5).mean()),
                ('blend 50/50',lambda ix:.5*(y[ix]-.5).mean()+.5*sl2*(pred[ix]-.5).mean()),
                ('blend 30R/70P',lambda ix:.3*(y[ix]-.5).mean()+.7*sl2*(pred[ix]-.5).mean())):
    rs=[]
    for rep in range(40):
        np.random.seed(rep)
        A,B=half_means(fn)
        rs.append(np.corrcoef(A,B)[0,1])
    print('%-20s split-half reliability (mean of 40 splits) %.3f'%(name,np.mean(rs)))

# --- sequential back-test: games-to-95% for result-only vs blended (weight w on eval part) and false early crossings
print('\n--- sequential back-test (pairs in log order, checkpoints every 25 pairs) ---')
SL=1.83
def seq(fi,w):
    s=np.where(grp==fi)[0]
    pr={}
    for i in s: pr.setdefault(book[i],[]).append(i)
    order=[v for v in pr.values() if len(v)==2]
    Rp=np.array([(y[a]+y[b])/2-.5 for a,b in order]); Pp=np.array([SL*((pred[a]+pred[b])/2-.5) for a,b in order])
    T=(1-w)*Rp+w*Pp
    return Rp,T
def firstcross(v,thr=1.645,step=25,minp=50):
    for n in range(minp,len(v)+1,step):
        x=v[:n]; sd=x.std(ddof=1)
        if sd>0 and x.mean()/(sd/math.sqrt(n))>=thr: return n
    return None
for w in (0.0,0.3,0.5):
    cross=[];fals=0;pos=0;tot=0
    for fi in sorted(set(grp)):
        Rp,T=seq(fi,w)
        if len(Rp)<150: continue
        tot+=1
        zf=Rp.mean()/(Rp.std(ddof=1)/math.sqrt(len(Rp)))
        c=firstcross(T)
        if c is not None:
            pos+=1
            if zf<1.0: fals+=1       # crossed 95% early but the full gate ended below z 1.0
            cross.append((c,len(Rp),zf))
    print('w=%.1f gates %d, crossed 95%%: %d, of which full-gate z<1.0 (false early cross): %d | mean pairs at crossing %.0f'%(w,tot,pos,fals,np.mean([c for c,_,_ in cross]) if cross else 0))
    if w in (0.0,0.5): print('   crossings (pairs, gate pairs, final z):',[(c,n,round(z,1)) for c,n,z in cross][:12])
