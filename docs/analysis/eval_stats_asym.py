import glob,json,math,numpy as np
rows=[]   # (mover score cp, mover's eventual points, ply, mover is candidate)
for f in glob.glob('logs/gates/*.games.jsonl')+glob.glob('../logs/sbx_gate1/res/*.games.jsonl'):
    for l in open(f,errors='ignore'):
        try:g=json.loads(l)
        except: continue
        mv=g.get('moves') or []
        for m in mv:
            if m.get('mate') is not None or m.get('score') is None or abs(m['score'])>1500: continue
            p=m['ply']
            if p<20 or p>60: continue
            pts=g['candidate_points'] if m['mover']=='candidate' else 1-g['candidate_points']
            rows.append((m['score'],pts,p,m['mover']=='candidate'))
R=np.array(rows,dtype=float); s,pts=R[:,0],R[:,1]
print('moves',len(R))
bins=[-1500,-600,-400,-250,-150,-75,-25,25,75,150,250,400,600,1500]
print('mover score bin   n    actual win%   sigmoid(s/400) expected   gap (actual-expected)')
for a,b in zip(bins,bins[1:]):
    k=(s>=a)&(s<b); 
    if k.sum()<200: continue
    exp=(1/(1+np.exp(-s[k]/400))).mean(); act=pts[k].mean()
    print('%5d..%5d %7d   %.3f         %.3f                 %+.3f'%(a,b,k.sum(),act,exp,act-exp))
# trust test: how often is the sign right, by optimistic vs pessimistic
for name,k in (('optimistic s>150',s>150),('pessimistic s<-150',s<-150)):
    right=(pts[k]>0.5) if s[k].mean()>0 else (pts[k]<0.5)
    print('%-20s n %6d  sign right %.1f%%  mean |score| %.0f'%(name,k.sum(),100*right.mean(),np.abs(s[k]).mean()))
