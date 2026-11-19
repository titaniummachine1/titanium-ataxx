import glob,json,math,numpy as np
G={}
for f in sorted(glob.glob('logs/gates/*.games.jsonl')+glob.glob('../logs/sbx_gate1/res/*.games.jsonl')):
    P={}
    for l in open(f,errors='ignore'):
        try:g=json.loads(l)
        except: continue
        m=(g['final_x']-g['final_o'])*(1 if g['candidate_seat']=='x' else -1)
        P.setdefault(g['book'],[]).append((g['candidate_points'],m))
    pr=[v for v in P.values() if len(v)==2]
    if len(pr)>=150: G[f]=pr
print('gates',len(G))
def pairval(v,kind,k):
    r=(v[0][0]+v[1][0])/2-.5; t=(math.tanh(v[0][1]/k)+math.tanh(v[1][1]/k))/4
    return {'result':r,'tanh':t,'blend':0.5*r+0.5*t,'blend25':0.75*r+0.25*t}[kind]
def rel(kind,k,reps=200):
    rs=[]
    for rep in range(reps):
        rng=np.random.RandomState(rep);A=[];B=[]
        for pr in G.values():
            idx=rng.permutation(len(pr)); h=len(pr)//2
            A.append(np.mean([pairval(pr[i],kind,k) for i in idx[:h]])); B.append(np.mean([pairval(pr[i],kind,k) for i in idx[h:]]))
        rs.append(np.corrcoef(A,B)[0,1])
    return np.mean(rs),np.std(rs)
for kind,k in (('result',10),('tanh',5),('tanh',10),('tanh',20),('blend',10),('blend25',10)):
    m,s=rel(kind,k); print('%-8s k=%-3d split-half reliability %.3f (sd over splits %.3f)'%(kind,k,m,s))
