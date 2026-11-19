import sys
p=sys.argv[1]; n=s=0
with open(p,'rb') as f:
    while True:
        b=f.read(48*200000)
        if not b: break
        v=b[40::48]; n+=len(v); s+=sum(v)
print(p, 'records', n, 'stm=1', s, 'share %.4f'%(s/n))
