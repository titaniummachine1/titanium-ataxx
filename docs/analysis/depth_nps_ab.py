"""D502: nodes and wall time at fixed depth (go depth D) on 57 real game positions, engines one after the other.
usage: python -I depth_nps_ab.py <games.jsonl> <depth> "<cmd1>" "<cmd2>" ...   (2 passes)"""
import sys,subprocess,time,json,re
g,dep,cmds=sys.argv[1],int(sys.argv[2]),sys.argv[3:]
pos=[]
for i,l in enumerate(open(g)):
    if i>=12: break
    d=json.loads(l); mv=[m["move"] for m in d["moves"]]
    for p in [10,25,40,55,70]:
        if p<len(mv)-2: pos.append("fen "+d["start"]+" moves "+" ".join(mv[:p]))
def run(c):
    pr=subprocess.Popen(c,shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
    tot=0;t=time.time()
    for ps in pos:
        pr.stdin.write("uainewgame\nposition "+ps+"\ngo depth %d\n"%dep);pr.stdin.flush()
        while True:
            l=pr.stdout.readline()
            m=re.search(r"nodes (\d+)",l)
            if m: tot+=int(m.group(1)); pr.stdout.readline(); break
    el=time.time()-t; pr.stdin.write("quit\n");pr.stdin.flush();pr.wait()
    return tot,el
for rep in range(2):
    for c in cmds:
        n,el=run(c); print("%-60s nodes %d  %.2f s  %.0f knps"%(c[-60:],n,el,n/el/1000),flush=True)
