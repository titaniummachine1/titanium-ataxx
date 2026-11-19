# D748 (2026-11-13): are the rows of a .tdat file still in game order?
# A row i is "linked" to row i-1 when its position is exactly one legal-looking move after row i-1:
#   new mover's stones  = old opponent minus the flipped stones           (subset test)
#   new opponent stones = old mover (minus at most the jump origin) + flipped + exactly ONE new square on an empty cell
# Two conventions are tested: A = occ0 is always the side to move; B = occ0 is colour 0 and `stm` says who moves.
# Streams the file in chunks (RAM-safe). Analysis only, nothing is written.
# Usage: powershell -File docs\analysis\tdat_chain_scan.ps1 -Path data\nnue\packed\train_v9.tdat
param([string]$Path = 'C:\gitProjects\titanium-ataxx\data\nnue\packed\train_v9.tdat')
Add-Type -TypeDefinition @'
using System; using System.IO;
public static class TdatScan {
  static int Pop(ulong x){ int c=0; while(x!=0){ x&=x-1; c++; } return c; }
  static bool Link(ulong pm, ulong po, ulong m, ulong o){
    // pm/po = previous mover/opponent, m/o = current mover/opponent (current mover = previous opponent side)
    if ((m & ~po) != 0) return false;                 // current mover only lost stones
    ulong flipped = po & ~m;
    if ((flipped & ~o) != 0) return false;            // every lost stone now belongs to the other side
    ulong gone = pm & ~o;                              // previous mover stones that vanished: 0 (clone) or 1 (jump origin)
    if (Pop(gone) > 1) return false;
    ulong fresh = o & ~pm & ~po;                       // exactly one new stone on a previously empty square
    if (Pop(fresh) != 1) return false;
    if ((o & ~(pm | flipped | fresh)) != 0) return false;
    return true;
  }
  public static string Run(string path){
    const int REC=48; const int CH=1<<16;
    var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1<<20);
    long rows = fs.Length / REC; var buf = new byte[REC*CH];
    long linkA=0, linkB=0, outOk=0, stm1=0; long i=0;
    ulong p0=0,p1=0; byte pstm=0; float pout=0;
    long run=1; long[] hist=new long[6]; long[] histRows=new long[6]; // 1, 2-9, 10-29, 30-59, 60-99, 100+
    long[] dec=new long[10]; long[] decN=new long[10];
    long[,] tk=new long[10,8]; long[] st=new long[10]; long[] prv=new long[10]; long[] c5=new long[10];
    Action<long> close = (len)=>{ int k = len==1?0: len<10?1: len<30?2: len<60?3: len<100?4:5; hist[k]++; histRows[k]+=len; };
    int n;
    while((n=fs.Read(buf,0,buf.Length))>0){
      int cnt=n/REC;
      for(int j=0;j<cnt;j++,i++){
        int b=j*REC;
        ulong o0=BitConverter.ToUInt64(buf,b), o1=BitConverter.ToUInt64(buf,b+8);
        float outc=BitConverter.ToSingle(buf,b+16); byte stm=buf[b+40]; if(stm!=0) stm1++;
        int d=(int)(i*10/rows); decN[d]++;
        tk[d, Math.Min(7,(int)buf[b+41])]++; if(stm!=0) st[d]++;
        if(Math.Abs(BitConverter.ToSingle(buf,b+20))>=98000f) prv[d]++;
        if(BitConverter.ToSingle(buf,b+32)!=0f) c5[d]++;
        if(i>0){
          bool a = Link(p0,p1,o0,o1);
          ulong pm = pstm==0?p0:p1, po = pstm==0?p1:p0, m = stm==0?o0:o1, o = stm==0?o1:o0;
          bool bb = (stm!=pstm) && Link(pm,po,m,o);
          if(a) linkA++; if(bb) linkB++;
          bool l = a || bb;
          if(l){ dec[d]++; if(Math.Abs((outc + pout) - 1.0f) < 0.02f) outOk++; run++; }
          else { close(run); run=1; }
        }
        p0=o0;p1=o1;pstm=stm;pout=outc;
      }
    }
    close(run); fs.Close();
    var sb=new System.Text.StringBuilder();
    sb.AppendFormat("rows {0}  stm=1 rows {1}\n", rows, stm1);
    sb.AppendFormat("linked to previous row: convention A {0} ({1:F1}%), convention B {2} ({3:F1}%)\n", linkA, 100.0*linkA/rows, linkB, 100.0*linkB/rows);
    sb.AppendFormat("linked rows whose outcome = 1 - previous outcome: {0} ({1:F1}% of linked)\n", outOk, 100.0*outOk/Math.Max(1,linkA+linkB));
    string[] nm={"1","2-9","10-29","30-59","60-99","100+"};
    for(int k=0;k<6;k++) sb.AppendFormat("chains of length {0,-6}: {1,9} chains, {2,10} rows ({3:F1}%)\n", nm[k], hist[k], histRows[k], 100.0*histRows[k]/rows);
    for(int k=0;k<10;k++) sb.AppendFormat("file decile {0}: linked {1:F1}% | teacher_kind 0/1/2/3 = {2:F0}/{3:F0}/{4:F0}/{5:F0}% | stm=1 {6:F0}% | proven {7:F0}% | cp5k present {8:F0}%\n", k, 100.0*dec[k]/Math.Max(1,decN[k]),
      100.0*tk[k,0]/decN[k], 100.0*tk[k,1]/decN[k], 100.0*tk[k,2]/decN[k], 100.0*tk[k,3]/decN[k], 100.0*st[k]/decN[k], 100.0*prv[k]/decN[k], 100.0*c5[k]/decN[k]);
    return sb.ToString();
  }
}
'@
[TdatScan]::Run($Path)
