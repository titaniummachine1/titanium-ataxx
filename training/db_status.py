from libsql_client import dbapi2 as sqlite3
import sys

p = sys.argv[1] if len(sys.argv) > 1 else r"C:\gitProjects\titanium-ataxx\data\nnue\titanium_train.db"
c = sqlite3.connect(p)
print("rows", c.execute("select count(1) from positions").fetchone()[0])
print("done", c.execute("select count(1) from positions where relabel_status='done'").fetchone()[0])
print("sources", list(c.execute("select source,count(1) from positions group by source")))
print("splits", list(c.execute("select split,count(1) from positions group by split")))
print("phases", list(c.execute("select phase,count(1) from positions group by phase")))
print("sample", list(c.execute("select id,engine_cp,outcome,engine_nodes from positions where engine_cp is not null limit 5")))
c.close()
