#!/bin/bash
# D1041: self-play generation on a Docker box (user 2026-11-17: 2 x 32 vCPU for ~3 h, then copy home + delete).
# Runs inside the box from /tmp/sn: ./ti (Linux hub build), fp.txt (main fingerprint made on the PC by
# `ti gen-fingerprint`), seq_dag.tgz (snapshot of the PC's sequence DAG, seq_dag.db + -log + -wal). Usage: d1041_box_gen.sh SHARD THREADS NAME
# Batches of 500 games so the work is in finished files; a STOP file ends the loop after the running batch.
set -u
K=$1; T=$2; NAME=$3
cd /tmp/sn
[ -f seq_dag.db ] || tar xzf seq_dag.tgz
mkdir -p out
chmod +x ti
echo "$(date -u +%FT%TZ) start shard $K/3 threads $T" >> out/run.log
i=1
while [ ! -f STOP ]; do
  b=$(printf 'batch_%04d' $i)
  echo "$(date -u +%FT%TZ) $b start" >> out/run.log
  ./ti gen --out out/$b --games 500 --nodes 5000 --threads $T --seq-dag seq_dag.db --src ${NAME}_$b \
     --main-fingerprint fp.txt --shard $K/3 > out/$b.log 2>&1
  rc=$?
  echo "$(date -u +%FT%TZ) $b end rc $rc" >> out/run.log
  [ $rc -ne 0 ] && break
  i=$((i+1))
done
echo "$(date -u +%FT%TZ) loop end" >> out/run.log
