#!/bin/sh
# D1611 (user order 'merge new rows from the proof hunt and upgrade existing ones'): finished proof batches of the D1604 hunt
# -> train_v17 = train_v16 + NEW proven positions (before the held-out tail) + every existing copy of a proven position
# upgraded (outcome 1/0, proof_kind 6, mate cp unless it already holds a same-sign proof; older opposite proofs overwritten
# and logged, D1607; contradictions between new proofs isolated, D16). NEW files only: train_v16 / train_v16h / val_v16tail
# untouched (sha before/after). STANDARD_TRAIN.txt is NOT changed (A13 = user order). Needs ti.exe NOT running a hunt.
set -e
TI=/c/gitProjects/titaniumtest/target_main/x86_64-pc-windows-gnu/release/ti.exe
cd /c/gitProjects/titanium-ataxx
P=data/nnue/packed; H=data/proofs/hunt_v16; W=data/datasets/v17/proofs; TAIL=1528091
BASE=${BASE:-train_v16}; OUT=${OUT:-train_v17}
mkdir -p $W
[ -e $P/$OUT.tdat ] && { echo "$P/$OUT.tdat exists, refusing"; exit 1; }
LIST=""; KEYS=""; NB=0
for r in $H/*.report.txt; do
  grep -q PLACEHOLDER $r && continue
  s=${r%.report.txt}; LIST="$LIST${LIST:+,}$s.tgam"; KEYS="$KEYS${KEYS:+,}$s.conflicts.keys"; NB=$((NB+1))
done
# EXTRA = comma list of proof .tgam whose batch was stopped before its report (lines already written are complete; D1611)
for x in $(echo "${EXTRA:-}" | tr ',' ' '); do LIST="$LIST,$x"; NB=$((NB+1)); done
echo "== $NB proof batches: $LIST"
sha() { sha256sum $1 | cut -c1-16; }
echo "== before: $BASE $(sha $P/$BASE.tdat)  ${BASE}h $(sha $P/${BASE}h.tdat)  val tail $(sha $P/val_v16tail.tdat)"
echo "== 1 every proof-line position (no dedup) $(date +%T)"
"$TI" data games-tdat --tgam "$LIST" --out $W/proof_all 2>&1 | tail -3
echo "== 2 contradictions between new proofs $(date +%T)"
"$TI" data proof-upgrade --proofs $W/proof_all.tdat --check-only --report $W/check --exclude-keys "$KEYS"
cat $(echo $KEYS | tr ',' ' ') $W/check.contradictions.keys | sort -u > $W/excluded.keys
echo "excluded keys: $(wc -l < $W/excluded.keys)"
echo "== 3 NEW positions only (dedup within the proofs and against the whole $BASE incl. the held-out tail) $(date +%T)"
"$TI" data games-tdat --tgam "$LIST" --dedup --exclude-keys $W/excluded.keys --dedup-against $P/$BASE.tdat --out $W/proof_new 2>&1 | tail -4
echo "== 4 checks of the new rows $(date +%T)"
"$TI" data check-leak --train $W/proof_new --val $P/val_v16tail --threads 6 2>&1 | tail -4
"$TI" data chain-audit --tdat $W/proof_new.tdat --chains $W/proof_new.chains 2>&1 | tail -4
echo "== 5 append before the held-out tail $(date +%T)"
"$TI" data append --base $P/$BASE --add $W/proof_new --out $P/${OUT}pre --before-tail $TAIL 2>&1 | tail -4
echo "== 6 upgrade diff $(date +%T)"
"$TI" data proof-upgrade --proofs $W/proof_all.tdat --exclude-keys $W/excluded.keys --base $P/${OUT}pre.tdat --preserve-tail $TAIL --out $W/${OUT}_upgrade.tdif --report $W/upgrade
echo "== 7 patch (fingerprint-verified) $(date +%T)"
"$TI" data patch --base $P/${OUT}pre.tdat --diff $W/${OUT}_upgrade.tdif --out $P/$OUT.tdat
for x in aux kata chains; do mv $P/${OUT}pre.$x $P/$OUT.$x; done
NR=$(( $(stat -c %s $P/$OUT.tdat) / 48 )); NN=$(( $(stat -c %s $W/proof_new.tdat) / 48 ))
NC=$(od -An -t u8 -j 32 -N 8 $W/${OUT}_upgrade.tdif | tr -d ' ')
printf '{"src":"%s","rows":%s,"rec":48,"kata_rec":16,"aux_rec":12,"holdout_tail_rows":%s}' \
  "$P/$BASE.tdat + $W/proof_new.tdat ($NN new proven positions of $NB D1604 proof batches, inserted before the held-out tail by ti data append; kata/aux zero-padded) + $W/${OUT}_upgrade.tdif ($NC rows upgraded by ti data proof-upgrade D1611, applied by ti data patch)" \
  $NR $TAIL > $P/$OUT.manifest.json
rm $P/${OUT}pre.tdat $P/${OUT}pre.manifest.json
echo "== 8 split $(date +%T)"
"$TI" data split --base $P/$OUT --tail $TAIL --head-out $P/${OUT}h --tail-out $P/val_v17tail 2>&1 | tail -3
echo "== 9 verify $(date +%T)"
A=$(sha $P/val_v16tail.tdat); B=$(sha $P/val_v17tail.tdat)
echo "val_v16tail $A  val_v17tail $B -> $( [ "$A" = "$B" ] && echo MATCH || echo MISMATCH )"
echo "rows $OUT $NR = $BASE $(( $(stat -c %s $P/$BASE.tdat) / 48 )) + new $NN"
echo "== after: $BASE $(sha $P/$BASE.tdat)  ${BASE}h $(sha $P/${BASE}h.tdat)"
python -I docs/analysis/d1611_check_v17.py $P/$BASE.tdat $P/$OUT.tdat $TAIL $NN $W/proof_all.tdat $W/excluded.keys
echo "== DONE $(date +%T)"
