#!/bin/bash
# Protocollo cross-sessione: rifà l'analisi sui risultati che ci sono, copia nel repo registro, sessioni, tabelle, figure,
# scelte dello pseudo-few-shot e log della coda, poi commit e push da ~/tesi_h1_git. Gli npz delle predizioni restano fuori.
# Uso: bash scripts/xs_commit.sh "messaggio"   (un lock evita due commit insieme da corsie diverse)
MSG=$1
W=$HOME/tesi_h1_work; G=$HOME/tesi_h1_git
XS=${TESI_XS:-$W/xs}
PY=/opt/miniconda3/envs/tesi-manifolds/bin/python
LOCK=$XS/queue/git.lock
until mkdir "$LOCK" 2>/dev/null; do sleep 5; done
trap 'rmdir "$LOCK"' EXIT
cd "$W" && $PY scripts/xs_analysis.py || exit 1
D=$G/results/crosssession
mkdir -p "$D/tune" "$D/models" "$D/logs"
cp "$W/results/h1_runs.csv" "$W/results/sessions.csv" "$G/results/"
cp "$W"/results/crosssession/*.csv "$W"/results/crosssession/*.md "$W"/results/crosssession/*.tex "$W"/results/crosssession/*.png "$D/" 2>/dev/null
cp "$XS"/tune/*.json "$D/tune/" 2>/dev/null
cp "$XS"/models/*.json "$D/models/" 2>/dev/null
cp "$XS"/queue/logs/*.log "$D/logs/" 2>/dev/null
cp "$XS"/queue/*.txt "$D/logs/" 2>/dev/null
cd "$G" || exit 1
git add results/
if git diff --cached --quiet; then echo "niente da committare"; exit 0; fi
git commit -q -m "$MSG" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && git log --oneline -1
