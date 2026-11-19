#!/bin/bash
# Cross-session protocol queue, one lane per process:  bash scripts/xs_queue.sh LANE
# Each line of $XS/queue/LANE.txt:  name|prerequisites (names separated by commas)|environment variables|command
# - a step starts when all its prerequisites (also from other lanes) are in done/;
# - finished steps (done/name) are skipped; a failed step is retried once; after 2 consecutive failures it stays
#   stopped (failed/name = 2) and so do the steps that depend on it;
# - the file is re-read after every step (lines can be added while it runs); the lane ends when the file
#   contains "#FINE" and nothing is left that could still start.
# Launch it detached from the session:  nohup caffeinate -i bash scripts/xs_queue.sh LANE >/dev/null 2>&1 &
# Restart: just relaunch. xs_run.py skips the sessions already in the registry and reloads the models from the cache.
XS=${TESI_XS:-$HOME/tesi_h1_work/xs}
LANE=$1
Q="$XS/queue"
mkdir -p "$Q/done" "$Q/failed" "$Q/logs"
cd "$HOME/tesi_h1_work" || exit 1
echo "$(date '+%F %T') [$LANE] lane started (pid $$)" >> "$Q/logs/coda.log"
echo $$ > "$Q/logs/$LANE.pid"
while true; do
  ran=0; waiting=0
  while IFS='|' read -r name needs envs cmd; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -e "$Q/done/$name" ]] && continue
    n=$(cat "$Q/failed/$name" 2>/dev/null || echo 0)
    (( n >= 2 )) && continue
    ready=1; dead=0
    IFS=',' read -ra req <<< "$needs"
    for r in "${req[@]}"; do
      [[ -z "$r" || -e "$Q/done/$r" ]] && continue
      ready=0
      m=$(cat "$Q/failed/$r" 2>/dev/null || echo 0)
      (( m >= 2 )) && dead=1
    done
    (( dead )) && continue
    if (( ! ready )); then waiting=1; continue; fi
    echo "$(date '+%F %T') [$LANE] INIZIO $name (attempt $((n + 1)))" >> "$Q/logs/coda.log"
    if env $envs bash -c "$cmd" < /dev/null >> "$Q/logs/$name.log" 2>&1; then
      touch "$Q/done/$name"; rm -f "$Q/failed/$name"
      echo "$(date '+%F %T') [$LANE] FINE   $name" >> "$Q/logs/coda.log"
    else
      echo $((n + 1)) > "$Q/failed/$name"
      echo "$(date '+%F %T') [$LANE] ERROR  $name (attempt $((n + 1)), see logs/$name.log)" >> "$Q/logs/coda.log"
    fi
    ran=1; break
  done < "$Q/$LANE.txt"
  if (( ran == 0 )); then
    if grep -q '^#FINE' "$Q/$LANE.txt" && (( waiting == 0 )); then break; fi
    sleep 60
  fi
done
echo "$(date '+%F %T') [$LANE] lane finished" >> "$Q/logs/coda.log"
