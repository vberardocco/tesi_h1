#!/bin/bash
# Queue of the replications. Each line of queue.txt: name|environment variables|command
# - skips the finished steps (file done/name) and those that failed 3 times (failed/name);
# - rereads queue.txt after every step: steps can be added while it runs;
# - if there is nothing to do it waits; it ends when queue.txt contains the line "#FINE".
# Restart: just relaunch this script. The 04 notebooks reload the already trained models from disk.
cd "$HOME/tesi_h1_work/rep" || exit 1
mkdir -p logs done failed
echo "$(date '+%F %T') queue started (pid $$)" >> logs/coda.log
while true; do
  ran=0
  while IFS='|' read -r name envs cmd; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -e "done/$name" ]] && continue
    n=$(cat "failed/$name" 2>/dev/null || echo 0)
    (( n >= 3 )) && continue
    echo "$(date '+%F %T') INIZIO $name (attempt $((n + 1)))" >> logs/coda.log
    if env $envs bash -c "$cmd" < /dev/null >> "logs/$name.log" 2>&1; then
      touch "done/$name"; echo "$(date '+%F %T') FINE   $name" >> logs/coda.log
    else
      echo $((n + 1)) > "failed/$name"; echo "$(date '+%F %T') ERRORE $name (see logs/$name.log)" >> logs/coda.log
    fi
    ran=1; break
  done < queue.txt
  if (( ran == 0 )); then
    grep -q '^#FINE' queue.txt && break
    sleep 120
  fi
done
echo "$(date '+%F %T') queue finished" >> logs/coda.log
