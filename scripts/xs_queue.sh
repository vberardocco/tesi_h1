#!/bin/bash
# Coda del protocollo cross-sessione, una corsia per processo:  bash scripts/xs_queue.sh CORSIA
# Ogni riga di $XS/queue/CORSIA.txt:  nome|prerequisiti (nomi separati da virgole)|variabili d'ambiente|comando
# - un passo parte quando tutti i prerequisiti (anche di altre corsie) sono in done/;
# - i passi finiti (done/nome) si saltano; un passo fallito si riprova una volta; dopo 2 fallimenti di fila resta
#   fermo (failed/nome = 2) e con lui i passi che ne dipendono;
# - il file si rilegge dopo ogni passo (si possono aggiungere righe mentre gira); la corsia finisce quando il file
#   contiene "#FINE" e non resta niente che possa ancora partire.
# Va lanciata staccata dalla sessione:  nohup caffeinate -i bash scripts/xs_queue.sh CORSIA >/dev/null 2>&1 &
# Ripartenza: rilanciare. xs_run.py salta le sessioni già nel registro e rilegge i modelli dalla cache.
XS=${TESI_XS:-$HOME/tesi_h1_work/xs}
LANE=$1
Q="$XS/queue"
mkdir -p "$Q/done" "$Q/failed" "$Q/logs"
cd "$HOME/tesi_h1_work" || exit 1
echo "$(date '+%F %T') [$LANE] corsia avviata (pid $$)" >> "$Q/logs/coda.log"
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
    echo "$(date '+%F %T') [$LANE] INIZIO $name (tentativo $((n + 1)))" >> "$Q/logs/coda.log"
    if env $envs bash -c "$cmd" < /dev/null >> "$Q/logs/$name.log" 2>&1; then
      touch "$Q/done/$name"; rm -f "$Q/failed/$name"
      echo "$(date '+%F %T') [$LANE] FINE   $name" >> "$Q/logs/coda.log"
    else
      echo $((n + 1)) > "$Q/failed/$name"
      echo "$(date '+%F %T') [$LANE] ERRORE $name (tentativo $((n + 1)), vedi logs/$name.log)" >> "$Q/logs/coda.log"
    fi
    ran=1; break
  done < "$Q/$LANE.txt"
  if (( ran == 0 )); then
    if grep -q '^#FINE' "$Q/$LANE.txt" && (( waiting == 0 )); then break; fi
    sleep 60
  fi
done
echo "$(date '+%F %T') [$LANE] corsia terminata" >> "$Q/logs/coda.log"
