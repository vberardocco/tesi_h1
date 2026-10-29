#!/bin/bash
# Coda delle repliche. Ogni riga di queue.txt: nome|variabili d'ambiente|comando
# - salta i passi finiti (file done/nome) e quelli falliti 3 volte (failed/nome);
# - rilegge queue.txt dopo ogni passo: si possono aggiungere passi mentre gira;
# - se non c'è niente da fare aspetta; finisce quando queue.txt contiene la riga "#FINE".
# Ripartenza: rilanciare questo script. I notebook del 04 rileggono dal disco i modelli già allenati.
cd "$HOME/tesi_h1_work/rep" || exit 1
mkdir -p logs done failed
echo "$(date '+%F %T') coda avviata (pid $$)" >> logs/coda.log
while true; do
  ran=0
  while IFS='|' read -r name envs cmd; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [[ -e "done/$name" ]] && continue
    n=$(cat "failed/$name" 2>/dev/null || echo 0)
    (( n >= 3 )) && continue
    echo "$(date '+%F %T') INIZIO $name (tentativo $((n + 1)))" >> logs/coda.log
    if env $envs bash -c "$cmd" < /dev/null >> "logs/$name.log" 2>&1; then
      touch "done/$name"; echo "$(date '+%F %T') FINE   $name" >> logs/coda.log
    else
      echo $((n + 1)) > "failed/$name"; echo "$(date '+%F %T') ERRORE $name (vedi logs/$name.log)" >> logs/coda.log
    fi
    ran=1; break
  done < queue.txt
  if (( ran == 0 )); then
    grep -q '^#FINE' queue.txt && break
    sleep 120
  fi
done
echo "$(date '+%F %T') coda terminata" >> logs/coda.log
