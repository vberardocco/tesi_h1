#!/usr/bin/env bash
# Costruisce l'immagine Docker per EvalAI (FALCON H1) senza `docker build`:
# container dalla base locale -> docker cp dei file -> docker commit --change.
# Motivo: con `docker build` Docker ricopia in cache l'intera base (~9 GB) e il disco del Mac si riempie;
# il commit scrive solo lo strato nuovo.
#
# Uso (30/10, decoder sklearn di esempio di FALCON):
#   bash evalai/make_image_h1.sh
# Per giovedì 5/11 (nostra ridge): cambiare DECODER_PKL, DECODE_PY, PHASE=test e IMAGE.
#
# Variabili (con i valori di default):
set -euo pipefail

BASE="${BASE:-falcon_b1_mlp:latest}"                    # base amd64 locale: pytorch 2.4.1, CUDA 12.1, falcon_challenge 1.0.2
IMAGE="${IMAGE:-sk_smoke_h1:latest}"                    # immagine prodotta
FALCON="${FALCON:-$HOME/tesi_h1_work/falcon-challenge}" # clone di snel-repo/falcon-challenge (fuori da iCloud)
DATA="${DATA:-$HOME/000954}"                             # dandiset H1 (montato con -v, non finisce nell'immagine)
DECODER_PKL="${DECODER_PKL:-$FALCON/local_data/sklearn_FalconTask.h1.pkl}"
DECODE_PY="${DECODE_PY:-$FALCON/decoder_demos/sklearn_sample.py}"
PHASE="${PHASE:-minival}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
CNAME="mkimg_h1_$$"

free_gb() { df -g / | awk 'NR==2 {print $4}'; }
check_space() {
  local f; f=$(free_gb)
  echo "[spazio] $1: ${f} GB liberi"
  if [ "$f" -lt "$MIN_FREE_GB" ]; then echo "STOP: meno di ${MIN_FREE_GB} GB liberi" >&2; exit 1; fi
}
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

check_space "inizio"
[ "$(docker image inspect "$BASE" --format '{{.Architecture}}')" = "amd64" ] || { echo "la base non è amd64" >&2; exit 1; }

# 1. Container dalla base, senza entrypoint, con i dati H1 montati in sola lettura.
docker run -d --name "$CNAME" --platform linux/amd64 --entrypoint "" \
  -v "$DATA":/dataset/evaluation_data/000954:ro \
  "$BASE" sleep infinity >/dev/null
dx() { docker exec -w /workspace "$CNAME" bash -c "$1"; }

# 2. Dove è installato falcon_challenge: se è editable dentro /workspace/falcon_b1, quella cartella resta.
echo "[falcon_challenge]"; dx "pip show -f falcon_challenge | grep -E '^(Version|Location|Editable project location)'"
KEEP_B1=0
if dx "pip show -f falcon_challenge" | grep -E '^(Location|Editable project location):' | grep -q '/workspace/falcon_b1'; then
  KEEP_B1=1; echo "  editable in /workspace/falcon_b1: la cartella resta"
fi

# 3. File dell'immagine (come sklearn_sample.Dockerfile, con le righe di H1).
dx "mkdir -p data"
docker cp "$DECODER_PKL"                  "$CNAME":/workspace/data/decoder.pkl
docker cp "$FALCON/decoder_demos"         "$CNAME":/workspace/decoder_demos
docker cp "$FALCON/data_demos"            "$CNAME":/workspace/data_demos
docker cp "$DECODE_PY"                    "$CNAME":/workspace/decode.py
docker cp "$FALCON/decoder_demos/filtering.py" "$CNAME":/workspace/filtering.py
[ "$KEEP_B1" = 1 ] || dx "rm -rf /workspace/falcon_b1"

# 4. Il decoder si carica con le versioni della base? Se no, riallena lì dentro (CV atteso 0.26).
if dx "python -c \"import pickle; pickle.load(open('data/decoder.pkl','rb')); print('pkl ok')\""; then :; else
  echo "[pkl] non si carica: rialleno nella base"
  dx "mkdir -p local_data && python decoder_demos/sklearn_decoder.py \
      --training_dir /dataset/evaluation_data/000954/sub-HumanPitt-held-in-calib/ \
      --calibration_dir /dataset/evaluation_data/000954/sub-HumanPitt-held-out-calib/ --mode all --task h1 \
      && cp local_data/sklearn_FalconTask.h1.pkl data/decoder.pkl && rm -rf local_data"
fi
dx "python -c 'import numpy, sklearn, falcon_challenge; print(\"numpy\", numpy.__version__, \"sklearn\", sklearn.__version__)'"
check_space "prima del commit"

# 5. Commit con configurazione esplicita: niente della base B1.
#    ENV non si può cancellare: le variabili di B1 vengono sovrascritte; KMP_DUPLICATE_LIB_OK (da B1) svuotato.
docker stop -t 1 "$CNAME" >/dev/null
docker commit \
  --change 'ENTRYPOINT []' \
  --change 'WORKDIR /workspace' \
  --change 'ENV PREDICTION_PATH=/submission/submission.csv' \
  --change 'ENV PREDICTION_PATH_LOCAL=/tmp/submission.pkl' \
  --change 'ENV GT_PATH=/tmp/ground_truth.pkl' \
  --change 'ENV EVALUATION_LOC=remote' \
  --change 'ENV MODEL_PATH=data/decoder.pkl' \
  --change 'ENV SPLIT=h1' \
  --change "ENV PHASE=$PHASE" \
  --change 'ENV BATCH_SIZE=4' \
  --change 'ENV EVAL_DATA_PATH=/dataset/evaluation_data' \
  --change 'ENV KMP_DUPLICATE_LIB_OK=' \
  --change 'CMD ["/bin/bash", "-c", "python decode.py --evaluation $EVALUATION_LOC --model-path $MODEL_PATH --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]' \
  "$CNAME" "$IMAGE" >/dev/null
check_space "dopo il commit"

# 6. Controllo finale della configurazione.
echo "[inspect] $IMAGE"
docker image inspect "$IMAGE" --format 'arch={{.Architecture}} | entrypoint={{json .Config.Entrypoint}} | cmd={{json .Config.Cmd}}'
docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -v -E '^(NV_|NVIDIA_|NCCL_|CUDA_|PATH=|LD_LIBRARY_PATH=|LIBRARY_PATH=|PYTORCH_VERSION=|NVARCH=)'
if docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i -E 'b1|mlp'; then
  echo "ATTENZIONE: resta un ENV di B1" >&2; exit 1
fi
echo "immagine pronta: $IMAGE"
# Prova locale (senza --gpus all, che sul Mac non c'è):
#   docker run --rm --platform linux/amd64 -e EVALUATION_LOC=local -v "$DATA":/dataset/evaluation_data/000954:ro "$IMAGE"
# Push: evalai push "$IMAGE" --phase few-shot-minival-2319 --private
