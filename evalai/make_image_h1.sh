#!/usr/bin/env bash
# Builds the Docker image for EvalAI (FALCON H1) without `docker build`:
# container from the local base -> docker cp of the files -> docker commit --change.
# Reason: with `docker build` Docker copies the whole base (~9 GB) again into the cache and the Mac disk fills up;
# the commit writes only the new layer.
#
# Usage (30/10, FALCON example sklearn decoder):
#   bash evalai/make_image_h1.sh
# For Thursday 5/11 (our ridge): change DECODER_PKL, DECODE_PY, PHASE=test and IMAGE.
#
# Variables (with their default values):
set -euo pipefail

BASE="${BASE:-falcon_b1_mlp:latest}"                    # local amd64 base: pytorch 2.4.1, CUDA 12.1, falcon_challenge 1.0.2
IMAGE="${IMAGE:-sk_smoke_h1:latest}"                    # image produced
FALCON="${FALCON:-$HOME/tesi_h1_work/falcon-challenge}" # clone of snel-repo/falcon-challenge (outside iCloud)
DATA="${DATA:-$HOME/000954}"                             # H1 dandiset (mounted with -v, does not end up in the image)
DECODER_PKL="${DECODER_PKL:-$FALCON/local_data/sklearn_FalconTask.h1.pkl}"
DECODE_PY="${DECODE_PY:-$FALCON/decoder_demos/sklearn_sample.py}"
PHASE="${PHASE:-minival}"
MIN_FREE_GB="${MIN_FREE_GB:-4}"
CNAME="mkimg_h1_$$"

free_gb() { df -g / | awk 'NR==2 {print $4}'; }
check_space() {
  local f; f=$(free_gb)
  echo "[space] $1: ${f} GB free"
  if [ "$f" -lt "$MIN_FREE_GB" ]; then echo "STOP: less than ${MIN_FREE_GB} GB free" >&2; exit 1; fi
}
cleanup() { docker rm -f "$CNAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

check_space "start"
[ "$(docker image inspect "$BASE" --format '{{.Architecture}}')" = "amd64" ] || { echo "the base is not amd64" >&2; exit 1; }

# 1. Container from the base, without entrypoint, with the H1 data mounted read-only.
docker run -d --name "$CNAME" --platform linux/amd64 --entrypoint "" \
  -v "$DATA":/dataset/evaluation_data/000954:ro \
  "$BASE" sleep infinity >/dev/null
dx() { docker exec -w /workspace "$CNAME" bash -c "$1"; }

# 2. Where falcon_challenge is installed: if it is editable inside /workspace/falcon_b1, that folder stays.
echo "[falcon_challenge]"; dx "pip show -f falcon_challenge | grep -E '^(Version|Location|Editable project location)'"
KEEP_B1=0
if dx "pip show -f falcon_challenge" | grep -E '^(Location|Editable project location):' | grep -q '/workspace/falcon_b1'; then
  KEEP_B1=1; echo "  editable in /workspace/falcon_b1: the folder stays"
fi

# 3. Image files (like sklearn_sample.Dockerfile, with the H1 lines).
dx "mkdir -p data"
docker cp "$DECODER_PKL"                  "$CNAME":/workspace/data/decoder.pkl
docker cp "$FALCON/decoder_demos"         "$CNAME":/workspace/decoder_demos
docker cp "$FALCON/data_demos"            "$CNAME":/workspace/data_demos
docker cp "$DECODE_PY"                    "$CNAME":/workspace/decode.py
docker cp "$FALCON/decoder_demos/filtering.py" "$CNAME":/workspace/filtering.py
[ "$KEEP_B1" = 1 ] || dx "rm -rf /workspace/falcon_b1"

# 4. Does the decoder load with the versions of the base? If not, retrain inside it (expected CV 0.26).
if dx "python -c \"import pickle; pickle.load(open('data/decoder.pkl','rb')); print('pkl ok')\""; then :; else
  echo "[pkl] does not load: retraining in the base"
  dx "mkdir -p local_data && python decoder_demos/sklearn_decoder.py \
      --training_dir /dataset/evaluation_data/000954/sub-HumanPitt-held-in-calib/ \
      --calibration_dir /dataset/evaluation_data/000954/sub-HumanPitt-held-out-calib/ --mode all --task h1 \
      && cp local_data/sklearn_FalconTask.h1.pkl data/decoder.pkl && rm -rf local_data"
fi
dx "python -c 'import numpy, sklearn, falcon_challenge; print(\"numpy\", numpy.__version__, \"sklearn\", sklearn.__version__)'"
check_space "before the commit"

# 5. Commit with explicit configuration: nothing from the B1 base.
#    ENV cannot be deleted: the B1 variables are overwritten; KMP_DUPLICATE_LIB_OK (from B1) emptied.
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
check_space "after the commit"

# 6. Final check of the configuration.
echo "[inspect] $IMAGE"
docker image inspect "$IMAGE" --format 'arch={{.Architecture}} | entrypoint={{json .Config.Entrypoint}} | cmd={{json .Config.Cmd}}'
docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -v -E '^(NV_|NVIDIA_|NCCL_|CUDA_|PATH=|LD_LIBRARY_PATH=|LIBRARY_PATH=|PYTORCH_VERSION=|NVARCH=)'
if docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -i -E 'b1|mlp'; then
  echo "WARNING: a B1 ENV remains" >&2; exit 1
fi
echo "image ready: $IMAGE"
# Local test (without --gpus all, which is not available on the Mac):

#   docker run --rm --platform linux/amd64 -e EVALUATION_LOC=local -v "$DATA":/dataset/evaluation_data/000954:ro "$IMAGE"
# Push: evalai push "$IMAGE" --phase few-shot-minival-2319 --private
