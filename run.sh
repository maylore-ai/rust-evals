#!/bin/bash
set -e
ROOT=$(dirname "$(realpath "$0")")
SUITES=humaneval,mbpp GPUS=0 OUT=results MEMORY=0.8 EXTRA=()
usage() {
    echo "usage: bash run.sh [-s humaneval,mbpp] [-g 0[,1]] [-o results] [-m 0.8] [-a '<vllm args>'] <model>..."
    exit 1
}
while getopts s:g:o:m:a: opt; do
    case $opt in
        s) SUITES=$OPTARG ;;
        g) GPUS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        m) MEMORY=$OPTARG ;;
        a) read -ra MORE <<< "$OPTARG"; EXTRA+=("${MORE[@]}") ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
[ $# -gt 0 ] || usage
[[ $GPUS =~ ^[0-9]+(,[0-9]+)*$ ]] || usage
for SUITE in ${SUITES//,/ }; do
    [[ $SUITE =~ ^(humaneval|mbpp)$ ]] || usage
done
for TOOL in rustc curl; do
    command -v $TOOL > /dev/null || { echo "$TOOL is not on PATH"; exit 1; }
done
systemd-run --user --scope -q true 2> /dev/null || { echo "systemd-run --user is required to cap memory of generated programs"; exit 1; }
[ -x "$ROOT/.venv/bin/python" ] || { echo "no $ROOT/.venv, run setup.sh first"; exit 1; }

source_of() {
    if [ -d "$1" ]; then realpath "$1"; else echo "$1"; fi
}
SHARED=$(for MODEL in "$@"; do basename "$(source_of "$MODEL")"; done | sort | uniq -d | paste -sd ' ')
[ -z "$SHARED" ] || { echo "models share a result name: $SHARED"; exit 1; }
OUT=$(realpath -m "$OUT")
PORT=$((8000 + 10#${GPUS%%,*}))
API=http://localhost:$PORT/v1
TP=$(tr , '\n' <<< "$GPUS" | wc -l)
export PATH=$ROOT/.venv/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$OUT/logs"

VLLM= SCRATCH=
cleanup() {
    [ -z "$VLLM" ] || { kill -- -$VLLM 2> /dev/null; wait $VLLM 2> /dev/null || true; }
    VLLM=
    [ -z "$SCRATCH" ] || rm -rf "$SCRATCH"
    SCRATCH=
}
trap cleanup EXIT

serve() {
    ! curl -sf $API/models > /dev/null || { echo "port $PORT is already taken"; exit 1; }
    echo "== $2: starting vLLM on GPU $GPUS, log $OUT/logs/vllm-$2.log"
    setsid vllm serve "$1" --served-model-name "$2" --dtype bfloat16 --max-model-len 8192 --language-model-only \
        --gpu-memory-utilization $MEMORY --tensor-parallel-size $TP --port $PORT "${EXTRA[@]}" > "$OUT/logs/vllm-$2.log" 2>&1 < /dev/null &
    VLLM=$!
    until curl -sf $API/models > /dev/null; do
        kill -0 $VLLM 2> /dev/null || { tail -30 "$OUT/logs/vllm-$2.log"; exit 1; }
        sleep 5
    done
}

claim() {
    mkdir -p "$1"
    [ -f "$1/model" ] || echo "$2" > "$1/model"
    [ "$(cat "$1/model")" = "$2" ] || { echo "$1 holds results of $(cat "$1/model"), not $2"; exit 1; }
}

for MODEL in "$@"; do
    MODEL=$(source_of "$MODEL")
    NAME=$(basename "$MODEL")
    for SUITE in ${SUITES//,/ }; do
        claim "$OUT/$SUITE/$NAME" "$MODEL"
    done
    for SUITE in ${SUITES//,/ }; do
        DIR=$OUT/$SUITE/$NAME
        [ ! -f "$DIR/generations.json" ] || continue
        [ -n "$VLLM" ] || serve "$MODEL" "$NAME"
        echo "== $NAME: generating $SUITE"
        python "$ROOT/generate.py" --suite $SUITE --model "$NAME" --api $API --out "$DIR" \
            > "$OUT/logs/generate-$SUITE-$NAME.log" 2>&1 || { tail -20 "$OUT/logs/generate-$SUITE-$NAME.log"; exit 1; }
    done
    cleanup
    for SUITE in ${SUITES//,/ }; do
        DIR=$OUT/$SUITE/$NAME
        [ ! -f "$DIR/scores.json" ] || continue
        echo "== $NAME: scoring $SUITE"
        SCRATCH=$(mktemp -d)
        TMPDIR=$SCRATCH systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 -p OOMPolicy=continue -q -- \
            python "$ROOT/score.py" --suite $SUITE --out "$DIR" > "$OUT/logs/score-$SUITE-$NAME.log" 2>&1 \
            || { tail -20 "$OUT/logs/score-$SUITE-$NAME.log"; exit 1; }
        cleanup
    done
done
python "$ROOT/summary.py" "$OUT"
