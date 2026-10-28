#!/bin/bash
set -e
cd "$(dirname "$(realpath "$0")")"
VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.30.0/vllm-0.30.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
HARNESS_COMMIT=8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd
export VIRTUAL_ENV=$PWD/.venv
[ -d .venv ] || uv venv --python 3.12 .venv
if [ "$1" = --score-only ]; then
    uv pip install torch transformers --torch-backend=cpu
else
    uv pip install "vllm @ $VLLM_WHEEL" --torch-backend=cu129
fi
[ -d harness ] || git clone https://github.com/bigcode-project/bigcode-evaluation-harness harness
git -C harness checkout -q $HARNESS_COMMIT
uv pip install --no-deps -e harness
uv pip install datasets==5.1.0 evaluate==0.4.6 numpy==2.3.5 pytest
