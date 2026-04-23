#!/usr/bin/env bash
# Train/eval launcher: CTC-heavy robust hybrid Mamba+Attention model

set -e
set -u
set -o pipefail

export CUDA_HOME=${CUDA_HOME:-/esat/audioslave/r0883470/miniconda3/envs/cuda128}
export CUDA_PATH=$CUDA_HOME
export CONDA_PREFIX=$CUDA_HOME

export PATH=$CUDA_HOME/bin:/esat/audioslave/r0883470/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}
export CUDACXX=$CUDA_HOME/bin/nvcc

export CFLAGS="-I$CONDA_PREFIX/include ${CFLAGS:-}"
export CXXFLAGS="-I$CUDA_HOME/include ${CXXFLAGS:-}"
export LDFLAGS="-L$CONDA_PREFIX/lib ${LDFLAGS:-}"

export NUMBA_DISABLE_JIT=1
export NUMBA_NUM_THREADS=1
export NUMBA_CPU_FEATURES=""
export NUMBA_CPU_NAME=generic
export NUMBA_THREADING_LAYER=workqueue

export PYTHONFAULTHANDLER=1
export ESPNET_SKIP_CONDA_ACTIVATION=1

VALIDATOR_TIMEOUT_SEC=${VALIDATOR_TIMEOUT_SEC:-40}
set +e
timeout "${VALIDATOR_TIMEOUT_SEC}"s python3 - <<'PY'
import ssl
import numpy
import torch

print("validator: ssl", ssl.OPENSSL_VERSION)
print("validator: numpy", numpy.__version__)
print("validator: torch", torch.__version__)
print("validator: cuda", torch.version.cuda)

if not torch.cuda.is_available():
    raise RuntimeError("validator: torch.cuda.is_available() is False")

import espnet2.utils.types  # noqa: F401
import espnet2.bin.launch  # noqa: F401
import espnet2.bin.asr_train  # noqa: F401

print("validator: espnet imports ok")
PY
validator_rc=$?
set -e
if [ "${validator_rc}" -ne 0 ]; then
    echo "validator: failed with exit code ${validator_rc}; requesting reschedule"
    echo "validator: incompatible worker for cuda128 runtime; requesting reschedule"
    exit 86
fi

PARALLEL_NJ=${NCPU:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
if [ "${PARALLEL_NJ}" -lt 1 ]; then
    PARALLEL_NJ=1
fi
if [ "${PARALLEL_NJ}" -gt 16 ]; then
    PARALLEL_NJ=16
fi
INFER_NJ=1

echo "parallel config: nj=${PARALLEL_NJ}, inference_nj=${INFER_NJ}"

lang=nl
train_set="train_cased"
valid_set="val_cased"
test_sets="val_cased"
nbpe=500

exp=exp/hybrid_mamba_attention_ctc_robust
asr_config=conf/streaming_hybrid_mamba_attention_ctc_robust.yaml
inference_config=conf/inference_hybrid_mamba_attention_ctc_robust.yaml
inference_args="--ctc_weight 0.8"
inference_asr_model="valid.loss.best.pth"

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN=1 set; skipping asr.sh run"
    exit 0
fi

./asr.sh \
    --ngpu ${GPU_COUNT:-1} \
    --nbpe ${nbpe} \
    --stage ${STAGE:-11} \
    --stop_stage ${STOP_STAGE:-12} \
    --nj "${PARALLEL_NJ}" \
    --inference_nj "${INFER_NJ}" \
    --gpu_inference true \
    --asr_config "${asr_config}" \
    --use_lm false \
    --lang ${lang} \
    --use_ngram false \
    --token_type bpe \
    --feats_type raw \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --expdir "${exp}" \
    "$@"

