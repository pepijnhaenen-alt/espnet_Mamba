#!/usr/bin/env bash
# Train/eval launcher: non-transformer (RNN-decoder) hybrid Mamba+Attention model

set -e
set -u
set -o pipefail

export CUDA_HOME=${CUDA_HOME:-/esat/audioslave/r0883470/miniconda3/envs/cuda128_mamba}
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

# Enable Python faulthandler for debugging
export PYTHONFAULTHANDLER=1
export ESPNET_SKIP_CONDA_ACTIVATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================================
# ENVIRONMENT VALIDATION
# ============================================================================

# Validate binary compatibility of worker with cuda128 environment
if ! python3 - <<'PY'
import ssl
import numpy
import soundfile
import scipy
import typeguard
import torch
from triton.runtime import driver
from importlib.metadata import version

print("validator: ssl", ssl.OPENSSL_VERSION)
print("validator: numpy", numpy.__version__)
print("validator: soundfile", soundfile.__version__)
print("validator: scipy", scipy.__version__)
print("validator: typeguard", version("typeguard"))

if not torch.cuda.is_available():
    raise RuntimeError("validator: torch.cuda.is_available() is False")

# Ensure Triton can see active CUDA driver for hybrid attention kernels
driver.active.get_current_target()
print("validator: cuda", torch.version.cuda)
print("validator: triton driver ok")
PY
then
    echo "validator: incompatible worker for cuda128 runtime; requesting reschedule"
    exit 86
fi

# ============================================================================
# PARALLELISM CONFIGURATION
# ============================================================================

PARALLEL_NJ=${NCPU:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
if [ "${PARALLEL_NJ}" -lt 1 ]; then
    PARALLEL_NJ=1
fi
if [ "${PARALLEL_NJ}" -gt 16 ]; then
    PARALLEL_NJ=16
fi
INFER_NJ=1
echo "parallel config: nj=${PARALLEL_NJ}, inference_nj=${INFER_NJ}"

echo "parallel config: nj=${PARALLEL_NJ}, inference_nj=${INFER_NJ}"

lang=nl
train_set="train_cased_cleaned"
valid_set="val_cased_cleaned"
test_sets="test_cased_cleaned_small"
nbpe=5000

exp=exp/hybrid_mamba_attention_nontransformer
asr_config=conf/streaming_hybrid_mamba_attention_nontransformer.yaml
inference_config=conf/inference_hybrid_mamba_attention_nontransformer.yaml
inference_args="--ctc_weight 0.55 --sim_chunk_length 2048"
inference_asr_model="valid.loss.best.pth"

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN=1 set; skipping asr.sh run"
    exit 0
fi

./asr.sh \
    --ngpu ${GPU_COUNT:-1} \
    --nbpe ${nbpe} \
    --stage ${STAGE:-12} \
    --stop_stage ${STOP_STAGE:-13} \
    --nj "${PARALLEL_NJ}" \
    --inference_nj "${INFER_NJ}" \
    --gpu_inference true \
    --asr_config "${asr_config}" \
    --use_lm false \
    --lang ${lang} \
    --use_ngram false \
    --token_type bpe \
    --feats_type raw \
    --compute_streaming_metrics true \
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --expdir "${exp}" \
    "$@"

