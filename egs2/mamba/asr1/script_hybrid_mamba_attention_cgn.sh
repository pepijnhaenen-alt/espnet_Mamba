#!/usr/bin/env bash
# Training script for Streaming Hybrid Mamba+Attention ASR Model
# Based on: "Advancing Streaming ASR with Chunk-wise Attention and Trans-chunk 
# Selective State Spaces"
#
# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Configure CUDA and conda environment
export CUDA_HOME=${CUDA_HOME:-/esat/audioslave/r0883470/miniconda3/envs/cuda128}
export CUDA_PATH=$CUDA_HOME
export CONDA_PREFIX=$CUDA_HOME

export PATH=$CUDA_HOME/bin:/esat/audioslave/r0883470/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}
export CUDACXX=$CUDA_HOME/bin/nvcc

export CFLAGS="-I$CONDA_PREFIX/include ${CFLAGS:-}"
export CXXFLAGS="-I$CUDA_HOME/include ${CXXFLAGS:-}"
export LDFLAGS="-L$CONDA_PREFIX/lib ${LDFLAGS:-}"

# Fix for numba/llvmlite segfaults during resampy import
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
# Keep a few CPU workers for post-training decoding/scoring without oversubscribing.
if [ "${PARALLEL_NJ}" -gt 4 ]; then
    INFER_NJ=4
else
    INFER_NJ=${PARALLEL_NJ}
fi
echo "parallel config: nj=${PARALLEL_NJ}, inference_nj=${INFER_NJ}"

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Language setting
lang=nl

# Dataset names (e.g., from CGN corpus for Dutch, or use your own)
train_set="train_cased_cleaned"
valid_set="val_cased_cleaned"
test_sets="test_cased_cleaned"

# Tokenization
nbpe=5000
pretrain=false

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

exp=exp/hybrid_mamba_attention_mamba2

# Training configuration for Streaming Hybrid Mamba+Attention model
# Best performing configuration from paper:
# - 16 layers, hidden_size=512 (wide)
# - Within-chunk attention + trans-chunk Mamba
# - Performance: 7.3% WER on Tedlium2, 0.40 RTF
asr_config=conf/streaming_hybrid_mamba_attention.yaml

# Inference configuration for streaming CTC decoding
inference_config=conf/inference_hybrid_mamba_attention_ctc_streaming.yaml

# Inference arguments for CTC-only model
inference_args="--ctc_weight 1.0"
inference_asr_model="valid.loss.ave.pth"

# ============================================================================
# TRAINING PIPELINE EXECUTION
# ============================================================================

# Check for dry-run mode
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN=1 set; skipping asr.sh run"
    exit 0
fi

# Execute training/inference pipeline
# Stages:
#   1-2:  Data preparation
#   3-4:  Data preprocessing (convert to proper format)
#   5:    Generate token list (vocabulary)
#   10:   ASR collects data statistics
#   11:   ASR model training (GPU required)
#   12:   ASR evaluation and scoring
#   13-16: Upload model (optional)

./asr.sh \
    --ngpu ${GPU_COUNT:-1} \
    --nbpe ${nbpe} \
    --stage ${STAGE:-11} \
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
    --inference_config "${inference_config}" \
    --inference_asr_model "${inference_asr_model}" \
    --inference_args "${inference_args}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --bpe_train_text "data/${train_set}/text" \
    --expdir "${exp}" \
    "$@"

echo "Training and evaluation completed!"
echo "Model directory: ${exp}"
echo "Best model: ${exp}/1/checkpoints/${inference_asr_model}"
