#!/usr/bin/env bash
# Train/eval launcher: non-transformer (RNN-decoder) hybrid Mamba+Attention model

set -e
set -u
set -o pipefail

# lang=nl
# train_set="train_cased_cleaned"
# valid_set="val_cased_cleaned"
# test_sets="test_cased_cleaned_small"
# exp=exp/baseline/asr_conformer_steven/cgn
lang=en
train_set="train_lib360_copy"
valid_set="dev_lib360"
test_sets="test_lib360_small"
exp=exp/transformer_encoder_ctc/libri
nbpe=5000

export CUDA_HOME=${CUDA_HOME:-/esat/audioslave/r0883470/miniconda3/envs/cuda128_mamba}
export CUDA_PATH=$CUDA_HOME
export CONDA_PREFIX=$CUDA_HOME

export PATH=$CUDA_HOME/bin:/esat/audioslave/r0883470/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH:-}
export CUDACXX=$CUDA_HOME/bin/nvcc

export CFLAGS="-I$CONDA_PREFIX/include ${CFLAGS:-}"
export CXXFLAGS="-I$CUDA_HOME/include ${CXXFLAGS:-}"
export LDFLAGS="-L$CONDA_PREFIX/lib ${LDFLAGS:-}"
export LD_LIBRARY_PATH="/esat/audioslave/r0883470/espnet_Mamba/tools/warp-transducer/build:${LD_LIBRARY_PATH}"
export PYTHONPATH="/esat/audioslave/r0883470/espnet_Mamba/tools/warp-transducer/pytorch_binding:${PYTHONPATH:-}"

export NUMBA_DISABLE_JIT=1
export NUMBA_NUM_THREADS=1
export NUMBA_CPU_FEATURES=""
export NUMBA_CPU_NAME=generic
export NUMBA_THREADING_LAYER=workqueue

# Enable Python faulthandler for debugging
export PYTHONFAULTHANDLER=1
export ESPNET_SKIP_CONDA_ACTIVATION=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SHARED_RUN_DIR=$(pwd)
SHARED_ASR_EXP="${SHARED_RUN_DIR}/${exp}"
LOCAL_RECIPE_DIR="${SHARED_RUN_DIR}"
LOCAL_ASR_EXP="${SHARED_ASR_EXP}"

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

asr_config=conf/streaming_transformer.yaml
inference_config=conf/inference_transformer_ctc_streaming.yaml
inference_args="--ctc_weight 1.0"
inference_asr_model="valid.loss.best.pth" #"valid.acc.best.pth" #This should be adjusted

if [ "${RUN_ON_SCRATCH:-0}" != "0" ]; then
    if [ -n "${_CONDOR_SCRATCH_DIR:-}" ] && [ -d "${_CONDOR_SCRATCH_DIR}" ]; then
        LOCAL_RUN_DIR="${_CONDOR_SCRATCH_DIR}/mamba_rnnt_local"
        mkdir -p "${LOCAL_RUN_DIR}"
        LOCAL_RECIPE_DIR="${LOCAL_RUN_DIR}/recipe"
        mkdir -p "${LOCAL_RECIPE_DIR}"
        cat > "${LOCAL_RECIPE_DIR}/path.sh" <<'EOF'
export PATH="$PWD/utils/:$PATH"
export LC_ALL=C
export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"
if [ -f local/path.sh ]; then
    . local/path.sh
fi
EOF
        for item in asr.sh cmd.sh conf utils steps pyscripts scripts data dump; do
            if [ -e "${SHARED_RUN_DIR}/${item}" ] && [ ! -e "${LOCAL_RECIPE_DIR}/${item}" ]; then
                ln -s "${SHARED_RUN_DIR}/${item}" "${LOCAL_RECIPE_DIR}/${item}"
            fi
        done
        mkdir -p "${LOCAL_RECIPE_DIR}/local"
        cat > "${LOCAL_RECIPE_DIR}/local/path.sh" <<'EOF'
:
EOF
        LOCAL_ASR_EXP="${LOCAL_RECIPE_DIR}/${exp}"
        mkdir -p "${LOCAL_ASR_EXP}"
        sync_back_local_exp() {
            if [ -d "${LOCAL_ASR_EXP}" ]; then
                mkdir -p "${SHARED_ASR_EXP}"
                if command -v rsync >/dev/null 2>&1; then
                    rsync -a "${LOCAL_ASR_EXP}/" "${SHARED_ASR_EXP}/"
                else
                    cp -a "${LOCAL_ASR_EXP}/." "${SHARED_ASR_EXP}/"
                fi
            fi
        }
        trap sync_back_local_exp EXIT
        cd "${LOCAL_RECIPE_DIR}"
        export PYTHONPATH="${SHARED_RUN_DIR}/../../..:${PYTHONPATH:-}"
        echo "scratch mode: local recipe dir ${LOCAL_RECIPE_DIR}"
        echo "scratch mode: tensorboard/checkpoints on ${LOCAL_ASR_EXP}"
        echo "scratch mode: results will sync back to ${SHARED_ASR_EXP}"
    else
        echo "scratch mode requested but _CONDOR_SCRATCH_DIR is unavailable; using shared run dir"
    fi
fi

if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN=1 set; skipping asr.sh run"
    exit 0
fi

./asr.sh \
    --ngpu ${GPU_COUNT:-1} \
    --nbpe ${nbpe} \
    --stage ${STAGE:-11} \
    --stop_stage ${STOP_STAGE:-16} \
    --audio-format flac \
    --nj "${PARALLEL_NJ}" \
    --inference_nj "${INFER_NJ}" \
    --gpu_inference true \
    --asr_exp "${LOCAL_ASR_EXP}" \
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

