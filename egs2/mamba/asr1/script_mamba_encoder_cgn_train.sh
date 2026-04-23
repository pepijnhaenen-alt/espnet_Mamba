#scl enable devtoolset-7 bash &&
#export PATH=/users/spraak/spch/prog/spch/cuda-11.0/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/users/spraak/spch/prog/spch/cuda-11.0/lib64:/users/spraak/spch/prog/spch/cudnn-7.6/lib64:/.singularity.d/libs:${LD_LIBRARY_PATH}
#export CFLAGS="-I/users/spraak/spch/prog/spch/cuda-11.0/include $CFLAGS" 
#export CUDA_HOME=/users/spraak/spch/prog/spch/cuda-11.0 
#export CUDA_PATH=/users/spraak/spch/prog/spch/cuda-11.0 

export CUDA_HOME=/esat/audioslave/r0883470/miniconda3/envs/cuda128
export CUDA_PATH=$CUDA_HOME
export CONDA_PREFIX=$CUDA_HOME

export PATH=$CUDA_HOME/bin:/esat/audioslave/r0883470/miniconda3/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

export CFLAGS="-I$CONDA_PREFIX/include $CFLAGS"
export CXXFLAGS="-I$CUDA_HOME/include $CXXFLAGS"
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

# Fix for numba/llvmlite segfaults during resampy import
export NUMBA_DISABLE_JIT=1
export NUMBA_NUM_THREADS=1
export NUMBA_CPU_FEATURES=""
export NUMBA_CPU_NAME=generic
export NUMBA_THREADING_LAYER=workqueue

# Enable Python faulthandler for segfault traces
export PYTHONFAULTHANDLER=1

export ESPNET_SKIP_CONDA_ACTIVATION=1
# source /users/students/r0883470/.bashrc
# source /esat/audioslave/r0883470/miniconda3/bin/activate cuda128
#echo $CPPFLAGS 
#echo $LDFLAGS 
#echo $LD_LIBRARY_PATH
#source /esat/audioslave/r0883470/espnet_Mamba/tools/venv/bin/activate

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# Validate binary compatibility of the assigned worker with cuda128 env.
if ! python3 - <<'PY'
import ssl
import asyncio
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
# Ensure Triton can see an active CUDA driver for Mamba2 kernels.
driver.active.get_current_target()
print("validator: cuda", torch.version.cuda)
print("validator: triton driver ok")
PY
then
    echo "validator: incompatible worker for cuda128 runtime; requesting reschedule"
    exit 86
fi

# Parallelism settings: default to NCPU from job environment, then host CPU count.
PARALLEL_NJ=${NCPU:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}
if [ "${PARALLEL_NJ}" -lt 1 ]; then
    PARALLEL_NJ=1
fi
if [ "${PARALLEL_NJ}" -gt 16 ]; then
    PARALLEL_NJ=16
fi
INFER_NJ=8
echo "parallel config: nj=${PARALLEL_NJ}, inference_nj=${INFER_NJ}"

lang=nl

train_set="train_cased_cleaned"
valid_set="val_cased_cleaned"
#test_sets="test_cased" #in quotation marks because multiple can be given seperated by a space
test_sets="test_cased_cleaned"

#old_lang=rian
nbpe=5000
pretrain=false
pt_tag=
asr_suffix=

exp=exp/cgn_mamba_encoder
# exp=exp/common_voice

asr_config=conf/streaming_mamba.yaml

inference_config=conf/inference_mamba_ctc_streaming.yaml

# CTC-only model: keep CTC enabled during decoding
inference_args="--ctc_weight 1.0"
inference_asr_model="valid.loss.ave.pth"

# ./asr.sh \
#         --ngpu 1 \
#         --nbpe ${nbpe} \
#         --stage 3 \ 
#         --stop_stage 5 \
#     --lang ${lang} \
#         --asr_config "${asr_config}" \
#         --use_lm false \
#         --use_ngram false \
#         --token_type bpe \
#         --inference_config "${inference_config}" \
#         --train_set "${train_set}" \
#         --valid_set "${valid_set}" \
#         --test_sets "${test_sets}" \
#         --bpe_train_text "data/${train_set}/text" "$@"
if [ "${DRYRUN:-0}" = "1" ]; then
    echo "DRYRUN=1 set; skipping asr.sh run"
    exit 0
fi

./asr.sh \
           --ngpu 1 \
           --nbpe ${nbpe} \
           --stage 10 \
           --stop_stage 13 \
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
           --bpe_train_text "data/${train_set}/text" "$@" \
           --expdir "${exp}"
#        --asr_tag "train_asr_conformer_finetuning_raw_lang_bpe5000_eta05" \

# 1-2: Data preparation. We won't need this (for now); our data is prepared elsewhere. 
# 3-4: Data preprocessing. Before you can train a model on a dataset, you first have to run these stages using a CPU job. But you only need to do it once (provided you don't make changes to your data). 
# 5: Generate token list (i.e. vocabulary). You'll need to do this once (on Librispeech) and then use this vocabulary for all models, using a CPU job (or locally without Condor job).
# 6-9: LM and Ngram training: we won't need this.
# ​10: ASR collects stats regarding the data. For each dataset, you always have to run this stage before you can run stage 11. This is also done using a GPU. 
# 11: ASR model training. Here is where you train the ASR model, using a GPU. 
# 12-13: to evaluate and score the model. This will be done after training, using a separate job (without GPU). 
# ​​14-16: to upload the model to website, we won't need this.
