#scl enable devtoolset-7 bash &&
#export PATH=/users/spraak/spch/prog/spch/cuda-11.0/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/users/spraak/spch/prog/spch/cuda-11.0/lib64:/users/spraak/spch/prog/spch/cudnn-7.6/lib64:/.singularity.d/libs:${LD_LIBRARY_PATH}
#export CFLAGS="-I/users/spraak/spch/prog/spch/cuda-11.0/include $CFLAGS" 
#export CUDA_HOME=/users/spraak/spch/prog/spch/cuda-11.0 
#export CUDA_PATH=/users/spraak/spch/prog/spch/cuda-11.0 
source /users/students/r0883470/.bashrc

export CUDA_HOME=/esat/audioslave/r0883470/miniconda3/envs/cuda128
export CUDA_PATH=$CUDA_HOME
export CONDA_PREFIX=$CUDA_HOME

. $CONDA_PREFIX/bin/activate

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc

export CFLAGS="-I$CONDA_PREFIX/include $CFLAGS"
export CXXFLAGS="-I$CUDA_HOME/include $CXXFLAGS"
export LDFLAGS="-L$CONDA_PREFIX/lib $LDFLAGS"

# Fix for numba/llvmlite segfaults during resampy import
# export NUMBA_DISABLE_JIT=1
# export NUMBA_NUM_THREADS=1
# export NUMBA_CPU_FEATURES=""
# export NUMBA_CPU_NAME=generic
# export NUMBA_THREADING_LAYER=workqueue

# Enable Python faulthandler for segfault traces
export PYTHONFAULTHANDLER=1

# Avoid conda shell activation in Condor jobs; it loads libmambapy and may segfault

export ESPNET_SKIP_CONDA_ACTIVATION=1
# source /esat/audioslave/r0883470/miniconda3/bin/activate cuda128
# source /users/students/r0883470/.bashrc
# echo $CPPFLAGS
# echo $LDFLAGS
# echo $LD_LIBRARY_PATH
# source /esat/audioslave/r0883470/espnet_Mamba/tools/venv/bin/activate

#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang=nl

#train_set="training_data"
#valid_set="val_data"
#test_sets="test_data" #in quotation marks because multiple can be given seperated by a space

train_set="train_cased"
valid_set="val_cased"
test_sets="test_cased" #in quotation marks because multiple can be given seperated by a space

#old_lang=rian
nbpe=300
pretrain=false
pt_tag=
asr_suffix=

exp=exp/cgn
# exp=exp/common_voice

asr_config=conf/train_asr_s4_2.yaml

inference_config=conf/decode_s4.yaml

# disable CTC at decoding time (model was trained with ctc_weight=0.0)
inference_args="--ctc_weight 0.0"

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
           --ngpu 0 \
           --nbpe ${nbpe} \
           --stage 12 \
           --stop_stage 13 \
           --audio-format wav \
	   --inference_nj 4 \
           --asr_config "${asr_config}" \
           --use_lm false \
	   --lang ${lang} \
           --use_ngram false \
           --token_type bpe \
           --feats_type raw \
           --inference_config "${inference_config}" \
           --inference_args "${inference_args}" \
           --train_set "${train_set}" \
           --valid_set "${valid_set}" \
           --test_sets "${test_sets}" \
           --bpe_train_text "data/${train_set}/text" "$@" 
#        --asr_tag "train_asr_conformer_finetuning_raw_lang_bpe5000_eta05" \

# 1-2: Data preparation. We won't need this (for now); our data is prepared elsewhere. 
# 3-4: Data preprocessing. Before you can train a model on a dataset, you first have to run these stages using a CPU job. But you only need to do it once (provided you don't make changes to your data). 
# 5: Generate token list (i.e. vocabulary). You'll need to do this once (on Librispeech) and then use this vocabulary for all models, using a CPU job (or locally without Condor job).
# 6-9: LM and Ngram training: we won't need this.
# ​10: ASR collects stats regarding the data. For each dataset, you always have to run this stage before you can run stage 11. This is also done using a GPU. 
# 11: ASR model training. Here is where you train the ASR model, using a GPU. 
# 12-13: to evaluate and score the model. This will be done after training, using a separate job (without GPU). 
# ​​14-16: to upload the model to website, we won't need this.
