#!/bin/bash

DATASET_DIR='/mnt/dataset/wmt_ende/'
RESULTS_DIR='gnmt_wmt16'
PROFILE_DIR='./profile'

SEED=1
TARGET=21.80

if [ -d ${PROFILE_DIR} ]; then
    rm -r ${PROFILE_DIR}
fi
mkdir ${PROFILE_DIR}

rm *.nvvp
rm *.log
rm *.json

PREFIX="python -u "
SUFFIX=
single_gpu=1
nvprof_on=0
multi=0
dist=0

for var in "$@"
do
    if [ $var = "cupti" ]; then
        SUFFIX="${SUFFIX} --cupti"
    fi
    if [ $var = "multi" ]; then # run multi-GPU profile
        single_gpu=0
        multi=1
        PREFIX="${PREFIX}-m multiproc "
    fi
    if [ $var = "nvprof" ]; then # use nvprof, cannot coexist with cupti
        nvprof_on=1
    fi
    if [ $var = "profile" ]; then # turn on NeuralTap profile
        SUFFIX="${SUFFIX} --profile"
    fi
    if [ $var = "dist" ]; then
        single_gpu=0
        dist=1
        #export NCCL_SOCKET_IFNAME=enp3s0f0
        #export MASTER_ADDR=eco-11
        #export MASTER_PORT=23451
        PREFIX="${PREFIX}-m dist_multiproc "
        SUFFIX="${SUFFIX} --dist-url env://"
    fi
done

if [ $nvprof_on = 1 ] && [ $single_gpu = 0 ]; then
    if [ $multi = 1 ]; then
        PREFIX="nvprof --export-profile seq2seq_multi_%p.nvvp -f --profile-child-processes ${PREFIX}"
    fi
    if [ $dist = 1 ]; then
        PREFIX="nvprof --export-profile seq2seq_dist_%p.nvvp -f --profile-child-processes ${PREFIX}"
    fi
elif [ $nvprof_on = 1 ]; then
    PREFIX="nvprof --export-profile seq2seq.nvvp -f ${PREFIX}"
fi

echo "running: $PREFIX train.py --save ${RESULTS_DIR} --dataset-dir ${DATASET_DIR} --seed $SEED --target-bleu $TARGET --epochs 8 --batch-size 64 $SUFFIX"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# run training
if [ $single_gpu = 1 ]; then
    echo "using single GPU"
    export CUDA_VISIBLE_DEVICES=0
fi

$PREFIX train.py \
  --save ${RESULTS_DIR} \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --epochs 8 \
  --batch-size 64 $SUFFIX \
  --num-minibatches 20

python ../../merge_log_to_json.py
