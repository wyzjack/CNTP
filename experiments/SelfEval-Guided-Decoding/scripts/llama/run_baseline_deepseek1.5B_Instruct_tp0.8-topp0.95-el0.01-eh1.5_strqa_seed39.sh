#!/bin/bash

set -x


split=test
dtname=strategyqa

EXEHOME=/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/src
DATAHOME=/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/data
OUTPUTHOME=/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/outputs/${dtname}/${split}_outputs

mkdir -p ${OUTPUTHOME}

cd ${EXEHOME}

CUDA_VISIBLE_DEVICES=0 python generate_code_baseline_deepseek_seed_strqa.py --verbal \
    --dt_name ${dtname} \
    --input_file ${DATAHOME}/${dtname}_${split}.jsonl \
    --output_dir ${OUTPUTHOME} \
    --max_tokens 512 \
    --temperature 0.8 \
    --top_p 0.95 \
    --mini_n_samples 1 --n_samples 1 \
    --batch_size 1 \
    --model_name 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B' \
    --entropy_threshold_low 0.01 \
    --entropy_threshold_high 1.5 \
    --max_trials 10 \
    --seed 39