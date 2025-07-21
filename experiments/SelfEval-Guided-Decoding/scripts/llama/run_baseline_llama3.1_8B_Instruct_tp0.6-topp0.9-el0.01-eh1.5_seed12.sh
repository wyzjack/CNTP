#!/bin/bash

set -x


split=test
dtname=gsm8k

EXEHOME=/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/src
DATAHOME=/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/data
OUTPUTHOME=/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/outputs/${dtname}/${split}_outputs

mkdir -p ${OUTPUTHOME}

cd ${EXEHOME}

CUDA_VISIBLE_DEVICES=4 python generate_code_baseline_llama3.1_cautious.py --verbal \
    --dt_name ${dtname} \
    --input_file ${DATAHOME}/${dtname}_${split}.jsonl \
    --output_dir ${OUTPUTHOME} \
    --max_tokens 512 \
    --temperature 0.6 \
    --top_p 0.9 \
    --mini_n_samples 1 --n_samples 1 \
    --batch_size 1 \
    --model_name 'meta-llama/Meta-Llama-3.1-8B-Instruct' \
    --entropy_threshold_low 0.01 \
    --entropy_threshold_high 1.5 \
    --max_trials 13 \
    --seed 12 \
    # --resume \
    # --resume_dt_string '01_19_23_06'