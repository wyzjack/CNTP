# mkdir -p data/downloads
mkdir -p data/eval

# # MMLU dataset
# wget -O data/downloads/mmlu_data.tar https://people.eecs.berkeley.edu/~hendrycks/data.tar
# mkdir -p data/downloads/mmlu_data
# tar -xvf data/downloads/mmlu_data.tar -C data/downloads/mmlu_data
# mv data/downloads/mmlu_data/data data/eval/mmlu && rm -r data/downloads/mmlu_data data/downloads/mmlu_data.tar


# # Big-Bench-Hard dataset
# wget -O data/downloads/bbh_data.zip https://github.com/suzgunmirac/BIG-Bench-Hard/archive/refs/heads/main.zip
# mkdir -p data/downloads/bbh
# unzip data/downloads/bbh_data.zip -d data/downloads/bbh
# mv data/downloads/bbh/BIG-Bench-Hard-main/ data/eval/bbh && rm -r data/downloads/bbh data/downloads/bbh_data.zip


# # TyDiQA-GoldP dataset
# mkdir -p data/eval/tydiqa
# wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-dev.json
# wget -P data/eval/tydiqa/ https://storage.googleapis.com/tydiqa/v1.1/tydiqa-goldp-v1.1-train.json


# # GSM dataset
# wget -P data/eval/gsm/ https://github.com/openai/grade-school-math/raw/master/grade_school_math/data/test.jsonl

# MATH dataset
mkdir -p data/eval/MATH
wget -P data/eval/MATH/ https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Math/main/evaluation/datasets/math/test.jsonl

# # Codex HumanEval
# wget -P data/eval/codex_humaneval https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
# wget -P data/eval/codex_humaneval https://huggingface.co/datasets/bigcode/humanevalpack/raw/main/data/python/data/humanevalpack.jsonl

# HumanEval+
wget -P data/eval/codex_humaneval https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.9/HumanEvalPlus-OriginFmt.jsonl.gz

# Alpaca Farm reference
wget -P data/eval/alpaca_farm https://huggingface.co/datasets/hamishivi/alpaca-farm-davinci-003-2048-token/resolve/main/davinci_003_outputs.json


# # TruthfulQA
# wget -P data/eval/truthfulqa https://github.com/sylinrl/TruthfulQA/raw/main/TruthfulQA.csv


# # Toxigen data
# mkdir -p data/eval/toxigen
# for minority_group in asian black chinese jewish latino lgbtq mental_disability mexican middle_east muslim native_american physical_disability trans women
# do
#     wget -O data/eval/toxigen/hate_${minority_group}.txt https://raw.githubusercontent.com/microsoft/TOXIGEN/main/prompts/hate_${minority_group}_1k.txt
# done


# IFEVAL data
wget -P data/eval/ifeval https://github.com/google-research/google-research/raw/master/instruction_following_eval/data/input_data.jsonl


# # XSTest data
# wget -P data/eval/xstest https://github.com/paul-rottger/exaggerated-safety/raw/main/xstest_v2_prompts.csv


# # we use self-instruct test set, and vicuna test set for our human evaluation
# mkdir -p data/eval/creative_tasks 
# wget -O data/eval/creative_tasks/self_instruct_test.jsonl https://github.com/yizhongw/self-instruct/raw/main/human_eval/user_oriented_instructions.jsonl
# wget -O data/eval/creative_tasks/vicuna_test.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/vicuna_bench/question.jsonl