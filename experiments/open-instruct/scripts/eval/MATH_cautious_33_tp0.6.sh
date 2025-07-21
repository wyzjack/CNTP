# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0



# Evaluating llama2 chat model using chain-of-thought and chat format
# python -m eval.MATH.run_eval \
#     --data_dir data/eval/MATH/ \
#     --max_num_examples 200 \
#     --save_dir results/MATH/llama2-chat-7B-cot-4shot \
#     --model ../hf_llama2_models/7B-chat \
#     --tokenizer ../hf_llama2_models/7B-chat \
#     --n_shot 4 \
#     --use_chat_format \
#     --chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format \
#     --use_vllm

python -m eval.MATH.run_eval_tp0dot6 \
    --data_dir ./scripts/data/eval/MATH/ \
    --max_num_examples 200 \
    --save_dir results/MATH/llama3.1-8B-Instruct_tp0.6_catious_33 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --n_shot 4 \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_llama3_1_chat_format


