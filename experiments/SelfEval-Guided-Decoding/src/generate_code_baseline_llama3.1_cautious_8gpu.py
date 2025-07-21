import os
import argparse
from datetime import datetime
from collections import Counter
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, set_seed
import jsonlines
from tqdm import tqdm

from utils.tool import load_llama_model_and_tokenizer, safe_execute, floatify_ans, finqa_equal
from utils.prompt import get_prompts, get_prompt_inputs
from utils.dataset import jsonlines_load


def parse_args():
    parser = argparse.ArgumentParser()
    ##=== prompting hyperparameters ===##
    parser.add_argument("--model_name", default='meta-llama/Meta-Llama-3.1-8B-Instruct', type=str)
    parser.add_argument("--auth_token", default="YOUR_HF_TOKEN", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=600, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument("--use_mini_n", default=False, action='store_true')
    parser.add_argument("--mini_n_samples", default=4, type=int, help='value of n for mini code generation sampling')
    parser.add_argument("--sleep_time", default=3, type=int)
    parser.add_argument("--max_stuck_time", default=8, type=int)
    ##=== prompt settings ===##
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--greedy", default=False, action='store_true')
    parser.add_argument("--chatgpt", default=False, action='store_true')
    ##=== input data ===##
    parser.add_argument("--dt_name", required=True, type=str, 
                        choices=[
                            'gsm8k', 'aqua', 'svamp', 'asdiv', 'mawps', 'tabmwp', 'finqa',
                            'object_counting', 'repeat_copy', 'colored_object', 'penguin',
                            'date_understanding', 'sports', 'csqa', 'saycan', 'strategyqa',
                            'gsm8k_cot',
                        ],
                        help='the dataset to test')
    parser.add_argument("--input_file", required=True, type=str)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    ##=== multi-GPU arguments (NEW) ===##
    parser.add_argument("--multi_gpu", action="store_true", help="Whether to split the dataset across multiple GPUs.")  # <-- NEW
    parser.add_argument("--gpu_index", type=int, default=0, help="Index of the GPU.")  # <-- NEW
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs.")  # <-- NEW

    ##=== others ===##
    parser.add_argument("--reverse", default=False, action='store_true')
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--verbal", default=False, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_dt_string", default="", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--entropy_threshold_low", default=0.01, type=float)
    parser.add_argument("--entropy_threshold_high", default=1.5, type=float)
    args = parser.parse_args()
    
    args.prompts = get_prompts(args.dt_name, return_eval=False, use_chatgpt=args.chatgpt)
    
    return args


@torch.no_grad()
def get_generations(model, tokenizer, input_ids, attention_mask, generation_config):
    sequences = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        generation_config=generation_config,
        output_scores=True, 
        return_dict_in_generate=True,
        tokenizer=tokenizer
    )
    
    sequences, scores = sequences.sequences.cpu(), sequences.scores
    scores = torch.stack(scores, dim=0).transpose(0, 1).cpu()
    sequences = (
        sequences.contiguous()
        .view(
            input_ids.size(0),
            generation_config.num_return_sequences,
            -1,
        )
        .transpose(0, 1)
    )
    
    if sequences.size(1) == 1:
        texts = tokenizer.batch_decode(sequences[:, 0, input_ids.size(-1):], skip_special_tokens=True)
        sequence_ids = sequences[:, 0, :]
    elif sequences.size(0) == 1:
        texts = tokenizer.batch_decode(sequences[0, :, input_ids.size(-1):], skip_special_tokens=True)
        sequence_ids = sequences[0, :, :]
    
    result = []
    for text, ids, _scores in zip(texts, sequence_ids, scores):
        gen_ids = ids[input_ids.size(-1):]
        tokens = [x.replace('▁', ' ').replace('<0x0A>', '\n') for x in tokenizer.convert_ids_to_tokens(gen_ids)]
        g = {
            'text': text, 
            'logprobs': {
                'tokens': tokens,
                'token_logprobs': gather_log_probabilities(_scores, gen_ids).tolist(),
            }
        }
        if sequences.size(0) == 1:
            g['logprobs']['top_logprobs'] = []
            log_probs = F.log_softmax(_scores.float(), dim=-1)
            for logprobs in log_probs:
                topk_logprobs, topk_ids = logprobs.topk(5)
                topk_tokens = tokenizer.batch_decode(
                    topk_ids.unsqueeze(1), skip_special_tokens=True,
                )
                g['logprobs']['top_logprobs'].append({tok: lp for tok, lp in zip(topk_tokens, topk_logprobs.tolist())})
        result.append(g)
    
    return result


def gather_log_probabilities(scores, gen_ids):
    """
    Utility for extracting the logprobs from the model outputs.
    """
    log_probs = F.log_softmax(scores.float(), dim=-1)
    # shape: [sequence_length, vocab_size]
    indices = gen_ids.unsqueeze(0).transpose(0, 1)  # shape: [sequence_length, 1]
    gathered = torch.gather(log_probs, 1, indices).squeeze(1)  # shape: [sequence_length]
    return gathered


def prompt_the_result(model, tokenizer, prompts, attn_masks, generation_config, n):
    results = []
    for _ in tqdm(range(n // generation_config.num_return_sequences)):
        results.extend(get_generations(model, tokenizer, 
                                       prompts, attn_masks, 
                                       generation_config))
    return {'choices': results}


def parse_api_result(raw_results, llama=True, return_prob=False):
    """
    Modify or keep this function as is to parse your raw generation results.
    """
    # For llama=True, raw_results is { 'choices': [ { 'text': ..., 'logprobs': { ... } }, ... ] }
    # Return a list of strings (the 'text') or code segments
    ret = []
    for c in raw_results['choices']:
        text = c['text'].strip()
        ret.append(text)
    return ret


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    
    ### ==================== Load Input Data ==================== ###
    data_test = jsonlines_load(args.input_file)
    for i, _ in enumerate(data_test):
        data_test[i]['index'] = i

    # If user gave --end -1, interpret as all data
    args.end = len(data_test) if args.end == -1 else args.end + 1
    
    # Take the slice [start, end) of data
    data_test = data_test[args.start:args.end]
    
    # Handle multi-GPU splitting if requested  # <-- NEW
    if args.multi_gpu and args.num_gpus > 1:
        total_len = len(data_test)
        chunk_size = (total_len + args.num_gpus - 1) // args.num_gpus
        subset_start = chunk_size * args.gpu_index
        subset_end = min(total_len, subset_start + chunk_size)
        data_test = data_test[subset_start:subset_end]
        print(f"[GPU {args.gpu_index}] Handling examples from {args.start + subset_start} to {args.start + subset_end} "
              f"(slice size = {len(data_test)})")
    else:
        print('number of examples (single GPU or no splitting):', len(data_test))

    # Prepare output filename
    if args.resume:
        dt_string = args.resume_dt_string
    else:
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%H_%M")

    # Build model type string
    mtype = 'Llama-3.1-Instruct' if 'Instruct' in args.model_name else 'Llama-3.1'
    if '8B' in args.model_name:
        mtype += '-8B'
    elif '70B' in args.model_name:
        mtype += '-70B'
    elif '405B' in args.model_name:
        mtype += '-405B'
    else:
        raise ValueError("The specified model name does not match known variants (8B, 70B, 405B).")

    # Build output filename - we append gpu_index if multi-gpu is used  # <-- MODIFIED
    def build_filename():
        suffix = f"_gpu{args.gpu_index}" if args.multi_gpu else ""
        if args.greedy:
            fname = f"{args.output_dir}/{args.dt_name}_{mtype}{suffix}_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}_entropy_low{args.entropy_threshold_low}_entropy_high{args.entropy_threshold_high}.jsonl"
        else:
            if args.n_samples > 1:
                fname = f"{args.output_dir}/{args.dt_name}_sc_{mtype}{suffix}_tp{args.temperature}_topp{args.top_p}_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}_entropy_low{args.entropy_threshold_low}_entropy_high{args.entropy_threshold_high}_cautious.jsonl"
            else:
                fname = f"{args.output_dir}/{args.dt_name}_vanilla_{mtype}{suffix}_tp{args.temperature}_topp{args.top_p}_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}_entropy_low{args.entropy_threshold_low}_entropy_high{args.entropy_threshold_high}_cautious.jsonl"
        if args.reverse:
            fname = fname.replace('.jsonl', '') + '_reverse.jsonl'
        return fname

    filename = build_filename()
    
    # If the file already exists, track which indexes are done
    if os.path.exists(filename):
        prev = jsonlines_load(filename)
        indexes = [x['index'] for x in prev if 'index' in x]
    else:
        indexes = []
        # Save the prompt into the file header
        with jsonlines.open(filename, mode='w') as writer:
            writer.write(args.prompts)
    
    # Filter out examples that are already processed
    inputs = []
    for example in data_test:
        if example['index'] in indexes:
            continue
        inputs.append(example)

    # Reverse if needed
    if args.reverse:
        inputs = inputs[::-1]

    # Load model & tokenizer
    model, tokenizer = load_llama_model_and_tokenizer(args.model_name, args.auth_token)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        num_return_sequences=args.mini_n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stop_strings = '\n\n\n',
        entropy_threshold_low=args.entropy_threshold_low,
        entropy_threshold_high=args.entropy_threshold_high,
    )
    
    correct, wrong = 0, 0
    
    # Iterate in batches
    for batch_idx in tqdm(range((len(inputs) + args.batch_size - 1) // args.batch_size)):
        batch = inputs[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
        
        contexts = []
        for exp in batch:
            full_prompt, _ = get_prompt_inputs(args.dt_name, args.prompts, exp, use_chatgpt=args.chatgpt)
            contexts.append(full_prompt)
        batch_inputs = tokenizer(contexts, return_tensors="pt", padding=True)
        prompts, attn_masks = batch_inputs.input_ids, batch_inputs.attention_mask
        
        if args.verbal:
            for exp in batch:
                print('======================')
                print(f'Index: {exp["index"]}\nQuestion: {exp["question"]}')
        
        raw_results = prompt_the_result(model, tokenizer, prompts, attn_masks, generation_config, args.n_samples)
        results = parse_api_result(raw_results, llama=True, return_prob=False)
        
        # If batch_size = 1 and n_samples > 1, we gather multiple generations
        if len(batch) == 1 and args.n_samples > 1:
            exp = batch[0]
            result_counter = Counter()
            for code in results:
                ans = safe_execute(code)
                ans = floatify_ans(ans)
                if ans is not None:
                    result_counter.update([ans])
            prediction = None
            if len(result_counter) > 0:
                prediction = result_counter.most_common(1)[0][0]
            gt_ans = exp.get('answer', None)
            if finqa_equal(prediction, gt_ans, False):
                correct += 1
            else:
                wrong += 1
            exp.update({
                'executed': prediction, 
                'generated': results,  # multiple solutions
            })
            with jsonlines.open(filename, mode='a') as writer:
                writer.write(exp)
        else:
            # 1 generation per item in the batch
            for rst, exp in zip(results, batch):
                code = rst
                ans = safe_execute(code)
                prediction = floatify_ans(ans)
                gt_ans = exp.get('answer', None)
                if finqa_equal(prediction, gt_ans, False):
                    correct += 1
                else:
                    wrong += 1
                exp.update({
                    'executed': prediction, 
                    'generated': [rst],
                })
                with jsonlines.open(filename, mode='a') as writer:
                    writer.write(exp)

        torch.cuda.empty_cache()

    accuracy = correct / max(1, (correct + wrong))
        
    print('======================')
    print(f"Accuracy: {accuracy:.4f} ({correct}/{correct + wrong})")

    # Save accuracy to a summary file
    summary_filename = filename.replace('.jsonl', '_summary.txt')
    with open(summary_filename, 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f} ({correct}/{correct + wrong})\n')
