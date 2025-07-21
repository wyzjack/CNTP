import os
import argparse
from datetime import datetime

import torch
from transformers import GenerationConfig, set_seed

from utils.tool import *
from utils.prompt import *
from utils.dataset import jsonlines_load
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    ##=== prompting hyperparameters ===##
    parser.add_argument("--model_name", default='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', type=str)
    parser.add_argument("--auth_token", default="YOUR_HF_TOKEN", type=str)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--max_tokens", default=600, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--logprobs", default=1, type=int)
    parser.add_argument("--use_mini_n", default=False, action='store_true')
    parser.add_argument("--mini_n_samples", default=4, type=int, help='value of n for mini code generation sampling (when token rate is limited)')
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
    ##=== others ===##
    parser.add_argument("--reverse", default=False, action='store_true')
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--verbal", default=False, action='store_true')
    parser.add_argument("--resume", default=False, action='store_true')
    parser.add_argument("--resume_dt_string", default="", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--entropy_threshold_low", default=0.1, type=float)
    parser.add_argument("--entropy_threshold_high", default=1.0, type=float)
    parser.add_argument("--num_beams", default=5, type=int)
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
        # print(f"the text is: \n {text}")
        # print("the end of the text")
        gen_ids = ids[input_ids.size(-1):]
        tokens = [x.replace('▁', ' ').replace('<0x0A>', '\n') for x in tokenizer.convert_ids_to_tokens(gen_ids)]
        gen_ids = gen_ids[:-1]
        tokens = tokens[:-1]
        g = {
            'text': text, 
            'logprobs': {
                'tokens': tokens,
                'token_logprobs': gather_log_probabilities(_scores, gen_ids).tolist()
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


def prompt_the_result(model, tokenizer, prompts, attn_masks, generation_config, n):
    results = []
    for _ in tqdm(range(n // generation_config.num_return_sequences)):
        results.extend(get_generations(model, tokenizer, 
                                       prompts, attn_masks, 
                                       generation_config))
    return {'choices': results}


if __name__ == "__main__":

    args = parse_args()

    # Set all seeds for reproducibility
    # seed = args.seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    set_seed(args.seed)
    
    ### ==================== Load Input Data ==================== ###
    data_test = jsonlines_load(args.input_file)
    for i, _ in enumerate(data_test):
        data_test[i]['index'] = i

    ### ==================== Prepare Output Filename ==================== ###
    if args.resume:
        dt_string = args.resume_dt_string
    else:
        now = datetime.now()
        dt_string = now.strftime("%m_%d_%H_%M")

    correct, wrong = 0, 0

    args.end = len(data_test) if args.end == -1 else args.end + 1
    data_test = data_test[args.start:args.end]
    print('number of examples: ', len(data_test))

    mtype = 'DeepSeek-R1-Distill-Qwen-1.5B'
    if args.model_name.count('1.5B'):
        mtype += '-1.5B'
    else:
        raise ValueError("The specified model does not exist.")
    
    if args.greedy:
        filename = f'{args.output_dir}/{args.dt_name}_{mtype}_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}_bm{args.num_beams}.jsonl'
        assert not args.temperature, "In greedy decoding, temperature should be 0.0"
    else:
        if args.n_samples > 1:
            filename = f'{args.output_dir}/{args.dt_name}_sc_{mtype}_tp{args.temperature}_topp{args.top_p}_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}_bm{args.num_beams}.jsonl'
        else:
            filename = f'{args.output_dir}/{args.dt_name}_vanilla_{mtype}_tp{args.temperature}_topp{args.top_p}_s{args.start}_e{args.end}_{dt_string}_seed{args.seed}_bm{args.num_beams}.jsonl'
    if args.reverse:
        filename = filename.replace('.jsonl', '') + '_reverse.jsonl'
    
    if os.path.exists(filename):
        prev = jsonlines_load(filename)
        indexes = [x['index'] for x in prev if 'index' in x]
    else:
        indexes = []
        with jsonlines.open(filename, mode='w') as writer:
            writer.write(args.prompts)
    
    inputs = []
    for example in tqdm(data_test):
        index = example['index']
        if index in indexes: continue
        rst = {'index': index}
        rst.update(example)
        
        inputs.append(rst)
    
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
        num_beams=args.num_beams
    )
    
    if args.reverse:
        inputs = inputs[::-1]
    
    for batch_idx in tqdm(range((len(inputs) + args.batch_size - 1) // args.batch_size)):
        batch = inputs[batch_idx * args.batch_size: (batch_idx + 1) * args.batch_size]
        
        contexts, prompts, attn_masks = [], [], []
        for exp in batch:
            full_prompt, _ = get_prompt_inputs(args.dt_name, args.prompts, exp, use_chatgpt=args.chatgpt)
            contexts.append(full_prompt)
        batch_inputs = tokenizer(contexts, return_tensors="pt", padding=True)        
        prompts, attn_masks = batch_inputs.input_ids, batch_inputs.attention_mask
        
        if args.verbal:
            print('======================')
            print(f'Index: {exp["index"]}\nQuestion: {exp["question"]}')
        
        raw_results = prompt_the_result(model, tokenizer, prompts, attn_masks, generation_config, args.n_samples)
        # print(f"raw_results: {raw_results}")
        results = parse_api_result(raw_results, llama=True, return_prob=False)
        # print(f"results: {results}")
        
        if len(batch) == 1:
            result_counter = Counter()
            for code in results:
                # print(f"code: {code}")
                ans = safe_execute(code)
                # print(f"ans: {ans}")
                ans = floatify_ans(ans)
                # print(f"floatified ans: {ans}")
                print(f"ans: {ans}")
                if ans is not None:
                    result_counter.update([ans])
            
            # print(f"num of counter: {sum(result_counter.values())}")  
            
            prediction = None
            
            if len(result_counter) > 0:
                prediction = result_counter.most_common(1)[0][0]
            gt_ans = exp.get('answer', None)
            if finqa_equal(prediction, gt_ans, False):
                correct += 1
            else:
                wrong += 1
            exp = batch[0]
            exp.update({
                'executed': prediction, 'generated': results,
            })
            with jsonlines.open(filename, mode='a') as writer:
                writer.write(exp)
        else:
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
                    'executed': prediction, 'generated': [rst],
                })
                with jsonlines.open(filename, mode='a') as writer:
                    writer.write(exp)

        torch.cuda.empty_cache()

    accuracy = correct / (correct + wrong)
        
    print('======================')
    print(correct / (correct + wrong), '(', correct, '/', correct + wrong, ')')

    # Save accuracy to a summary file
    summary_filename = filename.replace('.jsonl', '_summary.txt')
    with open(summary_filename, 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f} ({correct}/{correct + wrong})\n')
