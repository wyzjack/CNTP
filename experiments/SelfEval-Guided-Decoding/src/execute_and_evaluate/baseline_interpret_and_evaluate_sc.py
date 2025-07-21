import sys
import regex
import glob
from collections import Counter

sys.path.append('/sensei-fs/users/yizhouw/projects/vlm_self_correct/SelfEval-Guided-Decoding/src')
from utils.tool import *
from utils.dataset import jsonlines_load, jsonlines_dump
from execute_and_evaluate.interpret_and_evaluate import check_eq


if __name__ == '__main__':
    N = 40
    
    # Get base_pattern from command line argument
    if len(sys.argv) < 2:
        print("Usage: python baseline_interpret_and_evaluate_sc.py <base_pattern>")
        print("Example: python baseline_interpret_and_evaluate_sc.py \"./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_*_seed*_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl\"")
        sys.exit(1)
    
    base_pattern = sys.argv[1]
    
    # Find all matching files using the base pattern
    all_files = glob.glob(base_pattern)
    
    if not all_files:
        print("No input files found!")
        sys.exit(1)
        
    assert len(all_files) == N, f"Expected {N} files, but found {len(all_files)}"
    
    # Initialize combined data structure
    combined_results = {}  # index -> list of predictions
    accu, dur = {}, []
    
    # Load and process all files
    for fname in all_files:
        data = jsonlines_load(fname)
        
        # dtname = fname.strip().split('/')[4]
        dtname = 'strategyqa'
        if dtname == 'asdiv':
            real_test = jsonlines_load('/hdd2/yuxi/math_word_problem/nlu-asdiv-dataset/dataset/asdiv.jsonl')
            real_test = [x['input'] for x in real_test]
        
        for d in tqdm(data, desc=f"Processing {fname}"):
            if 'index' not in d: continue
            if dtname == 'asdiv' and all(not x.startswith(d['Body']) for x in real_test): continue
            
            gt_ans = d['answer']
            
            # Process single prediction for this file
            if len(d['generated']) == 1 and d.get('executed', None) is not None and False:
                prd = d['executed']
            else:
                prds, scores, probs = [], [], []
                for ii, g in enumerate(d['generated']):
                    if dtname in ['csqa', 'strategyqa', 'sports', 'saycan']:
                        # print(dtname)
                        if isinstance(g, dict):
                            g['generated'] = g['generated'][:-1] if 'return result' in g['generated'][-1] else g['generated']
                    
                    if isinstance(g, list):
                        code, p = g[0], (g[2] if len(g) > 2 else [])
                    elif isinstance(g, str):
                        code, p = g, []
                        if 'def solution():' in code:
                            code = code[code.index('def solution():'):]
                        if 'return result' in code:
                            code = code[:code.index('return result') + len('return result')]
                    else:
                        code, p = '\n'.join(g['generated']), g['prob']
                    
                    if dtname in ['csqa']:
                        rst = [x[0].strip('()') for x in regex.finditer(r'\([a-z\s]+\)', code.strip().split('\n')[-1])]
                        prd = [x.strip() for r in rst for x in r.split() if regex.match(r'^[a-z]$', x.strip())]
                        prd = tuple(set(prd))
                        if not len(prd) and ' answer is ' in code:
                            ans = [x for x in code.strip().split('\n') if x.count(' answer is ')]
                            prd = [x.strip() for r in ans[-1:] for x in r.split() if regex.match(r'^[a-z]$', x.strip())]
                            prd = tuple(set(prd))
                    elif dtname in ['strategyqa', 'sports']:
                        splited = code.strip().split()
                        prd = splited[-1].strip('.') if splited else None
                    elif dtname in ['gsm8k_cot']:
                        ans = code.strip().split('\n')[-1].replace('So the answer is ', '')
                        prd = [x[0] for x in regex.finditer(r'[\d\.,]+', ans) if regex.search(r'\d', x[0])]
                        if len(prd) > 2: prd = prd[-1]
                        elif len(prd): prd = prd[0]
                        else: prd = None
                        try:
                            prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
                        except:
                            prd = None
                    else:
                        exe_rst = safe_execute(code)
                        prd = floatify_ans(exe_rst)
                        if type(prd) not in [float, int, bool, str, dict, set, list, tuple]:
                            prd = floatify_ans(simplify_ans(exe_rst))
                    
                    if len(d['generated']) == 1: d['executed'] = prd
                    
                    prds.append(prd)
                    if len(p): probs.append(nor_prod(x[0]**(1/max(1, x[1])) for x in p))
                    if isinstance(g, dict) and len(p):
                        s = nor_prod(aggregate_conf_and_prob(c, p[0]**(1/max(1, p[0])), r=0.5) for c, p in zip(g['conf'], g['prob']))
                        scores.append(s)
                
                if len(probs): p_idx = probs.index(max(probs))
                else: p_idx = 0
                
                prd = prds[p_idx]
            
            # Store prediction in combined results
            if d['index'] not in combined_results:
                combined_results[d['index']] = {'predictions': [], 'gt_ans': gt_ans, 'question': d['question']}
            combined_results[d['index']]['predictions'].append(prd)
            
            if 'run_time' in d: dur.append(d['run_time'])
    
    # Evaluate using self-consistency across all files
    for idx, result in combined_results.items():
        predictions = result['predictions']
        result_counter = Counter()
        result_counter.update([x for x in predictions if x is not None])
        final_pred = result_counter.most_common(1)[0][0] if len(result_counter) else None
        
        accu[idx] = check_eq(final_pred, result['gt_ans'], dtname=dtname, percent_check=result['question'])
    
    print('accu ({}):'.format(len(accu)), sum(accu.values()) / len(accu) * 100)
    if len(dur): print('avg running time:', sum(dur)/len(dur) if isinstance(dur[0], float) else sum(sum(x) for x in dur)/len(dur))

    # Save results to a new file based on the pattern name
    base_output = base_pattern.replace('*', 'combined')
    with open(base_output.replace('.jsonl', '.txt'), 'w') as f:
        f.write('accu ({}):'.format(len(accu)))
        f.write(str(sum(accu.values()) / len(accu) * 100))


    jsonlines_dump(data, fname)
    