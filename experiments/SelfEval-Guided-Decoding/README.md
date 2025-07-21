### GSM8K and StrategyQA

To reproduce the results of Llama-3.1-8B Instruct and DeepSeek-R1-Distill-Qwen-1.5B on GSM 8K and StrategyQA datasets, you need to activate the conda environment:

```bash
conda activate gsm8k_strategyqa
```
Then you can run the following scripts to reproduce the results. You need to specify the `auth_token` as your own Hugging Face token in the `src/generate_code_baseline_llama3.1_cautious.py`, `src/generate_code_baseline_deepseek_seed.py`, and `src/generate_code_baseline_deepseek_seed_strqa.py` scripts. You might alsp need to change the `date_time` in the command to the actual date and time when you run the script and modify the `CUDA_VISIBLE_DEVICES` variable to the GPU you want to use.

The output files of the following scripts will be saved in `outputs`. All the output files of the following experiments are provided in the [Dropbox Link](https://www.dropbox.com/scl/fo/an6t4pyvddp3h58m9wrux/AKDW1dhYbOEIzeJNu5Tw0Ck?rlkey=cpgw17b6x7dwcqpdufzix7iwx&st=xk8y8rex&dl=0).

#### LLama-3.1-8B Instruct on GSM 8K

##### 5 independent runs 

```bash
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed0.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed1.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed2.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed3.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed4.sh
```
##### Self Consistency (40 runs)

```bash
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed0.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed1.sh
...
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp1.2-topp0.9-el0.01-eh1.5_maxtrial10_seed39.sh

```
and then do majority voting by running:
```bash
python src/print_accuracy_sc.py "gsm8k_vanilla_Llama-3.1-Instruct-8B_tp1.2_topp0.9_s0_e1319_*_seed{}_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl" 
```

#### DeepSeek-R1-Distill-Qwen-1.5B on GSM 8K

##### 5 independent runs 

```bash
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed0.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed1.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed2.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed3.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed4.sh
```
##### Self Consistency (40 runs)

```bash
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed0.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed1.sh
...
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp1.2-topp0.95-el0.01-eh1.5_seed39.sh

```
and then do majority voting by running:
```bash
python src/print_accuracy_sc.py "gsm8k_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp1.2_topp0.95_s0_e1319_*_seed{}_entropy_low0.01_entropy_high1.5_maxtrial10.jsonl" 
```


#### LLama-3.1-8B Instruct on StrategyQA

##### 5 independent runs 

```bash
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed0.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed1.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed2.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed3.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed4.sh
```
then we can get the accuracy by running:
```bash
python src/execute_and_evaluate/baseline_interpret_and_evaluate.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_{date_time}_seed0_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_{date_time}_seed1_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_{date_time}_seed2_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_{date_time}_seed3_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_{date_time}_seed4_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl
```

##### Self Consistency (40 runs)

```bash
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed0.sh
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed1.sh
...
bash scripts/llama/run_baseline_llama3.1_8B_Instruct_tp0.8_topp0.9_cautious_strqa_seed39.sh

```
and then do majority voting by running:
```bash
python src/execute_and_evaluate/baseline_interpret_and_evaluate_sc.py "./outputs/strategyqa/test_outputs/strategyqa_vanilla_Llama-3.1-Instruct-8B_tp0.8_topp0.9_s0_e2290_*_seed{}_entropy_low0.01_entropy_high1.5_trials10_cautious_perplexity.jsonl" 
```

#### DeepSeek-R1-Distill-Qwen-1.5B on StrategyQA

##### 5 independent runs 

```bash
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed0.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed1.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed2.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed3.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed4.sh
```
then we can get the accuracy by running:
```bash
python src/execute_and_evaluate/baseline_interpret_and_evaluate_deepseek.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp0.8_topp0.95_s0_e2290_{date_time}_seed0_entropy_low0.01_entropy_high1.5_trials10.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate_deepseek.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp0.8_topp0.95_s0_e2290_{date_time}_seed1_entropy_low0.01_entropy_high1.5_trials10.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate_deepseek.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp0.8_topp0.95_s0_e2290_{date_time}_seed2_entropy_low0.01_entropy_high1.5_trials10.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate_deepseek.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp0.8_topp0.95_s0_e2290_{date_time}_seed3_entropy_low0.01_entropy_high1.5_trials10.jsonl
python src/execute_and_evaluate/baseline_interpret_and_evaluate_deepseek.py ./outputs/strategyqa/test_outputs/strategyqa_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp0.8_topp0.95_s0_e2290_{date_time}_seed4_entropy_low0.01_entropy_high1.5_trials10.jsonl
```

##### Self Consistency (40 runs)

```bash
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed0.sh
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed1.sh
...
bash scripts/llama/run_baseline_deepseek1.5B_Instruct_tp0.8-topp0.95-el0.01-eh1.5_strqa_seed39.sh

```
and then do majority voting by running:
```bash
python src/execute_and_evaluate/baseline_interpret_and_evaluate_deepseek_sc.py "./outputs/strategyqa/test_outputs/strategyqa_vanilla_DeepSeek-R1-Distill-Qwen-1.5B_tp0.8_topp0.95_s0_e2290_*_seed{}_entropy_low0.01_entropy_high1.5_trials10.jsonl" 
```








