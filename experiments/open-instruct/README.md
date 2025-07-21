### MATH and TruthfulQA

To reproduce the results of Llama-3.1-8B Instruct and DeepSeek-R1-Distill-Qwen-1.5B on MATH dataset and Llama-2-7B-Chat, you need to activate the conda environment:

```bash
conda activate math_truthfulqa
```
Then you can run the following scripts to reproduce the results. You might need to modify the `CUDA_VISIBLE_DEVICES` variable to the GPU you want to use. You also need to specify the `hf_token` as your own Hugging Face token in the `eval/MATH/run_eval_tp0dot6.py`, `eval/MATH/run_eval_tp0dot6_topp0dot95.py`, and `eval/truthfulqa/run_eval_cautious_tp1dot2.py` scripts.

The output files of the following scripts will be saved in `results`. All the output files of the following experiments are provided in the [Dropbox Link](https://www.dropbox.com/scl/fo/vvla3vm4lp56b245qq6bh/AAOoZWvcLOqnQhy_uqPygFk?rlkey=3zpioh4gkh7cla2d77lkv1dat&dl=0).

#### LLama-3.1-8B Instruct on MATH

##### 5 independent runs 

```bash
bash scripts/eval/MATH_cautious_0_tp0.6.sh
bash scripts/eval/MATH_cautious_1_tp0.6.sh
bash scripts/eval/MATH_cautious_2_tp0.6.sh
bash scripts/eval/MATH_cautious_3_tp0.6.sh
bash scripts/eval/MATH_cautious_4_tp0.6.sh
```


##### Self Consistency (40 runs)

```bash
bash scripts/eval/MATH_cautious_0_tp0.6.sh
bash scripts/eval/MATH_cautious_1_tp0.6.sh
...
bash scripts/eval/MATH_cautious_39_tp0.6.sh

```
and then do majority voting by running:
```bash
python results/print_sc.py "./MATH/Llama-3.1-8B-Instruct_tp0.6_catious_*/predictions.jsonl" 
```

#### DeepSeek-R1-Distill-Qwen-1.5B on MATH

##### 5 independent runs 

```bash
bash scripts/eval/MATH_deepseek_cautious_0_tp0.6.sh
bash scripts/eval/MATH_deepseek_cautious_1_tp0.6.sh
bash scripts/eval/MATH_deepseek_cautious_2_tp0.6.sh
bash scripts/eval/MATH_deepseek_cautious_3_tp0.6.sh
bash scripts/eval/MATH_deepseek_cautious_4_tp0.6.sh
```
##### Self Consistency (40 runs)

```bash
bash scripts/eval/MATH_deepseek_cautious_0_tp0.6.sh
bash scripts/eval/MATH_deepseek_cautious_1_tp0.6.sh
...
bash scripts/eval/MATH_deepseek_cautious_39_tp0.6.sh

```
and then do majority voting by running:
```bash
python results/print_sc.py "./MATH/deepseek-r1-distill-qwen-1.5B_cautious_*/predictions.jsonl" 
```

The results will be saved in the `results/MATH` directory.

#### Llama-2-7B-Chat on TruthfulQA

##### 5 independent runs 

```bash
bash scripts/eval/truthfulqa_cautious_perplexity_0.sh
bash scripts/eval/truthfulqa_cautious_perplexity_1.sh
bash scripts/eval/truthfulqa_cautious_perplexity_2.sh
bash scripts/eval/truthfulqa_cautious_perplexity_3.sh
bash scripts/eval/truthfulqa_cautious_perplexity_4.sh
```
The results will be saved in the `results/truthfulqa` directory.


