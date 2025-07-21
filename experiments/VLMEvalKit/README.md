### MMVet and MathVista

To reproduce the results of Llama-3.2-11B-Vision-Instruct and LLaVA-CoT on MM Vet and MathVista datasets, you need to activate the conda environment:

```bash
conda activate mmvet_mathvista
```
Then you can run the following scripts to reproduce the results. You need to specify the `login(token="YOUR_HF_TOKEN")` as your own Hugging Face token in the `run.py` script.

```bash
torchrun --nproc-per-node=8 run.py --data MMVet MathVista_MINI --model LLaVA-CoT Llama-3.2-11B-Vision-Instruct --verbose
```

The output results will be saved in the `outputs/LLaVA-CoT` and `outputs/Llama-3.2-11B-Vision-Instruct` directories.








