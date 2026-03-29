<h2 align="center">Mitigating Mismatch within Reference-based Preference Optimization</h2>
<p align="center"><b>ICLR 2026 Poster</b> | <a href="https://openreview.net/pdf?id=k79Un1LSXy">[Paper]</a> | <a href="https://github.com/tmllab/2026_ICLR_HyPO">[Code]</a> </p>
<p align="center"> <a href="https://suqinyuan.github.io">Suqin Yuan</a>, <a href="https://xingruiyu.github.io">Xingrui Yu</a>, <a href="https://scholar.google.com/citations?user=pM9DLNIAAAAJ&hl=en">Jiyang Zheng</a>,  <a href="https://lfeng1995.github.io">Lei Feng</a>, <a href="https://people.csiro.au/W/D/Dadong-Wang">Dadong Wang</a>, <a href="https://www.a-star.edu.sg/cfar/about-cfar/management/prof-ivor-tsang">Ivor Tsang</a>, <a href="https://tongliang-liu.github.io">Tongliang Liu</a> </p>

### TL;DR
HyPO is a simple drop-in modification to DPO that prevents the reference model from weakening the learning signal when it prefers the rejected response, while still retaining the useful regularization provided by the reference model.

### BibTeX
```bibtex
@inproceedings{
yuan2026mitigating,
title={Mitigating Mismatch within Reference-based Preference Optimization},
author={Suqin Yuan and Xingrui Yu and Jiyang Zheng and Lei Feng and Dadong Wang and Ivor Tsang and Tongliang Liu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026}
}
```

### Reproducing HyPO with Alignment Handbook

Our implementation builds on the Alignment Handbook, which itself is built around Hugging Face Transformers and `trl`. HyPO is implemented as a thin modification of TRL's `DPOTrainer`: the core change is to replace the standard DPO reference margin inside the DPO loss with the HyPO clipped reference margin.

#### Environment
- `trl==0.9.6`
- `transformers==4.45.2`
- `accelerate==1.6.0`
- `deepspeed==0.15.4`
- `vllm==0.8.2`
- `flash-attn==2.7.4.post1`
- `torch==2.6.0+cu124`
- `python==3.10.16`
- `cuda==12.4`

#### To reproduce HyPO:

1. Clone the <a href="https://github.com/huggingface/alignment-handbook">Alignment Handbook</a>.


2. Add the provided HyPO-specific python files into `./alignment-handbook/scripts`:
   - run_hypo.py
   - hypo_trainer.py 
   - hypo_config.py

   In our release, HyPO is integrated into the standard DPO training flow through these three components.
 
3. Launch training with the provided YAML config:

   - HyPO_Llama-3-8B-Instruct.yaml 
   - HyPO_Llama-3-Base-8B.yaml 
   - HyPO_Mistral-7B-Base.yaml 
   - HyPO_Mistral-7B-Instruct-v0.2.yaml 

	For the main HyPO recipe used in the paper, we use a same-architecture checkpoint pre-aligned with SimPO as the stronger reference model together with the HyPO objective. For base-model experiments, we also use the SFT checkpoint released by SimPO. They all can be downloaded from Hugging Face according to the model paths specified in the YAML files.

#### Example launch command

For example, to train HyPO_Llama-3-8B-Instruct, you can run:

```bash
ACCELERATE_LOG_LEVEL=info \
accelerate launch \
  --config_file recipes/accelerate_configs/deepspeed_zero3.yaml \
  --num_processes=2 \
  scripts/run_hypo.py \
  recipes/HyPO_Llama-3-8B-Instruct.yaml \
  --push_to_hub=False \
  --overwrite_output_dir=True \
  --logging_strategy=steps \
  --logging_steps=10 \
  --report_to=tensorboard
```
### Evaluation

We follow the evaluation protocol and scripts released in the <a href="https://github.com/princeton-nlp/SimPO">SimPO repository</a>. After training a HyPO checkpoint in Alignment Handbook, we directly plug the resulting policy model into the SimPO evaluation pipeline, while keeping the benchmark setup, rollout configuration, judge model, and reporting protocol unchanged.

In practice, reproducing the evaluation only requires replacing the policy checkpoint path in the original SimPO evaluation commands with the HyPO-trained checkpoint produced by Alignment Handbook. 

#### Example workflow

1. Train HyPO in Alignment Handbook and obtain the target checkpoint.
2. Enter the SimPO repository.
3. Replace the policy model path in the official SimPO evaluation commands with your HyPO checkpoint path.
4. Run the original SimPO evaluation scripts with the same settings.

Please note that different model settings, such as model family/architecture and whether the model is base or instruction-tuned, require different training YAMLs and evaluation files, following the corresponding conventions in prior practice.

#### Contact: Suqin Yuan (suqinyuan.cs@gmail.com).
