# KVFlow

This repository contains the source code for ***[NeurIPS'25]** KVFlow: Efficient prefix caching for accelerating LLM-based multi-agent workflows* and ***[Preprint]** ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management*. It provides:
- **SScheduler layer** (`/SScheduler`): Pluggable mid-layer for agent simulations / workflows; it can call the scheduler to expose request metadata that helps the serving engine optimize memory management.
- **SGLang-based serving engine**(`/python/sglang`): Implements priority-based eviction and overlapped prefetch for both LoRA and KV payloads.

## News
- [2025/09] [KVFlow](https://arxiv.org/abs/2507.07400) accepted to NeurIPS 2025.
- [2026/01] [ScaleSim](https://arxiv.org/abs/2601.21473) preprint released on arXiv.
- [2026/02] ScaleSim codebase released on GitHub.

## Getting Started
### Install
```bash
git clone git@github.com:PanZaifeng/KVFlow.git
cd KVFlow
pip install "python[all]"
```
### Serve
- With config (YAML/JSON):
	```bash
	python -m sglang.launch_server --config ./python/sglang/configs/example.yaml
	```
- Without config (inline args):
	```bash
	python -m sglang.launch_server --model-path <model> --port 8001 --enable-lora --lora-target-modules all --max-lora-rank 64 --max-loras-per-batch 100 --max-total-tokens 100000 --enable-hierarchical-cache --hicache-size 20
	```

Key params:
- `model_path`: HF repo or local path to weights.
- `port`/`host`: HTTP endpoint for serving.
- `enable_lora`, `lora_target_modules`, `max_lora_rank`, `max_loras_per_batch`: LoRA batching knobs.
- `load_ahead_step`, `evict_pri_level`, `enable_holding`, `enable_interrupt`, `disable_prefetch`, `disable_lr_pf`, `disable_kv_pf`: prefetch/eviction controls.

More options live in `python/sglang/srt/server_args.py`; CLI flags match config keys.

## Citation

If you find this work useful, please cite:

```bibtex
@article{pan2025kvflow,
  title={KVFlow: Efficient prefix caching for accelerating LLM-based multi-agent workflows},
  author={Pan, Zaifeng and Patel, Ajjkumar and Hu, Zhengding and Shen, Yipeng and Guan, Yue and Li, Wan-Lu and Qin, Lianhui and Wang, Yida and Ding, Yufei},
  journal={arXiv preprint arXiv:2507.07400},
  year={2025}
}

@article{pan2026scalesim,
  title={ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management},
  author={Pan, Zaifeng and Shen, Yipeng and Hu, Zhengding and Wang, Zhuang and Manocha, Aninda and Wang, Zheng and Yu, Zhongkai and Guan, Yue and Ding, Yufei},
  journal={arXiv preprint arXiv:2601.21473},
  year={2026}
}
```
