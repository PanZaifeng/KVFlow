<div align="center" id="sglangtop">
<h1>KVFlow & ScaleSim</h1>

[KVFlow](https://arxiv.org/abs/2507.07400) · [ScaleSim](https://arxiv.org/abs/2601.21473)

</div>

## News
- [2025/09] KVFlow accepted to NeurIPS 2025.
- [2026/01] ScaleSim preprint released on arXiv.
- [2026/02] ScaleSim codebase released on GitHub.


## About
This repo hosts ScaleSim (and KVFlow) code. It provides:
- **SScheduler layer** (`/SScheduler`): Pluggable mid-layer for agent simulations / workflows; it can call the scheduler to expose request metadata that helps the serving engine optimize memory management.
- **SGLang-based serving engine**(`/python/sglang`): Implements priority-based eviction and overlapped prefetch for both LoRA and KV payloads.

## Getting Started
### Install
```bash
conda create -n scalesim python=3.12
conda activate scalesim
pip install -e ./python[all]
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

## Cite
```bibtex
@misc{pan2025kvflowefficientprefixcaching,
title={KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows},
author={Zaifeng Pan and Ajjkumar Patel and Zhengding Hu and Yipeng Shen and Yue Guan and Wan-Lu Li and Lianhui Qin and Yida Wang and Yufei Ding},
year={2025},
eprint={2507.07400},
archivePrefix={arXiv},
primaryClass={cs.DC},
url={https://arxiv.org/abs/2507.07400},
}@misc{pan2026scalesimservinglargescalemultiagent,
title={ScaleSim: Serving Large-Scale Multi-Agent Simulation with Invocation Distance-Based Memory Management},
author={Zaifeng Pan and Yipeng Shen and Zhengding Hu and Zhuang Wang and Aninda Manocha and Zheng Wang and Zhongkai Yu and Yue Guan and Yufei Ding},
year={2026},
eprint={2601.21473},
archivePrefix={arXiv},
primaryClass={cs.AI},
url={https://arxiv.org/abs/2601.21473},
}
```