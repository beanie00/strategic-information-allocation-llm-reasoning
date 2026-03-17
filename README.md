# Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty

## Overview

This repository contains the analysis code for the paper **"Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty"** ([arXiv:2603.15500](https://arxiv.org/abs/2603.15500)).

Much of our codebase builds upon **LIMO: Less Is More for Reasoning** ([paper](https://arxiv.org/pdf/2502.03387), [code](https://github.com/GAIR-NLP/LIMO)), which in turn uses [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) for training. For installation and environment setup, please refer to the [LLaMA-Factory repository](https://github.com/hiyouga/LlamaFactory).

For MI Peak experiments, our implementation is based on **"Demystifying Reasoning Dynamics with Mutual Information: Thinking Tokens are Information Peaks in LLM Reasoning"** ([paper](https://arxiv.org/abs/2506.02867), [code](https://github.com/ChnQ/MI-Peaks)).

We thank all teams for open-sourcing their work.


## Repository Structure

- **`/train`** — SFT training scripts for LIMO distillation experiments
- **`/eval`** — Math benchmark evaluation
- **`/mi_peak`** — Analysis for Section 3.4.1
- **`/distillation_without_epistemic_verbalization`** — Analysis for Section 4.1
- **`/analysis`** — Analysis for Section 4.2
- **`/example_eval_outputs`** — Example evaluation outputs

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation
```bibtex
@misc{kim2026understanding,
    title={Understanding Reasoning in LLMs through Strategic Information Allocation under Uncertainty},
    author={Kim, Jeonghye and Luo, Xufang and Kim, Minbeom and Lee, Sangmook and Kim, Dohyung and Li, Dongsheng and Yang, Yuqing},
    year={2026},
    eprint={2603.15500},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```