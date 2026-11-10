# mdlARC

ARC-AGI from scratch: small transformers trained only on ARC puzzles, with no language-model pretraining.

## Versions

| Version | Date | Result | Code | Post |
| --- | --- | --- | --- | --- |
| **v3: Looped mdlARC** | Nov 2026 | 37.08% on ARC-AGI-2 (Kaggle public leaderboard, no pretrained LLM); 71.88 / 81.50 top-2 / oracle on ARC-AGI-1 at 19M parameters | `main` (tag `v3-nov2026`) | [Looped mdlARC](https://mvakde.github.io/blog/looped-mdlarc/) |
| v2 | Mar 2026 | 44% on ARC-AGI-1 for 67 cents, 75M unlooped transformer | tag [`v2-mar2026`](https://github.com/mvakde/mdlARC/tree/v2-mar2026) | [44% on ARC-AGI-1 in 67 cents](https://mvakde.github.io/blog/44-on-arc-1/) |
| v1 | Dec 2025 | 27.5% on ARC-AGI-1, 28M transformer, ~$2 | tag [`v1-dec2025`](https://github.com/mvakde/mdlARC/tree/v1-dec2025) | [New Pareto Frontier on ARC-AGI](https://mvakde.github.io/blog/new-pareto-frontier-arc-agi/) |

## v3: Looped mdlARC
This repository holds the code for **"Looped mdlARC: ARC-AGI Without Language Models"**,
our ARC Prize 2026 entry. The submitted model scored **37.08% on the Kaggle ARC-AGI-2
public leaderboard** (submission 57908007).

It is a snapshot of the research code, plus the exact training specs and the submitted Kaggle
notebook. The research-branch notes are in [README_research.md](README_research.md).

## Model

A decoder-only transformer trained from scratch on ARC grids, with no language-model
pretraining: width 1024, 16 attention heads, SwiGLU FFN width 2688 and 8 physical
blocks. Blocks 3-6 (zero-based `loop_start_layer=2`, `loop_end_layer=6`) are run 6
times with shared weights and input injection, so each token passes through 28 block
applications. Training uses four-token multi-token prediction and deep supervision
on the early loop exits. The model has 99.8M parameters. See
[specs/README_looping_next_latent.md](specs/README_looping_next_latent.md) for the loop
schedule.

## What is where

- `specs/arc-prize-2026/loopv3-full-d1024f2688x6ds-dc4-b160-e270.json`: pretraining from
  scratch for 270 epochs on 8 GPUs (micro-batch 20 per GPU, global batch 160, NorMuon
  0.00693 / AdamW 0.00125, at most 4 context demos per training sequence).
- `specs/arc-prize-2026/loopv3-full-d1024f2688x6ds-dc4-b160-e440-r270.json`: the
  continuation that produced the submitted checkpoint. It resumes from the epoch-270
  checkpoint with optimizer state, re-warms the learning rates to 0.8x (NorMuon 0.005544,
  AdamW 0.001), and trains to epoch 440 with decay from epoch 352. Its `checkpoint`
  field is the path used on the training machine; point it at your own epoch-270
  checkpoint.
- `notebooks/submit_d1024_e440_ddp4_ft180.ipynb`: the submitted Kaggle notebook (4x L4,
  internet off). It fine-tunes the epoch-440 checkpoint on each task's demonstrations for
  180 epochs with one 4-GPU DDP team, then decodes with augmentation voting. The notebook
  embeds its own copy of the runtime source. Its first markdown cell still carries some
  text from earlier notebook versions; the settings in the code cells are authoritative.
- `src/`, `launch_run.py`, `dataset_building_scripts/`: model, training, decoding and
  data building.

Both specs train on a pinned family/composition data package
(`data/compositions/releases/counts-v1-20261024/release.json` and
`assets_tmp/arc2-family-full-counts-v1-20261024-r1.zip`), which is not part of this
branch. [specs/README_family_data.md](specs/README_family_data.md) describes how such
packages are prepared.

## Training and running

Follow [README_research.md](README_research.md) for requirements, the dataset build and
`launch_run.py --config <spec>`, [specs/README.md](specs/README.md) for spec fields, and
[specs/README_ddp.md](specs/README_ddp.md) for multi-GPU runs. The remote-launch tooling
under `workflows/` and `scripts/` was written for our own Colab, Vast, SF Compute and
RunPod accounts and needs your own credentials and storage.

## Links

- Blog post: https://mvakde.github.io/blog/looped-mdlarc/
- Model weights: https://www.kaggle.com/models/mithilavakde/mdlarc
- ARC Data Squeeze: https://github.com/mvakde/arc-data-squeeze

Released under the MIT License (see [LICENSE](LICENSE)).
