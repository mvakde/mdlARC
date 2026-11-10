# Experiment specs

A spec is a JSON file that describes one run: the training data, the model, the training schedule and
the evaluations. `launch_run.py` merges it with the defaults in `launch_run.py` and `src/research_options.py`;
any key a spec leaves out takes the default. Every result in the paper has a spec here, and each spec
sets every key whose default changed since its run, so it reproduces the settings that run used.

## Running a spec

```bash
python launch_run.py --config specs/data_squeeze/arc1_d256x6.json --preview   # print the resolved plan
python launch_run.py --config specs/data_squeeze/arc1_d256x6.json
```

The launcher builds the dataset, trains, then runs the spec's evaluations. Everything goes to
`runs/<run_name>/` (set `MDLARC_RUNS_DIR` to use another folder): `pretrain/tiny.pt` and its training log,
and for each evaluation `evals/<name>/score.json` (top-2), `oracle_score.json` (any candidate),
`submission.json` and `candidates.json.gz`.

Some specs start from another run's output, so run that one first:

| Spec | Needs |
| --- | --- |
| [`paper/rearc_and_scale/arc1_d384x6_scale_aug_rescore_eval.json`](paper/rearc_and_scale/arc1_d384x6_scale_aug_rescore_eval.json) | 2.3 (`runs/loopL-d384-exit-mid6-scale12-e330/`: checkpoint and `final_scale_views` candidates) |
| [`paper/rearc_and_scale/arc1_d384x6_rescore_eval.json`](paper/rearc_and_scale/arc1_d384x6_rescore_eval.json) | 1.3 (`runs/loopL-d384-exit-mid6-e330/`: checkpoint and `final` candidates) |
| [`paper/loop_ablations/d256x6_eval_loop_counts.json`](paper/loop_ablations/d256x6_eval_loop_counts.json) | 1.1 checkpoint |
| [`paper/loop_ablations/d384x6_eval_loop_counts.json`](paper/loop_ablations/d384x6_eval_loop_counts.json) | 1.3 checkpoint |
| [`paper/loop_ablations/d384_loop_x6_ds_continue_e440.json`](paper/loop_ablations/d384_loop_x6_ds_continue_e440.json) | 1.3 checkpoint (resumed with its optimizer state) |
| [`paper/kaggle_d1024/continue_e440_from_e270.json`](paper/kaggle_d1024/continue_e440_from_e270.json) | K1a final checkpoint (epoch 270) |
| [`paper/kaggle_d768/continue_e180_from_e080.json`](paper/kaggle_d768/continue_e180_from_e080.json) | K2a checkpoint `tiny.epoch080.pt` (stop K2a after epoch 80) |

The Kaggle specs read the training package `assets_tmp/arc2-family-full-counts-v1-20261024-r1.zip`;
`python data_generation/fetch.py` downloads it there. Specs with `"distributed": {"nproc_per_node": N}`
start one process per GPU on one machine; `batch_size` is per GPU.

## Layout

| Folder | Contents |
| --- | --- |
| `data_squeeze/` | ARC-AGI-1 and ARC-AGI-2 baselines, 0.71M to 18.9M parameters |
| `paper/rearc_and_scale/` | ReARC data, scale augmentation, scale views, likelihood ranking |
| `paper/loop_ablations/` | loop count, deep supervision (DS), input injection, width, the unlooped depth ladder, decoding at another loop count, continued training |
| `paper/kaggle_d1024/` | the d1024 Kaggle model: pretraining to epoch 270, then continuation to 440 |
| `paper/kaggle_d768/` | the d768 Kaggle model: epochs 1–80 on 4 GPUs, then 81–180 on 8 GPUs |

## Defaults that changed

These defaults differ from earlier versions of this code. Every paper spec sets the value its run used.

| Key | Default now | Earlier | Notes |
| --- | --- | --- | --- |
| `scale_augmentation` | `false` | `true` | ReARC specs (2.1, 2.2) and 2.3 set `true` |
| `inference_scale_views` | `false` | `true` | set in 2.1, 2.2 and the 2.4 / 2.5 evaluations |
| `inference_symbolic_guards` | `false` | `true` | set in 2.1, 2.2 |
| `inference_rescore` | `false` | `true` | set in 2.1, 2.2 (in-decode likelihood ranking) |
| `dataset.add_rearc` | `null` | `1000` | set to `1000` in 2.1, 2.2 |
| `compile_scope` | `"blocks"` | whole model (before per-block compilation) | `"blocks"` compiles each transformer block once; specs whose run compiled the whole model set `"model"` |
| `inference_speculative` | `true` | `false` (before the decode speed-ups) | self-speculative greedy decoding; specs whose run decoded with plain greedy steps set `false` |
| `stable_cross_entropy` | `false` | `true` in the Kaggle code | the Kaggle specs set `true` |
| `max_training_demos` | `null` (all demos) | `6` in the Kaggle code | the Kaggle specs set 4 or 6 |
| `kept_checkpoint_every`, `rolling_checkpoint_every` | `50`, `10` | none | extra checkpoint files only; training is unchanged |

Unchanged defaults include FlashAttention 2 (`FA: 2`), dropout and attention dropout 0.1, seed 42,
80 augmented decode views and the pair-group attention options used by every paper spec.

## Paper results

Scores are top-2 / oracle in percent, as reported in the paper. DS = deep supervision (loop exits
trained with weight 0.3); ×K = the looped blocks run K times. All rows train from scratch on the
ARC Data Squeeze set (ARC-AGI-1 training + ARC-AGI-2-only tasks + ConceptARC + the evaluation tasks'
demonstrations; the ARC-AGI-2 rows swap the two ARC sets) unless stated.

### ARC Data Squeeze baselines (ARC-AGI-1: % of 400 tasks; ARC-AGI-2: % of 120 tasks)

| Row | Spec | Model | Epochs | Top-2 / oracle |
| --- | --- | --- | --- | --- |
| 1.1 | [`data_squeeze/arc1_d256x6.json`](data_squeeze/arc1_d256x6.json) | d256, 8 blocks, [2,6) ×6 + DS (8.4M) | 330 | 66.25 / 77.25 |
| 1.2 | [`data_squeeze/arc1_d256x6_seed43.json`](data_squeeze/arc1_d256x6_seed43.json) | as 1.1, seed 43 | 330 | 62.00 / 76.12 |
| 1.3 | [`data_squeeze/arc1_d384x6.json`](data_squeeze/arc1_d384x6.json) | d384 ×6 + DS (18.9M) | 330 | 68.38 / 79.62 |
| 1.4 | [`data_squeeze/arc1_d144x6.json`](data_squeeze/arc1_d144x6.json) | d144 ×6 + DS, 3 heads, FFN 384 (2.0M) | 330 | 55.63 / 68.00 |
| 1.5 | [`data_squeeze/arc1_d96x6.json`](data_squeeze/arc1_d96x6.json) | d96 ×6 + DS, 4 heads, FFN 320 (1.05M) | 330 | 49.00 / 60.62 |
| 1.6 | [`data_squeeze/arc1_d96x6_ffn256.json`](data_squeeze/arc1_d96x6_ffn256.json) | d96 ×6 + DS, 3 heads, FFN 256 (0.90M) | 330 | 41.25 / 55.50 |
| 1.7 | [`data_squeeze/arc1_d128x6.json`](data_squeeze/arc1_d128x6.json) | d128 ×6 + DS, 2 heads, FFN 512 (2.1M) | 330 | 51.25 / 65.25 |
| 1.8 | [`data_squeeze/arc1_d80x6.json`](data_squeeze/arc1_d80x6.json) | d80 ×6 + DS, 4 heads, FFN 256 (0.71M) | 330 | 33.38 / 48.12 |
| 1.9 | [`data_squeeze/arc1_d96x6_ffn296.json`](data_squeeze/arc1_d96x6_ffn296.json) | d96 ×6 + DS, 4 heads, FFN 296 (0.99M) | 330 | 41.50 / 55.88 |
| 1.10 | [`data_squeeze/arc2_d384x6.json`](data_squeeze/arc2_d384x6.json) | ARC-AGI-2, d384 ×6 + DS | 330 | 13.89 / 20.56 (16.67 / 24.67 of 120) |
| 1.11 | [`data_squeeze/arc2_d256x8.json`](data_squeeze/arc2_d256x8.json) | ARC-AGI-2, d256 ×8 + DS | 330 | 12.64 / 18.89 (15.17 / 22.67 of 120) |

### ReARC and scale augmentation

| Row | Spec | Model | Epochs | Top-2 / oracle |
| --- | --- | --- | --- | --- |
| 2.1 | [`paper/rearc_and_scale/arc1_d384x6_rearc.json`](paper/rearc_and_scale/arc1_d384x6_rearc.json) | d384 ×6 + DS, ReARC, scale aug, scale views, guards, likelihood | 330 | 81.38 / 88.00 (plain votes 78.75) |
| 2.2 | [`paper/rearc_and_scale/arc2_d384x6_rearc.json`](paper/rearc_and_scale/arc2_d384x6_rearc.json) | as 2.1 on ARC-AGI-2 | 330 | 20.14 / 32.08 (24.17 / 38.5 of 120) |
| 2.3 | [`paper/rearc_and_scale/arc1_d384x6_scale_aug.json`](paper/rearc_and_scale/arc1_d384x6_scale_aug.json) | d384 ×6 + DS, scale aug; eval `final` | 330 | 70.38 / 79.75 |
| 2.4 | [`paper/rearc_and_scale/arc1_d384x6_scale_aug.json`](paper/rearc_and_scale/arc1_d384x6_scale_aug.json) | same run; eval `final_scale_views` | 330 | 71.00 / 81.50 |
| 2.5 | [`paper/rearc_and_scale/arc1_d384x6_scale_aug_rescore_eval.json`](paper/rearc_and_scale/arc1_d384x6_scale_aug_rescore_eval.json) | eval only: likelihood ranking of the 2.4 candidates, then `tools/guards_eval.py` | – | 71.88 / 81.50 |
| 2.6 | [`paper/rearc_and_scale/arc1_d384x6_rescore_eval.json`](paper/rearc_and_scale/arc1_d384x6_rescore_eval.json) | eval only: likelihood ranking of the 1.3 candidates, then `tools/guards_eval.py` | – | 70.88 / 79.88 |

### Loop ablations (d256, 8 blocks, loop over blocks [2,6), unless stated)

| Row | Spec | Model | Epochs | Top-2 / oracle |
| --- | --- | --- | --- | --- |
| 3.1 | [`paper/loop_ablations/unlooped_8blocks.json`](paper/loop_ablations/unlooped_8blocks.json) | no loop, 8 blocks (8.4M) | 330 | 47.25 / – |
| 3.2 | [`paper/loop_ablations/unlooped_8blocks_seed43.json`](paper/loop_ablations/unlooped_8blocks_seed43.json) | as 3.1, seed 43 | 330 | 47.63 / – |
| 3.3 | [`paper/loop_ablations/loop_x2.json`](paper/loop_ablations/loop_x2.json) | ×2, no DS | 330 | 55.88 / – |
| 3.4 | [`paper/loop_ablations/loop_x3.json`](paper/loop_ablations/loop_x3.json) | ×3, no DS | 330 | 60.13 / – |
| 3.5 | [`paper/loop_ablations/loop_x4.json`](paper/loop_ablations/loop_x4.json) | ×4, no DS | 330 | 61.88 / 72.88 |
| 3.6 | [`paper/loop_ablations/loop_x3_ds.json`](paper/loop_ablations/loop_x3_ds.json) | ×3 + DS | 330 | 59.38 / – |
| 3.7 | [`paper/loop_ablations/loop_x4_ds.json`](paper/loop_ablations/loop_x4_ds.json) | ×4 + DS | 330 | 64.25 / 75.50 (see notes) |
| 3.8 | [`paper/loop_ablations/loop_x8_ds.json`](paper/loop_ablations/loop_x8_ds.json) | ×8 + DS | 330 | 66.50 / 75.25 |
| 3.9 | [`paper/loop_ablations/unlooped_task_embeddings.json`](paper/loop_ablations/unlooped_task_embeddings.json) | no loop + task embeddings | 330 | 48.13 / 59.00 |
| 3.10 | [`paper/loop_ablations/loop_x4_ds_task_embeddings.json`](paper/loop_ablations/loop_x4_ds_task_embeddings.json) | ×4 + DS + task embeddings | 330 | 56.63 / 68.38 |
| 3.11 | [`paper/loop_ablations/unlooped_10blocks.json`](paper/loop_ablations/unlooped_10blocks.json) | no loop, 10 blocks | 330 | 52.13 / – |
| 3.12 | [`paper/loop_ablations/unlooped_12blocks.json`](paper/loop_ablations/unlooped_12blocks.json) | no loop, 12 blocks | 330 | 54.38 / – |
| 3.13 | [`paper/loop_ablations/unlooped_12blocks_seed43.json`](paper/loop_ablations/unlooped_12blocks_seed43.json) | no loop, 12 blocks, seed 43 | 330 | 52.75 / – |
| 3.14 | [`paper/loop_ablations/unlooped_16blocks.json`](paper/loop_ablations/unlooped_16blocks.json) | no loop, 16 blocks | 330 | 56.13 / – |
| 3.15 | [`paper/loop_ablations/unlooped_16blocks_seed43.json`](paper/loop_ablations/unlooped_16blocks_seed43.json) | no loop, 16 blocks, seed 43 | 330 | 55.38 / – |
| 3.16 | [`paper/loop_ablations/unlooped_24blocks.json`](paper/loop_ablations/unlooped_24blocks.json) | no loop, 24 blocks | 330 | 56.25 / – |
| 3.17 | [`paper/loop_ablations/unlooped_32blocks.json`](paper/loop_ablations/unlooped_32blocks.json) | no loop, 32 blocks | 330 | 58.63 / – |
| 3.18 | [`paper/loop_ablations/d384_loop_x4_ds.json`](paper/loop_ablations/d384_loop_x4_ds.json) | d384 ×4 + DS | 330 | 66.13 / 77.88 |
| 3.19 | [`paper/loop_ablations/d768_loop_x4_ds.json`](paper/loop_ablations/d768_loop_x4_ds.json) | d768 ×4 + DS, 12 heads | 330 | 62.75 / 74.38 |
| 3.20 | [`paper/loop_ablations/d768_loop_x4_ds_combo.json`](paper/loop_ablations/d768_loop_x4_ds_combo.json) | d768 ×4 + DS + 4 streams, dense mixing, iteration embeddings, 10× loop LR | 330 | 65.63 / 74.63 |
| 3.21 | [`paper/loop_ablations/d768_8heads_loop_x4_ds.json`](paper/loop_ablations/d768_8heads_loop_x4_ds.json) | d768 ×4 + DS, 8 heads | 330 | 65.62 / 76.25 |
| 3.22 | [`paper/loop_ablations/d768_unlooped.json`](paper/loop_ablations/d768_unlooped.json) | d768, no loop (75.6M) | 330 | 53.75 / – |
| 3.23 | [`paper/loop_ablations/d256x6_eval_loop_counts.json`](paper/loop_ablations/d256x6_eval_loop_counts.json) | eval only: 1.1 decoded at K′ = 3, 4, 5, 7, 8 | – | 57.63 / 70.63, 63.50 / 75.75, 66.00 / 77.50, 66.00 / 75.38, 65.25 / 74.88 |
| 3.24 | [`paper/loop_ablations/d384x6_eval_loop_counts.json`](paper/loop_ablations/d384x6_eval_loop_counts.json) | eval only: 1.3 decoded at K′ = 4, 5, 7, 8 | – | 67.38 / 79.13, 68.75 / 79.38, 67.00 / 78.88, 67.50 / 78.38 |
| 3.25 | [`paper/loop_ablations/loop_x6_ds_no_injection.json`](paper/loop_ablations/loop_x6_ds_no_injection.json) | ×6 + DS, no input injection | 330 | 64.13 / 76.13 |
| 3.26 | [`paper/loop_ablations/loop_x6_ds_no_injection_fixed_shortcut.json`](paper/loop_ablations/loop_x6_ds_no_injection_fixed_shortcut.json) | as 3.25 with a fixed token-embedding shortcut | 330 | 61.50 / 73.13 |
| 3.27 | [`paper/loop_ablations/loop_x6_no_ds.json`](paper/loop_ablations/loop_x6_no_ds.json) | ×6, no DS | 330 | 61.75 / 75.13 |
| 3.28 | [`paper/loop_ablations/loop_x6_plain.json`](paper/loop_ablations/loop_x6_plain.json) | ×6, no DS, no injection (plain loop) | 330 | 62.00 / 75.38 |
| 3.29 | [`paper/loop_ablations/loop_x6_ds_weight1.json`](paper/loop_ablations/loop_x6_ds_weight1.json) | ×6 + DS weight 1.0 | 330 | 64.63 / 75.88 |
| 3.30 | [`paper/loop_ablations/loop_x6_ds_dropout02.json`](paper/loop_ablations/loop_x6_ds_dropout02.json) | ×6 + DS, dropout 0.2 | 330 | 68.50 / 76.13 |
| 3.31 | [`paper/loop_ablations/loop_x6_plain_combo.json`](paper/loop_ablations/loop_x6_plain_combo.json) | plain loop + 4 streams, dense mixing, iteration embeddings | 330 | 62.75 / 76.63 |
| 3.32 | [`paper/loop_ablations/loop_x6_ds_combo.json`](paper/loop_ablations/loop_x6_ds_combo.json) | ×6 + DS + 4 streams, dense mixing, iteration embeddings | 330 | 63.38 / 75.88 |
| 3.33 | [`paper/loop_ablations/d384_loop_x6_ds_dropout02.json`](paper/loop_ablations/d384_loop_x6_ds_dropout02.json) | d384 ×6 + DS, dropout 0.2 | 330 | 67.75 / 78.75 |
| 3.34 | [`paper/loop_ablations/d384_loop_x8_ds.json`](paper/loop_ablations/d384_loop_x8_ds.json) | d384 ×8 + DS | 330 | 67.88 / 78.50 |
| 3.35 | [`paper/loop_ablations/d384_12blocks_loop_x10_ds.json`](paper/loop_ablations/d384_12blocks_loop_x10_ds.json) | d384, 12 blocks, [4,8) ×10 + DS | 330 | 69.00 / 79.12 |
| 3.36 | [`paper/loop_ablations/d96_loop_x11_ds.json`](paper/loop_ablations/d96_loop_x11_ds.json) | d96 ×11 + DS, 4 heads, FFN 320 | 330 | 48.88 / 60.75 |
| 3.37 | [`paper/loop_ablations/d384_loop_x6_ds_continue_e440.json`](paper/loop_ablations/d384_loop_x6_ds_continue_e440.json) | 1.3 continued from epoch 330 to 440 | 330→440 | 69.50 / 79.38 |

### Kaggle models (ARC-AGI-2 task-family package with all public answers; pretraining only, no in-run score)

| Row | Spec | Model | Epochs | Top-2 / oracle |
| --- | --- | --- | --- | --- |
| K1a | [`paper/kaggle_d1024/pretrain_e270.json`](paper/kaggle_d1024/pretrain_e270.json) | d1024 ×6 + DS, 16 heads, FFN 2688, ≤4 demos, 8 GPUs | 270 | – |
| K1b | [`paper/kaggle_d1024/continue_e440_from_e270.json`](paper/kaggle_d1024/continue_e440_from_e270.json) | K1a continued at 0.8× learning rate | 270→440 | – |
| K2a | [`paper/kaggle_d768/pretrain_e180.json`](paper/kaggle_d768/pretrain_e180.json) | d768 ×6 + DS, 12 heads, ≤6 demos, 4 GPUs; stopped after epoch 80 | 1→80 of 180 | – |
| K2b | [`paper/kaggle_d768/continue_e180_from_e080.json`](paper/kaggle_d768/continue_e180_from_e080.json) | K2a continued on 8 GPUs | 80→180 | – |

### Notes

- **1.1, 3.9.** The oracle comes from a second decode of the same checkpoint (same top-2). With this
  code one decode writes both scores.
- **1.2.** Same recipe as 1.1 with seed 43, but run on a later, faster version of the code.
- **2.3 / 2.4.** The run was trained in two parts with the same effective batch of 64: epochs 1–130 at
  micro-batch 16 × 4 accumulation, epochs 131–330 at 32 × 2. The spec runs all 330 epochs at 16 × 4.
- **2.5.** The spec ranks the 2.4 candidates by likelihood (71.25 top-2; plain votes 70.50). The
  paper's 71.88 then applies the symbolic guards to the likelihood ranking:
  [`tools/guards_eval.py`](../tools/guards_eval.py) over the 2.4 candidates and this spec's
  `rescore.json` (its `ll_top20+guards` line; the docstring has the exact command).
- **2.6.** Likelihood ranking gives 70.25 (plain votes 68.13); `tools/guards_eval.py` on the 1.3
  candidates and this spec's `rescore.json` gives the paper's 70.88. The original ranked candidates
  from a second decode of 1.3; the spec uses the candidates of the 1.3 spec's own `final` evaluation.
- **3.7.** 64.25 is the run's own decode. Three later decodes of the same checkpoint scored 63.50 to
  63.88 top-2, 74.50 to 75.50 oracle.
- **3.19.** The run's own decode ran out of memory on an 80 GB GPU. The score comes from a second
  decode with a prefill budget of 196,608 tokens, which the spec sets (`prefill_token_budget`).
- **3.23 / 3.24.** `eval_loop_iterations` decodes a looped checkpoint at another loop count; K′ = 6 is
  the training count (rows 1.1 and 1.3).
- **K2a / K2b.** Stop K2a after epoch 80 (it saves `tiny.epoch080.pt`), then run K2b on 8 GPUs with
  `MDLARC_RESUME_RESEED_RANKS=1`, which lets a resume change the number of GPUs. Epochs 81–180 then
  draw different per-GPU random streams, so the result is not bit-identical to the original run.
- **Kaggle scores.** K1 and K2 were scored on Kaggle after test-time fine-tuning in the submission
  notebooks; see [`kaggle_submissions/`](../kaggle_submissions/).

Results depend on GPU type and library versions: rerunning a spec gives the same settings, not
bit-identical numbers. The paper's two d256 seeds (1.1, 1.2) differ by 4 points top-2.
