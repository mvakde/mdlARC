# mdlARC

ARC-AGI from scratch: small transformers trained only on ARC puzzles, with no language-model pretraining.

## Versions

| Version | Date | Result | Code | Post |
| --- | --- | --- | --- | --- |
| **v3: Looped mdlARC** | Nov 2026 | 37.08% on ARC-AGI-2 (Kaggle public score¹), no pretrained LLM; 71.88 / 81.50 top-2 / oracle on ARC-AGI-1 at 19M parameters | `main` (tag `v3.1-nov2026`; first release `v3-nov2026`) | [Looped mdlARC](https://mvakde.github.io/blog/looped-mdlarc/) |
| v2 | Mar 2026 | 44% on ARC-AGI-1 for 67 cents, 75M unlooped transformer | tag [`v2-mar2026`](https://github.com/mvakde/mdlARC/tree/v2-mar2026) | [44% on ARC-AGI-1 in 67 cents](https://mvakde.github.io/blog/44-on-arc-1/) |
| v1 | Dec 2025 | 27.5% on ARC-AGI-1, 28M transformer, ~$2 | tag [`v1-dec2025`](https://github.com/mvakde/mdlARC/tree/v1-dec2025) | [New Pareto Frontier on ARC-AGI](https://mvakde.github.io/blog/new-pareto-frontier-arc-agi/) |

¹ The Kaggle public score of the final submission (57908007). It was submitted before the deadline but scored after the leaderboard froze, so the official board shows the previous looped model at 35.69% (rank 141).

## v3: Looped mdlARC

This repository reproduces **"Looped mdlARC: ARC-AGI Without Language Models"**, our ARC Prize 2026
entry. Every result in the paper has a spec here. The training data for the Kaggle model can be
downloaded or regenerated from the released generators, and the submitted Kaggle notebooks are
included unchanged.

### Main results

Scores are top-2 / oracle (any of the candidates correct), in %. ARC-AGI-1 is the 400-task public
evaluation set and ARC-AGI-2 the 120-task public evaluation set. Each model is trained from scratch
on public puzzles only: the training tasks of the evaluated benchmark, the other benchmark's tasks
that do not overlap it, ConceptARC, and the demonstration pairs of the evaluation tasks. The test
outputs being scored are never seen in training. The ReARC rows add generated ReARC tasks, and the
Kaggle model trains on the generated task families in [`data_generation/`](data_generation/).

| Model | Parameters | ARC-AGI-1 | ARC-AGI-2 | Spec |
| --- | --- | --- | --- | --- |
| d384, middle blocks looped ×6 | 18.9M | 68.38 / 79.62 | 13.89 / 20.56 | [`arc1_d384x6`](specs/data_squeeze/arc1_d384x6.json), [`arc2_d384x6`](specs/data_squeeze/arc2_d384x6.json) |
| + scale augmentation, scale views, symbolic guards, likelihood ranking | 18.9M | **71.88 / 81.50** | | [`specs/paper/rearc_and_scale/`](specs/paper/rearc_and_scale/) |
| + ReARC training data | 18.9M | 81.38 / 88.00 | 20.14 / 32.08 | [`specs/paper/rearc_and_scale/`](specs/paper/rearc_and_scale/) |
| d256, looped ×6 | 8.4M | 66.25 / 77.25 | | [`arc1_d256x6`](specs/data_squeeze/arc1_d256x6.json) |
| d144, looped ×6 | 2.0M | 55.63 / 68.00 | | [`arc1_d144x6`](specs/data_squeeze/arc1_d144x6.json) |
| d96, looped ×6 | 1.05M | 49.00 / 60.62 | | [`arc1_d96x6`](specs/data_squeeze/arc1_d96x6.json) |
| d1024, looped ×6, Kaggle model (with test-time fine-tuning) | 99.8M | | 37.08 (Kaggle public score¹) | [`specs/paper/kaggle_d1024/`](specs/paper/kaggle_d1024/), [`kaggle_submissions/`](kaggle_submissions/) |

The two d384 ×6 specs also rank each output's 20 most-voted candidates by likelihood (the ARC Data
Squeeze record recipe): their `score.json` gives 70.25 on ARC-AGI-1 and 15.14 on ARC-AGI-2, and
`oracle_score.json` keeps the plain-vote scores above. The 71.88 row ranks the decoded candidates by
likelihood, then applies the symbolic guards with [`tools/guards_eval.py`](tools/guards_eval.py). The loop ablations, the unlooped depth ladder, the
sub-1M models and the other paper rows are listed with their specs and scores in
[`specs/README.md`](specs/README.md).

## Quick start

One NVIDIA GPU with FlashAttention 2 (A100, H100 or H200 class). The ARC-AGI-1 baseline below trains
the 8.4M-parameter d256 ×6 looped model from scratch, then decodes the evaluation set.

```bash
git clone https://github.com/mvakde/mdlARC.git && cd mdlARC
pip install -r requirements.txt                  # PyTorch + flash-attn 2 matching your CUDA
python dataset_building_scripts/download_and_group.py
python launch_run.py --config specs/data_squeeze/arc1_d256x6.json
```

`download_and_group.py` fetches ARC-AGI-1, ARC-AGI-2, ConceptARC and Re-ARC at pinned commits into
`assets_tmp/`. `launch_run.py` builds the dataset, trains, decodes and scores, all in one run. Add
`--preview` to print the resolved plan without running anything.

A run writes everything to `runs/<run_name>/` (set `MDLARC_RUNS_DIR` to put runs elsewhere):

```
runs/<run_name>/
  pretrain/tiny.pt              final checkpoint (plus kept checkpoints every 50 epochs)
  pretrain/training_log.txt
  evals/final/submission.json   two attempts per test input, Kaggle format
  evals/final/score.json        top-2 score
  evals/final/oracle_score.json any-candidate (oracle) score
  evals/final/candidates.json.gz  every de-augmented candidate with its votes
  timing.txt
```

The d256 run took about 3 hours on one H100. Expect roughly 62 to 66 top-2 and 76 to 77 oracle:
two runs of this recipe scored 66.25 / 77.25 and 62.00 / 76.12.

## Reproducing the paper

Each paper result has one spec, listed with its scores in [`specs/README.md`](specs/README.md):

- [`specs/data_squeeze/`](specs/data_squeeze/): ARC-AGI-1 and ARC-AGI-2 baselines from 0.71M to 18.9M
  parameters. These are also the reference runs for the
  [ARC Data Squeeze](https://github.com/mvakde/arc-data-squeeze) benchmark.
- [`specs/paper/rearc_and_scale/`](specs/paper/rearc_and_scale/): ReARC data, scale augmentation and
  views, symbolic guards and likelihood ranking.
- [`specs/paper/loop_ablations/`](specs/paper/loop_ablations/): loop count, deep supervision, input
  injection, width, the unlooped depth ladder, and decoding at another loop count.
- [`specs/paper/kaggle_d1024/`](specs/paper/kaggle_d1024/) and
  [`specs/paper/kaggle_d768/`](specs/paper/kaggle_d768/): the two models behind the Kaggle
  submissions.

Specs that only decode a checkpoint (for example likelihood ranking or another loop count) read the
checkpoint of the training spec they name, from `runs/<that run>/pretrain/tiny.pt`.

### The Kaggle model (37.08 on ARC-AGI-2)

1. **Data.** Download the training package (182 MB, 161,798 puzzles in 3,826 task families) from
   this repository's Releases page with `python data_generation/fetch.py`, or regenerate it from the
   released generators; see [`data_generation/`](data_generation/).
2. **Pretrain.** `specs/paper/kaggle_d1024/pretrain_e270.json`: 270 epochs from scratch on 8 GPUs
   (DDP, global batch 160). This took 8.6 hours on 8 × B300.
3. **Continue.** `specs/paper/kaggle_d1024/continue_e440_from_e270.json`: resume the epoch-270
   checkpoint with its optimizer state, re-warm the learning rates to 0.8×, train to epoch 440
   (5.4 hours on 8 × B300).
4. **Test-time fine-tune and decode.** [`kaggle_submissions/`](kaggle_submissions/) holds the
   submitted notebooks and explains their inputs and recipe.

The epoch-440 checkpoint (99.8M parameters) is published on Kaggle as the model variation
[`mithilavakde/mdlarc-loopv3-private/pyTorch/d1024-e440`](https://www.kaggle.com/models/mithilavakde/mdlarc-loopv3-private/pyTorch/d1024-e440).

Multi-GPU specs set `"distributed": {"nproc_per_node": N}`; `launch_run.py` then starts one process per
GPU on a single machine.

## How it works

Every puzzle is serialized as a token sequence: the demonstration pairs, then the test input, then
the output to predict. The model is a small decoder-only transformer trained from scratch on these
sequences.

- **Looped blocks.** The model has 8 blocks. Blocks 0–1 run once (prelude), blocks 2–5 run K times
  in a loop (K = 6 in the main models), and blocks 6–7 run once (coda). So 8 blocks of parameters
  give 28 layers of computation.
- **Input injection.** Each loop iteration adds the prelude's output back in, so later iterations
  still see the puzzle.
- **Deep supervision.** Extra output exits inside the loop are trained with weight 0.3, so every
  iteration learns to move towards the answer.
- **Multi-token prediction.** Four heads predict the next 1–4 tokens (weight 0.3).
- **Pair-group attention.** Each demonstration pair attends only to itself; the test input and
  output attend to all of them. Demonstration order therefore does not matter.
- **Context-palette augmentation.** Training and decoding see each puzzle under rotations,
  reflections and colour permutations chosen per context.
- **Leave-one-out decoding with voting.** At test time the model decodes each test input under many
  augmented views and demonstration subsets. Every candidate is mapped back to the original frame,
  and the two most-voted grids are the two attempts.

These options are off by default and switched on in the specs that use them:

| Option | Spec keys |
| --- | --- |
| Scale augmentation (train on up- and down-scaled grids) | `scale_augmentation` |
| Scale views at decode time | `inference_scale_views` |
| Symbolic guards (reject candidates that break simple invariants of the demonstrations; rules from transductARC) | `inference_symbolic_guards` |
| Likelihood ranking of the most-voted candidates | `inference_rescore` |
| ReARC generated tasks in training | `dataset.add_rearc` |

Likelihood ranking normally runs inside the decode (`inference_rescore`). An evaluation with
`"mode": "rescore"` and a `candidates_path` instead ranks the candidates saved by a finished decode
(rows 2.5 and 2.6 in [`specs/README.md`](specs/README.md) do this). It does not use the decode's
speed-ups and is several times slower, so use `inference_rescore` for new runs.

[`specs/README.md`](specs/README.md) lists every default and how it differs from earlier versions of
this code.

## Repository layout

```
launch_run.py              one entry point: build data -> train -> decode -> score
src/                       model, training (single GPU and DDP), decoding, scoring
dataset_building_scripts/  ARC-AGI-1/2, ConceptARC and ReARC downloads at pinned commits; dataset builds
data_generation/           generators, sampler, validator and packager for the Kaggle model's training data
specs/                     data_squeeze/ baselines; paper/ one spec per paper result
kaggle_submissions/        the submitted Kaggle notebooks, their wheel build and a runtime extractor
tools/                     guard re-ranking of saved candidates; decode batch sizing for your GPU
visualiser/                browser viewer for tasks and model outputs
tests/                     unit tests (CPU): python -m pytest -q tests
```

## Built on

- [ARC-AGI-1](https://github.com/fchollet/ARC-AGI) and [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2):
  the tasks and evaluation sets.
- [ConceptARC](https://github.com/victorvikram/ConceptARC) and
  [Re-ARC](https://github.com/michaelhodel/re-arc): extra training tasks, downloaded at pinned commits.
- transductARC: the rules of the symbolic guards.
- [NVARC](https://github.com/1ytic/NVARC): the section format of the task summaries used as examples in the generator prompts
  ([`data_generation/authoring/`](data_generation/authoring/)).

## Links

- Paper: <!-- TODO(paper link) -->
- Blog post: https://mvakde.github.io/blog/looped-mdlarc/
- Model weights (99.8M, the 37.08% Kaggle model): https://www.kaggle.com/models/mithilavakde/mdlarc-loopv3-private/pyTorch/d1024-e440
- ARC Data Squeeze benchmark: https://github.com/mvakde/arc-data-squeeze

<!-- TODO(licence): decided last, with the user -->
