# Submit the scaling-ladder epoch-330 checkpoint

For a timing/accuracy check on the 40 held-out development tasks, use the
separate [validation notebook and private dataset](README_val40.md).

Import [submit_arc2_600m.ipynb](submit_arc2_600m.ipynb) into a new private Kaggle
notebook for [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).
This is the user's “600M” model from branch `ddp-scaling-ladder`, session
`vast-ddp8-b200-b24a1-lr061-e70-noon-20261025-r5`.
It uses the final **pretraining epoch 330** checkpoint, followed by **90 fresh
adaptation epochs** on the scoring tasks' provided demonstrations. It does not
start from a checkpoint already adapted to the 40 development-validation tasks.
The notebook uses **all four L4s**: synchronized DDP adaptation, then four
independent inference workers over disjoint tasks.
Only transformer blocks **1, 2, 23 and 24** are trainable. All other blocks,
embeddings, output heads and the final norm are frozen. The visible settings
`TRAIN_FIRST_LAYERS = 2` and `TRAIN_LAST_LAYERS = 2` control this selection.

## Attach inputs

Attach these three inputs:

1. The private [600M epoch-330 checkpoint dataset](https://www.kaggle.com/datasets/mithilavakde/mdlarc-600m-epoch330).
   Version 1 is processed and contains only `pretrain__tiny.pt`, with its size
   and private visibility verified on 2026-10-26.
2. Your existing [FlashAttention wheel dataset](https://www.kaggle.com/datasets/mithilavakde/flash-attn-cp313-torch211-cu128).
   Keep `wheel-manifest.json` and the `einops` wheel attached. The notebook
   verifies the build checksum and restores the `+` in the wheel filename
   automatically. It prints the current Python/PyTorch/CUDA/ABI and checks import
   compatibility; a previous successful notebook does not verify a new image.
3. The competition data. Keep `TEST_CHALLENGES = None`; discovery selects the
   competition's `arc-agi_test_challenges.json` for both adaptation and decoding.

Attach only this model checkpoint dataset. The smaller epoch-25 checkpoint has
the same filename and would make automatic discovery ambiguous.

The pinned checkpoint identity is:

```text
Size:   5,726,752,055 bytes
SHA256: a17e48c09c7d4c99755faa4a5eaa02d96cda88f137b1471194d1ffd20b661582
Epoch:  330
```

The processed Kaggle copy was streamed back and verified against this full
SHA-256 on a CPU Colab VM on 2026-10-26. Its version-1 inventory has one file.

The [Drive checkpoint](https://drive.google.com/file/d/DRIVE_ID_03_PLACEHOLDER/view)
and [canonical training run](https://drive.google.com/drive/folders/DRIVE_ID_04_PLACEHOLDER)
remain preserved. Transfer is CPU VM → Kaggle, with no checkpoint relay through
the Mac. File size and SHA-256 are checked before loading its trusted pickle.

## Run the smoke version

1. In Kaggle Settings choose **GPU L4 × 4** and **Internet off**. All four GPUs
   must be visible. DDP keeps one complete model on each GPU.
2. Import `submit_arc2_600m.ipynb` and attach the inputs above. Leave
   `ATTENTION_BACKEND = "flash"` and `FLASH_ATTN_WHEEL = None` for discovery.
3. Leave the new default `SMOKE_TEST = True`, restart the session, and Run All.
4. Wait for `SMOKE TEST PASSED`. Its output is `smoke_submission.json` and is
   deliberately unsuitable for a full competition submission.

The smoke run selects four tasks with large demonstration/input contexts, uses
two adaptation epochs and two inference augmentations, and exercises all four
DDP ranks with the same four-block freeze policy as full mode. Training uses
microbatch one per GPU and accumulation one in smoke
mode (effective batch four). After training workers exit, four fresh inference
workers each decode a task with batch one. It checks selected-block backward,
optimizer updates, final checkpoint save/reload, generation and JSON merging.
It reports per-GPU peak allocated/reserved memory in notebook output and
`submission_run_info.json`. This does not establish full-workload speed or accuracy.

## Run the full version and submit

1. Set `SMOKE_TEST = False` and restart the session.
2. Leave the selected recipe at:

   | Setting | Value |
   | --- | ---: |
   | Adaptation epochs | 90 |
   | First trainable transformer blocks | 2 |
   | Last trainable transformer blocks | 2 |
   | GPUs | 4 |
   | Training microbatch per GPU | 1 |
   | Accumulation | 10 |
   | Effective batch | 40 |
   | NorMuon LR | 0.0016 |
   | AdamW LR | 0.00028 |
   | Training augmentations | 80 |
   | Inference augmentations | 80 |
   | Inference batch per GPU | 1 |
   | Training data-loader workers per GPU | 0 |

3. Select **Save Version → Save & Run All**. Wait for successful completion and
   `READY: /kaggle/working/submission.json`.
4. From that saved version, select **Submit to Competition** and submit
   **`submission.json`**. Check the competition's Submissions page for the
   hidden-task rerun and score. The notebook's interactive test file is a
   placeholder; its timing does not guarantee the hidden rerun's timing.

The notebook preserves width 1536, 24 layers, 24 heads, feed-forward width 4096,
four prediction heads, the checkpoint's task-embedding setting, pair-group
two-lookup attention, and context-palette augmentation. Adaptation starts a fresh
optimizer containing only parameters in the first two and last two blocks.
For a demonstration target, the other demos
provide context. Query decoding uses balanced leave-one-out demonstration
contexts, as in the completed 600M development adaptation. Only train pairs
provide adaptation targets; test outputs are stripped and no solutions file is
loaded. Private-task factors start fresh, and shared weights are retained.

The historical full-model 90-epoch development adaptation scored 32.08% top-2
on 40 held-out public tasks; 330 epochs scored 28.33%. These are development
scores for the earlier full-model recipe. The new four-block recipe has no
measured Kaggle or development accuracy yet.
The move from two B200 GPUs to four L4s with accumulation preserves the nominal
effective batch; it is not a claim of numerical or runtime equivalence.

The model was pretrained on the development data split, with 40 public
evaluation families withheld. This notebook submits the completed model as
requested; it does not retrain it on the full public corpus.

## Memory and runtime

Training prints startup stages and every optimizer update (`TRAIN_LOG_MODE="step"`).
The `MDLARC_STARTUP_EVENT first_optimizer_step` marker confirms the first update
has finished. First-use compilation can take time before that marker appears.
Output is relayed on the notebook thread; messages every 30 seconds during silence
report only worker liveness, not training progress. Raw worker logs remain under
`/kaggle/working/runs/kaggle_600m_ddp/` (or the smoke run folder).

Training starts at microbatch one per GPU. DDP synchronizes gradients and the
existing global token loss normalization; full effective batch is
`1 × 4 × 10 = 40`. Training workers exit completely before inference, releasing
optimizer, activations, compiler and CUDA allocations.

Freezing the middle blocks removes their parameter gradients and optimizer
updates. They still pass gradients back to the first two blocks; they are not
detached or placed inside `no_grad`. All 24 blocks still run during inference,
so freezing does not speed up decoding. The actual training speedup must be
timed on Kaggle. Training logs and per-rank JSON report the selected zero-based
block indices, trainable parameter count and fraction.

Inference launches four independent processes, each with one visible GPU.
Each receives complete tasks and keeps all 80 augmentations for every assigned
query together, preserving AAIVR aggregation. Their disjoint predictions merge
into one validated submission. Inference starts at batch one per GPU; increase
`EVAL_BATCH_SIZE` only after observing memory use. A model-only inference
checkpoint avoids loading four copies of unused optimizer state into host RAM.

Each L4 still holds a complete model. If microbatch-one training fails, retain
the error and memory report; increasing accumulation cannot reduce per-example
activation memory. Worker errors stop the stage and terminate sibling workers.
There are no automatic training restarts or retries.

The notebook reports a training ETA after warmup and rejects a training estimate
over its eight-hour budget. Decoding takes additional time. The full run's peak
memory and completion within the competition budget remain unverified until
tested on Kaggle. Consult the current [code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)
before submitting.

Worker logs and memory summaries are under
`/kaggle/working/runs/smoke_600m_ddp/` or `kaggle_600m_ddp/`. The final
`submission_run_info.json` includes all four training and inference memory
reports. Keep a failed run's displayed traceback and corresponding worker log
when reporting a Kaggle issue.

## Rebuild

```bash
venv/bin/python kaggle/make_notebook.py --profile 600m-e330
venv/bin/python -m pytest -q tests/test_kaggle_submission.py tests/test_kaggle_multi_gpu.py tests/test_distributed_training.py
```

For the four-block update, 32 package checks and 11 worker checks against the
exact packaged model runtime passed. These include a real two-process Gloo
adaptation/save followed by task-sharded decoding and strict submission merge.
The reduced six-layer test verifies that all four selected blocks update while
middle-block weights, embeddings, final norm and heads remain bitwise unchanged.
It also verifies gradients reach the first block through the frozen middle.
Full-model CUDA, NCCL, FA2 compilation, and L4 memory/runtime are for the
user's Kaggle test. No Kaggle GPU run was started while preparing this version.

The 600M builder now preserves the model source already embedded in this
notebook and updates its Kaggle controls. This avoids importing model changes
from another checked-out branch. `--runtime-notebook <path>` explicitly selects
an existing notebook snapshot; `--refresh-runtime` deliberately replaces the
model implementation with current checkout sources and needs fresh validation.

The old `submit_arc2.ipynb` is a separate, frozen deliverable. Building this
profile leaves that notebook untouched. Keep reusable builder/helper/profile
source, this guide, tests, and the new notebook in Git. Keep transfer scripts,
downloaded previous notebooks, receipts, logs, and run records under ignored
`results/ddp-scaling-ladder/`.
