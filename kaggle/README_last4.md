# 600M last-four-block test and competition submission on four L4s

Use these **new** notebooks; the original `submit_arc2_600m.ipynb` remains frozen:

- [validate_arc2_600m_val240_last4.ipynb](validate_arc2_600m_val240_last4.ipynb):
  the uploaded 240-task timing workload, with post-generation scores.
- [submit_arc2_600m_last4.ipynb](submit_arc2_600m_last4.ipynb):
  official competition runtime data and `submission.json`.

No Kaggle GPU execution or competition submission was performed during preparation.
Full L4 memory use, duration and accuracy for this recipe remain unmeasured.

## Five-epoch full-workload timing test

Use [time_arc2_600m_val240_last4_e5.ipynb](time_arc2_600m_val240_last4_e5.ipynb)
to measure full decoding after only **five fresh adaptation epochs**. Attach the
same three validation inputs below, choose **L4 × 4 / Internet off**, restart,
leave **`SMOKE_TEST=False`**, and **Run All**. No settings changes are needed.

It trains all240/726demo pairs with the same last-four-block 2/2 batch and LRs,
then decodes all336 queries at80 augmentations: **26,880 cases**. Decode remains
batch1/GPU, prefill4096, native four-head speculation **on**. Full decode includes
worker startup, load, compilation, generation, AAIVR and merge; answer scoring
is excluded from that timer. This measures the five-epoch checkpoint, whose
output lengths may differ from the 330-epoch adaptation.

Look for **TIMING BENCHMARK COMPLETE**, displayed train/decode time and aggregate
cases/second, and download **`timing240_benchmark.json`** plus
`timing240_run_info.json`, `timing240_predictions.json`, `timing240_scores.json`.
Logs/checkpoints are under `/kaggle/working/runs/last4_val240_timing_e5/`.
This timing notebook does not generate a top-level competition `submission.json`.
Rebuild with `venv/bin/python kaggle/make_notebook.py --profile 600m-val240-last4-timing5`.

## Inputs to attach

| Input | Validation notebook | Competition notebook |
| --- | --- | --- |
| [Original epoch-330 checkpoint](https://www.kaggle.com/datasets/mithilavakde/mdlarc-600m-epoch330) | Required | Required |
| [FlashAttention 2 wheel](https://www.kaggle.com/datasets/mithilavakde/flash-attn-cp313-torch211-cu128) | Required for default flash backend | Required for default flash backend |
| [Private timing240, version 1](https://www.kaggle.com/datasets/mithilavakde/mdlarc-arc2-timing240-v1/versions/1) | Required | Do not attach |
| [ARC Prize 2026 ARC-AGI-2 competition data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data) | Unnecessary | Required; accept competition rules first |

Attach only this model checkpoint dataset: other `pretrain__tiny.pt` / `tiny.pt`
inputs make discovery ambiguous. Leave `CHECKPOINT_PATH=None`,
`TEST_CHALLENGES=None`, `ATTENTION_BACKEND="flash"`, `FLASH_ATTN_WHEEL=None`
and `ENABLE_COMPILE=True`. Paths are discovered under `/kaggle/input/`.
The environment cell checks the actual Python, PyTorch, CUDA and C++ ABI before
loading the attached wheel. A future Kaggle image change may require a compatible
replacement wheel; native attention is an explicit runtime change.
The uploaded data README preserves an earlier preparation proposal; the notebooks
and recipe table here define this requested last-four-block 2/2 run.

## Recipe in both notebooks

| Setting | Value |
| --- | ---: |
| Initial weights | Original pretraining epoch 330, fresh adaptation state |
| Trainable blocks, zero based | 20, 21, 22, 23 only |
| Full adaptation epochs | 330 fresh epochs |
| GPU processes | 4 synchronized DDP training ranks |
| Training microbatch per GPU | 2 |
| Gradient accumulation per GPU | 2 |
| Effective batch | 2 × 2 × 4 = 16 |
| NorMuon LR | 0.0003 |
| Auxiliary AdamW LR | 0.00028 |
| Training augmentations | 80 |
| Full inference augmentations | 80 |
| Inference workers | 4 independent processes, complete tasks sharded |
| Inference batch per GPU | 1 |
| Prefill token budget | 4096 |
| Native speculative decoding | Enabled, four prediction heads, greedy |
| Data-loader workers per training rank | 0 |
| Attention / compiler | FlashAttention 2 / torch.compile enabled |

Everything outside blocks 20–23 stays frozen, including token embeddings, final norm
and heads. Parameters disabled by the architecture stay disabled even inside
selected blocks. Optimizer groups are checked against the exact trainable set.
This checkpoint has `task_embeddings=False`; task conditioning remains disabled
during adaptation and decoding. Inactive task tables retained in its checkpoint
format are neither used nor trained.
Frozen blocks still execute. Each L4 has a separate model replica and memory;
the four cards do not provide one shared 96-GB allocation.

The notebooks preserve the original 24-layer, width-1536, 24-head, FF-4096,
four-prediction-head architecture, pair-group two-lookup attention, context-palette
augmentation, all-demo training schedule, leave-one-out inference contexts and
other optimizer/schedule defaults. The packaged model source is preserved byte
for byte from `submit_arc2_600m.ipynb`, whose SHA-256 is
`71ad055eb93ce682d470d695d95dd98bacbd375a2799774b08c419aea10a23ba`.
Only Kaggle control helpers change. The original pretrained checkpoint is checked
by **size 5,726,752,055 bytes**, **epoch 330**, and full SHA-256
`a17e48c09c7d4c99755faa4a5eaa02d96cda88f137b1471194d1ffd20b661582`
before it is trusted and loaded.

## 1. Smoke test the validation notebook

1. On Kaggle create a notebook and **File → Import Notebook → Upload**
   `validate_arc2_600m_val240_last4.ipynb`.
2. Use **Add Input** to attach the three validation inputs in the table.
3. Set **Accelerator: GPU L4 × 4**, **Internet: off**. All four GPUs must be visible.
4. Keep **`SMOKE_TEST = True`**, restart the session, and **Run All**.
5. Observe `MDLARC_STARTUP_EVENT first_optimizer_step`, then
   `Epoch 1/1 compute completed`, `CHECKPOINT SAVED AND RELOADED`, and finally
   **`SMOKE TEST PASSED`**. Liveness messages during compilation alone do not
   establish optimizer progress.

Smoke does **exactly one real training epoch over all 240 tasks / 726 demo pairs**
with 80 training augmentations, microbatch 2 and accumulation 2 on every rank.
It then decodes the **four tasks with largest demo-plus-query input contexts**
(IDs and query count printed) with **two inference augmentations**, batch 1 per
GPU. All queries of each selected task are decoded. This tests large contexts
and every inference worker while bounding decode work. It does not measure
full 80-augmentation decode cost or full-dataset accuracy.

For pinned timing240 version 1, the four smoke task IDs are `981571dc`,
`4ef15c79`, `b4b304c8`, `d8e07eb2`: two originals and two compositions,
seven queries total, hence 14 augmented query decodes.

Success requires completed optimizer steps on every rank, the requested epoch,
checkpoint save/reload, four-worker decoding, strict coverage/merge and no
malformed-attempt replacements. Per-rank JSON and notebook output report actual
optimizer-update counts and peak allocated/reserved VRAM for training and decode.
Model-only inference checkpoints avoid loading unused optimizer state four times.

Expected smoke files under `/kaggle/working/`:

- `smoke_validation_predictions.json`
- `smoke_validation_scores.json`
- `smoke_validation_run_info.json`
- `runs/smoke_last4_val240/` with `train-ddp.log`, `decode-gpu*.log`,
  `train-*.json`, `decode-*.json`, `request.json`, `finetuned.pt`, `inference.pt`.

## 2. Test all 240 tasks

1. After smoke passes, **restart the session** and set **`SMOKE_TEST = False`**.
2. Keep the full recipe above, then **Save Version → Save & Run All**.
   The notebook loads the original verified pretrained checkpoint and creates a
   fresh optimizer/schedule; it never resumes a smoke or previously adapted model.
3. Wait for **`VALIDATION COMPLETE`** and the original40 / synthetic200 scores.
4. Download `validation_predictions.json`, `validation_scores.json` and
   `validation_run_info.json` from that version's Output tab.

Full mode trains all **240 tasks**, **330 epochs**, **80 training augmentations**,
then decodes all **336 queries × 80 augmentations = 26,880 augmented cases**.
The 240 tasks comprise 40 unchanged held-out originals plus 200 compositions
from 200 distinct families excluded by the exact checkpoint's saved pretraining
catalog. Both direct parents are in original40; historical parent-generator
authoring context includes additional originals, disclosed in `timing_manifest.json`.
This is a timing workload, not 240 independent original ARC puzzles.

`validation_scores.json` reports **original40**, **synthetic200** and
**combined_timing_workload** separately. Top-1/top-2 accuracy averages query
correctness within each task, then averages across tasks; query-weighted counts
are included too. Workers receive cleaned challenge data only. The scoring parent
first opens query solutions **after every decode worker has exited and predictions
have merged**; answers never train the model or rank candidates.

## 3. Run the competition notebook and submit

1. Create a separate Kaggle notebook and import
   **`submit_arc2_600m_last4.ipynb`**.
2. Attach the competition data, original epoch-330 model and FA2 wheel inputs.
   Leave timing240 and its solutions out of this notebook.
3. Set **GPU L4 × 4**, **Internet off**, keep **`SMOKE_TEST=True`**, restart,
   and **Run All**. The same one-epoch 2/2 recipe trains on **all official runtime
   tasks' demonstration pairs**, then decodes a four-task large-context subset.
4. Wait for **`SMOKE TEST PASSED`**. Its files are `smoke_submission.json`,
   `smoke_submission_run_info.json`, and `runs/smoke_last4_competition/`.
   **Do not submit `smoke_submission.json`.**
5. Set **`SMOKE_TEST=False`**, keep the full recipe, and open the notebook
   editor's **competition / Submit to Competition panel → Submit**. Specify
   **`submission.json`** as the output filename and confirm submission.
   This saves the code version and starts the code-submission workflow; a
   separate completed interactive or saved full run is not required first.
6. Check the competition's **Submissions** page for the submitted version's
   queued/running entry and eventual result. Kaggle runs the submitted code with
   hidden runtime data; it must finish within the competition's execution limits
   and generate `/kaggle/working/submission.json`.
7. A separate **Save Version → Save & Run All** followed by submission from the
   completed version is an optional testing route. It is not a prerequisite for
   the editor's direct submission route. Never use validation predictions here.

[Kaggle's staff announcement](https://www.kaggle.com/discussions/product-feedback/269953)
explains that saving a version and submitting it proceed concurrently for code
competitions. The output filename tells Kaggle which file to score after execution.

The competition input guard accepts only the competition mount's official test
challenge file. Only demonstration pairs supply adaptation targets; query outputs
are stripped even from an interactive placeholder. All official queries must
appear in the final validated JSON with two attempts in original query order.
The hidden rerun uses the official runtime data; interactive placeholder timing
does not establish its duration. Consult the current
[competition code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements).

The inherited **eight-hour adaptation ETA guard** estimates after three epochs
and stops if training exceeds that budget; decoding needs additional time.
It never silently reduces epochs/batches or retries training. A prefill budget
limits batching, not an individual prompt's length or memory. Retain traceback,
worker logs and memory reports if smoke fails; full workload fit and duration
remain for the user's Kaggle test. No full-runtime estimate is claimed here.

## Rebuild and bounded CPU verification

```bash
venv/bin/python kaggle/make_notebook.py --profile 600m-last4
venv/bin/python kaggle/make_notebook.py --profile 600m-val240-last4
venv/bin/python -m pytest -q tests/test_kaggle_last4.py tests/test_kaggle_submission.py tests/test_kaggle_holdout.py tests/test_kaggle_multi_gpu.py
```

Both new profiles require the pinned original notebook and reject model-runtime
refresh. Keep reusable controls/profiles/tests/docs and the two notebooks in Git.
Keep upload staging, downloaded verification files, receipts, extracted sources,
logs and research records ignored under the current branch's `results/` directory.
