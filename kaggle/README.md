# Submit the ARC2 public/composition checkpoint

Use **[submit_arc2.ipynb](submit_arc2.ipynb)**. It contains its own source code, uses
one GPU, finetunes every parameter, and writes `/kaggle/working/submission.json`.
You do not need to upload the repository or any of the pretraining datasets.

## 1. Use the uploaded private checkpoint

The checkpoint is available in the private Kaggle dataset
**[mdlARC ARC2 Public v3 Epoch 25](https://www.kaggle.com/datasets/mithilavakde/mdlarc-arc2-public-v3-epoch25)**,
owned by `mithilavakde`. Version 1 has finished processing and contains only
`pretrain__tiny.pt`. The transfer ran entirely on a Colab CPU VM; the checkpoint
was never downloaded or relayed through the Mac.

The original [Drive checkpoint](https://drive.google.com/file/d/DRIVE_ID_01_PLACEHOLDER/view)
and [completed run folder](https://drive.google.com/drive/folders/DRIVE_ID_02_PLACEHOLDER)
are preserved.

This is the completed 25-epoch run `vast-54994367-public-v3-b112-e25-bw0` on branch
`codex/self-speculative-decoding`. Its recipe includes ARC2 train **and public
evaluation** originals/variations, ConceptARC, filtered ARC1, and v3 compositions
from ARC2 public evaluation. The recovered completion manifest and backup ledger
record success and verified backups. The notebook verifies the final file:

```text
Size:   4,364,936,923 bytes (about 4.36 GB)
SHA256: d5912ada04df6010369a59a7c5b6043261411825f95d5f3c304ad4461068a0dd
```

On 2026-10-10, both the Drive download and the processed Kaggle copy passed
these size and SHA-256 checks on the remote CPU VM. Kaggle private visibility
and the single-file inventory were also verified.

Leave `CHECKPOINT_PATH = None` to find the attached `pretrain__tiny.pt`
automatically. The notebook retains its size, SHA-256, and saved-epoch checks.
If you attach multiple matching checkpoints, set `CHECKPOINT_PATH` to this
dataset's exact path shown in the notebook's Input panel.

## 2. Create the competition notebook

1. Open [ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2).
   Join the competition if you have not already joined.
2. Create a new notebook from the competition's **Code** tab. In the notebook's
   **File → Import Notebook** menu, import `kaggle/submit_arc2.ipynb` from this checkout.
   If your interface labels this **Upload Notebook**, use that option.
3. Sign in as `mithilavakde`. Under **Add Input**, search your datasets for
   `mdlarc-arc2-public-v3-epoch25` and attach
   **[mdlARC ARC2 Public v3 Epoch 25](https://www.kaggle.com/datasets/mithilavakde/mdlarc-arc2-public-v3-epoch25)**.
   Keep the dataset private; it is available to notebooks in its owner's account.
4. Make sure the competition data is attached too. The notebook needs
   `arc-agi_test_challenges.json` from the competition mount. It supports both
   `/kaggle/input/competitions/<competition>/...` and the older mount layout.
5. Under notebook **Settings**, choose **GPU L4 × 4** and set **Internet off**.
   The code makes only GPU 0 visible to PyTorch and creates no DDP processes.

The current competition requirements are: notebook submission, at most **12
hours**, Internet disabled, and a file named `submission.json`. Its L4 pool is
presented as L4 × 4; selecting that machine is compatible with this single-GPU
notebook. [Kaggle code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements).

## 3. Check the environment and choose your attention wheel

Run the **settings** and **environment** cells first. The environment cell prints
Python, PyTorch, PyTorch's CUDA build, C++ ABI, and the GPU. With the default
`ATTENTION_BACKEND="flash"`, it may then stop because a compatible wheel is not
attached yet; that is the point to choose your wheel using the printed versions.

For FlashAttention:

1. Obtain/build a **FlashAttention 2** Linux wheel for that environment. As you
   mentioned, you can handle updating the wheel yourself. The old wheel's Python
   tag alone does not establish compatibility with the current PyTorch/CUDA ABI.
2. Upload it as a Kaggle dataset and attach that dataset to this notebook.
3. Set `FLASH_ATTN_WHEEL` to the exact attached `.whl` path, or leave it `None`
   if there is only one wheel matching the current Python tag.
4. Restart the session and Run All. Installation is offline, with `--no-index
   --no-deps`, and does not replace Kaggle's PyTorch. Any additional dependencies
   reported by pip/import must also be supplied offline.

The notebook probes the import in a fresh Python process, then tests an actual
model forward/backward pass on CUDA before finetuning. An incompatible wheel
causes an error; it does not silently select another backend. A wheel update alone
cannot guarantee that every compiler/kernel combination works, so the smoke run
below also exercises checkpoint saving and compiled decoding.

To check the workflow without a wheel, explicitly set:

```python
ATTENTION_BACKEND = "native"
```

This uses the model's existing PyTorch attention path: SDPA for training and
PyTorch FlexAttention/SDPA for decoding. It preserves the bidirectional input
prefix and causal output attention. It can require more memory and time than
FlashAttention, so time it before a full submission.

## 4. Run a small smoke test

1. Set `SMOKE_TEST = True` in the first cell.
2. Restart the session and **Run All**.
3. Wait for **SMOKE TEST PASSED**. It checks two runtime tasks, two finetune
   epochs, and two inference augmentations. It should create
   `/kaggle/working/smoke_submission.json`.

Smoke mode intentionally writes a different filename. **Do not submit the smoke
output.** This checks execution and JSON coverage, without measuring useful
accuracy or proving the full hidden-task workload fits the time limit.

If the GPU runs out of memory, lower the **training microbatch** and raise
accumulation to preserve the effective batch. Examples:

| Training microbatch | Accumulation | Effective batch |
| ---: | ---: | ---: |
| 8 | 4 | 32 |
| 4 | 8 | 32 |
| 2 | 16 | 32 |

For inference OOM, lower `EVAL_BATCH_SIZE` to 4 or 2. If multiprocessing fails,
set `NUM_WORKERS = 0`. If an error originates in TorchDynamo/Inductor compilation,
try `ENABLE_COMPILE = False` in a fresh session, then recheck runtime.

## 5. Save the full version and submit it

1. Set **`SMOKE_TEST = False`**. Leave test-file discovery enabled;
   do not replace it with a public evaluation file.
2. Restart the session. Use the default settings initially:
   **330 epochs**, effective batch **32**, and **80** train/inference augmentations.
3. Click **Save Version → Save & Run All**. This executes a clean saved version.
   Wait for it to succeed. The last cell must print:
   `READY: /kaggle/working/submission.json`.
4. Open that saved version's output and verify `submission.json` exists.
5. Click **Submit to Competition**, select the successful version and
   **`submission.json`**, and confirm the submission.
6. Monitor the competition's **Submissions** page for the scored rerun. Kaggle
   reruns the notebook with scoring tasks, so the interactive placeholder output
   does not establish that the hidden run will fit the deadline.

Kaggle currently describes **240 unseen scoring tasks**. The test file visible
during development is a public-evaluation placeholder, so its workload and
apparent accuracy can differ from the scored run.
[Kaggle data description](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data).

The notebook requires exact task coverage, the correct number of test inputs in
input order, and two rectangular grids with colors 0–9 for each test input.
Malformed decoded attempts are replaced with the other valid attempt, or the
test input grid if both attempts are malformed; replacements are reported in the
logs and `submission_run_info.json`. Missing tasks/pairs cause an error.
These are the competition's required submission structure and ordering.
[Kaggle evaluation format](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/evaluation).

Finetuning prints an ETA after compilation warmup. It stops early with an
actionable error if the estimated finetune exceeds `MAX_FINETUNE_HOURS=8`, leaving
room within 12 hours for setup and decoding. That estimate is based on training;
it does **not** measure the decode budget. If needed, restart with fewer epochs
or fewer inference augmentations after timing a run. Full L4 runtime and peak
memory have not been validated from this Mac.

## What changed from the old submission cell

- **All parameters trainable.** No freeze mode; transformer, shared latent
  tables, task factors, norms, output heads, and MTP heads participate in training.
- **One GPU / one process.** No torchrun, NCCL setup, DDP, or distributed inference.
  Gradient accumulation gives an effective batch of 32 using a microbatch of 8.
- **Checkpoint architecture and training format preserved.** This checkpoint
  uses `single_pair_bidirectional_v1` with `single_grid_prefix` augmentations,
  width 768, eight layers, 3072 task latents, and four prediction heads. The old
  1B model settings and conditioning-only learning rates are not used.
- **Fresh optimizer/schedule.** The default finetune NorMuon LR is `0.0003` and
  AdamW LR is `0.003`, from the repository's ARC2 finetune template. The saved
  dropout, MTP configuration, optimizer multipliers and augmentation semantics
  are preserved. These settings have not been tuned for this checkpoint on L4.
- **Old task rows removed.** Pretraining has 99,201 task-factor rows, about 304.7M
  parameters. They and their optimizer state are unnecessary for new competition
  tasks. Only fresh runtime factors go onto the GPU; the learned shared latent
  tables remain intact. The model has about **87.4M shared parameters**, including
  the shared task/value latent tables, plus `3072 × runtime task count` factors.
- **Source embedded and verified.** The notebook includes the current runtime
  with the fix that preserves latent width when finetuning on fewer tasks. It
  needs no source dataset, Drive credential, Colab launcher, or pretraining archive.
- **Same challenge file for finetuning and decoding.** Only its train outputs are
  used for updates; test outputs are stripped and no solution file is read.

DDP averages gradients; it does not inherently degrade them. The effective batch,
loss weighting, and optimizer schedule can change when switching configurations.
This notebook uses your requested single-GPU workflow.
[PyTorch DDP documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

Because public ARC2 evaluation answers were included in pretraining, public-eval
scores would reflect training overlap. The Kaggle test rerun is the relevant
assessment for this checkpoint. The competition also has external-data and
open-source eligibility requirements; see its
[rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules).

## Maintaining this notebook

The upload artifact is `submit_arc2.ipynb`. `submission_utils.py` contains the
Kaggle-specific checks and adaptation; `make_notebook.py` embeds it and the eight
required source modules. After an intentional source/helper change, regenerate:

```bash
venv/bin/python kaggle/make_notebook.py
```

Re-upload the notebook and repeat the smoke test after regeneration. Rebuilding
uses the current checkout's source; it does not change the expected checkpoint
identity or fetch/update any wheel.

Local validation: **113 tests passed, 1 skipped**, across the submission helpers,
single-pair format, MTP, and attention backend tests. The integration test uses a
small synthetic checkpoint on CPU with a test optimizer; actual NorMuon/CUDA,
the full checkpoint, wheel compatibility, and L4 performance are checked by the
Kaggle preflight/smoke run rather than claimed as locally verified.
