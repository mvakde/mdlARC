# Test the 600M model on the 40-task holdout

Use [validate_arc2_600m_val40.ipynb](validate_arc2_600m_val40.ipynb).
The private [holdout dataset](https://www.kaggle.com/datasets/mithilavakde/mdlarc-arc2-dev-val40-v1)
contains the exact `arc2_dev80_val40_v1` selection: **40 tasks, 121 demonstration
pairs, and 56 query inputs**. Version 1's privacy, four-file inventory and full
file checksums were verified after processing on 2026-10-27.

## Run on Kaggle

1. Import the validation notebook and attach the private holdout dataset.
2. Keep the [epoch-330 checkpoint dataset](https://www.kaggle.com/datasets/mithilavakde/mdlarc-600m-epoch330)
   and [FlashAttention wheel dataset](https://www.kaggle.com/datasets/mithilavakde/flash-attn-cp313-torch211-cu128)
   attached. The competition dataset is unnecessary for validation.
3. Select **L4 × 4**, **Internet off**, and restart the session after importing.
4. Leave **`SMOKE_TEST = False`** to run all 40 tasks, then Run All.

Defaults match the current submission recipe: 90 adaptation epochs, only blocks
1/2/23/24 trainable, microbatch one per GPU with accumulation ten (effective
batch 40), the same learning rates, and 80 train/inference augmentations.
Inference starts at batch one per GPU. To test a larger inference batch, change
`EVAL_BATCH_SIZE` explicitly and record it when comparing timings.

Training output is relayed on the notebook thread and logs every optimizer update
(`TRAIN_LOG_MODE = "step"`). Startup messages identify runtime import, DDP setup,
checkpoint loading, model/data construction and the backward preflight. The marker
`MDLARC_STARTUP_EVENT first_optimizer_step` appears after the first optimizer update
finishes; first-use compilation can delay it. During silence, a message every 30
seconds reports process liveness only. Raw training output is also saved to
`/kaggle/working/runs/kaggle_600m_val40/train-ddp.log`; decode workers have separate
`decode-gpu*.log` files in the same folder. These logging changes require importing
the updated notebook; they do not change an already-running cell.

The challenge file contains demonstration inputs/outputs and input-only queries.
Workers receive only that cleaned challenge data. Query answers are opened in
the notebook parent only after decoding and prediction merging have completed.
The scoring file is never used to update weights or rank predictions.

## Results

The final cell prints **VALIDATION COMPLETE**, top-1/top-2 task-weighted scores,
adaptation time and parallel decoding time. Files in `/kaggle/working/`:

- `validation_predictions.json`: two attempts per query, in original query order.
- `validation_scores.json`: task-weighted scores, query-weighted counts and per-task breakdown.
- `validation_run_info.json`: settings, phase times and per-GPU memory/epoch timing.

Task-weighted scores average correctness across each task's queries and then
across the 40 tasks, matching the earlier development evaluation. These are
development scores. Use [submit_arc2_600m.ipynb](submit_arc2_600m.ipynb) for the
competition submission; validation outputs are named separately.

This holdout has 56 queries. Project decoding cost using query/candidate counts
and output lengths, not simply a sixfold task-count multiplier for 240 tasks.
Training time also depends on the number and size of demonstration pairs.

## Rebuild

```bash
venv/bin/python kaggle/make_notebook.py --profile 600m-val40
```

If creating this artifact for the first time, select the preserved submission
runtime explicitly with `--runtime-notebook kaggle/submit_arc2_600m.ipynb`.
Model source stays byte-identical to the submission notebook; only validation
controls and Kaggle helpers differ. Reusable builder/profile/helper/notebook,
documentation and tests belong in Git. Exported puzzle JSON, upload receipts
and verification records stay in ignored results; Kaggle version 1 is the
uploaded data artifact.
