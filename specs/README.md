# Experiment specs

`ddp-scaling-ladder` adds [single-node DDP training](README_ddp.md), including
[the two-demo DDP recipe](arc2_family_development_two_demo_ddp.json) with two
GPUs, per-GPU batch 11, accumulation 3, and effective batch 66.

On `scaling-ladder`, `pretrain_eval.json` is the optimized **Context color +
schedule + two lookup** recipe. `python launch_run.py --preview` resolves it
without training. `example.json` has the same settings apart from run name.
The defaults use `FA=4`, `attention_dropout=0`, and
`inference_speculative="auto"`; embedding/feedforward dropout stay 0.1.
`context_palette_schedule_two_lookup.json` retains the historical FA2/dropout-0.1
reference, which scored **215/400 (53.75%)**. The optimized default has no new
measured result. See the root README for pinned FA4 setup and native preflight.

The reference preserves ARC1 + ConceptARC + filtered data, 330 epochs, batch 64,
width 768, 8 layers, 12 heads, FFN 3072, FA2, seed 42, NorMuon 0.004, AdamW
0.00072, four prediction heads, and `mtp_decay=[0.8,0.9,1.0]`. It sets
`attention_pattern="pair_group"`, `training_demo_schedule="half_or_more"`,
`augmentation_sampling_mode="context_palette"`, task/dihedral embeddings off,
and no simplex identities. Full-decode evaluation uses balanced leave-one-out
and 80 attempts. See [exact provenance and color policy](README_context_palette.md).

The planned ladder data is ARC2 train + ARC2 eval + ARC2 compositions from both
splits + ConceptARC + filtered. Data volumes, composition snapshots and further
GPU tuning will be chosen separately. The default entrypoint retains the
reference model/data size with the explicit backend/dropout changes above;
the large variants below provide an opt-in model size for the ladder.

`arc2_family_development.json` and `arc2_family_full.json` apply the default
model/training recipe to the ARC2 family-sampled data modes. Development uses the
frozen 80/40 public split and evaluates `pretrain.final` on the held-out 40 tasks;
full mode includes all public answers and disables evaluation. Both keep the
default's 330 epochs and disabled fine-tuning stage. All recipes cap context at
six demos and use seeded balanced selection; see the
[demo sampling contract](README_demo_ablations.md). See the
[family-data guide](README_family_data.md) for the sampling contract and prepared
package-pinned launch specs.

## Large-default specs

These reusable variants use 24 layers, width 1920, SwiGLU FFN width 5120
(8/3 times the model width), and 30 attention heads with 64 dimensions per head.
They have 1,063,588,608 trainable parameters; disabled frozen task-embedding
tables are additional. The ARC2 development large spec uses physical batch 12
and gradient accumulation 5, for an effective batch of 60, following successful
B300 pretraining and fine-tuning. The other large specs retain physical batch
32 and accumulation 2 (effective 64); their native capacity remains unverified.

| Large spec | Source recipe | Data and evaluation |
| --- | --- | --- |
| [pretrain_eval_large_default.json](pretrain_eval_large_default.json) | `pretrain_eval.json` | ARC1 + ConceptARC + filtered; ARC1 public evaluation |
| [arc2_family_development_large_default.json](arc2_family_development_large_default.json) | `arc2_family_development.json` | ARC2 development families; frozen 40-task validation |
| [arc2_family_full_large_default.json](arc2_family_full_large_default.json) | `arc2_family_full.json` | All public ARC2 families and answers; evaluation disabled |

Each variant changes the run name, model dimensions, training batch size
and gradient accumulation. The development large spec also sets evaluation
batch size 150 and `prefill_token_budget=262144`, the settings that completed
full-decode evaluation on the B300. It saves every 10 epochs with
`checkpoint_keep_last=1`, retaining only the latest local/VM checkpoint.
Each retains its source recipe's 330 epochs,
learning rates, initialization, FA4, augmentation, MTP and data boundaries.
The two family variants remain unbound reusable recipes; prepare a
package-pinned temp spec through the [family-data workflow](README_family_data.md)
before launching them.

The B300 settings above were verified with the 24-layer, width-1920 model,
FA4, context-palette augmentation and 80 evaluation augmentations. Batch 32
and batch 16 exhausted memory in the development test; batch 12 with accumulation
5 completed two pretraining epochs and 90 fine-tuning epochs. These settings
are preserved in the reusable development large spec, independently of the
ignored exact run snapshots. A changed model or longer contexts need a fresh
capacity check. Keep B300 instance 57431449 powered on between runs.

Preview a large recipe without preparing data or starting training:

```bash
venv/bin/python launch_run.py --config specs/pretrain_eval_large_default.json --preview
```

## Additional recipes and capacity probes

[arc2_family_development_two_demo.json](arc2_family_development_two_demo.json)
is the main two-demo development spec, copied from the saved 330-epoch B300
launch with `training_demo_schedule="two"`, `inference_demo_policy="two"`,
width 1536, FFN width 4096, and 24 heads (64 dimensions per head). It retains
24 layers, physical batch 22 with accumulation 3, the pinned counts-v1 data
package, and all other launch settings. The package remains an ignored input
under `assets_tmp/`; the spec itself belongs in Git.

Evaluation stages accept `prefill_token_budget` in `eval_defaults` or individual
`evals` entries (default 524288 padded prompt tokens). Together with `batch_size`,
this bounds prefill work: each length bucket uses the smaller of the example
cap and the token budget divided by its padded prompt length. It does not cap
all KV cache or intermediate memory. Saved-checkpoint evaluations can use
`eval_only=true`; see the [Vast workflow](../workflows/vast/README.md) for repeating
an evaluation on the same instance without retraining.

`finetune_only=true` requires a top-level checkpoint, `enable_finetune=true`, and
`dataset.style="finetune"`. It skips pretraining and starts each configured
fine-tune from that checkpoint with fresh optimizer/schedule state. Fine-tune
epoch counts are new adaptation epochs. Cached-data specs use the managed Vast
`continue` command, which verifies and reuses existing on-VM inputs.

To resume an interrupted adaptation, add `finetune_resume=true` together with
`finetune_only=true` and use the saved fine-tune checkpoint. Optimizer, scheduler,
per-rank RNG and completed-epoch progress are restored; fine-tune `epochs` is
the total target in this mode. Keep the dataset, objective and GPU count fixed.

For independent middle-block looping and NextLat variants of the context-palette
recipe, see [looping and NextLat](README_looping_next_latent.md). The two specs
keep the baseline protocol and change one treatment each.

`group_attention_pretrain_eval.json` is the fresh ARC1 + filtered + ConceptARC
all-demo group-attention recipe. `defaults.attention_pattern="pair_group"`
selects pair-local demos with the target input as the shared hub, direct output
access to all demos through separately normalized target/demo lookups and
learned per-head mixing gates, demo RoPE roles 1/2, and target RoPE roles 3/4. No simplex
labels or summary compression are used. Checkpoints preserve the mode; an existing dense
checkpoint cannot be converted by overriding the spec. The batch sizes are
recipe defaults, not a measured all-demo GPU capacity. See the root README for
the attention rules and implementation details.

For a bounded training capacity probe, add a top-level `training_benchmark`
object to a fresh training spec and set `enable_finetune=false`,
`skip_evals=true`, `evals={}`, and `gradient_accumulation_steps=1`. Launch it
through the normal cached-authentication Colab workflow. The default probe
uses a 70 **decimal GB** ceiling, searches batch multiples of eight, and
measures sampled original pairs plus synthetic 20×20 and 30×30 pairs. Each
candidate runs in a fresh process with the existing BF16, compiled model,
NorMuon optimizer, backward, gradient clipping, training metrics, and device
prefetch path. Expected capacity rejections are recorded without failing the
overall run; unexpected errors fail and preserve the normal recovery workflow.

Use `modes=["longest"]` for batches filled with the longest complete original
training contexts. For group attention, `modes=["variation_longest"]` scans all
selected task-variation files using the target plus its six longest eligible
demos as an upper bound. The
variation scan is saved once and reused by every candidate. `grid_30` means
every input and output is 30×30; group probes use the largest eligible training
demo count across the original and variation pools, capped at six.
Real-context length profiles use identity geometry, before augmentation.
The default allocator margin is 1 GB for CUDA allocations outside PyTorch.
For an explicit spot check, set `search=false` and equal start/max batch sizes.
Unstable search boundaries back off and repeat before being confirmed.

Results are saved under the run's `training_benchmark/results.json`, including
allocated/reserved/device memory, throughput, sequence lengths, the adjacent
rejected batch, and an independent confirmation of each largest batch. Memory
includes warmup and initialized optimizer state; timing excludes compilation.
The throughput measurement repeats pre-collated CPU batches and includes their
device transfer, but excludes dataset augmentation/collation and checkpoint
I/O. These probes establish hardware capacity, not optimal learning rates or
decoded accuracy. Keep individual probe specs in ignored `specs/temp specs/`.

For end-to-end timing, use `modes=["epoch"]`, `search=false`, and equal start/max
physical batch sizes. One warmup epoch and two measured epochs are the defaults;
a second fresh process confirms the result. This includes augmentation, CPU
layout preparation, workers, device transfers, compiled forward/backward,
clipping, optimizer and metrics. It preserves the full recipe's LR, NorMuon and
MTP schedules while stopping after the bounded epochs, without checkpoints or
evaluation. Initial compilation is excluded by warmup; later compilation stays
in the measured epoch time.

On `codex/two-demo-context`, each target gains two other demonstration pairs.
Fresh training in `two_demo_bidirectional_v1` is required. Historical outcomes
discussed below are recipe references; new run evidence belongs under
`results/<branch>/` (the run-layout helper encodes slashes in branch names).

## Two-demo context and task conditioning

Before `scaling-ladder`, `pretrain_eval.json` kept the saved R1 1K/1K batch 64
and 330 epochs, with the user-selected historical NorMuon LR 0.004 and AdamW
LR 0.00072. That earlier recipe preserved the remaining R1 settings, including
80 augmentations,
`augmentation_sampling_mode="single_grid_prefix"`, and four prediction heads
with constant MTP weight 0.3. Its defaults added two demos, disabled
dihedral embeddings, and exposed `defaults.task_embeddings` (true or false).
Evaluation inherits the saved task-embedding switch.

Demo sampling does not change target selection, order, batching, or augmentation
RNG. Each target has independent seed-based shuffled cycles. Every prefix of
the schedule balances each demo's total use and use in either position to
within one occurrence; a full cycle puts each demo once in each position.
Resume selects by visit count without mutable worker RNG. Training demos come
from other solved pairs in the same task or saved variation; test prompts use
only training demonstrations. When fewer than two exist, all available distinct
demos are used. Every demo and target shares the same geometry/color mapping.
The entire prefix through the target separator is bidirectional; only the
causal target output is supervised. Capacity is 5,581 tokens with grid roles
0 through 6.

For the requested historical comparison, `e161fd7` (the parent of the
single-pair change `ced7ce0`) used the following pretraining recipe in
`specs/pretrain_eval.json`:

| Setting | Saved R1 1K/1K | Historical whole-task context |
| --- | ---: | ---: |
| Batch / accumulation | 64 / 1 | 64 / 1 |
| NorMuon LR | 0.002 | 0.004 |
| AdamW LR | 0.00036 | 0.00072 |
| Epochs | 330 | 90 |
| Training / eval augmentations | 80 / 80 | 80 / 80 |
| Eval batch | 300 | 300 |
| RoPE spatial / role bases | 1000 / 1000 | 10000 / 10000 |
| Prediction heads | 4, MTP weight 0.3 | 1, no MTP |
| Augmentation policy | Protected output-only colors, unique target inputs | Full-task color policy |
| Dihedral embedding | Enabled | Absent |

Width 768, 8 layers, 12 heads, FFN 3072, dropout 0.1, weight decay 0.1,
attention/token/task decay 0.01, gradient clip 1, warmup 0.5%, decay start
80%, and LR floor 0.0001 were the same in those two specs. The older context
contained up to nine regular demos (seven for REARC/NVARC), rather than two.
The historical ARC-2 variation recipe used pretrain batch 20 with LRs
0.0022 / 0.0034 for 84 epochs, followed by finetune batch 8 with LRs
0.0003 / 0.003 for 330 epochs. These are source/spec comparisons, not results
from this branch. Only the historical learning rates have been adopted here;
batch 64, 330 epochs, and the remaining R1 settings are retained.

Add reusable base templates here. For a sweep, keep each trial's compact
override record in `results/<branch>/experiments/trials/` and generate its executable JSON in
the ignored `specs/temp specs/` folder. See the
[trial workflow](trials/README.md).

One-off Drive continuation configs also belong in the ignored
`specs/temp specs/` folder, rather than alongside reusable baselines. Copy the
source run's exact recovered spec, choose a new `run_name`, and set its Drive
`checkpoint` reference and total `pretrain.epochs` target. These use the normal
Colab launcher without a trial record; sweep-generated staged specs still
require one. See [continuation instructions](../scripts/README_colab_drive.md#continue-pretraining-from-any-saved-checkpoint).

The September 2026 `arc2_ft_autoresearch_20260929*.json` execution snapshots
live under the corresponding `results/colab-restart/colab_runs/<session>/specs/` folders. The
[sweep note](../results/colab-restart/experiments/notes/arc2_finetune_sweep_2026-09-29.md) links each
copy. Old run journals retain the paths used on the VMs; those uploaded source
archives are unchanged. New trials should not add permanent stage files here.

The finetune phases in `arc2_variations_finetune.json` and
`conceptarc_variations_finetune.json` use batch size 8, Normuon LR 0.0003,
AdamW LR 0.003, and 330 epochs. These are the best tested full-decode
settings in the [28–29 September ARC-2 sweep](../results/colab-restart/experiments/notes/arc2_finetune_sweep_2026-09-29.md),
which scored 10⅙/120 (8.47%) on public eval. ConceptARC adopts the ARC-2
settings; they have not been validated by a ConceptARC sweep. Pretraining
settings and evaluation checkpoints are unchanged.

`arc2_variations_originals_big_batch_finetune.json` mixes original ARC-2 train,
ConceptARC, and filtered ARC-1 tasks with all 42 variations per parent every
epoch. Pretraining defaults to 90 epochs and batch 96. Its model, optimizer,
augmentation, and evaluation settings copy the recovered
`r1-b96-linear-20261009-022757` ARC-1 run (184/400, 46%, versus 177/400 for the
sqrt-LR trial): NorMuon 0.003, AdamW 0.00054, 8 layers, Xavier initialization,
learned shortcuts, no QK or branch normalization, dihedral embeddings, and
constant MTP weight 0.3 with four prediction heads. That result is recipe
provenance, not a measured result for this new ARC-2 data mix. ARC-2 finetuning
retains batch 8, 330 epochs, and the existing 0.0003/0.003 learning rates.
Final evaluation enables greedy self-speculative decoding with the trained MTP
heads via `eval_defaults.inference_speculative=true`.

Variation sampling accepts `embedding_scope="variation"` for separate task
identities of the form `<parent>/variation/seed<N>`. Each original and its
variants have independent trainable task-factor rows, initialized by copying
the normally initialized original row to its variants on a fresh model only.
The shared latent projections therefore also produce identical initial input
and value embeddings for each family. Resume restores the learned rows without
copying them again. The default `embedding_scope="parent"` keeps existing specs
and checkpoints compatible.

Set `sampling.deduplicate_variation_inputs=false` to bypass unique-input pool
construction for variations while keeping the selected color-protection policy.
Original tasks and inference still use their normal `single_grid_prefix`
duplicate checks. Omitted flags preserve the previous variation behavior.
`variations_per_parent_per_epoch=42` includes every solved variation pair once
per epoch; original solved pairs are also included once, with shuffled batch
order and color/geometry augmentation. This does not equalize the exposure of
originals and variations, and batch-96 GPU capacity for this larger embedding
table still needs to be measured.

For a Kaggle checkpoint that trains on public answers, explicitly set
`dataset.pretrain_data.public_pretraining=true`, `dataset.style="pretrain"`,
`enable_finetune=false`, and `skip_evals=true`. This opt-in permits
`arc2_eval` in `sources` and `original_sources`, alongside `arc2_train`,
`conceptarc`, and `arc1_filtered`. Every selected original and variation uses
both demonstration and solved test pairs. Set
`eval_support_repeats_per_epoch=0`; public eval originals already contribute
their solved pairs. A held-out parent split is incompatible with this mode.
Other specs keep their existing separation of evaluation answers.

Saved compositions can be packaged from a sibling `arc-synth-data` checkout:

```bash
venv/bin/python scripts/package_compositions.py \
  --synth-root ../arc-synth-data --pipeline v3 --include-author-completed \
  --output assets_tmp/task-compositions-v3-<snapshot>.zip
```

The packager copies saved puzzles, verifies them against their generation
checkpoints, and records their source, parent IDs, seeds, and completed stage.
It invokes no generation models. `--include-author-completed` explicitly
includes locally checked v3 author output whose review/improvement stage has
not run; omit it to require completed families. `--pipeline v3` excludes v2.
Choose a new output filename for each snapshot; existing archives are retained.

Add `pretrain_data.compositions` with the printed `archive_path`, `size`, and
`sha256`. Public pretraining verifies this ZIP and its per-sample checksums,
and includes every packaged composition sample once per epoch, with all solved
pairs. Each family has an independent identity; with variation embedding scope,
each sample has its own task-factor row initialized from that family. The Vast
source snapshot includes the ZIP directly, so a separate Drive upload is not
required. Composition lineage is limited to sources actually present in the
saved collection; packaging does not create compositions for missing sources.

`conceptarc_variations_finetune.json` is the fixed ConceptARC iteration baseline:
84 epochs of variations from 120 parents, then 330 epochs of finetuning on the
original demonstrations from all 40 held-out tasks. It evaluates `familiar`
(30 tasks) and `heldout_concept` (10 TopBottom3D tasks) separately. Both use the
same final finetune checkpoint. The fixed task lists live in
`dataset_building_scripts/dataset_splits/conceptarc_topbottom3d_v1.json`; training seed changes do not
alter them. Model and optimizer settings are explicit in the spec.

`conceptarc_originals_finetune.json` is the non-varied comparison. It uses the
same fixed 120/30/10 split: 120 original training tasks (672 solved pairs),
30 familiar held-out tasks, and all 10 TopBottom3D tasks. Original training
parents contribute both their demonstration and solved test pairs, matching
the variation pipeline's `all_solved` policy. The 40 held-out parents contribute
only original demonstrations during finetuning; test outputs are kept in the
solution files for evaluation. Pretraining uses 330 epochs, 80 augmentations,
batch 64, Normuon LR 0.0022, and AdamW LR 0.0034. Model, scheduler, dropout,
finetune, and evaluation settings match the uploaded
`conceptarc-672-20261001-002225` spec, except for the pretrain epoch/checkpoint
schedule and batch size. Finetuning remains batch 8, 330 epochs, and
0.0003/0.003; full decoding remains batch 300 with 80 augmentations and seed 42.

The batch-64 default retains the variation run's LRs to limit changes in the
comparison. Batch 32 at the same rates is a useful alternative with twice as
many optimizer updates per epoch. The older `conceptarc_pretrain_eval.json`
uses batch 64 and LRs 0.004/0.00072; that is another candidate, but it changes
both optimizer rates as well as the data and training budget. None of these
pretraining choices is established as best by a ConceptARC sweep, and batch-64
GPU memory use still needs to be measured. The prepared optional specs
`specs/temp specs/conceptarc_originals_b32_matched.json` and
`specs/temp specs/conceptarc_originals_b64_legacy.json` use pretrain-only
overrides and leave finetuning/evaluation unchanged. These manually authored
temporary specs require `--standalone-spec` when launching.

This comparison uses different training budgets (330 original epochs versus
672 variation epochs, with different pair counts and batch sizes). It tests
the two practical recipes; it does not isolate the dataset effect at equal
optimizer updates or token exposure. `max_augments=80` is the augmentation
schedule size, not an 80-fold replication of each pair within an epoch.

## Augmentation and sampling modes

Use `defaults.augmentation_sampling_mode` (or a stage/evaluation override):

```json
"augmentation_sampling_mode": "single_grid_prefix"
```

- `full_task_prefix` preserves the augmentation and sampling policy from
  before this experiment: regular ARC recolors all known task colors into
  `0–9`, with the existing balanced color/geometry schedules and synthetic
  sampling policies. This remains the fallback when no policy is specified.
- `single_grid_prefix` reserves colors found in demonstration outputs but
  absent from all known task inputs (demonstration and test inputs). These
  labels remain fixed and are excluded from recoloring destinations. Black is
  treated like every other color. Remaining colors can map to any unreserved
  label, including unused labels, without collisions. REARC, NVARC and saved
  variations apply the same protection to their known training pairs. NVARC
  in this mode also gains destinations outside its existing palette.

The switch controls augmentation and sampling independently of the two-demo
sequence context. Pair/task sampling and the balanced training geometry
schedule are retained. Single-grid mode also restores the March unique-input
pool policy for training and inference:

- Keep every original pair, including duplicate originals.
- Reject added inputs matching any known original or any previously accepted
  augmentation of another pair in that task. Compare input content and shape;
  changing colors absent from the input does not make a new variant.
- Try new input-color mappings first, then remaining geometries of accepted
  mappings. `max_augments`/inference attempts are caps: small or symmetric
  grids can yield fewer distinct inputs. Training can reuse these pools across
  epochs; inference evaluates each accepted entry once.

Color proposals use balanced blocks over unreserved labels (nine labels when
one is protected), but filtering can change the final color frequencies.
Pools favor balanced geometries. Training uses its requested geometry whenever
the unique pool contains it; when none exists, it uses an available pool entry
and its actual geometry. The fallback mode retains its original schedules.

Pool building caches the eight input geometries, uses native byte translation
for recoloring and exact byte comparison for duplicates, and caches selection
orders. No duplicate checks run inside epoch selection. REARC/NVARC pools are
built once per task on first use; variation pools use a bounded cache.

Inference uses the same reserved-color rule, inferred from demonstrations and
known inputs; test outputs and solution files never determine augmentation.
Both policies apply one injective mapping to input and output, and inference
undoes the mapping before voting. The restricted policy completes a full
permutation of the ten labels, keeping reserved labels fixed.

Omitted/null mode inherits the checkpoint's policy; fresh runs and legacy
checkpoints default to `full_task_prefix`. New checkpoints save the policy so
evaluation and continuation inherit it. An explicit mode overrides it. Set
`full_task_prefix` to return to the fallback policy.

The complete pre-change tracked working tree, including the existing MTP
decay edits, is also saved locally on branch
`codex/augmentation-sampling-fallback-20261005`. Switching the policy requires
only the spec field above; checking out that branch recovers the earlier code.

## Multi-token prediction

Fresh runs use four prediction heads: the existing next-token head plus three
linear heads for tokens 2, 3, and 4 ahead. They share the same transformer pass.
The training objective is
`output_loss + mtp_weight * mtp_multiplier * mean(auxiliary losses)`,
with `mtp_weight=0.3` and `mtp_decay=[0.8, 0.9, 1.0]` by default, matching
R1's constant auxiliary weight of 0.3 throughout training. `mtp_decay` is an array
`[start, end, final_multiplier]` of finite values with
`0 <= start < end <= 1` and `0 <= final_multiplier <= 1`; set it in `defaults`,
`pretrain`, or a finetune override. A final multiplier of 0 removes the MTP
contribution after `end`; 1 keeps its weight constant.
Progress uses the fractional epoch at each minibatch's start divided
by the configured total epochs, including completed epochs on resume. Each
finetune uses its own epoch budget. Only queries from the input/output separator
onward receive loss; padding and offsets beyond each example's end are masked.
The next-token loss keeps weight 1. Logs retain `loss=` for next-token CE and
add `mtp_loss=` and `total_loss=` when MTP is active. `mtp_loss` reports the raw
mean auxiliary loss; `total_loss` includes the effective MTP weight.

Set `"n_predict": 4, "mtp_weight": 0.3` in `defaults`, `pretrain`, or a finetune
override to make the settings explicit. Set `"n_predict": 1` for the original
objective; `"mtp_weight": 0` also skips auxiliary head computation. These heads
use the existing AdamW output-head settings, including under NorMuon.

Omitted/null `n_predict` and `mtp_weight` settings inherit the saved checkpoint's
values when resuming or finetuning. `mtp_decay` uses the selected stage's
schedule, defaulting to `[0.8, 0.9, 1.0]`. Legacy checkpoints without MTP fields
remain single-head models. Resume rejects changes to the saved head count or
loss weight. Finetuning may change them, keeping existing weights, initializing
added heads, and starting a fresh optimizer/schedule. For the standalone
checkpoint workflow this means
`pretrain.reset_training_state=true`; an explicit `checkpoint_mode='resume'`
in the lower-level builder always preserves the saved training state.

Generation defaults to `"inference_speculative": "auto"`, selecting speculative
decoding for greedy inference and ordinary decoding for sampling. Set `false`
to use the next-token head alone, or `true` to require greedy speculation in
`defaults`, `eval_defaults`, or an eval override. With four trained heads,
the auxiliary heads propose three tokens and the existing cached inference
path verifies them together with the pending main-head token. Only the
contiguous prefix matching the main head is accepted; the first disagreement
is replaced by the main-head token. Rejected cache entries and grid positions
are rolled back independently for each batch row. Explicit `true` rejects sampling; use
`inference_temperature=0` or leave both sampling settings unset. Single-head
checkpoints retain ordinary decoding, and accuracy-only evaluation never runs
auxiliary heads. There is no runtime speed monitor or adaptive draft policy.

Evaluation's output limit is 930 tokens: 900 colors, 29 row separators, and
one end token. The end token replaces the final newline during serialization,
so it is already included; 931 would add an unnecessary output slot. The
default prefill token budget is 524,288, separately limiting the sum of padded
prompt lengths in a batch. The default evaluation `batch_size` is 120 and is
an additional cap. Two-demo prompts can reach 4,651 tokens, giving an effective
batch of 112; prompts padded to 4,096 tokens reach the cap of 120. Teacher-forced
accuracy uses the same token budget and batch cap, counting the full sequence.
New runs use these defaults; already uploaded Colab snapshots keep their saved
code and settings.

Training, prefill, and inference precision/backend choices are preserved.
Different verification shapes can still change floating-point rounding, so
compare exact generated tokens and warmed throughput on the intended GPU and
checkpoint before relying on numerical parity. This does not claim bitwise
invariance across batching, hardware, or software versions.

For an implementation-time comparison, an eval-only checkpoint spec with
`enable_finetune=false` can include `"speculative_benchmark": {}`. This runs
fixed-prefix one-token/block logit comparisons and exact free-generation
comparisons, warms both routes, then reports median end-to-end speed over three
repeats. Optional positive integer fields are `task_count` (8),
`augmentation_attempts` (2), `batch_size` (8), `max_new_tokens` (128),
`parity_cases` (4), `parity_tokens` (32), and `repeats` (3). Launch with the normal
cached Colab wrapper and `--stop-after-success`. Results go to
`speculative_benchmark/results.json` inside the run folder. A disagreement fails
the diagnostic after saving the comparison; existing VM retention/recovery
rules apply. This requires a CUDA GPU and a checkpoint with auxiliary heads.
The newer harness also accepts `max_seconds` for a bounded time budget,
`compile`, `compare_legacy` for the original full cache, and `compiled_probe`.
It reports numerical drift separately from generated-token disagreements and
saves partial results if the time budget ends. No benchmark starts automatically.

For a bounded batching comparison, use an eval-only checkpoint spec with
`enable_finetune=false` and `"evaluation_budget_benchmark": {}`. It compares
65,536, 262,144, 393,216, and 524,288 prompt-token budgets with caps 100 and 512.
It selects 48 tasks (half the longest prompts, half a seeded sample), uses 16
augmentation attempts, warms ordinary and speculative routes, and times three
repeats. Separate ordinary stress probes force all 930 output slots for both
prompt buckets, preserving the model weights, precision, and attention paths.
Candidates must retain 15% estimated device memory headroom. Exact tokens,
timings, peak allocated/reserved memory, and batch shapes are saved under
`evaluation_budget_benchmark/`; greedy output differences are reported.
Optional fields are `variants` (objects with `token_budget` and `batch_size`),
`task_count`, `augmentation_attempts`, `max_new_tokens`, `repeats`,
`trial_timeout_seconds` (1800), and `headroom_fraction` (0.15). Workers isolate
CUDA compilation and allocations between candidates. Use the normal cached
Colab launcher with `--stop-after-success`; this diagnostic performs no training
and adds no production speed or memory instrumentation.

GPU throughput and ARC accuracy need to be measured
for the chosen training recipe; MTP is a different objective, not a guarantee
of higher accuracy.

## Original-task data mode

Use `dataset.pretrain_data.kind="original_tasks"` with a nonempty `sources`
list drawn from `conceptarc`, `arc2_train`, and `arc1_filtered`. Sources can be
mixed. The mode needs only the grouped originals, without the variation ZIP.
Use `dataset.evaluation.source` to choose official `arc1_eval`/`arc2_eval`,
or a parent source with a `split_manifest` for a fixed held-out split. Task
IDs, group routing, manifest/source hash checks, and run-local data files use
the same rules as the variation mode. Pretrain checkpoints record the dataset
identity and reject mismatched original data on resume; finetuning preserves
that identity.

With `style="finetune"`, held-out support is used only during finetuning.
With `style="pretrain"` and an official eval source, support is included once
per epoch, alongside the original training sources. A fixed parent split
always excludes held-out support from pretraining. The optional
`eval_support_repeats_per_epoch` accepts 0 or 1 for original tasks, subject
to those style/split rules. Variation-only `snapshot`, `sampling`, and
`original_sources` settings are rejected in this mode.

Launch the comparison from the repository root:

```bash
RUN_SPEC="specs/conceptarc_originals_finetune.json"
# Optional: select one of the two temporary specs above instead.
RUN_SESSION="conceptarc-originals-$(date +%Y%m%d-%H%M%S)"
venv/bin/python scripts/colab_oauth/launch.py --config "$RUN_SPEC" --standalone-spec --dry-run
colab new -s "$RUN_SESSION" --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/launch.py \
  --config "$RUN_SPEC" --standalone-spec --session "$RUN_SESSION" \
  --gpu A100 --high-mem --min-gpu-memory-gb 75 --stop-after-success \
  --intent "Compare original ConceptARC with the 672-epoch variation run on the same split" \
  --changed-from-previous "Original training pairs; 330 pretrain epochs; selected pretrain batch/LRs; matching finetune and eval"
```

See [the Drive workflow](../scripts/README_colab_drive.md) for authentication,
backup verification, and recovery details.

## Conditioning, shortcuts and diagnostics

The default experiment sets `task_embeddings=false` and `dihedral_embeddings=false`.
Low-level fresh-model construction still uses task conditioning when no setting
is supplied, preserving explicit alternative specs and checkpoint inheritance.
Set `task_embeddings=false` for a fresh model to disable input task conditioning,
task-dependent value embeddings, and their shortcuts. Disabled tables and gates
remain frozen, outside optimizer groups; shared initial weights are identical
for a given seed. Resume and evaluation inherit this architecture.

The optional `dihedral_embeddings=true` setting adds eight learned vectors,
one for each existing transform index (0 is identity, 1–3 are rotations, and
4–7 are reflections). The actual augmentation ID is supplied for every pair
in training, accuracy evaluation, prefill, and cached decoding. The vector
uses the historical sigmoid gate initialized at -2 and a separate per-block
shortcut initialized at zero. Adding the table preserves shared initial
weights for the same seed. Omitted/null settings inherit a checkpoint's architecture; older
checkpoints without the field keep their original model without this table.
Changing it requires fresh training. Direct model calls with embeddings
enabled must supply `dihedral_ids` with one integer in 0–7 per sequence.
The table has its own AdamW group, configured by
`dihedral_embedding_weight_decay`, `adamw_dihedral_embedding_lr_multiplier`,
and `adamw_dihedral_embedding_lr_floor`; its gate and shortcut scalars use
the existing group without decay.

`input_shortcuts="fixed"` freezes the existing block shortcut scalars at
identity (hidden=1, token=0, example=0, dihedral=0 when enabled). `learned` retains the normal trainable
shortcuts. Both settings are checkpointed. Omitted/null settings inherit the
saved architecture. A resumed or
finetuned checkpoint cannot change these fresh-model settings. Loading a
checkpoint restores its learned weights.

`normuon_residual_output_lr_multiplier` (default `1.0`) multiplies the
configured NorMuon learning rates of each block's `attention.out_proj.weight`
and `ff.fc_out.weight` only. It composes with the existing attention/FC output
group multipliers. QKV, value-embedding gates, FFN input weights, and AdamW
parameters retain their configured rates. When the multiplier differs from
one, attention output weights form a separate optimizer group that inherits
the attention weight decay and LR floor. The default preserves historical
optimizer group topology. Specify the same setting in the shared defaults to
apply it to both pretraining and finetuning; a changed optimizer group layout
on continuation triggers the normal optimizer-state reset rules.

`depth_diagnostics_every=N` enables periodic per-layer residual output weight
norms, actual optimizer update norms/weight ratios, shortcut scalars, and the
gradient norm before clipping in the training console. Zero disables this
instrumentation. With diagnostics enabled, nonfinite clipped gradient norms
fail before the optimizer update, and sampled losses/updates are checked too.
Snapshots exist only for sampled updates; diagnostics consume extra memory and
time on those updates. NorMuon's normalized updates make these measurements
useful when testing smaller initial residual weights. Diagnostic intervals count
optimizer updates when gradient accumulation is used (global progress is
estimated from the training minibatch counter).

## R1 1k/1k model defaults

The default architecture follows the saved `r1-rope-xy1k-z1k-20261013-140632`
recipe: 3D RoPE, a role base of 1000, ordinary Xavier initialization, original
row-end boundary coordinates, no extra grid-role embeddings, and no query/key
or internal branch normalization. Query/key normalization and the other retired
positioning ablations remain unavailable. Two independent, optional controls
are available in shared `defaults` (or a specific training/evaluation stage):

```json
"defaults": {
  "residual_init": "depth_scaled",
  "branch_norm": "hidden_rms"
}
```

Set either key alone to enable only that option. Omitted or `null` settings use
`residual_init="xavier"` and `branch_norm="none"` for fresh models, and inherit
saved checkpoint settings for continuation/evaluation. Explicit checkpoint
overrides must match the saved values, including during finetuning.
Depth scaling multiplies each block's attention and FFN output weights by
`1/sqrt(2*n_layers)` once after ordinary initialization, with no runtime cost.
Internal RMS normalization acts on concatenated attention heads before the
attention output projection, and on gated FFN activations before hidden dropout
and the FFN output projection. It adds two learned RMS gains per block and
normalization work when enabled; native GPU throughput and BF16 parity need
GPU measurement. Neither option is enabled in the default reusable specs, and
their combination is not established as better by existing results.

The existing pre-RMSNorm blocks and learned input shortcuts remain.

`rope_spatial_base` remains configurable: it accepts a finite number greater
than one and defaults to **1000** for both columns and rows. A 64-channel head
keeps its original 20/20/24 column/row/role split. Changing the spatial base
requires fresh training; evaluation and continuation inherit the saved value.

The default experiment uses eight layers, width 768, twelve heads, FFN width
3072, no task conditioning, four prediction heads and MTP weight 0.3 with
`mtp_decay=[0.8,0.9,1.0]`.
The default task latent-bank size is `min(3072, number of task identities)`,
matching R1's effective 1307 latents on its original dataset. An explicit
`num_task_embedding_latents` overrides this default. Checkpoints retain their
saved latent-bank size even when adapting to fewer tasks.

New checkpoints record a fixed architecture version. Existing R1 checkpoints
with an explicit role base of 1000 load without changing their math. Checkpoints
with removed ablation treatments, or historical implicit role base 10000,
fail explicitly instead of being interpreted as the new model. Historical
run snapshots are retained unchanged. Run budgets, dropout, optimization,
augmentation and the remaining architectural settings remain configurable.

## Vast GPU templates

`vast/` contains infrastructure overlays for A100 40 GB, A100 80 GB, H100,
H200, and B200. Select them with the Vast launcher's `--gpu-template` option;
they inherit the saved infrastructure policy and are separate from experiment
specs. See the [Vast workflow guide](../workflows/vast/README.md#gpu-templates-and-automatic-ngc-container-selection).

## Frozen-checkpoint backend comparisons

An eval-only spec can include `backend_benchmark` to compare the current
FlashAttention/SDPA/FlexAttention paths against SDPA decoding and a precision
policy matching BF16-autocast training. It requires a top-level checkpoint and
`enable_finetune=false`. Run it through the normal cached Colab launcher with
`--stop-after-success`; no training or checkpoint updates occur.

```json
"backend_benchmark": {
  "variants": ["current_bf16", "sdpa_bf16", "current_amp", "sdpa_amp", "current_bf16_fp32_rope"],
  "task_count": 40,
  "augmentation_attempts": 8,
  "batch_size": 32,
  "max_new_tokens": 930,
  "parity_cases": 8,
  "parity_tokens": 64,
  "trial_timeout_seconds": 600
}
```

Each variant uses a separate CUDA process and the same seeded task selection
and augmentation schedule. `current_bf16` preserves the standard inference
policy; `sdpa_bf16` disables FlexAttention decoding. The AMP variants keep
parameters, residual streams, and positional tables in FP32 and use BF16
autocast. The optional RoPE variant changes only the positional table precision.
SDPA remains automatically dispatched, so this comparison does not pin its
internal kernel. Fixed-target-prefix probes compare native packed FlashAttention,
padded SDPA, and cached decoding; their tested token count can be shorter than
the full answer. Free decoding uses the existing generator and AAIVR scoring.

Results and partial failures are written under `backend_benchmark/` in the run
folder. The comparison reports sequence disagreements, gained/lost correct
test pairs, and task-averaged ARC scores. A bounded subset with fewer
augmentations does not establish the full evaluation score or predict the
effect of retraining with another backend.
