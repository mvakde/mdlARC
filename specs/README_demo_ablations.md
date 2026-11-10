# Opt-in demo ablations

The pair-group model caps context at six demos and retains two-lookup attention.
These settings are independent, reusable fields in `defaults`/stage overrides:

| Field | Default | Treatment |
| --- | --- | --- |
| `training_demo_schedule` | `all` | `all` uses up to six; `half_or_more` cycles equally over counts from half the eligible demos (rounded up, at least two) to the available total, with both bounds capped at six; `two` always selects `min(2, M)` of M eligible demos. |
| `inference_demo_policy` | `all` | `all` selects `min(6, M)` of M demos; `leave_one_out` selects `min(6, max(0, M-1))`; `two` selects `min(2, M)`, with at most `max_augments` attempts. |
| `pair_identity_vertices` | `0` | Integer N >= 2 enables a fixed centered regular simplex. Use enough vertices for the actual built dataset. |
| `pair_identity_strength` | `0.25` | Fixed positive norm of the additive pair signal, before input normalization. |

LOO with no demos retains the target input; with one demo it omits that demo.
The augmentation schedule, its RNG, decoding, and voting are unchanged. Unique
input augmentation can exhaust before the requested attempt cap, as before.
Demo inclusion counts differ by at most one over every attempt prefix.
Set both fields to `two` for two-demo mode throughout training and evaluation.
One eligible demo is kept, and zero eligible demos leave only the target.
Training always excludes the target pair. Two-demo mode retains the same seeded
balanced selection, shared augmentation, target supervision, and voting.
The six-demo cap and seeded balancing apply to every pair-group data mode,
including family development/full data, ordinary variations, and fine-tuning.
`max_training_demos` lowers that training cap (integer 1–6; default 6) while
retaining the selected count schedule, target exclusion and seeded balancing.
For example, `half_or_more` with cap 3 selects two or three eligible demos when
four are available, and three when five or more are available. Inference's
existing demo policy is independent of this training-only cap.
The nondefault training cap is recorded in family/variation checkpoint metadata.
All targets in a complete puzzle pass share a count and seeded cyclic offsets;
each physical pair is therefore used equally often within that pass. For each
target, use of its eligible demos differs by at most one over every visit prefix.
This replaces the archived per-target subset schedule; historical uploaded
sources and results retain their original behavior.

The 11-vertex palette uses a deterministic Helmert isometry into the first ten
model coordinates. Unit vertices have off-diagonal dot product -0.1. The gain
0.25 is fixed across future simplex comparisons: its signal norm is about 45%
of the initial expected width-768 token norm (`0.02 * sqrt(768)`, about 0.55).
No parameters or RNG draws are added during initialization, preserving every
shared initial weight. Row zero is neutral for the start token and the target.
Demo input/output tokens and their boundaries receive the same ID. Roles 3/4,
attention masks, normalization, and the original model objective are retained.

Assignments use randomized Latin cycles keyed by puzzle, target, and physical
demo identities. Each pair receives each vertex once per N occurrences; each
vertex appears M times per complete N-occurrence cycle. Assignment is injective
within a context and independent of augmentation/initialization RNG. For varying
contexts, occurrences are counted within each visible subset, preserving these
guarantees conditional on context. Training uses the same per-target visit index
as the historical schedule; replay does not rely on mutable worker state. IDs
are supplied explicitly for prefill; cached output tokens always use zero.

For sequential checkpoint evaluations on one managed VM, an `eval_only` spec
can include `evaluation_checkpoints`, a map of safe names to ordinary Drive
checkpoint identities. Reference them in `evals` with `input.NAME`. A top-level
`checkpoint` remains required and is referenced with `checkpoint`. Each input
is downloaded and verified directly on the VM, recorded, excluded from output
checkpoint backups, and never relayed through the Mac. One normal launcher
owns the entire sequence and shuts down only after all stages and recovery.
For an input from another historical branch, include its explicit `source_branch`
in the checkpoint identity. The downloader verifies the encoded branch folder
name and its ancestry under the configured `mdlARC/runs` parent. Omitting this
field retains the original same-branch restriction.

An evaluation's optional `reference_submission` names a repository-relative
JSON file with saved baseline predictions. It is included in the source archive.
After scoring, `all_demo_comparison.json` records both scores, each task's change,
and aggregates by demonstration count. Reference answers enter scoring only.

Historical target-roles and half-or-more checkpoints retain combined attention;
the loader accepts their explicit serialization formats and constructs their
saved architecture, with strict state loading. They are never assigned gates.

Set top-level `validate_cuda: true` to run the bounded native FA2, compiled
packed/padded forward/backward and generation checks in a separate process
before executing science. Its JSON report is part of the recovered run records.
All other Colab launch, credential, backup, recovery, and shutdown rules remain
in [the Colab guide](../scripts/README_colab.md).

The normal launch wrapper also accepts `--reuse-session OLD --session NEW`
for a subsequent job on a retained VM. It requires verified local result
recovery, a finalized matching remote manifest/status, the original idle kernel,
and no active watcher. It preserves the entire old workspace, source, results
and uploader control directory by renaming them before registering a fresh
local alias for the same VM/kernel. It allocates no VM, installs no credentials,
and never resubmits the old execution. An interrupted rename is inspected,
never blindly retried. A failed initialization with an incomplete backup flag
can be preserved only after metadata inspection proves there is no checkpoint,
partial download or checkpoint ledger. Its original incomplete flag is retained
in the reuse record. Runs with unverified checkpoints require recovery first.
New runs clear stale record-publishing mode and invalidate modules from the
previous uploaded source before importing the new one. The existing original
identities and canonical Drive run folders remain unchanged.
