# Context-aware color augmentation

`augmentation_sampling_mode: "context_palette"` is the `scaling-ladder` default
for regular ARC training and evaluation. `pretrain_eval.json`, `example.json`,
and `context_palette_schedule_two_lookup.json` retain the
330-epoch, seed-42, batch-64 schedule-plus-two-lookup recipe and evaluate with
balanced leave-one-out, at most 80 attempts. Explicit historical modes remain
available.

The reference is the completed
`contextcolor-schedule-e330-b64-20261020-233101-r2` run on
`codex/context-color-augmentation`, which scored **215/400 (53.75%)** on ARC1
public eval. Its canonical [Drive run](https://drive.google.com/drive/folders/DRIVE_ID_17_PLACEHOLDER)
preserves the source archive with SHA-256
`b16d4612bf7ea905fb672605dc6d6824674cb3268223105c041402535b2b9a3e` and launch
spec with SHA-256
`7374574510378ff5a2fef0c7ecb0c0065bd3d2221bb199cbe9e5083cb5c07d52`.
The named `context_palette_schedule_two_lookup.json` template preserves those
settings, with a reusable run name, no one-off reference-submission path, and
explicit ordinary decoding. That path generated a paired report and did not
affect training, generation, or scoring. `pretrain_eval.json` and `example.json`
now explicitly select FA4, attention dropout 0, and automatic speculative
decoding. The 53.75% score belongs to the historical reference, not this update.

This branch starts at the committed context-color implementation `e698e12`.
The uploaded source also contained inactive shifted-attention work; its switches
were off for the reference run. The branch now adapts CPU-prepared layouts,
batched noncausal attention calls, target-only gate projection, a combined-demo
decode cache and greedy/speculative decoding improvements. It retains one
softmax over all selected demos and the original target/demo sigmoid gate;
shifted routing and partner-comparison architectures are not introduced.

For each actual selected context, K contains colors in the available task
grids and V contains colors in the selected complete demos plus target input.
Fix H=K-V, reserve those labels as destinations, and bijectively permute all
remaining labels. Unused labels are legal destinations. Output-only colors
move whenever visible. Gold target output never contributes to V; evaluation
test outputs never contribute to K or to scheduling. This uses the agreed
assumption that required test-output colors occur somewhere in known demo
pairs or test inputs. Deliberately omitted evidence still requires protection.

Neither target-input nor full-prompt deduplication is imposed. Repeated target
inputs with different demos are valid; complete repeated prompts are permitted.
Removing the old target-input-only exhaustion can restore the full 80 attempts
on symmetric tasks. This is part of the treatment and must be recorded when
comparing to historical runs that produced fewer attempts.

The unchanged training sampler owns epoch order, physical target visits, and
the exact shuffle-balanced D4 schedule, including the baseline drop-last policy.
The [capped demo selector](README_demo_ablations.md) owns count and subset balance.
It uses at most six demos and matches leave-one-out evaluation after applying
the cap. The augmenter reads
these plans after the training loop finalizes the sampler and before workers start;
selection replays an explicit visit, without mutable worker counters. Resume
with the same planned horizon reproduces every color mapping and geometry.
Dropped targets retain their original demo-visit IDs on subsequent epochs;
color histograms count only the visits actually present in the finalized plan.

Within each physical target and protection set, complete Latin blocks balance
every unprotected source label across every legal destination. Pools retain
complete blocks up to max_augments (80 becomes 72 for nine legal labels); they
repeat without accumulating marginal imbalance. The pre-existing balanced
inference color generator and minimum-multiplicity color/D4 assignment pair
the multiset with the fixed geometry slots. Conditional source-destination
counts differ by at most one over the planned horizon; this is not a claim
about every intermediate prefix or every full joint contingency table.

Training plans are lazy cached per target in each worker. All schedule RNGs
are local, derived from target identity and protection set; model initialization,
dropout, batch shuffle, and geometry RNGs are untouched. Demo subsets use the
shared seeded capped cycle; the color plan observes those exact subsets.
Inference keeps a single fixed context and mapping throughout each decode and
inverts the complete ten-color bijection before ordinary voting. Model topology,
objective, optimizer, loss weighting, and decoding are unchanged.

RE-ARC, NVARC and variation training reject this mode until they have explicit
matched schedule support. Reusable implementation, tests, this guide and the
template belong in Git. Individual execution specs, reference submissions,
research notes, validation outputs and run records remain ignored.
