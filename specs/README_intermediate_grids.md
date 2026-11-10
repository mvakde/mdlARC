# Intermediate-grid demonstrations

This branch supports up to two intermediate canvas grids per demonstration.
Each chain is one attention group: `input -> intermediate(s) -> output`.
By default only the target output is supervised/generated. Target intermediates
are then never serialized, including during leave-one-out fine-tuning. All grids receive the
same geometry and color augmentation; intermediate colors enter the visible
context palette. Pair-group RoPE positions are demo input 1, intermediate 1 at 2,
intermediate 2 at 3, and output 4; target input/output use 5/6. With one
intermediate it occupies 3 and position 2 is unused. With none, positions 2/3
are unused. Both demos reuse these positions within their own attention groups.
No placeholder grid tokens are added. Fresh chain models select and save
`ordered_grid_positions=true`; training, prefill and cached decoding use the
same mapping. The vocabulary is unchanged. Checkpoints use a distinct chain
format identity. Pair simplex identities are currently disabled for chains.

Preparation executes exact-IO canonical programs through `../dsl-v2`'s native
engine and verifies their final outputs. New packages store ordered candidate
pools (`intermediate_grid_candidates`), rather than two fixed snapshots. On each
training visit or inference attempt, one candidate is sampled from each half of
the pool, in execution order. Pools with zero, one or two states use them all.
The context still contains at most two demos and two intermediates per demo.
Existing packages with `intermediate_grids` retain their fixed selections.

Filtering removes input/output copies, repeated grids, and background-only grids
created by canvas resets or full clear operations (`erase`, `cut`, `paint`,
`recolor`, `fill`). Loop prefixes and bound calls such as `patch = cut(...)`
are recognized. Metadata-only frames cannot reintroduce a filtered blank grid.
Visible detours are removed only when the complete execution state returns to
an earlier state, including the clipboard, variables, selection and pointer.
Work preserved in those fields prevents this removal. Useful partial erasures,
small crops, monochrome crops, and partial object constructions remain eligible.
There is no size, entropy, pixel-change or similarity-to-output threshold.

Sampling is stateless, seeded by the dataset seed, task, target and sampler
visit. Both demos share the early/late random quantiles; this aligns relative
progress, without assuming independent programs have identical semantic stages.
Resume and worker order cannot change a selection or consume model RNG. The
context palette uses exactly the selected grids, and all serialized grids share
the same geometry/color mapping. Candidate pools are not target supervision;
no target intermediate is serialized. No program text enters the learner package.

For ARC1 plus the old recipe's filtered ARC2 supplement, without ConceptARC:

```sh
venv/bin/python scripts/prepare_intermediate_grids.py \
  --package assets_tmp/arc1-filtered-intermediate-grids-NEW.zip \
  --audit-dir results/codex%2Fintermediate-grid-demos/preparation-NEW \
  --output-spec 'specs/temp specs/arc1-filtered-intermediate-grids-NEW.json'
```

Add `--base-spec 'specs/temp specs/EXISTING.json'` to preserve an approved
recipe's hyperparameters, including its epoch count, in a separate new spec.

The emitted spec selects at most two demos for pretraining and evaluation with
no fine-tuning. Without `--base-spec`, it defaults to 1,000 pretraining epochs
and derives other settings from the current `pretrain_eval.json`; it does not
claim to reproduce an archived run. With `--base-spec`, it preserves that
recipe's epoch count and other hyperparameters.
Preparation includes only pairs with exact matching, executable labels. Missing
pairs are listed in `preparation.json`, and completely unlabeled tasks are omitted.
Lookup includes full original-program runs pinned by the catalog's provenance,
since the shared catalog excludes official ARC2 evaluation-query rows even when
their programs exist. Those ARC2 query pairs belong to the filtered training
supplement in this ARC1 experiment. Remaining pairs also try the owning release's
existing hash-pinned repair compilers. Every recovered program is replayed by the
native engine and must match the complete output before its grids are retained.
The canonical catalog is never edited, and ARC1 held-out query programs are never
requested. Audits distinguish saved labels, compiler recovery and unresolved pairs.
All 400 ARC1 evaluation tasks contribute their demos to pretraining. Their query
inputs and outputs stay out of the training file. After pretraining, evaluation
uses the final pretraining checkpoint directly on those queries, with their
outputs kept in the separate scoring file.

The immutable ZIP is self-contained and pinned by size/SHA-256 in the spec.
Launches only extract it and do not need the sibling engine or label catalog.
The explicit preparation cache is SQLite, keyed by input, program, native
semantics and filter revision, and survives interrupted builds. Reusing an
audit directory avoids replaying completed executions. Packages, caches,
diagnostics and execution specs stay ignored; this source and guide belong in Git.

## Supervised target computation

Set `defaults.predict_intermediate_grids=true` for fresh training on
`input -> selected intermediate(s) -> final output`. The same filtered pool and
per-visit random quantiles select both demonstration and target stages. Targets
without an intermediate remain direct input/output examples. Intermediate grids
end with the existing IO separator; only the final grid ends the chain. Original
input and demonstrations remain accessible throughout the causal target branch.

Target RoPE z planes are input 5, then generated grids 6/7/8 in order. With one
intermediate the final grid is at 7; with none it is at 6. This numbering depends
only on consumed separators and cannot reveal how many future grids remain.
Each grid resets x/y. Cached greedy and speculative decoding use the same raw
grid counter and original target-input anchor. Generation permits at most two
intermediate separators and reserves a three-grid token budget. Only the final
grid enters ARC scoring/voting; generated intermediates remain in candidate
artifacts for analysis. Capacity with two demos rises from 9,301 to 11,161 tokens.

For CE and every MTP head, average token losses within each grid, average grids
within each example, then average examples. This prevents a large workspace or
long trace from outweighing a small final answer. Distributed loss normalization
uses participating examples rather than token counts. Teacher-forced color,
structural and exact-output accuracy continue to describe only the final grid.
Checkpoints use `supervised_intermediate_grid_chain_v1` and cannot be silently
reinterpreted as the earlier final-output-only format.

Prepare a true withheld-demo diagnostic without rerunning any DSL programs:

```sh
venv/bin/python scripts/prepare_intermediate_grids.py \
  --base-spec 'specs/temp specs/APPROVED-FA4.json' \
  --diagnostic-holdouts 64 \
  --package assets_tmp/arc1-filtered-supervised-HOLDOUTS.zip \
  --audit-dir results/codex%2Fintermediate-grid-demos/supervised-preparation \
  --output-spec 'specs/temp specs/supervised-targets-NEW.json'
```

This reserves one labeled pair from each of 64 evaluation tasks with at least
four demos and an available intermediate. Exact I/O duplicates are removed
globally from pretraining. The original evaluation demonstrations, all 419 query
inputs, and scoring-only query solutions remain byte-identical in their own
files. The prepared manifest records parent SHA-256 and withheld I/O identities.
For the current package this leaves 4,457 training pairs across 1,145 tasks.

The additional `intermediate-diagnostic` evaluation runs these same 64 withheld
demo targets twice, using generated intermediates and then supplied gold
intermediates. The original target pair is excluded from its own context. Gold
tokens enter through normal causal cache updates; they are never bidirectionally
prefilled or supplied for official test queries. The artifact reports final-grid
accuracy and its oracle gain separately from the official ARC score. It uses no
augmentation and has a bounded 64-pair budget. Main query evaluation retains the
approved augmentation/voting settings.
