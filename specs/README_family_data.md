# ARC2 family-sampled data

`arc2_family_development.json` uses the approved `arc2_dev80_val40_v1` split.
The tracked split is a byte-for-byte copy of
`results/scaling-ladder/experiments/arc2-development-split-v1.json`, including its
public-source content pin. The split is never selected again during preparation.
`arc2_family_full.json` uses every public ARC2 training and evaluation family.
Public query answers belong to training in full mode; public evaluation and
adaptation are disabled in that config.

Both configs derive their model, attention/context, optimizer, augmentation,
training duration and decoding settings from the default scaling-ladder spec,
`pretrain_eval.json`: FA4, zero attention dropout, two-lookup pair-group attention,
half-or-more demo sampling, context-palette augmentation, no task/dihedral/simplex
embeddings, batch 64, 330 epochs, NorMuon 0.004, AdamW 0.00072, and automatic
speculative decoding. Embedding/feedforward dropout remain 0.1, and native CUDA
validation is enabled. All default training and evaluation settings are preserved;
the variants change the dataset, run name and evaluation routing.

`arc2_family_development_large_default.json` and
`arc2_family_full_large_default.json` provide the same respective data modes
with the [large-default model](README.md#large-default-specs): 24 layers, width
1920, SwiGLU FFN 5120, 30 heads, and approximately 1.064B trainable parameters.
The development large spec uses the verified B300 settings: physical batch 12
with accumulation 5 (effective 60), evaluation batch cap 150 and prefill token
budget 262144. The full large spec retains batch 32 with accumulation 2
(effective 64), whose native capacity remains unverified. Use the large recipe
as `--spec` during package preparation to emit its executable temp spec. Changed
model or context sizes still require a fresh capacity check.

The main [two-demo development spec](arc2_family_development_two_demo.json)
retains the saved B300 launch's pinned counts-v1 package and training settings,
with width 1536, FFN width 4096, and 24 heads. Both training and evaluation use
two eligible demonstrations, keeping fewer when necessary. This spec is already
package-bound and can be selected directly without preparing a temp spec.

Context-palette training uses each selected candidate's known palette and only
its sampled demonstrations plus target input as visible evidence. Its color
plans balance actual visits to that physical candidate, with replayable D4
cycles. Plans and puzzle payloads are cached lazily; family selection, target
pairs and demo visits are unchanged by augmentation or worker assignment.

Development scores `pretrain.final` on the 40 held-out tasks with the default
leave-one-out, 80-augmentation decoding recipe. As in the default spec, there is
no fine-tuning/adaptation stage. Held-out demonstrations are available as query
context, and query outputs remain separate scoring data. Full mode includes all
public answers in pretraining and disables evaluation and scoring. These modes
retain the approved family sampling and leakage boundaries below. Neither config
changes any infrastructure or spending policy.

The four reusable family recipes pin the composition count release
`counts-v1-20261024`, while remaining unbound to an individual training package.
New executable full/development versions are:

- `specs/temp specs/arc2-scaling-ladder-full-counts-v1-20261024-r1.json`
- `specs/temp specs/arc2-scaling-ladder-development-counts-v1-20261024-r1.json`
- `specs/temp specs/arc2-scaling-ladder-full-large-default-counts-v1-20261024-r1.json`
- `specs/temp specs/arc2-scaling-ladder-development-large-default-counts-v1-20261024-r1.json`

Historical executable versions using the earlier Oct-24 packages remain:

- `specs/temp specs/arc2-scaling-ladder-development-20261024-r1.json`
- `specs/temp specs/arc2-scaling-ladder-full-20261024-r1.json`

These launch specs reuse the existing checksum-pinned puzzle ZIPs. The earlier
`arc2-family-*-20261024-r1.json` launch specs preserve the previous FA2 recipe;
they are retained as historical preparation records. Changes to model settings
do not require recovering or regenerating the same puzzle package.

## Epoch contract

Each enabled original parent has one candidate pool: its solved original plus
all saved variation instances. Choose exactly one candidate each epoch without
replacement within a cycle. Each family gets a fresh seeded shuffle every cycle;
consecutive orders differ whenever the pool has more than one candidate. The
family ID and cycle determine the order, so resume and out-of-order access replay
the same selections. Every eligible composition is a separate family with
its own cycle over saved instances. The original has the same frequency as
one saved seed; it is not an additional static example. Every solved train and
test pair in the chosen instance is a training target, using the existing
context/augmentation recipe. Batch boundaries do not define instances.
A final short batch/accumulation window is retained so pairs cannot disappear.

With only ARC2 enabled, development has 1,000 + 80 original-family selections
per epoch, plus one selection for each eligible composition family. Full mode
has 1,000 + 120 plus its eligible compositions. Pair and token counts vary with
the selected instances. Candidate counts, composition counts and seed counts
are inventory facts, not constants in the sampler.

`pretrain_data.sources` explicitly enables sources. Append `conceptarc` and/or
`arc1_filtered` to include those planned stacks, with the same original-plus-
variations family rule. They are disabled in the supplied ARC2-only configs,
not discarded by the implementation. The stage policy explicitly accepts
`author_completed` and `completed`. Author checks are not independent review;
per-instance `stage` provenance and eligible-stage counts survive packaging.
Changing to reviewed-only is an explicit recipe change.

## Prepare a new immutable inventory

Inventory is an explicit local action; this offline count-release preparation
requires no record sync. It never reserves work, generates samples, executes generators,
starts a watcher, accesses a checkpoint or starts training:

```sh
venv/bin/python scripts/prepare_family_data.py \
  --spec specs/arc2_family_development.json \
  --synth-root ../arc-synth-data \
  --audit-dir results/scaling-ladder/family-data-NEW
```

The source configs are complete recipes but intentionally have no fabricated
package identity. An unbound recipe cannot train or enter a remote source
snapshot. To make an executable pinned spec once the full corpus is recovered,
repeat discovery into a fresh audit directory and add:

```sh
  --package assets_tmp/family-data-NEW.zip \
  --output-spec 'specs/temp specs/arc2-family-NEW.json' \
  --run-name arc2-family-NEW
```

Select the full config for the final-data inventory. Without an explicit composition release pin, fresh builds discover accepted
composition job statuses beneath the selected v2/v3 result roots. The current
release-pinned recipes include exactly the frozen family set; newer accepted
families require a fresh release and package. No IDs, expected
composition count, or historical composition ZIP is baked into the recipe.
Mixed-split `arc2/mixed` plans resolve their qualified `arc2/training/<id>` and
`arc2/evaluation/<id>` parents separately and report those composition families
under `arc2_mixed`. Bare or unknown mixed-parent IDs fail closed; both parents
must belong to enabled sources. The usual held-out/context closure still applies.
Only accepted saved stages are eligible. Planned/running jobs without an
accepted saved stage are not training data.

Packaging requires every selected hash-pinned sample and required lineage
snapshot. It never substitutes the locally available subset. The audit catalog
lists exact missing paths and expected SHA-256 values, corrupt payloads,
metadata blockers and quarantine reasons. Bulk samples are ignored in
arc-synth-data and are deliberately absent from record sync. Having generators
or one representative puzzle per family does not mean every recorded seed is
materialized. A metadata inventory is not corpus clearance.

## One-time recovery and executable package

Use an explicit recovery cache to reuse saved copies/ZIP members by checksum,
reconstruct redundant protected text/example copies and, when requested, run
the saved generators for missing recorded seeds. This creates separate derived
objects under ignored `results/`; it never edits the source manifests or
arc-synth-data files. Generator code is checked against its saved hash, the first
generated seed is repeated as a sanity check, and every output must match the existing sample hash. A
mismatch remains missing and blocks packaging. No new seed pool is invented,
and no model calls, generation/authoring queues, training or VM launches occur.
Some historical generators iterate string sets, so the puzzle seed alone does
not determine output across processes. Recovery can try up to 128 fixed Python
hash seeds within 120 seconds, still accepting only the original sample hash.
This never repins different output. The final ZIP removes that runtime dependency
from training and resume; generator execution belongs only to preparation.

For example, from the repository root (choose fresh package/audit/spec names):

```sh
venv/bin/python scripts/prepare_family_data.py \
  --spec specs/arc2_family_full.json \
  --synth-root ../arc-synth-data \
  --audit-dir results/scaling-ladder/family-data-NEW/full \
  --recovery-cache results/scaling-ladder/family-data-NEW/recovery \
  --recovery-archive /path/to/existing/task-variations-v1.zip \
  --regenerate-missing --recovery-workers 4 \
  --package assets_tmp/arc2-family-full-NEW.zip \
  --output-spec 'specs/temp specs/arc2-family-full-NEW.json'
```

`--recovery-archive` is optional and repeatable. Omit `--regenerate-missing` for
copy/reconstruction only. Repeat with the development recipe and fresh output
paths, reusing the same recovery cache; all development leakage checks still
run on every recovered sample. The reusable source recipes intentionally stay
unbound. Launch the emitted **temp spec**, whose package identity is pinned.
Both source snapshot builders already upload the selected ZIP, and the remote
data builder verifies and extracts it. Training and resume never run generators
or rescan the sibling checkout. Freeze and reuse this package for that run.

Recovery stores one copy of each needed byte hash, streams one file/ZIP member
at a time, and limits subprocesses to `--recovery-workers` (default 2, maximum
8) with 300 seconds per generator family. A resumed recovery skips cached
objects after verifying them. `generation.jsonl` records every attempted family
and any mismatches/timeouts. Allow space for the recovered raw corpus plus the
compressed package; no second full raw staging tree is created. ZIP timestamps
and permissions are fixed so unchanged inventories produce the same bytes with
the same Python/zlib runtime. Recovery of historical model sessions is separate;
unreproducible bytes still require a backup. Checkpoints never pass through the Mac.

The package contains the frozen catalog and deduplicated-by-byte-hash JSON
instance files. It contains no checkpoints, logs, generators or credentials.
Only one instance payload is streamed into the ZIP at a time. Catalog metadata
is O(saved instances + provenance records); payload disk cost is one compressed
copy of each distinct selected instance. Building the training dataset extracts
one raw copy of each distinct packaged payload under the ignored run data
directory. Peak preparation disk is the existing saved corpus plus the new
package and catalog; peak run extraction adds the raw selected corpus. No
full-corpus temporary copy is staged during packaging. Check actual recovered
byte sizes before a future full build; missing bytes prevent a reliable total
size estimate today.

## Leakage and resume

Preparation pins all original source contents, the exact split bytes, sample
membership/hashes, stage policy and provenance records. Held-out original
families and their exact-pair aliases are excluded. Composition dependencies
are followed recursively, including consumed author/review context and parent
generator context. Either held-out parent excludes the composition and all its
descendants. These are directed dependencies; unrelated families are never
merged through an undirected composition graph. Missing dependencies, cycles,
unknown ancestry and changed provenance fail closed. Incomplete original
source families block readiness instead of quietly shrinking the source.

Historical direct-parent jobs use their saved plan/status agreement, protected
stage inputs, original task snapshots and recorded parent authoring context.
A nontrivial composition can provide `lineage.json` with `complete: true` and
an `ancestors` list of namespaced original/composition family IDs; every listed
ancestor must resolve recursively. This adds dependencies and cannot override
inferred ones. Missing protected payloads with known hash pins are reported as
pending verification and block packaging. Unknown lineage is quarantined.

All present selected payloads are compared against held-out/alias solved-pair
fingerprints. Any match excludes its entire family and descendants. A package
cannot be created until this check covers the entire selected corpus. Exact
fingerprints do not establish absence of semantic or transformed duplicates.

A package's size/SHA-256, catalog hash, seed and sampling settings enter checkpoint
metadata. Completed epoch position restores the same shuffled-cycle visits;
new data is never scanned during training or resume. Run data cannot switch
catalogs in place, including when `skip_data_build` is used. Refresh means an
explicit new inventory/package and a new run identity. Colab/Vast source
snapshot builders carry and verify the selected package without changing any
rental, spending, credential or cleanup policies.


## Explicit composition count releases

`pretrain_data.discovery.composition_release` selects a separate immutable
arc-synth-data release using `{"path":"data/compositions/releases/NAME/release.json","sha256":"..."}`.
Preparation verifies that manifest, its frozen inventory, every family archive,
the original generator/plan/status/ancestry pins and each selected puzzle hash.
It preserves the existing composition family IDs, stage provenance, parent
closure, held-out exclusions and exact-pair leakage checks. A newly accepted
source family cannot silently join a pinned release. Original-family variation
pools retain their existing meaning.

The release freezes all 42 recorded stage seeds and their explicit demo/query
counts at a pinned Python/NumPy/hash-seed runtime. A v2 release separately pins
authorized sample omissions, retaining their exact seed, counts and failure
evidence in the catalog. Every retained puzzle receives full primary validation.
Earlier families receive full independent replay; the user-approved later policy
replays five representative seeds per family. Unreplayed seeds have reduced
determinism coverage, recorded explicitly in the release manifest.
Preparation reads its ZIP members directly and packages ordinary puzzle JSON;
it does not fabricate historical checkpoints or execute release generators.
Training/source snapshots remain self-contained and use the new package pin.
Historical r1 packages, catalogs, launch specs, records and experiments are
preserved. An unmodified old recipe without this optional field still uses
historical checkpoint discovery.

The count policy has demos 2..9 at 15/24/24/16/12/5/3/1 percent, queries 1/2/3
at 60/35/5 percent, with demos greater than queries. Marginals apply to the
complete release; development leakage exclusions can shift the retained
distribution slightly. Generator-owned validation and bounded small-count
inspection do not establish universal teaching sufficiency.

Use `scripts/prepare_family_data.py` with a release-selecting recipe and fresh
audit, ZIP and temp-spec names as above. Reuse existing exact-byte recovery
objects or r1 ZIPs for unchanged original variations/lineage. No regeneration
of those assets is necessary when their byte pins are available. Packages and
audit catalogs are generated/ignored; reusable source/tests/docs stay in Git.
This offline workflow performs no record synchronization, model calls,
training, remote launches or transfers. Synchronization remains pending when
the user has explicitly requested local-only work.

The active release retains 113,638 validated composition puzzles in 2,706
families. Its general authorized minimum is 40 samples per family: ten families
have one diagnosed omission and two have two. Original variations, held-out
family definitions and historical packages retain their original byte pins.

Final package verification is recorded in
`results/scaling-ladder/experiments/arc2-family-counts-v1-ready-20261024.json`.
Full mode has 3,826 families and 161,798 candidate instances, including all
113,638 composition samples. Development has 2,546 families and 108,008
candidates, including 61,568 composition samples. Existing original-family
records, historical ZIP hashes and development exclusion records are unchanged.
The two ZIPs total 299,376,500 bytes. Large-default execution configs reuse the
same mode-specific ZIPs; no duplicate dataset package is needed.

With `--recovery-cache`, preparation now reads cached bytes during the first
audit. It repeats discovery only if missing payloads actually required recovery.
A complete cache is audited once; packaging still hashes every selected payload
and rechecks source metadata before publication.
