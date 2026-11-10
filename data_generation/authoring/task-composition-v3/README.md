# Compose unique task pairs at scale (v3)

For Claude Code web batches on **Claude Code usage**, use the checked-in
`/arc-compose` command instead of the Codex shell launcher:

```text
/arc-compose --dataset arc2/training --batch-size 100 --workers 4
```

See [the Claude web guide](CLAUDE-WEB.md). It dispatches native Opus 5.5 workers,
uses these local v3 checks and needs no API key or local Claude installation.

V3 is the default composition pipeline. `pipelines/task-composition` links to this
folder, so its `run`, `run-fast`, `run-normal` and `export.py` entry points use v3.
Runs continue to save under `results/task-composition-v3`; existing plans keep
their original paths. V1/v2 code and saved runs are preserved locally in the
Git-ignored `archive/compositions/` folder, using their original relative paths.

Create a requested number of composition families from a chosen dataset, with any
number of concurrent workers. Every launcher excludes previously reserved unordered
pairs across the dataset ledger. Automatic batches choose each new pair from the
least-used tasks while forecasting participation across the batch; shared reservations
can instead use an explicitly requested pair list. Each job receives a unique **unordered** pair: A+B
and B+A count as the same combination. Individual parents can be reused in other
pairs. With N eligible tasks there are N(N−1)/2 available combinations.

```sh
# Run just the author stage for the next 100 pairs, up to 8 concurrently.
pipelines/task-composition-v3/run run \
  --dataset arc2/evaluation --batch-size 100 --workers 8 --stage author

# Another dataset and separate output folder.
pipelines/task-composition-v3/run run \
  --dataset conceptarc/training --count 100 --workers 8 \
  --output-root results/task-composition-v3-conceptarc
```

Defaults: `gpt-6.1-sol / xhigh`, the task-variations Python environment, its pinned
SDK and matching bundled Codex CLI, and existing local login. `ARC_PIPELINE_PYTHON` and `CODEX_BIN` select
alternative runtimes. No model calls happen until `run`; no training or deployment
is included.

To continue a saved Claude batch locally with Codex, stop its Claude coordinator
and use a separate output root:

```sh
pipelines/task-composition-v3/run-normal run \
  --dataset arc2/training \
  --continue-from results/task-composition-v3-claude-arc2-training \
  --output-root results/task-composition-v3-codex-arc2-training \
  --count 300 --workers 8 --stage author
```

With `--continue-from`, `--count` is the **dataset total**, including successfully
checked authors in the existing runs. For example, 84 existing compositions leave
216 for Codex. Repeat the same command to resume; increasing the total extends the
Codex plan. Use `prepare` in place of `run` for an offline handoff and pair/evidence
preparation without model calls. Without `--continue-from`, `--count` retains its
usual meaning: the total number of families in that output root.

The handoff rebases cloned paths, verifies the published author artifacts, and
keeps their original models and files in the source folder. Unfinished source
jobs leave the active plan but their drafts stay in that folder and their pairs
remain reserved. The new output keeps the prior plan/ledger snapshots and a
recoverable handoff record. Both folders share the dataset ledger, so allocation
uses successful parent appearances across both runs and never repeats a pair.
Ignored historical seed copies need not be rebuilt merely to count an existing
author; resuming or reviewing those historical jobs still requires their full
evidence. `--stage all` reviews and improves the new Codex families; it does not
review author-only families retained in the source folder.

Use `python3 tools/codex.py update --local` from the repository root to update
all active Codex pipelines and the local CLI. After pulling a version bump on
another device, use `python3 tools/codex.py sync`. Both run compatibility and
isolation checks without model turns. See [updating Codex](../../README.md#update-codex).

For an additional fixed batch on another device while this machine finishes its
current queue, use the [device-batch workflow](DEVICE-BATCHES.md). It reserves
explicit non-overlapping pairs and restores repo-relative paths after cloning.

For a hand-selected list, use the shared worker command's
`reserve --kind composition --pairs-file PATH`, followed by
`work --assignment REQUEST_ID`. This preserves dataset-wide unordered pair
exclusions while choosing exactly the requested pairs instead of the automatic
least-used order. Add `--upgrade-pipeline` when deliberately creating new work
with current code from an older source plan. See the
[exact-pair instructions and checked-in 38-pair command](../../tools/SHARED-WORKERS.md#composition-batches).
Shared composition assignments run the author stage only.

## Selecting a dataset and reserving pairs

`--parents-root` defaults to `results/task-variations`. `--dataset` selects a folder
relative to it, e.g. `arc2/evaluation`, `arc2/training`, `arc1-filtered/training` or
`conceptarc/training`. For a custom flat collection use `--parents-root PATH
--dataset .`. Inputs must be complete **task-variation packages**, not raw ARC JSON:

```text
<parents-root>/<dataset>/<task-id>/
  task.json, summary.xml, generator_v1.py, review.md,
  generator_v2.py, review_coverage.md, provenance.json
```

Incomplete folders are excluded. At least two complete parents are required. The
candidate collection and its source hashes are pinned in `plan.json`. Every entry
point reserves explicit pairs in the shared dataset ledger before preparing evidence.
A+B and B+A are the same pair, including across different output roots. Reservations
and archived attempts keep their pairs unavailable, so retries continue the existing
composition rather than assigning the pair to a second composition.

Task appearances count only successfully checked author stages. Empty prepared
folders, running attempts and failed attempts do not count as generated compositions.
For each new pair, the allocator chooses the lowest-count tasks, forecasts the two
increments, and recalculates. If no unused pair exists within the lowest pool, it uses
the lowest-count available partner. If a lowest-count task cannot get any unused
partner, preparation stops with an explanation rather than skipping that level.
The saved dataset task order breaks ties consistently across seeds and batch sizes.

After allocation, the worker coordinator fills all available worker slots from
the reserved batch. Pairs may run and finish in any order, including pairs sharing
a parent or spanning different forecast participation levels. Least-used selection
applies when choosing pairs, so execution has no participation-level barrier. A failed generation
stops the queue; resuming retries that same job. Only one unfinished author batch is
allowed per dataset, and the dataset lock prevents overlapping runs from racing.
`coverage.json` reports dataset-wide successful appearances.

Use `--batch-size 100 --stage author` for the **next batch of 100 author stages**.
Repeat the exact same command tomorrow: it resumes an unfinished batch or, once
all requested stages have finished, appends the next 100 unique pairs. `--stage
all` (the default) runs author plus review/improvement; switching from author to
all finishes the existing families before a new batch is added.

`--count` remains an alternative **total target number of families**. A new run
requires either `--count` or `--batch-size`; they are mutually exclusive. To
inspect pair choices and evidence without contacting models:

```sh
pipelines/task-composition-v3/run prepare \
  --dataset arc2/evaluation --count 100 --seed 42
```

Preparation executes unchanged parent generators locally to save a varied solved
example for each parent, using v2's bounded broader-mode selection or reproducible
random fallback. It snapshots originals, variations, lineage and four worked review
examples distinct from both parents. Each parent-sampling subprocess has a 300-second
timeout. All requested jobs are prepared before any model work starts.

```sh
# Resume the saved plan; completed stages/jobs skip model work.
pipelines/task-composition-v3/run run --workers 8

# Append 100 new unique pairs to the existing 100-job plan.
pipelines/task-composition-v3/run run --count 200 --workers 8
```

Increasing `--count` extends the saved plan with new unique least-used pairs without changing old jobs.
Decreasing it, exceeding the dataset's pair capacity, changing pinned settings/code,
or changing a source parent needed by an extension raises an error. Use a new output
root for a different dataset, seed, model, effort or review sample count. A folder
lock prevents simultaneous prepare/run invocations on the same output root. Pair
reservations survive failures; a failed job retries its existing pair.

## Exactly two model stages

1. **Author:** studies both originals and their selected variations, develops **one**
   creative composition and writes `composition.md`, the rule and first generator.
   The prompt encourages new connecting concepts and broad structural variation,
   while requiring both parents to affect the answer. Local checks run on 42 seeds.
2. **Review and improve, in one call:** a fresh session receives the first generator,
   rule, parents, **five solved sampled puzzles** by default, and **four worked human
   reviews**. It first writes a complete review with corrections and creative,
   family-specific diversity requests. Examples establish specificity; the model is
   strongly encouraged to go beyond their variation axes. Only after the review is
   complete does it implement the improved generator, final rule and measured coverage.

Choose a different small review set with `--review-samples N` (1..42) on the initial
prepare/run. Samples are deterministically spaced through the 42 recorded author
seeds, including seed 0. The reviewer can generate targeted diagnostic cases when
useful. The host still verifies 42 seeds for each generator.

**Review-before-improvement barrier:** the second workspace starts without an editable
final generator or final specification. The model completes `review.md`, then runs
`finish_review.py` in that same call. The helper records the review's hash/time and
opens the editable starting generator/specification. It refuses to start if improved
deliverables already exist. Host checks verify the checkpoint and preserve the review
across repairs/resumes. The model then writes `generator.py`, `task_spec.md` and
`review_coverage.md`. This enforces the recorded artifact workflow; it is not a
claim about the model's internal reasoning order.

Missing deliverables or execution/check failures permit up to two additional repair
turns per stage per invocation, using the same session. Thus the normal successful
path has two model calls; repairs or interrupted-stage resumes can add calls. There
is no separate review call, independent solver, blind solving or approval verdict.
A resumed second stage preserves a completed review and continues improvement.

## Checks, results and cancellation

Generators retain the v2 interface:

```python
generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict
solve(input_grid: np.ndarray) -> np.ndarray
validate_generated_puzzle(puzzle: dict) -> None
self_test() -> None
```

Local checks cover determinism, 42 recorded seeds, legal explicit counts (2..10
train, 1..4 test, train > test), illegal counts, grid bounds/colors, duplicate inputs,
exact parent copies, agreement with the generator-owned input-only solver, its
validator and `self_test()`. Descriptive structure measurements and request-by-request
review coverage remain inspectable. These checks are not independent semantic proof.

Saved samples explicitly request 3–6 demos and one query, with target percentages
40/40/15/5, balanced per family. Generator signature defaults do not select saved
sample counts. Historical samples/checkpoints retain their original bytes.
For an existing completed assignment, resample its unchanged generators offline:

```sh
pipelines/task-variations/.venv/bin/python pipelines/task-composition-v3/resample_batch.py \
  --assignment ASSIGNMENT_ID --release data/compositions/releases/RELEASE_NAME \
  --demo-weights 3:40,4:40,5:15,6:5 --workers 20
```

This keeps all 42 recorded seeds per family, stores one ZIP per family and one
primary puzzle under `tasks/`, and checks every puzzle using the generator-owned
validator and solver. The primary puzzles are also balanced across families.
Count boundaries and primary seeds get separate-process byte replay. Source
generators, prior samples and checkpoints stay unchanged; no model calls occur.

```text
results/task-composition-v3/
  plan.json                    pinned dataset, candidate hashes and assigned jobs
  000001-<A>-<B>/
    parents/A/, parents/B/     originals and selected variations
    composition.md
    generator_v1.py, task_spec_v1.md, task_v1.json, validation_v1.json
    review.md, review-complete.json
    generator_v2.py, generator.py, task_spec.md, review_coverage.md
    validation.json, task.json
    samples/                   30 final solved puzzles, JSON + digit-row text
    checkpoints.json, provenance.json, status.json
    llm-inputs/evidence/examples/
    llm-inputs/author/{inputs,workspace,checked-samples}/
    llm-inputs/review-improve/{inputs,workspace,checked-samples}/
    saved-outputs/<stage>/attempt-N/
```

`--workers` limits concurrent compositions, with at most one active model call per
worker. Ctrl+C stops active worker/Codex subprocess groups and preserves completed
stages. Runtime/authentication failures stop scheduling and preserve progress.
Rerun the same command to resume. Completed artifacts are hash-checked before skip.
States are `running`, `author_completed`, `completed`, `failed`, `interrupted`; `completed` means local
workflow checks passed, not independent semantic approval.

Git retains published artifacts, plans, coverage, parent snapshots, saved inputs,
repair history, model turns and dataset pair schedules/reservations. Bulk seed
samples, copied worked examples, locks and temporary workspace contents remain
ignored. Backup copies of existing model turns and draft deliverables live in
`llm-inputs/<stage>/saved-output/`, with source hashes in `backup-manifest.json`;
these copies are separate from the live workflow state. Keep a separate full artifact archive for exact historical verification
or interrupted-session recovery; a clone alone is not a complete run snapshot.

Export final samples without running generators or contacting models:

```sh
python3 pipelines/task-composition-v3/export.py
# For a separate trial:
python3 pipelines/task-composition-v3/export.py \
  --source-root results/task-composition-v3-conceptarc
```

Exports go to `results/exported-datasets/task-composition-v3/<dataset group>/<job>/`.
Known dataset groups retain the existing export names; custom dataset paths retain
that relative name (`.` becomes `custom`). The exporter checks saved sample hashes.
Keep parent dataset/split lineage, particularly evaluation-set lineage, when using
the output to train or evaluate a model.

```sh
pipelines/task-variations/.venv/bin/python \
  pipelines/task-variations/tools/visualiser/server.py \
  --source results/task-composition-v3 --port 8767

# Offline toy fixtures, real local generator checks, no model/network calls:
pipelines/task-variations/.venv/bin/python -m unittest discover \
  -s pipelines/task-composition-v3/tests -v
```

Only final samples appear in top-level `samples/`; v1/history remain in excluded
model-input folders. **Show originals** resolves the two parent originals.

## How this differs from the older review workflows

Task-variations makes all 42 saved samples available, but its prompt asks for
representative/unusual inspection and does not prescribe reading every one. V2
copies all 42 and explicitly asks to read all samples. V3 provides five by default,
uses four worked reviews instead of two, and merges review plus improvement into
one model call after the author stage. V1 and v2 remain available unchanged.

## Explicit Fast service tier

Use `run-fast` in place of `run` to explicitly request Codex Fast for every worker:

```sh
pipelines/task-composition-v3/run-fast run \
  --output-root results/task-composition-v3-next-30-batch-3 \
  --stage author --workers 10
```

The launcher sets `CODEX_BIN` to the local `codex-fast` wrapper, which prepends
`-c 'service_tier="fast"'` before the SDK arguments. Keeping every config
override before `app-server` preserves the global permission override list. This preserves the model,
reasoning effort, permissions and existing prepared-plan hashes. The Codex config
reference defines Fast as the `priority` request tier:
https://developers.openai.com/codex/config-reference/.
Fast and normal wrappers default to the same pinned CLI through `pipelines/codex`.
`ARC_CODEX_REAL_BIN` can select another installed Codex executable.

For explicit normal speed, use `run-normal` instead. It prepends
`-c 'service_tier="default"'` with the same permission-preserving argument order,
so a user-level priority preference does not turn the batch into Fast mode:

```sh
pipelines/task-composition-v3/run-normal run \
  --output-root results/task-composition-v3-next-60-batch-5 \
  --stage author --workers 20
```

## Mandatory scheduling and cleanup

The normal `run`, `run-fast`, `run-normal` and `prepare_next.py` entry points all use
the same mandatory unique-pair / least-used policy. There is no round-robin bypass.
For 400 more author compositions, use:

```sh
pipelines/task-composition-v3/run run \
  --dataset arc2/evaluation --batch-size 400 --stage author --workers 20
```

Repeating this command resumes unfinished work; after all requested stages finish,
it adds the next 400. Preparations forecast usage to select a balanced batch, but
only successful author checkpoints appear in the reported counts. Once selected,
the reserved pairs run concurrently up to the requested worker limit.

`prepare_next.py` is an optional way to put a new batch in a separate output folder.
It uses exactly the same ledger and policy, and requires the previous author queue
to be finished. The compatible `--selection least-used` flag is optional; other
selection modes are rejected.

```sh
pipelines/task-variations/.venv/bin/python pipelines/task-composition-v3/prepare_next.py \
  --source-root results/task-composition-v3 --count 60 \
  --output-root results/task-composition-v3-next-batch
```

`cleanup.py` archives unfinished jobs, removes them from live plans, retains their
pair history to prevent reuse, and rebuilds counts from verified author checkpoints.
It also migrates the supported legacy plans to the mandatory policy. Completed
generators and all their existing evidence remain unchanged. Cleanup requires a
fresh archive folder and refuses to touch a running dataset. Archives preserve
original plans, coverage, the ledger and all cancelled job folders locally; keep
that archive separately if recovery on another machine matters.

```sh
pipelines/task-variations/.venv/bin/python pipelines/task-composition-v3/cleanup.py \
  --source-root results/task-composition-v3 \
  --archive-root results/.composition-v3-archive/cleanup-example
```
