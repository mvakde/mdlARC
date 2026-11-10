# Data generation: synthetic ARC-AGI-2 task families

This folder rebuilds the training data of the mdlARC v3 Kaggle model
(d1024, 37.08 on ARC-AGI-2). That model was trained on one package,
`arc2-family-full-counts-v1-20261024-r1.zip`: 161,798 ARC puzzles in 3,826
task families, all produced by Python generators that language models wrote.
Every generator is here as a plain `.py` file, with the seeds and counts that
produced each puzzle, so the package can be regenerated and checked: all but
79 of its generated puzzles come back byte for byte (see
[Verification](#verification-of-this-release)).

> **Contamination warning.** These generators encode the rules of ARC-AGI-2
> **public evaluation** tasks. 120 variation families are built directly on the
> 120 public evaluation tasks, 1,988 compositions combine two evaluation tasks
> and 418 combine an evaluation task with a training task. Every family,
> including the training-task ones, lists public evaluation tasks among its
> ancestors, because the hand-written few-shot demonstrations used by the
> authoring prompts are evaluation tasks. **Do not train on this data, or on
> anything derived from it, if you report scores on the ARC-AGI-2 public
> evaluation set.** It was built for the ARC Prize 2026 Kaggle competition,
> which scores on hidden tasks that are not in it.

Licence: to be decided.

## What the data is

Two kinds of family, both based on ARC-AGI-2 tasks:

- **Variation families** (1,120 in the package). One per ARC-AGI-2 task (1,000
  training, 120 public evaluation). A family holds the original task plus 42
  new puzzles from a generator that varies the task's inputs and surface while
  keeping its rule.
- **Composition families** (2,706). A generator written for an unordered pair
  of ARC-AGI-2 tasks that invents one new rule using concepts from both
  parents, with 40 to 42 puzzles each. No pair is used twice.

| Package content | Families | Puzzles |
| --- | ---: | ---: |
| Original ARC-AGI-2 tasks (inside the variation families) | | 1,120 |
| Variations of ARC-AGI-2 training tasks | 1,000 | 42,000 |
| Variations of ARC-AGI-2 public evaluation tasks | 120 | 5,040 |
| Compositions, evaluation + evaluation (`eval-eval`) | 1,988 | 83,486 |
| Compositions, training + training (`train-train`) | 300 | 12,600 |
| Compositions, training + evaluation (`mixed`) | 418 | 17,552 |
| **Total** | **3,826** | **161,798** |

The release also includes 187 variation generators that are not in the
package (27 for a filtered ARC-AGI-1 set, 160 for ConceptARC). They were
pinned by the same catalog and are marked `"in_package": false` in the
manifest; `sample.py --all` regenerates them as well.

Sizes: the 4,013 generators total 111.8 MB as source; with the 20 MB manifest
they make a 41.0 MB ZIP;
the package is 181,720,007 bytes zipped and 1.97 GB unzipped (1.87 GB of
puzzle JSON plus a 99 MB catalog).

## How it was made

Every generator was written by a language model and checked by machine. No
person reviewed the individual generators, with the exceptions listed in
[`authoring/README.md`](authoring/README.md#human-input): the six hand-written
summaries and six hand-written reviews used as few-shot demonstrations, and
two small manual repairs.

- **Variations** came from a four-stage pipeline per task: summarise the task,
  write a first generator, review it on its samples, write an improved
  generator. The models were mainly GPT "Sol-class" models (`gpt-5.6-sol`,
  `gpt-6-sol`, some `gpt-6-astra`) at reasoning effort `xhigh`, run in Codex
  CLI. The released generator is the improved one.
- **Compositions** came from a two-stage pipeline per pair: an author stage
  writes the composed rule and generator, and an optional review-and-improve
  stage revises it. 2,622 families were authored by `gpt-6.1-sol` (`xhigh`)
  in Codex CLI. One batch of 84 families,
  `task-composition-v3-claude-arc2-training`, was authored with Claude
  (`claude-opus-5-5`, `xhigh`) in Claude Code; 10 of those also went through
  review and improvement. All other families use the author-stage generator.

The prompts, few-shot examples and stage code are in [`authoring/`](authoring/)
as a reference. LLM outputs are not deterministic, so re-running authoring
produces different generators; the generators released here are the record.

Every generator is self-contained: it imports only the Python standard library
and NumPy, and no generator imports another generator or a helper module, so
each runs on its own from this tree (`manifest/families.jsonl` lists every
generator's imports and records `"dependencies": []`). Every generator
defines `generate_puzzle(seed, num_train, num_test)`,
`validate_generated_puzzle(puzzle)` and `self_test()`; composition generators
also define an input-only `solve(input_grid)`.

### Seeds and counts

- **Variations**: seeds 0 to 41. The generator chooses the number of
  demonstrations (2 to 10) and queries (1 to 4) from the seed.
- **Compositions**: 42 recorded seeds per family, with demonstration/query
  counts from the count policy `composition-counts-v1`. Demonstrations are
  weighted 2:15%, 3:24%, 4:24%, 5:16%, 6:12%, 7:5%, 8:3%, 9:1%; queries
  1:60%, 2:35%, 3:5%; always more demonstrations than queries. The quotas are
  balanced across dataset/stage cohorts and families with schedule seed
  20261024. Of the 113,652 scheduled puzzles, 14 recorded seeds failed on
  every replay and were dropped (at most two per family, 12 families); no
  generator, seed or other count was changed. `validate.py` recomputes the
  whole schedule from the policy.

### Checks that were run

On every packaged puzzle, when the package was built:

- compositions: exact requested counts; grid schema (rectangular, 1 to 30 per
  side, integer colours 0 to 9); no duplicate input within a puzzle; no exact
  copy of a parent pair; the generator's own validator; agreement between the
  generator's input-only solver and every pair; no repeated puzzle across a
  family's seeds; not every pair an identity. Every seed was replayed in a
  second process for 753 families and five representative seeds for the
  other 1,953 (41,389 replays, no mismatches).
- variations: the generator's `self_test()`, its validator and the grid schema.
  At authoring time each variation generator also passed host checks for
  determinism, legal and illegal counts, duplicate inputs and pairs, exact
  copies of the source task, and at least 40 distinct puzzles from 42 seeds.

These checks are structural. They do not prove that a rule is correct or that
the demonstrations are enough to learn it.

## Files

```text
data_generation/
  README.md
  generators/
    variations/<source>/<shard>/<task>.py          1,307 generators
    compositions/<pairing>/<shard>/<A>-<B>.py      2,706 generators
  manifest/
    families.jsonl      one line per family (4,013)
    release.json        count policy, schedule seed, runtime pins, catalog header
  sample.py             re-run generators on the recorded seeds (multiprocess)
  validate.py           schema, counts, duplicates, parent copies, solvers, schedule
  package.py            build the training ZIP from the samples
  fetch.py              download and verify the Release files
  authoring/            LLM prompts, few-shot examples and stage code (reference)
```

`<source>` is `arc2_train`, `arc2_eval`, `arc1_filtered` or `conceptarc`;
`<pairing>` is `eval-eval`, `train-train` or `mixed`; `<shard>` is the first
character of the task id (of parent A for compositions), which keeps every
folder under 400 files. `<A>-<B>` are the two parent task ids.

Each line of `manifest/families.jsonl` has:

- `id`: the family id used in the package catalog, for example
  `arc2_eval/0934a4d8` or
  `composition/task-composition-v3/arc2_eval/000001-446ef5d2-6e4f6532`;
- `kind` (`variation` or `composition`), `in_package`, `source` or
  `pairing`/`dataset`, `batch`, `job`;
- `parents` (the two parent tasks of a composition) and `ancestors` (every
  original task in its lineage, as in the package catalog);
- `generator`: `path`, `sha256`, `bytes`, `stage`, `imports`, `dependencies`;
- `authoring`: models and effort per stage;
- `sampling`: protocol; for compositions also the 42 recorded `seeds` in
  schedule order, the `omitted` rows with their errors, and
  `parent_pair_sha256` (truncated hashes of the parent pairs the original
  check excluded);
- `original`: SHA-256 and pair count of the original task (variation families);
- `samples`: `[seed, demos, queries, sha256]` per puzzle in generation order.
  A family's package instances are its original task (if any) followed by
  these samples, stored as `instances/<sha256>.json`.

## Regenerate and verify

Use the pinned runtime: **CPython 3.13.16, NumPy 2.5.4**, `PYTHONHASHSEED=0`
(the scripts set it for every generator process). The original ARC-AGI-2
tasks come from the public repository
[arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) (`data/`).

```bash
uv venv --python 3.13.16 .venv-data
uv pip install --python .venv-data/bin/python numpy==2.5.4
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git

# 1. Regenerate into data_generation/samples/ (one fresh process per family;
#    --resume continues a stopped run)
.venv-data/bin/python data_generation/sample.py --workers 8

# 2. Check the samples, re-run the composition solvers, recompute the schedule
.venv-data/bin/python data_generation/validate.py \
    --originals ARC-AGI-2/data --solver --require-all

# 3. Build the training package and compare it with the released one
python data_generation/fetch.py --name arc2-family-full-counts-v1-20261024-r1.zip
.venv-data/bin/python data_generation/package.py --originals ARC-AGI-2/data \
    --out data_generation/rebuilt/arc2-family-rebuilt.zip \
    --compare assets_tmp/arc2-family-full-counts-v1-20261024-r1.zip
```

`samples/`, `release-assets/` and `rebuilt/` inside this folder are ignored by
Git.

`sample.py` prints how many puzzles match the manifest hashes and exits
non-zero on any difference; a full run reports the 79 known differences
described under [Verification](#verification-of-this-release), and
`validate.py` lists those three families as hash mismatches.
`package.py` refuses differing bytes unless you pass `--allow-mismatch`, in
which case the rebuilt package holds the regenerated versions of those 79
puzzles. Useful options: `--stratified 60` for a quick
check spread over all family types, `--family ID` (repeatable), `--kind`,
`--pairing`, `--source`, `--all` to include the 187 families outside the
package, `--out DIR` for another output folder. Each family is written to
`samples/families/variations/...` or `samples/families/compositions/...` (one ZIP per family) with a
`records.json` of seeds, counts, hashes and the checks that ran. A different
Python or NumPy version prints a warning (`--strict-runtime` refuses to run)
and may change bytes.

`package.py` writes the same layout the mdlARC loader reads:
`family_catalog.json` plus `instances/<sha256>.json`. It rebuilds each
family's catalog entry (ancestors, instance records, sampling omissions) from
the manifest and the sample bytes; for families whose samples match, the
entry equals the released one. The released catalog also pinned hashes of
private authoring records (`metadata_sha256`); by default the rebuilt catalog
pins the public generator files there instead, so the ZIP hash differs. Pass
`--metadata-from` with the released package to copy those pins as well. To
train on a rebuilt package, put it under `assets_tmp/` and set the spec's
`dataset.pretrain_data.catalog` (`archive_path`, `size`, `sha256`) to the
values `package.py` prints.

Measured on an Apple M5 laptop (10 cores, 8 workers): regenerating all 4,013
families took 91 minutes of wall time and 5.1 CPU-hours. The 3,826 package
families are about 95% of that. A `--stratified 98` check takes under two
minutes.

### Verification of this release

With the pinned runtime, `sample.py --all` regenerated all 4,013 families
with no failures: **168,426 of 168,532 samples are byte-identical** to the
recorded hashes. Every composition puzzle (113,638) matches, and every one
passed the generator-owned checks again (validator, input-only solver,
counts, schema). `validate.py` recomputes the full composition count schedule
from the policy and seed 20261024 and finds all 2,706 families identical to
the recorded schedule.

Three package families (`arc2_train/97c75046`, `arc2_train/d6e50e54`,
`arc2_train/da2b0fe3`; 79 of the package's 160,678 generated puzzles) iterate
over Python sets, so their output depends on the per-process hash seed (in
`97c75046`, a set of direction names). The variation sampler did not pin `PYTHONHASHSEED`, so these puzzles
differ on regeneration, with the same demo/query counts. The released package
keeps the original bytes. One ConceptARC family outside the package
(`conceptarc/ExtendToBoundary6`, 27 puzzles) behaves the same way. All other
variation puzzles, and the 1,120 original tasks rebuilt from ARC-AGI-2
task files, match.

## Download the Release files

The repository's Releases page (tag `v3.1-nov2026`) has:

- `arc2-family-full-counts-v1-20261024-r1.zip`: the training package
  (181,720,007 bytes, SHA-256
  `51d8450883db3467d738b5dbebcd60a763e55fac91335b0afcf0d91f79a3a809`), the file
  the d1024 spec pins;
- `mdlarc-generators-v3.1.zip`: `generators/` and `manifest/` from this folder
  (40,996,904 bytes, SHA-256
  `5e3ecd55f98c5b5b77b90990104f9943c0706df3e8251cbffe8c5035d03eba99`);
- `SHA256SUMS`.

```bash
python data_generation/fetch.py              # both ZIPs into assets_tmp/ (where training reads the package)
python data_generation/fetch.py --name mdlarc-generators-v3.1.zip --dest downloads
```

`fetch.py` downloads from
`https://github.com/mvakde/mdlARC/releases/download/<tag>/<name>` (default tag
`v3.1-nov2026`) and checks each file against the hashes pinned in `fetch.py`
and the published `SHA256SUMS`.

## Training with the package

The d1024 spec reads only the package: `dataset.pretrain_data.catalog` names
`assets_tmp/arc2-family-full-counts-v1-20261024-r1.zip` with its size and
SHA-256, and the loader refuses any other bytes. The spec's
`discovery.composition_release` entry is an identifier that must equal the
catalog's own `discovery` field; no file is read from it. Nothing else from
the generator repository is needed for training.

## Not included

- the development package (the same data with 40 public evaluation tasks
  and their descendants held out, used for tuning) and the later
  80-composition batch; the d1024 model trained on neither;
- the 10,540 per-task prompt and recorded-input records of the authoring
  runs. They are the prompt templates in `authoring/` filled with each task's
  grids, and they contain local paths.

Licence: to be decided.
