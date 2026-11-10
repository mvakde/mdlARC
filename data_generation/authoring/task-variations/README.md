# Task variations

Use `./run STAGE [ACTION] --dataset DATASET --task NAME`.

| Stage | Output |
| --- | --- |
| `summarise-task` | `summary.xml` |
| `generate-from-summary` | `generator_v1.py` |
| `review-first-generation` | `review.md` |
| `generate-from-reviews` | `generator_v2.py` and review coverage |
| `produce-programs` | Canonical supervised DSL programs, traces and inspection view (pilot) |

The optional program-label stage reads existing reviewed artifacts and has its own
`dry-run`, `run`, `list`, `inspect`, and `verify` actions. See the
[program-label guide](src/05-programs/README.md). Its `verify` checks only newly generated
program artifacts; it does not repeat the task-variation provenance audit.

From the repository root, run ten reviewed families with ten workers:

```sh
pipelines/task-variations/run produce-programs --batch-size 10 --workers 10
```

`--batch-size` selects families; `--count` remains puzzles per family (default 42).
Use `--all --workers 10` for every reviewed family, or repeat `--task NAME` and add
`--workers`. Slots refill as families finish, each family's compiler cache and
artifacts stay isolated, and inspection renders once after the batch completes.
`produce-programs list --batch-size 10` previews selection without model calls.

For the full 400-task ARC1 training pool, use
`produce-programs --dataset arc1 --split training --batch-size 10 --workers 10`.
This selects original ARC1 training IDs from the existing reviewed ARC2 storage,
with matching IDs in ARC1 filtered storage as a fallback, and saves program
artifacts under `arc1/training`; `arc1-filtered` remains a
separate collection. Only `produce-programs` supports this full ARC1 view.

For all 400 ARC1 evaluation families, label their saved reviewed v2 variations:

```sh
pipelines/task-variations/run produce-programs --dataset arc1 --split evaluation \
  --all --resume --workers 30 --count 42 --output-root results/task-programs-v2
```

This searches both source splits in ARC2 and ARC1-filtered storage by original
ARC1 evaluation membership, saving compilers and bundles under `arc1/evaluation`.
Add `list` after `produce-programs` to preview selection without model calls.
Repeat the command to resume. Add `--original` in a later run to reuse the same
evaluation family compilers for all original demo/query pairs.

The dataset flag selects family IDs; saved variations remain the default label
source. Add `--original` to label each original task's demo and query pairs once:

```sh
pipelines/task-variations/run produce-programs --dataset arc1 --split training \
  --original --all --resume --workers 10 --output-root results/task-programs-v2
```

Use your existing program output root to reuse its family compilers. Original
bundles are saved separately under `runs/original/`; existing variation bundles
remain available. `--original` ignores `--count` and `--start-seed`, and cannot be
combined with `--fresh`. Add `list` to preview selection without model calls.
See the [program guide](src/05-programs/README.md#programs-for-the-original-tasks)
for source ordering, verification and compiler repair behavior.

Continue the remaining full ARC1 training families with at most 30 workers using
`produce-programs --dataset arc1 --split training --all --resume --workers 30`.
`--resume` skips verified bundles for the requested puzzle batch and preserves
unsuccessful attempts before replacement. Use the same output root as before.

For large program batches, start the local viewer with
`produce-programs serve --output-root results/task-programs --port 8766` and open
the printed URL. It loads one family and selected execution traces at a time,
and includes failed-family diagnostics.

Actions: `run`, `dry-run`, `sample` (generator stages), `list`.
Use `--all` for bulk work, `--force` to replace a selected output, and
`--model` / `--effort` to select model settings.

The code lives directly in `src/`, in pipeline order:

```text
src/
  01-summary/    # summarise the original task
  02-generate/   # build generator_v1.py from the summary
  03-review/     # review the first generator and write review.md
  04-improve/    # build generator_v2.py using the review
  05-programs/   # author family compilers and verify supervised DSL labels
  common/        # shared loading, storage, execution and isolation
  cli.py         # shared command-line interface
```

`src/task_variations.py` is a small import adapter so Python can keep using
`task_variations.summary`, etc., despite the numbered folder names.

Supporting folders:

- `acceptance/`: protected reference solvers and coverage requirements for selected
  ARC-2 evaluation tasks. Improvement workspaces receive these as read-only
  checks, and the host independently verifies correctness and variation.
- `configs/`: one settings file per dataset, including models, prompts, examples
  and sampling settings.
- `datasets/`: original input datasets, selected/prepared inputs, and `prepare.py`
  for preparing result folders offline.
- `examples/`: fixed solved summaries and human review demonstrations, including
  before/after generators and samples, supplied to the models as examples.
- `prompts/`: reusable model instructions for generation, review and improvement.
  Summary instructions are defined in `src/01-summary/prompt.py`.
- `tests/`: automated regression checks for validation, acceptance, output
  preservation, sampling and the four-step workflow.
- `tools/`: helper scripts for isolation checks, dataset building, the visualiser
  and the manual review interface.

The shared environment is `.venv`. All Codex stages, including program authoring
and repairs, use the CLI bundled with the SDK pinned in `pyproject.toml`.
From the repo root, `python3 tools/codex.py update --local` updates the shared
SDK/CLI and the local CLI; `python3 tools/codex.py sync` installs/checks the
checkout's existing pin. See [updating Codex](../../README.md#update-codex).

Export saved final v2 puzzle JSONs after sampling:

```sh
python3 pipelines/task-variations/export.py
```

Run from the repository root. This collects `samples/seed-*.json` into
`results/exported-datasets/task-variation/<dataset-group>/<task>/<task>seedN.json`.
See [the export guide](../../results/exported-datasets/README.md) for the dataset
groups, custom paths and rerun behavior.

All outputs and saved model input packages live under
`../../results/task-variations/<dataset>/<split>/<task-name>/`.
Successful replacements preserve previous outputs in `saved-outputs`.
Failed runs preserve selected outputs and keep their attempted inputs for inspection.
Dry runs use `llm-inputs/prepared-<stage>`.

Model work happens in a fresh temporary Git folder with restricted filesystem
permissions. Codex cannot read other tasks or the repository. The temporary folder
is removed when the run ends; the exact starting files and prompts are retained.

See the repository README for setup, commands, results and verification.
