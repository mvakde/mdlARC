# Authoring pipelines (reference copy)

This folder shows how the released puzzle generators were written by language
models. It is a reference copy of the two authoring pipelines, kept so readers
can see the prompts, the few-shot examples, the stage order and the checks each
generator had to pass.

You do not need anything here to regenerate the data. The released generators
in `../generators` are the record, and `../sample.py` regenerates the puzzles
from them.

## Language model outputs are not deterministic

Re-running these pipelines will not reproduce the released generators. The same
prompt, inputs and model settings give different generators on each run. Treat
this code as documentation of the method, not as a way to rebuild the release.

## Pipeline 1: task variations (`task-variations/`)

For each source ARC task, four model stages run in order. Each stage runs in a
fresh, isolated temporary workspace that can see only that task's files.

| Order | Stage name | Output | Code |
| --- | --- | --- | --- |
| 1 | `summarise-task` | `summary.xml` (rule, input generation, variation guidance, train/test shift, solution steps, key insight) | `src/01-summary/` |
| 2 | `generate-from-summary` | `generator_v1.py` | `src/02-generate/` |
| 3 | `review-first-generation` | `review.md` (a written review of v1 and its samples) | `src/03-review/` |
| 4 | `generate-from-reviews` | `generator_v2.py` plus `review_coverage.md` | `src/04-improve/` |

`generator_v2.py` is the final generator for a variation family. Shared loading,
storage, sampling, Codex invocation and workspace isolation live in
`src/common/`; `src/cli.py` is the command-line entry point (`run`, `run.py`).
`configs/` holds one settings file per source dataset (ARC-2, a filtered ARC-1
set and ConceptARC).

Prompts:

- Summary instructions are written in code, in `src/01-summary/prompt.py`.
- `prompts/generate.txt`: first generator, from the task and its summary.
- `prompts/review.md`: review instructions (the task prompt itself is in
  `src/03-review/workspace.py`).
- `prompts/improve.txt`: improved generator, from the review (the workspace
  prompt is assembled in `src/04-improve/prompt.py`).

Checks each generator had to pass before a stage was accepted:

- static checks on the code (required functions and signatures, banned imports);
- determinism (the same seed and counts give the same puzzle);
- legal and illegal train/test counts (illegal requests must raise);
- grid bounds (1 to 30 per side) and colours (0 to 9);
- no duplicate inputs or pairs within a puzzle;
- no exact copy of the source task's pairs;
- the generator's own `self_test()`;
- for three selected tasks, protected reference solvers in `acceptance/`, given
  to the improvement stage as read-only checks and re-run by the host.

Up to two repair turns were allowed per stage when a check failed.

## Pipeline 2: task composition v3 (`task-composition-v3/`)

Each job takes an unordered pair of finished task-variation families (A and B)
and asks a model to invent one new task family that uses concepts from both.
No pair is used twice. Two model stages run in order:

1. Author (`prompts/author.md`): studies both parents' original tasks,
   selected variations, summaries and generators, then writes
   `composition.md`, a rule specification and a first generator.
2. Review and improve, in one call (`prompts/review-improve.md`): a fresh
   session reviews the first generator using five solved samples and four worked
   human reviews, writes `review.md`, runs `finish_review.py` to checkpoint the
   review, and only then writes the improved `generator.py`, final specification
   and `review_coverage.md`.

`run.py` holds the stage logic and checks; `pair_policy.py` allocates unused
pairs from the least-used tasks; `parent_sample.py` runs a parent generator to
save a solved example; `verify.py` runs the host checks; `v2_storage.py` holds
file helpers; `inspect_puzzles.py` is a helper staged into model workspaces.
`entrypoint.py` and the `run` shell script are the original launcher.

Checks each composed generator had to pass:

- determinism over the 42 recorded seeds;
- legal and illegal train/test counts;
- grid bounds and colours;
- no duplicate inputs;
- no exact copy of a parent's pairs;
- agreement between the generator's own validator and its own input-only solver;
- the generator's own `self_test()`.

These checks are structural. There is no independent semantic proof that a
composed rule is correct, and no human reviewed individual composed generators.

### Claude Code batch (`task-composition-v3/claude-code/`)

One batch was written with Claude Code on the web instead of Codex CLI.
`claude_web.py` and the `claude-web` launcher prepare assignments and run the
same v3 checks; the model work was done by Claude Code's own subagents.
The coordinator procedure (`claude-code/skills/arc-compose/SKILL.md`) and the
worker definition (`claude-code/agents/arc-composition-worker.md`) are included
as inert reference files. They are deliberately not under a `.claude` folder,
so they do not activate in this repository.

## Few-shot examples

Both sets of examples were written by hand and supplied to the models as
demonstrations.

- `task-variations/examples/summary/`: six hand-written task summaries
  (`summary.xml`), each with its source task. Used as few-shot demonstrations
  in the summary stage. The summary format and the choice of these six tasks
  follow NVARC (github.com/1ytic/NVARC); the wording is our own.
- `task-variations/examples/review/`: six hand-written review demonstrations.
  Each folder holds the task, its summary, the generator before review, the
  human review (`manual_review.md`), the generator after improvement, and
  samples from both generators. All six are used in the task-variation review
  stage; the composition review stage uses four of them, chosen to differ from
  both parents.

## Human input

The generators were checked by machine only, with these exceptions:

- The six summary examples and six review examples below were written by hand.
  For those tasks the hand-written summary or review replaced the model stage,
  so their variation families were built from it.
- `arc2_train/27a28665`: all four stage outputs were replaced by a manual
  source-fidelity correction after the original summary misread the task.
- `arc2_train/25e02866`: a stray note after the code block in the model's
  response was removed by hand so the response could be parsed; the
  generator code (`gpt-6-sol`) was not changed.

## Models

All runs used reasoning effort `xhigh`.

Task variations, mostly GPT "Sol-class" models run through Codex CLI:

- summaries: `gpt-5.6-sol` (1,113 tasks) and `gpt-6-sol` (187);
- first generators: `gpt-5.6-sol` (1,094) and `gpt-6-sol` (201);
- reviews: `gpt-6-sol` (913) and `gpt-6-astra` (387);
- improved generators: `gpt-6-sol` (1,186) and `gpt-6-astra` (120).

Counts are over the 1,307 released variation generators. The exceptions are
listed under "Human input" above; a few early records (11 first-generator
stages) do not name the model.

Task compositions:

- `gpt-6.1-sol` in Codex CLI for 2,622 families;
- `claude-opus-5-5` through Claude Code for the 84-family batch
  `task-composition-v3-claude-arc2-training`.

## Running this code

This is a reference copy. It expects the original repository layout
(`pipelines/task-variations` and `pipelines/task-composition-v3` side by side,
with a `results/` folder two levels up), a Python environment with NumPy and
the pinned Codex SDK, and a signed-in Codex CLI or Claude Code session. Some
parts will not run as copied:

- `task-composition-v3/entrypoint.py` (and so the `run` script) imports
  `device_batch` and `codex_handoff`, which are not included. `run.py` can
  be called directly.
- `claude_web.py` hashes the skill and agent files at `.claude/...` paths;
  they are stored here under `claude-code/` instead.
- The program-label stage (`src/05-programs`), `src/shared`, the visualiser
  and manual review tools, tests and datasets are not included, so the
  `produce-programs`, `visualise` and `review-ui` commands and the packaging
  in `pyproject.toml` refer to missing files.
- Some READMEs mention other files or folders from the original repository
  that are not included here.

Licence: MIT-0, like the rest of this repository; see
[Licence](../../README.md#licence).
