# Run compositions in Claude Code on the web

This workflow uses the active Claude Code session and its native Opus 5.5
subagents at extra-high (`xhigh`) reasoning effort by default. Python prepares task
pairs, accepts the workers' files, runs the
existing v3 checks and publishes results. It never contacts an inference API,
launches a second Claude/Codex CLI or asks for an API key. Native workers count
toward the signed-in account's Claude Code usage. Your account's extra-usage
settings still apply if you have enabled them; this repository cannot change
account billing settings.

Push these repository changes to GitHub before starting a **new** cloud session
on that branch. Project skills and agent definitions load from the cloned repo.
Connect GitHub and select Opus 5.5 in Claude Code yourself.

In the **Claude Code message box**, enter:

```text
/arc-compose --dataset arc2/training --batch-size 100 --workers 4
```

That requests 100 unique composition families, each with authoring followed by
review and improvement. To run only authors:

```text
/arc-compose --dataset arc2/training --batch-size 100 --workers 4 --stage author
```

This slash command orchestrates the batch. A shell command cannot launch native
Claude workers: the Python helper returns assignments for the current Claude
conversation to dispatch. The default output is
`results/task-composition-v3-claude-arc2-training`. Use `--output-root
results/task-composition-v3-claude-MY-RUN` to choose another folder; continue using
that same folder for resume. Keep the `task-composition-v3-` prefix so existing
Git ignore rules cover temporary workspaces.

Repeat the same slash command to resume an unfinished batch. Once its requested
stages finish, repeating it appends the next batch. To complete an author-only
batch, repeat it with `--stage all`; it reviews those existing families before
allocating another batch. With no arguments, the initial command defaults to
ARC2 training, 100 pairs, 4 workers and both stages. For a small trial use
`--batch-size 2 --workers 2`.

Other options are `--count N` (a total target, mutually exclusive with
`--batch-size`), `--seed N`, `--review-samples N` (1..42, default 5), and
`--parents-root PATH`. `--effort high` requests high instead of xhigh effort for
this and later assignments of the run; receipts record each stage's requested
effort, while `plan.json` keeps its pinned xhigh. `--defer JOB` (a composition
name or its number, repeatable) stops scheduling a pair without discarding its
reservation or drafts. A deferred pair is also skipped when ignored samples are
rebuilt and is left out of status counts, so one unreproducible pair can be set
aside; `--undefer JOB` schedules and checks it again. Parents and outputs must be inside the checkout. Inputs
are the existing complete task-variation packages, including their final parent
generators. Composing them with Claude does not regenerate the source variations.

The skill creates the shared Python environment when needed and installs NumPy
only. You can also provision it in a cloud SessionStart hook:

```sh
python3 -m venv pipelines/task-variations/.venv
pipelines/task-variations/.venv/bin/python -m pip install 'numpy>=2,<3'
```

No Codex or Anthropic SDK is required by the Claude helper. Do not run the usual
`pipelines/task-composition-v3/run` entry point for Claude: it still uses Codex.

## How the command runs

`.claude/skills/arc-compose/SKILL.md` is the coordinator procedure.
`.claude/agents/arc-composition-worker.md` defines a worker pinned to
`claude-opus-5-5` and `effort: xhigh` with local file/command tools. The coordinator
also passes each assignment's effort (`xhigh` unless the run uses `--effort high`)
when launching/resuming native workers. Author and reviewer receive fresh
contexts; the coordinator sends only the assigned workspace and its stage prompt.
Repair turns may resume the same stage's worker, up to three submissions before
the queue stops. An explicit rerun resumes the same failed composition.

Scheduling matches the Codex runner: at most `--workers` stages are active, and a
freed slot is refilled as soon as its worker's submission is accepted, without
waiting for the others. A slot follows one composition, so an accepted author is
followed by that pair's review, and a finished pair hands its slot to the next
pending pair. Stages for one composition never overlap.
Python applies the shared dataset ledger, unordered pair uniqueness and
least-used parent allocation. It uses v3's existing 42-seed verifier, checks
protected evidence and enforces the review checkpoint before improvement. No
native model output is accepted merely because a worker reports success.

The bridge feeds **one already-completed native worker response** through the
existing v3 validation/publication loop. When that loop requests another model
turn, the bridge returns a repair assignment to the native coordinator. It does
not execute the Codex client. Native response receipts retain the real session
and agent IDs. Token usage and the runtime's actual model are not independently
measured; records distinguish the requested model and effort from verified runtime usage.

Worker tool restrictions and stage instructions scope their work. Protected
inputs are hash-checked. This uses Claude's cloud/session permissions and does
not reproduce Codex's OS-enforced task-only filesystem profile.

## Persistence and cloud restarts

After preparation and after each accepted stage, the coordinator checkpoints drafts and
commits/pushes only the selected output root and dataset ledger/order on its
current session branch. It never force-adds ignored samples or workspaces.
Cloud usage interruptions can leave a batch incomplete; reopen the same branch
and repeat the slash command. Only pushed progress is available to a fresh clone.
Unsubmitted work lost before checkpoint/push must be generated again.

An updated helper, skill or worker definition takes over an existing output root
between stages, or when `start` resumes it; `claude-web.json` keeps the earlier
hashes under `previous_implementation_hashes`. `start` gives active stages new
tickets, so a worker launched under the previous workflow can no longer submit;
its saved drafts carry over to the new worker. `next` and `submit` still refuse
a changed workflow while stages are active. The v3 validation code stays pinned
by `plan.json`.

Use **one coordinator session/branch per dataset**. Dataset locks serialize
commands within one checkout; independent cloud VMs and branches do not share
those locks. Concurrent coordinators on different clones can allocate duplicate
pairs. Resume the current branch instead of starting a competing session.

On a fresh clone the helper rebases its absolute plan/ledger paths to the new
checkout. It rebuilds missing copied human examples from their canonical files
and regenerates ignored seed samples from the published generators. Every
reconstructed file must match its recorded hash. Changed or unreconstructable
evidence stops the workflow. Submitted drafts, completed review checkpoints and
feedback are preserved under `llm-inputs/<stage>/saved-output/native/`, so workers
can continue them without reopening the completed review. No historical native
agent transcript is reconstructed or claimed to be retained by Git.

## Helper commands

These are local artifact operations, useful for the coordinator or debugging:

```sh
pipelines/task-composition-v3/claude-web start --dataset arc2/training --batch-size 100 --workers 4
pipelines/task-composition-v3/claude-web next
pipelines/task-composition-v3/claude-web submit --ticket TICKET --session-id SESSION_ID --agent-id AGENT_ID
pipelines/task-composition-v3/claude-web checkpoint
pipelines/task-composition-v3/claude-web status
```

Supply the same `--output-root` to each helper when using a custom run.
stdout is one JSON object; progress goes to stderr. `start` reserves/resumes the
batch and returns every active assignment. `next` fills free slots without
extending the batch; it returns only the new assignments, lists already running
tickets under `running`, and reports `wait` while every slot is busy. `submit` verifies/publishes one native worker's artifacts. A worker must write
`native-response.md` before submission. `checkpoint` preserves active drafts and
returns `persist_paths` for Git; it does not itself commit or push. `status` also
checks/restores missing evidence. Repeated tickets cannot publish a second time.

Completed results use the usual v3 generators, specifications, reviews,
validation reports and samples. Existing export code accepts the custom root:

```sh
python3 pipelines/task-composition-v3/export.py \
  --source-root results/task-composition-v3-claude-arc2-training
```

Passing checks means generator-owned structural and solver checks passed. It
does not assert independent semantic correctness of the invented ARC rule.

Official references: [project skills in cloud sessions](https://code.claude.com/docs/en/skills#use-skills-in-cowork-and-cloud-sessions),
[native subagents](https://code.claude.com/docs/en/sub-agents),
and [cloud session limits](https://code.claude.com/docs/en/claude-code-on-the-web#limitations).
