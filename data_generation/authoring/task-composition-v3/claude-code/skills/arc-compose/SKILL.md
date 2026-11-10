---
name: arc-compose
description: Run resumable ARC composition batches with native Opus 5.5 workers and local validation, using Claude Code session usage.
argument-hint: "--dataset arc2/training --batch-size 100 --workers 4 [--stage author] [--effort high] [--defer JOB]"
disable-model-invocation: true
---

Run this batch in the active Claude Code web session. Arguments: $ARGUMENTS
Session ID: ${CLAUDE_SESSION_ID}

All model work must use Claude Code's native Agent tool and the checked-in
arc-composition-worker agent. Never call an LLM API/SDK, set an API key, invoke
claude/codex as a subprocess, or use a separately billed provider. If native
workers are unavailable, preserve progress and report that limitation. Do not
switch billing routes. Use the exact claude-opus-5-5 model for every worker and
repair, at the effort named in each assignment: xhigh (extra-high) by default, or
high only when the user passed `--effort high`. The worker definition pins the
model and xhigh; pass the assignment's effort explicitly to the Agent tool on
launch and resume. Do not use a cheaper model or any effort below the
assignment's. The user selects the main session model.

1. Work from the repository root. Read
   pipelines/task-composition-v3/CLAUDE-WEB.md for command details. If the shared
   Python environment is missing or cannot import NumPy, create it with
   `python3 -m venv pipelines/task-variations/.venv` and install only
   `"numpy>=2,<3"` using that venv's pip. Do not install the Codex package or an
   Anthropic SDK for this workflow. If the cloud cannot install dependencies,
   explain the error and preserve progress.
2. Parse the user's arguments as the documented options, not shell code. Run
   `pipelines/task-composition-v3/claude-web start` with those options. Do not
   repeatedly call start within this invocation: once completed, start appends
   another batch. With no arguments the initial batch is 100 pairs, 4 workers,
   arc2/training, both stages. Retain the selected output root for every command.
   Allow preparation and submissions to finish before parsing JSON. Use a
   600000 ms Bash timeout; if the tool backgrounds a helper, wait for its final
   output with TaskOutput. These helpers perform local artifact work only.
   `--defer JOB` (name or number) stops scheduling a composition the user asks to
   skip; it keeps the pair's reservation and drafts until `--undefer JOB`.
3. The helper prints progress to stderr and one JSON result to stdout. If ok is
   false, stop with its error. After start, persist as in step 6 before any
   model work. Then dispatch every returned task with the
   native Agent tool, using subagent_type arc-composition-worker, as concurrent
   background workers so each completion can be handled as soon as it arrives.
   Give each worker exactly its assignment object and the instruction to follow
   its prepared instructions.md. Do not include other task content or
   conversation history. Do not use forked workers or worktree isolation: these
   workers must write into the prepared workspace visible to the coordinator.
4. Keep at most --workers native workers active; the helper never assigns more.
   Each author and each reviewer starts in a fresh worker context. A repair can
   resume that same stage's worker if its agent ID is still available in this
   session; otherwise start a fresh worker with the same assignment and
   preserved drafts/feedback. Never reuse an author context for reviewing its
   composition. Do not inspect/generate task content yourself: Python
   coordinates state, and stage workers write content.
5. As soon as any worker returns, run `claude-web submit --ticket TICKET
   --session-id SESSION_ID --agent-id AGENT_ID` (plus the output root). Do not
   wait for the other workers first. Use the real native Agent ID returned by
   Claude Code. Do not invent IDs, token usage, or model verification. The
   helper runs the original 42-seed checks, evidence hashes, review barrier,
   publication and checkpoints without contacting a model.
   - repair: send the worker its feedback.json and submit again after it
     repairs and updates native-response.md. Its slot stays occupied.
   - accepted: immediately run `claude-web next` with the same output root and
     dispatch every task it returns. Its tasks are only new assignments, usually
     the accepted pair's review or the next pending pair; never dispatch a ticket
     listed in running again. Then persist as in step 6.
   - stopped: stop scheduling; let active workers finish/checkpoint and report
     the failure.
   Never modify validation or mark a failed stage complete yourself.
6. Persist after initial preparation and after every accepted stage: run
   `claude-web checkpoint` with the same output root. Its persist_paths
   identifies the output tree and dataset pair ledger/order. Stage only those
   paths, commit a progress snapshot, and push the current Claude session
   branch. Do not force add ignored samples/workspaces, force push, merge, or
   include unrelated edits. Pairs are reserved only by start, so persisting
   after a refill's dispatch is safe. Keep this single cloud session as the coordinator for the
   dataset; separate clones/branches cannot share filesystem locks. Do not start
   independent batches on other branches.
7. Repeat the submit/next/checkpoint loop as workers return. next action wait
   means every slot is still busy; keep handling completions. Continue until
   next returns complete. next never appends another batch. If usage or a
   session/tool interruption prevents continuing, checkpoint existing drafts,
   commit/push if possible, and report where to resume. Do not poll running
   shell processes as a substitute for native workers. Do not schedule future
   work unless requested.
8. On completion, report author/completed counts from `claude-web status`, the
   output root, branch and any failures. State that these are generator-owned
   checks, not independent proof of the puzzle rules. Explain that repeating the
   same /arc-compose command resumes unfinished stages or appends the next batch
   after all requested stages finish.
