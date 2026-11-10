# Run this repository on Colab from the terminal

The launcher uses Google's [Colab CLI](https://github.com/googlecolab/google-colab-cli).
New Drive-backed runs use a fresh Colab VM and the default
[`scripts/colab_oauth/launch.py`](colab_oauth/README.md) entry point. It verifies
and installs the saved Drive-only `your-drive-account@example.com` credential, then uses
`scripts/run_on_colab.py` for execution and recovery. The launcher sends
current code and small data to that VM, downloads FlashAttention directly from
private Drive, and runs `launch_run.py`. A separate process uploads completed
checkpoints to `mdlARC/runs/<branch>/<session>` while training continues. Non-checkpoint
results return to your Mac. No Git commit or GitHub PAT is required.

Follow the [shared Drive setup](README_colab_drive.md) before the first real run.
That guide explains cached VM authorization, backup frequency, recovery, and
what gets downloaded. Set up the separate experiment-account token once using
[the OAuth guide](colab_oauth/README.md); renew it when the grant expires.
`--dry-run` requires no credentials and allocates no VM.

Choose the Colab compute account with `--account 1` or `--account 2`:

| Option | Colab login |
| --- | --- |
| `--account 1` (default) | `your-colab-account@example.com` |
| `--account 2` | `your-drive-account@example.com` |

The native `workflows/vast/Run.command` chooser asks which account to use after
you select Colab. Account 1 keeps the existing CLI login and session aliases.
Account 2 has its own token, session store, and history outside the checkout at
`~/.config/mdlarc-colab-accounts/2/`. No global login is switched, so runs from
both accounts can execute concurrently within each account's available quota.
Use unique session names across both accounts; local and Drive run records are
keyed by branch and session. The compute login is separate from the dedicated
Drive-only credential used for all run backups.

Sign into account 2 once (choose **your-drive-account@example.com**), then create a fresh
VM and launch it with the same account number:

```bash
venv/bin/python scripts/colab_accounts.py --account 2 -- sessions
venv/bin/python scripts/colab_accounts.py --account 2 -- new -s <new-session> --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/launch.py --account 2 \
  --config specs/pretrain_eval.json --session <new-session>
```

The account helper accepts the normal Colab CLI commands after `--`, including
`status`, `upload`, `download`, `auth` (explicit manual VM authentication), and
`stop`. It verifies account 2's login before any VM operation. Plain `colab`
commands continue to use account 1. Account 2 needs its own Colab CLI consent;
the Drive-only token cannot be used to allocate VMs. An expired grant requires
renewing the selected login, not switching to another account.

Launch journals and kernel identities save the compute account. The normal
`--watch-session` and `colab_run_docs.py ... refresh --session ...` commands
automatically restore it; an explicit conflicting account is rejected before
contacting Colab. Older records retain account 1 ownership. For an untracked
account 2 session, refresh with `refresh --session <session> --account 2`.

One-time setup on this computer:

```bash
uv tool install --force google-colab-cli==0.7.5
venv/bin/python scripts/fix_colab_cli.py
colab sessions
```

`--force` also upgrades an existing installation pinned to an older version. Version 0.7.5 supports its required PyPI `jupyter-kernel-client==0.9.0`; the Git kernel-client override needed by 0.7.2 is no longer required. The repair command remains necessary: it fixes the CLI's stdin handler, which otherwise reads an OAuth verification code without sending it to the VM. It also hides verification-code input and keeps input values out of CLI history. The repair is idempotent, keeps a backup beside the installed file, and refuses unrecognized versions. Repeat it after reinstalling or upgrading the CLI. The last command starts Google sign-in if needed. Complete it using the Google account that has A100 access. The CLI stores its credential in your user profile; no credential belongs in this repository.

From the repository root, choose an experiment spec explicitly. For example, the existing `specs/pretrain_eval.json` runs ARC-1 pretraining and evaluation:

```bash
venv/bin/python scripts/colab_oauth/launch.py --config specs/pretrain_eval.json --dry-run
colab new -s <new-session> --gpu A100
venv/bin/python scripts/colab_oauth/launch.py --config specs/pretrain_eval.json \
  --session <new-session>
```

For an A100 80 GB run, request Colab's high-memory shape when creating the VM
and verify the GPU memory before installing packages or training:

```bash
colab new -s <new-session> --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/launch.py --config specs/conceptarc_pretrain_eval.json \
  --session <new-session> --min-gpu-memory-gb 75
```

The wrapper defaults to `--auth-mode cached`; a valid saved grant avoids
per-VM Google sign-in. It validates the launch before touching the VM and
refuses sessions with existing run records or manual credentials. Cached-token
failure requires renewal through the OAuth setup command. For an explicitly
requested manual fallback, run `colab auth -s <new-session>` and use
`scripts/colab_oauth/launch.py --auth-mode manual` with the same launch arguments.
See [manual fallback details](README_colab_drive.md#manual-authentication-fallback).

The CLI currently uses Colab's default runtime version. To select a past
runtime such as 2026.01, use the Runtime > Change runtime type dialog in a
Colab notebook.

The bundled `wheels/flash_attn-2.8.3.post1-cp313-cp313-linux_x86_64.whl` was
built on an A100 for Python 3.13, PyTorch 2.11.0+cu128, CUDA 12.8, and GPU
compute capability 8.0. The launcher verifies its SHA-256 after downloading from Drive and checks
the remote Python, PyTorch, CUDA, C++ ABI, and GPU architecture before
installation. If Colab changes its default runtime, the run stops with a
compatibility error before training rather than compiling FlashAttention.

The default GPU is A100. Current `src/`, the selected spec,
`dataset_building_scripts/`, `assets/`, grouped ARC data, the small wheel
manifest, and Drive helper code are sent, including uncommitted edits. Other
specs are omitted. The wheel binary, old runs, and local
`assets_tmp/RE-ARC` tree are omitted. Existing data preparation may download
RE-ARC on a fresh ARC runtime even if the spec does not request it. Grouped ARC
uploads/compression have not changed in this update.

For a new parameter sweep, use the reusable template and trial record flow in
[`specs/trials/README.md`](../specs/trials/README.md). The selected
resolved spec is saved with each session's local records after packaging.

For a manually authored one-off spec under `specs/temp specs/`, add
`--standalone-spec` to both the dry run and launch command. This supports fresh
training without an input checkpoint. Sweep-generated specs still use their
matching `--trial-record`; the two options cannot be combined. Drive continuation
specs with a top-level checkpoint object do not require either option.

The full task-variation specs (`specs/arc2_variations_finetune.json` and
`specs/arc2_variations_joint.json`) automatically request their private ZIP
from restricted Drive. Launch with cached VM authentication as described in
[the Drive setup guide](README_colab_drive.md#new-runs-authorize-the-vm-then-launch). The ZIP
is verified and extracted under `assets_tmp/exported-datasets/` before training.
`--task-variations` remains available to stage the ZIP with an older spec;
older specs otherwise keep their existing behavior.

Non-checkpoint results and `run_status.json` are saved under `results/<branch>/colab_runs/<session>/`.
By default, the launcher leaves user-created shared-Drive sessions running.
Verify the checkpoint ledger and result recovery before stopping the session
yourself with `colab stop -s <session>`. Add `--stop-after-success` to stop it
automatically after training, finetuning, and evaluation succeed, all Drive
backups verify, and the non-checkpoint results are recovered locally:

```bash
venv/bin/python scripts/colab_oauth/launch.py \
  --config specs/conceptarc_variations_finetune.json \
  --session <new-session> --min-gpu-memory-gb 75 --stop-after-success
```

The launcher must stay running on your Mac through recovery and shutdown. It
automatically starts `caffeinate -i -w <launcher-pid>` on macOS and releases the
idle-sleep assertion when it exits. Keep the Mac powered with its lid open and
use a stable network; this assertion does not prevent lid-close sleep or Wi-Fi
loss. A failed run, unverified backup, or failed local recovery keeps an existing
VM available. A shutdown failure
reports an error and records `stop_failed` in the session journal.

## Dropped terminal connections

These protections are automatic in the normal launcher; no extra reconnect
flags or installed CLI patches are needed. `scripts/colab_connection.py` runs
with the CLI's Python and uses its existing local credentials. It preserves the
original VM, kernel, WebSocket client, and execution request identities in
`results/<branch>/colab_runs/_active/<session>.connection.json`; credentials are not saved there.
On a socket drop it refreshes runtime credentials and reconnects to the same
kernel, with bounded requests, ping/pong timeouts, and up to eight retries with
backoff. Training is submitted once. If delivery is uncertain, the launcher
observes files instead of sending another execution request. Reattachment never
creates a new VM/kernel or restarts training. Socket status is recorded in
`_active/<session>.connection.status.json`.

New launches also independently check the VM's saved console and result manifest
every 30 seconds using bounded file requests. If the CLI output stream ends or
stays quiet, the terminal follows the saved console and prints check/retry
messages. An unavailable network retries without submitting training again.
Result recovery and `--stop-after-success` do not depend on the output stream
returning. The VM publishes its finalized result manifest after the final
Drive summary upload has verified or recorded failure.

If the launcher or terminal itself closes, open another terminal in this
repository and attach to the existing run:

```bash
venv/bin/python scripts/run_on_colab.py --watch-session <session>
```

This validates the preserved launch record, attaches to the original kernel,
follows VM logs, and recovers results. It uploads no source and executes no
training. It inherits the original
`--stop-after-success` setting; that flag can also be added explicitly.
An OS lock enforces one watcher per session and releases automatically after a
crash. The local socket helper also exits if its launcher disappears, releasing
its socket lock without interrupting the remote job. A persistent result-download
failure leaves the VM available and reports the reattach command.

A network or authentication failure is recorded as unknown and retried. If two
successful authenticated server checks omit the original VM, the watcher stops
with state `unavailable`, preserves local records and Drive backups, and reports
that the VM is gone. It never silently launches a replacement. Do not delete or
edit saved connection identities to try again; inspect ambiguous submissions or
missing kernels using saved VM files and backup records.

Existing VMs keep their uploaded code. Runs without a saved connection identity
use file-only monitoring. Older uploaded snapshots may lack the finalized
completion marker; their watcher recovers results but retains the VM until you
manually verify the final Drive summary and stop it.

If a fresh pretrain/evaluation launch failed **before** an execution identity
was saved, `--watch-session` will not submit it. After read-only inspection has
confirmed the original VM and idle kernel, explicitly finish that first submission:

```bash
venv/bin/python scripts/colab_oauth/launch.py --branch YOUR_BRANCH \
  --resume-unsubmitted SESSION --expected-endpoint ORIGINAL_VM_ENDPOINT \
  --expected-kernel-id ORIGINAL_KERNEL_ID
```

This requires retained local records, an unchanged saved spec/source archive,
the identical uploaded archive, and no unpacked workspace or result files. It
uses the archived execution code and original idle kernel, transfers no
credentials or source, and inherits `--stop-after-success`. It supports fresh
pretraining/evaluation specs with `enable_finetune=false`, without checkpoint
or launch-only overrides. Any existing execution identity, including ambiguous
submission, rejects this mode; use ordinary attachment/inspection instead.
Read-only kernel lookups retry bounded transient failures; kernel creation and
execution are never blindly retried.

For an explicitly authorized live transport check after changing this code:

```bash
venv/bin/python scripts/check_colab_connection.py
```

This creates one fresh CPU-only VM and runs a three-minute diagnostic. It drops
only its local WebSocket for 45 seconds, checks progress through saved files,
reconnects, closes the local watcher, and attaches a replacement without
resubmission. It verifies one execution, the same kernel PID, complete progress,
archive checksums, and shutdown of that CPU VM. Its evidence is saved at
`results/<branch>/colab_runs/<diagnostic-session>/validation.json`. It accesses no private Drive
data and does not test Drive authentication or long-duration Colab retention.

## Run notes and agent handoff

For every real launch, the command first opens two macOS dialogs: one asks why
you are running the experiment, and the next asks what changed from the previous
run. Cancelling either stops the launcher before it uses the prepared VM. Dry runs do not ask for
notes. For a noninteractive launch, provide both `--intent` and
`--changed-from-previous` on the command line; providing only one is an error.

Your answers are written to `results/<branch>/experiments/notes/<session>.md`. Add observations,
interpretation, and a next action there at any time. The launcher never edits
that manual note again. It writes a separate deterministic report to
`results/<branch>/experiments/generated/<session>.md` and updates `results/<branch>/experiments/INDEX.md` from
the saved run records. All of these files, including raw records and live logs, are ignored by Git. The note text is included
in the Colab source archive.

While a launch runs, `results/<branch>/colab_runs/_active/<session>.json` records its last known
state, stage, and epoch. `_active/<session>.log` captures the CLI output and
`_active/<session>.vm.log` saves the independently fetched training console.
The generated report uses the VM console when available and labels progress as
last observed because a
launcher or network interruption can leave it stale. A read-only CLI check can
update the observation without claiming that training is still progressing:

```bash
venv/bin/python scripts/colab_run_docs.py refresh --session <session>
```

Regenerate reports from all locally recovered runs, including runs made before
this documentation feature, with:

```bash
venv/bin/python scripts/colab_run_docs.py render
```

The report lists the exact uploaded source SHA-256, the saved experiment spec,
checkpoint backup count, evaluation scores, and concise failure evidence when
those files are available. `AGENTS.md` explains the reading order for agents.

For the previous behavior (local wheel upload and checkpoint downloads), opt in
explicitly with `--local-artifacts`. Missing VM-side Drive authorization never
triggers a silent fallback to downloading large checkpoints.

These measures recover transport failures and protect saved results. Colab
availability and runtime duration still depend on account quota and backend
limits; successful HTTP polling does not guarantee VM retention. No artificial
activity or keepalive cell is used.

To continue pretraining from a saved Drive checkpoint, put its identity in the
spec's top-level `checkpoint` object and set `pretrain.epochs` to the desired
total. Use the same `--config ... --session ...` command; no resume flags are
needed. See [the continuation schema and examples](README_colab_drive.md#continue-pretraining-from-any-saved-checkpoint).

For an evaluation-only diagnostic of a public-pretraining checkpoint, use
`eval_only=true`, `enable_finetune=false`, and a top-level Drive checkpoint.
Choose the ordinary evaluation dataset so the evaluation can be scored. If the
checkpoint learned namespaced task IDs, set `eval_defaults.checkpoint_task_id_prefix`
to that namespace (for example `"arc2_eval/"`) and set
`extend_checkpoint_task_ids=false`. This matches native evaluation IDs to their
existing trained embedding rows, preserving all weights and rejecting missing
rows or identity collisions. A score on public answers used during pretraining
measures performance on seen tasks and must be labeled accordingly.

Evaluation indexes only tasks present in its challenge file and any requested
task subset. It retains the checkpoint embedding table and row order for exact
seen-task conditioning, loading the factors in bulk instead of copying rows
individually. New-task Kaggle submissions use `kaggle/submission_utils.py` to
discard the old task IDs, task-factor rows, and optimizer state before finetuning;
the learned shared embeddings and transformer weights are retained.

For several checkpoint evaluations on one VM, use an evaluation-only spec with
named `evaluation_checkpoints` and `input.NAME` evaluation references. The normal
launcher stages each input directly from Drive and runs the evaluations in
isolated processes, with one final recovery/shutdown. An explicit checkpoint
`source_branch` permits a historical input from that branch after verifying its
Drive ancestry. See [the demo ablation settings guide](../specs/README_demo_ablations.md).

When a retained VM is needed for a subsequent job, the same cached-auth wrapper
supports `--reuse-session OLD --session NEW --config SPEC`. This creates a fresh
local run alias for the same original VM/kernel, without allocating or changing
credentials. It first verifies that the old job is terminal, its kernel is idle,
and its finalized results were recovered; it preserves the old workspace and
all result/control files. Unverified checkpoints block reuse. A failed startup
without checkpoints or a backup ledger may be preserved after bounded metadata
inspection; its incomplete backup flag is recorded honestly. The new job gets
its own source/spec/execution identity, standard watcher, and success shutdown.
Never use this option to retry an ambiguous execution or restart an active job.
