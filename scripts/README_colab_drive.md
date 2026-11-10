# Private Drive storage for Colab experiments

New runs use the dedicated `your-drive-account@example.com` account for Drive storage.
Colab compute may use account 1 (`your-colab-account@example.com`) or account 2
(`your-drive-account@example.com`), selected with `--account`; see the
[two-account commands](README_colab.md). The CLI login stays separate from
the Drive-only credential. The Drive account has
Viewer access to the private [mdlARC folder](https://drive.google.com/drive/folders/DRIVE_ID_05_PLACEHOLDER)
and Editor access to its `runs` child. Each new run creates a fresh folder at
`mdlARC/runs/<branch>/<session>`. The wheel, environment, and dataset files remain in
their existing locations. Their fixed IDs are in `colab_drive.json`.

For notes and draft execution specs shared across laptops, use the separate
[small-record sync command](README_record_sync.md). It preserves branch paths
and stores immutable updates in `mdlARC/runs/shared-records`; existing per-run
backups and the standalone baselines file retain their locations.

## New runs: authorize the VM, then launch

Cached experiment-account authentication is the default. Complete the
[OAuth setup](colab_oauth/README.md) once, then reuse the verified, Drive-only
`your-drive-account@example.com` token across fresh VMs. The wrapper uses the same
execution launcher and recovery checks, and keeps the Google OAuth app in
Testing. Renew the grant when needed; there is no per-VM consent prompt while
it remains valid. Manual sign-in is an [explicit fallback](#manual-authentication-fallback).

Create a **new** Colab session using the
Google account with the desired GPU quota. Launch with the cached Drive credential:

```bash
colab new -s <new-session> --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/launch.py --config specs/conceptarc_smoke.json \
  --session <new-session> --intent "Smoke run" \
  --changed-from-previous "Using the shared runs folder"
```

The wrapper validates the spec and flags, verifies the saved token's account
and scopes, and transfers only that dedicated cached credential separately
from source and records. The Colab CLI login and historical uploader token are
never transferred. Use a fresh session for every experiment; reconnecting to
an existing run uses the watcher without installing credentials again.

The launcher sends only the small source snapshot through the Mac. The VM reads
the wheel and dataset ZIP from Drive, creates its run folder under `runs/<branch>`, and
uploads new checkpoints and result records straight to that folder. It verifies
Drive upload size and MD5; the checkpoint ledger also records SHA-256. Non-checkpoint
result records come back to the Mac. By default, the session is user-managed
and stays running; stop it yourself only after confirming needed backups. Add
`--stop-after-success` to the launcher command for automatic shutdown after the
run succeeds, all checkpoint and result-summary backups verify on Drive, and
local result recovery completes. Keep the Mac launcher running until it exits;
it automatically prevents Mac idle sleep while running. Keep the lid open,
power connected, and the network stable. Failed runs or incomplete recovery keep
an existing VM available even with this
flag.

To finetune from an existing pretrain checkpoint, first move its **old run
folder** into `mdlARC/runs` after that old run has finished. Moving the folder
keeps the checkpoint's Drive file ID. Pass its recorded identity to the launcher:

```bash
venv/bin/python scripts/colab_oauth/launch.py --config specs/YOUR_FINETUNE_SPEC.json \
  --session <new-session> \
  --pretrain-drive-id DRIVE_ID_06_PLACEHOLDER \
  --pretrain-name pretrain__tiny.pt --pretrain-size 658954291 \
  --pretrain-sha256 fbb9d881e9b22afb71a172cf326dc9d9db1aa4620844180ac5595f1227950718 \
  --finetune-target YOUR_TARGET --eval-target YOUR_EVAL \
  --intent "Finetune from original pretrain" \
  --changed-from-previous "New finetune settings"
```

Use a spec with `skip_data_build=false` so the VM rebuilds its dataset from the
shared Drive assets. The launcher rejects a finetune-only spec with
`skip_data_build=true`. The checkpoint is downloaded
to a temporary VM file, checked against Drive size and MD5 plus the supplied
SHA-256, and only then made available to training. A reference to the original
Drive file is recorded in the new run's `experiment.json`; no duplicate input
checkpoint is uploaded to the new run folder. Finetune and evaluation targets
must be names present in the selected spec. Omit `--eval-target` to train only.

### Default checkpoint downloads

Input checkpoints download directly from Drive to the VM through the cached-auth
API by default. The downloader requests **256 MiB byte ranges** with
`Accept-Encoding: identity`, streams each response to disk in 1 MiB blocks,
and retries transient failures from the last byte written within a bounded
retry budget. It uses the same verified credential as the Drive API client.
No Drive mount or browser authorization is needed while the cached grant is valid.

Each input still requires its verified Drive ID, run/branch ancestry, name,
size, MD5, and SHA-256. The complete file is published only after verification.
The input receipt records the range size, stream-block size, encoding, transfer
time, and verification time. Checkpoint bytes stay off the Mac.

### Copy checkpoints from a mounted Drive

For an unused runtime with Drive already mounted, add
`--checkpoint-mount /content/drive/MyDrive/mdlARC/runs` to the normal launcher.
Use the actual mounted location of the existing `mdlARC/runs` folder. This
applies to spec checkpoint objects, finetune-only checkpoint flags, and named
evaluation inputs. Checkpoint bytes are copied straight from the mount to the
VM's local staging file, without the serial Drive API media requests.

Mounting is a separate explicit Colab action and may require frontend consent.
In a notebook connected to that unused runtime, mount the experiment account
`your-drive-account@example.com`, for example:

```python
from google.colab import drive
drive.mount('/content/drive', readonly=True)
```

The launcher uses the existing mount without invoking sign-in or remounting it.
It still verifies the checkpoint's Drive ID, run/branch ancestry, name, size,
MD5, and SHA-256 before publishing the local copy. It resolves the source path
from those verified ancestor names and does not search the Drive. A missing or
corrupt mounted file fails rather than falling back to another transfer.
The input receipt in `experiment.json` records `transfer_method`,
`transfer_seconds`, and `verification_seconds`; measure real transfer speed
from these values. Mounted copies remain an explicit option; omitting it uses
the cached-auth API range downloader described above.

Existing run folders are **not** moved by this code. Current sessions keep the
exact source and destination settings they uploaded before these edits.

## Manual authentication fallback

Use this only when manual authentication is explicitly selected. Create a fresh
VM, then authenticate it interactively as `your-drive-account@example.com`:

```bash
colab new -s <new-session> --gpu A100 --high-mem
venv/bin/python scripts/fix_colab_cli.py
colab auth -s <new-session>
venv/bin/python scripts/colab_oauth/launch.py --auth-mode manual \
  --config specs/conceptarc_smoke.json --session <new-session> \
  --intent "Manual authentication smoke run" \
  --changed-from-previous "Explicit legacy authentication fallback"
```

`colab auth` requires Google sign-in in your browser and an authorization
response pasted at the terminal. If code entry hangs, press Ctrl+C, apply
`scripts/fix_colab_cli.py`, and retry authentication with the new link and code.
The repair should be reapplied after CLI reinstall or upgrade. The manual mode
reads and transfers no cached token; `scripts/run_on_colab.py` also remains
available directly for a VM you have already authenticated. Both routes verify
the Drive account and use the same backup and recovery safeguards. A cached
credential failure does not automatically select this fallback.

## Continue pretraining from any saved checkpoint

Copy the original experiment spec into the ignored `specs/temp specs/` folder,
choose a new `run_name`, and set
`pretrain.epochs` to the **total** target. Replace or add the top-level
`checkpoint` with the saved Drive file's identity:

```json
"checkpoint": {
  "drive_id": "DRIVE_ID_07_PLACEHOLDER",
  "name": "pretrain__tiny.epoch067.pt",
  "size": 609386904,
  "sha256": "drive"
}
```

All four fields are required. `size` is the expected byte count. `sha256`
accepts a recorded lowercase SHA-256 digest, or `"drive"` to obtain the
expected SHA-256 from Drive metadata on the VM. Missing provider checksums
or mismatched downloaded bytes stop the run before training. Downloads verify
size, MD5, SHA-256, and that the source belongs to a run folder under
`mdlARC/runs`; checkpoints do not pass through the Mac.

An optional `source_branch` names a different historical branch explicitly.
The downloader verifies that checkpoint's original run folder is beneath the
matching encoded branch folder in the configured `mdlARC/runs`, without moving
it. Omitting this field retains the same-branch requirement. Evaluation-only
specs may also supply named `evaluation_checkpoints` with these identities;
`input.NAME` references evaluate them sequentially on one managed VM. Inputs
are recorded and excluded from output checkpoint backups. See the
[settings guide](../specs/README_demo_ablations.md) for the complete workflow.

Existing local checkpoints keep their string form:
`"checkpoint": "/path/to/saved.pt"`. No checkpoint means fresh pretraining.
Drive objects are staged by the Colab launcher; `launch_run.py --preview`
validates and resolves them without network access. Direct execution of an
unstaged Drive spec reports that it must be launched through `run_on_colab.py`.

The completed epoch is read from checkpoint contents, never the filename or
a manually supplied starting epoch. Training restores model, optimizer,
global step, scheduler progress, and available RNG state. The total target
must be greater than the completed epoch. Input files are staged separately
under `runs/<run_name>/inputs/pretrain.pt`, leaving the final pretrain output
free to save and back up. Inputs are referenced in `experiment.json` and are
not duplicated in the new Drive run folder.

The future LR schedule uses the target in the continuation spec. Keeping the
original total preserves the schedule; changing the total deliberately
changes future cooling. Earlier warmup and training history remain unchanged.
Use `resume_warmup_pct=0.0` to avoid adding a new warmup. Dataset and optimizer
compatibility checks remain enforced. GPU continuations are not promised
bitwise identical to uninterrupted training.

Use the normal launcher, without resume or checkpoint flags:

```bash
# Choose a spec and create a fresh VM.
RUN_SPEC="specs/temp specs/conceptarc_resume67_to672.json"
RUN_SESSION="conceptarc-continuation-$(date +%Y%m%d-%H%M%S)"
venv/bin/python scripts/colab_oauth/launch.py --config "$RUN_SPEC" --dry-run
colab new -s "$RUN_SESSION" --gpu A100 --high-mem
venv/bin/python scripts/colab_oauth/launch.py \
  --config "$RUN_SPEC" --session "$RUN_SESSION" \
  --gpu A100 --high-mem --min-gpu-memory-gb 75 --stop-after-success
```

Standalone Drive continuation specs in `specs/temp specs/` do not need a
`--trial-record`. Sweep-generated staged specs still require their matching
record. The launcher saves the exact selected continuation spec with the run.

The existing `--pretrain-drive-id` / `--finetune-target` commands still support
finetune-only runs. Do not combine those flags with a checkpoint object in
the spec. Interrupted finetune resumption is outside this pretraining flow.

For the interrupted ConceptARC run, both prepared specs use the exact recovered
672-epoch spec and saved epoch-67 checkpoint:

| Spec | Total pretrain epochs | Additional epochs | LR cooling starts |
| --- | --- | --- | --- |
| `specs/temp specs/conceptarc_resume67_to672.json` | 672 | 605 | 537.6 |
| `specs/temp specs/conceptarc_resume67_to84.json` | 84 | 17 | 67.2 |

Set `RUN_SPEC` to either file above. Each retains the original 330-epoch
heldout finetune, both evaluations, and checkpoint schedule (67, 268, plus
final; past and out-of-range saves are skipped). The 84 branch inherits the
672-run warmup history (3.36 epochs rather than 0.42 for an 84-from-scratch run).
The source checkpoint's contents have not been downloaded or inspected on
this Mac; VM validation must pass before continuation. No new run was started
while preparing these specs.

`--stop-after-success` stops the VM only after all stages, Drive backup
verification, and local non-checkpoint result recovery succeed. Keep the Mac launcher
running until it exits. A failure or lost connection retains the VM for
recovery; inspect backup records before stopping it.

## Branch records and local cache

Each Git branch has a separate folder beneath `mdlARC/runs`, named after the
branch. Slashes and percent signs in branch names are percent-encoded, matching
the single local directory name. The launcher captures the branch before launch
and passes it to the VM; it does not guess from a subsequently switched branch.
It reuses the exact branch folder or creates one if absent. Duplicate branch
folders fail preflight. New sessions always create a fresh child folder.

The existing whole-task runs are in `runs/colab-restart`; single-pair runs belong
in `runs/newreason`. Locally, `results/<branch>/` contains `colab_runs/`,
`experiments/`, and `runs/`. All artifacts are ignored by Git and remain visible
when switching branches. Reusable specs and examples in `specs/trials/` remain
committed. Actual trial records belong in `results/<branch>/experiments/trials/`.

Drive is canonical. Each run folder contains the exact uploaded source, selected
spec, checkpoints, and non-checkpoint result archive. Its immutable
`local-records-<SHA256>.tar.gz` backups also preserve the local journal, manual
note, generated report, index, saved spec/provenance, and recovery evidence.
The initial note and journal are included in the source snapshot and backed up
before training. After recovery the launcher publishes the full local record
bundle using the original VM's credentials and verifies size, MD5, and SHA-256
before automatic shutdown. It uploads no Mac OAuth token. A failed record backup
retains the VM and reports recovery required. Binary checkpoints and historical
upload chunks are excluded from the local record bundles.

Reattach or render another branch explicitly:

```bash
venv/bin/python scripts/run_on_colab.py --branch colab-restart --watch-session SESSION
venv/bin/python scripts/colab_run_docs.py --branch colab-restart render
```

To publish edits made after the VM stopped, ask Codex to sync the notes through
connected Google Drive. Build the same portable bundle without credentials with:

```bash
venv/bin/python scripts/run_records.py --branch newreason --session SESSION --output /tmp/records.tar.gz
```

Upload it into the verified session folder using an immutable name containing its
SHA-256, then verify Drive metadata and record the file ID, checksum, and date in
`results/<branch>/drive_sync/`. A branch-wide bundle (omit `--session`) includes
sweep notes and trial records as well. Extract a downloaded record bundle into
`results/<branch>/`; it uses relative paths under that branch directory.
Local observations made after VM shutdown, including the shutdown result, remain
pending until this sync. Never mark an unsynchronized local edit as canonical.

## Historical Mac-token authorization (for recovery of older runs)

The following setup describes the older `drive.file` token path. New runs above
use VM-side account authorization instead.

1. In [Google Cloud Console](https://console.cloud.google.com/), create a
   dedicated project for this uploader. Enable **Google Drive API** and
   **Google Picker API**.
2. Configure Google Auth Platform for your own account. For a personal project,
   choose External and add your Google account as a test user if the app is in
   Testing. The only requested permission is `drive.file`.
3. Create an OAuth client with application type **Desktop app**. Download its
   JSON outside this repository, for example into Downloads. Use a dedicated
   client that has never been granted broad Drive access.
4. Install the optional Drive dependencies in the Python environment that will
   launch runs. For this repo's existing local environment, from the repo root:

   ```bash
   uv pip install --python venv/bin/python -r scripts/requirements_colab_drive.txt
   venv/bin/python scripts/setup_colab_drive.py \
     --client-secrets /absolute/path/to/downloaded-client.json
   ```

5. In the Google picker, select exactly these seven configured items: `mdlARC`,
   `fa2_wheels`, its dated subfolder, the `.whl`, `environment.json`, `datasets`,
   and `task-variations-v1.zip`. The
   helper filters the picker to those IDs and rejects extra or missing selections.

If you already authorized the previous five-item setup, its token does not gain
access to the new ZIP automatically. Re-run setup with the same dedicated Desktop
OAuth client and a **new** token path, for example:

```bash
venv/bin/python scripts/setup_colab_drive.py \
  --client-secrets /absolute/path/to/downloaded-client.json \
  --token ~/.config/mdlarc-drive/token-task-variations.json
```

That token is only relevant to historical scripts and manual recovery. New
launches use the separate cached experiment-account credential described above.

Google's `drive.file` permission is **per file**, not a folder-only OAuth scope.
Selecting the parent does not automatically grant access to existing files
inside it, which is why the wheel files and their parent folders are selected
explicitly. Newly created backup files are accessible to this dedicated app.
On top of this narrow authorization, code checks ancestry and writes only under
the fixed `mdlARC` folder ID. It has no Drive delete, replace, move, share, or
whole-drive listing calls. Each run creates a fresh child folder; existing Drive
files are never overwritten. Do not reuse this dedicated OAuth client to authorize
unrelated personal files.

The historical helper saves credentials with mode `600` at
`~/.config/mdlarc-drive/token.json`, outside this repo, after verifying the actual
Google token has only `drive.file` access. Historical launchers verified access
before allocating a GPU and transferred this small credential separately.
Current launchers do not use or transfer it. If an older run requires manual
token-based recovery and authorization has expired, repeat setup with a new
`--token` path. Do not grant broader access.

Official details: [desktop file-picker authorization](https://developers.google.com/workspace/drive/picker/guides/desktop-mobile-picker),
[Drive permissions](https://developers.google.com/workspace/drive/api/guides/api-specific-auth),
[resumable uploads](https://developers.google.com/workspace/drive/api/guides/manage-uploads).

Suggested names, **not applied automatically**: rename the project folder to
`mdlARC-colab`, the wheel collection to `fa2-wheels-colab`, and the dated subfolder
to `colab-a100-80gb-2026-09-24`. The date is the environment's build/upload date,
not the date of each experiment. All code references use IDs, not these names.

## Choosing backup frequency

Every training stage applies two checkpoint cadences by default, even when a
spec sets `checkpoint_epochs: []`:

- `kept_checkpoint_every: 50` saves standalone `tiny.epoch050.pt`, `tiny.epoch100.pt`
  and so on. These are never replaced, and each is backed up to Drive like any
  checkpoint. The final epoch is the ordinary final checkpoint.
- `rolling_checkpoint_every: 10` overwrites one local `tiny.rolling.pt` every
  10 epochs, so a crashed or stopped run can resume from at most 10 epochs back.
  It is never queued, uploaded, or counted toward backup completeness, and it is
  deleted once the final checkpoint is saved.

Set either cadence to 0 to disable it. To resume from the rolling file, use its local
path as a continuation spec's `"checkpoint"`. Extra epochs can still be added
explicitly; for example, `"pretrain": {"epochs": 90, "checkpoint_epochs": 10}` saves
kept checkpoints at epochs 10, 20, and so on, plus the final checkpoint.

Training saves to a temporary local file, publishes the complete checkpoint,
and adds a tiny queue entry. The separate uploader starts transferring it
immediately without making training wait for the network. It uses resumable
uploads and verifies Drive's file size and checksum. Checkpoint filenames cannot
be reused during a Drive-backed run. Uploads consume some CPU/disk/network
bandwidth; a runtime crash can still lose an upload that has not completed.

Each new Drive run folder under `mdlARC/runs/<branch>` contains checkpoints and a `source.tar.gz` backup of the
exact uploaded code, configs, and data. Checkpoints are named using their local
stage path, e.g. `pretrain__tiny.epoch10.pt`. A small
`results-without-checkpoints.tar.gz` stores the final records too.

## What comes back to your Mac

Non-checkpoint records are downloaded into `results/<branch>/colab_runs/<session>/`:

- `run_status.json`: success/failure, timestamps, and Drive folder location.
- `experiment.json`: exact spec, GPU/software details, and the source backup.
- `checkpoint_backups.json`: uploaded checkpoint locations, checksums, and failures.
- `console.log`, `failure.txt` when applicable, and `drive_upload.log`.
- Evaluation predictions, scores, accuracy metrics, timing, and other small outputs.

These are the small experiment notes Codex can read without downloading the model.
The exact source bundle is backed up on Drive too, including uncommitted changes.

Checkpoint extensions (`.pt`, `.pth`, `.ckpt`, `.safetensors`) are excluded
from local result downloads in Drive mode. All other run files are included
in both the Drive result archive and the local cache, regardless of size.
Credentials and queue control files are outside the run folder and never
included. Avoid syncing the Drive project folder onto your Mac if you want to
avoid downloading checkpoints through Drive for Desktop.

## Failures and recovery

The normal launcher saves the original VM/kernel/execution identity and
automatically reconnects with refreshed credentials after a socket drop. It
never repeats execution, restarts a kernel, or allocates a replacement during
recovery. It also polls the VM's saved console and completion manifest every
30 seconds independently of terminal streaming. It retries file requests after
network failures and recovers results even if streaming stays stuck. A new
run's finalized manifest is published after the last Drive summary upload;
shutdown still requires successful stages, verified backups, and verified
local result recovery.

To follow an existing shared Drive run after its launcher closes or stops
updating, run from the repository root:

```bash
venv/bin/python scripts/run_on_colab.py --watch-session <session>
```

This attaches using the saved launch identity and never uploads or starts
training. It inherits the original automatic-shutdown preference. A local lock
enforces one watcher per session; abrupt launcher closure releases the lock and
closes its local socket helper while the remote job continues. Historical runs
without a saved kernel identity use files only. Older uploaded snapshots may
have no finalized
completion marker; the watcher recovers their results but leaves the VM for a
manual final Drive-summary check before shutdown.

Two successful authenticated server listings must confirm the original VM's
absence before the watcher exits as `unavailable`. Network/auth failures are
unknown and do not trigger a replacement run. Local records and existing Drive
backups are preserved. These defaults protect monitoring and recovery but
cannot prevent Colab backend eviction or extend quota/runtime limits. See
[the connection workflow and CPU validation](README_colab.md#dropped-terminal-connections).

Before remote execution finishes, the launcher waits for all checkpoint uploads to
finish, verifies every new saved checkpoint is recorded, and retrieves non-checkpoint results.
A training failure still attempts to drain and recover completed checkpoints.
If a backup or local recovery fails, the user-managed session stays available,
including when `--stop-after-success` was requested.
Inspect `checkpoint_backups.json` and `drive_upload.log` before recovery.
An upload without confirmed completion is not blindly started again, to avoid
duplicate backups. Check the Drive run folder before retrying.

For new runs, VM-side account authorization stays on that VM; no Mac token is
transferred. Stop only after recovering required files: `colab stop -s <session>`.
Retained sessions can continue using compute units. For an old run using the
token path, consult its exact uploaded source and records before manual recovery.

For previous behavior (local wheel upload and checkpoint downloads), explicitly
use `--local-artifacts`. A `--dry-run` needs no Drive credentials and allocates no
VM. Failed VM-side Drive authorization never silently falls back to full
checkpoint downloads.

## Task-variation archive

The approved `mdlARC/datasets/task-variations-v1.zip` is used automatically by
specs with `dataset.pretrain_data.kind="task_variations"`. Run an existing
experiment with `--task-variations` to stage the archive without selecting the
variation training mode:

```bash
venv/bin/python scripts/colab_oauth/launch.py --config specs/conceptarc_smoke.json \
  --session <new-session> --task-variations
```

The launcher checks the ZIP's Drive parent, name, size, and SHA-256. The Colab VM
downloads it to `assets_tmp/task-variations-v1.zip` and extracts the data to
`assets_tmp/exported-datasets/`. It omits macOS metadata from extraction. The
archive is not included in the code upload or returned results; the run's
`experiment.json` records its file ID and SHA-256. The two full training specs
are `specs/arc2_variations_finetune.json` and `specs/arc2_variations_joint.json`.
