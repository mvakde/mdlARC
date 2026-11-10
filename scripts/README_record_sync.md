# Research records on both laptops

Source and reusable specs stay in the private GitHub repository. Small research
records and draft execution specs synchronize through Drive independently of the
checked-out code branch. Existing run artifacts remain canonical in
`mdlARC/runs/<branch>/<session>`.

The sync command uses the already authorized, Drive-only
`your-drive-account@example.com` credential at
`~/.config/mdlarc-colab-oauth/your-drive-account/token.json`. It accesses no VM and uses
neither Colab compute login nor Vast credentials. It verifies the account,
granted scope, and existing `mdlARC/runs` parent before writing. Its destination
is the [shared-records](https://drive.google.com/drive/folders/DRIVE_ID_08_PLACEHOLDER)
child of that existing runs folder; no permissions or
existing run folders are changed.

## Install on each laptop

From the checkout containing this script:

```bash
venv/bin/python scripts/sync_records.py install
venv/bin/mdlarc-records inventory
venv/bin/mdlarc-records sync --dry-run
venv/bin/mdlarc-records sync
```

The installed program lives in the ignored `venv/bin/` directory and contains
its own implementation. It continues working when another code branch does not
contain `scripts/sync_records.py`. Reinstall from an updated checkout to upgrade
it, or after rebuilding the environment. The existing Drive requirements are
in `scripts/requirements_colab_drive.txt`.

Run from the project folder. To use another working directory, put
`--root /absolute/path/to/mdlARC` before the command. An explicit `--token` path
also goes before the command. Keep tokens outside every Git checkout.

If authorization expires, renew the existing dedicated credential using the
[OAuth setup](colab_oauth/README.md). The OAuth client file on both laptops is
`~/.config/mdlarc-oauth-clients/mdlarc-oauth-client.json`:

```bash
venv/bin/python scripts/colab_oauth/setup.py \
  --client-secrets ~/.config/mdlarc-oauth-clients/mdlarc-oauth-client.json --reauthorize
```

The cached token remains at the path above. Moving the client file does not
require renewal. A failed grant never selects another account or requests
manual VM authentication.

## Switching laptops

1. On the laptop you are leaving, save your notes/specs and run
   `venv/bin/mdlarc-records sync`.
2. On the other laptop, run the same command before editing or preparing a run.
3. Run it again after writing notes or preparing draft specs.

Agents follow the same standing workflow in `AGENTS.md`: sync before using or
editing synchronized records/specs, then sync any changes before handing the
task back. Routine sync needs no additional request or confirmation. If it fails,
preserve local edits and report the pending sync instead of claiming the other
laptop is up to date. Tasks that do not use or change research records/specs do
not need to run it.

This is an explicit sync command, not a background service. Changes made while
offline stay local until it succeeds. Each command covers every supported branch
directory, preserving the original branch paths; it ignores `MDLARC_RUN_BRANCH`
for synchronization. These records still describe their original branches and
must not be treated as evidence for the currently checked-out branch.

The one-time transfer archive is still useful for the complete local cache,
historical source archives, older Git history bundle, and recovery identities.
Those larger/private operational records are outside this small-file sync.

## What synchronizes

| Content | Location and handling |
| --- | --- |
| Research notes, generated reports, trial records | Markdown/JSON under `results/<branch>/experiments/{notes,generated,trials}/` |
| Experiment index, journal, top-level summaries | Markdown/JSON directly under `results/<branch>/experiments/` |
| Exact execution specs, completion summaries, experiment metadata, checkpoint identities, final scores | Selected JSON from `results/<branch>/{colab_runs,vast_runs,runpod_runs,sfcompute_runs}/<session>/`, including recovered `runs/<run>/` records |
| Unused draft and continuation specs | JSON under `specs/temp specs/` |
| Historical `results/codex/<branch>/` records | Same selection rules; original paths preserved |

Files must be regular UTF-8 text, at most 4 MiB each. Each upload batch is at
most 32 MiB. Invalid JSON, credential fields, and recognizable token/private-key
content stop synchronization. Never put credentials in research notes or specs.
The selector deliberately excludes nested data audits, source workspaces,
datasets, logs, checkpoints, archives, caches, lock files, and live `_active/`
connection/session identities. Raw run evidence remains in the existing Drive
run backups; use the launcher recovery workflow when it is needed locally.

The shared baseline journal is a separate existing artifact:
[baselines.md](https://drive.google.com/file/d/DRIVE_ID_09_PLACEHOLDER/view).
It is deliberately excluded here so no second canonical copy is created.
Follow `AGENTS.md`: reconcile local `results/baselines.md` against that existing
Drive file and update it in place when a baseline update is requested.

## Conflicts and preservation

Uploads are immutable, checksum-verified change bundles. They record which
previous versions each edited file replaces. New uploads never overwrite earlier
Drive files. Independent changes to different records merge on the next sync;
simultaneous edits to the same record retain both versions and report a conflict.
Different existing contents on a laptop's first sync also require review.

On conflict, no planned local replacement or new upload occurs. Remote copies
are saved under ignored `results/_shared/record_sync/conflicts/`; `record.json`
identifies the original file. Compare those versions with the untouched local
file, edit the local file to contain the intended result, then explicitly resolve:

```bash
venv/bin/mdlarc-records sync \
  --resolve 'results/codex%2Fexample/experiments/notes/session.md'
```

Use the exact path reported by the command. Repeat `--resolve` for multiple
reviewed files. This publishes the reviewed local contents as the successor of
all observed versions. A concurrent later upload can still require another
review; it cannot erase either version.

Deleting a local record does **not** delete it on Drive: the next sync restores
it. This tool has no deletion/retention operation. Preserve the shared folder
and its history; deleting older bundles makes their descendant history
incomplete. The tool stops on incomplete/corrupt history rather than guessing.

Uploads and folder creation save pre-generated Drive IDs before submission.
If a response is lost, retrying reconciles that same identity instead of
creating another copy. Pending uploads remain under
`results/_shared/record_sync/`; preserve them until reconciled. A dry run changes
no Drive content or research files, but can populate the ignored download cache
and local process lock. Regular sync also maintains an ignored baseline cache.
Neither that baseline nor downloaded run statuses prove a VM is currently live.

Keep reusable sync code, tests, and this guide in source Git. Keep all research
contents and sync state ignored. No second Git repository is needed.
