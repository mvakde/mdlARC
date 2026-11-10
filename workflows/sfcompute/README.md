# SF Compute Autoresearch workflow

SF Compute is an additional provider in `workflows/vast/Run.command` and
`workflows/vast/choose.py`. It uses Autoresearch's `autor` CLI and command/file
API rather than SSH. The scientific experiment JSON is unchanged.

The workflow implementation is repository source, like Vast, Runpod and Colab.
The installed provider CLI, its refreshed login and the saved infrastructure
policy live outside Git and survive branch switches. Another branch needs this
workflow code merged or cherry-picked to expose the provider in its chooser;
credentials alone do not install a workflow into another branch.

## Setup

Install the official CLI once, then sign in in a browser:

```bash
curl -fsSL https://autoresearch.sfcompute.com/cli.sh | bash
~/.local/bin/autor login
```

For Firefox or a browser on another device, use `autor login --device-code` and
open its newly generated URL there. `autor` refreshes its own login; this
backend does not read OAuth tokens itself. Credentials remain in the CLI's
user configuration, normally `~/.config/gman/`, never in a spec or source
snapshot. The account credential is never uploaded to a node.

Save the infrastructure policy separately:

```bash
mkdir -p ~/.config/mdlarc-sfcompute
chmod 700 ~/.config/mdlarc-sfcompute
cp workflows/sfcompute/policy.example.json ~/.config/mdlarc-sfcompute/policy.json
chmod 600 ~/.config/mdlarc-sfcompute/policy.json
```

The launcher also uses the existing dedicated, verified Drive-only remote-run
credential at `~/.config/mdlarc-vast/drive-token.json`. Follow
[the shared Drive setup](../vast/README.md) if it needs renewal. Provider login
does not replace Drive authorization. No credit purchase or automatic top-up
is performed by this workflow.

The default API URL is `https://givemeanode.com`, which autor 0.29 uses for its
stored login. The official `https://autoresearch.sfcompute.com` alias is also
supported; CLI logins are scoped by URL. The policy's API URL and workspace are
frozen in each saved launch. Recovery rejects a conflicting workspace or URL.

## Launch and observe

The native chooser offers **SF Compute**, then the spec, policy and H100 shape.
The command-line chooser also accepts:

```bash
venv/bin/python workflows/vast/choose.py --backend sfcompute -- \
  --config specs/conceptarc_smoke.json \
  --policy workflows/sfcompute/policy.example.json --dry-run
```

That preview is offline: it resolves the experiment and packages the exact
working-tree source without a login, rental or Drive write. Launch a selected
experiment using the saved policy:

```bash
venv/bin/python workflows/sfcompute/run.py launch \
  --config specs/your_experiment.json --gpu-template h100 --detach

venv/bin/python workflows/sfcompute/run.py status --session SESSION
venv/bin/python workflows/sfcompute/run.py watch --session SESSION
venv/bin/python workflows/sfcompute/run.py sync --session SESSION
```

Use `--branch BRANCH` to observe another branch's explicitly selected session.
`h100` and `h100-1` select one H100; `h100-8` selects a whole eight-GPU node.
The eight-GPU shape requires an appropriate saved price/total allowance. It
does not silently enlarge the one-GPU budget. The existing DDP GPU-count
validation runs before rental. Use `--policy-override FIELD=JSON_VALUE` for
explicit changes, or edit your private saved policy.

The launcher checks available credit, upload limits, pinned input hashes,
scientific settings and Drive access before creating a node. It then verifies
the actual node UUID, unique name, image, shape, container CPU/RAM budget and
whole-node rate before uploading credentials or running work. A rejected
allocation is stopped with its disk retained. Price is quoted by creation,
so a price rejection can incur a brief charge before this recheck.

After confirmed handoff, the same detached supervisor used by Vast and Runpod
owns native runtime checks, training, checkpoint uploads, verified Drive
backups and the original deadline independently of the Mac. Handoff is not
proof of an optimizer update: use the console and `startup_timing.json`.

## Runtime and persistence

The example uses the catalog PyTorch/CUDA 13 image for FA4. A persistent venv
inherits the image's Python/Torch. Bootstrap constrains Torch to that exact
version and refuses a replacement. It installs only ordinary data/Drive
helpers; the shared worker installs pinned FA4 only when selected, then runs
the bounded native CUDA/BF16/compile/attention validation. An unchanged runtime
may reuse validation only when its full fingerprint matches.

FA2 and FA3 specs require those attention packages **already installed in the
selected image**. Use an appropriately prepared custom image and its Python
path; incompatible catalog image choices are rejected before rental. The
workflow refuses to start a long source build on paid capacity.
The small `conceptarc_smoke.json` is a useful offline packaging preview but
does not itself select FA4. Custom images must include Python 3.11+, PyTorch,
pip/venv support and the required attention backend; uv is used when present.
For custom images the launcher explicitly mounts persistent storage at `/home/dev`; do not bake the
only copy of your application into that mount point.

Workspaces, environments and checkpoints live below
`/home/dev/mdlarc-sfcompute/SESSION`. Disposable dataset/compiler caches use
local `/tmp`; optional `/scratch` is destroyed at stop. Only persistent files
survive a stop/wake. The supervisor's lifecycle control uses the node's own
metadata service and verifies its identity; it needs no account API key.

By default, success with verified backups stops the node; failures retain its
disk for inspection. The hard deadline still stops the node, and the provider
also has an idle shutdown rule. `keep` does not disable that provider rule.
Stopping does not delete the node or its files. This backend never deletes a
disk. Waking can queue for capacity and is not guaranteed to be immediate.

Published rates and provider behavior are documented at
[Autoresearch](https://autoresearch.sfcompute.com/) and in its
[manual](https://autoresearch.sfcompute.com/docs). Policy budgets are launcher
allowances, not account-wide provider spending caps. The original runtime
includes setup and reserves time for backups.

## Recovery and records

Creation has a unique name saved **before** submission. An uncertain create
is reconciled by that name and UUID; no second node is created:

```bash
venv/bin/python workflows/sfcompute/run.py reconcile --session SESSION
venv/bin/python workflows/sfcompute/run.py handoff --session SESSION
```

Source/file transfers are checksum-verified and refuse different existing
bytes. The remote dispatch marker and supervisor marker prevent a second
training submission even when a response is lost. If bootstrap or training
already failed, inspect its saved state; repeating handoff does not restart it.

Use `sync` for verified Drive recovery without waking the node. Explicit
`start` wakes the original node; `recover-backups` dispatches only backup
recovery using its saved venv and never starts training. These operations
retain the original deadline. A continuation is a new session launched from
the exact saved spec with a verified Drive checkpoint identity, following the
same continuation rules as other providers.

Small records and bulk outputs live in ignored
`results/<encoded-branch>/sfcompute_runs/SESSION/`. Drive remains canonical at
`mdlARC/runs/<branch>/<session>`. Source code, tests, this guide and the policy
example belong in Git. Credentials, source snapshots, launch identities,
checkpoints, raw logs and one-off specs do not. The shared records selector
includes portable SF Compute specs/results while excluding live identities
and private files; reinstall `scripts/sync_records.py install` on each laptop
after updating it. Routine record synchronization follows
[`scripts/README_record_sync.md`](../../scripts/README_record_sync.md).

`autor cp` has an account-specific maximum file size. The launcher checks the
source archive and split family inputs against it before rental; increase the
provider upload limit or reduce the package before attempting a larger run.

Offline regression tests live in `workflows/vast/tests/test_sfcompute.py` and
run with the shared workflow tests. The earlier tiny H100 probe established
native FA4/BF16 optimizer updates; it did not establish full-model memory use,
throughput, or an end-to-end Drive-backed training launch of this integration.
