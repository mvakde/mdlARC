# Independent Vast.ai experiment workflow

**Never stop or destroy B300 instance 57431449, at any cost.** Keep it powered
on through retries, errors, completed runs, budget limits, and deadlines.
The user explicitly requires preserving this scarce allocation. Training may
end, but no automatic cleanup or error handler may shut down its VM/container.
The Vast API client blocks stop/destroy requests for this instance ID.

The reusable [ARC2 development large spec](../../specs/arc2_family_development_large_default.json)
preserves the tested B300 settings: physical batch 12, gradient accumulation 5
(effective 60), evaluation batch cap 150 and prefill token budget 262144.
It retains 80 evaluation augmentations. Exact executed specs and source snapshots
remain in the local run cache and verified Drive provenance when old GPU
workspaces are explicitly cleared. Clearing workspaces never requires stopping
or recreating this instance; retain its installed runtime and SSH access.

All implementation and tests for this workflow live in this directory. The
Colab execution launcher and `scripts/colab_oauth/` remain independent. Vast and
cached-authentication Colab are the defaults for new remote experiments; manual
Colab authentication remains an explicit legacy fallback.

After initial setup, launch an experiment and let the remote supervisor own its
training, checkpoint uploads, periodic artifact backups, finalization, and cleanup.
The Mac can disconnect or close after the launcher confirms remote handoff.

The shared remote worker also accepts `backend="ssh"` in a prepared launch
plan for an externally managed VM. This mode reuses the detached training and
verified Drive backup machinery. It has no provider API client, always keeps
the VM after success or failure, and its deadline guard cannot stop the VM.
The training process still obeys the plan's runtime allowance. Credentials stay
in the workspace's private directory outside source and result archives.
The Vast rental CLI does not allocate or manage an external SSH reservation.
Google Drive is canonical at `mdlARC/runs/<branch>/<session>`. Local records are a
Git-ignored cache at `results/<branch>/vast_runs/<session>/`.

## Choose a workflow

Double-click `workflows/vast/Run.command` on macOS. It opens a chooser for the
Vast (preselected), Colab (cached OAuth), or legacy manual-authentication Colab,
then asks for the experiment spec. Vast uses your saved infrastructure policy.
The chooser also includes [Runpod Pods](../runpod/README.md), which share the
detached supervisor and Drive backup/recovery machinery and use their own
Runpod infrastructure policy and API key.
It also includes [SF Compute Autoresearch](../sfcompute/README.md), using the
installed `autor` login and command/file API with the same detached supervisor
and verified Drive backups. Its policy and credentials stay outside Git.
The Vast chooser also offers separate A100 40 GB, A100 80 GB, H100, H200, and
B200 templates, with A100 80 GB preselected, plus the saved policy's GPU choice.
Each template changes only the GPU and container settings; your saved CPU,
disk, price, runtime, transfer, location, and cleanup limits still apply.
Colab offers account 1 (`your-colab-account@example.com`) or account 2
(`your-drive-account@example.com`), asks for a fresh, unused session created under that
account, and installs the saved dedicated Drive credential;
the legacy choice requires a session you have already manually authenticated.

The existing direct commands still work. A command-line chooser is also available:

```bash
venv/bin/python workflows/vast/choose.py --backend colab --account 2 -- \
  --config specs/conceptarc_smoke.json --session YOUR_COLAB_SESSION

venv/bin/python workflows/vast/choose.py --backend vast -- \
  --config specs/conceptarc_smoke.json
```

`--backend colab` now uses cached authentication. `--backend colab-oauth` remains
an alias. For an explicitly requested manual fallback, use `--backend colab-manual`
with the same Colab arguments after running `colab auth -s YOUR_COLAB_SESSION`.
Cached-token failures are reported; the chooser does not switch to manual sign-in.

See the [Colab guide](../../scripts/README_colab.md) and
[cached-OAuth Colab guide](../../scripts/colab_oauth/README.md) for their own setup.
Create the Colab VM before using the chooser: `colab new` uses account 1;
`venv/bin/python scripts/colab_accounts.py --account 2 -- new ...` uses account 2.
The selected cached launcher handles Drive authentication. Both Colab accounts
can run concurrently within their quotas; use distinct session names. Saved
run records select the same account automatically when reconnecting.

## Initial local and container setup

Install the small Drive dependency set into the local environment:

```bash
uv pip install --python venv/bin/python -r workflows/vast/requirements.txt
```

Prepare your specific container with Python, the mdlARC training dependencies,
and FlashAttention already installed. Including the small Drive dependency set
avoids setup downloads; otherwise the worker installs only those helper packages.
It does **not** install Torch, download a FlashAttention wheel, compile
FlashAttention, or substitute another image. Existing dataset-building behavior
inside `launch_run.py` is retained. Use an immutable image tag or digest for
repeatability. The image must support Vast's SSH launch mode as root, `/workspace`
storage (or your configured `remote_root`), and the per-instance control variables.

Before installing helper packages or running training, the worker removes empty
and current-directory entries from inherited compiler, library, and executable
search paths. Other entries are preserved. This prevents GCC from treating the
repository's `specs/` directory as a compiler specs file. Once provenance is
backed up, a CUDA preflight initializes Triton and executes a tiny
`torch.compile` operation before preparing datasets or checkpoints. It also
checks BF16 and the required Torch compiler API. The preflight is limited to
five minutes and the remaining training allowance; a failure uses the normal
artifact backup and failure cleanup workflow. `runtime_preflight.log` and
`runtime_environment.json` record the result and path-removal counts, without
recording environment values. The supervisor stays independent of the Mac
during this check; remote handoff can be confirmed in `preflighting_runtime`.

Copy `policy.example.json` to `~/.config/mdlarc-vast/policy.json`. The default
image is `nvcr.io/nvidia/pytorch:26.05-py3`, with `min_cuda_version=13.2`.
Set your allowed GPUs, disk, price,
runtime, transfer allowances, and cleanup preference. The example values are
editable examples, not measured requirements or current market prices. You can
also pass `--policy /path/to/policy.json` and `--image YOUR_IMAGE` explicitly.

Use repeatable `--policy-override FIELD=JSON_VALUE` arguments for per-launch
changes without editing the saved policy. Overrides apply after the saved
policy, GPU template, and image options. All fields use their normal type and
constraint validation; unknown or repeated field names are rejected. The
explicit overrides and resolved policy appear in the dry-run output and saved
launch record. Recovery and retries use that recorded policy.

For an existing instance whose download speed is below the saved minimum:

```bash
venv/bin/python workflows/vast/run.py launch \
  --config specs/conceptarc_smoke.json --instance-id YOUR_INSTANCE_ID \
  --policy-override min_download_mbps=0 --dry-run
```

Zero disables the existing-instance download-capacity check. Other runtime,
price, transfer, and cleanup checks remain active. Numeric values, booleans,
lists, and null use JSON syntax. For a string, retain JSON's double quotes:
`--policy-override 'success_action="keep"'`. Remove `--dry-run` to launch.

### GPU templates and automatic NGC container selection

List the saved policy and all five templates without network access or a rental:

```bash
venv/bin/python workflows/vast/run.py templates
# Use --json for complete resolved policies, or --policy to select another base.
```

The separate JSON templates live in [`specs/vast/`](../../specs/vast/).
Select one with `--gpu-template a100-40gb`, `a100-80gb`, `h100`, `h200`, or `b200`:

```bash
venv/bin/python workflows/vast/run.py launch \
  --config specs/conceptarc_smoke.json --gpu-template h200 --dry-run
```

All templates use one GPU, CUDA >=13.2, and `26.05-py3` as the base image.
A100 40 GB accepts advertised memory between 35 and 45 GB; A100 80 GB accepts
75 to 85 GB. This distinguishes the two memory variants while allowing for
reported usable capacity and provider units. H100 includes SXM, PCIe, and NVL
with at least 75 GB; H200 includes H200 and H200 NVL with at least 130 GB;
B200 requires at least 170 GB. Both the search and local offer recheck enforce
memory bounds. Higher GPUs still have to fit your saved spending limits.

For the supported NGC PyTorch tags, `image_selection` defaults to `"cuda"`,
including existing policies without that field. The launcher selects the
highest supported tier from the chosen offer's advertised `cuda_max_good`:

| Host CUDA support | Container |
| --- | --- |
| 13.2 to below 13.3 | `nvcr.io/nvidia/pytorch:26.05-py3` |
| 13.3 to below 13.4 | `nvcr.io/nvidia/pytorch:26.07-py3` |
| 13.4 or higher | `nvcr.io/nvidia/pytorch:26.09-py3` |

NVIDIA documents CUDA 13.2.1, 13.3.1, and 13.4.1 in these releases. See
[26.07 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html),
[26.09 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-08.html),
and the [NGC published collection](https://catalog.ngc.nvidia.com/orgs/nvidia/-/collections/nvidia_dlfw/-/artifacts).
NVIDIA currently links its 26.09 notes at the `rel-26-08.html` URL.

An automatic policy with a newer base tag never selects an older container.
Custom images outside these three supported tags stay fixed. Use
`--image-selection fixed` or set `"image_selection": "fixed"` to pin the base
image. An explicit `--image` also pins it unless `--image-selection cuda` is
provided. Keep `min_cuda_version` at or above the selected image's requirement.

Selection occurs once before rental. The launch record and Drive provenance
save the resolved image, its CUDA minimum, the template name, and the offer's
CUDA/driver metadata. Watch, reconnect, and recovery use that saved image.
An offline dry run reports the selection rules; it cannot choose the final
image without an offer. The least expensive qualifying offer is still selected
using the existing allowance calculation. The runtime CUDA preflight verifies
actual operation of the selected container before training.

### Host compatibility and resource filters

Set `min_cuda_version` to the host-driver CUDA support required by your base
image. A new rental requires this explicit setting. Offer search filters on
Vast's `cuda_max_good` field, and the launcher independently rejects returned
offers with a lower, missing, or invalid value before renting. It records the
chosen offer's advertised CUDA support and NVIDIA driver version in provenance.
This is the host driver's CUDA support, separate from GPU compute capability
and the CUDA toolkit installed inside the container.

The default `13.2` is a conservative match for
`nvcr.io/nvidia/pytorch:26.05-py3`, which contains CUDA 13.2.1; CUDA selection
raises the saved minimum if it selects a newer image. Set the minimum to match
another base image when changing the container. NVIDIA supports some older-driver
compatibility configurations, but JIT/PTX compilation can require a newer
driver. The pre-rental filter uses advertised metadata; the remote CUDA
preflight still verifies actual operation. See the
[Vast offer fields](https://docs.vast.ai/api-reference/search/search-offers),
[NVIDIA PyTorch 26.05 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-05.html),
and [CUDA compatibility limits](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Set `min_download_mbps` and `min_cpu_cores_effective` to exclude slow network
offers and small CPU allocations. The example uses at least 2000 Mbps download
and 16 effective vCPUs, alongside its 32 GB allocated system RAM minimum. Search
filters on `inet_down` and `cpu_cores_effective`; the launcher also rejects lower,
missing, or invalid values in the returned offers before creating a rental.
If no offer satisfies the saved policy, it fails without relaxing these limits.
Older policies default the new CPU minimum to zero, disabling that constraint.

`cpu_cores_effective` is the CPU allocation for your offer; `cpu_cores` is the
host's total logical CPU count. For example, 8/32 means an 8-vCPU baseline on a
32-thread host. CPU capacity is shared, with a baseline proportional to the GPU
allocation and bursts possible when other workloads leave spare cycles. Filter
on the allocated count rather than the host total. CPU model and single-thread
speed still matter, so the count is a starting floor rather than a performance
guarantee. See [Vast resource allocation](https://docs.vast.ai/guides/instances/docker-environment).

Vast displays network speeds in Mbps (megabits per second): 2000 Mbps is
250 MB/s before overhead. The shared network figure is advertised machine
capacity, not reserved bandwidth or a measurement of your dataset source.
Contention, the source server and route, sequential transfers, decompression,
and disk work can all lower observed throughput. Raising this filter screens
out low advertised rates but does not guarantee 2000 Mbps transfers. See
[Vast offer card units](https://docs.vast.ai/guides/instances/choosing/find-and-rent).

Configure a Vast API key in `VAST_API_KEY` or the existing CLI location
`~/.config/vastai/vast_api_key`; alternatively pass `--api-key-file`. It is used
only locally and is never sent to the GPU. Instance-management access is enough;
account billing/key-management access is unnecessary. Keep the key outside Git.

The default private key is `~/.ssh/id_ed25519`; use `--ssh-key` for another key,
with its matching `<private-key-path>.pub` file. Before its first SSH connection,
the launcher verifies the saved rental identity and attaches only the public key
directly to that instance. This also supports team accounts, which reject
account-wide SSH key registration. The private key remains on the Mac.
The launcher uses
bounded SSH requests and a separate `known_hosts` file per run. It accepts a new
host key once and refuses a changed key on subsequent connections.
If the proxy route is slow, `MDLARC_SSH_DIRECT=1` selects the instance's
provider-advertised public IP and mapped SSH port when present. It changes only
the connection route; saved run identity and at-most-once submission checks remain.

Official API contracts used here: [offer search](https://docs.vast.ai/api-reference/search/search-offers),
[rent an instance](https://docs.vast.ai/api-reference/instances/create-instance),
[instance management](https://docs.vast.ai/api-reference/instances/manage-instance),
[paginated instance listing](https://docs.vast.ai/api-reference/instances/show-instances),
[instance SSH keys](https://docs.vast.ai/api-reference/instances/attach-ssh-key),
[per-instance control](https://docs.vast.ai/guides/instances/docker-environment).

## Authorize Google Drive once

This workflow uses its own OAuth client and refresh token for
`your-drive-account@example.com`. It does not use Vast Cloud Copy, Gmail passwords, a
service account, the Colab VM credential, or the existing Mac Colab token.

1. In Google Cloud Console, create/select an OAuth project and enable Google Drive API.
2. Configure the OAuth consent screen. For an external app in Testing, add
   `your-drive-account@example.com` as a test user. Google expires Testing refresh tokens
   after seven days; reauthorize with a new token path when needed. For unattended
   longer runs, configure publishing appropriately for your personal app before
   issuing the token. Tokens can still expire or be revoked.
3. Create an OAuth **Desktop app** client. Store its downloaded JSON outside this
   repository at `~/.config/mdlarc-oauth-clients/mdlarc-vast-oauth-client.json`,
   moving it out of Downloads. Both laptops use this path. Keep the containing
   directory mode `700` and the file mode `600`. Preserve it for future browser
   reauthorization; do not put it in Git or run records. It is separate from
   the Colab client, `~/.config/mdlarc-oauth-clients/mdlarc-oauth-client.json`.
4. Run the setup command and sign in as the experiment account:

```bash
venv/bin/python workflows/vast/setup_drive.py \
  --client-secrets ~/.config/mdlarc-oauth-clients/mdlarc-vast-oauth-client.json
```

The browser consent requests Drive-only offline access. Setup rejects a different
account and verifies access to the existing `mdlARC/runs` folder without writing
Drive files. It creates a mode-600 token at
`~/.config/mdlarc-vast/drive-token.json`. Existing credentials are not overwritten;
use a new `--token` path for reauthorization, then pass that path to launch/watch/sync.

For example, if this renewed-token destination does not already exist:

```bash
venv/bin/python workflows/vast/setup_drive.py \
  --client-secrets ~/.config/mdlarc-oauth-clients/mdlarc-vast-oauth-client.json \
  --token ~/.config/mdlarc-vast/drive-token-renewed.json
```

Choose a different unused token filename for later renewals. After the account
and Drive access checks succeed, pass the new token file through `--drive-token`
to the relevant Vast commands. Keep the existing token for saved runs that use
its path. Moving the OAuth client JSON into the private config directory does
not require reauthorization and does not change any cached or VM-side token.

At launch, the token transfers over SSH into a private directory outside source
and result folders. The VM can refresh it independently of the Mac. Code restricts
writes to the verified `runs` folder and selected branch/session; reads of dataset
inputs must stay under mdlARC, and continuation checkpoints must stay under the
selected branch. Existing backup files are immutable; this workflow has no Drive
delete, share, move, or overwrite operation. The experiment account needs enough
Drive storage for the saved checkpoints and artifacts.

The existing folder IDs and reviewed task-variation archive identity are in
`drive.example.json`. It contains no wheel reference. Pass `--drive-config` for
another approved configuration. A saved run uses its launch-time configuration
when recovering, even if this example changes later.

Google references: [offline OAuth](https://developers.google.com/identity/protocols/oauth2/web-server#offline),
[token expiration](https://developers.google.com/identity/protocols/oauth2#expiration),
[resumable uploads](https://developers.google.com/workspace/drive/api/guides/manage-uploads).

## Preview and launch

Preview is offline, requires no API keys/OAuth/SSH, and rents nothing:

```bash
venv/bin/python workflows/vast/run.py launch \
  --config specs/conceptarc_smoke.json \
  --policy workflows/vast/policy.example.json --dry-run
```

The scientific spec is validated by the existing `launch_run.py` schema.
Infrastructure settings remain in the separate policy. The selected working-tree
source, including uncommitted changes, is packaged; only the selected experiment
spec, relevant code/assets/grouped data, and the Vast worker are included. Wheels,
credentials, old results, checkpoints, unrelated specs, and the other workflow's
implementation are omitted. There is no GitHub token or requirement to commit/push.

Launch using your configured policy:

```bash
venv/bin/python workflows/vast/run.py launch \
  --config specs/conceptarc_smoke.json \
  --intent "Validate Vast training and backups" \
  --changed-from-previous "Run the selected experiment on the prepared Vast image"
```

Omit both note arguments to use the existing two Mac notes dialogs. `--session`
is optional; a unique Vast-prefixed session is generated. `MDLARC_RUN_BRANCH` or
the active Git branch determines the run branch; `--branch` overrides it explicitly.
The branch is captured once and never guessed again after a branch switch.

To use a manually rented, idle instance, add `--instance-id <ID>` to `launch`.
This path never searches offers or creates another rental. It verifies the
selected GPU, runtime compatibility, allocated CPU/network capacity, and price
against the saved limits. The rental's actual container and disk allocation are
recorded rather than replaced; CUDA image selection applies only to new rentals.
The runtime allowance starts at the original rental time, including time already
spent loading or idle. Total cost is estimated over the full declared allowance
with the usual transfer reserves. Existing managed labels, local run identities,
remote run markers, or GPU compute processes prevent a new submission.
The launcher assigns and verifies a unique run label, then uses the normal
handoff, Drive backups, deadline guard, and cleanup policy. An uncertain adoption
is inspected by its saved identity; never launch it again under a new session.

An explicitly selected, kept B300 can also receive a fresh experiment after its
previous runs finish. The launcher requires its current label to match a saved
run, verifies finalized completion and the canonical Drive result archive for
every saved run on that instance, and checks GPU/process idleness before claiming
it. Previous records and source snapshots remain intact. The new run uses the
current policy while counting runtime from the original rental start; it never
stops the B300 or creates a replacement rental.

The launcher selects the least expensive qualifying on-demand offer, saves the
exact source/spec and a credential-free identity, rents once, verifies the SSH
transfer, and starts an at-most-once detached worker. It confirms handoff after
the worker verifies Drive access and backs up its provenance, or after training
starts. Input preparation then proceeds remotely. `--detach` exits then; otherwise it
watches and recovers artifact snapshots. Closing the watcher does not stop or
resubmit training. Keep the laptop awake until handoff has been confirmed.

For manually authored temporary specs, use `--standalone-spec`. Sweep-generated
specs require their matching `--trial-record`; the original trial/template are
preserved in launch provenance. A Drive checkpoint continuation does not require
either flag. Set `skip_data_build=false` for the fresh workspace. Existing local
checkpoint strings are rejected; use the normal Drive identity object to stage
the checkpoint directly on the GPU with size/MD5/SHA-256 verification. Derive a
continuation from the original saved spec and leave scientific choices unchanged
except those you authorize; `pretrain.epochs` remains the total epoch target.

Large Drive inputs use the cached credential to stream 256 MiB API byte ranges,
with 8 MiB streaming buffers and bounded recovery from the last received byte.
This is the same transfer method used for Colab checkpoint inputs. The final
size, MD5 and pinned SHA-256 must match before the input is published locally.

Checkpoint cadence comes from your spec. The workflow does not change it. The
existing atomic checkpoint hook queues completed files for the background worker.
NVARC or other private dataset access can use `--secrets-file` pointing to a private
JSON outside the repository containing `KAGGLE_API_TOKEN`, `KAGGLE_USERNAME`,
`KAGGLE_KEY`, or `HF_TOKEN`. It transfers separately, is excluded from artifacts,
and known credential values are redacted from training console output.

## Backups, cleanup, and cost controls

For explicitly requested local checkpoints, launch with
`--policy-override 'checkpoint_upload_mode="manual"'`. Automatic checkpoint
uploads, including finalization/recovery uploads, are disabled. Existing
`checkpoint_keep_last` retention applies directly on the VM; logs, source and
non-checkpoint result records continue backing up normally. The completion
record distinguishes successful record backups from checkpoints left only on
the VM, and automatic cleanup keeps the instance. Upload individual checkpoints
only when the user explicitly requests them. This run-specific override does
not change the saved infrastructure policy or older runs.

Drive receives the exact source/spec/provenance, completed checkpoints, logs,
metrics/predictions, checkpoint ledger, and periodic non-checkpoint snapshots.
Resumable uploads preserve their private session state and use preallocated Drive
file IDs. An uncertain response is reconciled by that same ID, avoiding duplicate
files. Uploads verify size, MD5 and available provider SHA-256; recorded SHA-256 is
also used on recovery. Partial checkpoints are excluded from artifact snapshots.

After training exits, the worker drains checkpoint uploads, checks that every
completed output checkpoint is covered, uploads a final result archive, and
publishes a verified completion manifest before cleanup. Local recovery is not a
cleanup prerequisite. Success uses `success_action` (`destroy`, `stop`, or `keep`);
failure uses `failure_action` (`stop` or `keep`). **Unverified backups never allow
destruction.** The example destroys verified successful runs and stops failed runs
with their disk preserved. Stopped instances continue storage charges; persistent
volumes are not created by this workflow.

B300 rentals follow the standing rule in `AGENTS.md`: keep the VM powered on
after success, failure, transport errors, and runtime expiry. This overrides
saved stop/destroy cleanup actions, including the remote deadline guard, local
watcher, and independent watchdog. Training still ends at its original time
limit and finalizes backups; the VM continues incurring compute charges until
the user explicitly requests shutdown. The `stop` command is reserved for that
explicit request.

`max_runtime_hours` includes startup, training, and final uploads. Training stops
before the deadline to reserve `backup_grace_seconds`. A separate remote deadline
guard can stop compute even if the training supervisor crashes. File-transfer
allowances are shared by the supervisor and uploader. The offer is rechecked
against the policy, including a bandwidth reservation, before rental.

`max_total_usd` is an **estimated allowance**, using the quoted hourly price,
runtime, and configured upload/download reserves. It is not a provider-enforced
billing cap: network retransmissions, dataset-builder traffic outside managed
transfers, storage retained after stopping, or API/host
failure can exceed that estimate. A dead host cannot execute its own deadline
guard. For independent deadline enforcement, deploy the included `watchdog.py` on
an always-on machine with its own instance-management API key:

```bash
python3 workflows/vast/watchdog.py --leases /path/to/shared/leases
```

Then pass `--lease-export-dir /path/to/shared/leases` when launching, using a
mounted/shared directory that the controller can actually read while the Mac is
offline. The lease is exported before rental submission, including the unique
label, so an ambiguous submission can also be inspected by the controller. The
watchdog stops only matching rentals at their deadline; it never starts training,
rents a replacement, or destroys data. `--once` supports an external scheduler.
This implementation supplies the controller, but does not provision an always-on
machine or claim a hard spending guarantee during a Vast API outage.

## Observe and recover the original run

Use the saved session and branch, rather than launching again:

```bash
venv/bin/python workflows/vast/run.py status --branch newreason --session YOUR_SESSION
venv/bin/python workflows/vast/run.py watch --branch newreason --session YOUR_SESSION
venv/bin/python workflows/vast/run.py sync --branch newreason --session YOUR_SESSION
```

`status` labels its cache as the last local observation. `watch` checks Drive and
the saved instance independently. Network/authentication failures remain unknown;
two successful authenticated listings must omit the original instance to mark it
unavailable. `sync` reads Drive only and works after the GPU has been destroyed.
It downloads verified non-checkpoint archives into immutable
`artifacts/<SHA256>/` directories and records the latest location in
`latest_artifacts.json`. It does not download checkpoints or overwrite your note.

For an uncertain rental response, `reconcile --session ...` looks up the unique
saved label and records the original instance ID. It creates no rental and
submits no training. After an incomplete transfer, `handoff --session ...`
continues setup on that same instance, checks immutable uploaded files, and never
restarts a run with an existing started marker. `start --session ...` explicitly
restarts a stopped original instance within its original allowance.

For pending checkpoint backups on a retained, running instance, use
`recover-backups --session ...`. It retries saved transfer identities and scans
completed checkpoints without running training. Recovered manifests are saved
under immutable names; later `sync` selects the latest one. If the original
allowance has expired, the command refuses a new paid recovery interval rather
than silently extending it. The remote private directory and retained disk must
be preserved for this recovery. `stop --session ...` requests a stop of only the
saved, matching instance and preserves its disk.

Manual notes stay in `results/<branch>/vast_runs/<session>/note.md`; their launch
contents are in the backed-up provenance. Later local note edits are not silently
made canonical. Sync them to the verified run folder when requested.

### Explicit retry after a finalized setup failure

The `continue` command also resumes pretraining from a verified output checkpoint
on the same powered-on instance. Derive the temporary spec from the parent's exact
saved spec, set `checkpoint` to its uploaded Drive identity and `skip_data_build=true`,
and retain the parent dataset. Set `pretrain.epochs` to the total target, not the
number of additional epochs. To cool a stable run from completed epoch S through
total epoch T, use `wsd_decay_start_pct=S/T` and keep `resume_warmup_pct=0`.
Use `--reuse-checkpoint pretrain/<saved-file>.pt`; nested cached dataset payloads
are verified and hard-linked alongside the checkpoint. A manually backed-up parent
may have been intentionally stopped: finalized record backups plus a separately
verified checkpoint in that parent's Drive folder are sufficient. Training source
comes from the parent's saved snapshot; source fixes require explicit selection.

For a newly requested fine-tune using the checkpoint and data already on a
running VM, derive a temp spec from its uploaded source with `finetune_only=true`,
`enable_finetune=true`, `skip_data_build=true`, the exact Drive checkpoint identity,
and the requested fine-tune stages. Use `continue --session PARENT --new-session CHILD
--config SPEC --reuse-checkpoint inputs/pretrain.pt --source-update launch_run.py`.
The parent must have finalized verified backups and no active compute. The worker
checks the local checkpoint against its size/SHA and Drive metadata, then hard-links
it and the unchanged cached JSON data into the new workspace. It downloads no
checkpoint bytes and runs no pretraining. `--runtime-hours` grants a bounded window
for a newly authorized continuation, capped by the original total allowance,
quoted hourly price, elapsed rental time, and reserved bandwidth. It never changes
the B300 no-stop rule. Without that option, the original deadline is preserved.

For an interrupted fine-tune, use `finetune_resume=true` and the verified Drive
identity of its saved fine-tune output. Set each fine-tune stage's `epochs` to
the total target and pass that output's relative path with `--reuse-checkpoint`.
The continuation preserves the parent dataset and restores optimizer, scheduler,
epoch and per-rank RNG progress instead of starting a fresh adaptation.

For evaluation alone, set `eval_only=true`, `enable_finetune=false`,
`finetune_only=false` and `finetune_resume=false`, and point the evaluation stage
at `checkpoint`. The same `continue` command verifies and reuses the saved output
checkpoint and cached data without downloading it or running training again.

An explicitly authorized continuation can extend the original allowance with
`--deadline-at 2026-10-25T12:30:00+05:30` and
`--policy-override max_runtime_hours=... --policy-override max_total_usd=...`.
The deadline is the hard rental deadline; the saved backup grace is reserved
before it. The launcher checks the current rental price and transfer reserves,
retains cumulative transfer usage, and records the selected overrides in the new
launch without editing the parent. Backup allowance overrides use the same
`--policy-override` option. A running supervisor holds its plan in memory, so
this extension takes effect through a saved-checkpoint continuation.

Set `max_upload_gb=null` to remove the upload-volume cap and provide a positive
`upload_reserve_gb` for cost estimation; that reserve does not limit transfers.
`checkpoint_upload_every_epochs=60` uploads periodic checkpoints only at epoch
multiples of 60 while retaining the latest local save. Scheduled uploads are
protected until verification, and finalization also backs up the latest saved
checkpoint. The scientific spec still controls local save frequency.

`start` powers on the original instance; it does not rerun a finalized worker.
For a failed run with verified final Drive backups and no completed
checkpoints, use an explicit retry with a fresh session name:

```bash
venv/bin/python workflows/vast/run.py retry --branch YOUR_BRANCH \
  --session ORIGINAL_SESSION --retry-session NEW_RETRY_SESSION --dry-run
venv/bin/python workflows/vast/run.py retry --branch YOUR_BRANCH \
  --session ORIGINAL_SESSION --retry-session NEW_RETRY_SESSION --detach
```

The retry accepts a stopped instance or an idle, powered-on B300. It verifies
that the GPU has no compute process and no training/supervisor process remains
before reusing a powered-on B300, and never stops it to prepare a retry.

The preview is offline. The retry uses the same saved instance, image, scientific
spec, source, original rental deadline, cleanup policy, and cumulative transfer
usage. It adds `scripts/colab_drive.json` if a historical variations bundle
omitted that non-secret dependency, verifying its archive identity against the
saved Drive configuration. It records a new label, workspace, and Drive session
while preserving the failed run's files and markers. It never rents a replacement
or resets the allowance. Checkpoint-bearing failures require a continuation spec.
An uncertain retry must be inspected using its new session; repeated `retry`
commands refuse existing records. Use `start` or `handoff` for that saved retry
only after reconciling its actual state.

When the user explicitly asks to skip repeated native CUDA/FA4 tests, add
`--skip-native-validation` to the retry. Its runtime record marks validation as
skipped, never passed. Required FA4 package presence and the normal identity,
backup and submission safeguards remain enforced. Explicit supervisor fixes can
be included with `--source-update workflows/vast/worker.py`.

Remote training defaults to expandable CUDA allocator segments for variable
family-batch shapes; an existing `PYTORCH_ALLOC_CONF` or
`PYTORCH_CUDA_ALLOC_CONF` override takes precedence.

To repeat only a failed evaluation, pass `--eval-checkpoint /path/to/identity.json`,
`--eval-batch-size N`, and `--eval-token-budget N` to `retry`. The identity uses
the normal spec fields `drive_id`, `name`, `size`, and exact `sha256` (optionally
`source_branch`). It must match a completed checkpoint in the parent's verified
Drive backup ledger. The retry derives an `eval_only=true` spec from the uploaded
snapshot, retains its evaluation groups, augmentation and decoding settings,
and skips all training stages. Checkpoint bytes transfer directly from Drive to
the VM. Explicit `--source-update launch_run.py --source-update src/evaluate.py`
can add budget configuration support to historical snapshots; the exact file
changes remain recorded. No checkpoint bytes pass through the Mac.

Before the first upload of a B300 workspace, handoff refreshes only the shared
controller helper, worker, and Vast API client to enforce the no-stop rule.
It retains the prior source archive and launch record and records both file
hashes. All training source, scientific spec, and pinned data remain frozen.
An already uploaded or dispatched historical worker requires inspection rather
than an automatic replacement.

For a diagnosed source bug, repeat `--source-update src/FILE.py` on the preview
and retry commands to replace only those named training Python files from the
current checkout. Each file must already exist in the original snapshot.
Repository-relative Python paths under `src/` or `dataset_building_scripts/`,
and `launch_run.py`, are accepted; links, traversal, duplicate paths, specs and
other files are rejected. The retry records each old and new file hash. The
saved spec, pinned dataset, other source files, budget, deadline and transfer
accounting retain the same checks as an unchanged-source retry.

After a memory failure, pass both `--batch-size 16` and
`--gradient-accumulation-steps 4` to explicitly change the pretraining
microbatch. Their product must match the saved effective batch. Only these two
pretraining fields change; model, data, epochs, evaluation, source and rental
limits retain their saved values. The preview and new run record include the
old/new values and new spec checksum. The original spec is preserved. All
finalized-failure, backup, no-checkpoint and same-instance checks still apply.

When an effective-batch change is intentional, also provide its exact value:
`--batch-size 12 --gradient-accumulation-steps 5 --effective-batch-size 60`.
Without that explicit value the original effective batch remains mandatory.
The value must equal the requested physical batch times accumulation and is
recorded with the two changed pretraining fields. It changes no learning rate
or other training setting automatically.

## Offline verification

```bash
venv/bin/python -m pytest -q workflows/vast/tests
venv/bin/python -m compileall -q workflows/vast
```

Tests cover constrained offer selection, ambiguous rental/upload responses,
account/folder/branch restrictions, checksum failures, shared transfer allowances,
safe archive extraction, checkpoint exclusion, credential redaction, disconnected
watchers, the three-workflow selector, and a CPU simulation of detached training,
checkpoint backup, final artifact verification, and cleanup. They use no rental,
GPU training, private Drive data, or existing session. Live validation requires
the chosen container and initial account consent.
