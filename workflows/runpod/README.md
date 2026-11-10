# Runpod experiment workflow

Runpod Pods use the same detached training supervisor, runtime preflight,
checkpoint uploader, verified Drive backups, and checkpoint-free recovery as
the [Vast workflow](../vast/README.md). The experiment JSON is unchanged.
Choose **Runpod** in `workflows/vast/Run.command`, or launch from the command line.
The Mac can disconnect after the launcher confirms remote handoff.

Local run records are ignored at `results/<encoded-branch>/runpod_runs/<session>/`.
Drive remains canonical at `mdlARC/runs/<branch>/<session>`. Each launch preserves
the exact working-tree source, scientific spec, resolved infrastructure policy,
pod ID, unique pod name, and original deadline. Existing Vast and Colab records
remain separate. No Runpod SDK or CLI installation is required.

## Fast startup defaults

The launch path prioritizes time to the first optimizer update. Routine launches
do not run the full test suite or optional model ablation validation. Required
pod ownership, idle-workload, credential, pinned-input and backup checks remain;
do them once, and investigate only a phase that actually fails.

- Keep tiny dataset files on local disk. `/workspace` can be a network volume:
  writing 108,008 family JSON files there was the main deployment bottleneck.
  Runpod sets `TMPDIR=/tmp`; family payloads live in
  `/tmp/mdlarc-family-cache/<package-sha256>/`. A completion marker lets later
  runs reuse the cache without unpacking or rereading all members. The pinned
  package is still verified and training checks each selected sample's hash.
  Inventories, checkpoints, logs and credentials retain their existing storage.
- Transfer data separately from source. New Runpod family launches keep the
  pinned ZIP out of `source.tar.gz`. The pod keeps one checksummed archive at
  `/workspace/mdlarc-runpod/.inputs/<sha256>.zip` (under the configured root).
  Each source snapshot references that exact input through `source_inputs` in
  `launch.json`. A kept pod can seed this cache from a verified earlier run;
  changing code does not upload the same dataset from the Mac again. The pod
  backs up the ZIP directly to Drive; `source_input_backups.json` records its
  verified identity for recovery alongside the code snapshot and spec.
- Prepare the NGC environment automatically. SSH does not inherit all image
  entrypoint variables. The supervisor restores CUDA headers, library and tool
  paths, including Triton's `ptxas`, `cuobjdump`, `nvdisasm` and libdevice.
  Missing Drive helpers and pinned FA4 handle the image's PEP 668 marker during
  installation, retaining the supplied Torch installation.
- Reuse validation when the runtime matches. `cache_runtime_validation=true`
  is the Runpod default. A passed CUDA/BF16/compile/FA4 report is cached on local
  disk, keyed by pod, container start, host boot/driver, Python, package versions,
  device visibility, compiler files and validation code. A changed runtime or
  missing cache requires one bounded native validation; an unchanged cached
  runtime does not rerun those tests. Set the policy field to `false` to force
  validation. No benchmark or full test suite belongs in this startup path.
- Make model validation opt-in. `validate_model_on_start=false` is the Runpod
  default; it skips `launch_run.py`'s optional model CUDA validation while
  retaining the native runtime validation above. To honor a spec's
  `validate_cuda=true`, explicitly override `validate_model_on_start=true`.
  Training, optimizer, dropout, MTP, data sampling and evaluation settings are
  unaffected by this startup policy.
- Measure actual training. Handoff, data preparation, model setup, compilation
  and training are separate states. `startup_timing.json` records elapsed time
  from the launch request through the first optimizer update. The supervisor
  reports `training` only after that first update marker; an epoch heading or
  an alive supervisor alone is not evidence of an optimizer step. Older uploaded
  source without these markers retains its legacy `running` state, which means
  only that the experiment process was submitted.

Use an explicitly supplied existing rental with `--pod-id`, keep B300 capacity
powered on, and reconcile a saved session after an uncertain response. Never
allocate another pod or resubmit training to work around uncertainty. Paid GPU
time is for training; optional diagnostics need an explicit reason.

The local-disk data path reached pretraining in the observed deployment. These
additional warm-cache and split-transfer defaults have not yet been timed in a
new paid launch; they do not modify an already uploaded/running session.

## Initial setup

Install the existing small Drive dependency set if it is not already installed:

```bash
uv pip install --python venv/bin/python -r workflows/vast/requirements.txt
mkdir -p ~/.config/mdlarc-runpod
chmod 700 ~/.config/mdlarc-runpod
cp workflows/runpod/policy.example.json ~/.config/mdlarc-runpod/policy.json
```

Edit the policy for your GPU types, image, CPU/RAM requirements, disk, network,
cost allowances, allowed CUDA versions, and cleanup preference. Runpod has its
own policy because its GPU identifiers and allocation filters differ from Vast.
Example prices and limits are editable allowances, not live quotes or measured
requirements. A misspelled or missing policy path fails instead of using defaults.

Set `RUNPOD_API_KEY` in your local environment, or save the key in
`~/.config/mdlarc-runpod/api-key` with mode `600`. Alternatively use
`--api-key-file /path/outside/the/repository`. Keep the key out of command-line
arguments, specs, Git, and run records. The launcher uses this key only locally.
It never sends the account key or your private SSH key to a pod.
API requests identify the client as `mdlARC-Runpod/1.0`. Runpod's Cloudflare
front end can reject Python's default client signature with HTTP 403/error 1010;
the error is reported separately from an ordinary API permission failure.

Use `~/.ssh/id_ed25519` and its paired `.pub` file, or pass `--ssh-key`. The pod
receives only the public key. Creation exposes public TCP SSH on port 22 and
starts an SSH daemon with password login disabled; an Ubuntu/Debian image can
install `openssh-server` at startup if missing. Other images must include
`/usr/sbin/sshd`. The image's entrypoint is retained and must execute its command
arguments (as the NGC entrypoint does). Use a pinned image with Python, Torch, training dependencies,
and FA2 already installed. The shared supervisor installs only Drive helpers
and the pinned FA4 package when the scientific spec selects FA4; it does not
replace Torch. CUDA/BF16/compilation are checked remotely before input preparation.
The Runpod example uses `nvcr.io/nvidia/pytorch:25.11-py3` with host CUDA `13.0`.
NVIDIA documents CUDA 13.0.2 in that container; see the
[25.11 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html).
Images are fixed on Runpod, with explicit `allowed_cuda_versions` sent to its
scheduler. The launcher checks the live API's accepted versions before requesting
a pod and rejects a known NGC image with incompatible allowed host versions.
Existing user policies are never rewritten or silently downgraded. The original
26.05/CUDA 13.2–13.4 example was rejected by Runpod's live API, which advertised
versions only through 13.0 during the 2026-10-24 smoke attempt.

Reuse the separately verified, Drive-only `your-drive-account@example.com` remote-run
credential at `~/.config/mdlarc-vast/drive-token.json`, or pass `--drive-token`.
This is the dedicated infrastructure Drive credential, separate from Colab
compute login. It is transferred privately to the supervisor outside the source
snapshot. For initial authorization or renewal, follow
[Vast Drive setup](../vast/README.md) and use its existing external OAuth client
and a fresh token path. The launcher verifies the Drive account and scope before
renting, and does not fall back to manual authentication.

Runpod provides `RUNPOD_POD_ID` and a **pod-scoped** `RUNPOD_API_KEY` in the pod
environment. Startup saves just those values in a private control file for SSH
sessions. The supervisor requires a matching pod ID before handoff. Training
and dependency installation receive no Runpod or Vast control key. If scoped
control credentials are unavailable, setup fails before training; the launcher
attempts to stop only the saved pod. It never substitutes the account key.
The SSH host key is generated on the persistent volume so reconnecting after a
stop/start retains the per-run known-hosts check.

## Launch

Preview packaging and validation offline, without credentials or a rental:

```bash
venv/bin/python workflows/runpod/run.py templates

venv/bin/python workflows/runpod/run.py launch \
  --config specs/conceptarc_smoke.json \
  --policy workflows/runpod/policy.example.json --gpu-template h200 --dry-run
```

Launch using the saved policy:

```bash
venv/bin/python workflows/runpod/run.py launch \
  --config specs/conceptarc_smoke.json --gpu-template a100-80gb --detach

venv/bin/python workflows/vast/choose.py --backend runpod -- \
  --config specs/conceptarc_smoke.json --gpu-template h200 --detach
```

The GPU templates are `a100-40gb`, `a100-80gb`, `h100`, `h200`, and `b200`.
They change only the GPU IDs/count. All other saved constraints remain active.
The historical A100 40 GB ID is usable only when the live Runpod catalog reports
it with stock; absence is an error and never substitutes an 80 GB card.
Use `--image` or repeatable `--policy-override FIELD=JSON_VALUE` for explicit
per-launch changes. For example, `--policy-override max_hourly_usd=4` or
`--policy-override 'success_action="stop"'`. Overrides are saved and do not
change the policy file. Headless launches can supply both `--intent` and
`--changed-from-previous`; otherwise the existing Mac notes dialogs are used.

The launcher queries the Runpod GPU catalog and chooses the cheapest allowed
GPU with reported stock and a quote within the policy. It requests that exact
GPU, on-demand allocation, public SSH, the CPU/RAM/network floors, locations,
and CUDA versions. Constraints are never relaxed after an allocation failure.
The allocated pod's identity, GPU/count, image, CPU/RAM, disk, cloud tier,
data center, advertised network rates, and price are checked again before any
source or Drive credential upload. An out-of-policy pod is stopped with its
volume retained. Runpod's catalog quote is an estimate: creation has no price
cap parameter, so a price change can incur brief rental cost before the recheck.
The total allowance reserves `storage_reserve_usd` in addition to compute;
it is not a provider-enforced account billing cap. Stopped volume storage
continues to be billed until the pod is deleted.

Checkpoints move directly between the pod and Drive. Continuations use Drive
checkpoint identities in an ignored `specs/temp specs/` config and the same
launcher. `pretrain.epochs` remains the total target. A sweep config needs its
matching `--trial-record`; manually authored configs can use `--standalone-spec`.
Pass `--pod-id POD_ID` to adopt an existing idle rental instead of creating a
pod. Set explicit GPU, image and cost policy overrides matching that rental.
Adoption verifies its API identity and SSH endpoint, requires finalized verified
backups for previous managed sessions, rejects active GPU/training processes,
and uses a new dedicated workspace.
Its training allowance starts at adoption. Adopted pods are always kept powered
on after errors, completion and deadlines; only training ends at its limit.
No remote lifecycle credential is needed for preserved adoption. The account
key stays local, and the dedicated Drive token still enables verified backups.
The shared supervisor restores installed CUDA toolkit paths for SSH sessions
that do not inherit the container entrypoint environment.
Use `handoff` on the saved session to reconcile an interrupted setup.

For a finalized failure with verified backups, `retry --session ORIGINAL
--retry-session NEW --source-update dataset_building_scripts/family_data.py
--skip-cuda-validation --detach` reuses the exact saved source/spec, applying
only the named source fixes. It retains the original pod and deadline.
`--skip-cuda-validation` reuses the same pod's passed CUDA/FA4 report and skips
the optional model validation; it requires a matching runtime fingerprint.
An existing legacy report without a fingerprint can be migrated without rerunning validation when
the same container predates its passed validation, Python/Torch/FA4 and CUDA
paths match, attention validation code matches, and package metadata has not
changed since the proof. Otherwise it fails closed.
Use `--restart-config "specs/temp specs/replacement.json"` for an explicitly
requested fresh restart with a replacement spec, even when the failed parent
saved checkpoints. All parent checkpoint backups are verified and retained;
the replacement spec must not load a checkpoint. The original pod, source
snapshot, infrastructure policy, and deadline are retained. Only named
`--source-update` files and the replacement spec may change training source.
Remote
family payloads use a disposable local disk cache by default for every remote
run, including fresh launches. Runpod training processes use `TMPDIR=/tmp`, so
the cache is `/tmp/mdlarc-family-cache/<package-sha256>/`; matching payloads
are reused without rewriting them. The pinned package, inventories, logs,
checkpoints, and backup credentials remain on persistent `/workspace` and keep
their existing Drive backup workflow. The payload cache can be rebuilt from
the pinned package after the container's local disk is lost.

## Observe and recover

Use the original branch and session printed at launch:

```bash
venv/bin/python workflows/runpod/run.py status --branch BRANCH --session SESSION
venv/bin/python workflows/runpod/run.py watch --branch BRANCH --session SESSION
venv/bin/python workflows/runpod/run.py sync --branch BRANCH --session SESSION
```

`status` reports the last local observation. `watch` polls the original pod and
verified Drive manifests; `sync` recovers only checkpoint-free artifacts from
Drive, even after the pod is deleted. Network/authentication failures remain
unknown; absence requires two successful authenticated listings. One watcher
per local session is enforced. None of these commands creates a pod or training.

If creation loses its response, use `reconcile --branch BRANCH --session SESSION`.
It finds the unique saved name, records that pod's ID, and starts no training.
Never repeat `launch` to resolve an uncertain submission. If setup was incomplete,
use `handoff` on that same branch/session. The permanent remote submission marker
and worker lock prevent duplicate training after a lost SSH response. Handoff
uses the saved source, spec, policy, and deadline, even after local edits.

Use `stop` to retain the original pod's volume. Use `start` only to restart that
same pod within its original allowance, then `recover-backups` to finish missing
Drive uploads without training. Re-provide `--drive-token` on `handoff` if you
used a non-default token path. Missing GPU capacity on restart is reported;
there is no replacement rental or automatic zero-GPU fallback.

Success cleanup follows `success_action`; destruction requires a successful
run and finalized, verified source, checkpoint, and artifact backups. Failure
cleanup supports `stop` or `keep`. Run files live on `/workspace` so stopping
preserves them. A separate pod-side deadline process stops the pod at the
original deadline, including when training or backup finalization fails.
The training deadline reserves `backup_grace_seconds` for finalization.

For enforcement independent of a hung pod, pass a Runpod-only shared directory
using `--lease-export-dir` and run this on an always-on machine:

```bash
venv/bin/python workflows/runpod/watchdog.py --leases /path/to/runpod-leases
```

The watchdog verifies the saved pod ID/name before requesting a stop; an
uncertain creation can be found by its saved unique name. A pod-side process
alone cannot enforce a deadline if the pod or its network is unresponsive.

Small Runpod research records participate in the standing
[cross-laptop record sync](../../scripts/README_record_sync.md). After upgrading
the selector, reinstall it on each laptop with
`venv/bin/python scripts/sync_records.py install`; launch identities, credentials,
checkpoints, and raw logs remain excluded.

## Validation and API references

Runpod's offline tests live with the shared workflow tests so the normal full
suite includes them:

```bash
venv/bin/python -m pytest -q workflows/vast/tests/test_runpod.py
venv/bin/python -m pytest -q tests workflows/vast/tests
```

Offline tests do not establish live pod allocation, scoped-key API permissions,
SSH readiness, CUDA compatibility, or training throughput. Fresh-pod allocation
and automatic cleanup have not been validated on a paid Runpod pod. Preserved
adoption of an existing B300 exercised SSH handoff, Drive provenance/artifact
backups, native CUDA/BF16/compile/FA4 validation, and entry into pretraining.
The newer fast-start refinements above have not yet been timed through a fresh
launch.

- [Runpod REST pod creation](https://docs.runpod.io/api-reference/pods/POST/pods)
- [Pod response and connection fields](https://docs.runpod.io/api-reference/pods/GET/pods/podId)
- [GPU catalog, stock, and pricing queries](https://docs.runpod.io/sdks/graphql/manage-pods)
- [Pod-scoped environment credentials](https://docs.runpod.io/pods/templates/environment-variables)
- [Public SSH and file transfers](https://docs.runpod.io/pods/configuration/use-ssh)
