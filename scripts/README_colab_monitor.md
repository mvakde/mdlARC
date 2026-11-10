# Local Colab monitor

Double-click `scripts/Colab Monitor.command`, or run from the checkout:

```bash
venv/bin/python scripts/colab_monitor.py --open
```

Open <http://127.0.0.1:8766>. Keep its terminal running; Ctrl-C stops only the
monitor UI. Starting it again reuses the same monitor. Use `--port 8767` if the
default port belongs to another application.

The page lists current VMs on both saved Colab compute accounts. It does not
allocate, stop or prune VMs, execute kernel code, change launcher identities,
or interfere with existing training watchers. Authentication uses the two
existing Colab CLI token files, separately from the Drive-only token. Missing
or expired grants are reported; renew them with the existing account workflow.
There are no browser login prompts inside the monitor.

Each VM has two independent buttons:

- **Speed / GPU** reads current GPU memory and utilization immediately, then
  takes a second progress sample after 30 seconds. Speed is an estimate from
  logged epochs/steps (or evaluation batches), including any pause within the
  sample. Coarse logging limits precision. A new stage or changing epoch length
  invalidates the sample. No progress is reported as zero observed progress,
  never proof that training stalled. Memory is a snapshot, not peak allocation.
- **Training metrics** reads the live console and available scored artifacts.
  Latest step losses and completed-epoch accuracies are labeled separately.
  Training exact-output accuracy, teacher-forced evaluation accuracy and
  held-out full-decode scores are distinct. A run without scored output shows
  pending, including runs with no configured intermediate evaluation.

Saved comparison references at
`results/<branch>/status_checks/<session>/comparison_references.json` are
picked up automatically. The table compares the same completed epoch and shows
historical final scores separately. It retains the reference caveats about
datasets, context format, conditioning, learning rates and schedule differences.
Historical branches are comparison sources, never evidence of a live VM.

The UI uses plain HTML/CSS/JavaScript and a standard-library Python HTTP server.
It has no build, framework, model/API calls, scheduled polls or new dependencies.
Colab imports live in two lazy account workers, which exit after 20 seconds idle.
The workers reuse connections while warm and read bounded console tails; a
server ignoring HTTP Range is streamed with a 4 MiB transfer ceiling. The server
keeps compact metric/reference caches in memory. It never downloads checkpoints
or predictions. Refreshing the VM list is parallel across the two accounts;
buttons work independently. All Colab VM operations are HTTP GET requests;
OAuth token refresh uses the credential provider's normal authentication flow.
For tracked runs, the exact saved upload spec and matching original VM identity
avoid redundant remote directory/spec reads. Untracked VMs are discovered from
their actual files. Reused aliases cannot attach an old VM's saved spec.

The server binds to `127.0.0.1`. Host/origin checks and a per-launch request
token protect the local API; runtime credentials remain inside account workers
and never reach the browser, process arguments or logs. The monitor serves only
its three UI assets. It does not expose the repository as a file server.

Reusable code/UI, this guide, the launcher and tests belong in Git. Development
logs, measurements and other run evidence belong under ignored
`results/<branch>/monitor/`; the monitor itself creates no persistent files.

Targeted verification:

```bash
venv/bin/python -m pytest -q tests/test_colab_monitor.py
```
