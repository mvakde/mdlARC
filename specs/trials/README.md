# Sweep trial records

Keep reusable base settings in `specs/`. Give each planned trial one stable ID
and a small JSON record under `results/<branch>/experiments/trials/`. `template` is a repository-relative path. The
`overrides` object merges into that template; nested objects merge, while
arrays and scalar values replace the template value. A record must select
exactly one finetune named by `trial_id`; its optional evaluation uses the same
ID. The template leaves `finetunes` and `evals` empty.

`example_arc2_b8_e330.json` is a **new-run example**, not a record of the
September 2026 historical sweep. Copy it, choose a new trial ID, and adjust
the parameters before using it. Do not reuse an ID for a different setting.

From the repository root:

```bash
venv/bin/python scripts/materialize_sweep_spec.py \
  --trial results/<branch>/experiments/trials/YOUR_TRIAL.json
venv/bin/python launch_run.py \
  --config 'specs/temp specs/YOUR_TRIAL_ID.json' --preview
venv/bin/python scripts/colab_oauth/launch.py \
  --config 'specs/temp specs/YOUR_TRIAL_ID.json' \
  --trial-record results/<branch>/experiments/trials/YOUR_TRIAL.json --dry-run
```

For an authorized run, use that same `--config` and `--trial-record` with the
default `scripts/colab_oauth/launch.py` session and checkpoint options in
`scripts/README_colab_drive.md`. The launcher checks that the staged spec
matches the record before upload. It saves the **spec bytes from the source
archive** to `results/<branch>/colab_runs/<session>/spec.json`, plus `trial.json`,
`template.json`, and hashes in `spec_provenance.json`. The run's generated
report links these files. The archive includes the selected spec and omits
other specs, including unrelated staged trials. The staged file is ignored by
Git; remove it when finished.

Record the trial ID, changed settings, result, and session link in a human
sweep note under `results/<branch>/experiments/notes/`. The session's generated report and raw
run files provide the evidence. The September 2026 stage snapshots are kept
with their historical run folders under `results/<branch>/colab_runs/<session>/specs/`; their
older VM paths remain in the unchanged journals, recovery records, and source
archives where present.
