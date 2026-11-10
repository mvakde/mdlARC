# Adapt a pinned submission checkpoint on a validation set

The `checkpoint_adaptation` experiment mode runs through the ordinary
`scripts/colab_oauth/launch.py` launcher, including original-kernel recovery,
direct Drive checkpoint verification, asynchronous checkpoint backups and
verified result recovery. It uses the exact Python model runtime embedded in
a hash-pinned submission notebook. It does not refresh the notebook runtime
from the current checkout.

Author one-off specs under ignored `specs/temp specs/`. Set a top-level
`checkpoint` identity, `enable_finetune=false`, `skip_data_build=true`, and
`skip_evals=true`; this mode owns its adaptation and decoding stages. Pin the
notebook and separate challenge/solution files in `checkpoint_adaptation.inputs`.
Each identity has repository-relative `path`, byte `size`, and full `sha256`.
The source snapshot includes and verifies all three inputs. Challenges must
have input-only query pairs. Only the parent scoring process opens solutions,
after successful decoding.

The mode accepts `method="last_blocks"` with `last`, or `method="lora"` with
`rank`, `alpha`, `lora_lr`, and `lora_weight_decay`. LoRA targets every block's
packed attention QKV projection, attention output projection, and both SwiGLU
MLP projections. Original parameters remain frozen. Last-block adaptation
preserves parameters disabled by the checkpoint architecture. Embeddings,
heads and norms outside selected blocks stay frozen.

Training controls are `epochs`, per-GPU `batch_size`, `accumulation`,
`train_augments`, `normuon_lr`, `adamw_lr`, `checkpoint_epochs`, and `seed`.
Decoding uses `eval_augments` and `eval_batch_size`. `enable_compile` and
`attention_backend` select the execution path. Defaults and architecture come
from the pinned notebook and checkpoint; adaptation always starts fresh.
LoRA uses AdamW with a 1% relative LR floor; last-block tuning uses the
notebook's NorMuon/auxiliary AdamW optimizer and schedule.

Training and inference run in separate processes, releasing optimizer and
compiler allocations between phases. LoRA saves ordinary merged model
checkpoints for decoding plus separate `.adapters.pt` files containing the
unmerged adapter parameters, optimizer, scheduler, RNG state and source
checkpoint identity. These small files preserve the information needed for
recovery; automatic LoRA training resume is not implemented. Merged checkpoints
omit incompatible adapter optimizer state. Checkpoints publish atomically to
the normal backup queue.

For an already adapted checkpoint, set `decode_only=true` and `input_epoch`
to its saved adaptation epoch. The input still passes the ordinary launcher's
Drive size/hash/ancestry checks. No training stage executes. An optional pinned
`inputs.training_stats` preserves the original training measurements in the new
validation summary. The submission notebook still supplies the architecture
and exact model implementation.

`inference_speculative=true` enables the checkpoint's native greedy
self-speculative path. Set both `eval_batch_size` and `prefill_token_budget`:
raising only the batch cap can leave long prompts limited by the token budget.
Check the longest contexts and leave room for full-length outputs; a short
generation probe can use less KV-cache capacity than the full decode.

`decode_probe=true` checks the longest available contexts without opening query
solutions. It compares four serial reference prefixes against the larger
batch, compares ordinary/speculative outputs, and records warmed timings and
peak memory. Controls are `probe_cases`, `probe_tokens`, and
`probe_max_seconds`. Compilation is reported separately in warmup rows.
`wait_for_decode_settings=true` pauses after the probe so a comparison can use
common settings on both VMs. Supply a `decode_settings.json` file in each run
directory with exactly `eval_batch_size` (positive integer) and
`inference_speculative` (boolean). The accepted settings are saved in the
execution request and final summary. This configuration handoff never starts
training or creates another VM/kernel.

Outputs under the managed run directory include `training_progress.json`,
`training_stats.json`, `predictions.json`, `generated_candidates.json.gz`,
`decoding_stats.json`, `phase_times.json`, `validation_scores.json`, and
`validation_run_info.json`. Scores report both task-weighted top-1/top-2 and
query-weighted counts, plus a per-task breakdown. A worker failure stops the
experiment without automatic resubmission.

Keep the reusable source, tests and this guide in Git. Keep execution specs,
input data copies, source snapshots, checkpoints, logs and comparison records
under ignored temporary-spec/results paths. Sync notes and draft specs before
and after preparing a run.
