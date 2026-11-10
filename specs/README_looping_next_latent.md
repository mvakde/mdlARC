# Independent looping and NextLat experiments

Both specs derive from `context_palette_schedule_two_lookup.json`. They keep
ARC1 + filtered + ConceptARC, context-palette augmentation, half-or-more demo
sampling, separate target/demo attention, disabled task/dihedral conditioning,
batch 64, 330 epochs, the existing optimizer settings, and balanced leave-one-out
evaluation with 80 augmentations. They are fresh-training recipes.

| Spec | Change |
| --- | --- |
| `context_palette_loop2.json` | Repeat the four middle blocks twice; retain four prediction heads and MTP weight 0.3. |
| `context_palette_next_latent.json` | Retain ordinary depth; replace MTP with one-step NextLat. |

Build and inspect either through the existing launcher:

```bash
venv/bin/python launch_run.py --config specs/context_palette_loop2.json --preview
venv/bin/python launch_run.py --config specs/context_palette_next_latent.json --preview
```

Use the ordinary backend launcher when ready to run. These changes do not launch
or allocate compute. Native GPU memory capacity and throughput have not been
established by CPU tests; the loop increases executed depth and cache storage.

## Depth recurrence

`n_layers` remains the number of physical blocks. Layer bounds are zero-based
and end-exclusive. The looping spec uses `loop_start_layer=2`,
`loop_end_layer=6`, `loop_iterations=2`:

```text
physical blocks: 0 1 | 2 3 4 5 | 6 7
execution:      0 1 | 2 3 4 5 | 2 3 4 5 | 6 7
```

The model executes twelve block applications from the first optimizer update.
Each occurrence of a repeated block has its own KV cache, while attention,
feed-forward, shortcut and conditioning weights are shared. The same schedule
is used for padded and packed training, prompt prefill, incremental generation,
and speculative verification. Existing positional and attention rules remain.

With `loop_input_injection=true`, subsequent iterations begin with
`z + alpha * RMSNorm(e)`, where `e` is the fixed prelude output. `alpha` is a
learned zero-initialized scalar; normalization has no extra learned parameters.
The first iteration receives `e` directly. Losses use the final coda exit only.
There is no temporal feedback, intermediate-exit loss, variable-depth schedule,
early stopping, or KV sharing between iterations.

Set `loop_iterations=1` to retain the original forward computation and parameter
layout. Active loops require explicit valid layer bounds. The configured spec
uses two prelude and two coda blocks; the generic span also supports other
placements. `cache_layer_count` reports logical cache entries, independently of
the physical `config.n_layers`.

## NextLat

The implementation follows the one-step objective and residual dynamics MLP in
the [NextLat paper](https://arxiv.org/html/2511.05963v4#S3.SS2) and
[reference implementation](https://github.com/JaydenTeoh/NextLat/blob/main/models/model_nextlat.py).
For final normalized states `h`, the dynamics predicts
`h_hat[t+1] = h[t] + MLP(LayerNorm(concat(E[x[t+1]], h[t])))`.
It uses three bias-free linear layers, two GELUs, concatenation LayerNorm, and
hidden width `next_latent_hidden_dim` (default `2*d_model`; 1536 in the spec).
Dynamics weights initialize with normal standard deviation 0.02. Initialization
of this optional module preserves the trunk's seeded weights and RNG stream.

```text
loss = ordinary next-token CE
     + next_latent_regression_weight * SmoothL1(h_hat[t+1], stopgrad(h[t+1]))
     + next_latent_kl_weight * KL(teacher next-token distribution || predicted distribution)
```

Both auxiliary weights are 1.0 in the spec. Smooth L1 averages over valid
transition/hidden-coordinate pairs; KL averages over eligible next-token
distributions. Both accumulate in float32 under autocast. Teacher states,
teacher logits, and vocabulary-head weights in the auxiliary KL branch are
detached. Gradients reach the current state, dynamics and next-token embedding;
ordinary CE still trains the vocabulary head normally.

`next_latent_horizon` defaults to 1, preserving the original objective and
checkpoint behavior. Larger horizons recursively reuse the same dynamics MLP:
step two consumes the first predicted state and the second actual input token,
and so on. Every intermediate prediction is supervised against the detached
transformer state. Gradients propagate through the predicted rollout. Each
auxiliary loss is averaged over eligible transitions at each offset, then over
the configured horizon, matching equations (3)-(4) in the paper. Rollouts stop
contributing at END, an ignored label, padding, or a packed-example boundary.
Additional horizon steps add compute and activations, but no parameters.

The paper uses different coefficients by benchmark. Its Countdown experiments
at horizons 1/4/8 use next-token/regression/KL weights 1/2/1; its common default
on several other benchmarks is 1/1/1 (Appendix F, Table 5). Neither setting is
established as optimal for ARC. Individual horizon launch configs belong in
ignored `specs/temp specs/`; retain the baseline's data and evaluation protocol.

Only supervised causal output transitions participate, starting at the target
separator's prediction of the first output token. Prefix latent targets, padding,
ignored source labels, terminal END states and cross-example transitions are
excluded. KL additionally requires a supervised next-token query at the target
state. Conditions use the actual teacher-forced input token, including when a
caller supplies different labels. The expensive dynamics MLP runs only on
output transitions. Native packed training reuses the existing attention-layout
output indices without a second data-dependent selection; invalid/terminal
transitions receive zero weight. Padded/fallback index selection runs eagerly;
model and dynamics tensor work remain compatible with ordinary `torch.compile`.

The spec sets `n_predict=1` and `mtp_weight=0`, so it allocates no MTP heads and
applies no MTP loss. NextLat has its own constant coefficients and does not use
the MTP decay schedule. Logs report ordinary CE, latent regression, latent KL,
the weighted latent objective, and total loss separately. Empty transition sets
produce finite zero auxiliary losses and zero dynamics gradients.

Ordinary autoregressive generation retains the original transformer path.
Teacher-forced evaluation disables both auxiliary objectives explicitly.

## NextLat speculative decoding

Set `"inference_speculative": true` in an evaluation's settings (or in
`eval_defaults`) to draft with a saved NextLat checkpoint. Use greedy inference
(`inference_temperature=0`). The saved model's `next_latent_horizon` controls
the number of proposals: four/eight recursive proposals for d4/d8, plus one
pending token selected by the main transformer. The default remains ordinary
decoding. No checkpoint conversion or retraining is needed.

The dynamics consumes the pending token embedding and the last verified final
normalized hidden state, predicts the next state, and applies the shared main
vocabulary head to propose a token. Further proposals recursively consume the
predicted state and previous proposed token. A single causal transformer pass
verifies the whole block. Only its agreeing prefix is accepted, followed by
the main head's correction (or bonus token). Rejected KV entries and grid
coordinates are rolled back independently for each batch row; EOS, token
budgets, padding, and separate target/demo attention use the same verifier as
MTP speculation. NextLat takes precedence when a model also has MTP heads.

This preserves greedy token selection algorithmically; different BF16 block
shapes can still cause numerical tie changes. Proposal acceptance and runtime
depend on the checkpoint and workload; a longer horizon does not guarantee a
speedup. Startup logs identify NextLat drafting and its proposal count. Models
without NextLat continue using MTP proposals when available, or ordinary decode
when they have one head. Auxiliary training coefficients do not affect this
inference rule.

## Checkpoints and controls

Old checkpoints default to no looping and no NextLat, with no added tensors.
New checkpoint configs preserve each option. Evaluation always uses the saved
architecture; resume requires the saved options and optimizer topology.
An explicit `checkpoint_mode='finetune'` / `reset_training_state=true` allows
adding/removing these options with a fresh optimizer and schedule. Shared model
weights are retained, and new optional parameters are initialized. Corrupted
saved research tensors still fail validation even when dropping an objective.

These two specs deliberately change different factors. Compare decoded grid
scores under the same evaluation policy and report training compute/time,
generation throughput and memory separately. Looping retains stored block
count but adds compute and KV storage; NextLat adds a dynamics module used in
training and optional speculative drafting. Neither spec is a compute-matched
claim against the baseline.

## Learned-loop extensions

Optional fields extend the depth loop. Each starts as the identity, so a model with
any of them initially computes the plain loop; all default off and old checkpoints
load unchanged. They apply in padded and packed training, prompt prefill, cached
decoding and speculative verification, and are position-wise, so cached decoding
needs no state beyond the per-logical-layer KV caches.

| Field | Effect |
| --- | --- |
| `loop_exit_weight` | Deep supervision: after every loop iteration but the last, run the coda and head and add this weight times the mean early-exit cross-entropy (training only). |
| `loop_streams` | `n >= 2` residual streams written once per iteration: loop-specific read/write weights and a diagonal keep gate (`loop_stream_gate`: `static` or `dynamic`). Streams start equal; one-hot readers break the symmetry. |
| `loop_dense_mixing` | Later restarts and the coda add learned per-channel mixes of RMS-normalised earlier iteration latents. |
| `loop_iteration_embedding` | A learned offset added at each restart. |
| `loop_gate`, `loop_gate_init` | `static` or per-`token` gates on every looped block update, `2*sigmoid(...)` initialised to `loop_gate_init`; a gated loop over the whole stack can learn its own span. |
| `loop_ffn` | `first`: iterations after the first run attention only. |
| `loop_iteration_norms` | Per-iteration multiplicative gains on the shared blocks' norms. |
| `adamw_loop_lr_multiplier` | Separate AdamW group for the `loop_*` parameters with this LR multiplier. At 1.0 the historical optimizer topology is kept. |

Dynamic gate terms are scaled by `1/sqrt(d_model)` so per-coordinate AdamW steps
do not saturate them. Deeper loops raise activation and KV-cache memory: very
deep configurations may need gradient accumulation and a smaller evaluation
batch. Evaluation keeps one KV cache per logical layer, so prefill memory also
grows with loop depth; set the `MDLARC_PREFILL_TOKEN_BUDGET` environment variable
(default 524288 tokens per prefill chunk) lower when prefill runs out of memory.
It changes chunking only, not the decoded outputs. Run records and measured outcomes are kept outside Git under
`results/se-exp/experiments/notes/`.
