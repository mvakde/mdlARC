# DDP scaling ladder

On `ddp-scaling-ladder`, the ordinary experiment launcher supports single-node
[PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).
The coordinator prepares the data once, uses
`torchrun` for each training stage, waits for all workers, and evaluates in a
fresh single process. The existing Vast/Runpod supervisor and Drive checkpoint
backup workflow still own the run. Worker failures fail the stage; torchrun
does not automatically retry or resubmit training.

[arc2_family_development_two_demo_ddp.json](arc2_family_development_two_demo_ddp.json)
derives from the two-demo development spec. It keeps the pinned counts-v1
package, frozen development split, 330 epochs, 24 layers, width 1536, FFN 4096,
24 heads, FA4, context-palette augmentation, MTP, and two-demo evaluation policy.
Its default is two GPUs, batch 11 **per GPU**, accumulation 3: global microbatch
22 and effective batch 66, matching the source spec. Learning rates are unchanged.
The dataset ZIP remains an ignored input under `assets_tmp/`.

```bash
venv/bin/python launch_run.py --config specs/arc2_family_development_two_demo_ddp.json --preview
venv/bin/python launch_run.py --config specs/arc2_family_development_two_demo_ddp.json
```

The second command requires two visible CUDA GPUs on the same machine. Each
worker holds a complete model and optimizer; DDP does not combine GPU memory.
CUDA/NCCL/FA4 capacity and throughput need validation on the intended hardware.

To enable DDP on another reusable recipe, set top-level
`"distributed": {"nproc_per_node": 2}`. Training `batch_size` is per GPU and
`gradient_accumulation_steps` is per optimizer update. Effective batch is their
product times `nproc_per_node`. An omitted distributed block keeps the existing
single-GPU path. DDP currently supports training stages, including fine-tuning;
evaluation and benchmark modes run separately. Multi-node launch is unsupported.

For remote launches, explicitly request the same GPU count in the infrastructure
policy. GPU templates default to one device, so override the count after choosing
a template. For example, using an already configured Runpod policy:

```bash
venv/bin/python workflows/runpod/run.py launch \
  --config specs/arc2_family_development_two_demo_ddp.json \
  --policy-override num_gpus=2 --dry-run
```

Review the dry run's GPU, image, price, runtime and backup policy before launching
without `--dry-run`. The Vast launcher accepts the same GPU-count override.
Launch fails before rental if the requested count is smaller than the DDP count.
An existing single-GPU rental cannot run this two-GPU recipe.

Remote single-node DDP uses internal NCCL socket bootstrap over loopback and
keeps GPU P2P/NVLink enabled, avoiding image-specific external fabric plugins.
Explicit NCCL transport environment settings take precedence. Initialization
diagnostics are logged unless an explicit NCCL logging setting overrides them.

The sampler chooses the same global family instances and visit identities on
every rank, then splits each global microbatch across workers. Family sampling
keeps every selected solved pair exactly once, including epoch tails. A tail
smaller than the GPU count is rebalanced with the preceding microbatch to keep
all ranks participating. At per-GPU batch 1 this can require one additional
example on a rank. Epochs with fewer pairs than ranks fail explicitly. Existing
regular-data drop-last behavior remains in place. Geometry, demo selection and
context-palette schedules are prepared from the original sampler's visits.
DDP data-loader workers use spawn so they do not fork an initialized NCCL runtime.

Gradients synchronize at accumulation boundaries. CE and each MTP head use
global valid-token counts, so unequal output lengths across GPUs do not alter
the global-microbatch objective. Loss logs average across ranks and teacher-forced
metric counts are summed globally. Only rank zero writes logs/checkpoints;
checkpoints contain ordinary model keys, optimizer/scheduler state and each
rank's RNG state. Evaluation loads these through the ordinary single-GPU path.
Checkpoint RNG-object gathering and the post-save barrier use a dedicated
CPU/Gloo group; gradient and training-metric collectives continue to use NCCL.
This keeps checkpoint metadata coordination out of CUDA object buffers.
Resuming a DDP checkpoint requires its saved GPU count; fresh fine-tuning can
change it. Different GPU counts, dropout streams and floating-point reduction order can
produce different training trajectories even at the same effective batch.

Local verification includes a two-process CPU/Gloo comparison against unsplit
microbatches with two-demo contexts, unequal token lengths, MTP, NorMuon/AdamW,
gradient clipping, accumulation, and a short final accumulation window. This
checks numerical correctness and portable checkpoints, not native CUDA/FA4
performance or memory capacity.
