"""Single-node DDP helpers; importing this module never starts workers."""

from contextlib import contextmanager, redirect_stdout
from datetime import timedelta
import os
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

_checkpoint_group = None


def active():
    return dist.is_available() and dist.is_initialized()


def rank():
    return dist.get_rank() if active() else 0


def world_size():
    return dist.get_world_size() if active() else 1


def unwrap(model):
    while isinstance(model, DistributedDataParallel) or hasattr(model, "_orig_mod"):
        model = model.module if isinstance(model, DistributedDataParallel) else model._orig_mod
    return model


def configure_single_node_nccl(env):
    """Use local bootstrap instead of image-specific external fabric plugins."""
    if env.get("MDLARC_REMOTE") != "1":
        return
    transport = ("NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME")
    if not any(key in env for key in transport):
        env.update(NCCL_NET="Socket", NCCL_NET_PLUGIN="none",
                   NCCL_IB_DISABLE="1", NCCL_SOCKET_IFNAME="lo")
    # These affect bootstrap/network selection; intra-node P2P/NVLink remains enabled.
    env.setdefault("NCCL_DEBUG", "INFO")
    env.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,GRAPH")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")


@contextmanager
def training_process(expected_world_size):
    """Bind CUDA before NCCL initialization and always release the group."""
    if expected_world_size == 1:
        yield
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != expected_world_size:
        raise RuntimeError("DDP training must be started by the launch_run.py coordinator (torchrun).")
    configure_single_node_nccl(os.environ)
    local_rank = int(os.environ["LOCAL_RANK"])
    if not torch.cuda.is_available() or torch.cuda.device_count() < expected_world_size:
        raise RuntimeError(f"DDP requires {expected_world_size} visible CUDA GPUs.")
    global _checkpoint_group
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        # RNG snapshots are CPU Python objects. Keeping their gather and the
        # post-save wait on Gloo avoids CUDA serialization/synchronization in
        # the checkpoint control path; model gradients still use NCCL.
        _checkpoint_group = dist.new_group(backend="gloo", timeout=timedelta(minutes=5))
        # Keep tracebacks on every rank, but only one copy of ordinary logs.
        with open(os.devnull, "w") as quiet:
            with redirect_stdout(quiet if rank() else sys.stdout):
                yield
    finally:
        if _checkpoint_group is not None:
            dist.destroy_process_group(_checkpoint_group)
            _checkpoint_group = None
        dist.destroy_process_group()


def _checkpoint_control_group():
    if _checkpoint_group is None and dist.get_backend() == "nccl":
        raise RuntimeError("NCCL checkpoint coordination requires the CPU control group")
    return _checkpoint_group


def gather_checkpoint_rng_state(state):
    """Collect per-rank CPU RNG snapshots without allocating CUDA object buffers."""
    if not active():
        return None
    states = [None] * world_size() if rank() == 0 else None
    dist.gather_object(state, states, dst=0, group=_checkpoint_control_group())
    return states


def checkpoint_barrier():
    if active():
        dist.barrier(group=_checkpoint_control_group())


class DistributedBatchSampler:
    """Split deterministic global batches without repeating or dropping pairs.

    The underlying sampler owns all family/demo/augmentation visit identities.
    A tiny final batch is rebalanced with its predecessor so every rank can
    participate. Only one epoch's index batches are cached, never data payloads.
    """

    def __init__(self, sampler, *, rank, world_size):
        if not 0 <= rank < world_size:
            raise ValueError("Invalid DDP rank/world_size")
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size
        self._cached_batches = None

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    @property
    def drop_last(self):
        return self.sampler.drop_last

    @drop_last.setter
    def drop_last(self, value):
        self.sampler.drop_last = value
        self._cached_batches = None

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self._cached_batches = None

    def _batches(self):
        if self._cached_batches is None:
            batches = [list(batch) for batch in self.sampler]
            if batches and len(batches[-1]) < self.world_size:
                if len(batches) == 1:
                    raise ValueError("DDP needs at least one selected training pair per rank per epoch.")
                tail = batches.pop()
                combined = batches.pop() + tail
                if len(combined) >= 2 * self.world_size:
                    middle = len(combined) // 2
                    batches.extend([combined[:middle], combined[middle:]])
                else:
                    batches.append(combined)
            self._cached_batches = batches
        return self._cached_batches

    def __len__(self):
        if self._cached_batches is None and hasattr(self.sampler, "__len__"):
            length = len(self.sampler)
            # Rebalancing keeps the batch count when a full batch can supply
            # two nonempty rank groups. Avoid shuffling/materializing hundreds
            # of future family epochs just to plan the optimizer schedule.
            if length != 1 and self.sampler.batch_size >= 2 * self.world_size:
                return length
        return len(self._batches())

    def __iter__(self):
        return (batch[self.rank::self.world_size] for batch in self._batches())


def shard_batch_sampler(sampler):
    if world_size() == 1:
        return sampler
    return DistributedBatchSampler(sampler, rank=rank(), world_size=world_size())


def normalize_losses(outputs):
    """Undo local token means, then normalize over the global microbatch.

    DDP averages gradients, hence the world-size factor. Each MTP head has
    its own valid-token denominator, including ranks with zero valid labels.
    The collectives stay outside the compiled model's forward graph.
    """
    if world_size() == 1:
        return outputs
    units = outputs.get("grid_loss_units")
    counts = (outputs["num_output_tokens"] if units is None else units).reshape(1).float()
    mtp_counts = outputs.get("mtp_token_counts")
    if mtp_counts is not None:
        counts = torch.cat((counts, mtp_counts.float()))
    exit_loss = outputs.get("loop_exit_loss")
    if exit_loss is not None:
        # Deep-supervision exits average over the final head's output tokens.
        counts = torch.cat((counts, outputs["num_output_tokens"].reshape(1).float()))
    totals = counts.detach().clone()
    dist.all_reduce(totals)
    scales = counts * world_size() / totals.clamp(min=1)
    result = dict(outputs)
    result["output_loss"] = outputs["output_loss"] * scales[0]
    if mtp_counts is not None:
        result["mtp_loss"] = (outputs["mtp_losses"] * scales[1:1 + mtp_counts.numel()]).mean()
    if exit_loss is not None:
        result["loop_exit_loss"] = exit_loss * scales[-1]
    return result


def summed(tensor):
    result = tensor.clone()
    if active():
        dist.all_reduce(result)
    return result
