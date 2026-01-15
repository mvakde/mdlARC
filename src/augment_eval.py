from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from augment import Augmentor
from inference import DEFAULT_MAX_NEW_TOKENS, run_split_inference_augmented as _run_split_inference_augmented
from tinytransformer import TinyTransformer


@torch.inference_mode()
def run_split_inference_augmented(
    model: TinyTransformer,
    dataset,
    split: str,
    device: torch.device,
    *,
    augmentor: Augmentor,
    batch_size: int = 16,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    task_ids: Optional[Sequence[str]] = None,
    pair_index: Optional[int] = None,
    log_prompts: bool = False,
    include_targets: bool = True,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, List[List[int]]], bool]:
    return _run_split_inference_augmented(
        model=model,
        dataset=dataset,
        split=split,
        device=device,
        augmentor=augmentor,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        task_ids=task_ids,
        pair_index=pair_index,
        log_prompts=log_prompts,
        include_targets=include_targets,
        temperature=temperature,
        top_k=top_k,
    )
