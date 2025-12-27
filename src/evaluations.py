from typing import Callable, Dict, List, Optional, Sequence

import torch

from inference import DEFAULT_MAX_NEW_TOKENS, run_split_inference
from tinytransformer import TinyTransformer
from utils import END_TOKEN_ID, IO_SEPARATOR_TOKEN_ID, NEXT_LINE_TOKEN_ID


def _has_correct_shape(
    sequence: Sequence[int],
    predicted_tokens: Sequence[int],
    target_tokens: Sequence[int],
) -> bool:
    if not target_tokens:
        return False
    if len(predicted_tokens) != len(target_tokens):
        return False

    target_newlines = [
        idx for idx, tok in enumerate(target_tokens) if tok == NEXT_LINE_TOKEN_ID
    ]
    predicted_newlines = [
        idx for idx, tok in enumerate(predicted_tokens) if tok == NEXT_LINE_TOKEN_ID
    ]
    if target_newlines != predicted_newlines:
        return False

    try:
        sep_idx = sequence.index(IO_SEPARATOR_TOKEN_ID)
        end_idx = sequence.index(END_TOKEN_ID, sep_idx + 1)
    except ValueError:
        return False
    return (end_idx - (sep_idx + 1)) == len(target_tokens)


def _pixel_accuracy(
    predicted_tokens: Sequence[int], target_tokens: Sequence[int]
) -> Optional[float]:
    predicted_digits = [tok for tok in predicted_tokens if 0 <= tok <= 9]
    target_digits = [tok for tok in target_tokens if 0 <= tok <= 9]
    if not target_digits or len(predicted_digits) != len(target_digits):
        return None
    correct = sum(1 for p, t in zip(predicted_digits, target_digits) if p == t)
    return correct / len(target_digits)


def summarize_split_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    num_shape_correct = 0
    num_fully_correct = 0
    accuracies: List[float] = []
    fully_correct_results: List[Dict[str, object]] = []

    for res in results:
        predicted_tokens = res.get("output_tokens", [])
        target_tokens = res.get("target_output_tokens", [])
        sequence = res.get("sequence", [])
        shape_ok = _has_correct_shape(sequence, predicted_tokens, target_tokens)
        res["shape_correct"] = shape_ok
        if not shape_ok:
            continue
        num_shape_correct += 1
        acc = _pixel_accuracy(predicted_tokens, target_tokens)
        if acc is not None:
            res["pixel_accuracy"] = acc
            accuracies.append(acc)
        if predicted_tokens == target_tokens:
            num_fully_correct += 1
            fully_correct_results.append(res)

    avg_pixel_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return {
        "total_sequences": len(results),
        "num_shape_correct": num_shape_correct,
        "avg_pixel_accuracy": avg_pixel_accuracy,
        "num_fully_correct": num_fully_correct,
        "fully_correct_results": fully_correct_results,
    }


def evaluate_model_on_dataset(
    model: TinyTransformer,
    dataset,
    device: torch.device,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 16,
    splits: Sequence[str] = ("train", "test"),
    log_prompts: bool = False,
    color_mappings: Optional[Sequence[Sequence[int]]] = None,
    color_apply_fn: Optional[Callable[[str], bool]] = None,
    task_ids: Optional[Sequence[str]] = None,
    include_targets: bool = True,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:
    evaluation: Dict[str, Dict[str, object]] = {}
    # Determine if we should look for targets for this specific split
    # Usually we want targets for 'train' (to debug) but maybe not for 'test' if doing blind submission
    split_include_targets = include_targets

    # If the dataset split itself doesn't have outputs (like test) and we forced include_targets=True,
    # run_split_inference will look for solutions.json. If we forced False, it won't.
    for split in splits:
        split_results = run_split_inference(
            model=model,
            dataset=dataset,
            split=split,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            log_prompts=log_prompts,
            include_targets=split_include_targets,
            color_mappings=color_mappings,
            color_apply_fn=color_apply_fn,
            task_ids=task_ids,
            temperature=temperature,
            top_k=top_k,
        )
        summary = summarize_split_results(split_results)
        evaluation[split] = {"results": split_results, "summary": summary}
    return evaluation
