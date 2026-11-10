"""Frozen-checkpoint backend/precision comparisons through the normal launcher."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("current_bf16", "sdpa_bf16", "current_amp", "sdpa_amp",
            "current_bf16_fp32_rope")
DEFAULTS = {
    "variants": list(VARIANTS[:4]), "task_count": 40,
    "augmentation_attempts": 8, "batch_size": 32, "max_new_tokens": 1024,
    "parity_cases": 8, "parity_tokens": 128, "trial_timeout_seconds": 900,
}


def validate_config(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULTS):
        raise ValueError("Invalid backend_benchmark object or unknown keys.")
    config = {**DEFAULTS, **value}
    variants = config["variants"]
    if (not isinstance(variants, list) or not variants
            or any(not isinstance(v, str) or v not in VARIANTS for v in variants)
            or len(set(variants)) != len(variants)):
        raise ValueError("backend_benchmark.variants must be unique supported names.")
    for key in set(DEFAULTS) - {"variants"}:
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"backend_benchmark.{key} must be a positive integer.")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def compare_logits(reference, actual):
    import numpy as np

    reference, actual = np.asarray(reference, dtype=np.float32), np.asarray(actual, dtype=np.float32)
    if reference.shape != actual.shape or reference.ndim != 2 or not reference.size:
        raise ValueError("Logit comparisons require equal nonempty [token, vocab] arrays.")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Nonfinite logits in backend comparison.")
    delta = np.abs(reference - actual)
    ref_top = reference.argmax(-1)
    changed = ref_top != actual.argmax(-1)
    top_two = np.sort(reference, axis=-1)[:, -2:]
    margins = top_two[:, 1] - top_two[:, 0]
    return {"tokens": len(reference), "argmax_disagreements": int(changed.sum()),
            "max_abs_logit_delta": float(delta.max()), "mean_abs_logit_delta": float(delta.mean()),
            "minimum_reference_margin": float(margins.min()),
            "changed_reference_margins": margins[changed].tolist()}


def compare_runs(reference, actual):
    if reference["selection_sha256"] != actual["selection_sha256"]:
        raise ValueError("Variants evaluated different examples or augmentation schedules.")
    def keyed(rows):
        return {(r["task_id"], r["pair_index"], r["dihedral_index"],
                 tuple(r["color_mapping"])): r for r in rows}
    left, right = keyed(reference["predictions"]), keyed(actual["predictions"])
    if left.keys() != right.keys() or len(left) != len(reference["predictions"]):
        raise ValueError("Prediction identity mismatch or duplicate attempts.")
    changed = [key for key in left if left[key]["output_tokens"] != right[key]["output_tokens"]]
    ref_cases = {(r["task_id"], r["pair_index"]): r for r in reference["cases"]}
    new_cases = {(r["task_id"], r["pair_index"]): r for r in actual["cases"]}
    if ref_cases.keys() != new_cases.keys():
        raise ValueError("Scored case mismatch.")
    gained = [list(k) for k in ref_cases if not ref_cases[k]["correct"] and new_cases[k]["correct"]]
    lost = [list(k) for k in ref_cases if ref_cases[k]["correct"] and not new_cases[k]["correct"]]
    ref_parity = {r["case"]: r for r in reference["parity"]}
    new_parity = {r["case"]: r for r in actual["parity"]}
    if ref_parity.keys() != new_parity.keys():
        raise ValueError("Parity case mismatch.")
    return {"attempts": len(left), "changed_generated_sequences": len(changed),
            "gained_correct_cases": gained, "lost_correct_cases": lost,
            "arc_score_delta": actual["score"]["arc_score"] - reference["score"]["arc_score"],
            "parity": {k: {route: compare_logits(ref_parity[k][route], new_parity[k][route])
                            for route in ("packed_flash_logits", "cached_logits")}
                       for k in ref_parity}}


def run_benchmark(plan):
    config = plan["backend_benchmark"]
    directory = plan["run_dir"] / "backend_benchmark"
    directory.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "checkpoint": str(plan["pretrain"]["checkpoint_path"]),
               "variants": {}, "comparisons": {}, "failures": {}}
    for variant in config["variants"]:
        print(f"Backend comparison: {variant}", flush=True)
        output = directory / f"{variant}.json"
        try:
            subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                            "--config", str(plan["spec_path"]), "--variant", variant,
                            "--output", str(output)], cwd=ROOT, check=True,
                           timeout=config["trial_timeout_seconds"])
            results["variants"][variant] = json.loads(output.read_text())
        except (subprocess.SubprocessError, OSError, ValueError) as exc:
            results["failures"][variant] = str(exc)
        # Save partial evidence even if a later variant fails.
        write_json(directory / "results.json", results)
    variants = results["variants"]
    for ref, actual in (("current_bf16", "sdpa_bf16"), ("current_bf16", "current_amp"),
                        ("current_amp", "sdpa_amp"), ("current_bf16", "sdpa_amp"),
                        ("current_bf16", "current_bf16_fp32_rope")):
        if ref in variants and actual in variants:
            results["comparisons"][f"{ref} -> {actual}"] = compare_runs(variants[ref], variants[actual])
    write_json(directory / "results.json", results)
    print(json.dumps({v: r["score"] for v, r in variants.items()}, indent=2), flush=True)
    if results["failures"]:
        raise RuntimeError("Backend comparison incomplete; inspect backend_benchmark/results.json.")


def cached_target_logits(model, attempt, device, max_new_tokens, token_limit, *, compile_decode=True):
    """Use the real prefill, padding, cache, and compiled decode on fixed target prefixes."""
    import torch
    import evaluate

    item = attempt.inference
    prefix_len = len(item.prompt)
    pad_len = evaluate._prompt_bucket_length(prefix_len, max_seq_len=model.config.max_seq_len,
                                             max_new_tokens=max_new_tokens)
    input_ids, mask = evaluate._left_pad_sequences([item.prompt], evaluate.END_PAIR_TOKEN_ID,
                                                  device, padded_length=pad_len)
    positions = evaluate._pad_cached_positions([item.positions], pad_len, device)
    ids = torch.tensor([item.task_embedding_ids], dtype=torch.int32, device=device)
    dihedrals = torch.tensor([item.dihedral_index], dtype=torch.int32, device=device)
    conditioning = model.compute_conditioning_embeds(ids, dihedrals).to(model.example_embedding.weight.dtype)
    values = model.compute_value_embeds(ids)
    shared = dict(example_ids=ids, dihedral_ids=dihedrals,
                  conditioning_embeds=conditioning, value_embeds=values)
    output = model.forward_generate(input_ids=input_ids, positions_3d=positions,
                                    attention_mask=mask, **shared)
    capacity = min(((pad_len + max_new_tokens + 127) // 128) * 128, model.config.max_seq_len)
    cache = tuple(tuple(torch.cat([v, v.new_zeros((*v.shape[:2], capacity - pad_len, v.shape[-1]))], dim=2)
                        for v in pair) for pair in output["past_key_values"])
    full_mask = torch.zeros((1, capacity), dtype=torch.bool, device=device)
    full_mask[:, :pad_len] = mask
    count = min(token_limit, len(attempt.tokens) - prefix_len, capacity - pad_len)
    logits = [output["logits"][0, -1].float().cpu()]
    if compile_decode and not hasattr(model, "_compiled_decode"):
        model._compiled_decode = torch.compile(model.forward_generate, mode="default", fullgraph=True)
    decode = model._compiled_decode if compile_decode else model.forward_generate
    position = torch.tensor([pad_len], device=device, dtype=torch.long)
    for offset in range(count - 1):
        torch.compiler.cudagraph_mark_step_begin()
        full_mask[:, position] = True
        block_mask = (model.build_decode_block_mask_for_step(full_mask, position, capacity, device)
                      if device.type == "cuda" and model.blocks[0].attention._has_flex_attention else None)
        index = prefix_len + offset
        output = decode(input_ids=attempt.tokens[None, index:index + 1].to(device),
                        positions_3d=attempt.positions[None, index:index + 1].to(device),
                        past_key_values=cache, attention_mask=full_mask, cache_position=position,
                        decode_block_mask=block_mask, **shared)
        logits.append(output["logits"][0, -1].float().cpu())
        position.add_(1)
    return torch.stack(logits)


def worker(plan, variant, output_path):
    import torch
    import evaluate
    from build import build_model_and_data
    from common import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Backend benchmark requires a CUDA GPU.")
    config = plan["backend_benchmark"]
    eval_cfg = next(iter(plan["evals"].values()))
    cfg = argparse.Namespace(**{**plan["shared"], **eval_cfg,
                               "checkpoint_path": plan["pretrain"]["checkpoint_path"]})
    started = time.perf_counter()
    model, dataset, _, device, _ = build_model_and_data(cfg, is_eval=True,
                                                       eval_splits=("test",), build_dataloader=False)
    model.eval()
    model._loaded_checkpoint = None
    if not model._has_varlen_flash_attention():
        raise RuntimeError("Native FlashAttention is required for the training-path comparison.")
    if not all(block.attention._has_flex_attention for block in model.blocks):
        raise RuntimeError("FlexAttention is required to validate the current CUDA decode path.")
    amp = variant.endswith("_amp")
    if not amp:
        ropes = [(block.attention.rope,
                  {name: value.clone() for name, value in block.attention.rope.named_buffers()})
                 for block in model.blocks] if variant.endswith("fp32_rope") else []
        model.to(dtype=torch.bfloat16)
        for rope, buffers in ropes:
            for name, value in buffers.items():
                setattr(rope, name, value)
    if variant.startswith("sdpa"):
        for block in model.blocks:
            block.attention._has_flex_attention = False
    all_tasks = sorted({example.task_id for example in dataset.iter_examples(split="test")})
    random.Random(cfg.seed).shuffle(all_tasks)
    selected = sorted(all_tasks[:config["task_count"]])
    solutions = json.loads(Path(cfg.solutions_path).read_text())
    work = evaluate._build_inference_attempts(dataset, "test", selected, None,
                                            config["augmentation_attempts"], True)
    schedule = [{"task": w.task_id, "pair": w.pair_index, "prompt": w.prompt,
                 "ids": w.task_embedding_ids, "dihedral": w.dihedral_index,
                 "mapping": w.color_mapping} for w in work]
    identity = hashlib.sha256(json.dumps(schedule, sort_keys=True).encode()).hexdigest()
    parity_work, seen = [], set()
    for item in work:
        key = (item.task_id, item.pair_index)
        if key not in seen:
            seen.add(key)
            parity_work.append(item)
        if len(parity_work) >= config["parity_cases"]:
            break
    attempts = evaluate._build_accuracy_attempts(dataset, "test", parity_work, solutions,
                                                 model.config.max_seq_len)
    parity = []
    precision = lambda: torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    with torch.inference_mode(), precision():
        for attempt in attempts:
            item = attempt.inference
            length, sep = len(attempt.tokens), len(item.prompt) - 1
            kwargs = dict(example_ids=torch.tensor([item.task_embedding_ids], device=device),
                          dihedral_ids=torch.tensor([item.dihedral_index], device=device),
                          sep_indices=torch.tensor([sep], device=device), compute_mtp_loss=False,
                          compute_next_latent_loss=False)
            packed = model(input_ids=attempt.tokens.to(device), positions_3d=attempt.positions.to(device),
                           cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device=device),
                           max_seqlen=length, **kwargs)["logits"][sep:-1].float().cpu()
            padded = model(input_ids=attempt.tokens[None].to(device),
                           positions_3d=attempt.positions[None].to(device), **kwargs)["logits"][0, sep:-1].float().cpu()
            cached = cached_target_logits(model, attempt, device, config["max_new_tokens"], config["parity_tokens"])
            count = len(cached)
            packed, padded = packed[:count], padded[:count]
            parity.append({"case": f"{item.task_id}:{item.pair_index}",
                           "target_tokens": length - sep - 1, "tested_tokens": count,
                           "packed_vs_padded": compare_logits(packed, padded),
                           "packed_vs_cached": compare_logits(packed, cached),
                           "packed_flash_logits": packed.tolist(), "padded_sdpa_logits": padded.tolist(),
                           "cached_logits": cached.tolist()})
            print(f"{variant} parity {parity[-1]['case']}: {parity[-1]['packed_vs_cached']}", flush=True)
        # This process owns the model's precision policy; don't recast AMP weights.
        original_prepare = evaluate._prepare_model_for_inference
        evaluate._prepare_model_for_inference = lambda model: model
        try:
            predictions = evaluate.run_split_inference(
                model, dataset, "test", device, batch_size=config["batch_size"],
                max_new_tokens=config["max_new_tokens"], task_ids=selected,
                augmentation_attempts=config["augmentation_attempts"])
        finally:
            evaluate._prepare_model_for_inference = original_prepare
    set_seed(cfg.seed)
    submission = evaluate._build_submission_from_aaivr(evaluate.run_aaivr_on_results(predictions))
    cases, task_scores = [], []
    for task_id in selected:
        truths = solutions[task_id]
        correct_count = 0
        for index, truth in enumerate(truths):
            entry = submission.get(task_id, [])
            entry = entry[index] if index < len(entry) else {}
            correct = any(entry.get(key) == truth for key in ("attempt_1", "attempt_2"))
            correct_count += correct
            cases.append({"task_id": task_id, "pair_index": index, "correct": correct,
                          "attempt_1": entry.get("attempt_1"), "attempt_2": entry.get("attempt_2")})
        task_scores.append(correct_count / len(truths))
    score = {"tasks": len(selected), "cases": len(cases),
             "correct_cases": sum(row["correct"] for row in cases),
             "arc_score": sum(task_scores), "arc_score_percent": 100 * sum(task_scores) / len(selected),
             "generated_attempts": len(predictions), "seconds": time.perf_counter() - started}
    write_json(output_path, {"variant": variant, "score": score, "selected_task_ids": selected,
                            "selection_sha256": identity, "parity": parity, "cases": cases,
                            "predictions": predictions, "model_config": vars(model.config),
                            "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                            "gpu": torch.cuda.get_device_name(),
                                            "parameters_dtype": str(next(model.parameters()).dtype),
                                            "rope_dtype": str(model.blocks[0].attention.rope.cos_x_cache.dtype),
                                            "decode_backend": "sdpa" if variant.startswith("sdpa") else "flex"}})
    print(f"{variant}: {json.dumps(score)}", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    import launch_run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    worker(launch_run.build_plan(launch_run.load_spec(args.config), args.config), args.variant, args.output)
