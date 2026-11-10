"""Pinned last-four-block comparison cells for validation and competition use."""

import json


def last_four_cells(cells, selected, defaults, cell):
    validation = selected.get("validation")
    mode = "validation" if validation else "competition"
    full_transition = (
        "After `SMOKE TEST PASSED`, **restart the session**, set **SMOKE_TEST=False**, then Run All."
        if validation else
        "For competition submission, set **SMOKE_TEST=False** and use the notebook editor's **competition panel → Submit**."
    )
    cells[0] = cell("markdown", f"""
        # mdlARC 600M: last four blocks — {mode}, four L4 GPUs

        Start from the verified original **pretraining epoch 330** checkpoint.
        Adapt afresh for **{selected['epochs']} epochs**, training only zero-based transformer blocks
        **20, 21, 22, 23**. Token embeddings, heads, final norm, earlier blocks and any
        architecture-disabled parameters stay frozen. DDP uses **four GPUs**,
        microbatch **2 per GPU**, accumulation **2 per GPU**, effective batch **16**.
        NorMuon LR **0.0003**, auxiliary AdamW LR **0.00028**; **80** training and
        **80** inference augmentations in full mode.

        Select **L4 × 4**, **Internet off**. Each GPU holds its own model replica.
        Four independent task-sharded inference workers start after training exits.
        Decode starts at batch **1 per GPU**, prefill budget **4096 tokens**, native
        **four-head speculative decoding**. Settings and per-GPU peak VRAM are printed.

        Default **SMOKE_TEST=True** performs exactly **one real training epoch** over
        the complete selected demonstration set, with the same 2/2 training batch and
        80 training augmentations, then decodes four tasks selected by large input
        context with two inference augmentations. Success requires optimizer updates
        on every rank, checkpoint save/reload and successful decoding/merge without
        malformed-output replacements. Smoke outputs must not be submitted.

        {full_transition}
        Full adaptation always reloads the original pretrained weights;
        no smoke checkpoint is carried into it. See `kaggle/README_last4.md`.
    """)
    if validation:
        intro = f"""
        Attach [private timing240](https://www.kaggle.com/datasets/{validation['dataset_ref']}),
        [epoch330 model](https://www.kaggle.com/datasets/mithilavakde/mdlarc-600m-epoch330),
        and [FA2 wheel](https://www.kaggle.com/datasets/mithilavakde/flash-attn-cp313-torch211-cu128).
        Competition data is unnecessary. Full mode uses all **240 tasks**, **726 demos**,
        **336 queries** and **26,880 augmented query decodes**. Solutions are opened only
        by the parent after generation/merge. Scores keep **original40** and
        **synthetic200** separate. Direct composition parents are original40; historical
        generator authoring references are disclosed in the manifest. This is a timing
        workload, not 240 independent original ARC tasks. Validation creates
        `validation_predictions.json`, `validation_scores.json`, `validation_run_info.json`.
        Use **submit_arc2_600m_last4.ipynb** for competition submission.
        """
    else:
        intro = """
        Attach [competition data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data),
        [epoch330 model](https://www.kaggle.com/datasets/mithilavakde/mdlarc-600m-epoch330),
        and [FA2 wheel](https://www.kaggle.com/datasets/mithilavakde/flash-attn-cp313-torch211-cu128).
        Adaptation uses only the official runtime tasks' demonstration pairs, then
        decodes their official query inputs. No timing240 data or solutions are used.
        Full mode produces **submission.json**. With SMOKE_TEST=False, open the
        notebook editor's **competition panel → Submit** and specify **submission.json**.
        This saves and submits the code version without waiting for a separate full
        run to finish. Kaggle executes the submitted notebook on hidden runtime data
        and scores the generated file. See the [Kaggle staff workflow explanation](https://www.kaggle.com/discussions/product-feedback/269953).
        Check the competition's current
        [code requirements](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements).
        """
    cells[0]["source"] += cell("markdown", intro)["source"]
    # Reuse environment/bootstrap cells; these contain no model source changes.
    settings = "".join(cells[1]["source"])
    settings = settings.replace("# True: four tasks/two epochs; False: full selected dataset.",
                                "# Exactly one epoch on all demos; then four-task decode.")
    if validation:
        settings = settings.replace('"FULL SUBMISSION"', '"FULL VALIDATION"')
    settings += f"\nPREFILL_TOKEN_BUDGET = {selected['prefill_token_budget']}\nINFERENCE_SPECULATIVE = True\n"
    settings += f"PINNED_SOURCE_NOTEBOOK_SHA256 = {selected['runtime_source_sha256']!r}\n"
    cells[1] = cell("code", settings)
    cells[2] = cell("markdown", """
        ## Offline CUDA / FlashAttention environment

        Verify four L4 GPUs and the printed Python/PyTorch/CUDA/ABI against the
        attached wheel. Keep FlashAttention 2 and torch.compile enabled for this
        comparison. A changed Kaggle image may require a new compatible wheel.
        Native attention is an explicit alternative and changes the runtime comparison.
    """)
    cells[4] = cell("markdown", """
        ## Embedded source

        This payload contains the frozen mdlARC runtime and Kaggle submission
        helpers. Its checksum is verified before extraction.

        This checkpoint has **task_embeddings=False**. Task conditioning stays
        disabled during adaptation and decoding. The runtime retains inactive task
        tables in its checkpoint format; they are neither used nor trained.
        Token embeddings remain frozen. Only architecture-enabled parameters in
        transformer blocks **20–23** are trained.
    """)
    cells[6] = cell("markdown", """
        ## Verify inputs, then isolate demonstration training and query decoding

        Checkpoint size and full SHA-256 are verified before trusted loading, then
        saved architecture/epoch/augmentation metadata are checked. Training always
        uses all selected tasks' demos. Smoke decoding selects four tasks by demo plus
        largest query input context; outputs/solutions cannot affect that selection.
    """)
    discovery = ('ku.find_validation_challenges(Path("/kaggle/input"), TEST_CHALLENGES, VALIDATION_IDENTITY)'
                 if validation else 'ku.find_test_challenges(Path("/kaggle/input"), TEST_CHALLENGES)')
    cells[7] = cell("code", f'''
        import gc
        import json
        from time import perf_counter
        import multi_gpu as mg

        CHECKPOINT_PATH = ku.find_checkpoint(Path("/kaggle/input"), CHECKPOINT_PATH)
        TEST_CHALLENGES = {discovery}
        print("Verifying checkpoint size and full SHA-256 before load...", flush=True)
        checkpoint_hash = ku.verify_checkpoint_file(CHECKPOINT_PATH, CHECKPOINT_IDENTITY)
        original = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False, mmap=True)
        ku.verify_checkpoint_metadata(original, CHECKPOINT_IDENTITY)
        print("Original pretrained epoch:", original["epoch"], "SHA-256:", checkpoint_hash)
        print("Architecture:", original["config"])
        del original
        gc.collect()
        data_path = WORKING / "runtime_training_challenges.json"
        training_challenges = ku.prepare_challenges(TEST_CHALLENGES, data_path)
        decode_data_path = WORKING / "runtime_decode_challenges.json"
        challenges = ku.prepare_challenges(TEST_CHALLENGES, decode_data_path,
            smoke_test=SMOKE_TEST, smoke_task_count=NUM_GPUS, smoke_strategy="largest")
        print("Training tasks:", len(training_challenges), "demonstration pairs:",
              sum(len(t["train"]) for t in training_challenges.values()))
        print("Decode tasks:", list(challenges) if SMOKE_TEST else len(challenges),
              "queries:", sum(len(t["test"]) for t in challenges.values()))
        print("Decode batch per GPU:", EVAL_BATCH_SIZE, "prefill token budget:",
              PREFILL_TOKEN_BUDGET, "four-head speculative:", INFERENCE_SPECULATIVE)
        print("Frozen source notebook SHA-256:", PINNED_SOURCE_NOTEBOOK_SHA256,
              "packaged runtime SHA-256:", RUNTIME_SHA256)
    ''')
    if validation:
        cells[7]["source"] += cell("code", '''
            if (len(training_challenges) != VALIDATION_IDENTITY["tasks"] or
                sum(len(t["train"]) for t in training_challenges.values()) != VALIDATION_IDENTITY["demonstration_pairs"] or
                sum(len(t["test"]) for t in training_challenges.values()) != VALIDATION_IDENTITY["query_pairs"]):
                raise ValueError("Pinned validation dataset count mismatch.")
            manifest_path = TEST_CHALLENGES.with_name(VALIDATION_IDENTITY["manifest"]["filename"])
            ku.verify_dataset_file(manifest_path, VALIDATION_IDENTITY["manifest"])
            validation_groups = json.loads(manifest_path.read_text())["groups"]
        ''')["source"]
    cells[8] = cell("markdown", f"""
        ## Synchronized DDP adaptation — only blocks 20–23

        Exactly one smoke epoch / {selected['epochs']} full epochs, both with microbatch 2 ×
        accumulation 2 × four GPUs = effective batch 16. Both use 80 training
        augmentations and all available demos. A fresh optimizer/schedule is created
        from original pretrained weights every time. Architecture-disabled parameters
        remain disabled; optimizer groups contain exactly the trainable parameters.

        `MDLARC_STARTUP_EVENT first_optimizer_step` confirms a completed update.
        Per-rank reports count actual optimizer steps and completed epochs. Liveness
        messages only report process health. The eight-hour training ETA guard reserves
        time for decoding; a rejected estimate stops the run without changing epochs.
        Workers exit completely before inference releases/allocates its CUDA memory.
    """)
    run_name = selected.get("run_name", "last4_val240" if validation else "last4_competition")
    cells[9] = cell("code", f'''
        SAVED_DEFAULTS = json.loads({json.dumps(json.dumps(defaults))})
        run_dir = WORKING / "runs" / ("smoke_{run_name}" if SMOKE_TEST else "{run_name}")
        run_dir.mkdir(parents=True, exist_ok=True)
        # Remove prior stage artifacts so failed reruns cannot accept stale results.
        for stale in ["finetuned.pt", "inference.pt", "train-*.json", "decode-*.json", "shard-*.json"]:
            for path in run_dir.glob(stale):
                path.unlink()
        save_path = run_dir / "finetuned.pt"
        inference_path = run_dir / "inference.pt"
        request_path = run_dir / "request.json"
        request = {{
            "working": str(WORKING), "run_dir": str(run_dir), "num_gpus": NUM_GPUS,
            "checkpoint_path": str(CHECKPOINT_PATH), "checkpoint_identity": CHECKPOINT_IDENTITY,
            "data_path": str(data_path), "decode_data_path": str(decode_data_path),
            "save_path": str(save_path), "inference_checkpoint": str(inference_path),
            "defaults": SAVED_DEFAULTS, "attention_backend": ATTENTION_BACKEND,
            "enable_compile": ENABLE_COMPILE, "train_log_mode": TRAIN_LOG_MODE,
            "progress_interval_seconds": PROGRESS_INTERVAL_SECONDS,
            "smoke_test": SMOKE_TEST, "max_finetune_hours": MAX_FINETUNE_HOURS,
            "trainable_blocks": {{"first": TRAIN_FIRST_LAYERS, "last": TRAIN_LAST_LAYERS}},
            "eval_augments": 2 if SMOKE_TEST else EVAL_AUGMENTS,
            "eval_batch_size": EVAL_BATCH_SIZE, "prefill_token_budget": PREFILL_TOKEN_BUDGET,
            "inference_speculative": INFERENCE_SPECULATIVE,
            "finetune": {{
                "epochs": 1 if SMOKE_TEST else EPOCHS,
                "batch_size": TRAIN_BATCH_SIZE, "accumulation": GRADIENT_ACCUMULATION_STEPS,
                "max_augments": TRAIN_AUGMENTS, "num_workers": NUM_WORKERS,
                "normuon_lr": NORMUON_LR, "adamw_lr": ADAMW_LR,
            }},
        }}
        mg.write_json(request_path, request)
        print("ADAPTATION REQUEST:", json.dumps(request, indent=2), flush=True)
        started = perf_counter()
        mg.launch_training(request_path)
        adaptation_hours = (perf_counter() - started) / 3600
        training_stats = [json.loads((run_dir / f"train-{{rank}}.json").read_text())
                          for rank in range(NUM_GPUS)]
        if not all(s["optimizer_updates"] > 0 and s["epochs"] == request["finetune"]["epochs"]
                   and len(s["epoch_seconds"]) == request["finetune"]["epochs"]
                   and s["trainable_blocks"] == [20, 21, 22, 23]
                   and s["effective_batch_size"] == 16 for s in training_stats):
            raise RuntimeError("Training reports did not verify the complete requested adaptation.")
        mg.make_inference_checkpoint(save_path, inference_path, request["finetune"]["epochs"])
        print("CHECKPOINT SAVED AND RELOADED:", inference_path)
        print("Adaptation hours:", adaptation_hours, "Per-GPU training:", training_stats)
    ''')
    cells[10] = cell("markdown", """
        ## Independent task-sharded decode, merge and final checks

        Four fresh processes, one visible GPU each, reload the adapted model-only
        checkpoint. All augmentations of a task stay together for native AAIVR ranking.
        Each uses batch 1, budget 4096 and greedy four-head speculative decoding by
        default. A single long prompt still needs to fit its GPU; the budget caps
        batching and does not truncate prompts. Missing shards, failed workers and
        malformed output replacements prevent a successful smoke marker.
    """)
    prefix = selected.get("output_prefix", "validation" if validation else "submission")
    output_name = f"{prefix}_predictions.json" if validation else "submission.json"
    info_name = f"{prefix}_run_info.json"
    cells[11] = cell("code", f'''
        started = perf_counter()
        mg.launch_inference(request_path)
        output = WORKING / ("smoke_{output_name}" if SMOKE_TEST else "{output_name}")
        repairs, inference_stats = mg.merge_shards(request, challenges, output)
        decoding_hours = (perf_counter() - started) / 3600
        summary = {{
            "mode": {mode!r}, "smoke_test": SMOKE_TEST, "checkpoint_sha256": checkpoint_hash,
            "runtime_sha256": RUNTIME_SHA256, "source_notebook_sha256": PINNED_SOURCE_NOTEBOOK_SHA256,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "attention_backend": ATTENTION_BACKEND, "enable_compile": ENABLE_COMPILE,
            "request": request, "training_tasks": len(training_challenges),
            "training_demonstration_pairs": sum(len(t["train"]) for t in training_challenges.values()),
            "decode_task_ids": list(challenges), "decode_queries": sum(len(t["test"]) for t in challenges.values()),
            "training": training_stats, "inference": inference_stats,
            "adaptation_hours": adaptation_hours, "decoding_hours": decoding_hours,
            "malformed_attempt_replacements": repairs,
        }}
        mg.write_json(WORKING / ("smoke_{info_name}" if SMOKE_TEST else "{info_name}"), summary)
        print("Parallel decode hours:", decoding_hours, "Per-GPU inference:", inference_stats)
        if SMOKE_TEST and repairs:
            raise RuntimeError("Smoke decoding produced malformed attempts; inspect logs before full mode.")
    ''')
    if validation:
        cells[11]["source"] += cell("code", f'''
            # First answer-file read: parent only, after every decode worker and merge.
            solutions_path = TEST_CHALLENGES.with_name(VALIDATION_IDENTITY["solutions"]["filename"])
            ku.verify_dataset_file(solutions_path, VALIDATION_IDENTITY["solutions"])
            scores = ku.score_validation_groups(json.loads(output.read_text()), challenges,
                json.loads(solutions_path.read_text()), validation_groups)
            score_path = WORKING / ("smoke_{prefix}_scores.json" if SMOKE_TEST else "{prefix}_scores.json")
            mg.write_json(score_path, scores)
            for group, result in scores.items():
                print(group, "tasks:", result["tasks"], "queries:", result["queries"],
                      "top1/top2 %:", result.get("top1", {{}}).get("percent"), result.get("top2", {{}}).get("percent"))
            print("VALIDATION COMPLETE:", output, "Scores:", score_path)
        ''')["source"]
    else:
        cells[11]["source"] += cell("code", '''
            if not SMOKE_TEST:
                ku.validate_submission(json.loads(output.read_text()), json.loads(TEST_CHALLENGES.read_text()))
                print(f"READY: {output} ({output.stat().st_size:,} bytes), all official runtime queries covered.")
        ''')["source"]
    cells[11]["source"] += cell("code", '''
        if SMOKE_TEST:
            print("SMOKE TEST PASSED: one real epoch, all ranks updated, checkpoint save/reload, four-worker decode/merge.")
            print("Restart the session, set SMOKE_TEST=False. Full mode loads the original epoch330 checkpoint afresh.")
            print("Do not submit smoke outputs.")
    ''')["source"]
    if selected.get("timing_benchmark"):
        cells[0] = cell("markdown", f"""
            # Five-epoch training + full 240-task decoding speed test

            This timing notebook starts fresh from the original verified epoch-330
            pretrained checkpoint, trains exactly **five epochs on all 240 tasks /
            726 demonstration pairs**, then decodes **all 336 queries with 80
            augmentations each: 26,880 augmented cases**. Default **SMOKE_TEST=False**;
            there is no four-task subset in this benchmark.

            Attach [timing240](https://www.kaggle.com/datasets/{validation['dataset_ref']}),
            [epoch330 checkpoint](https://www.kaggle.com/datasets/mithilavakde/mdlarc-600m-epoch330),
            and [FA2 wheel](https://www.kaggle.com/datasets/mithilavakde/flash-attn-cp313-torch211-cu128).
            Select **L4 × 4**, **Internet off**, restart the session and **Run All**.

            Only blocks **20–23** train; task embeddings stay disabled. Training is
            microbatch **2/GPU**, accumulation **2/GPU**, effective batch **16**,
            NorMuon **0.0003**, AdamW **0.00028**, training augmentations **80**.
            Four independent inference workers use **batch 1/GPU**, prefill budget
            **4096**, and **native four-head speculative decoding enabled**.

            The final cell prints **TIMING BENCHMARK COMPLETE**, measured training
            time, full decoding wall time and aggregate augmented-cases/second.
            Decode time includes process startup, checkpoint loading, compilation,
            generation, AAIVR and merging; post-generation answer scoring is separate.
            These are measurements for five-epoch-adapted weights; prediction lengths
            and decoding duration can change after 330 adaptation epochs.

            Outputs: `timing240_benchmark.json`, `timing240_run_info.json`,
            `timing240_predictions.json`, `timing240_scores.json`, plus per-GPU logs
            and memory reports in `runs/{run_name}/`. Original40 and synthetic200
            scores remain separate. Query solutions are opened only after decoding
            and merging. **This notebook is a timing test, not a competition submission.**
        """)
        settings = "".join(cells[1]["source"])
        settings = settings.replace("# Exactly one epoch on all demos; then four-task decode.",
                                    "# Timing benchmark: keep False for five epochs and full240 / 80-augs decode.")
        settings += '\nif SMOKE_TEST:\n    raise ValueError("Keep SMOKE_TEST=False for the requested five-epoch full240 timing benchmark.")\n'
        cells[1] = cell("code", settings)
        cells[6] = cell("markdown", """
            ## Verify inputs and prepare all 240 tasks

            Checkpoint size, full SHA-256 and architecture are checked before use.
            Both training and decoding use the entire pinned 240-task challenge set.
            Only demonstrations supply training targets; queries contain inputs only.
            The separate query-answer file is opened after all generation and merging.
        """)
        training_description = "".join(cells[8]["source"])
        training_description = training_description.replace(
            "Exactly one smoke epoch / 5 full epochs, both with microbatch 2 ×",
            "Five fresh adaptation epochs, with microbatch 2 ×")
        training_description = training_description.replace("Both use 80 training", "This test uses 80 training")
        cells[8] = cell("markdown", training_description)
        cells[11]["source"] += cell("code", '''
            query_count = sum(len(t["test"]) for t in challenges.values())
            augmented_cases = query_count * request["eval_augments"]
            if len(challenges) != 240 or query_count != 336 or augmented_cases != 26880:
                raise RuntimeError("Timing benchmark did not decode the complete requested workload.")
            decode_seconds = decoding_hours * 3600
            benchmark = {
                "training_epochs": request["finetune"]["epochs"],
                "training_tasks": len(training_challenges), "decode_tasks": len(challenges),
                "decode_queries": query_count, "inference_augments": request["eval_augments"],
                "augmented_query_cases": augmented_cases,
                "adaptation_seconds": adaptation_hours * 3600,
                "decode_wall_seconds": decode_seconds, "decode_wall_hours": decoding_hours,
                "aggregate_augmented_cases_per_second": augmented_cases / decode_seconds,
                "decode_scope": "Worker startup, load, compile, generation, AAIVR and merge; excludes answer scoring.",
                "num_gpus": NUM_GPUS, "decode_batch_per_gpu": EVAL_BATCH_SIZE,
                "prefill_token_budget": PREFILL_TOKEN_BUDGET,
                "inference_speculative": INFERENCE_SPECULATIVE,
                "checkpoint_sha256": checkpoint_hash, "runtime_sha256": RUNTIME_SHA256,
                "training": training_stats, "inference": inference_stats,
            }
            mg.write_json(WORKING / "timing240_benchmark.json", benchmark)
            print("TIMING BENCHMARK COMPLETE")
            print(f"Training: {adaptation_hours * 60:.2f} minutes (5 fresh adaptation epochs).")
            print(f"Full240 decode: {decode_seconds / 60:.2f} minutes / {decoding_hours:.3f} hours.")
            print(f"Aggregate throughput: {augmented_cases / decode_seconds:.3f} augmented query cases/second across four GPUs.")
            print("Timing report:", WORKING / "timing240_benchmark.json")
        ''')["source"]
    return cells
