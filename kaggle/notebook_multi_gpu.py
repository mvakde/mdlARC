"""Four-GPU notebook cells; keep the existing single-GPU profile separate."""

import json


def distributed_cells(cells, selected, defaults, cell):
    count = selected["num_gpus"]
    validation = selected.get("validation")
    cells[0] = cell("markdown", f"""
        # mdlARC: {selected['title']} — four L4 GPUs

        Attach the competition, `mithilavakde/mdlarc-600m-epoch330`, and the
        FlashAttention 2 wheel dataset. Select **L4 × 4**, **Internet off**.
        This notebook uses **all four GPUs** for DDP adaptation and parallel decoding.

        The checkpoint is pretraining **epoch {selected['epoch']}**. Full adaptation is
        **{selected['epochs']} epochs**, microbatch **1 per GPU**, accumulation **10**,
        effective batch **40**, with **80** training and inference augmentations.
        Only the first **2** and last **2** transformer blocks train (1, 2, 23, 24).
        Embeddings, output heads, final norm, and all other blocks are frozen.
        Each inference worker processes complete tasks, retaining every augmentation
        together for the existing AAIVR ranking.

        Start with the default **SMOKE_TEST=True**. It exercises four large-context
        runtime tasks, all four training ranks, checkpoint reload, and parallel decode.
        After `SMOKE TEST PASSED`, restart with **SMOKE_TEST=False** and use
        **Save Version → Save & Run All**. Submit that version's `submission.json`.

        The source is embedded; no repository dataset or credentials are needed.
        Instructions: `kaggle/README_600m.md`. GPU memory and runtime are to be tested
        on Kaggle; CPU validation does not establish that the full workload fits.
    """)
    cells[1] = cell("code", f'''
        import os
        import sys
        from pathlib import Path

        NUM_GPUS = {count}
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            if sys.modules["torch"].cuda.device_count() != NUM_GPUS:
                raise RuntimeError("Restart the session before selecting all four GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if int(os.environ.get("WORLD_SIZE", "1")) != 1 or "LOCAL_RANK" in os.environ:
            raise RuntimeError("Run the notebook directly; it launches its own torchrun workers.")

        CHECKPOINT_PATH = None
        TEST_CHALLENGES = None
        ATTENTION_BACKEND = "flash"
        FLASH_ATTN_WHEEL = None
        ENABLE_COMPILE = True
        TRAIN_LOG_MODE = "step"  # Log every optimizer update, including epoch/step/loss.
        PROGRESS_INTERVAL_SECONDS = 30  # Report quiet worker processes without claiming training progress.
        SMOKE_TEST = {selected.get('smoke_test', True)}  # True: four tasks/two epochs; False: full selected dataset.
        EPOCHS = {selected['epochs']}
        TRAIN_FIRST_LAYERS = {selected['train_first_layers']}
        TRAIN_LAST_LAYERS = {selected['train_last_layers']}
        TRAIN_BATCH_SIZE = {selected['batch_size']}  # Per GPU.
        GRADIENT_ACCUMULATION_STEPS = {selected['accumulation']}
        EVAL_BATCH_SIZE = {selected['eval_batch_size']}  # Per GPU; raise after checking memory.
        TRAIN_AUGMENTS = 80
        EVAL_AUGMENTS = 80
        NUM_WORKERS = 0  # Four training processes already supply CPU parallelism.
        NORMUON_LR = {selected['normuon_lr']}
        ADAMW_LR = {selected['adamw_lr']}
        MAX_FINETUNE_HOURS = 8.0

        if ATTENTION_BACKEND not in {{"flash", "native"}}:
            raise ValueError("Choose flash or native attention.")
        if not ENABLE_COMPILE:
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
        else:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
        WORKING = Path("/kaggle/working")
        WORKING.mkdir(parents=True, exist_ok=True)
        os.chdir(WORKING)
        if (WORKING / "submission.json").exists():
            (WORKING / "submission.json").replace(WORKING / "previous_submission.json")
        os.environ.pop("MDLARC_CHECKPOINT_QUEUE", None)
        os.environ.pop("MDLARC_RUN_DIR", None)
        print("Mode:", "SMOKE TEST" if SMOKE_TEST else "FULL SUBMISSION")
        print("Training GPUs:", NUM_GPUS, "full effective batch:",
              NUM_GPUS * TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    ''')
    environment = "".join(cells[3]["source"])
    environment = environment.replace('torch.cuda.device_count() != 1', 'torch.cuda.device_count() != NUM_GPUS')
    environment = environment.replace('print("Using:", torch.cuda.get_device_name(0))',
                                      'print("Using:", [torch.cuda.get_device_name(i) for i in range(NUM_GPUS)])')
    cells[3] = cell("code", environment)
    cells[6] = cell("markdown", """
        ## Verify the checkpoint and prepare runtime data

        Size and SHA-256 are checked before loading the trusted checkpoint. Epoch,
        architecture, and augmentation format are then verified. Only provided train
        pairs become adaptation targets. Test outputs are removed. Smoke mode selects
        four tasks with large demonstration/input contexts without reading test answers.
    """)
    cells[7] = cell("code", '''
        import gc
        import json
        from time import perf_counter
        import multi_gpu as mg

        CHECKPOINT_PATH = ku.find_checkpoint(Path("/kaggle/input"), CHECKPOINT_PATH)
        TEST_CHALLENGES = ku.find_test_challenges(Path("/kaggle/input"), TEST_CHALLENGES)
        print("Verifying checkpoint size and SHA-256...", flush=True)
        checkpoint_hash = ku.verify_checkpoint_file(CHECKPOINT_PATH, CHECKPOINT_IDENTITY)
        original = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False, mmap=True)
        ku.verify_checkpoint_metadata(original, CHECKPOINT_IDENTITY)
        print("Verified checkpoint epoch:", original["epoch"], "SHA-256:", checkpoint_hash)
        print("Saved architecture:", original["config"])
        del original
        gc.collect()
        data_path = WORKING / "runtime_challenges.json"
        challenges = ku.prepare_challenges(TEST_CHALLENGES, data_path, smoke_test=SMOKE_TEST,
                                          smoke_task_count=NUM_GPUS, smoke_strategy="largest")
        print("Tasks:", len(challenges), "train pairs:", sum(len(t["train"]) for t in challenges.values()),
              "test inputs:", sum(len(t["test"]) for t in challenges.values()))
    ''')
    cells[8] = cell("markdown", """
        ## Adapt on four GPUs

        Torchrun starts four processes, one complete model per GPU. DDP synchronizes
        gradients of the four trainable edge blocks. All other parameters are frozen.
        Gradients propagate through the frozen middle blocks to the first two blocks;
        freezing them reduces parameter-gradient/optimizer work but does not skip
        their forward computation or all of their backward computation.
        Full effective batch is 1 × 4 × 10 = 40. Smoke uses microbatch
        one and accumulation one on all four ranks. Workers exit after saving the final
        checkpoint, releasing optimizer, activation, compiler, and CUDA memory before
        inference starts. Worker failures stop the stage and are never retried.

        Output shows startup stages, then every optimizer update and epoch timing.
        A message every 30 seconds during silence reports only process liveness.
        `MDLARC_STARTUP_EVENT first_optimizer_step` confirms the first update finished.
        First-use compilation can delay that marker. Raw logs persist in the run folder.
    """)
    cells[9] = cell("code", f'''
        SAVED_DEFAULTS = json.loads({json.dumps(json.dumps(defaults))})
        run_dir = WORKING / "runs" / ("smoke_600m_ddp" if SMOKE_TEST else "kaggle_600m_ddp")
        run_dir.mkdir(parents=True, exist_ok=True)
        save_path = run_dir / "finetuned.pt"
        inference_path = run_dir / "inference.pt"
        request_path = run_dir / "request.json"
        request = {{
            "working": str(WORKING), "run_dir": str(run_dir), "num_gpus": NUM_GPUS,
            "checkpoint_path": str(CHECKPOINT_PATH), "checkpoint_identity": CHECKPOINT_IDENTITY,
            "data_path": str(data_path), "save_path": str(save_path),
            "inference_checkpoint": str(inference_path), "defaults": SAVED_DEFAULTS,
            "attention_backend": ATTENTION_BACKEND, "enable_compile": ENABLE_COMPILE,
            "train_log_mode": TRAIN_LOG_MODE, "progress_interval_seconds": PROGRESS_INTERVAL_SECONDS,
            "smoke_test": SMOKE_TEST, "max_finetune_hours": MAX_FINETUNE_HOURS,
            "trainable_blocks": {{"first": TRAIN_FIRST_LAYERS, "last": TRAIN_LAST_LAYERS}},
            "eval_augments": 2 if SMOKE_TEST else EVAL_AUGMENTS,
            "eval_batch_size": 1 if SMOKE_TEST else EVAL_BATCH_SIZE,
            "finetune": {{
                "epochs": 2 if SMOKE_TEST else EPOCHS,
                "batch_size": 1 if SMOKE_TEST else TRAIN_BATCH_SIZE,
                "accumulation": 1 if SMOKE_TEST else GRADIENT_ACCUMULATION_STEPS,
                "max_augments": 2 if SMOKE_TEST else TRAIN_AUGMENTS,
                "num_workers": 0 if SMOKE_TEST else NUM_WORKERS,
                "normuon_lr": NORMUON_LR, "adamw_lr": ADAMW_LR,
            }},
        }}
        mg.write_json(request_path, request)
        print("Launching DDP adaptation:", request["finetune"]["epochs"], "epochs on", NUM_GPUS,
              "GPUs; raw log:", run_dir / "train-ddp.log", flush=True)
        started = perf_counter()
        mg.launch_training(request_path)
        adaptation_hours = (perf_counter() - started) / 3600
        print("DDP adaptation hours:", adaptation_hours, flush=True)
        print("Preparing model-only checkpoint for inference...", flush=True)
        mg.make_inference_checkpoint(save_path, inference_path, request["finetune"]["epochs"])
        training_stats = [json.loads((run_dir / f"train-{{rank}}.json").read_text())
                          for rank in range(NUM_GPUS)]
        print("Per-GPU training memory:", training_stats)
    ''')
    cells[10] = cell("markdown", """
        ## Decode on four GPUs and merge

        Four independent processes load the same adapted weights. Each sees one GPU
        and receives a disjoint task list. All augmentations for a query stay with one
        worker, preserving the existing ranking. A model-only checkpoint avoids loading
        four copies of the unused optimizer state into host memory. Missing tasks,
        overlapping shards, invalid grids, or a failed worker prevent final submission.
    """)
    cells[11] = cell("code", '''
        print("Launching parallel decoding on", NUM_GPUS, "GPUs; logs:", run_dir, flush=True)
        started = perf_counter()
        mg.launch_inference(request_path)
        output = WORKING / ("smoke_submission.json" if SMOKE_TEST else "submission.json")
        repairs, inference_stats = mg.merge_shards(request, challenges, output)
        decoding_hours = (perf_counter() - started) / 3600
        summary = {
            "smoke_test": SMOKE_TEST, "checkpoint_sha256": checkpoint_hash,
            "runtime_sha256": RUNTIME_SHA256, "num_gpus": NUM_GPUS,
            "attention_backend": ATTENTION_BACKEND, "epochs": request["finetune"]["epochs"],
            "train_log_mode": TRAIN_LOG_MODE, "progress_interval_seconds": PROGRESS_INTERVAL_SECONDS,
            "trainable_blocks": request["trainable_blocks"],
            "effective_batch_size": NUM_GPUS * request["finetune"]["batch_size"] * request["finetune"]["accumulation"],
            "train_augments": request["finetune"]["max_augments"], "eval_augments": request["eval_augments"],
            "tasks": len(challenges), "training": training_stats, "inference": inference_stats,
            "adaptation_hours": adaptation_hours, "decoding_hours": decoding_hours,
            "malformed_attempt_replacements": repairs,
        }
        mg.write_json(WORKING / "submission_run_info.json", summary)
        print("Parallel decode hours:", decoding_hours)
        print("Per-GPU inference memory:", inference_stats)
        if SMOKE_TEST:
            print("SMOKE TEST PASSED. Restart, set SMOKE_TEST=False, then Save & Run All.")
        else:
            full_tasks = json.loads(TEST_CHALLENGES.read_text())
            ku.validate_submission(json.loads(output.read_text()), full_tasks)
            print(f"READY: {output} ({output.stat().st_size:,} bytes), all runtime test inputs covered.")
    ''')
    if validation:
        cells[0] = cell("markdown", """
            # mdlARC: 40-task held-out development validation

            Attach `mithilavakde/mdlarc-arc2-dev-val40-v1`, the epoch-330 model
            `mithilavakde/mdlarc-600m-epoch330`, and the FlashAttention wheel dataset.
            Select **L4 × 4**, **Internet off**. Competition data is unnecessary.

            Defaults run **all 40 tasks**, **90 adaptation epochs**, effective batch
            **40**, and **80** train/inference augmentations. Only transformer blocks
            **1, 2, 23, 24** train. Set SMOKE_TEST=True only for a four-task smoke run.

            Challenges contain 121 demonstration pairs and 56 input-only queries.
            Query answers are in a separate file, read only after decoding completes.
            The final cell reports task-weighted top-1/top-2 scores and run timing.

            Outputs: `validation_predictions.json`, `validation_scores.json`,
            `validation_run_info.json`. **This notebook is for validation.**
            Use `submit_arc2_600m.ipynb` for the final competition submission.
        """)
        settings = "".join(cells[1]["source"])
        settings = settings.replace('if (WORKING / "submission.json").exists():\n    (WORKING / "submission.json").replace(WORKING / "previous_submission.json")', '')
        cells[1] = cell("code", settings + f"\nVALIDATION_IDENTITY = {validation!r}\n")
        source = "".join(cells[7]["source"])
        source = source.replace('ku.find_test_challenges(Path("/kaggle/input"), TEST_CHALLENGES)',
                                'ku.find_validation_challenges(Path("/kaggle/input"), TEST_CHALLENGES, VALIDATION_IDENTITY)')
        cells[7] = cell("code", source)
        source = "".join(cells[9]["source"])
        source = source.replace('"smoke_600m_ddp" if SMOKE_TEST else "kaggle_600m_ddp"',
                                '"smoke_600m_val40" if SMOKE_TEST else "kaggle_600m_val40"')
        cells[9] = cell("code", source)
        source = "".join(cells[11]["source"])
        source = source.replace('"smoke_submission.json" if SMOKE_TEST else "submission.json"',
                                '"smoke_validation_predictions.json" if SMOKE_TEST else "validation_predictions.json"')
        source = source.replace('"submission_run_info.json"', '"validation_run_info.json"')
        source = source[:source.index('if SMOKE_TEST:\n')]
        source += "".join(cell("code", '''
        # Solutions are first opened here, after every prediction has finished.
        solutions_path = TEST_CHALLENGES.with_name(VALIDATION_IDENTITY["solutions"]["filename"])
        ku.verify_dataset_file(solutions_path, VALIDATION_IDENTITY["solutions"])
        scores = ku.score_validation(json.loads(output.read_text()), challenges,
                                     json.loads(solutions_path.read_text()))
        score_path = WORKING / ("smoke_validation_scores.json" if SMOKE_TEST else "validation_scores.json")
        mg.write_json(score_path, scores)
        print("Task-weighted top-1:", scores["top1"]["percent"], "%")
        print("Task-weighted top-2:", scores["top2"]["percent"], "%")
        print("VALIDATION COMPLETE:", len(challenges), "tasks,", scores["queries"], "queries.")
        print("Predictions:", output, "Scores:", score_path)
        ''')["source"])
        cells[11] = cell("code", source)
    return cells
