"""Pinned checkpoint identities and adaptation recipes for shipped notebooks."""

PROFILES = {
    "public-v3-e25": {
        "filename": "submit_arc2.ipynb",
        "title": "ARC2 public/composition epoch 25",
        "session": "vast-54994367-public-v3-b112-e25-bw0",
        "epoch": 25,
        "size": 4364936923,
        "sha256": "d5912ada04df6010369a59a7c5b6043261411825f95d5f3c304ad4461068a0dd",
        "augmentation_sampling_mode": "single_grid_prefix",
        "config": {"d_model": 768, "n_layers": 8, "num_task_embedding_latents": 3072},
        "epochs": 330, "batch_size": 8, "accumulation": 4,
        "eval_batch_size": 8, "normuon_lr": 0.0003, "adamw_lr": 0.003,
        "training_demo_schedule": "all", "inference_demo_policy": "all",
    },
    "600m-e330": {
        "filename": "submit_arc2_600m.ipynb",
        "title": "600M scaling ladder epoch 330",
        "session": "vast-ddp8-b200-b24a1-lr061-e70-noon-20261025-r5",
        "epoch": 330,
        "size": 5726752055,
        "sha256": "a17e48c09c7d4c99755faa4a5eaa02d96cda88f137b1471194d1ffd20b661582",
        "augmentation_sampling_mode": "context_palette",
        "config": {
            "d_model": 1536, "d_ff": 4096, "n_layers": 24, "n_heads": 24,
            "num_task_embedding_latents": 3072, "task_embeddings": False,
            "attention_pattern": "pair_group", "split_target_demo_attention": True,
            "n_predict": 4,
        },
        "num_gpus": 4,
        "preserve_runtime": True,
        "train_first_layers": 2, "train_last_layers": 2,
        "epochs": 90, "batch_size": 1, "accumulation": 10,
        "eval_batch_size": 1, "normuon_lr": 0.0016, "adamw_lr": 0.00028,
        "training_demo_schedule": "all", "inference_demo_policy": "leave_one_out",
    },
}

PROFILES["600m-val40"] = dict(
    PROFILES["600m-e330"],
    filename="validate_arc2_600m_val40.ipynb",
    title="600M epoch 330 — 40-task development validation",
    smoke_test=False,
    validation={
        "dataset_ref": "mithilavakde/mdlarc-arc2-dev-val40-v1",
        "tasks": 40, "demonstration_pairs": 121, "query_pairs": 56,
        "challenges": {"filename": "arc2_val40_challenges.json", "size": 247515,
                       "sha256": "ce706884dff93678a7295cc69474c7dbe90afddc55044c1bc23b278a7279b0f5"},
        "solutions": {"filename": "arc2_val40_solutions.json", "size": 51494,
                      "sha256": "3af6959c9b527ab0916dd572d20e9377687856cb5adddccdc9fcf19c03b63570"},
    },
)

# Separate distributable artifacts: never overwrite the runtime pinned by remote runs.
PROFILES["600m-last4"] = dict(
    PROFILES["600m-e330"], filename="submit_arc2_600m_last4.ipynb",
    title="600M epoch 330 — last four blocks", recipe="last4",
    runtime_source="kaggle/submit_arc2_600m.ipynb",
    runtime_source_sha256="71ad055eb93ce682d470d695d95dd98bacbd375a2799774b08c419aea10a23ba",
    train_first_layers=0, train_last_layers=4, epochs=330,
    batch_size=2, accumulation=2, normuon_lr=0.0003, adamw_lr=0.00028,
    eval_batch_size=1, prefill_token_budget=4096, inference_speculative=True,
    smoke_test=True,
)
PROFILES["600m-val240-last4"] = dict(
    PROFILES["600m-last4"], filename="validate_arc2_600m_val240_last4.ipynb",
    title="600M epoch 330 — last four blocks, 240-task validation",
    validation={
        "dataset_ref": "mithilavakde/mdlarc-arc2-timing240-v1",
        "tasks": 240, "demonstration_pairs": 726, "query_pairs": 336,
        "challenges": {"filename": "arc2_val240_challenges.json", "size": 1951766,
                       "sha256": "4692772246e8778ca30dc2cadb9d9807b0ad580fa8f837340095fca2f07f2813"},
        "solutions": {"filename": "arc2_val240_solutions.json", "size": 352752,
                      "sha256": "551d9d2fe02194a7d924123c394a94ec161410f24159e5ce7f3909b140bedfd2"},
        "manifest": {"filename": "timing_manifest.json", "size": 1250346,
                     "sha256": "b051f8b0eda100f09356ada6f5339e6532cbdd64bbc72d7bb1cac5b51b4af826"},
    },
)

PROFILES["600m-val240-last4-timing5"] = dict(
    PROFILES["600m-val240-last4"],
    filename="time_arc2_600m_val240_last4_e5.ipynb",
    title="600M last four blocks — five epochs and full timing240 decode",
    epochs=5, smoke_test=False, timing_benchmark=True,
    run_name="last4_val240_timing_e5", output_prefix="timing240",
)
