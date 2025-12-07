from pathlib import Path
import argparse
import importlib
import utils, tinytransformer, train

importlib.reload(utils)  # pick up code changes during iteration
importlib.reload(tinytransformer)
importlib.reload(train)

args = {
    # run config
    "num_workers": 0,
    "device": "cuda",  # 'cuda' | 'mps' | 'cpu'
    # paths - must pass as Path("<path_to_dir>")
    "save_path": Path("runs/tiny4.pt"),
    "checkpoint_path": None,  # Path("runs/tiny.pt"),  # or None to start from scratch
    "data_path": Path("assets/script-tests/grouped-tasks/challenges.json"),
    # "data_path": Path("assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_train/challenges.json"),  # this dataset has dihedral augments only on the train sequences (use this for training)
    # "data_path": Path("assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_both/challenges.json"), # this has dihedral augments on train and test sequences (only use for evaluation)
    # hyperparameters
    "epochs": 1,
    "batch_size": 110,
    "val_batch_size": 60,
    "enable_color_aug_train": False,
    "max_color_augments_train": 15,
    "color_aug_seed": 42,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "seed": 42,
    # Visibility toggles
    "log_train_strings": False,
    "log_train_limit": 10,
    "log_inference_prompt": False,
}
cfg = argparse.Namespace(**args)

model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)


# Training only
train.train_model(
    cfg,
    model=model,
    dataloader=dataloader,
    dataset=dataset,
    device=device,
    data_path=data_path,
)

from pathlib import Path
import argparse
import importlib
import utils, tinytransformer, train

importlib.reload(utils)  # pick up code changes during iteration
importlib.reload(tinytransformer)
importlib.reload(train)

args = {
    # run config
    "num_workers": 0,
    "device": "cuda",  # 'cuda' | 'mps' | 'cpu'
    # paths - must pass as Path("<path_to_dir>")
    "save_path": Path("runs/tiny4.pt"),
    "checkpoint_path": None,  # Path("runs/tiny.pt"),  # or None to start from scratch
    "data_path": Path("assets/script-tests/grouped-tasks/challenges.json"),
    # "data_path": Path("assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_train/challenges.json"),  # this dataset has dihedral augments only on the train sequences (use this for training)
    # "data_path": Path("assets/ARC-1/grouped-tasks/concept_plus_combined_dihedral_both/challenges.json"), # this has dihedral augments on train and test sequences (only use for evaluation)
    # hyperparameters
    "epochs": 1,
    "batch_size": 110,
    "val_batch_size": 60,
    "enable_color_aug_train": False,
    "max_color_augments_train": 15,
    "color_aug_seed": 42,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "seed": 42,
    # Visibility toggles
    "log_train_strings": False,
    "log_train_limit": 10,
    "log_inference_prompt": False,
}
cfg = argparse.Namespace(**args)

model, dataset, dataloader, device, data_path = train.build_model_and_data(cfg)

# ## AAIVR and visualise
# importlib.reload(utils)


# cfg.visualize_augmented_outputs = True
# cfg.visualize_task_ids = ["00d62c1b", "e0fb7511" , "00576224" , "3aa6fb7a"] # always pass as list,  # e.g. ["00d62c1b"]
# cfg.visualize_split = "train"
# cfg.visualize_pair_index =  None  # None = all pairs/augments
# cfg.visualize_plot = True
# cfg.visualize_log_prompts = False
# cfg.visualize_aaivr_top_k = 2


# test_results = evaluation.get("test", {}).get("results", [])
# aaivr_results = utils.run_aaivr_on_results(test_results)

# print("\nAAIVR selections (pass@2) for test split:")
# if not aaivr_results:
#     print("  no test results available for AAIVR voting")

# summary = utils.summarize_aaivr_pass_at_k(aaivr_results)
# evaluated = summary.get("evaluated", 0)
# hits = summary.get("hits", 0)

# print("AAIVR pass@2 with targets:", f"{hits} / {evaluated} original test pairs")
# x = set([])
# for sel in aaivr_results:
#     if sel.pass_at_k:
#         x.add(sel.task_id)

# print("Unique tasks: ", len(set(x)))

# # Optional: Visualize augmented outputs for selected tasks (dihedral + color)
# if cfg.visualize_augmented_outputs and cfg.visualize_task_ids:
#     selected_split = cfg.visualize_split
#     pair_idx = cfg.visualize_pair_index
#     vis_plot = cfg.visualize_plot
#     vis_log_prompts = cfg.visualize_log_prompts
#     top_k = max(1, int(cfg.visualize_aaivr_top_k))

#     split_results = evaluation.get(selected_split, {}).get("results", [])
#     filtered = [
#         res
#         for res in split_results
#         if res.get("task_id") in cfg.visualize_task_ids
#         and (pair_idx is None or res.get("pair_index") == pair_idx)
#     ]
#     if not filtered:
#         print(
#             f"\n[visualize] No results found for tasks {cfg.visualize_task_ids} "
#             f"in split '{selected_split}' (pair_index={pair_idx})."
#         )
#     else:
#         pair_label = pair_idx if pair_idx is not None else "all"
#         print(
#             f"\n[visualize] Showing augmented predictions for tasks {cfg.visualize_task_ids} "
#             f"in split '{selected_split}' (pair_index={pair_label})"
#         )
#         filtered.sort(
#             key=lambda r: (
#                 r.get("task_id", ""),
#                 r.get("pair_index", -1),
#                 r.get("color_permutation_index", -1),
#             )
#         )
#         for res in filtered:
#             task_id = res.get("task_id")
#             pair_index = res.get("pair_index")
#             color_idx = res.get("color_permutation_index", None)
#             color_label = f", color_perm={color_idx}" if color_idx is not None else ""
#             print(f"\nTask {task_id} pair {pair_index} ({selected_split}{color_label})")
#             if vis_log_prompts:
#                 print("Prompt tokens:", utils.tokens_to_string(res["prompt_tokens"]))
#             print(
#                 "Generated output tokens:", utils.tokens_to_string(res["output_tokens"])
#             )
#             if res.get("target_output_tokens"):
#                 print(
#                     "Target output tokens:",
#                     utils.tokens_to_string(res["target_output_tokens"]),
#                 )
#             print("Predicted grid:")
#             for row in res["output_grid"]:
#                 print(row)
#             if res.get("target_grid"):
#                 print("Target grid:")
#                 for row in res["target_grid"]:
#                     print(row)
#             if vis_plot:
#                 prompt_grids = utils.split_grids_from_tokens(res["prompt_tokens"])
#                 input_grid = prompt_grids[0] if prompt_grids else []
#                 to_plot = [input_grid, res["output_grid"]]
#                 if res.get("target_grid"):
#                     to_plot.append(res["target_grid"])
#                 try:
#                     utils.plot_grids(
#                         to_plot,
#                         title=(
#                             f"{task_id} pair {pair_index} ({selected_split}{color_label})"
#                         ),
#                     )
#                 except Exception as e:
#                     print(
#                         f"  skipping visualization for {task_id} pair {pair_index}: {e}"
#                     )

#         if selected_split == "test":
#             print(f"\n[visualize] AAIVR top-{top_k} for selected tasks")
#             aaivr_subset = [res for res in filtered if res.get("split") == "test"]
#             selections = utils.run_aaivr_on_results(
#                 aaivr_subset, top_k=top_k, discard_input_copies=True
#             )
#             if not selections:
#                 print("  no AAIVR selections for the chosen subset.")
#             else:
#                 for sel in selections:
#                     if sel.pass_at_k is None:
#                         pass_str = "N/A"
#                     else:
#                         pass_str = "PASS" if sel.pass_at_k else "MISS"
#                     print(
#                         f"  Task {sel.task_id} base_pair {sel.original_pair_index}: "
#                         f"{pass_str} (generated={sel.num_generated}, valid={sel.num_valid})"
#                     )
#                     if sel.target_grid is not None:
#                         print("    Target grid:")
#                         for row in sel.target_grid:
#                             print(f"    {row}")
#                     for idx, cand in enumerate(sel.ranked_candidates[:top_k]):
#                         grid = cand["grid"]
#                         count = cand["count"]
#                         print(f"    Candidate {idx + 1} (count={count}):")
#                         for row in grid:
#                             print(f"      {row}")
#                         if vis_plot:
#                             try:
#                                 utils.plot_grids(
#                                     [grid],
#                                     title=(
#                                         f"{sel.task_id} pair {sel.original_pair_index} "
#                                         f"AAIVR cand {idx + 1}"
#                                     ),
#                                 )
#                             except Exception as e:
#                                 print(f"      plot failed: {e}")
