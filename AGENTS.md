# MDL ARC Model – Agent Guide

This file explains how the main code pieces fit together and how they use the ARC datasets under `assets/`. Its scope is the entire repository.

## Code Structure

- `tinytransformer.py`
  - Defines `TinyTransformerConfig` (dimensions, layers, dropout, max sequence length, number of ARC tasks/examples).
  - Implements the Transformer:
    - `MultiHeadSelfAttention` with query/key/value projections, causal masking, and dropout.
    - `RotaryEmbedding3D`, which applies 3D rotary positional embeddings to Q/K using per-token `(x, y, z)` coordinates.
    - `TransformerBlock` (LayerNorm → self-attention → residual + LayerNorm → feed-forward → residual).
    - `TinyTransformer`, which:
      - Embeds tokens (`token_embedding`) and a per-task example id (`example_embedding`).
      - Adds the per-task example embedding to every token in a sequence, so task identity is present throughout the serialized ARC grids.
      - Builds a 3D position tensor via `_compute_positions_3d` from token ids and the ARC serialization scheme.
      - Applies a stack of `TransformerBlock`s, final LayerNorm, and a linear head to produce logits over the vocabulary.
      - Computes autoregressive cross-entropy loss by shifting targets one token to the right and masking padded positions with `IGNORE_INDEX`.

- `utils.py`
  - Vocabulary and tokens:
    - Uses digits `0–9` plus four special tokens: `<start>`, `<next_line>`, `<input_output_separator>`, `<end>`.
    - Exposes constants like `VOCAB_SIZE`, `START_TOKEN_ID`, `NEXT_LINE_TOKEN_ID`, `IO_SEPARATOR_TOKEN_ID`, `END_TOKEN_ID`, `MAX_SEQ_LEN`, `IGNORE_INDEX`.
  - Grid/token conversion helpers:
    - `grid_to_tokens`: flattens a 2D integer grid into a 1D token list, inserting `<next_line>` between rows.
    - `encode_example`: serializes one ARC pair into a single sequence:
      - `<start>` + input grid tokens + `<input_output_separator>` + optional output grid tokens + optional `<end>`.
    - `tokens_to_grid`, `split_grids_from_tokens`, `extract_output_tokens`, `tokens_to_string`: turn token ids back into human-readable strings and 2D grids.
  - Visualization:
    - `DEFAULT_COLORS`: fixed color palette for digits 0–9.
    - `plot_grids`: uses matplotlib to show one or more grids as colored images.
  - Dataset & dataloader:
    - `SequenceExample`: small dataclass holding the serialized token tensor, `example_id` (task index), `task_id` string, `split` (`"train"` or `"test"`), `pair_index`, and `has_output`.
    - `ARCExampleDataset`:
      - Expects a *grouped-tasks* style `challenges.json` (see Dataset Structure below).
      - Loads all tasks from that file (optionally restricted via `task_whitelist`) and turns each `{input, output}` pair into an encoded token sequence.
      - Tracks a stable integer `example_id` per `task_id` so that all pairs from the same ARC task share the same example embedding.
      - Supports querying examples by split and whether they have outputs (`iter_examples`).
    - `collate_examples`: pads variable-length token sequences into a batch (`input_ids`, `attention_mask`, `example_ids`, plus metadata lists), using the `<end>` token as padding.
    - `create_dataloader`: wraps the dataset in a standard PyTorch `DataLoader` with `collate_examples`.

- `train.py`
  - CLI entry point:
    - `parse_args` defines training/eval options such as `--data-path`, `--batch-size`, `--epochs`, `--device`, `--checkpoint-path`, `--eval-only`, and logging/plotting flags.
    - Note: `DEFAULT_DATA_PATH` defaults to `assets/ARC-2/grouped-tasks/training/challenges.json`, but you can override it with `--data-path`.
  - Utilities:
    - `set_seed`: seeds PyTorch (and CUDA if available) for reproducibility.
    - `resolve_device`: chooses `cpu`, `cuda`, or `mps`.
  - Training:
    - `train_one_epoch`:
      - Iterates over the `ARCExampleDataset` dataloader.
      - Moves tensors to the device, feeds them to `TinyTransformer`, backpropagates the loss, clips gradients, and updates parameters.
      - Optionally logs a few serialized training sequences using `tokens_to_string`.
  - Inference and evaluation:
    - `greedy_generate`: autoregressively generates tokens, starting from a prompt sequence and stopping at `<end>` or `max_new_tokens`.
    - `run_inference`:
      - Takes a single ARC task (`task_id`, `pair_index`), finds the corresponding *test* example with `has_output == False`, and treats its token sequence as a prompt.
      - Calls `greedy_generate`, then decodes the output portion of the sequence into a grid with `extract_output_tokens` and `tokens_to_grid`.
      - Optionally plots the input/output grids via `split_grids_from_tokens` and `plot_grids`.
    - `evaluate_dataset`:
      - Assumes there is a `solutions.json` alongside the chosen `challenges.json`.
      - For every test example without outputs in the dataset, looks up the ground-truth output grid in `solutions.json` and compares it to the model’s decoded grid.
      - Reports exact-match accuracy over all evaluable test pairs.
  - Checkpointing:
    - `maybe_save_model`: saves model weights, config, `task_ids`, and the `data_path` into a checkpoint.
    - `load_checkpoint`: loads weights from disk; supports checkpoints that either contain a full dict or just a raw `state_dict`.
    - `infer_num_examples_from_checkpoint`: extracts the number of tasks/examples expected by the checkpointed model.
  - Orchestration:
    - `run`:
      - Resolves `data_path` (preferring CLI args, then checkpoint metadata).
      - Builds an `ARCExampleDataset` from the specified `challenges.json`.
      - Ensures that dataset task count (`dataset.num_examples`) and model `num_examples` agree.
      - Constructs the `TinyTransformer`, optionally loads checkpoint weights, then:
        - Runs training + evaluation (`evaluate_dataset`) when `eval_only` is `False`.
        - Runs either a single-task inference (`run_inference`) or full evaluation (`evaluate_dataset`) when `eval_only` is `True`.
    - `main` is the CLI entry; `python train.py --data-path <path/to/challenges.json>` is the primary way to run the model outside notebooks.

- `run_experiments.ipynb`
  - Jupyter notebook wrapper around `utils` and `train`.
  - Lets you:
    - Configure arguments in a Python dict (including `data_path` pointing to small `assets/script-tests/.../challenges.json` subsets).
    - Launch training + auto-eval or eval-only runs by calling `train.run` with an `argparse.Namespace`.
    - Optionally run single-example inference and plot predicted grids.

## Dataset Structure (`assets/`)

### High-level layout

- `assets/ARC-1`
  - `data/training/*.json` and `data/evaluation/*.json`:
    - One file per original ARC task.
    - Each file has the canonical ARC structure:
      - `"train"`: list of examples with `"input"` and `"output"` grids (2D lists of integers 0–9).
      - `"test"`: list of examples; some have `"output"`, others omit it (challenge test pairs).
  - `data/rearc-training/*.json`:
    - Alternate training data for ARC-1 tasks.
    - Each file is a *flat list* of `{ "input": grid, "output": grid }` objects (no `"train"`/`"test"` split).
    - Not used directly by the current `ARCExampleDataset`, but useful for creating custom grouped-tasks files.
  - `grouped-tasks/training/challenges.json` and `grouped-tasks/training/solutions.json`
  - `grouped-tasks/evaluation/challenges.json` and `grouped-tasks/evaluation/solutions.json`

- `assets/ARC-2`
  - `data/training/*.json` and `data/evaluation/*.json`: same per-task ARC format as `ARC-1` (with `"train"` and `"test"` arrays of `{input, output}` pairs).
  - `grouped-tasks/training/challenges.json` and `grouped-tasks/training/solutions.json`
  - `grouped-tasks/evaluation/challenges.json` and `grouped-tasks/evaluation/solutions.json`
  - `train.py` uses `assets/ARC-2/grouped-tasks/training/challenges.json` as its default dataset.

- `assets/script-tests`
  - Small, curated subsets designed for quick experiments and debugging.
  - `grouped-tasks*/challenges.json` and `grouped-tasks*/solutions.json`:
    - Same grouped-tasks structure as the ARC-1/ARC-2 grouped files (see below), but containing only a handful of tasks.
    - Examples:
      - `grouped-tasks-00d62c1b/` focuses on a single ARC task.
      - `grouped-tasks-0-4x/`, `grouped-tasks-2x/`, `grouped-tasks-4x/` provide progressively larger subsets.
  - `separate-tasks/*.json`:
    - Per-task ARC files in the original `"train"`/`"test"` format, similar to `assets/ARC-1/data/training`.
    - Useful when you want to work with one task at a time or build new grouped-tasks files.

### Grouped-tasks JSON format

All `grouped-tasks/*/challenges.json` files share the same schema, which is what `ARCExampleDataset` expects:

- Top-level object: mapping from ARC `task_id` (e.g. `"00d62c1b"`) to a task description:
  - `"train"`: list of examples, each `{ "input": grid, "output": grid }`.
  - `"test"`: list of examples, each `{ "input": grid }` or `{ "input": grid, "output": grid }`.
    - For *challenge* test pairs, `"output"` is omitted or set to `null`; these are the ones the model must predict.
  - Each `grid` is a 2D list of integers `0–9`, matching the model’s digit tokens and color palette.

The companion `grouped-tasks/*/solutions.json` files contain the ground-truth outputs for challenge test pairs:

- Top-level object: mapping from the same `task_id` to a list of output grids.
- Each entry is a list indexed by `pair_index` corresponding to the test examples inside `"test"` for that task.
- Each element is a single 2D integer grid, which `evaluate_dataset` compares against the model’s prediction.

## How the Code Uses the Dataset

- `ARCExampleDataset` loads `challenges.json`:
  - For each `task_id`:
    - Iterates over `"train"` and `"test"` lists (for the specified `splits`).
    - Encodes each `{input, output}` pair into a token sequence with `encode_example`.
    - Sets `has_output` based on whether `"output"` is present.
  - Assigns a unique integer `example_id` per `task_id` so that all pairs from the same task share the same example embedding in `TinyTransformer`.

- `train.py` training loop (`train_one_epoch`):
  - Consumes batches from `create_dataloader`, which yields:
    - `input_ids`: padded token sequences.
    - `attention_mask`: marks real tokens vs padding.
    - `example_ids`: per-sequence task indices.
  - For each batch:
    - Calls `model(input_ids, example_ids, attention_mask=attention_mask)`.
    - Uses the returned `loss` (autoregressive cross-entropy over the serialized ARC sequences).

- `TinyTransformer._compute_positions_3d`:
  - Inspects `input_ids` token-by-token and computes `(x, y, z)` coordinates that reflect the ARC grid layout:
    - `<start>` is anchored at `(0, 0, 0)`.
    - Input grid tokens (digits and `<next_line>`) live on layer `z=1`; `x` and `y` follow column and row indices (with `<next_line>` stepping to a new row).
    - `<input_output_separator>` sits at `(0, 0, 2)` and switches subsequent tokens to the output layer.
    - Output grid tokens live on `z=3`, again laid out by rows/columns.
    - `<end>` is placed at `(0, 0, 4)`.
  - These coordinates drive the 3D RoPE module (`RotaryEmbedding3D`), giving the attention mechanism structured positional information that respects input vs output regions and grid geometry.

- Inference and evaluation:
  - `run_inference`:
    - Picks a test example with `has_output == False` (no output grid in `challenges.json`).
    - Uses its encoded sequence as a prompt, runs `greedy_generate`, and decodes only the tokens after `<input_output_separator>` as the predicted output grid.
  - `evaluate_dataset`:
    - For each challenge test example, it:
      - Generates an output grid as above.
      - Looks up the reference grid from `solutions.json` (`solutions[task_id][pair_index]`).
      - Compares the two grids for exact equality and aggregates accuracy.

## Practical Notes for Future Changes

- When adding new datasets:
  - Prefer to create new `grouped-tasks/.../challenges.json` and `solutions.json` files that follow the existing schema.
  - Ensure all grid values are integers in `[0, 9]` so they are compatible with the current vocabulary and color palette.
  - If you change the format, also update `ARCExampleDataset` and any evaluation logic that assumes the grouped-tasks structure.

- When modifying the model:
  - Preserve the semantics of the special tokens and how sequences are constructed—other components (dataset, position computation, evaluation) depend on them.
  - If you change `num_examples` or the set of `task_ids`, update checkpoints or regenerate them to avoid mismatches between `TinyTransformerConfig.num_examples` and `dataset.num_examples`.
