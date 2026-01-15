# AGENTS.md - mdlARC Codebase Documentation

This document provides comprehensive documentation for the mdlARC codebase, a transformer-based solution for the ARC-AGI (Abstraction and Reasoning Corpus) challenge.

## Overview

mdlARC is a lightweight (~60M parameter) transformer model designed to solve ARC puzzles through self-supervised compression. The approach tokenizes 2D grids into sequences and trains a transformer to predict outputs autoregressively, using color permutation and dihedral (geometric) augmentations for data efficiency.

**Key Results:**
- ~36% accuracy on ARC-1 public evaluation
- ~$2 total compute cost on A100 GPU
- 127 min training + 49 min inference

---

## Directory Structure

```
mdlARC/
├── src/                              # Core source code
│   ├── common.py                     # Shared utilities (vocab, tokenization, dataset, transforms)
│   ├── build.py                      # Model and data building (build_model_and_data)
│   ├── train.py                      # Training pipeline (train_model)
│   ├── evaluate.py                   # Evaluation pipeline (run_evaluation_configs)
│   ├── utils.py                      # Visualization, scoring, memory cleanup
│   ├── tinytransformer.py            # Transformer model architecture
│   ├── inference.py                  # Token generation with KV caching
│   ├── augment.py                    # Color and dihedral augmentation system
│   ├── aaivr.py                      # Augmentation inverse voting
│   └── normuon.py                    # NorMuon optimizer (Muon + AdamW hybrid)
├── dataset_building_scripts/
│   └── build_datasets.py             # Dataset downloader and builder
├── assets/                           # Dataset files (after building)
│   ├── challenges.json               # Combined train/eval challenges
│   └── solutions.json                # Ground truth solutions
├── runs/                             # Training outputs (created at runtime)
│   ├── tiny.pt                       # Model checkpoint
│   ├── training_log.txt              # Training logs
│   └── <eval_name>/submission.json   # Evaluation outputs
├── interactive-run.ipynb             # Main execution notebook
└── requirements.txt                  # Dependencies
```

**File Organization:**
The codebase is organized around 3 main functionalities:
1. **build.py** - Building model and data (`build_model_and_data`)
2. **train.py** - Training loop (`train_model`)
3. **evaluate.py** - Evaluation pipeline (`run_evaluation_configs`)

Shared code used by multiple modules lives in **common.py**, while **utils.py** contains standalone utilities (visualization, scoring, memory cleanup).

---

## Main Entry Point: interactive-run.ipynb

The notebook `interactive-run.ipynb` is the primary interface for training, evaluation, scoring, and visualization. It contains the following cells:

### Cell 1: Environment Setup
```python
root_folder, mount_folder = "content", "content/drive/MyDrive"  # for colab
%cd /$root_folder/
!git clone https://github.com/mvakde/mdlARC.git
%cd /$root_folder/mdlARC
```
Sets up the environment for Colab or Modal execution.

### Cell 2: Dataset Building
```python
!python dataset_building_scripts/build_datasets.py --datasets arc1 conceptarc --splits train eval --with-solutions --cleanup none
```
Downloads and builds the ARC datasets.

### Cell 3: Training Setup & Execution
Configures all hyperparameters and builds model/data:
```python
from build import build_model_and_data
cfg = argparse.Namespace(**args)
model, dataset, dataloader, device, data_path = build_model_and_data(cfg)
```

### Cell 4: Training Loop
```python
train.train_model(cfg, model=model, dataloader=dataloader, dataset=dataset, device=device, data_path=data_path)
```

### Cell 5: Memory Cleanup
```python
utils.cleanup_memory(globals())
```

### Cell 6: Evaluation
```python
import evaluate
eval_results = evaluate.run_evaluation_configs(cfg, EVAL_CONFIGS, ...)
```

### Cell 7: Visualization
```python
utils.visualize_submissions(Path("runs") / EVAL_SUB_FOLDER / "submission.json", ...)
```

### Cell 8: Scoring
```python
score = utils.score_arc_submission(SOLUTIONS_FILE, SUBMISSION_FILE)
```

---

## Source Modules

### common.py - Shared Utilities

**Purpose:** Central module containing all shared code used by build, train, and evaluate modules.

**Vocabulary & Constants:**

```python
VOCAB_SIZE = 14  # 0-9 digits + 4 special tokens
SPECIAL_TOKENS = {
    "<start>": 10,
    "<next_line>": 11,
    "<input_output_separator>": 12,
    "<end>": 13
}
MAX_SEQ_LEN = 1863
IGNORE_INDEX = -100
```

**Key Functions:**

| Function | Description |
|----------|-------------|
| `encode_example(input_grid, output_grid)` | Convert grid pair to token sequence |
| `grid_to_tokens(grid)` | Flatten 2D grid to 1D token sequence |
| `tokens_to_grid(tokens)` | Reconstruct grid from tokens |
| `compute_positions_3d(tokens)` | Compute (x,y,z) coordinates per token (Numba JIT) |
| `apply_dihedral_transform(grid, transform_id)` | Apply geometric transformation |
| `apply_inverse_dihedral_transform(grid, transform_id)` | Reverse geometric transformation |
| `apply_color_permutation_to_tokens(tokens, mapping)` | Apply color remapping to tokens |
| `apply_color_permutation_to_grid(grid, mapping)` | Apply color remapping to grid |
| `load_challenges(path)` | Load JSON dataset |
| `extract_output_tokens(tokens)` | Extract output portion from token sequence |
| `set_seed(seed)` | Set all random seeds for reproducibility |
| `resolve_device(device_str)` | Resolve device string to torch.device |
| `capture_rng_state(device)` | Capture current RNG state for checkpointing |
| `restore_rng_state(state, device)` | Restore RNG state from checkpoint |

**Dihedral Transforms (8 operations):**
- `identity`, `rot90`, `rot180`, `rot270`
- `flip_horizontal`, `flip_vertical`
- `flip_main_diagonal`, `flip_anti_diagonal`

**Dataset Classes:**

`SequenceExample` - Dataclass representing a single tokenized example

`ARCExampleDataset` - Main dataset class:
- Loads from ARC JSON format
- Pre-computes all 8 dihedral variants
- Caches 3D positional encodings
- Supports task whitelist filtering

`LengthBucketBatchSampler` - Groups examples by sequence length for efficient batching

`collate_examples` - Collate function for DataLoader

`create_dataloader` - Factory for creating DataLoaders with proper batching

---

### build.py - Model and Data Building

**Purpose:** Constructs dataset, dataloader, and model from configuration. Entry point for setup.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `build_model_and_data(cfg, checkpoint, reuse_dataset, is_eval)` | Main builder function |
| `load_checkpoint(checkpoint_path)` | Load model checkpoint from disk |
| `infer_num_examples_from_checkpoint(checkpoint)` | Extract task count from checkpoint |

**Usage:**
```python
from build import build_model_and_data
model, dataset, dataloader, device, data_path = build_model_and_data(cfg)
```

---

### train.py - Training Pipeline

**Purpose:** Orchestrates the full training loop with checkpoint management, optimizer handling, and validation.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `train_model(cfg, model, dataloader, dataset, device, data_path)` | Full training loop |
| `train_one_epoch(...)` | Single epoch training |
| `validate_one_epoch(model, dataloader, device)` | Validation pass |
| `maybe_save_model(...)` | Conditional checkpoint saving |

**Features:**
- Automatic Mixed Precision (AMP) with bfloat16
- Gradient clipping and accumulation
- WSD learning rate schedule (warmup, stable, decay)
- NorMuon + AdamW hybrid optimizer support
- Comprehensive RNG state management

---

### utils.py - Standalone Utilities

**Purpose:** Visualization, scoring, and memory cleanup functions.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `plot_grids(grids)` | Visualize grids with matplotlib |
| `visualize_submissions(submission, solutions, mode)` | Visual comparison of predictions |
| `score_arc_submission(solutions, submission)` | Compute official ARC score |
| `cleanup_memory(globals_dict)` | Free GPU memory and clean caches |

**Note:** For backward compatibility, common.py exports are re-exported from utils.py

---

### tinytransformer.py - Model Architecture

**Purpose:** Lightweight transformer optimized for ARC with 3D rotary position embeddings.

**Key Classes:**

**`TinyTransformerConfig`** - Model configuration:
```python
vocab_size: int = 14
max_seq_len: int = 1863
d_model: int = 128      # Embedding dimension
n_heads: int = 4        # Attention heads
d_ff: int = 512         # FFN hidden dimension
n_layers: int = 4       # Transformer blocks
dropout: float = 0.1
num_examples: int       # Number of unique tasks
```

**`TinyTransformer`** - Main model:
- Token embeddings + task-specific example embeddings
- Stack of TransformerBlocks
- RMSNorm final layer
- Linear lm_head for vocabulary prediction

**`RotaryEmbedding3D`** - Custom 3D RoPE:
- Splits head_dim into 3 sections for (x, y, z)
- x: column position (0-32)
- y: row position (0-32)
- z: phase (input/separator/output/end)

**`MultiHeadSelfAttention`**:
- Flash Attention via `F.scaled_dot_product_attention`
- KV caching for efficient generation
- Two forward modes: full sequence (training) and cached (inference)

**`FeedForward`** - Gated Linear Units (GLU):
- SiLU activation
- Projects to 2x hidden dim, gates, projects back

**Forward Signatures:**
```python
# Training
logits, loss, input_loss, output_loss, num_output_tokens = model.forward(tokens, positions_3d, example_ids, targets)

# Generation (with KV cache)
logits = model.forward_generate(tokens, positions_3d, example_id, caches, decode_mode)
```

---

### evaluate.py - Evaluation Pipeline

**Purpose:** Run inference across dataset splits, aggregate results, compute metrics.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_evaluation_configs(cfg, eval_configs, ...)` | Main evaluation orchestrator |
| `summarize_split_results(results)` | Aggregate metrics per split |

**Evaluation Config Format:**
```python
EVAL_CONFIGS = [
    ("eval_100color_both", 100, PATH_BOTH),  # (name, max_augments, dataset_path)
]
```

**Output Metrics:**
- Shape correctness (dimensions match)
- Pixel accuracy (correct cells / total cells)
- Full correctness (exact match)
- Per-task pass rate (ARC official metric)

---

### inference.py - Token Generation

**Purpose:** Autoregressive generation with KV caching for efficient inference.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_split_inference(model, dataset, split, cfg, augmentor)` | Inference on dataset split |
| `DEFAULT_MAX_NEW_TOKENS = 931` | Maximum tokens to generate |

**Features:**
- Compiled grid state tracking (`@torch.compile`)
- Color permutation + dihedral transforms during inference
- Temperature and top-k sampling support
- Batch generation with early stopping

---

### augment.py - Augmentation System

**Purpose:** Color permutation and dihedral augmentation with intelligent deduplication.

**Key Exports:**

| Function/Class | Description |
|----------------|-------------|
| `build_augmentor(dataset, cfg)` | Factory function |
| `Augmentor` | Manages augment selection per example |
| `Augments` | Dataclass storing augment indices and color maps |

**Color Permutation Strategy:**
- Permutes colors 1-9 (0 = background is fixed)
- Uses unranking algorithm for permutation enumeration
- Filters duplicates via hashing
- Respects "output-only colors" that don't appear in inputs

**Dihedral Augmentation:**
- All 8 geometric operations available
- Configurable application to test examples

**Selection:**
```python
augmentor.select_for_example(example, epoch, index)
# Returns: (color_mapping, dihedral_index)
```
Epoch-aware selection ensures different augments per epoch, consistent within epoch.

---

### aaivr.py - Augmentation Inverse Voting

**Purpose:** Aggregate predictions from augmented variants through voting.

AAIVR = Automated Augmentation Inverse Voting and Ranking

**Key Functions:**

| Function | Description |
|----------|-------------|
| `run_aaivr_on_results(results, ...)` | Main voting aggregation |
| `AAIVRSelection` | Dataclass with selected outputs + ranking |

**Process:**
1. Invert dihedral transforms on predictions
2. Invert color permutations if mappings provided
3. Filter to rectangular grids only
4. Optionally discard input copies
5. Vote across augmented variants
6. Select top-2 candidates (Pass@2)

---

### normuon.py - Optimizer

**Purpose:** NorMuon optimizer combining Muon (for linear weights) + AdamW (for other params).

**Key Features:**
- Splits params: Muon-eligible (2D linear weights) vs. others (AdamW)
- Muon: orthogonal updates via Newton-Schulz iteration
- Per-group learning rates and weight decay
- SVD-free orthogonalization for efficiency

```python
optimizer = SingleDeviceNorMuonWithAuxAdam(
    muon_params=linear_weights,
    adam_params=other_params,
    lr=1.66e-3,
    momentum=0.95,
    beta2=0.95,
)
```

---

## Dataset Building

### build_datasets.py

**Usage:**
```bash
python dataset_building_scripts/build_datasets.py \
    --datasets arc1 conceptarc \
    --splits train eval \
    --with-solutions \
    --cleanup none
```

**Supported Datasets:**
- `arc1` - Official ARC-1 benchmark
- `arc2` - ARC-2 (extended)
- `conceptarc` - ConceptARC

**Options:**
- `--datasets`: Which datasets to download
- `--splits`: train, eval, or both
- `--with-solutions`: Include solution files
- `--cleanup`: none/partial/full - cleanup intermediate files

**Output:** Combined JSON files in `assets/` directory

---

## Configuration Reference

### Training Configuration

```python
args = {
    # Run identification
    "name": "experiment-name",
    "GPU": "A100",  # For logging

    # Paths
    "data_path": Path("assets/challenges.json"),
    "train_log_file": Path("runs/training_log.txt"),
    "save_path": Path("runs/tiny.pt"),
    "checkpoint_path": None,  # Or Path("runs/tiny.pt") to resume
    "checkpoint_epochs": [300, 400, 500],  # When to checkpoint

    # Training
    "epochs": 20,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "do_validate": True,
    "val_batch_size": 140,

    # Augmentation
    "enable_aug": True,
    "max_augments": 10,
    "enable_color_aug": True,
    "color_apply_to_test": True,
    "enable_dihedral_aug": True,
    "dihedral_apply_to_test": True,

    # Optimizer
    "optimizer": "normuon",  # "adamw" | "normuon"
    "normuon_lr": 1.66e-3,
    "normuon_momentum": 0.95,
    "normuon_beta2": 0.95,
    "lr": 3e-4,  # AdamW learning rate

    # Learning rate schedule
    "warmup_pct": 0.02,
    "wsd_decay_start_pct": 0.8,
    "lr_floor": 0.0,

    # Regularization
    "weight_decay": 0.1,
    "attention_weight_decay": 0.01,
    "token_embedding_weight_decay": 0.01,
    "task_embedding_weight_decay": 0.01,
    "grad_clip": 1.0,
    "dropout": 0.1,
    "seed": 42,

    # Model architecture
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "n_layers": 4,

    # Inference
    "inference_temperature": None,
    "inference_top_k": None,
}
```

### Evaluation Configuration

```python
EVAL_CONFIGS = [
    ("eval_100color_both", 100, PATH_BOTH),  # (name, max_augments, path)
]

EVAL_BATCH_SIZE = 1300
SPLITS = ["test"]
CHECKPOINT_PATH = Path("runs/tiny.pt")
SOLUTIONS_PRESENT = False  # True if solutions.json available
EVAL_TASK_IDS = None  # None for all, or ["00576224", ...] for specific tasks
LOG_CORRECT_GRIDS = False
```

---

## Token Encoding Scheme

Grids are encoded as token sequences:

```
[<start>] [input_tokens] [<next_line>]* [<sep>] [output_tokens] [<next_line>]* [<end>]
```

**Token IDs:**
- 0-9: Grid cell color values
- 10: `<start>` - Sequence start
- 11: `<next_line>` - Row delimiter
- 12: `<input_output_separator>` - Separates input from output
- 13: `<end>` - Sequence end

**Example:**
```
Input grid:     Output grid:
[[1,2],         [[3,4],
 [3,4]]          [5,6]]

Tokens: [10, 1, 2, 11, 3, 4, 12, 3, 4, 11, 5, 6, 13]
        [start, 1,2,nl, 3,4, sep, 3,4,nl, 5,6, end]
```

---

## 3D Positional Encoding

Each token receives (x, y, z) coordinates:

- **x**: Column position within current grid (0-30)
- **y**: Row position within current grid (0-29)
- **z**: Phase indicator:
  - 0: `<start>` token
  - 1: Input grid tokens
  - 2: `<input_output_separator>`
  - 3: Output grid tokens / transition
  - 4: `<end>` token

These coordinates are used by `RotaryEmbedding3D` to provide position-aware attention.

---

## Task Embeddings (Example Embeddings)

Task embeddings provide the model with task-specific context, allowing it to recognize which ARC puzzle it's solving and adapt its behavior accordingly.

### How It Works

1. **Assignment:** Each unique task ID (e.g., `"00576224"`) is assigned a unique integer `example_id` (0, 1, 2, ..., N-1) during dataset loading:
   ```python
   for example_id, task_id in enumerate(task_ids):
       self.task_id_to_example_id[task_id] = example_id
   ```

2. **Embedding Layer:** The model contains a learnable embedding table:
   ```python
   self.example_embedding = nn.Embedding(num_examples, d_model)
   # num_examples = total unique tasks in dataset (~1280 for ARC-1 + ConceptARC)
   # d_model = embedding dimension (e.g., 768)
   ```

3. **Application:** During forward pass, the task embedding is looked up and **added** to every token embedding in the sequence:
   ```python
   token_embeds = self.token_embedding(input_ids)        # [B, seq_len, D]
   example_embeds = self.example_embedding(example_ids)  # [B, D]
   hidden_states = token_embeds + example_embeds.unsqueeze(1)  # Broadcast add
   ```

### Why Task Embeddings Matter

- **Task Recognition:** The model learns to associate specific patterns with each task, helping it understand the implicit rule being demonstrated
- **Shared Context:** All examples from the same task (train pairs + test pairs) share the same embedding, reinforcing that they follow the same transformation rule
- **Bias Signal:** Acts as a task-specific bias that shifts the model's behavior based on which puzzle it's solving

### Configuration

Task embeddings have their own weight decay parameter for regularization:
```python
"task_embedding_weight_decay": 0.01  # Separate from other weight decay
```

This is typically set lower than the main weight decay to allow task embeddings more freedom to specialize.

### Relationship to Other Embeddings

The model combines three types of information for each token:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| Token Embedding | `[B, seq_len, D]` | What the token is (0-9, special tokens) |
| Task Embedding | `[B, 1, D]` → broadcast | Which task this sequence belongs to |
| 3D RoPE | Applied in attention | Where the token is (x, y, z position) |

The token and task embeddings are summed before being passed through the transformer blocks, while 3D RoPE is applied within the attention mechanism.

---

## Workflow Examples

### Training a Model

```python
import train
from build import build_model_and_data
from pathlib import Path
import argparse

args = {
    "data_path": Path("assets/challenges.json"),
    "save_path": Path("runs/model.pt"),
    "epochs": 100,
    "batch_size": 32,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 4,
    # ... other config
}
cfg = argparse.Namespace(**args)

model, dataset, dataloader, device, data_path = build_model_and_data(cfg)
train.train_model(cfg, model, dataloader, dataset, device, data_path)
```

### Running Evaluation

```python
import evaluate
from pathlib import Path

EVAL_CONFIGS = [
    ("eval_100aug", 100, Path("assets/challenges.json")),
]

eval_results = evaluate.run_evaluation_configs(
    cfg,
    EVAL_CONFIGS,
    eval_batch_size=1300,
    splits=["test"],
    checkpoint_path=Path("runs/model.pt"),
    include_targets=False,
)
```

### Scoring Results

```python
import utils
from pathlib import Path

score = utils.score_arc_submission(
    Path("assets/solutions.json"),
    Path("runs/eval_100aug/submission.json")
)
```

### Visualizing Predictions

```python
import utils
from pathlib import Path

utils.visualize_submissions(
    Path("runs/eval_100aug/submission.json"),
    solutions_file="assets/solutions.json",
    mode="!"  # Compare vs solutions
)
```

---

## ARC Challenge Context

The **ARC (Abstraction and Reasoning Corpus)** consists of puzzle tasks where:

- **Input**: A 2D grid of colors (0-9)
- **Output**: A transformed grid following implicit rules
- **Training examples**: 2-5 input/output pairs demonstrating the rule
- **Test**: Apply the inferred rule to new inputs
- **Evaluation**: Output must exactly match ground truth (pixel-perfect)

This solution:
1. Tokenizes grids into sequences
2. Trains a transformer to predict outputs autoregressively
3. Uses color/geometric augmentations for data efficiency
4. Aggregates predictions via voting across augmented variants

---

## Dependencies

```
torch          # Core deep learning
numpy          # Numerical operations
numba          # JIT compilation for position computation
matplotlib     # Grid visualization
```

---

## Tips for Development

1. **Memory Management**: Use `utils.cleanup_memory(globals())` between training and evaluation to free GPU memory.

2. **Checkpointing**: Set `checkpoint_epochs` to a list for specific epochs or an integer for periodic saving.

3. **Augmentation Tuning**: Higher `max_augments` increases diversity but slows training. 10-100 is typical.

4. **Validation**: Enable `do_validate` and provide `solutions.json` to monitor output loss during training.

5. **Inference Speed**: Larger `eval_batch_size` improves throughput but requires more GPU memory.

6. **Debugging Tasks**: Set `EVAL_TASK_IDS` to specific task IDs for targeted debugging.
