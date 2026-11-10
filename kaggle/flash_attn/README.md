# FlashAttention 2 for the Kaggle L4 environment

These scripts build FlashAttention `2.8.3.post1` for regular CPython 3.13.15,
PyTorch `2.11.0+cu128`, CUDA 12.8, Linux x86_64, and the C++11 ABI. The build
targets `sm_80`; NVIDIA documents Ampere cubin compatibility with Ada GPUs,
including the L4. This is a target-specific wheel, not a general wheel for
other Python, PyTorch, or CUDA versions.

## Dedicated CPU builder

Use Ubuntu 24.04 with the CUDA 12.8 development toolkit and GCC 13. The
validated builder has 32 CPUs, a 128 GB container memory limit, and a 100 GB
container disk. A GPU is unnecessary for compilation and CPU import checks.
The scripts use `/opt/mdlarc-fa2`; that directory must have at least 50 GiB
free when starting setup.

Copy `setup_env.sh` and `build_wheel.sh` to `/opt/mdlarc-fa2/`. Run setup and
build in separate tmux sessions, preserving logs:

```bash
mkdir -p /opt/mdlarc-fa2/logs
tmux new-session -d -s fa2-setup 'bash /opt/mdlarc-fa2/setup_env.sh > /opt/mdlarc-fa2/logs/setup.log 2>&1'
```

After setup exits successfully and prints `ENVIRONMENT READY`:

```bash
tmux new-session -d -s fa2-build 'bash /opt/mdlarc-fa2/build_wheel.sh > /opt/mdlarc-fa2/logs/build.log 2>&1'
```

The builder uses eight compiler jobs and two NVCC threads per job. Inspect
`logs/build.exitcode` and `logs/build.log` before considering any retry. A
quiet pip log can occur while CUDA compilation continues. The build script
refuses to submit a second build when `logs/build.started` exists.

Setup changes out of the CPython source directory before importing
setuptools. This avoids its `pybuilddir.txt` special case, which can disable
the bundled distutils. `bash setup_env.sh --verify-only` checks an existing
environment without compiling or reinstalling packages.

Successful output is in `/opt/mdlarc-fa2/out/`. The wheel name is:

```text
flash_attn-2.8.3.post1+cu128torch2.11cxx11abitrue.sm80-cp313-cp313-linux_x86_64.whl
```

The accompanying manifest records its checksum, compiler and runtime
versions, ELF dependencies, GPU architecture, and CPU import result. Success
requires `libcudart.so.12` and excludes `libcudart.so.13`. GPU numerical
validation remains a separate step.

## Private Kaggle attachment and GPU validation

Upload the wheel, `wheel-manifest.json`, `validate_on_kaggle.py`, and the
small `einops-0.8.1-py3-none-any.whl` dependency as a private Kaggle Dataset.
Include the supplied license files. Attach that Dataset to the notebook,
select an L4 accelerator, and leave Internet off.

For `submit_arc2.ipynb`, its environment cell already installs the selected
wheel and checks the import. Set `FLASH_ATTN_WHEEL` to the attached wheel path.
The standalone installation below is unnecessary for that notebook.

The attached filename may lose the `+` that separates the local version.
For example, `2.8.3.post1cu128torch2.11cxx11abitrue.sm80` is invalid; the
wheel's actual metadata version is
`2.8.3.post1+cu128torch2.11cxx11abitrue.sm80`. If this happens, copy the file
to `/kaggle/working` using the original name from the manifest. Put this
cell after the settings cell and before the environment cell, retaining it
in the saved notebook:

```python
from pathlib import Path
import hashlib
import json
import shutil

FA2_DATASET = Path('/kaggle/input/datasets/mithilavakde/flash-attn-cp313-torch211-cu128')
manifest = json.loads((FA2_DATASET / 'wheel-manifest.json').read_text())
sources = list(FA2_DATASET.glob('flash_attn-*.whl'))
assert len(sources) == 1, sources
with sources[0].open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['wheel_sha256']
wheel = Path('/kaggle/working') / manifest['wheel']
shutil.copyfile(sources[0], wheel)
FLASH_ATTN_WHEEL = str(wheel)
```

The optional one-time GPU numerical check can then run after the environment
cell, before loading the model:

```python
subprocess.run([sys.executable, str(FA2_DATASET / 'validate_on_kaggle.py')], check=True)
```

For a separate validation notebook, use the full standalone cell below.
Replace the Dataset folder with the actual attached path shown in Kaggle's
Input panel:

```python
from pathlib import Path
import hashlib
import json
import subprocess
import sys

FA2_DATASET = Path('/kaggle/input/datasets/mithilavakde/flash-attn-cp313-torch211-cu128')
manifest = json.loads((FA2_DATASET / 'wheel-manifest.json').read_text())
sources = list(FA2_DATASET.glob('flash_attn-*.whl'))
assert len(sources) == 1, sources
with sources[0].open('rb') as stream:
    assert hashlib.file_digest(stream, 'sha256').hexdigest() == manifest['wheel_sha256']
import shutil
wheel = Path('/kaggle/working') / manifest['wheel']
shutil.copyfile(sources[0], wheel)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
    str(FA2_DATASET / 'einops-0.8.1-py3-none-any.whl'),
], check=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
    '--force-reinstall', str(wheel),
], check=True)
subprocess.run([sys.executable, str(FA2_DATASET / 'validate_on_kaggle.py')], check=True)
```

The fresh validation process checks the exact target environment and L4,
then compares variable-length attention output and gradients with a float32
reference for fp16/bf16, causal attention, and grouped query attention. It
also checks dropout forward/backward for finite results. It loads no model
checkpoint and performs no model training. A successful run writes
`/kaggle/working/flash_attn_gpu_validation.json` and prints
`FLASHATTENTION L4 FORWARD/BACKWARD VALIDATION PASSED`.

In the submission notebook, set `ATTENTION_BACKEND = 'flash'` and
`FLASH_ATTN_WHEEL` to the exact attached wheel path. Restart the notebook
session first if it already imported a different FlashAttention extension.

Sources: [pinned FlashAttention build configuration](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post1/setup.py),
[NVIDIA Ampere/Ada compatibility](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html#compatibility-between-ampere-and-ada).
