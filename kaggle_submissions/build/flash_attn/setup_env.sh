#!/usr/bin/env bash
# Run inside the dedicated Linux builder. This does not build FlashAttention.
set -euo pipefail
trap 'printf "Setup failed at line %s. See the setup log.\n" "$LINENO" >&2' ERR

[[ $# == 0 || ( $# == 1 && $1 == --verify-only ) ]] || {
    echo "Usage: setup_env.sh [--verify-only]" >&2; exit 1;
}

[[ "$(uname -sm)" == "Linux x86_64" ]] || {
    echo "This script requires Linux x86_64." >&2; exit 1;
}
[[ "$(id -u)" == "0" ]] || {
    echo "Run as root inside the dedicated builder container." >&2; exit 1;
}
[[ -x /usr/local/cuda/bin/nvcc ]] || {
    echo "CUDA devel toolkit is missing." >&2; exit 1;
}
/usr/local/cuda/bin/nvcc --version | grep -q 'release 12.8,' || {
    echo "Expected CUDA compiler 12.8." >&2; exit 1;
}

FA2_ROOT=/opt/mdlarc-fa2
mkdir -p "$FA2_ROOT"/{sources,logs,out,tmp}
export DEBIAN_FRONTEND=noninteractive
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PYTHONHOME PYTHONPATH

if [[ ${1:-} != --verify-only ]]; then
    FA2_FREE_KIB=$(df -Pk "$FA2_ROOT" | awk 'NR == 2 {print $4}')
    (( FA2_FREE_KIB >= 50 * 1024 * 1024 )) || {
        echo "At least 50 GiB free disk space is required for this setup." >&2; exit 1;
    }
    apt-get update
    apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl xz-utils binutils tmux \
        libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
        libffi-dev liblzma-dev libncurses-dev libgdbm-dev \
        libgdbm-compat-dev uuid-dev

    cd "$FA2_ROOT/sources"
    curl --fail --location --retry 3 \
        https://www.python.org/ftp/python/3.13.15/Python-3.13.15.tar.xz \
        --output Python-3.13.15.tar.xz
    printf '%s  %s\n' \
        1e66a7945a48390ee4c2a4268a0e4185884059a13c4aab6d148aa208deea4a76 \
        Python-3.13.15.tar.xz | sha256sum --check -
    tar -xf Python-3.13.15.tar.xz
    cd Python-3.13.15
    ./configure --prefix="$FA2_ROOT/python-3.13.15" --with-ensurepip=install
    make -j16
    make install

    # Setuptools treats a directory containing pybuilddir.txt as CPython's own
    # build tree and suppresses its bundled distutils there. Leave that tree
    # before installing packages or importing the PyTorch extension helper.
    cd "$FA2_ROOT"
    "$FA2_ROOT/python-3.13.15/bin/python3.13" -m venv "$FA2_ROOT/venv"
    FA2_PYTHON="$FA2_ROOT/venv/bin/python"
    # Bootstrap the patched installer before any other package downloads.
    "$FA2_PYTHON" -m pip install --no-cache-dir --only-binary=:all: \
        "pip @ https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"
    "$FA2_PYTHON" -m pip install --no-cache-dir --only-binary=:all: \
        --index-url https://pypi.org/simple \
        'certifi>=2026.10.19' setuptools==80.9.0 wheel==0.45.1 packaging==25.0 \
        ninja==1.13.0 psutil==7.0.0 einops==0.8.1 numpy==2.3.3
    "$FA2_PYTHON" -m pip install --no-cache-dir --only-binary=:all: \
        --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.11.0+cu128'
fi

cd "$FA2_ROOT"
FA2_PYTHON="$FA2_ROOT/venv/bin/python"
"$FA2_PYTHON" -m pip check

export CUDA_HOME=/usr/local/cuda
"$FA2_PYTHON" - <<'PY'
import bz2, ctypes, hashlib, json, lzma, pathlib, platform, sqlite3
import ssl, subprocess, sys, sysconfig, zlib
import torch
from packaging.tags import sys_tags
from torch.utils.cpp_extension import CUDA_HOME

assert sys.version_info[:3] == (3, 13, 15), sys.version
assert not sysconfig.get_config_var('Py_GIL_DISABLED'), 'Need regular CPython, not cp313t'
assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
assert torch.__version__ == '2.11.0+cu128', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert CUDA_HOME is not None
assert any(str(t) == 'cp313-cp313-linux_x86_64' for t in sys_tags())

root = pathlib.Path('/opt/mdlarc-fa2')
manifest = {
    'stage': 'environment_ready',
    'python': sys.version,
    'python_soabi': sysconfig.get_config_var('SOABI'),
    'python_source_sha256': hashlib.file_digest(
        (root / 'sources/Python-3.13.15.tar.xz').open('rb'), 'sha256'
    ).hexdigest(),
    'torch': torch.__version__,
    'torch_cuda': torch.version.cuda,
    'torch_cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
    'platform': platform.platform(),
    'glibc': platform.libc_ver(),
    'cuda_home': CUDA_HOME,
    'nvcc': subprocess.check_output(['/usr/local/cuda/bin/nvcc', '--version'], text=True),
    'gcc': subprocess.check_output(['gcc', '--version'], text=True).splitlines()[0],
    'cuda_device_available': torch.cuda.is_available(),
    'container_image': 'nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43',
}
(root / 'out/build-environment.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
print('ENVIRONMENT READY')
PY
"$FA2_PYTHON" -m pip freeze > "$FA2_ROOT/out/environment-freeze.txt"
dpkg-query -W > "$FA2_ROOT/out/system-packages.txt"

cat > "$FA2_ROOT/build_env.sh" <<'ENV'
source /opt/mdlarc-fa2/venv/bin/activate
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export BUILD_TARGET=cuda
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export MAX_JOBS=8
export NVCC_THREADS=2
export TMPDIR=/opt/mdlarc-fa2/tmp
ENV

printf 'Setup complete. Build profile: %s/build_env.sh\n' "$FA2_ROOT"
