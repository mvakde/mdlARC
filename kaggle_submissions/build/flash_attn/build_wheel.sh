#!/usr/bin/env bash
# Build on the dedicated Linux CPU builder after setup_env.sh succeeds.
set -euo pipefail
[[ "$(uname -sm)" == "Linux x86_64" ]] || {
    echo "This script requires Linux x86_64." >&2; exit 1;
}
FA2_ROOT=/opt/mdlarc-fa2
source "$FA2_ROOT/build_env.sh"
cd "$FA2_ROOT"
[[ -f "$FA2_ROOT/out/build-environment.json" ]] || {
    echo "Run setup_env.sh first." >&2; exit 1;
}
[[ ! -e "$FA2_ROOT/logs/build.started" ]] || {
    echo "A build was already submitted. Inspect its log/exitcode before retrying." >&2; exit 1;
}
date -u +%FT%TZ > "$FA2_ROOT/logs/build.started"
trap 'FA2_EXIT=$?; printf "%s\n" "$FA2_EXIT" > "$FA2_ROOT/logs/build.exitcode"; exit "$FA2_EXIT"' EXIT
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PYTHONHOME PYTHONPATH
export FLASH_ATTN_LOCAL_VERSION=cu128torch2.11cxx11abitrue.sm80

python - <<'PY'
import os, platform, sys, torch
assert sys.version_info[:3] == (3, 13, 15)
assert platform.machine() == 'x86_64'
assert torch.__version__ == '2.11.0+cu128'
assert torch.version.cuda == '12.8'
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert os.environ['FLASH_ATTN_CUDA_ARCHS'] == '80'
assert os.environ['MAX_JOBS'] == '8'
assert os.environ['NVCC_THREADS'] == '2'
print('Build environment verified; targeting sm_80 for Ampere/Ada.')
PY

cd "$FA2_ROOT/sources"
curl --fail --location --retry 3 \
    https://files.pythonhosted.org/packages/01/7a/92a46e7cd6bbb4d7b2855a457c3b855df54a97af5656d98fc92e58e61065/flash_attn-2.8.3.post1.tar.gz \
    --output flash_attn-2.8.3.post1.tar.gz
printf '%s  %s\n' \
    55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348 \
    flash_attn-2.8.3.post1.tar.gz | sha256sum --check -
tar -xf flash_attn-2.8.3.post1.tar.gz
python -m pip wheel --no-build-isolation --no-deps --no-index \
    --wheel-dir "$FA2_ROOT/out" ./flash_attn-2.8.3.post1

cd "$FA2_ROOT"
FA2_WHEELS=("$FA2_ROOT"/out/flash_attn-2.8.3.post1+cu128torch2.11cxx11abitrue.sm80-cp313-cp313-linux_x86_64.whl)
[[ -f ${FA2_WHEELS[0]} ]] || { echo "Expected wheel is missing." >&2; exit 1; }
python -m pip install --no-index --no-deps --force-reinstall "${FA2_WHEELS[0]}"
python -m pip check

python - <<'PY'
import email, hashlib, importlib.metadata, json, pathlib, re, subprocess, zipfile
import torch
import flash_attn
from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from packaging.utils import parse_wheel_filename

root = pathlib.Path('/opt/mdlarc-fa2')
out = root / 'out'
wheels = list(out.glob('flash_attn-2.8.3.post1+cu128torch2.11cxx11abitrue.sm80-cp313-cp313-linux_x86_64.whl'))
assert len(wheels) == 1
wheel = wheels[0]
name, version, _, tags = parse_wheel_filename(wheel.name)
assert {str(t) for t in tags} == {'cp313-cp313-linux_x86_64'}
assert str(version) == importlib.metadata.version('flash_attn')
assert all(callable(f) for f in (
    flash_attn_varlen_func, flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
))

audit = root / 'audit'
audit.mkdir(exist_ok=True)
with zipfile.ZipFile(wheel) as z:
    wheel_info = z.read(next(n for n in z.namelist() if n.endswith('.dist-info/WHEEL'))).decode()
    assert 'Tag: cp313-cp313-linux_x86_64' in wheel_info
    metadata = email.message_from_bytes(z.read(next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))))
    extensions = [n for n in z.namelist() if n.endswith('.so')]
    assert len(extensions) == 1
    extension = audit / pathlib.Path(extensions[0]).name
    extension.write_bytes(z.read(extensions[0]))

dynamic = subprocess.check_output(['readelf', '-dW', str(extension)], text=True)
needed = re.findall(r'Shared library: \[(.*?)\]', dynamic)
assert 'libcudart.so.12' in needed, needed
assert 'libcudart.so.13' not in needed, needed
versions = subprocess.check_output(['readelf', '--version-info', str(extension)], text=True)
symbols = sorted(set(re.findall(r'Name: ((?:GLIBCXX|GLIBC|CXXABI)_[0-9.]+)', versions)))
elfs = subprocess.check_output(['/usr/local/cuda/bin/cuobjdump', '--list-elf', str(extension)], text=True)
assert 'sm_80' in elfs
(out / 'elf-needed.txt').write_text(dynamic)
(out / 'elf-symbol-versions.txt').write_text(versions)
(out / 'cuda-elf-list.txt').write_text(elfs)

digest = hashlib.file_digest(wheel.open('rb'), 'sha256').hexdigest()
(out / (wheel.name + '.sha256')).write_text(digest + '  ' + wheel.name + '\n')
manifest = json.loads((out / 'build-environment.json').read_text())
manifest.update({
    'stage': 'wheel_built_cpu_import_passed',
    'flash_attn_module_version': flash_attn.__version__,
    'flash_attn_distribution_version': str(version),
    'wheel': wheel.name,
    'wheel_bytes': wheel.stat().st_size,
    'wheel_sha256': digest,
    'wheel_tags': sorted(str(t) for t in tags),
    'wheel_requires_dist': metadata.get_all('Requires-Dist', []),
    'flash_attn_source_sha256': '55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348',
    'flash_attn_setup_sha256': hashlib.file_digest((root / 'sources/flash_attn-2.8.3.post1/setup.py').open('rb'), 'sha256').hexdigest(),
    'max_jobs': 8,
    'nvcc_threads': 2,
    'flash_attn_cuda_archs': ['80'],
    'elf_needed': needed,
    'elf_symbol_versions': symbols,
    'cpu_import_passed': True,
    'gpu_numerical_validation_passed': False,
})
(out / 'wheel-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
print('WHEEL BUILT; CPU IMPORT AND LINKAGE CHECKS PASSED. GPU VALIDATION STILL REQUIRED.')
PY
python -m pip freeze > "$FA2_ROOT/out/environment-freeze.txt"
