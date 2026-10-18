#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_debug_env.sh [--build-native] [--help]

Configure Linux NVIDIA GPU debugging in the repository's .venv (Python 3.12).
Default: editable Python sources with precompiled native extensions.
--build-native: compile C++/CUDA extensions with CMAKE_BUILD_TYPE=Debug;
                requires a CUDA toolkit and a working native build toolchain.

Optional environment variables:
  VLLM_PRECOMPILED_WHEEL_VARIANT   e.g. cu129 (otherwise auto-detected)
  VLLM_PRECOMPILED_WHEEL_LOCATION explicit matching wheel URL or local path
  MAX_JOBS                       limit native compilation parallelism

After setup: source .venv/debug-activate
Then: vllm serve <MODEL> --enforce-eager
For VS Code, install the Python and Python Debugger extensions. The script
creates .vscode/launch.json only if it does not already exist.
EOF
}

build_native=0
for arg in "$@"; do
    case "$arg" in
        --build-native) build_native=1 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
fail() { echo "[debug setup] $*" >&2; exit 1; }
trap 'echo "[debug setup] Failed at line $LINENO; fix the error above and rerun." >&2' ERR

[[ "$(uname -s)" == Linux ]] || fail "This script requires Linux."
command -v uv >/dev/null || fail "Install uv first: https://docs.astral.sh/uv/"
command -v nvidia-smi >/dev/null || fail "Install a working NVIDIA driver first."
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if (( build_native )); then
    command -v nvcc >/dev/null || fail "--build-native requires nvcc on PATH."
fi

if [[ ! -e .venv ]]; then
    uv venv --python 3.12 .venv
fi
[[ -x .venv/bin/python ]] || fail ".venv exists but has no usable Python."
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use a Python 3.12 .venv"'
uv pip install --python .venv/bin/python -r requirements/lint.txt debugpy
.venv/bin/pre-commit install

if (( build_native )); then
    VLLM_USE_PRECOMPILED=0 VLLM_USE_PRECOMPILED_RUST=0 CMAKE_BUILD_TYPE=Debug \
        uv pip install --python .venv/bin/python -e . --torch-backend=auto
else
    VLLM_USE_PRECOMPILED=1 \
        VLLM_PRECOMPILED_WHEEL_COMMIT=6cf7b26bd4bff60bf378e1af14044280ac0d214c \
        uv pip install --python .venv/bin/python -e . --torch-backend=auto
fi

.venv/bin/python - <<'PY'
import pathlib
import torch
import vllm
import vllm._C_stable_libtorch

assert torch.cuda.is_available(), "PyTorch cannot access the GPU"
assert pathlib.Path(vllm.__file__).resolve().parent == pathlib.Path.cwd() / "vllm"
print("vLLM:", vllm.__version__, "source:", vllm.__file__)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

cat > .venv/debug-activate <<'EOF'
# Source this file from Bash to activate vLLM Python debugging.
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/bin/activate"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_LOGGING_LEVEL=DEBUG
EOF

mkdir -p .vscode
if [[ ! -e .vscode/launch.json ]]; then
    # noclobber also protects against a file created after the existence check.
    (
        set -o noclobber
        cat > .vscode/launch.json <<'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "vLLM: debug current Python file",
            "type": "debugpy",
            "request": "launch",
            "python": "${workspaceFolder}/.venv/bin/python",
            "program": "${file}",
            "cwd": "${workspaceFolder}",
            "console": "integratedTerminal",
            "justMyCode": false,
            "subProcess": true,
            "env": {
                "PATH": "${workspaceFolder}/.venv/bin:${env:PATH}",
                "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                "VLLM_LOGGING_LEVEL": "DEBUG"
            }
        }
    ]
}
EOF
    )
else
    echo "[debug setup] Keeping existing .vscode/launch.json."
fi

echo "[debug setup] Ready. Run: source '$repo_root/.venv/debug-activate'"
echo "[debug setup] Use --enforce-eager with vllm serve, or enforce_eager=True with LLM."
echo "[debug setup] Native builds use Debug symbols; VS Code launch config debugs Python."
