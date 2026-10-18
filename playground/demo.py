# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import sys

os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
os.environ["HUMMING_DISABLE_PARALLEL_BUILD"] = "1"

venv_bin = os.path.dirname(sys.executable)
os.environ["PATH"] = f"{venv_bin}:{os.environ.get('PATH', '')}"

driver_lib = "/usr/lib/wsl/lib"
cuda_lib = "/usr/local/cuda/lib64"
library_path = f"{driver_lib}:{cuda_lib}"
if existing_library_path := os.environ.get("LIBRARY_PATH"):
    library_path = f"{library_path}:{existing_library_path}"
os.environ["LIBRARY_PATH"] = library_path

from vllm import LLM, SamplingParams  # noqa: E402

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)


if __name__ == "__main__":
    main()
