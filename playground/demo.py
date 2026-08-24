# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)




def main():
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    outputs = llm.generate(prompts, sampling_params)


if __name__ == "__main__":
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] =  "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    main()
