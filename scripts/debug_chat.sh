#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash scripts/debug_chat.sh

Send a non-streaming chat request to a running vLLM server.
Optional environment variables: BASE_URL (default: http://127.0.0.1:8000),
VLLM_API_KEY (if the server requires authentication).

Start the server in another terminal from the repository root:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 .venv/bin/python -m \
    vllm.entrypoints.openai.api_server --model Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 8000 --enforce-eager

For debugging, launch that module with your Python debugger and set a
breakpoint in vllm/entrypoints/openai/chat_completion/api_router.py,
in create_chat_completion(). Then run this script.
If using --served-model-name, update the model field below to match.
EOF
    exit 0
fi

headers=(-H 'Content-Type: application/json')
if [[ -n "${VLLM_API_KEY:-}" ]]; then
    headers+=(-H "Authorization: Bearer ${VLLM_API_KEY}")
fi

curl --silent --show-error --fail-with-body \
    "${BASE_URL:-http://127.0.0.1:8000}/v1/chat/completions" \
    "${headers[@]}" \
    --data-binary @- <<'JSON'
{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [
    {"role": "user", "content": "你好，请用一句话介绍你自己。"}
  ],
  "temperature": 0,
  "max_tokens": 128,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}
JSON
printf '\n'
