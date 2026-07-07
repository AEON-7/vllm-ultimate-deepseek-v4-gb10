#!/usr/bin/env bash
set -euo pipefail

: "${IMAGE:?Set IMAGE, e.g. ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:test-001}"
: "${MODEL_DIR:?Set MODEL_DIR, e.g. /models/deepseek-ai/DeepSeek-V4-Flash}"
: "${NODE_RANK:?Set NODE_RANK to 0 on head, 1 on worker}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the head node fabric IP}"
: "${VLLM_HOST_IP:?Set VLLM_HOST_IP to this node fabric IP}"
: "${FABRIC_IF:?Set FABRIC_IF to the RoCE fabric interface}"
: "${NCCL_IB_HCA:?Set NCCL_IB_HCA to the RoCE HCA name}"

NAME="${NAME:-vllm-ds4}"
MASTER_PORT="${MASTER_PORT:-29519}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-$HOME/.cache/flashinfer-deepseek-v4-gb10}"
TRITON_CACHE="${TRITON_CACHE:-$HOME/.cache/triton-deepseek-v4-gb10}"

mkdir -p "${FLASHINFER_CACHE}" "${TRITON_CACHE}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

EXTRA_ARGS=()
if [[ "${NODE_RANK}" != "0" ]]; then
  EXTRA_ARGS+=(--headless)
fi

docker run -d --name "${NAME}" \
  --restart unless-stopped \
  --runtime nvidia --gpus all --ipc host --network host \
  --shm-size 16g --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --ulimit memlock=-1:-1 \
  --device=/dev/infiniband \
  -v "${MODEL_DIR}:${MODEL_DIR}:ro" \
  -v "${FLASHINFER_CACHE}:/root/.cache/flashinfer" \
  -v "${TRITON_CACHE}:/root/.triton" \
  -e VLLM_HOST_IP="${VLLM_HOST_IP}" \
  -e NCCL_IB_HCA="${NCCL_IB_HCA}" \
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}" \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_SOCKET_IFNAME="${FABRIC_IF}" \
  -e GLOO_SOCKET_IFNAME="${FABRIC_IF}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_DEEP_GEMM_WARMUP=skip \
  -e VLLM_RPC_TIMEOUT=3600000 \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  --entrypoint vllm \
  "${IMAGE}" \
  serve "${MODEL_DIR}" \
    --served-model-name DeepSeek-V4-Flash \
    --trust-remote-code \
    --tokenizer-mode deepseek_v4 \
    --tensor-parallel-size 2 \
    --enable-expert-parallel \
    --moe-backend marlin \
    --linear-backend triton \
    --distributed-executor-backend mp \
    --nnodes 2 \
    --node-rank "${NODE_RANK}" \
    --master-addr "${MASTER_ADDR}" \
    --master-port "${MASTER_PORT}" \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --enable-prefix-caching \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --no-enable-flashinfer-autotune \
    --compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}' \
    --reasoning-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --load-format safetensors \
    --host 0.0.0.0 \
    --port "${PORT}" \
    "${EXTRA_ARGS[@]}"

echo "Started ${NAME} on node rank ${NODE_RANK}"
echo "Logs: docker logs -f ${NAME}"

