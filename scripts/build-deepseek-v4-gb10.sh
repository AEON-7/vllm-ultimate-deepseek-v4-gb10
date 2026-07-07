#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
IMAGE="${IMAGE:-ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:local-${GIT_SHA}}"

echo "Building ${IMAGE}"
docker build \
  -t "${IMAGE}" \
  -f Dockerfile.deepseek-v4-gb10-layer \
  .

echo
echo "Built: ${IMAGE}"
echo "Smoke import check:"
docker run --rm --entrypoint python3 "${IMAGE}" - <<'PY'
import cutlass
import flashinfer
import deep_gemm
import tilelang
import tvm_ffi
import vllm.models.deepseek_v4
print("imports ok")
print("cutlass", getattr(cutlass, "__version__", "unknown"))
print("flashinfer", getattr(flashinfer, "__version__", "unknown"))
print("tilelang", getattr(tilelang, "__version__", "unknown"))
PY

