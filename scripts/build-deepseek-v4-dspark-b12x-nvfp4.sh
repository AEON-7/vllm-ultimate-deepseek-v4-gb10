#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
BASE_IMAGE="${BASE_IMAGE:-aeon-vllm-ultimate:deepseek-v4-gb10-ab}"
IMAGE="${IMAGE:-ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:local-${GIT_SHA}}"

echo "Building ${IMAGE}"
echo "Base image: ${BASE_IMAGE}"
docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  -t "${IMAGE}" \
  -f Dockerfile.deepseek-v4-dspark-b12x-nvfp4-layer \
  .

echo
echo "Built: ${IMAGE}"
echo "Smoke import check:"
docker run --rm --entrypoint python3 "${IMAGE}" - <<'PY'
import b12x
import cutlass
import flashinfer
import tilelang
from vllm.config.cache import CacheDType
from vllm.v1.kv_cache_interface import get_kv_quant_mode
import vllm.v1.spec_decode.dspark
import vllm.v1.attention.backends.mla.b12x_mla_sparse

assert "nvfp4_ds_mla" in CacheDType.__args__
assert get_kv_quant_mode("nvfp4_ds_mla").name == "NVFP4"
print("imports ok")
print("b12x", getattr(b12x, "__version__", "unknown"))
print("cutlass", getattr(cutlass, "__version__", "unknown"))
print("flashinfer", getattr(flashinfer, "__version__", "unknown"))
print("tilelang", getattr(tilelang, "__version__", "unknown"))
PY
