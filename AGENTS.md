# AGENTS.md — DeepSeek-V4 GB10 Experimental vLLM Image

This repository is for the experimental DeepSeek-V4 / DeepSeek-V4-Flash GB10 TP=2 image:

```text
ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:experimental-pr4
```

Do not confuse it with the production Qwen/Gemma image:

```text
ghcr.io/aeon-7/aeon-vllm-ultimate:latest
```

## Agent Operating Rules

- Treat this repo as experimental.
- Do not promote B12X / `nvfp4_ds_mla` images unless they pass boot, quality, and benchmark validation.
- Do not merge DeepSeek-specific overlays into the general AEON image without a Qwen/Gemma non-regression pass.
- Stop any running vLLM containers before building or benchmarking on a DGX Spark.
- Preserve `/root/.cache/flashinfer` and `/root/.triton` between runs where possible; first-run JIT can be long.
- Use `PIECEWISE` cudagraph mode for TP=2 DeepSeek tests.
- Do not use FULL decode graph replay unless the explicit task is to reproduce/debug NCCL desync.

## Recommended Image

Use:

```bash
docker pull ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:experimental-pr4
```

Avoid:

```text
ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:*
```

unless the task is specifically to fix the B12X/DSpark lane. The current known blocker is a mismatched DeepSeek overlay where `common/ops/__init__.py` imports `build_flashinfer_mixed_sparse_indices` but the packaged `cache_utils.py` does not define it.

## TP=2 Launch

Use `scripts/serve-deepseek-v4-gb10-tp2.sh`.

Head node:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:experimental-pr4 \
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
NODE_RANK=0 \
MASTER_ADDR=<head-fabric-ip> \
VLLM_HOST_IP=<head-fabric-ip> \
FABRIC_IF=<fabric-interface> \
NCCL_IB_HCA=<roce-hca> \
./scripts/serve-deepseek-v4-gb10-tp2.sh
```

Worker node:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:experimental-pr4 \
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
NODE_RANK=1 \
MASTER_ADDR=<head-fabric-ip> \
VLLM_HOST_IP=<worker-fabric-ip> \
FABRIC_IF=<fabric-interface> \
NCCL_IB_HCA=<roce-hca> \
./scripts/serve-deepseek-v4-gb10-tp2.sh
```

## Validation Checklist

After launch, verify:

- `docker logs -f vllm-ds4` reaches server startup.
- `curl http://127.0.0.1:8000/v1/models` returns `DeepSeek-V4-Flash`.
- `python3 scripts/validate-endpoint.py --base-url http://127.0.0.1:8000/v1` succeeds.
- No NCCL timeout.
- No `EngineCore encountered an issue`.
- Output is coherent.

## Benchmark Requirements

For any benchmark report, capture:

- Image tag and digest.
- Repo commit.
- Model checkpoint and revision.
- Node count and fabric settings.
- Prompt categories: coding, math, reasoning, prose, summary, dialogue.
- TTFT p50/p95.
- TPOT p50/p95.
- Median decode tok/s.
- Peak decode tok/s.
- Aggregate tok/s.
- Request error count.
- GPU temperature and power if available.

## Current Known Numbers

Single DGX Spark Qwen compatibility smoke, not a DeepSeek TP2 benchmark:

| Image | Single TTFT | Single TPOT | Single Decode | c=4 Aggregate |
|---|---:|---:|---:|---:|
| Current AEON vLLM | 331.1 ms | 39.54 ms | 25.30 tok/s | 63.87 tok/s |
| DeepSeek PR4 image | 328.1 ms | 37.50 ms | 26.67 tok/s | 68.49 tok/s |

DeepSeek TP2 PR-layer validation:

| Test | Result |
|---|---:|
| Single stream | 16.6 tok/s |
| 10 concurrent multi-turn | 66.4 aggregate tok/s |
| Benchmark turns | 48/48 |
| Engine errors | 0 |

## Future Universal Image

The long-term target is one universal `aeon-vllm-ultimate` package with:

- common TP/runtime stability fixes,
- Qwen/Gemma hot path preserved,
- DeepSeek sparse-MLA overlays isolated,
- optional DSpark support,
- optional B12X MXFP4 MoE path after validation.

Until that exists, keep DeepSeek work in this dedicated repository.

