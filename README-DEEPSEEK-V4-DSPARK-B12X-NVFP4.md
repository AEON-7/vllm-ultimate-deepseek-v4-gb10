# DeepSeek-V4 DSpark + B12X + NVFP4 MLA on DGX Spark

Experimental tester image for DeepSeek-V4-Flash on 2x DGX Spark.

This is the next layer after the PR #4 GB10 compatibility image. It adds the
three missing pieces called out in the TP2 review:

1. `b12x==0.30.0`, installed with `--no-deps` so Torch, Triton, CUDA, and the
   base runtime are not replaced.
2. B12X MoE/MLA overlays for native MXFP4 DeepSeek-V4 weights on SM120/GB10.
3. `nvfp4_ds_mla` KV-cache support using the DeepSeek-V4 584-byte sparse-MLA
   cache envelope.

Status: build and import smoke validated on one DGX Spark. Full runtime
validation requires a 2x Spark rig with the DeepSeek-V4-Flash checkpoint and
RoCE fabric.

## Why This Is Separate

The Qwen/Gemma A/B test did not show PR #4 as a general performance win:

| Image / graph mode | TTFT p50 | TPOT p50 | Decode p50 | c=4 aggregate p50 |
|---|---:|---:|---:|---:|
| Current Qwen default, FULL_AND_PIECEWISE | 308 ms | 45.77 ms | 22.09 tok/s | 53.21 tok/s |
| Current Qwen + PIECEWISE | 340 ms | 42.16 ms | 24.03 tok/s | 45.10 tok/s |
| PR #4 layer, FULL_AND_PIECEWISE | 334 ms | 42.43 ms | 24.02 tok/s | 46.10 tok/s |
| PR #4 layer + PIECEWISE | 311 ms | 46.97 ms | 21.60 tok/s | 45.78 tok/s |

Conclusion: keep Qwen/Gemma on the current default image and graph policy.
Use this repo only for DeepSeek-V4 TP2/DSpark experiments.

## Build

First build or pull the PR #4 base layer:

```bash
IMAGE=aeon-vllm-ultimate:deepseek-v4-gb10-ab \
  ./scripts/build-deepseek-v4-gb10.sh
```

Then build the DSpark/B12X/NVFP4 layer:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:test-001 \
BASE_IMAGE=aeon-vllm-ultimate:deepseek-v4-gb10-ab \
  ./scripts/build-deepseek-v4-dspark-b12x-nvfp4.sh
```

Local validation performed on Spark:

```text
nvfp4_ds_mla True
mode NVFP4
B12X_MXFP4 True
b12x_mhc False
woproj False
rocm_hipbmm_attr True
dspark vllm.v1.spec_decode.dspark
proposer vllm.v1.spec_decode.dspark_proposer
b12x_mla vllm.v1.attention.backends.mla.b12x_mla_sparse
```

## Serve TP2

Run on both Sparks. Head node:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:test-001 \
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
NODE_RANK=0 \
MASTER_ADDR=<head-fabric-ip> \
VLLM_HOST_IP=<head-fabric-ip> \
FABRIC_IF=<fabric-interface> \
NCCL_IB_HCA=<roce-hca> \
./scripts/serve-deepseek-v4-dspark-b12x-tp2.sh
```

Worker node:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:test-001 \
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
NODE_RANK=1 \
MASTER_ADDR=<head-fabric-ip> \
VLLM_HOST_IP=<worker-fabric-ip> \
FABRIC_IF=<fabric-interface> \
NCCL_IB_HCA=<roce-hca> \
./scripts/serve-deepseek-v4-dspark-b12x-tp2.sh
```

The script uses:

```text
--tensor-parallel-size 2
--enable-expert-parallel
--moe-backend flashinfer_b12x
--linear-backend triton
--kv-cache-dtype nvfp4_ds_mla
--block-size 256
--enable-prefix-caching
--async-scheduling
--enable-chunked-prefill
--max-model-len 1048576
--max-num-seqs 6
--max-num-batched-tokens 8192
--max-cudagraph-capture-size max_num_seqs * (MTP_NUM_TOKENS + 1)
--compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}'
--speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
```

Keep `PIECEWISE` for TP2 DeepSeek. PR testing showed FULL decode graph replay
can capture/desync NCCL collectives between the two ranks.

## Critical Environment

The serve script sets:

```text
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1
VLLM_DSV4_B12X_COMPRESSED_MLA=0
MTP_NUM_TOKENS=3
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_DEEP_GEMM_WARMUP=skip
```

Do not install `b12x` without `--no-deps`. Its package metadata can pull a
different Torch/Triton stack, which would invalidate this container.

## What To Test

Minimum tester report:

| Test | Required output |
|---|---|
| Build | Image builds from clean checkout |
| Import smoke | `b12x`, `dspark`, `B12X_MXFP4`, `nvfp4_ds_mla` all present |
| Launch | Both TP2 ranks initialize without NCCL timeout |
| c=1 | Natural language, coding, math, reasoning, prose |
| c=4/c=8 | Same categories, no engine errors |
| Long context | 9K+ prompt with 800+ token output |
| Soak | 30-60 minutes PIECEWISE |
| Negative check | FULL graph replay either reproduces or clears the PR's NCCL issue |

Capture TTFT p50/p95, TPOT p50/p95, median decode tok/s, peak decode tok/s,
aggregate tok/s, error count, GPU temperature, exact image tag, exact model
revision, NCCL settings, and fabric topology.

## Caveats

- Experimental. Not a default community image.
- Designed for DeepSeek-V4-Flash on 2x DGX Spark, not Qwen/Gemma.
- Full runtime quality and speed still need independent 2-node validation.
- The image deliberately preserves the base `envs.py` and adds DSpark knobs;
  it does not wholesale replace `envs.py` from the upstream DSpark overlay.
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0` is the conservative default. Enable the
  compressed MLA path only as a separate A/B after the base path is stable.
