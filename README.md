# AEON DeepSeek-V4-Flash GB10 Private Tester Build

Private experimental repo for trusted multi-rig testers validating DeepSeek-V4-Flash on DGX Spark GB10 / sm_121a.

This repository is intentionally separate from the public `aeon-vllm-ultimate` Qwen/Gemma build path. It carries the PR #4 DeepSeek-V4 enablement layer in isolation so we can test it without changing the default community image.

## Status

Experimental. Do not treat this as a production default yet.

There are now two tester tracks:

| Track | Image family | Purpose | Status |
|---|---|---|---|
| PR #4 GB10 compatibility | `vllm-ultimate-deepseek-v4-gb10` | Make DeepSeek-V4-Flash boot on GB10 with stock Cutlass 4.6.0, DeepGEMM sm120, FlashInfer 0.6.14, and PIECEWISE TP2 safety | Build validated; PR author reports 66.4 aggregate tok/s on 2x Spark |
| DSpark + B12X + NVFP4 MLA | `vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4` | Add `b12x==0.30.0`, native MXFP4 B12X MoE path, DSpark proposer, and `nvfp4_ds_mla` KV cache | Build/import smoke validated on Spark; needs 2-node tester runtime validation |

Important A/B result: PR #4 is not a Qwen/Gemma default upgrade. On Qwen3.6
DFlash fast-path, the current default image remains better at c=4 aggregate
throughput than the PR #4 layer. Keep this repo isolated for DeepSeek-V4 TP2.

| Qwen3.6 A/B | TTFT p50 | TPOT p50 | Decode p50 | c=4 aggregate p50 |
|---|---:|---:|---:|---:|
| Current default, FULL_AND_PIECEWISE | 308 ms | 45.77 ms | 22.09 tok/s | 53.21 tok/s |
| Current default, PIECEWISE | 340 ms | 42.16 ms | 24.03 tok/s | 45.10 tok/s |
| PR #4 layer, FULL_AND_PIECEWISE | 334 ms | 42.43 ms | 24.02 tok/s | 46.10 tok/s |
| PR #4 layer, PIECEWISE | 311 ms | 46.97 ms | 21.60 tok/s | 45.78 tok/s |

Raw A/B harness and JSON results are in `benchmarks/qwen36-pr4-ab/`.

The layer is designed for:

- 2x DGX Spark
- GB10 / sm_121a
- TP2 plus expert parallel
- RoCE fabric
- DeepSeek-V4-Flash fp8 checkpoint
- vLLM base image `ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0`

The PR author reports:

- 66.4 aggregate tok/s over 10 concurrent multi-turn conversations
- 16.6 tok/s single-stream
- 48/48 benchmark turns completed
- zero engine errors under PIECEWISE cudagraph mode
- known NCCL desync with FULL decode graph replay on 2-node TP2

We need trusted testers to reproduce or falsify those numbers on independent multi-Spark rigs.

## What This Adds

The DeepSeek layer lives in:

- `Dockerfile.deepseek-v4-gb10-layer`
- `overlay-deepseek-v4-gb10/`
- `thrmma_shim.py`
- `README-DEEPSEEK-V4-GB10.md`

The newer DSpark/B12X/NVFP4 layer lives in:

- `Dockerfile.deepseek-v4-dspark-b12x-nvfp4-layer`
- `overlay-dspark-b12x-nvfp4/`
- `patches/official-main-b12x-nvfp4-python.patch`
- `patches/apply-dspark-envs.py`
- `patches/apply-dspark-stage-c.py`
- `scripts/build-deepseek-v4-dspark-b12x-nvfp4.sh`
- `scripts/serve-deepseek-v4-dspark-b12x-tp2.sh`
- `README-DEEPSEEK-V4-DSPARK-B12X-NVFP4.md`

It keeps the base image's stock `nvidia-cutlass-dsl 4.6.0` and layers in:

- `ThrMma` / `TiledMma` compatibility shim for `cutlass.cute.core`
- DeepGEMM `nv_dev` sm120 kernels
- FlashInfer 0.6.14 sparse-MLA API
- `apache-tvm-ffi==0.1.11` plus `tilelang==0.1.11`
- DeepSeek-V4 `o_proj` sm12x adapter
- sm12x route to `persistent_topk`
- Triton E8M0 scale upcast
- CUTE `nvvm.fmax` feature detection overlay

See [README-DEEPSEEK-V4-GB10.md](README-DEEPSEEK-V4-GB10.md) for the original technical breakdown.
See [README-DEEPSEEK-V4-DSPARK-B12X-NVFP4.md](README-DEEPSEEK-V4-DSPARK-B12X-NVFP4.md) for the B12X/NVFP4 DSpark track.

## Build

Build on a DGX Spark or another arm64 CUDA-capable build host.

```bash
git clone https://github.com/AEON-7/vllm-ultimate-deepseek-v4-gb10.git
cd vllm-ultimate-deepseek-v4-gb10

./scripts/build-deepseek-v4-gb10.sh
```

Default output tag:

```text
ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:local-<git-sha>
```

Override it when needed:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:test-001 \
  ./scripts/build-deepseek-v4-gb10.sh
```

Build the DSpark/B12X/NVFP4 experimental image:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:test-001 \
BASE_IMAGE=aeon-vllm-ultimate:deepseek-v4-gb10-ab \
  ./scripts/build-deepseek-v4-dspark-b12x-nvfp4.sh
```

## Push To GHCR

Login first:

```bash
echo "$GHCR_PAT" | docker login ghcr.io -u AEON-7 --password-stdin
```

Then push the image:

```bash
./scripts/push-ghcr.sh ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:test-001
./scripts/push-ghcr.sh ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-dspark-b12x-nvfp4:test-001
```

Keep GHCR package visibility private unless/until this build graduates from experimental.

## Launch TP2

Run this on both Sparks after the image is available locally or pullable from GHCR.

Head node:

```bash
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:test-001 \
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
IMAGE=ghcr.io/aeon-7/vllm-ultimate-deepseek-v4-gb10:test-001 \
MODEL_DIR=/models/deepseek-ai/DeepSeek-V4-Flash \
NODE_RANK=1 \
MASTER_ADDR=<head-fabric-ip> \
VLLM_HOST_IP=<worker-fabric-ip> \
FABRIC_IF=<fabric-interface> \
NCCL_IB_HCA=<roce-hca> \
./scripts/serve-deepseek-v4-gb10-tp2.sh
```

The script uses the currently recommended PR profile:

- `--tensor-parallel-size 2`
- `--enable-expert-parallel`
- `--moe-backend marlin`
- `--linear-backend triton`
- `--kv-cache-dtype fp8`
- `--max-model-len 65536`
- `--max-num-seqs 8`
- `--max-num-batched-tokens 4096`
- `--gpu-memory-utilization 0.80`
- `--compilation-config '{"cudagraph_mode":"PIECEWISE","custom_ops":["all"]}'`

FULL decode graph replay is intentionally not used because the PR documents NCCL desync under FULL graph replay on TP2.

## Smoke Test

After the server is live on the head node:

```bash
python3 scripts/validate-endpoint.py --base-url http://127.0.0.1:8000/v1
```

Expected:

- `/v1/models` returns `DeepSeek-V4-Flash`
- short chat completes
- no engine timeout
- no NCCL error
- no output garbling

## Validation Matrix

For testers, a useful report includes all of the following:

| Test | Target |
|---|---|
| Build | Image builds from clean checkout |
| Import smoke | `cutlass`, `deep_gemm`, `flashinfer`, `tilelang`, `vllm.models.deepseek_v4` import cleanly |
| Health | `/v1/models` returns model metadata |
| c=1 | natural language, coding, math, reasoning, prose |
| c=2,4,8 | same prompt categories, no engine errors |
| Long context | 9K+ prompt with 800+ token output |
| Multi-turn | 10 concurrent conversations if the rig can sustain it |
| Stability | 30-60 minute PIECEWISE soak |
| Negative check | FULL decode graph replay reproduces or does not reproduce NCCL issue |

Capture:

- TTFT p50/p95
- TPOT p50/p95
- median decode tok/s
- peak decode tok/s
- aggregate tok/s
- completion/error count
- GPU temp and power
- exact image tag and git commit
- exact fabric settings

## Tester Report Template

```text
Tester:
Hardware:
OS / driver:
CUDA runtime:
Image tag:
Repo commit:
Model checkpoint:
Fabric interface:
NCCL settings:

Build result:
Import smoke:
Launch command:
Health:

c=1 results:
c=2 results:
c=4 results:
c=8 results:
Long-context result:
Soak duration:

Errors / stack traces:
Output quality notes:
Thermals:
Recommendation:
```

## Safety Notes

- Do not merge this into the public default image path yet.
- Do not run it on a production OpenClaw/Qwen/Gemma host unless the existing vLLM containers are stopped.
- Do not expose the API publicly during validation.
- Keep tester access private and revocable.
- Persist `/root/.cache/flashinfer` and `/root/.triton` on test rigs to avoid repeating first-run JIT cost.
