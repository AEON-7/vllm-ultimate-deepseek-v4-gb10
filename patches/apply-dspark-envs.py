from __future__ import annotations

from pathlib import Path


ENVS = Path("/usr/local/lib/python3.12/site-packages/vllm/envs.py")
MARKER = "# AEON DSpark/B12X overlay environment compatibility"

BLOCK = r'''

# AEON DSpark/B12X overlay environment compatibility.
# Keep the base image envs.py intact and add only the knobs required by the
# experimental DeepSeek-V4 DSpark overlay.
def _aeon_env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


environment_variables.setdefault(
    "VLLM_USE_B12X_SPARSE_INDEXER",
    lambda: _aeon_env_bool("VLLM_USE_B12X_SPARSE_INDEXER"),
)
environment_variables.setdefault(
    "VLLM_USE_B12X_MHC",
    lambda: _aeon_env_bool("VLLM_USE_B12X_MHC"),
)
environment_variables.setdefault(
    "VLLM_USE_B12X_FP8_GEMM",
    lambda: _aeon_env_bool("VLLM_USE_B12X_FP8_GEMM"),
)
environment_variables.setdefault(
    "VLLM_USE_B12X_WO_PROJECTION",
    lambda: _aeon_env_bool("VLLM_USE_B12X_WO_PROJECTION"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_POSITION0_DIAGNOSTICS",
    lambda: _aeon_env_bool("VLLM_DSPARK_POSITION0_DIAGNOSTICS"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK",
    lambda: _aeon_env_bool("VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT",
    lambda: _aeon_env_bool("VLLM_DSPARK_REFERENCE_KV_QUANT_DEQUANT"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_FUSED_MARKOV_ARGMAX",
    lambda: _aeon_env_bool("VLLM_DSPARK_FUSED_MARKOV_ARGMAX"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS",
    lambda: _aeon_env_bool("VLLM_DSPARK_COLLECT_CONFIDENCE_DIAGNOSTICS"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_CONFIDENCE_SCHEDULER",
    lambda: _aeon_env_bool("VLLM_DSPARK_CONFIDENCE_SCHEDULER"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_CONFIDENCE_THRESHOLD",
    lambda: float(os.getenv("VLLM_DSPARK_CONFIDENCE_THRESHOLD", "0.0")),
)
environment_variables.setdefault(
    "VLLM_DSPARK_EXPORT_DRAFT_PROBS",
    lambda: _aeon_env_bool("VLLM_DSPARK_EXPORT_DRAFT_PROBS"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_FORCE_DRAFT_LENGTH",
    lambda: int(os.getenv("VLLM_DSPARK_FORCE_DRAFT_LENGTH", "0")),
)
environment_variables.setdefault(
    "VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP",
    lambda: _aeon_env_bool("VLLM_DSPARK_HARDWARE_SCHEDULER_EARLY_STOP", "1"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_STAGE_TIMING",
    lambda: _aeon_env_bool("VLLM_DSPARK_STAGE_TIMING"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_STAGE_TIMING_LOG_EVERY",
    lambda: int(os.getenv("VLLM_DSPARK_STAGE_TIMING_LOG_EVERY", "20")),
)
environment_variables.setdefault(
    "VLLM_DSPARK_ITER_TIMING",
    lambda: _aeon_env_bool("VLLM_DSPARK_ITER_TIMING"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_ITER_TIMING_LOG_EVERY",
    lambda: int(os.getenv("VLLM_DSPARK_ITER_TIMING_LOG_EVERY", "20")),
)
environment_variables.setdefault(
    "VLLM_DSPARK_TARGET_TIMING",
    lambda: _aeon_env_bool("VLLM_DSPARK_TARGET_TIMING"),
)
environment_variables.setdefault(
    "VLLM_DSPARK_TARGET_TIMING_LOG_EVERY",
    lambda: int(os.getenv("VLLM_DSPARK_TARGET_TIMING_LOG_EVERY", "20")),
)
environment_variables.setdefault(
    "VLLM_DSPARK_SPS_CURVE",
    lambda: os.getenv("VLLM_DSPARK_SPS_CURVE", ""),
)
environment_variables.setdefault(
    "VLLM_DSV4_B12X_COMPRESSED_MLA",
    lambda: _aeon_env_bool("VLLM_DSV4_B12X_COMPRESSED_MLA"),
)
environment_variables.setdefault(
    "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE",
    lambda: _aeon_env_bool("VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE"),
)
environment_variables.setdefault(
    "VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT",
    lambda: _aeon_env_bool("VLLM_DSV4_DSPARK_DEFER_TARGET_CAPTURE_EXACT"),
)
environment_variables.setdefault(
    "VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP",
    lambda: _aeon_env_bool("VLLM_ENABLE_DEEPSEEK_V4_SPARSE_MLA_WARMUP", "1"),
)
'''


text = ENVS.read_text()
if MARKER not in text:
    ENVS.write_text(text.rstrip() + BLOCK + "\n")

print("DSpark/B12X env compatibility OK")
