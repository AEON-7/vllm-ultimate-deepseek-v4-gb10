from __future__ import annotations

from pathlib import Path


ROOT = Path("/usr/local/lib/python3.12/site-packages/vllm")


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"missing patch anchor in {target}: {old!r}")
    target.write_text(text.replace(old, new, 1))


def require(path: str, needle: str) -> None:
    target = ROOT / path
    text = target.read_text()
    if needle not in text:
        raise SystemExit(f"missing required text in {target}: {needle!r}")


if "cache_dtype in (\"nvfp4\", \"nvfp4_ds_mla\")" not in (
    ROOT / "models/deepseek_v4/attention.py"
).read_text():
    replace_once(
        "models/deepseek_v4/attention.py",
        """        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )
""",
        """        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )
        if (
            cache_config is not None
            and cache_config.cache_dtype in ("nvfp4", "nvfp4_ds_mla")
        ):
            head_bytes = 584
""",
    )

kv_text = (ROOT / "v1/kv_cache_interface.py").read_text()
if 'self.cache_dtype_str == "nvfp4_ds_mla"' in kv_text and "return self.storage_block_size * 416" in kv_text:
    replace_once(
        "v1/kv_cache_interface.py",
        """        if self.cache_dtype_str == "nvfp4_ds_mla":
            return self.storage_block_size * 416
        if self.cache_dtype_str == "fp8_ds_mla":
""",
        """        if self.cache_dtype_str == "nvfp4_ds_mla":
            if self.model_version == "deepseek_v4":
                return self.storage_block_size * 584
            return self.storage_block_size * 416
        if self.cache_dtype_str == "fp8_ds_mla":
""",
    )

require("config/cache.py", '"nvfp4_ds_mla"')
require("config/vllm.py", 'self.cache_config.cache_dtype = "nvfp4_ds_mla"')
require("models/deepseek_v4/attention.py", '"nvfp4_ds_mla"')
require("models/deepseek_v4/sparse_mla.py", '"nvfp4_ds_mla"')
require("models/deepseek_v4/sparse_mla.py", "return (num_blocks, block_size, 584)")
require("v1/kv_cache_interface.py", '"nvfp4_ds_mla"')
require("v1/attention/backends/mla/flashmla_sparse.py", '"nvfp4_ds_mla"')
require("model_executor/layers/fused_moe/oracle/mxfp4.py", "B12X_MXFP4")
require("model_executor/layers/fused_moe/experts/b12x_mxfp4_moe.py", "class B12xExperts")

print("DSpark Stage C B12X/NVFP4 DeepSeek V4 validation OK")
