# Upstream PR #4 Archive

This directory preserves the two patch files generated from
[AEON-7/vllm-ultimate-dgx-spark PR #4](https://github.com/AEON-7/vllm-ultimate-dgx-spark/pull/4).

The patches are archived here so this repository clearly records the origin of
the DeepSeek-V4 / DSpark GB10 enablement layer while keeping the dedicated
container documentation repo focused and easy to audit.

Original PR author:

- [@gilby](https://github.com/gilby) / Kevin Gilbertson

Original commits archived:

- `b83c65cd1432e435e00987e8e2971ce605bdb2e2` — DeepSeek-V4-Flash enablement layer for DGX Spark (GB10/sm_121a), TP2
- `5e2420e5e0c8d5034aa728965b04f1b11eb55adf` — Keep stock cutlass-dsl 4.6.0; fix the concurrency deadlock

The patch archive commits in this repository preserve Kevin Gilbertson as the
Git author, so GitHub can attribute the imported PR work to the original
contributor.
