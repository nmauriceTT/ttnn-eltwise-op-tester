# Performance Microbenchmark

This document describes how `bench.py` measures the performance of TTNN element-wise operations.

## What is measured

- **Device kernel time, via tracy.** Only on-device kernel execution is timed; host-side launch and dispatch are excluded via tracy signposts around each iteration.
- **Per-core duration, scaled by core count.** The representative per-core kernel duration is scaled by the core count and divided by the total element count to yield `cycles_per_datum`. `cycles_per_tile = cycles_per_datum * 1024`.
- **Arch-aware clock.** `ns` is converted to cycles using `1.0 GHz` on Wormhole and `1.35 GHz` on Blackhole.

## Tensor layout

All inputs and outputs are **L1-resident, height-sharded across the full 8×8 core grid, in tile layout**. Shapes are chosen to be as large as possible while still fitting in L1, so the kernel runs at steady state with no DRAM traffic during the measurement.

Per-core shard shapes:

| Operation type | `bfloat16` shard | `float32` shard |
|----------------|------------------|-----------------|
| Unary          | `[256, 256]`     | `[256, 128]`    |
| Binary         | `[256, 128]`     | `[128, 128]`    |

## Warmup

There is **no explicit warmup iteration**. The benchmark runs 10 iterations back-to-back and reports the fastest, which implicitly absorbs any first-iteration overhead.

## Supported operation types

`bench.py` supports the two element-wise operation families registered in `src/operations.py`:

- **Unary element-wise** (`--type unary`, default) — implementations from `UNARY_OPERATIONS`.
- **Binary element-wise** (`--type binary`) — implementations from `BINARY_OPERATIONS`.

Unary backward operations (`UNARY_BW_OPERATIONS`) are **not** benchmarked by this script.

When `-k <base_op>` is passed, all registered implementation variants for that base op are benchmarked in sequence, so variants (e.g. `exp` vs `exp-fast-approx`) can be compared directly.
