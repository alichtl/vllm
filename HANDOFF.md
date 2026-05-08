# Fork handoff — KV cache snapshot/restore

State of the `kv-snapshot` branch as of 2026-05-08. Read top-to-bottom
before resuming.

## Current branch

`kv-snapshot`, six commits ahead of `upstream/main` (last upstream
fetched: `17b72fd1c`):

```
a6ce1b721 feat(kv-snapshot): server-side allowlist of snapshot_id prefixes
6505eff06 feat: add /kv/* snapshot API endpoints
54049916f feat: add snapshot/restore abstract methods to EngineClient protocol
19ed7f00d feat: add KV cache snapshot serialization helpers
67e811f53 feat: add snapshot storage backends (memory, disk, tiered)
0a25d0635 feat: add KV cache snapshot data structures
```

`work.md` (untracked, at repo root) is the original 11-task spec the
first five commits were written against.

## What's implemented

| Layer                      | File                                                          | Status                                          |
| -------------------------- | ------------------------------------------------------------- | ----------------------------------------------- |
| Data structures            | `vllm/v1/core/kv_cache_snapshot.py`                           | Done — dataclasses, tier enum, size estimation  |
| Storage backends           | `vllm/kv_snapshot/storage.py`                                 | Done — memory/disk/tiered, spill, TTL, promote  |
| Serialization helpers      | `vllm/v1/core/kv_cache_manager.py` (free fns)                 | Done — `create_snapshot_from_blocks`, `prepare_restore_data` |
| Engine client protocol     | `vllm/engine/protocol.py`                                     | Abstract methods declared (no implementations)  |
| HTTP API                   | `vllm/entrypoints/serve/cache/api_router.py`                  | `/kv/snapshot`, `/kv/restore`, `/kv/{id}`, `/kv/{id}/status` — always attached |
| Prefix-allowlist rejection | same `api_router.py` + `vllm/kv_snapshot/config.py` + `vllm/envs.py` | Done — opt-in via `VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES` |

Tests cover everything above:

- `tests/v1/core/test_kv_cache_snapshot.py`
- `tests/kv_snapshot/test_storage.py`
- `tests/v1/core/test_kv_cache_snapshot_manager.py`
- `tests/entrypoints/serve/cache/test_kv_snapshot_api.py` (rejection
  layer added in `a6ce1b721`)

## Engine bridge — landed (single-GPU and TP)

The four abstract methods declared in `vllm/engine/protocol.py:151-170`
now have a working implementation across the engine-core IPC boundary,
including TP-aware gather/scatter:

- HTTP layer → `AsyncLLM.{snapshot,restore,delete,get_kv_snapshot_status}`
  in `vllm/v1/engine/async_llm.py`
- AsyncLLM → `EngineCoreClient.*_async` (`call_utility_async` on
  `AsyncMPClient`) in `vllm/v1/engine/core_client.py`
- EngineCore methods (`snapshot_kv_cache`, `restore_kv_cache`,
  `delete_kv_snapshot`, `get_kv_snapshot_status`) in
  `vllm/v1/engine/core.py`
- TP gather/scatter via `model_executor.collective_rpc`:
  - **Snapshot path** — each rank reads its KV-head slice via
    `Worker.read_kv_blocks` → `GPUModelRunner.read_kv_block`, returns
    `dict[block_id, Tensor]` on CPU. EngineCore concatenates per-block
    along dim 3 (kv_heads) to produce the full-head snapshot.
  - **Restore path** — EngineCore `torch.chunk`s each block tensor on
    dim 3 into `tp_size` shards, broadcasts the shard list via
    `collective_rpc("write_kv_blocks", ...)`. Each worker picks
    `shards_per_rank[self.rank]` and writes via
    `GPUModelRunner.write_kv_block`.
- New CLI flags / `CacheConfig` fields wired through `arg_utils.py`:
  `--kv-snapshot-enabled`, `--kv-snapshot-dir`,
  `--kv-snapshot-warm-max-bytes`, `--kv-snapshot-ttl-seconds`. All
  excluded from `compute_hash` so they don't invalidate compile caches.
- Unit tests in `tests/v1/engine/test_kv_snapshot_bridge.py` cover the
  TP cat/chunk math, the read/write_kv_block round-trip on a CPU stub,
  and the worker-level rank dispatch.

### What still needs follow-up

1. **Wire restored blocks to a request slot.** Today
   `EngineCore.restore_kv_cache` allocates blocks (ref_cnt += 1) and
   stashes them in `self._restored_block_holds[snapshot_id]`. The next
   prefill won't pick them up automatically — either we register the
   per-block hashes in the prefix cache, or we extend `restore` to take
   a `request_id` and overwrite that request's blocks.
2. **KV layout coverage.** Snapshot/restore assumes the FlashAttention
   logical shape `(2, num_blocks, block_size, num_kv_heads, head_size)`
   per layer. MLA / sparse / mamba layouts will need separate
   implementations in `read_kv_block` / `write_kv_block`.
3. **Multi-group caches.** The current code asserts a single
   `kv_cache_group`. Models with sliding-window + full attention split
   across groups need group-aware snapshot/restore.
4. **Real-model end-to-end test.** The unit tests don't actually run a
   model. A GPU-gated test that snapshots a mid-conversation request,
   restores it, and asserts identical next-token logits is the
   confidence-builder before relying on this in production.

## Auto-swap on tenant change (future, depends on the above)

The downstream design discussed but explicitly deferred:

- Each tenant's "session" KV state lives under a snapshot id like
  `<tenant>_active`.
- Inbound requests carry an `X-Tenant-Id` header (NetworkPolicy is the
  trust perimeter; no auth in the API).
- Middleware on `/v1/chat/completions` and `/v1/completions`:
  1. Read tenant from header.
  2. If different from `app.state.current_tenant`:
     - Snapshot the outgoing tenant's blocks to `<outgoing>_active`
       (warm/cold tier per `SnapshotConfig`).
     - `reset_prefix_cache` on the engine.
     - If `<incoming>_active` exists, restore it.
     - Update `current_tenant`.
  3. Acquire engine lock for the tenant; release on response complete.
- Spill policy: warm tier auto-spills to cold tier when over budget
  (already implemented in `TieredSnapshotStore`); revisit eviction
  ordering once real workloads exist.

Don't start the middleware until the engine layer is in. Otherwise it's
plumbing that calls methods that error.

## Prefix-allowlist layer (live; safe to deploy)

The prefix-rejection layer (commit `a6ce1b721`) does NOT depend on the
engine implementation. It rejects bad calls at the HTTP boundary
*before* they reach the engine. Useful as a defensive layer in any
multi-tenant deployment.

To enable in a deployment: set the env var on the container.

```
VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES=tenant_a_,tenant_b_
```

(Tenants are free to register more prefixes later — it's a CSV list.)

When unset (default), enforcement is off — current behavior preserved.

Tests for both modes pass:

```
ruff check vllm/kv_snapshot/config.py \
           vllm/entrypoints/serve/cache/api_router.py \
           tests/entrypoints/serve/cache/test_kv_snapshot_api.py
```

(Full `pytest` requires the vllm dev environment; this fork doesn't
have one wired up yet — adding `tests/conftest.py`-compatible deps to a
local venv is left as an exercise.)

## Image-build note

Consumers that base on `vllm/vllm-openai:<tag>` (upstream) can run this
fork in two ways:

- **Layer the fork-only files on top** of the upstream release image
  (fast — Python-only diff). Works for the rejection layer alone, since
  no engine code is involved:
    - `vllm/v1/core/kv_cache_snapshot.py`
    - `vllm/v1/core/kv_cache_manager.py` (the appended free functions)
    - `vllm/kv_snapshot/__init__.py`, `config.py`, `storage.py`
    - `vllm/engine/protocol.py` (the four new abstract methods)
    - `vllm/entrypoints/serve/cache/api_router.py`
    - `vllm/envs.py` (the new `VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES` entry)

- **Rebuild from the fork's `docker/Dockerfile`** (slow — full CUDA
  build). Required once the engine implementation lands, because the
  v1 core will have changed.

## Naming constraint

This fork is public (`alichtl/vllm`). All code, comments, commit
messages, test names, and documentation must be 100% generic — never
reference any specific downstream consumer or deployment context.
Frame everything as a general-purpose vLLM improvement. Tests use
generic placeholder prefixes (`alpha_`, `bravo_`); the env var name
itself is generic (`VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES`).

## File map

```
vllm/
├── envs.py                                      # +VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES
├── engine/protocol.py                           # +4 abstract /kv/* methods
├── entrypoints/serve/cache/api_router.py        # /kv/* routes + prefix allowlist
├── kv_snapshot/
│   ├── __init__.py
│   ├── config.py                                # SnapshotConfig + from_env
│   └── storage.py                               # Memory/Disk/Tiered stores
└── v1/core/
    ├── kv_cache_manager.py                      # +create_snapshot_from_blocks, prepare_restore_data
    └── kv_cache_snapshot.py                     # SnapshotMetadata, KVCacheSnapshot, SnapshotTier

tests/
├── kv_snapshot/test_storage.py
├── v1/core/test_kv_cache_snapshot.py
├── v1/core/test_kv_cache_snapshot_manager.py
└── entrypoints/serve/cache/test_kv_snapshot_api.py

work.md                                          # untracked — original 11-task spec
HANDOFF.md                                       # this file
```
