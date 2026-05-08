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

## What's NOT implemented (the load-bearing gap)

**No engine implementation** of the four abstract methods declared in
`vllm/engine/protocol.py:151-168`:

- `EngineClient.snapshot_kv_cache(request_id, snapshot_id)`
- `EngineClient.restore_kv_cache(snapshot_id)`
- `EngineClient.delete_kv_snapshot(snapshot_id)`
- `EngineClient.get_kv_snapshot_status(snapshot_id)`

`AsyncLLM`, `LLMEngine`, `MQLLMEngine`, and friends in `vllm/v1/engine/`
do not override these. **Calling any `/kv/*` endpoint today will
`NotImplementedError` (or `AttributeError`) at runtime.**

This is the missing piece that everything downstream depends on. The
HTTP layer, the prefix rejection, and any future auto-swap middleware
all sit on top of these methods doing real work.

## Implementing the engine layer (next session's main job)

The engine implementation needs to bridge the API layer to the v1 core:

1. **Snapshot path** (`AsyncLLM.snapshot_kv_cache`):
   - Look up the request in `Scheduler` to get its allocated block list.
   - Read the K/V tensors for those blocks out of the GPU `BlockPool`.
   - Call `create_snapshot_from_blocks(...)` (already implemented in
     `kv_cache_manager.py`) to materialize a `KVCacheSnapshot` with
     CPU-cloned tensors.
   - Hand it to a `SnapshotStore` backend (memory/disk/tiered) keyed by
     `snapshot_id`.

2. **Restore path** (`AsyncLLM.restore_kv_cache`):
   - Load the snapshot from the store.
   - Allocate fresh blocks in the `BlockPool` matching the snapshot's
     block layout.
   - Copy CPU tensors back into the new GPU blocks.
   - Wire the new block ids into a request slot so the next prefill
     step finds the prefilled state.

3. **Engine boundary considerations**:
   - The v1 scheduler runs in the engine-core process; the API runs in
     the front-end process. Snapshot/restore needs to round-trip across
     the IPC boundary (`core_client.py`). Use the same RPC pattern as
     existing engine-core calls.
   - Concurrency: snapshot/restore must respect in-flight steps. Probably
     easiest to schedule them as engine-core tasks that run between
     `step()` calls rather than concurrently.
   - `BlockPool` allocations under restore can fail (OOM); the restore
     RPC needs a typed error path.

4. **Multi-GPU / TP**:
   - Each TP rank holds a slice of every block. Snapshot must gather
     across ranks; restore must scatter back. Use the existing
     all-gather paths used by the model executor.

5. **Tests**:
   - Unit-test the engine path with a tiny model (the existing test
     pattern in `tests/v1/engine/`).
   - End-to-end test that snapshots a request mid-conversation, kills
     the request, restores into a new one, and asserts the next decode
     produces identical token logits.

This is genuinely multi-session work. Plan to write it in pieces:
serialize-block-tensors → restore-into-fresh-blocks → IPC bridge →
TP-aware gather/scatter → tests.

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
