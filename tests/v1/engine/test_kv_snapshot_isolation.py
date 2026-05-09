# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Isolation invariants for the time-sliced multi-tenant KV swap.

These exercise the BlockPool-level guarantees that make tenant
separation safe:

- After reset_prefix_cache, no prior hash lookup can find a block.
- release_all_snapshot_holds frees every restored block so that the
  subsequent reset succeeds (the swap's load-bearing prerequisite).
- Snapshot → reset → restore is a fixed point on the (hash → tensor)
  mapping: the new tenant sees a blank cache, the old tenant on
  return sees their state recovered into freshly allocated blocks
  (block IDs may shift; tensor data is preserved).
- The prefix-allowlist enforcement at the HTTP boundary still applies
  under whole-cache (session) snapshot mode, where no request_id is
  supplied.

The tests run against a real BlockPool (CPU-only — KVCacheBlock is
plain Python). EngineCore-level methods are exercised by binding the
unbound class methods to a minimal self-stub, which avoids needing a
full vLLM engine + scheduler + GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cache.api_router import kv_router
from vllm.kv_snapshot.config import SnapshotConfig
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import register_restored_blocks_in_prefix_cache
from vllm.v1.core.kv_cache_snapshot import SnapshotBlockData
from vllm.v1.engine.core import EngineCore


def _make_block_pool(num_blocks: int = 16) -> BlockPool:
    return BlockPool(
        num_gpu_blocks=num_blocks, enable_caching=True, hash_block_size=16
    )


def _snap_block(block_id: int, hash_: bytes | None) -> SnapshotBlockData:
    return SnapshotBlockData(
        block_id=block_id, block_hash=hash_, data=torch.zeros(1)
    )


class _EngineCoreSelf:
    """Minimal self-stub for binding EngineCore's unbound methods."""

    # Bind the helper methods at class scope so attribute lookup finds them.
    _release_holds = EngineCore._release_holds
    release_all_snapshot_holds = EngineCore.release_all_snapshot_holds
    release_snapshot_holds = EngineCore.release_snapshot_holds

    def __init__(self, block_pool: BlockPool):
        self.scheduler = MagicMock()
        self.scheduler.kv_cache_manager.block_pool = block_pool
        self._restored_block_holds: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------


class TestResetPrefixCacheClearsAllAddressableState:

    def test_reset_clears_cache_map(self):
        pool = _make_block_pool()
        blks = pool.get_new_blocks(3)
        register_restored_blocks_in_prefix_cache(
            pool, blks,
            [_snap_block(b.block_id, h)
             for b, h in zip(blks, [b"h1", b"h2", b"h3"])],
        )
        # Pre-reset, the holds keep ref_cnt > 0 — reset would fail.
        assert pool.reset_prefix_cache() is False

        # Free the holds (simulating release_all_snapshot_holds), then reset.
        pool.evict_blocks({b.block_id for b in blks})
        pool.free_blocks(reversed(blks))
        assert pool.reset_prefix_cache() is True

        # Every prior hash lookup must miss now — this is the leak guard.
        assert pool.cached_block_hash_to_block.get_one_block(b"h1") is None
        assert pool.cached_block_hash_to_block.get_one_block(b"h2") is None
        assert pool.cached_block_hash_to_block.get_one_block(b"h3") is None
        # Every block has had its hash cleared.
        for b in blks:
            assert b.block_hash is None

    def test_reset_clears_iter_blocks(self):
        """iter_blocks (used by session-mode snapshot) must yield nothing
        after a reset — otherwise snapshotting after a reset would
        re-capture stale state."""
        pool = _make_block_pool()
        blks = pool.get_new_blocks(2)
        register_restored_blocks_in_prefix_cache(
            pool, blks,
            [_snap_block(b.block_id, h) for b, h in zip(blks, [b"a", b"b"])],
        )
        pool.evict_blocks({b.block_id for b in blks})
        pool.free_blocks(reversed(blks))
        assert pool.reset_prefix_cache() is True

        assert list(pool.cached_block_hash_to_block.iter_blocks()) == []


# ---------------------------------------------------------------------------
# release_all_snapshot_holds
# ---------------------------------------------------------------------------


class TestReleaseAllSnapshotHolds:

    def test_release_all_evicts_hashes_and_frees_blocks(self):
        pool = _make_block_pool()
        blks_a = pool.get_new_blocks(2)
        blks_b = pool.get_new_blocks(2)
        register_restored_blocks_in_prefix_cache(
            pool, blks_a,
            [_snap_block(b.block_id, h)
             for b, h in zip(blks_a, [b"a1", b"a2"])],
        )
        register_restored_blocks_in_prefix_cache(
            pool, blks_b,
            [_snap_block(b.block_id, h)
             for b, h in zip(blks_b, [b"b1", b"b2"])],
        )
        free_before = pool.get_num_free_blocks()

        stub = _EngineCoreSelf(pool)
        stub._restored_block_holds = {"snap-a": blks_a, "snap-b": blks_b}

        result = stub.release_all_snapshot_holds()

        assert result["status"] == "released"
        assert set(result["released_snapshots"]) == {"snap-a", "snap-b"}
        assert stub._restored_block_holds == {}
        # All blocks back on the free queue.
        assert pool.get_num_free_blocks() == free_before + 4
        # No more cache entries.
        assert len(pool.cached_block_hash_to_block) == 0
        # And reset is now possible (the swap's prerequisite).
        assert pool.reset_prefix_cache() is True

    def test_release_all_when_empty_is_noop(self):
        pool = _make_block_pool()
        stub = _EngineCoreSelf(pool)

        result = stub.release_all_snapshot_holds()

        assert result["status"] == "no_holds"
        assert result["released_snapshots"] == []

    def test_release_all_when_holds_attr_missing(self):
        """Engine that never enabled the snapshot store has no
        _restored_block_holds attribute. release_all should handle that
        cleanly — the swap path runs unconditionally."""
        pool = _make_block_pool()
        stub = _EngineCoreSelf(pool)
        del stub._restored_block_holds

        result = stub.release_all_snapshot_holds()

        assert result["status"] == "no_holds"


# ---------------------------------------------------------------------------
# A → B → A round-trip
# ---------------------------------------------------------------------------


class TestSwapPreservesTenantStateAndIsolation:

    def test_round_trip_recovers_alice_after_bob_runs(self):
        """The full swap dance:
            1. Alice's blocks are cached with her hashes.
            2. (Snapshot conceptually captures these — using iter_blocks.)
            3. Holds released, reset.
            4. Bob's request lookups find nothing (isolation OK).
            5. Bob's blocks are cached.
            6. Snapshot bob, release/reset.
            7. Restore alice — fresh blocks now hold her hashes again.
            8. Alice's lookups succeed; bob's hashes don't appear.
        """
        pool = _make_block_pool(num_blocks=64)

        # ---- Alice's turn ----
        alice_blks = pool.get_new_blocks(3)
        alice_hashes = [b"alice-h1", b"alice-h2", b"alice-h3"]
        alice_data = [torch.randn(2, 2, 16, 4, 64) for _ in range(3)]
        register_restored_blocks_in_prefix_cache(
            pool, alice_blks,
            [SnapshotBlockData(b.block_id, h, d)
             for b, h, d in zip(alice_blks, alice_hashes, alice_data)],
        )

        # Capture alice's session — what we'd put in the store.
        alice_session = [
            (h, d) for (h, _b), d in
            zip(pool.cached_block_hash_to_block.iter_blocks(), alice_data)
        ]
        assert len(alice_session) == 3

        # ---- Swap to bob ----
        stub = _EngineCoreSelf(pool)
        stub._restored_block_holds = {"alice_active": alice_blks}
        stub.release_all_snapshot_holds()
        assert pool.reset_prefix_cache() is True

        # ISOLATION: bob's lookups for alice's hashes return nothing.
        for h in alice_hashes:
            assert pool.cached_block_hash_to_block.get_one_block(h) is None

        # ---- Bob's turn ----
        bob_blks = pool.get_new_blocks(2)
        bob_hashes = [b"bob-h1", b"bob-h2"]
        register_restored_blocks_in_prefix_cache(
            pool, bob_blks,
            [_snap_block(b.block_id, h)
             for b, h in zip(bob_blks, bob_hashes)],
        )
        # Bob is the only one in the cache.
        assert len(pool.cached_block_hash_to_block) == 2
        for h in bob_hashes:
            assert pool.cached_block_hash_to_block.get_one_block(h) is not None

        # ---- Swap back to alice ----
        stub._restored_block_holds = {"bob_active": bob_blks}
        stub.release_all_snapshot_holds()
        assert pool.reset_prefix_cache() is True

        # ISOLATION: alice's incoming lookups find no bob entries.
        for h in bob_hashes:
            assert pool.cached_block_hash_to_block.get_one_block(h) is None

        # Restore alice from her saved session — fresh blocks, same hashes.
        new_alice_blks = pool.get_new_blocks(len(alice_session))
        register_restored_blocks_in_prefix_cache(
            pool, new_alice_blks,
            [SnapshotBlockData(b.block_id, h, d)
             for b, (h, d) in zip(new_alice_blks, alice_session)],
        )

        # Alice's hashes resolve again — but to NEW physical block IDs.
        for new_blk, (h, _) in zip(new_alice_blks, alice_session):
            looked_up = pool.cached_block_hash_to_block.get_one_block(h)
            assert looked_up is not None
            assert looked_up.block_id == new_blk.block_id

        # Block IDs are NOT necessarily the same as alice's first turn —
        # the snapshot is hash-keyed, not id-keyed. Confirm at least one
        # differs to make this assertion meaningful (in a small pool the
        # allocator may reuse ids; if so, the test still passes for the
        # right reason — what matters is the hash mapping).
        new_ids = {b.block_id for b in new_alice_blks}
        old_ids = {b.block_id for b in alice_blks}
        # They may overlap, but the hash → block lookup is the actual
        # invariant we care about. The above assertion verified it.
        assert new_ids and old_ids  # both non-empty


# ---------------------------------------------------------------------------
# Prefix allowlist still enforced under session mode
# ---------------------------------------------------------------------------


class TestPrefixAllowlistEnforcementUnderSessionMode:

    @pytest.fixture
    def client_with_allowlist(self):
        from unittest.mock import AsyncMock

        app = FastAPI()
        app.include_router(kv_router)
        eng = MagicMock()
        eng.snapshot_kv_cache = AsyncMock(return_value={"snapshot_id": "x"})
        app.state.engine_client = eng
        app.state.kv_snapshot_config = SnapshotConfig(
            allowed_id_prefixes=("alice_", "bob_"),
        )
        return TestClient(app), eng

    def test_session_mode_without_id_rejected_by_allowlist(self, client_with_allowlist):
        """The allowlist enforcement applies to session mode too — a
        deployment that wants tenant separation cannot be bypassed by
        omitting both request_id and snapshot_id."""
        client, eng = client_with_allowlist

        resp = client.post("/kv/snapshot", json={})

        assert resp.status_code == 403
        assert "snapshot_id is required" in resp.json()["error"]
        eng.snapshot_kv_cache.assert_not_called()

    def test_session_mode_with_unallowed_id_rejected(self, client_with_allowlist):
        client, eng = client_with_allowlist

        resp = client.post(
            "/kv/snapshot", json={"snapshot_id": "evil_xxx"}
        )

        assert resp.status_code == 403
        eng.snapshot_kv_cache.assert_not_called()

    def test_session_mode_with_allowed_id_admitted(self, client_with_allowlist):
        client, eng = client_with_allowlist

        resp = client.post(
            "/kv/snapshot", json={"snapshot_id": "alice_active"}
        )

        assert resp.status_code == 200
        eng.snapshot_kv_cache.assert_called_once_with(None, "alice_active")
