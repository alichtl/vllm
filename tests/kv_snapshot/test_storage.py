"""Tests for KV cache snapshot storage backends."""

import time

import pytest
import torch

from vllm.kv_snapshot.storage import (
    DiskSnapshotStore,
    MemorySnapshotStore,
    TieredSnapshotStore,
)
from vllm.v1.core.kv_cache_snapshot import (
    KVCacheSnapshot,
    SnapshotBlockData,
    SnapshotMetadata,
    SnapshotTier,
)


def _make_snapshot(snapshot_id: str, request_id: str = "req-1",
                   num_blocks: int = 2, created_at: float | None = None,
                   num_layers: int = 4, block_size: int = 16,
                   num_kv_heads: int = 2,
                   head_size: int = 64) -> KVCacheSnapshot:
    meta = SnapshotMetadata(
        snapshot_id=snapshot_id,
        request_id=request_id,
        num_tokens=num_blocks * block_size,
        num_blocks=num_blocks,
        block_size=block_size,
        dtype=torch.float16,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        num_layers=num_layers,
    )
    if created_at is not None:
        meta.created_at = created_at
    blocks = [
        SnapshotBlockData(
            block_id=i,
            block_hash=f"hash-{i}".encode() if i % 2 == 0 else None,
            data=torch.randn(2, num_layers, block_size, num_kv_heads,
                             head_size, dtype=torch.float16),
        )
        for i in range(num_blocks)
    ]
    return KVCacheSnapshot(metadata=meta, blocks=blocks)


class TestMemorySnapshotStore:

    def test_save_load_roundtrip(self):
        store = MemorySnapshotStore()
        snap = _make_snapshot("snap-1")
        store.save(snap)
        loaded = store.load("snap-1")
        assert loaded is not None
        assert loaded.metadata.snapshot_id == "snap-1"
        assert len(loaded.blocks) == 2
        for block in loaded.blocks:
            assert block.data.device.type == "cpu"

    def test_load_missing_returns_none(self):
        store = MemorySnapshotStore()
        assert store.load("nonexistent") is None

    def test_delete(self):
        store = MemorySnapshotStore()
        snap = _make_snapshot("snap-del")
        store.save(snap)
        assert store.delete("snap-del") is True
        assert store.load("snap-del") is None
        assert store.delete("snap-del") is False

    def test_status(self):
        store = MemorySnapshotStore()
        snap = _make_snapshot("snap-st")
        store.save(snap)
        st = store.status("snap-st")
        assert st is not None
        assert st["tier"] == "warm"
        assert st["snapshot_id"] == "snap-st"
        assert st["num_blocks"] == 2
        assert store.status("nonexistent") is None

    def test_list_snapshots(self):
        store = MemorySnapshotStore()
        store.save(_make_snapshot("a"))
        store.save(_make_snapshot("b"))
        ids = store.list_snapshots()
        assert set(ids) == {"a", "b"}


class TestDiskSnapshotStore:

    def test_save_load_roundtrip(self, tmp_path):
        store = DiskSnapshotStore(str(tmp_path / "snapshots"))
        snap = _make_snapshot("disk-1")
        store.save(snap)
        loaded = store.load("disk-1")
        assert loaded is not None
        assert loaded.metadata.snapshot_id == "disk-1"
        for orig, restored in zip(snap.blocks, loaded.blocks):
            assert torch.allclose(orig.data, restored.data)

    def test_directory_creation(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        store = DiskSnapshotStore(str(nested))
        snap = _make_snapshot("nested-1")
        store.save(snap)
        assert nested.exists()
        assert store.load("nested-1") is not None

    def test_status_reads_sidecar(self, tmp_path):
        store = DiskSnapshotStore(str(tmp_path / "snap_st"))
        snap = _make_snapshot("disk-st")
        store.save(snap)
        st = store.status("disk-st")
        assert st is not None
        assert st["tier"] == "cold"
        assert st["snapshot_id"] == "disk-st"

    def test_delete(self, tmp_path):
        store = DiskSnapshotStore(str(tmp_path / "snap_del"))
        snap = _make_snapshot("disk-del")
        store.save(snap)
        assert store.delete("disk-del") is True
        assert store.load("disk-del") is None
        assert store.delete("disk-del") is False

    def test_list_snapshots(self, tmp_path):
        store = DiskSnapshotStore(str(tmp_path / "snap_list"))
        store.save(_make_snapshot("x"))
        store.save(_make_snapshot("y"))
        assert set(store.list_snapshots()) == {"x", "y"}


class TestTieredSnapshotStore:

    def _make_tiered(self, tmp_path, warm_max_bytes=1024 * 1024 * 1024,
                     ttl_seconds=86400):
        warm = MemorySnapshotStore()
        cold = DiskSnapshotStore(str(tmp_path / "cold"))
        return TieredSnapshotStore(warm, cold, warm_max_bytes, ttl_seconds)

    def test_save_load_warm(self, tmp_path):
        store = self._make_tiered(tmp_path)
        snap = _make_snapshot("tiered-1")
        store.save(snap)
        loaded = store.load("tiered-1")
        assert loaded is not None
        st = store.status("tiered-1")
        assert st["tier"] == "warm"

    def test_spill_to_cold(self, tmp_path):
        # Set warm budget very small so it spills immediately
        store = self._make_tiered(tmp_path, warm_max_bytes=1)
        store.save(_make_snapshot("spill-1"))
        # After save, spill should have moved it to cold
        st = store.status("spill-1")
        assert st is not None
        assert st["tier"] == "cold"

    def test_cold_to_warm_promotion(self, tmp_path):
        store = self._make_tiered(tmp_path, warm_max_bytes=1)
        store.save(_make_snapshot("promo-1"))
        # It's in cold now
        assert store.status("promo-1")["tier"] == "cold"
        # Increase budget so promotion can happen
        store._warm_max_bytes = 1024 * 1024 * 1024
        loaded = store.load("promo-1")
        assert loaded is not None
        assert store.status("promo-1")["tier"] == "warm"

    def test_cross_tier_listing(self, tmp_path):
        warm = MemorySnapshotStore()
        cold = DiskSnapshotStore(str(tmp_path / "cold"))
        store = TieredSnapshotStore(warm, cold,
                                    warm_max_bytes=1024 * 1024 * 1024,
                                    ttl_seconds=86400)
        # Put one in warm directly
        warm.save(_make_snapshot("w1"))
        # Put one in cold directly
        cold.save(_make_snapshot("c1"))
        ids = store.list_snapshots()
        assert set(ids) == {"w1", "c1"}

    def test_ttl_expiry(self, tmp_path):
        store = self._make_tiered(tmp_path, ttl_seconds=1)
        snap = _make_snapshot("expire-1", created_at=time.time() - 10)
        store.save(snap)
        removed = store.cleanup_expired()
        assert removed == 1
        assert store.load("expire-1") is None
