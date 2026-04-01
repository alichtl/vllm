"""Tests for KV cache snapshot data structures."""

import time

import pytest
import torch

from vllm.v1.core.kv_cache_snapshot import (
    KVCacheSnapshot,
    SnapshotBlockData,
    SnapshotMetadata,
    SnapshotTier,
)


class TestSnapshotTier:

    def test_tier_ordering(self):
        assert SnapshotTier.HOT < SnapshotTier.WARM < SnapshotTier.COLD

    def test_tier_values(self):
        assert SnapshotTier.HOT == 0
        assert SnapshotTier.WARM == 1
        assert SnapshotTier.COLD == 2


class TestSnapshotMetadata:

    def test_metadata_creation(self):
        before = time.time()
        meta = SnapshotMetadata(
            snapshot_id="snap-001",
            request_id="req-1",
            num_tokens=128,
            num_blocks=8,
            block_size=16,
            dtype=torch.float16,
            num_kv_heads=4,
            head_size=64,
            num_layers=12,
        )
        after = time.time()
        assert meta.snapshot_id == "snap-001"
        assert meta.request_id == "req-1"
        assert meta.num_tokens == 128
        assert before <= meta.created_at <= after

    def test_estimated_size_bytes_float16(self):
        meta = SnapshotMetadata(
            snapshot_id="snap-002",
            request_id="req-2",
            num_tokens=64,
            num_blocks=4,
            block_size=16,
            dtype=torch.float16,
            num_kv_heads=2,
            head_size=64,
            num_layers=6,
        )
        # 4 blocks * 16 tokens * 2 heads * 64 dim * 2 (K+V) * 6 layers * 2 bytes
        expected = 4 * 16 * 2 * 64 * 2 * 6 * 2
        assert meta.estimated_size_bytes == expected

    def test_estimated_size_bytes_float32(self):
        meta = SnapshotMetadata(
            snapshot_id="snap-003",
            request_id="req-3",
            num_tokens=32,
            num_blocks=2,
            block_size=16,
            dtype=torch.float32,
            num_kv_heads=4,
            head_size=128,
            num_layers=24,
        )
        expected = 2 * 16 * 4 * 128 * 2 * 24 * 4
        assert meta.estimated_size_bytes == expected


class TestSnapshotBlockData:

    def test_block_with_hash(self):
        data = torch.randn(2, 6, 16, 2, 64)
        block = SnapshotBlockData(
            block_id=5,
            block_hash=b"abc123",
            data=data,
        )
        assert block.block_id == 5
        assert block.block_hash == b"abc123"
        assert torch.equal(block.data, data)

    def test_block_without_hash(self):
        data = torch.randn(2, 6, 16, 2, 64)
        block = SnapshotBlockData(
            block_id=10,
            block_hash=None,
            data=data,
        )
        assert block.block_hash is None


class TestKVCacheSnapshot:

    def test_snapshot_creation(self):
        meta = SnapshotMetadata(
            snapshot_id="snap-100",
            request_id="req-100",
            num_tokens=32,
            num_blocks=2,
            block_size=16,
            dtype=torch.float16,
            num_kv_heads=2,
            head_size=64,
            num_layers=4,
        )
        blocks = [
            SnapshotBlockData(block_id=0, block_hash=None,
                              data=torch.randn(2, 4, 16, 2, 64)),
            SnapshotBlockData(block_id=1, block_hash=b"hash1",
                              data=torch.randn(2, 4, 16, 2, 64)),
        ]
        snap = KVCacheSnapshot(metadata=meta, blocks=blocks)
        assert snap.metadata.snapshot_id == "snap-100"
        assert len(snap.blocks) == 2

    def test_default_tier_is_warm(self):
        meta = SnapshotMetadata(
            snapshot_id="snap-101",
            request_id="req-101",
            num_tokens=16,
            num_blocks=1,
            block_size=16,
            dtype=torch.float16,
            num_kv_heads=2,
            head_size=64,
            num_layers=4,
        )
        snap = KVCacheSnapshot(metadata=meta, blocks=[])
        assert snap.tier == SnapshotTier.WARM

    def test_explicit_tier(self):
        meta = SnapshotMetadata(
            snapshot_id="snap-102",
            request_id="req-102",
            num_tokens=16,
            num_blocks=1,
            block_size=16,
            dtype=torch.float16,
            num_kv_heads=2,
            head_size=64,
            num_layers=4,
        )
        snap = KVCacheSnapshot(
            metadata=meta, blocks=[], tier=SnapshotTier.COLD
        )
        assert snap.tier == SnapshotTier.COLD
