# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the KV snapshot/restore engine bridge.

These tests exercise the TP-aware gather/scatter math, the CacheConfig
plumbing, and the worker-level batch read/write dispatch without
requiring a real engine or GPU. The integration test for a live
snapshot/restore round-trip lives in tests/kv_snapshot/test_integration.py
(GPU-gated).
"""

from __future__ import annotations

import pytest
import torch

from vllm.v1.core.kv_cache_manager import (
    create_snapshot_from_blocks,
    prepare_restore_data,
    register_restored_blocks_in_prefix_cache,
)
from vllm.v1.core.kv_cache_snapshot import SnapshotBlockData
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker


def test_tp_gather_scatter_round_trip():
    """Concatenating per-rank shards on dim 3 then chunking back is identity.

    This is the math at the heart of EngineCore.snapshot_kv_cache (gather)
    and restore_kv_cache (scatter): the kv_heads dimension lives at index 3
    of the per-block tensor [2, num_layers, block_size, kv_heads, head_size].
    """
    full = torch.randn(2, 4, 16, 8, 64)
    tp_size = 4

    per_rank = torch.chunk(full, tp_size, dim=3)
    assert all(s.shape == (2, 4, 16, 2, 64) for s in per_rank)

    gathered = torch.cat(per_rank, dim=3)
    assert gathered.shape == full.shape
    assert torch.equal(gathered, full)


def test_tp_gather_preserves_per_rank_slices():
    """A two-rank gather should keep each rank's data in the right kv_heads
    range — rank N's slice lands at heads [N*local : (N+1)*local]."""
    rank0 = torch.randn(2, 4, 16, 2, 64)
    rank1 = torch.randn(2, 4, 16, 2, 64)
    gathered = torch.cat([rank0, rank1], dim=3)

    assert gathered.shape == (2, 4, 16, 4, 64)
    torch.testing.assert_close(gathered[:, :, :, :2, :], rank0)
    torch.testing.assert_close(gathered[:, :, :, 2:, :], rank1)


def test_tp_size_one_gather_is_identity():
    """When TP=1 the gather/scatter should be a no-op aside from the chunk."""
    data = torch.randn(2, 2, 16, 8, 64)
    per_rank = torch.chunk(data, 1, dim=3)
    assert len(per_rank) == 1
    assert torch.equal(per_rank[0], data)
    assert torch.equal(torch.cat(per_rank, dim=3), data)


def test_create_snapshot_and_prepare_restore_round_trip():
    """The standalone serialization helpers preserve tensor data exactly."""
    block_id = 7
    data = torch.randn(2, 4, 16, 8, 64)
    snap = create_snapshot_from_blocks(
        snapshot_id="t1",
        request_id="r1",
        block_ids=[block_id],
        block_data={block_id: data},
        block_hashes={block_id: None},
        num_tokens=16,
        block_size=16,
        dtype=data.dtype,
        num_kv_heads=8,
        head_size=64,
        num_layers=4,
    )
    assert snap.metadata.num_blocks == 1
    assert snap.metadata.num_kv_heads == 8

    new_block_id = 42
    restored = prepare_restore_data(snap, [new_block_id])
    assert new_block_id in restored
    torch.testing.assert_close(restored[new_block_id], data)


class _ModelRunnerStub:
    """Just enough of GPUModelRunner to call read/write_kv_block as classmethods."""

    def __init__(self, kv_caches: list[torch.Tensor]):
        self.kv_caches = kv_caches


def test_read_kv_block_returns_stacked_layers():
    caches = [torch.randn(2, 8, 16, 4, 64) for _ in range(3)]
    stub = _ModelRunnerStub(caches)

    out = GPUModelRunner.read_kv_block(stub, block_id=5)

    assert out.shape == (2, 3, 16, 4, 64)
    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(out[:, layer_idx], cache[:, 5])


def test_write_kv_block_round_trip():
    caches = [torch.randn(2, 8, 16, 4, 64) for _ in range(3)]
    stub = _ModelRunnerStub(caches)

    new_data = torch.randn(2, 3, 16, 4, 64)
    GPUModelRunner.write_kv_block(stub, block_id=5, data=new_data)

    out = GPUModelRunner.read_kv_block(stub, block_id=5)
    torch.testing.assert_close(out, new_data)


def test_read_kv_block_validates_layout():
    """A tensor that doesn't match the (2, num_blocks, ...) FlashAttention
    logical layout should be rejected with a clear error."""
    bad_cache = torch.randn(8, 16, 4, 64)  # missing the leading-2 dim
    stub = _ModelRunnerStub([bad_cache])

    with pytest.raises(RuntimeError, match="2, num_blocks, block_size"):
        GPUModelRunner.read_kv_block(stub, block_id=0)


def test_read_kv_block_rejects_out_of_range_id():
    caches = [torch.randn(2, 4, 16, 4, 64)]
    stub = _ModelRunnerStub(caches)

    with pytest.raises(IndexError):
        GPUModelRunner.read_kv_block(stub, block_id=100)


def test_write_kv_block_rejects_layer_count_mismatch():
    caches = [torch.randn(2, 4, 16, 4, 64) for _ in range(3)]
    stub = _ModelRunnerStub(caches)
    wrong_layers = torch.randn(2, 5, 16, 4, 64)  # 5 layers, but stub has 3

    with pytest.raises(ValueError, match="layers but model has"):
        GPUModelRunner.write_kv_block(stub, block_id=0, data=wrong_layers)


class _WorkerStub:
    """Just enough of Worker to drive write_kv_blocks rank dispatch."""

    def __init__(self, rank: int, captured: list):
        self.rank = rank
        self._captured = captured

        captured_ref = self._captured

        class _MR:
            @staticmethod
            def write_kv_block(block_id, data):
                captured_ref.append((block_id, data))

            @staticmethod
            def read_kv_block(block_id):
                return torch.full((2, 1, 1, 1, 1), float(block_id))

        self.model_runner = _MR()


def test_worker_write_kv_blocks_dispatches_to_own_rank():
    block_ids = [10, 20]
    rank0_shards = [torch.zeros(2, 1, 1, 1, 1), torch.zeros(2, 1, 1, 1, 1)]
    rank1_shards = [torch.ones(2, 1, 1, 1, 1), torch.full((2, 1, 1, 1, 1), 2.0)]

    captured: list = []
    stub = _WorkerStub(rank=1, captured=captured)

    Worker.write_kv_blocks(
        stub, block_ids=block_ids, shards_per_rank=[rank0_shards, rank1_shards]
    )

    assert [c[0] for c in captured] == [10, 20]
    torch.testing.assert_close(captured[0][1], torch.ones(2, 1, 1, 1, 1))
    torch.testing.assert_close(captured[1][1], torch.full((2, 1, 1, 1, 1), 2.0))


def test_worker_write_kv_blocks_rejects_rank_out_of_range():
    captured: list = []
    stub = _WorkerStub(rank=2, captured=captured)

    with pytest.raises(RuntimeError, match="rank 2"):
        Worker.write_kv_blocks(
            stub, block_ids=[0], shards_per_rank=[[torch.zeros(2, 1, 1, 1, 1)]]
        )


def test_worker_write_kv_blocks_rejects_count_mismatch():
    captured: list = []
    stub = _WorkerStub(rank=0, captured=captured)

    with pytest.raises(ValueError, match="!="):
        Worker.write_kv_blocks(
            stub,
            block_ids=[0, 1],
            shards_per_rank=[[torch.zeros(2, 1, 1, 1, 1)]],  # only one shard
        )


def test_worker_read_kv_blocks_returns_dict_keyed_by_id():
    captured: list = []
    stub = _WorkerStub(rank=0, captured=captured)

    out = Worker.read_kv_blocks(stub, block_ids=[3, 7])

    assert set(out.keys()) == {3, 7}
    # Each entry is the per-block tensor returned by read_kv_block; the stub
    # fills it with the block_id value so we can identify the mapping.
    torch.testing.assert_close(out[3], torch.full((2, 1, 1, 1, 1), 3.0))
    torch.testing.assert_close(out[7], torch.full((2, 1, 1, 1, 1), 7.0))


def _make_block_pool(num_blocks: int = 16, enable_caching: bool = True):
    """Construct a real BlockPool — no GPU needed, KVCacheBlock is plain Python."""
    from vllm.v1.core.block_pool import BlockPool

    return BlockPool(
        num_gpu_blocks=num_blocks,
        enable_caching=enable_caching,
        hash_block_size=16,
    )


def _snap_block(block_id: int, hash_: bytes | None) -> SnapshotBlockData:
    """A snapshot block with the bare minimum the wiring helper inspects."""
    return SnapshotBlockData(
        block_id=block_id, block_hash=hash_, data=torch.zeros(1)
    )


def test_register_restored_blocks_inserts_hashes_into_prefix_cache():
    """A restored block whose snapshot recorded a hash becomes discoverable
    via cached_block_hash_to_block, mirroring the path normal cache_blocks
    takes for newly-full blocks."""
    pool = _make_block_pool()
    # Allocate two fresh blocks — these stand in for what restore_kv_cache
    # got back from get_new_blocks.
    restored = pool.get_new_blocks(2)
    snap_blocks = [
        _snap_block(restored[0].block_id, b"hash-a-with-group-id"),
        _snap_block(restored[1].block_id, b"hash-b-with-group-id"),
    ]

    n = register_restored_blocks_in_prefix_cache(pool, restored, snap_blocks)

    assert n == 2
    assert restored[0].block_hash == b"hash-a-with-group-id"
    assert restored[1].block_hash == b"hash-b-with-group-id"
    # Cache lookup returns the same block objects we registered.
    assert pool.cached_block_hash_to_block.get_one_block(
        b"hash-a-with-group-id"
    ) is restored[0]
    assert pool.cached_block_hash_to_block.get_one_block(
        b"hash-b-with-group-id"
    ) is restored[1]


def test_register_skips_blocks_with_none_hash():
    """A snapshot block with block_hash=None (the partial trailing block of a
    request) gets data restored but no prefix-cache registration — there's
    nothing for a future request to look it up by."""
    pool = _make_block_pool()
    restored = pool.get_new_blocks(2)
    snap_blocks = [
        _snap_block(restored[0].block_id, b"hash-a"),
        _snap_block(restored[1].block_id, None),  # partial block
    ]

    n = register_restored_blocks_in_prefix_cache(pool, restored, snap_blocks)

    assert n == 1
    assert restored[1].block_hash is None
    assert len(pool.cached_block_hash_to_block) == 1


def test_register_is_noop_when_caching_disabled():
    """No prefix cache exists to register into; helper must early-out cleanly."""
    pool = _make_block_pool(enable_caching=False)
    restored = pool.get_new_blocks(1)
    snap_blocks = [_snap_block(restored[0].block_id, b"hash-a")]

    n = register_restored_blocks_in_prefix_cache(pool, restored, snap_blocks)

    assert n == 0
    assert restored[0].block_hash is None
    assert len(pool.cached_block_hash_to_block) == 0


def test_register_rejects_length_mismatch():
    pool = _make_block_pool()
    restored = pool.get_new_blocks(2)
    snap_blocks = [_snap_block(restored[0].block_id, b"hash-a")]

    with pytest.raises(ValueError, match="restored vs"):
        register_restored_blocks_in_prefix_cache(pool, restored, snap_blocks)


def test_evict_blocks_removes_restored_hashes_from_prefix_cache():
    """Mirrors what delete_kv_snapshot does: evict the hashes the restore
    registered before freeing the blocks. Without this step the cache map
    would point at a block that's about to be re-allocated to someone else."""
    pool = _make_block_pool()
    restored = pool.get_new_blocks(2)
    snap_blocks = [
        _snap_block(restored[0].block_id, b"hash-a"),
        _snap_block(restored[1].block_id, b"hash-b"),
    ]
    register_restored_blocks_in_prefix_cache(pool, restored, snap_blocks)
    assert len(pool.cached_block_hash_to_block) == 2

    pool.evict_blocks({b.block_id for b in restored})

    assert len(pool.cached_block_hash_to_block) == 0
    assert restored[0].block_hash is None
    assert restored[1].block_hash is None


def test_cache_config_kv_snapshot_defaults():
    """The new fields should default to disabled with sane sizes."""
    from vllm.config.cache import CacheConfig

    cc = CacheConfig()
    assert cc.kv_snapshot_enabled is False
    assert cc.kv_snapshot_dir == "/var/cache/vllm/kv"
    assert cc.kv_snapshot_warm_max_bytes == 8 * 1024 * 1024 * 1024
    assert cc.kv_snapshot_ttl_seconds == 86400


def test_cache_config_kv_snapshot_factors_excluded_from_compute_hash():
    """The new fields shouldn't affect the compiled-graph hash — they only
    govern runtime snapshot serialization. This guards against a future
    refactor accidentally invalidating compile caches when the snapshot
    config changes."""
    from vllm.config.cache import CacheConfig

    cc1 = CacheConfig()
    cc2 = CacheConfig(
        kv_snapshot_enabled=True,
        kv_snapshot_dir="/tmp/other",
        kv_snapshot_warm_max_bytes=1,
        kv_snapshot_ttl_seconds=1,
    )
    assert cc1.compute_hash() == cc2.compute_hash()
