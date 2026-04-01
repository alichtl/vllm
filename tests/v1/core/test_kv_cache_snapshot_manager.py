"""Tests for KV cache snapshot serialization helpers."""

import torch

from vllm.v1.core.kv_cache_manager import (
    create_snapshot_from_blocks,
    prepare_restore_data,
)
from vllm.v1.core.kv_cache_snapshot import SnapshotTier


NUM_LAYERS = 4
BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 64
DTYPE = torch.float16
SHAPE = (2, NUM_LAYERS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)


def _fake_block_data(block_ids: list[int]) -> dict[int, torch.Tensor]:
    return {bid: torch.randn(*SHAPE, dtype=DTYPE) for bid in block_ids}


class TestCreateSnapshotFromBlocks:

    def test_correct_metadata(self):
        block_ids = [3, 7]
        data = _fake_block_data(block_ids)
        hashes: dict[int, bytes | None] = {3: b"hash3", 7: None}
        snap = create_snapshot_from_blocks(
            snapshot_id="snap-1",
            request_id="req-1",
            block_ids=block_ids,
            block_data=data,
            block_hashes=hashes,
            num_tokens=32,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            num_layers=NUM_LAYERS,
        )
        assert snap.metadata.snapshot_id == "snap-1"
        assert snap.metadata.request_id == "req-1"
        assert snap.metadata.num_blocks == 2
        assert snap.metadata.num_tokens == 32
        assert snap.tier == SnapshotTier.WARM

    def test_tensors_are_cpu_copies(self):
        block_ids = [0, 1]
        data = _fake_block_data(block_ids)
        snap = create_snapshot_from_blocks(
            snapshot_id="snap-2",
            request_id="req-2",
            block_ids=block_ids,
            block_data=data,
            block_hashes={},
            num_tokens=32,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            num_layers=NUM_LAYERS,
        )
        for block in snap.blocks:
            assert block.data.device.type == "cpu"
        # Must be a distinct copy (different data_ptr)
        assert snap.blocks[0].data.data_ptr() != data[0].data_ptr()
        # But values should match
        assert torch.allclose(snap.blocks[0].data, data[0])

    def test_block_hashes_preserved(self):
        block_ids = [5]
        data = _fake_block_data(block_ids)
        snap = create_snapshot_from_blocks(
            snapshot_id="snap-3",
            request_id="req-3",
            block_ids=block_ids,
            block_data=data,
            block_hashes={5: b"myhash"},
            num_tokens=16,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            num_layers=NUM_LAYERS,
        )
        assert snap.blocks[0].block_hash == b"myhash"


class TestPrepareRestoreData:

    def test_mapping_works(self):
        block_ids = [0, 1]
        data = _fake_block_data(block_ids)
        snap = create_snapshot_from_blocks(
            snapshot_id="snap-r1",
            request_id="req-r1",
            block_ids=block_ids,
            block_data=data,
            block_hashes={},
            num_tokens=32,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            num_layers=NUM_LAYERS,
        )
        new_ids = [10, 20]
        mapping = prepare_restore_data(snap, new_ids)
        assert set(mapping.keys()) == {10, 20}
        assert torch.allclose(mapping[10], snap.blocks[0].data)
        assert torch.allclose(mapping[20], snap.blocks[1].data)

    def test_length_mismatch_raises(self):
        block_ids = [0]
        data = _fake_block_data(block_ids)
        snap = create_snapshot_from_blocks(
            snapshot_id="snap-r2",
            request_id="req-r2",
            block_ids=block_ids,
            block_data=data,
            block_hashes={},
            num_tokens=16,
            block_size=BLOCK_SIZE,
            dtype=DTYPE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            num_layers=NUM_LAYERS,
        )
        import pytest
        with pytest.raises(ValueError, match="Block count mismatch"):
            prepare_restore_data(snap, [10, 20])
