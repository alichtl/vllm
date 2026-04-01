"""Data structures for KV cache snapshot/restore."""

import time
from dataclasses import dataclass, field
from enum import IntEnum

import torch


class SnapshotTier(IntEnum):
    """Storage tier for KV cache snapshots."""
    HOT = 0   # GPU VRAM
    WARM = 1  # Host RAM
    COLD = 2  # Disk


@dataclass
class SnapshotMetadata:
    """Metadata describing a KV cache snapshot."""
    snapshot_id: str
    request_id: str
    num_tokens: int
    num_blocks: int
    block_size: int
    dtype: torch.dtype
    num_kv_heads: int
    head_size: int
    num_layers: int
    created_at: float = field(default_factory=time.time)

    @property
    def estimated_size_bytes(self) -> int:
        """Estimate total size: blocks * block_size * heads * head_dim
        * 2 (K+V) * layers * element_size."""
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        return (self.num_blocks * self.block_size * self.num_kv_heads
                * self.head_size * 2 * self.num_layers * elem_size)


@dataclass
class SnapshotBlockData:
    """Data for a single KV cache block.

    data shape: [2, num_layers, block_size, num_kv_heads, head_size]
      - dim 0: key (0) and value (1)
    """
    block_id: int
    block_hash: bytes | None
    data: torch.Tensor


@dataclass
class KVCacheSnapshot:
    """A complete KV cache snapshot for a request."""
    metadata: SnapshotMetadata
    blocks: list[SnapshotBlockData]
    tier: SnapshotTier = SnapshotTier.WARM
