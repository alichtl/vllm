"""Snapshot storage backends: memory, disk, and tiered."""

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path

import torch

from vllm.v1.core.kv_cache_snapshot import (
    KVCacheSnapshot,
    SnapshotBlockData,
    SnapshotMetadata,
    SnapshotTier,
)


class SnapshotStore(ABC):
    """Abstract base class for snapshot storage backends."""

    @abstractmethod
    def save(self, snapshot: KVCacheSnapshot) -> None:
        """Persist a snapshot."""
        ...

    @abstractmethod
    def load(self, snapshot_id: str) -> KVCacheSnapshot | None:
        """Load a snapshot by ID. Returns None if not found."""
        ...

    @abstractmethod
    def delete(self, snapshot_id: str) -> bool:
        """Delete a snapshot. Returns True if it existed."""
        ...

    @abstractmethod
    def status(self, snapshot_id: str) -> dict | None:
        """Return lightweight status dict or None if not found."""
        ...

    @abstractmethod
    def list_snapshots(self) -> list[str]:
        """Return all snapshot IDs."""
        ...


class MemorySnapshotStore(SnapshotStore):
    """In-memory (warm tier) snapshot store backed by a dict."""

    def __init__(self) -> None:
        self._store: dict[str, KVCacheSnapshot] = {}

    def save(self, snapshot: KVCacheSnapshot) -> None:
        # Ensure all tensors are on CPU
        for block in snapshot.blocks:
            if block.data.device.type != "cpu":
                block.data = block.data.cpu()
        snapshot.tier = SnapshotTier.WARM
        self._store[snapshot.metadata.snapshot_id] = snapshot

    def load(self, snapshot_id: str) -> KVCacheSnapshot | None:
        return self._store.get(snapshot_id)

    def delete(self, snapshot_id: str) -> bool:
        return self._store.pop(snapshot_id, None) is not None

    def status(self, snapshot_id: str) -> dict | None:
        snap = self._store.get(snapshot_id)
        if snap is None:
            return None
        return {
            "tier": "warm",
            "snapshot_id": snap.metadata.snapshot_id,
            "request_id": snap.metadata.request_id,
            "num_tokens": snap.metadata.num_tokens,
            "num_blocks": snap.metadata.num_blocks,
            "size_bytes": snap.metadata.estimated_size_bytes,
            "created_at": snap.metadata.created_at,
        }

    def list_snapshots(self) -> list[str]:
        return list(self._store.keys())

    @property
    def total_bytes(self) -> int:
        return sum(
            s.metadata.estimated_size_bytes for s in self._store.values()
        )


class DiskSnapshotStore(SnapshotStore):
    """Disk (cold tier) snapshot store using torch.save/load."""

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)

    def _snapshot_path(self, snapshot_id: str) -> Path:
        return self._base_dir / f"{snapshot_id}.pt"

    def _meta_path(self, snapshot_id: str) -> Path:
        return self._base_dir / f"{snapshot_id}.meta"

    def save(self, snapshot: KVCacheSnapshot) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        snapshot.tier = SnapshotTier.COLD
        # Ensure CPU tensors before saving
        for block in snapshot.blocks:
            if block.data.device.type != "cpu":
                block.data = block.data.cpu()
        # Save as a plain dict so we don't depend on the dataclass being
        # importable from the exact same module path at load time.
        meta = snapshot.metadata
        save_dict = {
            "metadata": {
                "snapshot_id": meta.snapshot_id,
                "request_id": meta.request_id,
                "num_tokens": meta.num_tokens,
                "num_blocks": meta.num_blocks,
                "block_size": meta.block_size,
                "dtype": str(meta.dtype),
                "num_kv_heads": meta.num_kv_heads,
                "head_size": meta.head_size,
                "num_layers": meta.num_layers,
                "created_at": meta.created_at,
            },
            "blocks": [
                {
                    "block_id": b.block_id,
                    "block_hash": b.block_hash,
                    "data": b.data,
                }
                for b in snapshot.blocks
            ],
        }
        torch.save(save_dict, self._snapshot_path(meta.snapshot_id))
        # Write JSON metadata sidecar for lightweight status reads
        meta_dict = {
            "tier": "cold",
            "snapshot_id": meta.snapshot_id,
            "request_id": meta.request_id,
            "num_tokens": meta.num_tokens,
            "num_blocks": meta.num_blocks,
            "size_bytes": meta.estimated_size_bytes,
            "created_at": meta.created_at,
        }
        self._meta_path(meta.snapshot_id).write_text(json.dumps(meta_dict))

    def load(self, snapshot_id: str) -> KVCacheSnapshot | None:
        path = self._snapshot_path(snapshot_id)
        if not path.exists():
            return None
        raw = torch.load(path, weights_only=False)
        md = raw["metadata"]
        # Resolve dtype string back to torch.dtype
        dtype = getattr(torch, md["dtype"].replace("torch.", ""))
        metadata = SnapshotMetadata(
            snapshot_id=md["snapshot_id"],
            request_id=md["request_id"],
            num_tokens=md["num_tokens"],
            num_blocks=md["num_blocks"],
            block_size=md["block_size"],
            dtype=dtype,
            num_kv_heads=md["num_kv_heads"],
            head_size=md["head_size"],
            num_layers=md["num_layers"],
        )
        metadata.created_at = md["created_at"]
        blocks = [
            SnapshotBlockData(
                block_id=b["block_id"],
                block_hash=b["block_hash"],
                data=b["data"],
            )
            for b in raw["blocks"]
        ]
        return KVCacheSnapshot(
            metadata=metadata, blocks=blocks, tier=SnapshotTier.COLD
        )

    def delete(self, snapshot_id: str) -> bool:
        pt_path = self._snapshot_path(snapshot_id)
        meta_path = self._meta_path(snapshot_id)
        existed = pt_path.exists()
        if pt_path.exists():
            pt_path.unlink()
        if meta_path.exists():
            meta_path.unlink()
        return existed

    def status(self, snapshot_id: str) -> dict | None:
        meta_path = self._meta_path(snapshot_id)
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text())

    def list_snapshots(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        return [p.stem for p in self._base_dir.glob("*.pt")]


class TieredSnapshotStore(SnapshotStore):
    """Two-tier store: warm (memory) + cold (disk) with automatic spill."""

    def __init__(self, warm: MemorySnapshotStore, cold: DiskSnapshotStore,
                 warm_max_bytes: int, ttl_seconds: int) -> None:
        self._warm = warm
        self._cold = cold
        self._warm_max_bytes = warm_max_bytes
        self._ttl_seconds = ttl_seconds

    def save(self, snapshot: KVCacheSnapshot) -> None:
        self._warm.save(snapshot)
        self._spill_if_needed()

    def _spill_if_needed(self) -> None:
        """Spill oldest snapshots from warm to cold when over budget."""
        while self._warm.total_bytes > self._warm_max_bytes:
            ids = self._warm.list_snapshots()
            if not ids:
                break
            # Find oldest by created_at
            oldest_id = min(
                ids,
                key=lambda sid: self._warm._store[sid].metadata.created_at,
            )
            snap = self._warm.load(oldest_id)
            if snap is not None:
                self._cold.save(snap)
                self._warm.delete(oldest_id)

    def load(self, snapshot_id: str) -> KVCacheSnapshot | None:
        snap = self._warm.load(snapshot_id)
        if snap is not None:
            return snap
        # Promote from cold to warm
        snap = self._cold.load(snapshot_id)
        if snap is not None:
            self._warm.save(snap)
            self._cold.delete(snapshot_id)
        return snap

    def delete(self, snapshot_id: str) -> bool:
        warm_del = self._warm.delete(snapshot_id)
        cold_del = self._cold.delete(snapshot_id)
        return warm_del or cold_del

    def status(self, snapshot_id: str) -> dict | None:
        s = self._warm.status(snapshot_id)
        if s is not None:
            return s
        return self._cold.status(snapshot_id)

    def list_snapshots(self) -> list[str]:
        warm_ids = set(self._warm.list_snapshots())
        cold_ids = set(self._cold.list_snapshots())
        return list(warm_ids | cold_ids)

    def cleanup_expired(self) -> int:
        """Remove snapshots older than TTL. Returns count removed."""
        now = time.time()
        removed = 0
        for sid in self.list_snapshots():
            st = self.status(sid)
            if st and now - st["created_at"] > self._ttl_seconds:
                self.delete(sid)
                removed += 1
        return removed
