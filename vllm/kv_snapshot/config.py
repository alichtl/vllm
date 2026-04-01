"""Configuration for KV cache snapshot storage."""

from dataclasses import dataclass


@dataclass
class SnapshotConfig:
    """Configuration for KV cache snapshot storage."""
    enabled: bool = False
    snapshot_dir: str = "/var/cache/vllm/kv"
    warm_max_bytes: int = 8 * 1024 * 1024 * 1024  # 8 GB
    ttl_seconds: int = 86400  # 24 hours
