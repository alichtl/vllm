"""Tests for KV cache snapshot API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cache.api_router import kv_router
from vllm.kv_snapshot.config import SnapshotConfig


def _build_app(allowed_id_prefixes: tuple[str, ...] = ()) -> FastAPI:
    app = FastAPI()
    app.include_router(kv_router)
    mock_engine = MagicMock()
    mock_engine.snapshot_kv_cache = AsyncMock()
    mock_engine.restore_kv_cache = AsyncMock()
    mock_engine.delete_kv_snapshot = AsyncMock()
    mock_engine.get_kv_snapshot_status = AsyncMock()
    app.state.engine_client = mock_engine
    app.state.kv_snapshot_config = SnapshotConfig(
        allowed_id_prefixes=allowed_id_prefixes)
    return app


@pytest.fixture
def app():
    return _build_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def engine(app):
    return app.state.engine_client


class TestSnapshotEndpoint:

    def test_snapshot_success(self, client, engine):
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "snap-abc",
            "size_bytes": 1024,
            "tier": "warm",
            "latency_ms": 5.0,
        }
        resp = client.post("/kv/snapshot",
                           json={"request_id": "req-1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["snapshot_id"] == "snap-abc"
        engine.snapshot_kv_cache.assert_called_once_with("req-1", None)

    def test_snapshot_custom_id(self, client, engine):
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "my-snap",
            "size_bytes": 512,
            "tier": "warm",
            "latency_ms": 3.0,
        }
        resp = client.post("/kv/snapshot",
                           json={"request_id": "req-2",
                                 "snapshot_id": "my-snap"})
        assert resp.status_code == 200
        engine.snapshot_kv_cache.assert_called_once_with("req-2", "my-snap")

    def test_snapshot_session_mode_omitted_request_id(self, client, engine):
        """Omitting request_id triggers whole-prefix-cache (session) mode."""
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "snap-sess",
            "mode": "session",
            "num_blocks": 12,
            "tier": "warm",
            "latency_ms": 4.0,
        }
        resp = client.post("/kv/snapshot", json={})
        assert resp.status_code == 200
        engine.snapshot_kv_cache.assert_called_once_with(None, None)

    def test_snapshot_session_mode_explicit_null_request_id(self, client, engine):
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "snap-sess2",
            "mode": "session",
            "num_blocks": 0,
            "tier": "warm",
            "latency_ms": 1.0,
        }
        resp = client.post(
            "/kv/snapshot", json={"request_id": None, "snapshot_id": "snap-sess2"}
        )
        assert resp.status_code == 200
        engine.snapshot_kv_cache.assert_called_once_with(None, "snap-sess2")

    def test_snapshot_empty_string_request_id_treated_as_session(self, client, engine):
        """Empty string is normalized to None — both mean session mode."""
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "snap-sess3",
            "mode": "session",
            "num_blocks": 0,
            "tier": "warm",
            "latency_ms": 1.0,
        }
        resp = client.post("/kv/snapshot", json={"request_id": ""})
        assert resp.status_code == 200
        engine.snapshot_kv_cache.assert_called_once_with(None, None)

    def test_snapshot_not_found(self, client, engine):
        engine.snapshot_kv_cache.side_effect = ValueError("request not found")
        resp = client.post("/kv/snapshot",
                           json={"request_id": "bad-id"})
        assert resp.status_code == 404

    def test_snapshot_runtime_error(self, client, engine):
        engine.snapshot_kv_cache.side_effect = RuntimeError("not enabled")
        resp = client.post("/kv/snapshot",
                           json={"request_id": "req-1"})
        assert resp.status_code == 503


class TestRestoreEndpoint:

    def test_restore_success(self, client, engine):
        engine.restore_kv_cache.return_value = {
            "status": "restored",
            "tier": "warm",
            "latency_ms": 8.0,
        }
        resp = client.post("/kv/restore",
                           json={"snapshot_id": "snap-1"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "restored"

    def test_restore_not_found(self, client, engine):
        engine.restore_kv_cache.return_value = None
        resp = client.post("/kv/restore",
                           json={"snapshot_id": "no-such"})
        assert resp.status_code == 404


class TestDeleteEndpoint:

    def test_delete_success(self, client, engine):
        engine.delete_kv_snapshot.return_value = {"status": "deleted"}
        resp = client.delete("/kv/snap-del")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


class TestStatusEndpoint:

    def test_status_found(self, client, engine):
        engine.get_kv_snapshot_status.return_value = {
            "tier": "warm",
            "snapshot_id": "snap-st",
            "num_blocks": 4,
        }
        resp = client.get("/kv/snap-st/status")
        assert resp.status_code == 200
        assert resp.json()["tier"] == "warm"

    def test_status_not_found(self, client, engine):
        engine.get_kv_snapshot_status.return_value = None
        resp = client.get("/kv/no-such/status")
        assert resp.status_code == 404


class TestPrefixEnforcement:
    """Server-side enforcement of snapshot_id prefix when configured."""

    @pytest.fixture
    def app(self):
        return _build_app(allowed_id_prefixes=("alpha_", "bravo_"))

    @pytest.fixture
    def client(self, app):
        return TestClient(app)

    @pytest.fixture
    def engine(self, app):
        return app.state.engine_client

    def test_snapshot_rejects_missing_id(self, client, engine):
        resp = client.post("/kv/snapshot", json={"request_id": "req-1"})
        assert resp.status_code == 403
        body = resp.json()
        assert "snapshot_id is required" in body["error"]
        assert body["allowed_prefixes"] == ["alpha_", "bravo_"]
        engine.snapshot_kv_cache.assert_not_called()

    def test_snapshot_rejects_unprefixed_id(self, client, engine):
        resp = client.post(
            "/kv/snapshot",
            json={"request_id": "req-1", "snapshot_id": "evil-snap"})
        assert resp.status_code == 403
        assert resp.json()["allowed_prefixes"] == ["alpha_", "bravo_"]
        engine.snapshot_kv_cache.assert_not_called()

    def test_snapshot_accepts_first_prefix(self, client, engine):
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "alpha_x",
            "size_bytes": 1,
            "tier": "warm",
            "latency_ms": 1.0,
        }
        resp = client.post(
            "/kv/snapshot",
            json={"request_id": "req-a", "snapshot_id": "alpha_x"})
        assert resp.status_code == 200
        engine.snapshot_kv_cache.assert_called_once_with("req-a", "alpha_x")

    def test_snapshot_accepts_second_prefix(self, client, engine):
        engine.snapshot_kv_cache.return_value = {
            "snapshot_id": "bravo_y",
            "size_bytes": 1,
            "tier": "warm",
            "latency_ms": 1.0,
        }
        resp = client.post(
            "/kv/snapshot",
            json={"request_id": "req-b", "snapshot_id": "bravo_y"})
        assert resp.status_code == 200
        engine.snapshot_kv_cache.assert_called_once_with("req-b", "bravo_y")

    def test_restore_rejects_unprefixed_id(self, client, engine):
        resp = client.post("/kv/restore",
                           json={"snapshot_id": "evil-snap"})
        assert resp.status_code == 403
        engine.restore_kv_cache.assert_not_called()

    def test_restore_accepts_prefixed_id(self, client, engine):
        engine.restore_kv_cache.return_value = {
            "status": "restored",
            "tier": "warm",
            "latency_ms": 1.0,
        }
        resp = client.post("/kv/restore",
                           json={"snapshot_id": "alpha_x"})
        assert resp.status_code == 200

    def test_delete_rejects_unprefixed_id(self, client, engine):
        resp = client.delete("/kv/evil-snap")
        assert resp.status_code == 403
        engine.delete_kv_snapshot.assert_not_called()

    def test_delete_accepts_prefixed_id(self, client, engine):
        engine.delete_kv_snapshot.return_value = {"status": "deleted"}
        resp = client.delete("/kv/alpha_x")
        assert resp.status_code == 200

    def test_status_rejects_unprefixed_id(self, client, engine):
        resp = client.get("/kv/evil-snap/status")
        assert resp.status_code == 403
        engine.get_kv_snapshot_status.assert_not_called()

    def test_status_accepts_prefixed_id(self, client, engine):
        engine.get_kv_snapshot_status.return_value = {
            "tier": "warm",
            "snapshot_id": "alpha_x",
            "num_blocks": 1,
        }
        resp = client.get("/kv/alpha_x/status")
        assert resp.status_code == 200


class TestSnapshotConfigFromEnv:
    """SnapshotConfig.from_env parses VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES correctly."""

    def test_unset_means_no_enforcement(self, monkeypatch):
        monkeypatch.delenv("VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES", raising=False)
        cfg = SnapshotConfig.from_env()
        assert cfg.allowed_id_prefixes == ()

    def test_empty_string_means_no_enforcement(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES", "")
        cfg = SnapshotConfig.from_env()
        assert cfg.allowed_id_prefixes == ()

    def test_single_prefix(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES", "alpha_")
        cfg = SnapshotConfig.from_env()
        assert cfg.allowed_id_prefixes == ("alpha_",)

    def test_multiple_prefixes_with_whitespace(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES",
                           "alpha_, bravo_ ,charlie_")
        cfg = SnapshotConfig.from_env()
        assert cfg.allowed_id_prefixes == ("alpha_", "bravo_", "charlie_")

    def test_dropped_empty_segments(self, monkeypatch):
        monkeypatch.setenv("VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES",
                           ",alpha_,,bravo_,")
        cfg = SnapshotConfig.from_env()
        assert cfg.allowed_id_prefixes == ("alpha_", "bravo_")
