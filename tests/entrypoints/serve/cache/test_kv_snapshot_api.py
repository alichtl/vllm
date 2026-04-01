"""Tests for KV cache snapshot API endpoints."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cache.api_router import kv_router


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(kv_router)
    mock_engine = MagicMock()
    mock_engine.snapshot_kv_cache = AsyncMock()
    mock_engine.restore_kv_cache = AsyncMock()
    mock_engine.delete_kv_snapshot = AsyncMock()
    mock_engine.get_kv_snapshot_status = AsyncMock()
    app.state.engine_client = mock_engine
    return app


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

    def test_snapshot_missing_request_id(self, client, engine):
        resp = client.post("/kv/snapshot", json={})
        assert resp.status_code == 422

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
