# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the time-sliced tenant swap layer.

Covers both the bare TenantSwitcher concurrency primitive (no FastAPI
involved) and the FastAPI middleware that wraps it. The engine is
mocked end-to-end — these are unit tests, not integration. The real
TP=2 round-trip lives in tests/kv_snapshot/test_integration.py.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.cache.tenant_switcher import (
    DEFAULT_TENANT_HEADER,
    TenantSwitcher,
    install_tenant_switcher,
    session_snapshot_id,
)


def _make_engine() -> MagicMock:
    """Engine stub with the exact set of async methods the switcher calls."""
    eng = MagicMock()
    eng.snapshot_kv_cache = AsyncMock(
        return_value={"snapshot_id": "x", "num_blocks": 4}
    )
    eng.release_all_snapshot_holds = AsyncMock(return_value={"status": "released"})
    eng.reset_prefix_cache = AsyncMock(return_value=True)
    eng.get_kv_snapshot_status = AsyncMock(return_value=None)
    eng.restore_kv_cache = AsyncMock(return_value=None)
    return eng


# ---------------------------------------------------------------------------
# TenantSwitcher
# ---------------------------------------------------------------------------


class TestTenantSwitcher:

    @pytest.mark.asyncio
    async def test_cold_start_does_not_snapshot(self):
        """First request from any tenant has no outgoing — no snapshot call."""
        eng = _make_engine()
        sw = TenantSwitcher(eng)

        async with sw.for_tenant("alice"):
            pass

        eng.snapshot_kv_cache.assert_not_called()
        eng.reset_prefix_cache.assert_called_once()
        eng.get_kv_snapshot_status.assert_called_once_with("alice_active")
        # No previous snapshot, so no restore.
        eng.restore_kv_cache.assert_not_called()
        assert sw.current_tenant == "alice"

    @pytest.mark.asyncio
    async def test_cold_start_restores_when_snapshot_exists(self):
        """If alice_active is already in the store, the cold start restores it."""
        eng = _make_engine()
        eng.get_kv_snapshot_status.return_value = {
            "tier": "warm",
            "snapshot_id": "alice_active",
        }
        eng.restore_kv_cache.return_value = {
            "snapshot_id": "alice_active",
            "num_blocks": 7,
            "num_registered_in_prefix_cache": 7,
            "tier": "warm",
            "latency_ms": 2.0,
        }
        sw = TenantSwitcher(eng)

        async with sw.for_tenant("alice"):
            pass

        eng.restore_kv_cache.assert_called_once_with("alice_active")

    @pytest.mark.asyncio
    async def test_same_tenant_concurrent_requests_do_not_swap(self):
        """Two concurrent requests for the same tenant share the engine
        without re-running the swap."""
        eng = _make_engine()
        sw = TenantSwitcher(eng)

        async def one_request():
            async with sw.for_tenant("alice"):
                await asyncio.sleep(0.01)

        await asyncio.gather(one_request(), one_request(), one_request())

        # Exactly one swap (the first/cold-start one). Subsequent same-tenant
        # entries should pass through without resetting/snapshotting.
        eng.reset_prefix_cache.assert_called_once()
        eng.snapshot_kv_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_different_tenant_triggers_swap(self):
        """When B arrives after A has been served, snapshot A → release all
        holds → reset → conditionally restore B. Order matters."""
        eng = _make_engine()
        # Track call order across the engine methods.
        call_order: list[str] = []
        eng.snapshot_kv_cache.side_effect = lambda *a, **k: (
            call_order.append("snapshot") or {"num_blocks": 5}
        )
        eng.release_all_snapshot_holds.side_effect = lambda: (
            call_order.append("release_all") or {"status": "released"}
        )
        eng.reset_prefix_cache.side_effect = lambda *a, **k: (
            call_order.append("reset") or True
        )
        eng.get_kv_snapshot_status.side_effect = lambda sid: (
            call_order.append(f"status({sid})") or None
        )

        sw = TenantSwitcher(eng)

        async with sw.for_tenant("alice"):
            pass
        async with sw.for_tenant("bob"):
            pass

        # release_all is called uniformly on every swap — on cold start
        # the hold dict is empty so it's a no-op, but the call shape is
        # the same (keeps the swap flow branch-free).
        assert call_order == [
            "release_all",
            "reset",
            "status(alice_active)",
            "snapshot",
            "release_all",
            "reset",
            "status(bob_active)",
        ]
        # Confirm the alice snapshot used session mode (request_id=None).
        snap_calls = eng.snapshot_kv_cache.call_args_list
        assert len(snap_calls) == 1
        assert snap_calls[0].args == (None, "alice_active")

    @pytest.mark.asyncio
    async def test_swap_blocked_until_in_flight_drains(self):
        """A long-running A request must complete before B's swap can fire."""
        eng = _make_engine()
        sw = TenantSwitcher(eng)

        a_release = asyncio.Event()
        b_admitted = asyncio.Event()

        async def alice_long_request():
            async with sw.for_tenant("alice"):
                await a_release.wait()

        async def bob_request():
            async with sw.for_tenant("bob"):
                b_admitted.set()

        alice_task = asyncio.create_task(alice_long_request())
        await asyncio.sleep(0.01)  # let alice acquire
        bob_task = asyncio.create_task(bob_request())
        await asyncio.sleep(0.01)  # let bob queue

        # Bob should NOT have been admitted yet.
        assert not b_admitted.is_set()
        # Tenant is still alice.
        assert sw.current_tenant == "alice"

        # Releasing alice should unblock the swap and admit bob.
        a_release.set()
        await asyncio.wait_for(asyncio.gather(alice_task, bob_task), timeout=2.0)
        assert b_admitted.is_set()
        assert sw.current_tenant == "bob"

    @pytest.mark.asyncio
    async def test_reset_failure_aborts_swap_and_releases_flag(self):
        """If reset returns False the swap raises — and a subsequent caller
        must be able to retry rather than wedge on swap_in_progress=True."""
        eng = _make_engine()
        eng.reset_prefix_cache.return_value = False
        sw = TenantSwitcher(eng)

        with pytest.raises(RuntimeError, match="reset_prefix_cache returned False"):
            async with sw.for_tenant("alice"):
                pass

        # Switcher should not be wedged: a retry can re-try the swap.
        assert sw._swap_in_progress is False
        eng.reset_prefix_cache.return_value = True
        async with sw.for_tenant("alice"):
            pass
        assert sw.current_tenant == "alice"

    @pytest.mark.asyncio
    async def test_engine_exception_during_swap_releases_flag(self):
        """If any engine call raises mid-swap, the flag must clear so the
        switcher doesn't deadlock subsequent requests."""
        eng = _make_engine()
        eng.snapshot_kv_cache.side_effect = ConnectionError("engine down")
        sw = TenantSwitcher(eng)

        # First call (cold start) succeeds since there's no snapshot to take.
        async with sw.for_tenant("alice"):
            pass

        # Second call triggers a swap; the snapshot fails. ConnectionError is
        # not RuntimeError so it propagates rather than being caught.
        with pytest.raises(ConnectionError):
            async with sw.for_tenant("bob"):
                pass

        assert sw._swap_in_progress is False
        # Recovery: replace the broken side_effect, retry.
        eng.snapshot_kv_cache.side_effect = None
        eng.snapshot_kv_cache.return_value = {"num_blocks": 0}
        async with sw.for_tenant("bob"):
            pass
        assert sw.current_tenant == "bob"

    @pytest.mark.asyncio
    async def test_empty_prefix_cache_snapshot_is_handled_gracefully(self):
        """The swap proceeds even if the outgoing tenant had nothing to
        snapshot (e.g., they got admitted but never sent any inference)."""
        eng = _make_engine()
        sw = TenantSwitcher(eng)

        async with sw.for_tenant("alice"):
            pass

        # Now make snapshot raise the "empty" runtime error and try a swap.
        eng.snapshot_kv_cache.side_effect = RuntimeError(
            "prefix cache is empty; nothing to snapshot"
        )
        async with sw.for_tenant("bob"):
            pass

        # Swap should have completed: bob is current.
        assert sw.current_tenant == "bob"

    @pytest.mark.asyncio
    async def test_empty_tenant_id_rejected(self):
        eng = _make_engine()
        sw = TenantSwitcher(eng)

        with pytest.raises(ValueError, match="non-empty"):
            sw.for_tenant("")

    @pytest.mark.asyncio
    async def test_pile_up_of_waiters_for_same_new_tenant(self):
        """Multiple bob requests piled up behind alice all admit after one swap."""
        eng = _make_engine()
        sw = TenantSwitcher(eng)
        async with sw.for_tenant("alice"):
            pass

        # Now stack three concurrent bob requests.
        # Reset the snapshot mock to return cleanly for the swap.
        eng.snapshot_kv_cache.return_value = {"num_blocks": 3}
        eng.snapshot_kv_cache.reset_mock()
        eng.reset_prefix_cache.reset_mock()

        admitted = []

        async def bob_request(tag):
            async with sw.for_tenant("bob"):
                admitted.append(tag)
                await asyncio.sleep(0.005)

        await asyncio.gather(bob_request("b1"), bob_request("b2"), bob_request("b3"))

        # Exactly one swap should have happened — first bob triggered it,
        # the rest joined as same-tenant.
        eng.snapshot_kv_cache.assert_called_once()
        eng.reset_prefix_cache.assert_called_once()
        assert set(admitted) == {"b1", "b2", "b3"}
        assert sw.current_tenant == "bob"


# ---------------------------------------------------------------------------
# Naming convention
# ---------------------------------------------------------------------------


class TestSessionSnapshotId:

    def test_format(self):
        assert session_snapshot_id("alice") == "alice_active"
        assert session_snapshot_id("bob") == "bob_active"

    def test_compatible_with_prefix_allowlist(self):
        """Names produced are guaranteed to start with '<tenant>_', so a
        deployment that allowlists '<tenant>_' for each tenant via
        VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES will pass the boundary check."""
        for tenant in ("alpha", "bravo", "charlie"):
            assert session_snapshot_id(tenant).startswith(f"{tenant}_")


# ---------------------------------------------------------------------------
# FastAPI middleware
# ---------------------------------------------------------------------------


class TestMiddleware:

    @pytest.fixture
    def app_and_engine(self):
        engine = _make_engine()
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {"choices": []}

        @app.post("/v1/completions")
        async def completions():
            return {"choices": []}

        @app.get("/health")
        async def health():
            return {"ok": True}

        install_tenant_switcher(app, engine_client=engine)
        return app, engine

    def test_request_with_header_admits_and_triggers_swap(self, app_and_engine):
        app, engine = app_and_engine
        client = TestClient(app)

        resp = client.post(
            "/v1/chat/completions", headers={DEFAULT_TENANT_HEADER: "alice"}
        )

        assert resp.status_code == 200
        # Cold start: reset called, no outgoing snapshot.
        engine.reset_prefix_cache.assert_called_once()
        engine.snapshot_kv_cache.assert_not_called()
        engine.get_kv_snapshot_status.assert_called_once_with("alice_active")

    def test_request_without_header_rejected(self, app_and_engine):
        app, engine = app_and_engine
        client = TestClient(app)

        resp = client.post("/v1/chat/completions")

        assert resp.status_code == 400
        assert "X-Tenant-Id" in resp.json()["error"]
        engine.reset_prefix_cache.assert_not_called()

    def test_non_gated_path_bypasses_switcher(self, app_and_engine):
        """A /health request shouldn't trigger a swap or require a tenant header."""
        app, engine = app_and_engine
        client = TestClient(app)

        resp = client.get("/health")

        assert resp.status_code == 200
        engine.reset_prefix_cache.assert_not_called()
        engine.snapshot_kv_cache.assert_not_called()

    def test_two_tenants_swap(self, app_and_engine):
        app, engine = app_and_engine
        client = TestClient(app)

        client.post("/v1/chat/completions", headers={DEFAULT_TENANT_HEADER: "alice"})
        client.post("/v1/chat/completions", headers={DEFAULT_TENANT_HEADER: "bob"})

        # Cold start (alice): no snapshot of "outgoing" since there isn't one.
        # Second entry (bob): snapshot alice as outgoing.
        assert engine.snapshot_kv_cache.call_count == 1
        assert engine.snapshot_kv_cache.call_args.args == (None, "alice_active")
        # release_all + reset are called uniformly per swap (cold + tenant switch),
        # so each is invoked twice across the two requests.
        assert engine.release_all_snapshot_holds.call_count == 2
        assert engine.reset_prefix_cache.call_count == 2

    def test_custom_header_name(self):
        engine = _make_engine()
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat():
            return {}

        install_tenant_switcher(app, engine_client=engine, tenant_header="X-Org")
        client = TestClient(app)

        # Wrong header → rejected.
        resp = client.post("/v1/chat/completions", headers={"X-Tenant-Id": "alice"})
        assert resp.status_code == 400

        # Right header → admitted.
        resp = client.post("/v1/chat/completions", headers={"X-Org": "alice"})
        assert resp.status_code == 200
