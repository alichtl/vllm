# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tenant-aware KV cache swap for time-sliced multi-tenant deployments.

Designed for the model: a single vLLM instance (typically on a TP=N GPU
pair) serves N tenants one-at-a-time, with each tenant's KV cache state
captured between turns and restored on return. The trust perimeter is
the network — middleware reads ``X-Tenant-Id`` from inbound requests
and treats it as ground truth.

Concurrency model:

- One active tenant at a time. Multiple concurrent requests *from the
  same tenant* share the engine — no serialization between them.
- A request from a *different* tenant blocks until the current tenant's
  in-flight requests drain, then atomically: snapshots outgoing →
  releases outgoing's restore-holds → resets prefix cache → restores
  incoming's session if one exists. Only one swap can be in progress
  at a time.

The TenantSwitcher object is stateful and not safe to share across
processes — there's exactly one per FastAPI app.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient

logger = init_logger(__name__)


def session_snapshot_id(tenant_id: str) -> str:
    """The snapshot id under which a tenant's session state is stored.

    Naming this consistently matters: the prefix-allowlist enforcement in
    api_router.py rejects ids that don't carry an allowed prefix, so the
    deployment must include ``<tenant>_`` in VLLM_KV_SNAPSHOT_ALLOWED_PREFIXES
    for each tenant it intends to serve.
    """
    return f"{tenant_id}_active"


class TenantSwitcher:
    """Single-active-tenant scheduler that wraps the engine for fast swaps.

    Use as::

        switcher = TenantSwitcher(engine_client)
        async with switcher.for_tenant(tenant_id):
            ... # call the engine for inference

    The async-with block holds an admission lease; releasing it lets the
    switcher know the request finished. When the lease count for the
    current tenant drops to zero and a different tenant is waiting, the
    swap fires.
    """

    def __init__(
        self,
        engine_client: "EngineClient",
        snapshot_id_for: Callable[[str], str] = session_snapshot_id,
    ) -> None:
        self._engine = engine_client
        self._snapshot_id_for = snapshot_id_for

        self._cond = asyncio.Condition()
        self._current_tenant: str | None = None
        self._in_flight = 0
        self._swap_in_progress = False

    @property
    def current_tenant(self) -> str | None:
        return self._current_tenant

    def for_tenant(self, tenant_id: str) -> "_TenantLease":
        """Return an async context manager that holds admission for tenant_id.

        The actual wait/swap happens on ``__aenter__``; ``__aexit__``
        releases the lease and may unblock a waiter.
        """
        if not tenant_id:
            raise ValueError("tenant_id must be a non-empty string")
        return _TenantLease(self, tenant_id)

    async def _acquire(self, tenant_id: str) -> None:
        # Phase 1 — wait until either we can join the current tenant
        # (no swap pending, current == us) OR we get to initiate a swap
        # to ourselves (no swap pending, in_flight == 0).
        do_swap = False
        old_tenant: str | None = None
        async with self._cond:
            while True:
                if self._swap_in_progress:
                    # Someone else is mid-swap. Wait.
                    await self._cond.wait()
                    continue
                if self._current_tenant == tenant_id:
                    # Already serving us — admit and proceed.
                    self._in_flight += 1
                    return
                # Either current is None (cold start) or it's a different
                # tenant. We need a swap. Wait for in-flight to drain
                # before we can take the swap-in-progress flag.
                if self._in_flight > 0:
                    await self._cond.wait()
                    continue
                # We're claiming the swap. Hold the flag while we await
                # the engine — no other coroutine will start one.
                self._swap_in_progress = True
                old_tenant = self._current_tenant
                do_swap = True
                break

        # Phase 2 — perform the swap without holding the lock. We're
        # safe because swap_in_progress is True; everyone else waits.
        if do_swap:
            try:
                await self._do_swap(old_tenant, tenant_id)
            except BaseException:
                async with self._cond:
                    self._swap_in_progress = False
                    self._cond.notify_all()
                raise

        # Phase 3 — install the new state and admit ourselves.
        async with self._cond:
            self._current_tenant = tenant_id
            self._in_flight = 1
            self._swap_in_progress = False
            self._cond.notify_all()

    async def _release(self) -> None:
        async with self._cond:
            self._in_flight -= 1
            if self._in_flight == 0:
                # A waiter for a different tenant may now proceed.
                self._cond.notify_all()

    async def _do_swap(
        self, old_tenant: str | None, new_tenant: str
    ) -> None:
        """Snapshot outgoing → release outgoing holds → reset → restore incoming.

        Each step is best-effort with engine errors logged: we'd rather lose
        cache continuity than block the swap entirely. The reset is the
        load-bearing isolation step — if it fails, we surface the error.
        """
        if old_tenant is not None:
            old_id = self._snapshot_id_for(old_tenant)
            try:
                result = await self._engine.snapshot_kv_cache(
                    None, old_id
                )
                logger.info(
                    "[tenant-swap] snapshotted %s: %s blocks → %s",
                    old_tenant,
                    result.get("num_blocks"),
                    old_id,
                )
            except RuntimeError as exc:
                # "prefix cache is empty; nothing to snapshot" is fine —
                # the outgoing tenant just had no cached state.
                if "empty" not in str(exc):
                    logger.warning(
                        "[tenant-swap] snapshot of %s failed: %s",
                        old_tenant,
                        exc,
                    )

        # Free every block held by every prior restore — this includes
        # the outgoing tenant's session restore plus any per-frame
        # restores still alive. Without this, reset_prefix_cache would
        # fail with blocks still allocated. On-store snapshots are
        # preserved so a future restore can re-allocate fresh blocks.
        try:
            await self._engine.release_all_snapshot_holds()
        except Exception as exc:
            logger.warning(
                "[tenant-swap] release_all_snapshot_holds failed: %s", exc
            )

        ok = await self._engine.reset_prefix_cache()
        if not ok:
            raise RuntimeError(
                "reset_prefix_cache returned False during tenant swap "
                f"({old_tenant!r} → {new_tenant!r}); some blocks remain "
                "allocated. Refusing to swap to avoid leaking KV state "
                "across tenants."
            )

        if new_tenant is not None:
            new_id = self._snapshot_id_for(new_tenant)
            status = await self._engine.get_kv_snapshot_status(new_id)
            if status is not None:
                result = await self._engine.restore_kv_cache(new_id)
                logger.info(
                    "[tenant-swap] restored %s: %s blocks (registered %s in prefix cache)",
                    new_tenant,
                    result and result.get("num_blocks"),
                    result and result.get("num_registered_in_prefix_cache"),
                )
            else:
                logger.info(
                    "[tenant-swap] no prior snapshot for %s — cold start",
                    new_tenant,
                )


class _TenantLease:
    """Admission lease for a tenant. Acquire on __aenter__, release on __aexit__."""

    __slots__ = ("_switcher", "_tenant_id", "_held")

    def __init__(self, switcher: TenantSwitcher, tenant_id: str) -> None:
        self._switcher = switcher
        self._tenant_id = tenant_id
        self._held = False

    async def __aenter__(self) -> "_TenantLease":
        await self._switcher._acquire(self._tenant_id)
        self._held = True
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._held:
            self._held = False
            await self._switcher._release()


# ---------------------------------------------------------------------------
# FastAPI integration
# ---------------------------------------------------------------------------

DEFAULT_TENANT_HEADER = "X-Tenant-Id"

# Routes the tenant gate applies to. We intentionally don't gate the
# /kv/* admin endpoints — those are operational tools and shouldn't
# trigger swaps.
DEFAULT_GATED_PATHS = (
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
)


def install_tenant_switcher(
    app,
    *,
    engine_client: "EngineClient",
    tenant_header: str = DEFAULT_TENANT_HEADER,
    gated_paths: tuple[str, ...] = DEFAULT_GATED_PATHS,
) -> TenantSwitcher:
    """Install the TenantSwitcher and the request middleware on a FastAPI app.

    Returns the switcher so the caller can attach hooks (e.g. for tests).
    Stored at ``app.state.tenant_switcher``.

    The middleware reads ``tenant_header`` from inbound requests on
    ``gated_paths``. Requests missing the header are rejected with 400.
    Requests on non-gated paths bypass the switcher entirely.
    """
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    switcher = TenantSwitcher(engine_client)
    app.state.tenant_switcher = switcher
    gated_set = set(gated_paths)

    async def _tenant_swap_middleware(request, call_next):
        path = request.url.path
        if path not in gated_set:
            return await call_next(request)

        tenant_id = request.headers.get(tenant_header)
        if not tenant_id:
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                    f"missing required {tenant_header!r} header on {path}"
                },
            )

        async with switcher.for_tenant(tenant_id):
            return await call_next(request)

    # FastAPI 0.116+/Starlette 0.40+ raise on add_middleware once the
    # ASGI middleware stack has been built. The build happens lazily on
    # first request, but install_tenant_switcher runs after init_app_state
    # has already touched the app in ways that materialize the stack on
    # current versions. Install via the user_middleware list directly and
    # invalidate the cached stack so it gets rebuilt on first call —
    # functionally equivalent to the @app.middleware("http") decorator
    # but doesn't trip the post-start guard.
    #
    # insert(0, ...) puts this middleware at the head of the user list,
    # which makes it the OUTERMOST wrapper at request time — what we want
    # so the swap completes before any of the stock middleware (CORS,
    # exception handlers, request logging) sees the request.
    app.user_middleware.insert(
        0, Middleware(BaseHTTPMiddleware, dispatch=_tenant_swap_middleware)
    )
    app.middleware_stack = None

    return switcher
