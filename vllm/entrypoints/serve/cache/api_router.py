# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)

router = APIRouter()


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/reset_prefix_cache")
async def reset_prefix_cache(
    raw_request: Request,
    reset_running_requests: bool = Query(default=False),
    reset_external: bool = Query(default=False),
):
    """
    Reset the local prefix cache.

    Optionally, if the query parameter `reset_external=true`
    also resets the external (connector-managed) prefix cache.

    Note that we currently do not check if the prefix cache
    is successfully reset in the API server.

    Example:
       POST /reset_prefix_cache?reset_external=true
    """
    logger.info("Resetting prefix cache...")

    await engine_client(raw_request).reset_prefix_cache(
        reset_running_requests, reset_external
    )
    return Response(status_code=200)


@router.post("/reset_mm_cache")
async def reset_mm_cache(raw_request: Request):
    """
    Reset the multi-modal cache. Note that we currently do not check if the
    multi-modal cache is successfully reset in the API server.
    """
    logger.info("Resetting multi-modal cache...")
    await engine_client(raw_request).reset_mm_cache()
    return Response(status_code=200)


@router.post("/reset_encoder_cache")
async def reset_encoder_cache(raw_request: Request):
    """
    Reset the encoder cache. Note that we currently do not check if the
    encoder cache is successfully reset in the API server.
    """
    logger.info("Resetting encoder cache...")
    await engine_client(raw_request).reset_encoder_cache()
    return Response(status_code=200)


class SnapshotRequest(BaseModel):
    request_id: str
    snapshot_id: str | None = None


class RestoreRequest(BaseModel):
    snapshot_id: str


# KV snapshot router — always attached (production-available)
kv_router = APIRouter(prefix="/kv", tags=["kv-snapshot"])


@kv_router.post("/snapshot")
async def snapshot_kv_cache(body: SnapshotRequest, raw_request: Request):
    """Snapshot the KV cache for a request."""
    try:
        result = await engine_client(raw_request).snapshot_kv_cache(
            body.request_id, body.snapshot_id
        )
        return JSONResponse(content=result)
    except ValueError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    except RuntimeError as exc:
        return JSONResponse(status_code=503, content={"error": str(exc)})


@kv_router.post("/restore")
async def restore_kv_cache(body: RestoreRequest, raw_request: Request):
    """Restore a KV cache snapshot."""
    try:
        result = await engine_client(raw_request).restore_kv_cache(
            body.snapshot_id
        )
    except ValueError as exc:
        return JSONResponse(status_code=404, content={"error": str(exc)})
    if result is None:
        return JSONResponse(status_code=404,
                            content={"error": "snapshot not found"})
    return JSONResponse(content=result)


@kv_router.delete("/{snapshot_id}")
async def delete_kv_snapshot(snapshot_id: str, raw_request: Request):
    """Delete a KV cache snapshot."""
    result = await engine_client(raw_request).delete_kv_snapshot(snapshot_id)
    return JSONResponse(content=result)


@kv_router.get("/{snapshot_id}/status")
async def get_kv_snapshot_status(snapshot_id: str, raw_request: Request):
    """Get the status of a KV cache snapshot."""
    result = await engine_client(raw_request).get_kv_snapshot_status(
        snapshot_id
    )
    if result is None:
        return JSONResponse(status_code=404,
                            content={"error": "snapshot not found"})
    return JSONResponse(content=result)


def attach_router(app: FastAPI):
    # KV snapshot endpoints are always available (production-ready)
    app.include_router(kv_router)
    # Dev-mode cache management endpoints
    if not envs.VLLM_SERVER_DEV_MODE:
        return
    app.include_router(router)
