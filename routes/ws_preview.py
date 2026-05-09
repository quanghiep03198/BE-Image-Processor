from __future__ import annotations

import asyncio
import json

import cv2
from fastapi import APIRouter, WebSocket
from fastapi.logger import logger
from fastapi.websockets import WebSocketDisconnect
from starlette.websockets import WebSocketState

from core.processor import (
    optimize_preview_bytes,
    process_frame_cuda,
    upload_image_to_gpu,
)
from store.image_store import image_store

router = APIRouter(prefix="/ws")
gpu_semaphore = asyncio.Semaphore(4)


@router.websocket("/preview/{session_id}")
async def ws_preview(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()

    raw_bytes = image_store.get(session_id)
    if raw_bytes is None:
        await websocket.close(code=4404)
        return

    working_bytes = raw_bytes
    if len(raw_bytes) > 1 * 512 * 1024:
        try:
            working_bytes = optimize_preview_bytes(raw_bytes)
        except ValueError as exc:
            logger.warning(
                "Preview optimize failed for session %s: %s",
                session_id,
                exc,
            )

    try:
        gpu_src = upload_image_to_gpu(working_bytes)
    except ValueError:
        await websocket.close(code=4400)
        return

    params: dict = {}
    latest_preview: bytes | None = None

    try:
        async for text in websocket.iter_text():
            if websocket.client_state != WebSocketState.CONNECTED:
                return
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            if isinstance(payload.get("params"), dict):
                params = dict(payload["params"])
            elif isinstance(payload.get("data"), dict):
                params = dict(payload["data"])
            else:
                params = dict(payload)

            if not params:
                continue

            try:
                async with gpu_semaphore:
                    webp = await asyncio.to_thread(process_frame_cuda, gpu_src, params)
                latest_preview = webp
                image_store.set(session_id, webp)
                await websocket.send_bytes(webp)
            except (ValueError, RuntimeError, cv2.error) as exc:
                logger.warning(
                    "Preview frame failed for session %s: %s",
                    session_id,
                    exc,
                )
    except TypeError:
        return
    except WebSocketDisconnect:
        return

    finally:
        if latest_preview is not None:
            image_store.set(session_id, latest_preview)
        gpu_src.release()


# @router.websocket("/preview/v2/{session_id}")
# async def ws_preview(websocket: WebSocket, session_id: str) -> None:
#     await websocket.accept()

#     raw_bytes = image_store.get(session_id)
#     if raw_bytes is None:
#         await websocket.close(code=4404)
#         return

#     try:
#         gpu_src = upload_image_to_gpu(raw_bytes)
#     except ValueError:
#         await websocket.close(code=4400)
#         return

#     params: dict = {}

#     try:

#         async def receive_loop() -> None:
#             nonlocal params
#             try:
#                 async for text in websocket.iter_text():
#                     try:
#                         payload = json.loads(text)
#                     except json.JSONDecodeError:
#                         continue
#                     if isinstance(payload, dict):
#                         if isinstance(payload.get("params"), dict):
#                             params = dict(payload["params"])
#                         elif isinstance(payload.get("data"), dict):
#                             params = dict(payload["data"])
#                         else:
#                             params = dict(payload)
#             except TypeError:
#                 return
#             except WebSocketDisconnect:
#                 return

#         async def send_loop() -> None:
#             try:
#                 while True:
#                     if websocket.client_state != WebSocketState.CONNECTED:
#                         return
#                     if params:
#                         current_params = dict(params)
#                         try:
#                             async with gpu_semaphore:
#                                 jpeg = await asyncio.to_thread(
#                                     process_frame_cuda, gpu_src, current_params
#                                 )
#                             await websocket.send_bytes(jpeg)
#                         except (ValueError, RuntimeError, cv2.error) as exc:
#                             logger.warning(
#                                 "Preview frame failed for session %s: %s",
#                                 session_id,
#                                 exc,
#                             )
#                     await asyncio.sleep(1 / 30)
#             except WebSocketDisconnect:
#                 return

#         await asyncio.gather(receive_loop(), send_loop())

#     finally:
#         gpu_src.release()
#         image_store.delete(session_id)
