from fastapi import FastAPI, APIRouter, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.logger import logger
import asyncio
import numpy as np
import uuid
from pathlib import Path
from datetime import datetime
import cv2

from core.processor import process_frame_cuda, upload_image_to_gpu
from routes.ws_preview import router as ws_router
from store.image_store import image_store

# import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3198"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/api")


@app.get("/")
def read_root():
    return {
        "opencvVersion": cv2.__version__,
        "deviceCount": cv2.cuda.getCudaEnabledDeviceCount(),
        "currentDevice": (
            cv2.cuda.getDevice() if cv2.cuda.getCudaEnabledDeviceCount() > 0 else None
        ),
    }


# Tạo folder để lưu ảnh nếu chưa tồn tại
IMAGES_FOLDER = Path("images")
IMAGES_FOLDER.mkdir(exist_ok=True)


@router.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@router.post("/upload")
async def upload_image(
    file: UploadFile,
    old_session_id: str | None = Form(
        None, description="Session ID cũ để xóa ảnh đã upload trước đó"
    ),
):
    if old_session_id:
        image_store.delete(old_session_id)
    data = await file.read()
    session_id = str(uuid.uuid4())
    image_store.set(session_id, data)
    return {"session_id": session_id}


@router.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    blur: float | None = Form(
        None, ge=0, le=5, description="Gaussian Blur 0-5 (kernel 3px→11px)"
    ),
    sharpen: float | None = Form(
        None, ge=0, le=5, description="Sharpen 0-5 (nhẹ→sắc nét rõ)"
    ),
    enhance: float | None = Form(
        None, ge=0, le=5, description="CLAHE enhance 0-5 (clipLimit 1.0→4.0)"
    ),
    denoise: float | None = Form(
        None, ge=0, le=5, description="Denoise 0-5 (h 3→20, giữ chi tiết)"
    ),
    brightness: float | None = Form(
        None, ge=-1, le=1, description="Brightness -1.0 đến 1.0 (âm=tối, dương=sáng)"
    ),
    grayscale: float | None = Form(
        None, ge=0, le=1, description="Grayscale 0.0 đến 1.0 (0=màu, 1=xám hoàn toàn)"
    ),
    jpeg_quality: int = Form(
        80, ge=1, le=100, description="JPEG quality cho preview frame"
    ),
):
    """
    ! deprecated
    API tạo phiên preview realtime qua WebSocket.
    """
    logger.debug(
        "Received image: %s, blur=%s, sharpen=%s, enhance=%s, denoise=%s, brightness=%s, grayscale=%s",
        file.filename,
        blur,
        sharpen,
        enhance,
        denoise,
        brightness,
        grayscale,
    )

    try:
        contents = await file.read()
        gpu_img = upload_image_to_gpu(contents)

        params = {
            "blur": blur if blur is not None else 0,
            "sharpen": sharpen if sharpen is not None else 0,
            "enhance": enhance if enhance is not None else 0,
            "denoise": denoise if denoise is not None else 0,
            "brightness": brightness if brightness is not None else 0,
            "grayscale": grayscale if grayscale is not None else 0,
            "jpeg_quality": jpeg_quality,
        }

        preview_frame = await asyncio.to_thread(process_frame_cuda, gpu_img, params)

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Không thể đọc file ảnh. Vui lòng kiểm tra định dạng."
                },
            )

        session_id = uuid.uuid4().hex
        image_store.set(session_id, contents)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Preview session created",
                "session_id": session_id,
                "preview_ws_path": f"/ws/preview/{session_id}",
                "image": {
                    "filename": file.filename,
                    "width": int(image.shape[1]),
                    "height": int(image.shape[0]),
                },
                "initial_preview": {
                    "size_bytes": len(preview_frame),
                    "content_type": "image/jpeg",
                    "jpeg_quality": jpeg_quality,
                },
                "params": params,
            },
        )

    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi xử lý ảnh: {str(e)}"}
        )


@router.post("/save-image")
async def save_image(
    file: UploadFile = File(...),
    filename: str = Query(
        None, description="Tên file (optional, nếu không cung cấp sẽ tự tạo)"
    ),
):
    """
    API lưu ảnh vào folder images trên backend.
    """
    try:
        contents = await file.read()

        # Tạo tên file nếu không cung cấp
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = Path(file.filename).suffix or ".png"
            filename = f"image_{timestamp}{ext}"
        else:
            # Đảm bảo filename có phần mở rộng
            if not Path(filename).suffix:
                filename += ".png"

        # Kiểm tra an toàn - chỉ cho phép các ký tự an toàn trong tên file
        if "/" in filename or "\\" in filename:
            return JSONResponse(
                status_code=400, content={"error": "Tên file không hợp lệ"}
            )

        # Đường dẫn đầy đủ để lưu file
        file_path = IMAGES_FOLDER / filename

        # Ghi file
        with open(file_path, "wb") as f:
            f.write(contents)

        return JSONResponse(
            status_code=200,
            content={
                "message": "Ảnh đã được lưu thành công",
                "filename": filename,
                "path": str(file_path),
            },
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi lưu ảnh: {str(e)}"}
        )


@router.get("/get-saved-images")
async def get_saved_images():
    """
    API lấy danh sách các ảnh đã lưu.
    """
    try:
        if not IMAGES_FOLDER.exists():
            return {"images": []}

        images = []
        for file in IMAGES_FOLDER.iterdir():
            if file.is_file() and file.suffix.lower() in [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".bmp",
            ]:
                images.append(
                    {
                        "filename": file.name,
                        "size": file.stat().st_size,
                        "created": datetime.fromtimestamp(
                            file.stat().st_ctime
                        ).isoformat(),
                    }
                )

        return {"images": sorted(images, key=lambda x: x["created"], reverse=True)}

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi lấy danh sách ảnh: {str(e)}"}
        )


@router.get("/download-image/{filename}")
async def download_image(filename: str):
    """
    API tải ảnh đã lưu từ backend.
    """
    try:
        # Kiểm tra an toàn - chỉ cho phép tên file an toàn
        if "/" in filename or "\\" in filename or ".." in filename:
            return JSONResponse(
                status_code=400, content={"error": "Tên file không hợp lệ"}
            )

        file_path = IMAGES_FOLDER / filename

        if not file_path.exists():
            return JSONResponse(
                status_code=404, content={"error": "File không tồn tại"}
            )

        return FileResponse(file_path, media_type="image/png")

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi tải ảnh: {str(e)}"}
        )


app.include_router(router)
app.include_router(ws_router)
