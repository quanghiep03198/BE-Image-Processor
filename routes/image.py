from fastapi import APIRouter, UploadFile, File, Query, Form
from fastapi.responses import FileResponse, JSONResponse
import uuid
from datetime import datetime
from store.image_store import image_store
from pathlib import Path

router = APIRouter(prefix="/api/image")

IMAGES_FOLDER = Path("images")


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


@router.post("/save")
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
