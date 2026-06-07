import uuid
from datetime import datetime
from imghdr import what as detect_image_format
from pathlib import Path

from beanie import PydanticObjectId
from fastapi import APIRouter, Depends, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from middlewares.auth_middleware import auth_middleware
from models.image import ImageModel, ImageWithUser, RenameImagePayload
from store.image_store import image_store

router = APIRouter(prefix="/api/images")

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
    print(session_id)
    return {"session_id": session_id}


@router.post("/save/{session_id}")
async def save_image(
    session_id: str,
    req=Depends(auth_middleware),
):
    """
    API lưu ảnh vào folder images trên backend.
    """
    try:
        # Lấy dữ liệu ảnh từ session store
        image_bytes = session_id and image_store.get(session_id)
        if image_bytes is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Session ID không hợp lệ hoặc đã hết hạn"},
            )

        # Phát hiện định dạng ảnh để xác định phần mở rộng và MIME type
        detected_format = detect_image_format(None, h=image_bytes)
        ext_by_format = {
            "png": ".png",
            "jpeg": ".jpeg",
            "jpg": ".jpg",
            "webp": ".webp",
            "svg": ".svg",
            "bmp": ".bmp",
        }
        mime_by_format = {
            "png": "image/png",
            "jpeg": "image/jpeg",
            "jpg": "image/jpg",
            "webp": "image/webp",
            "svg": "image/svg+xml",
            "bmp": "image/bmp",
        }

        if detected_format not in ext_by_format:
            return JSONResponse(
                status_code=400,
                content={"error": "Dữ liệu trong session không phải ảnh hợp lệ"},
            )

        # Lấy thông tin user từ request state (được set bởi auth middleware)
        user = req.state.user
        user_id = (
            user.get("id") if isinstance(user, dict) else getattr(user, "id", None)
        )
        if user_id is None:
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})

        # Tạo folder lưu ảnh nếu chưa tồn tại, và lưu ảnh vào filesystem
        IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)

        ext = ext_by_format[detected_format]
        filename = f"{session_id}{ext}"

        file_path = IMAGES_FOLDER / filename

        with open(file_path, "wb") as f:
            f.write(image_bytes)

        # Lưu thông tin ảnh vào database
        new_image = ImageModel(
            user_created=user_id,
            url=str(file_path),
            name=filename,
            mime_type=mime_by_format[detected_format],
            size=len(image_bytes),
            created_at=datetime.now(),
        )
        await new_image.insert()

        # Xóa ảnh đã upload sau khi đã lưu vào database và filesystem
        image_store.delete(session_id)

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


@router.get("/saved")
async def get_saved_images(req=Depends(auth_middleware)):
    """
    API lấy danh sách các ảnh đã lưu.
    """
    try:
        if not IMAGES_FOLDER.exists():
            return {"images": []}

        images = await ImageModel.aggregate(
            aggregation_pipeline=[
                {"$match": {"user_created": PydanticObjectId(req.state.user["id"])}},
                {
                    "$lookup": {
                        "from": "users",
                        "localField": "user_created",
                        "foreignField": "_id",
                        "as": "owner",
                    }
                },
                {"$unwind": {"path": "$owner"}},
            ],
            projection_model=ImageWithUser,
        ).to_list()

        images = [image.model_dump(mode="json") for image in images]

        return JSONResponse(status_code=200, content=images)

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi lấy danh sách ảnh: {str(e)}"}
        )


@router.get("/download/{filename}")
async def download_image(filename: str, req=Depends(auth_middleware)):
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

        in_mem_file = (
            await ImageModel.find_one(
                {
                    "url": str(file_path),
                    "user_created": PydanticObjectId(req.state.user["id"]),
                }
            )
        ).model_dump(mode="json")

        # print("in_mem_file :>> f{}", in_mem_file.model_dump(mode='json'))

        return FileResponse(
            file_path,
            media_type=(
                in_mem_file["mime_type"] if in_mem_file else "application/octet-stream"
            ),
            filename=in_mem_file["name"] if in_mem_file else filename,
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi tải ảnh: {str(e)}"}
        )


@router.patch("/rename/{image_id}")
async def rename_image(
    image_id: str,
    payload: RenameImagePayload,
    req: Request = Depends(auth_middleware),
):
    """
    API đổi tên ảnh đã lưu.
    """

    name = payload.name.strip()

    try:
        image = await ImageModel.get(PydanticObjectId(image_id))
        if not image:
            return JSONResponse(status_code=404, content="Ảnh không tồn tại")

        # Kiểm tra quyền sở hữu ảnh
        if str(image.user_created) != req.state.user["id"]:
            return JSONResponse(
                status_code=403, content="Bạn không có quyền đổi tên ảnh này"
            )

        if not name:
            return JSONResponse(status_code=400, content="Tên ảnh không hợp lệ")

        # Đổi tên file trên filesystem, luôn giữ extension gốc của ảnh
        old_file_path = Path(image.url)
        old_ext = old_file_path.suffix
        safe_name = Path(name).name
        new_stem = Path(safe_name).stem
        new_filename = f"{new_stem}{old_ext}"
        new_file_path = old_file_path.with_name(new_filename)

        if new_file_path.exists():
            return JSONResponse(status_code=400, content="Tên file mới đã tồn tại")

        old_file_path.rename(new_file_path)

        # Cập nhật thông tin ảnh trong database
        image.url = str(new_file_path)
        image.name = new_filename
        await image.save()

        return JSONResponse(status_code=200, content="Ảnh đã được đổi tên thành công")

    except Exception as e:
        return JSONResponse(status_code=500, content=f"Lỗi đổi tên ảnh: {str(e)}")


@router.delete("/{image_id}")
async def delete_image(image_id: str, req=Depends(auth_middleware)):
    """
    API xóa ảnh đã lưu.
    """
    try:
        image = await ImageModel.get(PydanticObjectId(image_id))
        if not image:
            return JSONResponse(status_code=404, content={"error": "Ảnh không tồn tại"})

        # Kiểm tra quyền sở hữu ảnh
        if str(image.user_created) != req.state.user["id"]:
            return JSONResponse(
                status_code=403, content={"error": "Bạn không có quyền xóa ảnh này"}
            )

        # Xóa file ảnh khỏi filesystem
        file_path = Path(image.url)
        if file_path.exists():
            file_path.unlink()

        # Xóa thông tin ảnh khỏi database
        await image.delete()

        return JSONResponse(
            status_code=200, content={"message": "Ảnh đã được xóa thành công"}
        )

    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi xóa ảnh: {str(e)}"}
        )
