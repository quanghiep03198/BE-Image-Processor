from fastapi import FastAPI, APIRouter, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.logger import logger
import cv2
import numpy as np
import os
import io
from pathlib import Path
from datetime import datetime
import logging

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
    return {"Hello": "World"}


# Tạo folder để lưu ảnh nếu chưa tồn tại
IMAGES_FOLDER = Path("images")
IMAGES_FOLDER.mkdir(exist_ok=True)


@router.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


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
):
    """
    API xử lý ảnh từ frontend. Hỗ trợ áp dụng nhiều filter cùng lúc.

    Ví dụ: ?blur=2&sharpen=1&enhance=3
    Thứ tự áp dụng: denoise → blur → sharpen → enhance
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
        # Đọc file ảnh
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Không thể đọc file ảnh. Vui lòng kiểm tra định dạng."
                },
            )

        processed_image = image

        # Nếu không có filter nào được bật, trả về ảnh gốc.
        has_active_filter = any(
            [
                denoise is not None and denoise > 0,
                blur is not None and blur > 0,
                sharpen is not None and sharpen > 0,
                enhance is not None and enhance > 0,
                brightness is not None and brightness != 0,
                grayscale is not None and grayscale > 0,
            ]
        )

        if not has_active_filter:
            success, encoded_image = cv2.imencode(".png", image)
            if not success:
                return JSONResponse(
                    status_code=500, content={"error": "Lỗi khi mã hóa ảnh"}
                )
            return StreamingResponse(
                io.BytesIO(encoded_image.tobytes()),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename=original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                },
            )

        # Thứ tự áp dụng: denoise -> blur -> sharpen -> enhance -> brightness -> grayscale
        if denoise is not None and denoise > 0:
            # h = 3 + denoise * 3.4 → range: 3 (nhẹ) đến ~20 (mạnh)
            h_val = 3.0 + denoise * 3.4
            processed_image = cv2.fastNlMeansDenoisingColored(
                processed_image,
                None,
                h=h_val,
                hColor=h_val,
                templateWindowSize=7,
                searchWindowSize=21,
            )

        if blur is not None and blur > 0:
            # blur=1→kernel=5, blur=5→kernel=13. Luôn là số lẻ.
            kernel_size = max(3, int(round(3 + blur * 2)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            processed_image = cv2.GaussianBlur(
                processed_image, (kernel_size, kernel_size), 0
            )

        if sharpen is not None and sharpen > 0:
            # center = 9 + sharpen → range: 9 (nhẹ) đến 14 (sắc nét mạnh)
            center = 9.0 + sharpen
            kernel = np.array([[-1, -1, -1], [-1, center, -1], [-1, -1, -1]])
            processed_image = cv2.filter2D(processed_image, -1, kernel)

        if enhance is not None and enhance > 0:
            # clipLimit = 1.0 + enhance * 0.6 → range: 1.6 (nhẹ) đến 4.0 (rõ nét)
            lab = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.0 + enhance * 0.6, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            processed_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        if brightness is not None and brightness != 0:
            offset = 255.0 * brightness
            processed_image = np.clip(
                processed_image.astype(np.float32) + offset, 0, 255
            ).astype(np.uint8)

        if grayscale is not None and grayscale > 0:
            gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            alpha = max(0.0, min(1.0, grayscale))
            processed_image = cv2.addWeighted(
                gray_bgr, alpha, processed_image, 1.0 - alpha, 0
            )

        # Mã hóa ảnh thành PNG để gửi trở lại
        success, encoded_image = cv2.imencode(".png", processed_image)
        if not success:
            return JSONResponse(
                status_code=500, content={"error": "Lỗi khi mã hóa ảnh"}
            )

        # Trả về ảnh đã xử lý
        return StreamingResponse(
            io.BytesIO(encoded_image.tobytes()),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            },
        )

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
