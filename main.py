from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import cv2
import numpy as np
import os
import io
from pathlib import Path
from datetime import datetime

app = FastAPI()

# Tạo folder để lưu ảnh nếu chưa tồn tại
IMAGES_FOLDER = Path("images")
IMAGES_FOLDER.mkdir(exist_ok=True)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    filter_type: str = Query("blur", description="blur, sharpen, enhance, denoise"),
    intensity: int = Query(1, ge=1, le=5, description="Mức độ xử lý từ 1-5"),
):
    """
    API xử lý ảnh từ frontend.

    filter_type: blur (mịn), sharpen (sắc nét), enhance (tăng chất lượng), denoise (khử nhiễu)
    intensity: Mức độ xử lý (1-5)
    """
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

        # Áp dụng bộ lọc dựa trên loại được yêu cầu
        if filter_type == "blur":
            # Mịn ảnh bằng Gaussian Blur
            kernel_size = 3 + (intensity - 1) * 2  # 3, 5, 7, 9, 11
            processed_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

        elif filter_type == "sharpen":
            # Tăng chất lượng/sắc nét ảnh
            kernel = np.array([[-1, -1, -1], [-1, intensity + 8, -1], [-1, -1, -1]])
            processed_image = cv2.filter2D(image, -1, kernel)

        elif filter_type == "enhance":
            # Tăng cường chất lượng ảnh (CLAHE - Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0 + intensity, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            processed_image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        elif filter_type == "denoise":
            # Khử nhiễu
            processed_image = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=10 * intensity,
                hForColorComponents=10 * intensity,
                templateWindowSize=7,
                searchWindowSize=21,
            )
        else:
            return JSONResponse(
                status_code=400,
                content={"error": f"Loại filter '{filter_type}' không hỗ trợ"},
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
        return JSONResponse(
            status_code=500, content={"error": f"Lỗi xử lý ảnh: {str(e)}"}
        )


@app.post("/save-image")
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


@app.get("/get-saved-images")
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


@app.get("/download-image/{filename}")
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
