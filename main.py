from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, Request
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse

from databases.mongo import close_database, connect_database
from exceptions import AppException
from routes import auth_router, img_router, ws_router

try:
    # Tạo folder để lưu ảnh nếu chưa tồn tại
    IMAGES_FOLDER = Path("images")
    IMAGES_FOLDER.mkdir(exist_ok=True)

    # import cv2

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cv2.cuda.printShortCudaDeviceInfo(cv2.cuda.getDevice())
        await connect_database()
        yield
        await close_database()

    app = FastAPI(
        lifespan=lifespan,
        title="Image Processing API",
        description="API for image processing tasks",
    )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3198", "http://localhost:1205"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

    @app.get("/")
    def read_root():

        return {
            "opencv_version": cv2.__version__,
            "cuda_device_count": cv2.cuda.getCudaEnabledDeviceCount(),
            "current_cude_device": (
                cv2.cuda.getDevice()
                if cv2.cuda.getCudaEnabledDeviceCount() > 0
                else None
            ),
        }

    @app.get("/images/{filename}")
    async def show_image(
        filename: str,
    ):
        """
        API hiển thị ảnh đã lưu từ backend.
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
                status_code=500, content={"error": f"Lỗi hiển thị ảnh: {str(e)}"}
            )

    app.include_router(ws_router)
    app.include_router(img_router)
    app.include_router(auth_router)
except Exception as e:
    logger.error(f"Error during app initialization: {e}")

except KeyboardInterrupt:
    print("Shutting down application...")
