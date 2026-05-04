from fastapi import FastAPI, APIRouter, UploadFile, File, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
import cv2
from routes import ws_router, img_router, auth_router
from databases import *
from fastapi.middleware.gzip import GZipMiddleware

# Tạo folder để lưu ảnh nếu chưa tồn tại
IMAGES_FOLDER = Path("images")
IMAGES_FOLDER.mkdir(exist_ok=True)

# import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3198"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

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


app.include_router(router)
app.include_router(ws_router)
app.include_router(img_router)
app.include_router(auth_router)
