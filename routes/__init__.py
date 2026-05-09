from routes.auth import router as auth_router
from routes.image import router as img_router
from routes.ws_preview import router as ws_router

# image_router = __img_router
# ws_router = __ws_router
__all__ = ["img_router", "ws_router", "auth_router"]
