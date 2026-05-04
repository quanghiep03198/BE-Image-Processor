from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPBearer
from typing import Callable
from jwt import decode
from configs import env

app = FastAPI()

security = HTTPBearer()


def auth_middleware(request: Request, token=Depends(security)):
    # Lấy token từ header Authorization
    auth_header = request.headers.get("Authorization")
    if auth_header is None or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, content={"detail": "Unauthorized"})

    token = auth_header.split(" ")[1]
    payload = decode(token, key=env("JWT_SECRET_KEY"), algorithms=["HS256"])
    request.state.user = payload
    return request
