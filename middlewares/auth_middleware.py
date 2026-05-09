from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import ExpiredSignatureError, InvalidTokenError, decode

from configs import env
from models.revoke_token import RevokeTokenModel

app = FastAPI()

security = HTTPBearer()


async def auth_middleware(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    # Lấy token từ header Authorization
    try:
        if credentials is None:
            raise HTTPException(
                status_code=403, content={"message": "Invalid credentials"}
            )

        token = credentials.credentials

        # Kiểm tra xem token có bị thu hồi hay không
        revoked_token = await RevokeTokenModel.find_one({"token": token})
        if revoked_token is not None:
            raise HTTPException(status_code=403, detail="Token has been revoked")

        payload = decode(token, key=env("JWT_SECRET_KEY"), algorithms="HS256")
        request.state.user = payload
        return request
    except Exception as e:
        if isinstance(e, ExpiredSignatureError):
            raise HTTPException(status_code=401, detail="Token has expired")
        elif isinstance(e, InvalidTokenError):
            raise HTTPException(status_code=403, detail="Invalid token")
        else:
            raise HTTPException(status_code=500, detail="Authentication failed")
