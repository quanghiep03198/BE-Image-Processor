from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

import jwt
from bcrypt import checkpw, gensalt, hashpw
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import *

from configs import env
from libs.utils import extract_payload_from_token
from middlewares.auth_middleware import auth_middleware
from models.revoke_token import RevokeTokenModel
from models.user import SigninBodySchema, SignupBodySchema, UserModel

router = APIRouter(
    prefix="/api/auth",
)

security = HTTPBearer()


@router.post("/signup", status_code=201)
async def signup(item: SignupBodySchema):
    try:

        user = await UserModel.find_one({"email": item.email})

        if user is not None:
            return {"error": "User already exists"}

        hashed_password = hashpw(
            item.password.encode("utf-8"),
            gensalt(env(key="SALT_ROUNDS", default=10, serializer=int)),
        )
        new_user = UserModel(
            email=item.email,
            password=hashed_password,
            display_name=item.display_name,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        created_user = (await new_user.insert()).model_dump(mode="json")

        return JSONResponse(
            content={
                "message": "User created successfully",
                "user": {
                    "id": created_user["id"],
                    "email": created_user["email"],
                    "display_name": created_user["display_name"],
                },
            }
        )
    except Exception as e:
        logger.error(f"Error during signup: {e}")
        return {"error": e}


@router.post("/signin", status_code=200)
async def signin(
    payload: SigninBodySchema,
):

    user_doc = await UserModel.find_one({"email": payload.email})

    user = user_doc.model_dump(mode="json") if user_doc else None

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    pwd_hash = user["password"]
    if isinstance(pwd_hash, str):
        pwd_hash = pwd_hash.encode("utf-8")

    if not checkpw(payload.password.encode("utf-8"), pwd_hash):
        raise HTTPException(status_code=400, detail="Invalid password")

    now = datetime.now(timezone.utc)
    access_token = jwt.encode(
        payload={
            "id": user["id"],
            "email": user["email"],
            "display_name": user["display_name"],
            "iat": now,
            "exp": now + timedelta(minutes=60),
            "jti": str(uuid4()),
        },
        key=env("JWT_SECRET_KEY"),
        headers={
            "alg": "HS256",
            "typ": "JWT",
        },
        algorithm="HS256",
    )

    return JSONResponse(
        content={
            "access_token": access_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "display_name": user["display_name"],
            },
        }
    )


@router.post("/signout", status_code=200)
async def signout(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):

    await RevokeTokenModel(
        token=credentials.credentials, revoked_at=datetime.now()
    ).insert()
    # revoked_token = RevokeTokenModel(token=credentials.credentials)

    # revoked_token = await revoked_token.insert()
    return {"message": "User signed out successfully", "token": credentials.credentials}


@router.get("/refresh-token", status_code=200)
async def refresh_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
):
    exp_token = credentials.credentials
    payload = extract_payload_from_token(exp_token)
    user_id = payload.get("id")
    user = await UserModel.get(user_id)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    user = user.model_dump(mode="json")

    now = datetime.now(timezone.utc)
    new_token = jwt.encode(
        payload={
            "id": user["id"],
            "email": user["email"],
            "display_name": user["display_name"],
            "iat": now,
            "exp": now + timedelta(minutes=60),
            "jti": str(uuid4()),
        },
        key=env("JWT_SECRET_KEY"),
        algorithm="HS256",
    )
    return JSONResponse(content={"access_token": new_token})


@router.get("/profile", status_code=200)
async def profile(req=Depends(auth_middleware)):
    return JSONResponse(content=req.state.user)


@router.get("/users")
async def get_users():
    try:
        users = await UserModel.find_all().to_list()
        users = [user.model_dump(mode="json") for user in users]
        return JSONResponse(content=users, status_code=200)
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
