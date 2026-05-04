from fastapi import APIRouter, Form, Depends
from fastapi.exceptions import HTTPException
from fastapi.logger import logger
from fastapi.responses import JSONResponse
from databases import db
from pydantic import BaseModel, EmailStr, field_validator
from typing import Annotated
from bcrypt import hashpw, gensalt, checkpw
import jwt
from configs import env
from middlewares.auth_middleware import auth_middleware
from fastapi.security import HTTPBearer
from asyncio import *
from datetime import datetime, timezone

router = APIRouter(prefix="/api/auth")


class SigninBody(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class SignupBody(SigninBody):
    display_name: str

    @field_validator("display_name")
    @classmethod
    def display_name(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Display name should be at least 6 characters")
        return v


@router.post("/signup", status_code=201)
async def signup(item: SignupBody):
    try:
        user_collection = db.get_collection("users")

        user = user_collection.find_one({"email": item.email})

        if user is not None:
            return {"error": "User already exists"}

        hashed_password = hashpw(
            item.password.encode("utf-8"),
            gensalt(env(key="SALT_ROUNDS", default=10, serializer=int)),
        )

        user_collection.insert_one(
            {
                "email": item.email,
                "password": hashed_password,
                "display_name": item.display_name,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            },
        )

        new_user = user_collection.find_one({"email": item.email})
        return JSONResponse(
            content={
                "message": "User created successfully",
                "user": {
                    "_id": str(new_user["_id"]),
                    "email": new_user["email"],
                    "display_name": new_user["display_name"],
                },
            }
        )
    except Exception as e:
        logger.error(f"Error during signup: {e}")
        return {"error": e}


@router.post("/signin", status_code=200)
def login(item: SigninBody):
    user_collection = db.get_collection("users")

    user = user_collection.find_one({"email": item.email})

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not checkpw(item.password.encode("utf-8"), user["password"]):
        raise HTTPException(status_code=400, detail="Invalid password")

    access_token = jwt.encode(
        payload={
            "_id": str(user["_id"]),
            "email": user["email"],
            "display_name": user["display_name"],
        },
        key=env("JWT_SECRET_KEY"),
        algorithm="HS256",
    )

    return JSONResponse(
        content={
            "access_token": access_token,
            "user": {
                "_id": str(user["_id"]),
                "email": user["email"],
                "display_name": user["display_name"],
            },
        }
    )
    # return {"access_token": access_token, "user": user}


@router.get("/profile", status_code=200)
def profile(user=Depends(auth_middleware)):
    return user
