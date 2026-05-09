from typing import Any

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
from pymongo import ASCENDING, IndexModel


class UserModel(Document):
    email: EmailStr
    password: str
    display_name: str

    class Settings:
        name = "users"
        indexes = [
            IndexModel(
                [("email", ASCENDING)],
                unique=True,
            )
        ]

    class Config:
        # Tự động serialize ObjectId → string
        json_encoders = {PydanticObjectId: str}


class SigninBodySchema(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_strength(cls, value: str) -> str:
        if len(value) < 6:
            raise ValueError("Password must be at least 6 characters")
        return value


class SignupBodySchema(SigninBodySchema):
    display_name: str

    @field_validator("display_name")
    @classmethod
    def display_name(cls, value: str) -> str:
        if len(value) < 6:
            raise ValueError("Display name should be at least 6 characters")
        return value


class UserInfo(BaseModel):
    id: str = Field(alias="_id")
    email: EmailStr
    display_name: str

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("id", mode="before")
    @classmethod
    def convert_id(cls, v: Any) -> str:
        return str(v)


class UserModel(Document):
    email: EmailStr
    password: str
    display_name: str

    class Settings:
        name = "users"
        indexes = [
            IndexModel(
                [("email", ASCENDING)],
                unique=True,
            )
        ]

    class Config:
        # Tự động serialize ObjectId → string
        json_encoders = {PydanticObjectId: str}
