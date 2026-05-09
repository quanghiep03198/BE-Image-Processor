from datetime import datetime
from typing import Any, Optional

from beanie import Document, PydanticObjectId
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import ASCENDING, DESCENDING, IndexModel

from models.user import UserInfo


class ImageModel(Document):
    user_created: PydanticObjectId
    url: str
    name: str
    mime_type: str
    size: int
    created_at: datetime

    class Settings:
        name = "images"
        indexes = [
            IndexModel([("user", ASCENDING), ("created_at", DESCENDING)]),
        ]

    class Config:
        json_encoders = {PydanticObjectId: str}

class ImageWithUser(BaseModel):
    id: str = Field(alias="_id")
    name: str
    size: int
    url: str
    created_at: datetime
    mime_type: str
    user_created: Optional[str]
    owner: Optional[UserInfo] = None

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("id", "user_created", mode="before")
    @classmethod
    def convert_objectid(cls, v: Any) -> str:
        return str(v)

class RenameImagePayload(BaseModel):
    name: str