from asyncio import *

# import asyncio
from typing import Optional

from beanie import init_beanie
from fastapi.logger import logger
from pymongo import AsyncMongoClient

from configs import env
from models.image import ImageModel
from models.revoke_token import RevokeTokenModel
from models.user import UserModel

global db


class Database:
    client: Optional[AsyncMongoClient] = None


db = Database()


async def connect_database():
    try:
        db.client = AsyncMongoClient(env("MONGO_URI"))
        logger.info("Connected to MongoDB")
        # database = db.client["image_processor"]
        # * Initialize Beanie with the database and document models
        await init_beanie(
            database=db.client.get_database("image_processor"),
            document_models=[UserModel, RevokeTokenModel, ImageModel],
        )
        logger.info("Beanie initialized with MongoDB")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e.args}")


# asyncio.run(connect_database())


async def close_database() -> None:
    if db.client:
        await db.client.close()
        logger.info("MongoDB connection closed")


def get_database(name="image_processor"):
    if db.client:
        return db.client.get_database(name)
