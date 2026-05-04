from pymongo import MongoClient
from dotenv import dotenv_values, load_dotenv
from fastapi.logger import logger
from configs import env

global mongo_client, db
try:
    mongo_client = MongoClient(
        host=[
            "127.0.0.1:27017",
            "127.0.0.1:27018",
            "127.0.0.1:27019",
        ],
        connect=True,
        replicaSet="rs0",
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=5000,
    )
    db = mongo_client.get_database(name="image_processor")
    logger.info("Connected to MongoDB")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {e}")
    mongo_client = None
    db = None
