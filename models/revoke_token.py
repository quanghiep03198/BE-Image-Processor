from datetime import datetime

from beanie import Document
from pymongo import ASCENDING, IndexModel


class RevokeTokenModel(Document):
    token: str
    revoked_at: datetime

    class Settings:
        name = "revoke_tokens"
        indexes = [
            IndexModel(
                [
                    ("token", ASCENDING),
                ],
                unique=True,
            )
        ]
