from typing import Any, Callable, Literal

from dotenv import dotenv_values, load_dotenv

load_dotenv()


def env(
    key: Literal["JWT_SECRET_KEY", "MONGO_URI", "SALT_ROUNDS"],
    default=None,
    serializer: Callable[[str], Any] | None = None,
):
    if serializer and callable(serializer):
        return serializer(dotenv_values(".env").get(key, default))
    return dotenv_values(".env").get(key, default)
