from dotenv import load_dotenv, dotenv_values
from typing import Callable, Any
from os import path
from pathlib import Path

load_dotenv()


def env(
    key: str,
    default=None,
    serializer: Callable[[str], Any] | None = None,
):
    if serializer and callable(serializer):
        return serializer(dotenv_values(".env").get(key, default))
    return dotenv_values(".env").get(key, default)
