def extract_payload_from_token(token: str) -> dict:
    import base64
    import json

    try:
        payload = base64.b64decode(token.split(".")[1] + "==").decode("utf-8")
        return json.loads(payload)
    except Exception as e:
        raise ValueError(f"Invalid token format: {e}")
