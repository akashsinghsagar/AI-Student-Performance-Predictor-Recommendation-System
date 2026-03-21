from typing import Any, Dict, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from app.utils.logger import get_logger


logger = get_logger(__name__)


class AppException(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        code: str = "APP_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


def app_exception_handler(_: Request, exc: AppException) -> JSONResponse:
    payload = {
        "error": {
            "code": exc.code,
            "message": exc.message,
            "details": exc.details,
        }
    }
    return JSONResponse(status_code=exc.status_code, content=payload)


def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error: %s", exc)
    payload = {
        "error": {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected server error occurred.",
            "details": {},
        }
    }
    return JSONResponse(status_code=500, content=payload)
