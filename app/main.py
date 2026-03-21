from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.health import router as health_router
from app.routes.prediction import router as prediction_router
from app.utils.config import get_settings
from app.utils.exceptions import (
    AppException,
    app_exception_handler,
    unhandled_exception_handler,
)
from app.utils.logger import configure_logging


configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Production-ready API for student performance prediction, NLP insights, "
        "and recommendation generation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppException, app_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(health_router)
app.include_router(prediction_router)
