import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from app import config
from app.routes import face as face_router
from app.services import arcface as arc_svc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    arc_svc.init_model()
    logger.info("SIABDI Face Service siap.")
    yield
    logger.info("SIABDI Face Service dihentikan.")


app = FastAPI(
    title="SIABDI Face Service",
    description="ArcFace verification service untuk SIABDI. Internal — jangan expose ke publik.",
    version="1.0.0",
    lifespan=lifespan,
    # Matikan docs di production jika tidak diperlukan
    # docs_url=None, redoc_url=None,
)

app.include_router(face_router.router)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.FACE_SERVICE_HOST,
        port=config.FACE_SERVICE_PORT,
        reload=False,
        workers=1,  # InsightFace model tidak thread-safe di multi-worker tanpa forking
    )
