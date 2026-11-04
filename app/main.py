"""FastAPI entry point for Zara V2."""
from fastapi import FastAPI

from app.api.chat import router as chat_router

app = FastAPI(
    title="Zara V2 API",
    version="2.0.0",
    description="Local API for the Zara V2 assistant. Run on localhost unless you add authentication.",
)
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "zara_v2"}