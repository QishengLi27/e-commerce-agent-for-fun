from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.config import settings

app = FastAPI(
    title="E-Commerce Support Agent API",
    description="AI-powered customer service backend",
    version="0.1.0",
)

# Allow frontend to call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {"message": "E-Commerce Support Agent API", "docs": "/docs"}
