from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from app.quiz.router import router as quiz_router
from app.quiz.item_pool import load_item_pool
from app.database import init_db, close_db
from app.convert.router import router as convert_router, preload_models
from app.recommend.router import router as recommend_router
from app.convert.embedder import build_embeddings

load_dotenv()

@asynccontextmanager
async def lifespan(_app):
    await init_db()
    load_item_pool()
    await preload_models()
    await build_embeddings()
    yield
    await close_db()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz_router)
app.include_router(convert_router)
app.include_router(recommend_router)

@app.get("/")
def root():
    return {"message": "NewsBee AI Server Running"}
