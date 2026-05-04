from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from app.quiz.router import router as quiz_router
from app.database import init_db, close_db

load_dotenv()

@asynccontextmanager
async def lifespan(app):
    await init_db()
    yield
    await close_db()

app = FastAPI(lifespan=lifespan)

# CORS 설정 (프론트 연결용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 연결
app.include_router(quiz_router)

# 기본 라우트 (테스트용)
@app.get("/")
def root():
    return {"message": "NewsBee AI Server Running"}