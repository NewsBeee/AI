from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.quiz.router import router as quiz_router
from app.convert.router import router as convert_router
from app.recommend.router import router as recommend_router

load_dotenv()

app = FastAPI(title="NewsBee AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 나중에 프론트 URL로 변경
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(quiz_router, prefix="/quiz", tags=["quiz"])
app.include_router(convert_router, prefix="/convert", tags=["convert"])
app.include_router(recommend_router, prefix="/recommend", tags=["recommend"])

@app.get("/")
def health_check():
    return {"status": "ok", "service": "NewsBee AI Server"}
