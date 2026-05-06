from typing import Optional, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.recommend.service import get_recommendations

router = APIRouter(prefix="/api/recommend", tags=["recommend"])


class ReadingHistoryItem(BaseModel):
    article_id: int
    title: str
    link: str
    category: str = ""
    keywords: List[str] = []
    embedding: Optional[List[float]] = None
    read_at: str


class RecommendRequest(BaseModel):
    user_id: str
    grade: int
    history: List[ReadingHistoryItem]


class RecommendedArticle(BaseModel):
    title: str
    link: str
    summary: str
    similarity_score: float
    published_at: str


class RecommendResponse(BaseModel):
    user_id: str
    count: int
    recommendations: List[RecommendedArticle]



@router.post("/", response_model=RecommendResponse)
async def recommend_articles(req: RecommendRequest):
    try:
        history_dicts = [item.model_dump() for item in req.history]

        recommendations = await get_recommendations(
            reading_history=history_dicts,
            user_grade=req.grade,
        )

        return RecommendResponse(
            user_id=req.user_id,
            count=len(recommendations),
            recommendations=recommendations,
        )

    except Exception as e:
        print(f"[recommend] 에러: {e}")
        raise HTTPException(status_code=500, detail=str(e))