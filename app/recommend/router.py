from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.recommend.service import get_recommendations

router = APIRouter(prefix="/api/recommend", tags=["recommend"])


class ReadingHistoryItem(BaseModel):
    article_id: int
    title: str
    link: str
    category: str
    keywords: list[str] = []
    embedding: list[float] | None = None
    read_at: str


class RecommendRequest(BaseModel):
    user_id: str
    grade: int
    history: list[ReadingHistoryItem]


class RecommendedArticle(BaseModel):
    title: str
    link: str
    category: str
    summary: str
    similarity_score: float
    published_at: str


class RecommendResponse(BaseModel):
    user_id: str
    count: int
    recommendations: list[RecommendedArticle]


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