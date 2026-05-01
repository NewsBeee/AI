from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.convert.tagger import tag_article, _get_kiwi, _load_vocab
from app.convert.replacer import build_replacement_map
from app.convert.rewriter import rewrite_article
from app.convert.summarizer import summarize_article, summarize_with_keywords
from app.convert.embedder import get_collection, get_model

router = APIRouter(prefix="/api/convert", tags=["convert"])


def preload_models() -> None:
    """서버 시작 시 무거운 모델을 미리 로딩해 첫 요청 지연 방지."""
    _get_kiwi()
    _load_vocab()
    get_model()
    get_collection()


class ConvertRequest(BaseModel):
    text: str
    target_level: int = 3    # 목표 어휘 등급 (대체어·리라이팅·요약 공통)
    min_word_level: int = 4  # 이 등급 이상인 단어를 어려운 단어로 태깅


class SummarizeRequest(BaseModel):
    text: str
    target_level: int = 3
    max_sentences: int = 5


@router.post("/process")
async def process_article(req: ConvertRequest):
    """
    기사 변환 전체 파이프라인.
    tag → replace → rewrite → summarize 순서로 실행.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어 있습니다.")

    tagged = tag_article(req.text, min_level=req.min_word_level)
    rmap = build_replacement_map(tagged, target_max_level=req.target_level)
    rewritten = rewrite_article(req.text, rmap, target_level=req.target_level)
    summary = summarize_with_keywords(rewritten, target_level=req.target_level)

    return {
        "success": True,
        "original": req.text,
        "rewritten": rewritten,
        "summary": summary["summary"],
        "keywords": summary["keywords"],
        "tagged_words": [
            {
                "word": t["word"],
                "level": t["level"],
                "meaning": t["meaning"],
                "sentence_index": t["sentence_index"],
            }
            for t in tagged
        ],
        "replacement_map": {
            word: {
                "original_level": info["original_level"],
                "best_replacement": info["best_replacement"],
                "candidates": [
                    {"word": c["word"], "level": c["level"], "similarity": c["similarity"]}
                    for c in info["candidates"]
                ],
            }
            for word, info in rmap.items()
        },
    }


@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    """기사 요약만 단독 실행."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text가 비어 있습니다.")

    result = summarize_with_keywords(
        req.text,
        target_level=req.target_level,
        max_sentences=req.max_sentences,
    )
    return {
        "success": True,
        "summary": result["summary"],
        "keywords": result["keywords"],
    }
