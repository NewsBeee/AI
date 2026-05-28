import asyncio
import time

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.convert.crawler import fetch_article_text
from app.convert.tagger import tag_article, _get_kiwi, _load_vocab
from app.convert.replacer import build_replacement_map
from app.convert.rewriter import build_convert_article_markdown, rewrite_and_summarize
from app.convert.summarizer import summarize_with_keywords
from app.convert.categorizer import classify_category
from app.convert.embedder import get_collection, get_model

router = APIRouter(prefix="/api/convert", tags=["convert"])


async def preload_models() -> None:
    """서버 시작 시 무거운 모델을 미리 로딩해 첫 요청 지연 방지."""
    _get_kiwi()
    await _load_vocab()
    get_model()
    get_collection()


class ConvertRequest(BaseModel):
    text: str = ""
    url: str | None = None
    target_level: int = 3    # 목표 어휘 등급 (대체어·리라이팅·요약 공통)
    min_word_level: int = 4  # 이 등급 이상인 단어를 어려운 단어로 태깅


class SummarizeRequest(BaseModel):
    text: str
    target_level: int = 3
    max_sentences: int = 5


@router.post("/process")
async def process_article(req: ConvertRequest):
    """기사 변환 전체 파이프라인: tag → replace → rewrite → summarize"""
    if req.url:
        try:
            text = await fetch_article_text(req.url)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=502, detail=f"URL 요청 실패: {e.response.status_code}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"URL 접근 오류: {e}")
        if not text:
            raise HTTPException(status_code=422, detail="본문 텍스트를 추출할 수 없습니다.")
    elif req.text.strip():
        text = req.text
    else:
        raise HTTPException(status_code=400, detail="text 또는 url 중 하나는 필수입니다.")

    try:
        t0 = time.perf_counter()

        tagged = await asyncio.to_thread(tag_article, text, req.min_word_level)
        t1 = time.perf_counter()
        print(f"[TIMING] tag_article: {t1 - t0:.2f}s  ({len(tagged)} words tagged)")

        rmap = await asyncio.to_thread(build_replacement_map, tagged, req.target_level)
        t2 = time.perf_counter()
        print(f"[TIMING] build_replacement_map: {t2 - t1:.2f}s  ({len(rmap)} words in rmap)")

        rewrite_result, category = await asyncio.gather(
            asyncio.to_thread(rewrite_and_summarize, text, rmap, req.target_level),
            asyncio.to_thread(classify_category, text),
        )
        t3 = time.perf_counter()
        print(f"[TIMING] rewrite_and_summarize: {t3 - t2:.2f}s")
        print(f"[TIMING] total: {t3 - t0:.2f}s")

    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {
        "success": True,
        "original": text,
        "rewritten": rewrite_result.get("convert_article")
        or build_convert_article_markdown(rewrite_result["rewritten"], rmap),
        "summary": rewrite_result["summary"],
        "keywords": rewrite_result["keywords"],
        "category": category,
        "tagged_words": [
            {
                "word": t["word"],
                "level": t["level"],
                "meaning": t["meaning"],
                "sentence_index": t["sentence_index"],
            }
            for t in tagged
            if (rep := rmap.get(t["word"], {}).get("best_replacement"))
            and rep != t["word"]
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

    try:
        result = await asyncio.to_thread(
            summarize_with_keywords, req.text, None, req.target_level, req.max_sentences
        )
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"data": result["summary"]}
