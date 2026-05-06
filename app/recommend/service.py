import numpy as np
from collections import Counter

from app.recommend.embedder import embed_texts, embed_articles
from app.recommend.news_fetcher import fetch_candidate_articles
from app.recommend.recommender import compute_interest_profile, rank_candidates
from app.recommend.config import (
    RECOMMEND_COUNT,
    SEARCH_KEYWORD_COUNT,
    FALLBACK_KEYWORDS,
)


def extract_search_keywords(
    reading_history: list[dict],
    top_n: int = SEARCH_KEYWORD_COUNT,
) -> list[str]:
    all_keywords = []
    for item in reading_history:
        all_keywords.extend(item.get("keywords", []))

    if not all_keywords:
        return FALLBACK_KEYWORDS

    counter = Counter(all_keywords)
    return [kw for kw, _ in counter.most_common(top_n)]


async def get_recommendations(
    reading_history: list[dict],
    user_grade: int,
    top_n: int = RECOMMEND_COUNT,
) -> list[dict]:
    if not reading_history:
        print("[service] 읽기 이력이 없어 기본 추천으로 대체")
        return await _fallback_recommend(top_n)

    history_embeddings = []
    texts_to_embed = []
    embed_indices = []

    for i, item in enumerate(reading_history):
        if item.get("embedding") is not None:
            history_embeddings.append((i, np.array(item["embedding"])))
        else:
            texts_to_embed.append(item.get("title", ""))
            embed_indices.append(i)

    # 임베딩 없는 항목 일괄 생성
    if texts_to_embed:
        print(f"[service] 이력 {len(texts_to_embed)}건 임베딩 생성")
        new_embeddings = embed_texts(texts_to_embed)
        for idx, emb in zip(embed_indices, new_embeddings):
            history_embeddings.append((idx, emb))

    # 원래 순서로 정렬
    history_embeddings.sort(key=lambda x: x[0])
    ordered_embeddings = [emb for _, emb in history_embeddings]
    read_times = [item.get("read_at", "") for item in reading_history]

    # 관심사 프로필 계산 
    profile = compute_interest_profile(ordered_embeddings, read_times)
    print("[service] 관심사 프로필 계산 완료")


    search_keywords = extract_search_keywords(reading_history)
    print(f"[service] 검색 키워드: {search_keywords}")

    # 후보 기사 수집
    read_urls = {item["link"] for item in reading_history if item.get("link")}
    candidates = await fetch_candidate_articles(search_keywords, read_urls)

    if not candidates:
        print("[service] 후보 기사가 없습니다.")
        return []

    print(f"[service] 후보 {len(candidates)}건 임베딩 중...")
    candidate_embeddings = embed_articles(candidates)

    recommendations = rank_candidates(
        profile=profile,
        candidate_embeddings=candidate_embeddings,
        candidates=candidates,
        top_n=top_n,
    )

    result = []
    for rec in recommendations:
        result.append({
            "title": rec.get("title", ""),
            "link": rec.get("link", ""),
            "summary": rec.get("summary", ""),
            "similarity_score": rec.get("similarity_score", 0.0),
            "published_at": rec.get("published_at", ""),
        })

    print(f"[service] 추천 {len(result)}건 반환")
    return result


async def _fallback_recommend(top_n: int) -> list[dict]:
    """
    읽기 이력이 없는 신규 사용자용 폴백.
    기본 키워드로 최신 기사를 가져옴.
    """
    candidates = await fetch_candidate_articles(FALLBACK_KEYWORDS)

    # 이력이 없으므로 최신순 정렬
    candidates.sort(key=lambda x: x.get("published_at", ""), reverse=True)

    result = []
    for c in candidates[:top_n]:
        result.append({
            "title": c.get("title", ""),
            "link": c.get("link", ""),
            "summary": c.get("summary", ""),
            "similarity_score": 0.0,
            "published_at": c.get("published_at", ""),
        })

    return result