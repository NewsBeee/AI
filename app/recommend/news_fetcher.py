import re
import httpx
from typing import Optional

from app.recommend.config import (
    NAVER_CLIENT_ID,
    NAVER_CLIENT_SECRET,
    NAVER_SEARCH_URL,
    NEWS_MAX_PER_KEYWORD,
    NEWS_SORT,
)


def _clean_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).replace("&quot;", '"').replace("&amp;", "&")


async def search_news(
    query: str,
    display: int = NEWS_MAX_PER_KEYWORD,
    sort: str = NEWS_SORT,
) -> list[dict]:
    """
    Args:
        query: 검색 키워드
        display: 가져올 기사 수 (최대 100)
        sort: 정렬 기준 (date: 최신순, sim: 정확도순)

    Returns:
        기사 리스트 [{title, link, summary, published_at}, ...]
    """
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
    }
    params = {
        "query": query,
        "display": display,
        "start": 1,
        "sort": sort,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(NAVER_SEARCH_URL, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()

        articles = []
        for item in data.get("items", []):
            articles.append({
                "title": _clean_html(item.get("title", "")),
                "link": item.get("originallink", "") or item.get("link", ""),
                "summary": _clean_html(item.get("description", "")),
                "published_at": item.get("pubDate", ""),
            })

        print(f"[news] '{query}': {len(articles)}건 수집")
        return articles

    except httpx.HTTPStatusError as e:
        print(f"[news] HTTP 에러 ({query}): {e.response.status_code}")
        return []
    except httpx.RequestError as e:
        print(f"[news] 요청 실패 ({query}): {e}")
        return []


async def fetch_candidate_articles(
    keywords: list[str],
    read_urls: Optional[set[str]] = None,
) -> list[dict]:
    """
    키워드 리스트로 네이버 뉴스 검색 후 병합.
    이미 읽은 URL은 제외.

    Args:
        keywords: 검색 키워드 리스트 (사용자 이력 TF-IDF에서 추출)
        read_urls: 이미 읽은 기사 URL 집합

    Returns:
        중복 제거된 후보 기사 리스트
    """
    if read_urls is None:
        read_urls = set()

    all_articles = []
    seen_links = set()

    for keyword in keywords:
        articles = await search_news(keyword)
        for article in articles:
            link = article["link"]
            if link not in read_urls and link not in seen_links:
                seen_links.add(link)
                all_articles.append(article)

    print(f"[news] 총 후보 기사: {len(all_articles)}건 (중복·읽은 기사 제외)")
    return all_articles