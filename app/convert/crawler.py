"""
httpx + BeautifulSoup 기반 신문 기사 본문 크롤러.
"""
import httpx
from bs4 import BeautifulSoup

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# 주요 언론사별 본문 컨테이너 선택자 (우선순위 순)
_SELECTORS = [
    "article",
    "#dic_area",            # 네이버 뉴스
    "#newsct_article",      # 네이버 뉴스 (구버전)
    "#articeBody",          # 조선일보
    "#article-view-content-div",  # 오마이뉴스
    ".article-body",
    ".article_body",
    ".article-content",
    "#articleBody",
    "#article-body",
    "main",
]


async def fetch_article_text(url: str) -> str:
    async with httpx.AsyncClient(
        headers=_HEADERS, follow_redirects=True, timeout=10.0
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "figure"]):
        tag.decompose()

    container = None
    for selector in _SELECTORS:
        container = soup.select_one(selector)
        if container:
            break

    if container is None:
        container = soup.body or soup

    paragraphs = [
        p.get_text(separator=" ", strip=True)
        for p in container.find_all("p")
        if len(p.get_text(strip=True)) > 30
    ]
    text = "\n".join(paragraphs)

    if not text.strip():
        text = container.get_text(separator="\n", strip=True)

    return text.strip()
