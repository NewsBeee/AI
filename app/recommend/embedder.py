import numpy as np

from app.core.embedder import embed_text, embed_texts


def embed_article(article: dict) -> np.ndarray:
    parts = [
        article.get("title", ""),
        article.get("summary", ""),
    ]
    text = " ".join(p for p in parts if p).strip()

    if not text:
        raise ValueError("임베딩할 텍스트가 없습니다.")

    return embed_text(text)


def embed_articles(articles: list[dict], batch_size: int = 64) -> np.ndarray:
    texts = []

    for article in articles:
        parts = [
            article.get("title", ""),
            article.get("summary", ""),
        ]
        text = " ".join(p for p in parts if p).strip()
        texts.append(text if text else article.get("title", ""))

    return embed_texts(texts, batch_size=batch_size)