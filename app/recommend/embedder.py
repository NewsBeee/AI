import numpy as np
from sentence_transformers import SentenceTransformer

from app.recommend.config import EMBED_MODEL_NAME

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[recommend/embedder] Loading model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    # 단일 텍스트 임베딩
    model = get_model()
    return model.encode(text, convert_to_numpy=True)


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    # 복수 텍스트 일괄 임베딩
    model = get_model()
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True)


def embed_article(article: dict) -> np.ndarray:
    parts = [
        article.get("title", ""),
        article.get("summary", ""),
        article.get("content", ""),
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
            article.get("content", ""),
        ]
        text = " ".join(p for p in parts if p).strip()
        texts.append(text if text else article.get("title", ""))

    return embed_texts(texts, batch_size=batch_size)