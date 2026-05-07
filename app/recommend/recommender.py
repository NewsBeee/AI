import numpy as np

from app.recommend.config import (
    DECAY_FACTOR,
    HISTORY_WINDOW,
    RECOMMEND_COUNT,
)


def compute_interest_profile(
    history_embeddings: list[np.ndarray],
    read_times: list[str],
    decay: float = DECAY_FACTOR,
    window: int = HISTORY_WINDOW,
) -> np.ndarray:
    """
    Args:
        history_embeddings: 읽은 기사들의 임베딩 벡터 리스트
        read_times: 읽은 시각 ISO 문자열 리스트 (최신순 정렬 가정)
        decay: 감쇠율 (0~1, 1에 가까울수록 과거 기사도 반영)
        window: 최근 N개까지만 사용

    Returns:
        관심사 프로필 벡터
    """
    if not history_embeddings:
        raise ValueError("읽기 이력이 없어 관심사 프로필을 계산할 수 없습니다.")

    embeddings = history_embeddings[:window]
    times = read_times[:window]

    # 시간순 정렬
    pairs = sorted(
        zip(times, embeddings),
        key=lambda x: x[0],
        reverse=True,
    )

    # 가중치 계산
    weights = np.array([decay ** i for i in range(len(pairs))])

    stacked = np.stack([emb for _, emb in pairs])  
    weighted = stacked * weights[:, np.newaxis]   
    profile = weighted.sum(axis=0)

    # L2 정규화
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm

    return profile



def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # 두 벡터 간 코사인 유사도
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def rank_candidates(
    profile: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidates: list[dict],
    top_n: int = RECOMMEND_COUNT,
) -> list[dict]:
    """
    Args:
        profile: 사용자 관심사 프로필 벡터 (dim,)
        candidate_embeddings: 후보 기사 임베딩 (N, dim)
        candidates: 후보 기사 메타데이터 리스트
        top_n: 상위 몇 개 반환

    Returns:
        유사도 점수가 포함된 추천 기사 리스트 (내림차순)
    """
    if len(candidates) == 0:
        return []

    norms = np.linalg.norm(candidate_embeddings, axis=1)
    profile_norm = np.linalg.norm(profile)

    if profile_norm == 0:
        return []

    similarities = candidate_embeddings @ profile / (norms * profile_norm + 1e-10)

    scored = []
    for i, candidate in enumerate(candidates):
        scored.append({
            **candidate,
            "similarity_score": round(float(similarities[i]), 4),
        })

    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_n]