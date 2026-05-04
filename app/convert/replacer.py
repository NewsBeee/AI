"""
높은 등급 단어에 대해 ChromaDB 벡터 유사도 검색으로 낮은 등급 대체 후보를 찾음.
"""
import re
import numpy as np
from app.convert.embedder import get_collection, get_model


def _first_meaning(full_meaning: str) -> str:
    """
    여러 정의 중 첫 번째만 추출.
    '「1」정의1.; 「2」정의2.' → '정의1.'
    의미 오염 방지: 여러 동음이의어 중 가장 흔한 첫 번째 의미로만 쿼리.
    """
    cleaned = re.sub(r"「\d+」", "", full_meaning)
    first = cleaned.split(";")[0].strip()
    return first if first else full_meaning


def _pos_base(pos: str) -> str:
    """'명사/동사' 같은 복합 POS에서 첫 번째 품사만 반환"""
    return pos.split("/")[0].strip()


def find_replacements(
    word: str,
    meaning: str,
    sentence: str = "",
    source_pos: str = "",
    target_max_level: int = 3,
    top_k: int = 5,
) -> list[dict]:
    """
    word/meaning과 의미적으로 유사하면서 target_max_level 이하인 단어 후보 반환.
    sentence가 주어지면 문장 전체를 쿼리에 포함해 문맥 기반 검색.
    동일 품사 후보를 유사도보다 우선해 정렬.

    Args:
        word: 대체할 단어
        meaning: 해당 단어의 의미
        sentence: 단어가 등장한 원문 문장 (문맥 기반 검색에 사용)
        source_pos: 원래 단어의 품사 (POS 우선 정렬에 사용)
        target_max_level: 대체어의 최대 등급 (기본 3등급 이하)
        top_k: 반환할 후보 수

    Returns:
        list of {word, level, pos, meaning, similarity}
    """
    model = get_model()
    collection = get_collection()

    # DB 엔트리는 "단어: 의미" 짧은 형식 → 쿼리도 같은 공간에 있어야 함
    # 문장 임베딩(0.3)과 단어+의미 임베딩(0.7)을 가중 합산 후 L2 정규화
    # → 문맥으로 다의어 구분하면서도 DB 벡터 공간과 형식을 맞춤
    emb_meaning = model.encode(
        f"{word}: {_first_meaning(meaning)}", convert_to_numpy=True
    )
    if sentence:
        emb_sentence = model.encode(sentence, convert_to_numpy=True)
        combined = 0.3 * emb_sentence + 0.7 * emb_meaning
        combined = combined / np.linalg.norm(combined)
        query_embedding = combined.tolist()
    else:
        query_embedding = emb_meaning.tolist()

    fetch_k = min(top_k * 8, collection.count())
    raw_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where={"level": {"$lte": target_max_level}},
        include=["metadatas", "distances"],
    )

    if not raw_results["metadatas"] or not raw_results["metadatas"][0]:
        return []

    src_pos_base = _pos_base(source_pos)
    candidates: list[dict] = []

    for meta, dist in zip(raw_results["metadatas"][0], raw_results["distances"][0]):
        if meta["word"] == word:
            continue

        similarity = round(1.0 - float(dist), 4)
        pos_match = bool(src_pos_base and src_pos_base in meta["pos"])

        candidates.append({
            "word": meta["word"],
            "level": meta["level"],
            "pos": meta["pos"],
            "meaning": meta["meaning"],
            "similarity": similarity,
            "_pos_match": pos_match,
        })

    # 동일 품사 우선, 그 다음 유사도 내림차순
    candidates.sort(key=lambda c: (not c["_pos_match"], -c["similarity"]))

    # 내부 정렬 키 제거 후 반환
    for c in candidates:
        del c["_pos_match"]

    return candidates[:top_k]


def build_replacement_map(
    tagged_words: list[dict],
    target_max_level: int = 3,
    top_k: int = 3,
) -> dict[str, dict]:
    """
    태깅된 단어 목록에서 각 단어의 대체 후보 맵 생성.

    Returns:
        {word: {original_level, meaning, candidates, best_replacement}}
    """
    result: dict[str, dict] = {}
    for item in tagged_words:
        word = item["word"]
        if word in result:
            continue
        candidates = find_replacements(
            word=word,
            meaning=item["meaning"],
            sentence=item.get("sentence", ""),
            source_pos=item.get("pos", ""),
            target_max_level=target_max_level,
            top_k=top_k,
        )
        result[word] = {
            "original_level": item["level"],
            "meaning": item["meaning"],
            "candidates": candidates,
            "best_replacement": candidates[0]["word"] if candidates else None,
        }
    return result
