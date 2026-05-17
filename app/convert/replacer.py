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


def _query_collection(
    word: str,
    source_pos: str,
    query_embedding: list[float],
    target_max_level: int,
    top_k: int,
    collection,
) -> list[dict]:
    """사전 계산된 임베딩으로 ChromaDB 검색 후 후보 반환."""
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
    for c in candidates:
        del c["_pos_match"]

    return candidates[:top_k]


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
    """
    model = get_model()
    collection = get_collection()

    # DB 엔트리는 "단어: 의미" 짧은 형식 → 쿼리도 같은 공간에 있어야 함
    # 문장 임베딩(0.3)과 단어+의미 임베딩(0.7)을 가중 합산 후 L2 정규화
    # → 문맥으로 다의어 구분하면서도 DB 벡터 공간과 형식을 맞춤
    texts = [f"{word}: {_first_meaning(meaning)}"]
    if sentence:
        texts.append(sentence)
    embs = model.encode(texts, convert_to_numpy=True)

    if sentence:
        combined = 0.3 * embs[1] + 0.7 * embs[0]
        combined = combined / np.linalg.norm(combined)
        query_embedding = combined.tolist()
    else:
        query_embedding = embs[0].tolist()

    return _query_collection(word, source_pos, query_embedding, target_max_level, top_k, collection)


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
    model = get_model()
    collection = get_collection()

    # 중복 단어 제거
    seen: dict[str, dict] = {}
    for item in tagged_words:
        if item["word"] not in seen:
            seen[item["word"]] = item
    items = list(seen.values())

    # 배치 인코딩: meaning 텍스트 전체를 한 번에 encode
    meaning_texts = [f"{item['word']}: {_first_meaning(item['meaning'])}" for item in items]
    meaning_embs = model.encode(meaning_texts, convert_to_numpy=True)

    # sentence가 있는 항목만 모아 한 번에 encode
    sentence_indices = [i for i, item in enumerate(items) if item.get("sentence")]
    if sentence_indices:
        sentence_embs_list = model.encode(
            [items[i]["sentence"] for i in sentence_indices], convert_to_numpy=True
        )
        sentence_embs = dict(zip(sentence_indices, sentence_embs_list))
    else:
        sentence_embs = {}

    result: dict[str, dict] = {}
    for i, item in enumerate(items):
        emb_meaning = meaning_embs[i]
        if i in sentence_embs:
            combined = 0.3 * sentence_embs[i] + 0.7 * emb_meaning
            combined = combined / np.linalg.norm(combined)
            query_embedding = combined.tolist()
        else:
            query_embedding = emb_meaning.tolist()

        candidates = _query_collection(
            item["word"], item.get("pos", ""), query_embedding, target_max_level, top_k, collection
        )
        result[item["word"]] = {
            "original_level": item["level"],
            "meaning": item["meaning"],
            "candidates": candidates,
            "best_replacement": candidates[0]["word"] if candidates else None,
        }

    return result
