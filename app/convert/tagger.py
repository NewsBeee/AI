"""
기사 텍스트에서 형태소 분석(kiwipiepy)으로 단어를 추출하고 어휘 등급을 태깅.
"""
import aiomysql
from kiwipiepy import Kiwi

from app.convert.config import VOCAB_TABLE
from app.database import get_pool

_KIWI_TAG_TO_POS: dict[str, str] = {
    "NNG": "명사", "NNP": "명사", "NNB": "명사",
    "VV": "동사",
    "VA": "형용사",
    "MAG": "부사", "MAJ": "부사",
    "XR": "어근",
}

# 사전 검색 대상 품사 (조사·어미·접사 등 문법 형태소 제외)
_LOOKUP_TAGS = frozenset({
    "NNG", "NNP", "NNB",
    "VV", "VA", "VX", "VCP", "VCN",
    "MAG", "MAJ",
    "XR",
})

# 복합명사(합성어) 2-gram 탐지에 사용할 태그 (NNB 제외)
_NOUN_TAGS = frozenset({"NNG", "NNP"})

_vocab_dict: dict[str, list[dict]] | None = None
_kiwi: Kiwi | None = None


async def _load_vocab() -> dict[str, list[dict]]:
    """word → 엔트리 리스트(level 오름차순) 최초 1회 로딩"""
    global _vocab_dict
    if _vocab_dict is not None:
        return _vocab_dict

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT word, level, pos, origin, meaning, source FROM {VOCAB_TABLE}"
            )
            rows = await cur.fetchall()

    vocab: dict[str, list[dict]] = {}
    for row in rows:
        word = str(row["word"]).strip()
        entry = {
            "level": int(row["level"]) if str(row["level"]).isdigit() else 0,
            "pos": str(row.get("pos") or ""),
            "origin": str(row.get("origin") or ""),
            "meaning": str(row.get("meaning") or ""),
            "source": str(row.get("source") or ""),
        }
        vocab.setdefault(word, []).append(entry)

    for word in vocab:
        vocab[word].sort(key=lambda e: e["level"])

    _vocab_dict = vocab
    return _vocab_dict


def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
    return _kiwi


def _normalize(form: str, tag: str) -> str:
    """동사/형용사 어간에 '다'를 붙여 표제어 형태로 변환"""
    if tag in ("VV", "VA", "VX", "VCP", "VCN"):
        return form + "다"
    return form


def _best_entry(entries: list[dict], kiwi_tag: str) -> dict:
    """POS가 일치하는 엔트리 우선, 없으면 level 최솟값 반환"""
    target_pos = _KIWI_TAG_TO_POS.get(kiwi_tag, "")
    for e in entries:
        if target_pos and target_pos in e["pos"]:
            return e
    return entries[0]


def _make_tagged(word: str, entry: dict, sent_idx: int, sentence: str) -> dict:
    return {
        "word": word,
        "level": entry["level"],
        "pos": entry["pos"],
        "meaning": entry["meaning"],
        "sentence_index": sent_idx,
        "sentence": sentence,
    }


def tag_article(text: str, min_level: int = 4) -> list[dict]:
    """
    기사 텍스트에서 단어 추출 및 어휘 등급 태깅.

    - 조사·어미·접사 등 문법 형태소는 제외
    - 복합명사(NNG+NNG) 2-gram 우선 탐지
    - 기사 전체에서 동일 단어는 첫 등장만 반환

    Args:
        text: 기사 본문
        min_level: 이 등급 이상인 단어만 반환 (기본 4등급 이상)

    Returns:
        list of {word, level, pos, meaning, sentence_index, sentence}
    """
    if _vocab_dict is None:
        raise RuntimeError("vocab이 초기화되지 않았습니다. 서버 시작 시 await _load_vocab()을 호출하세요.")
    vocab = _vocab_dict
    kiwi = _get_kiwi()

    sentences = kiwi.split_into_sents(text)
    results: list[dict] = []
    seen: set[str] = set()  # 기사 전체 중복 제거

    for sent_idx, sent in enumerate(sentences):
        tokens = kiwi.tokenize(sent.text)
        i = 0

        while i < len(tokens):
            token = tokens[i]
            tag = str(token.tag)

            # 복합명사 2-gram 탐지 (NNG+NNG, NNG+NNP)
            if (
                tag in _NOUN_TAGS
                and i + 1 < len(tokens)
                and str(tokens[i + 1].tag) in _NOUN_TAGS
            ):
                bigram = token.form + tokens[i + 1].form
                if bigram in vocab:
                    if bigram not in seen:
                        entry = _best_entry(vocab[bigram], tag)
                        if entry["level"] >= min_level:
                            seen.add(bigram)
                            results.append(_make_tagged(bigram, entry, sent_idx, sent.text))
                    i += 2
                    continue  # 복합어 인식 시 개별 토큰 처리 스킵

            # 단일 토큰: 허용된 품사만 검색
            if tag in _LOOKUP_TAGS:
                normalized = _normalize(token.form, tag)
                if normalized in vocab and normalized not in seen:
                    entry = _best_entry(vocab[normalized], tag)
                    if entry["level"] >= min_level:
                        seen.add(normalized)
                        results.append(_make_tagged(normalized, entry, sent_idx, sent.text))

            i += 1

    return results
