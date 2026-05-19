"""
OpenAI API를 사용해 기사를 학습자 수준에 맞게 요약.
"""
import os
from openai import OpenAI, OpenAIError

from app.convert.config import LLM_MODEL

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _summarize_raw(text: str, max_sentences: int, keywords: list[str] | None = None) -> str:
    """1단계: 원문에서 핵심 내용 추출 (어휘 수준 무관)."""
    keyword_hint = ""
    if keywords:
        keyword_hint = f"\n- 다음 핵심 단어를 요약에 자연스럽게 포함하세요: {', '.join(keywords)}"

    prompt = f"""다음 기사의 핵심 내용을 {max_sentences}문장 이내로 요약해주세요.

【조건】
- 누가, 무엇을, 왜/어떻게를 중심으로 핵심 사실만 담으세요.
- 번호, 불릿 포인트 없이 자연스러운 문단으로 작성하세요.
- 요약문만 출력하고 설명은 쓰지 마세요.{keyword_hint}

【기사】
{text}

【요약】"""

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        raise RuntimeError(f"_summarize_raw OpenAI 오류: {e}") from e


def _simplify_and_extract(raw_summary: str, target_level: int) -> dict:
    """2단계: 요약문을 target_level 이하 어휘로 변환하고 핵심 어휘 추출."""
    prompt = f"""다음 요약문에서 어휘 {target_level}등급을 초과하는 어려운 단어만 골라 쉬운 표현으로 교체하고, 핵심 어휘도 추출해주세요.

【조건】
- 원문 표현을 최대한 유지하고, 꼭 필요한 어려운 단어만 선택적으로 교체하세요.
- 쉬운 단어·고유명사·숫자·날짜는 그대로 두세요.
- 문장 구조와 문장 수를 바꾸지 마세요.
- 번호, 불릿 포인트 없이 자연스러운 문단으로 작성하세요.
- 핵심 어휘: 요약 이해에 중요한 단어 5개를 쉼표로 구분해 나열하세요.

【출력 형식】(이 형식 그대로 출력)
요약: <변환된 요약문>
핵심어휘: <단어1>, <단어2>, <단어3>, <단어4>, <단어5>

【요약문】
{raw_summary}"""

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except OpenAIError as e:
        raise RuntimeError(f"_simplify_and_extract OpenAI 오류: {e}") from e

    content = response.choices[0].message.content.strip()
    summary = ""
    keywords: list[str] = []

    for line in content.splitlines():
        if line.startswith("요약:"):
            summary = line[len("요약:"):].strip()
        elif line.startswith("핵심어휘:"):
            keywords = [k.strip() for k in line[len("핵심어휘:"):].split(",")]

    return {"summary": summary or content, "keywords": keywords}


def summarize_article(
    text: str,
    target_level: int = 3,
    max_sentences: int = 5,
) -> str:
    raw = _summarize_raw(text, max_sentences)
    return _simplify_and_extract(raw, target_level)["summary"]


def summarize_with_keywords(
    text: str,
    keywords: list[str] | None = None,
    target_level: int = 3,
    max_sentences: int = 5,
) -> dict:
    """
    1단계 원문 요약 → 2단계 어휘 수준 변환 후 핵심 어휘와 함께 반환.

    Returns:
        {"summary": str, "keywords": list[str]}
    """
    raw = _summarize_raw(text, max_sentences, keywords)
    return _simplify_and_extract(raw, target_level)
