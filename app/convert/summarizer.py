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


def summarize_article(
    text: str,
    target_level: int = 3,
    max_sentences: int = 5,
) -> str:
    """
    기사를 학습자 수준에 맞게 요약.

    Args:
        text: 기사 원문 또는 리라이팅된 본문
        target_level: 목표 어휘 등급 (기본 3등급 이하)
        max_sentences: 요약 최대 문장 수

    Returns:
        요약 텍스트
    """
    prompt = f"""다음 기사를 초등학교 고학년~중학생이 이해할 수 있도록 쉽게 요약해주세요.

【조건】
- {max_sentences}문장 이내의 자연스러운 문단으로 작성하세요.
- 어휘 {target_level}등급 이하의 쉬운 표현을 사용하세요.
- 기사의 핵심 내용(누가, 무엇을, 왜)을 담아주세요.
- 번호, 불릿 포인트 없이 흐르는 문장으로 작성하세요.
- 요약문만 출력하고 설명은 쓰지 마세요.

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
        raise RuntimeError(f"summarize_article OpenAI 오류: {e}") from e


def summarize_with_keywords(
    text: str,
    keywords: list[str] | None = None,
    target_level: int = 3,
    max_sentences: int = 5,
) -> dict:
    """
    요약 + 핵심 어휘 목록 함께 반환.

    Returns:
        {"summary": str, "keywords": list[str]}
    """
    keyword_hint = ""
    if keywords:
        keyword_hint = f"\n- 다음 핵심 어휘를 요약에 자연스럽게 포함하세요: {', '.join(keywords)}"

    prompt = f"""다음 기사를 초등학교 고학년~중학생이 이해할 수 있도록 쉽게 요약하고, 핵심 어휘도 추출해주세요.

【조건】
- 요약: {max_sentences}문장 이내, 어휘 {target_level}등급 이하의 쉬운 표현{keyword_hint}
- 핵심 어휘: 기사 이해에 중요한 단어 5개를 쉼표로 구분해 나열

【출력 형식】(이 형식 그대로 출력)
요약: <요약문>
핵심어휘: <단어1>, <단어2>, <단어3>, <단어4>, <단어5>

【기사】
{text}"""

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except OpenAIError as e:
        raise RuntimeError(f"summarize_with_keywords OpenAI 오류: {e}") from e

    content = response.choices[0].message.content.strip()
    summary = ""
    extracted_keywords: list[str] = []

    for line in content.splitlines():
        if line.startswith("요약:"):
            summary = line[len("요약:"):].strip()
        elif line.startswith("핵심어휘:"):
            extracted_keywords = [k.strip() for k in line[len("핵심어휘:"):].split(",")]

    return {"summary": summary or content, "keywords": extracted_keywords}
