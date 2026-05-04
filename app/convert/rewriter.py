"""
OpenAI API를 사용해 기사 문장을 쉬운 어휘로 자연스럽게 리라이팅.
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


def rewrite_article(
    original_text: str,
    replacement_map: dict[str, dict],
    target_level: int = 3,
) -> str:
    """
    기사를 낮은 등급 어휘로 자연스럽게 리라이팅.

    - 어려운 단어마다 후보 목록을 제시하되, 맞지 않으면 LLM이 직접 선택하도록 유도
    - 임베딩 유사도 기반 후보의 품질이 낮을 수 있으므로 LLM 판단에 우선권을 줌

    Args:
        original_text: 원문 기사
        replacement_map: replacer.build_replacement_map()의 반환값
        target_level: 목표 어휘 등급 (기본 3등급 이하)

    Returns:
        리라이팅된 기사 텍스트
    """
    word_lines: list[str] = []
    for word, info in replacement_map.items():
        cand_words = [c["word"] for c in info.get("candidates", [])]
        cand_str = "ㆍ".join(cand_words) if cand_words else ""
        line = f"- {word}({info['original_level']}등급)"
        if cand_str:
            line += f"  [참고 후보: {cand_str}]"
        word_lines.append(line)

    word_section = "\n".join(word_lines)

    prompt = f"""다음 기사를 초등학교 고학년~중학생이 이해할 수 있도록 쉬운 어휘로 자연스럽게 다시 써주세요.

【바꿔야 할 어려운 단어 목록】
각 단어 옆의 [참고 후보]는 사전에서 찾은 쉬운 단어 예시입니다.
문맥에 맞는 후보가 있으면 사용하고, 없으면 직접 더 적절한 쉬운 표현을 선택하세요.

{word_section}

【조건】
- 원문의 사실·정보·논조를 정확히 유지하세요.
- 어휘만 조정하고 문장 구조는 최대한 유지하세요.
- 전체 어휘 수준 {target_level}등급 이하를 목표로 하세요.
- 리라이팅된 기사만 출력하고 설명은 쓰지 마세요.

【원문】
{original_text}

【리라이팅】"""

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        raise RuntimeError(f"rewrite_article OpenAI 오류: {e}") from e


def rewrite_sentence(
    sentence: str,
    difficult_words: list[str],
    replacements: dict[str, str] | None = None,
) -> str:
    """
    단일 문장 리라이팅.

    Args:
        sentence: 원문 문장
        difficult_words: 쉽게 바꿔야 할 단어 목록
        replacements: {원래 단어: 대체어} 힌트 (선택)
    """
    if not difficult_words:
        return sentence

    word_list = ", ".join(difficult_words)
    hint_text = ""
    if replacements:
        hint_text = "\n힌트: " + ", ".join(f"{k}→{v}" for k, v in replacements.items())

    prompt = f"""다음 문장에서 어려운 단어를 쉬운 말로 자연스럽게 바꿔주세요. 바뀐 문장만 출력하세요.

어려운 단어: {word_list}{hint_text}

원문: {sentence}
결과:"""

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        raise RuntimeError(f"rewrite_sentence OpenAI 오류: {e}") from e
