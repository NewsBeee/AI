"""
OpenAI API를 사용해 기사 문장을 쉬운 어휘로 자연스럽게 리라이팅.
"""
import os
import re
from openai import OpenAI, OpenAIError

from app.convert.config import LLM_MODEL

_client: OpenAI | None = None

_WORD_BOUNDARY = r"가-힣A-Za-z0-9_"

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def build_convert_article_markdown(
    rewritten_text: str,
    replacement_map: dict[str, dict],
) -> str:
    """Mark changed replacement words in markdown so the frontend can render them directly."""
    if not rewritten_text or not replacement_map:
        return rewritten_text

    marked_text = rewritten_text
    replacements: set[str] = set()
    for original_word, info in replacement_map.items():
        replacement = info.get("best_replacement")
        if replacement and replacement != original_word:
            replacements.add(replacement)

    for replacement in sorted(replacements, key=len, reverse=True):
        pattern = re.compile(
            rf"(?<![{_WORD_BOUNDARY}]){re.escape(replacement)}(?![{_WORD_BOUNDARY}])"
        )
        marked_text = pattern.sub(
            lambda match: f"**{match.group(0)}**",
            marked_text,
        )

    return marked_text


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


def rewrite_and_summarize(
    original_text: str,
    replacement_map: dict[str, dict],
    target_level: int = 3,
    max_sentences: int = 5,
) -> dict:
    """
    리라이팅과 요약을 단일 OpenAI 호출로 처리.

    Returns:
        {"rewritten": str, "convert_article": str, "summary": str, "keywords": list[str]}
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

    prompt = f"""다음 기사를 두 가지 작업으로 처리해주세요.

【작업 1: 리라이팅】
초등학교 고학년~중학생이 이해할 수 있도록 쉬운 어휘로 자연스럽게 다시 쓰세요.
- 전체 어휘 수준 {target_level}등급 이하를 목표로 하세요.
- 원문의 사실·정보·논조를 정확히 유지하세요.
- 어휘만 조정하고 문장 구조는 최대한 유지하세요.

【작업 1-1: 변환 표시】
리라이팅된 기사와 같은 내용을 한 번 더 쓰되, 원문보다 쉽게 바꾼 단어와 표현만 마크다운 굵게 표시(**표시**)하세요.
- 조사, 어미, 문장부호는 굵게 표시하지 마세요.
- 원문에 그대로 있던 단어는 표시하지 마세요.
- 후보 단어를 그대로 쓰지 않았더라도 실제로 쉽게 바꾼 표현이면 표시하세요.

【바꿔야 할 어려운 단어 목록】
각 단어 옆의 [참고 후보]는 사전에서 찾은 쉬운 단어 예시입니다.
문맥에 맞는 후보가 있으면 사용하고, 없으면 직접 더 적절한 쉬운 표현을 선택하세요.

{word_section}

【작업 2: 요약】
리라이팅된 텍스트를 바탕으로 {max_sentences}문장 이내로 요약하고 핵심 어휘 5개를 추출하세요.
- 누가, 무엇을, 왜/어떻게를 중심으로 핵심 사실만 담으세요.
- 번호, 불릿 포인트 없이 자연스러운 문단으로 작성하세요.

【출력 형식】(구분자를 반드시 그대로 사용)
===리라이팅===
<리라이팅된 기사 전문>
===변환표시===
<바뀐 단어와 표현만 **굵게 표시**한 리라이팅 기사 전문>
===요약===
<요약문>
===핵심어휘===
<단어1>, <단어2>, <단어3>, <단어4>, <단어5>

【원문】
{original_text}"""

    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
    except OpenAIError as e:
        raise RuntimeError(f"rewrite_and_summarize OpenAI 오류: {e}") from e

    content = response.choices[0].message.content.strip()
    parts = re.split(r"===(?:리라이팅|변환표시|요약|핵심어휘)===", content)

    rewritten = parts[1].strip() if len(parts) > 1 else content
    convert_article = parts[2].strip() if len(parts) > 2 else build_convert_article_markdown(rewritten, replacement_map)
    summary = parts[3].strip() if len(parts) > 3 else ""
    keywords_str = parts[4].strip() if len(parts) > 4 else ""
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]

    return {
        "rewritten": rewritten,
        "convert_article": convert_article,
        "summary": summary,
        "keywords": keywords,
    }


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
