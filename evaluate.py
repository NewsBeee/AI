"""
기사 변환 및 요약 성능 평가 스크립트.

평가 항목:
  변환: 치환 정확도, 치환 누락률, BERTScore F1 (원문 vs rewritten)
  요약: ROUGE-1/2/L, BERTScore F1 (원문 vs summary)

사용법:
  python evaluate.py [--base-url http://localhost:8000] [--target-level 3]
"""

import argparse
import json
import sys

import httpx
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

BASE_URL = "http://localhost:8000"
TARGET_LEVEL = 3

URLS = [
    "https://n.news.naver.com/mnews/article/001/0016083783?rc=N&ntype=RANKING",
    "https://n.news.naver.com/mnews/article/001/0016083121?rc=N&ntype=RANKING",
    "https://n.news.naver.com/mnews/article/421/0008951580",
    "https://n.news.naver.com/mnews/article/029/0003027213",
    "https://n.news.naver.com/mnews/article/001/0016083974",
]


# ── 변환 평가 ──────────────────────────────────────────────────────────────────

def calc_conversion_metrics(process_resp: dict, target_level: int) -> dict:
    rmap = process_resp.get("replacement_map", {})
    total = len(rmap)
    if total == 0:
        return {"substitution_accuracy": None, "missing_rate": None, "total_tagged": 0}

    correctly_replaced = sum(
        1 for info in rmap.values()
        if info["best_replacement"]
        and info["candidates"]
        and info["candidates"][0]["level"] <= target_level
    )
    missing = sum(1 for info in rmap.values() if not info["best_replacement"])

    return {
        "substitution_accuracy": correctly_replaced / total,
        "missing_rate": missing / total,
        "total_tagged": total,
    }


# ── 요약 평가 ──────────────────────────────────────────────────────────────────

def calc_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": round(scores["rouge1"].fmeasure, 4),
        "rouge2_f": round(scores["rouge2"].fmeasure, 4),
        "rougeL_f": round(scores["rougeL"].fmeasure, 4),
    }


def calc_bertscore(references: list[str], hypotheses: list[str]) -> list[float]:
    _, _, F1 = bert_score_fn(hypotheses, references, lang="ko", verbose=False)
    return [round(f.item(), 4) for f in F1]


# ── API 호출 ───────────────────────────────────────────────────────────────────

def call_process(base_url: str, url: str, target_level: int) -> dict:
    resp = httpx.post(
        f"{base_url}/api/convert/process",
        json={"url": url, "target_level": target_level},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def call_summarize(base_url: str, text: str, target_level: int) -> dict:
    resp = httpx.post(
        f"{base_url}/api/convert/summarize",
        json={"text": text, "target_level": target_level},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ── 출력 ──────────────────────────────────────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def print_metrics(label: str, metrics: dict) -> None:
    print(f"\n[{label}]")
    for k, v in metrics.items():
        if v is None:
            print(f"  {k}: N/A")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--target-level", type=int, default=TARGET_LEVEL)
    args = parser.parse_args()

    conversion_results = []
    summarization_results = []

    for i, url in enumerate(URLS, 1):
        print_section(f"기사 {i}/{len(URLS)}: {url}")

        # ── /process 호출
        try:
            proc = call_process(args.base_url, url, args.target_level)
        except Exception as e:
            print(f"  [ERROR] /process 실패: {e}")
            continue

        original = proc.get("original", "")
        rewritten = proc.get("rewritten", "")

        # ── 변환 평가
        conv_metrics = calc_conversion_metrics(proc, args.target_level)
        print_metrics("변환 지표", conv_metrics)
        conversion_results.append(conv_metrics)

        # ── /summarize 호출
        try:
            summ = call_summarize(args.base_url, original, args.target_level)
            summary = summ.get("data", "")
        except Exception as e:
            print(f"  [ERROR] /summarize 실패: {e}")
            summary = ""

        # ── BERTScore (변환 + 요약 묶어서 배치 계산)
        bert_refs = [original, original]
        bert_hyps = [rewritten, summary] if summary else [rewritten]
        bert_scores = calc_bertscore(bert_refs[:len(bert_hyps)], bert_hyps)

        conv_bert = bert_scores[0]
        conv_metrics["bertscore_f1"] = conv_bert
        print(f"  bertscore_f1 (원문 vs rewritten): {conv_bert:.4f}")

        if summary:
            rouge_metrics = calc_rouge(original, summary)
            summ_bert = bert_scores[1] if len(bert_scores) > 1 else None
            summ_metrics = {**rouge_metrics, "bertscore_f1": summ_bert}
            print_metrics("요약 지표", summ_metrics)
            summarization_results.append(summ_metrics)

    # ── 전체 평균
    print_section("전체 평균")

    if conversion_results:
        valid_conv = [r for r in conversion_results if r["substitution_accuracy"] is not None]
        if valid_conv:
            print("\n[변환 평균]")
            for key in ("substitution_accuracy", "missing_rate", "bertscore_f1"):
                values = [r[key] for r in valid_conv if r.get(key) is not None]
                if values:
                    print(f"  {key}: {sum(values) / len(values):.4f}")

    if summarization_results:
        print("\n[요약 평균]")
        for key in ("rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1"):
            values = [r[key] for r in summarization_results if r.get(key) is not None]
            if values:
                print(f"  {key}: {sum(values) / len(values):.4f}")


if __name__ == "__main__":
    main()
