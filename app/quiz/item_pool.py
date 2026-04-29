"""
문항 풀 로드 및 관리
현재: JSON 파일에서 로드
추후: DB 연동으로 변경
"""

import json
import os
from typing import Optional

_item_pool: list = []


def load_item_pool(path: str = None):
    """서버 시작 시 문항 풀 로드"""
    global _item_pool

    if path is None:
        # 기본 경로: 프로젝트 루트의 data/quiz_item_pool.json
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(base_dir, "data", "quiz_item_pool.json")

    with open(path, "r", encoding="utf-8") as f:
        _item_pool = json.load(f)

    print(f"문항 풀 로드 완료: {len(_item_pool)}개")

    # 등급별 통계
    stats = {}
    for item in _item_pool:
        grade = item["grade"]
        item_type = item["type"]
        key = f"{grade}등급-{item_type}"
        stats[key] = stats.get(key, 0) + 1

    for k in sorted(stats.keys()):
        print(f"   {k}: {stats[k]}개")


def get_all_items() -> list:
    """전체 문항 목록 반환"""
    return _item_pool


def get_item_by_id(item_id: str) -> Optional[dict]:
    """ID로 문항 조회"""
    for item in _item_pool:
        if item["id"] == item_id:
            return item
    return None