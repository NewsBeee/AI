import math
import random
from typing import Optional


# 7등급 logit 매핑
GRADE_LOGIT = {1: -3.0, 2: -2.0, 3: -1.0, 4: 0.0, 5: 1.0, 6: 2.0, 7: 3.0}

# 등급 경계값
GRADE_BOUNDARIES = [
    (1, -2.5),
    (2, -1.5),
    (3, -0.5),
    (4, 0.5),
    (5, 1.5),
    (6, 2.5),
]

TOTAL_QUESTIONS = 10


def estimate_grade(b_value: float) -> int:
    # B값을 등급으로 변환
    for grade, upper in GRADE_BOUNDARIES:
        if b_value < upper:
            return grade
    return 7


def create_session(quiz_type: str, current_grade: Optional[int] = None) -> dict:
    """
    CAT 세션 초기화

    quiz_type: "onboarding" | "upgrade"
    current_grade: 승급 퀴즈일 때 현재 등급
    """
    if quiz_type == "onboarding":
        initial_d = 0.0  # 4등급 중간에서 시작
    else:
        # 승급: 현재 등급의 logit에서 시작
        initial_d = GRADE_LOGIT.get(current_grade, 0.0)

    return {
        "quiz_type": quiz_type,
        "current_grade": current_grade,
        "D": initial_d,
        "L": 0, 
        "H": 0.0, 
        "R": 0, 
        "history": [], 
        "responses": [], 
        "completed": False,
    }
