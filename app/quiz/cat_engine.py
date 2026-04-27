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

def select_next_item(session:dict, items: list)->Optional[dict]:
    # 현재 세션 상태에 기반하여 다음 문항 선택
    # 홀수번째: 빈칸 채우기 문항
    # 짝수번째: 의미 선택 문항
    # 목표 난이도(D)에 가장 가까운 미출제 문항을 선택
    question_number=session["L"]+1

    if question_number > TOTAL_QUESTIONS:
        return None
    
    if question_number%2==1:
        target_type="fill_blank"
    else:
        target_type="meaning_choice"

    used_ids=set(session["history"])
    candidates=[
        item for item in items
        if item["id"] not in used_ids and item["typr"]==target_type
    ]
    if not candidates:
        candidates=[
            item for item in items
            if item["id"] not in used_ids
        ]

    if not candidates:
        return None
    
    target_d=session["D"]
    candidates.sort(key=lambda x: abs(x["b"]-target_d))

    return candidates[0]

def process_answer(session:dict, item:dict, selected_choice: int)->dict:
    # 답변 처리+CAT 업데이트
    is_correct=(selected_choice==item["answer"])

    #D를 해당 문항의 실제 b값으로 갱신
    session["D"]=item["b"]

    session["L"]+=1
    session["H"]+=item["b"]

    session["history"].append(item["id"])
    session["responses"].append({
        "item_id": item["id"],
        "word": item["word"],
        "grade": item["grade"],
        "type": item["type"],
        "is_correct": is_correct,
        "b": item["b"],
    })

    if is_correct:
        session["R"]+=1
        session["D"]=session["D"]+2/session["L"]
    else:
        session["D"]=session["D"]-2/session["L"]

    result_response={
        "is_correct": is_correct,
        "question_number": session["L"],
    }

    #10문항 완료 시
    if session["L"]>=TOTAL_QUESTIONS:
        session["completed"]=True
        result=compute_final_result(session)
        result_response["completed"]=True
        result_response["result"]=result
    else:
        result_response["completed"]=False
        result_response["result"]=None
    return result_response

def compute_final_result(session:dict)->dict:
    # 최종 결과 계산
    """
    B = H/L + ln(R/W)
    예외: W=0이면 B = H/L + ln((R-0.5)/0.5)
          R=0이면 B = H/L + ln(0.5/(W-0.5))
    """
    L=session["L"]
    H=session["H"]
    R=session["R"]
    W=L-R

    if W==0:
        B=H/L+math.log((R-0.5)/0.5)
    elif R==0:
        B=H/L+math.log(0.5/(W-0.5))
    else:
        B=H/L+math.log(R/W)
    
    assigned_grade=estimate_grade(B)
    # 승급 판정
    promoted=False
    if session["quiz_type"]=="upgrade" and session["current_grade"]:
        current_logit=GRADE_LOGIT.get(session["current_grade"], 0.0)
        if B>current_logit+0.5:
            promoted=True
    
    by_grade={}
    for resp in session["responses"]:
        grade_key = f"{resp['grade']}등급"
        if grade_key not in by_grade:
            by_grade[grade_key] = {"출제": 0, "정답": 0}
        by_grade[grade_key]["출제"] += 1
        if resp["is_correct"]:
            by_grade[grade_key]["정답"] += 1
 
    return {
        "estimated_theta": round(B, 2),
        "assigned_grade": assigned_grade,
        "total": L,
        "correct": R,
        "promoted": promoted,
        "by_grade": by_grade,
    }