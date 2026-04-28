#퀴즈 API 라우터
# POST /api/quiz/start - 세션 시작 + 첫 문항
# POST /api/quiz/answer - 답변 제출 + 다음 문항 or 결과
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import uuid

from app.quiz.cat_engine import create_session, select_next_item, process_answer
from app.quiz.item_pool import get_all_items, get_item_by_id  

router=APIRouter(prefix="/api/quiz", tags=["quiz"])

_sessions:dict={}

class QuizStartRequest(BaseModel):
  user_id:str
  quiz_type:str
  current_grade: Optional[int]=None

class QuizAnswerRequest(BaseModel):
  session_id:str
  choiceId:int

def format_item_for_frontend(item:dict, question_number:int)->dict:
  if item["type"]=="meaning_choice":  
    question_text=f"'{item['word']}'의 의미로 가장 적절한 것을 고르세요."
  else:
    question_text=item["sentence"]

  choices=[
    {"choiceId":i, "choiceText":opt}
    for i,opt in enumerate(item["options"])
  ]

  return{
    "questionId":item["id"],
    "questionNumber":question_number,
    "totalQuestions":10,
    "type":item["type"],
    "word":item["word"],
    "questionText":question_text,
    "choices":choices,
  }

@router.post("/start")
async def quiz_start(req:QuizStartRequest):
  if req.quiz_type not in ("onboarding", "upgrade"):
    raise HTTPException(status_code=400, detail="quiz_type은 'onboarding' 또는 'upgrade'여야 합니다.")
  
  if req.quiz_type=="upgrade" and req.current_grade is None:
    raise HTTPException(status_code=400, detail="승급 퀴즈는 current_grade가 필요합니다.")

  session_id=str(uuid.uuid4())
  session=create_session(req.quiz_type, req.current_grade)

  items=get_all_items()
  first_item=select_next_item(session, items)

  if not first_item:
    raise HTTPException(status_code=500, detail="출제할 문항이 없습니다.")

  session["current_item_id"]=first_item["id"]
  _sessions[session_id]=session
  
  return{
    "success":True,
    "session_id":session_id,
    "first_item": format_item_for_frontend(first_item, 1),
  }

@router.post("/answer")
async def quiz_answer(req: QuizAnswerRequest):
    session = _sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")

    if session["completed"]:
        raise HTTPException(status_code=400, detail="이미 완료된 퀴즈입니다.")

    current_item_id = session.get("current_item_id")
    if not current_item_id:
        raise HTTPException(status_code=400, detail="출제된 문항이 없습니다.")

    current_item = get_item_by_id(current_item_id)
    if not current_item:
        raise HTTPException(status_code=500, detail="문항을 찾을 수 없습니다.")

    result = process_answer(session, current_item, req.choiceId)

    response = {
        "success": True,
        "is_correct": result["is_correct"],
        "question_number": result["question_number"],
        "completed": result["completed"],
    }

    if result["completed"]:
        response["result"] = result["result"]
        response["next_item"] = None
        del _sessions[req.session_id]
    else:
        items = get_all_items()
        next_item = select_next_item(session, items)
        if next_item:
            session["current_item_id"] = next_item["id"]
            response["next_item"] = format_item_for_frontend(
                next_item, result["question_number"] + 1
            )
        else:
            response["next_item"] = None
        response["result"] = None

    return response