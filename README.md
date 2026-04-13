# NewsBee AI Server

RAG 기반 수준별 기사 변환 및 어휘력 강화 서비스 - AI 서버

## 기술 스택
- FastAPI
- LangChain
- ChromaDB
- OpenAI GPT-4o-mini

## 시작하기

### 1. 레포 클론
git clone https://github.com/NewsBeee/AI.git
cd AI

### 2. 패키지 설치
pip install -r requirements.txt

### 3. 환경변수 설정
.env.example 파일을 복사해서 .env 파일 생성
API 키는 팀 단톡방 참고

### 4. 서버 실행
python -m uvicorn app.main:app --reload --port 8001

### 5. 서버 확인
http://localhost:8001 접속 후 아래 메시지 확인
{"message": "NewsBee AI Server Running"}
