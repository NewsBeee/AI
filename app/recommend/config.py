import os
from dotenv import load_dotenv

load_dotenv()


NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID", "")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET", "")
NAVER_SEARCH_URL = "https://openapi.naver.com/v1/search/news.json"

NEWS_MAX_PER_KEYWORD = 10      # 키워드당 최대 수집 건수
NEWS_SORT = "date"             # date(최신순) | sim(정확도순)

SEARCH_KEYWORD_COUNT = 7       # 이력에서 추출할 검색 키워드 수
RECOMMEND_COUNT = 10           # 최종 추천 기사 수
HISTORY_WINDOW = 20            # 관심사 프로필 계산에 사용할 최근 이력 수
DECAY_FACTOR = 0.95            # 시간 기반 가중치 감쇠율


EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jhgan/ko-sroberta-multitask")

# 신규 사용자 폴백용 기본 키워드
FALLBACK_KEYWORDS = ["뉴스", "시사", "사회"]