"""
MySQL(RDS) 비동기 연결 관리
문항 조회 전용
"""

import os
import aiomysql
from dotenv import load_dotenv

load_dotenv()

_pool: aiomysql.Pool = None


async def init_db():
    global _pool
    _pool = await aiomysql.create_pool(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        db=os.getenv("DB_NAME"),
        charset="utf8mb4",
        autocommit=True,
        minsize=2,
        maxsize=10,
    )
    print("MySQL 커넥션 풀 생성 완료")


async def close_db():
    global _pool
    if _pool:
        _pool.close()
        await _pool.wait_closed()


def get_pool() -> aiomysql.Pool:
    if _pool is None:
        raise RuntimeError("DB 풀이 초기화되지 않았습니다.")
    return _pool