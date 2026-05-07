"""
aiomysql 기반 RDS 커넥션 풀 관리.
"""
import os
import aiomysql

_pool: aiomysql.Pool | None = None


async def init_db() -> None:
    global _pool
    _pool = await aiomysql.create_pool(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", ""),
        password=os.getenv("DB_PASSWORD", ""),
        db=os.getenv("DB_NAME", ""),
        charset="utf8mb4",
        autocommit=True,
        minsize=2,
        maxsize=10,
    )


def get_pool() -> aiomysql.Pool:
    if _pool is None:
        raise RuntimeError("DB 풀이 초기화되지 않았습니다.")
    return _pool


async def close_db() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        await _pool.wait_closed()
        _pool = None
