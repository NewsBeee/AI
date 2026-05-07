import json
from typing import Optional

import aiomysql
from app.database import get_pool

TABLE_NAME = "quiz_item_pool"


async def get_all_items() -> list:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT id, word, grade, type, b, sentence, options, answer FROM {TABLE_NAME}"
            )
            rows = await cur.fetchall()

    items = []
    for row in rows:
        item = dict(row)
        if isinstance(item["options"], str):
            item["options"] = json.loads(item["options"])
        items.append(item)
    return items


async def get_item_by_id(item_id: str) -> Optional[dict]:
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT id, word, grade, type, b, sentence, options, answer "
                f"FROM {TABLE_NAME} WHERE id = %s",
                (item_id,),
            )
            row = await cur.fetchone()

    if not row:
        return None

    item = dict(row)
    if isinstance(item["options"], str):
        item["options"] = json.loads(item["options"])
    return item