"""
RDS vocab_master 테이블에서 어휘 데이터를 읽어 sentence-transformers로 임베딩 후 ChromaDB에 저장.
최초 1회 또는 force_rebuild=True 시에만 실행.
"""

import aiomysql
import chromadb

from app.core.embedder import get_model
from app.convert.config import CHROMA_PATH, COLLECTION_NAME, VOCAB_TABLE
from app.database import get_pool

_client = None
_collection = None


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


async def build_embeddings(force_rebuild: bool = False) -> None:
    collection = get_collection()

    if collection.count() > 0 and not force_rebuild:
        print(f"[embedder] {collection.count()} entries already exist. Skipping rebuild.")
        return

    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                f"SELECT vocab_id, word, level, pos, origin, meaning, source "
                f"FROM {VOCAB_TABLE} WHERE word IS NOT NULL AND meaning IS NOT NULL"
            )
            rows = await cur.fetchall()

    print(f"[embedder] Loaded {len(rows)} vocab entries from DB")

    documents = [f"{row['word']}: {row['meaning']}" for row in rows]

    model = get_model()
    print("[embedder] Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(
        documents,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    batch_size = 5000

    for start in range(0, len(rows), batch_size):
        batch = rows[start: start + batch_size]
        metadatas = [
            {
                "word": str(row["word"]),
                "level": int(row["level"]) if str(row["level"]).isdigit() else 0,
                "pos": str(row["pos"] or ""),
                "origin": str(row["origin"] or ""),
                "meaning": str(row["meaning"]),
                "source": str(row["source"] or ""),
            }
            for row in batch
        ]

        collection.upsert(
            ids=[str(row["vocab_id"]) for row in batch],
            embeddings=embeddings[start: start + batch_size].tolist(),
            documents=documents[start: start + batch_size],
            metadatas=metadatas,
        )

        print(f"[embedder] Upserted {min(start + batch_size, len(rows))}/{len(rows)}")

    print("[embedder] Build complete.")