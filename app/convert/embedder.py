"""
CSV 어휘 사전을 읽어 sentence-transformers로 임베딩 후 ChromaDB에 저장.
최초 1회 또는 force_rebuild=True 시에만 실행.
"""
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "vocab_embeddings"
MODEL_NAME = "jhgan/ko-sroberta-multitask"
CSV_PATH = "data/vocab_dictionary_v3.csv"

_client: chromadb.PersistentClient | None = None
_collection = None
_model: SentenceTransformer | None = None


def get_client() -> chromadb.PersistentClient:
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


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def build_embeddings(force_rebuild: bool = False) -> None:
    collection = get_collection()

    if collection.count() > 0 and not force_rebuild:
        print(f"[embedder] {collection.count()} entries already exist. Skipping rebuild.")
        return

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["word", "meaning"]).reset_index(drop=True)
    print(f"[embedder] Loaded {len(df)} vocab entries")

    # word + meaning을 합쳐서 임베딩 (의미 기반 유사도 향상)
    documents = (df["word"].astype(str) + ": " + df["meaning"].astype(str)).tolist()

    model = get_model()
    print("[embedder] Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(
        documents,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    batch_size = 5000
    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start : start + batch_size].copy()
        batch_df = batch_df.fillna("")

        metadatas = [
            {
                "word": str(row["word"]),
                "level": int(row["level"]) if str(row["level"]).isdigit() else 0,
                "pos": str(row.get("pos", "")),
                "origin": str(row.get("origin", "")),
                "meaning": str(row["meaning"]),
                "source": str(row.get("source", "")),
            }
            for _, row in batch_df.iterrows()
        ]

        collection.upsert(
            ids=[str(i) for i in batch_df.index],
            embeddings=embeddings[start : start + batch_size].tolist(),
            documents=documents[start : start + batch_size],
            metadatas=metadatas,
        )
        print(f"[embedder] Upserted {min(start + batch_size, len(df))}/{len(df)}")

    print("[embedder] Build complete.")
