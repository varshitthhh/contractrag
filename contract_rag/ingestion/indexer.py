import uuid
import json
import logging
from pathlib import Path
from collections import Counter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    SparseVectorParams, SparseIndexParams,
    SparseVector
)

logger = logging.getLogger(__name__)
MANIFEST_PATH = "contract_rag/data/processed/manifest.json"

def load_manifest() -> dict:
    if Path(MANIFEST_PATH).exists():
        return json.loads(Path(MANIFEST_PATH).read_text())
    return {}

def save_manifest(manifest: dict):
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2))

def build_sparse_vector(text: str) -> SparseVector:
    words = text.lower().split()
    counts = Counter(words)
    vocab = sorted(set(words))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    indices = [word_to_id[w] for w in counts]
    values = [float(c) for c in counts.values()]
    return SparseVector(indices=indices, values=values)

def init_collection(client: QdrantClient, collection_name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": VectorParams(size=vector_size, distance=Distance.COSINE)},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams())}
        )
        logger.info(f"Created collection: {collection_name}")
    else:
        logger.info(f"Collection exists: {collection_name}")

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: list[dict], embeddings: list[list[float]], source_file: str):
    manifest = load_manifest()
    if source_file in manifest:
        logger.info(f"Skipping {source_file} — already indexed")
        return 0

    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source_file}_{chunk['chunk_index']}"))
        sparse = build_sparse_vector(chunk["text"])
        payload = {k: v for k, v in chunk.items() if k != "embedding"}
        points.append(PointStruct(
            id=point_id,
            vector={"dense": embedding, "sparse": sparse},
            payload=payload
        ))

    batch_size = 64
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=collection_name, points=points[i:i+batch_size])

    manifest[source_file] = {"chunks": len(chunks), "status": "indexed"}
    save_manifest(manifest)
    logger.info(f"Indexed {len(points)} chunks from {source_file}")
    return len(points)