
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import Optional


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PATH = "chroma_persistent"
COLLECTION_NAME = "ps4_collection"

embedder = SentenceTransformer(EMBED_MODEL_NAME)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

def encode_texts(texts, as_list=True):
    embs = embedder.encode(texts, show_progress_bar=False)
    return embs.tolist() if as_list else embs
