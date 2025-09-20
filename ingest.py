from typing import List, Dict
import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL_NAME, CHROMA_DB_PATH, COLLECTION_NAME

model = SentenceTransformer(EMBED_MODEL_NAME)

# Using PersistentClient to automatically save to disk
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)


collection = client.get_or_create_collection(name=COLLECTION_NAME)


def _gen_id(file_id: str, i: int) -> str:
    """Generates a unique ID for each document segment."""
    return f"{file_id}__seg_{i}"

def ingest_segments_to_chroma(segments: List[Dict], file_id: str):
    """
    Embeds document segments and adds them to the ChromaDB collection.
    Persistence is handled automatically by the PersistentClient.
    """
    docs, metadatas, ids = [], [], []

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            continue
        
        docs.append(text)
        
        md = {
            "start": seg.get("start", 0.0),
            "end": seg.get("end"),
            "speaker": seg.get("speaker"),
            "file_id": file_id,
            "source_type": seg.get("source_type", "text")
        }
        metadatas.append(md)
        ids.append(_gen_id(file_id, i))

    if not docs:
        return {"added": 0}

    # Generate embeddings
    embeddings = model.encode(docs).tolist()
    
    # Add the data to the collection
    collection.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metadatas)

    return {"added": len(docs)}
