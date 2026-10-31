import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

# Paths are built from this file's location, so the script works
# no matter which folder you run it from.
ROOT = Path(__file__).resolve().parent.parent
OFFERS_DIR = ROOT / "data" / "offers"

CHUNK_SIZE = 800      # max characters per chunk
CHUNK_OVERLAP = 150   # characters repeated from the previous chunk
DB_DIR = ROOT / "chroma_db"   # where the database is saved on disk
COLLECTION_NAME = "offres"    # like a "table" in the database
BATCH_SIZE = 64               # chunks sent to the API per request

def read_document(path: Path) -> str:
    """Return the text of a .txt file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of about `size` characters, overlapping by `overlap`."""
    # 1. Clean up: remove empty lines and extra spaces
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))

        # 2. Don't cut in the middle of a sentence: go back to the last
        #    line break or ". " if there's one in the second half of the chunk
        if end < len(text):
            cut = max(text.rfind("\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break
        # 3. The next chunk starts a bit before the end of this one (overlap)
        start = max(end - overlap, start + 1)

    return chunks

def embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    """Compute embeddings in batches."""
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors

def main() -> None:
    load_dotenv(ROOT / ".env")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # 1. Read and split every offer, keeping track of where each chunk comes from
    ids, documents, metadatas = [], [], []
    for path in sorted(OFFERS_DIR.glob("*.txt")):
        chunks = split_text(read_document(path))
        print(f"{path.name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            ids.append(f"{path.stem}-{i}")                       # unique id, e.g. "ateme_data_ia-3"
            documents.append(chunk)                              # the text
            metadatas.append({"source": path.name, "chunk": i})  # where it comes from

    # 2. Turn every chunk into a vector
    print(f"\nComputing {len(documents)} embeddings with {embedding_model}...")
    client = OpenAI()
    vectors = embed(client, embedding_model, documents)

    # 3. Store everything in ChromaDB (we rebuild it from scratch each time, to avoid duplicates)
    db = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        db.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = db.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    print(f"OK: {collection.count()} chunks indexed in {DB_DIR.name}/")

if __name__ == "__main__":
    main()