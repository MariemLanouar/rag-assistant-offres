import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "offres"
TOP_K = 6  # how many chunks we retrieve for each question
SYSTEM_PROMPT = """Tu es un assistant qui aide un étudiant à analyser des offres de stage.
Réponds uniquement à partir des extraits fournis.
Après chaque information, cite le ou les extraits utilisés avec leur numéro, par exemple [1] ou [2][4].
Si les extraits ne contiennent pas la réponse, dis-le clairement au lieu d'inventer.
Réponds en français, de façon concise."""

def retrieve(client: OpenAI, collection, question: str, embedding_model: str, k: int = TOP_K, source: str | None = None) -> list[dict]:
    """Return the k chunks closest to the question, optionally only from one offer (source)."""
    query_vector = client.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    where = {"source": source} if source else None  # metadata filter: search inside one offer only
    results = collection.query(query_embeddings=[query_vector], n_results=k, where=where)
    return [
        {"text": doc, "source": meta["source"], "chunk": meta["chunk"], "distance": dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

def build_context(chunks: list[dict]) -> str:
    """Number the chunks so the LLM can cite them."""
    return "\n\n".join(f"[{i}] (source : {c['source']})\n{c['text']}" for i, c in enumerate(chunks, 1))


def answer(client: OpenAI, chat_model: str, question: str, chunks: list[dict]) -> str:
    """Send the question + the chunks to the LLM and return its answer."""
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,  # 0 = most stable and factual answers, no creativity
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extraits :\n\n{build_context(chunks)}\n\nQuestion : {question}"},
        ],
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    client = OpenAI()
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION_NAME)

    question = " ".join(sys.argv[1:]) or "Quelles offres demandent Docker ?"
    chunks = retrieve(client, collection, question, embedding_model)

    print("\n" + answer(client, chat_model, question, chunks))
    print("\nSources :")
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}] {c['source']} (chunk {c['chunk']}, distance {c['distance']:.3f})")