import os
import sys
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from ingest import embed, split_text  # we reuse the functions from steps 2 and 3

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "offres"
CV_DIR = ROOT / "data" / "cv"
TOP_PAIRS = 3     # an offer's score = average of its 3 best (CV chunk, offer chunk) similarities
TOP_OFFERS = 3    # how many offers the LLM explains

EXPLAIN_PROMPT = """Tu aides un étudiant à évaluer si une offre de stage correspond à un CV.
Réponds uniquement à partir du CV et des extraits de l'offre fournis, sans rien inventer.
Donne :
1. Points forts : 2 ou 3 éléments du CV qui correspondent à l'offre.
2. Compétences manquantes : 2 ou 3 éléments demandés par l'offre qui n'apparaissent pas dans le CV.
Sois concis, en français, avec des listes à puces."""


def read_cv(path: Path) -> str:
    """Return the text of a CV in .pdf or .txt format."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every vector to length 1, so a dot product equals the cosine similarity."""
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def rank_offers(cv_vectors: np.ndarray, offer_data: dict) -> list[dict]:
    """Score every offer against the CV and sort them from best to worst match."""
    offer_vectors = normalize(np.array(offer_data["embeddings"]))
    cv_vectors = normalize(cv_vectors)

    # similarity[i, j] = how close CV chunk i is to offer chunk j (1 = same meaning)
    similarity = cv_vectors @ offer_vectors.T

    sources = [meta["source"] for meta in offer_data["metadatas"]]
    ranking = []
    for source in sorted(set(sources)):
        columns = [j for j, s in enumerate(sources) if s == source]   # this offer's chunks
        scores = similarity[:, columns].flatten()                    # every (CV chunk, offer chunk) pair
        best = np.argsort(scores)[::-1][:TOP_PAIRS]                   # the strongest pairs
        best_chunks = sorted({columns[k % len(columns)] for k in best})
        ranking.append({
            "source": source,
            "score": float(scores[best].mean()),
            "chunks": [offer_data["documents"][j] for j in best_chunks],
        })
    return sorted(ranking, key=lambda r: r["score"], reverse=True)


def explain(client: OpenAI, chat_model: str, cv_text: str, offer: dict) -> str:
    """Ask the LLM why this offer matches the CV and what is missing."""
    extracts = "\n\n".join(offer["chunks"])
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": EXPLAIN_PROMPT},
            {"role": "user", "content": f"CV :\n{cv_text}\n\nExtraits de l'offre ({offer['source']}) :\n{extracts}"},
        ],
    )
    return response.choices[0].message.content


def main() -> None:
    load_dotenv(ROOT / ".env")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

    # The CV path can be given in the terminal; otherwise we take the first file in data/cv/
    if len(sys.argv) > 1:
        cv_path = Path(sys.argv[1])
    else:
        cv_files = sorted(p for p in CV_DIR.glob("*") if p.suffix.lower() in {".pdf", ".txt"})
        if not cv_files:
            raise SystemExit("No CV found: put a .pdf or .txt file in data/cv/")
        cv_path = cv_files[0]

    # 1. Read and split the CV, then embed its chunks (with the SAME model as the offers)
    cv_text = read_cv(cv_path)
    cv_chunks = split_text(cv_text)
    client = OpenAI()
    cv_vectors = np.array(embed(client, embedding_model, cv_chunks))

    # 2. Load every offer chunk and its embedding from ChromaDB
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION_NAME)
    offer_data = collection.get(include=["embeddings", "metadatas", "documents"])

    # 3. Rank the offers
    ranking = rank_offers(cv_vectors, offer_data)
    print(f"\nRanking for {cv_path.name} ({len(cv_chunks)} CV chunks):\n")
    for i, offer in enumerate(ranking, 1):
        print(f"{i:2}. {offer['source']:<35} score {offer['score']:.3f}")

    # 4. Explain the top offers
    for offer in ranking[:TOP_OFFERS]:
        print(f"\n=== {offer['source']} ===")
        print(explain(client, chat_model, cv_text, offer))


if __name__ == "__main__":
    main()