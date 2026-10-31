# RAG Assistant for Internship Offers

A Python assistant that answers questions about a set of internship offers, **citing its sources**, and says when the answer is not in the documents instead of inventing one.

Built to learn Retrieval-Augmented Generation (RAG) from scratch: chunking, embeddings, vector search and grounded generation.

## Example

```
> python src/ask.py "Quelle est la gratification chez Ateme ?"
La gratification chez Ateme est de 1500 € brut mensuel [1].

> python src/ask.py "Quel est le salaire chez Google ?"
Les extraits fournis ne contiennent pas d'informations sur le salaire chez Google.
```

## How it works

```
INDEXING (once)                          QUESTION (each time)
offers (.txt) → chunks → embeddings      question → embedding
                      ↓                          ↓
                  ChromaDB  ←──── search the 6 closest chunks
                                                 ↓
                         LLM answers using only those chunks + cites them
```

1. **Chunking**: each offer is split into ~800-character chunks with a 150-character overlap, cut at line or sentence boundaries.
2. **Embeddings**: each chunk is turned into a vector with OpenAI `text-embedding-3-small` and stored in **ChromaDB** with its source.
3. **Retrieval**: the question is embedded with the same model, and the 6 closest chunks are retrieved (cosine similarity).
4. **Generation**: the LLM (`gpt-4o-mini`, temperature 0) answers **only from those chunks**, cites them as [1], [2]…, and says so when the information is missing.

## Stack
Python · OpenAI API · ChromaDB · python-dotenv

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI API key
```

## Usage

```bash
python src/ingest.py                                  # index the offers in data/offers/
python src/ask.py "Quelles offres demandent Docker ?"  # ask a question
```

## Tests and observations

| Question | Result |
|---|---|
| Internship stipend at Ateme | ✅ Correct answer (1 500 €), correct source cited |
| Offers requiring Docker | ✅ Ignored a misleading chunk about *shipping* containers · ⚠️ Missed 2 offers that mention Docker (not retrieved in the top 6) |
| Salary at Google (not in the data) | ✅ Says the information is not available, no hallucination |

## Limits and next steps
- **The answer is only as good as the retrieval**: if a relevant chunk is not retrieved, the LLM cannot use it. Next step: **hybrid search** (embeddings + keyword search) for exact terms like technology names.
- **CV matching** (in progress): rank the offers closest to a CV and explain the missing skills.
- **Evaluation** on a fixed set of questions to measure accuracy.
- **Provider-independent setup**: switch to a free or local model through configuration.

## Author
Mariem Lanouar · [LinkedIn](https://www.linkedin.com/in/mariem-lanouar-543911287)