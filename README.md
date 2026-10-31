# RAG Assistant for Internship Offers

A Python project that helps analyze internship offers with Retrieval-Augmented Generation (RAG):

- **Ask questions** about a set of offers and get answers that **cite their sources**, or say when the information is missing instead of inventing it.
- **Match a CV with the offers**: rank the offers by relevance and get, for the best ones, the **strong points** and the **missing skills**.

Built from scratch to learn RAG end to end: chunking, embeddings, vector search, grounded generation and evaluation.

## Examples

**Questions with sources**
```
> python src/ask.py "Quelle est la gratification chez Ateme ?"
La gratification chez Ateme est de 1500 € brut mensuel [1].

> python src/ask.py "Quel est le salaire chez Google ?"
Les extraits fournis ne contiennent pas d'informations sur le salaire chez Google.
```

**CV matching** (fictional supply chain profile)
```
> python src/match.py data/cv_examples/cv_supply_chain_fictif.txt
 1. shiseido_operations_analyst.txt     score 0.721
 2. danone_ia_consolidation.txt         score 0.632
 ...
19. adacore_ai_production.txt           score 0.420

=== shiseido_operations_analyst.txt ===
1. Points forts :
   - Spécialisation en Supply Chain Management
   - Excel avancé et Power BI
   - Consolidation des inventaires et suivi des KPI logistiques
2. Compétences manquantes : ...
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

1. **Chunking**: each offer is split into ~800-character chunks with a 150-character overlap, cut at line or sentence boundaries, so each chunk covers one topic.
2. **Embeddings**: each chunk is turned into a vector with OpenAI `text-embedding-3-small` and stored in **ChromaDB** with its source.
3. **Question answering** (`ask.py`): the question is embedded with the same model, the 6 closest chunks are retrieved (cosine similarity), and the LLM (`gpt-4o-mini`, temperature 0) answers **only from those chunks**, citing them as [1], [2]…
4. **CV matching** (`match.py`): the CV (PDF or TXT) is split and embedded the same way. Every CV chunk is compared with every offer chunk (cosine similarity matrix). An offer's score is the **average of its 3 best (CV chunk, offer chunk) pairs**, so generic parts of an offer (company description, benefits) don't lower its score. For the top 3 offers, the LLM explains the strong points and the missing skills, using only the CV and the offer's extracts.

## Stack
Python · OpenAI API · ChromaDB · NumPy · pypdf · python-dotenv

## Project structure

```
data/
├── offers/        # 19 real internship offers (.txt)
├── cv_examples/   # 2 fictional CVs used for testing
└── cv/            # your own CV (ignored by Git)
src/
├── ingest.py      # chunking + embeddings + ChromaDB indexing
├── ask.py         # questions with cited sources
└── match.py       # CV ↔ offers matching + explanations
```

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your OpenAI API key
```

## Usage

```bash
python src/ingest.py                                       # index the offers in data/offers/
python src/ask.py "Quelles offres demandent Docker ?"       # ask a question
python src/match.py data/cv_examples/cv_frontend_fictif.txt # rank the offers for a CV
python src/match.py                                         # uses the first CV found in data/cv/
```

## Evaluation

### Question answering

| Question | Result |
|---|---|
| Internship stipend at Ateme | ✅ Correct answer (1 500 €), correct source cited |
| Offers requiring Docker | ✅ Ignored a misleading chunk about *shipping* containers · ⚠️ Missed 2 offers that mention Docker (not retrieved in the top 6) |
| Salary at Google (not in the data) | ✅ Says the information is not available, no hallucination |

### CV matching

Tested with 3 very different profiles to check that the ranking really depends on the CV:

| CV | Top of the ranking | Bottom of the ranking | Verdict |
|---|---|---|---|
| Fictional supply chain analyst | Shiseido (supply chain, 0.721, clear lead), Danone, Saegus | AdaCore, L'Oréal, ScaleX | ✅ As expected |
| Fictional frontend developer | CGI, Coexya (Angular, JavaScript) | AXA, AdaCore, Shiseido | ✅ As expected |
| Real backend / data / AI profile | Coexya (Java Spring, Angular, SQL), Sopra Steria, Ateme | L'Oréal, AXA | ⚠️ See language bias below |

**Language bias found**: for all three CVs (written in French), the 4 offers written in **English** always end up in the last positions, even when they match the profile well. Texts in different languages get less similar embeddings, so English offers are systematically penalized.

**Explanations**: the ranking is reliable, but the LLM explanations should be read critically. They are sometimes inconsistent (e.g. saying testing skills are missing while the CV lists Jest and Cypress).

## Limits and next steps
- **Language bias**: translate offers and CVs into the same language before embedding, use a more multilingual embedding model, or let the LLM re-rank the top results.
- **The answer is only as good as the retrieval**: if a relevant chunk is not retrieved, the LLM cannot use it. Next step: **hybrid search** (embeddings + keyword search) for exact terms like technology names.
- **Automated evaluation** on a fixed set of questions with expected answers.
- **Bigger dataset**: collect offers through official APIs instead of manual copies.
- **Provider-independent setup**: switch to a free or local model through configuration.

## Author
Mariem Lanouar · [LinkedIn](https://www.linkedin.com/in/mariem-lanouar-543911287)