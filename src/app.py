import io
import os
import re
import unicodedata
from pathlib import Path

import chromadb
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# We reuse the functions we already wrote: the interface only adds a visual layer
from ask import answer, retrieve
from ingest import embed, split_text
from match import explain, rank_offers

ROOT = Path(__file__).resolve().parent.parent
DB_DIR = ROOT / "chroma_db"
OFFERS_DIR = ROOT / "data" / "offers"
COLLECTION_NAME = "offres"

load_dotenv(ROOT / ".env")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

st.set_page_config(page_title="StageMatch", page_icon="🎯", layout="centered")

@st.cache_resource  # created once, then reused at every interaction
def get_resources():
    client = OpenAI()
    collection = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION_NAME)
    return client, collection


def read_uploaded_cv(uploaded_file) -> str:
    """Return the text of an uploaded CV (.pdf or .txt)."""
    data = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return data.decode("utf-8", errors="ignore")


def to_filename(name: str) -> str:
    """'L'Oréal – Cloud & AI' -> 'l_oreal_cloud_ai.txt' (no accents, no spaces)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")
    return f"{slug}.txt"


def add_offer(client, collection, name: str, text: str) -> tuple[str, int]:
    """Save a pasted offer and index it right away (incremental indexing)."""
    filename = to_filename(name)
    OFFERS_DIR.mkdir(parents=True, exist_ok=True)
    (OFFERS_DIR / filename).write_text(text, encoding="utf-8")  # kept on disk, so ingest.py can rebuild everything

    collection.delete(where={"source": filename})  # if the offer already existed, replace its old chunks
    chunks = split_text(text)
    vectors = embed(client, EMBEDDING_MODEL, chunks)
    collection.add(
        ids=[f"{Path(filename).stem}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return filename, len(chunks)

def pretty(source: str) -> str:
    return Path(source).stem.replace("_", " ").capitalize()

def list_offers(collection) -> list[str]:
    """Names of all the offers currently in the database."""
    return sorted({meta["source"] for meta in collection.get(include=["metadatas"])["metadatas"]})


st.title("StageMatch")
st.caption("Pose des questions sur les offres de stage et compare-les à ton CV · RAG avec OpenAI et ChromaDB")

if not DB_DIR.exists():
    st.error("La base est vide : lance d'abord `python src/ingest.py`.")
    st.stop()

client, collection = get_resources()
col1, col2 = st.columns(2)
col1.metric("Offres indexées", len(list_offers(collection)))
col2.metric("Chunks dans ChromaDB", collection.count())
tab_questions, tab_matching, tab_add = st.tabs(["💬 Questions", "📄 Matching CV", "➕ Ajouter une offre"])

# ---------- Tab 1: questions with sources ----------
with tab_questions:
    offers = list_offers(collection)
    scope = st.selectbox("Rechercher dans", ["Toutes les offres"] + offers, format_func=lambda s: s if s == "Toutes les offres" else pretty(s))
    question = st.text_input("Pose une question sur les offres", placeholder="Quelles compétences sont demandées ?")

    if st.button("Rechercher", type="primary") and question.strip():
        source = None if scope == "Toutes les offres" else scope
        with st.spinner("Recherche des passages pertinents..."):
            chunks = retrieve(client, collection, question, EMBEDDING_MODEL, source=source)
            response = answer(client, CHAT_MODEL, question, chunks)

        st.markdown("### Réponse")
        st.write(response)

        st.markdown("### Sources")
        for i, c in enumerate(chunks, 1):
            # cosine distance = 1 - similarity, so 1 - distance gives a more readable relevance score
            with st.expander(f"[{i}] {pretty(c['source'])} · pertinence {1 - c['distance']:.0%}"):
                st.write(c["text"])

# ---------- Tab 2: CV matching ----------
with tab_matching:
    uploaded = st.file_uploader("Dépose un CV (PDF ou TXT)", type=["pdf", "txt"])
    nb_explained = st.slider("Nombre d'offres à expliquer", 1, 5, 3)

    if uploaded and st.button("Lancer le matching", type="primary"):
        with st.spinner("Analyse du CV et calcul des scores..."):
            cv_text = read_uploaded_cv(uploaded)
            cv_chunks = split_text(cv_text)
            cv_vectors = np.array(embed(client, EMBEDDING_MODEL, cv_chunks))
            offer_data = collection.get(include=["embeddings", "metadatas", "documents"])
            ranking = rank_offers(cv_vectors, offer_data)

        st.markdown("### Classement des offres")
        st.dataframe(
            [{"Rang": i, "Offre": pretty(o["source"]), "Score": o["score"]} for i, o in enumerate(ranking, 1)],
            hide_index=True,
            column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f")},
        )

        st.markdown(f"### Analyse des {nb_explained} meilleures offres")
        for offer in ranking[:nb_explained]:
            with st.spinner(f"Analyse de {offer['source']}..."):
                explanation = explain(client, CHAT_MODEL, cv_text, offer)
            with st.expander(f"{pretty(offer['source'])} · score {offer['score']:.3f}", expanded=True):
                st.markdown(explanation)

# ---------- Tab 3: add an offer ----------
with tab_add:
    st.write("Colle une offre de stage : elle est ajoutée à la base et devient disponible pour les questions et le matching.")
    name = st.text_input("Nom de l'offre", placeholder="L'Oréal – Cloud & AI Software Engineer")
    text = st.text_area("Texte de l'offre", height=250)

    if "added_msg" in st.session_state:
        st.success(st.session_state.pop("added_msg"))

    if st.button("Ajouter à la base", type="primary"):
        if not name.strip() or len(text.strip()) < 100:
            st.warning("Indique un nom et colle le texte complet de l'offre.")
        else:
            with st.spinner("Découpage et calcul des embeddings..."):
                filename, nb_chunks = add_offer(client, collection, name, text)
            # Rerun the page so the new offer appears in the "Rechercher dans" list right away
            st.session_state["added_msg"] = f"Offre ajoutée : {filename} ({nb_chunks} chunks). Elle est disponible dans les deux autres onglets."
            st.rerun()

st.divider()
st.caption("Projet personnel · [GitHub](https://github.com/MariemLanouar/rag-assistant-offres)")