import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from llm_generator import generate_llm_answer

st.set_page_config(page_title="DocuMind AI", layout="wide")

# ---------------- UI Styling ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #f4fbf6;
}

/* Headers */
h1, h2, h3 {
    color: #1b7f5c;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #e8f5ec;
}

/* Buttons */
.stButton>button {
    background-color: #1b7f5c;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
}

.stButton>button:hover {
    background-color: #145c42;
}

/* Confidence badge */
.confidence-box {
    padding: 8px;
    border-radius: 8px;
    background-color: #e6f7ee;
    color: #145c42;
    font-weight: 600;
    border: 1px solid #cdebd7;
}

/* Source card */
.source-card {
    padding: 10px;
    border-radius: 8px;
    background-color: #f0faf4;
    border: 1px solid #cdebd7;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #1b7f5c;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------

col1, col2 = st.columns([1,6])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712100.png", width=70)

with col2:
    st.markdown("""
    <h1>DocuMind AI</h1>
    <p style='color:#145c42;font-size:18px'>
    AI-powered document question answering assistant
    </p>
    """, unsafe_allow_html=True)

# ---------------- Models ----------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------- Session State ----------------

if "docs" not in st.session_state:
    st.session_state.docs = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "index" not in st.session_state:
    st.session_state.index = None

if "bm25" not in st.session_state:
    st.session_state.bm25 = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- PDF Reader ----------------

def extract_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text

# ---------------- Sidebar Upload ----------------

st.sidebar.header("📄 Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

# ---------------- Process Documents ----------------

if st.sidebar.button("Process Documents"):

    if not uploaded_files:
        st.sidebar.warning("Please upload documents first")

    else:

        docs = []
        sources = []

        for file in uploaded_files:

            if file.name.endswith(".pdf"):
                text = extract_pdf(file)
            else:
                text = file.read().decode("utf-8")

            if not text:
                continue

            chunk_size = 800
            overlap = 150

            for i in range(0, len(text), chunk_size - overlap):

                chunk = text[i:i + chunk_size]
                docs.append(chunk)
                sources.append(file.name)

        embeddings = embedding_model.encode(docs)
        embeddings = np.array(embeddings, dtype=np.float32)

        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        tokenized_docs = [doc.split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)

        st.session_state.docs = docs
        st.session_state.sources = sources
        st.session_state.index = index
        st.session_state.bm25 = bm25
        st.session_state.messages = []

        st.sidebar.success("Documents processed successfully!")

# ---------------- Clear System ----------------

if st.sidebar.button("Clear System"):

    st.session_state.docs = []
    st.session_state.sources = []
    st.session_state.index = None
    st.session_state.bm25 = None
    st.session_state.messages = []

    st.sidebar.success("System cleared")

# ---------------- System Info ----------------

st.sidebar.divider()
st.sidebar.subheader("📊 System Info")

st.sidebar.write("Documents loaded:", len(set(st.session_state.sources)))
st.sidebar.write("Total chunks:", len(st.session_state.docs))

# ---------------- Chat History ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat Input ----------------

question = st.chat_input("Ask a question about your documents")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.index is None:

            st.warning("Please upload and process documents first")

        else:

            with st.spinner("Searching documents..."):

                docs = st.session_state.docs
                sources = st.session_state.sources

                query_embedding = embedding_model.encode([question])
                query_embedding = np.array(query_embedding, dtype=np.float32)

                faiss.normalize_L2(query_embedding)

                distances, indices = st.session_state.index.search(query_embedding, 20)

                vector_indices = indices[0][:10]

                query_tokens = question.split()
                bm25_scores = st.session_state.bm25.get_scores(query_tokens)
                bm25_top = np.argsort(bm25_scores)[::-1][:10]

                combined_indices = list(set(vector_indices) | set(bm25_top))

                candidate_chunks = []
                candidate_sources = []

                for idx in combined_indices:
                    candidate_chunks.append(docs[idx])
                    candidate_sources.append(sources[idx])

                pairs = [[question, chunk] for chunk in candidate_chunks]

                scores = reranker.predict(pairs)

                ranked = sorted(
                    zip(scores, candidate_chunks, candidate_sources),
                    key=lambda x: x[0],
                    reverse=True
                )

                top_chunks = [x[1] for x in ranked[:5]]
                top_sources = [x[2] for x in ranked[:5]]

                best_score = ranked[0][0]

                if best_score > 0.8:
                    confidence = "high"
                elif best_score > 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"

                
                    context = "\n".join(top_chunks)

                    answer = generate_llm_answer(context, question)

                    st.markdown(answer)

                st.markdown(f"""
                <div class="confidence-box">
                Confidence: {confidence.upper()}
                </div>
                """, unsafe_allow_html=True)

                st.subheader("📚 Sources")

                for i in range(len(top_chunks)):

                    with st.expander(f"📄 {top_sources[i]}"):

                        st.markdown(
                            f"<div class='source-card'>{top_chunks[i]}</div>",
                            unsafe_allow_html=True
                        )

                        st.write("Relevance Score:", round(ranked[i][0], 3))

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )