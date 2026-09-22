import os
import streamlit as st
from pypdf import PdfReader
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.set_page_config(page_title="PDF RAG Assistant", page_icon="📄", layout="centered")

st.title("📄 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its contents.")

# Get Groq API key from Streamlit Secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("Groq API key is not configured. Add GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_embedding_model()


def extract_text(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    return "\n".join(pages)


def split_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_index(chunks):
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    return index


def retrieve_chunks(question, chunks, index, k=4):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        question_embedding.astype("float32"),
        min(k, len(chunks))
    )

    return [chunks[i] for i in indices[0] if i >= 0]


def ask_groq(question, context):
    prompt = f"""You are a helpful PDF question-answering assistant.

Answer the user's question using only the information provided in the context below.

If the answer cannot be found in the context, say:
"I could not find that information in the uploaded PDF."

Do not make up information.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You answer questions from retrieved PDF context accurately and concisely."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content


uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:
    if (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    ):
        with st.spinner("Processing PDF..."):
            text = extract_text(uploaded_file)

            if not text.strip():
                st.error("Could not extract text from this PDF.")
                st.stop()

            chunks = split_text(text)
            index = create_index(chunks)

            st.session_state.file_name = uploaded_file.name
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.messages = []

        st.success(f"PDF processed successfully — {len(chunks)} chunks created.")

    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask something about the PDF...")

    if question:
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDF..."):
                retrieved = retrieve_chunks(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

                context = "\n\n---\n\n".join(retrieved)
                answer = ask_groq(question, context)

            st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
else:
    st.info("Upload a PDF to get started.")
