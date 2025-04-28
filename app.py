import streamlit as st
import fitz  # PyMuPDF
import faiss
import pickle
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os

# --- Settings ---
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Load Sentence Transformer Model ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper Functions ---
def extract_text_from_pdf(pdf_file, filename):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text, filename




def chunk_text(text, chunk_size=500):
    # 1. Split nicely into sentences
    sentences = re.split(r'(?<=[.!?]) +', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        # If adding this sentence crosses limit
        if current_length + word_count > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += word_count

    # Add final leftover chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def retrieve_relevant_chunks(query, index, chunks, embeddings, top_k=5, final_k=3):
    query_vector = embed_model.encode(query)
    query_vector = np.array(query_vector).reshape(1, -1)  # <-- FIX

    # Step 1: FAISS fast retrieval
    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    retrieved_vectors = embeddings[indices[0]]

    # Step 2: Semantic re-ranking
    similarities = cosine_similarity(query_vector, retrieved_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1]

    # Step 3: Pick top 'final_k' chunks
    top_chunks = [retrieved_chunks[i] for i in ranked_indices[:final_k]]

    return top_chunks

def generate_answer(context, question):
    client = OpenAI(api_key=OPENAI_API_KEY)
    prompt = f"Answer the following question based ONLY on the context below. If the answer is not present, say 'The information is not available in the provided documents.'\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("📚 Multi-Paper Scientific Q&A")
st.write("Upload one or more scientific papers and ask questions about them!")

# --- Initialize session state for chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_files = st.file_uploader(
    "Upload your scientific papers (.pdf)", 
    type=["pdf"], 
    accept_multiple_files=True
)

all_chunks = []
all_chunk_sources = []  # NEW: Keep track of where each chunk came from

if uploaded_files:
    for uploaded_file in uploaded_files:
        full_text, filename = extract_text_from_pdf(uploaded_file, uploaded_file.name)
        chunks = chunk_text(full_text)
        all_chunks.extend(chunks)
        all_chunk_sources.extend([filename] * len(chunks))

index, embeddings = build_faiss_index(all_chunks)

st.success(f"✅ {len(uploaded_files)} paper(s) loaded and indexed!")  # <--- move INSIDE


question = st.text_input("❓ Ask a question:")

if question:
    with st.spinner('Synthesizing answer...'):
        retrieved_chunks = retrieve_relevant_chunks(question, index, all_chunks, embeddings)
        retrieved_sources = [all_chunk_sources[all_chunks.index(chunk)] for chunk in retrieved_chunks]

        combined_context = "\n\n".join(retrieved_chunks)
        answer = generate_answer(combined_context, question)

        if retrieved_sources:
            primary_source = max(set(retrieved_sources), key=retrieved_sources.count)
            tagged_answer = f"**Answer mostly based on {primary_source}:**\n\n{answer}"
        else:
            tagged_answer = answer

        st.session_state.chat_history.append({
            "question": question,
            "answer": tagged_answer,
            "supporting_chunks": retrieved_chunks
        })



# --- Display full chat history ---
# --- Display full chat history ---
if st.session_state.chat_history:
    for chat in reversed(st.session_state.chat_history):
        with st.expander(f"🧠 {chat['question']}", expanded=False):
            st.markdown(f"""
                <div style='padding: 10px; background-color: #f5f7fa; border-radius: 8px;'>
                    <h4 style='margin-bottom: 10px; color: #1a1a1a;'>🤖 Answer:</h4>
                    <div style='font-size: 16px; color: #333333;'>{chat['answer']}</div>
                </div>
            """, unsafe_allow_html=True)

            # --- Show Supporting Context Chunks ---
            if "supporting_chunks" in chat:
                st.markdown(f"<br><b>🔎 Supporting Context Chunks:</b>", unsafe_allow_html=True)
                for idx, chunk in enumerate(chat["supporting_chunks"]):
                    
                    # Split chunk into smaller paragraphs
                    paragraphs = re.split(r'(?<=[.!?])\s+', chunk.strip())
                    short_paragraphs = []
                    current = ""

                    for para in paragraphs:
                        if len(current.split()) + len(para.split()) < 80:  # 80 words per paragraph
                            current += " " + para
                        else:
                            short_paragraphs.append(current.strip())
                            current = para
                    if current:
                        short_paragraphs.append(current.strip())

                    # ✅ Now print nicely INSIDE the loop
                    st.markdown(f"""
                    <div style='padding: 8px; margin-top:5px; margin-bottom:5px; background-color: #ffffff; border-radius: 6px;'>
                        <b>Chunk {idx+1}:</b><br>
                        <div style='font-size: 17px; color: #111111; line-height: 1.6;'>
                            {"<br><br>".join(short_paragraphs)}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
