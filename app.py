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
from textwrap import dedent

# --- Settings ---
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Load Sentence Transformer Model ---
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()


def extract_text_from_pdf(pdf_file, filename):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    chunks = []
    for i, page in enumerate(doc):  # `i` is the page number
        text = page.get_text()
        page_chunks = chunk_text(text, source=filename)
        for chunk in page_chunks:
            chunk["page"] = i + 1  # Add page number (1-based)
        chunks.extend(page_chunks)
    return chunks


def chunk_text(text, chunk_size=500, source="unknown"):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = []
    current_length = 0
    start_char = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_length + word_count > chunk_size:
            chunk_text = ' '.join(current_chunk)
            end_char = start_char + len(chunk_text)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "start_char": start_char,
                "end_char": end_char
            })
            start_char = end_char
            current_chunk = []
            current_length = 0

        current_chunk.append(sentence)
        current_length += word_count

    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        end_char = start_char + len(chunk_text)
        chunks.append({
            "text": chunk_text,
            "source": source,
            "start_char": start_char,
            "end_char": end_char
        })

    return chunks



def build_faiss_index(chunks):
    texts = [chunk["text"] for chunk in chunks]
    
    if not texts:
        raise ValueError("No text chunks provided to build the FAISS index.")
    
    embeddings = embed_model.encode(texts)
    
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)  # reshape if only one embedding
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return index, embeddings

def score_chunks_with_gpt(chunks, question, model_name="gpt-3.5-turbo"):
    client = OpenAI(api_key=OPENAI_API_KEY)

    chunk_texts = [f"Chunk {i+1}:\n{chunk['text']}" for i, chunk in enumerate(chunks)]
    formatted_chunks = "\n\n".join(chunk_texts)

    prompt = dedent(f"""
    You are evaluating the usefulness of the following text chunks in answering the question:

    Question: {question}

    Score each chunk from 1 to 10 based on:
    - 10 = Directly answers or strongly supports the question
    - 7–9 = Partial answer or good support
    - 4–6 = General context, weakly related
    - 1–3 = Irrelevant or off-topic

    Return the scores as a plain list of numbers in order (e.g., 7, 6, 3, 10...)

    Chunks:
    {formatted_chunks}
    """)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )

    raw_scores = response.choices[0].message.content
    return parse_scores(raw_scores, len(chunks))

def synthesize_context(chunks, question, model_name="gpt-4-turbo"):
    client = OpenAI(api_key=OPENAI_API_KEY)
    chunk_texts = "\n\n".join([f"[Chunk {i+1}]: {chunk['text']}" for i, chunk in enumerate(chunks)])

    prompt = dedent(f"""
    You are a research assistant. Given the following chunks from a scientific document and a specific question, 
    synthesize the chunks into a concise, logically structured, and coherent context for answering the question.

    Only include information that is relevant to the question.
    Do NOT copy the chunks directly. Summarize, combine, and rephrase where necessary.

    Question: {question}

    Chunks:
    {chunk_texts}

    Synthesized Context:
    """)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def parse_scores(raw_output, expected_count):
    scores = re.findall(r"\b(10|[1-9])\b", raw_output)
    scores = list(map(int, scores))
    if len(scores) != expected_count:
        return [5] * expected_count  # fallback if GPT output is unexpected
    return scores

def retrieve_relevant_chunks(query, index, chunks, embeddings, top_k=8, final_k=3):
    query_vector = embed_model.encode(query)
    query_vector = np.array(query_vector).reshape(1, -1)

    distances, indices = index.search(query_vector, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    gpt_scores = score_chunks_with_gpt(retrieved_chunks, query, model_name=model_choice)

    ranked = sorted(zip(gpt_scores, retrieved_chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [chunk for score, chunk in ranked[:final_k]]
    top_scores = [score for score, chunk in ranked[:final_k]]

    return top_chunks, top_scores


model_choice = st.selectbox(
    "🧠 Choose OpenAI model:",
    ["gpt-3.5-turbo", "gpt-4-turbo"],
    index=1  # default to GPT-4 Turbo
)

def generate_answer(context, question, model_name="gpt-4-turbo"):
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""You are a highly skilled AI trained to extract scientific knowledge from academic papers.

Your task is to answer the following question using ONLY the context provided. Do not use external knowledge.

- If the answer is explicitly mentioned, quote and explain it.
- If it is implied, infer logically and provide reasoning.
- If the answer is not directly stated but can be reasonably inferred from the context, do so and explain your reasoning. Only if it cannot be inferred at all, say: "The answer is not available in the provided documents."

Think step by step if needed.

Context:
{context}

Question:
{question}

Answer:"""


    response = client.chat.completions.create(
        model=model_name,  # ✅ Now uses user's selected model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("📚 Scientific Paper Synthesizer & QA Engine")
st.write("Upload research papers, summarize key findings, and get instant answers from them!")

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
        filename = uploaded_file.name
        chunks = extract_text_from_pdf(uploaded_file, filename)
        all_chunks.extend(chunks)
        all_chunk_sources.extend([filename] * len(chunks))

    # Only build index if there are chunks
    if all_chunks:
        index, embeddings = build_faiss_index(all_chunks)
        st.success(f"✅ {len(uploaded_files)} paper(s) loaded and indexed!")
    else:
        st.warning("⚠️ No valid text was extracted from the uploaded files.")


st.success(f"✅ {len(uploaded_files)} paper(s) loaded and indexed!")

# 🔥 INSERT your "Summarize Selected Paper" UI here 🔥
selected_file = st.selectbox(
    "📄 Choose a paper to summarize:",
    list(set(all_chunk_sources))
)

if st.button("📝 Summarize Selected Paper"):
    if all_chunks and selected_file:
        with st.spinner('Summarizing selected paper...'):
            selected_chunks = [
                chunk for chunk, source in zip(all_chunks, all_chunk_sources)
                if source == selected_file
            ]
            if selected_chunks:
                full_context = "\n\n".join([chunk["text"] for chunk in selected_chunks])
                summary = summarize_document(full_context)
                st.success(f"✅ Summary for {selected_file} generated!")
                st.text_area("📜 Paper Summary:", value=summary, height=300)
            else:
                st.warning("⚠️ No content found for the selected file.")
    else:
        st.warning("⚠️ Please upload papers first.")


if st.button("📝 Summarize All Uploaded Papers"):
    if all_chunks:
        with st.spinner('Summarizing papers...'):
            full_context = "\n\n".join([chunk["text"] for chunk in all_chunks])
            summary = summarize_document(full_context, model_name=model_choice)
            st.success("✅ Summary Generated!")
            st.text_area("📜 Paper Summary:", value=summary, height=300)
    else:
        st.warning("⚠️ Please upload papers first.")
        
question = st.text_input("❓ Ask a question:")        


if question:
    with st.spinner('Synthesizing answer...'):
        retrieved_chunks, relevance_scores = retrieve_relevant_chunks(
            question, index, all_chunks, embeddings
        )
        retrieved_sources = [all_chunk_sources[all_chunks.index(chunk)] for chunk in retrieved_chunks]

        synthesized_context = synthesize_context(retrieved_chunks, question, model_name=model_choice)

        answer = generate_answer(synthesized_context, question, model_name=model_choice)


        if retrieved_sources:
            primary_source = max(set(retrieved_sources), key=retrieved_sources.count)
            tagged_answer = f"**Answer mostly based on {primary_source}:**\n\n{answer}"
        else:
            tagged_answer = answer

        st.session_state.chat_history.append({
            "question": question,
            "answer": tagged_answer,
            "supporting_chunks": retrieved_chunks,
            "relevance_scores": relevance_scores
        })




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

            # Optional toggle to show metadata
            if st.checkbox(f"🔎 Show Metadata for: {chat['question']}", key=f"meta_{chat['question']}_{chat['answer'][:10]}"):
                for idx, chunk in enumerate(chat["supporting_chunks"]):
                    st.markdown(f"""
                        <div style='font-size: 15px; color: #555; margin-bottom: 10px;'>
                            <b>Chunk {idx+1} Metadata:</b><br>
                            • <b>Source:</b> {chunk.get("source", "N/A")}<br>
                            • <b>Page:</b> {chunk.get("page", "N/A")}<br>
                            • <b>Characters:</b> {chunk.get("start_char", 0)} – {chunk.get("end_char", 0)}
                        </div>
                    """, unsafe_allow_html=True)

            # --- Show Supporting Context Chunks with Relevance Score ---
            st.markdown(f"<br><b>🔎 Supporting Context Chunks:</b>", unsafe_allow_html=True)
            for idx, chunk in enumerate(chat["supporting_chunks"]):
                paragraphs = re.split(r'(?<=[.!?])\s+', chunk["text"].strip())
                short_paragraphs = []
                current = ""

                for para in paragraphs:
                    if len(current.split()) + len(para.split()) < 80:
                        current += " " + para
                    else:
                        short_paragraphs.append(current.strip())
                        current = para
                if current:
                    short_paragraphs.append(current.strip())

                relevance_score = chat.get("relevance_scores", [None]*len(chat["supporting_chunks"]))[idx]
                score_display = f"<span style='font-size: 14px; color: #888;'>🔍 Relevance Score: {relevance_score}/10</span>"

                st.markdown(f"""
                    <div style='padding: 8px; margin-top:5px; margin-bottom:5px; background-color: #ffffff; border-radius: 6px;'>
                        <b>Chunk {idx+1}:</b> {score_display}<br>
                        <div style='font-size: 17px; color: #111111; line-height: 1.6;'>
                            {"<br><br>".join(short_paragraphs)}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
