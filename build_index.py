import os
import fitz  # PyMuPDF
import re
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def extract_sections_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text()

    sections = {
        "Abstract": "",
        "Introduction": "",
        "Methodology": "",
        "Results": "",
        "Conclusion": ""
    }

    current_section = None
    for line in full_text.split("\n"):
        line_stripped = line.strip()

        if re.match(r"^(Abstract|Introduction|Methodology|Methods|Results|Conclusion|Discussion)$", line_stripped, re.IGNORECASE):
            if "method" in line_stripped.lower():
                current_section = "Methodology"
            elif "discussion" in line_stripped.lower():
                current_section = "Conclusion"
            else:
                current_section = line_stripped.capitalize()
            continue

        if current_section:
            sections[current_section] += line + " "

    return sections

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_faiss_index(pdf_path, index_path="faiss.index", metadata_path="metadata.pkl"):
    sections = extract_sections_from_pdf(pdf_path)
    all_chunks = []
    metadata = []

    for section_name, text in sections.items():
        if text.strip():
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            metadata.extend([(section_name, i) for i in range(len(chunks))])

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ FAISS index built with {len(all_chunks)} chunks!")

if __name__ == "__main__":
    build_faiss_index("papers/SP1.pdf")
