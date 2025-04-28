import fitz  # PyMuPDF
import re

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

if __name__ == "__main__":
    pdf_file = "papers/SP1.pdf"
    sections = extract_sections_from_pdf(pdf_file)

    for sec, text in sections.items():
        print(f"\n========== {sec} ==========\n")
        print(text[:1000])  # only print first 1000 characters
