# Scientific Paper Synthesiser 📚

This project is an intelligent scientific paper assistant built with **Python**, **Streamlit**, and **OpenAI**.  
It allows users to **upload multiple PDFs**, **ask questions** about them, and **receive synthesised answers**.

---

## ✨ Features

- 📄 Upload multiple scientific papers in PDF format
- 🧠 Retrieve and synthesise answers from the uploaded papers
- ⚡ Fast semantic search using **FAISS** and **Sentence Transformers**
- 🛡️ OpenAI key protected via `.env` file
- 🖥️ Built with an interactive **Streamlit** web app
- 🔍 Shows the **supporting context chunks** used to generate each answer

---

## 🚀 How to Run Locally

1. **Clone this repository**:
   ```bash
   git clone https://github.com/AdityaIyer3004/scientific-paper-synthesiser.git
   cd scientific-paper-synthesiser


##  Create a virtual environment and activate it:

bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows
Install the requirements:

bash
Copy
Edit
pip install -r requirements.txt
Set up your .env file with your OpenAI API key:

ini
Copy
Edit
OPENAI_API_KEY=your-openai-api-key-here
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py

