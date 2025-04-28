
# Scientific Paper Synthesiser

This is a **Streamlit-based application** that allows users to:
- Upload multiple scientific papers (PDFs),
- Ask questions about the papers,
- Get synthesised answers based on document retrieval and Openai-powered summarization.

---

## Features
- 📄 Upload multiple PDFS.
- 🔍 Retrieve the most relevant chunks using FAISS and re-rank with cosine similarity.
- ✍️ Generate answers based only on the provided documents using OpenAI GPT-3.5.
- 🧠 Keep a full, expandable chat history.
- 🔒 API key is managed securely using the .env` file (no hardcoding).
- 📈 Clean, aesthetic layout with chunked paragraph formatting for easy reading.

---

## Folder Structure
```
scientific-paper-synthesizer/
│
├── app.py                 # Main Streamlit application
├── build_index.py          # (optional future feature) - to build custom indexes
├── parse_pdf.py            # (optional) utility functions to parse PDFs
├── query_engine.py         # Core query and retrieval engine
├── .gitignore              # Git ignore file
├── .env                    # Hidden environment variable file (NOT pushed to GitHub)
└── README.txt              # This README file
```

---

## How to Run Locally

1. Clone this repository:

```bash
git clone https://github.com/AdityaIyer3004/scientific-paper-synthesiser.git
cd scientific-paper-synthesizer
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file:

```
OPENAI_API_KEY=your-openai-api-key-here
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

---

## Deployment

- You can deploy this project for free using platforms like **Streamlit Community Cloud** or **Render**.
- Just make sure your `.env` secrets are properly set in deployment environment variables.

---

## Future Improvements
- Add more efficient retrieval (e.g., hybrid search)
- Include document summarisation before question answering
- Optimise for large document datasets

---

## Credits
- **📝 License**

This project is licensed under the MIT License.

👨‍💻 Developed by Aditya Iyer
🌟 If you found this useful, star ⭐ the repo! 🚀


🌟 Powered by OpenAI, FAISS, Sentence Transformers, and Streamlit.

