
# NeuroPaper AI

A production-grade Retrieval-Augmented Generation (RAG) system for scientific paper analysis. Upload research PDFs, query them with natural language, and get precise answers grounded in the source material — with LLM-as-a-Judge evaluation and input guardrails on every response.

**Stack:** FastAPI · FAISS · SentenceTransformers · Groq (Llama 3.3 70B) · Docker · Google Cloud Run

---

## What It Does

- Ingests multi-paper PDF corpora and indexes them into a FAISS vector store
- Retrieves semantically relevant chunks using dense embeddings (384-dim)
- Scores and re-ranks retrieved chunks using an LLM with a multi-criteria rubric
- Synthesizes ranked chunks via chain-of-thought before answering
- Evaluates every answer with a secondary LLM-as-Judge on 4 dimensions
- Runs input guardrails to block off-topic, unsafe, or injection-attempt queries
- Generates structured 7-section academic summaries of uploaded papers

---

## Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────┐
│              INGESTION PIPELINE              │
│  PyMuPDF → sentence chunking (500 words)    │
│  SentenceTransformer (all-MiniLM-L6-v2)     │
│  FAISS IndexFlatL2 (384-dim L2 search)      │
└─────────────────────────────────────────────┘
    │
    ▼ Query
┌─────────────────────────────────────────────┐
│              GUARDRAIL LAYER                 │
│  Llama 3.1 8B — checks:                     │
│  · safe (no injection / jailbreak)          │
│  · on_topic (academic content)              │
│  · specific (answerable from paper)         │
└─────────────────────────────────────────────┘
    │ pass
    ▼
┌─────────────────────────────────────────────┐
│          RETRIEVAL + LLM RE-RANKING          │
│  FAISS top-8 ANN search                     │
│  LLM multi-criteria scorer (Llama 3.3 70B)  │
│  → Topical Match / Evidence Quality /       │
│    Contextual Value (0–4 / 0–3 / 0–3)      │
│  Top-3 chunks selected                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│          CHAIN-OF-THOUGHT SYNTHESIS          │
│  Extract → Connect → Reconcile → Synthesize │
│  Dense analytical briefing as RAG context   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              ANSWER GENERATION               │
│  Expert researcher persona + reasoning      │
│  protocol (explicit/inferred/metric cite)   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│           LLM-AS-JUDGE EVALUATION            │
│  Llama 3.1 8B — scores:                     │
│  · Faithfulness   · Relevance               │
│  · Completeness   · Groundedness            │
│  + Hallucination Risk (low/medium/high)     │
│  + One-sentence verdict                     │
└─────────────────────────────────────────────┘
    │
    ▼
  Response + Eval Scores → Frontend
```

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | FastAPI | Async, type-safe, auto OpenAPI docs |
| Vector Search | FAISS IndexFlatL2 | In-memory ANN, zero infra overhead for demo scale |
| Embeddings | all-MiniLM-L6-v2 (384-dim) | Best speed/quality tradeoff for semantic search |
| LLM Inference | Groq API — Llama 3.3 70B | ~10x faster than OpenAI, free tier |
| PDF Parsing | PyMuPDF (fitz) | Fastest Python PDF library, accurate text extraction |
| Deployment | Google Cloud Run | Serverless containers, scales to zero, pay-per-request |
| Frontend | Vanilla JS + CSS | Zero framework overhead, full custom UI |

---

## RAG Pipeline — Technical Deep Dive

### 1. Chunking Strategy
Text is split on sentence boundaries (`(?<=[.!?]) +`) into 500-word windows. Sentence-aware chunking preserves semantic coherence — mid-sentence cuts degrade embedding quality and retrieval precision.

### 2. Embedding & Indexing
`all-MiniLM-L6-v2` produces 384-dimensional dense vectors. FAISS `IndexFlatL2` performs exact L2 nearest-neighbor search. For this scale (100–500 chunks), exact search outperforms ANN approximations in accuracy with negligible latency difference.

### 3. Multi-Criteria LLM Re-Ranking
FAISS retrieves top-8 candidates by vector similarity. A second LLM pass scores each chunk on three axes:
- **Topical Match (0–4):** Direct relevance to the question
- **Evidence Quality (0–3):** Specific data, theorems, experimental results
- **Contextual Value (0–3):** Definitions, background, causal mechanisms

This separates *semantic similarity* (what FAISS measures) from *answer utility* (what actually matters).

### 4. Chain-of-Thought Synthesis
Rather than feeding raw chunks to the answer model, a 4-step synthesis pass runs first:
1. **Extract** — facts, claims, metrics from each chunk
2. **Connect** — relationships and causal links between chunks
3. **Reconcile** — flag contradictions, identify most specific source
4. **Synthesize** — dense analytical briefing preserving exact numbers and terminology

### 5. LLM-as-Judge Evaluation
Every answer is evaluated by a fast secondary model (`llama-3.1-8b-instant`) on:
- **Faithfulness** — are all claims grounded in the retrieved context?
- **Relevance** — does the answer address the actual question?
- **Completeness** — does it cover all aspects the context supports?
- **Groundedness** — are specific numbers, names, theorems reproduced accurately?

The judge also assigns `hallucination_risk: low | medium | high` and a one-sentence verdict.

### 6. Input Guardrails
Before any expensive LLM call, a fast guardrail model classifies the query:
- `safe` — detects prompt injection, jailbreak attempts, harmful content
- `on_topic` — ensures the query is about academic/scientific content
- `specific` — checks the query is answerable from a document

---

## Advanced Prompting Techniques Used

| Technique | Where Applied |
|---|---|
| **System role / persona** | All 4 LLM calls — scopes model behavior precisely |
| **Chain-of-thought (CoT)** | Synthesis step — 4-step explicit reasoning chain |
| **Structured output** | Judge, guardrail — JSON-only responses with regex fallback |
| **Multi-criteria rubric** | Chunk scorer — explicit sub-score breakdown |
| **Reasoning protocol** | Answer generation — model reasons before writing |
| **Calibrated confidence** | Answer prompt — distinguishes evidence types explicitly |
| **Constitutional constraints** | All prompts — "use ONLY the context", "do NOT hallucinate" |
| **Temperature tuning** | 0.0 guardrail → 0.1 judge → 0.2 answer → 0.3 synthesis |

---

## API Reference

```
POST /api/upload         Upload and index one or more PDF files
POST /api/query          Query indexed papers with natural language
POST /api/summarize      Generate structured 7-section paper summary
GET  /api/files          List currently indexed files
```

### Query Response Schema
```json
{
  "answer": "string",
  "source": "filename.pdf",
  "guard": { "safe": true, "on_topic": true, "specific": true, "flag": null },
  "eval": {
    "faithfulness": 9, "relevance": 8, "completeness": 7, "groundedness": 9,
    "hallucination_risk": "low",
    "verdict": "Accurate, well-grounded answer citing specific experimental results."
  },
  "chunks": [{ "text": "...", "source": "filename.pdf", "page": 4, "score": 9 }]
}
```

---

## Local Setup

```bash
git clone https://github.com/Adityaiyer3004/scientific-paper-synthesiser
cd scientific-paper-synthesiser

python -m venv .venv
source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

echo "GROQ_API_KEY=your_key_here" > .env
# Free key at console.groq.com/keys

uvicorn app:app --host 0.0.0.0 --port 8080
# Open http://localhost:8080
```

---

## Deployment — Google Cloud Run

```bash
gcloud auth login
gcloud projects create neuropaper-ai --name="NeuroPaper AI"
gcloud config set project neuropaper-ai
gcloud services enable run.googleapis.com containerregistry.googleapis.com cloudbuild.googleapis.com

# Build container (~10-15 min first time)
gcloud builds submit --tag gcr.io/neuropaper-ai/neuropaper

# Deploy
gcloud run deploy neuropaper \
  --image gcr.io/neuropaper-ai/neuropaper \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 120 \
  --set-env-vars GROQ_API_KEY=your_key_here
```

---

## Interview Preparation — Lead / Senior Engineer

### System Design

**Q: Walk me through your RAG architecture and key design decisions.**

The pipeline has 5 stages: ingest → guardrail → retrieve+rerank → synthesize → evaluate. Key decisions: (1) sentence-boundary chunking over fixed-character splits to preserve semantic units; (2) two-stage retrieval where FAISS handles speed and an LLM handles quality — pure vector similarity conflates semantic proximity with answer utility; (3) a CoT synthesis pass between retrieval and answering so the answer model gets organized, dense context; (4) a secondary judge model instead of self-evaluation, since models are poorly calibrated when scoring their own outputs.

**Q: Why FAISS over managed vector stores like Pinecone or Weaviate?**

For hundreds to low thousands of chunks, FAISS `IndexFlatL2` gives exact nearest-neighbor search with sub-millisecond latency and zero infrastructure overhead. Managed stores add network latency, cost, and operational complexity that only pays off at millions of vectors or when you need persistence across deployments. The tradeoff is that the index resets on cold start — acceptable here, but would need addressing in production via a persistent vector store or serializing the index to Cloud Storage.

**Q: How would you scale this to thousands of concurrent users?**

Three changes: (1) replace in-memory FAISS with a persistent vector store so the index survives restarts and is shared across replicas; (2) add a document store (PostgreSQL or Firestore) to persist uploaded papers; (3) cache embeddings and search results for identical queries. Cloud Run handles horizontal scaling automatically. For high throughput, move embedding/indexing to an async background job queue so uploads don't block requests.

**Q: How would you add multi-turn conversation support?**

Maintain session-scoped conversation history. On each turn, use query rewriting — pass the history + current query to a rewrite model to produce a standalone search query before hitting FAISS. Without rewriting, follow-up questions like "what about the limitations?" retrieve irrelevant chunks because they lack context of what "it" refers to.

---

### Retrieval & Embeddings

**Q: Why all-MiniLM-L6-v2 over text-embedding-ada-002?**

Deliberate tradeoff. MiniLM runs at ~14,000 sentences/second on CPU with an 80MB model size. ada-002 requires an API call with ~200ms latency per batch and costs per token. For self-hosted RAG where embeddings run at both index time and query time, local inference wins on latency and cost. The quality gap is measurable on benchmarks but negligible in practice for domain-specific scientific text where vocabulary overlap between query and document is high.

**Q: When would you use keyword (BM25) search over semantic (dense) search?**

Dense search excels at semantic similarity — "what causes gradient vanishing" finds chunks about "exploding gradients" without exact term overlap. BM25 excels at exact-match recall — rare technical terms, specific model names, dataset names, author names, equation labels. Production RAG systems use hybrid search: BM25 + dense retrieval with reciprocal rank fusion, then reranking. This covers both semantic and lexical relevance.

**Q: How do you handle relevant content spread across many sections of a long paper?**

Hierarchical retrieval: first retrieve at section level, then at chunk level within relevant sections. Another approach is parent-child chunking — embed small chunks for precise retrieval, but return the surrounding larger context window to the LLM for richer context.

---

### Evaluation & Reliability

**Q: What are the limitations of LLM-as-a-Judge?**

Three failure modes: (1) **self-serving bias** — same model family tends to rate its own outputs highly; mitigated here by using a smaller, different-temperature model; (2) **position bias** — judges favor answers that are longer or presented first; (3) **calibration drift** — scores aren't absolute across calls. In production, validate judge scores against human labels on a calibration set.

**Q: How do you evaluate RAG quality beyond LLM-as-judge?**

Three dimensions: (1) **Retrieval** — precision@k and recall@k against ground truth relevant chunks; (2) **Generation** — faithfulness, answer relevance, answer correctness via RAGAS framework; (3) **End-to-end** — human evaluation on a test set with known correct answers.

**Q: How do you handle hallucination?**

Three layers: (1) **Prompt constraints** — explicit instruction to use only the provided context, with a fallback phrase for insufficient context; (2) **CoT synthesis** — pre-organized, fact-dense context reduces the model's need to fill gaps; (3) **Post-hoc evaluation** — the judge's faithfulness score flags unsupported claims. Production addition: extract claims from the answer and verify each against source chunks automatically.

---

### Prompt Engineering

**Q: Why use a system prompt instead of putting everything in the user message?**

System prompts establish persistent behavioral constraints the model treats as authoritative — role, output format, prohibited behaviors. User prompts provide dynamic per-request content. Mixing them causes the model to weight constraints and content equally, leading to instruction drift on longer prompts. System prompts are also cached by inference providers (Groq caches the prefix), reducing latency and cost.

**Q: How did you decide on temperature values for each call?**

Temperature controls the creativity/determinism tradeoff. Guardrail: `0.0` — classification needs determinism. Judge: `0.1` — above zero to avoid degenerate outputs, consistent scoring. Answer: `0.2` — low for factual grounding. Synthesis/summary: `0.3` — mild creativity for fluent prose while facts stay grounded. Using the same temperature everywhere is a common mistake.

**Q: What's the difference between your synthesis step and passing chunks directly?**

Directly passing raw chunks forces the answer model to simultaneously do retrieval comprehension, fact extraction, cross-chunk reasoning, and answer formulation in one pass. The synthesis step separates concerns — a dedicated call handles comprehension and organization, then the answer model focuses purely on answering with clean, structured context. This reduces missed information that was present in the chunks but not surfaced in the answer.

---

### MLOps & Production

**Q: How would you monitor this in production?**

Four pillars: (1) **Latency** — trace each pipeline stage separately; P95 per stage; (2) **Quality** — sample requests, log judge scores, alert if mean faithfulness drops; (3) **Reliability** — guardrail block rate (spike = abuse), error rate per endpoint, cold start frequency; (4) **Cost** — Groq token usage per request, flag unexpectedly expensive requests.

**Q: How would you optimize cold start time?**

Bake model weights into the Docker image during build: `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`. This increases image size but eliminates runtime download. For Cloud Run, set minimum instances to 1 to eliminate cold starts entirely for active deployments.

**Q: How would you A/B test a new prompting strategy?**

Use Cloud Run's traffic splitting to route 10% of traffic to a revision with the new prompts. Log judge scores, latency, and user feedback per revision. Run until statistical significance (~500–1000 requests per variant). Compare score distributions, not just means — a prompt that improves average faithfulness but increases variance may be worse in practice.

---

## Project Structure

```
scientific-paper-synthesiser/
├── app.py                 # FastAPI application — all endpoints and pipeline
├── backend.py             # Mirror of app.py
├── requirements.txt       # Python dependencies
├── Dockerfile             # Cloud Run container definition
├── .dockerignore          # Excludes .env, .venv, PDFs from image
├── .gitignore             # Excludes secrets and local artifacts
├── .env                   # GROQ_API_KEY (never committed)
└── static/
    └── index.html         # Frontend — vanilla JS/CSS, no framework
```

---

## Key Engineering Decisions & Tradeoffs

| Decision | Alternative | Why This Choice |
|---|---|---|
| FAISS in-memory | Pinecone / Weaviate | Zero infra, sub-ms search at this scale; tradeoff: no persistence |
| Groq + Llama 3.3 70B | OpenAI GPT-4 | ~10x faster inference, free tier, comparable quality for RAG |
| Separate scorer + answer model | Single LLM call | Separates retrieval quality from answer quality; easier to debug |
| CoT synthesis before answering | Direct chunk → answer | Reduces hallucination by pre-organizing context |
| Fast 8B judge model | Same model as generator | Avoids self-serving bias; 5x cheaper; low latency |
| Sentence-boundary chunking | Fixed character windows | Preserves semantic units; better embedding quality |
| CPU torch in Dockerfile | Full CUDA torch | 3GB vs 8GB image; Cloud Run has no GPU |
| Cloud Run | GKE / App Engine | Serverless, scales to zero, simplest path for this architecture |

---

*Built by Aditya Iyer · [github.com/Adityaiyer3004](https://github.com/Adityaiyer3004)*

