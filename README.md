# ClinicalMind — Multi-Agent Clinical Reasoning System

A production-grade multi-agent AI system that answers complex clinical questions by coordinating five specialised agents. Each agent has a distinct role — retrieval, guideline checking, analysis, hallucination detection, and synthesis. No agent ever diagnoses. Every answer is grounded in uploaded documents with cited sources.

**Live demo:** https://your-streamlit-url.streamlit.app

---

## What it does

Upload medical PDFs — clinical guidelines, WHO reports, research papers. Ask complex clinical questions. ClinicalMind coordinates five agents to retrieve evidence, check protocols, reason across documents, verify for hallucinations, and return a structured answer with confidence level and page citations.

**Example queries:**
- "Compare health outcomes between high income and low income countries and what interventions does WHO recommend to close this gap?"
- "What is the relationship between poverty and life expectancy based on the evidence in these documents?"
- "What does the document recommend for reducing child mortality in low income countries?"

---

## Architecture
```
User question
      |
      v
Orchestrator agent (LangGraph StateGraph)
      |
      |-----> Retrieval agent    — searches FAISS vector store, returns top 5 chunks with citations
      |-----> Guideline agent    — finds protocol and recommendation passages specifically
      |-----> Analysis agent     — reasons over retrieved evidence, draws clinical connections
      |-----> Critic agent       — checks analysis for hallucinations, assigns HIGH / MEDIUM / LOW confidence
      |
      v
Aggregator agent
      |
      v
Final answer — citations + confidence badge + disclaimer
      |
      v
Streamlit UI
```

---

## The five agents

| Agent | Role | Model |
|---|---|---|
| Orchestrator | Breaks query into sub-tasks, coordinates all agents via LangGraph StateGraph | GPT-4o-mini |
| Retrieval | Semantic search over FAISS vector store, returns top 5 chunks with source and page | GPT-4o-mini |
| Guideline | Targets protocol and recommendation passages specifically | GPT-4o-mini |
| Analysis | Reasons over retrieved evidence, connects findings across documents | GPT-4o-mini |
| Critic | Checks analysis output for hallucinations, flags anything not grounded in documents, assigns confidence | GPT-4o-mini |
| Aggregator | Synthesises all outputs into one final structured answer | GPT-4o-mini |

---

## Tech stack

| Component | Technology |
|---|---|
| Agent orchestration | LangGraph StateGraph |
| LLM | GPT-4o-mini |
| Vector database | FAISS |
| Embeddings | OpenAI text-embedding-3-small |
| Document loading | LangChain PyPDFLoader |
| Chunking | RecursiveCharacterTextSplitter |
| UI | Streamlit |
| Language | Python 3.11 |

---

## What makes this different from a chatbot

A standard RAG chatbot retrieves chunks and generates an answer in one step. ClinicalMind uses a multi-agent pipeline where each agent specialises in one part of the reasoning process:

1. Retrieval is separated from reasoning — the retrieval agent finds evidence, the analysis agent interprets it
2. Guidelines are checked independently — the guideline agent specifically targets protocol passages, not just any relevant text
3. Hallucination detection is built in — the critic agent verifies the analysis against the retrieved evidence before the answer reaches the user
4. Confidence is explicit — every answer carries a HIGH / MEDIUM / LOW confidence rating based on how well the evidence supports the conclusion

---

## Ethical design

- No agent ever generates a diagnosis
- Every answer is grounded in uploaded documents only — no internet access, no training data leakage
- If the answer is not in the documents the critic agent flags LOW confidence and the system says so explicitly
- Every response carries a disclaimer: research and informational purposes only, not medical advice
- No patient data is stored — documents are processed in-session only

---

## Project structure
```
clinicalmind/
├── app.py              # Streamlit UI
├── orchestrator.py     # LangGraph StateGraph — coordinates all agents
├── agents.py           # Five specialised agents
├── ingest.py           # PDF ingestion + FAISS vectorstore builder
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies for Streamlit Cloud
├── .env                # API keys (not committed)
├── .gitignore
├── data/               # Uploaded PDFs (not committed)
└── vectorstore/        # FAISS index (not committed)
```

---

## Running locally
```bash
# Clone the repo
git clone https://github.com/Umer-Mahmood-Khan/clinicalmind.git
cd clinicalmind

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key to .env
OPENAI_API_KEY=sk-your-key-here

# Add a medical PDF to data/
# Then run the app
python -m streamlit run app.py
```

---

## Streamlit Cloud deployment
```
requirements.txt  — Python packages
packages.txt      — System packages (poppler-utils, tesseract-ocr)
```

Add secrets in Streamlit Cloud settings:
```toml
OPENAI_API_KEY = "sk-your-key-here"
```

---

## Context

Built as the third project in a medical AI portfolio developed at the National Center for Artificial Intelligence (NCAI), Pakistan. The system extends the RAG pipeline from MedRAG into a full multi-agent reasoning architecture.

Related projects in the same pipeline:
- **RadiOCR** — OCR extraction layer for scanned radiology reports
- **MedRAG** — RAG knowledge retrieval layer with AWS S3 integration
- **ClinicalMind** — multi-agent reasoning layer (this project)

Together these three systems form a complete clinical document intelligence pipeline — from raw scanned reports to structured extraction to deep multi-agent reasoning.

---

## Built by

**Umer Mahmood Khan**
AI Research Engineer at National Center for Artificial Intelligence (NCAI), Pakistan

- RadiOCR: https://cfxf4o325p6lwux8me8ugz.streamlit.app
- MedRAG: https://medrag-hck3ynxroca5zi7fkj9vapp.streamlit.app
- GitHub: https://github.com/Umer-Mahmood-Khan
