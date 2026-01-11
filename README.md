<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.109-green?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenAI-GPT--4-orange?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/RAG-Powered-purple?style=for-the-badge" />
</p>

<h1 align="center">Customer Intelligence AI</h1>

<p align="center">
  <strong>Production-grade AI system that transforms raw customer feedback into actionable business intelligence using RAG (Retrieval-Augmented Generation)</strong>
</p>

<p align="center">
  <a href="#-the-problem">Problem</a> •
  <a href="#-the-solution">Solution</a> •
  <a href="#-live-demo">Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-reference">API</a>
</p>

---

##  The Problem

**Every SaaS company struggles with the same challenge:**

Companies collect customer feedback from multiple channels — surveys, support tickets, app reviews, social media. This data contains valuable insights, but:

| Challenge | Impact |
|-----------|--------|
|  **Volume** | Thousands of feedback items impossible to read manually |
|  **Unstructured** | Free-text complaints can't be queried with SQL |
|  **Slow Insights** | By the time trends are spotted, customers have churned |
|  **Hallucination Risk** | ChatGPT makes up answers not grounded in real data |
|  **No Tracking** | No way to know if AI quality is degrading over time |

**Result:** Companies are flying blind on what customers actually think.

---

## 💡 The Solution

**Customer Intelligence AI** is an end-to-end system that:

### Ingests Any Format
Upload CSV, PDF, or text files containing customer feedback — the system handles parsing automatically.

###  Understands Meaning, Not Just Keywords  
Uses **semantic embeddings** to understand that "app is sluggish" and "performance is terrible" are the same complaint.

###  Auto-Discovers Themes
**ML clustering** automatically groups similar complaints and identifies the top issues without manual labeling.

###  Answers Questions in Plain English
Ask "What are customers unhappy about?" and get a synthesized answer with **citations to real feedback**.

###  Never Hallucinates
**RAG (Retrieval-Augmented Generation)** ensures every answer is grounded in actual customer data.

###  Built for Production
Docker-ready with health checks, error handling, and **evaluation metrics** to catch quality regressions.

---

## Live Demo

### Step 1: Upload Customer Feedback
```bash
curl -X POST http://localhost:8000/upload-data \
  -F "file=@data/sample/feedback.csv"
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully processed 100 feedback records",
  "records_processed": 100,
  "total_indexed": 100
}
```

### Step 2: Ask Business Questions
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the top 3 customer complaints?"}'
```

**Response:**
```json
{
  "answer": "Based on the customer feedback, the top 3 complaints are:\n\n1. **Performance Issues** (28% of complaints): Customers report slow loading times and app freezes. Example: \"The app is extremely slow when loading my dashboard. Takes over 30 seconds!\"\n\n2. **Poor Support Response** (22%): Multiple customers frustrated with support wait times. Example: \"Customer support never responds to my tickets. Been waiting 2 weeks!\"\n\n3. **Bugs and Technical Issues** (18%): Users encountering data loss and crashes. Example: \"Found a bug: clicking save sometimes loses all my changes.\"",
  "citations": [
    {"text": "The app is extremely slow...", "source": "feedback.csv", "relevance": 0.89},
    {"text": "Customer support never responds...", "source": "feedback.csv", "relevance": 0.85}
  ],
  "num_sources": 5
}
```

### Step 3: Discover Themes Automatically
```bash
curl -X POST http://localhost:8000/cluster \
  -H "Content-Type: application/json" \
  -d '{"n_clusters": 5}'
```

**Response:**
```json
{
  "total_feedback": 100,
  "num_themes": 5,
  "quality_score": 0.412,
  "themes": [
    {"id": 0, "size": 28, "percentage": 28.0, "keywords": ["slow", "loading", "performance"]},
    {"id": 1, "size": 22, "percentage": 22.0, "keywords": ["support", "response", "ticket"]},
    {"id": 2, "size": 18, "percentage": 18.0, "keywords": ["bug", "error", "crash"]}
  ]
}
```

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CUSTOMER INTELLIGENCE AI                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   DATA INGESTION          ML PROCESSING           RAG ENGINE           │
│   ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       │
│   │ • CSV Parser    │──────▶│ • TF-IDF        │──────▶│ • Embeddings    │       │
│   │ • PDF Extractor │       │ • K-Means       │       │ • FAISS Index   │       │
│   │ • Text Loader   │       │ • Theme Extract │       │ • LLM + Context │       │
│   │ • Auto-Clean    │       │ • Silhouette    │       │ • Citations     │       │
│   └─────────────────┘       └─────────────────┘       └─────────────────┘       │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         ⚡ FASTAPI BACKEND                               │   │
│   │  POST /upload-data   POST /query   POST /cluster   GET /health/metrics  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         EVALUATION ENGINE                             │   │
│   │        Ground Truth Tests → Keyword Recall → Regression Detection       │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              🐳 Docker Container (Port 8000)
```

### How RAG Prevents Hallucination

```
User Question: "What do customers complain about?"
                    │
                    ▼
        ┌───────────────────────┐
        │ 1. EMBED THE QUESTION │  ──▶  [0.23, -0.15, 0.87, ...]
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 2. SEARCH FAISS INDEX │  ──▶  Find top 5 similar feedback
        └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ 3. BUILD CONTEXT      │  ──▶  "Feedback 1: App is slow..."
        └───────────────────────┘       "Feedback 2: Support is bad..."
                    │
                    ▼
        ┌───────────────────────┐
        │ 4. LLM + CONTEXT      │  ──▶  Answer ONLY from retrieved docs
        └───────────────────────┘
                    │
                    ▼
            Grounded Answer + Citations
```

---

##  Tech Stack

| Layer | Technology | Why This Choice |
|-------|------------|-----------------|
| **API** | FastAPI | Async, auto-docs, type validation |
| **Embeddings** | sentence-transformers | Fast, accurate semantic search |
| **Vector DB** | FAISS | Facebook-scale similarity search |
| **LLM** | OpenAI GPT-4o-mini | Cost-effective reasoning |
| **ML** | Scikit-learn | Production-tested clustering |
| **Data** | Pandas | Industry-standard data processing |
| **Deploy** | Docker | One-command deployment |

---

## ⚡ Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone
git clone https://github.com/Waqar53/customer_intelligence_ai.git
cd customer_intelligence_ai

# Build
docker build -t customer-ai .

# Run (add your OpenAI key)
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-xxx customer-ai

# Open http://localhost:8000/docs for interactive API
```

### Option 2: Local Development
```bash
# Clone
git clone https://github.com/Waqar53/customer_intelligence_ai.git
cd customer_intelligence_ai

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
export OPENAI_API_KEY=sk-xxx

# Run
uvicorn app.main:app --reload --port 8000
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload-data` | POST | Upload CSV/PDF/TXT feedback files |
| `/query` | POST | Ask questions about feedback |
| `/cluster` | POST | Run ML clustering analysis |
| `/health` | GET | Service health check |
| `/metrics` | GET | Performance & evaluation metrics |

📖 **Full interactive documentation:** http://localhost:8000/docs

---

##  Evaluation System

**Built-in quality assurance to prevent AI regression:**

```python
# Ground truth test cases
{
  "question": "What are customers unhappy about?",
  "expected_keywords": ["slow", "support", "bug", "problem"]
}

# Metrics tracked
- Keyword Recall: % of expected terms in answer
- Pass Rate: % of test cases passing threshold
- Token Usage: Cost monitoring
```

**Run evaluation:**
```bash
curl http://localhost:8000/metrics
```

---

## 📁 Project Structure

```
customer_intelligence_ai/
├── app/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Environment configuration
│   ├── api/
│   │   ├── routes.py        # API endpoints
│   │   └── models.py        # Request/response schemas
│   ├── core/
│   │   ├── data_ingestion.py    # CSV, PDF, TXT loaders
│   │   ├── data_cleaning.py     # Text preprocessing
│   │   ├── clustering.py        # TF-IDF + K-Means
│   │   ├── embeddings.py        # Sentence transformers
│   │   ├── vector_store.py      # FAISS operations
│   │   ├── llm_client.py        # OpenAI integration
│   │   └── rag_pipeline.py      # RAG orchestration
│   └── evaluation/
│       ├── evaluator.py         # Quality metrics
│       └── ground_truth.json    # Test cases
├── data/sample/             # Sample feedback data
├── tests/                   # Unit tests (13 passing)
├── Dockerfile               # Production container
├── docker-compose.yml       # Easy deployment
└── requirements.txt         # Dependencies
```

---

##  Testing

```bash
# Run all tests
pytest tests/ -v

# Results: 13 passed 
```

---

##  Skills Demonstrated

This project showcases production-ready AI engineering:

| Skill | Implementation |
|-------|----------------|
| **RAG Architecture** | Retrieval + LLM for grounded responses |
| **ML Engineering** | TF-IDF, K-Means clustering, embeddings |
| **Vector Databases** | FAISS indexing and similarity search |
| **API Development** | FastAPI with Pydantic validation |
| **LLM Integration** | OpenAI with prompt engineering |
| **Evaluation-Driven AI** | Ground truth testing, metrics |
| **Production Practices** | Docker, error handling, config mgmt |
| **Clean Architecture** | Modular, testable, documented code |

---

##  Author

**Waqar Azim**
- GitHub: [@Waqar53](https://github.com/Waqar53)

---

## 📄 License

MIT License - Use freely for learning or production.

---

<p align="center">
  <strong>⭐ If you find this useful, please star the repo!</strong>
</p>
