# RAG-Multi-Modal-Scientific-Literature-Mining

## Table of Contents
 
1. [Prerequisites](#1-prerequisites)
2. [Project Directory Structure](#2-project-directory-structure)
3. [Environment Setup](#3-environment-setup)
4. [Configuration Layer](#4-configuration-layer)
5. [Data Ingestion Pipeline](#5-data-ingestion-pipeline)
6. [Multi-Modal Parser](#6-multi-modal-parser)
7. [Embedding Pool](#7-embedding-pool)
8. [Hybrid Index Setup](#8-hybrid-index-setup)
9. [Index Writer](#9-index-writer)
10. [Retrieval Engine](#10-retrieval-engine)
11. [LLM Generator](#11-llm-generator)
12. [Contradiction Detector](#12-contradiction-detector)
13. [Evaluation Pipeline](#13-evaluation-pipeline)
14. [API Server](#14-api-server)
15. [Celery Worker](#15-celery-worker)
16. [Docker Compose — Full Stack](#16-docker-compose--full-stack)
17. [Running the Project End-to-End](#17-running-the-project-end-to-end)
 
---
 
## 1. Prerequisites
 
### Software required
 
| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Core runtime |
| Docker + Docker Compose | Latest | Qdrant, Neo4j, Redis, Elasticsearch |
| CUDA (optional) | 11.8+ | GPU acceleration for embeddings |
| Java 11+ | JDK 11 | Required by GROBID server |
| Node.js 18+ | Optional | GROBID client tooling |
 
### API keys needed
 
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` — for LLM generation
- `NCBI_API_KEY` — for PubMed E-utilities (free, increases rate limit)
- No key needed for ArXiv
 
---
 
## 2. Project Directory Structure
 
```
scirag/
│
├── config/
│   ├── settings.py              # Central config via pydantic-settings
│   └── logging_config.py        # Structured JSON logging
│
├── ingestion/
│   ├── __init__.py
│   ├── pubmed_client.py         # Async PubMed E-utilities downloader
│   ├── arxiv_client.py          # Async ArXiv API downloader
│   ├── semantic_scholar.py      # Semantic Scholar citation fetcher
│   ├── pdf_downloader.py        # Async PDF download + MinIO upload
│   └── deduplicator.py          # DOI-based dedup using Redis SET
│
├── parsing/
│   ├── __init__.py
│   ├── grobid_client.py         # GROBID REST wrapper — section extraction
│   ├── nougat_client.py         # Nougat model — equation LaTeX extraction
│   ├── table_extractor.py       # pdfplumber + pandas table extraction
│   ├── figure_extractor.py      # PyMuPDF bounding box + CLIP captioning
│   ├── chunker.py               # Section-aware recursive text chunker
│   └── models.py                # ParsedDocument, Chunk, Figure dataclasses
│
├── embeddings/
│   ├── __init__.py
│   ├── base.py                  # Abstract BaseEmbedder interface
│   ├── text_embedder.py         # SciBERT / E5-large text encoder
│   ├── equation_embedder.py     # Math2Vec / LaTeX tokenizer
│   ├── table_embedder.py        # TAPAS table encoder
│   ├── figure_embedder.py       # BioMedCLIP image+text encoder
│   └── pool.py                  # EmbedderPool — routes by modality type
│
├── index/
│   ├── __init__.py
│   ├── qdrant_store.py          # Qdrant collection CRUD + upsert
│   ├── neo4j_store.py           # Citation graph node/edge management
│   ├── elasticsearch_store.py   # BM25 index writer and searcher
│   └── writer.py                # IndexWriter — orchestrates all three stores
│
├── retrieval/
│   ├── __init__.py
│   ├── hyde.py                  # HyDE — hypothetical document generation
│   ├── dense_retriever.py       # Qdrant semantic search
│   ├── graph_retriever.py       # Neo4j multi-hop citation traversal
│   ├── keyword_retriever.py     # Elasticsearch BM25 search
│   ├── reranker.py              # Cross-encoder re-ranking (ms-marco)
│   ├── self_rag.py              # Self-RAG relevance + faithfulness critic
│   └── engine.py                # RetrievalEngine — orchestrates all retrievers
│
├── generation/
│   ├── __init__.py
│   ├── prompts.py               # All system + user prompt templates
│   ├── generator.py             # LLM generation with evidence chain building
│   ├── contradiction.py         # Cross-paper contradiction detector
│   └── models.py                # GeneratedAnswer, EvidenceChain dataclasses
│
├── evaluation/
│   ├── __init__.py
│   ├── ragas_eval.py            # RAGAS faithfulness + context precision
│   ├── benchmark.py             # Benchmark runner on curated question set
│   └── questions.json           # 100 curated scientific benchmark questions
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── routes/
│   │   ├── ingest.py            # POST /ingest — trigger paper ingestion
│   │   ├── query.py             # POST /query — RAG query endpoint
│   │   └── eval.py              # POST /eval — run evaluation suite
│   └── schemas.py               # Pydantic request/response models
│
├── workers/
│   ├── __init__.py
│   ├── celery_app.py            # Celery + Redis broker config
│   └── tasks.py                 # Async tasks: ingest_paper, embed_chunk
│
├── tests/
│   ├── unit/
│   │   ├── test_chunker.py
│   │   ├── test_embedders.py
│   │   └── test_retrieval.py
│   └── integration/
│       ├── test_ingestion.py
│       └── test_full_pipeline.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_retrieval_evaluation.ipynb
│
├── docker/
│   ├── grobid/
│   │   └── Dockerfile
│   └── nougat/
│       └── Dockerfile
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── pyproject.toml
└── README.md
```
 
---
 
## 3. Environment Setup
 
### Step 3.1 — Clone and create virtual environment
 
```bash
mkdir scirag && cd scirag
git init
python3.11 -m venv .venv
source .venv/bin/activate
```
 
### Step 3.2 — requirements.txt
 
```txt
# Web framework
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.1
pydantic-settings==2.2.1
 
# Async HTTP
aiohttp==3.9.5
httpx==0.27.0
 
# PDF parsing
pymupdf==1.24.3
pdfplumber==0.11.0
grobid-client-python==0.8.4
 
# ML / Embeddings
torch==2.3.0
transformers==4.41.0
sentence-transformers==3.0.0
open-clip-torch==2.24.0
# Math2Vec — install from source (see Step 7.3)
 
# Vector store
qdrant-client==1.9.1
 
# Graph database
neo4j==5.20.0
 
# Keyword search
elasticsearch==8.13.0
 
# Task queue
celery==5.4.0
redis==5.0.4
 
# Object storage
minio==7.2.7
 
# LLM clients
openai==1.30.1
anthropic==0.28.0
 
# RAG framework (optional, for orchestration)
llama-index==0.10.43
llama-index-vector-stores-qdrant==0.2.8
 
# Evaluation
ragas==0.1.9
datasets==2.19.1
 
# Data processing
pandas==2.2.2
numpy==1.26.4
scipy==1.13.0
 
# Utilities
python-dotenv==1.0.1
loguru==0.7.2
tqdm==4.66.4
tenacity==8.3.0
orjson==3.10.3
```
 
```bash
pip install -r requirements.txt
```
 
### Step 3.3 — .env.example
 
```env
# LLM
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openai            # or anthropic
LLM_MODEL=gpt-4o
 
# PubMed
NCBI_API_KEY=your_ncbi_key
 
# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION=scirag_chunks
 
# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
 
# Elasticsearch
ES_HOST=http://localhost:9200
ES_INDEX=scirag_bm25
 
# Redis
REDIS_URL=redis://localhost:6379/0
 
# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=scirag-pdfs
 
# GROBID
GROBID_URL=http://localhost:8070
 
# Nougat
NOUGAT_URL=http://localhost:8071
 
# Embedding models
TEXT_EMBED_MODEL=allenai/scibert_scivocab_uncased
TABLE_EMBED_MODEL=google/tapas-base
CLIP_MODEL=microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
 
# Retrieval settings
TOP_K_DENSE=50
TOP_K_KEYWORD=20
TOP_K_GRAPH=15
RERANK_TOP_N=10
MAX_HOPS=3
```
 
Copy and fill in:
```bash
cp .env.example .env
```
