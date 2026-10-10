# Telecom-RAG

Question-answering over telecom/satellite documents with citations.
Python · Azure OpenAI (embeddings + chat) · ChromaDB · Docker

## Architecture

```mermaid
flowchart LR
    subgraph Ingestion["Ingestion (ingest.py)"]
        A[PDF / TXT / MD docs] --> B[Chunk<br/>200 words, 40 overlap]
        B --> C[Azure OpenAI<br/>Embeddings]
        C --> D[(ChromaDB<br/>vector store)]
    end
    subgraph Query["Query (ask.py)"]
        Q[User question] --> E[Azure OpenAI<br/>Embeddings]
        E --> F[Top-k similarity search]
        D --> F
        F --> G[Prompt: context + question]
        G --> H[Azure OpenAI<br/>GPT chat]
        H --> I[Answer + citations]
    end
```

## Query Flow

```mermaid
sequenceDiagram
    actor U as User
    participant App as ask.py / rag.py
    participant Emb as Azure Embeddings
    participant DB as ChromaDB
    participant LLM as Azure GPT
    U->>App: Question
    App->>Emb: Embed question
    Emb-->>App: Query vector
    App->>DB: Top-k nearest chunks
    DB-->>App: Chunks + sources
    App->>LLM: System rules + context + question
    LLM-->>App: Answer with [file#chunk] citations
    App-->>U: Answer + sources
```

## Setup
1. Create an Azure OpenAI resource; deploy a chat model (e.g. `gpt-4o-mini`) and `text-embedding-3-small`.
2. `cp .env.example .env` and fill in endpoint, key, deployment names.
3. Put public documents (e.g. ITU / 3GPP specs) in `data/`.
4. Run:
```bash
pip install -r requirements.txt
python ingest.py
python ask.py "What is the difference between C-band and Ku-band?"
```
Docker: `docker build -t telecom-rag . && docker run --env-file .env -v $(pwd)/data:/app/data -it telecom-rag python ask.py "your question"`

## Evaluation
Edit `eval.json` with 20 questions + expected keywords, then `python eval.py`.

| Metric | Result |
|---|---|
| Retrieval hit-rate | _fill in_ |
| Answer accuracy | _fill in_ |

## Design notes
- Answers constrained to retrieved context; says "don't know" otherwise (reduces hallucination).
- `upsert` makes re-ingestion idempotent.
- Temperature 0 for reproducible answers.

## Ideas
Hybrid search (BM25 + vector), re-ranking, Azure AI Search as the vector store, FastAPI endpoint.
