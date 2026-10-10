"""Core RAG logic: chunk -> embed (Azure OpenAI) -> store/search (ChromaDB) -> answer."""
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
)
CHAT = os.environ["AZURE_CHAT_DEPLOYMENT"]
EMBED = os.environ["AZURE_EMBED_DEPLOYMENT"]
collection = chromadb.PersistentClient(path="chroma_db").get_or_create_collection(
    "docs", metadata={"hnsw:space": "cosine"}
)


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words), 1), step)]


def embed(texts: list[str]) -> list[list[float]]:
    out = []
    for i in range(0, len(texts), 64):  # batch to respect API limits
        res = client.embeddings.create(model=EMBED, input=texts[i:i + 64])
        out += [d.embedding for d in res.data]
    return out


def ingest(data_dir: str = "data") -> int:
    ids, docs, metas = [], [], []
    for f in sorted(Path(data_dir).glob("*")):
        if f.suffix.lower() not in {".pdf", ".txt", ".md"}:
            continue
        for n, c in enumerate(chunk(read_file(f))):
            ids.append(f"{f.name}-{n}")
            docs.append(c)
            metas.append({"source": f.name, "chunk": n})
    if docs:
        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
    return len(docs)


def retrieve(question: str, k: int = 4):
    res = collection.query(query_embeddings=embed([question]), n_results=k)
    return res["documents"][0], res["metadatas"][0]


def answer(question: str, k: int = 4):
    docs, metas = retrieve(question, k)
    context = "\n\n".join(f"[{m['source']}#{m['chunk']}] {d}" for d, m in zip(docs, metas))
    msg = [
        {"role": "system", "content": (
            "Answer ONLY from the context. Cite sources like [file#chunk]. "
            "If the answer is not in the context, say you don't know.")},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    res = client.chat.completions.create(model=CHAT, messages=msg, temperature=0)
    return res.choices[0].message.content, metas
