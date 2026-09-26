import os

import ollama
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector

load_dotenv()

PG_DSN = dict(
    host=os.getenv("POSTGRES_HOST", "127.0.0.1"),
    port=int(os.getenv("POSTGRES_PORT", 5433)),
    dbname=os.getenv("POSTGRES_DB", "transcript_db"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "password"),
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")  # 384-dim -> matches VECTOR(384)

RRF_K = int(os.getenv("RRF_K", "60"))
TOP_N = int(os.getenv("TOP_N", "20"))
TOP_K = int(os.getenv("TOP_K", "5"))

_client = ollama.Client(host=OLLAMA_HOST)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    source_file TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    speakers    TEXT,
    start_time  DOUBLE PRECISION,
    end_time    DOUBLE PRECISION,
    text        TEXT NOT NULL,
    embedding   VECTOR(384),
    UNIQUE (source_file, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def get_connection():
    conn = psycopg2.connect(**PG_DSN)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(SCHEMA_SQL)
        conn.commit()

    register_vector(conn)
    return conn


def embed_text(text: str) -> list[float]:
    return _client.embeddings(model=EMBED_MODEL, prompt=text)["embedding"]