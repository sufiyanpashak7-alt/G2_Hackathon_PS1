CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    source_file   TEXT NOT NULL,
    chunk_index   INT NOT NULL,
    speakers      TEXT,
    start_time    DOUBLE PRECISION,
    end_time      DOUBLE PRECISION,
    text          TEXT NOT NULL,
    embedding     VECTOR(384),
    UNIQUE (source_file, chunk_index)
);



CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);