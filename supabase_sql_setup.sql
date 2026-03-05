-- First, make sure you have the vector extension enabled
-- First, make sure you have the vector extension enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the documents table if it doesn't exist
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    metadata JSONB,
    embedding VECTOR(768)
);

-- Create the similarity search function that LangChain expects
CREATE OR REPLACE FUNCTION match_documents_langchain_2(
    query_embedding VECTOR(768),
    match_threshold FLOAT DEFAULT 0.6,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
$$;

-- Create an index on the embedding column for better performance
CREATE INDEX IF NOT EXISTS documents_embedding_idx 
ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);