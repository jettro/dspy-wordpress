ingest_transformation_query = """
SELECT
    VECTOR_ENFORCE(embedding, 1536, 'float') as chunk_embedding,
    document_id,
    chunk_id,
    text,
    total_chunks,
    title,
    url,
    updated_at,
    tags,
    categories
FROM
    _input
WHERE
    title IS NOT NULL
"""
