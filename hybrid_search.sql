-- Удаляем существующие функции с таким же именем в разных схемах
DROP FUNCTION IF EXISTS extensions.hybrid_search(text, extensions.vector, int, float, float, int);
DROP FUNCTION IF EXISTS public.hybrid_search(text, extensions.vector, int, float, float, int);
DROP FUNCTION IF EXISTS public.hybrid_search(text, vector, int, float, float, int);
DROP FUNCTION IF EXISTS public.hybrid_search(text, vector(512), int, float, float, int);

-- Создаём функцию в схеме public с правильным типом и оператором
CREATE OR REPLACE FUNCTION public.hybrid_search(
  query_text TEXT,
  query_embedding EXTENSIONS.VECTOR(1024),
  match_count INT,
  full_text_weight FLOAT DEFAULT 1,
  semantic_weight FLOAT DEFAULT 1,
  rrf_k INT DEFAULT 50
)
RETURNS SETOF documents
LANGUAGE SQL
AS $$
WITH full_text AS (
  SELECT
    id,
    -- Note: ts_rank_cd is not indexable but will only rank matches of the where clause
    -- which shouldn't be too big
    ROW_NUMBER() OVER(ORDER BY ts_rank_cd(fts, websearch_to_tsquery(query_text)) DESC) AS rank_ix
  FROM
    documents
  WHERE
    fts @@ websearch_to_tsquery(query_text)
  ORDER BY rank_ix
  LIMIT LEAST(match_count, 30) * 2
),
semantic AS (
  SELECT
    id,
    ROW_NUMBER() OVER (ORDER BY embedding <=> query_embedding) AS rank_ix
  FROM
    documents
  ORDER BY rank_ix
  LIMIT LEAST(match_count, 30) * 2
)
SELECT
  documents.*
FROM
  full_text
  FULL OUTER JOIN semantic
    ON full_text.id = semantic.id
  JOIN documents
    ON COALESCE(full_text.id, semantic.id) = documents.id
ORDER BY
  COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
  COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
  DESC
LIMIT
  LEAST(match_count, 30)
$$;
