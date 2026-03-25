# Data Models

## documents
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | Auto-generated |
| filename | TEXT | Original filename or URL |
| source_type | VARCHAR(20) | pdf, docx, image, csv, url, text |
| content_hash | TEXT UNIQUE | SHA-256 for deduplication |
| metadata | JSONB | Extractor-specific metadata |
| pipeline_id | UUID | Optional pipeline association |
| status | VARCHAR(20) | queued, processing, complete, failed |
| chunk_count | INT | Set on completion |
| created_at | TIMESTAMPTZ | |

## chunks
| Column | Type | Description |
|--------|------|-------------|
| id | UUID PK | |
| document_id | UUID FK | Cascades on delete |
| content | TEXT | Raw chunk text |
| embedding | vector(1536) | text-embedding-3-small |
| chunk_index | INT | Order within document |
| token_count | INT | tiktoken count |
| metadata | JSONB | page, section, strategy, etc. |

Indexes: HNSW on embedding, GIN on to_tsvector(content), GIN on metadata

## pipeline_runs
Stores every query execution. Retrieved_chunk_ids is a UUID array linking to chunks. Evaluation scores are in the evaluations table (1:1 with runs).

## evaluations
Four RAGAS metrics + overall_score. user_rating (1-5) is optional human feedback. calibration_needed is flagged when automated and human scores diverge > 0.3.

## training_pairs
High-quality (score > threshold) run outputs formatted for fine-tuning. included_in_job tracks which finetune_job consumed each pair.
