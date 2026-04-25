# SchemaAgent Architecture

## System Overview

SchemaAgent is a hybrid Text-to-SQL system. A user question is routed to one of three paths, relevant context is retrieved, and the final answer is synthesised in natural language.

```text
User Question
    |
    v
IntentRouter (Claude Haiku)
    |-- database -> SQLAgent
    |-- document -> DocRAG
    `-- hybrid   -> SQLAgent + DocRAG
                         |
                         v
                 Claude synthesis layer
                         |
                         v
              Natural-language final answer
```

---

## Thesis Contribution

The thesis studies three interacting mechanisms:

1. **Schema RAG** retrieves only the most relevant tables for the current question.
2. **Self-correction** retries failed SQL generations with explicit error feedback.
3. **BirdDescRAG** retrieves data-dictionary context from BIRD's `database_description/*.csv` files when schema alone is not enough.

The research claim is that enterprise Text-to-SQL needs more than schema serialization. It needs both structural grounding and semantic grounding.

---

## Core Components

### IntentRouter (`src/router/__init__.py`)

- Model: Claude Haiku
- Role: classify the question as `database`, `document`, or `hybrid`
- Fallback: returns `hybrid` on parsing or API failure

### SQLAgent (`src/agent.py`)

`SQLAgent` is the main database reasoning component.

```text
1. Retrieve schema context with SchemaRAG
2. Generate SQL with the configured LLM
3. Validate syntax with sqlglot
4. Execute on SQLite
5. If syntax, execution, or empty-result failure occurs, retry with correction feedback
6. If RAG omitted needed tables and execution fails, escalate to the full schema
```

This schema-escalation step is the mechanism behind the super-additive behavior of RAG plus self-correction.

### SchemaRAG (`src/schema/`)

SchemaRAG is a three-part retrieval pipeline:

| Component | File | Role |
|-----------|------|------|
| `SchemaSerializer` | `serializer.py` | Converts SQLite schema into retrievable text chunks |
| `SchemaIndexer` | `indexer.py` | Embeds and stores schema chunks in ChromaDB |
| `SchemaRetriever` | `retriever.py` | Retrieves top-k schema chunks for a question |

`SQLAgent` uses `schema_rag.get_schema_context(question, db_id)` to fetch prompt context.

### DocRAG (`src/doc_rag.py`)

- Indexes PDF, DOCX, and TXT files
- Supports HyDE retrieval for document-heavy questions
- Returns passage-level evidence for document or hybrid queries

### BirdDescRAG (`src/bird_desc_rag.py`)

BirdDescRAG is the Config E addition.

- Indexes BIRD `database_description/*.csv` files per database
- Treats each table description CSV as a retrievable data-dictionary chunk
- Returns formatted column descriptions as an evidence string for SQL generation

This turns BIRD's structured metadata into a realistic proxy for enterprise data dictionaries.

### HybridFusion (`src/fusion.py`)

HybridFusion orchestrates routing, retrieval, and answer synthesis.

| Intent | SQL path | Document path | Synthesis model |
|--------|----------|---------------|-----------------|
| `database` | yes | no | Claude Haiku |
| `document` | no | yes | Claude Haiku |
| `hybrid` | yes | yes | Claude Sonnet |

The design uses two clients:

- `client` for SQL generation
- `synth_client` for final natural-language synthesis

This keeps SQL generation pluggable while holding synthesis quality constant.

### UnifiedClient (`src/llm_client.py`)

Unified adapter for Anthropic and OpenAI-compatible providers. This allows the same SQLAgent interface to drive Claude or Qwen backends.

---

## Evaluation Design

The ablation framework now has five configurations:

| Config | Schema RAG | Self-correction | Domain knowledge source |
|--------|------------|-----------------|-------------------------|
| A | no | no | none |
| B | yes | no | none |
| C | no | yes | none |
| D | yes | yes | gold BIRD evidence hint |
| E-noDoc | yes | yes | none |
| E-withDoc | yes | yes | BirdDescRAG over description CSVs |

Config E is the thesis bridge to real deployments:

- `E-noDoc` measures the cost of removing the perfect hint.
- `E-withDoc` measures how much of that loss can be recovered from a data dictionary.
- `D - E-withDoc` measures the residual gap between ideal gold knowledge and retrieved documentation.

The benchmark metric is **execution accuracy (EX)**: the predicted SQL and gold SQL must return the same result set on the same database.

---

## Data Flow Example

Question: `Which year recorded the most consumption of gas paid in CZK?`

```text
1. IntentRouter -> database
2. SchemaRAG retrieves likely tables
3. BirdDescRAG optionally retrieves value mappings and column descriptions
4. SQLAgent generates SQL
5. SQLAgent validates and executes the query
6. If needed, SQLAgent retries with correction feedback or full-schema escalation
7. Final SQL result is returned or synthesised for the UI
```

This is the full thesis story in one path: structural retrieval, semantic retrieval, and agentic repair.
