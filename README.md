# Smart Q&A Assistant with Hybrid Search

A Retrieval-Augmented Generation (RAG) project that answers technical questions in Artificial Intelligence and Data Science. It combines keyword search (BM25), semantic search (MiniLM embeddings) and an LLM that writes answers with numbered source citations.

Developed for the Advanced Deep Learning final project.

## Tech Stack

| Component | Technology / Method |
|---|---|
| Keyword search | BM25 (`rank_bm25`) |
| Semantic search | `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings, cosine similarity) |
| Hybrid ranking | Reciprocal Rank Fusion (k=60) and min-max normalized score fusion (50/50 weights) |
| Answer generation | `Qwen/Qwen2.5-3B-Instruct` (local, default) or OpenAI `gpt-4o-mini` (optional, needs `OPENAI_API_KEY`) |
| Interface | Gradio |
| Evaluation | MAP@10, MRR@10, nDCG@10 |

## Data

Two StackExchange JSON files, containing question posts only (no answer text):

- Artificial Intelligence: 750 questions (`top_ai_questions.json`)
- Data Science: 750 questions (`top_datascience_questions.json`)

After HTML cleaning and word-based chunking (180 words, 40 overlap), the corpus has **1,500 questions → 2,026 searchable passages**. Each passage keeps its title, tags, link and score metadata.

## How It Works

1. **Question:** the user asks a question in the Gradio app and picks a retrieval mode (`bm25`, `semantic`, `hybrid_rrf` or `hybrid_score`).
2. **Retrieval:** BM25 and MiniLM each rank the passages. The hybrid modes fuse the two rankings.
3. **Evidence enrichment:** the corpus has no answers, so for each retrieved question the system fetches its accepted and top-voted answers from the live StackExchange API. If the API is unavailable, it falls back to the retrieved question passage.
4. **Generation:** the evidence is labeled `[S1]`, `[S2]`, … and passed to the LLM, which writes a short answer with inline citations and a "Sources used" line.
5. **Citation check:** the system checks that every cited label (`[S1]` …) matches a retrieved source. Answers with no citation, or that look malformed, are replaced with a safe fallback message. This is a format check, not a check that the answer is factually supported.
6. **Display:** the app shows the answer and a table of the retrieved sources.

```
Question → BM25 + MiniLM retrieval → Fusion → StackExchange answer lookup
        → LLM generation → Cited answer + source table
```

## Retrieval Evaluation

Evaluated on 10 known-item queries. Each query is a paraphrase of a question title in the corpus, with exactly one relevant document. Scores are computed at document level (duplicate chunks removed) with custom metric code.

| Retriever | MAP@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| BM25 | 0.858 | 0.858 | 0.893 |
| Semantic (MiniLM) | **1.000** | **1.000** | **1.000** |
| Hybrid (RRF) | 0.900 | 0.900 | 0.926 |
| Hybrid (score fusion) | 0.950 | 0.950 | 0.963 |

Both hybrid methods beat BM25, but semantic search alone scored highest on this set. The queries are close paraphrases of titles and the embeddings include the title, so the set favors semantic matching. It is small and does not measure answer quality on unseen questions.

## Limitations

- Answer quality depends on the StackExchange API being reachable at query time.
- Citation checking validates labels only. It does not verify faithfulness to the sources.
- The OpenAI generation path is implemented but was not part of the evaluated runs.
- Evaluation covers retrieval only. Generated answers have not been systematically scored.
