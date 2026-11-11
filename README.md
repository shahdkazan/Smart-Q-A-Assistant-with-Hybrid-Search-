# Smart Q&A Assistant with Hybrid Search

A Retrieval-Augmented Generation (RAG) assistant that answers technical questions in Artificial Intelligence and Data Science. It combines BM25 keyword search, MiniLM semantic search, rank/score fusion, and an LLM that writes answers with inline source citations.

Built for the Advanced Deep Learning final project.

## Tech Stack

| Component | Technology / Method |
|---|---|
| Keyword search | BM25 (`rank_bm25`, BM25Okapi) with a custom tokenizer that keeps technical tokens (e.g. `fit_transform`, `c++`) |
| Semantic search | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized, cosine similarity) |
| Fusion | Reciprocal Rank Fusion (k=60) and min-max normalized weighted score fusion (0.5 / 0.5) |
| Answer generation | `Qwen/Qwen2.5-3B-Instruct` (local, default) or `gpt-4o-mini` (if `OPENAI_API_KEY` is set) |
| Answer evidence | Top answers fetched from the public StackExchange API for each retrieved question |
| Interface | Gradio |
| Evaluation | MAP@10, MRR@10, nDCG@10 |

## Data

Two StackExchange-style JSON files:

- `top_ai_questions.json`: 750 questions (AI StackExchange)
- `top_datascience_questions.json`: 750 questions (Data Science StackExchange)

Each record has title, body (HTML), tags, link, score, view count, answer count, and question ID.

**Preprocessing:** HTML is stripped (code blocks are kept as readable text), then each document's title, tags and body are combined. Documents are split into 180-word passages with a 40-word overlap, giving **2,026 searchable passages** from 1,500 questions.

> **Note:** the corpus contains question posts only, not answer bodies. To give the generator real explanatory content, the RAG step fetches the top two answers (accepted answers first, then highest-voted) for each retrieved question from the StackExchange API. If the API is unreachable, it falls back to the retrieved question text. This step requires internet access.

## How It Works

1. **Question** is entered in the Gradio interface.
2. **Retrieval:** BM25 and MiniLM each score all passages independently.
3. **Fusion:** RRF or score fusion merges the two rankings (`hybrid_rrf` is the default).
4. **Context building:** the top-k passages (default 5) are labelled `[S1]`, `[S2]`, ... and enriched with StackExchange answer text.
5. **Generation:** the LLM answers using only the labelled sources, with inline citations and a final "Sources used:" line.
6. **Output checks:** the answer is cleaned (URLs and metadata removed) and validated. A safe fallback message is shown if the output is unusable. A citation check confirms every `[S#]` label maps to a retrieved source, and a source table lets users inspect the evidence.

The interface offers four retrieval modes (`bm25`, `semantic`, `hybrid_rrf`, `hybrid_score`), a top-k slider (1-10), and a generator choice (`auto`, `local`, `openai`).

## Retrieval Evaluation

Evaluated at document level (chunks from the same question are deduplicated) on 10 queries, each with one labelled relevant question:

| Retriever | MAP@10 | MRR@10 | nDCG@10 |
|---|---|---|---|
| BM25 | 0.858 | 0.858 | 0.893 |
| Semantic (MiniLM) | **1.000** | **1.000** | **1.000** |
| Hybrid RRF | 0.900 | 0.900 | 0.926 |
| Hybrid score fusion | 0.950 | 0.950 | 0.963 |

**Takeaways**
- Both hybrid methods beat BM25 alone, and score fusion beat RRF.
- Semantic search alone was best on this set. The queries are close paraphrases of corpus titles, so this is effectively a known-item lookup test and favors dense retrieval. It is a small sanity benchmark, not a general claim. Hybrid search is expected to help more on keyword-heavy or exact-term queries.

## Limitations

- The evaluation set is small (10 queries) and title-derived.
- Citation checking validates that labels are well-formed and map to retrieved sources. It does **not** verify that each claim is factually supported.
- Answer quality depends on the StackExchange API being reachable.
- Local generation with a 3B model is best run on a GPU.

## Run It

1. Open the notebook in Google Colab (GPU recommended).
2. Upload `top_ai_questions.json` and `top_datascience_questions.json`.
3. Run all cells. Packages install automatically.
4. Open the `gradio.live` link printed at the end.

**Optional, to use OpenAI instead of the local model:**
```python
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```
Then set Generator to `auto` or `openai`.
