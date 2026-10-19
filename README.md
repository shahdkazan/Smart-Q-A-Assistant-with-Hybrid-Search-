# Smart Q&A Assistant with Hybrid Search

A **Retrieval-Augmented Generation (RAG)** project that answers technical questions in Artificial Intelligence and Data Science by combining traditional information retrieval, semantic search, and language models.

Developed for the **Advanced Deep Learning / Generative AI final project**.

## What Makes It Work?

The assistant uses a hybrid retrieval pipeline to find relevant information and generate answers grounded in retrieved sources.

| Component | Technology / Method |
|---|---|
| Keyword search | BM25 |
| Semantic search | MiniLM sentence embeddings |
| Result ranking | Reciprocal Rank Fusion (RRF) and score-based fusion |
| Answer generation | OpenAI API or local Hugging Face instruction model |
| User interface | Gradio |
| Evaluation | MAP@10, MRR@10, nDCG@10 |

## Data Behind the Assistant

The system indexes two StackExchange-style JSON datasets:

- **Artificial Intelligence:** 750 questions (`top_ai_questions.json`)
- **Data Science:** 750 questions (`top_datascience_questions.json`)

**Dataset statistics**
- 1,500 original question records
- 2,026 searchable passages after preprocessing

Each record includes question content and metadata such as title, body, tags, source link, score, view count, answer count, and question ID.

## From Question to Answer

1. **Input:** The user submits a technical question through the Gradio interface.
2. **Retrieval:** BM25 and MiniLM independently identify relevant passages.
3. **Fusion:** RRF or score-based fusion combines the retrieval results.
4. **Context building:** The highest-ranked passages are assembled into context for the language model.
5. **Generation:** The model produces an answer grounded in the retrieved information.
6. **Verification:** Inline source references (`[S1]`, `[S2]`, etc.) and a source table help users inspect the supporting documents.

## Core Objectives

- Improve technical information retrieval by combining lexical and semantic search.
- Generate context-aware answers using RAG.
- Improve transparency through source citations.
- Compare retrieval effectiveness using standard ranking metrics.
- Provide an interactive environment for testing and demonstration.

## System Workflow

```text
Question
   → Dual Retrieval
   → Hybrid Ranking
   → Context Assembly
   → LLM Generation
   → Answer with Sources
```

**Key takeaway:** The project integrates information retrieval, hybrid ranking, and LLM-based generation into a single technical question-answering system.
