# Wikiplot Search: Technical Overview

## Tech Stack
* **Backend:** FastAPI
* **Primary Database:** PostgreSQL
* **Vector Search:** OpenSearch
* **Embeddings:** all-MiniLM-L6-v2
* **Search Orchestration:** LangChain

## Architecture & Key Decisions

* **Separated Storage:** Stored the vector embeddings in OpenSearch and the full-text plots in PostgreSQL, linking them via a shared `book_id`. 
* **Hybrid Search (RRF):** Used LangChain's `EnsembleRetriever` to run a Vector search and a BM25 Keyword search giving each 50/50 weightage simultaneously.
  * *Why:* So that we can find documents that match the exact words the user typed, as well as documents that just match the conceptual vibe of the query.
* **Explicit Cosine Similarity:** Explicitly configured OpenSearch to use Cosine Similarity instead of its default L2 distance. The embedding model does still apply L2 normalization making the magnitude(length or size) of the user query and the chunk texts the same.
