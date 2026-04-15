# Wikiplot Search: Technical Overview

## Tech Stack
* **Backend:** FastAPI
* **Primary Database:** PostgreSQL
* **Vector Search:** OpenSearch
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Search Orchestration:** LangChain

## Architecture & Key Decisions

* **Separated Storage:** We store the vector embeddings in OpenSearch and the full-text plots in PostgreSQL, linking them via a shared `book_id`. 
* **Hybrid Search (RRF):** We use LangChain's `EnsembleRetriever` to run a Vector search and a BM25 Keyword search simultaneously.
  * *Why:* It guarantees we find documents that match the exact words the user typed, as well as documents that just match the conceptual vibe of the query.
* **Explicit Cosine Similarity:** We explicitly configured OpenSearch to use Cosine Similarity instead of its default L2 distance.
  * *Why:* This perfectly aligns the database's math with HuggingFace's native math, guaranteeing maximum accuracy.
* **LLM Removal (Low Latency):** We stripped out the generative AI step that was originally summarizing the plots.
  * *Why:* Generating text was causing a 15-second bottleneck. Fetching the raw PostgreSQL text directly dropped our latency to 0.5 seconds, achieving true chat-based search speed.
