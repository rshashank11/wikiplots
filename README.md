### Wikiplot rag

---

### System architecture

<img width="821" height="378" alt="Screenshot 2026-04-27 at 9 54 36 am" src="https://github.com/user-attachments/assets/326620b0-33ab-4933-ad83-2c41a5bc01a6" />

---

### Tech stack
* **FastAPI**: Web framework and API orchestration.
* **OpenSearch**: Dual-engine storage for Vector and BM25 search.
* **PostgreSQL**: Relational storage for full plot hydration.
* **GPT-4o**: Final generation and synthesis.
* **GPT-4o-mini**: Intent parsing and query optimization.
* **Jina AI v3**: Cross-encoder reranking.
* **LangChain**: RAG logic and LLM orchestration.

---

**1. Intent Detection & Query Optimization**
Isolates the primary subject to enable metadata fencing and refines the search query for the engine. 

**Example:**
* **User Input:** "Who finally kills Immortan Joe in Fury Road?"
* **System Output (SearchPlan):**
    * **subject:** "Mad Max: Fury Road"
    * **search_query:** "Immortan Joe death Furiosa final battle"
    * **is_specific_entity:** True

**2. Hybrid Retrieval (Lexical + Semantic)**
Executes a simultaneous search across the OpenSearch index.
* **Lexical (BM25)**: Targets specific nouns, names, and character jargon.
* **Semantic (kNN)**: Captures thematic meaning and conceptual relationships.

**3. Reranking (Jina AI v3)**
Analyzes the top candidate chunks from the hybrid search. While the initial Hybrid Search (RRF) uses "Bi-Encoders" to quickly score documents separately, the Reranker is a Cross-Encoder.

The Reason: RRF is a statistical merge of two lists; it doesn't actually understand the relationship between the query and the result. The Reranker performs a word by word semantic analysis, processing the query and the document chunk simultaneously.

Listwise Reranking: Jina v3 is a listwise reranker, meaning it doesn't just score query document pairs in isolation. It looks at the whole set of candidates at once to decide which chunk is truly the best fit relative to the others.

The Benefit: It catches nuances that keyword and vector searches miss. By reading the top 20 results deeply, it can filter out "hallucinations by proximity" where a chunk contains the right keywords but the wrong meaning.

Example:
Input: A chunk mentions "Furiosa" and "Immortan Joe" but is actually describing a scene from the middle of the movie rather than the death scene.
Reranker Action: It recognizes that while the keywords match, the semantic intent of "final death" is missing, and drops the chunk's rank in favor of the actual climax text.

**4. Deduplication Logic**
Filters the reranked chunks to ensure unique media items.
* **Mechanism**: Uses the `book_id` field to remove duplicate references.
* **Purpose**: Prevents the UI and LLM from processing redundant data from the same movie or book.

**5. Full-Plot Hydration (PostgreSQL)**
Retrieves the complete, full raw plot text from the relational database for the top 5 unique results. This ensures the generator has full context beyond a simple text snippet.

**6. Generation (GPT-4o)**
The final synthesis step. The model is constrained to the hydrated context and must provide strict bracketed citations [Source X] for every fact presented.
