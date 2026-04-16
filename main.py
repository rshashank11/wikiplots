# --- Web framework + DB session handling ---
# FastAPI: the web framework that turns Python functions into HTTP endpoints.
# Depends: FastAPI's dependency-injection helper. We use it to hand a fresh DB
# session to each request handler without having to open/close it manually.
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
# Session is the SQLAlchemy ORM session type — think of it as a "conversation"
# with the database. We only import it for type hinting here.
from sqlalchemy.orm import Session

# `text` lets us write raw SQL strings safely. SQLAlchemy refuses to execute
# a plain Python string for safety, so we wrap it in text() first.
from sqlalchemy import text

# get_session is our own factory (defined in database.py) that yields a
# scoped DB session per request. Paired with Depends() below.
from database import get_session

# Standard library: os for env vars, time for measuring request latency.
import os
import time

# Reads a .env file from disk and loads its KEY=VALUE pairs into os.environ.
# Lets us keep secrets (DB URLs, API keys) out of source code.
from dotenv import load_dotenv

# Official OpenSearch Python client — we use it for the "keyword" half of
# our hybrid retrieval (BM25-style lexical search).
from opensearchpy import OpenSearch

# docsearch is the pre-built vector store
# created in load_wikiplots.py. It handles the "semantic" half of retrieval.
from load_wikiplots import docsearch

# Our SQLAlchemy ORM model for the books table — holds the full plot text
# that's too big to stash inside OpenSearch metadata.
from models import BookMetadata

# Pydantic's BaseModel gives us request/response schemas with free validation.
# FastAPI uses these to parse JSON bodies and generate OpenAPI docs.
from pydantic import BaseModel

# HuggingFaceEmbeddings wraps a sentence-transformer model so LangChain can
# call it like any other embedding provider. We import it here because it's
# handy to have the embedding object around even if docsearch already uses it.
from langchain_huggingface import HuggingFaceEmbeddings

# EnsembleRetriever merges results from multiple retrievers using a weighted
# reciprocal-rank-fusion-style combination. It's how we do "hybrid search".
from langchain_classic.retrievers import EnsembleRetriever

# BaseRetriever is the abstract class every LangChain retriever must subclass.
# We inherit from it to make our custom OpenSearch retriever play nicely
# inside the EnsembleRetriever.
from langchain_core.retrievers import BaseRetriever

# Document is LangChain's standard container: page_content (the text) plus
# a metadata dict. Every retriever must return a list of these.
from langchain_core.documents import Document

# Type hints. `Any` is an escape hatch for the os_client field because the
# OpenSearch client class isn't a Pydantic-friendly type.
from typing import List, Any

import uuid

from schemas import SearchQuery, BookCreate, BookResponse

# Actually run the .env loader now, before anything reads os.environ.
load_dotenv()

# Create the FastAPI app instance. The title shows up in the /docs Swagger UI.
app = FastAPI(title="Wikiplot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=False,  
    allow_methods=["*"],      
    allow_headers=["*"],
)

# Instantiate the embedding model once at startup. `all-MiniLM-L6-v2` is a
# small (~22M param) sentence-transformer — fast, 384-dim vectors, good
# quality-per-dollar for semantic search on short passages.
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Build the raw OpenSearch client we'll use for keyword search.
os_client = OpenSearch(
    # `hosts` takes a list so the client can fail over between nodes — we
    # only have one URL, pulled from the environment so it's not hardcoded.
    hosts=[os.environ.get("OPENSEARCH_URL")],
    # use_ssl=True: talk HTTPS, not plain HTTP.
    use_ssl=True,
    # verify_certs=False: skip cert-chain validation. Fine for a local dev
    # cluster with a self-signed cert; you'd flip this to True in prod.
    verify_certs=False,
    # Don't enforce that the cert's hostname matches — again, self-signed dev.
    ssl_assert_hostname=False,
    # Mute the "you're doing insecure stuff" warning that would otherwise
    # spam the logs every request.
    ssl_show_warn=False
)

# Custom retriever that lets LangChain query OpenSearch via the BaseRetriever
# interface. We need this because LangChain doesn't ship a first-class
# OpenSearch keyword retriever that plugs straight into EnsembleRetriever.
class OpenSearchKeywordRetriever(BaseRetriever):
    # These are Pydantic fields (BaseRetriever is a Pydantic model under the
    # hood). Declaring them as class attributes with type annotations is how
    # you give a Pydantic model its schema.
    os_client: Any                        # the OpenSearch client instance
    index_name: str = "wikiplots_index"   # which index to hit; default provided
    k: int = 5                            # how many hits to return per query

    # LangChain calls this method internally when you do `.invoke(query)`.
    # The leading underscore marks it as the "protected" hook to override.
    # `*,` forces run_manager to be keyword-only — LangChain passes it by name.
    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        # Fire a search request against OpenSearch. `body` is the raw
        # OpenSearch Query DSL — same JSON you'd send with curl.
        response = self.os_client.search(
            index=self.index_name,
            body={
                "query": {
                    # `match` runs the query through the analyzer (tokenize,
                    # lowercase, etc.) and scores docs via BM25. This is the
                    # lexical/keyword half of hybrid search.
                    "match": {
                        "text": query   # "text" = the field we indexed the plot into
                    }
                },
                "size": self.k   # cap the number of hits we pull back
            }
        )

        # Reshape the raw OpenSearch response into LangChain Documents so the
        # rest of the pipeline doesn't have to know about OpenSearch's format.
        docs = []
        # response["hits"]["hits"] is the actual list of matched documents;
        # the outer "hits" also holds totals/metadata we don't need here.
        for hit in response["hits"]["hits"]:
            docs.append(Document(
                # _source is the original document we indexed. "text" is the
                # plot body; "metadata" is the dict of title/book_id/etc.
                page_content=hit["_source"]["text"],
                metadata=hit["_source"]["metadata"]
            ))
        return docs

# Spin up one instance of our keyword retriever, wired to the OpenSearch client.
keyword_retriever = OpenSearchKeywordRetriever(os_client=os_client)

# docsearch is already a vector store; `.as_retriever()` wraps it in the
# BaseRetriever interface so EnsembleRetriever can consume it.
vector_retriever = docsearch.as_retriever()

# The hybrid retriever: combine lexical + semantic. Weights 0.5/0.5 means we
# trust both equally — tune this once we see real query performance.
hybrid_retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# Serve the frontend at the root URL.
@app.get("/")
def read_root():
    return FileResponse("app.html")

# /health pings our two downstream dependencies so ops/monitoring can tell
# whether the API itself is alive but something behind it is broken.
@app.get("/health")
def health_check(db: Session = Depends(get_session)):
    # Assume everything is down until proven otherwise.
    health_status = {
        "api": "online",
        "postgresql": "disconnected",
        "opensearch": "disconnected"
    }
    # Try Postgres: `SELECT 1` is the cheapest possible query that still
    # forces a real round-trip through the connection.
    try:
        db.execute(text("SELECT 1"))
        health_status["postgresql"] = "connected"
    except Exception as e:
        # We swallow the exception so one broken dep doesn't 500 the health
        # endpoint — but we log it so we can actually debug it.
        print(f"Postgresql database error: {e}")

    # Same idea for OpenSearch. `.ping()` is the client's built-in cheap check.
    try:
        if os_client.ping():
            health_status['opensearch'] = "connected"
    except Exception as e:
        print(f"Opensearch error: {e}")

    return health_status

# The main retrieval endpoint. POST because the body can be arbitrarily long
# and we don't want it in the URL.
@app.post("/search")
def search_books(request: SearchQuery, db: Session = Depends(get_session)):
    # Wall-clock start so we can report total latency back to the caller.
    start_time = time.time()

    # Push the requested top_k down into both retrievers. They each need it
    # set separately because EnsembleRetriever doesn't forward it for us.
    keyword_retriever.k = request.top_k
    # `search_kwargs` is the standard LangChain knob for vector retrievers —
    # it's forwarded to the underlying similarity_search call.
    vector_retriever.search_kwargs = {"k": request.top_k}

    # Run hybrid retrieval. `.invoke` is LangChain's unified call interface;
    # it fans out to both retrievers and fuses the results.
    results = hybrid_retriever.invoke(request.query)

    # We'll collect the book IDs from the retrieved chunks so we can do ONE
    # bulk Postgres query at the end instead of N sequential ones (N+1 trap).
    title_to_id_map = {}
    book_ids_to_fetch = []

    for doc in results:
        # `.get(key, default)` avoids a KeyError if the field is missing.
        title = doc.metadata.get('title', 'Unknown Title')
        book_id = doc.metadata.get('book_id')

        # Only remember ids that actually exist — skip malformed docs.
        if book_id:
            title_to_id_map[title] = book_id
            book_ids_to_fetch.append(book_id)

    # Single SQL query: SELECT * FROM books WHERE id IN (...). `.all()` runs
    # the query and materializes the rows into BookMetadata objects.
    # SELECT * FROM book_metadata WHERE id IN ('uuid1', 'uuid2', 'uuid3', 'uuid4', 'uuid5');
    db_results = db.query(BookMetadata).filter(BookMetadata.id.in_(book_ids_to_fetch)).all()
    # Dict comprehension: id -> plot, so we can O(1) look up the full plot
    # text while assembling the response below. str(book.id) because the
    # book_ids from the retriever are strings and we want the keys to match.
    books = {str(book.id): book.plot for book in db_results}

    # Build the final response list in the same order the retriever returned.
    final_payload = []
    for doc in results:
        # Extract the ID once and ensure it's a string
        raw_id = doc.metadata.get('book_id')
        book_id_str = str(raw_id) if raw_id else "N/A"
        title = doc.metadata.get('title', 'Unknown Title')

        # Pull the text from our Postgres map using the string ID.
        # If the UUID is stale (OpenSearch and Postgres out of sync),
        # fall back to a title-based lookup.
        full_plot = books.get(book_id_str)
        if not full_plot:
            fallback = db.query(BookMetadata).filter(BookMetadata.title == title).first()
            if fallback:
                full_plot = fallback.plot
                book_id_str = str(fallback.id)
            else:
                full_plot = doc.page_content

        final_payload.append({
            "id": book_id_str,
            "title": title,
            "snippet": f"{doc.page_content[:200]}...",
            "full_plot_text": full_plot.strip()
        })

    # Stop the timer and round to 2 decimal places — don't need microseconds.
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    # Echo the query back so the client can correlate responses, include the
    # timing for debugging, and of course the actual results.
    return {
        "user_query": request.query,
        "execution_time_seconds": execution_time,
        "results": final_payload
    }

@app.get("/books/search", response_model=List[BookResponse])
def search_books_by_title(title: str, db: Session = Depends(get_session)):
    search_pattern = f"%{title}%"
    books = db.query(BookMetadata).filter(BookMetadata.title.ilike(search_pattern)).limit(10).all()
    return books

@app.post("/books", response_model=BookResponse)
def create_book(book: BookCreate, db: Session = Depends(get_session)):
    new_book = BookMetadata(title=book.title, plot=book.plot)
    db.add(new_book)
    db.commit()
    db.refresh(new_book)
    return new_book

@app.get("/books/{book_id}", response_model=BookResponse)
def get_book(book_id: str, db: Session = Depends(get_session)):
    book = db.query(BookMetadata).filter(BookMetadata.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="book not found")
    return book

@app.put("/books/{book_id}", response_model=BookResponse)
def update_book(book_id: str, book_data: BookCreate, db: Session = Depends(get_session)):
    book = db.query(BookMetadata).filter(BookMetadata.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="book not found")
    
    book.title = book_data.title
    book.plot = book_data.plot
    db.commit()
    db.refresh(book)
    return book

@app.delete("/books/{book_id}")
def delete_book(book_id: str, db: Session = Depends(get_session)):
    book = db.query(BookMetadata).filter(BookMetadata.id == book_id).first()
    if not book:
        raise HTTPException(status_code=404, detail="book not found")
    
    db.delete(book)
    db.commit()
    return {"status": "success"}