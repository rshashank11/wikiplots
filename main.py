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

# docsearch is the pre-built vector store created in load_wikiplots.py. 
# It handles the "semantic" half of retrieval.
from load_wikiplots import text_splitter, INDEX_NAME

# Our SQLAlchemy ORM model for the books table — holds the full plot text
# that's too big to stash inside OpenSearch metadata.
from models import BookMetadata

# Pydantic's BaseModel gives us request/response schemas with free validation.
from pydantic import BaseModel

# HuggingFaceEmbeddings wraps a sentence-transformer model so LangChain can
# call it like any other embedding provider.
from langchain_huggingface import HuggingFaceEmbeddings

# EnsembleRetriever merges results from multiple retrievers using a weighted
# reciprocal-rank-fusion-style combination. It's how we do "hybrid search".
from langchain_classic.retrievers import EnsembleRetriever

# BaseRetriever is the abstract class every LangChain retriever must subclass.
from langchain_core.retrievers import BaseRetriever

# Document is LangChain's standard container: page_content (the text) plus
# a metadata dict. Every retriever must return a list of these.
from langchain_core.documents import Document

# LangChain's high-level wrapper around OpenSearch as a vector store.
from langchain_community.vectorstores import OpenSearchVectorSearch

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Type hints. `Any` is an escape hatch for the os_client field.
from typing import List, Any
import uuid
import requests

from schemas import SearchQuery, BookCreate, BookResponse, SearchPlan
from langchain_openai import ChatOpenAI

# Actually run the .env loader now, before anything reads os.environ.
load_dotenv()

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL")
JINA_API_KEY = os.environ.get("JINA_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Create the FastAPI app instance.
app = FastAPI(title="Wikiplot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=False,  
    allow_methods=["*"],      
    allow_headers=["*"],
)

# Instantiate the embedding model once at startup.
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Build the raw OpenSearch client we'll use for keyword search.
os_client = OpenSearch(
    hosts=[OPENSEARCH_URL],
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

# Build the vector store handle.
docsearch = OpenSearchVectorSearch(
    index_name=INDEX_NAME,
    embedding_function=embeddings,
    opensearch_url=OPENSEARCH_URL,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    engine="lucene",
)

# Custom retriever for OpenSearch keyword search via BaseRetriever interface.
class OpenSearchKeywordRetriever(BaseRetriever):
    os_client: Any
    index_name: str = INDEX_NAME
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        response = self.os_client.search(
            index=self.index_name,
            body={
                "query": {"match": {"text": query}},
                "size": self.k
            }
        )
        docs = []
        for hit in response["hits"]["hits"]:
            docs.append(Document(
                page_content=hit["_source"]["text"],
                metadata=hit["_source"]["metadata"]
            ))
        return docs

# Spin up instances for hybrid retrieval.
keyword_retriever = OpenSearchKeywordRetriever(os_client=os_client)
vector_retriever = docsearch.as_retriever()

hybrid_retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

# --- AI Model Initialization ---

intent_llm = ChatOpenAI(model='gpt-4o-mini', api_key=OPENAI_API_KEY, temperature=0)
final_llm = ChatOpenAI(model='gpt-4o', api_key=OPENAI_API_KEY, temperature=0.3)

# --- Core RAG Functions ---

def parse_user_query(user_input: str) -> SearchPlan:
    intent_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert Query Parser. "
            "1. If the query is broad (e.g., 'Harry Potter'), set detected_intent to 'ambiguous'. "
            "2. If specific, extract the subject and optimized keywords for search_query. "
            "3. If ambiguous, the generator will later ask the user for clarification."
        )),
        ("human", "{input}")
    ])
    chain = intent_prompt | intent_llm.with_structured_output(SearchPlan)
    return chain.invoke({"input": user_input})

def rerank_results(query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
    # rerank_results uses Jina AI's listwise reranker to rank candidate chunks.
    if not documents: return []
    
    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "jina-reranker-v3",
        "query": query,
        "documents": [doc.page_content for doc in documents],
        "top_n": top_n * 2 # Extra padding for deduplication
    }

    try:
        response = requests.post("https://api.jina.ai/v1/rerank", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        rankings = response.json().get("results", [])
        return [documents[item['index']] for item in rankings]
    except Exception as e:
        print(f"Rerank Error: {e}")
        return documents[:top_n]

def generate_final_answer(query: str, plan: SearchPlan, contexts: List[dict]) -> str:
    # Synthesizes the final answer using the full plots hydrated from Postgres.
    context_block = ""
    for i, item in enumerate(contexts):
        context_block += f"\n---\n[Source {i+1}]: {item['title']}\nPLOT: {item['full_plot_text']}\n"

    system_prompt = (
        "You are an expert media analyst. Answer accurately using ONLY provided plots.\n\n"
        "STRICT CITATION RULES:\n"
        "1. Every claim MUST cite the source using: [Source X].\n"
        "2. If multiple sources support a claim, combine them: [Source 1, 2].\n"
        "3. If the query is 'ambiguous', define the subject and ask clarifying questions."
    )
    
    response = final_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Intent: {plan.detected_intent}\nQuestion: {query}\n\nContext:\n{context_block}")
    ])
    return response.content

# --- API Endpoints ---

@app.get("/")
def read_root():
    return FileResponse("app.html")

@app.post("/search")
def search_books(request: SearchQuery, db: Session = Depends(get_session)):
    # measures total time for the search and generation process.
    start_time = time.time()
    
    # 1. Parsing: Turn natural language into structured search parameters.
    plan = parse_user_query(request.query)
    
    # 2. Dynamic Filtering: Setup the 'Hard Fence' metadata filter.
    os_filter = {"match_phrase": {"metadata.title": plan.subject}} if plan.is_specific_entity and plan.subject else None

    # 3. Retrieval: Pull candidate chunks from OpenSearch.
    keyword_retriever.k = 20
    
    # --- CRITICAL FIX ---
    # We only include the 'filter' key if it exists. 
    # OpenSearch kNN will crash with 'VALUE_NULL' if we pass filter: None.
    v_search_kwargs = {"k": 20}
    if os_filter:
        v_search_kwargs["filter"] = os_filter
    
    vector_retriever.search_kwargs = v_search_kwargs
    
    candidates = hybrid_retriever.invoke(plan.search_query)

    # 4. Rerank: Order the chunks by true semantic relevance.
    best_docs = rerank_results(plan.search_query, candidates, top_n=request.top_k)

    # 5. Deduplication & Hydration: Ensure unique results and fetch full text.
    seen_ids = set()
    unique_docs = []
    
    for doc in best_docs:
        bid = str(doc.metadata.get('book_id'))
        if bid not in seen_ids:
            seen_ids.add(bid)
            unique_docs.append(doc)
        if len(unique_docs) >= request.top_k:
            break

    book_ids = [doc.metadata.get('book_id') for doc in unique_docs if doc.metadata.get('book_id')]
    db_results = db.query(BookMetadata).filter(BookMetadata.id.in_(book_ids)).all()
    plot_map = {str(b.id): b.plot for b in db_results}

    final_payload = []
    for doc in unique_docs:
        bid = str(doc.metadata.get('book_id'))
        title = doc.metadata.get('title', 'Unknown Title')
        full_plot = plot_map.get(bid)
        
        if not full_plot:
            fallback = db.query(BookMetadata).filter(BookMetadata.title == title).first()
            full_plot = fallback.plot if fallback else doc.page_content

        final_payload.append({
            "title": title,
            "full_plot_text": full_plot.strip(),
            "snippet": doc.page_content
        })

    # 6. Generation: Synthesis step.
    answer = generate_final_answer(request.query, plan, final_payload)
    
    execution_time = round(time.time() - start_time, 2)

    return {
        "answer": answer,
        "user_query": request.query,
        "search_query": plan.search_query,
        "execution_time_seconds": execution_time,
        "results": final_payload
    }

# Standard CRUD omitted for space...