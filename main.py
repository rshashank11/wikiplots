from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import get_session
import os
import time
from dotenv import load_dotenv
from opensearchpy import OpenSearch
from load_wikiplots import docsearch
from models import BookMetadata

from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List, Any

load_dotenv()

app = FastAPI(title="Wikiplot API")

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

os_client = OpenSearch(
    hosts=[os.environ.get("OPENSEARCH_URL")],
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False
)

class OpenSearchKeywordRetriever(BaseRetriever):
    os_client: Any
    index_name: str = "wikiplots_index"
    k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        response = self.os_client.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "text": query 
                    }
                },
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

keyword_retriever = OpenSearchKeywordRetriever(os_client=os_client)
vector_retriever = docsearch.as_retriever()

hybrid_retriever = EnsembleRetriever(
    retrievers=[keyword_retriever, vector_retriever],
    weights=[0.5, 0.5]
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "sanity check"}

@app.get("/health")
def health_check(db: Session = Depends(get_session)):
    health_status = {
        "api": "online",
        "postgresql": "disconnected",
        "opensearch": "disconnected"
    }
    try:
        db.execute(text("SELECT 1"))
        health_status["postgresql"]= "connected"
    except Exception as e:
        print(f"Postgresql database error: {e}")
    
    try:
        if os_client.ping():
            health_status['opensearch'] = "connected"
    except Exception as e:
        print(f"Opensearch error: {e}")

    return health_status

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search")
def search_plots(request: SearchQuery, db: Session = Depends(get_session)):
    start_time = time.time()
    
    keyword_retriever.k = request.top_k
    vector_retriever.search_kwargs = {"k": request.top_k}

    results = hybrid_retriever.invoke(request.query)
    
    title_to_id_map = {}
    book_ids_to_fetch = []
    
    for doc in results:
        title = doc.metadata.get('title', 'Unknown Title')
        book_id = doc.metadata.get('book_id')
        
        if book_id:
            title_to_id_map[title] = book_id
            book_ids_to_fetch.append(book_id)
            
    full_plots_from_db = db.query(BookMetadata).filter(BookMetadata.id.in_(book_ids_to_fetch)).all()
    postgres_lookup = {str(book.id): book.plot for book in full_plots_from_db}

    final_payload = []
    for doc in results:
        title = doc.metadata.get('title', 'Unknown title')
        book_id = str(doc.metadata.get('book_id', 'N/A'))
        snippet = doc.page_content[:200] + "..."
        full_plot = postgres_lookup.get(book_id, "Full text not found in database.")
        
        final_payload.append({
            "title": title,
            "snippet": snippet,
            "full_plot_text": full_plot
        })
    
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)

    return {
        "user_query": request.query, 
        "execution_time_seconds": execution_time,
        "results": final_payload
    }