import os
import uuid
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from models import BookMetadata
from database import Base, engine, SessionLocal

load_dotenv()

OPENSEARCH_URL=os.getenv("OPENSEARCH_URL")

INDEX_NAME = "wikiplots_index"

Base.metadata.create_all(bind=engine)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

docsearch = OpenSearchVectorSearch(
    index_name=INDEX_NAME,
    embedding_function=embeddings,
    opensearch_url=OPENSEARCH_URL,
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    engine="lucene",
    # space_type="cosinesimil"
)

def ingest_data():
    db = SessionLocal()

    with open("data/titles", "r", encoding="utf-8") as t, open("data/plots", "r", encoding="utf-8") as p:
        titles = t.read().splitlines()
        plots = p.read().split("<EOS>")

    limit = 1000

    for i in range(limit, len(titles)):
        title_text=titles[i].strip()
        plot_text=plots[i].strip()

        if not plot_text:
            continue

        book_id = uuid.uuid4()
        new_book = BookMetadata(id=book_id, title=title_text, plot=plot_text)
        db.add(new_book)

        chunks = text_splitter.split_text(plot_text)

        metadatas=[{"book_id": str(book_id), "title": title_text} for _ in chunks]

        docsearch.add_texts(
            texts=chunks,
            metadatas=metadatas
        )

        if i % 10 == 0:
            db.commit()
            print(f"Progress: {i}/{limit} - Loaded: {title_text}")

    db.commit()
    db.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data()