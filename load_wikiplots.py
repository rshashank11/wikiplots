# --- Standard library ---
# os: used to pull environment variables (OPENSEARCH_URL) out of the shell/.env.
import os
# uuid: generates unique IDs for each book row. Using UUIDs instead of
# auto-increment ints means we can generate the id on the Python side BEFORE
# inserting, which lets us attach the same id to the vector chunks right away.
import uuid

# Loads .env into os.environ so os.getenv() below actually finds the URL.
from dotenv import load_dotenv

# RecursiveCharacterTextSplitter breaks long text into smaller chunks by
# trying progressively smaller separators (paragraph -> sentence -> word).
# We need this because embedding models have a token limit AND because
# smaller chunks give more precise retrieval hits.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LangChain's high-level wrapper around OpenSearch as a vector store.
# It handles: embedding the text, creating the index, storing vectors +
# metadata, and exposing a .similarity_search() interface.
from langchain_community.vectorstores import OpenSearchVectorSearch

# Same sentence-transformer wrapper we use in main.py — critical that both
# files use the SAME model, otherwise the query vectors won't be comparable
# to the indexed vectors.
from langchain_huggingface import HuggingFaceEmbeddings

# Our SQLAlchemy ORM model — one row per book (id, title, full plot text).
from models import BookMetadata
# Base: SQLAlchemy declarative base, needed to create tables.
# engine: the DB connection pool.
# SessionLocal: factory that produces a new Session when called.
from database import Base, engine, SessionLocal

# Actually do the .env load now, before we read OPENSEARCH_URL below.
load_dotenv()

# Pull the OpenSearch URL out of the env. `getenv` returns None if missing,
# which is fine here because OpenSearchVectorSearch will blow up loudly
# downstream if the URL is bad — no silent mis-configuration.
OPENSEARCH_URL=os.getenv("OPENSEARCH_URL")

# Name of the OpenSearch index we'll write vectors into. Kept as a constant
# so main.py and this file can't drift — though right now main.py hardcodes
# the same string separately (TODO: share it via a constants module).
INDEX_NAME = "wikiplots_index"

# Tell SQLAlchemy to CREATE TABLE for any model that inherits from Base,
# if the table doesn't already exist. Safe to run repeatedly — it's a no-op
# when the schema is already in place. This gives us a one-shot bootstrap.
Base.metadata.create_all(bind=engine)

# Instantiate the embedding model once. Fully-qualified name
# "sentence-transformers/all-MiniLM-L6-v2" is the HuggingFace hub path;
# main.py uses the short form 'all-MiniLM-L6-v2' but they resolve to the same
# model, so query-time and index-time embeddings stay aligned.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Chunking config:
#   chunk_size=1000    -> ~1000 characters per chunk (not tokens!)
#   chunk_overlap=100  -> each chunk shares 100 chars with its neighbor so
#                         we don't lose context at chunk boundaries.
# Good defaults for prose; tune later if retrieval quality is off.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Build the vector store handle. This doesn't upload anything yet — it just
# gives us an object we can call .add_texts() on later.
docsearch = OpenSearchVectorSearch(
    index_name=INDEX_NAME,             # index to write into / read from
    embedding_function=embeddings,     # how to turn text -> 384-dim vector
    opensearch_url=OPENSEARCH_URL,     # where the cluster lives
    # SSL flags mirror main.py — dev cluster with a self-signed cert, so we
    # turn on HTTPS but skip all the verification steps.
    use_ssl=True,
    verify_certs=False,
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    # engine="lucene" picks OpenSearch's native Lucene kNN backend. The other
    # option is "nmslib" or "faiss"; lucene is the modern default and works
    # out of the box on AWS-managed OpenSearch.
    engine="lucene",
    # space_type="cosinesimil"   # left commented — defaults to l2 for now;
                                 # flip this on if cosine similarity ends up
                                 # scoring better for our prose embeddings.
)

# Main ingestion routine — reads the raw dataset, writes to Postgres + OpenSearch.
def ingest_data():
    # Open a DB session manually (we're not inside a FastAPI request, so no
    # Depends injection). We're responsible for closing it at the end.
    db = SessionLocal()

    # Open both files in ONE `with` statement — comma-separated context
    # managers. Both close automatically at the end of the block, even on error.
    # encoding="utf-8" because plot summaries contain accented chars, emoji, etc.
    with open("data/titles", "r", encoding="utf-8") as t, open("data/plots", "r", encoding="utf-8") as p:
        # titles file: one title per line, so splitlines() gives us a list.
        titles = t.read().splitlines()
        # plots file: single blob with "<EOS>" (end-of-story) markers between
        # each plot. split() gives us a list aligned index-for-index with titles.
        plots = p.read().split("<EOS>")

    # Cap how much we ingest per run. Currently we start AT `limit` and go
    # to the end of the dataset — i.e. we're picking up where an earlier
    # run of the first 1000 left off. (A bit of a footgun: rename to
    # `start_index` next time for clarity.)
    limit = 1000

    # Walk through every (title, plot) pair past the start index.
    for i in range(limit, len(titles)):
        # .strip() kills leading/trailing whitespace/newlines from the file.
        title_text=titles[i].strip()
        plot_text=plots[i].strip()

        # Skip empty plots — no point indexing a zero-length document, and
        # it would pollute search results.
        if not plot_text:
            continue

        # Generate a fresh v4 UUID on the Python side. We need the id NOW so
        # we can stamp it on both the Postgres row AND the vector chunks.
        book_id = uuid.uuid4()
        # Build the ORM object. Not persisted yet — db.add() stages it in
        # the session's "to be INSERTed" queue; commit() flushes it.
        new_book = BookMetadata(id=book_id, title=title_text, plot=plot_text)
        db.add(new_book)

        # Split the full plot into retrieval-sized chunks. Returns a list of
        # strings — the splitter handles the overlap sliding window for us.
        chunks = text_splitter.split_text(plot_text)

        # Build a parallel list of metadata dicts, one per chunk. Every chunk
        # carries the SAME book_id/title so at query time we can trace any
        # retrieved chunk back to its source book in Postgres.
        # `_` because we don't actually need the loop variable — we only want
        # the length to match `chunks`.
        metadatas=[{"book_id": str(book_id), "title": title_text} for _ in chunks]

        # Embed all chunks and push them into OpenSearch in one call. The
        # vector store handles the embedding under the hood via the
        # `embedding_function` we wired up at construction.
        docsearch.add_texts(
            texts=chunks,
            metadatas=metadatas
        )

        # Batch-commit to Postgres every 10 books instead of every single one.
        # Trades a bit of crash-safety for a LOT of throughput — committing
        # per-row would be dominated by transaction overhead.
        if i % 10 == 0:
            db.commit()
            print(f"Progress: {i}/{limit} - Loaded: {title_text}")

    # Final commit catches the last <10 rows that didn't hit the modulo check.
    db.commit()
    # Return the connection to the pool. Without this we'd leak connections
    # on every run.
    db.close()
    print("Ingestion complete.")

# Standard Python idiom: only run ingest_data() when this file is executed
# directly (`python load_wikiplots.py`). When main.py does `from load_wikiplots
# import docsearch`, this block is SKIPPED — we just want the `docsearch`
# object, not a full re-ingest on every API startup.
if __name__ == "__main__":
    ingest_data()
