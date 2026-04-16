from pydantic import BaseModel
import uuid

# Pydantic schema for the /search POST body. FastAPI will:
#   1. parse the incoming JSON,
#   2. validate types (reject if `query` isn't a string),
#   3. hand us a typed Python object.
class SearchQuery(BaseModel):
    query: str        # required — the user's natural-language search
    top_k: int = 5    # optional — defaults to 5 if the client omits it


class BookCreate(BaseModel):
    title: str
    plot: str

class BookResponse(BaseModel):
    id: uuid.UUID
    title: str
    plot: str

    class Config:
        from_attributes = True