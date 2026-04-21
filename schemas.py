from pydantic import BaseModel, Field
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

class SearchPlan(BaseModel):
    subject: str = Field(
        description="The specific title or main character mentioned (e.g., 'Harry Potter', 'Tim Cook')"
    )
    search_query: str = Field(
        description="An optimized version of the query for a search engine, focusing only on keywords"
    )
    is_specific_entity: bool = Field(
        description="True if the user is asking about specific book or movie or person or character, etc., False for general topics"
    )
    detected_intent: str = Field(
        description="What the user wants to know (e.g., 'family_info', 'plot_summary', 'biography')"
    )