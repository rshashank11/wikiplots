from sqlalchemy import Column, String, Text, UUID
import uuid
from database import Base

class BookMetadata(Base):
    __tablename__='book_metadata'
    id=Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title=Column(String, nullable=False)
    plot=Column(Text, nullable=False)
