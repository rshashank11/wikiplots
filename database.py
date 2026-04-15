from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

import os
from dotenv import load_dotenv

load_dotenv()

POSTGRESQL_URL = os.getenv("DATABASE_URL", "")
engine = create_engine(POSTGRESQL_URL) # Create a physical connection to postgres DB

SessionLocal = sessionmaker(autoflush=False, bind=engine) # Individual workspace connected to the postgres DB

class Base(DeclarativeBase):
    pass

def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()