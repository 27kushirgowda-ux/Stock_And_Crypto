from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")

# ✅ Local SQLite fallback
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./users.db"

# ✅ Fix postgres URL
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql://",
        1
    )

# ✅ SQLite (Local)
if "sqlite" in DATABASE_URL:

    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )

# ✅ PostgreSQL (Render)
else:

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"sslmode": "require"}
    )

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()