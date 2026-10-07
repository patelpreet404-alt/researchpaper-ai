"""
SQLAlchemy engine/session management.

SQLite is used as a lightweight, zero-configuration persistence layer for
document metadata and conversation history. The vector embeddings
themselves live in the FAISS index (see ``app.services.vector_store_service``),
not in SQLite.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base class for all ORM models."""


settings = get_settings()

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not already exist."""
    # Import models so they are registered on Base.metadata before create_all.
    from app.infrastructure import models_db  # noqa: F401

    Base.metadata.create_all(bind=engine)
    # Upgrade databases created before browser-session isolation was added.
    # Existing rows are assigned to an inaccessible legacy owner so they are
    # never exposed to a newly-created visitor session.
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table in ("documents", "conversations"):
            columns = {column["name"] for column in inspector.get_columns(table)}
            if "owner_id" not in columns:
                connection.execute(
                    text(
                        f"ALTER TABLE {table} ADD COLUMN owner_id VARCHAR(36) NOT NULL "
                        "DEFAULT 'legacy'"
                    )
                )
    logger.info("Database initialized at %s", settings.database_url)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for a database session outside of FastAPI's DI (e.g. in services)."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
