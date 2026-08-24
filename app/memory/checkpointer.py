"""Checkpointer persistente de LangGraph usando SQLite."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data" / "memory"

DEFAULT_DB_PATH = DB_DIR / "checkpoints.sqlite"
DB_PATH = Path(
    os.getenv(
        "CHECKPOINT_DB_PATH",
        str(DEFAULT_DB_PATH),
    )
)


@asynccontextmanager
async def checkpointer_context():
    """Abre y cierra el checkpointer SQLite durante la vida de la aplicación."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        yield checkpointer