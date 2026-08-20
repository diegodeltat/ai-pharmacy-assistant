"""Checkpointer persistente de LangGraph usando SQLite."""

from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


BASE_DIR = Path(__file__).resolve().parents[2]
DB_DIR = BASE_DIR / "data" / "memory"
DB_PATH = DB_DIR / "checkpoints.sqlite"


@asynccontextmanager
async def checkpointer_context():
    """Abre y cierra el checkpointer SQLite durante la vida de la aplicación."""

    DB_DIR.mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as checkpointer:
        yield checkpointer