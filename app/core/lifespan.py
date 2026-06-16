from app.core.rag_proccesor import RagManager,VectorStore 
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):

    rag = RagManager()
    app.state.rag = rag

    await app.state.rag.store._init_db()

    yield

    del app.state.rag