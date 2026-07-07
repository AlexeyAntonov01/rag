from app.core.rag_proccesor import RagManager,VectorStore 
from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
from loguru import logger

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info('=== Запуск приложения ===')
    try:
        rag = RagManager()
        app.state.rag = rag
        await app.state.rag.store._init_db()
        logger.info('База данных Qdrant успешно инициализирована')
    except Exception as e:
        logger.critical(f'Критическая ошибка при старте: не удалось настроить RAG или БД: {e}')
        raise e

    yield
    logger.info('=== Остановка приложения FastAPI ===')
    if hasattr(app.state, 'rag'):
        del app.state.rag
        
    logger.info('Ресурсы успешно очищены')