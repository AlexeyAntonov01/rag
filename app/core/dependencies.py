from fastapi import Request
from .rag_proccesor import RagManager


async def get_rag(request:Request) -> RagManager:

	return request.app.state.rag