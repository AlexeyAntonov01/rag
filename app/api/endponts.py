from fastapi import APIRouter, Request
from app.schemas.schemas import Question
import os
import asyncio

router = APIRouter()


@router.post("/ask")
async def ask_bot(request: Request, question: Question):

    rag = request.app.state.rag
    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, rag.ask, question.text)
    return {"answer": answer}


@router.post("/upload_pdf")
async def upload_pdf_store(request: Request,file_to_upload: str):

    rag = request.app.state.rag

    if not os.path.exists(file_to_upload):
        error_msg = f"ФАЙЛ НЕ НАЙДЕН: {os.path.abspath(file_to_upload)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

    try:
        rag.upload_file(file_to_upload)
        return {"status": "success", "message": f"Файл {file_to_upload} успешно загружен"}
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return {"status": "error", "message": str(e)}
    

@router.post("/drop_db")
async def drop_database(request: Request):

    rag = request.app.state.rag
    try:
        rag.store.clear_db()
        return {"status": "success", "message": f"БД удалена"}
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return {"status": "error", "message": str(e)}



    

