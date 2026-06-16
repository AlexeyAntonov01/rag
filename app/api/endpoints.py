from fastapi import (
    APIRouter, Request 
    )
from app.schemas.schemas import Question
import os
import asyncio

router = APIRouter()

@router.post("/ask")
async def ask_bot(request: Request, question: Question):

    rag = request.app.state.rag
    answer = await rag.ask(question.text)
    return {"answer": answer}


@router.post("/upload_pdf")
async def upload_pdf_store(request: Request,file_to_upload: str):

    rag = request.app.state.rag

    if not os.path.exists(file_to_upload):
        error_msg = f"ФАЙЛ НЕ НАЙДЕН: {os.path.abspath(file_to_upload)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

    try:

        await rag.upload_file(file_to_upload)
        return {"status": "success", "message": f"Файл {file_to_upload} успешно загружен"}
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return {"status": "error", "message": str(e)}
    

@router.post("/upload_pdf_batch")
async def upload_pdf_batch_store(request: Request,file_path_to_upload: str):

    rag = request.app.state.rag

    #указываю папку с файлами
    if not os.path.exists(file_path_to_upload):
        error_msg = f"Каталог НЕ НАЙДЕН: {os.path.abspath(file_path_to_upload)}"

        print(error_msg)
        return {"status": "error", "message": error_msg}

    try:

        loop = asyncio.get_running_loop()
        

        def find_file_name():
            
            file_list = []
            for entry in os.scandir(file_path_to_upload):
                if entry.is_file() and entry.name.lower().endswith('.pdf'):
                    file_list.append(entry.path)

            print(file_list,flush=True)
            return file_list
        file_list = await loop.run_in_executor(None,find_file_name)

        await rag.upload_file_multi(file_list)
        return {"status": "success", "message": f"Файлы из каталога {file_path_to_upload} успешно загружены"}
    
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return {"status": "error", "message": str(e)}
    

@router.post("/drop_db")
async def drop_database(request: Request):

    rag = request.app.state.rag
    try:
        await rag.store.clear_db()
        return {"status": "success", "message": f"БД удалена"}
    except Exception as e:
        print(f"ОШИБКА: {str(e)}")
        return {"status": "error", "message": str(e)}
    