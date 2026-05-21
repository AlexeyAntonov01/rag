import os
from ollama import AsyncClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import(
    Distance,
    VectorParams,
    PointStruct
    )
import asyncio
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer
from docling.document_converter import(
    DocumentConverter,
    PdfFormatOption
    )
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
import uuid
from pathlib import Path
import pymupdf
import hashlib
from functools import partial
import re


class DocumentProcessor:
    # Принять документ -> Вернуть чанки с методанными

    def __init__(self):

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.generate_page_images = False 
        pdf_options = PdfFormatOption(pipeline_options=pipeline_options)

        self.docs_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_options
            }
            )    
        self.chunker = HybridChunker(tokenizer=os.getenv("EMBED_MODEL"),
                                    max_tokens=512,
                                    merge_peers=True)
        

    async def getDocument(self, file_path):

        if not os.path.exists(file_path):
            return

        loop = asyncio.get_running_loop()

        def proccess_pdf():

            try:
                filename = os.path.basename(file_path)
                temp_pdf_path = f"data/temp_no_images_{filename}"

                with pymupdf.open(file_path) as doc:
                    
                    for num, page in enumerate(doc):

                        images = page.get_image_info(hashes=True)

                        try:

                            for img in images:

                                filename_hash = f"{img['digest'].hex()}.png"
                                save_path = f"data/output_images/{filename_hash}"

                                pix = page.get_pixmap(clip = img['bbox'],matrix=pymupdf.Matrix(3,3))
                                pix.save(save_path)
                                pix = None

                                page.add_redact_annot(img['bbox'], fill = (1,1,1))
                                page.apply_redactions()

                                point = (img['bbox'][0], img['bbox'][1])

                                page.insert_text(point,
                                                f'<image_ref>{filename_hash}</image_ref>',
                                                fontsize=7, 
                                                color = (1,0,0))
                                
                        except Exception as page_error:
                            print(f'Ошибка при разборе картинки на {num}:{page_error}')
                            continue

                    doc.save(temp_pdf_path, garbage=4, deflate=True, clean=True)
                    

            except Exception as e:

                print(f"Критическая ошибка при работе с файлом {filename}: {e}")

            result = self.docs_converter.convert(temp_pdf_path)
            chunks = list(self.chunker.chunk(result.document))

            return chunks, filename

        return await loop.run_in_executor(None,proccess_pdf)
    

class VectorStore:
    # Превращает чанки в векторы -> Insert в БД
    def __init__(self):

        self.collection_name = os.getenv("COLLECTION_NAME")
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.client = AsyncQdrantClient(host = self.qdrant_host,port=int(os.getenv("QDRANT_PORT")))
       
        self.emb_fn = SentenceTransformer(
            os.getenv("EMBED_MODEL"),
            trust_remote_code=True
        )

    async def _init_db(self):

        if not await self.client.collection_exists(self.collection_name):

            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config = VectorParams(size=768,distance=Distance.COSINE))


    async def chunks2Collection(self,chunks,file_name):
        
        points = []
        batch_size = 16 
        all_texts = [chunk.text for chunk in chunks]
        
        for chunk in chunks:

            text_fixed = chunk.text
            text_fixed = re.sub(r'\s+(<image_ref>)', r' \1', text_fixed)

            chunk.text = text_fixed

        all_texts = [chunk.text for chunk in chunks]

        loop = asyncio.get_running_loop()
        
        for i in range(0, len(all_texts), batch_size):

            batch_texts = all_texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            encode_func = partial(self.emb_fn.encode, batch_texts,convert_to_tensor=False,prompt="search_document: ")

            batch_vectors = await loop.run_in_executor(None,encode_func
                )

            for j, vector in enumerate(batch_vectors):

                chunk = batch_chunks[j]

                text_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
                stable_id = str(uuid.UUID(text_hash))

                points.append(
                    PointStruct(
                        id=stable_id,
                        vector=vector.tolist(),
                        payload={
                            'text': chunk.text,
                            'metadatas': {
                                'filename': file_name,
                                'title': chunk.meta.headings if chunk.meta.headings else None
                            }
                        }
                    )
                )
        
        await self.client.upsert(collection_name=self.collection_name, points=points)


    async def clear_db(self):

        try:
            await self.client.delete_collection(self.collection_name)
            await self.client.create_collection(collection_name = os.getenv("COLLECTION_NAME"),
                vectors_config=VectorParams(size=768, distance=Distance.COSINE))

            return True

        except Exception as e:
            return e


class RagManager:

   
    def __init__(self):
       
        self.histories = {}
        self.processor  = DocumentProcessor()
        self.store = VectorStore()
        self.ollama_host = os.getenv("OLLAMA_HOST")
        self.client = AsyncClient(host=self.ollama_host)
        self.semaphore = asyncio.Semaphore(10)
        self.emb_lock = asyncio.Lock() 

    async def upload_file(self,file_path):

        async with self.semaphore:
            chunks,file_name = await self.processor.getDocument(file_path)
            if chunks:
                await self.store.chunks2Collection(chunks,file_name)
            else:
                print('Нет чанков!',flush=True)

    async def upload_file_multi(self,file_path_list:list):

        #последовательно иначе oom
        for file_nm in file_path_list:    
            chunks,file_name = await self.processor.getDocument(file_nm)
            if chunks:
                await self.store.chunks2Collection(chunks,file_name)
            else:
                print('Нет чанков!',flush=True)

    def clearHistory(self,user_id):

        self.histories.pop(user_id,None)


    async def ask(self, query,user_id=0):

        user_history = self.histories.get(user_id, [])
        loop = asyncio.get_running_loop()
        emb_qiery_func = partial(self.store.emb_fn.encode,query)
        async with self.emb_lock:
            emb_qiery_raw = await loop.run_in_executor(
                None,
                emb_qiery_func
            )
        emb_qiery = emb_qiery_raw.tolist() if hasattr(emb_qiery_raw, 'tolist') else list(emb_qiery_raw)

        search_results = (await self.store.client.query_points(query=emb_qiery,collection_name = os.getenv("COLLECTION_NAME"),limit=15)).points
        
        if not search_results:
            return "Информация не найдена"
      
        docs = [f"Документ: {hit.payload['metadatas']['filename']}\n{hit.payload['text']}" for hit in search_results]
        context_text = "\n\n".join(docs)

        history_text = ""
        if user_history:
            history_text = "Предыдущий диалог:\n"
            for turn in user_history[-5:]:  
                history_text += f"Пользователь: {turn['user']}\n"
                history_text += f"Ассистент: {turn['assistant']}\n"
       
        prompt = f"""Ты - ассистент технической поддержки Axapta. Твоя единственная задача - дать точный, структурированный ответ на вопрос пользователя, опираясь исключительно на предоставленный КОНТЕКСТ.

                ПРАВИЛА:
                1. Максимально полный ответ: используй ВСЮ информацию из контекста (все шаги, примечания, условия)
                2. При уточняющих вопросах ("почему?", "подробнее") используй ИСТОРИЮ ДИАЛОГА, но ответ строй по текущему КОНТЕКСТУ
                3. Работа с картинками: XML-теги <image_ref>...</image_ref> - копируй в ответ ТОЧНО на тех же местах
                4. Запрещены внешние знания. Если нет ответа: "Информация не найдена в базе знаний Axapta"
                5. В конце: "Источник: <название файла>"

                КОНТЕКСТ ДЛЯ ОТВЕТА:
                {context_text}

                ИСТОРИЯ ДИАЛОГА:
                {history_text}

                ВОПРОС ПОЛЬЗОВАТЕЛЯ: {query}

                ПОШАГОВЫЙ ОТВЕТ АССИСТЕНТА:"""

        response = await self.client.generate(model="qwen2.5:7b", prompt=prompt, options={'temperature': 0, "num_ctx": 12288})

        if user_id not in self.histories:
            self.histories[user_id] = []
        
        self.histories[user_id].append({
            'user': query,
            'assistant': response["response"]
        })
        
        self.histories[user_id] = self.histories[user_id][-5:]

        return response['response']




if __name__ == "__main__":
    
    rag = RagManager()
    file_to_upload = "data/СКУД/test.pdf" 

    if os.path.exists(file_to_upload):
        print(f"Загрузка файла: {file_to_upload}")
       
        rag.upload_file(file_to_upload)
        print("Файл успешно загружен в базу Qdrant")

        
