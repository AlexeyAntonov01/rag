import os
from ollama import AsyncClient
from qdrant_client import AsyncQdrantClient,models
from qdrant_client.models import(
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    Prefetch,
    RrfQuery
    )
from fastembed import TextReranker
from fastembed.sparse import SparseTextEmbedding
import asyncio
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

            return chunks, filename, result

        return await asyncio.to_thread(proccess_pdf)
    

class VectorStore:
    # Превращает чанки в векторы -> Insert в БД
    def __init__(self):

        self.collection_name = os.getenv("COLLECTION_NAME")
        self.qdrant_host = os.getenv("QDRANT_HOST")
        self.client = AsyncQdrantClient(host = self.qdrant_host,port=int(os.getenv("QDRANT_PORT")))
        self.sparse_emb_fn = SparseTextEmbedding(model_name="Qdrant/bm25")
       
        self.emb_fn = SentenceTransformer(
            os.getenv("EMBED_MODEL"),
            trust_remote_code=True
        )

    async def _init_db(self):

        if not await self.client.collection_exists(self.collection_name):

            await self.client.create_collection(
                collection_name=self.collection_name,
               

                vectors_config = {'dense': VectorParams(size=768,distance=Distance.COSINE)},

                sparse_vectors_config = {'sparse':SparseVectorParams()}

                )

    async def chunks2Collection(self,chunks,file_name,response_generated_topic):
        
        points = []
        batch_size = 16 
        custom_prompt = f"search_document: Этот текст взят из документа {file_name}. Текст: "

        for chunk in chunks:

            text_fixed = chunk.text
            text_fixed = re.sub(r'\s+(<image_ref>)', r' \1', text_fixed)
            chunk.text = text_fixed
            
        all_texts = [chunk.text for chunk in chunks]
        
        for i in range(0, len(all_texts), batch_size):

            batch_texts = all_texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            encode_func = partial(self.emb_fn.encode, batch_texts,convert_to_tensor=False,prompt=custom_prompt)
            encode_func_sparse = partial(self.sparse_emb_fn.encode,batch_texts)

            batch_vectors = await asyncio.to_thread(encode_func)
            batch_sparse_raw = await asyncio.to_thread(encode_func_sparse)
            batch_sparse = list(batch_sparse_raw)

            for j, vector in enumerate(batch_vectors):

                chunk = batch_chunks[j]
                current_sparse = batch_sparse[j]

                text_hash = hashlib.md5(chunk.text.encode('utf-8')).hexdigest()
                stable_id = str(uuid.UUID(text_hash))

                vector_sparse = models.SparseVector(

                        indices = current_sparse.indices.tolist(),
                        values = current_sparse.values.tolist()
                    )

                points.append(
                    PointStruct(
                        id=stable_id,
                        vector=
                            {
                                'dense' :vector.tolist(),
                                'sparse':vector_sparse
                            }
                        ,
                        payload={
                            'text': chunk.text,
                            'metadatas': {
                                'filename': file_name,
                                'title': response_generated_topic if response_generated_topic else 'Неизвестно'
                            }
                        }
                    )
                )
        
        await self.client.upsert(collection_name=self.collection_name, points=points)


    async def clear_db(self):

        try:
            await self.client.delete_collection(self.collection_name)
            await self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config = {'dense': VectorParams(size=768,distance=Distance.COSINE)},
                        sparse_vectors_config = {'sparse':SparseVectorParams()}
            )

            return True

        except Exception as e:
            raise  e


class UserHistory:


    def __init__(self):

        self.turn = []
        self.user_lock = asyncio.Lock()

    def add_turn(self,query,response):

        self.turn.append(
                {
                    'user':query,
                    'assistant':response
                }
            )

        self.turn = self.turn[-4:]

    def clear_history(self):

        self.turn.clear()

    def get_text(self) -> str:

        if not self.turn:
            return ''

        history_text = "Предыдущий диалог:\n"
        for elem in self.turn[-4:]:

            history_text += f"Пользователь: {elem['user']}\n"
            history_text += f"Ассистент: {elem['assistant']}\n"

        return history_text


class RagManager:

   
    def __init__(self):

        self.processor  = DocumentProcessor()
        self.store = VectorStore()
        self.ollama_host = os.getenv("OLLAMA_HOST")
        self.client = AsyncClient(host=self.ollama_host)
        self.semaphore = asyncio.Semaphore(1)
        self.emb_lock = asyncio.Lock()
        self.reranker = TextReranker(model_name='BAAI/bge-reranker-v2-m3',normalize=True)
        self.histories = {}


    async def upload_file(self,file_path):

        async with self.semaphore:
            chunks,file_name,result = await self.processor.getDocument(file_path)

            full_text_markdown = result.document.export_to_markdown()

            if chunks:
                response_generated_topic = await self.generate_topic_prompt(full_text_markdown)
                await self.store.chunks2Collection(chunks,file_name,response_generated_topic)
            else:
                print('Нет чанков!',flush=True)

    async def upload_file_multi(self,file_path_list:list):

        async with self.semaphore:
            for file_nm in file_path_list:    
                chunks,file_name,result = await self.processor.getDocument(file_nm)
                full_text_markdown = result.document.export_to_markdown()

                if chunks:
                    response_generated_topic = await self.generate_topic_prompt(full_text_markdown)
                    await self.store.chunks2Collection(chunks,file_name,response_generated_topic)
                else:
                    print('Нет чанков!',flush=True)

    async def generate_topic_prompt(self,full_text_markdown):

        topic_prompt = f"""
                Ты - опытный аналитик технической документации и инструкций. Твоя задача — изучить текст документа и сформулировать ОДНУ главную тему этого документа на русском языке. ВАЖНО: Умести тему в одно предложение\n\n
                Текст документа:
                {full_text_markdown}
            """
        response_generated_topic = await self.client.generate(model='qwen2.5:7b', prompt=topic_prompt)

        return response_generated_topic['response']

    async def ask(self, query,user_id=0):

      
        if user_id not in self.histories:

            self.histories[user_id] = UserHistory()

        # Лок для пользователя, чтобы не ломалось история и небыло конкурентных запросов. 
        async with self.histories[user_id].user_lock:

            emb_qiery_func = partial(self.store.emb_fn.encode,query)
            emb_qiery_sparse_func = partial(self.store.sparse_emb_fn.encode,[query])

            ## Потенциально узкое место, модель эмбеддингов для плотных вектора не потокобезопасна
            ## Пришлось обернуть в асинхронный лок

            emb_qiery_sparse_raw_task =  asyncio.to_thread(
                    emb_qiery_sparse_func
                    )

            async with self.emb_lock:
                emb_qiery_raw =  await asyncio.to_thread(
                    emb_qiery_func
                )

            emb_qiery_sparse_raw = await emb_qiery_sparse_raw_task

            emb_qiery = emb_qiery_raw.tolist() if hasattr(emb_qiery_raw, 'tolist') else list(emb_qiery_raw)

            emb_qiery_sparse_list = list(emb_qiery_sparse_raw)
            current_emb_qiery_sparse = emb_qiery_sparse_list[0]

            emb_qiery_sparse = models.SparseVector(
                            indices = current_emb_qiery_sparse.indices.tolist(),
                            values = current_emb_qiery_sparse.values.tolist()
                )

            search_results = (await self.store.client.query_points(
                                                collection_name = os.getenv("COLLECTION_NAME"),
                                                prefetch = [
                                                        Prefetch(query = emb_qiery, using ='dense',limit=20),      
                                                        Prefetch(query = emb_qiery_sparse, using ='sparse',limit=20)],
                                                query= RrfQuery(rrf=models.Rrf()),
                                                limit = 30
                                                )

                            ).points
            
            if not search_results:
                return "Информация не найдена"

            docs = [f"---Документ: {hit.payload['metadatas']['filename']}\nКраткое описание документа: {hit.payload['metadatas']['title']}\nТекст: {hit.payload['text']}" for hit in search_results]

            reranker_score_partial = partial(self.reranker.rerank,query,docs)
            reranker_score = list(await asyncio.to_thread(reranker_score_partial))

            if len(reranker_score) > 0 and reranker_score[0].score <= 0.35:

                context_text = 'Не найдено релевантных ответов'

            else:
                context_text = "\n\n".join(docs[node.index] for node in reranker_score[:6])


            history_text = self.histories[user_id].get_text()
           
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

            response = await self.client.generate(model="qwen2.5:7b", prompt=prompt, options={'temperature': 0, "num_ctx": 14000})

            self.histories[user_id].add_turn(query,response['response'])

            return response['response']
            