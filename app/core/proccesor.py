import os
from ollama import Client
from qdrant_client import QdrantClient
from qdrant_client.models import(
    Distance,
    VectorParams,
    PointStruct
    )
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
        self.chunker = HybridChunker(tokenizer=os.getenv("EMBED_MODEL"),max_tokens=512)
        

    def getDocument(self, file_path):

        if not os.path.exists(file_path):
            return

        try:
            filename = os.path.basename(file_path)
            temp_pdf_path = f"data/temp_no_images_{filename}"

            with pymupdf.open(filename) as doc:
                
                for num, page in enumerate(doc):

                    images = page.get_image_info(hashes=True)
                    dict_[num] = images

                    try:

                        for img in images:

                            filename_hash = f"{filename}_{img['digest'].hex()}.png"
                            save_path = f"extracted_images/{filename_hash}"

                            pix = page.get_pixmap(clip = img['bbox'],matrix=pymupdf.Matrix(3,3))
                            pix.save(save_path)
                            pix = None

                            page.add_redact_annot(img['bbox'], fill = (1,1,1))
                            page.apply_redactions()

                            point = (img['bbox'][0],img['bbox'][1])

                            page.insert_text(point,
                                            f'[IMAGE_REF:{filename_hash}]',
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

        return chunks, file_name
    

class VectorStore:
    # Превращает чанки в векторы -> Insert в БД
    def __init__(self):

        self.collection_name = os.getenv("COLLECTION_NAME")
        qdrant_host = os.getenv("QDRANT_HOST")
        self.client = QdrantClient(host = qdrant_host,port=int(os.getenv("QDRANT_PORT")))

        if not self.client.collection_exists(self.collection_name):

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config = VectorParams(size=768,distance=Distance.COSINE))
           
        self.emb_fn = SentenceTransformer(
            os.getenv("EMBED_MODEL"),
            trust_remote_code=True
        )


    def old_chunks2Collection(self,chunks,file_name):
        
        points = []

        for chunk in chunks:
        
            title = chunk.meta.headings[0] if chunk.meta.headings else "Название отсутсвует"

            points.append(
                    PointStruct(
                        id = str(uuid.uuid4()),
                        vector = self.emb_fn.encode(chunk.text).tolist(),
                        payload = {'text':chunk.text,
                                        'metadatas':
                                        {'filename':file_name,
                                         'title':title}
                                        
                                }
                        )
                    )                                           
                    
        self.client.upsert(collection_name = self.collection_name,points = points)


    def chunks2Collection(self,chunks,file_name):
        
        points = []
        batch_size = 16 
        all_texts = [chunk.text for chunk in chunks]

        for i in range(0, len(all_texts), batch_size):

            batch_texts = all_texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            batch_vectors = self.emb_fn.encode(
                batch_texts,
                convert_to_tensor=False,
                prompt="search_document: "
            )

            for j, vector in enumerate(batch_vectors):
                chunk = batch_chunks[j]
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector.tolist(),
                        payload={
                            'text': chunk.text,
                            'metadatas': {
                                'filename': file_name,
                                'title': chunk.meta.headings if chunk.meta.headings else "Инструкция"
                            }
                        }
                    )
                )
        
        self.client.upsert(collection_name=self.collection_name, points=points)


    def clear_db(self):

        try:
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(collection_name = os.getenv("COLLECTION_NAME"),
                vectors_config=VectorParams(size=768, distance=Distance.COSINE))

        except Exception as e:
            return e


class RagManager:

   
    def __init__(self):
       
        self.histories = {}
        self.processor  = DocumentProcessor()
        self.store = VectorStore()

    def upload_file(self,file_path):

        chunks,file_name = self.processor.getDocument(file_path)
        if chunks:
            self.store.chunks2Collection(chunks,file_name)
        else:
            print('Нет чанков!')

    def clearHistory(self,user_id):

        try:

            if self.histories.get(user_id):

                self.histories.pop(user_id)

        except Exception as e:

             raise e


    def ask(self, query,user_id=0):

        user_history = self.histories.get(user_id, [])
        emb_qiery = self.store.emb_fn.encode(query).tolist()
        search_results = self.store.client.query_points(query=emb_qiery,collection_name = os.getenv("COLLECTION_NAME"),limit=8).points
        
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
       
        prompt = f"""Ты — база знаний, ассистент специалиста технической поддержки Axapta. Твоя задача: ответить на вопрос, используя данные из базы данных, при выводе ответа укажи тему инструкции(название файла) и её дату откуда был взят ответ
                ПРАВИЛА:
                1. Отвечай ТОЛЬКО на основе предоставленных документов
                2. Если вопрос уточняющий ("не понял", "подробнее") - используй ТЕ ЖЕ документы, что и в предыдущем ответе
                3. Всегда указывай название документа-источника
                4. Если информация не найдена - скажи "Информация не найдена в базе знаний Axapta"
                КОНТЕКСТ:
                {context_text}

                ИСТОРИЯ ДИАЛОГА:
                {history_text}

                ВОПРОС:
                {query}

                ОТВЕТ:"""
        
        ollama_host = os.getenv("OLLAMA_HOST")
        client = Client(host=ollama_host)

        response = client.generate(model="qwen2.5:14b", prompt=prompt, options={'temperature': 0})

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

        
