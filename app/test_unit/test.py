from app.core.proccesor import DocumentProcessor, VectorStore, RagManager
import pytest
import os



@pytest.fixture(scope='session')
def shared_processor():
    return DocumentProcessor()

@pytest.fixture(scope='session')
def shared_store():

    return VectorStore(db_path="./test_db_tmp")

@pytest.fixture()
def store(shared_store):
    shared_store.clear_db()
    return shared_store

@pytest.fixture()
def rag():
    
    manager = RagManager()
    manager.store.clear_db()
    return manager

class MockChunk:
    def __init__(self, text):
        self.text = text



def test_getDocument(shared_processor):
    path = "data/СКУД/test.pdf"
    if os.path.exists(path):
        chunks, file_name = shared_processor.getDocument(path)
        assert chunks is not None
        assert len(chunks) > 0
    
    assert shared_processor.getDocument('fake.pdf') is None

def test_store_add(store):
    initial_count = store.client.count(collection_name="my_docs").count
    store.chunks2Collection([MockChunk("Тест заливки в БД")], "test.pdf")
    new_count = store.client.count(collection_name="my_docs").count
    assert new_count == initial_count + 1

def test_upload_file(rag):
    path = "data/СКУД/test.pdf"
    if os.path.exists(path):
        rag.upload_file(path)
        count = rag.store.client.count(collection_name="my_docs").count
        assert count > 0

def test_chunks2Collection(store):

    filename = "secret_manual.pdf"
    store.chunks2Collection([MockChunk("Тест метаданных")], filename)
    
    points, _ = store.client.scroll(collection_name="my_docs", limit=1, with_payload=True)
    assert points[0].payload['metadatas'] == filename
