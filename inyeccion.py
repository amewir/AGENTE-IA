import sys
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

root_path = Path(__file__).resolve().parent
sys.path.append(str(root_path))
from src.config import VECTOR_DB_DIR, EMBEDDING_MODEL_NAME

ruta_pdf = root_path / "data" / "Docs" / "decreto" / "Decreto-5-2021-Simplificacion-de-Tramites.pdf"

print("Cargando pdf...")
loader = PyPDFLoader(str(ruta_pdf))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# Limpieza de fuente metadados
for chunk in chunks:
    chunk.metadata["source"] = "Decreto-5-2021"

print(" Conectando a ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vector_db = Chroma(collection_name="Conocimiento_Chatbot_GAE", embedding_function=embeddings, persist_directory=str(VECTOR_DB_DIR))

print("Inyectado de vectores...")
vector_db.add_documents(chunks)

# ─── FORZAMOS EL GUARDADO EN DISCO DURO ───
if hasattr(vector_db, 'persist'):
    vector_db.persist()

print("Exito")