import sys
import os
import json
import random
import re
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama 
from pathlib import Path, PureWindowsPath
import unicodedata

# Configuraciones
sys.path.append(str(Path(__file__).parent.parent.parent))
from dotenv import load_dotenv
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from src.config import DATA_RAW_PATH, PROCESSED_DATA_DIR, VECTOR_DB_DIR, MODEL_NAME, TEMPERATURE, OLLAMA_HOST

# --- INICIALIZACIÓN GLOBAL ---
def cargar_base_datos():
    from src.config import EMBEDDING_MODEL_NAME, VECTOR_DB_DIR
    print(f"--- Cargando Base de Datos: {VECTOR_DB_DIR} ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return Chroma(collection_name="Conocimiento_Chatbot_GAE", embedding_function=embeddings, persist_directory=str(VECTOR_DB_DIR))

# Inicializamos la base de datos y el modelo una sola vez
vector_db_global = cargar_base_datos()
llm_global = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, base_url=OLLAMA_HOST)

# --- PROMPT ---
prompt_agent_template = """
Eres el Consultor Senior de la GAE. Tu cerebro es EXCLUSIVAMENTE el [CONTEXTO RECUPERADO].

INSTRUCCIONES:
1. Si la consulta es sobre un Decreto, Ley o Acuerdo, HAZ UN RESUMEN BREVE de su contenido principal.
2. Si la consulta es sobre gestión o estrategia, busca la información operativa o técnica.
3. Si la respuesta NO existe en el contexto, responde únicamente: "Ese punto específico no figura en los registros documentados hasta la fecha."

REGLAS DE FORMATO:
- NO uses frases introductorias ni de relleno.
- NO imprimas títulos, ni etiquetas.
- Si hay listas o ejes, usa viñetas (*).
- NUNCA inventes información.

[CONTEXTO RECUPERADO]:
{context}

[CONSULTA DEL USUARIO]:
{question}
"""
prompt_global = ChatPromptTemplate.from_template(prompt_agent_template)

# --- UTILIDADES ---
def limpiar_consulta(texto: str) -> str:
    if not texto: return ""
    texto = str(texto).lower()[:500]
    texto = "".join(c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c))
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s\-]', '', texto)).strip()

def formatear_docs(docs):
    if not docs: return "SIN_CONTEXTO_DISPONIBLE"
    context = []
    for doc in docs:
        nombre = PureWindowsPath(Path(doc.metadata.get('source', 'Desconocido'))).name
        context.append(f"--- FUENTE: {nombre} ---\n{doc.page_content}\n")
    return "\n\n".join(context)

def realizar_consulta(query_crudo: str):
    query_limpia = limpiar_consulta(query_crudo)
    if not query_limpia: return "Consulta no válida."

    # 1. DETECCIÓN DE INTENCIÓN (Legal vs Operativo)
    keywords_legales = ["decreto", "ley", "acuerdo", "norma", "presupuesto"]
    es_legal = any(k in query_limpia.lower() for k in keywords_legales)
    
    # 2. FILTRO DINÁMICO
    # Si es consulta legal, buscamos en TODO. Si es operativa, filtramos para reducir ruido.
    filtro = None if es_legal else {"categoria": {"$in": ["informes", "manuales", "plan", "memoria"]}}
    
    # 3. CREAR RETRIEVER DINÁMICO
    retriever = vector_db_global.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 15, "filter": filtro}
    )

    # 4. DEBUG (Ver qué está leyendo realmente)
    docs = retriever.get_relevant_documents(query_limpia)
    print(f"\n--- DEBUG: Documentos encontrados para '{query_limpia}' ---")
    for d in docs: print(f"Archivo: {d.metadata.get('source')} | Cat: {d.metadata.get('categoria')}")

    # 5. EJECUCIÓN
    rag_chain = (
        {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
        | prompt_global
        | llm_global
        | StrOutputParser()
    )
    
    try:
        return rag_chain.invoke(query_limpia)
    except Exception as e:
        return f"[ERROR] Al procesar: {e}"