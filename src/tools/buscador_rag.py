import sys
import os
import json
import random
import re
from pathlib import Path, PureWindowsPath
import unicodedata
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama 

# CONFIGURACIÓN DE RUTAS 
root_path = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_path))

from src.config import VECTOR_DB_DIR, MODEL_NAME, TEMPERATURE, OLLAMA_HOST, EMBEDDING_MODEL_NAME

#CARGA DE BASE DE DATOS VECTORIAL
def cargar_base_datos():
    print(f"--- Cargando Base de Datos de Vectores desde: {VECTOR_DB_DIR} ---")
    
    embeddings = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME,
        model_kwargs = {'local_files_only': False}
    )
    
    vector_db = Chroma(
        collection_name = "Conocimiento_Chatbot_GAE",
        embedding_function = embeddings,
        persist_directory = str(VECTOR_DB_DIR)
    )
    return vector_db

# INICIALIZACIÓN
vector_db_global = cargar_base_datos()
llm_global = ChatOllama(model=MODEL_NAME, temperature=TEMPERATURE, base_url=OLLAMA_HOST)

# PROMPT TEMPLATE OPTIMIZADO 
prompt_agent_template = """
Eres el Asistente Virtual Ciudadano de la GAE. Tu cerebro es EXCLUSIVAMENTE el [CONTEXTO RECUPERADO].
Tu objetivo es traducir el lenguaje legal seco en derechos y beneficios claros para el ciudadano.

INSTRUCCIONES DE REDACCIÓN:
1. TRADUCCIÓN DE PROHIBICIONES: Si el texto legal dice que "queda prohibido a las dependencias cobrar por formularios", tradúcelo SIEMPRE como un derecho positivo para el ciudadano: "Derecho a recibir formularios y solicitudes de forma totalmente gratuita".
2. REGLA DE ORO DE GRATUIDAD: Bajo ninguna circunstancia redactes que los formularios o solicitudes TIENEN COSTO o que NO son gratuitos. La ley obliga a que sean 100% gratuitos y digitales. Corrige cualquier fragmento que contradiga esto.
3. COHERENCIA Y SENTIDO COMÚN: Revisa críticamente lo que redactas. Nunca digas que un derecho del ciudadano es "no recibir servicios gratuitos" o "restringir accesos". Invierte la redacción para que tenga coherencia con el espíritu de la ley.
4. NO hagas copy-paste de fragmentos rotos. Redacta oraciones completas, amables y con perfecto sentido gramatical.

EJEMPLO DE TRADUCCIÓN CORRECTA:
- Texto legal confuso: "Prohibido cobrar por formularios o solicitudes en medios físicos o electrónicos."
- Tu redacción ciudadana: "* Obtener de forma 100% gratuita todos los formularios y solicitudes, tanto en formato físico como digital."

REGLAS DE FORMATO Y CONOCIMIENTO BASE:
- CONTEXTO INSTITUCIONAL: Ten siempre en cuenta que el "Decreto 5-2021" es la "Ley para la Simplificación de Requisitos y Trámites Administrativos". Su objetivo es modernizar la gestión pública.
- IGNORA TOTALMENTE los artículos sobre entrada en vigencia (como el Art. 43) a menos que se te pregunte específicamente por fechas.
- SÍNTESIS COMPLETA: Analiza TODOS los fragmentos recuperados antes de responder. No te quedes solo con el primer dato que encuentres.
- NO uses frases introductorias ni de relleno ("Según el contexto...", "El documento dice...").
- Si hay varios puntos, usa viñetas (*) para estructurarlos de forma limpia.

[CONTEXTO RECUPERADO]:
{context}

[CONSULTA DEL CIUDADANO]:
{question}
"""
prompt_global = ChatPromptTemplate.from_template(prompt_agent_template)

#FUNCIONES DE UTILIDAD Y MÓDULO ESTÁTICO 
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

def respuesta_estatica(query: str):
    path_respuestas = Path(__file__).resolve().parent.parent.parent / "data" / "static" / "intents.json"
    try:
        with open(path_respuestas, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    
    query_limpia = limpiar_consulta(query)
    for intent in data['intents']:
        for pattern in intent["patterns"]:
            p_limpio = limpiar_consulta(pattern)
            if not p_limpio: continue
            patron_exacto = rf"\b{re.escape(p_limpio)}\b"
            if re.search(patron_exacto, query_limpia):
                return random.choice(intent['responses'])
    return None

# FUNCIÓN PRINCIPAL DE CONSULTA
def realizar_consulta(query_crudo: str):
    # Intentar resolver mediante respuestas estáticas fijas primero
    res_json = respuesta_estatica(query_crudo)
    if res_json: return res_json

    query_limpia = limpiar_consulta(query_crudo)
    if not query_limpia: return "Consulta no válida o vacía."

    # Detección de intención legal general
    keywords_legales = [
        "decreto", "ley", "acuerdo", "norma", "presupuesto", 
        "5-2021", "36-2024", "simplificacion", "tramite", "tramitres"
    ]
    es_legal = any(k in query_crudo.lower() for k in keywords_legales)
    
    search_kwargs = {"k": 7}
    query_busqueda = query_limpia

    # FILTROS

    if "5-2021" in query_crudo.lower() or "simplificacion" in query_crudo.lower():
        query_busqueda = "Artículo 1 Objeto de la ley proposito simplificacion de requisitos y tramites administrativos reducir dependencias del estado"
        search_kwargs["filter"] = {
            "source": "Decreto-5-2021"
        }
    elif "36-2024" in query_crudo.lower():
        query_busqueda = f"{query_limpia} presupuesto general de ingresos y egresos del estado finanzas"
        search_kwargs["filter"] = {
            "source": "36-2024"
        }
    elif "gae" in query_limpia and ("que es" in query_limpia or "que hace" in query_limpia or "funcion" in query_limpia):
        # Le inyectamos los conceptos institucionales y bloqueamos el documento de la ONU
        query_busqueda = "funciones atribuciones misión visión Comisión Presidencial de Gobierno Abierto y Electrónico"
        search_kwargs["filter"] = {
            "$and": [
                {"source": {"$not_contains": "UN-E-Government"}},
                {"categoria": {"$in": ["manuales", "memoria"]}}
            ]
        }




    elif es_legal:
        # Búsqueda legal general (sin restricciones de nombre de archivo)
        search_kwargs["filter"] = {
            "$and": [
                {"source": {"$not_contains": ".xlsx"}},
                {"source": {"$not_contains": "Alertas"}},
                {"source": {"$not_contains": "Covid"}}
            ]
        }
    else:
        # Búsqueda operativa genérica
        search_kwargs["filter"] = {
            "$and": [
                {"source": {"$not_contains": ".xlsx"}},
                {"source": {"$not_contains": "Alertas"}},
                {"source": {"$not_contains": "Covid"}},
                {"categoria": {"$in": ["informes", "manuales", "plan", "memoria"]}}
            ]
        }

    retriever = vector_db_global.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )

    try:
        docs = retriever.invoke(query_busqueda)
        print(f"\n[DEBUG RAG] ======= CONSULTA PROCESADA: '{query_limpia}' =======")
        print(f"[DEBUG RAG] Filtros aplicados a Chroma: {search_kwargs.get('filter')}")
        print(f"[DEBUG RAG] Documentos recuperados de la DB: {len(docs)}")
        for i, d in enumerate(docs):
            print(f"    -> [{i+1}] {PureWindowsPath(d.metadata.get('source', '---')).name}")
        print("==================================================\n")
    except Exception as ed:
        print(f"[DEBUG ERROR] Error en canal de despliegue de logs: {ed}")

    rag_chain = (
        {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
        | prompt_global
        | llm_global
        | StrOutputParser()
    )
    
    try:
        return rag_chain.invoke(query_limpia)
    except Exception as e:
        return f"[ERROR] Ocurrió una anomalía al procesar la consulta en el modelo: {e}"