import time
from pathlib import Path
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import warnings

# Rutas y configuraciones
from dotenv import load_dotenv
from config import DATA_RAW_PATH, PROCESSED_DATA_DIR, VECTOR_DB_DIR, MODEL_NAME, TEMPERATURE

# Cargaremos dotenv para cargar las variables del entorno
load_dotenv()

# Silenciamos advertencias para mantener la terminal limpia
warnings.filterwarnings("ignore", category=UserWarning)


ruta_documentos = r"C:\Users\angel.hernandez\Documents\Progra\Agente_IA_GAE\data\Docs"
DATA_RAW_PATH = ruta_documentos

ruta_tesseract = r"C:\Program Files\Tesseract-OCR"
ruta_poppler = r"C:\Program Files\poppler\Library\bin"  # Ajusta esta ruta a donde hayas descomprimido poppler

os.environ["PATH"] += os.pathsep + ruta_tesseract + os.pathsep + ruta_poppler


categorias_totales = {
    "acuerdo-gubernativo", "acuerdo-interno",
    "cartas", "constitucion",
    "decreto", "egdi", "guias",
    "informes", "leyes", "manuales",
    "memoria", "normas","notas", "plan"
}

categorias_legales = {
    "acuerdo-gubernativo", "acuerdo-interno",
    "constitucion", "decreto", "leyes", "normas"
}


def obtener_documentos(directorio_base=DATA_RAW_PATH):
    archivos_finales = []
    # Cargar los documentos desde el directorio de datos en crudo
    print(f"--- Cargando los documentos desde: {directorio_base} ---")
    
    # diccionario para guardar los archivos por nombre, extension y evitar duplicidad de archivos
    archivos_por_nombre = {}

    # diccionario de extensiones permitidas para entender el contenido de los archivos.
    extensiones_permitidas = {
        ".pdf", 
        ".docx", 
    }

    permitidas = extensiones_permitidas
    for ruta in Path(directorio_base).rglob("*"):
        if ruta.is_file() and ruta.suffix.lower() in permitidas:
            nombre_ruta = ruta.stem
            extension_ruta = ruta.suffix.lower()

            if nombre_ruta not in archivos_por_nombre:
                archivos_por_nombre[nombre_ruta] = {}
            
            archivos_por_nombre[nombre_ruta][extension_ruta] = ruta
            
    archivos_finales = []

    for nombre, extensiones in archivos_por_nombre.items():
        if ".pdf" in extensiones:
            archivos_finales.append(extensiones[".pdf"])
        elif ".docx" in extensiones:
            archivos_finales.append(extensiones[".docx"])
        elif ".xlsx" in extensiones:
            archivos_finales.append(extensiones[".xlsx"])
        else:
            mejor_opcion = list(extensiones.keys())[0]
            archivos_finales.append(extensiones[mejor_opcion])

    # Mostrar en pantalla los archivos que se han cargado:
    for archivo in archivos_finales:
        # Imprime solo el nombre del archivo y la carpeta donde está, no la ruta entera
        print(f" [FILE] {archivo.parent.name}/{archivo.name}")
        
    print(f"[INFO] Cantidad de archivos encontrados: {len(archivos_finales)}")
    return archivos_finales

# Para la ingesta de datos, se procesaran los documentos encontrados, segun los parametros que se definieron anteriormente
# la variable que engloba los archivos procesados se llama "documentos_encontrados"
def ingesta_datos_chroma(ruta_ingesta, ruta_vector_db = VECTOR_DB_DIR):
    # Procesar los documentos encontrados
    print("\n--- Procesando los documentos encontrados ---")

    if not ruta_ingesta:
        print("[ERROR] No se encontraron documentos para procesar. Por favor, revisa la ruta de los datos.")
        return
        
    inicio_proceso = time.time()
    
    # Se carga el modelo de embeddings para convertir los fragmentos de texto en vectores numericos
    # HuggingFaceEmbeddings, es una clase que permite cargar modelos 
    # Tambien se tiene gemma3, de google, y este se carga con HuggingFaceEmbeddings, ya que es compatible
    print("[INFO] Cargando modelo de embeddings...")
    modeloIA = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Se procede a crear la base de datos vectorial usando chroma para el control de los vectores de información
    vector_db = Chroma(
        collection_name="Conocimiento_Chatbot_GAE",
        embedding_function=modeloIA,
        persist_directory=str(ruta_vector_db)
    )

    # Implementacion de procesamiento por lotes para optimizar memoria RAM
    

    divisor_legal = RecursiveCharacterTextSplitter(
        chunk_size = 1800,
        chunk_overlap = 300
    )

    divisor_general = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150
    )
    tamano_lote = 100
    for i in range(0, len(ruta_ingesta), tamano_lote):
        lote_actual = ruta_ingesta[i:i + tamano_lote]
        print(f"\n[LOTE {i//tamano_lote + 1}] Procesando archivos del {i} al {i+len(lote_actual)}...")

        # Cargar los documentos sin procesar utilizando unstructured, que es una libreria
        # que permite cargar diferentes tipos de archivos como pdf, docx, xlsx, txt, html, json, xml, php, entre otros.
        documentos_para_procesar = []

        for ruta in lote_actual:
            try:
                categoria_carpeta = ruta.parent.name.lower()
                print(f" Cargando el documento: {ruta.name}...")
                # Agregamos strategy='fast' para evitar errores de Poppler/OCR y agilizar el proceso
                loader = UnstructuredFileLoader(str(ruta), languages=["spa"], 
                                                strategy="hi_res")
                documentos_carga = loader.load()

                #agregar nueva funcion de categorizacion de docs
                for doc in documentos_carga:
                    doc.metadata["categoria"] = categoria_carpeta
                    doc.metadata["fuente"] = ruta.name
                
                if categoria_carpeta in categorias_legales:
                    fragmentos = divisor_legal.split_documents(documentos_carga)
                else:
                    fragmentos = divisor_general.split_documents(documentos_carga)
                
                documentos_para_procesar.extend(fragmentos)


            except Exception as e:
                print(f"[ERROR] Al cargar el documento {ruta.name}: {e}")
        
        if documentos_para_procesar:
            # Ahora toca los chunks
            
            # Fragmentos de textos, son los pedazos de texto para conocimiento del chatbot

            
            # Se agregan los fragmentos a la base de datos respetando el limite maximo de ChromaDB
            max_batch_chroma = 5000
            for j in range(0, len(documentos_para_procesar), max_batch_chroma):
                sub_lote = documentos_para_procesar[j:j + max_batch_chroma]
                vector_db.add_documents(sub_lote)
                
            print(f" [OK] Lote guardado. Fragmentos procesados: {len(documentos_para_procesar)}")

    fin = time.time()
    print(f"\n[INFO] Proceso de ingesta de datos finalizado en {fin - inicio_proceso:.2f} segundos.")

if __name__ == "__main__":
    # Iniciando el proceso de extraccion de datos
    print("--- Iniciando el proceso de extraccion de datos ---")
    archivos_encontrados = obtener_documentos()
    
    # Iniciando el proceso de ingesta de datos
    print("\n--- Iniciando el proceso de ingesta de datos ---")
    ingesta_datos_chroma(archivos_encontrados)
    
    print("--- Proceso finalizado con exito ---")