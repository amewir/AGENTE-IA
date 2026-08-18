import os 
import platform
from pathlib import Path
from dotenv import load_dotenv

#Cargaremos dotenv para cargar las variables del entorno

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Intentamos obtener la ruta genérica de Docker
DATA_RAW_PATH = os.getenv("DATA_RAW_PATH")

# Si no estamos en Docker (no hay DATA_RAW_PATH)
if not DATA_RAW_PATH:
    sys_platform = platform.system()
    if sys_platform == "Windows":
        DATA_RAW_PATH = os.getenv("DATA_RAW_PATH_WINDOWS")
    elif sys_platform == "Darwin":
        DATA_RAW_PATH = os.getenv("DATA_RAW_PATH_MACOS")
    elif sys_platform == "Linux":
        DATA_RAW_PATH = os.getenv("DATA_RAW_PATH_LINUX")

# Si por alguna razón sigue vacío, usamos una ruta relativa segura
if not DATA_RAW_PATH:
    DATA_RAW_PATH = str(BASE_DIR / "data" / "raw")

VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db_2"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
STATIC_DATA_DIR = BASE_DIR / "data" / "static"

# Crear directorios
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.getenv("MODEL_NAME", "gemma3")
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1)) # Bajamos a 0.1 para producción
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
