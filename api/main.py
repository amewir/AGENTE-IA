import sys
from pathlib import Path

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
sys.path.append(str(root_path / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from src.tools.buscador_rag import respuesta_estatica, realizar_consulta
except ModuleNotFoundError:
    from tools.buscador_rag import respuesta_estatica, realizar_consulta

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agente IA - GAE / API")

# Uso de cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def end_chatpoint(request: ChatRequest):
    query = request.message.strip()

    res_stc = respuesta_estatica(query)

    if res_stc:
        return {
            "response": res_stc,
            "source": "static"
        }
    
    try:
        respuesta_rag = realizar_consulta(query)

        return {
            "response": respuesta_rag,
            "source": "rag"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))