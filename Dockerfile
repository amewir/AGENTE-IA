FROM python:3.11-slim

WORKDIR /app

#Dependencias del sistema 
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar e instalar librerías de Python 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Descargar modelos de IA 
RUN python -m spacy download en_core_web_sm
RUN python -c "from langchain_huggingface import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')"

#El código fuente se copia HASTA EL PURO FINAL
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]