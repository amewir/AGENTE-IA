# Agente IA Institucional (RAG)

![Status](https://img.shields.io/badge/Status-Deployed-success)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![AI](https://img.shields.io/badge/AI-LangChain%20%7C%20HuggingFace-orange)

Proyecto destinado a crear un **Agente de Inteligencia Artificial** dentro de una institución, enfocado en procesar, analizar y resumir documentos oficiales para brindar asistencia a la ciudadanía y al personal interno.

## 🎯 Objetivo
Se espera que este proyecto sea capaz de leer los documentos y dar resúmenes de forma interna. Su enfoque principal está destinado a **atender consultas** y procesar datos que el personal externo (o ciudadanos) solicite, permitiendo buscar información de forma pertinente.

El agente procesa el mapa del sitio, la documentación subida, y brinda una ayuda precisa, guiada y transparente a la población mediante el uso de Modelos de Lenguaje Grandes (LLMs).

## 🚀 Características Principales

- **Búsqueda RAG (Retrieval-Augmented Generation)**: Extrae contexto de los documentos de la institución para fundamentar las respuestas del modelo de lenguaje, garantizando que la información no sea inventada (alucinaciones).
- **Procesamiento NLP Avanzado**: Utiliza librerías como `SpaCy` y embeddings de `HuggingFace` (`sentence-transformers/all-MiniLM-L6-v2`) para vectorizar los textos.
- **Pipeline de Ingesta**: Scripts dedicados como `inyeccion.py` y `extraerpdf.ipynb` para convertir los PDFs institucionales a bases de datos vectoriales.
- **Despliegue Contenerizado**: Totalmente preparado para entornos de producción utilizando `Docker` y `docker-compose.yml`.

## 🏗️ Arquitectura
El sistema consta de:
1. Una API principal (`api/main.py`).
2. Herramientas de RAG y búsquedas (`src/tools/buscador_rag.py`).
3. Interfaz ciudadana conectada al backend.

*(Actualización 20/05/2026: Arquitectura RAG desplegada, filtro de documentos institucionales funcionando, interfaz ciudadana terminada)*

## 🛠️ Tecnologías
- **Backend:** Python 3.11, FastAPI / Flask (según la configuración de despliegue).
- **AI / NLP:** LangChain, HuggingFaceEmbeddings, SpaCy.
- **Infraestructura:** Docker, Docker Compose.

---
**Design by:** Angel Hernández
- **Fecha de creación del repositorio:** 04 de Mayo de 2026
- **Fecha de actualización del README:** 18 de Agosto de 2026