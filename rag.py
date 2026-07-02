"""
rag.py — Núcleo del Asistente de Optimización Administrativa
y Orientación Normativa UNEFA Lara

Contiene:
- Búsqueda semántica de los fragmentos más relevantes ante una pregunta
- Generación de la respuesta final vía Groq, basada SOLO en esos fragmentos
"""

import os
import json
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GEMINI_KEY or not GROQ_KEY:
    raise SystemExit(
        "ERROR: Falta GEMINI_API_KEY o GROQ_API_KEY en el archivo .env"
    )

INDICE_DIR = "indice"
CHUNKS_PATH = os.path.join(INDICE_DIR, "chunks.json")
EMBEDDINGS_PATH = os.path.join(INDICE_DIR, "embeddings.npy")

if not os.path.exists(CHUNKS_PATH) or not os.path.exists(EMBEDDINGS_PATH):
    raise SystemExit(
        "ERROR: No se encontró el índice. Corre primero 'python indexar.py' "
        "para procesar los PDFs de la carpeta 'documentos/'."
    )

with open(CHUNKS_PATH, encoding="utf-8") as f:
    CHUNKS = json.load(f)
EMBEDDINGS = np.load(EMBEDDINGS_PATH)

# Prompt del sistema: define el rol institucional del asistente
SYSTEM_PROMPT = """Eres el Asistente de Optimización Administrativa y Orientación Normativa
de la UNEFA Núcleo Lara. Tu función es ayudar a estudiantes, profesores y personal
administrativo a entender y aplicar correctamente los reglamentos, normativas y
procedimientos institucionales, basándote EXCLUSIVAMENTE en los documentos oficiales
que se te proporcionan como contexto.

Reglas que debes seguir siempre:
1. Responde SOLO con base en el CONTEXTO proporcionado. No inventes artículos,
   procedimientos, plazos ni requisitos que no estén en el contexto.
2. Si la información solicitada no aparece en el contexto, dilo claramente:
   indica que no encontraste esa información en los documentos disponibles y
   recomienda consultar directamente con la unidad administrativa correspondiente.
3. Cuando cites una norma o procedimiento, menciona el documento fuente y,
   si está disponible, la página o artículo correspondiente.
4. Usa un tono formal, claro y orientado a resolver trámites administrativos
   de forma práctica (qué hacer, en qué orden, qué se necesita).
5. Si la pregunta es ambigua, pide la aclaración mínima necesaria antes de responder,
   o responde a la interpretación más probable y ofrece la alternativa.
"""


def embedding_consulta(texto):
    """Genera el embedding de la pregunta del usuario."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
    params = {"key": GEMINI_KEY}
    body = {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": texto}]},
        "task_type": "RETRIEVAL_QUERY",
    }
    r = requests.post(url, params=params, json=body, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["embedding"]["values"], dtype=np.float32)


def buscar_relevantes(pregunta, k=5):
    """Devuelve los k fragmentos más relevantes para la pregunta, por similitud coseno."""
    q_emb = embedding_consulta(pregunta)

    normas_docs = np.linalg.norm(EMBEDDINGS, axis=1)
    norma_q = np.linalg.norm(q_emb)
    similitudes = (EMBEDDINGS @ q_emb) / (normas_docs * norma_q + 1e-8)

    top_indices = np.argsort(similitudes)[::-1][:k]
    resultados = []
    for idx in top_indices:
        item = dict(CHUNKS[idx])
        item["similitud"] = float(similitudes[idx])
        resultados.append(item)
    return resultados


def generar_respuesta(pregunta, contexto_chunks):
    """Genera la respuesta final vía Groq usando solo el contexto recuperado."""
    contexto = "\n\n---\n\n".join(
        f"[Documento: {c['fuente']} | Fragmento {c['chunk_id']}]\n{c['texto']}"
        for c in contexto_chunks
    )

    prompt_usuario = f"""CONTEXTO (extraído de documentos oficiales de UNEFA):

{contexto}

PREGUNTA DEL USUARIO: {pregunta}

Responde siguiendo las reglas del sistema. Cita el documento fuente cuando corresponda."""

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_usuario},
        ],
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def responder(pregunta, k=5):
    """Función principal: busca contexto relevante y genera la respuesta."""
    relevantes = buscar_relevantes(pregunta, k=k)
    respuesta = generar_respuesta(pregunta, relevantes)
    return respuesta, relevantes
