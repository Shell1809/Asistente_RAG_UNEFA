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

# 1. ACTUALIZACIÓN DEL PROMPT PARA SER ESTRICTO
SYSTEM_PROMPT = """Eres el Asistente de Optimización Administrativa de la UNEFA Núcleo Lara.

REGLAS OBLIGATORIAS:
1. Evalúa si la pregunta tiene respuesta clara en el CONTEXTO proporcionado.
2. Si la respuesta es SÍ, comienza directamente con "SÍ" y explica detalladamente citando el documento y artículo.
3. Si la respuesta es NO, o si el contexto NO contiene la información, debes responder obligatoriamente: "NO dispongo de esa información específica en los reglamentos actuales".
4. EN TODOS LOS CASOS, finaliza tu respuesta con esta frase exacta: "Por favor, para proceder con este trámite o aclarar dudas adicionales, dirígete a la Secretaría o a la Coordinación de tu carrera correspondiente."
5. Usa un tono formal y administrativo.
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
def generar_respuesta(pregunta, contexto_chunks):
    """Genera la respuesta final vía Groq usando solo el contexto recuperado."""
    contexto = "\n\n---\n\n".join(
        f"[Documento: {c['fuente']} | Fragmento {c['chunk_id']}]\n{c['texto']}"
        for c in contexto_chunks
    )

    # 2. LÓGICA DE REFUERZO EN EL PROMPT
    prompt_usuario = f"""CONTEXTO (extraído de documentos oficiales de UNEFA):
{contexto}

PREGUNTA DEL USUARIO: {pregunta}

Responde siguiendo estrictamente las REGLAS del sistema. Si la información no está en el CONTEXTO, no intentes inventarla."""

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
        "temperature": 0.1, # 3. TEMPERATURA BAJA PARA RESPUESTAS DETERMINÍSTICAS
    }
    r = requests.post(url, headers=headers, json=body, timeout=60)
    r.raise_for_status()
    
    respuesta = r.json()["choices"][0]["message"]["content"]
    
    # 4. LOGICA DE CONTROL EXTRA (Si el modelo ignora el "NO", lo forzamos)
    # Si detectamos que no hay contexto suficiente, podemos pre-procesar, 
    # pero con el nuevo SYSTEM_PROMPT el modelo Llama 3.3 es muy bueno siguiendo instrucciones.
    
    return respuesta
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
