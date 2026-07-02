"""
Script de indexación — Asistente de Optimización Administrativa
y Orientación Normativa UNEFA Lara

Este script:
1. Lee todos los PDFs de la carpeta 'documentos/'
2. Extrae el texto
3. Lo divide en fragmentos (chunks)
4. Genera el embedding de cada fragmento vía Gemini API
5. Guarda todo en la carpeta 'indice/' para que rag.py lo use después

Se corre UNA SOLA VEZ, y cada vez que agregues, quites o cambies un PDF
en la carpeta 'documentos/'.
"""

import os
import io
import json
import time
import base64
import fitz  # PyMuPDF
import requests
import numpy as np
from dotenv import load_dotenv

# --- Nota: pytesseract ya no es obligatorio con este método de OCR vía Gemini,
# pero lo dejamos disponible por si quieres volver al OCR local más adelante.
# import pytesseract
# from PIL import Image

# Si vuelves a usar Tesseract localmente, descomenta la línea de arriba y esta:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

MIN_CARACTERES_PARA_CONSIDERAR_TEXTO = 20

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise SystemExit(
        "ERROR: No se encontró GEMINI_API_KEY en el archivo .env. "
        "Revisa que el archivo .env exista y tenga la línea GEMINI_API_KEY=..."
    )

DOCS_DIR = "documentos"
OUT_DIR = "indice"
CHUNK_SIZE = 800       # caracteres por fragmento
CHUNK_OVERLAP = 150    # solapamiento entre fragmentos para no cortar ideas a la mitad

os.makedirs(OUT_DIR, exist_ok=True)


def ocr_pagina(pagina, zoom=2, reintentos=3):
    """Convierte una página del PDF en imagen y le pide a Gemini que transcriba
    el texto exactamente (alternativa a Tesseract, más precisa en documentos
    institucionales/legales en español)."""
    matriz = fitz.Matrix(zoom, zoom)
    pix = pagina.get_pixmap(matrix=matriz)
    img_bytes = pix.tobytes("png")
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_KEY}
    body = {
        "contents": [{
            "parts": [
                {"text": "Transcribe EXACTAMENTE todo el texto visible en esta imagen "
                          "(documento institucional en español). No traduzcas, no "
                          "resumas, no agregues comentarios. Si hay tablas, "
                          "transcribe fila por fila. Si no hay texto legible, "
                          "responde solo con: [SIN TEXTO]"},
                {"inline_data": {"mime_type": "image/png", "data": img_b64}}
            ]
        }]
    }

    for intento in range(reintentos):
        try:
            r = requests.post(url, headers=headers, params=params, json=body, timeout=60)
            if r.status_code == 429:
                espera = 20 * (intento + 1)
                print(f"      Límite alcanzado en OCR, esperando {espera}s...")
                time.sleep(espera)
                continue
            r.raise_for_status()
            data = r.json()
            texto = data["candidates"][0]["content"]["parts"][0]["text"]
            return "" if "[SIN TEXTO]" in texto else texto
        except (requests.exceptions.RequestException, KeyError, IndexError) as e:
            print(f"      (reintento OCR {intento + 1}/{reintentos} por error: {e})")
            time.sleep(5)
    print("      No se pudo hacer OCR de esta página, se deja vacía.")
    return ""


def extraer_texto(ruta_pdf):
    """Extrae el texto completo de un PDF, página por página.
    Si una página no tiene texto extraíble (PDF escaneado/imagen),
    usa OCR automáticamente como respaldo."""
    doc = fitz.open(ruta_pdf)
    texto_completo = ""
    paginas_con_ocr = 0

    for num_pagina, pagina in enumerate(doc, start=1):
        texto_pagina = pagina.get_text().strip()

        if len(texto_pagina) < MIN_CARACTERES_PARA_CONSIDERAR_TEXTO:
            print(f"    Página {num_pagina} sin texto extraíble, aplicando OCR...")
            texto_pagina = ocr_pagina(pagina)
            paginas_con_ocr += 1

        texto_completo += f"\n[Página {num_pagina}]\n{texto_pagina}"

    doc.close()

    if paginas_con_ocr:
        print(f"    ({paginas_con_ocr} página(s) procesada(s) con OCR)")

    return texto_completo


def dividir_en_chunks(texto, tam=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide un texto largo en fragmentos con solapamiento."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tam
        fragmento = texto[inicio:fin].strip()
        if fragmento:
            chunks.append(fragmento)
        inicio += tam - overlap
    return chunks


BATCH_SIZE = 25  # fragmentos por solicitud a la API (menos solicitudes = menos riesgo de 429)


def obtener_embeddings_lote(textos, reintentos=5):
    """Obtiene los embeddings de una LISTA de textos en una sola llamada a la API,
    usando el endpoint de lote de Gemini. Reduce drásticamente el número de
    solicitudes comparado con pedir un embedding a la vez."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents"
    headers = {"Content-Type": "application/json"}
    params = {"key": GEMINI_KEY}
    body = {
        "requests": [
            {
                "model": "models/gemini-embedding-001",
                "content": {"parts": [{"text": t}]},
                "task_type": "RETRIEVAL_DOCUMENT",
            }
            for t in textos
        ]
    }

    for intento in range(reintentos):
        try:
            r = requests.post(url, headers=headers, params=params, json=body, timeout=60)
            if r.status_code == 429:
                espera = 20 * (intento + 1)
                print(f"    Límite de frecuencia alcanzado, esperando {espera}s antes de reintentar...")
                time.sleep(espera)
                continue
            r.raise_for_status()
            data = r.json()
            return [e["values"] for e in data["embeddings"]]
        except requests.exceptions.RequestException as e:
            print(f"    (reintento {intento + 1}/{reintentos} por error: {e})")
            time.sleep(5)
    raise RuntimeError("No se pudo obtener el lote de embeddings tras varios intentos.")


def main():
    if not os.path.isdir(DOCS_DIR):
        raise SystemExit(f"ERROR: No existe la carpeta '{DOCS_DIR}'. Créala y coloca ahí tus PDFs.")

    archivos_pdf = [f for f in os.listdir(DOCS_DIR) if f.lower().endswith(".pdf")]

    if not archivos_pdf:
        raise SystemExit(f"No se encontró ningún PDF dentro de '{DOCS_DIR}/'. Copia tus reglamentos ahí.")

    print(f"Se encontraron {len(archivos_pdf)} PDF(s): {', '.join(archivos_pdf)}")
    print()

    todos_chunks = []
    for archivo in archivos_pdf:
        ruta = os.path.join(DOCS_DIR, archivo)
        print(f"Procesando: {archivo}")
        texto = extraer_texto(ruta)

        if len(texto.strip()) < 50:
            print(f"  AVISO: '{archivo}' parece no tener texto extraíble (¿es un PDF escaneado/imagen?). Se omite.")
            continue

        chunks = dividir_en_chunks(texto)
        print(f"  -> {len(chunks)} fragmentos generados")

        for i, c in enumerate(chunks):
            todos_chunks.append({
                "texto": c,
                "fuente": archivo,
                "chunk_id": i,
            })

    if not todos_chunks:
        raise SystemExit("No se generó ningún fragmento de texto. Revisa tus PDFs.")

    print(f"\nTotal de fragmentos a indexar: {len(todos_chunks)}")
    print("Generando embeddings (esto usa la API de Gemini, puede tardar unos minutos)...\n")

    vectores = []
    total = len(todos_chunks)
    for inicio in range(0, total, BATCH_SIZE):
        lote = todos_chunks[inicio:inicio + BATCH_SIZE]
        textos_lote = [ch["texto"] for ch in lote]
        fin = min(inicio + BATCH_SIZE, total)
        print(f"  Embeddings {inicio + 1}-{fin} de {total}...")
        embs_lote = obtener_embeddings_lote(textos_lote)
        vectores.extend(embs_lote)
        time.sleep(2)  # pausa breve entre lotes

    matriz = np.array(vectores, dtype=np.float32)
    np.save(os.path.join(OUT_DIR, "embeddings.npy"), matriz)

    with open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(todos_chunks, f, ensure_ascii=False, indent=2)

    print("\n✅ Índice creado correctamente en la carpeta 'indice/'.")
    print(f"   - {len(todos_chunks)} fragmentos indexados de {len(archivos_pdf)} documento(s).")
    print("   Ya puedes correr: streamlit run app.py")


if __name__ == "__main__":
    main()
