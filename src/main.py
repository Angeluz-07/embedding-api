from pydantic import BaseModel
from typing import List
from pathlib import Path
from fastembed import TextEmbedding
from fastapi import FastAPI, HTTPException

# 🛠️ Apuntamos a la carpeta .data en el top-level de tu app
DATA_DIR = Path(__file__).resolve().parent.parent / ".data"
CACHE_DIR = str(DATA_DIR / "fastembed_cache")

# El modelo nativo equivalente y optimizado en fastembed
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Flujo simple: fastembed busca en el cache_dir de forma automática.
# Si el Dockerfile empaquetó los pesos ahí, los levanta en milisegundos con ONNX Runtime.
# Si no existiera la carpeta (ej. local nuevo), descarga el .onnx plano de internet de inmediato.
print(f"🔍 Inicializando TextEmbedding usando el caché en: {CACHE_DIR}")
try:
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=CACHE_DIR)
    print(f"Model loaded successfully: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Error crítico cargando el modelo: {e}")
    raise e

app = FastAPI()


class TextRequest(BaseModel):
    text: str


class TextBatchRequest(BaseModel):
    texts: List[str]


@app.post("/embed")
def get_embedding(request: TextRequest):
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="El texto no puede estar vacío")

        # .embed() espera un iterable y devuelve un generador de arrays de NumPy
        generator = model.embed([request.text])
        vector = list(generator)[0].tolist()
        return {"vector": vector}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed_batch")
def get_embeddings(request: TextBatchRequest):
    try:
        if not request.texts:
            return {"vectors": []}

        # 🚀 Forzamos a fastembed a procesar la lista completa como un lote único continuo
        # Pasamos de forma explícita todos los elementos a una lista nativa primero
        embeddings_generator = model.embed(request.texts)

        # Convertimos cada vector individual (Numpy Array) a lista de floats nativos
        vectors = [vector.tolist() for vector in embeddings_generator]

        # Validación de seguridad para tu tranquilidad en los logs
        print(
            f"📦 Batch procesado: Recibidos {len(request.texts)} textos -> Generados {len(vectors)} vectores."
        )

        return {"vectors": vectors}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en procesamiento por lote: {str(e)}"
        )


@app.get("/dimension")
async def get_dimension():
    # Retornamos la dimensión fija (MiniLM-L6-v2 siempre es 384)
    # Útil para inicializar tu base de datos de vectores Qdrant
    return {"dimension": 384}
