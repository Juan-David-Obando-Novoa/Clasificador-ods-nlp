"""
API de clasificación de opiniones por Objetivo de Desarrollo Sostenible (ODS 3, 4, 5).

    uvicorn main:app --reload
    Documentación interactiva: http://localhost:8000/docs
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from DataModel import PredictionInput, RetrainInput
from Utils import model_info, predict_proba, retrain_model

logging.basicConfig(
    filename="model_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(
    title="Clasificador de opiniones por ODS",
    description="Clasifica texto en español en los ODS 3 (salud), 4 (educación) y 5 (igualdad de género).",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Estado del servicio y versión del modelo activo."""
    return {"status": "ok", "model": model_info()}


# El frontend llama /predict/ y /retrain (con y sin barra final).
# Registrar ambas rutas evita el redirect 307 innecesario.
@app.post("/predict/")
@app.post("/predict")
async def predict(input_data: PredictionInput):
    """Clasifica una lista de textos. Devuelve la etiqueta y las probabilidades por clase."""
    if not input_data.texts:
        raise HTTPException(status_code=422, detail="La lista de textos está vacía.")

    log.info("Predicción solicitada para %d textos", len(input_data.texts))
    try:
        # El pipeline normaliza el texto internamente: no se preprocesa aquí.
        predictions, probabilities = predict_proba(input_data.texts)
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "classes": model_info()["classes"],
            "model_version": model_info()["version"],
        }
    except Exception as exc:
        log.exception("Error en predicción")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/retrain/")
@app.post("/retrain")
async def retrain(input_data: RetrainInput):
    """
    Reentrena el modelo con ejemplos etiquetados nuevos.

    La versión resultante solo se activa si iguala o supera al modelo actual
    en el holdout fijo. La respuesta incluye `promoted` para indicarlo.
    """
    if len(input_data.texts) != len(input_data.labels):
        raise HTTPException(
            status_code=422,
            detail=f"Los textos ({len(input_data.texts)}) y las etiquetas "
                   f"({len(input_data.labels)}) no coinciden en cantidad.",
        )
    if not input_data.texts:
        raise HTTPException(status_code=422, detail="No se recibieron ejemplos.")

    log.info("Reentrenamiento solicitado con %d ejemplos", len(input_data.texts))
    try:
        return retrain_model(input_data.texts, input_data.labels)
    except Exception as exc:
        log.exception("Error en reentrenamiento")
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
