"""
Carga del modelo, predicción y reentrenamiento versionado.

Diferencias con la versión anterior:

  - Un solo artefacto (`Pipeline`) en lugar de vectorizador y modelo sueltos.
    El pipeline recibe texto crudo, así que ya no hay preprocesamiento
    duplicado ni riesgo de que entrenamiento e inferencia difieran.

  - El reentrenamiento vuelve a ajustar el vectorizador. Antes se hacía
    `transform` sin `fit`, de modo que el vocabulario quedaba congelado y
    las palabras nuevas se descartaban en silencio.

  - Cada reentrenamiento crea una versión nueva y solo la promueve si supera
    a la actual en un holdout fijo. Antes se sobrescribía el archivo del
    modelo, sin forma de volver atrás si el resultado empeoraba.
"""

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from train import build_pipeline

log = logging.getLogger(__name__)

BASE = Path(__file__).parent
MODELS = BASE / "models"
DATA = BASE / "data"

_lock = threading.Lock()  # el reentrenamiento no debe solaparse consigo mismo
_model = None
_meta = None


def _load() -> None:
    global _model, _meta
    meta_path = MODELS / "current.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "No hay modelo entrenado. Ejecuta primero:  python train.py"
        )
    _meta = json.loads(meta_path.read_text())
    _model = joblib.load(MODELS / _meta["file"])
    log.info("Modelo cargado: %s (f1=%s)", _meta["file"], _meta["f1_macro_holdout"])


_load()


def model_info() -> dict:
    return {**_meta, "classes": [int(c) for c in _model.classes_]}


def predict_proba(texts: list[str]):
    """
    Predice la etiqueta de ODS de cada texto.

    Los textos entran en crudo: el pipeline los normaliza internamente.
    """
    predictions = _model.predict(texts)
    probabilities = _model.predict_proba(texts)
    return predictions, probabilities


def _evaluate(pipeline) -> dict:
    """Evalúa contra el holdout fijo, el mismo para todas las versiones."""
    holdout = pd.read_csv(DATA / "holdout.csv")
    y_true = holdout["sdg"]
    y_pred = pipeline.predict(holdout["Textos_espanol"].astype(str))
    return {
        "precision": float(precision_score(y_true, y_pred, average="macro")),
        "recall": float(recall_score(y_true, y_pred, average="macro")),
        "f1_score": float(f1_score(y_true, y_pred, average="macro")),
    }


def retrain_model(texts: list[str], labels: list[int]) -> dict:
    """
    Reentrena desde cero sobre los datos base más todo lo acumulado.

    Se reentrena por completo en lugar de usar `partial_fit` porque el
    vocabulario del TF-IDF también tiene que aprender los términos nuevos.
    Con un corpus de este tamaño el reentrenamiento tarda segundos.

    La versión nueva solo se activa si iguala o supera a la actual en el
    holdout. Si no, queda guardada en disco pero el servicio sigue
    respondiendo con la anterior.
    """
    global _model, _meta

    with _lock:
        # 1. Acumular los datos nuevos, para que no se pierdan en el siguiente ciclo
        incoming = pd.DataFrame({"Textos_espanol": texts, "sdg": labels})
        inc_path = DATA / "incremental.csv"
        if inc_path.exists():
            incoming = pd.concat([pd.read_csv(inc_path), incoming], ignore_index=True)
        incoming.to_csv(inc_path, index=False)

        # 2. Entrenar sobre base + acumulados (el holdout queda fuera, siempre)
        base = pd.read_csv(DATA / "train_base.csv")
        full = pd.concat([base, incoming], ignore_index=True).drop_duplicates(
            subset="Textos_espanol"
        )

        pipeline = build_pipeline()
        pipeline.fit(full["Textos_espanol"].astype(str), full["sdg"])

        # 3. Evaluar contra el holdout fijo
        metrics = _evaluate(pipeline)

        # 4. Guardar versionado y promover solo si no empeora
        version = _meta["version"] + 1
        filename = f"modelo_v{version}.joblib"
        joblib.dump(pipeline, MODELS / filename)

        # Se compara con la misma precisión con la que se guardó la métrica,
        # para que un empate no se lea como un empeoramiento.
        promoted = round(metrics["f1_score"], 4) >= _meta["f1_macro_holdout"]
        if promoted:
            _meta = {
                "version": version,
                "file": filename,
                "f1_macro_holdout": round(metrics["f1_score"], 4),
                "f1_macro_cv": None,
                "n_train": int(len(full)),
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            (MODELS / "current.json").write_text(json.dumps(_meta, indent=2))
            _model = pipeline
            log.info("Versión %s promovida (f1=%.4f)", version, metrics["f1_score"])
        else:
            log.warning(
                "Versión %s descartada: f1=%.4f < %.4f del modelo activo",
                version, metrics["f1_score"], _meta["f1_macro_holdout"],
            )

        return {
            **metrics,
            "version": version,
            "promoted": promoted,
            "active_version": _meta["version"],
            "n_train": int(len(full)),
        }
