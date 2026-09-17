"""
Entrena el clasificador de ODS y guarda la primera versión del modelo.

    python train.py --data data/ODScat_345.xlsx

Produce:
    models/modelo_v1.joblib   pipeline completo (TF-IDF + clasificador)
    models/current.json       qué versión está activa y con qué métricas
    data/holdout.csv          conjunto de evaluación fijo, nunca se entrena con él

El holdout fijo es lo que permite comparar versiones entre sí: sin un
conjunto de evaluación estable, "el reentrenamiento mejoró el modelo" no
significa nada.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import normalize_text

BASE = Path(__file__).parent
MODELS = BASE / "models"
DATA = BASE / "data"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    """
    La configuración ganadora, elegida por validación cruzada de 5 folds.
    Ver la tabla de resultados en el README.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=normalize_text,  # la limpieza vive dentro del pipeline
                ngram_range=(1, 2),           # bigramas: "salud mental", "igualdad de genero"
                min_df=2,                     # descarta términos que aparecen una sola vez
                sublinear_tf=True,            # log(tf), estándar en clasificación de texto
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def main(data_path: str) -> None:
    MODELS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)

    df = pd.read_excel(data_path)
    X, y = df["Textos_espanol"].astype(str), df["sdg"]
    print(f"Corpus: {len(df)} documentos | clases: {dict(y.value_counts())}")

    # Holdout fijo. Se guarda en disco para que cada reentrenamiento futuro
    # se evalúe exactamente contra los mismos documentos.
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pd.DataFrame({"Textos_espanol": X_hold, "sdg": y_hold}).to_csv(
        DATA / "holdout.csv", index=False
    )
    pd.DataFrame({"Textos_espanol": X_train, "sdg": y_train}).to_csv(
        DATA / "train_base.csv", index=False
    )

    pipe = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"\nValidación cruzada (5 folds): f1_macro = {scores.mean():.4f} ± {scores.std():.4f}")

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_hold)
    f1 = f1_score(y_hold, y_pred, average="macro")

    print(f"\nHoldout: f1_macro = {f1:.4f}\n")
    print(classification_report(y_hold, y_pred, digits=3))

    path = MODELS / "modelo_v1.joblib"
    joblib.dump(pipe, path)

    meta = {
        "version": 1,
        "file": path.name,
        "f1_macro_holdout": round(float(f1), 4),
        "f1_macro_cv": round(float(scores.mean()), 4),
        "n_train": int(len(X_train)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (MODELS / "current.json").write_text(json.dumps(meta, indent=2))
    print(f"\nGuardado: {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ODScat_345.xlsx")
    main(p.parse_args().data)
