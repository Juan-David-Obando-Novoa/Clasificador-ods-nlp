# Clasificador de opiniones por ODS

Clasifica texto en español en tres Objetivos de Desarrollo Sostenible: **ODS 3** (salud y bienestar), **ODS 4** (educación de calidad) y **ODS 5** (igualdad de género).

Desarrollado para la materia *ISIS3301 Inteligencia de Negocios*, Ingeniería de Sistemas y Computación, Universidad de los Andes.

**Autores:** Miguel Angel Ariza Jimenez · Julian Escobar Rivera · Juan David Obando Novoa

| | |
|---|---|
| Corpus | 4.049 textos en español (ODS 3: 1.244 · ODS 4: 1.354 · ODS 5: 1.451) |
| Modelo | TF-IDF (1-2 gramas) + regresión logística, en un único `Pipeline` de scikit-learn |
| F1 macro | **0.980** en holdout · 0.981 ± 0.005 en validación cruzada de 5 folds |
| Stack | FastAPI · scikit-learn · React |

---

## Cómo ejecutar el proyecto

### Backend (API)

1. **Ubicarse en la carpeta `api`:**

   ```bash
   cd etapa2/api
   ```

2. **Crear un entorno virtual:**

   ```bash
   python -m venv .venv
   ```

3. **Activar el entorno virtual:**

   - **Windows:**

     ```bash
     .venv\Scripts\activate
     ```

   - **macOS/Linux:**

     ```bash
     source .venv/bin/activate
     ```

4. **Instalar las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Entrenar el modelo** *(solo la primera vez)*:

   ```bash
   python train.py --data data/ODScat_345.xlsx
   ```

   Genera `models/modelo_v1.joblib`, el conjunto de evaluación fijo `data/holdout.csv` e imprime las métricas. Si el repositorio ya trae un modelo entrenado, este paso se puede omitir.

6. **Ejecutar el servidor:**

   ```bash
   uvicorn main:app --reload
   ```

   Disponible en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

### Frontend (Web)

1. **Ubicarse en la carpeta `web`:**

   ```bash
   cd etapa2/web
   ```

2. **Instalar dependencias:**

   ```bash
   npm install
   ```

3. **Iniciar la aplicación:**

   ```bash
   npm start
   ```

   Disponible en `http://localhost:3000`.

---

## Endpoints

### `GET /health`

Estado del servicio y versión del modelo activo.

```json
{"status": "ok", "model": {"version": 1, "f1_macro_holdout": 0.9804, "classes": [3, 4, 5]}}
```

### `POST /predict/`

```json
{"texts": ["El acceso a la educación primaria aumentó en zonas rurales."]}
```

```json
{
  "predictions": [4],
  "probabilities": [[0.11, 0.76, 0.13]],
  "classes": [3, 4, 5],
  "model_version": 1
}
```

Los textos se envían **en crudo**: la normalización ocurre dentro del pipeline.

### `POST /retrain/`

```json
{"texts": ["La cobertura de vacunación infantil llegó al 95%."], "labels": [3]}
```

```json
{
  "precision": 0.981, "recall": 0.980, "f1_score": 0.980,
  "version": 2, "promoted": true, "active_version": 2, "n_train": 3240
}
```

`promoted` indica si la versión nueva reemplazó a la anterior. Si no supera al modelo activo en el holdout, queda guardada en disco pero el servicio sigue respondiendo con la versión anterior.

---

## Versión 2: qué cambió y por qué

La primera versión usaba TF-IDF limitado a 1.000 términos con Naive Bayes multinomial. Para decidir qué valía la pena cambiar, cada modificación se midió por separado con validación cruzada estratificada de 5 folds, en lugar de aplicarlas todas a la vez.

### Resultados

| Configuración | F1 macro | Δ |
|---|---|---|
| Original: 1.000 features + Naive Bayes | 0.9641 | — |
| Sin límite de `max_features` | 0.9696 | +0.55 |
| \+ bigramas, `min_df=2`, `sublinear_tf` | 0.9706 | +0.10 |
| **\+ clasificador lineal en vez de Naive Bayes** | **0.9814** | **+1.08** |

**La tasa de error baja de 3.6% a 1.9%: se reduce prácticamente a la mitad.**

El hallazgo que no era evidente de antemano: el trabajo sobre las características aporta medio punto, mientras que cambiar el modelo aporta el doble. Naive Bayes asume independencia entre palabras, lo cual es falso en texto; con 4.000 documentos ese sesgo ya estorba más de lo que ayuda.

### Cambios que se probaron y se descartaron

Se documentan porque el resultado fue contrario a lo esperado:

- **Stemming en español** (`SnowballStemmer`): **empeora** el modelo en 0.3 puntos. Colapsa palabras que sí distinguían clases entre sí.
- **Eliminar stopwords**: irrelevante. Quitar el filtro incluso rinde marginalmente mejor, así que se eliminó una dependencia de NLTK que no aportaba nada.
- **Corregir el mojibake con `ftfy` en lugar de la tabla de reemplazos**: 0.05 puntos de diferencia, dentro del ruido. Se conservó por corrección del texto, no por exactitud.

### Configuración final

```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        preprocessor=normalize_text,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced')),
])
```

Se eligió regresión logística sobre `LinearSVC` a pesar de que este último saca una centésima más: la API expone probabilidades por clase y `LinearSVC` no implementa `predict_proba`. La diferencia es ruido estadístico; la funcionalidad no.

---

## Correcciones de ingeniería

Además de la exactitud, se corrigieron cuatro defectos del código anterior.

**1. Preprocesamiento duplicado.** `main.py` llamaba `preprocess_text()` y después `predict_proba()`, que volvía a preprocesar el mismo texto. Ahora la normalización vive dentro del `TfidfVectorizer` (parámetro `preprocessor`), en un único módulo `preprocessing.py`. El pipeline recibe texto crudo, así que es imposible limpiarlo dos veces o que entrenamiento e inferencia difieran.

**2. El reentrenamiento descartaba vocabulario nuevo.** El endpoint hacía `tfidf_vectorizer.transform(...)` sin volver a ajustar el vectorizador: el vocabulario quedaba congelado en el del entrenamiento original y las palabras nuevas se perdían en silencio. Ahora se reajusta el pipeline completo sobre los datos base más los acumulados.

**3. No había forma de revertir un reentrenamiento.** Se hacía `joblib.dump()` sobre el mismo archivo, así que un reentrenamiento malo destruía el modelo bueno de forma irreversible. Ahora cada versión se guarda aparte (`modelo_vN.joblib`), se evalúa contra un holdout fijo y solo se promueve si no empeora. `models/current.json` registra cuál está activa.

**4. Las métricas de reentrenamiento no eran comparables.** Se calculaban sobre una partición del propio lote de reentrenamiento, distinta en cada llamada, de modo que dos ejecuciones no se podían comparar entre sí. Ahora todas las versiones se evalúan contra el mismo `data/holdout.csv`, que nunca entra al entrenamiento.

**5. Las tildes rompían la predicción en entradas escritas a mano.** El corpus contiene *género* y *educación* con tilde, así que una consulta escrita como `genero` o `educacion` no coincidía con ningún término del vocabulario: el vector TF-IDF salía vacío y el modelo respondía con su sesgo por defecto (ODS 3) en lugar de con una predicción real. `género` daba ODS 5 con probabilidad 1.00 y `genero` daba ODS 3. La normalización ahora elimina tildes —conservando la eñe, que es una letra y no un acento— de modo que ambas formas colapsan en el mismo término. Medido, el F1 no cambia: las tildes no aportaban capacidad discriminativa, solo fragilidad frente a lo que escribe un usuario.

**6. La interfaz reventaba al agregar una opinión después de predecir.** `PredictionResults` paginaba sobre la lista de opiniones, pero las predicciones seguían siendo las del lote anterior; al llegar a un índice sin predicción, `undefined.map()` tumbaba la página. Ahora agregar o borrar una opinión descarta los resultados anteriores, y el componente solo renderiza las opiniones que tienen predicción. Las etiquetas de la barra de probabilidades se toman de `results.classes` en vez de estar fijas como `i + 3`.

También: el orden de limpieza estaba invertido (se eliminaba la puntuación antes de reparar el mojibake, partiendo secuencias que ya no se podían reemplazar), la tabla de reemplazos convertía *ñ* en *n* mientras restauraba las demás tildes, y `LinearSVC` sustituye a `SVC(kernel='linear')`, que es órdenes de magnitud más lento sobre el mismo problema.

En el notebook de la etapa 1, el `fit_transform` del vectorizador se hacía sobre el dataset completo antes de dividir en entrenamiento y prueba, lo que filtra información del conjunto de prueba al vocabulario y a los pesos IDF. Medido, infla el resultado apenas 0.12 puntos, pero es incorrecto: en el código nuevo el vectorizador se ajusta dentro del `Pipeline`, de modo que la validación cruzada lo reajusta en cada fold.

---

## Estructura

```
etapa1/
  Proyecto1_Etapa1.ipynb      análisis exploratorio y comparación de modelos
etapa2/
  api/
    preprocessing.py          normalización de texto (única definición)
    train.py                  entrenamiento inicial y creación del holdout
    Utils.py                  carga del modelo, predicción, reentrenamiento versionado
    main.py                   endpoints de FastAPI
    DataModel.py              esquemas de entrada y validación
    models/                   modelos versionados + current.json
    data/                     dataset, holdout fijo y datos acumulados
  web/                        interfaz en React
```

## Notas

- El dataset viene con mojibake (`nÃºmero` en lugar de `número`): se guardó en UTF-8 y se leyó como Latin-1. `preprocessing.fix_encoding()` lo repara antes de cualquier otra limpieza.
- Un modelo serializado con `joblib` no es portable entre versiones de scikit-learn. Si al arrancar aparece un `AttributeError` sobre un atributo que el estimador no tiene, basta con volver a ejecutar `python train.py` en el entorno local.
- `data/` incluye el dataset para que el proyecto sea reproducible de principio a fin.
