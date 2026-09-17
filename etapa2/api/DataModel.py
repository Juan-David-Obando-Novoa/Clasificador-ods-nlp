from typing import List, Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Textos en español a clasificar."""

    texts: List[str] = Field(..., min_length=1, examples=[["El acceso a la educación primaria aumentó un 12%."]])


class RetrainInput(BaseModel):
    """
    Ejemplos etiquetados para reentrenar.

    `labels` está restringido a las clases conocidas: una etiqueta fuera de
    {3, 4, 5} se rechaza en la validación en lugar de corromper el modelo.
    """

    texts: List[str] = Field(..., min_length=1)
    labels: List[Literal[3, 4, 5]] = Field(..., min_length=1)
