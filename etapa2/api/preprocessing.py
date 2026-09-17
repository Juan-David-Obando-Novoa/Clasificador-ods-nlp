"""
Normalización de texto en español.

Este módulo es la ÚNICA definición del preprocesamiento en todo el proyecto.
Se inyecta dentro del TfidfVectorizer (parámetro `preprocessor`), de modo que
el pipeline serializado recibe texto crudo y lo limpia él mismo.

Consecuencia: es imposible que el entrenamiento y la inferencia limpien el
texto de forma distinta, y es imposible limpiar dos veces el mismo texto.
"""

import re
import string
import unicodedata

try:
    import ftfy

    _HAS_FTFY = True
except ImportError:  # pragma: no cover
    _HAS_FTFY = False


# Respaldo cuando ftfy no está instalado. El dataset original viene con
# mojibake: se guardó en UTF-8 y se leyó como Latin-1, así que 'número'
# aparece como 'nÃºmero'. ftfy revierte eso de forma general; esta tabla
# solo cubre los casos vistos en este corpus.
_MOJIBAKE = {
    "Ãº": "ú", "Ã±": "ñ", "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó",
    "Ã¼": "ü", "Ã¤": "ä", "Ã«": "ë", "Ã¯": "ï", "Ã§": "ç", "Ã“": "Ó",
    "Ã‘": "Ñ", "Ã\x81": "Á", "Ã‰": "É", "Ã\x8d": "Í", "Ãš": "Ú",
}

_PUNCT = re.compile(f"[{re.escape(string.punctuation)}]")
_DIGITS = re.compile(r"\d+")
_SPACES = re.compile(r"\s+")


def fix_encoding(text: str) -> str:
    """Repara el mojibake. Debe correr ANTES de cualquier otra limpieza."""
    if _HAS_FTFY:
        return ftfy.fix_text(text)
    for bad, good in _MOJIBAKE.items():
        text = text.replace(bad, good)
    return text


def strip_accents(text: str) -> str:
    """
    Elimina tildes conservando la eñe.

    Sin esto, "genero" escrito por un usuario no coincide con "género" del
    corpus: el término queda fuera del vocabulario, el vector TF-IDF sale
    vacío y el modelo responde con su sesgo por defecto en lugar de con una
    predicción real. La eñe se preserva porque es una letra distinta, no un
    acento: "año" y "ano" no son la misma palabra.
    """
    text = text.replace("ñ", "\x00").replace("Ñ", "\x01")
    text = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return text.replace("\x00", "ñ").replace("\x01", "Ñ")


def normalize_text(text: str) -> str:
    """
    Pipeline de normalización completo.

    mojibake → minúsculas → sin tildes → sin puntuación ni dígitos.

    Al vivir dentro del TfidfVectorizer, se aplica igual al entrenar que al
    predecir, así que el vocabulario y la consulta siempre coinciden.
    """
    text = fix_encoding(str(text))
    text = text.lower()
    text = strip_accents(text)
    text = _PUNCT.sub(" ", text)
    text = _DIGITS.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
