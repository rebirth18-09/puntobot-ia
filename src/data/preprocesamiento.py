"""Una sola transformación de texto para entrenamiento e inferencia."""
import unicodedata

def normalizar_texto(texto):
    if not isinstance(texto, str):
        raise TypeError("El texto debe ser una cadena")
    return unicodedata.normalize("NFC", texto).lower().strip()

def normalizar_textos(textos):
    return [normalizar_texto(t) for t in textos]
