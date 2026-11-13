"""Clasificador elegido en la etapa 4, aún sin ajustar al construirlo."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from src.data.preprocesamiento import normalizar_texto

def crear_vectorizador(config):
    m = config["modelo"]
    return FeatureUnion([
        ("palabras", TfidfVectorizer(preprocessor=normalizar_texto, lowercase=False,
            ngram_range=tuple(m["ngramas_palabras"]), max_features=m["max_features"])),
        ("caracteres", TfidfVectorizer(preprocessor=normalizar_texto, lowercase=False,
            analyzer="char_wb", ngram_range=tuple(m["ngramas_caracteres"]), max_features=m["max_features"]))
    ])

def crear_modelo(config):
    m = config["modelo"]
    return Pipeline([
        ("tfidf", crear_vectorizador(config)),
        ("clasificador", LogisticRegression(C=m["C"], max_iter=m["max_iter"],
            class_weight=m["class_weight"], random_state=config["entrenamiento"]["semilla"]))
    ])
