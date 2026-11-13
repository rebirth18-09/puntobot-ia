"""Métricas de intención, extracción y decisión de PuntoBot."""
from difflib import SequenceMatcher

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score, log_loss,
)

from src.data.preprocesamiento import normalizar_texto
from src.inference.predictor import candidatos_producto


def medir_intencion(y, pred, clases, probabilidades=None, orden_prob=None):
    reporte = classification_report(y, pred, labels=clases, output_dict=True, zero_division=0)
    return {
        "exactitud": float(accuracy_score(y, pred)),
        "f1_macro": float(f1_score(y, pred, labels=clases, average="macro", zero_division=0)),
        "perdida": None if probabilidades is None else float(log_loss(y, probabilidades, labels=orden_prob)),
        "por_clase": {c: reporte[c] for c in clases},
        "matriz": confusion_matrix(y, pred, labels=clases).tolist(),
    }


def predecir_filas(df, config, modelo, catalogo):
    columnas = config["datos"]
    textos = df[columnas["columna_texto"]].tolist()
    probs = modelo.predict_proba(textos)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1):
        raise ValueError("El modelo produjo probabilidades inválidas")
    filas = []
    for (_, fila), texto, p in zip(df.iterrows(), textos, probs):
        mejor = int(p.argmax())
        candidatos = candidatos_producto(texto, catalogo)
        candidato = candidatos[0] if candidatos else {"nombre": "", "similitud": 0.0}
        filas.append({
            "id": int(fila[columnas["columna_id"]]),
            "texto": texto,
            "intencion_real": fila[columnas["columna_clase"]],
            "intencion_predicha": str(modelo.classes_[mejor]),
            "confianza": float(p[mejor]),
            "producto_real": normalizar_texto(fila[columnas["columna_producto"]]),
            "producto_candidato": normalizar_texto(candidato["nombre"]),
            "similitud": candidato["similitud"],
        })
    return filas, probs


def decidir(intencion, confianza, similitud, umbral_intencion, umbral_producto):
    if intencion == "otra" or confianza < umbral_intencion:
        return "reformular"
    if similitud < umbral_producto:
        return "aclarar_producto"
    return "consultar"


def aplicar_regla(filas, umbral_intencion, umbral_producto):
    detalladas = []
    for fila in filas:
        accion = decidir(fila["intencion_predicha"], fila["confianza"], fila["similitud"],
                         umbral_intencion, umbral_producto)
        producto = fila["producto_candidato"] if fila["similitud"] >= umbral_producto else ""
        correcta = (fila["intencion_real"] != "otra"
                    and fila["intencion_real"] == fila["intencion_predicha"]
                    and bool(fila["producto_real"]) and fila["producto_real"] == producto)
        detalladas.append({**fila, "producto_predicho": producto, "accion": accion,
                           "consulta_correcta": correcta if accion == "consultar" else None})
    return detalladas


def medir_regla(filas):
    n = len(filas)
    autorizadas = sum(f["accion"] == "consultar" for f in filas)
    correctas = sum(f["consulta_correcta"] is True for f in filas)
    elegibles = sum(f["intencion_real"] != "otra" and bool(f["producto_real"]) for f in filas)
    return {
        "filas": n, "consultas_elegibles": elegibles, "autorizadas": autorizadas,
        "correctas": correctas, "incorrectas": autorizadas - correctas,
        "reformular": sum(f["accion"] == "reformular" for f in filas),
        "aclarar_producto": sum(f["accion"] == "aclarar_producto" for f in filas),
        "cobertura": autorizadas / n,
        "precision_autorizadas": correctas / autorizadas if autorizadas else None,
        "resolucion_elegibles": correctas / elegibles if elegibles else None,
    }


def medir_producto(filas):
    tp = sum(bool(f["producto_real"]) and f["producto_real"] == f["producto_predicho"] for f in filas)
    fp = sum(bool(f["producto_predicho"]) and f["producto_real"] != f["producto_predicho"] for f in filas)
    fn = sum(bool(f["producto_real"]) and f["producto_real"] != f["producto_predicho"] for f in filas)
    return {
        "aciertos": tp, "falsos_positivos": fp, "falsos_negativos": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def auditar_similitud(filas, textos_train):
    """Semejanza de caracteres, diagnóstico descriptivo posterior."""
    corpus = [normalizar_texto(t) for t in textos_train]
    parecidos = [
        max(SequenceMatcher(None, normalizar_texto(f["texto"]), t).ratio() for t in corpus)
        for f in filas
    ]
    return {"umbral": 0.80, "filas": len(filas), "sobre_umbral": sum(p >= 0.80 for p in parecidos)}
