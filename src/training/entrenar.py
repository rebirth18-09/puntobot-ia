"""Ajuste con train, selección y decisión con val, prueba final separada."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier

from src.data.division import cargar_particiones
from src.evaluation.evaluar import (
    aplicar_regla, auditar_similitud, decidir, medir_intencion, medir_producto,
    medir_regla, predecir_filas,
)
from src.inference.predictor import predecir
from src.models.modelo import crear_modelo
from src.models.pesos import calcular_pesos

RAIZ = Path(__file__).resolve().parents[2]


def cargar_config():
    # config.yaml usa la sintaxis JSON admitida por YAML 1.2.
    return json.loads((RAIZ / "config.yaml").read_text("utf-8"))


def escribir_json(ruta, datos):
    temporal = ruta.with_suffix(".tmp")
    temporal.write_text(json.dumps(datos, ensure_ascii=False, indent=2, allow_nan=False) + "\n", "utf-8")
    temporal.replace(ruta)


def opciones(config, experimento):
    cfg = deepcopy(config)
    cfg["modelo"].update(C=experimento["C"], class_weight=experimento["class_weight"])
    return cfg


def ajustar(train, config):
    d = config["datos"]
    return crear_modelo(config).fit(train[d["columna_texto"]], train[d["columna_clase"]])


def medidas(modelo, frame, config):
    d = config["datos"]
    probs = modelo.predict_proba(frame[d["columna_texto"]])
    pred = modelo.classes_[probs.argmax(axis=1)]
    return medir_intencion(frame[d["columna_clase"]], pred, config["clases"], probs, modelo.classes_)


def curva(train, val, config, directorio):
    d = config["datos"]
    semilla = config["entrenamiento"]["semilla"]
    grupos = [g.sample(frac=1, random_state=semilla) for _, g in train.groupby(d["columna_clase"])]
    puntos = []
    for fraccion in config["ajuste_fino"]["fracciones_curva"]:
        muestra = pd.concat([g.iloc[:max(1, round(len(g) * fraccion))] for g in grupos])
        modelo = ajustar(muestra, config)
        puntos.append({
            "n": len(muestra),
            "perdida_train": medidas(modelo, muestra, config)["perdida"],
            "perdida_val": medidas(modelo, val, config)["perdida"],
        })
    fig, ax = plt.subplots(figsize=(7.1, 3.6), layout="constrained")
    for llave, etiqueta, color in [("perdida_train", "Entrenamiento", "#117A69"),
                                  ("perdida_val", "Validación", "#284E8A")]:
        ax.plot([p["n"] for p in puntos], [p[llave] for p in puntos],
                "o-", label=etiqueta, color=color, linewidth=2)
    ax.set(xlabel="Consultas usadas para entrenar", ylabel="Pérdida logarítmica")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(directorio / "curva-aprendizaje.png", dpi=180)
    plt.close(fig)
    return puntos


def seleccionar_politica(filas, decision):
    referencia_i = decision["referencia_intencion"]
    referencia_p = decision["referencia_producto"]
    grilla = []
    for i in decision["umbrales_intencion"]:
        for p in decision["umbrales_producto"]:
            detalle = aplicar_regla(filas, i, p)
            grilla.append({
                "umbral_intencion": i, "umbral_producto": p,
                "operacion": medir_regla(detalle), "producto": medir_producto(detalle),
            })
    def orden(candidato):
        m = candidato["operacion"]
        i, p = candidato["umbral_intencion"], candidato["umbral_producto"]
        distancia = round(abs(i - referencia_i) + abs(p - referencia_p), 6)
        return (m["incorrectas"], -m["correctas"], distancia, i, p)
    return min(grilla, key=orden), grilla


def sondear(config, modelo, catalogo, i, p):
    resultados = []
    for texto in config["decision"]["sondeos"]:
        r = predecir(texto, modelo, catalogo, config["version"])
        intencion = max(r["probabilidades"], key=r["probabilidades"].get)
        confianza = r["probabilidades"][intencion]
        similitud = r["candidatos"][0]["similitud"] if r["candidatos"] else 0.0
        resultados.append({
            "texto": texto, "intencion": intencion, "confianza": confianza,
            "producto": r["producto_texto"], "similitud": similitud,
            "accion": decidir(intencion, confianza, similitud, i, p),
        })
    return resultados


def entrenar(salida):
    salida.mkdir(parents=True, exist_ok=True)
    if (salida / "prueba.json").exists():
        raise FileExistsError("Esta corrida ya fue evaluada. Usa otra carpeta para una reproducción.")
    config = cargar_config()
    train, val, auditoria = cargar_particiones(config, RAIZ)
    d = config["datos"]
    y_train, y_val = train[d["columna_clase"]], val[d["columna_clase"]]
    dummy = DummyClassifier(strategy="most_frequent").fit(np.zeros(len(train)), y_train)
    baseline = medir_intencion(y_val, dummy.predict(np.zeros(len(val))), config["clases"])
    experimentos = []
    ajustados = {}
    for e in config["ajuste_fino"]["experimentos"]:
        modelo = ajustar(train, opciones(config, e))
        ajustados[e["id"]] = modelo
        experimentos.append({
            **e, "iteraciones": int(modelo.named_steps["clasificador"].n_iter_.max()),
            "train": medidas(modelo, train, config), "val": medidas(modelo, val, config),
        })
    def orden(e):
        m = e["val"]
        return (-m["f1_macro"], -m["por_clase"]["otra"]["recall"], m["perdida"])
    ganador = min(experimentos, key=orden)
    modelo = ajustados[ganador["id"]]
    catalogo = sorted({p for p in train[d["columna_producto"]] if p})
    filas_val, _ = predecir_filas(val, config, modelo, catalogo)
    politica, grilla = seleccionar_politica(filas_val, config["decision"])
    i, p = politica["umbral_intencion"], politica["umbral_producto"]
    detalle_val = aplicar_regla(filas_val, i, p)
    modelo_path = salida / "puntobot-v0.7.joblib"
    joblib.dump({"modelo": modelo, "baseline": dummy, "catalogo": catalogo}, modelo_path, compress=3)
    resultado = {
        "version": config["version"], "origen": d["origen"],
        "hora_managua": datetime.now(timezone(timedelta(hours=-6))).isoformat(timespec="seconds"),
        "entorno": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
        "config": config, "auditoria": auditoria,
        "conteos_train": y_train.value_counts().to_dict(),
        "conteos_val": y_val.value_counts().to_dict(),
        "pesos_train": calcular_pesos(y_train, config["clases"]),
        "productos_catalogo": len(catalogo), "mayoritaria": str(dummy.classes_[int(dummy.class_prior_.argmax())]),
        "baseline": baseline, "experimentos": experimentos,
        "seleccion": ganador["id"], "politica": politica, "grilla": grilla,
        "curva": curva(train, val, opciones(config, ganador), salida),
        "validacion_filas": detalle_val, "sondeos": sondear(config, modelo, catalogo, i, p),
        "sha256_modelo": hashlib.sha256(modelo_path.read_bytes()).hexdigest(),
    }
    escribir_json(salida / "seleccion.json", resultado)
    return resultado


def evaluar_prueba(salida):
    destino = salida / "prueba.json"
    if destino.exists():
        raise FileExistsError("La prueba de esta corrida ya está registrada en prueba.json.")
    seleccion = json.loads((salida / "seleccion.json").read_text("utf-8"))
    cfg = seleccion["config"]
    d = cfg["datos"]
    origen = RAIZ / d["directorio"]
    archivo = origen / "test.csv"
    contenido = archivo.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(contenido).hexdigest() != seleccion["auditoria"]["sha256"]["test.csv"]:
        raise ValueError("La partición de prueba cambió después de la selección")
    modelo_path = salida / "puntobot-v0.7.joblib"
    if hashlib.sha256(modelo_path.read_bytes()).hexdigest() != seleccion["sha256_modelo"]:
        raise ValueError("El artefacto cambió después de la selección")
    artefacto = joblib.load(modelo_path)
    test = pd.read_csv(archivo, keep_default_na=False)
    y = test[d["columna_clase"]]
    if set(y) != set(cfg["clases"]):
        raise ValueError("Las clases de prueba no coinciden con la configuración")
    modelo = artefacto["modelo"]
    filas, probs = predecir_filas(test, cfg, modelo, artefacto["catalogo"])
    i = seleccion["politica"]["umbral_intencion"]
    p = seleccion["politica"]["umbral_producto"]
    detalle = aplicar_regla(filas, i, p)
    # Se registra la evidencia antes de cualquier análisis cualitativo.
    resultado = {
        "version": seleccion["version"], "origen": seleccion["origen"],
        "hora_managua": datetime.now(timezone(timedelta(hours=-6))).isoformat(timespec="seconds"),
        "seleccion": seleccion["seleccion"], "clases": cfg["clases"],
        "sha256_prueba": seleccion["auditoria"]["sha256"]["test.csv"],
        "sha256_modelo": seleccion["sha256_modelo"],
        "umbral_intencion": i, "umbral_producto": p,
        "conteos": y.value_counts().to_dict(),
        "intencion": medir_intencion(y, [f["intencion_predicha"] for f in filas],
                                    cfg["clases"], probs, modelo.classes_),
        "baseline": medir_intencion(y, artefacto["baseline"].predict(np.zeros(len(test))), cfg["clases"]),
        "producto": medir_producto(detalle), "operacion": medir_regla(detalle),
        "filas": detalle,
    }
    textos_train = pd.read_csv(origen / "train.csv", usecols=[d["columna_texto"]],
                               keep_default_na=False)[d["columna_texto"]]
    resultado["similitud_train"] = auditar_similitud(detalle, textos_train)
    escribir_json(destino, resultado)
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fase", choices=["validacion", "prueba"])
    parser.add_argument("--salida", type=Path, default=RAIZ / "runtime" / "etapa7")
    args = parser.parse_args()
    if args.fase == "validacion":
        resultado = entrenar(args.salida)
        print("Selección congelada", resultado["seleccion"],
              resultado["politica"]["umbral_intencion"], resultado["politica"]["umbral_producto"])
    else:
        resultado = evaluar_prueba(args.salida)
        print("Prueba final", resultado["intencion"]["exactitud"], resultado["intencion"]["f1_macro"])


if __name__ == "__main__":
    main()
