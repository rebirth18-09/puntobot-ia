"""Verifica la etapa 6 sin evaluar el conjunto de prueba."""
from pathlib import Path
import json
import math
import io
import sys
import platform
import importlib.metadata as metadata
from datetime import datetime, timezone, timedelta
import numpy as np
import yaml
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.dummy import DummyClassifier

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
from src.data.division import cargar_particiones
from src.models.modelo import crear_modelo
from src.models.pesos import calcular_pesos
from src.inference.predictor import predecir

def verificar(incluir_keras=True, guardar=True):
    config = yaml.safe_load((RAIZ / "config.yaml").read_text("utf-8"))
    train, val, auditoria = cargar_particiones(config, RAIZ)
    texto = config["datos"]["columna_texto"]
    etiqueta = config["datos"]["columna_clase"]
    clases = config["clases"]
    x_train, y_train = train[texto].tolist(), train[etiqueta].tolist()
    x_val, y_val = val[texto].tolist(), val[etiqueta].tolist()
    pesos = calcular_pesos(y_train, clases)
    modelo = crear_modelo(config)
    assert not hasattr(modelo.named_steps["clasificador"], "coef_")
    # Ajuste de cordura, no búsqueda de hiperparámetros ni entrenamiento final.
    modelo.fit(x_train, y_train)
    vectorizador = modelo.named_steps["tfidf"]
    X_train = vectorizador.transform(x_train)
    X_val = vectorizador.transform(x_val)
    assert np.isfinite(X_train.data).all() and np.isfinite(X_val.data).all()
    assert X_train.shape[1] == X_val.shape[1]
    palabras = dict(vectorizador.transformer_list)["palabras"]
    assert "tokenexclusivodevalidacion" not in palabras.vocabulary_
    assert palabras.transform(["tokenexclusivodevalidacion"]).nnz == 0
    catalogo = sorted(set(p for p in train["producto_mencionado"] if p))
    respuesta = predecir(x_val[0], modelo, catalogo, config["version"])
    lote = modelo.predict_proba([x_val[0]])[0]
    inferencia = np.array([respuesta["probabilidades"][c] for c in modelo.classes_])
    diferencia = float(np.max(np.abs(lote - inferencia)))
    assert diferencia == 0.0
    probabilidades = modelo.predict_proba(x_val)
    assert np.isfinite(probabilidades).all()
    assert ((probabilidades >= 0) & (probabilidades <= 1)).all()
    assert np.allclose(probabilidades.sum(axis=1), 1)
    assert set(respuesta) == {"modelo_version", "probabilidades", "producto_texto", "candidatos"}
    assert set(respuesta["probabilidades"]) == set(clases)
    # La respuesta es JSON y los datos de inventario no se calculan aquí.
    json.dumps(respuesta, allow_nan=False)
    for invalido in ["", " " * 5, "x" * 201]:
        try:
            predecir(invalido, modelo, catalogo, config["version"])
        except ValueError:
            pass
        else:
            raise AssertionError("Aceptó una entrada inválida")
    # El artefacto incluye el vectorizador ajustado y el clasificador juntos.
    buffer = io.BytesIO()
    joblib.dump(modelo, buffer)
    buffer.seek(0)
    restaurado = joblib.load(buffer)
    assert np.allclose(restaurado.predict_proba(x_val), probabilidades)
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train)
    pred_base = baseline.predict(np.zeros((len(y_val), 1)))
    pred = modelo.predict(x_val)
    ejecucion = datetime.now(timezone(timedelta(hours=-6)))
    resultados = {
        "fecha": ejecucion.date().isoformat(), "ejecutado_en": ejecucion.isoformat(),
        "zona_horaria": "America/Managua",
        "fuente": "lote sintetico suministrado", "semilla": config["entrenamiento"]["semilla"],
        "python": platform.python_version(),
        "versiones": {p: metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "PyYAML"]},
        "filas": {"entrenamiento": len(train), "validacion": len(val), "prueba_sellada": auditoria["filas_prueba"]},
        "conteos": {"entrenamiento": train[etiqueta].value_counts().to_dict(), "validacion": val[etiqueta].value_counts().to_dict()},
        "auditoria": auditoria, "pesos_solo_entrenamiento": pesos,
        "formas": {"train": list(X_train.shape), "val": list(X_val.shape)},
        "rango_tfidf_no_cero": [float(X_train.data.min()), float(X_train.data.max())],
        "diferencia_pipeline_inferencia": diferencia,
        "baseline": {"clase": str(baseline.classes_[np.argmax(baseline.class_prior_)]), "accuracy": float(accuracy_score(y_val, pred_base)),
            "f1_macro": float(f1_score(y_val, pred_base, labels=clases, average="macro", zero_division=0))},
        "clasificador_verificacion": {"accuracy": float(accuracy_score(y_val, pred)), "f1_macro": float(f1_score(y_val, pred, labels=clases, average="macro", zero_division=0)),
            "por_clase": classification_report(y_val, pred, labels=clases, output_dict=True, zero_division=0)},
        "contrato_ejemplo": respuesta,
        "checks": {"particiones_sin_solapamiento": True, "fit_solo_train": True, "probabilidades_validas": True, "contrato_json": True, "entradas_invalidas": True, "serializacion": True},
        "keras": None,
    }
    if incluir_keras:
        import tensorflow as tf
        import keras
        from src.models.puente_keras import crear_modelo_puente
        keras.utils.set_random_seed(config["entrenamiento"]["semilla"])
        # Ocho consultas de cada clase para memorizar un lote balanceado.
        indices = np.concatenate([np.flatnonzero(np.array(y_train) == c)[:8] for c in clases])
        xb = X_train[indices].toarray().astype("float32")
        mapa = {c:i for i,c in enumerate(clases)}
        yb = np.array([mapa[y_train[i]] for i in indices], dtype="int32")
        red = crear_modelo_puente(xb.shape[1], config,
            learning_rate=config["entrenamiento"]["learning_rate_sobreajuste"])
        inicial = red.evaluate(xb, yb, return_dict=True, verbose=0)
        assert math.isfinite(inicial["loss"]) and inicial["loss"] < 10
        historial = red.fit(xb, yb, epochs=config["entrenamiento"]["epocas_sobreajuste"], batch_size=len(yb), verbose=0)
        final = red.evaluate(xb, yb, return_dict=True, verbose=0)
        assert final["accuracy"] >= 0.99 and final["loss"] < 0.1, "El lote no se memorizó"
        salida = red.predict(xb, verbose=0)
        assert salida.shape == (len(yb), len(clases)) and np.allclose(salida.sum(1), 1, atol=1e-5)
        resultados["keras"] = {"tensorflow": tf.__version__, "keras": keras.__version__, "forma_lote": list(xb.shape),
            "parametros": red.count_params(), "ln_k": math.log(len(clases)), "perdida_inicial": float(inicial["loss"]),
            "perdida_final": float(final["loss"]), "accuracy_final": float(final["accuracy"]), "epocas": len(historial.history["loss"])}
        keras.backend.clear_session()
    if guardar:
        carpeta = RAIZ / "runtime/verificacion"
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    return resultados

if __name__ == "__main__":
    resultados = verificar(incluir_keras="--sin-keras" not in sys.argv)
    print(json.dumps(resultados, ensure_ascii=False, indent=2))
