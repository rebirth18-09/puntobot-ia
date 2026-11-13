"""Carga las particiones existentes; no vuelve a dividir el corpus."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from .preprocesamiento import normalizar_textos

def cargar_particiones(config, raiz):
    d = config["datos"]
    ruta = Path(raiz) / d["directorio"]
    train = pd.read_csv(ruta / "train.csv", keep_default_na=False)
    val = pd.read_csv(ruta / "val.csv", keep_default_na=False)
    # Prueba solo se inspecciona para auditar identificadores y duplicados.
    test_audit = pd.read_csv(ruta / "test.csv", usecols=[d["columna_id"], d["columna_texto"]], keep_default_na=False)
    for nombre, df in [("train", train), ("val", val)]:
        assert set(df[d["columna_clase"]]) == set(config["clases"]), f"Clases ausentes o desconocidas en {nombre}"
    conjuntos = [train, val, test_audit]
    for df in conjuntos:
        assert df[d["columna_id"]].is_unique, "ID duplicado"
        textos = normalizar_textos(df[d["columna_texto"]])
        assert all(textos), "Texto vacío"
        assert len(textos) == len(set(textos)), "Texto duplicado dentro de la partición"
    for a, b in [(0,1), (0,2), (1,2)]:
        assert not set(conjuntos[a][d["columna_id"]]) & set(conjuntos[b][d["columna_id"]]), "ID compartido"
        assert not set(normalizar_textos(conjuntos[a][d["columna_texto"]])) & set(normalizar_textos(conjuntos[b][d["columna_texto"]])), "Texto compartido"
    # Normalizo los saltos LF/CRLF que Git convierte al clonar en Windows.
    # Comparo el contenido con el manifiesto y rechazo cualquier otro cambio.
    hashes = {n: hashlib.sha256((ruta / n).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for n in ["train.csv", "val.csv", "test.csv"]}
    manifest = ruta.parent / "manifest.json"
    if manifest.exists():
        assert hashes == json.loads(manifest.read_text("utf-8"))["sha256"], "Las particiones cambiaron"
    return train, val, {"filas_prueba": len(test_audit), "solapamiento_ids": 0, "solapamiento_textos": 0, "sha256": hashes}
