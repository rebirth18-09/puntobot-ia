"""Demostración independiente: ajusta solo con train y clasifica texto."""
import argparse
import json
from pathlib import Path

import yaml

from src.data.division import cargar_particiones
from src.inference.predictor import predecir
from src.models.modelo import crear_modelo

RAIZ = Path(__file__).resolve().parent
EJEMPLOS = (
    "hay paracetamol disponible",
    "cuanto cuesta el ibuprofeno",
    "cuando vendieron por ultima vez el ibuprofeno",
)


def crear_demo():
    config = yaml.safe_load((RAIZ / "config.yaml").read_text("utf-8"))
    train, _, _ = cargar_particiones(config, RAIZ)
    columnas = config["datos"]
    modelo = crear_modelo(config).fit(
        train[columnas["columna_texto"]], train[columnas["columna_clase"]]
    )
    catalogo = sorted({p for p in train[columnas["columna_producto"]] if p})
    return config, modelo, catalogo


def clasificar(texto, config, modelo, catalogo):
    respuesta = predecir(texto, modelo, catalogo, config["version"] + "-demo")
    intencion = max(respuesta["probabilidades"], key=respuesta["probabilidades"].get)
    return {
        "texto": texto,
        "intencion": intencion,
        "confianza": respuesta["probabilidades"][intencion],
        "modo": "demostracion_con_datos_sinteticos",
        **respuesta,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texto", help="Consulta de entre 1 y 200 caracteres")
    args = parser.parse_args()
    if args.texto is not None and (not args.texto.strip() or len(args.texto) > 200):
        parser.error("La consulta debe tener entre 1 y 200 caracteres")
    config, modelo, catalogo = crear_demo()
    consultas = [args.texto] if args.texto is not None else EJEMPLOS
    for texto in consultas:
        resultado = clasificar(texto, config, modelo, catalogo)
        print(json.dumps(resultado, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
