"""Ejecuta los tres notebooks completos sin modificar las fuentes versionadas."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import argparse
import hashlib
import json
import os
import sys

import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]


def ejecutar(salida):
    salida = Path(salida).resolve()
    salida.mkdir(parents=True, exist_ok=True)
    # Uso el mismo entorno para el ejecutor y los kernels.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ["MPLBACKEND"] = "Agg"
    informe = {}
    for ruta in sorted((RAIZ / "notebooks").glob("06-*.ipynb")):
        contenido = ruta.read_bytes()
        notebook = nbformat.reads(contenido.decode("utf-8"), as_version=4)
        nbformat.validate(notebook)
        cantidad = sum(c.cell_type == "code" for c in notebook.cells)
        # Agrego la evidencia solo a la copia ejecutada; conservo la fuente sin salidas.
        notebook.cells.append(nbformat.v4.new_code_cell(
            'import json\nprint("EVIDENCIA_JSON=" + json.dumps(globals().get("resumen", globals().get("resultados", {})), ensure_ascii=False, allow_nan=False))'))
        print(f"Ejecutando {ruta.name} ({cantidad} celdas de código)", flush=True)
        NotebookClient(notebook, timeout=1200, kernel_name="python3",
            resources={"metadata": {"path": str(RAIZ)}}).execute()
        resumen = None
        for output in notebook.cells[-1].get("outputs", []):
            for linea in output.get("text", "").splitlines():
                if linea.startswith("EVIDENCIA_JSON="):
                    resumen = json.loads(linea.removeprefix("EVIDENCIA_JSON="))
        if resumen is None:
            raise RuntimeError(f"Falta evidencia en {ruta.name}")
        informe[ruta.name] = {
            "ejecutado_en": datetime.now(timezone(timedelta(hours=-6))).isoformat(),
            "sha256_fuente": hashlib.sha256(contenido.replace(b"\r\n", b"\n")).hexdigest(),
            "normalizacion_hash": "LF",
            "celdas_ejecutadas": cantidad, "sin_errores": True, "resultados": resumen,
        }
        nbformat.write(notebook, salida / ruta.name)
        (salida / "ejecucion-notebooks.json").write_text(
            json.dumps(informe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if contenido != ruta.read_bytes():
            raise RuntimeError("Se modificó un notebook fuente durante la ejecución")
        print(f"Correcto: {ruta.name}", flush=True)
    return informe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=RAIZ / "runtime/notebooks")
    args = parser.parse_args()
    ejecutar(args.salida)
