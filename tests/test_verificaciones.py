"""Regresiones sobre fecha de ejecución e integridad de las particiones."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import patch
import shutil
import yaml

from scripts import verificar as verificacion
from src.data.division import cargar_particiones

RAIZ = Path(__file__).resolve().parents[1]


class VerificacionesTest(TestCase):
    def setUp(self):
        self.config = yaml.safe_load((RAIZ / "config.yaml").read_text("utf-8"))

    def test_fecha_corresponde_a_ejecucion_en_managua(self):
        fecha = datetime(2030, 1, 2, 23, 45, tzinfo=timezone(timedelta(hours=-6)))
        with patch.object(verificacion, "datetime") as reloj:
            reloj.now.return_value = fecha
            resultado = verificacion.verificar(incluir_keras=False, guardar=False)
        self.assertEqual(resultado["fecha"], "2030-01-02")
        self.assertEqual(resultado["ejecutado_en"], fecha.isoformat())
        self.assertEqual(resultado["zona_horaria"], "America/Managua")
        self.assertEqual(reloj.now.call_args.args[0].utcoffset(None), timedelta(hours=-6))

    def test_crlf_y_lf_conservan_la_misma_identidad(self):
        with TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            destino = raiz / self.config["datos"]["directorio"]
            shutil.copytree(RAIZ / "data/fixtures", destino.parent)
            for ruta in destino.glob("*.csv"):
                contenido = ruta.read_bytes().replace(b"\r\n", b"\n")
                ruta.write_bytes(contenido.replace(b"\n", b"\r\n"))
            train, val, auditoria = cargar_particiones(self.config, raiz)
            self.assertEqual((len(train), len(val), auditoria["filas_prueba"]), (116, 25, 25))
            for ruta in destino.glob("*.csv"):
                ruta.write_bytes(ruta.read_bytes().replace(b"\r\n", b"\n"))
            self.assertEqual(cargar_particiones(self.config, raiz)[2]["sha256"], auditoria["sha256"])

    def test_particiones_alteradas_se_rechazan(self):
        with TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            destino = raiz / self.config["datos"]["directorio"]
            shutil.copytree(RAIZ / "data/fixtures", destino.parent)
            ruta = destino / "train.csv"
            ruta.write_text(ruta.read_text("utf-8").replace("a que precio anda el complejo b", "precio diferente de complejo b"), encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "particiones cambiaron"):
                cargar_particiones(self.config, raiz)

    def test_id_compartido_se_rechaza(self):
        with TemporaryDirectory() as temporal:
            raiz = Path(temporal)
            destino = raiz / self.config["datos"]["directorio"]
            shutil.copytree(RAIZ / "data/fixtures", destino.parent)
            ruta = destino / "val.csv"
            lineas = ruta.read_text("utf-8").splitlines()
            id_train = (destino / "train.csv").read_text("utf-8").splitlines()[1].split(",")[0]
            lineas[1] = id_train + "," + lineas[1].split(",", 1)[1]
            ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "ID compartido"):
                cargar_particiones(self.config, raiz)


if __name__ == "__main__":
    main()
