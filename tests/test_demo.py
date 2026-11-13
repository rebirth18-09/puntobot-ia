"""Comprueba la demo con el modelo real y el contrato de inferencia."""
import json
from unittest import TestCase, main

from demo import clasificar, crear_demo


class DemoTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.modelo, cls.catalogo = crear_demo()

    def test_clasifica_intenciones_y_devuelve_json(self):
        casos = {
            "hay paracetamol disponible": "consulta_stock",
            "cuanto cuesta el ibuprofeno": "consulta_precio",
            "cuando vendieron por ultima vez el ibuprofeno": "consulta_ultima_venta",
        }
        for texto, esperada in casos.items():
            with self.subTest(texto=texto):
                resultado = clasificar(texto, self.config, self.modelo, self.catalogo)
                self.assertEqual(resultado["intencion"], esperada)
                self.assertEqual(set(resultado["probabilidades"]), set(self.config["clases"]))
                self.assertAlmostEqual(sum(resultado["probabilidades"].values()), 1.0)
                self.assertIn(resultado["producto_texto"], self.catalogo)
                json.dumps(resultado, allow_nan=False)

    def test_rechaza_consultas_invalidas(self):
        for texto in ("", "   ", "x" * 201):
            with self.subTest(texto=texto), self.assertRaises(ValueError):
                clasificar(texto, self.config, self.modelo, self.catalogo)


if __name__ == "__main__":
    main()
