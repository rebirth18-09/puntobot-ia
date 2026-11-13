# PuntoBot IA

Herramienta de inteligencia artificial para clasificar consultas de inventario y ventas de Punto Farma. Este repositorio reúne el módulo de IA y sus evidencias para evaluación académica. Se ejecuta de forma independiente en Python.

## Qué hace la herramienta

Recibe una consulta de texto, estima su intención y propone nombres de productos por similitud con un catálogo de demostración.

| Intención | Ejemplo |
|---|---|
| `consulta_stock` | «hay paracetamol disponible» |
| `consulta_precio` | «cuanto cuesta el ibuprofeno» |
| `consulta_ultima_venta` | «cuando vendieron por ultima vez el ibuprofeno» |
| `otra` | Consulta ajena a esas tres intenciones |

El modelo usa **TF-IDF de palabras y caracteres + regresión logística**. Devuelve probabilidades y candidatos de producto. Los precios, existencias y ventas se obtendrían desde el sistema de gestión al integrarlo; la IA no genera esos valores.

## Probar la IA

Recomendado: Python 3.12 o 3.13 de 64 bits. No requiere GPU, base de datos ni instalar el sistema de gestión.

```bash
git clone https://github.com/rebirth18-09/puntobot-ia.git
cd puntobot-ia
python -m venv .venv
```

Activa el entorno según tu sistema:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
python demo.py
python demo.py --texto "hay paracetamol disponible"
```

`demo.py` ajusta el modelo en memoria usando únicamente las 116 consultas de entrenamiento y devuelve JSON con `intencion`, `confianza`, `probabilidades`, `producto_texto` y `candidatos`. No necesita descargar pesos ni usar credenciales. Sin `--texto` ejecuta tres ejemplos.

## Guía para el evaluador

| Aspecto evaluable | Archivos |
|---|---|
| Demostración funcional | [demo.py](demo.py) |
| Preparación de datos y separación de conjuntos | [src/data/](src/data/), [manifiesto](data/fixtures/manifest.json) |
| Construcción y justificación del modelo | [src/models/modelo.py](src/models/modelo.py), [diseño](docs/06-desarrollo-modelo/diseno-del-modelo.md), [ADR-003](docs/06-desarrollo-modelo/decisiones/ADR-003-tfidf-regresion-logistica.md) |
| Entrenamiento y selección con validación | [src/training/entrenar.py](src/training/entrenar.py), [config.yaml](config.yaml) |
| Inferencia y extracción de candidatos | [src/inference/predictor.py](src/inference/predictor.py) |
| Métricas y reglas de decisión | [src/evaluation/evaluar.py](src/evaluation/evaluar.py) |
| Verificaciones y evidencia | [scripts/verificar.py](scripts/verificar.py), [tests/](tests/), [alcance y reproducción](docs/evaluacion.md) |
| Prácticas académicas en Keras | [notebooks/](notebooks/), [puente Keras](src/models/puente_keras.py) |

## Verificar y entrenar

```bash
python scripts/verificar.py --sin-keras
python -m unittest discover -s tests -v
python -m src.training.entrenar validacion
```

La verificación guarda un informe en `runtime/verificacion/resultados.json`. El entrenamiento compara cuatro configuraciones usando train y val, selecciona una y guarda el modelo, los umbrales y la curva de aprendizaje en `runtime/etapa7/`. Estas salidas quedan fuera de Git.

La evaluación final de test es un paso explícito y posterior a decidir el modelo:

```bash
python -m src.training.entrenar prueba
```

Ese comando registra `runtime/etapa7/prueba.json` y rechaza repetir la evaluación en la misma carpeta. La demostración, las pruebas automatizadas y la selección con validación no miden el desempeño en test.

## Notebooks académicos

TensorFlow y Keras son opcionales para la herramienta principal. Para reproducir las tres prácticas:

```bash
python -m pip install -r requirements-notebooks.txt
python scripts/ejecutar_notebooks.py
```

Las copias ejecutadas quedan en `runtime/notebooks/`. Las fuentes publicadas no contienen salidas. En Google Colab, abre los notebooks desde este repositorio y, si es privado, autoriza un secreto `GITHUB_TOKEN` con permiso de lectura; la primera celda descarga el código e instala las dependencias académicas.

## Datos y resultados

Incluye **166 consultas sintéticas**: 116 de entrenamiento, 25 de validación y 25 de prueba, conservadas con hashes SHA-256. El vocabulario, IDF y catálogo de demostración se aprenden solo con entrenamiento.

La evidencia histórica de la etapa 6 registra **96 % de exactitud y F1 macro de 0.970175 en validación**, frente al 40 % del baseline mayoritario. Corresponde a una comprobación con datos sintéticos y a las versiones consignadas en [resultados.json](docs/06-desarrollo-modelo/resultados.json); no demuestra rendimiento con usuarios reales ni es una evaluación final en test. Consulta [los límites y la evidencia de esta entrega](docs/evaluacion.md).

La [verificación de esta entrega](docs/entrega-verificada.json) registra seis pruebas correctas, demo funcional y selección con validación ejecutada el 12 de noviembre de 2026. La configuración E3 obtuvo exactitud y F1 macro de 1.0 en las 25 consultas sintéticas de validación; test sigue reservado. Estos resultados requieren validación posterior con consultas reales.

## Contenido de la entrega

El repositorio contiene código de IA, configuración, dependencias, datos sintéticos, notebooks, pruebas y documentación del modelo. Los datos reales, credenciales, entornos locales y modelos entrenados están excluidos mediante `.gitignore`. La interfaz, autenticación, base de datos y módulos administrativos del programa completo quedan fuera de esta entrega.

## Equipo

- Ricardo José Romero Blandón
- Josias Beeri Zeledón Pastrán
- Pablo Joshua Castro

Docente: Norman Ariel Gutiérrez Hernández.

Procedencia: módulo de IA de PuntoBot desarrollado en las etapas 6 y 7. Los documentos de `docs/06-desarrollo-modelo/` conservan la evidencia y las decisiones de la etapa 6; los resultados de nuevas ejecuciones se guardan por separado en `runtime/`.
