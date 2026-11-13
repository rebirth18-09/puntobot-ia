# Desarrollo del modelo de PuntoBot

La etapa 6 convierte el enfoque y los contratos de las etapas anteriores en componentes ejecutables. El modelo elegido continúa siendo TF-IDF y regresión logística; Keras se utiliza para completar la práctica de construcción de una red y la prueba de memorización de un lote.

| Entregable | Contenido |
|---|---|
| [Diseño del modelo](diseno-del-modelo.md) | Decisiones, dimensiones, pérdida y justificación |
| [ADR-003](decisiones/ADR-003-tfidf-regresion-logistica.md) | Elección de TF-IDF y regresión logística |
| [Adaptación de la guía](modelo.md) | Relación entre el ejemplo dermatológico y PuntoBot |
| [Verificaciones](verificaciones.md) | Resultados obtenidos al ejecutar las comprobaciones |
| [Resultados JSON](resultados.json) | Evidencia numérica y versiones |
| [Puente de Keras](../../notebooks/06-1-puente-keras.ipynb) | Definir, compilar, entrenar y predecir |
| [Construcción del modelo](../../notebooks/06-2-modelo-puntobot.ipynb) | Pipeline elegido en la etapa 4 |
| [Pipeline y verificaciones](../../notebooks/06-3-verificaciones.ipynb) | Auditoría, baseline y pruebas de cordura |

Los notebooks se publican sin salidas. Los valores de la corrida se conservan en documentos y JSON para que otro integrante pueda reproducirlos.

[Integrantes, docente y responsabilidades](trabajo-en-equipo.md).

[Ejecución completa de los notebooks](../../scripts/ejecutar_notebooks.py): instala `requirements-notebooks.txt` y ejecuta `python scripts/ejecutar_notebooks.py`. Conserva las fuentes sin salidas y guarda las copias ejecutadas en `runtime/notebooks/`, fuera de Git.
