# Alcance de la evaluación de PuntoBot IA

Esta entrega permite revisar y ejecutar la herramienta de IA sin depender del sistema de gestión de la farmacia.

## Flujo del modelo

1. Normalizar Unicode, minúsculas y espacios.
2. Representar la consulta mediante TF-IDF de palabras y caracteres.
3. Clasificar con regresión logística en cuatro intenciones.
4. Buscar candidatos de producto por similitud de texto.
5. En el entrenamiento de la etapa 7, comparar configuraciones y seleccionar umbrales con validación.

Las probabilidades de intención proceden del clasificador. La similitud de producto procede de una regla con `SequenceMatcher`; no es una probabilidad calibrada ni un modelo de lenguaje generativo.

## Evidencia y reproducción

- [Verificación de esta entrega](entrega-verificada.json), 12 de noviembre de 2026: seis pruebas correctas, demo funcional y selección con validación ejecutada en Python 3.13.4. La configuración E3 (`C=4`, pesos balanceados) obtuvo exactitud y F1 macro de 1.0 sobre las 25 consultas sintéticas de validación. Test no se evaluó. Los notebooks se revisaron sin salidas y con sintaxis válida; TensorFlow no se volvió a ejecutar en esta entrega.
- `docs/06-desarrollo-modelo/resultados.json` y `ejecucion-notebooks.json` son evidencia histórica del 1 de noviembre de 2026. Sus versiones y hashes pertenecen a esa ejecución original.
- Los notebooks de esta entrega adaptan el arranque al repositorio `puntobot-ia`; sus hashes ya no deben compararse con los del informe histórico.
- `requirements.txt` fija las dependencias del clasificador y del entrenamiento de la etapa 7. `requirements-keras.txt` y `requirements-notebooks.txt` añaden la práctica académica opcional.
- `python scripts/verificar.py --sin-keras` genera evidencia nueva en `runtime/verificacion/`, sin sobrescribir la histórica.
- `python -m unittest discover -s tests -v` comprueba integridad de particiones, equivalencia LF/CRLF, fecha en Managua e inferencia de la demo.
- `python -m src.training.entrenar validacion` ajusta con train, selecciona con val y conserva la configuración y el modelo en `runtime/etapa7/`.
- `python -m src.training.entrenar prueba` evalúa el modelo seleccionado sobre test y registra el resultado una vez por carpeta de ejecución. Debe usarse solo al cerrar la selección del modelo.

La copia de evaluación no incluye pesos entrenados. El evaluador puede generarlos localmente con el código y las particiones sintéticas incluidas.

## Límites

El corpus es pequeño y sintético. Algunas frases comparten plantillas; la ausencia de duplicados exactos no garantiza independencia. Validación incluye solo dos ejemplos de `otra`. Las probabilidades y la similitud no garantizan una decisión correcta.

La demo devuelve intención, probabilidades y productos candidatos; no muestra precios ni stock porque no conecta una base de datos real. La calidad de la intención, la extracción de producto y las decisiones se miden por separado en el código de evaluación.

La comprobación histórica de 96 % en validación no constituye una medición en producción. La evaluación final de prueba y una validación con consultas reales requieren su propia evidencia. La revisión académica por otro integrante sigue pendiente según la documentación original.
