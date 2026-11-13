# Verificaciones de PuntoBot en la Etapa 6

Fecha de la nueva ejecución: 1 de noviembre de 2026. Semilla: 42. Ejecución local en Python 3.12.14, TensorFlow 2.20.0 y Keras 3.11.3. Datos: lote sintético suministrado, particiones preservadas de la etapa 3.

## División y pesos

| Conjunto | Consultas | Stock | Precio | Última venta | Otra |
|---|---:|---:|---:|---:|---:|
| Entrenamiento | 116 | 49 | 33 | 24 | 10 |
| Validación | 25 | 10 | 7 | 6 | 2 |
| Prueba | 25 | Sellado | Sellado | Sellado | Sellado |

IDs compartidos entre conjuntos: 0. Textos normalizados idénticos compartidos: 0. Los hashes SHA-256 del contenido con saltos LF coinciden con el manifiesto. La conversión LF/CRLF de Git en Windows no altera esa identidad; cualquier otro cambio se rechaza. Se auditan identificadores y textos de test sin leer sus etiquetas ni ejecutar una evaluación sobre él.

| Clase | Peso calculado solo con entrenamiento |
|---|---:|
| consulta_stock | 0.591837 |
| consulta_precio | 0.878788 |
| consulta_ultima_venta | 1.208333 |
| otra | 2.900000 |

## Verificaciones de cordura

| Comprobación | Resultado obtenido |
|---|---|
| Preparación compartida entre pipeline e inferencia | Diferencia máxima 0.0 |
| Forma de características train | 116 × 1675 |
| Forma de características val | 25 × 1675 |
| Forma del lote de Keras | 32 × 1675 |
| Rango TF-IDF | Ceros y valores no nulos entre 0.051230 y 1.000000; todos finitos |
| Probabilidades | Cuatro valores entre 0 y 1; suman 1 |
| Vocabulario e IDF | Ajustados exclusivamente con train |
| Contrato JSON | Campos, clases y tipos correctos |
| Entradas inválidas | Rechaza vacío, espacios y más de 200 caracteres |
| Serialización del pipeline | Predicciones iguales antes y después de guardar/cargar con joblib |
| Pérdida inicial de la red descartable | 1.384109; ln(4) = 1.386294 |
| Sobreajuste de un lote | Pérdida 1.384109 → 0.00003959, exactitud 1.00, 250 épocas |
| Parámetros de la red | 26988; coincide con el cálculo manual |

El modelo usado para memorizar el lote se descarta. Esa prueba demuestra que la red y las características permiten aprender; no mide generalización.

## Baseline y ajuste de verificación

El baseline siempre predice `consulta_stock`, la clase mayoritaria de entrenamiento. Se mide en validación, con etiquetas de test selladas.

| Modelo o ejercicio | Exactitud en val | F1 macro en val |
|---|---:|---:|
| Baseline trivial | 0.400000 | 0.142857 |
| TF-IDF y regresión logística | 0.960000 | 0.970175 |
| Puente didáctico de Keras | 0.920000 | 0.874709 |

Estos valores se obtuvieron ejecutando el código; no son estimaciones ni resultados de producción. El puente no sustituye el clasificador de la etapa 4. No se buscan hiperparámetros en esta etapa.

## Práctica de Keras

Se ejecutaron los tres notebooks completos mediante `scripts/ejecutar_notebooks.py`, cada uno en un kernel independiente: 7, 3 y 3 celdas de código, respectivamente. El informe registra la hora con desplazamiento UTC−06:00 y el SHA-256 de cada notebook fuente con saltos normalizados a LF. Las salidas de los notebooks publicados se borraron. La evidencia numérica se conserva en [resultados.json](resultados.json) y [ejecucion-notebooks.json](ejecucion-notebooks.json).

También pasaron cuatro pruebas de regresión: fecha real de ejecución en Managua, equivalencia de hashes LF/CRLF, rechazo de cambios de contenido y rechazo de IDs compartidos. Se ejecutan con `python -m unittest discover -s tests -v`.

Los ejercicios de romper y reparar comprobaron una forma de entrada incorrecta, una salida lineal que no produce probabilidades y una fuga de vocabulario mostrada con textos artificiales. Se conservaron la forma correcta, softmax y el ajuste exclusivo sobre train.

## Límites y continuidad

El corpus actual es sintético y comparte plantillas. Aunque no hay textos idénticos entre particiones, las plantillas similares pueden inflar la validación. Solo dos consultas de `otra` aparecen en val; no permiten concluir desempeño fiable para esa clase. No se confirma todavía la meta de exactitud sobre consultas reales ni la meta de F1 de extracción de producto.

La etapa 7 debe usar el corpus real, evaluar separación por origen o sesión cuando exista esa información, ajustar con train, calibrar con val y abrir test una sola vez al final. La revisión por otro integrante debe realizarse antes de la entrega académica; no se afirma que ya haya ocurrido.
