# Revisión de la etapa 6 contra las guías

Fecha: 2026-11-01. Referencias: sesión 22 (primer modelo Keras), guía del modelo dermatológico (sesión 23 en su contenido), pipeline y verificaciones (sesión 24 en su contenido) y resumen de la etapa 6. El nombre de descarga de las dos últimas guías difiere del número que muestran dentro.

## Entregables técnicos

| Requisito | Evidencia | Resultado |
|---|---|---|
| Continuidad con etapas anteriores | Enlaces en el README a etapas 3, 4 y 5 | Mantengo las etapas en repositorios separados y enlazo los documentos previos. |
| README, configuración y dependencias fijadas | README.md, config.yaml, requirements.txt | Presentes; se añadió el bloque de capas y tasa del ejercicio de sobreajuste. |
| Exclusión de datos y pesos | .gitignore, data/.gitkeep | Solo se permiten los fixtures sintéticos enumerados; datos reales, credenciales y artefactos quedan fuera. |
| Puente de Keras y romper/reparar | notebooks/06-1-puente-keras.ipynb | 7 celdas ejecutadas sin errores; cálculo manual de parámetros y coherencia softmax/pérdida comprobados. |
| Modelo importable y comprobación de salida | src/models/modelo.py, notebooks/06-2-modelo-puntobot.ipynb | 3 celdas ejecutadas; salida (4, 4), probabilidades válidas, 1675 características y 6704 coeficientes/interceptos. |
| Diseño justificado | diseno-del-modelo.md | Añadido con decisiones, pesos, dimensiones y límites. |
| ADR-003 | decisiones/ADR-003-tfidf-regresion-logistica.md | Añadido; conserva TF-IDF y regresión logística de las etapas 4 y 5. |
| Preparación compartida | src/data/preprocesamiento.py | Misma función en TF-IDF e inferencia; diferencia máxima de probabilidades 0. |
| División e integridad | src/data/division.py y manifiesto de fixtures | Sin IDs/textos idénticos compartidos; corregida la incompatibilidad LF/CRLF de Windows. No se han redistribuido las filas. |
| Aumentación solo en entrenamiento | modelo.md y diseño | Se decide no aumentar texto para no cambiar intención o medicamento. |
| Cordura y baseline | notebooks/06-3-verificaciones.ipynb | 3 celdas ejecutadas; pérdida inicial cercana a ln(4), lote memorizado con exactitud 1.00, baseline de validación 0.40. |
| Evidencia reproducible | verificaciones.md, resultados.json, ejecucion-notebooks.json | Nueva ejecución completa con hora, versiones y hashes de fuentes con saltos LF; fecha de ejecución dinámica. |
| Fuentes de notebooks sin salidas | Los tres .ipynb versionados | Sin outputs ni execution_count; copias ejecutadas fuera del repositorio. |
| Regresiones | tests/test_verificaciones.py | Cuatro pruebas aprobadas: fecha, LF/CRLF, alteración de datos y solapamiento de IDs. |
| Versión de la etapa | v0.6-modelo-construido y v0.6.1-modelo-verificado | Se conserva la etiqueta inicial; la nueva etiqueta identifica las correcciones verificadas. |

## Adaptación al proyecto propio

La clínica enseña una arquitectura para imágenes con ResNet50. PuntoBot recibe texto y ya seleccionó un clasificador distinto. Se conserva la finalidad de cada comprobación y se documenta la equivalencia; no se cambia la entrada del proyecto para copiar el ejemplo. El notebook 2 y el ADR llevan nombres de PuntoBot en lugar de ResNet50. No existen pacientes, una base congelada de imágenes ni normalización de píxeles en estos datos.

## Pendientes que requieren evidencia del equipo

Al revisar GitHub encontré cero Pull Requests previos de la etapa 6. Falta que otro integrante ejecute y revise los notebooks desde su cuenta, y que se cumpla la revisión circular de las guías. Registro esta ejecución como comprobación técnica; la aprobación académica corresponde a los compañeros.

El corpus es sintético y no permite acreditar las metas del proyecto con usuarios reales. No tiene identificador de sesión, persona o plantilla para garantizar separación por origen. La extracción de producto no está evaluada con un catálogo real. Esos límites quedan declarados para las etapas 7 a 9; no se publica una evaluación final de prueba.
