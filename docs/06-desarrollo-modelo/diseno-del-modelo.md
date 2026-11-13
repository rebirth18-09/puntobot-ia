# Diseño del modelo · Etapa 6

Fecha de revisión: 2026-11-01. Proyecto: PuntoBot, consultas de inventario y ventas de Punto Farma.

| Decisión | Valor | Justificación |
|---|---|---|
| Entrada | `texto_consulta`, entre 1 y 200 caracteres | Es la información disponible cuando el empleado pregunta; no utiliza la etiqueta, el producto anotado, el ID ni el rol como características. |
| Clases | `consulta_stock`, `consulta_precio`, `consulta_ultima_venta`, `otra` | Conserva las cuatro intenciones de las etapas 3 a 5. |
| Preparación | Unicode NFC, minúsculas y recorte de espacios | Una sola función importable se ejecuta en entrenamiento e inferencia. |
| Representación | TF-IDF de palabras (1, 2) y caracteres dentro de palabras (3, 5), máximo 3000 características por representación | Combina frases y fragmentos de nombres; vocabulario e IDF se aprenden solo con entrenamiento. |
| Clasificador | Regresión logística, `C=1.0`, `max_iter=1000` | Continúa el enfoque seleccionado en la etapa 4 y el servicio scikit-learn de la etapa 5; ver ADR-003. |
| Salida y pérdida | Cuatro probabilidades, entropía cruzada multiclase con regularización | Las intenciones son excluyentes. Se usa `modelo.classes_` para asociar cada probabilidad a su clase. |
| Balanceo | `class_weight=balanced` | Los pesos dependen exclusivamente de los conteos de entrenamiento: `n / (k × n_clase)`. |
| Pesos actuales | Stock 0.591837; precio 0.878788; última venta 1.208333; otra 2.900000 | Con 116 filas y conteos 49/33/24/10, evita que la clase mayoritaria domine el ajuste de cordura. |
| Tamaño ajustado en la comprobación | 1675 características; coeficientes `(4, 1675)` y 4 interceptos: 6704 valores | Se obtiene del vocabulario de train. No hay una base preentrenada ni parámetros congelados en este clasificador. |
| Puente de Keras | Dense 16 ReLU → Dense 8 ReLU → Dense 4 softmax | Practica la API funcional de la sesión 22; usa etiquetas enteras y `sparse_categorical_crossentropy`. No reemplaza la regresión logística. |
| Parámetros del puente | `(1675×16+16)+(16×8+8)+(8×4+4)=26988` | El notebook comprueba el cálculo manual contra `count_params()`. Las capas y tasas están en `config.yaml`. |
| División | Particiones preservadas de la etapa 3, con manifiesto SHA-256 | Entrenamiento 116, validación 25 y prueba 25. Se auditan IDs y textos; las etiquetas de prueba no se cargan para evaluar. |
| Agrupación | No disponible en el corpus actual | No existe un ID de empleado, sesión o plantilla. La ausencia de duplicados exactos no garantiza independencia entre frases sintéticas. |
| Aumentación | Desactivada | Cambiar palabras podría alterar la intención o el medicamento. Validación e inferencia son deterministas. |
| Artefacto previsto | Pipeline completo `.joblib`, fuera de Git | Mantiene juntos el vectorizador y el clasificador, según el contrato de arquitectura. |
| Productos | Candidatos por similitud contra un catálogo de demostración obtenido de train | No utiliza productos etiquetados de validación o prueba. La extracción debe evaluarse con el catálogo real en la etapa 7. |
| Decisión de negocio | Umbrales iniciales 0.60 y 0.80, aplicados por el backend | El modelo propone probabilidades y candidatos; no consulta el inventario ni inventa precios o ventas. |

Configuración: [config.yaml](../../config.yaml). Código: [modelo.py](../../src/models/modelo.py), [pesos.py](../../src/models/pesos.py) y [puente_keras.py](../../src/models/puente_keras.py).

La explicación de la adaptación de las guías está en [modelo.md](modelo.md). Los resultados son comprobaciones con datos sintéticos, sin selección de hiperparámetros ni evaluación final de prueba. El entrenamiento y la calibración corresponden a la etapa 7; la integración HTTP, a la etapa 8.
