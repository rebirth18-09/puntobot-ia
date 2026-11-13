# Modelo de clasificación de intenciones

## Entrada y salida

El modelo recibe `texto_consulta` y devuelve probabilidades de `consulta_stock`, `consulta_precio`, `consulta_ultima_venta` y `otra`. El producto mencionado y la intención etiquetada no forman parte de sus entradas. Tampoco se usan rol, hora, ID o longitud para predecir la intención.

## Representación y modelo elegido

Una función compartida normaliza Unicode, minúsculas y espacios extremos. TF-IDF combina palabras y bigramas con n-gramas de caracteres dentro de palabras. El vocabulario y la frecuencia inversa se ajustan únicamente con train. El clasificador es regresión logística con `C=1`, hasta 1000 iteraciones y pesos balanceados. Esta es la familia elegida en la etapa 4, compatible con el servicio scikit-learn y artefacto `.joblib` de la etapa 5.

La pérdida es entropía cruzada multiclase regularizada. El orden de probabilidades se obtiene de `modelo.classes_`, sin suponer que coincide con un orden numérico fijo. Cada respuesta incluye nombres de clase y versión.

## Puente de Keras

Para practicar la sesión 22 se construye otra red: entrada TF-IDF, Dense 16 ReLU, Dense 8 ReLU y Dense 4 softmax. Usa Adam y `sparse_categorical_crossentropy`. Si la entrada tiene n características, hay `(n × 16 + 16) + (16 × 8 + 8) + (8 × 4 + 4)` parámetros. Las etiquetas son enteros de 0 a 3. No se usa una sigmoide binaria con cuatro clases.

El entrenamiento del puente y el sobreajuste de un lote son ejercicios de verificación. No seleccionan ni reemplazan el modelo final del proyecto.

## Adaptación del pipeline de la guía

| Ejemplo de imágenes | Aplicación en PuntoBot |
|---|---|
| Cargar y normalizar imágenes | Normalizar texto y aplicar el mismo vectorizador ajustado |
| Modelo ResNet50 | Pipeline TF-IDF más regresión logística elegido en la etapa 4 |
| Clases de urgencia | Cuatro intenciones de consulta |
| División por paciente | Preservar particiones y auditar IDs y textos; falta identificador de origen para agrupar |
| Aumentación solo en train | No se aplica aumentación: modificar palabras podría cambiar intención o producto |
| Mostrar imágenes de un lote | Mostrar consultas con sus etiquetas |
| Forma y rango de píxeles | Forma del vector TF-IDF, valores finitos y probabilidades válidas |
| Memorización de un lote | Red Keras descartable con 32 consultas balanceadas |

## Contrato con el backend

`predecir` devuelve el contenido del contrato `/clasificar` documentado en la etapa 5. La coincidencia de productos usa similitud por fragmentos de palabras contra un catálogo; es una implementación de demostración, todavía sin evaluar contra un catálogo real ni demostrar la meta de F1 0.80. El servicio propone candidatos y el backend decide si aplica los umbrales 0.60 y 0.80. Aquí no se ejecuta SQL ni se inventan valores de inventario. El endpoint HTTP corresponde a la integración posterior.

## Siguiente etapa

El corpus real debe reemplazar el lote sintético. La etapa 7 entrenará el modelo, calibrará con validación y abrirá test una sola vez para la evaluación final. Debe reportar F1 macro y métricas por clase, especialmente `otra`. Las plantillas compartidas del lote actual pueden inflar la validación.

## Referencias técnicas

- [TF-IDF en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).
- [API de entrenamiento de Keras](https://keras.io/api/models/model_training_apis/).
- [Pérdidas probabilísticas de Keras](https://keras.io/api/losses/probabilistic_losses/).
