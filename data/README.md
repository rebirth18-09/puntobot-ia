# Datos de la Etapa 6

`fixtures/splits/` conserva exactamente los archivos sintéticos `train.csv`, `val.csv` y `test.csv` entregados en la etapa 3. [Origen](https://github.com/rebirth18-09/proyecto-ia-puntobot/tree/main/data/splits).

El manifiesto permite verificar sus hashes. El vocabulario, IDF, pesos de clase y catálogo de demostración se calculan únicamente con entrenamiento. La etapa 6 usa validación para comprobaciones y mantiene `test.csv` sellado para la etapa 7. Solo se leen sus identificadores y textos para auditar solapamientos; no se leen sus etiquetas ni se mide su desempeño.

Las consultas reales deben guardarse fuera de Git, en `data/splits/`. Configura esa carpeta en `config.yaml` cuando exista el corpus real. Las cifras actuales no demuestran la meta de 85 % en la farmacia.

Para comparar hashes normalizo únicamente los saltos LF/CRLF que Git puede convertir al clonar en Windows; los valores del manifiesto corresponden al contenido LF original. No se adjuntó una licencia de redistribución del CSV; los fixtures se conservan como material sintético suministrado para la práctica, sin atribuirles una licencia externa.
