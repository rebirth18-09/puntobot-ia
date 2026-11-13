# ADR-003 · TF-IDF y regresión logística para PuntoBot

**Estado:** aceptada como decisión técnica de desarrollo; revisión académica del equipo pendiente.
**Fecha de registro:** 2026-11-01 · **Etapa:** 6.

## Contexto

PuntoBot clasifica texto en cuatro intenciones. La etapa 4 seleccionó TF-IDF y regresión logística y la etapa 5 definió un servicio local scikit-learn que devuelve probabilidades y candidatos de producto. El corpus disponible es pequeño y sintético. El ejemplo dermatológico de las guías clasifica imágenes; su ResNet50 no corresponde a la entrada de este proyecto.

Antecedentes: [selección del enfoque](https://github.com/rebirth18-09/proyecto-ia-puntobot-etapa5/blob/main/docs/04-seleccion-enfoque.md), [ADR-001 de despliegue local](https://github.com/rebirth18-09/proyecto-ia-puntobot-etapa5/blob/main/docs/05-arquitectura/decisiones/ADR-001-despliegue-local.md) y [ADR-002 de inferencia inmediata](https://github.com/rebirth18-09/proyecto-ia-puntobot-etapa5/blob/main/docs/05-arquitectura/decisiones/ADR-002-inferencia-inmediata.md).

## Opciones consideradas

| Opción | Consideración |
|---|---|
| Reglas o palabras clave | Útiles como referencia, pero exigen mantener expresiones y resolver ambigüedades manualmente. |
| TF-IDF y regresión logística | Compatible con la arquitectura, probabilidades por intención y ejecución local sin GPU. |
| Red densa de Keras | Permite completar la práctica; sus resultados didácticos no justifican cambiar el enfoque elegido. |
| Modelo de lenguaje preentrenado | Se puede reconsiderar si el corpus real exige más capacidad y hay recursos para medirlo. |
| ResNet50 | Procesa imágenes, mientras que PuntoBot recibe texto. Se conserva como ejemplo del curso, no como dependencia del producto. |

## Decisión

Mantengo un `Pipeline` con una `FeatureUnion` de TF-IDF de palabras y caracteres y una regresión logística balanceada. La preparación se importa desde un único módulo. Los hiperparámetros viven en `config.yaml`. El puente Keras se conserva como ejercicio separado.

## Consecuencias y criterio de revisión

Se mantiene el contrato `/clasificar`, el despliegue local y el artefacto `.joblib`. No se necesita entrenar una base de imágenes ni aumentar el texto. El vocabulario solo aprende de train y el orden de probabilidades se consulta en `classes_`.

Los n-gramas no garantizan comprensión de contexto, y las probabilidades requieren evaluar su calibración. El catálogo y la similitud de productos siguen siendo una demostración. Si el corpus real no alcanza las metas de exactitud de intención y F1 de producto, la etapa 7 deberá comparar alternativas con validación, registrar la decisión y mantener prueba sellada hasta elegir el modelo final. Las métricas sintéticas actuales no justifican afirmar que se alcanzaron esas metas.
