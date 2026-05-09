# Resumen Ejecutivo — Análisis de Sentimientos NLP

## 1. Objetivo

Este proyecto construye y evalúa modelos de clasificación NLP para predecir si reseñas de películas expresan sentimiento positivo o negativo.

El objetivo no es solo clasificar texto, sino demostrar un flujo completo de NLP: limpieza de texto, vectorización TF-IDF, comparación de modelos, validación cruzada, evaluación final en test, interpretación de tokens y análisis de errores.

## 2. Contexto de negocio

El análisis de sentimientos ayuda a las organizaciones a entender opiniones a escala.

Un clasificador de sentimiento puede apoyar el monitoreo de feedback de clientes, análisis de reseñas de productos, escucha social, análisis de percepción de marca e investigación de mercado.

Sin embargo, los modelos de sentimiento no deben tratarse como jueces perfectos de opinión. Pueden fallar con sarcasmo, negación, sentimiento mixto, frases ambiguas y lenguaje específico de dominio.

## 3. Alcance del dataset

El proyecto usa el IMDB Dataset of 50K Movie Reviews.

El dataset contiene:

| Dataset | Filas | Columnas |
|---|---:|---:|
| Reseñas crudas IMDB | 50,000 | 2 |
| Reseñas limpias deduplicadas | 49,582 | 7 |
| Train split | 34,706 | 7 |
| Validation split | 7,438 | 7 |
| Test split | 7,438 | 7 |

La variable objetivo es `sentiment`.

La variable binaria de modelado es `sentiment_label`, donde:

| Etiqueta | Significado |
|---:|---|
| 0 | Negativo |
| 1 | Positivo |

## 4. Hallazgos de calidad de datos

El dataset crudo estaba balanceado, pero requería limpieza textual.

| Validación | Resultado |
|---|---:|
| Filas crudas | 50,000 |
| Valores faltantes | 0 |
| Filas duplicadas exactas | 418 |
| Textos de reseña duplicados | 418 |
| Reseñas con HTML | 29,202 |
| Reseñas positivas antes de limpieza | 25,000 |
| Reseñas negativas antes de limpieza | 25,000 |

Las 418 reseñas duplicadas fueron eliminadas antes de dividir los datos. Esto evita fuga de información si una misma reseña apareciera en train y también en validation/test.

Después de deduplicar, el dataset limpio contiene 49,582 reseñas.

## 5. Preprocesamiento de texto

El pipeline de preprocesamiento realiza los siguientes pasos:

1. Convierte reseñas y sentimientos a formato string consistente.
2. Valida que reseñas duplicadas no tengan etiquetas contradictorias.
3. Elimina textos de reseña duplicados.
4. Decodifica entidades HTML.
5. Elimina etiquetas HTML como `<br />`.
6. Convierte texto a minúsculas.
7. Elimina URLs.
8. Conserva caracteres alfabéticos y apóstrofes.
9. Normaliza espacios repetidos.
10. Crea etiquetas binarias de sentimiento.

La limpieza conserva apóstrofes porque las contracciones y negaciones pueden ser importantes en análisis de sentimientos.

## 6. Balance de clases después del preprocesamiento

La distribución de clases se mantiene balanceada después de deduplicar:

| Clase | Cantidad | Proporción |
|---|---:|---:|
| Positivo | 24,884 | 50.19% |
| Negativo | 24,698 | 49.81% |

Las particiones train, validation y test preservan este balance mediante split estratificado.

## 7. Longitud del texto

Las reseñas limpias varían bastante en longitud:

| Métrica | Clean Word Count |
|---|---:|
| Media | 229.94 |
| Mediana | 172 |
| Mínimo | 6 |
| Máximo | 2,462 |

Esta variabilidad importa porque las reseñas cortas pueden carecer de contexto, mientras que las reseñas largas pueden contener opiniones mixtas.

## 8. Modelos comparados

El proyecto comparó los siguientes modelos:

| Modelo | Descripción |
|---|---|
| Baseline most frequent | Clasificador de clase mayoritaria |
| Logistic Regression + TF-IDF | Clasificador lineal con features TF-IDF |
| Linear SVM + TF-IDF | Clasificador lineal de margen con TF-IDF |
| Naive Bayes + TF-IDF | Clasificador probabilístico para texto |
| Random Forest + TF-IDF | Ensamble de árboles sobre features sparse |

TF-IDF usó unigramas y bigramas, eliminación de stop words en inglés y escalamiento sublinear de frecuencia.

## 9. Resultados en validación

En la primera partición de validación, Logistic Regression obtuvo el mayor F1-score:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression TF-IDF | 0.9013 | 0.8892 | 0.9178 | 0.9032 | 0.9637 | 0.9633 |
| Linear SVM TF-IDF | 0.9010 | 0.8959 | 0.9084 | 0.9021 | 0.9640 | 0.9630 |
| Naive Bayes TF-IDF | 0.8778 | 0.8698 | 0.8896 | 0.8796 | 0.9466 | 0.9455 |
| Random Forest TF-IDF | 0.8477 | 0.8239 | 0.8859 | 0.8537 | 0.9268 | 0.9247 |
| Baseline Most Frequent | 0.5019 | 0.5019 | 1.0000 | 0.6683 | 0.5000 | 0.5019 |

El recall del baseline es engañoso porque el modelo predice todas las reseñas como positivas.

## 10. Resultados de validación cruzada

La validación cruzada cambió la decisión final del modelo.

| Modelo | Accuracy promedio | F1 promedio | ROC-AUC promedio | Average Precision promedio |
|---|---:|---:|---:|---:|
| Linear SVM TF-IDF | 0.8955 | mayor entre candidatos | 0.9608 | 0.9594 |
| Logistic Regression TF-IDF | 0.8946 | segundo mayor | 0.9605 | 0.9596 |
| Naive Bayes TF-IDF | 0.8738 | menor | 0.9456 | 0.9448 |
| Random Forest TF-IDF | 0.8432 | menor | 0.9237 | 0.9213 |
| Baseline Most Frequent | 0.5019 | baseline engañoso | 0.5000 | 0.5019 |

Linear SVM obtuvo el mejor F1 promedio en validación cruzada. Por eso fue seleccionado como modelo final antes de evaluar en test.

## 11. Evaluación final en test

El modelo final seleccionado es:

```text
Linear SVM + TF-IDF
```

Resultados finales en test:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression TF-IDF | 0.9000 | 0.8893 | 0.9145 | 0.9017 | 0.9650 | 0.9649 |
| Linear SVM TF-IDF | 0.9000 | 0.8963 | 0.9054 | 0.9009 | 0.9642 | 0.9637 |
| Naive Bayes TF-IDF | 0.8790 | 0.8766 | 0.8832 | 0.8799 | 0.9495 | 0.9479 |
| Random Forest TF-IDF | 0.8463 | 0.8209 | 0.8875 | 0.8529 | 0.9277 | 0.9258 |
| Baseline Most Frequent | 0.5019 | 0.5019 | 1.0000 | 0.6683 | 0.5000 | 0.5019 |

Logistic Regression tiene un desempeño ligeramente mejor en test, pero Linear SVM se mantiene como modelo final porque fue elegido con validación cruzada antes de mirar los resultados de test.

Esto evita seleccionar el modelo usando el set de prueba.

## 12. Matriz de confusión del modelo final

El modelo final Linear SVM clasifica correctamente 6,694 de 7,438 reseñas de test.

| Real / Predicho | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 3,314 | 391 |
| Actual Positive | 353 | 3,380 |

El modelo genera:

- 391 falsos positivos,
- 353 falsos negativos.

La distribución de errores es relativamente balanceada.

## 13. Interpretación de tokens

Se extrajeron los coeficientes del Linear SVM para identificar tokens asociados a cada clase.

Tokens positivos fuertes:

- excellent,
- great,
- perfect,
- best,
- amazing,
- brilliant,
- wonderful,
- enjoyable,
- loved,
- superb.

Tokens negativos fuertes:

- worst,
- awful,
- waste,
- bad,
- boring,
- disappointment,
- terrible,
- fails,
- forgettable,
- mediocre,
- poor,
- dull,
- horrible.

Estos coeficientes muestran asociaciones en el espacio TF-IDF. No son explicaciones causales.

## 14. Análisis de errores

El modelo final produjo:

| Resultado | Cantidad |
|---|---:|
| Predicciones correctas | 6,694 |
| Falsos positivos | 391 |
| Falsos negativos | 353 |

Tasa de error por sentimiento real:

| Sentimiento real | Tasa de error |
|---|---:|
| Negativo | 10.55% |
| Positivo | 9.46% |

El modelo comete ligeramente más errores en reseñas negativas. Esto sugiere que es marginalmente más probable que clasifique algunas reseñas negativas como positivas.

La longitud de la reseña no explica por sí sola los errores. Las predicciones incorrectas tienen una mediana de palabras similar a las predicciones correctas.

## 15. Recomendaciones de negocio

1. Usar el modelo para monitoreo escalable de sentimientos, no como juez perfecto de opinión.
2. Usar tendencias agregadas de sentimiento en lugar de depender solo de predicciones individuales.
3. Revisar manualmente predicciones de baja confianza cuando la decisión tenga alto impacto.
4. Monitorear falsos positivos y falsos negativos por separado.
5. Reentrenar el modelo con datos específicos del dominio antes de aplicarlo fuera de reseñas de películas.
6. Añadir revisión humana para casos ambiguos, sarcásticos o de sentimiento mixto.
7. Evitar usar el modelo como sistema totalmente automático de moderación o decisión.

## 16. Limitaciones

- El modelo fue entrenado en reseñas de películas y puede no generalizar a otros dominios.
- TF-IDF no comprende profundamente contexto, sarcasmo, ironía o negación compleja.
- Los coeficientes lineales sirven para interpretación, pero no son explicaciones causales.
- El modelo no usa embeddings contextuales ni transformers en esta versión.
- El texto crudo de reseñas no se guarda en outputs del repositorio para evitar redistribución innecesaria del dataset.
- El desempeño en escenarios reales puede cambiar con texto más corto, ruidoso o informal.

## 17. Próximos pasos

- Agregar calibración de probabilidades para ajustar umbrales de decisión.
- Añadir un flujo de revisión manual para predicciones de baja confianza.
- Probar el modelo en otro dominio de reseñas.
- Comparar TF-IDF con word embeddings o sentence embeddings.
- Agregar modelos transformer en una versión futura.
- Construir una API ligera de inferencia en un proyecto posterior de despliegue.
