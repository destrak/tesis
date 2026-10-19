# Dinámica del piojo de mar en centros de cultivo de *Salmo salar* de Chile y Noruega (2014–2024)

Código de investigación para el **estudio comparativo de la dinámica del piojo de mar** en seis centros seleccionados de cultivo de salmón del Atlántico (*Salmo salar*), tres ubicados en Chile y tres en Noruega. El proyecto integra registros parasitarios y variables oceanográficas mediante **procesamiento de datos, homologación temporal, análisis estadístico, agrupamiento K-means y aprendizaje automático**. Su objetivo es caracterizar patrones parasitarios y ambientales y evaluar modelos capaces de predecir la carga parasitaria total de la semana siguiente.

Los centros constituyen **casos de estudio** y no una muestra representativa de la salmonicultura de ambos países. Los análisis comparativos y predictivos no permiten, por sí solos, establecer relaciones causales.

## Metodología

El flujo de trabajo comprende cinco etapas:

1. **Obtención de datos:** recopilación de registros parasitarios desde **SERNAPESCA** (Chile) y **BarentsWatch** (Noruega), y extracción de temperatura y salinidad superficiales desde **Copernicus Marine**.
2. **Preparación de las bases:** limpieza y estandarización de los CSV, procesamiento de archivos NetCDF, agregación semanal e integración de información ambiental y parasitaria para los centros seleccionados.
3. **Comparación temporal y ambiental:** homologación estacional mediante un **desfase de 26 semanas aplicado a Chile** y comparación de tres pares de macrozonas: Los Lagos–Sur/Oeste, Aysén–Centro y Magallanes–Norte. La comparabilidad ambiental principal se evalúa con **|ΔT| ≤ 3 °C** y **|ΔS| ≤ 3 PSU**; también se estudian escenarios de sensibilidad de ±2/±2 y ±4/±4.
4. **Exploración de patrones:** generación de series temporales, diagramas de caja y mapas de correlación; identificación de grupos mediante **K-means** y representación de resultados con **PCA**.
5. **Predicción temporal:** desarrollo, ajuste y evaluación de modelos **Random Forest** y **LSTM** para predecir la carga parasitaria total en la semana **t+1**, considerando análisis separados para Chile y Noruega y un conjunto combinado.

## Archivos del proyecto

Las descripciones corresponden a los **33 scripts Python revisados**. Los modelos predictivos definitivos son **`random_forest_final.py`** y **`lstm_final.py`**. Los demás scripts de Random Forest y LSTM documentan pruebas, selección de hiperparámetros y análisis complementarios.

### Obtención y preparación de datos

| **Script** | **Función** |
| --- | --- |
| `limpiarcsvchile.py` | Filtra y organiza los registros parasitarios chilenos de un centro seleccionado, estandariza las semanas y prepara un CSV para el análisis. |
| `limpiarcsvbarentwatch.py` | Limpia y adapta los registros de piojo de mar obtenidos desde BarentsWatch, normalizando columnas para los análisis posteriores. |
| `datosnoruega(1).py` | Consulta la API de BarentsWatch y descarga registros semanales de infestación para una localidad noruega (configurada en el script como **32677**). **Requiere credenciales propias.** |
| `datoscentrolagos102424(1).py` | Descarga desde Copernicus Marine archivos NetCDF de temperatura y salinidad para el centro **102424**, en Los Lagos. |
| `datoscentroayzen110758(1).py` | Descarga variables oceanográficas desde Copernicus Marine para el centro **110758**, en Aysén. |
| `datoscentromagallanes120128(1).py` | Descarga temperatura y salinidad para el centro **120128**, en Magallanes. |
| `datossuroeste33077(1).py` | Descarga variables ambientales para el centro noruego **33077**, en Sur/Oeste. |
| `datoszonacentronoruega32677(1).py` | Descarga variables ambientales para el centro noruego **32677**, en la zona Centro. |
| `datoszonanortenoruega24175(1).py` | Descarga variables ambientales para el centro noruego **24175**, en la zona Norte. |
| `procesar_nc.py` | Lee los archivos NetCDF descargados y consolida los valores de temperatura y salinidad en una base tabular; su configuración original corresponde al centro **120128**. |
| `promedio_semanal.py` | Agrega los registros ambientales a promedios semanales de temperatura y salinidad. |
| `juntardatasets.py` | Une los registros parasitarios y ambientales mediante las claves temporales correspondientes para construir una base integrada; el archivo de ejemplo está configurado para un centro específico. |
| `agregarestacionyparasitostotales.py` | Asigna estaciones del año según el país y la semana, calcula `parasitos_totales` a partir de las categorías parasitarias y genera bases consolidadas para Chile, Noruega y el conjunto. **Sobrescribe algunos CSV originales** según su configuración. |

### Comparaciones, gráficos y agrupamiento

| **Script** | **Función** |
| --- | --- |
| `comparar(1).py` | Compara un par de centros de Chile y Noruega tras aplicar el desfase de 26 semanas y calcula diferencias y cumplimiento de los umbrales ambientales de **±3 °C** y **±3 PSU**. |
| `comparartodo(1).py` | Extiende la comparación a los tres pares de macrozonas y analiza la sensibilidad de las semanas comparables frente a umbrales **±2/±2**, **±3/±3** y **±4/±4**. |
| `Wilcoxonsigned-rank.py` | Homologa temporalmente las observaciones de Chile y Noruega, contrasta la normalidad de las diferencias mediante Shapiro–Wilk y aplica una prueba pareada (t de Student o Wilcoxon, según corresponda), además de resumir la comparabilidad ambiental. |
| `graficar.py` | Genera gráficos de temperatura, salinidad y diferencias ambientales de un par de centros, mostrando los umbrales de comparación. |
| `graficartodo.py` | Genera series temporales y visualizaciones de temperatura y salinidad para Chile, Noruega y el conjunto de observaciones. |
| `parasitosenconjunto.py` | Integra los centros de ambos países, aplica homologación estacional y produce bases procesadas, **diagramas de caja** y **series temporales** de carga parasitaria. |
| `parasitosseparadosporzona.py` | Construye gráficos de evolución parasitaria por región o macrozona, organizados por país y sobre semanas homologadas. |
| `clustering(1).py` | Aplica **K-means** a variables parasitarias y ambientales; evalúa el número de grupos con métodos de codo y silueta, y presenta los conglomerados mediante **PCA** y otros gráficos descriptivos. |
| `heatmap_correlacion.py` | Genera un mapa de calor de correlaciones entre variables del conjunto de datos utilizado en Random Forest. |
| `heatmap_shap.py` | Compara visualmente las importancias de variables obtenidas mediante **SHAP** para los modelos de Chile, Noruega y el conjunto combinado. |

### Random Forest

| **Script** | **Función** |
| --- | --- |
| `randomforest(1).py` | Versión exploratoria que evalúa variables originales y variables de identificación temporal o territorial, con interpretación de importancia mediante **SHAP agrupado**. |
| `radomforestmejorado.py` | Desarrolla un Random Forest con variables históricas, rezagos, estadísticas móviles y análisis **SHAP** para la predicción semanal. Se conserva la escritura original del nombre del archivo. |
| `random_forest_gridsearch.py` | Evalúa combinaciones de hiperparámetros mediante **GridSearchCV** y genera resultados de evaluación e importancia de variables. |
| `random_forest_reducido_1.py` | Prueba una configuración reducida de predictores, excluyendo variables de identificación temporal y territorial (*proxies*). |
| `random_forest_reducido_2.py` | Evalúa un modelo reducido basado principalmente en variables del momento actual, sin rezagos ni estadísticas móviles. |
| `randomforest_comparacion_bloques.py` | Compara el rendimiento de Random Forest utilizando diferentes bloques de variables explicativas y referencias predictivas. |
| **`random_forest_final.py`** | **Modelo Random Forest definitivo:** utiliza hiperparámetros seleccionados, predice la carga parasitaria en **t+1**, calcula métricas, importancia de variables, resultados **SHAP** cuando está disponible y tiempos de ejecución. |

### Redes LSTM

| **Script** | **Función** |
| --- | --- |
| `LSTM.py` | Primera implementación de redes **LSTM** para predecir la carga parasitaria semanal mediante secuencias de observaciones. |
| `LSTM_GRIDSEARCH.py` | Explora configuraciones e hiperparámetros del modelo LSTM y compara sus resultados mediante evaluación temporal. |
| **`lstm_final.py`** | **Modelo LSTM definitivo:** utiliza los hiperparámetros seleccionados, entrena y evalúa redes para Chile, Noruega y el conjunto combinado, y reporta métricas, predicciones, figuras y tiempos/complejidad computacional. |

## Disponibilidad de datos y reproducibilidad

> **Este repositorio documenta los códigos de investigación, pero no es una aplicación de ejecución inmediata.** La reproducción de los análisis exige disponer de los datos originales o de las bases procesadas equivalentes y revisar las rutas, las columnas y las configuraciones de cada script.

Los registros de origen provienen de **SERNAPESCA**, **BarentsWatch** y **Copernicus Marine**, sujetos a las condiciones de acceso y utilización correspondientes. Los scripts trabajan principalmente con archivos **CSV** y **NetCDF**; varios contienen rutas locales absolutas del entorno donde se realizó el estudio.

Para reutilizar los códigos es necesario:

1. Disponer de los archivos de entrada de los centros de estudio y adaptar las rutas locales.
2. Verificar las variables, unidades, semanas y convenciones de nombres antes de combinar las bases.
3. Preparar las bases procesadas requeridas por cada análisis; **no todos los scripts forman una única cadena de ejecución automática**.
4. En los modelos predictivos, respetar el orden temporal de las observaciones y las particiones de entrenamiento y prueba, evitando utilizar información futura.

**Dependencia importante:** `lstm_final.py` está configurado originalmente para leer un archivo `datasetT1.csv` ubicado en la carpeta de resultados de una versión mejorada de Random Forest. Este archivo debe existir y conservar el esquema esperado; ejecutar únicamente `random_forest_final.py` no garantiza que se genere en esa misma ruta. Los dos modelos finales pueden documentarse como resultados definitivos sin asumir que uno sustituye automáticamente los datos de entrada del otro.

Las bibliotecas utilizadas incluyen **pandas, NumPy, matplotlib, scikit-learn, SciPy, xarray, Copernicus Marine Toolbox, requests, SHAP, joblib y TensorFlow/Keras**, según el script. No se proporciona un entorno de ejecución único verificado para todo el proyecto.

**Seguridad:** las credenciales de acceso a servicios externos deben configurarse fuera del código fuente (por ejemplo, mediante variables de entorno). No deben publicarse tokens, claves o contraseñas en el repositorio.

## Agradecimientos

**ANID FONDECYT de Iniciación 11261470**.
