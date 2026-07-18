# -*- coding: utf-8 -*-

# ============================================================
# RANDOM FOREST REGRESSOR FINAL SIN PROXIES + SHAP + TIEMPOS
# Predicción de carga parasitaria de la semana siguiente
#
# Modelo final:
# - No usa anio, zona, localidad ni estacion como predictoras.
# - Usa pais solo en el modelo conjunto Chile--Noruega.
# - Usa hiperparámetros seleccionados previamente mediante GridSearchCV.
# - Registra tiempos de ejecución.
# - Estima complejidad temporal aproximada.
# ============================================================

import os
from pathlib import Path
import unicodedata
import time
import math

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

import joblib

try:
    import shap
except ModuleNotFoundError:
    shap = None
    print("SHAP no está instalado. El código correrá sin SHAP.")


# ============================================================
# 1. Rutas
# ============================================================

carpeta_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\dataset\datasetchile"
carpeta_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\dataset\datasetnoruega"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_final_sin_proxies"
os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. Configuración general
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

CALCULAR_IMPORTANCIA = True
CALCULAR_SHAP = shap is not None

N_MUESTRA_SHAP = 300
N_REPEATS_PERMUTACION = 5

TIEMPO_INICIO_SCRIPT = time.perf_counter()


# ============================================================
# 3. Hiperparámetros finales seleccionados por GridSearchCV
# ============================================================

MEJORES_PARAMETROS_RF = {
    "conjunto_chile_noruega": {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 8,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "bootstrap": True
    },
    "chile": {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_split": 8,
        "min_samples_leaf": 2,
        "max_features": "log2",
        "bootstrap": True
    },
    "noruega": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "max_features": "log2",
        "bootstrap": True
    }
}


# ============================================================
# 4. Funciones auxiliares generales
# ============================================================

def normalizar_columna(col):
    col = str(col).strip().lower()
    col = unicodedata.normalize("NFKD", col)
    col = "".join(c for c in col if not unicodedata.combining(c))
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace(".", "_")
    col = col.replace("/", "_")

    while "__" in col:
        col = col.replace("__", "_")

    return col


def leer_csv_seguro(ruta):
    for enc in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            return pd.read_csv(ruta, encoding=enc, sep=None, engine="python")
        except Exception:
            pass

    raise ValueError(f"No se pudo leer el archivo: {ruta}")


def cargar_carpeta(carpeta, pais):
    carpeta = Path(carpeta)

    archivos = list(carpeta.glob("*.csv"))

    archivos = [
        f for f in archivos
        if "piojos_temp_sal" in f.name.lower()
    ]

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos válidos en {carpeta}")

    bases = []

    for archivo in archivos:
        print(f"Leyendo {pais}: {archivo.name}")

        df_temp = leer_csv_seguro(archivo)
        df_temp.columns = [normalizar_columna(c) for c in df_temp.columns]

        df_temp["pais"] = pais
        df_temp["archivo_origen"] = archivo.name

        bases.append(df_temp)

    return pd.concat(bases, ignore_index=True)


def convertir_numerico(serie):
    return pd.to_numeric(
        serie.astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )


def crear_fecha_iso(anio, semana):
    return pd.to_datetime(
        anio.astype("Int64").astype(str)
        + "-W"
        + semana.astype("Int64").astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce"
    )


def media_minima(df_temp, columnas, minimo=2):
    conteo = df_temp[columnas].notna().sum(axis=1)
    media = df_temp[columnas].mean(axis=1, skipna=True)
    media[conteo < minimo] = np.nan
    return media


def maximo_minimo(df_temp, columnas, minimo=2):
    conteo = df_temp[columnas].notna().sum(axis=1)
    maximo = df_temp[columnas].max(axis=1, skipna=True)
    maximo[conteo < minimo] = np.nan
    return maximo


def obtener_nombres_variables(preprocesador):
    try:
        nombres = preprocesador.get_feature_names_out()
        nombres = [
            nombre.replace("num__", "").replace("cat__", "")
            for nombre in nombres
        ]
        return nombres
    except Exception:
        return None


def formatear_tiempo(segundos):
    minutos = int(segundos // 60)
    seg = segundos % 60

    if minutos > 0:
        return f"{minutos} min {seg:.2f} s"

    return f"{seg:.2f} s"


# ============================================================
# 5. Funciones para complejidad temporal aproximada
# ============================================================

def obtener_numero_variables_preprocesadas(modelo, X):
    """
    Calcula cuántas variables quedan después del preprocesamiento
    numérico + One-Hot Encoding.
    """
    preprocesador = modelo.named_steps["preprocesamiento"]
    X_prep = preprocesador.transform(X)

    if hasattr(X_prep, "shape"):
        return X_prep.shape[1]

    return np.nan


def calcular_mtry(max_features, p):
    """
    Aproxima el número de variables consideradas en cada división.
    Para max_features='log2', se usa floor(log2(p)).
    """
    if p <= 0 or pd.isna(p):
        return np.nan

    if max_features == "sqrt":
        return max(1, int(np.sqrt(p)))

    if max_features == "log2":
        return max(1, int(np.log2(p)))

    if max_features is None:
        return p

    if isinstance(max_features, int):
        return min(max_features, p)

    if isinstance(max_features, float):
        return max(1, int(max_features * p))

    return np.nan


def estimar_profundidad_efectiva(max_depth, n_train):
    """
    Si max_depth=None, se usa una aproximación conservadora basada en log2(n).
    """
    if n_train <= 1:
        return 1

    if max_depth is None:
        return int(np.ceil(np.log2(n_train)))

    return int(max_depth)


def estimar_complejidad_rf(
    nombre_modelo,
    n_train,
    n_test,
    p_preprocesadas,
    params,
    n_repeats_permutacion=5,
    n_muestra_shap=300
):
    """
    Complejidades aproximadas:

    Entrenamiento Random Forest:
        O(B * n * m_try * d * log(n))

    Predicción:
        O(B * n_test * d)

    Permutation importance:
        O(R * p_original * costo_prediccion)

    SHAP TreeExplainer:
        Aproximación práctica:
        O(n_shap * B * d^2)

    Donde:
        B = número de árboles
        n = número de observaciones de entrenamiento
        p = número de variables después del preprocesamiento
        m_try = variables consideradas por split
        d = profundidad efectiva del árbol
        R = repeticiones de permutación
    """

    B = params["n_estimators"]
    d = estimar_profundidad_efectiva(params["max_depth"], n_train)
    m_try = calcular_mtry(params["max_features"], p_preprocesadas)

    log_n = np.log2(max(n_train, 2))

    complejidad_entrenamiento = B * n_train * m_try * d * log_n
    complejidad_prediccion = B * n_test * d

    n_shap = min(n_muestra_shap, n_test)

    complejidad_shap = n_shap * B * (d ** 2)

    return {
        "modelo": nombre_modelo,
        "n_train": n_train,
        "n_test": n_test,
        "n_variables_preprocesadas": p_preprocesadas,
        "n_arboles_B": B,
        "m_try_aprox": m_try,
        "profundidad_efectiva_d": d,
        "complejidad_entrenamiento_teorica": "O(B * n * m_try * d * log(n))",
        "operaciones_entrenamiento_aprox": complejidad_entrenamiento,
        "complejidad_prediccion_teorica": "O(B * n_test * d)",
        "operaciones_prediccion_aprox": complejidad_prediccion,
        "complejidad_permutacion_teorica": "O(R * p * B * n_test * d)",
        "n_repeats_permutacion": n_repeats_permutacion,
        "complejidad_shap_teorica_aprox": "O(n_shap * B * d^2)",
        "n_muestra_shap": n_shap,
        "operaciones_shap_aprox": complejidad_shap
    }


# ============================================================
# 6. Cargar bases
# ============================================================

tiempo_carga_inicio = time.perf_counter()

df_chile = cargar_carpeta(carpeta_chile, "Chile")
df_noruega = cargar_carpeta(carpeta_noruega, "Noruega")

df = pd.concat([df_chile, df_noruega], ignore_index=True)

tiempo_carga_fin = time.perf_counter()

print("\nColumnas cargadas:")
print(df.columns.tolist())

print(
    "\nTiempo de carga de datos:",
    formatear_tiempo(tiempo_carga_fin - tiempo_carga_inicio)
)


# ============================================================
# 7. Homologar columnas
# ============================================================

if "ano" in df.columns:
    df = df.rename(columns={"ano": "anio"})

columnas_necesarias = [
    "anio",
    "semana",
    "localidad",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "temperatura",
    "salinidad",
    "parasitos_totales",
    "pais"
]

faltantes = [c for c in columnas_necesarias if c not in df.columns]

if len(faltantes) > 0:
    raise ValueError(f"Faltan columnas necesarias: {faltantes}")


# ============================================================
# 8. Conversión numérica
# ============================================================

columnas_numericas_base = [
    "anio",
    "semana",
    "localidad",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "temperatura",
    "salinidad",
    "parasitos_totales"
]

for col in columnas_numericas_base:
    df[col] = convertir_numerico(df[col])


# ============================================================
# 9. Identificar centro y zona
# ============================================================

df["localidad_id"] = df["localidad"].astype("Int64")

mapa_zonas = {
    102424: "Los Lagos",
    110758: "Aysén",
    120128: "Magallanes",
    24175: "Norte",
    32677: "Centro",
    33077: "Sur/Oeste"
}

df["zona"] = df["localidad_id"].map(mapa_zonas)

if df["zona"].isna().sum() > 0:
    print("\nAdvertencia: hay localidades sin zona asignada:")
    print(df[df["zona"].isna()]["localidad_id"].unique())

df["zona"] = df["zona"].fillna("Sin zona")
df["localidad"] = df["localidad_id"].astype(str)


# ============================================================
# 10. Estación
# ============================================================

if "estacion" not in df.columns:
    df["estacion"] = "Sin estación"

df["estacion"] = df["estacion"].astype(str)


# ============================================================
# 11. Semana homologada Chile -> Noruega
# ============================================================

df["semana_homologada"] = df["semana"].copy()
df["anio_homologado"] = df["anio"].copy()

mask_chile = df["pais"] == "Chile"

df.loc[mask_chile, "semana_homologada"] = (
    df.loc[mask_chile, "semana"] + 26
)

cambio_anio = (
    mask_chile &
    (df["semana_homologada"] > 52)
)

df.loc[cambio_anio, "semana_homologada"] = (
    df.loc[cambio_anio, "semana_homologada"] - 52
)

df.loc[cambio_anio, "anio_homologado"] = (
    df.loc[cambio_anio, "anio_homologado"] + 1
)

df["semana_homologada"] = df["semana_homologada"].astype("Int64")
df["anio_homologado"] = df["anio_homologado"].astype("Int64")

print("\nEjemplo de homologación temporal Chile -> Noruega:")
print(
    df[["pais", "anio", "semana", "anio_homologado", "semana_homologada"]]
    .dropna()
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 12. Eliminar duplicados por país-centro-año-semana
# ============================================================

print("\nFilas antes de eliminar duplicados:", df.shape[0])

df = df.drop_duplicates(
    subset=["pais", "localidad_id", "anio", "semana"],
    keep="first"
).copy()

print("Filas después de eliminar duplicados:", df.shape[0])


# ============================================================
# 13. Crear fechas
# ============================================================

df = df.sort_values(["pais", "localidad_id", "anio", "semana"]).copy()

df["fecha_t"] = crear_fecha_iso(df["anio"], df["semana"])


# ============================================================
# 14. Crear variable objetivo t+1 real
# ============================================================

df["parasitos_totales_t1"] = (
    df.groupby(["pais", "localidad_id"])["parasitos_totales"].shift(-1)
)

df["anio_objetivo"] = (
    df.groupby(["pais", "localidad_id"])["anio"].shift(-1)
)

df["semana_objetivo"] = (
    df.groupby(["pais", "localidad_id"])["semana"].shift(-1)
)

df["fecha_t1"] = crear_fecha_iso(
    df["anio_objetivo"],
    df["semana_objetivo"]
)

df["es_semana_siguiente"] = (
    (df["fecha_t1"] - df["fecha_t"]).dt.days == 7
)


# ============================================================
# 15. Crear rezagos con verificación de continuidad
# ============================================================

grupo = df.groupby(["pais", "localidad_id"], group_keys=False)

variables_rezago = [
    "parasitos_totales",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "temperatura",
    "salinidad"
]

for lag in [1, 2, 3, 4]:

    df[f"fecha_lag{lag}"] = grupo["fecha_t"].shift(lag)
    diferencia_dias = (df["fecha_t"] - df[f"fecha_lag{lag}"]).dt.days

    es_lag_valido = diferencia_dias == (7 * lag)

    for var in variables_rezago:
        nueva_col = f"{var}_lag{lag}"
        df[nueva_col] = grupo[var].shift(lag)
        df.loc[~es_lag_valido, nueva_col] = np.nan


# ============================================================
# 16. Medias móviles, máximos y cambios semanales
# ============================================================

for var in variables_rezago:
    columnas_media = [
        var,
        f"{var}_lag1",
        f"{var}_lag2",
        f"{var}_lag3"
    ]

    df[f"{var}_media_4"] = media_minima(
        df,
        columnas_media,
        minimo=2
    )


variables_parasitarias = [
    "parasitos_totales",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles"
]

for var in variables_parasitarias:
    columnas_max = [
        var,
        f"{var}_lag1",
        f"{var}_lag2",
        f"{var}_lag3"
    ]

    df[f"{var}_max_4"] = maximo_minimo(
        df,
        columnas_max,
        minimo=2
    )


df["delta_parasitos_1"] = (
    df["parasitos_totales"] - df["parasitos_totales_lag1"]
)

df["delta_hembras_1"] = (
    df["hembras_ovigeras"] - df["hembras_ovigeras_lag1"]
)

df["delta_adultos_1"] = (
    df["adultos_moviles"] - df["adultos_moviles_lag1"]
)

df["delta_juveniles_1"] = (
    df["juveniles"] - df["juveniles_lag1"]
)

df["delta_temperatura_1"] = (
    df["temperatura"] - df["temperatura_lag1"]
)

df["delta_salinidad_1"] = (
    df["salinidad"] - df["salinidad_lag1"]
)


# ============================================================
# 17. Comparación de tamaños
# ============================================================

columnas_clustering = [
    "pais",
    "zona",
    "localidad",
    "localidad_id",
    "anio",
    "semana",
    "semana_homologada",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales",
    "temperatura",
    "salinidad"
]

columnas_rf = columnas_clustering + [
    "parasitos_totales_t1",
    "anio_objetivo",
    "semana_objetivo"
]

df_clustering_equivalente = df.dropna(subset=columnas_clustering).copy()

df_rf_sin_continuidad = df.dropna(subset=columnas_rf).copy()

df_rf_con_continuidad = df[
    df["es_semana_siguiente"]
].dropna(
    subset=columnas_rf
).copy()

print("\n======================================")
print("COMPARACIÓN DE TAMAÑOS")
print("======================================")

print("\nRegistros tipo clustering:")
print(df_clustering_equivalente["pais"].value_counts())
print("Total:", df_clustering_equivalente.shape[0])

print("\nRF sin exigir semana siguiente real:")
print(df_rf_sin_continuidad["pais"].value_counts())
print("Total:", df_rf_sin_continuidad.shape[0])

print("\nRF exigiendo semana siguiente real:")
print(df_rf_con_continuidad["pais"].value_counts())
print("Total:", df_rf_con_continuidad.shape[0])


# ============================================================
# 18. Base final del modelo
# ============================================================

df_modelo = df_rf_con_continuidad.copy()

df_modelo["semana_sin"] = np.sin(
    2 * np.pi * df_modelo["semana_homologada"] / 52
)

df_modelo["semana_cos"] = np.cos(
    2 * np.pi * df_modelo["semana_homologada"] / 52
)

df_modelo.to_csv(
    os.path.join(carpeta_salida, "datasetT1.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nDatasetT1 guardado correctamente.")


# ============================================================
# 19. Función para entrenar Random Forest final
# ============================================================

def entrenar_random_forest(nombre_modelo, datos):

    tiempo_inicio_modelo = time.perf_counter()

    print("\n======================================")
    print("Modelo:", nombre_modelo)
    print("Observaciones:", datos.shape[0])
    print("======================================")

    # --------------------------------------------------------
    # Variables predictoras finales sin proxies
    # --------------------------------------------------------
    # Se excluyen como predictoras:
    # - anio
    # - zona
    # - localidad
    # - estacion
    #
    # La variable pais solo se usa en el modelo conjunto.
    # --------------------------------------------------------

    variables_numericas = [
        "temperatura",
        "salinidad",
        "hembras_ovigeras",
        "adultos_moviles",
        "juveniles",
        "parasitos_totales",
        "semana_sin",
        "semana_cos",

        "parasitos_totales_lag1",
        "parasitos_totales_lag2",
        "parasitos_totales_lag3",
        "parasitos_totales_lag4",
        "parasitos_totales_media_4",
        "parasitos_totales_max_4",
        "delta_parasitos_1",

        "hembras_ovigeras_lag1",
        "hembras_ovigeras_lag2",
        "hembras_ovigeras_lag3",
        "hembras_ovigeras_lag4",
        "hembras_ovigeras_media_4",
        "hembras_ovigeras_max_4",
        "delta_hembras_1",

        "adultos_moviles_lag1",
        "adultos_moviles_lag2",
        "adultos_moviles_lag3",
        "adultos_moviles_lag4",
        "adultos_moviles_media_4",
        "adultos_moviles_max_4",
        "delta_adultos_1",

        "juveniles_lag1",
        "juveniles_lag2",
        "juveniles_lag3",
        "juveniles_lag4",
        "juveniles_media_4",
        "juveniles_max_4",
        "delta_juveniles_1",

        "temperatura_lag1",
        "temperatura_lag2",
        "temperatura_lag3",
        "temperatura_lag4",
        "temperatura_media_4",
        "delta_temperatura_1",

        "salinidad_lag1",
        "salinidad_lag2",
        "salinidad_lag3",
        "salinidad_lag4",
        "salinidad_media_4",
        "delta_salinidad_1"
    ]

    variables_numericas = [
        col for col in variables_numericas
        if col in datos.columns
    ]

    if nombre_modelo == "conjunto_chile_noruega" and datos["pais"].nunique() > 1:
        variables_categoricas = ["pais"]
    else:
        variables_categoricas = []

    variables_predictoras = variables_numericas + variables_categoricas

    print("\nVariables predictoras utilizadas:")
    print(variables_predictoras)
    print("Número de variables predictoras originales:", len(variables_predictoras))

    # --------------------------------------------------------
    # División temporal entrenamiento/prueba
    # --------------------------------------------------------

    datos_train = datos[datos["anio_objetivo"] <= 2022].copy()
    datos_test = datos[datos["anio_objetivo"] >= 2023].copy()

    print("Entrenamiento:", datos_train.shape[0])
    print("Prueba:", datos_test.shape[0])

    if datos_train.shape[0] == 0 or datos_test.shape[0] == 0:
        print(f"No hay datos suficientes para {nombre_modelo}")
        return None, None, None, None, None, None

    X_train = datos_train[variables_predictoras].copy()
    y_train = datos_train["parasitos_totales_t1"].copy()

    X_test = datos_test[variables_predictoras].copy()
    y_test = datos_test["parasitos_totales_t1"].copy()

    # --------------------------------------------------------
    # Pipeline de preprocesamiento
    # --------------------------------------------------------

    transformers = [
        (
            "num",
            SimpleImputer(strategy="median"),
            variables_numericas
        )
    ]

    if len(variables_categoricas) > 0:
        transformers.append(
            (
                "cat",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                variables_categoricas
            )
        )

    preprocesamiento = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # --------------------------------------------------------
    # Hiperparámetros óptimos según GridSearchCV sin proxies
    # --------------------------------------------------------

    params = MEJORES_PARAMETROS_RF[nombre_modelo]

    print("\nHiperparámetros usados:")
    print(params)

    rf = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        random_state=42,
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        bootstrap=params["bootstrap"],
        n_jobs=1
    )

    modelo = Pipeline(steps=[
        ("preprocesamiento", preprocesamiento),
        ("random_forest", rf)
    ])

    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------

    tiempo_inicio_entrenamiento = time.perf_counter()

    modelo.fit(X_train, y_train)

    tiempo_fin_entrenamiento = time.perf_counter()
    tiempo_entrenamiento = tiempo_fin_entrenamiento - tiempo_inicio_entrenamiento

    # --------------------------------------------------------
    # Predicción y evaluación final
    # --------------------------------------------------------

    tiempo_inicio_prediccion = time.perf_counter()

    y_pred = modelo.predict(X_test)

    tiempo_fin_prediccion = time.perf_counter()
    tiempo_prediccion = tiempo_fin_prediccion - tiempo_inicio_prediccion

    y_base = datos_test["parasitos_totales"].values

    mae_rf = mean_absolute_error(y_test, y_pred)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_rf = r2_score(y_test, y_pred)

    mae_base = mean_absolute_error(y_test, y_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_base))
    r2_base = r2_score(y_test, y_base)

    # --------------------------------------------------------
    # Variables después del preprocesamiento
    # --------------------------------------------------------

    p_preprocesadas = obtener_numero_variables_preprocesadas(modelo, X_train)

    complejidad = estimar_complejidad_rf(
        nombre_modelo=nombre_modelo,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        p_preprocesadas=p_preprocesadas,
        params=params,
        n_repeats_permutacion=N_REPEATS_PERMUTACION,
        n_muestra_shap=N_MUESTRA_SHAP
    )

    metricas = pd.DataFrame({
        "modelo": [nombre_modelo],
        "n_entrenamiento": [X_train.shape[0]],
        "n_prueba": [X_test.shape[0]],
        "n_variables_originales": [len(variables_predictoras)],
        "n_variables_preprocesadas": [p_preprocesadas],
        "MAE_RF": [mae_rf],
        "RMSE_RF": [rmse_rf],
        "R2_RF": [r2_rf],
        "MAE_base_persistencia": [mae_base],
        "RMSE_base_persistencia": [rmse_base],
        "R2_base_persistencia": [r2_base],
        "n_estimators": [params["n_estimators"]],
        "max_depth": [params["max_depth"]],
        "min_samples_split": [params["min_samples_split"]],
        "min_samples_leaf": [params["min_samples_leaf"]],
        "max_features": [params["max_features"]],
        "bootstrap": [params["bootstrap"]],
        "tiempo_entrenamiento_s": [tiempo_entrenamiento],
        "tiempo_prediccion_s": [tiempo_prediccion],
        "tiempo_entrenamiento": [formatear_tiempo(tiempo_entrenamiento)],
        "tiempo_prediccion": [formatear_tiempo(tiempo_prediccion)],
        "variables_proxy_excluidas": ["anio, zona, localidad, estacion"]
    })

    print("\nMétricas:")
    print(metricas.to_string(index=False))

    metricas.to_csv(
        os.path.join(
            carpeta_salida,
            f"metricas_rf_final_sin_proxies_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Guardar complejidad temporal
    # --------------------------------------------------------

    complejidad_df = pd.DataFrame([complejidad])

    complejidad_df.to_csv(
        os.path.join(
            carpeta_salida,
            f"complejidad_temporal_rf_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nComplejidad temporal aproximada:")
    print(complejidad_df.to_string(index=False))

    # --------------------------------------------------------
    # Guardar predicciones
    # --------------------------------------------------------

    predicciones = datos_test[[
        "pais",
        "zona",
        "localidad",
        "localidad_id",
        "anio",
        "semana",
        "anio_objetivo",
        "semana_objetivo",
        "parasitos_totales",
        "parasitos_totales_t1"
    ]].copy()

    predicciones["prediccion_rf"] = y_pred
    predicciones["prediccion_base_persistencia"] = y_base

    predicciones["error_rf"] = (
        predicciones["parasitos_totales_t1"] -
        predicciones["prediccion_rf"]
    )

    predicciones["error_abs_rf"] = np.abs(
        predicciones["error_rf"]
    )

    predicciones.to_csv(
        os.path.join(
            carpeta_salida,
            f"predicciones_rf_final_sin_proxies_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Gráfico observado vs predicho
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    limite_max = max(y_test.max(), y_pred.max())

    if limite_max == 0:
        limite_max = 1

    plt.plot([0, limite_max], [0, limite_max], linestyle="--")

    plt.xlabel("Carga parasitaria observada")
    plt.ylabel("Carga parasitaria predicha")
    plt.title(f"Random Forest final sin proxies: observado vs predicho ({nombre_modelo})")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"rf_final_sin_proxies_observado_vs_predicho_{nombre_modelo}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Serie temporal observada y predicha
    # --------------------------------------------------------

    serie = predicciones.groupby(
        ["anio_objetivo", "semana_objetivo"],
        as_index=False
    )[["parasitos_totales_t1", "prediccion_rf"]].mean()

    serie["fecha"] = crear_fecha_iso(
        serie["anio_objetivo"],
        serie["semana_objetivo"]
    )

    serie = serie.dropna(subset=["fecha"]).sort_values("fecha")

    plt.figure(figsize=(11, 5))

    plt.plot(
        serie["fecha"],
        serie["parasitos_totales_t1"],
        label="Observado"
    )

    plt.plot(
        serie["fecha"],
        serie["prediccion_rf"],
        label="Predicho"
    )

    plt.xlabel("Fecha objetivo")
    plt.ylabel("Carga parasitaria promedio")
    plt.title(f"Serie temporal observada y predicha ({nombre_modelo})")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"rf_final_sin_proxies_serie_temporal_{nombre_modelo}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Importancia por permutación
    # --------------------------------------------------------

    importancia_df = None
    tiempo_importancia = np.nan

    if CALCULAR_IMPORTANCIA:

        print("\nCalculando importancia por permutación...")

        tiempo_inicio_importancia = time.perf_counter()

        importancia = permutation_importance(
            modelo,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=N_REPEATS_PERMUTACION,
            random_state=42,
            n_jobs=1
        )

        tiempo_fin_importancia = time.perf_counter()
        tiempo_importancia = tiempo_fin_importancia - tiempo_inicio_importancia

        importancia_df = pd.DataFrame({
            "variable": variables_predictoras,
            "importancia_media": importancia.importances_mean,
            "importancia_std": importancia.importances_std
        }).sort_values(
            "importancia_media",
            ascending=False
        )

        importancia_df.to_csv(
            os.path.join(
                carpeta_salida,
                f"importancia_permutacion_rf_final_sin_proxies_{nombre_modelo}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        importancia_plot = importancia_df.head(20).sort_values(
            "importancia_media"
        )

        plt.figure(figsize=(8, 7))

        plt.barh(
            importancia_plot["variable"],
            importancia_plot["importancia_media"]
        )

        plt.xlabel("Incremento del MAE al permutar la variable")
        plt.ylabel("Variable")
        plt.title(f"Importancia de variables sin proxies ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"rf_final_sin_proxies_importancia_variables_{nombre_modelo}.png"
            ),
            dpi=300
        )

        plt.close()

        print("Tiempo importancia por permutación:", formatear_tiempo(tiempo_importancia))

    # --------------------------------------------------------
    # SHAP values
    # --------------------------------------------------------

    shap_df = None
    tiempo_shap = np.nan

    if CALCULAR_SHAP:

        print("\nCalculando SHAP values...")

        tiempo_inicio_shap = time.perf_counter()

        preprocesador = modelo.named_steps["preprocesamiento"]
        rf_entrenado = modelo.named_steps["random_forest"]

        X_test_prep = preprocesador.transform(X_test)

        if hasattr(X_test_prep, "toarray"):
            X_test_prep = X_test_prep.toarray()

        nombres_variables = obtener_nombres_variables(preprocesador)

        if nombres_variables is None:
            nombres_variables = [
                f"var_{i}" for i in range(X_test_prep.shape[1])
            ]

        X_test_prep_df = pd.DataFrame(
            X_test_prep,
            columns=nombres_variables
        )

        n_muestra_shap = min(
            N_MUESTRA_SHAP,
            X_test_prep_df.shape[0]
        )

        X_shap = X_test_prep_df.sample(
            n=n_muestra_shap,
            random_state=42
        )

        explainer = shap.TreeExplainer(rf_entrenado)

        shap_values = explainer.shap_values(X_shap)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_df = pd.DataFrame({
            "variable": X_shap.columns,
            "shap_importancia_media_abs": np.abs(shap_values).mean(axis=0)
        }).sort_values(
            "shap_importancia_media_abs",
            ascending=False
        )

        shap_df.to_csv(
            os.path.join(
                carpeta_salida,
                f"shap_importancia_rf_final_sin_proxies_{nombre_modelo}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        shap.summary_plot(
            shap_values,
            X_shap,
            plot_type="bar",
            max_display=20,
            show=False
        )

        plt.title(f"Importancia SHAP - Random Forest sin proxies ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_bar_rf_final_sin_proxies_{nombre_modelo}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        shap.summary_plot(
            shap_values,
            X_shap,
            max_display=20,
            show=False
        )

        plt.title(f"SHAP summary plot - Random Forest sin proxies ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_beeswarm_rf_final_sin_proxies_{nombre_modelo}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        tiempo_fin_shap = time.perf_counter()
        tiempo_shap = tiempo_fin_shap - tiempo_inicio_shap

        print("SHAP values calculados correctamente.")
        print("Tiempo SHAP:", formatear_tiempo(tiempo_shap))

    # --------------------------------------------------------
    # Guardar modelo
    # --------------------------------------------------------

    joblib.dump(
        modelo,
        os.path.join(
            carpeta_salida,
            f"modelo_rf_final_sin_proxies_{nombre_modelo}.joblib"
        )
    )

    tiempo_fin_modelo = time.perf_counter()
    tiempo_total_modelo = tiempo_fin_modelo - tiempo_inicio_modelo

    tiempos_modelo = pd.DataFrame({
        "modelo": [nombre_modelo],
        "tiempo_entrenamiento_s": [tiempo_entrenamiento],
        "tiempo_prediccion_s": [tiempo_prediccion],
        "tiempo_importancia_permutacion_s": [tiempo_importancia],
        "tiempo_shap_s": [tiempo_shap],
        "tiempo_total_modelo_s": [tiempo_total_modelo],
        "tiempo_entrenamiento": [formatear_tiempo(tiempo_entrenamiento)],
        "tiempo_prediccion": [formatear_tiempo(tiempo_prediccion)],
        "tiempo_importancia_permutacion": [
            formatear_tiempo(tiempo_importancia)
            if not pd.isna(tiempo_importancia)
            else "No calculado"
        ],
        "tiempo_shap": [
            formatear_tiempo(tiempo_shap)
            if not pd.isna(tiempo_shap)
            else "No calculado"
        ],
        "tiempo_total_modelo": [formatear_tiempo(tiempo_total_modelo)]
    })

    tiempos_modelo.to_csv(
        os.path.join(
            carpeta_salida,
            f"tiempos_rf_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTiempo total del modelo:", formatear_tiempo(tiempo_total_modelo))

    return metricas, predicciones, importancia_df, shap_df, tiempos_modelo, complejidad_df


# ============================================================
# 20. Entrenar modelos
# ============================================================

metricas_todas = []
shap_todos = []
tiempos_todos = []
complejidades_todas = []

met_conjunto, pred_conjunto, imp_conjunto, shap_conjunto, tiempo_conjunto, comp_conjunto = entrenar_random_forest(
    "conjunto_chile_noruega",
    df_modelo
)

if met_conjunto is not None:
    metricas_todas.append(met_conjunto)

if shap_conjunto is not None:
    shap_conjunto["modelo"] = "conjunto_chile_noruega"
    shap_todos.append(shap_conjunto)

if tiempo_conjunto is not None:
    tiempos_todos.append(tiempo_conjunto)

if comp_conjunto is not None:
    complejidades_todas.append(comp_conjunto)


met_chile, pred_chile, imp_chile, shap_chile, tiempo_chile, comp_chile = entrenar_random_forest(
    "chile",
    df_modelo[df_modelo["pais"] == "Chile"].copy()
)

if met_chile is not None:
    metricas_todas.append(met_chile)

if shap_chile is not None:
    shap_chile["modelo"] = "chile"
    shap_todos.append(shap_chile)

if tiempo_chile is not None:
    tiempos_todos.append(tiempo_chile)

if comp_chile is not None:
    complejidades_todas.append(comp_chile)


met_noruega, pred_noruega, imp_noruega, shap_noruega, tiempo_noruega, comp_noruega = entrenar_random_forest(
    "noruega",
    df_modelo[df_modelo["pais"] == "Noruega"].copy()
)

if met_noruega is not None:
    metricas_todas.append(met_noruega)

if shap_noruega is not None:
    shap_noruega["modelo"] = "noruega"
    shap_todos.append(shap_noruega)

if tiempo_noruega is not None:
    tiempos_todos.append(tiempo_noruega)

if comp_noruega is not None:
    complejidades_todas.append(comp_noruega)


# ============================================================
# 21. Resumen final
# ============================================================

if len(metricas_todas) > 0:

    resumen_metricas = pd.concat(metricas_todas, ignore_index=True)

    resumen_metricas.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_random_forest_final_sin_proxies.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    resumen_metricas_redondeado = resumen_metricas.copy()

    columnas_redondear = [
        "MAE_RF",
        "RMSE_RF",
        "R2_RF",
        "MAE_base_persistencia",
        "RMSE_base_persistencia",
        "R2_base_persistencia",
        "tiempo_entrenamiento_s",
        "tiempo_prediccion_s"
    ]

    for col in columnas_redondear:
        if col in resumen_metricas_redondeado.columns:
            resumen_metricas_redondeado[col] = resumen_metricas_redondeado[col].round(4)

    resumen_metricas_redondeado.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_random_forest_final_sin_proxies_redondeado.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL DE MÉTRICAS")
    print("======================================")
    print(resumen_metricas_redondeado.to_string(index=False))

else:
    print("\nNo se generaron modelos válidos.")


if len(tiempos_todos) > 0:

    resumen_tiempos = pd.concat(tiempos_todos, ignore_index=True)

    tiempo_total_script = time.perf_counter() - TIEMPO_INICIO_SCRIPT

    fila_total = pd.DataFrame({
        "modelo": ["TOTAL_SCRIPT"],
        "tiempo_entrenamiento_s": [resumen_tiempos["tiempo_entrenamiento_s"].sum()],
        "tiempo_prediccion_s": [resumen_tiempos["tiempo_prediccion_s"].sum()],
        "tiempo_importancia_permutacion_s": [resumen_tiempos["tiempo_importancia_permutacion_s"].sum(skipna=True)],
        "tiempo_shap_s": [resumen_tiempos["tiempo_shap_s"].sum(skipna=True)],
        "tiempo_total_modelo_s": [tiempo_total_script],
        "tiempo_entrenamiento": [formatear_tiempo(resumen_tiempos["tiempo_entrenamiento_s"].sum())],
        "tiempo_prediccion": [formatear_tiempo(resumen_tiempos["tiempo_prediccion_s"].sum())],
        "tiempo_importancia_permutacion": [
            formatear_tiempo(resumen_tiempos["tiempo_importancia_permutacion_s"].sum(skipna=True))
        ],
        "tiempo_shap": [
            formatear_tiempo(resumen_tiempos["tiempo_shap_s"].sum(skipna=True))
        ],
        "tiempo_total_modelo": [formatear_tiempo(tiempo_total_script)]
    })

    resumen_tiempos = pd.concat(
        [resumen_tiempos, fila_total],
        ignore_index=True
    )

    resumen_tiempos.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_tiempos_random_forest_final_sin_proxies.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL DE TIEMPOS")
    print("======================================")
    print(resumen_tiempos.to_string(index=False))


if len(complejidades_todas) > 0:

    resumen_complejidad = pd.concat(
        complejidades_todas,
        ignore_index=True
    )

    columnas_redondear_complejidad = [
        "operaciones_entrenamiento_aprox",
        "operaciones_prediccion_aprox",
        "operaciones_shap_aprox"
    ]

    for col in columnas_redondear_complejidad:
        resumen_complejidad[col] = resumen_complejidad[col].round(2)

    resumen_complejidad.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_complejidad_temporal_random_forest_final_sin_proxies.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL DE COMPLEJIDAD TEMPORAL")
    print("======================================")
    print(resumen_complejidad.to_string(index=False))


if len(shap_todos) > 0:

    resumen_shap = pd.concat(shap_todos, ignore_index=True)

    resumen_shap.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_shap_random_forest_final_sin_proxies.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL SHAP")
    print("======================================")
    print(resumen_shap.head(30).to_string(index=False))


print("\nArchivos guardados en:")
print(carpeta_salida)

print("\nTiempo total del script:")
print(formatear_tiempo(time.perf_counter() - TIEMPO_INICIO_SCRIPT))

plt.close("all")