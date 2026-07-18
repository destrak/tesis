# -*- coding: utf-8 -*-

# ============================================================
# RANDOM FOREST REGRESSOR MEJORADO + GRIDSEARCH + SHAP
# Predicción de carga parasitaria de la semana siguiente
# Versión final sin variables proxy:
# NO usa anio, zona, localidad ni estacion como predictoras
# ============================================================

import os
from pathlib import Path
import unicodedata

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
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import joblib

try:
    import shap
except ModuleNotFoundError:
    shap = None
    print("SHAP no está instalado en este entorno. El código correrá sin SHAP.")


# ============================================================
# 1. Rutas
# ============================================================

carpeta_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\dataset\datasetchile"
carpeta_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\dataset\datasetnoruega"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_gridsearch_sin_proxies"
os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. Configuración
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

CALCULAR_IMPORTANCIA = True
CALCULAR_SHAP = shap is not None

N_MUESTRA_SHAP = 300


# ============================================================
# 3. Funciones auxiliares
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


# ============================================================
# 4. Cargar bases
# ============================================================

df_chile = cargar_carpeta(carpeta_chile, "Chile")
df_noruega = cargar_carpeta(carpeta_noruega, "Noruega")

df = pd.concat([df_chile, df_noruega], ignore_index=True)

print("\nColumnas cargadas:")
print(df.columns.tolist())


# ============================================================
# 5. Homologar columnas
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
# 6. Conversión numérica
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
# 7. Identificar centro y zona
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
# 8. Estación
# ============================================================

if "estacion" not in df.columns:
    df["estacion"] = "Sin estación"

df["estacion"] = df["estacion"].astype(str)


# ============================================================
# 9. Semana homologada Chile -> Noruega
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
# 10. Eliminar duplicados por país-centro-año-semana
# ============================================================

print("\nFilas antes de eliminar duplicados:", df.shape[0])

df = df.drop_duplicates(
    subset=["pais", "localidad_id", "anio", "semana"],
    keep="first"
).copy()

print("Filas después de eliminar duplicados:", df.shape[0])


# ============================================================
# 11. Crear fechas
# ============================================================

df = df.sort_values(["pais", "localidad_id", "anio", "semana"]).copy()

df["fecha_t"] = crear_fecha_iso(df["anio"], df["semana"])


# ============================================================
# 12. Crear variable objetivo t+1 real
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

df["fecha_t1"] = crear_fecha_iso(df["anio_objetivo"], df["semana_objetivo"])

df["es_semana_siguiente"] = (
    (df["fecha_t1"] - df["fecha_t"]).dt.days == 7
)


# ============================================================
# 13. Crear rezagos con verificación de continuidad
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
# 14. Medias móviles, máximos y cambios
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
# 15. Comparación de tamaños
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
# 16. Base final del modelo
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
# 17. Función para entrenar Random Forest
# ============================================================

def entrenar_random_forest(nombre_modelo, datos):

    print("\n======================================")
    print("Modelo:", nombre_modelo)
    print("Observaciones:", datos.shape[0])
    print("======================================")

    # --------------------------------------------------------
    # Variables predictoras finales sin proxies
    # --------------------------------------------------------
    # Se excluyen:
    # - anio
    # - zona
    # - localidad
    # - estacion
    #
    # pais se usa únicamente en el modelo conjunto Chile--Noruega.
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
    print("Número de variables predictoras:", len(variables_predictoras))

    # --------------------------------------------------------
    # División temporal entrenamiento/prueba
    # --------------------------------------------------------

    datos_train = datos[datos["anio_objetivo"] <= 2022].copy()
    datos_test = datos[datos["anio_objetivo"] >= 2023].copy()

    print("Entrenamiento total hasta 2022:", datos_train.shape[0])
    print("Prueba desde 2023:", datos_test.shape[0])

    if datos_train.shape[0] == 0 or datos_test.shape[0] == 0:
        print(f"No hay datos suficientes para {nombre_modelo}")
        return None, None, None, None, None

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

    # ========================================================
    # GridSearchCV con validación temporal
    # entrenamiento: año objetivo <= 2021
    # validación: año objetivo == 2022
    # ========================================================

    print("\nBuscando mejores hiperparámetros con GridSearchCV sin proxies...")

    tiene_train_busqueda = (datos_train["anio_objetivo"] <= 2021).sum()
    tiene_validacion = (datos_train["anio_objetivo"] == 2022).sum()

    if tiene_train_busqueda == 0 or tiene_validacion == 0:

        print("\nNo hay datos suficientes para GridSearch temporal.")
        print("Se usará configuración base fija.")

        rf_base = RandomForestRegressor(
            n_estimators=800,
            random_state=42,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            n_jobs=1
        )

        modelo = Pipeline(steps=[
            ("preprocesamiento", preprocesamiento),
            ("random_forest", rf_base)
        ])

        modelo.fit(X_train, y_train)

        mejores_parametros_limpios = {
            "n_estimators": 800,
            "max_depth": None,
            "min_samples_split": 4,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "bootstrap": True
        }

        mejor_rmse_validacion = np.nan

        resultados_grid = pd.DataFrame()

    else:

        test_fold = []

        for anio_obj in datos_train["anio_objetivo"]:
            if anio_obj <= 2021:
                test_fold.append(-1)
            elif anio_obj == 2022:
                test_fold.append(0)
            else:
                test_fold.append(-1)

        ps = PredefinedSplit(test_fold=test_fold)

        rf_base = RandomForestRegressor(
            random_state=42,
            bootstrap=True,
            n_jobs=1
        )

        modelo_base = Pipeline(steps=[
            ("preprocesamiento", preprocesamiento),
            ("random_forest", rf_base)
        ])

        param_grid = {
            "random_forest__n_estimators": [300, 500, 800],
            "random_forest__max_depth": [None, 10, 20],
            "random_forest__min_samples_split": [2, 4, 8],
            "random_forest__min_samples_leaf": [1, 2],
            "random_forest__max_features": ["sqrt", "log2"]
        }

        grid = GridSearchCV(
            estimator=modelo_base,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=ps,
            n_jobs=1,
            verbose=2,
            refit=True
        )

        grid.fit(X_train, y_train)

        print("\nMejores hiperparámetros:")
        print(grid.best_params_)
        print("Mejor RMSE de validación:", -grid.best_score_)

        resultados_grid = pd.DataFrame(grid.cv_results_)

        resultados_grid.to_csv(
            os.path.join(
                carpeta_salida,
                f"gridsearch_rf_sin_proxies_{nombre_modelo}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        modelo = grid.best_estimator_

        mejores_parametros_limpios = {
            "n_estimators": grid.best_params_["random_forest__n_estimators"],
            "max_depth": grid.best_params_["random_forest__max_depth"],
            "min_samples_split": grid.best_params_["random_forest__min_samples_split"],
            "min_samples_leaf": grid.best_params_["random_forest__min_samples_leaf"],
            "max_features": grid.best_params_["random_forest__max_features"],
            "bootstrap": True
        }

        mejor_rmse_validacion = -grid.best_score_

    # ========================================================
    # Evaluación final en prueba
    # ========================================================

    y_pred = modelo.predict(X_test)

    y_base = datos_test["parasitos_totales"].values

    mae_rf = mean_absolute_error(y_test, y_pred)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_rf = r2_score(y_test, y_pred)

    mae_base = mean_absolute_error(y_test, y_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_base))
    r2_base = r2_score(y_test, y_base)

    metricas = pd.DataFrame({
        "modelo": [nombre_modelo],
        "n_entrenamiento": [X_train.shape[0]],
        "n_prueba": [X_test.shape[0]],
        "mejor_RMSE_validacion": [mejor_rmse_validacion],
        "MAE_RF": [mae_rf],
        "RMSE_RF": [rmse_rf],
        "R2_RF": [r2_rf],
        "MAE_base_persistencia": [mae_base],
        "RMSE_base_persistencia": [rmse_base],
        "R2_base_persistencia": [r2_base],
        "n_estimators": [mejores_parametros_limpios["n_estimators"]],
        "max_depth": [mejores_parametros_limpios["max_depth"]],
        "min_samples_split": [mejores_parametros_limpios["min_samples_split"]],
        "min_samples_leaf": [mejores_parametros_limpios["min_samples_leaf"]],
        "max_features": [mejores_parametros_limpios["max_features"]],
        "bootstrap": [mejores_parametros_limpios["bootstrap"]],
        "variables_proxy_excluidas": ["anio, zona, localidad, estacion"]
    })

    print("\nMétricas:")
    print(metricas.to_string(index=False))

    metricas.to_csv(
        os.path.join(
            carpeta_salida,
            f"metricas_rf_sin_proxies_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

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
            f"predicciones_rf_sin_proxies_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # Gráfico observado vs predicho
    # ========================================================

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    limite_max = max(y_test.max(), y_pred.max())

    if limite_max == 0:
        limite_max = 1

    plt.plot([0, limite_max], [0, limite_max], linestyle="--")

    plt.xlabel("Carga parasitaria observada")
    plt.ylabel("Carga parasitaria predicha")
    plt.title(f"Random Forest sin proxies: observado vs predicho ({nombre_modelo})")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"rf_sin_proxies_observado_vs_predicho_{nombre_modelo}.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # Serie temporal observada y predicha
    # ========================================================

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
            f"rf_sin_proxies_serie_temporal_{nombre_modelo}.png"
        ),
        dpi=300
    )

    plt.close()

    # ========================================================
    # Importancia por permutación
    # ========================================================

    importancia_df = None

    if CALCULAR_IMPORTANCIA:

        print("\nCalculando importancia por permutación...")

        importancia = permutation_importance(
            modelo,
            X_test,
            y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=5,
            random_state=42,
            n_jobs=1
        )

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
                f"importancia_permutacion_rf_sin_proxies_{nombre_modelo}.csv"
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
                f"rf_sin_proxies_importancia_variables_{nombre_modelo}.png"
            ),
            dpi=300
        )

        plt.close()

    # ========================================================
    # SHAP values
    # ========================================================

    shap_df = None

    if CALCULAR_SHAP:

        print("\nCalculando SHAP values...")

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
                f"shap_importancia_rf_sin_proxies_{nombre_modelo}.csv"
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
                f"shap_bar_rf_sin_proxies_{nombre_modelo}.png"
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
                f"shap_beeswarm_rf_sin_proxies_{nombre_modelo}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("SHAP values calculados correctamente.")

    # ========================================================
    # Guardar modelo
    # ========================================================

    joblib.dump(
        modelo,
        os.path.join(
            carpeta_salida,
            f"modelo_rf_sin_proxies_{nombre_modelo}.joblib"
        )
    )

    return metricas, predicciones, importancia_df, shap_df, mejores_parametros_limpios


# ============================================================
# 18. Entrenar modelos
# ============================================================

metricas_todas = []
shap_todos = []
hiperparametros_todos = []

met_conjunto, pred_conjunto, imp_conjunto, shap_conjunto, params_conjunto = entrenar_random_forest(
    "conjunto_chile_noruega",
    df_modelo
)

if met_conjunto is not None:
    metricas_todas.append(met_conjunto)

if shap_conjunto is not None:
    shap_conjunto["modelo"] = "conjunto_chile_noruega"
    shap_todos.append(shap_conjunto)

if params_conjunto is not None:
    fila_params = params_conjunto.copy()
    fila_params["modelo"] = "conjunto_chile_noruega"
    hiperparametros_todos.append(fila_params)


met_chile, pred_chile, imp_chile, shap_chile, params_chile = entrenar_random_forest(
    "chile",
    df_modelo[df_modelo["pais"] == "Chile"].copy()
)

if met_chile is not None:
    metricas_todas.append(met_chile)

if shap_chile is not None:
    shap_chile["modelo"] = "chile"
    shap_todos.append(shap_chile)

if params_chile is not None:
    fila_params = params_chile.copy()
    fila_params["modelo"] = "chile"
    hiperparametros_todos.append(fila_params)


met_noruega, pred_noruega, imp_noruega, shap_noruega, params_noruega = entrenar_random_forest(
    "noruega",
    df_modelo[df_modelo["pais"] == "Noruega"].copy()
)

if met_noruega is not None:
    metricas_todas.append(met_noruega)

if shap_noruega is not None:
    shap_noruega["modelo"] = "noruega"
    shap_todos.append(shap_noruega)

if params_noruega is not None:
    fila_params = params_noruega.copy()
    fila_params["modelo"] = "noruega"
    hiperparametros_todos.append(fila_params)


# ============================================================
# 19. Resumen final
# ============================================================

if len(metricas_todas) > 0:

    resumen_metricas = pd.concat(metricas_todas, ignore_index=True)

    resumen_metricas.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_random_forest_sin_proxies_gridsearch.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    resumen_metricas_redondeado = resumen_metricas.copy()

    columnas_redondear = [
        "mejor_RMSE_validacion",
        "MAE_RF",
        "RMSE_RF",
        "R2_RF",
        "MAE_base_persistencia",
        "RMSE_base_persistencia",
        "R2_base_persistencia"
    ]

    for col in columnas_redondear:
        resumen_metricas_redondeado[col] = resumen_metricas_redondeado[col].round(4)

    resumen_metricas_redondeado.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_random_forest_sin_proxies_gridsearch_redondeado.csv"
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


if len(hiperparametros_todos) > 0:

    resumen_hiperparametros = pd.DataFrame(hiperparametros_todos)

    columnas_orden = [
        "modelo",
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap"
    ]

    resumen_hiperparametros = resumen_hiperparametros[columnas_orden]

    resumen_hiperparametros.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_hiperparametros_random_forest_sin_proxies_gridsearch.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("MEJORES HIPERPARÁMETROS SIN PROXIES")
    print("======================================")
    print(resumen_hiperparametros.to_string(index=False))


if len(shap_todos) > 0:

    resumen_shap = pd.concat(shap_todos, ignore_index=True)

    resumen_shap.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_shap_random_forest_sin_proxies_gridsearch.csv"
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

plt.close("all")