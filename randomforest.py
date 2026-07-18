# -*- coding: utf-8 -*-

# ============================================================
# RANDOM FOREST CON VARIABLES ORIGINALES + SHAP AGRUPADO
#
# Objetivo:
# Generar una prueba diagnóstica usando variables originales,
# incluyendo variables proxy:
#
# anio, zona, localidad, estacion
#
# Este modelo NO es el modelo final.
# Sirve para justificar la decisión de excluir variables proxy
# en el modelo final.
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

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_variables_originales"
os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. Configuración general
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


def crear_estacion_desde_semana_homologada(semana):
    """
    Respaldo por si los archivos originales no traen estacion.
    Usa semana homologada para crear estación aproximada.
    """
    if pd.isna(semana):
        return "Sin estación"

    semana = int(semana)

    if 1 <= semana <= 13:
        return "Invierno"
    elif 14 <= semana <= 26:
        return "Primavera"
    elif 27 <= semana <= 39:
        return "Verano"
    elif 40 <= semana <= 52:
        return "Otoño"
    else:
        return "Sin estación"


def agrupar_variable_original(nombre):
    """
    Agrupa variables One-Hot a su variable original.
    Ejemplo:
    zona_Aysén -> zona
    localidad_102424 -> localidad
    estacion_Verano -> estacion
    pais_Chile -> pais
    """
    nombre = str(nombre)

    if nombre.startswith("pais_"):
        return "pais"
    elif nombre.startswith("zona_"):
        return "zona"
    elif nombre.startswith("localidad_"):
        return "localidad"
    elif nombre.startswith("estacion_"):
        return "estacion"
    else:
        return nombre


def tipo_variable(variable):
    proxies = ["anio", "zona", "localidad", "estacion"]

    if variable in proxies:
        return "proxy"
    else:
        return "no_proxy"


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
# 8. Semana homologada Chile -> Noruega
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


# ============================================================
# 9. Estación
# ============================================================

if "estacion" not in df.columns:
    print("\nAdvertencia: no existe columna 'estacion'.")
    print("Se creará estación aproximada usando semana_homologada.")

    df["estacion"] = df["semana_homologada"].apply(
        crear_estacion_desde_semana_homologada
    )

else:
    df["estacion"] = df["estacion"].astype(str).fillna("Sin estación")


print("\nEjemplo de homologación temporal Chile -> Noruega:")
print(
    df[["pais", "anio", "semana", "anio_homologado", "semana_homologada", "estacion"]]
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
# 13. Comparación de tamaños
# ============================================================

columnas_base = [
    "pais",
    "zona",
    "localidad",
    "localidad_id",
    "anio",
    "semana",
    "semana_homologada",
    "estacion",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales",
    "temperatura",
    "salinidad"
]

columnas_rf = columnas_base + [
    "parasitos_totales_t1",
    "anio_objetivo",
    "semana_objetivo"
]

df_rf_sin_continuidad = df.dropna(subset=columnas_rf).copy()

df_rf_con_continuidad = df[
    df["es_semana_siguiente"]
].dropna(
    subset=columnas_rf
).copy()

print("\n======================================")
print("COMPARACIÓN DE TAMAÑOS")
print("======================================")

print("\nRF sin exigir semana siguiente real:")
print(df_rf_sin_continuidad["pais"].value_counts())
print("Total:", df_rf_sin_continuidad.shape[0])

print("\nRF exigiendo semana siguiente real:")
print(df_rf_con_continuidad["pais"].value_counts())
print("Total:", df_rf_con_continuidad.shape[0])


# ============================================================
# 14. Base final del modelo diagnóstico
# ============================================================

df_modelo = df_rf_con_continuidad.copy()

df_modelo["semana_sin"] = np.sin(
    2 * np.pi * df_modelo["semana_homologada"] / 52
)

df_modelo["semana_cos"] = np.cos(
    2 * np.pi * df_modelo["semana_homologada"] / 52
)

df_modelo.to_csv(
    os.path.join(carpeta_salida, "datasetT1_variables_originales.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("\nDataset diagnóstico con variables originales guardado correctamente.")


# ============================================================
# 15. Función para entrenar Random Forest diagnóstico
# ============================================================

def entrenar_random_forest(nombre_modelo, datos):

    print("\n======================================")
    print("Modelo:", nombre_modelo)
    print("Observaciones:", datos.shape[0])
    print("======================================")

    # ========================================================
    # Variables originales para diagnóstico
    # Sin rezagos, sin medias móviles, sin máximos móviles,
    # sin deltas.
    #
    # Incluye variables proxy:
    # anio, zona, localidad, estacion
    # ========================================================

    variables_numericas = [
        "temperatura",
        "salinidad",
        "hembras_ovigeras",
        "adultos_moviles",
        "juveniles",
        "parasitos_totales",
        "anio",
        "semana_sin",
        "semana_cos"
    ]

    variables_categoricas = [
        col for col in ["pais", "zona", "localidad", "estacion"]
        if col in datos.columns and datos[col].nunique() > 1
    ]

    variables_predictoras = variables_numericas + variables_categoricas

    print("\nVariables numéricas usadas:")
    print(variables_numericas)

    print("\nVariables categóricas usadas:")
    print(variables_categoricas)

    pd.DataFrame({
        "variable": variables_predictoras
    }).to_csv(
        os.path.join(
            carpeta_salida,
            f"variables_usadas_rf_variables_originales_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # División temporal
    # ========================================================

    datos_train = datos[datos["anio_objetivo"] <= 2022].copy()
    datos_test = datos[datos["anio_objetivo"] >= 2023].copy()

    print("Entrenamiento:", datos_train.shape[0])
    print("Prueba:", datos_test.shape[0])

    if datos_train.shape[0] == 0 or datos_test.shape[0] == 0:
        print(f"No hay datos suficientes para {nombre_modelo}")
        return None, None, None, None, None

    X_train = datos_train[variables_predictoras].copy()
    y_train = datos_train["parasitos_totales_t1"].copy()

    X_test = datos_test[variables_predictoras].copy()
    y_test = datos_test["parasitos_totales_t1"].copy()

    # ========================================================
    # Preprocesamiento
    # ========================================================

    transformadores = [
        (
            "num",
            SimpleImputer(strategy="median"),
            variables_numericas
        )
    ]

    if len(variables_categoricas) > 0:
        transformadores.append(
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
        transformers=transformadores,
        remainder="drop"
    )

    # ========================================================
    # Random Forest con configuración final GridSearchCV
    # ========================================================

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=2,
        max_features="log2",
        bootstrap=True,
        n_jobs=1
    )

    modelo = Pipeline(steps=[
        ("preprocesamiento", preprocesamiento),
        ("random_forest", rf)
    ])

    modelo.fit(X_train, y_train)

    # ========================================================
    # Predicción y métricas
    # ========================================================

    y_pred = modelo.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

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
        "n_variables_predictoras": [len(variables_predictoras)],
        "MAE_RF": [mae_rf],
        "RMSE_RF": [rmse_rf],
        "R2_RF": [r2_rf],
        "MAE_base_persistencia": [mae_base],
        "RMSE_base_persistencia": [rmse_base],
        "R2_base_persistencia": [r2_base],
        "n_estimators": [300],
        "max_depth": [20],
        "min_samples_split": [2],
        "min_samples_leaf": [2],
        "max_features": ["log2"],
        "bootstrap": [True]
    })

    print("\nMétricas:")
    print(metricas.to_string(index=False))

    metricas.to_csv(
        os.path.join(
            carpeta_salida,
            f"metricas_rf_variables_originales_{nombre_modelo}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # Guardar predicciones
    # ========================================================

    predicciones = datos_test[[
        "pais",
        "zona",
        "localidad",
        "localidad_id",
        "anio",
        "semana",
        "estacion",
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

    predicciones["error_abs_rf"] = np.abs(predicciones["error_rf"])

    predicciones.to_csv(
        os.path.join(
            carpeta_salida,
            f"predicciones_rf_variables_originales_{nombre_modelo}.csv"
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
    plt.plot([0, limite_max], [0, limite_max], linestyle="--")

    plt.xlabel("Carga parasitaria observada")
    plt.ylabel("Carga parasitaria predicha")
    plt.title(f"RF variables originales: observado vs predicho ({nombre_modelo})")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"rf_variables_originales_observado_vs_predicho_{nombre_modelo}.png"
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
            f"rf_variables_originales_serie_temporal_{nombre_modelo}.png"
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
        }).sort_values("importancia_media", ascending=False)

        importancia_df.to_csv(
            os.path.join(
                carpeta_salida,
                f"importancia_permutacion_rf_variables_originales_{nombre_modelo}.csv"
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
        plt.title(f"Importancia por permutación - variables originales ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"rf_variables_originales_importancia_permutacion_{nombre_modelo}.png"
            ),
            dpi=300
        )

        plt.close()

    # ========================================================
    # SHAP values y SHAP agrupado por variable original
    # ========================================================

    shap_df = None
    shap_df_agrupado = None

    if CALCULAR_SHAP:

        print("\nCalculando SHAP values...")

        preprocesador = modelo.named_steps["preprocesamiento"]
        rf_entrenado = modelo.named_steps["random_forest"]

        X_test_prep = preprocesador.transform(X_test)

        if hasattr(X_test_prep, "toarray"):
            X_test_prep = X_test_prep.toarray()

        nombres_variables = obtener_nombres_variables(preprocesador)

        if nombres_variables is None:
            nombres_variables = [f"var_{i}" for i in range(X_test_prep.shape[1])]

        X_test_prep_df = pd.DataFrame(
            X_test_prep,
            columns=nombres_variables
        )

        n_muestra_shap = min(N_MUESTRA_SHAP, X_test_prep_df.shape[0])

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
                f"shap_importancia_rf_variables_originales_{nombre_modelo}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        # ----------------------------------------------------
        # Gráfico SHAP normal, con variables One-Hot separadas
        # ----------------------------------------------------

        shap.summary_plot(
            shap_values,
            X_shap,
            plot_type="bar",
            max_display=20,
            show=False
        )

        plt.title(f"Importancia SHAP - variables originales ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_bar_rf_variables_originales_{nombre_modelo}.png"
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

        plt.title(f"SHAP summary plot - variables originales ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_beeswarm_rf_variables_originales_{nombre_modelo}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------------------------
        # SHAP agrupado por variable original
        # ----------------------------------------------------

        shap_df_agrupado = shap_df.copy()

        shap_df_agrupado["variable_original"] = (
            shap_df_agrupado["variable"].apply(agrupar_variable_original)
        )

        shap_df_agrupado = (
            shap_df_agrupado
            .groupby("variable_original", as_index=False)["shap_importancia_media_abs"]
            .sum()
            .sort_values("shap_importancia_media_abs", ascending=False)
        )

        shap_df_agrupado["tipo"] = shap_df_agrupado["variable_original"].apply(
            tipo_variable
        )

        shap_df_agrupado.to_csv(
            os.path.join(
                carpeta_salida,
                f"shap_importancia_agrupada_variables_originales_{nombre_modelo}.csv"
            ),
            index=False,
            encoding="utf-8-sig"
        )

        shap_plot = shap_df_agrupado.head(20).sort_values(
            "shap_importancia_media_abs"
        )

        colores = [
            "tomato" if v in ["anio", "zona", "localidad", "estacion"] else "steelblue"
            for v in shap_plot["variable_original"]
        ]

        plt.figure(figsize=(8, 7))
        plt.barh(
            shap_plot["variable_original"],
            shap_plot["shap_importancia_media_abs"],
            color=colores
        )

        plt.xlabel("Importancia SHAP media absoluta")
        plt.ylabel("Variable original")
        plt.title(f"SHAP agrupado por variable original ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_bar_agrupado_variables_originales_{nombre_modelo}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # ----------------------------------------------------
        # Gráfico específico solo de variables proxy
        # ----------------------------------------------------

        shap_proxy = shap_df_agrupado[
            shap_df_agrupado["variable_original"].isin(
                ["anio", "zona", "localidad", "estacion"]
            )
        ].copy()

        if shap_proxy.shape[0] > 0:

            shap_proxy = shap_proxy.sort_values("shap_importancia_media_abs")

            plt.figure(figsize=(7, 5))
            plt.barh(
                shap_proxy["variable_original"],
                shap_proxy["shap_importancia_media_abs"],
                color="tomato"
            )

            plt.xlabel("Importancia SHAP media absoluta")
            plt.ylabel("Variable proxy")
            plt.title(f"Importancia SHAP de variables proxy ({nombre_modelo})")
            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    carpeta_salida,
                    f"shap_variables_proxy_{nombre_modelo}.png"
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
            f"modelo_rf_variables_originales_{nombre_modelo}.joblib"
        )
    )

    return metricas, predicciones, importancia_df, shap_df, shap_df_agrupado


# ============================================================
# 16. Entrenar modelos
# ============================================================

metricas_todas = []
shap_todos = []
shap_agrupado_todos = []

met_conjunto, pred_conjunto, imp_conjunto, shap_conjunto, shap_agrupado_conjunto = entrenar_random_forest(
    "conjunto_chile_noruega",
    df_modelo
)

if met_conjunto is not None:
    metricas_todas.append(met_conjunto)

if shap_conjunto is not None:
    shap_conjunto["modelo"] = "conjunto_chile_noruega"
    shap_todos.append(shap_conjunto)

if shap_agrupado_conjunto is not None:
    shap_agrupado_conjunto["modelo"] = "conjunto_chile_noruega"
    shap_agrupado_todos.append(shap_agrupado_conjunto)


met_chile, pred_chile, imp_chile, shap_chile, shap_agrupado_chile = entrenar_random_forest(
    "chile",
    df_modelo[df_modelo["pais"] == "Chile"].copy()
)

if met_chile is not None:
    metricas_todas.append(met_chile)

if shap_chile is not None:
    shap_chile["modelo"] = "chile"
    shap_todos.append(shap_chile)

if shap_agrupado_chile is not None:
    shap_agrupado_chile["modelo"] = "chile"
    shap_agrupado_todos.append(shap_agrupado_chile)


met_noruega, pred_noruega, imp_noruega, shap_noruega, shap_agrupado_noruega = entrenar_random_forest(
    "noruega",
    df_modelo[df_modelo["pais"] == "Noruega"].copy()
)

if met_noruega is not None:
    metricas_todas.append(met_noruega)

if shap_noruega is not None:
    shap_noruega["modelo"] = "noruega"
    shap_todos.append(shap_noruega)

if shap_agrupado_noruega is not None:
    shap_agrupado_noruega["modelo"] = "noruega"
    shap_agrupado_todos.append(shap_agrupado_noruega)


# ============================================================
# 17. Resumen final de métricas
# ============================================================

if len(metricas_todas) > 0:

    resumen_metricas = pd.concat(metricas_todas, ignore_index=True)

    resumen_metricas.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_random_forest_variables_originales.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL DE MÉTRICAS")
    print("======================================")
    print(resumen_metricas.to_string(index=False))

else:
    print("\nNo se generaron modelos válidos.")


# ============================================================
# 18. Resumen SHAP normal
# ============================================================

if len(shap_todos) > 0:

    resumen_shap = pd.concat(shap_todos, ignore_index=True)

    resumen_shap.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_shap_random_forest_variables_originales.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL SHAP")
    print("======================================")
    print(resumen_shap.head(30).to_string(index=False))


# ============================================================
# 19. Resumen SHAP agrupado + heatmap
# ============================================================

if len(shap_agrupado_todos) > 0:

    resumen_shap_agrupado = pd.concat(
        shap_agrupado_todos,
        ignore_index=True
    )

    resumen_shap_agrupado.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_shap_agrupado_variables_originales.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN SHAP AGRUPADO VARIABLES ORIGINALES")
    print("======================================")
    print(resumen_shap_agrupado.head(40).to_string(index=False))

    # --------------------------------------------------------
    # Heatmap SHAP agrupado normalizado
    # --------------------------------------------------------

    tabla_heatmap = resumen_shap_agrupado.pivot_table(
        index="variable_original",
        columns="modelo",
        values="shap_importancia_media_abs",
        aggfunc="sum",
        fill_value=0
    )

    orden_variables = (
        tabla_heatmap.mean(axis=1)
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    tabla_heatmap = tabla_heatmap.loc[orden_variables]

    tabla_heatmap_norm = tabla_heatmap.copy()

    for col in tabla_heatmap_norm.columns:
        max_col = tabla_heatmap_norm[col].max()
        if max_col > 0:
            tabla_heatmap_norm[col] = tabla_heatmap_norm[col] / max_col

    tabla_heatmap_norm.to_csv(
        os.path.join(
            carpeta_salida,
            "heatmap_shap_agrupado_variables_originales_normalizado.csv"
        ),
        encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(
        figsize=(9, max(5, 0.38 * tabla_heatmap_norm.shape[0]))
    )

    im = ax.imshow(tabla_heatmap_norm.values, aspect="auto", cmap="Blues")

    ax.set_xticks(np.arange(tabla_heatmap_norm.shape[1]))
    ax.set_yticks(np.arange(tabla_heatmap_norm.shape[0]))

    ax.set_xticklabels(tabla_heatmap_norm.columns, rotation=45, ha="right")
    ax.set_yticklabels(tabla_heatmap_norm.index)

    ax.set_title("Importancia SHAP agrupada por variable original")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("Variable original")

    for i in range(tabla_heatmap_norm.shape[0]):
        for j in range(tabla_heatmap_norm.shape[1]):
            valor = tabla_heatmap_norm.iloc[i, j]
            ax.text(
                j,
                i,
                f"{valor:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Importancia SHAP normalizada")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            "heatmap_shap_agrupado_variables_originales.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Porcentaje de importancia asociado a variables proxy
    # --------------------------------------------------------

    resumen_proxy = (
        resumen_shap_agrupado
        .assign(es_proxy=lambda x: x["variable_original"].isin(
            ["anio", "zona", "localidad", "estacion"]
        ))
        .groupby(["modelo", "es_proxy"], as_index=False)["shap_importancia_media_abs"]
        .sum()
    )

    total_por_modelo = (
        resumen_proxy
        .groupby("modelo", as_index=False)["shap_importancia_media_abs"]
        .sum()
        .rename(columns={"shap_importancia_media_abs": "shap_total"})
    )

    proxy_por_modelo = resumen_proxy[
        resumen_proxy["es_proxy"] == True
    ].rename(
        columns={"shap_importancia_media_abs": "shap_proxy"}
    )[["modelo", "shap_proxy"]]

    resumen_porcentaje_proxy = total_por_modelo.merge(
        proxy_por_modelo,
        on="modelo",
        how="left"
    )

    resumen_porcentaje_proxy["shap_proxy"] = resumen_porcentaje_proxy[
        "shap_proxy"
    ].fillna(0)

    resumen_porcentaje_proxy["porcentaje_proxy"] = (
        100
        * resumen_porcentaje_proxy["shap_proxy"]
        / resumen_porcentaje_proxy["shap_total"]
    )

    resumen_porcentaje_proxy.to_csv(
        os.path.join(
            carpeta_salida,
            "porcentaje_importancia_shap_variables_proxy.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    plt.figure(figsize=(7, 5))
    plt.bar(
        resumen_porcentaje_proxy["modelo"],
        resumen_porcentaje_proxy["porcentaje_proxy"],
        color="tomato"
    )

    plt.ylabel("Importancia SHAP de variables proxy (%)")
    plt.xlabel("Modelo")
    plt.title("Porcentaje de importancia asociado a variables proxy")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            "porcentaje_importancia_shap_variables_proxy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


print("\nArchivos guardados en:")
print(carpeta_salida)

print("\nImágenes clave para justificar la decisión:")
print("1) shap_bar_agrupado_variables_originales_conjunto_chile_noruega.png")
print("2) shap_variables_proxy_conjunto_chile_noruega.png")
print("3) heatmap_shap_agrupado_variables_originales.png")
print("4) porcentaje_importancia_shap_variables_proxy.png")

plt.close("all")