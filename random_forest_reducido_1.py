# -*- coding: utf-8 -*-

# ============================================================
# RANDOM FOREST REDUCIDO 1 - COMPLETO SIN PROXIES + SHAP
# Sobrescribe resultados_random_forest_reducido_1
#
# Se eliminan como predictores:
# anio, zona, localidad, estacion
#
# Se mantienen:
# variables actuales + rezagos + medias móviles + máximos móviles
# + deltas + semana_sin + semana_cos + pais
# ============================================================

import os
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

carpeta_entrada = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_final"
archivo_dataset = os.path.join(carpeta_entrada, "datasetT1.csv")

# Esta carpeta sobrescribe el reducido 1
carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_reducido_1"
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


# ============================================================
# 4. Cargar dataset ya procesado
# ============================================================

df_modelo = pd.read_csv(archivo_dataset, encoding="utf-8-sig")

print("\nDataset cargado correctamente:")
print(df_modelo.shape)
print(df_modelo.columns.tolist())


# ============================================================
# 5. Variables predictoras base
# ============================================================

variables_numericas_base = [
    # Variables ambientales actuales
    "temperatura",
    "salinidad",

    # Variables parasitarias actuales
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales",

    # Estacionalidad
    "semana_sin",
    "semana_cos",

    # Parásitos totales
    "parasitos_totales_lag1",
    "parasitos_totales_lag2",
    "parasitos_totales_lag3",
    "parasitos_totales_lag4",
    "parasitos_totales_media_4",
    "parasitos_totales_max_4",
    "delta_parasitos_1",

    # Hembras ovígeras
    "hembras_ovigeras_lag1",
    "hembras_ovigeras_lag2",
    "hembras_ovigeras_lag3",
    "hembras_ovigeras_lag4",
    "hembras_ovigeras_media_4",
    "hembras_ovigeras_max_4",
    "delta_hembras_1",

    # Adultos móviles
    "adultos_moviles_lag1",
    "adultos_moviles_lag2",
    "adultos_moviles_lag3",
    "adultos_moviles_lag4",
    "adultos_moviles_media_4",
    "adultos_moviles_max_4",
    "delta_adultos_1",

    # Juveniles
    "juveniles_lag1",
    "juveniles_lag2",
    "juveniles_lag3",
    "juveniles_lag4",
    "juveniles_media_4",
    "juveniles_max_4",
    "delta_juveniles_1",

    # Temperatura
    "temperatura_lag1",
    "temperatura_lag2",
    "temperatura_lag3",
    "temperatura_lag4",
    "temperatura_media_4",
    "delta_temperatura_1",

    # Salinidad
    "salinidad_lag1",
    "salinidad_lag2",
    "salinidad_lag3",
    "salinidad_lag4",
    "salinidad_media_4",
    "delta_salinidad_1"
]

columnas_necesarias = variables_numericas_base + [
    "pais",
    "zona",
    "localidad",
    "localidad_id",
    "anio",
    "semana",
    "anio_objetivo",
    "semana_objetivo",
    "parasitos_totales_t1"
]

faltantes = [c for c in columnas_necesarias if c not in df_modelo.columns]

if len(faltantes) > 0:
    raise ValueError(f"Faltan columnas necesarias en datasetT1.csv: {faltantes}")


# ============================================================
# 6. Función para entrenar Random Forest
# ============================================================

def entrenar_random_forest(nombre_modelo, datos):

    print("\n======================================")
    print("Modelo:", nombre_modelo)
    print("Observaciones:", datos.shape[0])
    print("======================================")

    # ========================================================
    # Variables usadas
    # Se elimina: anio, zona, localidad, estacion
    # Se mantiene: pais solo si hay más de una categoría
    # ========================================================

    variables_numericas = variables_numericas_base.copy()

    variables_categoricas = [
        col for col in ["pais"]
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
            f"variables_usadas_rf_reducido_1_{nombre_modelo}.csv"
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
        return None, None, None, None

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
    # Random Forest con hiperparámetros seleccionados por GridSearch
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
            f"metricas_rf_reducido_1_{nombre_modelo}.csv"
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
            f"predicciones_rf_reducido_1_{nombre_modelo}.csv"
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
    plt.title(f"RF reducido 1 sin proxies: observado vs predicho ({nombre_modelo})")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"rf_reducido_1_observado_vs_predicho_{nombre_modelo}.png"
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
            f"rf_reducido_1_serie_temporal_{nombre_modelo}.png"
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
                f"importancia_permutacion_rf_reducido_1_{nombre_modelo}.csv"
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
        plt.title(f"Importancia de variables RF reducido 1 ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"rf_reducido_1_importancia_variables_{nombre_modelo}.png"
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
                f"shap_importancia_rf_reducido_1_{nombre_modelo}.csv"
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

        plt.title(f"Importancia SHAP - RF reducido 1 ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_bar_rf_reducido_1_{nombre_modelo}.png"
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

        plt.title(f"SHAP summary plot - RF reducido 1 ({nombre_modelo})")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"shap_beeswarm_rf_reducido_1_{nombre_modelo}.png"
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
            f"modelo_rf_reducido_1_{nombre_modelo}.joblib"
        )
    )

    return metricas, predicciones, importancia_df, shap_df


# ============================================================
# 7. Entrenar modelos
# ============================================================

metricas_todas = []
shap_todos = []

met_conjunto, pred_conjunto, imp_conjunto, shap_conjunto = entrenar_random_forest(
    "conjunto_chile_noruega",
    df_modelo
)

if met_conjunto is not None:
    metricas_todas.append(met_conjunto)

if shap_conjunto is not None:
    shap_conjunto["modelo"] = "conjunto_chile_noruega"
    shap_todos.append(shap_conjunto)


met_chile, pred_chile, imp_chile, shap_chile = entrenar_random_forest(
    "chile",
    df_modelo[df_modelo["pais"] == "Chile"].copy()
)

if met_chile is not None:
    metricas_todas.append(met_chile)

if shap_chile is not None:
    shap_chile["modelo"] = "chile"
    shap_todos.append(shap_chile)


met_noruega, pred_noruega, imp_noruega, shap_noruega = entrenar_random_forest(
    "noruega",
    df_modelo[df_modelo["pais"] == "Noruega"].copy()
)

if met_noruega is not None:
    metricas_todas.append(met_noruega)

if shap_noruega is not None:
    shap_noruega["modelo"] = "noruega"
    shap_todos.append(shap_noruega)


# ============================================================
# 8. Resumen final
# ============================================================

if len(metricas_todas) > 0:

    resumen_metricas = pd.concat(metricas_todas, ignore_index=True)

    resumen_metricas.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_random_forest_reducido_1.csv"
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


if len(shap_todos) > 0:

    resumen_shap = pd.concat(shap_todos, ignore_index=True)

    resumen_shap.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_shap_random_forest_reducido_1.csv"
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