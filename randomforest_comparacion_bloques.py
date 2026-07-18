# -*- coding: utf-8 -*-

# ============================================================
# COMPARACIÓN POR BLOQUES DE VARIABLES
# RANDOM FOREST REGRESSOR
# Predicción de carga parasitaria t+1
# Sin uso de to_latex ni jinja2
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


# ============================================================
# 1. Rutas
# ============================================================

ruta_dataset = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_mejorado\dataset_random_forest_mejorado_t1.csv"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\analisis_bloques_random_forest"
os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. Configuración
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

CALCULAR_IMPORTANCIA = True


# ============================================================
# 3. Cargar dataset
# ============================================================

df = pd.read_csv(ruta_dataset, encoding="utf-8-sig")

print("Dataset cargado:")
print(df.shape)
print(df.columns.tolist())


# ============================================================
# 4. Asegurar tipos de variables
# ============================================================

variables_numericas_forzar = [
    "temperatura",
    "salinidad",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales",
    "parasitos_totales_t1",
    "anio",
    "semana",
    "semana_homologada",
    "anio_objetivo",
    "semana_objetivo",
    "semana_sin",
    "semana_cos"
]

for col in variables_numericas_forzar:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["pais", "zona", "localidad", "estacion"]:
    if col in df.columns:
        df[col] = df[col].astype(str)


# ============================================================
# 5. Definir bloques de variables
# ============================================================

variables_ambientales_temporales = [
    "temperatura",
    "salinidad",
    "anio",
    "semana_sin",
    "semana_cos"
]

variables_parasitarias_actuales = [
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales"
]

variables_historicas = [
    col for col in df.columns
    if (
        ("_lag" in col or "_media_4" in col or "_max_4" in col or col.startswith("delta_"))
        and not col.startswith("fecha")
    )
]

variables_categoricas_base = [
    col for col in ["pais", "zona", "localidad", "estacion"]
    if col in df.columns
]

bloques = {
    "ambiental_temporal_territorial": {
        "numericas": variables_ambientales_temporales,
        "categoricas": variables_categoricas_base
    },

    "ambiental_temporal_mas_carga_actual": {
        "numericas": variables_ambientales_temporales + variables_parasitarias_actuales,
        "categoricas": variables_categoricas_base
    },

    "modelo_completo_historico": {
        "numericas": (
            variables_ambientales_temporales +
            variables_parasitarias_actuales +
            variables_historicas
        ),
        "categoricas": variables_categoricas_base
    }
}

# Eliminar variables no existentes o duplicadas
for nombre_bloque in bloques:
    bloques[nombre_bloque]["numericas"] = list(dict.fromkeys([
        v for v in bloques[nombre_bloque]["numericas"]
        if v in df.columns
    ]))

    bloques[nombre_bloque]["categoricas"] = list(dict.fromkeys([
        v for v in bloques[nombre_bloque]["categoricas"]
        if v in df.columns
    ]))

print("\nBloques definidos:")
for nombre, vars_bloque in bloques.items():
    print("\n", nombre)
    print("Numéricas:", len(vars_bloque["numericas"]))
    print("Categóricas:", vars_bloque["categoricas"])


# ============================================================
# 6. Métricas
# ============================================================

def calcular_metricas(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return mae, rmse, r2


# ============================================================
# 7. Crear modelo Random Forest
# ============================================================

def crear_pipeline_rf(variables_numericas, variables_categoricas):

    transformadores = []

    if len(variables_numericas) > 0:
        transformadores.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                variables_numericas
            )
        )

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

    rf = RandomForestRegressor(
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
        ("random_forest", rf)
    ])

    return modelo


# ============================================================
# 8. Función para entrenar por bloque
# ============================================================

def entrenar_bloque(
    nombre_base,
    datos,
    nombre_bloque,
    variables_numericas,
    variables_categoricas
):

    print("\n======================================")
    print("Base:", nombre_base)
    print("Bloque:", nombre_bloque)
    print("Observaciones:", datos.shape[0])
    print("======================================")

    variables_predictoras = variables_numericas + variables_categoricas

    datos_train = datos[datos["anio_objetivo"] <= 2022].copy()
    datos_test = datos[datos["anio_objetivo"] >= 2023].copy()

    print("Entrenamiento:", datos_train.shape[0])
    print("Prueba:", datos_test.shape[0])

    X_train = datos_train[variables_predictoras].copy()
    y_train = datos_train["parasitos_totales_t1"].copy()

    X_test = datos_test[variables_predictoras].copy()
    y_test = datos_test["parasitos_totales_t1"].copy()

    modelo = crear_pipeline_rf(variables_numericas, variables_categoricas)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    mae_rf, rmse_rf, r2_rf = calcular_metricas(y_test, y_pred)

    predicciones = datos_test[[
        "pais",
        "zona",
        "localidad",
        "anio",
        "semana",
        "anio_objetivo",
        "semana_objetivo",
        "parasitos_totales",
        "parasitos_totales_t1"
    ]].copy()

    predicciones["prediccion_rf"] = y_pred
    predicciones["error_rf"] = (
        predicciones["parasitos_totales_t1"] -
        predicciones["prediccion_rf"]
    )
    predicciones["error_abs_rf"] = np.abs(predicciones["error_rf"])

    predicciones.to_csv(
        os.path.join(
            carpeta_salida,
            f"predicciones_{nombre_base}_{nombre_bloque}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    metricas = {
        "base": nombre_base,
        "modelo": nombre_bloque,
        "n_entrenamiento": X_train.shape[0],
        "n_prueba": X_test.shape[0],
        "n_variables_numericas": len(variables_numericas),
        "n_variables_categoricas": len(variables_categoricas),
        "MAE": mae_rf,
        "RMSE": rmse_rf,
        "R2": r2_rf
    }

    print(pd.DataFrame([metricas]).to_string(index=False))

    joblib.dump(
        modelo,
        os.path.join(
            carpeta_salida,
            f"modelo_{nombre_base}_{nombre_bloque}.joblib"
        )
    )

    importancia_df = None

    if CALCULAR_IMPORTANCIA and nombre_bloque == "modelo_completo_historico":

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
                f"importancia_{nombre_base}_{nombre_bloque}.csv"
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
        plt.title(f"Importancia de variables: {nombre_base}")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                carpeta_salida,
                f"importancia_{nombre_base}_{nombre_bloque}.png"
            ),
            dpi=300
        )

        plt.close()

    return metricas, predicciones, importancia_df


# ============================================================
# 9. Modelo base de persistencia
# ============================================================

def calcular_persistencia(nombre_base, datos):

    datos_test = datos[datos["anio_objetivo"] >= 2023].copy()

    y_test = datos_test["parasitos_totales_t1"].copy()
    y_base = datos_test["parasitos_totales"].copy()

    mae, rmse, r2 = calcular_metricas(y_test, y_base)

    metricas = {
        "base": nombre_base,
        "modelo": "persistencia",
        "n_entrenamiento": np.nan,
        "n_prueba": datos_test.shape[0],
        "n_variables_numericas": 1,
        "n_variables_categoricas": 0,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

    print("\nPersistencia:", nombre_base)
    print(pd.DataFrame([metricas]).to_string(index=False))

    return metricas


# ============================================================
# 10. Entrenar modelos por base
# ============================================================

bases = {
    "conjunto_chile_noruega": df.copy(),
    "chile": df[df["pais"] == "Chile"].copy(),
    "noruega": df[df["pais"] == "Noruega"].copy()
}

metricas_todas = []

for nombre_base, datos_base in bases.items():

    # Modelo base de persistencia
    metricas_todas.append(
        calcular_persistencia(nombre_base, datos_base)
    )

    # Random Forest por bloques
    for nombre_bloque, vars_bloque in bloques.items():

        metricas, predicciones, importancia = entrenar_bloque(
            nombre_base=nombre_base,
            datos=datos_base,
            nombre_bloque=nombre_bloque,
            variables_numericas=vars_bloque["numericas"],
            variables_categoricas=vars_bloque["categoricas"]
        )

        metricas_todas.append(metricas)


# ============================================================
# 11. Guardar resumen de métricas
# ============================================================

resumen_metricas = pd.DataFrame(metricas_todas)

resumen_metricas.to_csv(
    os.path.join(
        carpeta_salida,
        "resumen_comparacion_bloques_rf.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\n======================================")
print("RESUMEN COMPARACIÓN POR BLOQUES")
print("======================================")
print(resumen_metricas.to_string(index=False))


# ============================================================
# 12. Gráficos comparativos por base
# ============================================================

for nombre_base in resumen_metricas["base"].unique():

    tabla = resumen_metricas[
        resumen_metricas["base"] == nombre_base
    ].copy()

    # Gráfico R2
    plt.figure(figsize=(10, 5))
    plt.bar(tabla["modelo"], tabla["R2"])
    plt.ylabel(r"$R^2$")
    plt.xlabel("Modelo")
    plt.title(f"Comparación de $R^2$ por bloque - {nombre_base}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"comparacion_R2_{nombre_base}.png"
        ),
        dpi=300
    )

    plt.close()

    # Gráfico MAE
    plt.figure(figsize=(10, 5))
    plt.bar(tabla["modelo"], tabla["MAE"])
    plt.ylabel("MAE")
    plt.xlabel("Modelo")
    plt.title(f"Comparación de MAE por bloque - {nombre_base}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"comparacion_MAE_{nombre_base}.png"
        ),
        dpi=300
    )

    plt.close()

    # Gráfico RMSE
    plt.figure(figsize=(10, 5))
    plt.bar(tabla["modelo"], tabla["RMSE"])
    plt.ylabel("RMSE")
    plt.xlabel("Modelo")
    plt.title(f"Comparación de RMSE por bloque - {nombre_base}")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"comparacion_RMSE_{nombre_base}.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 13. Guardar tabla redondeada en CSV
# ============================================================

tabla_redondeada = resumen_metricas.copy()

tabla_redondeada["MAE"] = tabla_redondeada["MAE"].round(3)
tabla_redondeada["RMSE"] = tabla_redondeada["RMSE"].round(3)
tabla_redondeada["R2"] = tabla_redondeada["R2"].round(3)

tabla_redondeada.to_csv(
    os.path.join(
        carpeta_salida,
        "tabla_comparacion_bloques_rf_redondeada.csv"
    ),
    index=False,
    encoding="utf-8-sig"
)

print("\nTabla redondeada guardada correctamente.")


# ============================================================
# 14. Final
# ============================================================

print("\n======================================")
print("ANÁLISIS FINALIZADO")
print("Archivos guardados en:")
print(carpeta_salida)
print("======================================")

plt.close("all")