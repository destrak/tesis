# -*- coding: utf-8 -*-

# ============================================================
# MODELO LSTM CON AJUSTE DE HIPERPARÁMETROS
# Predicción de carga parasitaria total t+1
# Sin variables proxy: anio, zona, localidad, estacion
# Validación temporal: <=2021 entrenamiento, 2022 validación, >=2023 prueba
# ============================================================

import os
import re
import itertools
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
except ImportError:
    raise ImportError(
        "No se pudo importar TensorFlow. Instálalo con: pip install tensorflow "
        "o ejecuta este script en Google Colab."
    )


# ============================================================
# 1. Rutas
# ============================================================

ruta_dataset = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_mejorado\datasetT1.csv"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_lstm_hiperparametros_sin_proxies"
os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. Configuración general
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

tf.random.set_seed(42)
np.random.seed(42)

EPOCHS = 200
PATIENCE = 25

# Grilla de hiperparámetros
GRID_HIPERPARAMETROS = {
    "ventana": [4, 8, 12],
    "lstm_units": [16, 32, 64],
    "dropout": [0.10, 0.20, 0.30],
    "dense_units": [8, 16],
    "learning_rate": [0.001, 0.0005],
    "batch_size": [16, 32]
}


# ============================================================
# 3. Funciones auxiliares
# ============================================================

def limpiar_nombre(texto):
    texto = str(texto).lower()
    texto = texto.replace(" ", "_")
    texto = re.sub(r"[^a-zA-Z0-9_]+", "", texto)
    return texto


def crear_fecha_iso(anio, semana):
    return pd.to_datetime(
        anio.astype("Int64").astype(str)
        + "-W"
        + semana.astype("Int64").astype(str).str.zfill(2)
        + "-1",
        format="%G-W%V-%u",
        errors="coerce"
    )


def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    return mae, rmse, r2


def imputar_y_escalar_3d(X_train, X_val=None, X_test=None):
    """
    Imputa NaN con mediana calculada solo en entrenamiento y luego escala con StandardScaler.
    Evita fuga de información desde validación o prueba.
    """

    n_features = X_train.shape[2]

    X_train_2d = X_train.reshape(-1, n_features)

    medianas = np.nanmedian(X_train_2d, axis=0)
    medianas = np.where(np.isnan(medianas), 0, medianas)

    def imputar(X):
        X_2d = X.reshape(-1, n_features)
        inds = np.where(np.isnan(X_2d))
        X_2d[inds] = np.take(medianas, inds[1])
        return X_2d.reshape(X.shape)

    X_train_imp = imputar(X_train.copy())

    scaler = StandardScaler()
    scaler.fit(X_train_imp.reshape(-1, n_features))

    X_train_s = scaler.transform(
        X_train_imp.reshape(-1, n_features)
    ).reshape(X_train.shape)

    salida = [X_train_s, scaler, medianas]

    if X_val is not None:
        X_val_imp = imputar(X_val.copy())
        X_val_s = scaler.transform(
            X_val_imp.reshape(-1, n_features)
        ).reshape(X_val.shape)
        salida.append(X_val_s)

    if X_test is not None:
        X_test_imp = imputar(X_test.copy())
        X_test_s = scaler.transform(
            X_test_imp.reshape(-1, n_features)
        ).reshape(X_test.shape)
        salida.append(X_test_s)

    return salida


def crear_modelo_lstm(ventana, n_features, lstm_units, dropout, dense_units, learning_rate):

    modelo = Sequential([
        Input(shape=(ventana, n_features)),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dense(1)
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return modelo


# ============================================================
# 4. Cargar dataset
# ============================================================

df = pd.read_csv(ruta_dataset, encoding="utf-8-sig")

print("Dataset cargado:")
print(df.shape)
print(df.columns.tolist())


# ============================================================
# 5. Asegurar tipos de datos
# ============================================================

columnas_numericas = [
    "anio",
    "semana",
    "anio_objetivo",
    "semana_objetivo",
    "temperatura",
    "salinidad",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales",
    "parasitos_totales_t1"
]

for col in columnas_numericas:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in ["pais", "zona", "localidad", "estacion"]:
    if col in df.columns:
        df[col] = df[col].astype(str)


# ============================================================
# 6. Homologación temporal Chile -> Noruega
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

df["semana_sin"] = np.sin(
    2 * np.pi * df["semana_homologada"] / 52
)

df["semana_cos"] = np.cos(
    2 * np.pi * df["semana_homologada"] / 52
)


# ============================================================
# 7. Crear fechas reales para secuencias
# ============================================================

df["fecha_t"] = crear_fecha_iso(df["anio"], df["semana"])
df["fecha_objetivo"] = crear_fecha_iso(
    df["anio_objetivo"],
    df["semana_objetivo"]
)

df = df.dropna(
    subset=[
        "fecha_t",
        "fecha_objetivo",
        "parasitos_totales_t1",
        "pais",
        "localidad"
    ]
).copy()


# ============================================================
# 8. Variables de entrada sin proxies
# ============================================================
# Se eliminan como predictoras:
# anio, zona, localidad, estacion
# pais se usa solo en el modelo conjunto Chile--Noruega

variables_numericas_lstm = [
    "temperatura",
    "salinidad",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles",
    "parasitos_totales",
    "semana_sin",
    "semana_cos"
]


def preparar_base_lstm(datos, nombre_base):

    datos = datos.copy()

    variables_entrada = variables_numericas_lstm.copy()

    # pais solo se incorpora en el modelo conjunto
    if nombre_base == "conjunto_chile_noruega":
        dummies_pais = pd.get_dummies(datos["pais"], prefix="pais", drop_first=False)
        datos = pd.concat([datos, dummies_pais], axis=1)
        variables_entrada += dummies_pais.columns.tolist()

    variables_entrada = [
        v for v in variables_entrada
        if v in datos.columns
    ]

    return datos, variables_entrada


# ============================================================
# 9. Crear secuencias para LSTM
# ============================================================

def crear_secuencias_lstm(datos, ventana, variables_entrada):

    datos = datos.sort_values(
        ["pais", "localidad", "fecha_t"]
    ).copy()

    X_lista = []
    y_lista = []
    persistencia_lista = []
    meta_lista = []

    grupos = datos.groupby(["pais", "localidad"])

    for (pais, localidad), grupo in grupos:

        grupo = grupo.sort_values("fecha_t").reset_index(drop=True)

        for i in range(ventana - 1, len(grupo)):

            ventana_df = grupo.iloc[i - ventana + 1:i + 1].copy()

            # Verificar continuidad semanal dentro de la ventana
            diferencias = ventana_df["fecha_t"].diff().dropna().dt.days

            if len(diferencias) > 0 and not (diferencias == 7).all():
                continue

            fila_actual = grupo.iloc[i]

            if pd.isna(fila_actual["parasitos_totales_t1"]):
                continue

            X = ventana_df[variables_entrada].values.astype(float)
            y = float(fila_actual["parasitos_totales_t1"])
            y_persistencia = float(fila_actual["parasitos_totales"])

            X_lista.append(X)
            y_lista.append(y)
            persistencia_lista.append(y_persistencia)

            meta_lista.append({
                "pais": fila_actual["pais"],
                "zona": fila_actual["zona"] if "zona" in fila_actual else "",
                "localidad": fila_actual["localidad"],
                "anio": fila_actual["anio"],
                "semana": fila_actual["semana"],
                "anio_objetivo": fila_actual["anio_objetivo"],
                "semana_objetivo": fila_actual["semana_objetivo"],
                "fecha_objetivo": fila_actual["fecha_objetivo"],
                "parasitos_totales": fila_actual["parasitos_totales"],
                "parasitos_totales_t1": fila_actual["parasitos_totales_t1"]
            })

    X_array = np.array(X_lista)
    y_array = np.array(y_lista)
    persistencia_array = np.array(persistencia_lista)
    meta = pd.DataFrame(meta_lista)

    if len(meta) > 0:
        orden = np.argsort(meta["fecha_objetivo"].values.astype("datetime64[ns]"))
        X_array = X_array[orden]
        y_array = y_array[orden]
        persistencia_array = persistencia_array[orden]
        meta = meta.iloc[orden].reset_index(drop=True)

    return X_array, y_array, persistencia_array, meta


# ============================================================
# 10. Evaluar una combinación de hiperparámetros
# ============================================================

def evaluar_combinacion(nombre_base, datos, variables_entrada, params):

    ventana = params["ventana"]

    X, y, y_persistencia, meta = crear_secuencias_lstm(
        datos=datos,
        ventana=ventana,
        variables_entrada=variables_entrada
    )

    if X.shape[0] == 0:
        return None

    mask_train = meta["anio_objetivo"] <= 2021
    mask_val = meta["anio_objetivo"] == 2022
    mask_test = meta["anio_objetivo"] >= 2023

    X_train = X[mask_train.values]
    y_train = y[mask_train.values]

    X_val = X[mask_val.values]
    y_val = y[mask_val.values]

    X_test = X[mask_test.values]
    y_test = y[mask_test.values]

    if X_train.shape[0] < 30 or X_val.shape[0] < 10 or X_test.shape[0] < 10:
        return None

    X_train_s, scaler, medianas, X_val_s, X_test_s = imputar_y_escalar_3d(
        X_train,
        X_val,
        X_test
    )

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    modelo = crear_modelo_lstm(
        ventana=ventana,
        n_features=X_train_s.shape[2],
        lstm_units=params["lstm_units"],
        dropout=params["dropout"],
        dense_units=params["dense_units"],
        learning_rate=params["learning_rate"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    )

    historial = modelo.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS,
        batch_size=params["batch_size"],
        callbacks=[early_stop],
        verbose=0
    )

    y_val_pred = modelo.predict(X_val_s, verbose=0).flatten()
    y_val_pred = np.clip(y_val_pred, 0, None)

    mae_val, rmse_val, r2_val = calcular_metricas(y_val, y_val_pred)

    mejor_epoca = int(np.argmin(historial.history["val_loss"]) + 1)

    resultado = {
        "base": nombre_base,
        "ventana": ventana,
        "lstm_units": params["lstm_units"],
        "dropout": params["dropout"],
        "dense_units": params["dense_units"],
        "learning_rate": params["learning_rate"],
        "batch_size": params["batch_size"],
        "n_train": X_train.shape[0],
        "n_val": X_val.shape[0],
        "n_test": X_test.shape[0],
        "MAE_val": mae_val,
        "RMSE_val": rmse_val,
        "R2_val": r2_val,
        "mejor_epoca": mejor_epoca,
        "epocas_entrenadas": len(historial.history["loss"])
    }

    return resultado


# ============================================================
# 11. Entrenar modelo final con mejores hiperparámetros
# ============================================================

def entrenar_modelo_final(nombre_base, datos, variables_entrada, params):

    print("\n======================================")
    print("ENTRENAMIENTO FINAL")
    print("Base:", nombre_base)
    print("Mejores hiperparámetros:")
    print(params)
    print("======================================")

    ventana = int(params["ventana"])

    X, y, y_persistencia, meta = crear_secuencias_lstm(
        datos=datos,
        ventana=ventana,
        variables_entrada=variables_entrada
    )

    mask_train_total = meta["anio_objetivo"] <= 2022
    mask_test = meta["anio_objetivo"] >= 2023

    X_train_total = X[mask_train_total.values]
    y_train_total = y[mask_train_total.values]

    X_test = X[mask_test.values]
    y_test = y[mask_test.values]
    y_persistencia_test = y_persistencia[mask_test.values]
    meta_test = meta[mask_test].reset_index(drop=True)

    X_train_s, scaler, medianas, X_test_s = imputar_y_escalar_3d(
        X_train_total,
        X_test=X_test
    )

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(42)

    modelo = crear_modelo_lstm(
        ventana=ventana,
        n_features=X_train_s.shape[2],
        lstm_units=int(params["lstm_units"]),
        dropout=float(params["dropout"]),
        dense_units=int(params["dense_units"]),
        learning_rate=float(params["learning_rate"])
    )

    mejor_epoca = int(params["mejor_epoca"])

    historial = modelo.fit(
        X_train_s,
        y_train_total,
        epochs=mejor_epoca,
        batch_size=int(params["batch_size"]),
        verbose=0
    )

    y_pred = modelo.predict(X_test_s, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, None)

    mae_lstm, rmse_lstm, r2_lstm = calcular_metricas(y_test, y_pred)
    mae_base, rmse_base, r2_base = calcular_metricas(
        y_test,
        y_persistencia_test
    )

    resultado_final = {
        "base": nombre_base,
        "modelo": "LSTM_optimizada_sin_proxies",
        "ventana_semanas": ventana,
        "lstm_units": int(params["lstm_units"]),
        "dropout": float(params["dropout"]),
        "dense_units": int(params["dense_units"]),
        "learning_rate": float(params["learning_rate"]),
        "batch_size": int(params["batch_size"]),
        "epocas_finales": mejor_epoca,
        "n_entrenamiento_total": X_train_total.shape[0],
        "n_prueba": X_test.shape[0],
        "MAE_LSTM": mae_lstm,
        "RMSE_LSTM": rmse_lstm,
        "R2_LSTM": r2_lstm,
        "MAE_persistencia": mae_base,
        "RMSE_persistencia": rmse_base,
        "R2_persistencia": r2_base
    }

    print(pd.DataFrame([resultado_final]).round(4).to_string(index=False))

    nombre_archivo_base = limpiar_nombre(nombre_base)

    # --------------------------------------------------------
    # Guardar predicciones
    # --------------------------------------------------------

    predicciones = meta_test.copy()
    predicciones["prediccion_lstm"] = y_pred
    predicciones["prediccion_persistencia"] = y_persistencia_test
    predicciones["error_lstm"] = (
        predicciones["parasitos_totales_t1"] -
        predicciones["prediccion_lstm"]
    )
    predicciones["error_abs_lstm"] = np.abs(predicciones["error_lstm"])

    predicciones.to_csv(
        os.path.join(
            carpeta_salida,
            f"predicciones_lstm_optimizada_{nombre_archivo_base}.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Gráfico observado vs predicho
    # --------------------------------------------------------

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)

    limite = max(np.max(y_test), np.max(y_pred))
    plt.plot([0, limite], [0, limite], linestyle="--")

    plt.xlabel("Carga parasitaria observada")
    plt.ylabel("Carga parasitaria predicha")
    plt.title(f"LSTM optimizada observado vs predicho - {nombre_base}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"lstm_optimizada_observado_vs_predicho_{nombre_archivo_base}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Serie temporal promedio observada y predicha
    # --------------------------------------------------------

    serie = predicciones.groupby(
        ["anio_objetivo", "semana_objetivo"],
        as_index=False
    )[["parasitos_totales_t1", "prediccion_lstm"]].mean()

    serie["fecha_objetivo"] = crear_fecha_iso(
        serie["anio_objetivo"],
        serie["semana_objetivo"]
    )

    serie = serie.dropna(subset=["fecha_objetivo"]).sort_values(
        "fecha_objetivo"
    )

    plt.figure(figsize=(11, 5))
    plt.plot(
        serie["fecha_objetivo"],
        serie["parasitos_totales_t1"],
        label="Observado"
    )
    plt.plot(
        serie["fecha_objetivo"],
        serie["prediccion_lstm"],
        label="Predicho LSTM"
    )

    plt.xlabel("Fecha objetivo")
    plt.ylabel("Carga parasitaria promedio")
    plt.title(f"Serie temporal LSTM optimizada - {nombre_base}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"lstm_optimizada_serie_temporal_{nombre_archivo_base}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Guardar modelo
    # --------------------------------------------------------

    modelo.save(
        os.path.join(
            carpeta_salida,
            f"modelo_lstm_optimizada_{nombre_archivo_base}.keras"
        )
    )

    return resultado_final


# ============================================================
# 12. Ejecutar ajuste de hiperparámetros
# ============================================================

bases = {
    "conjunto_chile_noruega": df.copy(),
    "chile": df[df["pais"] == "Chile"].copy(),
    "noruega": df[df["pais"] == "Noruega"].copy()
}

combinaciones = list(itertools.product(
    GRID_HIPERPARAMETROS["ventana"],
    GRID_HIPERPARAMETROS["lstm_units"],
    GRID_HIPERPARAMETROS["dropout"],
    GRID_HIPERPARAMETROS["dense_units"],
    GRID_HIPERPARAMETROS["learning_rate"],
    GRID_HIPERPARAMETROS["batch_size"]
))

resultados_tuning = []
resultados_finales = []

for nombre_base, datos_base in bases.items():

    print("\n##################################################")
    print("AJUSTE DE HIPERPARÁMETROS PARA:", nombre_base)
    print("##################################################")

    datos_lstm, variables_entrada = preparar_base_lstm(
        datos=datos_base,
        nombre_base=nombre_base
    )

    print("Variables de entrada:")
    print(variables_entrada)

    for idx, comb in enumerate(combinaciones, start=1):

        params = {
            "ventana": comb[0],
            "lstm_units": comb[1],
            "dropout": comb[2],
            "dense_units": comb[3],
            "learning_rate": comb[4],
            "batch_size": comb[5]
        }

        print(f"\nModelo {idx}/{len(combinaciones)} - {nombre_base}")
        print(params)

        resultado = evaluar_combinacion(
            nombre_base=nombre_base,
            datos=datos_lstm,
            variables_entrada=variables_entrada,
            params=params
        )

        if resultado is not None:
            resultados_tuning.append(resultado)
            print(
                "RMSE validación:",
                round(resultado["RMSE_val"], 4),
                "| R2 validación:",
                round(resultado["R2_val"], 4)
            )
        else:
            print("Combinación omitida por datos insuficientes.")

    # Guardar avance del tuning
    df_tuning_parcial = pd.DataFrame(resultados_tuning)

    if len(df_tuning_parcial) > 0:
        df_tuning_parcial.to_csv(
            os.path.join(carpeta_salida, "resumen_tuning_lstm_sin_proxies.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    # Seleccionar mejor combinación de esta base
    df_base = df_tuning_parcial[df_tuning_parcial["base"] == nombre_base].copy()

    if len(df_base) == 0:
        print("No se encontraron modelos válidos para:", nombre_base)
        continue

    mejor = df_base.sort_values("RMSE_val", ascending=True).iloc[0]

    print("\nMEJOR CONFIGURACIÓN PARA:", nombre_base)
    print(mejor.to_string())

    resultado_final = entrenar_modelo_final(
        nombre_base=nombre_base,
        datos=datos_lstm,
        variables_entrada=variables_entrada,
        params=mejor
    )

    resultados_finales.append(resultado_final)


# ============================================================
# 13. Guardar resumen final
# ============================================================

df_tuning = pd.DataFrame(resultados_tuning)

if len(df_tuning) > 0:

    df_tuning.to_csv(
        os.path.join(carpeta_salida, "resumen_tuning_lstm_sin_proxies.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    mejores = (
        df_tuning
        .sort_values(["base", "RMSE_val"], ascending=[True, True])
        .groupby("base", as_index=False)
        .first()
    )

    mejores.to_csv(
        os.path.join(carpeta_salida, "mejores_hiperparametros_lstm_sin_proxies.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("MEJORES HIPERPARÁMETROS POR BASE")
    print("======================================")
    print(mejores.round(4).to_string(index=False))


df_final = pd.DataFrame(resultados_finales)

if len(df_final) > 0:

    df_final.to_csv(
        os.path.join(carpeta_salida, "resumen_metricas_lstm_optimizada_sin_proxies.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    df_final_redondeado = df_final.copy()

    columnas_redondear = [
        "MAE_LSTM",
        "RMSE_LSTM",
        "R2_LSTM",
        "MAE_persistencia",
        "RMSE_persistencia",
        "R2_persistencia"
    ]

    for col in columnas_redondear:
        df_final_redondeado[col] = df_final_redondeado[col].round(3)

    df_final_redondeado.to_csv(
        os.path.join(carpeta_salida, "resumen_metricas_lstm_optimizada_sin_proxies_redondeado.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL LSTM OPTIMIZADA SIN PROXIES")
    print("======================================")
    print(df_final_redondeado.to_string(index=False))

else:
    print("\nNo se generaron resultados finales.")


print("\nArchivos guardados en:")
print(carpeta_salida)

plt.close("all")