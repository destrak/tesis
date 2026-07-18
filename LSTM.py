# -*- coding: utf-8 -*-

# ============================================================
# MODELO LSTM
# Predicción de carga parasitaria total t+1
# Chile, Noruega y conjunto Chile--Noruega
# ============================================================

import os
import re
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

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_lstm"
os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. Configuración general
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

tf.random.set_seed(42)
np.random.seed(42)

VENTANAS = [4, 8, 12]
EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 25


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


def escalar_3d(X_train, X_val, X_test):
    n_features = X_train.shape[2]

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    scaler.fit(X_train_2d)

    X_train_s = scaler.transform(
        X_train.reshape(-1, n_features)
    ).reshape(X_train.shape)

    X_val_s = scaler.transform(
        X_val.reshape(-1, n_features)
    ).reshape(X_val.shape)

    X_test_s = scaler.transform(
        X_test.reshape(-1, n_features)
    ).reshape(X_test.shape)

    return X_train_s, X_val_s, X_test_s, scaler


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
# Según la metodología de la tesis:
# S_N = S_C + 26, si S_C + 26 <= 52
# S_N = S_C + 26 - 52, si S_C + 26 > 52

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
# 8. Variables de entrada para LSTM
# ============================================================
# La LSTM ya aprende la historia mediante secuencias, por eso no se
# incorporan rezagos manuales como variables de entrada.

variables_numericas_lstm = [
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

variables_categoricas_lstm = [
    col for col in ["pais", "zona", "localidad", "estacion"]
    if col in df.columns
]

df_features = df.copy()

df_dummies = pd.get_dummies(
    df_features[variables_categoricas_lstm],
    drop_first=False
)

df_features = pd.concat(
    [
        df_features,
        df_dummies
    ],
    axis=1
)

variables_dummies = df_dummies.columns.tolist()

variables_entrada = variables_numericas_lstm + variables_dummies

variables_entrada = [
    v for v in variables_entrada
    if v in df_features.columns
]

print("\nNúmero de variables de entrada LSTM:", len(variables_entrada))
print(variables_entrada)


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
# 10. Crear modelo LSTM
# ============================================================

def crear_modelo_lstm(ventana, n_features):

    modelo = Sequential([
        Input(shape=(ventana, n_features)),
        LSTM(32, return_sequences=False),
        Dropout(0.20),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return modelo


# ============================================================
# 11. Entrenar y evaluar LSTM
# ============================================================

def entrenar_lstm(nombre_base, datos, ventana):

    print("\n======================================")
    print("Base:", nombre_base)
    print("Ventana:", ventana, "semanas")
    print("======================================")

    X, y, y_persistencia, meta = crear_secuencias_lstm(
        datos=datos,
        ventana=ventana,
        variables_entrada=variables_entrada
    )

    print("Secuencias generadas:", X.shape)

    if X.shape[0] == 0:
        print("No se generaron secuencias válidas.")
        return None

    mask_train_total = meta["anio_objetivo"] <= 2022
    mask_test = meta["anio_objetivo"] >= 2023

    X_train_total = X[mask_train_total.values]
    y_train_total = y[mask_train_total.values]
    meta_train_total = meta[mask_train_total].reset_index(drop=True)

    X_test = X[mask_test.values]
    y_test = y[mask_test.values]
    y_persistencia_test = y_persistencia[mask_test.values]
    meta_test = meta[mask_test].reset_index(drop=True)

    if X_train_total.shape[0] < 30 or X_test.shape[0] < 10:
        print("Datos insuficientes para entrenamiento/prueba.")
        return None

    # Validación temporal: último 20 % del entrenamiento
    n_val = max(1, int(0.20 * X_train_total.shape[0]))

    X_train = X_train_total[:-n_val]
    y_train = y_train_total[:-n_val]

    X_val = X_train_total[-n_val:]
    y_val = y_train_total[-n_val:]

    print("Entrenamiento:", X_train.shape[0])
    print("Validación:", X_val.shape[0])
    print("Prueba:", X_test.shape[0])

    X_train_s, X_val_s, X_test_s, scaler = escalar_3d(
        X_train,
        X_val,
        X_test
    )

    modelo = crear_modelo_lstm(
        ventana=ventana,
        n_features=X_train_s.shape[2]
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
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0
    )

    y_pred = modelo.predict(X_test_s, verbose=0).flatten()
    y_pred = np.clip(y_pred, 0, None)

    mae_lstm, rmse_lstm, r2_lstm = calcular_metricas(y_test, y_pred)
    mae_base, rmse_base, r2_base = calcular_metricas(
        y_test,
        y_persistencia_test
    )

    resultado = {
        "base": nombre_base,
        "modelo": "LSTM",
        "ventana_semanas": ventana,
        "n_entrenamiento": X_train.shape[0],
        "n_validacion": X_val.shape[0],
        "n_prueba": X_test.shape[0],
        "MAE_LSTM": mae_lstm,
        "RMSE_LSTM": rmse_lstm,
        "R2_LSTM": r2_lstm,
        "MAE_persistencia": mae_base,
        "RMSE_persistencia": rmse_base,
        "R2_persistencia": r2_base,
        "epocas_entrenadas": len(historial.history["loss"])
    }

    print(pd.DataFrame([resultado]).to_string(index=False))

    nombre_archivo_base = limpiar_nombre(nombre_base)

    predicciones = meta_test.copy()
    predicciones["prediccion_lstm"] = y_pred
    predicciones["prediccion_persistencia"] = y_persistencia_test
    predicciones["error_lstm"] = (
        predicciones["parasitos_totales_t1"] -
        predicciones["prediccion_lstm"]
    )
    predicciones["error_abs_lstm"] = np.abs(
        predicciones["error_lstm"]
    )

    predicciones.to_csv(
        os.path.join(
            carpeta_salida,
            f"predicciones_lstm_{nombre_archivo_base}_ventana_{ventana}.csv"
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
    plt.title(f"LSTM observado vs predicho - {nombre_base} - ventana {ventana}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"lstm_observado_vs_predicho_{nombre_archivo_base}_ventana_{ventana}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------------
    # Curva de pérdida
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))
    plt.plot(historial.history["loss"], label="Entrenamiento")
    plt.plot(historial.history["val_loss"], label="Validación")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.title(f"Curva de pérdida LSTM - {nombre_base} - ventana {ventana}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"lstm_loss_{nombre_archivo_base}_ventana_{ventana}.png"
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
    plt.title(f"Serie temporal LSTM - {nombre_base} - ventana {ventana}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            carpeta_salida,
            f"lstm_serie_temporal_{nombre_archivo_base}_ventana_{ventana}.png"
        ),
        dpi=300
    )

    plt.close()

    modelo.save(
        os.path.join(
            carpeta_salida,
            f"modelo_lstm_{nombre_archivo_base}_ventana_{ventana}.keras"
        )
    )

    return resultado


# ============================================================
# 12. Ejecutar modelos
# ============================================================

bases = {
    "conjunto_chile_noruega": df_features.copy(),
    "chile": df_features[df_features["pais"] == "Chile"].copy(),
    "noruega": df_features[df_features["pais"] == "Noruega"].copy()
}

resultados = []

for nombre_base, datos_base in bases.items():

    for ventana in VENTANAS:

        resultado = entrenar_lstm(
            nombre_base=nombre_base,
            datos=datos_base,
            ventana=ventana
        )

        if resultado is not None:
            resultados.append(resultado)


# ============================================================
# 13. Guardar resumen final
# ============================================================

resumen = pd.DataFrame(resultados)

if len(resumen) > 0:

    resumen.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_lstm.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    resumen_redondeado = resumen.copy()

    columnas_redondear = [
        "MAE_LSTM",
        "RMSE_LSTM",
        "R2_LSTM",
        "MAE_persistencia",
        "RMSE_persistencia",
        "R2_persistencia"
    ]

    for col in columnas_redondear:
        resumen_redondeado[col] = resumen_redondeado[col].round(3)

    resumen_redondeado.to_csv(
        os.path.join(
            carpeta_salida,
            "resumen_metricas_lstm_redondeado.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print("\n======================================")
    print("RESUMEN FINAL LSTM")
    print("======================================")
    print(resumen_redondeado.to_string(index=False))

else:
    print("\nNo se generaron resultados LSTM.")


print("\nArchivos guardados en:")
print(carpeta_salida)

plt.close("all")