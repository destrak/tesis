# -*- coding: utf-8 -*-

# ============================================================
# MODELO LSTM CON MEJORES HIPERPARÁMETROS
# Predicción de carga parasitaria total t+1
# Chile, Noruega y conjunto Chile--Noruega
# Sin proxies: anio, zona, localidad, estacion
#
# Incluye:
# - Tiempo total del programa
# - Tiempo por modelo
# - Tiempo de creación de secuencias
# - Tiempo de escalamiento
# - Tiempo de entrenamiento
# - Tiempo promedio por época
# - Tiempo de predicción
# - Complejidad temporal y espacial
# ============================================================

import os
from time import perf_counter
from datetime import timedelta

# Se inicia el cronómetro antes de importar las bibliotecas pesadas.
INICIO_PROGRAMA = perf_counter()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
plt.ioff()

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

try:
    import tensorflow as tf

    from tensorflow.keras.models import Sequential

    from tensorflow.keras.layers import (
        LSTM,
        Dense,
        Dropout,
        Input
    )

    from tensorflow.keras.callbacks import EarlyStopping

except ImportError:
    raise ImportError(
        "No se pudo importar TensorFlow. "
        "Instálalo con: pip install tensorflow "
        "o ejecuta este script en Google Colab."
    )


# ============================================================
# 1. RUTAS
# ============================================================

ruta_dataset = (
    r"C:\Users\jarpa\OneDrive\Escritorio"
    r"\datos tesis\tesis"
    r"\resultados_random_forest_mejorado"
    r"\datasetT1.csv"
)

carpeta_salida = (
    r"C:\Users\jarpa\OneDrive\Escritorio"
    r"\datos tesis\tesis"
    r"\resultados_lstm_mejores_hiperparametros_sin_proxies"
)

os.makedirs(
    carpeta_salida,
    exist_ok=True
)


# ============================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

tf.random.set_seed(42)
np.random.seed(42)

EPOCHS = 200
PATIENCE = 25


# ------------------------------------------------------------
# Detectar CPU o GPU
# ------------------------------------------------------------

DISPOSITIVOS_GPU = tf.config.list_physical_devices("GPU")

if len(DISPOSITIVOS_GPU) > 0:
    DISPOSITIVO = "GPU"
else:
    DISPOSITIVO = "CPU"

print("\n======================================")
print("DISPOSITIVO DETECTADO")
print("======================================")
print("Dispositivo utilizado:", DISPOSITIVO)

if DISPOSITIVOS_GPU:
    for gpu in DISPOSITIVOS_GPU:
        print("GPU detectada:", gpu.name)


# ============================================================
# 3. MEJORES HIPERPARÁMETROS OBTENIDOS
# ============================================================

MEJORES_HIPERPARAMETROS = {

    "conjunto_chile_noruega": {
        "ventana": 4,
        "lstm_units": 32,
        "dropout": 0.10,
        "dense_units": 8,
        "learning_rate": 0.0005,
        "batch_size": 32
    },

    "chile": {
        "ventana": 4,
        "lstm_units": 16,
        "dropout": 0.20,
        "dense_units": 8,
        "learning_rate": 0.001,
        "batch_size": 32
    },

    "noruega": {
        "ventana": 4,
        "lstm_units": 32,
        "dropout": 0.10,
        "dense_units": 16,
        "learning_rate": 0.001,
        "batch_size": 32
    }
}


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def formatear_tiempo(segundos):
    """
    Convierte segundos a un formato HH:MM:SS.
    """

    return str(
        timedelta(
            seconds=round(float(segundos))
        )
    )


def limpiar_nombre(texto):
    """
    Limpia un texto para utilizarlo como nombre de archivo.
    """

    texto = str(texto).lower()
    texto = texto.replace(" ", "_")

    texto = re.sub(
        r"[^a-zA-Z0-9_]+",
        "",
        texto
    )

    return texto


def crear_fecha_iso(anio, semana):
    """
    Crea una fecha a partir del año ISO y la semana ISO.
    Se utiliza el lunes como primer día de la semana.
    """

    return pd.to_datetime(

        anio.astype("Int64").astype(str)

        + "-W"

        + semana.astype("Int64")
        .astype(str)
        .str.zfill(2)

        + "-1",

        format="%G-W%V-%u",

        errors="coerce"
    )


def calcular_metricas(y_real, y_pred):
    """
    Calcula MAE, RMSE y R2.
    """

    mae = mean_absolute_error(
        y_real,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            y_pred
        )
    )

    r2 = r2_score(
        y_real,
        y_pred
    )

    return mae, rmse, r2


def imputar_y_escalar_3d(
    X_train,
    X_val,
    X_test
):
    """
    Imputa los valores NaN utilizando la mediana calculada
    exclusivamente sobre el conjunto de entrenamiento.

    Posteriormente escala las variables mediante StandardScaler.

    Los arreglos tienen la forma:

    muestras × ventana × variables
    """

    n_features = X_train.shape[2]

    # Convertir entrenamiento desde 3D a 2D
    X_train_2d = X_train.reshape(
        -1,
        n_features
    )

    # Medianas calculadas únicamente con entrenamiento
    medianas = np.nanmedian(
        X_train_2d,
        axis=0
    )

    # Si una variable contiene solamente NaN, usar cero
    medianas = np.where(
        np.isnan(medianas),
        0,
        medianas
    )

    def imputar(X):
        """
        Imputa NaN conservando la estructura original 3D.
        """

        X_2d = X.reshape(
            -1,
            n_features
        ).copy()

        indices_nan = np.where(
            np.isnan(X_2d)
        )

        X_2d[indices_nan] = np.take(
            medianas,
            indices_nan[1]
        )

        return X_2d.reshape(
            X.shape
        )

    X_train_imp = imputar(X_train)
    X_val_imp = imputar(X_val)
    X_test_imp = imputar(X_test)

    scaler = StandardScaler()

    scaler.fit(
        X_train_imp.reshape(
            -1,
            n_features
        )
    )

    X_train_s = scaler.transform(
        X_train_imp.reshape(
            -1,
            n_features
        )
    ).reshape(
        X_train.shape
    )

    X_val_s = scaler.transform(
        X_val_imp.reshape(
            -1,
            n_features
        )
    ).reshape(
        X_val.shape
    )

    X_test_s = scaler.transform(
        X_test_imp.reshape(
            -1,
            n_features
        )
    ).reshape(
        X_test.shape
    )

    return (
        X_train_s,
        X_val_s,
        X_test_s,
        scaler
    )


# ============================================================
# 5. CARGAR DATASET
# ============================================================

inicio_carga_dataset = perf_counter()

df = pd.read_csv(
    ruta_dataset,
    encoding="utf-8-sig"
)

tiempo_carga_dataset = (
    perf_counter()
    - inicio_carga_dataset
)

print("\n======================================")
print("DATASET CARGADO")
print("======================================")

print("Dimensiones:", df.shape)
print("Columnas:")
print(df.columns.tolist())

print(
    "Tiempo de carga:",
    formatear_tiempo(
        tiempo_carga_dataset
    )
)

inicio_preprocesamiento = perf_counter()


# ============================================================
# 6. ASEGURAR TIPOS DE DATOS
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

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


columnas_categoricas = [
    "pais",
    "zona",
    "localidad",
    "estacion"
]

for col in columnas_categoricas:

    if col in df.columns:

        df[col] = df[col].astype(str)


# ============================================================
# 7. HOMOLOGACIÓN TEMPORAL CHILE -> NORUEGA
# ============================================================

df["semana_homologada"] = df["semana"].copy()
df["anio_homologado"] = df["anio"].copy()

mask_chile = (
    df["pais"] == "Chile"
)

df.loc[
    mask_chile,
    "semana_homologada"
] = (
    df.loc[
        mask_chile,
        "semana"
    ]
    + 26
)

cambio_anio = (
    mask_chile
    &
    (
        df["semana_homologada"]
        > 52
    )
)

df.loc[
    cambio_anio,
    "semana_homologada"
] = (
    df.loc[
        cambio_anio,
        "semana_homologada"
    ]
    - 52
)

df.loc[
    cambio_anio,
    "anio_homologado"
] = (
    df.loc[
        cambio_anio,
        "anio_homologado"
    ]
    + 1
)

df["semana_sin"] = np.sin(
    2
    * np.pi
    * df["semana_homologada"]
    / 52
)

df["semana_cos"] = np.cos(
    2
    * np.pi
    * df["semana_homologada"]
    / 52
)


# ============================================================
# 8. CREAR FECHAS REALES PARA LAS SECUENCIAS
# ============================================================

df["fecha_t"] = crear_fecha_iso(
    df["anio"],
    df["semana"]
)

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

tiempo_preprocesamiento = (
    perf_counter()
    - inicio_preprocesamiento
)

print("\n======================================")
print("PREPROCESAMIENTO GENERAL")
print("======================================")

print(
    "Filas después del preprocesamiento:",
    len(df)
)

print(
    "Tiempo de preprocesamiento:",
    formatear_tiempo(
        tiempo_preprocesamiento
    )
)


# ============================================================
# 9. VARIABLES DE ENTRADA SIN PROXIES
# ============================================================

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


def preparar_base_lstm(
    datos,
    nombre_base
):
    """
    Se eliminan como predictoras:

    - anio
    - zona
    - localidad
    - estacion

    La variable pais se usa únicamente para la base conjunta
    Chile--Noruega mediante variables dummy.
    """

    datos = datos.copy()

    variables_entrada = (
        variables_numericas_lstm.copy()
    )

    if nombre_base == "conjunto_chile_noruega":

        dummies_pais = pd.get_dummies(
            datos["pais"],
            prefix="pais",
            drop_first=False,
            dtype=int
        )

        datos = pd.concat(
            [
                datos,
                dummies_pais
            ],
            axis=1
        )

        variables_entrada += (
            dummies_pais.columns.tolist()
        )

    variables_entrada = [
        variable
        for variable in variables_entrada
        if variable in datos.columns
    ]

    return (
        datos,
        variables_entrada
    )


# ============================================================
# 10. CREAR SECUENCIAS PARA LSTM
# ============================================================

def crear_secuencias_lstm(
    datos,
    ventana,
    variables_entrada
):
    """
    Crea secuencias temporales consecutivas para cada combinación
    de país y localidad.

    Solo se aceptan ventanas donde las observaciones están
    separadas exactamente por siete días.
    """

    datos = datos.sort_values(
        [
            "pais",
            "localidad",
            "fecha_t"
        ]
    ).copy()

    X_lista = []
    y_lista = []
    persistencia_lista = []
    meta_lista = []

    grupos = datos.groupby(
        [
            "pais",
            "localidad"
        ]
    )

    for (pais, localidad), grupo in grupos:

        grupo = grupo.sort_values(
            "fecha_t"
        ).reset_index(
            drop=True
        )

        for i in range(
            ventana - 1,
            len(grupo)
        ):

            ventana_df = grupo.iloc[
                i - ventana + 1:
                i + 1
            ].copy()

            diferencias = (
                ventana_df["fecha_t"]
                .diff()
                .dropna()
                .dt.days
            )

            # Verificar continuidad semanal
            if (
                len(diferencias) > 0
                and
                not (diferencias == 7).all()
            ):
                continue

            fila_actual = grupo.iloc[i]

            if pd.isna(
                fila_actual[
                    "parasitos_totales_t1"
                ]
            ):
                continue

            X = ventana_df[
                variables_entrada
            ].values.astype(float)

            y = float(
                fila_actual[
                    "parasitos_totales_t1"
                ]
            )

            y_persistencia = float(
                fila_actual[
                    "parasitos_totales"
                ]
            )

            X_lista.append(X)
            y_lista.append(y)

            persistencia_lista.append(
                y_persistencia
            )

            meta_lista.append({

                "pais":
                    fila_actual["pais"],

                "zona":
                    fila_actual["zona"]
                    if "zona" in fila_actual
                    else "",

                "localidad":
                    fila_actual["localidad"],

                "anio":
                    fila_actual["anio"],

                "semana":
                    fila_actual["semana"],

                "anio_objetivo":
                    fila_actual["anio_objetivo"],

                "semana_objetivo":
                    fila_actual["semana_objetivo"],

                "fecha_objetivo":
                    fila_actual["fecha_objetivo"],

                "parasitos_totales":
                    fila_actual["parasitos_totales"],

                "parasitos_totales_t1":
                    fila_actual[
                        "parasitos_totales_t1"
                    ]
            })

    X_array = np.array(
        X_lista
    )

    y_array = np.array(
        y_lista
    )

    persistencia_array = np.array(
        persistencia_lista
    )

    meta = pd.DataFrame(
        meta_lista
    )

    if len(meta) > 0:

        orden = np.argsort(
            meta[
                "fecha_objetivo"
            ].values.astype(
                "datetime64[ns]"
            )
        )

        X_array = X_array[orden]
        y_array = y_array[orden]

        persistencia_array = (
            persistencia_array[orden]
        )

        meta = meta.iloc[
            orden
        ].reset_index(
            drop=True
        )

    return (
        X_array,
        y_array,
        persistencia_array,
        meta
    )


# ============================================================
# 11. CREAR MODELO LSTM
# ============================================================

def crear_modelo_lstm(
    ventana,
    n_features,
    params
):
    """
    Construye y compila el modelo LSTM.
    """

    modelo = Sequential([

        Input(
            shape=(
                ventana,
                n_features
            )
        ),

        LSTM(
            int(
                params["lstm_units"]
            ),
            return_sequences=False
        ),

        Dropout(
            float(
                params["dropout"]
            )
        ),

        Dense(
            int(
                params["dense_units"]
            ),
            activation="relu"
        ),

        Dense(1)
    ])

    modelo.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=float(
                params["learning_rate"]
            )
        ),

        loss="mse",

        metrics=[
            "mae"
        ]
    )

    return modelo


# ============================================================
# 12. ENTRENAR Y EVALUAR LSTM
# ============================================================

def entrenar_lstm(
    nombre_base,
    datos,
    params
):
    """
    Entrena un modelo LSTM y mide los tiempos de cada etapa.

    Complejidad temporal dominante:

        O(E * S * W * U * (F + U))

    donde:

        E = épocas efectivamente entrenadas
        S = número de secuencias
        W = longitud de la ventana temporal
        U = número de unidades LSTM
        F = número de variables de entrada

    Complejidad espacial aproximada:

        O(S * W * F + U * (F + U))
    """

    inicio_total_modelo = perf_counter()

    ventana = int(
        params["ventana"]
    )

    print("\n\n======================================")
    print("BASE:", nombre_base)
    print("======================================")

    print(
        "Ventana:",
        ventana,
        "semanas"
    )

    print(
        "Hiperparámetros:",
        params
    )

    print(
        "Dispositivo:",
        DISPOSITIVO
    )


    # --------------------------------------------------------
    # Preparación de la base
    # --------------------------------------------------------

    inicio_preparacion_base = perf_counter()

    datos_lstm, variables_entrada = preparar_base_lstm(
        datos=datos,
        nombre_base=nombre_base
    )

    tiempo_preparacion_base = (
        perf_counter()
        - inicio_preparacion_base
    )

    print("\nVariables de entrada:")
    print(variables_entrada)


    # --------------------------------------------------------
    # Creación de secuencias
    # --------------------------------------------------------

    inicio_secuencias = perf_counter()

    X, y, y_persistencia, meta = crear_secuencias_lstm(
        datos=datos_lstm,
        ventana=ventana,
        variables_entrada=variables_entrada
    )

    tiempo_secuencias = (
        perf_counter()
        - inicio_secuencias
    )

    print(
        "\nForma de las secuencias:",
        X.shape
    )

    print(
        "Tiempo de creación de secuencias:",
        formatear_tiempo(
            tiempo_secuencias
        )
    )

    if X.shape[0] == 0:

        tiempo_fallido = (
            perf_counter()
            - inicio_total_modelo
        )

        print(
            "No se generaron secuencias válidas."
        )

        print(
            "Tiempo utilizado:",
            formatear_tiempo(
                tiempo_fallido
            )
        )

        return None


    # --------------------------------------------------------
    # División temporal
    # --------------------------------------------------------

    inicio_division = perf_counter()

    mask_train_total = (
        meta["anio_objetivo"]
        <= 2022
    )

    mask_test = (
        meta["anio_objetivo"]
        >= 2023
    )

    X_train_total = X[
        mask_train_total.values
    ]

    y_train_total = y[
        mask_train_total.values
    ]

    X_test = X[
        mask_test.values
    ]

    y_test = y[
        mask_test.values
    ]

    y_persistencia_test = y_persistencia[
        mask_test.values
    ]

    meta_test = meta[
        mask_test
    ].reset_index(
        drop=True
    )

    if (
        X_train_total.shape[0] < 30
        or
        X_test.shape[0] < 10
    ):

        tiempo_fallido = (
            perf_counter()
            - inicio_total_modelo
        )

        print(
            "Datos insuficientes "
            "para entrenamiento o prueba."
        )

        print(
            "Tiempo utilizado:",
            formatear_tiempo(
                tiempo_fallido
            )
        )

        return None

    # Último 20 % del entrenamiento para validación temporal
    n_val = max(
        1,
        int(
            0.20
            * X_train_total.shape[0]
        )
    )

    X_train = X_train_total[
        :-n_val
    ]

    y_train = y_train_total[
        :-n_val
    ]

    X_val = X_train_total[
        -n_val:
    ]

    y_val = y_train_total[
        -n_val:
    ]

    tiempo_division = (
        perf_counter()
        - inicio_division
    )

    print(
        "\nSecuencias de entrenamiento:",
        X_train.shape[0]
    )

    print(
        "Secuencias de validación:",
        X_val.shape[0]
    )

    print(
        "Secuencias de prueba:",
        X_test.shape[0]
    )


    # --------------------------------------------------------
    # Imputación y escalamiento
    # --------------------------------------------------------

    inicio_escalamiento = perf_counter()

    (
        X_train_s,
        X_val_s,
        X_test_s,
        scaler
    ) = imputar_y_escalar_3d(
        X_train,
        X_val,
        X_test
    )

    tiempo_escalamiento = (
        perf_counter()
        - inicio_escalamiento
    )


    # --------------------------------------------------------
    # Construcción del modelo
    # --------------------------------------------------------

    inicio_construccion = perf_counter()

    tf.keras.backend.clear_session()

    tf.random.set_seed(42)
    np.random.seed(42)

    modelo = crear_modelo_lstm(
        ventana=ventana,
        n_features=X_train_s.shape[2],
        params=params
    )

    tiempo_construccion = (
        perf_counter()
        - inicio_construccion
    )

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=PATIENCE,

        restore_best_weights=True
    )


    # --------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------

    print("\nIniciando entrenamiento...")

    inicio_entrenamiento = perf_counter()

    historial = modelo.fit(

        X_train_s,

        y_train,

        validation_data=(
            X_val_s,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=int(
            params["batch_size"]
        ),

        callbacks=[
            early_stop
        ],

        verbose=0
    )

    tiempo_entrenamiento = (
        perf_counter()
        - inicio_entrenamiento
    )

    epocas_entrenadas = len(
        historial.history["loss"]
    )

    if epocas_entrenadas > 0:

        segundos_por_epoca = (
            tiempo_entrenamiento
            / epocas_entrenadas
        )

    else:

        segundos_por_epoca = np.nan

    # Estimación para las 200 épocas basada en el promedio real
    tiempo_estimado_200_epocas = (
        segundos_por_epoca
        * EPOCHS
    )


    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    inicio_prediccion = perf_counter()

    y_pred = modelo.predict(
        X_test_s,
        verbose=0
    ).flatten()

    tiempo_prediccion = (
        perf_counter()
        - inicio_prediccion
    )

    # Evitar predicciones negativas
    y_pred = np.clip(
        y_pred,
        0,
        None
    )


    # --------------------------------------------------------
    # Métricas
    # --------------------------------------------------------

    inicio_metricas = perf_counter()

    (
        mae_lstm,
        rmse_lstm,
        r2_lstm
    ) = calcular_metricas(
        y_test,
        y_pred
    )

    (
        mae_base,
        rmse_base,
        r2_base
    ) = calcular_metricas(
        y_test,
        y_persistencia_test
    )

    tiempo_metricas = (
        perf_counter()
        - inicio_metricas
    )


    # --------------------------------------------------------
    # Complejidad teórica
    # --------------------------------------------------------

    n_features = int(
        X_train_s.shape[2]
    )

    lstm_units = int(
        params["lstm_units"]
    )

    dense_units = int(
        params["dense_units"]
    )

    n_secuencias_entrenamiento = int(
        X_train.shape[0]
    )

    n_secuencias_validacion = int(
        X_val.shape[0]
    )

    n_secuencias_prueba = int(
        X_test.shape[0]
    )

    # Este factor no representa FLOPs exactos.
    # Sirve para comparar el costo relativo entre los modelos.
    factor_complejidad_entrenamiento = (

        epocas_entrenadas

        * (
            n_secuencias_entrenamiento
            + n_secuencias_validacion
        )

        * ventana

        * lstm_units

        * (
            n_features
            + lstm_units
        )
    )

    factor_complejidad_prediccion = (

        n_secuencias_prueba

        * ventana

        * lstm_units

        * (
            n_features
            + lstm_units
        )
    )

    complejidad_temporal = (
        "O(E*S*W*U*(F+U))"
    )

    complejidad_espacial = (
        "O(S*W*F + U*(F+U))"
    )

    nombre_archivo_base = limpiar_nombre(
        nombre_base
    )


    # --------------------------------------------------------
    # Guardar resultados
    # --------------------------------------------------------

    inicio_guardado = perf_counter()

    predicciones = meta_test.copy()

    predicciones[
        "prediccion_lstm"
    ] = y_pred

    predicciones[
        "prediccion_persistencia"
    ] = y_persistencia_test

    predicciones[
        "error_lstm"
    ] = (

        predicciones[
            "parasitos_totales_t1"
        ]

        - predicciones[
            "prediccion_lstm"
        ]
    )

    predicciones[
        "error_abs_lstm"
    ] = np.abs(
        predicciones[
            "error_lstm"
        ]
    )

    predicciones.to_csv(

        os.path.join(
            carpeta_salida,
            (
                "predicciones_lstm_mejores_"
                f"{nombre_archivo_base}.csv"
            )
        ),

        index=False,

        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # Gráfico observado vs predicho
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 6)
    )

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    limite = max(
        np.max(y_test),
        np.max(y_pred)
    )

    if limite == 0:
        limite = 1

    plt.plot(
        [0, limite],
        [0, limite],
        linestyle="--"
    )

    plt.xlabel(
        "Carga parasitaria observada"
    )

    plt.ylabel(
        "Carga parasitaria predicha"
    )

    plt.title(
        "LSTM mejores hiperparámetros - "
        f"{nombre_base}"
    )

    plt.tight_layout()

    plt.savefig(

        os.path.join(
            carpeta_salida,
            (
                "lstm_mejores_observado_vs_predicho_"
                f"{nombre_archivo_base}.png"
            )
        ),

        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Curva de pérdida
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        historial.history["loss"],
        label="Entrenamiento"
    )

    plt.plot(
        historial.history["val_loss"],
        label="Validación"
    )

    plt.xlabel("Época")
    plt.ylabel("MSE")

    plt.title(
        "Curva de pérdida LSTM - "
        f"{nombre_base}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(

        os.path.join(
            carpeta_salida,
            (
                "lstm_mejores_loss_"
                f"{nombre_archivo_base}.png"
            )
        ),

        dpi=300
    )

    plt.close()


    # --------------------------------------------------------
    # Serie temporal promedio observada y predicha
    # --------------------------------------------------------

    serie = predicciones.groupby(

        [
            "anio_objetivo",
            "semana_objetivo"
        ],

        as_index=False

    )[
        [
            "parasitos_totales_t1",
            "prediccion_lstm"
        ]
    ].mean()

    serie["fecha_objetivo"] = crear_fecha_iso(

        serie["anio_objetivo"],

        serie["semana_objetivo"]
    )

    serie = serie.dropna(
        subset=[
            "fecha_objetivo"
        ]
    ).sort_values(
        "fecha_objetivo"
    )

    plt.figure(
        figsize=(11, 5)
    )

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

    plt.xlabel(
        "Fecha objetivo"
    )

    plt.ylabel(
        "Carga parasitaria promedio"
    )

    plt.title(
        "Serie temporal LSTM - "
        f"{nombre_base}"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(

        os.path.join(
            carpeta_salida,
            (
                "lstm_mejores_serie_temporal_"
                f"{nombre_archivo_base}.png"
            )
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
            (
                "modelo_lstm_mejores_"
                f"{nombre_archivo_base}.keras"
            )
        )
    )

    tiempo_guardado = (
        perf_counter()
        - inicio_guardado
    )

    tiempo_total_modelo = (
        perf_counter()
        - inicio_total_modelo
    )


    # --------------------------------------------------------
    # Crear diccionario de resultados
    # --------------------------------------------------------

    resultado = {

        "base":
            nombre_base,

        "modelo":
            "LSTM_mejores_hiperparametros_sin_proxies",

        "dispositivo":
            DISPOSITIVO,

        "ventana_semanas":
            ventana,

        "n_variables_entrada":
            n_features,

        "lstm_units":
            lstm_units,

        "dropout":
            float(
                params["dropout"]
            ),

        "dense_units":
            dense_units,

        "learning_rate":
            float(
                params["learning_rate"]
            ),

        "batch_size":
            int(
                params["batch_size"]
            ),

        "n_secuencias_totales":
            int(
                X.shape[0]
            ),

        "n_entrenamiento":
            n_secuencias_entrenamiento,

        "n_validacion":
            n_secuencias_validacion,

        "n_prueba":
            n_secuencias_prueba,

        "MAE_LSTM":
            mae_lstm,

        "RMSE_LSTM":
            rmse_lstm,

        "R2_LSTM":
            r2_lstm,

        "MAE_persistencia":
            mae_base,

        "RMSE_persistencia":
            rmse_base,

        "R2_persistencia":
            r2_base,

        "epocas_maximas":
            EPOCHS,

        "epocas_entrenadas":
            epocas_entrenadas,

        "early_stopping_activo":
            epocas_entrenadas < EPOCHS,

        "tiempo_preparacion_base_segundos":
            tiempo_preparacion_base,

        "tiempo_secuencias_segundos":
            tiempo_secuencias,

        "tiempo_division_segundos":
            tiempo_division,

        "tiempo_escalamiento_segundos":
            tiempo_escalamiento,

        "tiempo_construccion_segundos":
            tiempo_construccion,

        "tiempo_entrenamiento_segundos":
            tiempo_entrenamiento,

        "segundos_por_epoca":
            segundos_por_epoca,

        "tiempo_estimado_200_epocas_segundos":
            tiempo_estimado_200_epocas,

        "tiempo_prediccion_segundos":
            tiempo_prediccion,

        "tiempo_metricas_segundos":
            tiempo_metricas,

        "tiempo_guardado_segundos":
            tiempo_guardado,

        "tiempo_total_modelo_segundos":
            tiempo_total_modelo,

        "complejidad_temporal":
            complejidad_temporal,

        "complejidad_espacial":
            complejidad_espacial,

        "factor_complejidad_entrenamiento":
            factor_complejidad_entrenamiento,

        "factor_complejidad_prediccion":
            factor_complejidad_prediccion
    }


    # --------------------------------------------------------
    # Mostrar métricas
    # --------------------------------------------------------

    print("\n======================================")
    print("RESULTADOS DEL MODELO")
    print("======================================")

    columnas_mostrar = [
        "base",
        "n_entrenamiento",
        "n_validacion",
        "n_prueba",
        "epocas_entrenadas",
        "MAE_LSTM",
        "RMSE_LSTM",
        "R2_LSTM"
    ]

    print(

        pd.DataFrame(
            [resultado]
        )[
            columnas_mostrar
        ]
        .round(4)
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Mostrar tiempos
    # --------------------------------------------------------

    print("\n======================================")
    print("TIEMPOS DE EJECUCIÓN")
    print("======================================")

    print(
        "Preparación de la base:",
        formatear_tiempo(
            tiempo_preparacion_base
        )
    )

    print(
        "Creación de secuencias:",
        formatear_tiempo(
            tiempo_secuencias
        )
    )

    print(
        "División temporal:",
        formatear_tiempo(
            tiempo_division
        )
    )

    print(
        "Imputación y escalamiento:",
        formatear_tiempo(
            tiempo_escalamiento
        )
    )

    print(
        "Construcción del modelo:",
        formatear_tiempo(
            tiempo_construccion
        )
    )

    print(
        "Entrenamiento:",
        formatear_tiempo(
            tiempo_entrenamiento
        )
    )

    print(
        "Entrenamiento en segundos:",
        round(
            tiempo_entrenamiento,
            4
        )
    )

    print(
        "Épocas entrenadas:",
        epocas_entrenadas
    )

    print(
        "Promedio por época:",
        round(
            segundos_por_epoca,
            4
        ),
        "segundos"
    )

    print(
        "Tiempo estimado para 200 épocas:",
        formatear_tiempo(
            tiempo_estimado_200_epocas
        )
    )

    print(
        "Predicción:",
        formatear_tiempo(
            tiempo_prediccion
        )
    )

    print(
        "Cálculo de métricas:",
        formatear_tiempo(
            tiempo_metricas
        )
    )

    print(
        "Guardado de archivos:",
        formatear_tiempo(
            tiempo_guardado
        )
    )

    print(
        "TIEMPO TOTAL DEL MODELO:",
        formatear_tiempo(
            tiempo_total_modelo
        )
    )

    print(
        "Tiempo total del modelo en segundos:",
        round(
            tiempo_total_modelo,
            4
        )
    )


    # --------------------------------------------------------
    # Mostrar complejidad
    # --------------------------------------------------------

    print("\n======================================")
    print("COMPLEJIDAD COMPUTACIONAL")
    print("======================================")

    print(
        "Complejidad temporal dominante:",
        complejidad_temporal
    )

    print(
        "Complejidad espacial:",
        complejidad_espacial
    )

    print(
        "E =",
        epocas_entrenadas,
        "épocas"
    )

    print(
        "S =",
        (
            n_secuencias_entrenamiento
            + n_secuencias_validacion
        ),
        "secuencias de entrenamiento y validación"
    )

    print(
        "W =",
        ventana,
        "semanas"
    )

    print(
        "U =",
        lstm_units,
        "unidades LSTM"
    )

    print(
        "F =",
        n_features,
        "variables"
    )

    print(
        "Factor relativo de entrenamiento:",
        f"{factor_complejidad_entrenamiento:.3e}"
    )

    return resultado


# ============================================================
# 13. EJECUTAR LOS TRES MODELOS
# ============================================================

bases = {

    "conjunto_chile_noruega":
        df.copy(),

    "chile":
        df[
            df["pais"] == "Chile"
        ].copy(),

    "noruega":
        df[
            df["pais"] == "Noruega"
        ].copy()
}

resultados = []

for nombre_base, datos_base in bases.items():

    params = MEJORES_HIPERPARAMETROS[
        nombre_base
    ]

    resultado = entrenar_lstm(

        nombre_base=nombre_base,

        datos=datos_base,

        params=params
    )

    if resultado is not None:

        resultados.append(
            resultado
        )


# ============================================================
# 14. GUARDAR RESUMEN FINAL
# ============================================================

resumen = pd.DataFrame(
    resultados
)

if len(resumen) > 0:

    ruta_resumen_completo = os.path.join(

        carpeta_salida,

        (
            "resumen_metricas_tiempos_complejidad_"
            "lstm_sin_proxies.csv"
        )
    )

    resumen.to_csv(

        ruta_resumen_completo,

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
        "R2_persistencia",

        "tiempo_preparacion_base_segundos",
        "tiempo_secuencias_segundos",
        "tiempo_division_segundos",
        "tiempo_escalamiento_segundos",
        "tiempo_construccion_segundos",
        "tiempo_entrenamiento_segundos",
        "segundos_por_epoca",
        "tiempo_estimado_200_epocas_segundos",
        "tiempo_prediccion_segundos",
        "tiempo_metricas_segundos",
        "tiempo_guardado_segundos",
        "tiempo_total_modelo_segundos"
    ]

    for col in columnas_redondear:

        if col in resumen_redondeado.columns:

            resumen_redondeado[col] = (
                resumen_redondeado[col]
                .round(3)
            )

    ruta_resumen_redondeado = os.path.join(

        carpeta_salida,

        (
            "resumen_metricas_tiempos_complejidad_"
            "lstm_sin_proxies_redondeado.csv"
        )
    )

    resumen_redondeado.to_csv(

        ruta_resumen_redondeado,

        index=False,

        encoding="utf-8-sig"
    )

    print("\n\n======================================")
    print("RESUMEN FINAL LSTM")
    print("======================================")

    columnas_resumen_pantalla = [

        "base",
        "dispositivo",
        "n_secuencias_totales",
        "epocas_entrenadas",
        "MAE_LSTM",
        "RMSE_LSTM",
        "R2_LSTM",
        "tiempo_entrenamiento_segundos",
        "tiempo_total_modelo_segundos",
        "complejidad_temporal"
    ]

    print(

        resumen_redondeado[
            columnas_resumen_pantalla
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nNo se generaron resultados LSTM."
    )


# ============================================================
# 15. TIEMPO TOTAL DEL PROGRAMA
# ============================================================

tiempo_total_programa = (
    perf_counter()
    - INICIO_PROGRAMA
)

print("\n\n======================================")
print("TIEMPO TOTAL DE EJECUCIÓN DEL SCRIPT")
print("======================================")

print(
    "Dispositivo:",
    DISPOSITIVO
)

print(
    "Carga del dataset:",
    formatear_tiempo(
        tiempo_carga_dataset
    )
)

print(
    "Preprocesamiento general:",
    formatear_tiempo(
        tiempo_preprocesamiento
    )
)

print(
    "Tiempo total en segundos:",
    round(
        tiempo_total_programa,
        3
    )
)

print(
    "Tiempo total HH:MM:SS:",
    formatear_tiempo(
        tiempo_total_programa
    )
)

print("\n======================================")
print("COMPLEJIDAD TEMPORAL DOMINANTE")
print("======================================")

print(
    "O(E*S*W*U*(F+U))"
)

print(
    "E: épocas efectivamente entrenadas."
)

print(
    "S: número de secuencias."
)

print(
    "W: longitud de la ventana temporal."
)

print(
    "U: número de unidades LSTM."
)

print(
    "F: número de variables de entrada."
)

print(
    "\nOrdenación inicial: O(N log N)."
)

print(
    "Creación y escalamiento: O(S*W*F)."
)

print(
    "Entrenamiento dominante: O(E*S*W*U*(F+U))."
)

print(
    "Complejidad espacial: O(S*W*F + U*(F+U))."
)


# ============================================================
# 16. GUARDAR REPORTE DE TIEMPO Y COMPLEJIDAD
# ============================================================

ruta_reporte_tiempo = os.path.join(

    carpeta_salida,

    "reporte_tiempo_y_complejidad.txt"
)

with open(
    ruta_reporte_tiempo,
    "w",
    encoding="utf-8"
) as archivo:

    archivo.write(
        "REPORTE DE TIEMPO Y COMPLEJIDAD - MODELOS LSTM\n"
    )

    archivo.write(
        "=" * 60
        + "\n"
    )

    archivo.write(
        f"Dispositivo utilizado: {DISPOSITIVO}\n"
    )

    archivo.write(

        "Tiempo de carga del dataset: "

        f"{tiempo_carga_dataset:.6f} segundos "

        f"({formatear_tiempo(tiempo_carga_dataset)})\n"
    )

    archivo.write(

        "Tiempo de preprocesamiento general: "

        f"{tiempo_preprocesamiento:.6f} segundos "

        f"({formatear_tiempo(tiempo_preprocesamiento)})\n"
    )

    archivo.write(

        "Tiempo total del script: "

        f"{tiempo_total_programa:.6f} segundos "

        f"({formatear_tiempo(tiempo_total_programa)})\n\n"
    )

    archivo.write(
        "COMPLEJIDAD COMPUTACIONAL\n"
    )

    archivo.write(
        "-" * 60
        + "\n"
    )

    archivo.write(
        "Complejidad temporal dominante:\n"
    )

    archivo.write(
        "O(E*S*W*U*(F+U))\n\n"
    )

    archivo.write(

        "E = épocas efectivamente entrenadas.\n"
        "S = número de secuencias.\n"
        "W = longitud de la ventana temporal.\n"
        "U = unidades de la capa LSTM.\n"
        "F = número de variables de entrada.\n\n"
    )

    archivo.write(
        "Ordenación de los datos: O(N log N).\n"
    )

    archivo.write(
        "Creación de secuencias: O(S*W*F).\n"
    )

    archivo.write(
        "Imputación y escalamiento: O(S*W*F).\n"
    )

    archivo.write(
        "Entrenamiento LSTM: O(E*S*W*U*(F+U)).\n"
    )

    archivo.write(
        "Predicción: O(S_test*W*U*(F+U)).\n"
    )

    archivo.write(

        "Complejidad espacial: "
        "O(S*W*F + U*(F+U)).\n\n"
    )

    if len(resumen) > 0:

        archivo.write(
            "TIEMPOS POR MODELO\n"
        )

        archivo.write(
            "-" * 60
            + "\n"
        )

        for _, fila in resumen.iterrows():

            archivo.write(
                f"Base: {fila['base']}\n"
            )

            archivo.write(

                "  Dispositivo: "

                f"{fila['dispositivo']}\n"
            )

            archivo.write(

                "  Secuencias totales: "

                f"{int(fila['n_secuencias_totales'])}\n"
            )

            archivo.write(

                "  Secuencias de entrenamiento: "

                f"{int(fila['n_entrenamiento'])}\n"
            )

            archivo.write(

                "  Secuencias de validación: "

                f"{int(fila['n_validacion'])}\n"
            )

            archivo.write(

                "  Secuencias de prueba: "

                f"{int(fila['n_prueba'])}\n"
            )

            archivo.write(

                "  Épocas entrenadas: "

                f"{int(fila['epocas_entrenadas'])}\n"
            )

            archivo.write(

                "  Tiempo de creación de secuencias: "

                f"{fila['tiempo_secuencias_segundos']:.6f} s\n"
            )

            archivo.write(

                "  Tiempo de escalamiento: "

                f"{fila['tiempo_escalamiento_segundos']:.6f} s\n"
            )

            archivo.write(

                "  Tiempo de entrenamiento: "

                f"{fila['tiempo_entrenamiento_segundos']:.6f} s "
                f"({formatear_tiempo(fila['tiempo_entrenamiento_segundos'])})\n"
            )

            archivo.write(

                "  Promedio por época: "

                f"{fila['segundos_por_epoca']:.6f} s\n"
            )

            archivo.write(

                "  Tiempo estimado para 200 épocas: "

                f"{fila['tiempo_estimado_200_epocas_segundos']:.6f} s "
                f"({formatear_tiempo(fila['tiempo_estimado_200_epocas_segundos'])})\n"
            )

            archivo.write(

                "  Tiempo de predicción: "

                f"{fila['tiempo_prediccion_segundos']:.6f} s\n"
            )

            archivo.write(

                "  Tiempo total del modelo: "

                f"{fila['tiempo_total_modelo_segundos']:.6f} s "
                f"({formatear_tiempo(fila['tiempo_total_modelo_segundos'])})\n"
            )

            archivo.write(

                "  Complejidad temporal: "

                f"{fila['complejidad_temporal']}\n"
            )

            archivo.write(

                "  Factor relativo de entrenamiento: "

                f"{fila['factor_complejidad_entrenamiento']:.3e}\n\n"
            )


# ============================================================
# 17. FINALIZAR
# ============================================================

print("\n======================================")
print("ARCHIVOS GENERADOS")
print("======================================")

print(
    "Carpeta de salida:",
    carpeta_salida
)

print(
    "Reporte de tiempo:",
    ruta_reporte_tiempo
)

if len(resumen) > 0:

    print(
        "Resumen completo:",
        ruta_resumen_completo
    )

    print(
        "Resumen redondeado:",
        ruta_resumen_redondeado
    )

plt.close("all")

print("\nEjecución finalizada correctamente.")