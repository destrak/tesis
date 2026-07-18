import pandas as pd
import numpy as np
import os
from scipy.stats import shapiro, ttest_rel, wilcoxon

# ============================================================
# RUTAS PRINCIPALES
# ============================================================

ruta_chile = "dataset/datasetchile"
ruta_noruega = "dataset/datasetnoruega"

salida = "Resultados_Tests_Temperatura"
os.makedirs(salida, exist_ok=True)

# ============================================================
# ARCHIVOS CHILE Y NORUEGA
# ============================================================

archivos_chile = {
    "Los Lagos": "cargas_parasitarias_chile_102424_piojos_temp_sal_2014_2024.csv",
    "Aysén": "cargas_parasitarias_chile_110758_piojos_temp_sal_2014_2024_con_estacion_y_totales.csv",
    "Magallanes": "cargas_parasitarias_chile_120128_piojos_temp_sal_2014_2024_con_estacion_y_totales.csv"
}

archivos_noruega = {
    "Norte": "barentswatch_24175_piojos_temp_sal_2014_2024.csv",
    "Centro": "barentswatch_32677_piojos_temp_sal_2014_2024.csv",
    "Sur/Oeste": "barentswatch_33077_piojos_temp_sal_2014_2024.csv"
}

pares_comparativos = {
    "Los_Lagos_vs_Sur_Oeste": ("Los Lagos", "Sur/Oeste"),
    "Aysen_vs_Centro": ("Aysén", "Centro"),
    "Magallanes_vs_Norte": ("Magallanes", "Norte")
}

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_columnas(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("á", "a", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace("ñ", "n", regex=False)
        .str.replace("°", "", regex=False)
    )
    return df


def detectar_columna_anio(df):
    posibles = ["anio", "ano", "year", "año"]

    for col in posibles:
        if col in df.columns:
            return col

    for col in df.columns:
        if "year" in col or "anio" in col or "ano" in col:
            return col

    raise ValueError(f"No se encontró columna de año. Columnas disponibles: {list(df.columns)}")


def detectar_columna_semana(df):
    posibles = ["semana", "week", "semana_epidemiologica"]

    for col in posibles:
        if col in df.columns:
            return col

    for col in df.columns:
        if "semana" in col or "week" in col:
            return col

    raise ValueError(f"No se encontró columna de semana. Columnas disponibles: {list(df.columns)}")


def detectar_columna_temperatura(df):
    posibles = [
        "temperatura",
        "temperatura_c",
        "temperatura_agua",
        "temp",
        "temp_c",
        "temperature",
        "temperature_c",
        "sst",
        "sea_surface_temperature"
    ]

    for col in posibles:
        if col in df.columns:
            return col

    for col in df.columns:
        if "temp" in col or "temperatura" in col or "temperature" in col:
            return col

    raise ValueError(f"No se encontró columna de temperatura. Columnas disponibles: {list(df.columns)}")


def detectar_columna_salinidad(df):
    posibles = [
        "salinidad",
        "salinidad_psu",
        "salinity",
        "salinity_psu",
        "salt",
        "psu"
    ]

    for col in posibles:
        if col in df.columns:
            return col

    for col in df.columns:
        if "sal" in col or "salinity" in col or "psu" in col:
            return col

    raise ValueError(f"No se encontró columna de salinidad. Columnas disponibles: {list(df.columns)}")

# ============================================================
# CARGA DE DATOS CHILE
# ============================================================

def cargar_chile():
    lista = []

    for region, archivo in archivos_chile.items():
        path = os.path.join(ruta_chile, archivo)

        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo de Chile: {path}")

        df = pd.read_csv(path)
        df = normalizar_columnas(df)

        col_anio = detectar_columna_anio(df)
        col_semana = detectar_columna_semana(df)
        col_temp = detectar_columna_temperatura(df)
        col_sal = detectar_columna_salinidad(df)

        temp = df[[col_anio, col_semana, col_temp, col_sal]].copy()
        temp.columns = ["anio", "semana", "temperatura_chile", "salinidad_chile"]
        temp["region_chile"] = region

        temp["anio"] = pd.to_numeric(temp["anio"], errors="coerce")
        temp["semana"] = pd.to_numeric(temp["semana"], errors="coerce")
        temp["temperatura_chile"] = pd.to_numeric(temp["temperatura_chile"], errors="coerce")
        temp["salinidad_chile"] = pd.to_numeric(temp["salinidad_chile"], errors="coerce")

        lista.append(temp)

    return pd.concat(lista, ignore_index=True)

# ============================================================
# CARGA DE DATOS NORUEGA
# ============================================================

def cargar_noruega():
    lista = []

    for macrozona, archivo in archivos_noruega.items():
        path = os.path.join(ruta_noruega, archivo)

        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo de Noruega: {path}")

        df = pd.read_csv(path)
        df = normalizar_columnas(df)

        col_anio = detectar_columna_anio(df)
        col_semana = detectar_columna_semana(df)
        col_temp = detectar_columna_temperatura(df)
        col_sal = detectar_columna_salinidad(df)

        temp = df[[col_anio, col_semana, col_temp, col_sal]].copy()
        temp.columns = ["anio", "semana", "temperatura_noruega", "salinidad_noruega"]
        temp["macrozona_noruega"] = macrozona

        temp["anio"] = pd.to_numeric(temp["anio"], errors="coerce")
        temp["semana"] = pd.to_numeric(temp["semana"], errors="coerce")
        temp["temperatura_noruega"] = pd.to_numeric(temp["temperatura_noruega"], errors="coerce")
        temp["salinidad_noruega"] = pd.to_numeric(temp["salinidad_noruega"], errors="coerce")

        lista.append(temp)

    return pd.concat(lista, ignore_index=True)

# ============================================================
# HOMOLOGACIÓN TEMPORAL CHILE -> NORUEGA
# ============================================================

def homologar_semana_chile(df_chile):
    df = df_chile.copy()

    df["semana_noruega_equivalente"] = df["semana"] + 26
    df["anio_noruega_equivalente"] = df["anio"]

    mask = df["semana_noruega_equivalente"] > 52

    df.loc[mask, "semana_noruega_equivalente"] = (
        df.loc[mask, "semana_noruega_equivalente"] - 52
    )

    df.loc[mask, "anio_noruega_equivalente"] = (
        df.loc[mask, "anio_noruega_equivalente"] + 1
    )

    return df

# ============================================================
# EVALUACIÓN DE RESTRICCIÓN AMBIENTAL
# ============================================================

def evaluar_restriccion_ambiental(df):
    df = df.copy()

    df["delta_temperatura"] = df["temperatura_chile"] - df["temperatura_noruega"]
    df["delta_salinidad"] = df["salinidad_chile"] - df["salinidad_noruega"]

    df["delta_temperatura_abs"] = df["delta_temperatura"].abs()
    df["delta_salinidad_abs"] = df["delta_salinidad"].abs()

    df["cumple_temperatura_3C"] = df["delta_temperatura_abs"] <= 3
    df["cumple_salinidad_3PSU"] = df["delta_salinidad_abs"] <= 3

    df["cumple_restriccion_ambiental"] = (
        df["cumple_temperatura_3C"] &
        df["cumple_salinidad_3PSU"]
    )

    return df

# ============================================================
# TEST ESTADÍSTICO PAREADO DE TEMPERATURA
# ============================================================

def aplicar_test_temperatura(datos, nombre_comparacion):
    datos = datos.dropna(subset=["temperatura_chile", "temperatura_noruega"]).copy()

    temp_chile = datos["temperatura_chile"]
    temp_noruega = datos["temperatura_noruega"]

    diferencias = temp_chile - temp_noruega
    n = len(diferencias)

    if n < 3:
        return {
            "comparacion": nombre_comparacion,
            "n_pares": n,
            "temperatura_media_chile": np.nan,
            "temperatura_media_noruega": np.nan,
            "diferencia_media_chile_menos_noruega": np.nan,
            "desviacion_diferencias": np.nan,
            "p_shapiro": np.nan,
            "normalidad_diferencias": "No aplicable",
            "prueba_aplicada": "No aplicable",
            "estadistico": np.nan,
            "p_value": np.nan,
            "conclusion": "No hay suficientes datos"
        }

    stat_shapiro, p_shapiro = shapiro(diferencias)

    alpha = 0.05

    if p_shapiro >= alpha:
        prueba = "t de Student pareada"
        stat, p_value = ttest_rel(temp_chile, temp_noruega)
    else:
        prueba = "Wilcoxon signed-rank"
        stat, p_value = wilcoxon(temp_chile, temp_noruega)

    conclusion = (
        "Existe diferencia estadísticamente significativa"
        if p_value < alpha
        else "No existe diferencia estadísticamente significativa"
    )

    return {
        "comparacion": nombre_comparacion,
        "n_pares": n,
        "temperatura_media_chile": round(temp_chile.mean(), 4),
        "temperatura_media_noruega": round(temp_noruega.mean(), 4),
        "diferencia_media_chile_menos_noruega": round(diferencias.mean(), 4),
        "desviacion_diferencias": round(diferencias.std(), 4),
        "p_shapiro": round(p_shapiro, 8),
        "normalidad_diferencias": "Normal" if p_shapiro >= alpha else "No normal",
        "prueba_aplicada": prueba,
        "estadistico": round(stat, 6),
        "p_value": round(p_value, 8),
        "conclusion": conclusion
    }

# ============================================================
# RESUMEN DE RESTRICCIÓN AMBIENTAL
# ============================================================

def resumen_restriccion_ambiental(datos, nombre_comparacion):
    datos_validos = datos.dropna(
        subset=[
            "temperatura_chile",
            "temperatura_noruega",
            "salinidad_chile",
            "salinidad_noruega"
        ]
    ).copy()

    n_total = len(datos_validos)

    if n_total == 0:
        return {
            "comparacion": nombre_comparacion,
            "n_total_semanas": 0,
            "n_cumple_temperatura": 0,
            "porcentaje_cumple_temperatura": np.nan,
            "n_cumple_salinidad": 0,
            "porcentaje_cumple_salinidad": np.nan,
            "n_cumple_ambas": 0,
            "porcentaje_cumple_ambas": np.nan,
            "delta_temperatura_media_abs": np.nan,
            "delta_salinidad_media_abs": np.nan,
            "cumple_promedio_temperatura": "No aplicable",
            "cumple_promedio_salinidad": "No aplicable",
            "cumple_promedio_ambas": "No aplicable"
        }

    n_cumple_temp = datos_validos["cumple_temperatura_3C"].sum()
    n_cumple_sal = datos_validos["cumple_salinidad_3PSU"].sum()
    n_cumple_ambas = datos_validos["cumple_restriccion_ambiental"].sum()

    delta_temp_media_abs = datos_validos["delta_temperatura_abs"].mean()
    delta_sal_media_abs = datos_validos["delta_salinidad_abs"].mean()

    cumple_prom_temp = delta_temp_media_abs <= 3
    cumple_prom_sal = delta_sal_media_abs <= 3
    cumple_prom_ambas = cumple_prom_temp and cumple_prom_sal

    return {
        "comparacion": nombre_comparacion,
        "n_total_semanas": n_total,

        "n_cumple_temperatura": int(n_cumple_temp),
        "porcentaje_cumple_temperatura": round((n_cumple_temp / n_total) * 100, 2),

        "n_cumple_salinidad": int(n_cumple_sal),
        "porcentaje_cumple_salinidad": round((n_cumple_sal / n_total) * 100, 2),

        "n_cumple_ambas": int(n_cumple_ambas),
        "porcentaje_cumple_ambas": round((n_cumple_ambas / n_total) * 100, 2),

        "delta_temperatura_media_abs": round(delta_temp_media_abs, 4),
        "delta_salinidad_media_abs": round(delta_sal_media_abs, 4),

        "cumple_promedio_temperatura": "Sí" if cumple_prom_temp else "No",
        "cumple_promedio_salinidad": "Sí" if cumple_prom_sal else "No",
        "cumple_promedio_ambas": "Sí" if cumple_prom_ambas else "No"
    }

# ============================================================
# EJECUCIÓN PRINCIPAL
# ============================================================

df_chile = cargar_chile()
df_noruega = cargar_noruega()

df_chile_homologado = homologar_semana_chile(df_chile)

resultados_tests = []
resultados_restriccion = []

for nombre_comparacion, (region_chile, macrozona_noruega) in pares_comparativos.items():

    chile_pair = df_chile_homologado[
        df_chile_homologado["region_chile"] == region_chile
    ].copy()

    noruega_pair = df_noruega[
        df_noruega["macrozona_noruega"] == macrozona_noruega
    ].copy()

    comparacion = pd.merge(
        chile_pair,
        noruega_pair,
        left_on=["anio_noruega_equivalente", "semana_noruega_equivalente"],
        right_on=["anio", "semana"],
        how="inner",
        suffixes=("_chile", "_noruega")
    )

    comparacion = evaluar_restriccion_ambiental(comparacion)

    columnas_finales = [
        "region_chile",
        "macrozona_noruega",

        "anio_chile",
        "semana_chile",
        "anio_noruega_equivalente",
        "semana_noruega_equivalente",

        "anio_noruega",
        "semana_noruega",

        "temperatura_chile",
        "temperatura_noruega",
        "delta_temperatura",
        "delta_temperatura_abs",

        "salinidad_chile",
        "salinidad_noruega",
        "delta_salinidad",
        "delta_salinidad_abs",

        "cumple_temperatura_3C",
        "cumple_salinidad_3PSU",
        "cumple_restriccion_ambiental"
    ]

    # Renombrar columnas para evitar confusión después del merge
    comparacion = comparacion.rename(columns={
        "anio_chile": "anio_chile",
        "semana_chile": "semana_chile",
        "anio_noruega": "anio_noruega",
        "semana_noruega": "semana_noruega"
    })

    # Si pandas generó otros nombres, los ajustamos
    if "anio_x" in comparacion.columns:
        comparacion = comparacion.rename(columns={"anio_x": "anio_chile"})
    if "semana_x" in comparacion.columns:
        comparacion = comparacion.rename(columns={"semana_x": "semana_chile"})
    if "anio_y" in comparacion.columns:
        comparacion = comparacion.rename(columns={"anio_y": "anio_noruega"})
    if "semana_y" in comparacion.columns:
        comparacion = comparacion.rename(columns={"semana_y": "semana_noruega"})

    columnas_existentes = [col for col in columnas_finales if col in comparacion.columns]

    comparacion[columnas_existentes].to_csv(
        os.path.join(salida, f"dataset_comparativo_{nombre_comparacion}.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    resultado_test = aplicar_test_temperatura(comparacion, nombre_comparacion)
    resultado_restriccion = resumen_restriccion_ambiental(comparacion, nombre_comparacion)

    resultados_tests.append(resultado_test)
    resultados_restriccion.append(resultado_restriccion)

# ============================================================
# GUARDAR RESULTADOS FINALES
# ============================================================

df_tests = pd.DataFrame(resultados_tests)
df_restriccion = pd.DataFrame(resultados_restriccion)

df_tests.to_csv(
    os.path.join(salida, "resumen_tests_temperatura.csv"),
    index=False,
    encoding="utf-8-sig"
)

df_restriccion.to_csv(
    os.path.join(salida, "resumen_restriccion_ambiental.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# MOSTRAR RESULTADOS
# ============================================================

print("\n====================================================")
print("RESUMEN TESTS ESTADÍSTICOS DE TEMPERATURA")
print("====================================================\n")
print(df_tests)

print("\n====================================================")
print("RESUMEN CUMPLIMIENTO RESTRICCIÓN AMBIENTAL")
print("====================================================\n")
print(df_restriccion)

print("\nArchivos reemplazados/guardados en la carpeta:")
print(salida)