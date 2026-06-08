import pandas as pd
import os
from glob import glob

# ============================================================
# COMPARACIÓN GENERAL CHILE - NORUEGA
# Con centros, regiones/macrozona, desfase temporal,
# temperatura, salinidad y resumen por par y por región
# ============================================================

# ------------------------------------------------------------
# 1. RUTAS DE ENTRADA
# ------------------------------------------------------------

carpeta_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Chile"
carpeta_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Noruega"

# Carpeta de salida
carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Comparaciones_Tesis"
os.makedirs(carpeta_salida, exist_ok=True)

# ------------------------------------------------------------
# 2. PARÁMETROS DE COMPARACIÓN
# ------------------------------------------------------------

DESFASE_SEMANAS = 26
UMBRAL_TEMP = 3
UMBRAL_SAL = 3
CRITERIO_SIMILITUD = 90

# ============================================================
# 3. FUNCIONES AUXILIARES
# ============================================================

def obtener_codigo_archivo(ruta):
    """
    Extrae el nombre del archivo sin extensión.
    Ejemplo:
    102424_Semanal_2014_2024.csv -> 102424_Semanal_2014_2024
    """
    return os.path.splitext(os.path.basename(ruta))[0]


def asignar_region(pais, archivo_origen):
    """
    Asigna región o macrozona según el código del centro.
    """

    archivo = str(archivo_origen)

    if pais == "Chile":
        if "102424" in archivo:
            return "Los Lagos"
        elif "110758" in archivo:
            return "Aysén"
        elif "120128" in archivo:
            return "Magallanes"
        else:
            return "Chile_sin_region"

    elif pais == "Noruega":
        if "33077" in archivo:
            return "Sur/Oeste"
        elif "32677" in archivo:
            return "Centro"
        elif "24175" in archivo:
            return "Norte"
        else:
            return "Noruega_sin_region"

    return "Sin_region"


def normalizar_columnas(df):
    """
    Normaliza nombres de columnas para evitar problemas por espacios,
    mayúsculas o nombres distintos.
    """

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("á", "a", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace("ñ", "n", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    return df


def detectar_columna(df, posibles_nombres, nombre_final):
    """
    Busca una columna dentro de una lista de posibles nombres
    y la renombra a un nombre estándar.
    """

    for col in posibles_nombres:
        if col in df.columns:
            df = df.rename(columns={col: nombre_final})
            return df

    raise ValueError(
        f"No se encontró la columna necesaria '{nombre_final}'. "
        f"Columnas disponibles: {list(df.columns)}"
    )


def estandarizar_columnas(df):
    """
    Estandariza columnas mínimas necesarias:
    anio, semana, temperatura_media_C, salinidad_media_PSU.
    """

    df = normalizar_columnas(df)

    # Año
    df = detectar_columna(
        df,
        posibles_nombres=[
            "anio", "ano", "year"
        ],
        nombre_final="anio"
    )

    # Semana
    df = detectar_columna(
        df,
        posibles_nombres=[
            "semana", "week"
        ],
        nombre_final="semana"
    )

    # Temperatura
    df = detectar_columna(
        df,
        posibles_nombres=[
            "temperatura_media_c",
            "temperatura_c",
            "temperatura",
            "sea_temperature",
            "temperature",
            "temp",
            "temp_c"
        ],
        nombre_final="temperatura_media_C"
    )

    # Salinidad
    df = detectar_columna(
        df,
        posibles_nombres=[
            "salinidad_media_psu",
            "salinidad_psu",
            "salinidad",
            "salinity",
            "sal_psu"
        ],
        nombre_final="salinidad_media_PSU"
    )

    return df


def cargar_archivos(carpeta, pais):
    """
    Carga todos los CSV de una carpeta, asigna país y región,
    y une todo en una sola base.
    """

    archivos = glob(os.path.join(carpeta, "*.csv"))

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {carpeta}")

    lista_df = []

    for archivo in archivos:
        print(f"Cargando archivo: {archivo}")

        try:
            df = pd.read_csv(archivo)
        except UnicodeDecodeError:
            df = pd.read_csv(archivo, encoding="latin1")

        df = estandarizar_columnas(df)

        df["archivo_origen"] = obtener_codigo_archivo(archivo)
        df["pais"] = pais
        df["region"] = df["archivo_origen"].apply(lambda x: asignar_region(pais, x))

        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
        df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
        df["temperatura_media_C"] = pd.to_numeric(df["temperatura_media_C"], errors="coerce")
        df["salinidad_media_PSU"] = pd.to_numeric(df["salinidad_media_PSU"], errors="coerce")

        df = df.dropna(subset=["anio", "semana"])

        df["anio"] = df["anio"].astype(int)
        df["semana"] = df["semana"].astype(int)

        lista_df.append(df)

    if len(lista_df) == 0:
        raise ValueError(f"No se pudo cargar ningún archivo válido desde: {carpeta}")

    return pd.concat(lista_df, ignore_index=True)


def clasificar_variable(dif, umbral):
    """
    Clasifica una diferencia individual de temperatura o salinidad.
    """

    if pd.isna(dif):
        return "Dato faltante"
    elif dif <= umbral:
        return "Muy similar"
    elif dif <= 5:
        return "Diferente"
    else:
        return "No comparable"


def clasificar_comparabilidad(dif_temp, dif_sal):
    """
    Clasifica la comparabilidad general.
    Una semana es comparable solo si cumple:
    |ΔT| <= 3 °C y |ΔS| <= 3 PSU.
    """

    if pd.isna(dif_temp) or pd.isna(dif_sal):
        return "No comparable / datos faltantes"

    if dif_temp <= UMBRAL_TEMP and dif_sal <= UMBRAL_SAL:
        return "Comparable"
    else:
        return "No comparable / revisar"


def limpiar_nombre_archivo(nombre):
    """
    Limpia nombres para guardar archivos sin errores.
    """

    return (
        str(nombre)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(";", "_")
    )


# ============================================================
# 4. CARGAR BASES CHILE Y NORUEGA
# ============================================================

print("\n====================================================")
print(" CARGANDO ARCHIVOS DE CHILE")
print("====================================================")

chile = cargar_archivos(carpeta_chile, "Chile")

print("\n====================================================")
print(" CARGANDO ARCHIVOS DE NORUEGA")
print("====================================================")

noruega = cargar_archivos(carpeta_noruega, "Noruega")

# Guardar bases unificadas
ruta_chile_unido = os.path.join(carpeta_salida, "Base_Chile_Unificada.csv")
ruta_noruega_unido = os.path.join(carpeta_salida, "Base_Noruega_Unificada.csv")

chile.to_csv(ruta_chile_unido, index=False, encoding="utf-8-sig")
noruega.to_csv(ruta_noruega_unido, index=False, encoding="utf-8-sig")

print(f"\nBase Chile guardada en: {ruta_chile_unido}")
print(f"Base Noruega guardada en: {ruta_noruega_unido}")

# ============================================================
# 5. CREAR SEMANA EQUIVALENTE DE NORUEGA PARA CADA REGISTRO CHILENO
# ============================================================

chile_comp = chile.copy()

chile_comp["semana_noruega_equivalente"] = chile_comp["semana"] + DESFASE_SEMANAS

chile_comp.loc[
    chile_comp["semana_noruega_equivalente"] > 52,
    "semana_noruega_equivalente"
] -= 52

chile_comp["anio_noruega_equivalente"] = chile_comp["anio"]

chile_comp.loc[
    chile_comp["semana"] + DESFASE_SEMANAS > 52,
    "anio_noruega_equivalente"
] = chile_comp["anio"] + 1

# ============================================================
# 6. RENOMBRAR COLUMNAS PARA COMPARACIÓN
# ============================================================

chile_comp = chile_comp.rename(columns={
    "anio": "anio_chile",
    "semana": "semana_chile",
    "archivo_origen": "centro_chile",
    "region": "region_chile",
    "temperatura_media_C": "temperatura_chile_C",
    "salinidad_media_PSU": "salinidad_chile_PSU"
})

noruega_comp = noruega.rename(columns={
    "anio": "anio_noruega",
    "semana": "semana_noruega",
    "archivo_origen": "centro_noruega",
    "region": "region_noruega",
    "temperatura_media_C": "temperatura_noruega_C",
    "salinidad_media_PSU": "salinidad_noruega_PSU"
})

# ============================================================
# 7. COMPARAR TODOS LOS CENTROS DE CHILE CON TODOS LOS DE NORUEGA
# ============================================================

comparacion = pd.merge(
    chile_comp,
    noruega_comp,
    left_on=["anio_noruega_equivalente", "semana_noruega_equivalente"],
    right_on=["anio_noruega", "semana_noruega"],
    how="inner",
    suffixes=("_chile", "_noruega")
)

if comparacion.empty:
    raise ValueError(
        "La comparación quedó vacía. Revisa que existan años y semanas equivalentes "
        "entre Chile y Noruega después del desfase de 26 semanas."
    )

# ============================================================
# 8. CALCULAR DIFERENCIAS AMBIENTALES
# ============================================================

comparacion["dif_temperatura_C"] = (
    comparacion["temperatura_chile_C"] - comparacion["temperatura_noruega_C"]
).abs()

comparacion["dif_salinidad_PSU"] = (
    comparacion["salinidad_chile_PSU"] - comparacion["salinidad_noruega_PSU"]
).abs()

comparacion["comparabilidad_temperatura"] = comparacion["dif_temperatura_C"].apply(
    lambda x: clasificar_variable(x, UMBRAL_TEMP)
)

comparacion["comparabilidad_salinidad"] = comparacion["dif_salinidad_PSU"].apply(
    lambda x: clasificar_variable(x, UMBRAL_SAL)
)

comparacion["comparabilidad_general"] = comparacion.apply(
    lambda row: clasificar_comparabilidad(
        row["dif_temperatura_C"],
        row["dif_salinidad_PSU"]
    ),
    axis=1
)

# ============================================================
# 9. SELECCIONAR COLUMNAS PRINCIPALES
# ============================================================

columnas_principales = [
    "centro_chile",
    "region_chile",
    "centro_noruega",
    "region_noruega",
    "anio_chile",
    "semana_chile",
    "anio_noruega",
    "semana_noruega",
    "semana_noruega_equivalente",
    "anio_noruega_equivalente",
    "temperatura_chile_C",
    "temperatura_noruega_C",
    "dif_temperatura_C",
    "salinidad_chile_PSU",
    "salinidad_noruega_PSU",
    "dif_salinidad_PSU",
    "comparabilidad_temperatura",
    "comparabilidad_salinidad",
    "comparabilidad_general"
]

columnas_principales = [col for col in columnas_principales if col in comparacion.columns]

comparacion_final = comparacion[columnas_principales].copy()

# Redondear columnas numéricas
for col in comparacion_final.select_dtypes(include="number").columns:
    comparacion_final[col] = comparacion_final[col].round(3)

# ============================================================
# 10. GUARDAR COMPARACIÓN TOTAL
# ============================================================

ruta_comparacion_total = os.path.join(
    carpeta_salida,
    "Comparacion_Todos_Chile_vs_Todos_Noruega.csv"
)

comparacion_final.to_csv(ruta_comparacion_total, index=False, encoding="utf-8-sig")

print("\n====================================================")
print(" COMPARACIÓN TOTAL CHILE - NORUEGA FINALIZADA")
print("====================================================")
print(f"Archivo guardado en: {ruta_comparacion_total}")
print(f"Total de filas comparadas: {len(comparacion_final)}")

# ============================================================
# 11. RESUMEN POR PAR CENTRO CHILE - CENTRO NORUEGA
# ============================================================

resumen_pares = (
    comparacion_final
    .groupby(["centro_chile", "region_chile", "centro_noruega", "region_noruega"])
    .agg(
        total_semanas=("comparabilidad_general", "count"),
        semanas_comparables=("comparabilidad_general", lambda x: (x == "Comparable").sum()),
        promedio_dif_temp_C=("dif_temperatura_C", "mean"),
        promedio_dif_sal_PSU=("dif_salinidad_PSU", "mean")
    )
    .reset_index()
)

resumen_pares["porcentaje_comparabilidad"] = (
    resumen_pares["semanas_comparables"] / resumen_pares["total_semanas"] * 100
).round(2)

resumen_pares["conclusion"] = resumen_pares["porcentaje_comparabilidad"].apply(
    lambda x: "Similar" if x >= CRITERIO_SIMILITUD else "No similar"
)

resumen_pares["promedio_dif_temp_C"] = resumen_pares["promedio_dif_temp_C"].round(3)
resumen_pares["promedio_dif_sal_PSU"] = resumen_pares["promedio_dif_sal_PSU"].round(3)

ruta_resumen_pares = os.path.join(
    carpeta_salida,
    "Resumen_Comparabilidad_Por_Par_Chile_Noruega.csv"
)

resumen_pares.to_csv(ruta_resumen_pares, index=False, encoding="utf-8-sig")

print(f"Resumen por pares guardado en: {ruta_resumen_pares}")

# ============================================================
# 12. RESUMEN POR REGIÓN / MACROZONA
# ============================================================

resumen_regiones = (
    comparacion_final
    .groupby(["region_chile", "region_noruega"])
    .agg(
        total_semanas=("comparabilidad_general", "count"),
        semanas_comparables=("comparabilidad_general", lambda x: (x == "Comparable").sum()),
        promedio_dif_temp_C=("dif_temperatura_C", "mean"),
        promedio_dif_sal_PSU=("dif_salinidad_PSU", "mean")
    )
    .reset_index()
)

resumen_regiones["porcentaje_comparabilidad"] = (
    resumen_regiones["semanas_comparables"] / resumen_regiones["total_semanas"] * 100
).round(2)

resumen_regiones["conclusion"] = resumen_regiones["porcentaje_comparabilidad"].apply(
    lambda x: "Similar" if x >= CRITERIO_SIMILITUD else "No similar"
)

resumen_regiones["promedio_dif_temp_C"] = resumen_regiones["promedio_dif_temp_C"].round(3)
resumen_regiones["promedio_dif_sal_PSU"] = resumen_regiones["promedio_dif_sal_PSU"].round(3)

ruta_resumen_regiones = os.path.join(
    carpeta_salida,
    "Resumen_Comparabilidad_Por_Region_Chile_Noruega.csv"
)

resumen_regiones.to_csv(ruta_resumen_regiones, index=False, encoding="utf-8-sig")

print(f"Resumen por regiones guardado en: {ruta_resumen_regiones}")

# ============================================================
# 13. RESUMEN GLOBAL
# ============================================================

total_filas = len(comparacion_final)
semanas_comparables = (comparacion_final["comparabilidad_general"] == "Comparable").sum()

porcentaje_comparabilidad = round(
    (semanas_comparables / total_filas) * 100,
    2
)

resumen_global = pd.DataFrame({
    "total_filas_comparadas": [total_filas],
    "semanas_comparables": [semanas_comparables],
    "porcentaje_comparabilidad": [porcentaje_comparabilidad],
    "criterio_similitud": [f">= {CRITERIO_SIMILITUD} %"],
    "conclusion": [
        "Similar" if porcentaje_comparabilidad >= CRITERIO_SIMILITUD else "No similar"
    ]
})

ruta_resumen_global = os.path.join(
    carpeta_salida,
    "Resumen_Global_Comparabilidad_Chile_Noruega.csv"
)

resumen_global.to_csv(ruta_resumen_global, index=False, encoding="utf-8-sig")

print(f"Resumen global guardado en: {ruta_resumen_global}")

# ============================================================
# 14. GUARDAR COMPARACIONES INDIVIDUALES POR PAR DE CENTROS
# ============================================================

carpeta_pares = os.path.join(carpeta_salida, "Comparaciones_por_par")
os.makedirs(carpeta_pares, exist_ok=True)

for (centro_chile, centro_noruega), df_par in comparacion_final.groupby(["centro_chile", "centro_noruega"]):

    nombre_archivo = f"Comparacion_{centro_chile}_vs_{centro_noruega}.csv"
    nombre_archivo = limpiar_nombre_archivo(nombre_archivo)

    ruta_par = os.path.join(carpeta_pares, nombre_archivo)
    df_par.to_csv(ruta_par, index=False, encoding="utf-8-sig")

print(f"Comparaciones individuales por par guardadas en: {carpeta_pares}")

# ============================================================
# 15. GUARDAR COMPARACIONES POR REGIÓN / MACROZONA
# ============================================================

carpeta_regiones = os.path.join(carpeta_salida, "Comparaciones_por_region")
os.makedirs(carpeta_regiones, exist_ok=True)

for (region_chile, region_noruega), df_region in comparacion_final.groupby(["region_chile", "region_noruega"]):

    nombre_archivo = f"Comparacion_{region_chile}_vs_{region_noruega}.csv"
    nombre_archivo = limpiar_nombre_archivo(nombre_archivo)

    ruta_region = os.path.join(carpeta_regiones, nombre_archivo)
    df_region.to_csv(ruta_region, index=False, encoding="utf-8-sig")

print(f"Comparaciones por región guardadas en: {carpeta_regiones}")

# ============================================================
# 16. MOSTRAR RESÚMENES EN CONSOLA
# ============================================================

print("\n====================================================")
print(" RESUMEN GLOBAL")
print("====================================================")
print(resumen_global)

print("\n====================================================")
print(" RESUMEN POR REGIONES")
print("====================================================")
print(
    resumen_regiones
    .sort_values("porcentaje_comparabilidad", ascending=False)
)

print("\n====================================================")
print(" TOP 10 PARES DE CENTROS MÁS SIMILARES")
print("====================================================")
print(
    resumen_pares
    .sort_values("porcentaje_comparabilidad", ascending=False)
    .head(10)
)

print("\n====================================================")
print(" PROCESO FINALIZADO CORRECTAMENTE")
print("====================================================")