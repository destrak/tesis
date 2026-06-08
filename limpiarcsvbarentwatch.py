import pandas as pd

# ==========================
# ARCHIVO DE ENTRADA
# ==========================

archivo_entrada = "barentswatch_locality_33077_lice_2014_2024.csv"

# ==========================
# LEER CSV ORIGINAL
# ==========================

df = pd.read_csv(archivo_entrada, encoding="utf-8-sig")

print("Columnas originales:")
print(df.columns.tolist())

# ==========================
# FUNCIÓN PARA BUSCAR COLUMNAS
# ==========================

def buscar_columna(df, posibles_nombres):
    for posible in posibles_nombres:
        for col in df.columns:
            if posible.lower() in str(col).lower():
                return col
    raise ValueError(f"No se encontró ninguna columna para: {posibles_nombres}")

# ==========================
# IDENTIFICAR COLUMNAS
# ==========================

col_anio = buscar_columna(df, ["År", "Ã", "r"])
col_semana = buscar_columna(df, ["Uke"])
col_localidad = buscar_columna(df, ["Lokalitetsnummer"])
col_hembras = buscar_columna(df, ["Voksne hunnlus"])
col_adultos = buscar_columna(df, ["Lus i bevegelige stadier"])
col_juveniles = buscar_columna(df, ["Fastsittende lus"])

# ==========================
# EXTRAER Y RENOMBRAR
# ==========================

df_limpio = df[
    [
        col_anio,
        col_semana,
        col_localidad,
        col_hembras,
        col_adultos,
        col_juveniles
    ]
].copy()

df_limpio.columns = [
    "año",
    "semana",
    "localidad",
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles"
]

# ==========================
# CONVERTIR A NUMÉRICO
# ==========================

for col in df_limpio.columns:
    df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")

# ==========================
# ORDENAR Y GUARDAR
# ==========================

df_limpio = df_limpio.sort_values(["año", "semana"]).reset_index(drop=True)

archivo_salida = "barentswatch_33077_piojos_2014_2024_extraido.csv"

df_limpio.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

print("Archivo generado correctamente:")
print(archivo_salida)

print("Dimensiones:", df_limpio.shape)
print(df_limpio.head(10))