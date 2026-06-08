import pandas as pd

# ==========================
# ARCHIVO DE ENTRADA
# ==========================

archivo_entrada = "Cargas_parasitarias_2012_2024(2014-2024).csv"

codigo_centro = 120128

# ==========================
# LEER ARCHIVO
# ==========================

df = pd.read_csv(
    archivo_entrada,
    encoding="latin1",
    sep=None,
    engine="python"
)

# ==========================
# LIMPIAR CÓDIGO CENTRO
# ==========================

df["Código Centro"] = (
    df["Código Centro"]
    .astype(str)
    .str.strip()
)

# Filtrar centro específico
df = df[df["Código Centro"] == str(codigo_centro)].copy()

print("Filas encontradas:", len(df))

# ==========================
# EXTRAER COLUMNAS SIN TEMPERATURA NI SALINIDAD
# ==========================

df_final = df[
    [
        "Año",
        "semana",
        "Código Centro",
        "Prom. Hembras Ovígeras",
        "Prom. Adultos Móviles",
        "Prom. Juveniles"
    ]
].copy()

# ==========================
# RENOMBRAR COLUMNAS
# ==========================

df_final = df_final.rename(columns={
    "Año": "año",
    "semana": "semana",
    "Código Centro": "localidad",
    "Prom. Hembras Ovígeras": "hembras_ovigeras",
    "Prom. Adultos Móviles": "adultos_moviles",
    "Prom. Juveniles": "juveniles"
})

# ==========================
# CONVERTIR COMAS DECIMALES A PUNTO
# ==========================

for col in ["hembras_ovigeras", "adultos_moviles", "juveniles"]:
    df_final[col] = (
        df_final[col]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

df_final["año"] = pd.to_numeric(df_final["año"], errors="coerce")
df_final["semana"] = pd.to_numeric(df_final["semana"], errors="coerce")
df_final["localidad"] = pd.to_numeric(df_final["localidad"], errors="coerce")

# ==========================
# CREAR TODAS LAS SEMANAS 2014-2024
# ==========================

grilla = pd.MultiIndex.from_product(
    [
        range(2014, 2025),   # años 2014 a 2024
        range(1, 53)         # semanas 1 a 52
    ],
    names=["año", "semana"]
).to_frame(index=False)

grilla["localidad"] = codigo_centro

# ==========================
# UNIR CON LOS DATOS REALES
# ==========================

df_completo = grilla.merge(
    df_final,
    on=["año", "semana", "localidad"],
    how="left"
)

# ==========================
# ORDENAR
# ==========================

df_completo = df_completo.sort_values(
    ["año", "semana"]
).reset_index(drop=True)

# ==========================
# GUARDAR CON NA
# ==========================

archivo_salida = f"cargas_parasitarias_chile_{codigo_centro}_2014_2024_con_NA.csv"

df_completo.to_csv(
    archivo_salida,
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

print("Archivo generado:")
print(archivo_salida)

print("Dimensiones:")
print(df_completo.shape)

print(df_completo.head(20))
print(df_completo.tail(20))

print("NA por columna:")
print(df_completo.isna().sum())