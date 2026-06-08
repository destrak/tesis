import pandas as pd

# ==========================
# ARCHIVOS DE ENTRADA
# ==========================

archivo_piojos = "cargas_parasitarias_chile_110758_2014_2024_con_NA.csv"
archivo_ambiental = "110758_Semanal_2014_2024.csv"

# ==========================
# LEER ARCHIVOS
# ==========================

piojos = pd.read_csv(archivo_piojos, encoding="utf-8-sig")
amb = pd.read_csv(archivo_ambiental, encoding="utf-8-sig")

# ==========================
# SELECCIONAR TEMPERATURA Y SALINIDAD
# ==========================

amb_limpio = amb[
    [
        "anio",
        "semana",
        "temperatura_media_C",
        "salinidad_media_PSU"
    ]
].copy()

amb_limpio = amb_limpio.rename(columns={
    "anio": "año",
    "temperatura_media_C": "temperatura",
    "salinidad_media_PSU": "salinidad"
})

# ==========================
# UNIR POR AÑO Y SEMANA
# ==========================

df_final = piojos.merge(
    amb_limpio,
    on=["año", "semana"],
    how="left"
)

# ==========================
# ORDENAR COLUMNAS
# ==========================

df_final = df_final[
    [
        "año",
        "semana",
        "localidad",
        "hembras_ovigeras",
        "adultos_moviles",
        "juveniles",
        "temperatura",
        "salinidad"
    ]
]

# ==========================
# ORDENAR FILAS
# ==========================

df_final = df_final.sort_values(["año", "semana"]).reset_index(drop=True)

# ==========================
# GUARDAR CON NULOS COMO NA
# ==========================

df_final.to_csv(
    "cargas_parasitarias_chile_110758_piojos_temp_sal_2014_2024.csv",
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

print("Archivo generado correctamente")
print(df_final.head())
print(df_final.tail())
print(df_final.isna().sum())