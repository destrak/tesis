import pandas as pd
import os

# ============================================================
# PROMEDIO SEMANAL DE TEMPERATURA Y SALINIDAD
# Centro: ACS 54 B
# Locality: 120128
# ============================================================

# Archivo diario consolidado
archivo_entrada = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\120128_Consolidado_2014_2024.csv"

# Archivo semanal de salida
archivo_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\120128_Semanal_2014_2024.csv"

print("====================================================")
print(" PROMEDIO SEMANAL -120128")
print("====================================================")

# ============================================================
# 1. Leer CSV diario
# ============================================================

df = pd.read_csv(archivo_entrada)

print("Columnas encontradas:")
print(df.columns.tolist())

# ============================================================
# 2. Convertir fecha a formato datetime
# ============================================================

df["fecha"] = pd.to_datetime(df["fecha"])

# ============================================================
# 3. Crear año y semana ISO
# ============================================================
# ISO week: semanas estándar tipo calendario epidemiológico:
# lunes a domingo.
# Esto sirve bien para comparar con datos semanales de BarentsWatch.

df["anio"] = df["fecha"].dt.isocalendar().year
df["semana"] = df["fecha"].dt.isocalendar().week

# ============================================================
# 4. Promediar por año y semana
# ============================================================

df_semanal = (
    df.groupby(["anio", "semana"], as_index=False)
    .agg(
        temperatura_media_C=("temperatura_C", "mean"),
        salinidad_media_PSU=("salinidad_PSU", "mean"),
        temperatura_min_C=("temperatura_C", "min"),
        temperatura_max_C=("temperatura_C", "max"),
        salinidad_min_PSU=("salinidad_PSU", "min"),
        salinidad_max_PSU=("salinidad_PSU", "max"),
        n_dias=("fecha", "count")
    )
)

# ============================================================
# 5. Redondear valores
# ============================================================

df_semanal["temperatura_media_C"] = df_semanal["temperatura_media_C"].round(3)
df_semanal["salinidad_media_PSU"] = df_semanal["salinidad_media_PSU"].round(3)
df_semanal["temperatura_min_C"] = df_semanal["temperatura_min_C"].round(3)
df_semanal["temperatura_max_C"] = df_semanal["temperatura_max_C"].round(3)
df_semanal["salinidad_min_PSU"] = df_semanal["salinidad_min_PSU"].round(3)
df_semanal["salinidad_max_PSU"] = df_semanal["salinidad_max_PSU"].round(3)

# ============================================================
# 6. Guardar CSV semanal
# ============================================================

df_semanal.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

print("\nCONVERSION SEMANAL EXITOSA")
print(f"Total de semanas generadas: {len(df_semanal)}")
print(f"Archivo guardado en:")
print(archivo_salida)

print("\nVista previa:")
print(df_semanal.head(10))

print("\nUltimas semanas:")
print(df_semanal.tail(10))