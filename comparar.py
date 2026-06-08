import pandas as pd
import os

# ============================================================
# COMPARACIÓN CHILE - NORUEGA CON DESFASE HEMISFÉRICO
# Chile: Fiordo Blanco 100660
# Noruega: Verpeide 13837
# Desfase: 26 semanas
# ============================================================

# Archivos de entrada
archivo_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\102424_Semanal_2014_2024.csv"
archivo_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\33077_Semanal_2014_2024.csv"

# Archivo de salida
archivo_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Comparacion_Chile_Noruega_Desfase_26_semanas regionlagos_regionsuroeste.csv"

# ============================================================
# 1. Cargar datos
# ============================================================

chile = pd.read_csv(archivo_chile)
noruega = pd.read_csv(archivo_noruega)

# Asegurar que año y semana sean enteros
chile["anio"] = chile["anio"].astype(int)
chile["semana"] = chile["semana"].astype(int)

noruega["anio"] = noruega["anio"].astype(int)
noruega["semana"] = noruega["semana"].astype(int)

# ============================================================
# 2. Crear semana equivalente para Noruega
# ============================================================
# Para cada semana chilena, buscamos la semana noruega equivalente
# desplazada 26 semanas.

chile["semana_noruega_equivalente"] = chile["semana"] + 26

# Si pasa de 52, vuelve al inicio
chile.loc[chile["semana_noruega_equivalente"] > 52, "semana_noruega_equivalente"] -= 52

# ============================================================
# 3. Ajustar año equivalente de Noruega
# ============================================================
# Si la semana chilena + 26 pasa de 52, la semana equivalente noruega
# cae al inicio del año siguiente.

chile["anio_noruega_equivalente"] = chile["anio"]

chile.loc[chile["semana"] + 26 > 52, "anio_noruega_equivalente"] = chile["anio"] + 1

# Como tenemos datos solo hasta 2024, se eliminarán comparaciones que pidan Noruega 2025.

# ============================================================
# 4. Renombrar columnas para evitar confusión
# ============================================================

chile_comp = chile.rename(columns={
    "anio": "anio_chile",
    "semana": "semana_chile",
    "temperatura_media_C": "temperatura_chile_C",
    "salinidad_media_PSU": "salinidad_chile_PSU"
})

noruega_comp = noruega.rename(columns={
    "anio": "anio_noruega",
    "semana": "semana_noruega",
    "temperatura_media_C": "temperatura_noruega_C",
    "salinidad_media_PSU": "salinidad_noruega_PSU"
})

# ============================================================
# 5. Unir Chile con Noruega usando año y semana equivalente
# ============================================================

comparacion = pd.merge(
    chile_comp,
    noruega_comp,
    left_on=["anio_noruega_equivalente", "semana_noruega_equivalente"],
    right_on=["anio_noruega", "semana_noruega"],
    how="inner"
)

# ============================================================
# 6. Calcular diferencias ambientales
# ============================================================

comparacion["dif_temperatura_C"] = (
    comparacion["temperatura_chile_C"] - comparacion["temperatura_noruega_C"]
).abs()

comparacion["dif_salinidad_PSU"] = (
    comparacion["salinidad_chile_PSU"] - comparacion["salinidad_noruega_PSU"]
).abs()

# ============================================================
# 7. Clasificar comparabilidad
# ============================================================

def clasificar_temp(dif):
    if dif <= 3:
        return "Muy similar"
    elif dif <= 5:
        return "Diferente"
    else:
        return "No comparable"

def clasificar_sal(dif):
    if dif <= 3:
        return "Muy similar"
    elif dif <= 5:
        return "Diferente"
    else:
        return "No comparable"

comparacion["comparabilidad_temperatura"] = comparacion["dif_temperatura_C"].apply(clasificar_temp)
comparacion["comparabilidad_salinidad"] = comparacion["dif_salinidad_PSU"].apply(clasificar_sal)

# Criterio general
comparacion["comparabilidad_general"] = comparacion.apply(
    lambda row: "Comparable"
    if row["dif_temperatura_C"] <= 3 and row["dif_salinidad_PSU"] <= 3
    else "No comparable / revisar",
    axis=1
)

# ============================================================
# 8. Seleccionar columnas finales
# ============================================================

columnas_finales = [
    "anio_chile",
    "semana_chile",
    "anio_noruega",
    "semana_noruega",
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

# Mantener solo columnas que existan
columnas_finales = [col for col in columnas_finales if col in comparacion.columns]

comparacion_final = comparacion[columnas_finales]

# Redondear valores numéricos
for col in comparacion_final.select_dtypes(include="number").columns:
    comparacion_final[col] = comparacion_final[col].round(3)

# ============================================================
# 9. Guardar resultado semanal
# ============================================================

comparacion_final.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

print("====================================================")
print(" COMPARACIÓN CHILE - NORUEGA FINALIZADA")
print("====================================================")
print(f"Archivo guardado en:")
print(archivo_salida)
print(f"\nTotal de semanas comparadas: {len(comparacion_final)}")

print("\nVista previa:")
print(comparacion_final.head(15))

print("\nResumen de comparabilidad general:")
print(comparacion_final["comparabilidad_general"].value_counts())

# ============================================================
# 10. Calcular porcentaje de comparabilidad
# ============================================================

total_semanas = len(comparacion_final)

semanas_comparables = (
    comparacion_final["comparabilidad_general"] == "Comparable"
).sum()

porcentaje_comparabilidad = (semanas_comparables / total_semanas) * 100

if porcentaje_comparabilidad >= 90:
    conclusion_comparabilidad = "Similar"
else:
    conclusion_comparabilidad = "No similar"

print("\n====================================================")
print(" RESUMEN DE COMPARABILIDAD AMBIENTAL")
print("====================================================")
print(f"Total de semanas comparadas      : {total_semanas}")
print(f"Semanas comparables              : {semanas_comparables}")
print(f"Porcentaje de comparabilidad     : {porcentaje_comparabilidad:.2f} %")
print(f"Criterio                         : >= 90 %")
print(f"Conclusión                       : {conclusion_comparabilidad}")

# ============================================================
# 11. Guardar resumen de comparabilidad
# ============================================================

archivo_resumen = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Resumen_Comparabilidad_Chile_Noruega.csv"

df_resumen = pd.DataFrame({
    "total_semanas_comparadas": [total_semanas],
    "semanas_comparables": [semanas_comparables],
    "porcentaje_comparabilidad": [round(porcentaje_comparabilidad, 2)],
    "criterio_similitud": [">= 90 %"],
    "conclusion": [conclusion_comparabilidad]
})

df_resumen.to_csv(archivo_resumen, index=False, encoding="utf-8-sig")

print(f"\nResumen guardado en:")
print(archivo_resumen)