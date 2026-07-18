import pandas as pd

# ============================================================
# COMPARACIÓN CHILE - NORUEGA CON DESFASE HEMISFÉRICO
# Chile: Los Lagos, centro 102424
# Noruega: Sur/Oeste, centro 33077
# Desfase temporal: 26 semanas
# Periodo: 2014 - semana 26 de 2024
# ============================================================

# Archivos de entrada
archivo_chile = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis"
    r"\102424_Semanal_2014_2024.csv"
)

archivo_noruega = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis"
    r"\33077_Semanal_2014_2024.csv"
)

# Archivos de salida
archivo_salida = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis"
    r"\Comparacion_Chile_Noruega_Desfase_26_semanas_"
    r"regionlagos_regionsuroeste.csv"
)

archivo_resumen = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis"
    r"\Resumen_Comparabilidad_Los_Lagos_Sur_Oeste.csv"
)

# ============================================================
# 1. Cargar datos
# ============================================================

chile = pd.read_csv(archivo_chile)
noruega = pd.read_csv(archivo_noruega)

# Convertir año y semana a valores numéricos
for df in [chile, noruega]:
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
    df["semana"] = pd.to_numeric(df["semana"], errors="coerce")

# Eliminar registros sin año o semana
chile = chile.dropna(subset=["anio", "semana"]).copy()
noruega = noruega.dropna(subset=["anio", "semana"]).copy()

chile["anio"] = chile["anio"].astype(int)
chile["semana"] = chile["semana"].astype(int)

noruega["anio"] = noruega["anio"].astype(int)
noruega["semana"] = noruega["semana"].astype(int)

# ============================================================
# 2. Definir periodo de análisis
# ============================================================
# Se emplean ciclos de 52 semanas:
# - 2014 a 2023: semanas 1 a 52
# - 2024: semanas 1 a 26
#
# Total esperado:
# 10 años x 52 semanas + 26 semanas = 546 semanas.

chile = chile[
    (chile["semana"].between(1, 52))
    & (
        (chile["anio"].between(2014, 2023))
        | (
            (chile["anio"] == 2024)
            & (chile["semana"].between(1, 26))
        )
    )
].copy()

# Eliminar semanas 53 y datos fuera del periodo requerido
noruega = noruega[
    (noruega["anio"].between(2014, 2024))
    & (noruega["semana"].between(1, 52))
].copy()

# ============================================================
# 3. Crear semana equivalente para Noruega
# ============================================================

chile["semana_noruega_equivalente"] = chile["semana"] + 26
chile["anio_noruega_equivalente"] = chile["anio"]

# Identificar semanas que pasan al año siguiente
cambio_anio = chile["semana_noruega_equivalente"] > 52

chile.loc[
    cambio_anio,
    "semana_noruega_equivalente"
] -= 52

chile.loc[
    cambio_anio,
    "anio_noruega_equivalente"
] += 1

# ============================================================
# 4. Renombrar columnas
# ============================================================

chile_comp = chile.rename(
    columns={
        "anio": "anio_chile",
        "semana": "semana_chile",
        "temperatura_media_C": "temperatura_chile_C",
        "salinidad_media_PSU": "salinidad_chile_PSU"
    }
)

noruega_comp = noruega.rename(
    columns={
        "anio": "anio_noruega",
        "semana": "semana_noruega",
        "temperatura_media_C": "temperatura_noruega_C",
        "salinidad_media_PSU": "salinidad_noruega_PSU"
    }
)

# ============================================================
# 5. Homologar Chile y Noruega
# ============================================================

comparacion = pd.merge(
    chile_comp,
    noruega_comp,
    left_on=[
        "anio_noruega_equivalente",
        "semana_noruega_equivalente"
    ],
    right_on=[
        "anio_noruega",
        "semana_noruega"
    ],
    how="inner"
)

# Ordenar cronológicamente
comparacion = comparacion.sort_values(
    ["anio_chile", "semana_chile"]
).reset_index(drop=True)

# ============================================================
# 6. Calcular diferencias ambientales
# ============================================================

# Diferencias con signo
comparacion["delta_temperatura_C"] = (
    comparacion["temperatura_chile_C"]
    - comparacion["temperatura_noruega_C"]
)

comparacion["delta_salinidad_PSU"] = (
    comparacion["salinidad_chile_PSU"]
    - comparacion["salinidad_noruega_PSU"]
)

# Diferencias absolutas
comparacion["delta_temperatura_abs_C"] = (
    comparacion["delta_temperatura_C"].abs()
)

comparacion["delta_salinidad_abs_PSU"] = (
    comparacion["delta_salinidad_PSU"].abs()
)

# ============================================================
# 7. Evaluar los criterios operacionales
# ============================================================

comparacion["cumple_temperatura"] = (
    comparacion["delta_temperatura_abs_C"] <= 3
)

comparacion["cumple_salinidad"] = (
    comparacion["delta_salinidad_abs_PSU"] <= 3
)

# Una semana es comparable solo si cumple ambos criterios
comparacion["cumple_ambos_criterios"] = (
    comparacion["cumple_temperatura"]
    & comparacion["cumple_salinidad"]
)

comparacion["comparabilidad_general"] = comparacion[
    "cumple_ambos_criterios"
].map({
    True: "Comparable",
    False: "No comparable"
})

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
    "delta_temperatura_C",
    "delta_temperatura_abs_C",
    "salinidad_chile_PSU",
    "salinidad_noruega_PSU",
    "delta_salinidad_PSU",
    "delta_salinidad_abs_PSU",
    "cumple_temperatura",
    "cumple_salinidad",
    "cumple_ambos_criterios",
    "comparabilidad_general"
]

comparacion_final = comparacion[columnas_finales].copy()

# Redondear variables continuas
columnas_decimales = [
    "temperatura_chile_C",
    "temperatura_noruega_C",
    "delta_temperatura_C",
    "delta_temperatura_abs_C",
    "salinidad_chile_PSU",
    "salinidad_noruega_PSU",
    "delta_salinidad_PSU",
    "delta_salinidad_abs_PSU"
]

comparacion_final[columnas_decimales] = (
    comparacion_final[columnas_decimales].round(3)
)

# ============================================================
# 9. Verificar cantidad de pares
# ============================================================

total_semanas = len(comparacion_final)
total_esperado = 546

if total_semanas != total_esperado:
    print(
        f"ADVERTENCIA: se esperaban {total_esperado} pares, "
        f"pero se obtuvieron {total_semanas}."
    )
else:
    print(f"Verificación correcta: {total_semanas} pares homologados.")

# ============================================================
# 10. Calcular resumen de comparabilidad
# ============================================================

semanas_cumple_temperatura = int(
    comparacion_final["cumple_temperatura"].sum()
)

semanas_cumple_salinidad = int(
    comparacion_final["cumple_salinidad"].sum()
)

semanas_comparables = int(
    comparacion_final["cumple_ambos_criterios"].sum()
)

porcentaje_temperatura = (
    semanas_cumple_temperatura / total_semanas
) * 100

porcentaje_salinidad = (
    semanas_cumple_salinidad / total_semanas
) * 100

porcentaje_comparabilidad = (
    semanas_comparables / total_semanas
) * 100

diferencia_temperatura_media = (
    comparacion_final["delta_temperatura_C"].mean()
)

diferencia_temperatura_abs_media = (
    comparacion_final["delta_temperatura_abs_C"].mean()
)

diferencia_salinidad_abs_media = (
    comparacion_final["delta_salinidad_abs_PSU"].mean()
)

# ============================================================
# 11. Guardar resultados
# ============================================================

comparacion_final.to_csv(
    archivo_salida,
    index=False,
    encoding="utf-8-sig"
)

df_resumen = pd.DataFrame({
    "comparacion": ["Los_Lagos_vs_Sur_Oeste"],
    "total_semanas": [total_semanas],
    "semanas_cumple_temperatura": [
        semanas_cumple_temperatura
    ],
    "porcentaje_cumple_temperatura": [
        round(porcentaje_temperatura, 2)
    ],
    "semanas_cumple_salinidad": [
        semanas_cumple_salinidad
    ],
    "porcentaje_cumple_salinidad": [
        round(porcentaje_salinidad, 2)
    ],
    "semanas_cumplen_ambos": [
        semanas_comparables
    ],
    "porcentaje_comparabilidad_ambiental": [
        round(porcentaje_comparabilidad, 2)
    ],
    "diferencia_temperatura_media_C": [
        round(diferencia_temperatura_media, 4)
    ],
    "diferencia_temperatura_abs_media_C": [
        round(diferencia_temperatura_abs_media, 4)
    ],
    "diferencia_salinidad_abs_media_PSU": [
        round(diferencia_salinidad_abs_media, 4)
    ]
})

df_resumen.to_csv(
    archivo_resumen,
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 12. Mostrar resultados
# ============================================================

print("\n====================================================")
print(" COMPARACIÓN CHILE - NORUEGA FINALIZADA")
print("====================================================")
print(f"Archivo semanal: {archivo_salida}")
print(f"Archivo resumen: {archivo_resumen}")

print("\n====================================================")
print(" RESUMEN DE COMPARABILIDAD AMBIENTAL")
print("====================================================")
print(f"Total de semanas homologadas : {total_semanas}")
print(
    f"Cumplen temperatura          : "
    f"{semanas_cumple_temperatura} "
    f"({porcentaje_temperatura:.2f} %)"
)
print(
    f"Cumplen salinidad            : "
    f"{semanas_cumple_salinidad} "
    f"({porcentaje_salinidad:.2f} %)"
)
print(
    f"Cumplen ambos criterios      : "
    f"{semanas_comparables} "
    f"({porcentaje_comparabilidad:.2f} %)"
)
print(
    f"Diferencia media temperatura : "
    f"{diferencia_temperatura_media:.4f} °C"
)
print(
    f"Diferencia absoluta media T  : "
    f"{diferencia_temperatura_abs_media:.4f} °C"
)
print(
    f"Diferencia absoluta media S  : "
    f"{diferencia_salinidad_abs_media:.4f} PSU"
)