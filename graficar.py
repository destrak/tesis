import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ============================================================
# GRAFICAR TEMPERATURA Y SALINIDAD
# Chile vs Noruega con desfase hemisférico
# ============================================================

archivo_entrada = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Comparacion_Chile_Noruega_Desfase_26_semanas_regionmagallanes_regionnorte.csv"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Graficos_Comparacion_magallanes_norte"
os.makedirs(carpeta_salida, exist_ok=True)

# ============================================================
# 1. Leer datos
# ============================================================

df = pd.read_csv(archivo_entrada)

print("Columnas disponibles:")
print(df.columns.tolist())

# ============================================================
# 2. Crear eje temporal desde 0
# ============================================================

df = df.sort_values(["anio_chile", "semana_chile"]).reset_index(drop=True)
df["indice_semana"] = range(0, len(df))

# ============================================================
# 3. Gráfico de temperatura
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["indice_semana"],
    df["temperatura_chile_C"],
    label="Chile",
    linewidth=1.8
)

plt.plot(
    df["indice_semana"],
    df["temperatura_noruega_C"],
    label="Noruega equivalente",
    linewidth=1.8
)

plt.title("Comparación de temperatura media semanal: Chile vs Noruega")
plt.xlabel("Semana comparada")
plt.ylabel("Temperatura media (°C)")
plt.legend()
plt.grid(True, alpha=0.3)

# Ejes
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.yticks(range(0, 31, 5))

ruta_temp = os.path.join(carpeta_salida, "comparacion_temperatura_chile_noruega.png")
plt.savefig(ruta_temp, dpi=300, bbox_inches="tight")
plt.show()

print(f"Gráfico de temperatura guardado en: {ruta_temp}")

# ============================================================
# 4. Gráfico de salinidad
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["indice_semana"],
    df["salinidad_chile_PSU"],
    label="Chile",
    linewidth=1.8
)

plt.plot(
    df["indice_semana"],
    df["salinidad_noruega_PSU"],
    label="Noruega equivalente",
    linewidth=1.8
)

plt.title("Comparación de salinidad media semanal: Chile vs Noruega")
plt.xlabel("Semana comparada")
plt.ylabel("Salinidad media (PSU)")
plt.legend()
plt.grid(True, alpha=0.3)

# Ejes
plt.xlim(left=0)
plt.ylim(0, 40)
plt.yticks(range(0, 41, 10))

ruta_sal = os.path.join(carpeta_salida, "comparacion_salinidad_chile_noruega.png")
plt.savefig(ruta_sal, dpi=300, bbox_inches="tight")
plt.show()

print(f"Gráfico de salinidad guardado en: {ruta_sal}")

# ============================================================
# 5. Gráfico de diferencias absolutas de temperatura
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["indice_semana"],
    df["dif_temperatura_C"],
    label="Diferencia temperatura",
    linewidth=1.8
)

plt.axhline(
    y=3,
    linestyle="--",
    linewidth=1.5,
    label="Umbral temperatura = 3 °C"
)

plt.title("Diferencia absoluta de temperatura entre Chile y Noruega")
plt.xlabel("Semana comparada")
plt.ylabel("Diferencia absoluta de temperatura (°C)")
plt.legend()
plt.grid(True, alpha=0.3)

# Ejes
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.yticks(np.arange(0, df["dif_temperatura_C"].max() + 2, 1))

ruta_dif_temp = os.path.join(carpeta_salida, "diferencia_temperatura_chile_noruega.png")
plt.savefig(ruta_dif_temp, dpi=300, bbox_inches="tight")
plt.show()

print(f"Gráfico de diferencia de temperatura guardado en: {ruta_dif_temp}")

# ============================================================
# 6. Gráfico de diferencias absolutas de salinidad
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    df["indice_semana"],
    df["dif_salinidad_PSU"],
    label="Diferencia salinidad",
    linewidth=1.8
)

plt.axhline(
    y=3,
    linestyle="--",
    linewidth=1.5,
    label="Umbral salinidad = 3 PSU"
)

plt.title("Diferencia absoluta de salinidad entre Chile y Noruega")
plt.xlabel("Semana comparada")
plt.ylabel("Diferencia absoluta de salinidad (PSU)")
plt.legend()
plt.grid(True, alpha=0.3)

# Ejes
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.yticks(np.arange(0, df["dif_salinidad_PSU"].max() + 1, 0.5))

ruta_dif_sal = os.path.join(carpeta_salida, "diferencia_salinidad_chile_noruega.png")
plt.savefig(ruta_dif_sal, dpi=300, bbox_inches="tight")
plt.show()

print(f"Gráfico de diferencia de salinidad guardado en: {ruta_dif_sal}")

print("\nProceso finalizado.")