import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Ruta del datasetT1
# ============================================================

carpeta = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_mejorado_gridsearch"
archivo_dataset = os.path.join(carpeta, "datasetT1.csv")

df = pd.read_csv(archivo_dataset)

# ============================================================
# Variables a revisar
# ============================================================

variables_revisar = [
    "parasitos_totales_t1",
    "parasitos_totales",
    "parasitos_totales_lag1",
    "parasitos_totales_lag2",
    "parasitos_totales_lag3",
    "parasitos_totales_lag4",
    "parasitos_totales_media_4",
    "parasitos_totales_max_4",
    "delta_parasitos_1",

    "adultos_moviles",
    "adultos_moviles_lag1",
    "adultos_moviles_media_4",
    "adultos_moviles_max_4",
    "delta_adultos_1",

    "hembras_ovigeras",
    "hembras_ovigeras_lag1",
    "hembras_ovigeras_media_4",
    "hembras_ovigeras_max_4",
    "delta_hembras_1",

    "juveniles",
    "juveniles_lag1",
    "juveniles_lag2",
    "juveniles_media_4",
    "juveniles_max_4",
    "delta_juveniles_1",

    "temperatura",
    "temperatura_lag1",
    "temperatura_media_4",
    "delta_temperatura_1",

    "salinidad",
    "salinidad_lag1",
    "salinidad_media_4",
    "delta_salinidad_1",

    "semana_sin",
    "semana_cos",
    "anio"
]

variables_revisar = [v for v in variables_revisar if v in df.columns]

df_corr = df[variables_revisar].copy()

# Correlación Spearman: útil si la relación no es perfectamente lineal
corr = df_corr.corr(method="spearman")

# ============================================================
# Graficar heatmap de correlación con valores
# ============================================================

plt.figure(figsize=(18, 15))

imagen = plt.imshow(corr, aspect="auto", vmin=-1, vmax=1)

plt.colorbar(imagen, label="Correlación de Spearman")

plt.xticks(
    ticks=np.arange(len(corr.columns)),
    labels=corr.columns,
    rotation=90
)

plt.yticks(
    ticks=np.arange(len(corr.index)),
    labels=corr.index
)

# Agregar valores dentro de cada celda
for i in range(corr.shape[0]):
    for j in range(corr.shape[1]):
        valor = corr.iloc[i, j]

        # Para que el texto se vea según el color de fondo
        color_texto = "white" if valor < -0.3 else "black"

        plt.text(
            j,
            i,
            f"{valor:.2f}",
            ha="center",
            va="center",
            color=color_texto,
            fontsize=5
        )

plt.title("Heatmap de correlación entre variables del Random Forest")
plt.tight_layout()

salida = os.path.join(carpeta, "heatmap_correlacion_variables_rf_con_valores.png")
plt.savefig(salida, dpi=300, bbox_inches="tight")
plt.show()

print("Heatmap guardado en:")
print(salida)