import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Ruta donde están tus archivos SHAP
# ============================================================

carpeta = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\resultados_random_forest_mejorado_gridsearch"

archivo_chile = os.path.join(carpeta, "shap_importancia_rf_gridsearch_chile.csv")
archivo_conjunto = os.path.join(carpeta, "shap_importancia_rf_gridsearch_conjunto_chile_noruega.csv")
archivo_noruega = os.path.join(carpeta, "shap_importancia_rf_gridsearch_noruega.csv")

# ============================================================
# Cargar archivos
# ============================================================

shap_chile = pd.read_csv(archivo_chile)
shap_conjunto = pd.read_csv(archivo_conjunto)
shap_noruega = pd.read_csv(archivo_noruega)

shap_chile = shap_chile.rename(columns={"shap_importancia_media_abs": "Chile"})
shap_conjunto = shap_conjunto.rename(columns={"shap_importancia_media_abs": "Chile_Noruega"})
shap_noruega = shap_noruega.rename(columns={"shap_importancia_media_abs": "Noruega"})

# ============================================================
# Unir tablas
# ============================================================

df_shap = shap_chile[["variable", "Chile"]].merge(
    shap_conjunto[["variable", "Chile_Noruega"]],
    on="variable",
    how="outer"
).merge(
    shap_noruega[["variable", "Noruega"]],
    on="variable",
    how="outer"
)

df_shap = df_shap.fillna(0)

# ============================================================
# Normalizar por modelo para comparar importancia relativa
# ============================================================

for col in ["Chile", "Chile_Noruega", "Noruega"]:
    maximo = df_shap[col].max()
    if maximo > 0:
        df_shap[col + "_norm"] = df_shap[col] / maximo
    else:
        df_shap[col + "_norm"] = 0

df_shap["promedio_norm"] = df_shap[
    ["Chile_norm", "Chile_Noruega_norm", "Noruega_norm"]
].mean(axis=1)

# Elegir top 25 variables más importantes en promedio
df_top = df_shap.sort_values("promedio_norm", ascending=False)

heatmap_data = df_top.set_index("variable")[
    ["Chile_norm", "Chile_Noruega_norm", "Noruega_norm"]
]

# ============================================================
# Graficar heatmap con valores dentro de cada celda
# ============================================================

plt.figure(figsize=(9, 11))

imagen = plt.imshow(heatmap_data, aspect="auto", vmin=0, vmax=1)

plt.colorbar(imagen, label="Importancia SHAP normalizada")

plt.xticks(
    ticks=np.arange(len(heatmap_data.columns)),
    labels=["Chile", "Chile-Noruega", "Noruega"],
    rotation=0
)

plt.yticks(
    ticks=np.arange(len(heatmap_data.index)),
    labels=heatmap_data.index
)

# Agregar valores dentro de cada celda
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        valor = heatmap_data.iloc[i, j]

        # Color del texto según intensidad del fondo
        color_texto = "white" if valor < 0.45 else "black"

        plt.text(
            j,
            i,
            f"{valor:.2f}",
            ha="center",
            va="center",
            color=color_texto,
            fontsize=8
        )

plt.title("Heatmap de importancia SHAP por modelo")
plt.xlabel("Modelo")
plt.ylabel("Variable")

plt.tight_layout()

salida = os.path.join(carpeta, "heatmap_shap_importancia_modelos_con_valores.png")
plt.savefig(salida, dpi=300, bbox_inches="tight")
plt.show()

print("Heatmap guardado en:")
print(salida)