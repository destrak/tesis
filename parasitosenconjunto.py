import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from matplotlib.patches import Patch

# ==============================
# RUTA DEL DATASET
# ==============================

ruta = "dataset/base_chile_noruega_completa.csv"

# ==============================
# CARPETAS DE SALIDA
# ==============================

carpeta_boxplots = "boxplots_parasitos"
carpeta_lineas = "lineas_parasitos_homologadas"

os.makedirs(carpeta_boxplots, exist_ok=True)
os.makedirs(carpeta_lineas, exist_ok=True)

# ==============================
# CARGAR DATASET
# ==============================

df = pd.read_csv(ruta, na_values=["NA", ""])

# ==============================
# ASEGURAR FORMATOS
# ==============================

df["año"] = pd.to_numeric(df["año"], errors="coerce")
df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
df["parasitos_totales"] = pd.to_numeric(df["parasitos_totales"], errors="coerce")
df["localidad"] = df["localidad"].astype(str)

df = df.dropna(subset=["año", "semana", "pais", "parasitos_totales"])

df["año"] = df["año"].astype(int)
df["semana"] = df["semana"].astype(int)

# ==============================
# ASIGNAR REGIÓN / MACROZONA
# ==============================

mapa_zonas = {
    # Chile
    "102424": "Los Lagos",
    "110758": "Aysén",
    "120128": "Magallanes",

    # Noruega
    "33077": "Sur/Oeste Noruega",
    "32677": "Centro Noruega",
    "24175": "Norte Noruega"
}

df["zona"] = df["localidad"].map(mapa_zonas)

# ==============================
# ESTACIÓN BASE CHILE
# ==============================

def estacion_chile(semana):
    if pd.isna(semana):
        return np.nan

    semana = int(semana)

    if 1 <= semana <= 13:
        return "Verano"
    elif 14 <= semana <= 26:
        return "Otoño"
    elif 27 <= semana <= 39:
        return "Invierno"
    elif 40 <= semana <= 52:
        return "Primavera"
    else:
        return np.nan

# ==============================
# HOMOLOGACIÓN TEMPORAL
# ==============================

def homologar_semana(row):
    """
    Chile mantiene su semana original.
    Noruega se desplaza 26 semanas hacia atrás para comparar estaciones equivalentes.
    """

    semana = int(row["semana"])
    pais = row["pais"]

    if pais == "Chile":
        return semana

    elif pais == "Noruega":
        semana_homologada = semana - 26

        if semana_homologada <= 0:
            semana_homologada += 52

        return semana_homologada

    return np.nan


def homologar_anio(row):
    """
    Ajusta el año homologado para Noruega.
    Si al restar 26 semanas queda una semana menor o igual a cero,
    el registro se mueve al año anterior en el eje homologado.
    """

    anio = int(row["año"])
    semana = int(row["semana"])
    pais = row["pais"]

    if pais == "Chile":
        return anio

    elif pais == "Noruega":
        if semana - 26 <= 0:
            return anio - 1
        else:
            return anio

    return np.nan


df["semana_homologada"] = df.apply(homologar_semana, axis=1)
df["año_homologado"] = df.apply(homologar_anio, axis=1)

df["semana_homologada"] = df["semana_homologada"].astype(int)
df["año_homologado"] = df["año_homologado"].astype(int)

df["estacion_homologada"] = df["semana_homologada"].apply(estacion_chile)

# ==============================
# FECHA HOMOLOGADA
# ==============================

df["fecha_homologada"] = pd.to_datetime(
    df["año_homologado"].astype(str)
    + "-W"
    + df["semana_homologada"].astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

df = df.dropna(subset=["fecha_homologada"])

# ==============================
# FILTRAR PERÍODO COMÚN HOMOLOGADO
# ==============================

df_grafico = df[
    (df["año_homologado"] >= 2014) &
    (df["año_homologado"] <= 2024)
].copy()

# ==============================
# BOXPLOT 1: POR PAÍS
# ==============================

paises = ["Chile", "Noruega"]

plt.figure(figsize=(8, 6))

datos_pais = [
    df_grafico[df_grafico["pais"] == pais]["parasitos_totales"].dropna()
    for pais in paises
]

plt.boxplot(
    datos_pais,
    tick_labels=paises,
    showmeans=True
)

plt.title("Distribución de parásitos totales por país")
plt.xlabel("País")
plt.ylabel("Parásitos totales")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

ruta_boxplot_pais = os.path.join(
    carpeta_boxplots,
    "boxplot_parasitos_totales_por_pais.png"
)

plt.savefig(ruta_boxplot_pais, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Guardado: {ruta_boxplot_pais}")

# ==============================
# BOXPLOT 2: POR ESTACIÓN HOMOLOGADA Y PAÍS
# ==============================

orden_estaciones = ["Verano", "Otoño", "Invierno", "Primavera"]

plt.figure(figsize=(12, 6))

posiciones = []
datos_boxplot = []
etiquetas = []
pos = 1

for estacion in orden_estaciones:
    for pais in paises:
        datos = df_grafico[
            (df_grafico["estacion_homologada"] == estacion) &
            (df_grafico["pais"] == pais)
        ]["parasitos_totales"].dropna()

        datos_boxplot.append(datos)
        posiciones.append(pos)
        etiquetas.append(f"{estacion}\n{pais}")
        pos += 1

    pos += 0.7

plt.boxplot(
    datos_boxplot,
    positions=posiciones,
    showmeans=True
)

plt.xticks(posiciones, etiquetas)
plt.title("Distribución de parásitos totales por estación homologada y país")
plt.xlabel("Estación homologada y país")
plt.ylabel("Parásitos totales")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

ruta_boxplot_estacion = os.path.join(
    carpeta_boxplots,
    "boxplot_parasitos_totales_por_estacion_homologada_y_pais.png"
)

plt.savefig(ruta_boxplot_estacion, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Guardado: {ruta_boxplot_estacion}")

# ==============================
# BOXPLOT 3: POR REGIÓN / MACROZONA
# ==============================

orden_zonas = [
    "Los Lagos",
    "Aysén",
    "Magallanes",
    "Sur/Oeste Noruega",
    "Centro Noruega",
    "Norte Noruega"
]

df_zonas = df_grafico.dropna(subset=["zona"])

datos_zonas = [
    df_zonas[df_zonas["zona"] == zona]["parasitos_totales"].dropna()
    for zona in orden_zonas
]

plt.figure(figsize=(12, 6))

plt.boxplot(
    datos_zonas,
    tick_labels=orden_zonas,
    showmeans=True
)

plt.title("Distribución de parásitos totales por región o macrozona")
plt.xlabel("Región / macrozona")
plt.ylabel("Parásitos totales")
plt.xticks(rotation=30, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()

ruta_boxplot_zona = os.path.join(
    carpeta_boxplots,
    "boxplot_parasitos_totales_por_zona.png"
)

plt.savefig(ruta_boxplot_zona, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Guardado: {ruta_boxplot_zona}")

# ==============================
# GRÁFICO TEMPORAL HOMOLOGADO SIMPLE
# ==============================

df_temporal = (
    df_grafico
    .groupby(["pais", "fecha_homologada"], as_index=False)["parasitos_totales"]
    .mean()
)

plt.figure(figsize=(14, 6))

for pais in paises:
    datos = df_temporal[df_temporal["pais"] == pais].sort_values("fecha_homologada")

    plt.plot(
        datos["fecha_homologada"],
        datos["parasitos_totales"],
        linewidth=1.8,
        label=pais
    )

plt.title("Evolución temporal homologada de parásitos totales por país")
plt.xlabel("Año homologado")
plt.ylabel("Parásitos totales promedio")
plt.legend(title="País")
plt.grid(True, alpha=0.3)
plt.tight_layout()

ruta_linea_simple = os.path.join(
    carpeta_lineas,
    "linea_temporal_homologada_parasitos_totales_por_pais.png"
)

plt.savefig(ruta_linea_simple, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Guardado: {ruta_linea_simple}")

# ==============================
# GRÁFICO TEMPORAL HOMOLOGADO CON FONDO ESTACIONAL
# VERSIÓN LIMPIA Y SUAVIZADA
# ==============================

colores_estacion = {
    "Verano": "#FFB3B3",      # rojo suave
    "Otoño": "#F6D7B0",       # naranja suave
    "Invierno": "#BFD7EA",    # azul suave
    "Primavera": "#CDECCF"    # verde suave
}

colores_pais = {
    "Chile": "#1f77b4",
    "Noruega": "#ff7f0e"
}

# Promedio semanal por país
df_temporal_hom = (
    df_grafico
    .groupby(["pais", "fecha_homologada"], as_index=False)["parasitos_totales"]
    .mean()
    .sort_values(["pais", "fecha_homologada"])
)

# Suavizado con promedio móvil de 4 semanas
df_temporal_hom["parasitos_suavizados"] = (
    df_temporal_hom
    .groupby("pais")["parasitos_totales"]
    .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
)

fecha_min = df_temporal_hom["fecha_homologada"].min()
fecha_max = df_temporal_hom["fecha_homologada"].max()

fig, ax = plt.subplots(figsize=(15, 6))

# ==============================
# FONDO ESTACIONAL
# ==============================

rangos_estaciones = [
    ("Verano", 1, 13),
    ("Otoño", 14, 26),
    ("Invierno", 27, 39),
    ("Primavera", 40, 52)
]

for anio in range(2014, 2025):
    for estacion, semana_ini, semana_fin in rangos_estaciones:

        inicio = pd.to_datetime(
            f"{anio}-W{semana_ini:02d}-1",
            format="%G-W%V-%u",
            errors="coerce"
        )

        fin = pd.to_datetime(
            f"{anio}-W{semana_fin:02d}-7",
            format="%G-W%V-%u",
            errors="coerce"
        )

        if pd.isna(inicio) or pd.isna(fin):
            continue

        if fin < fecha_min or inicio > fecha_max:
            continue

        ax.axvspan(
            max(inicio, fecha_min),
            min(fin, fecha_max),
            facecolor=colores_estacion[estacion],
            alpha=0.35,
            zorder=0
        )

# ==============================
# LÍNEAS SUAVIZADAS
# ==============================

for pais in paises:
    datos = df_temporal_hom[
        df_temporal_hom["pais"] == pais
    ].sort_values("fecha_homologada")

    ax.plot(
        datos["fecha_homologada"],
        datos["parasitos_suavizados"],
        linewidth=2.3,
        color=colores_pais[pais],
        label=pais,
        zorder=3
    )

# ==============================
# FORMATO DEL GRÁFICO
# ==============================

ax.set_title("Evolución temporal homologada de parásitos totales por país")
ax.set_xlabel("Año homologado")
ax.set_ylabel("Parásitos totales promedio")
ax.set_xlim(fecha_min, fecha_max)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.grid(True, axis="y", alpha=0.25)

leyenda_estaciones = [
    Patch(facecolor=colores_estacion["Verano"], alpha=0.50, label="Verano"),
    Patch(facecolor=colores_estacion["Otoño"], alpha=0.50, label="Otoño"),
    Patch(facecolor=colores_estacion["Invierno"], alpha=0.50, label="Invierno"),
    Patch(facecolor=colores_estacion["Primavera"], alpha=0.50, label="Primavera")
]

handles_lineas, labels_lineas = ax.get_legend_handles_labels()

ax.legend(
    handles=handles_lineas + leyenda_estaciones,
    loc="upper right",
    fontsize=9,
    frameon=True
)

plt.tight_layout()

ruta_linea_estacional = os.path.join(
    carpeta_lineas,
    "linea_temporal_homologada_parasitos_totales_con_fondo_estacional.png"
)

plt.savefig(ruta_linea_estacional, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Guardado: {ruta_linea_estacional}")

# ==============================
# GUARDAR DATASET HOMOLOGADO
# ==============================

ruta_dataset_homologado = os.path.join(
    "dataset",
    "base_chile_noruega_completa_homologada.csv"
)

df_grafico.to_csv(
    ruta_dataset_homologado,
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

print(f"Dataset homologado guardado en: {ruta_dataset_homologado}")

print("\nListo. Se generaron los gráficos con homologación temporal corregida y suavizada.")