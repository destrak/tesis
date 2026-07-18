import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from matplotlib.patches import Patch
import re

# ============================================================
# RUTAS
# ============================================================

# Carpeta donde está guardado este script .py
carpeta_script = os.path.dirname(os.path.abspath(__file__))

# Dataset ubicado dentro de la misma carpeta del script
carpeta_base = os.path.join(carpeta_script, "dataset")
carpeta_chile = os.path.join(carpeta_base, "datasetchile")
carpeta_noruega = os.path.join(carpeta_base, "datasetnoruega")

# Carpetas de salida
carpeta_boxplots = os.path.join(carpeta_script, "boxplots_parasitos")
carpeta_lineas = os.path.join(carpeta_script, "lineas_parasitos_homologadas")

os.makedirs(carpeta_boxplots, exist_ok=True)
os.makedirs(carpeta_lineas, exist_ok=True)

print("Carpeta script:", carpeta_script)
print("Carpeta base:", carpeta_base)
print("Carpeta Chile:", carpeta_chile)
print("Carpeta Noruega:", carpeta_noruega)

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_columnas(df):
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("á", "a", regex=False)
        .str.replace("é", "e", regex=False)
        .str.replace("í", "i", regex=False)
        .str.replace("ó", "o", regex=False)
        .str.replace("ú", "u", regex=False)
        .str.replace("ñ", "n", regex=False)
        .str.replace("°", "", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(" ", "_", regex=False)
    )

    renombres = {
        "año": "anio",
        "ano": "anio",
        "year": "anio",

        "week": "semana",
        "sem": "semana",

        "localidad": "localidad",
        "codigo": "localidad",
        "codigo_localidad": "localidad",
        "centro": "localidad",

        "hembras_ovigeras": "hembras_ovigeras",
        "adult_female_lice": "hembras_ovigeras",
        "adult_female": "hembras_ovigeras",

        "adultos_moviles": "adultos_moviles",
        "lice_in_moving_stages": "adultos_moviles",
        "moving_lice": "adultos_moviles",

        "juveniles": "juveniles",
        "stuck_lice": "juveniles",

        "parasitos_totales": "parasitos_totales",
        "total_piojos": "parasitos_totales",
        "parasites_total": "parasitos_totales",
        "total_lice": "parasitos_totales",
        "carga_total_piojos": "parasitos_totales",

        "temperatura_c": "temperatura",
        "temperatura": "temperatura",
        "temperatura_c_": "temperatura",
        "temperatura_celsius": "temperatura",
        "sea_temperature": "temperatura",
        "temperature": "temperatura",

        "salinidad_psu": "salinidad",
        "salinidad": "salinidad",
        "salinity": "salinidad",

        "pais": "pais",
        "country": "pais",

        "estacion": "estacion",
        "season": "estacion"
    }

    df = df.rename(columns={k: v for k, v in renombres.items() if k in df.columns})

    return df


def inferir_localidad_desde_archivo(nombre_archivo):
    """
    Extrae códigos conocidos desde el nombre del archivo.
    """

    codigos = ["102424", "110758", "120128", "24175", "32677", "33077"]

    for codigo in codigos:
        if codigo in nombre_archivo:
            return codigo

    # Si no encuentra los códigos conocidos, intenta encontrar un número de 5 o 6 dígitos
    match = re.search(r"\d{5,6}", nombre_archivo)
    if match:
        return match.group(0)

    return np.nan


def cargar_archivos(carpeta, pais):
    if not os.path.exists(carpeta):
        raise FileNotFoundError(f"No existe la carpeta: {carpeta}")

    archivos = [
        os.path.join(carpeta, archivo)
        for archivo in os.listdir(carpeta)
        if archivo.lower().endswith(".csv")
    ]

    # Evita cargar bases generales si ya existen
    archivos = [
        archivo for archivo in archivos
        if not os.path.basename(archivo).lower().startswith("base_")
    ]

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {carpeta}")

    bases = []

    for archivo in archivos:
        print(f"Cargando {pais}: {archivo}")

        try:
            df = pd.read_csv(archivo, na_values=["NA", ""], encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(archivo, na_values=["NA", ""], encoding="latin1")

        df = normalizar_columnas(df)

        nombre_archivo = os.path.basename(archivo)

        # Asegurar columna país
        df["pais"] = pais

        # Asegurar columna localidad
        if "localidad" not in df.columns:
            df["localidad"] = inferir_localidad_desde_archivo(nombre_archivo)

        df["localidad"] = (
            df["localidad"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        # Convertir columnas numéricas
        columnas_numericas = [
            "anio",
            "semana",
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles",
            "temperatura",
            "salinidad",
            "parasitos_totales"
        ]

        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Verificar columnas mínimas
        columnas_minimas = [
            "anio",
            "semana",
            "localidad",
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles",
            "temperatura",
            "salinidad"
        ]

        faltantes = [col for col in columnas_minimas if col not in df.columns]

        if faltantes:
            print("Archivo omitido por columnas faltantes:")
            print(archivo)
            print("Faltan:", faltantes)
            print("Columnas disponibles:", df.columns.tolist())
            continue

        # Calcular parásitos totales si no existe
        columnas_parasitos = [
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles"
        ]

        if "parasitos_totales" not in df.columns:
            df["parasitos_totales"] = df[columnas_parasitos].sum(axis=1, skipna=True)

            # Si las tres columnas son NA, deja parasitos_totales como NA
            df["parasitos_totales"] = df["parasitos_totales"].where(
                df[columnas_parasitos].notna().any(axis=1),
                np.nan
            )

        bases.append(df)

    if len(bases) == 0:
        raise ValueError(f"No se pudo cargar ningún archivo válido desde: {carpeta}")

    return pd.concat(bases, ignore_index=True)


def guardar_csv(df, ruta):
    """
    Guarda el dataframe usando 'año' en vez de 'anio' para mantener formato de tesis.
    """

    df_guardar = df.copy()

    if "anio" in df_guardar.columns:
        df_guardar = df_guardar.rename(columns={"anio": "año"})

    df_guardar.to_csv(
        ruta,
        index=False,
        encoding="utf-8-sig",
        na_rep="NA"
    )


# ============================================================
# FUNCIONES DE ESTACIÓN
# ============================================================

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


def estacion_noruega(semana):
    if pd.isna(semana):
        return np.nan

    semana = int(semana)

    if 1 <= semana <= 13:
        return "Invierno"
    elif 14 <= semana <= 26:
        return "Primavera"
    elif 27 <= semana <= 39:
        return "Verano"
    elif 40 <= semana <= 52:
        return "Otoño"
    else:
        return np.nan


# ============================================================
# CARGAR DATASETS ORIGINALES
# ============================================================

base_chile = cargar_archivos(carpeta_chile, "Chile")
base_noruega = cargar_archivos(carpeta_noruega, "Noruega")

df = pd.concat([base_chile, base_noruega], ignore_index=True)

# ============================================================
# ASEGURAR FORMATOS
# ============================================================

df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
df["parasitos_totales"] = pd.to_numeric(df["parasitos_totales"], errors="coerce")

df["pais"] = df["pais"].astype(str).str.strip()
df["localidad"] = (
    df["localidad"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

df = df.dropna(subset=["anio", "semana", "pais", "parasitos_totales"])

df["anio"] = df["anio"].astype(int)
df["semana"] = df["semana"].astype(int)

# Mantener solo semanas válidas para homologación de 52 semanas
df = df[(df["semana"] >= 1) & (df["semana"] <= 52)].copy()

# ============================================================
# ASIGNAR REGIÓN / MACROZONA
# ============================================================

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

# ============================================================
# ASIGNAR ESTACIÓN ORIGINAL
# ============================================================

df["estacion"] = np.where(
    df["pais"] == "Chile",
    df["semana"].apply(estacion_chile),
    df["semana"].apply(estacion_noruega)
)

# ============================================================
# HOMOLOGACIÓN TEMPORAL
# ============================================================

def homologar_semana(row):
    """
    Criterio metodológico:

    Noruega mantiene su semana original.
    Chile se desplaza 26 semanas hacia adelante.

    S_N = S_C + 26

    Si S_C + 26 > 52, se resta 52.
    """

    semana = int(row["semana"])
    pais = row["pais"]

    if pais == "Noruega":
        return semana

    elif pais == "Chile":
        semana_homologada = semana + 26

        if semana_homologada > 52:
            semana_homologada -= 52

        return semana_homologada

    return np.nan


def homologar_anio(row):
    """
    Ajusta el año homologado para Chile.

    Si S_C + 26 > 52, el registro chileno pasa al año siguiente
    en el eje homologado.
    """

    anio = int(row["anio"])
    semana = int(row["semana"])
    pais = row["pais"]

    if pais == "Noruega":
        return anio

    elif pais == "Chile":
        if semana + 26 > 52:
            return anio + 1
        else:
            return anio

    return np.nan


df["semana_homologada"] = df.apply(homologar_semana, axis=1)
df["anio_homologado"] = df.apply(homologar_anio, axis=1)

df = df.dropna(subset=["semana_homologada", "anio_homologado"])

df["semana_homologada"] = df["semana_homologada"].astype(int)
df["anio_homologado"] = df["anio_homologado"].astype(int)

# Como el eje homologado queda en referencia noruega,
# la estación homologada se calcula con calendario noruego.
df["estacion_homologada"] = df["semana_homologada"].apply(estacion_noruega)

# ============================================================
# FECHA HOMOLOGADA
# ============================================================

df["fecha_homologada"] = pd.to_datetime(
    df["anio_homologado"].astype(str)
    + "-W"
    + df["semana_homologada"].astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

df = df.dropna(subset=["fecha_homologada"])

# ============================================================
# FILTRAR PERÍODO COMÚN HOMOLOGADO
# ============================================================

df_grafico = df[
    (df["anio_homologado"] >= 2014) &
    (df["anio_homologado"] <= 2024)
].copy()

# ============================================================
# GUARDAR BASES GENERADAS
# ============================================================

guardar_csv(
    base_chile,
    os.path.join(carpeta_chile, "base_chile_completa.csv")
)

guardar_csv(
    base_noruega,
    os.path.join(carpeta_noruega, "base_noruega_completa.csv")
)

guardar_csv(
    df,
    os.path.join(carpeta_base, "base_chile_noruega_completa.csv")
)

guardar_csv(
    df_grafico,
    os.path.join(carpeta_base, "base_chile_noruega_completa_homologada.csv")
)

print("\nBases guardadas correctamente.")
print("Base Chile:", base_chile.shape)
print("Base Noruega:", base_noruega.shape)
print("Base conjunta:", df.shape)
print("Base homologada:", df_grafico.shape)

# ============================================================
# BOXPLOT 1: POR PAÍS
# ============================================================

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

# ============================================================
# BOXPLOT 2: POR ESTACIÓN HOMOLOGADA Y PAÍS
# ============================================================

orden_estaciones = ["Invierno", "Primavera", "Verano", "Otoño"]

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

# ============================================================
# BOXPLOT 3: POR REGIÓN / MACROZONA
# ============================================================

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

# ============================================================
# GRÁFICO TEMPORAL HOMOLOGADO SIMPLE
# ============================================================

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

# ============================================================
# GRÁFICO TEMPORAL HOMOLOGADO CON FONDO ESTACIONAL
# ============================================================

colores_estacion = {
    "Invierno": "#BFD7EA",
    "Primavera": "#CDECCF",
    "Verano": "#FFB3B3",
    "Otoño": "#F6D7B0"
}

colores_pais = {
    "Chile": "#1f77b4",
    "Noruega": "#ff7f0e"
}

df_temporal_hom = (
    df_grafico
    .groupby(["pais", "fecha_homologada"], as_index=False)["parasitos_totales"]
    .mean()
    .sort_values(["pais", "fecha_homologada"])
)

df_temporal_hom["parasitos_suavizados"] = (
    df_temporal_hom
    .groupby("pais")["parasitos_totales"]
    .transform(lambda x: x.rolling(window=4, min_periods=1).mean())
)

fecha_min = df_temporal_hom["fecha_homologada"].min()
fecha_max = df_temporal_hom["fecha_homologada"].max()

fig, ax = plt.subplots(figsize=(15, 6))

rangos_estaciones = [
    ("Invierno", 1, 13),
    ("Primavera", 14, 26),
    ("Verano", 27, 39),
    ("Otoño", 40, 52)
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

ax.set_title("Evolución temporal homologada de parásitos totales por país")
ax.set_xlabel("Año homologado")
ax.set_ylabel("Parásitos totales promedio")
ax.set_xlim(fecha_min, fecha_max)

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

ax.grid(True, axis="y", alpha=0.25)

leyenda_estaciones = [
    Patch(facecolor=colores_estacion["Invierno"], alpha=0.50, label="Invierno"),
    Patch(facecolor=colores_estacion["Primavera"], alpha=0.50, label="Primavera"),
    Patch(facecolor=colores_estacion["Verano"], alpha=0.50, label="Verano"),
    Patch(facecolor=colores_estacion["Otoño"], alpha=0.50, label="Otoño")
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

print("\nListo. Se usaron los CSV originales de datasetchile y datasetnoruega.")
print("Homologación aplicada según el criterio: S_N = S_C + 26.")