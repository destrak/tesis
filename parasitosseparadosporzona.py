import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
import numpy as np

# ============================================================
# RUTAS AUTOMÁTICAS
# ============================================================

# Carpeta donde está guardado este script .py
carpeta_script = os.path.dirname(os.path.abspath(__file__))

# Carpetas de entrada
carpeta_base = os.path.join(carpeta_script, "dataset")
carpeta_chile_datos = os.path.join(carpeta_base, "datasetchile")
carpeta_noruega_datos = os.path.join(carpeta_base, "datasetnoruega")

# Carpeta principal de salida
carpeta_salida = os.path.join(carpeta_script, "parasitos_imagenes_homologadas")

# Subcarpetas por país
carpeta_chile = os.path.join(carpeta_salida, "Chile")
carpeta_noruega = os.path.join(carpeta_salida, "Noruega")

os.makedirs(carpeta_chile, exist_ok=True)
os.makedirs(carpeta_noruega, exist_ok=True)

print("Carpeta script:", carpeta_script)
print("Carpeta base:", carpeta_base)
print("Carpeta Chile:", carpeta_chile_datos)
print("Carpeta Noruega:", carpeta_noruega_datos)

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

        "pais": "pais",
        "country": "pais",
    }

    df = df.rename(columns={k: v for k, v in renombres.items() if k in df.columns})

    return df


def inferir_localidad_desde_archivo(nombre_archivo):
    codigos = ["102424", "110758", "120128", "24175", "32677", "33077"]

    for codigo in codigos:
        if codigo in nombre_archivo:
            return codigo

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

        df["pais"] = pais

        if "localidad" not in df.columns:
            df["localidad"] = inferir_localidad_desde_archivo(nombre_archivo)

        df["localidad"] = (
            df["localidad"]
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )

        columnas_necesarias = [
            "anio",
            "semana",
            "localidad",
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles"
        ]

        faltantes = [col for col in columnas_necesarias if col not in df.columns]

        if faltantes:
            print("Archivo omitido por columnas faltantes:")
            print(archivo)
            print("Faltan:", faltantes)
            print("Columnas disponibles:", df.columns.tolist())
            continue

        columnas_numericas = [
            "anio",
            "semana",
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles"
        ]

        for col in columnas_numericas:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        bases.append(df)

    if len(bases) == 0:
        raise ValueError(f"No se pudo cargar ningún archivo válido desde: {carpeta}")

    return pd.concat(bases, ignore_index=True)


def limpiar_nombre_archivo(nombre):
    nombre = nombre.replace("á", "a").replace("é", "e").replace("í", "i")
    nombre = nombre.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    nombre = nombre.replace("Á", "A").replace("É", "E").replace("Í", "I")
    nombre = nombre.replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
    nombre = nombre.replace("/", "_").replace(" ", "_")
    nombre = re.sub(r"[^A-Za-z0-9_]", "", nombre)
    return nombre

# ============================================================
# CARGAR DATASETS ORIGINALES
# ============================================================

base_chile = cargar_archivos(carpeta_chile_datos, "Chile")
base_noruega = cargar_archivos(carpeta_noruega_datos, "Noruega")

df = pd.concat([base_chile, base_noruega], ignore_index=True)

# ============================================================
# ASEGURAR FORMATOS
# ============================================================

df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
df["semana"] = pd.to_numeric(df["semana"], errors="coerce")

df["pais"] = df["pais"].astype(str).str.strip()
df["localidad"] = (
    df["localidad"]
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)

columnas_piojos = [
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles"
]

for col in columnas_piojos:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["anio", "semana", "pais", "localidad"])

df["anio"] = df["anio"].astype(int)
df["semana"] = df["semana"].astype(int)

# Mantener semanas válidas
df = df[(df["semana"] >= 1) & (df["semana"] <= 52)].copy()

# ============================================================
# ASIGNAR REGIÓN O MACROZONA
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

localidades_sin_zona = df[df["zona"].isna()]["localidad"].unique()

if len(localidades_sin_zona) > 0:
    print("Localidades sin zona asignada:")
    print(localidades_sin_zona)

# ============================================================
# HOMOLOGACIÓN TEMPORAL
# ============================================================

def homologar_semana(row):
    """
    Criterio metodológico:

    Noruega mantiene su semana original.
    Chile se desplaza 26 semanas hacia adelante.

    S_N = S_C + 26
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
    Si S_C + 26 > 52, el registro chileno pasa al año siguiente.
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

# ============================================================
# CREAR FECHA HOMOLOGADA
# ============================================================

df["fecha_homologada"] = pd.to_datetime(
    df["anio_homologado"].astype(str)
    + "-W"
    + df["semana_homologada"].astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

df = df.dropna(subset=["fecha_homologada", "zona"])

df = df.sort_values(["pais", "zona", "fecha_homologada"])

# Período común homologado
df = df[
    (df["anio_homologado"] >= 2014) &
    (df["anio_homologado"] <= 2024)
].copy()

# ============================================================
# PROMEDIAR POR PAÍS, ZONA Y SEMANA HOMOLOGADA
# ============================================================

df_zona = (
    df.groupby(["pais", "zona", "fecha_homologada"], as_index=False)[columnas_piojos]
    .mean()
)

# ============================================================
# FUNCIÓN PARA GRAFICAR
# ============================================================

def graficar_piojos_por_zona(datos, pais, zona):
    datos = datos.sort_values("fecha_homologada")

    plt.figure(figsize=(14, 6))

    plt.plot(
        datos["fecha_homologada"],
        datos["hembras_ovigeras"],
        label="Hembras ovígeras",
        linewidth=1.5
    )

    plt.plot(
        datos["fecha_homologada"],
        datos["adultos_moviles"],
        label="Adultos móviles",
        linewidth=1.5
    )

    plt.plot(
        datos["fecha_homologada"],
        datos["juveniles"],
        label="Juveniles",
        linewidth=1.5
    )

    plt.title(f"Evolución semanal homologada de piojo de mar - {zona}")
    plt.xlabel("Año homologado")
    plt.ylabel("Abundancia parasitaria")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()

    if pais == "Chile":
        carpeta_destino = carpeta_chile
    elif pais == "Noruega":
        carpeta_destino = carpeta_noruega
    else:
        carpeta_destino = carpeta_salida

    nombre_archivo = limpiar_nombre_archivo(zona) + ".png"
    ruta_salida = os.path.join(carpeta_destino, nombre_archivo)

    plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Guardado: {ruta_salida}")

# ============================================================
# GUARDAR BASE HOMOLOGADA USADA PARA LOS GRÁFICOS
# ============================================================

ruta_base_homologada = os.path.join(
    carpeta_base,
    "base_chile_noruega_series_parasitarias_homologada.csv"
)

df_guardar = df.copy()
df_guardar = df_guardar.rename(columns={"anio": "año", "anio_homologado": "año_homologado"})

df_guardar.to_csv(
    ruta_base_homologada,
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

print(f"Base homologada guardada en: {ruta_base_homologada}")

# ============================================================
# GENERAR GRÁFICOS
# ============================================================

for (pais, zona), datos in df_zona.groupby(["pais", "zona"]):
    graficar_piojos_por_zona(datos, pais, zona)

print("\nGráficos generados correctamente en la carpeta 'parasitos_imagenes_homologadas'.")
print("Homologación aplicada según el criterio: S_N = S_C + 26.")