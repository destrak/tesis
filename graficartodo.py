import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

# ============================================================
# GRAFICAR CHILE, NORUEGA Y TODOS JUNTOS
# Temperatura y salinidad
# ============================================================

# Carpetas de entrada
carpeta_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Chile"
carpeta_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Noruega"

# Carpeta de salida
carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Graficos_Comparacion_General"
os.makedirs(carpeta_salida, exist_ok=True)

# ============================================================
# 1. Asignar nombres de regiones según código de centro
# ============================================================

def asignar_region(pais, nombre_archivo):
    nombre = str(nombre_archivo)

    if pais == "Chile":
        if "102424" in nombre:
            return "Los Lagos"
        elif "110758" in nombre:
            return "Aysén"
        elif "120128" in nombre:
            return "Magallanes"
        else:
            return "Chile sin región"

    if pais == "Noruega":
        if "33077" in nombre:
            return "Sur/Oeste"
        elif "32677" in nombre:
            return "Centro"
        elif "24175" in nombre:
            return "Norte"
        else:
            return "Noruega sin región"

    return "Sin región"


def normalizar_columnas(df):
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
        .str.replace(" ", "_", regex=False)
    )
    return df


def detectar_columna(df, posibles, nombre_final):
    for col in posibles:
        if col in df.columns:
            df = df.rename(columns={col: nombre_final})
            return df

    raise ValueError(
        f"No se encontró la columna {nombre_final}. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )


def estandarizar_columnas(df):
    df = normalizar_columnas(df)

    df = detectar_columna(
        df,
        ["anio", "ano", "year"],
        "anio"
    )

    df = detectar_columna(
        df,
        ["semana", "week"],
        "semana"
    )

    df = detectar_columna(
        df,
        [
            "temperatura_media_c",
            "temperatura_c",
            "temperatura",
            "sea_temperature",
            "temperature",
            "temp",
            "temp_c"
        ],
        "temperatura_media_C"
    )

    df = detectar_columna(
        df,
        [
            "salinidad_media_psu",
            "salinidad_psu",
            "salinidad",
            "salinity",
            "sal_psu"
        ],
        "salinidad_media_PSU"
    )

    return df


def cargar_pais(carpeta, pais):
    archivos = glob(os.path.join(carpeta, "*.csv"))

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron CSV en {carpeta}")

    lista = []

    for archivo in archivos:
        print(f"Cargando {pais}: {archivo}")

        try:
            df = pd.read_csv(archivo)
        except UnicodeDecodeError:
            df = pd.read_csv(archivo, encoding="latin1")

        df = estandarizar_columnas(df)

        nombre_archivo = os.path.splitext(os.path.basename(archivo))[0]

        df["pais"] = pais
        df["centro"] = nombre_archivo
        df["region"] = asignar_region(pais, nombre_archivo)

        df["anio"] = pd.to_numeric(df["anio"], errors="coerce")
        df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
        df["temperatura_media_C"] = pd.to_numeric(df["temperatura_media_C"], errors="coerce")
        df["salinidad_media_PSU"] = pd.to_numeric(df["salinidad_media_PSU"], errors="coerce")

        df = df.dropna(subset=["anio", "semana"])
        df["anio"] = df["anio"].astype(int)
        df["semana"] = df["semana"].astype(int)

        lista.append(df)

    return pd.concat(lista, ignore_index=True)


# ============================================================
# 2. Cargar Chile y Noruega
# ============================================================

chile = cargar_pais(carpeta_chile, "Chile")
noruega = cargar_pais(carpeta_noruega, "Noruega")

# Unir todo
todo = pd.concat([chile, noruega], ignore_index=True)

# Crear eje temporal ordenado
todo["fecha_semana"] = todo["anio"].astype(str) + "-S" + todo["semana"].astype(str).str.zfill(2)

# Para ordenar por año y semana
todo = todo.sort_values(["pais", "region", "anio", "semana"]).reset_index(drop=True)

# Guardar base unificada usada para gráficos
ruta_base = os.path.join(carpeta_salida, "Base_Chile_Noruega_Graficos.csv")
todo.to_csv(ruta_base, index=False, encoding="utf-8-sig")

print(f"\nBase usada para gráficos guardada en: {ruta_base}")

# ============================================================
# 3. Función para graficar varias series
# ============================================================

def graficar_series(df, variable, ylabel, titulo, nombre_salida):
    plt.figure(figsize=(16, 7))

    for (pais, region), grupo in df.groupby(["pais", "region"]):
        grupo = grupo.sort_values(["anio", "semana"]).copy()
        grupo["indice"] = range(1, len(grupo) + 1)

        etiqueta = f"{pais} - {region}"

        plt.plot(
            grupo["indice"],
            grupo[variable],
            linewidth=1.8,
            label=etiqueta
        )

    plt.title(titulo, fontsize=16)
    plt.xlabel("Semana de la serie temporal")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(left=1)

    if "temperatura" in variable:
        plt.ylim(bottom=0)
        plt.yticks(range(0, 31, 5))

    if "salinidad" in variable:
        plt.ylim(0, 40)
        plt.yticks(range(0, 41, 5))

    ruta = os.path.join(carpeta_salida, nombre_salida)
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Gráfico guardado en: {ruta}")


# ============================================================
# 4. Gráficos solo Chile: 3 regiones en 1
# ============================================================

df_chile = todo[todo["pais"] == "Chile"].copy()

graficar_series(
    df=df_chile,
    variable="temperatura_media_C",
    ylabel="Temperatura media (°C)",
    titulo="Chile: temperatura media semanal por región",
    nombre_salida="01_Chile_3_regiones_temperatura.png"
)

graficar_series(
    df=df_chile,
    variable="salinidad_media_PSU",
    ylabel="Salinidad media (PSU)",
    titulo="Chile: salinidad media semanal por región",
    nombre_salida="02_Chile_3_regiones_salinidad.png"
)

# ============================================================
# 5. Gráficos solo Noruega: 3 macrozonas en 1
# ============================================================

df_noruega = todo[todo["pais"] == "Noruega"].copy()

graficar_series(
    df=df_noruega,
    variable="temperatura_media_C",
    ylabel="Temperatura media (°C)",
    titulo="Noruega: temperatura media semanal por macrozona",
    nombre_salida="03_Noruega_3_macrozones_temperatura.png"
)

graficar_series(
    df=df_noruega,
    variable="salinidad_media_PSU",
    ylabel="Salinidad media (PSU)",
    titulo="Noruega: salinidad media semanal por macrozona",
    nombre_salida="04_Noruega_3_macrozones_salinidad.png"
)

# ============================================================
# 6. Gráficos Chile + Noruega: 6 series en 1
# ============================================================

graficar_series(
    df=todo,
    variable="temperatura_media_C",
    ylabel="Temperatura media (°C)",
    titulo="Chile y Noruega: temperatura media semanal por región/macrozona",
    nombre_salida="05_Chile_Noruega_6_series_temperatura.png"
)

graficar_series(
    df=todo,
    variable="salinidad_media_PSU",
    ylabel="Salinidad media (PSU)",
    titulo="Chile y Noruega: salinidad media semanal por región/macrozona",
    nombre_salida="06_Chile_Noruega_6_series_salinidad.png"
)

print("\nProceso finalizado correctamente.")