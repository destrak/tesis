# -*- coding: utf-8 -*-
"""
ANÁLISIS DE SENSIBILIDAD DE LOS UMBRALES AMBIENTALES

Compara los tres pares territoriales utilizados en la tesis:

1) Los Lagos   - Sur/Oeste
2) Aysén       - Centro
3) Magallanes  - Norte

Escenarios evaluados:
- Estricto:  ±2 °C y ±2 PSU
- Principal: ±3 °C y ±3 PSU
- Flexible:  ±4 °C y ±4 PSU

Se mantiene el desfase temporal de 26 semanas.
"""

import os
from glob import glob
import pandas as pd


# ============================================================
# 1. RUTAS
# ============================================================

carpeta_chile = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis"
    r"\tesis\dataset\datasetchile"
)

carpeta_noruega = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis"
    r"\tesis\dataset\datasetnoruega"
)

carpeta_salida = (
    r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis"
    r"\tesis\dataset\sensibilidad_umbrales"
)

os.makedirs(carpeta_salida, exist_ok=True)


# ============================================================
# 2. PARÁMETROS
# ============================================================

DESFASE_SEMANAS = 26

ESCENARIOS = {
    "Estricto_2_2": (2, 2),
    "Principal_3_3": (3, 3),
    "Flexible_4_4": (4, 4),
}


# ============================================================
# 3. PARES TERRITORIALES DE LA TESIS
# ============================================================

PARES_VALIDOS = {
    ("Los Lagos", "Sur/Oeste"),
    ("Aysén", "Centro"),
    ("Magallanes", "Norte"),
}


# ============================================================
# 4. FUNCIONES AUXILIARES
# ============================================================

def obtener_codigo_archivo(ruta):
    """Obtiene el nombre del archivo sin extensión."""
    return os.path.splitext(os.path.basename(ruta))[0]


def asignar_region(pais, archivo_origen):
    """Asigna región o macrozona a partir del código del centro."""

    archivo = str(archivo_origen)

    if pais == "Chile":

        if "102424" in archivo:
            return "Los Lagos"

        elif "110758" in archivo:
            return "Aysén"

        elif "120128" in archivo:
            return "Magallanes"

        return "Chile_sin_region"

    elif pais == "Noruega":

        if "33077" in archivo:
            return "Sur/Oeste"

        elif "32677" in archivo:
            return "Centro"

        elif "24175" in archivo:
            return "Norte"

        return "Noruega_sin_region"

    return "Sin_region"


def normalizar_columnas(df):
    """Normaliza los nombres de las columnas."""

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


def detectar_columna(df, posibles_nombres, nombre_final):
    """Detecta y renombra una columna."""

    for col in posibles_nombres:

        if col in df.columns:
            return df.rename(columns={col: nombre_final})

    raise ValueError(
        f"No se encontró la columna '{nombre_final}'.\n"
        f"Columnas disponibles:\n{list(df.columns)}"
    )


def estandarizar_columnas(df):
    """Estandariza las columnas utilizadas en el análisis."""

    df = normalizar_columnas(df)

    # Año
    df = detectar_columna(
        df,
        [
            "anio",
            "ano",
            "year",
        ],
        "anio",
    )

    # Semana
    df = detectar_columna(
        df,
        [
            "semana",
            "week",
        ],
        "semana",
    )

    # Temperatura
    df = detectar_columna(
        df,
        [
            "temperatura_media_c",
            "temperatura_c",
            "temperatura",
            "sea_temperature",
            "temperature",
            "temp",
            "temp_c",
        ],
        "temperatura_media_C",
    )

    # Salinidad
    df = detectar_columna(
        df,
        [
            "salinidad_media_psu",
            "salinidad_psu",
            "salinidad",
            "salinity",
            "sal_psu",
        ],
        "salinidad_media_PSU",
    )

    return df


def cargar_archivos(carpeta, pais):
    """Carga todos los CSV de una carpeta."""

    archivos = glob(os.path.join(carpeta, "*.csv"))

    if len(archivos) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en:\n{carpeta}"
        )

    lista = []

    for archivo in archivos:

        print(f"Cargando: {archivo}")

        try:
            df = pd.read_csv(archivo)

        except UnicodeDecodeError:
            df = pd.read_csv(
                archivo,
                encoding="latin1",
            )

        df = estandarizar_columnas(df)

        df["archivo_origen"] = obtener_codigo_archivo(archivo)

        df["pais"] = pais

        df["region"] = df["archivo_origen"].apply(
            lambda x: asignar_region(pais, x)
        )

        # Convertir a numérico
        df["anio"] = pd.to_numeric(
            df["anio"],
            errors="coerce",
        )

        df["semana"] = pd.to_numeric(
            df["semana"],
            errors="coerce",
        )

        df["temperatura_media_C"] = pd.to_numeric(
            df["temperatura_media_C"],
            errors="coerce",
        )

        df["salinidad_media_PSU"] = pd.to_numeric(
            df["salinidad_media_PSU"],
            errors="coerce",
        )

        df = df.dropna(
            subset=[
                "anio",
                "semana",
            ]
        )

        df["anio"] = df["anio"].astype(int)
        df["semana"] = df["semana"].astype(int)

        lista.append(df)

    return pd.concat(
        lista,
        ignore_index=True,
    )


# ============================================================
# 5. CARGAR CHILE Y NORUEGA
# ============================================================

print("\n==============================================")
print("CARGANDO CHILE")
print("==============================================")

chile = cargar_archivos(
    carpeta_chile,
    "Chile",
)


print("\n==============================================")
print("CARGANDO NORUEGA")
print("==============================================")

noruega = cargar_archivos(
    carpeta_noruega,
    "Noruega",
)


# ============================================================
# 6. HOMOLOGACIÓN TEMPORAL DE 26 SEMANAS
# ============================================================

chile_comp = chile.copy()

chile_comp["semana_noruega_equivalente"] = (
    chile_comp["semana"]
    + DESFASE_SEMANAS
)

chile_comp.loc[
    chile_comp["semana_noruega_equivalente"] > 52,
    "semana_noruega_equivalente"
] -= 52


chile_comp["anio_noruega_equivalente"] = (
    chile_comp["anio"]
)


chile_comp.loc[
    chile_comp["semana"] + DESFASE_SEMANAS > 52,
    "anio_noruega_equivalente"
] = (
    chile_comp["anio"] + 1
)


# ============================================================
# 7. RENOMBRAR COLUMNAS
# ============================================================

chile_comp = chile_comp.rename(
    columns={
        "anio": "anio_chile",
        "semana": "semana_chile",
        "archivo_origen": "centro_chile",
        "region": "region_chile",
        "temperatura_media_C": "temperatura_chile_C",
        "salinidad_media_PSU": "salinidad_chile_PSU",
    }
)


noruega_comp = noruega.rename(
    columns={
        "anio": "anio_noruega",
        "semana": "semana_noruega",
        "archivo_origen": "centro_noruega",
        "region": "region_noruega",
        "temperatura_media_C": "temperatura_noruega_C",
        "salinidad_media_PSU": "salinidad_noruega_PSU",
    }
)


# ============================================================
# 8. UNIR POR AÑO Y SEMANA HOMOLOGADA
# ============================================================

comparacion = pd.merge(
    chile_comp,
    noruega_comp,
    left_on=[
        "anio_noruega_equivalente",
        "semana_noruega_equivalente",
    ],
    right_on=[
        "anio_noruega",
        "semana_noruega",
    ],
    how="inner",
)


if comparacion.empty:

    raise ValueError(
        "No se encontraron semanas equivalentes "
        "después del desfase de 26 semanas."
    )


# ============================================================
# 9. DEJAR SOLO LOS 3 PARES DE LA TESIS
# ============================================================

comparacion["par_valido"] = comparacion.apply(
    lambda fila: (
        fila["region_chile"],
        fila["region_noruega"],
    ) in PARES_VALIDOS,
    axis=1,
)


comparacion = comparacion[
    comparacion["par_valido"]
].copy()


print("\nPares territoriales incluidos:")

print(
    comparacion[
        [
            "region_chile",
            "region_noruega",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "region_chile",
            "region_noruega",
        ]
    )
)


# ============================================================
# 10. DIFERENCIAS AMBIENTALES
# ============================================================

comparacion["dif_temperatura_C"] = (
    comparacion["temperatura_chile_C"]
    - comparacion["temperatura_noruega_C"]
).abs()


comparacion["dif_salinidad_PSU"] = (
    comparacion["salinidad_chile_PSU"]
    - comparacion["salinidad_noruega_PSU"]
).abs()


# Solo semanas con ambas variables disponibles
comparacion_valida = comparacion.dropna(
    subset=[
        "dif_temperatura_C",
        "dif_salinidad_PSU",
    ]
).copy()


# ============================================================
# 11. ANÁLISIS DE SENSIBILIDAD
# ============================================================

resultados = []


for nombre_escenario, (umbral_temp, umbral_sal) in ESCENARIOS.items():

    temp = comparacion_valida.copy()

    temp["comparable"] = (
        (temp["dif_temperatura_C"] <= umbral_temp)
        &
        (temp["dif_salinidad_PSU"] <= umbral_sal)
    )

    resumen = (
        temp
        .groupby(
            [
                "region_chile",
                "region_noruega",
            ]
        )
        .agg(
            total_semanas=(
                "comparable",
                "count",
            ),
            semanas_comparables=(
                "comparable",
                "sum",
            ),
            promedio_dif_temp_C=(
                "dif_temperatura_C",
                "mean",
            ),
            promedio_dif_sal_PSU=(
                "dif_salinidad_PSU",
                "mean",
            ),
        )
        .reset_index()
    )

    resumen["porcentaje_comparabilidad"] = (
        resumen["semanas_comparables"]
        / resumen["total_semanas"]
        * 100
    )

    resumen["escenario"] = nombre_escenario

    resumen["umbral_temperatura_C"] = umbral_temp

    resumen["umbral_salinidad_PSU"] = umbral_sal

    resultados.append(resumen)


# ============================================================
# 12. UNIR RESULTADOS
# ============================================================

sensibilidad = pd.concat(
    resultados,
    ignore_index=True,
)


sensibilidad["porcentaje_comparabilidad"] = (
    sensibilidad["porcentaje_comparabilidad"]
    .round(2)
)

sensibilidad["promedio_dif_temp_C"] = (
    sensibilidad["promedio_dif_temp_C"]
    .round(3)
)

sensibilidad["promedio_dif_sal_PSU"] = (
    sensibilidad["promedio_dif_sal_PSU"]
    .round(3)
)


# Nombre del par
sensibilidad["par_territorial"] = (
    sensibilidad["region_chile"]
    + " - "
    + sensibilidad["region_noruega"]
)


# ============================================================
# 13. ORDENAR
# ============================================================

orden_escenarios = {
    "Estricto_2_2": 1,
    "Principal_3_3": 2,
    "Flexible_4_4": 3,
}


orden_pares = {
    "Los Lagos - Sur/Oeste": 1,
    "Aysén - Centro": 2,
    "Magallanes - Norte": 3,
}


sensibilidad["orden_escenario"] = (
    sensibilidad["escenario"]
    .map(orden_escenarios)
)

sensibilidad["orden_par"] = (
    sensibilidad["par_territorial"]
    .map(orden_pares)
)


sensibilidad = sensibilidad.sort_values(
    [
        "orden_escenario",
        "orden_par",
    ]
)


# ============================================================
# 14. TABLA RESUMIDA PARA LA TESIS
# ============================================================

tabla_tesis = sensibilidad[
    [
        "escenario",
        "umbral_temperatura_C",
        "umbral_salinidad_PSU",
        "par_territorial",
        "total_semanas",
        "semanas_comparables",
        "porcentaje_comparabilidad",
    ]
].copy()


# ============================================================
# 15. TABLA PIVOT
# ============================================================

tabla_pivot = sensibilidad.pivot(
    index="par_territorial",
    columns="escenario",
    values="porcentaje_comparabilidad",
)


tabla_pivot = tabla_pivot.reindex(
    [
        "Los Lagos - Sur/Oeste",
        "Aysén - Centro",
        "Magallanes - Norte",
    ]
)


tabla_pivot = tabla_pivot[
    [
        "Estricto_2_2",
        "Principal_3_3",
        "Flexible_4_4",
    ]
]


tabla_pivot = tabla_pivot.rename(
    columns={
        "Estricto_2_2": "2°C / 2 PSU",
        "Principal_3_3": "3°C / 3 PSU",
        "Flexible_4_4": "4°C / 4 PSU",
    }
)


# ============================================================
# 16. GUARDAR CSV
# ============================================================

ruta_detalle = os.path.join(
    carpeta_salida,
    "Sensibilidad_Umbrales_Detalle.csv",
)


ruta_tesis = os.path.join(
    carpeta_salida,
    "Sensibilidad_Umbrales_Tesis.csv",
)


ruta_pivot = os.path.join(
    carpeta_salida,
    "Sensibilidad_Umbrales_Resumen.csv",
)


sensibilidad.to_csv(
    ruta_detalle,
    index=False,
    encoding="utf-8-sig",
)


tabla_tesis.to_csv(
    ruta_tesis,
    index=False,
    encoding="utf-8-sig",
)


tabla_pivot.to_csv(
    ruta_pivot,
    encoding="utf-8-sig",
)


# ============================================================
# 17. RESULTADOS EN CONSOLA
# ============================================================

print("\n")
print("============================================================")
print(" ANÁLISIS DE SENSIBILIDAD DE UMBRALES")
print("============================================================")

print("\nPorcentaje de semanas comparables:\n")

print(tabla_pivot)


print("\n")
print("============================================================")
print(" DETALLE")
print("============================================================")

print(
    tabla_tesis.to_string(
        index=False
    )
)


print("\n")
print("============================================================")
print(" ARCHIVOS GUARDADOS")
print("============================================================")

print(ruta_detalle)
print(ruta_tesis)
print(ruta_pivot)

print("\nProceso finalizado correctamente.")