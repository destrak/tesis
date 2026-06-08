import pandas as pd
import numpy as np
import os
import glob

# ==============================
# CARPETAS
# ==============================

carpeta_base = "dataset"
carpeta_chile = os.path.join(carpeta_base, "datasetchile")
carpeta_noruega = os.path.join(carpeta_base, "datasetnoruega")

# ==============================
# FUNCIONES DE ESTACIÓN
# ==============================

def estacion_chile(semana):
    if pd.isna(semana):
        return "NA"

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
        return "NA"


def estacion_noruega(semana):
    if pd.isna(semana):
        return "NA"

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
        return "NA"


# ==============================
# FUNCIÓN PARA PROCESAR ARCHIVOS
# ==============================

def procesar_archivos(carpeta, pais):
    archivos = glob.glob(os.path.join(carpeta, "*.csv"))

    # Evita reprocesar bases generales si ya existen
    archivos = [
        archivo for archivo in archivos
        if not os.path.basename(archivo).startswith("base_")
    ]

    bases = []

    for archivo in archivos:
        print(f"Procesando y reemplazando: {archivo}")

        df = pd.read_csv(archivo)

        # Normalizar nombres de columnas
        df.columns = df.columns.str.strip()

        # Reemplazar celdas vacías o espacios por NA
        df = df.replace(r"^\s*$", np.nan, regex=True)

        # Agregar o actualizar país
        df["pais"] = pais

        # Agregar o actualizar estación
        if pais == "Chile":
            df["estacion"] = df["semana"].apply(estacion_chile)
        elif pais == "Noruega":
            df["estacion"] = df["semana"].apply(estacion_noruega)

        # Columnas parasitarias
        columnas_parasitos = [
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles"
        ]

        # Verificar columnas existentes
        columnas_existentes = [col for col in columnas_parasitos if col in df.columns]

        if len(columnas_existentes) == 3:
            # Convertir a numérico
            for col in columnas_parasitos:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sumar parásitos
            df["parasitos_totales"] = df[columnas_parasitos].sum(axis=1, skipna=True)

            # Si las tres columnas son NA, deja parasitos_totales como NA
            df["parasitos_totales"] = df["parasitos_totales"].where(
                df[columnas_parasitos].notna().any(axis=1),
                np.nan
            )
        else:
            print("Ojo: no se encontraron todas las columnas parasitarias en:")
            print(archivo)
            print("Columnas disponibles:")
            print(df.columns.tolist())

        # Reemplazar NaN por texto NA para guardar
        df = df.fillna("NA")

        # Reemplazar archivo original
        df.to_csv(archivo, index=False, encoding="utf-8-sig", na_rep="NA")

        bases.append(df)

    return bases


# ==============================
# PROCESAR CHILE Y NORUEGA
# ==============================

bases_chile = procesar_archivos(carpeta_chile, "Chile")
bases_noruega = procesar_archivos(carpeta_noruega, "Noruega")

# ==============================
# CREAR BASES GENERALES
# ==============================

if len(bases_chile) == 0:
    raise ValueError("No se encontraron archivos CSV en la carpeta de Chile.")

if len(bases_noruega) == 0:
    raise ValueError("No se encontraron archivos CSV en la carpeta de Noruega.")

base_chile = pd.concat(bases_chile, ignore_index=True)
base_noruega = pd.concat(bases_noruega, ignore_index=True)
base_total = pd.concat([base_chile, base_noruega], ignore_index=True)

# ==============================
# GUARDAR BASES GENERALES
# ==============================

base_chile.to_csv(
    os.path.join(carpeta_chile, "base_chile_completa.csv"),
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

base_noruega.to_csv(
    os.path.join(carpeta_noruega, "base_noruega_completa.csv"),
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

base_total.to_csv(
    os.path.join(carpeta_base, "base_chile_noruega_completa.csv"),
    index=False,
    encoding="utf-8-sig",
    na_rep="NA"
)

# ==============================
# MOSTRAR RESULTADO
# ==============================

print("\nArchivos originales reemplazados correctamente.")
print("Celdas vacías guardadas como NA.")
print("Base Chile:", base_chile.shape)
print("Base Noruega:", base_noruega.shape)
print("Base total:", base_total.shape)

print("\nVista previa:")
print(base_total.head())