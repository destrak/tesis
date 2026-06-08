import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# ==============================
# RUTA DEL DATASET
# ==============================

ruta = "dataset/base_chile_noruega_completa.csv"

# Carpeta principal de salida
carpeta_salida = "parasitos_imagenes"

# Subcarpetas por país
carpeta_chile = os.path.join(carpeta_salida, "Chile")
carpeta_noruega = os.path.join(carpeta_salida, "Noruega")

os.makedirs(carpeta_chile, exist_ok=True)
os.makedirs(carpeta_noruega, exist_ok=True)

# ==============================
# CARGAR DATASET
# ==============================

df = pd.read_csv(ruta, na_values=["NA", ""])

# ==============================
# ASEGURAR FORMATOS
# ==============================

df["año"] = pd.to_numeric(df["año"], errors="coerce")
df["semana"] = pd.to_numeric(df["semana"], errors="coerce")
df["localidad"] = df["localidad"].astype(str)

columnas_piojos = [
    "hembras_ovigeras",
    "adultos_moviles",
    "juveniles"
]

for col in columnas_piojos:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==============================
# ASIGNAR REGIÓN O MACROZONA
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

# Revisar si hay localidades sin zona asignada
localidades_sin_zona = df[df["zona"].isna()]["localidad"].unique()

if len(localidades_sin_zona) > 0:
    print("Localidades sin zona asignada:")
    print(localidades_sin_zona)

# ==============================
# CREAR FECHA APROXIMADA
# ==============================

df["fecha"] = pd.to_datetime(
    df["año"].astype("Int64").astype(str)
    + "-W"
    + df["semana"].astype("Int64").astype(str).str.zfill(2)
    + "-1",
    format="%G-W%V-%u",
    errors="coerce"
)

# Eliminar filas sin fecha o sin zona
df = df.dropna(subset=["fecha", "zona"])

df = df.sort_values(["pais", "zona", "fecha"])

# ==============================
# PROMEDIAR POR PAÍS, ZONA Y SEMANA
# ==============================

df_zona = (
    df.groupby(["pais", "zona", "fecha"], as_index=False)[columnas_piojos]
    .mean()
)

# ==============================
# FUNCIÓN PARA LIMPIAR NOMBRES
# ==============================

def limpiar_nombre_archivo(nombre):
    nombre = nombre.replace("á", "a").replace("é", "e").replace("í", "i")
    nombre = nombre.replace("ó", "o").replace("ú", "u").replace("ñ", "n")
    nombre = nombre.replace("Á", "A").replace("É", "E").replace("Í", "I")
    nombre = nombre.replace("Ó", "O").replace("Ú", "U").replace("Ñ", "N")
    nombre = nombre.replace("/", "_").replace(" ", "_")
    nombre = re.sub(r"[^A-Za-z0-9_]", "", nombre)
    return nombre

# ==============================
# FUNCIÓN PARA GRAFICAR
# ==============================

def graficar_piojos_por_zona(datos, pais, zona):
    datos = datos.sort_values("fecha")

    plt.figure(figsize=(14, 6))

    plt.plot(
        datos["fecha"],
        datos["hembras_ovigeras"],
        label="Hembras ovígeras",
        linewidth=1.5
    )

    plt.plot(
        datos["fecha"],
        datos["adultos_moviles"],
        label="Adultos móviles",
        linewidth=1.5
    )

    plt.plot(
        datos["fecha"],
        datos["juveniles"],
        label="Juveniles",
        linewidth=1.5
    )

    plt.title(f"Evolución semanal de piojo de mar - {zona}")
    plt.xlabel("Año")
    plt.ylabel("Abundancia parasitaria")
    plt.legend()
    plt.grid(True, alpha=0.3)
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

# ==============================
# GENERAR GRÁFICOS
# ==============================

for (pais, zona), datos in df_zona.groupby(["pais", "zona"]):
    graficar_piojos_por_zona(datos, pais, zona)

print("\nGráficos generados correctamente en la carpeta 'parasitos_imagenes'.")