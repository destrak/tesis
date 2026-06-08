import xarray as xr
import pandas as pd
import glob
import os
import numpy as np

# ============================================================
# CONSOLIDAR ARCHIVOS NETCDF DE COPERNICUS
# Centro: ACS 54 B
# Código establecimiento: 120128
# Periodo esperado: 2014-2024
# ============================================================

carpeta_origen = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Historico_120128"

archivo_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\120128_Consolidado_2014_2024.csv"

# Buscar archivos .nc
archivos_nc = glob.glob(os.path.join(carpeta_origen, "*.nc"))
archivos_nc = sorted(archivos_nc)

print("====================================================")
print(" BUSQUEDA DE ARCHIVOS NETCDF")
print("====================================================")
print("Carpeta revisada:")
print(carpeta_origen)
print(f"\nSe encontraron {len(archivos_nc)} archivos NetCDF.")

if archivos_nc:
    print("\nArchivos encontrados:")
    for archivo in archivos_nc:
        print(" -", os.path.basename(archivo))
else:
    print("\nNo se encontraron archivos .nc.")
    print("Revisa que la carpeta sea correcta.")
    exit()

print("====================================================")

lista_dataframes = []

# ============================================================
# PROCESAR ARCHIVOS
# ============================================================

for archivo in archivos_nc:
    nombre_base = os.path.basename(archivo)
    print(f"\nProcesando: {nombre_base}")

    try:
        ds = xr.open_dataset(archivo)

        print("Dimensiones:")
        print(ds.dims)

        print("Variables disponibles:")
        print(list(ds.data_vars))

        df = ds.to_dataframe().reset_index()

        print("Columnas del DataFrame:")
        print(list(df.columns))

        print(f"Filas originales: {len(df)}")

        # Verificar que existan las variables necesarias
        if "thetao" not in df.columns or "so" not in df.columns:
            print(f"[ERROR] {nombre_base} no contiene thetao y/o so.")
            ds.close()
            continue

        # Ver cuántos datos válidos hay antes de filtrar
        n_temp_validos = df["thetao"].notna().sum()
        n_sal_validos = df["so"].notna().sum()

        print(f"Datos válidos de temperatura: {n_temp_validos}")
        print(f"Datos válidos de salinidad  : {n_sal_validos}")

        # Eliminar filas donde temperatura y salinidad sean ambas NaN
        df_agua = df.dropna(subset=["thetao", "so"], how="all")

        print(f"Filas con datos de agua: {len(df_agua)}")

        if df_agua.empty:
            print(f"[ADVERTENCIA] {nombre_base} no tiene datos válidos de agua.")
            print("Posible causa: el buffer cayó sobre tierra o zona sin datos del modelo.")
            ds.close()
            continue

        # Filtro de profundidad superficial
        if "depth" in df_agua.columns:
            profundidad_superficie = df_agua["depth"].min()
            df_superficie = df_agua[df_agua["depth"] == profundidad_superficie]
            print(f"Profundidad superficial usada: {profundidad_superficie}")
        else:
            df_superficie = df_agua.copy()
            print("[AVISO] No existe columna depth. Se procesa sin filtro de profundidad.")

        # Promedio espacial diario
        df_diario = (
            df_superficie
            .groupby("time")
            .mean(numeric_only=True)
            .reset_index()
        )

        df_diario["archivo_origen"] = nombre_base
        df_diario["codigo_establecimiento"] = 120128
        df_diario["centro"] = "ACS 54 B"

        lista_dataframes.append(df_diario)

        ds.close()

        print(f"[OK] {nombre_base} procesado. Días: {len(df_diario)}")

    except Exception as e:
        print(f"[ERROR] No se pudo procesar {nombre_base}.")
        print(f"Detalle: {e}")

# ============================================================
# CONSOLIDACIÓN FINAL
# ============================================================

print("\n====================================================")
print(" UNIENDO TODOS LOS ARCHIVOS")
print("====================================================")

if lista_dataframes:
    df_maestro = pd.concat(lista_dataframes, ignore_index=True)

    df_maestro = df_maestro.sort_values("time").reset_index(drop=True)

    df_maestro = df_maestro.rename(columns={
        "time": "fecha",
        "thetao": "temperatura_C",
        "so": "salinidad_PSU",
        "latitude": "latitud_promedio",
        "longitude": "longitud_promedio",
        "depth": "profundidad_m"
    })

    df_maestro.to_csv(archivo_salida, index=False, encoding="utf-8-sig")

    print("\nCONSOLIDACION EXITOSA")
    print(f"Total de registros diarios procesados: {len(df_maestro)}")
    print("Archivo guardado en:")
    print(archivo_salida)

    print("\nVista previa:")
    columnas_vista = ["fecha", "temperatura_C", "salinidad_PSU"]

    columnas_vista = [col for col in columnas_vista if col in df_maestro.columns]

    print(df_maestro[columnas_vista].iloc[np.r_[0:3, -3:0]])

else:
    print("No se generó ningún DataFrame.")
    print("Esto significa que los archivos no tenían datos válidos de thetao/so.")
    print("Revisa si Copernicus descargó solo NaN para esa zona.")