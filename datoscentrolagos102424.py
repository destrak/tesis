import copernicusmarine
import time
import os

# ============================================================
# CENTRO CHILE
# Código establecimiento: 102424
# Coordenadas aproximadas desde polígono
# ============================================================

lat_centro = -42.737833
lon_centro = -73.013083

buffer = 0.15

min_lon = lon_centro - buffer
max_lon = lon_centro + buffer
min_lat = lat_centro - buffer
max_lat = lat_centro + buffer

# Carpeta de salida
directorio_hdd = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Historico_102424"
os.makedirs(directorio_hdd, exist_ok=True)

print("Iniciando extracción Copernicus para centro chileno 102424")
print(f"Latitud centro: {lat_centro}")
print(f"Longitud centro: {lon_centro}")
print(f"Buffer: {buffer} grados")
print(f"Área: lon {min_lon} a {max_lon}, lat {min_lat} a {max_lat}")

# ============================================================
# DESCARGA 2014-2024 COMPLETO
# ============================================================

for anio in range(2014, 2025):

    fecha_inicio = f"{anio}-01-01"
    fecha_fin = f"{anio}-12-31"

    nombre_archivo = f"Centro_102424_temp_sal_{anio}.nc"

    print(f"\n-> Extrayendo año {anio}...")
    print(f"   Periodo: {fecha_inicio} a {fecha_fin}")
    print(f"   Archivo: {nombre_archivo}")

    try:
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao", "so"],
            start_datetime=fecha_inicio,
            end_datetime=fecha_fin,
            minimum_longitude=min_lon,
            maximum_longitude=max_lon,
            minimum_latitude=min_lat,
            maximum_latitude=max_lat,
            minimum_depth=0,
            maximum_depth=1,
            output_directory=directorio_hdd,
            output_filename=nombre_archivo,
            force_download=True
        )

        print(f"[OK] Año {anio} guardado correctamente.")
        time.sleep(3)

    except Exception as e:
        print(f"[ERROR] Fallo en el año {anio}: {e}")

print("\nProceso finalizado.")