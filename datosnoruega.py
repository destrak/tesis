"""Descarga registros semanales de piojo de mar desde BarentsWatch.

Configuración obligatoria mediante variables de entorno:
    BARENTSWATCH_CLIENT_ID
    BARENTSWATCH_CLIENT_SECRET

Salida:
    barentswatch_locality_32677_lice_2014_2024.csv

Instalar dependencias (solo si faltan):
    python -m pip install requests pandas
"""

import os
from io import StringIO

import pandas as pd
import requests

# Configuración original de la consulta.
LOCALITY_NO = 32677
ANIO_INICIAL = 2014
ANIO_FINAL = 2024
TOKEN_URL = "https://id.barentswatch.no/connect/token"
BASE_URL = "https://www.barentswatch.no/bwapi/v1/geodata/download/fishhealth"
ARCHIVO_SALIDA = f"barentswatch_locality_{LOCALITY_NO}_lice_{ANIO_INICIAL}_{ANIO_FINAL}.csv"


def obtener_token(sesion):
    """Solicita un token OAuth2 sin exponer las credenciales en el código."""
    client_id = os.environ.get("BARENTSWATCH_CLIENT_ID", "").strip()
    client_secret = os.environ.get("BARENTSWATCH_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        raise RuntimeError(
            "Faltan las variables de entorno BARENTSWATCH_CLIENT_ID y/o "
            "BARENTSWATCH_CLIENT_SECRET. Configúralas antes de ejecutar el script."
        )

    respuesta = sesion.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "scope": "api",
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    print("Estado de autenticación:", respuesta.status_code)
    if not respuesta.ok:
        raise RuntimeError(
            "No se pudo autenticar en BarentsWatch. "
            "Comprueba las credenciales, los permisos y la conexión de red "
            f"(HTTP {respuesta.status_code})."
        )

    token = respuesta.json().get("access_token")
    if not token:
        raise RuntimeError("BarentsWatch no devolvió un access_token válido.")
    return token


def main():
    dataframes = []
    with requests.Session() as sesion:
        token = obtener_token(sesion)
        sesion.headers.update({"Authorization": f"Bearer {token}"})

        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1):
            params = {
                "filetype": "csv",
                "reporttype": "lice",
                "localityno": LOCALITY_NO,
                "fromyear": anio,
                "fromweek": 1,
                "toyear": anio,
                "toweek": 52,
            }

            try:
                respuesta = sesion.get(BASE_URL, params=params, timeout=60)
                print(f"Año {anio}: HTTP {respuesta.status_code}")
                if not respuesta.ok:
                    print(f"  No se obtuvieron datos (HTTP {respuesta.status_code}).")
                    continue
                if not respuesta.text.strip():
                    print("  Respuesta vacía.")
                    continue

                datos = pd.read_csv(
                    StringIO(respuesta.text), sep=None, engine="python"
                )
                datos["year_download"] = anio
                dataframes.append(datos)
                print("  Filas:", len(datos))
            except requests.RequestException as error:
                print(f"  Error de conexión para {anio}: {type(error).__name__}")
            except (ValueError, pd.errors.ParserError) as error:
                print(f"  Error al leer el CSV de {anio}: {type(error).__name__}")

    if not dataframes:
        print("No se descargaron datos.")
        return

    df_final = pd.concat(dataframes, ignore_index=True)
    df_final.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")
    print(f"CSV generado correctamente: {ARCHIVO_SALIDA}")
    print("Dimensiones:", df_final.shape)
    print(df_final.head())


if __name__ == "__main__":
    main()
