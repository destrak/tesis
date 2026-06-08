import requests
import pandas as pd
from io import StringIO

CLIENT_ID = "nijarpa2021@udec.cl:PINOX"
CLIENT_SECRET = "PINOX_2026_BW_locality24175_secure_access_ABC123"

# ==========================
# OBTENER TOKEN
# ==========================

token_url = "https://id.barentswatch.no/connect/token"

token_data = {
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "scope": "api",
    "grant_type": "client_credentials"
}

token_response = requests.post(token_url, data=token_data)

print("Estado token:", token_response.status_code)

if token_response.status_code != 200:
    print(token_response.text)
    raise SystemExit("No se pudo obtener token.")

access_token = token_response.json()["access_token"]

headers = {
    "Authorization": f"Bearer {access_token}"
}

# ==========================
# DESCARGA
# ==========================

base_url = "https://www.barentswatch.no/bwapi/v1/geodata/download/fishhealth"

localityno = 32677
dfs = []

for year in range(2014, 2025):
    params = {
        "filetype": "csv",
        "reporttype": "lice",
        "localityno": localityno,
        "fromyear": year,
        "fromweek": 1,
        "toyear": year,
        "toweek": 52
    }

    response = requests.get(base_url, headers=headers, params=params)

    print(year, response.status_code)
    print("URL:", response.url)

    if response.status_code == 200 and response.text.strip():
        try:
            df = pd.read_csv(StringIO(response.text), sep=None, engine="python")
            df["year_download"] = year
            dfs.append(df)
            print("Filas:", len(df))
        except Exception as e:
            print("Error leyendo CSV:", e)
            print(response.text[:1000])
    else:
        print(response.text[:1000])

if dfs:
    df_final = pd.concat(dfs, ignore_index=True)

    df_final.to_csv(
        "barentswatch_locality_32677_lice_2014_2024.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("CSV generado correctamente")
    print(df_final.shape)
    print(df_final.head())
else:
    print("No se descargaron datos.")