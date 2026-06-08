import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ============================================================
# CLUSTERING NO SUPERVISADO
# Chile completo, Noruega completo y Chile + Noruega
# Piojo de mar + temperatura + salinidad
# ============================================================

# ------------------------------------------------------------
# 1. Rutas
# ------------------------------------------------------------

carpeta_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\dataset\datasetchile"
carpeta_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\dataset\datasetnoruega"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\Clustering_General"
os.makedirs(carpeta_salida, exist_ok=True)

# ------------------------------------------------------------
# 2. Asignar región / macrozona según centro
# ------------------------------------------------------------

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

    elif pais == "Noruega":
        if "33077" in nombre:
            return "Sur/Oeste"
        elif "32677" in nombre:
            return "Centro"
        elif "24175" in nombre:
            return "Norte"
        else:
            return "Noruega sin región"

    return "Sin región"

# ------------------------------------------------------------
# 3. Normalizar columnas
# ------------------------------------------------------------

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

def renombrar_columnas(df):
    renombres = {
        "año": "anio",
        "ano": "anio",
        "year": "anio",
        "week": "semana",

        "hembras_ovigeras": "hembras_ovigeras",
        "adult_female_lice": "hembras_ovigeras",
        "adult_female": "hembras_ovigeras",

        "adultos_moviles": "adultos_moviles",
        "lice_in_moving_stages": "adultos_moviles",
        "moving_lice": "adultos_moviles",

        "juveniles": "juveniles",
        "stuck_lice": "juveniles",

        "temperatura_(°c)": "temperatura",
        "temperatura_c": "temperatura",
        "temperatura_°c": "temperatura",
        "temperatura": "temperatura",
        "sea_temperature": "temperatura",
        "temperature": "temperatura",

        "salinidad_(psu)": "salinidad",
        "salinidad_psu": "salinidad",
        "salinidad": "salinidad",
        "salinity": "salinidad"
    }

    df = df.rename(columns={k: v for k, v in renombres.items() if k in df.columns})
    return df

# ------------------------------------------------------------
# 4. Cargar archivos de un país
# ------------------------------------------------------------

def cargar_pais(carpeta, pais):
    archivos = glob(os.path.join(carpeta, "*.csv"))

    if len(archivos) == 0:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {carpeta}")

    lista = []

    for archivo in archivos:
        print(f"Cargando {pais}: {archivo}")

        try:
            df = pd.read_csv(archivo)
        except UnicodeDecodeError:
            df = pd.read_csv(archivo, encoding="latin1")

        df = normalizar_columnas(df)
        df = renombrar_columnas(df)

        nombre_archivo = os.path.splitext(os.path.basename(archivo))[0]

        df["pais"] = pais
        df["centro"] = nombre_archivo
        df["region"] = asignar_region(pais, nombre_archivo)

        columnas_necesarias = [
            "anio",
            "semana",
            "hembras_ovigeras",
            "adultos_moviles",
            "juveniles",
            "temperatura",
            "salinidad"
        ]

        faltantes = [col for col in columnas_necesarias if col not in df.columns]

        if faltantes:
            print(f"Archivo omitido por columnas faltantes: {archivo}")
            print(f"Faltan: {faltantes}")
            continue

        for col in columnas_necesarias:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["anio", "semana"])

        df["anio"] = df["anio"].astype(int)
        df["semana"] = df["semana"].astype(int)

        lista.append(df)

    if len(lista) == 0:
        raise ValueError(f"No se pudo cargar ningún archivo válido desde: {carpeta}")

    return pd.concat(lista, ignore_index=True)

# ------------------------------------------------------------
# 5. Crear variables adicionales
# ------------------------------------------------------------

def preparar_variables(df):
    df = df.copy()

    df = df.sort_values(["pais", "region", "anio", "semana"]).reset_index(drop=True)

    df["carga_total_piojos"] = (
        df["hembras_ovigeras"] +
        df["adultos_moviles"] +
        df["juveniles"]
    )

    df["adultos_totales"] = (
        df["hembras_ovigeras"] +
        df["adultos_moviles"]
    )

    df["indice_temporal"] = range(1, len(df) + 1)

    return df

# ------------------------------------------------------------
# 6. Función general de clustering
# ------------------------------------------------------------

def ejecutar_clustering(df, nombre_analisis, carpeta_base_salida):
    print("\n====================================================")
    print(f" EJECUTANDO CLUSTERING: {nombre_analisis}")
    print("====================================================")

    carpeta_out = os.path.join(carpeta_base_salida, nombre_analisis.replace(" ", "_"))
    os.makedirs(carpeta_out, exist_ok=True)

    variables_cluster = [
        "hembras_ovigeras",
        "adultos_moviles",
        "juveniles",
        "carga_total_piojos",
        "adultos_totales",
        "temperatura",
        "salinidad"
    ]

    datos = df[variables_cluster].copy()
    datos = datos.replace([np.inf, -np.inf], np.nan)
    datos = datos.dropna()

    df_modelo = df.loc[datos.index].copy()

    print(f"Filas usadas para clustering: {len(df_modelo)}")

    if len(df_modelo) < 10:
        print("Muy pocos datos para clustering. Se omite este análisis.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(datos)

    # --------------------------------------------------------
    # Evaluar k con codo y silhouette
    # --------------------------------------------------------

    inercias = []
    silhouettes = []
    k_values = range(2, 9)

    for k in k_values:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X_scaled)

        inercias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    # Gráfico codo
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, inercias, marker="o")
    plt.title(f"Método del codo - {nombre_analisis}")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.grid(True, alpha=0.3)

    ruta_codo = os.path.join(carpeta_out, "01_metodo_codo.png")
    plt.savefig(ruta_codo, dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico silhouette
    plt.figure(figsize=(10, 5))
    plt.plot(k_values, silhouettes, marker="o")
    plt.title(f"Coeficiente Silhouette - {nombre_analisis}")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Silhouette score")
    plt.grid(True, alpha=0.3)

    ruta_silhouette = os.path.join(carpeta_out, "02_silhouette_score.png")
    plt.savefig(ruta_silhouette, dpi=300, bbox_inches="tight")
    plt.close()

    k_optimo = list(k_values)[np.argmax(silhouettes)]

    print("\nResultados silhouette:")
    for k, s in zip(k_values, silhouettes):
        print(f"k = {k}: silhouette = {s:.3f}")

    print(f"k óptimo sugerido: {k_optimo}")

    # --------------------------------------------------------
    # Aplicar K-Means final
    # --------------------------------------------------------

    kmeans_final = KMeans(
        n_clusters=k_optimo,
        random_state=42,
        n_init=20
    )

    df_modelo["cluster"] = kmeans_final.fit_predict(X_scaled)

    # --------------------------------------------------------
    # PCA para visualización
    # --------------------------------------------------------

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df_modelo["PCA1"] = X_pca[:, 0]
    df_modelo["PCA2"] = X_pca[:, 1]

    plt.figure(figsize=(10, 7))

    for cluster in sorted(df_modelo["cluster"].unique()):
        sub = df_modelo[df_modelo["cluster"] == cluster]

        plt.scatter(
            sub["PCA1"],
            sub["PCA2"],
            label=f"Cluster {cluster}",
            alpha=0.75
        )

    plt.title(f"Clusters PCA - {nombre_analisis}")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ruta_pca = os.path.join(carpeta_out, "03_clusters_pca.png")
    plt.savefig(ruta_pca, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # PCA con país / región
    # --------------------------------------------------------

    plt.figure(figsize=(10, 7))

    for region in sorted(df_modelo["region"].unique()):
        sub = df_modelo[df_modelo["region"] == region]

        plt.scatter(
            sub["PCA1"],
            sub["PCA2"],
            label=region,
            alpha=0.75
        )

    plt.title(f"Distribución por región/macrozona - {nombre_analisis}")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ruta_pca_region = os.path.join(carpeta_out, "04_pca_por_region.png")
    plt.savefig(ruta_pca_region, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Resumen por cluster
    # --------------------------------------------------------

    resumen_cluster = (
        df_modelo
        .groupby("cluster")
        .agg(
            semanas=("semana", "count"),
            promedio_hembras_ovigeras=("hembras_ovigeras", "mean"),
            promedio_adultos_moviles=("adultos_moviles", "mean"),
            promedio_juveniles=("juveniles", "mean"),
            promedio_adultos_totales=("adultos_totales", "mean"),
            promedio_carga_total=("carga_total_piojos", "mean"),
            promedio_temperatura=("temperatura", "mean"),
            promedio_salinidad=("salinidad", "mean")
        )
        .reset_index()
    )

    for col in resumen_cluster.select_dtypes(include="number").columns:
        resumen_cluster[col] = resumen_cluster[col].round(3)

    q25 = resumen_cluster["promedio_carga_total"].quantile(0.25)
    q75 = resumen_cluster["promedio_carga_total"].quantile(0.75)

    def interpretar_cluster(row):
        carga = row["promedio_carga_total"]

        if carga >= q75:
            return "Alta carga parasitaria"
        elif carga <= q25:
            return "Baja carga parasitaria"
        else:
            return "Carga parasitaria intermedia"

    resumen_cluster["interpretacion"] = resumen_cluster.apply(interpretar_cluster, axis=1)

    ruta_resumen = os.path.join(carpeta_out, "05_resumen_clusters.csv")
    resumen_cluster.to_csv(ruta_resumen, index=False, encoding="utf-8-sig")

    print("\nResumen por cluster:")
    print(resumen_cluster)

    # --------------------------------------------------------
    # Distribución de regiones por cluster
    # --------------------------------------------------------

    tabla_region_cluster = pd.crosstab(
        df_modelo["region"],
        df_modelo["cluster"],
        margins=True
    )

    ruta_region_cluster = os.path.join(carpeta_out, "06_tabla_region_cluster.csv")
    tabla_region_cluster.to_csv(ruta_region_cluster, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Serie temporal de carga total por cluster
    # --------------------------------------------------------

    df_modelo = df_modelo.sort_values(["pais", "region", "anio", "semana"]).reset_index(drop=True)
    df_modelo["indice_temporal"] = range(1, len(df_modelo) + 1)

    plt.figure(figsize=(15, 6))

    for cluster in sorted(df_modelo["cluster"].unique()):
        sub = df_modelo[df_modelo["cluster"] == cluster]

        plt.scatter(
            sub["indice_temporal"],
            sub["carga_total_piojos"],
            label=f"Cluster {cluster}",
            s=25
        )

    plt.plot(
        df_modelo["indice_temporal"],
        df_modelo["carga_total_piojos"],
        linewidth=0.8,
        alpha=0.5
    )

    plt.title(f"Evolución temporal de la carga total por cluster - {nombre_analisis}")
    plt.xlabel("Índice temporal")
    plt.ylabel("Carga total de piojos")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ruta_serie = os.path.join(carpeta_out, "07_serie_temporal_clusters.png")
    plt.savefig(ruta_serie, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Boxplot de carga total por cluster
    # --------------------------------------------------------

    clusters_ordenados = sorted(df_modelo["cluster"].unique())

    datos_boxplot = [
        df_modelo[df_modelo["cluster"] == c]["carga_total_piojos"]
        for c in clusters_ordenados
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(
        datos_boxplot,
        labels=[f"Cluster {c}" for c in clusters_ordenados]
    )

    plt.title(f"Distribución de carga total por cluster - {nombre_analisis}")
    plt.xlabel("Cluster")
    plt.ylabel("Carga total de piojos")
    plt.grid(True, alpha=0.3)

    ruta_boxplot = os.path.join(carpeta_out, "08_boxplot_carga_total.png")
    plt.savefig(ruta_boxplot, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Promedio de variables parasitarias por cluster
    # --------------------------------------------------------

    variables_parasitarias = [
        "promedio_hembras_ovigeras",
        "promedio_adultos_moviles",
        "promedio_juveniles",
        "promedio_adultos_totales",
        "promedio_carga_total"
    ]

    resumen_parasitos = resumen_cluster.set_index("cluster")[variables_parasitarias]

    plt.figure(figsize=(12, 6))
    resumen_parasitos.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Promedio de variables parasitarias por cluster - {nombre_analisis}")
    plt.xlabel("Cluster")
    plt.ylabel("Valor promedio")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ruta_barras_parasitos = os.path.join(carpeta_out, "09_promedios_parasitarios.png")
    plt.savefig(ruta_barras_parasitos, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Promedio temperatura y salinidad por cluster
    # --------------------------------------------------------

    variables_ambientales = [
        "promedio_temperatura",
        "promedio_salinidad"
    ]

    resumen_ambiental = resumen_cluster.set_index("cluster")[variables_ambientales]

    plt.figure(figsize=(10, 6))
    resumen_ambiental.plot(kind="bar", figsize=(10, 6))
    plt.title(f"Promedio de temperatura y salinidad por cluster - {nombre_analisis}")
    plt.xlabel("Cluster")
    plt.ylabel("Valor promedio")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.legend(["Temperatura (°C)", "Salinidad (PSU)"])

    ruta_barras_ambiente = os.path.join(carpeta_out, "10_promedios_ambientales.png")
    plt.savefig(ruta_barras_ambiente, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Guardar dataset con clusters
    # --------------------------------------------------------

    ruta_dataset_cluster = os.path.join(carpeta_out, "11_dataset_con_clusters.csv")
    df_modelo.to_csv(ruta_dataset_cluster, index=False, encoding="utf-8-sig")

    print(f"\nArchivos guardados en: {carpeta_out}")

# ============================================================
# 7. Cargar datos
# ============================================================

chile = cargar_pais(carpeta_chile, "Chile")
noruega = cargar_pais(carpeta_noruega, "Noruega")

chile = preparar_variables(chile)
noruega = preparar_variables(noruega)

todo = pd.concat([chile, noruega], ignore_index=True)
todo = preparar_variables(todo)

# Guardar bases unificadas
chile.to_csv(os.path.join(carpeta_salida, "Base_Chile_Clustering.csv"), index=False, encoding="utf-8-sig")
noruega.to_csv(os.path.join(carpeta_salida, "Base_Noruega_Clustering.csv"), index=False, encoding="utf-8-sig")
todo.to_csv(os.path.join(carpeta_salida, "Base_Chile_Noruega_Clustering.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 8. Ejecutar clustering en 3 niveles
# ============================================================

ejecutar_clustering(
    df=chile,
    nombre_analisis="01_Chile_completo",
    carpeta_base_salida=carpeta_salida
)

ejecutar_clustering(
    df=noruega,
    nombre_analisis="02_Noruega_completo",
    carpeta_base_salida=carpeta_salida
)

ejecutar_clustering(
    df=todo,
    nombre_analisis="03_Chile_Noruega_conjunto",
    carpeta_base_salida=carpeta_salida
)

print("\n====================================================")
print("PROCESO FINALIZADO CORRECTAMENTE")
print("====================================================")
print(f"Resultados guardados en: {carpeta_salida}")