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
# Chile completo, Noruega completo y Chile-Noruega
# Piojo de mar + temperatura + salinidad
# ============================================================

# ------------------------------------------------------------
# 1. Rutas
# ------------------------------------------------------------

carpeta_chile = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\dataset\datasetchile"
carpeta_noruega = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\dataset\datasetnoruega"

carpeta_salida = r"C:\Users\jarpa\OneDrive\Escritorio\datos tesis\tesis\Clustering_General"
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

        "parasitos_totales": "parasitos_totales",
        "parasites_total": "parasitos_totales",
        "total_lice": "parasitos_totales",
        "carga_total_piojos": "parasitos_totales",

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
            "parasitos_totales",
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
# 5. Ordenar registros y crear índice temporal
# ------------------------------------------------------------

def preparar_variables(df):
    df = df.copy()

    df = df.sort_values(["pais", "region", "anio", "semana"]).reset_index(drop=True)

    df["indice_temporal"] = range(1, len(df) + 1)

    return df

# ------------------------------------------------------------
# 6. Reordenar clusters según carga parasitaria
# ------------------------------------------------------------

def reordenar_clusters_por_carga(labels, df_modelo):
    """
    Reordena las etiquetas de clusters para que:

    Cluster 0 = mayor promedio de parásitos totales
    Cluster 1 = segundo mayor promedio de parásitos totales
    ...
    Cluster k-1 = menor promedio de parásitos totales

    Esto se hace porque KMeans asigna etiquetas arbitrarias.
    """

    temp = df_modelo.copy()
    temp["cluster_original"] = labels

    resumen_carga = (
        temp
        .groupby("cluster_original")["parasitos_totales"]
        .mean()
        .sort_values(ascending=False)
    )

    mapa_clusters = {
        cluster_original: cluster_nuevo
        for cluster_nuevo, cluster_original in enumerate(resumen_carga.index)
    }

    labels_reordenados = (
        pd.Series(labels)
        .map(mapa_clusters)
        .astype(int)
        .to_numpy()
    )

    tabla_mapa = pd.DataFrame({
        "cluster_original": resumen_carga.index,
        "cluster_nuevo": [mapa_clusters[c] for c in resumen_carga.index],
        "promedio_parasitos_totales": resumen_carga.values
    })

    return labels_reordenados, mapa_clusters, tabla_mapa

# ------------------------------------------------------------
# 7. Función general de clustering
# ------------------------------------------------------------

def ejecutar_clustering(df, nombre_analisis, carpeta_base_salida, k_seleccionado):
    print("\n====================================================")
    print(f" EJECUTANDO CLUSTERING: {nombre_analisis}")
    print("====================================================")

    # Nombre limpio para carpeta de salida
    nombre_carpeta = (
        nombre_analisis
        .replace(" ", "-")
        .replace("_", "-")
    )

    carpeta_out = os.path.join(carpeta_base_salida, nombre_carpeta)
    os.makedirs(carpeta_out, exist_ok=True)

    variables_cluster = [
        "hembras_ovigeras",
        "adultos_moviles",
        "juveniles",
        "temperatura",
        "salinidad"
    ]

    # Se exige información completa en las variables del modelo y en
    # parasitos_totales, que se usa para ordenar e interpretar los clústeres.
    columnas_completas = variables_cluster + ["parasitos_totales"]

    datos_completos = df[columnas_completas].copy()
    datos_completos = datos_completos.replace([np.inf, -np.inf], np.nan)
    datos_completos = datos_completos.dropna()

    df_modelo = df.loc[datos_completos.index].copy()
    datos = datos_completos[variables_cluster]

    print(f"Filas usadas para clustering: {len(df_modelo)}")

    if len(df_modelo) < 10:
        print("Muy pocos datos para clustering. Se omite este análisis.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(datos)

    # --------------------------------------------------------
    # Evaluar k con codo y silhouette
    # --------------------------------------------------------

    resultados_k = []

    k_values_codo = range(1, 9)
    k_values_silhouette = range(2, 9)

    for k in k_values_codo:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )

        labels = kmeans.fit_predict(X_scaled)

        silhouette = (
            silhouette_score(X_scaled, labels)
            if k >= 2 else np.nan
        )

        tamanos = pd.Series(labels).value_counts().sort_index()

        resultados_k.append({
            "k": k,
            "inercia": kmeans.inertia_,
            "silhouette_promedio": silhouette,
            "tamano_minimo": int(tamanos.min()),
            "tamano_maximo": int(tamanos.max()),
            "tamanos_clusters": "; ".join(
                f"cluster_{cluster}={cantidad}"
                for cluster, cantidad in tamanos.items()
            )
        })

    tabla_validacion = pd.DataFrame(resultados_k)

    tabla_silhouette = tabla_validacion.dropna(
        subset=["silhouette_promedio"]
    )

    k_max_silhouette = int(
        tabla_silhouette.loc[
            tabla_silhouette["silhouette_promedio"].idxmax(),
            "k"
        ]
    )

    if k_seleccionado not in k_values_silhouette:
        raise ValueError("k_seleccionado debe encontrarse entre 2 y 8")

    tabla_validacion["seleccionado_final"] = (
        tabla_validacion["k"] == k_seleccionado
    )

    tabla_validacion["maximo_silhouette"] = (
        tabla_validacion["k"] == k_max_silhouette
    )

    ruta_validacion = os.path.join(carpeta_out, "00_validacion_k.csv")

    tabla_validacion.round(6).to_csv(
        ruta_validacion,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Gráfico del codo
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        tabla_validacion["k"],
        tabla_validacion["inercia"],
        marker="o"
    )

    plt.title(f"Método del codo - {nombre_analisis}")
    plt.xlabel("Número de clústeres (k)")
    plt.ylabel("Inercia")
    plt.grid(True, alpha=0.3)

    ruta_codo = os.path.join(carpeta_out, "01_metodo_codo.png")
    plt.savefig(ruta_codo, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Gráfico Silhouette
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        tabla_silhouette["k"],
        tabla_silhouette["silhouette_promedio"],
        marker="o"
    )

    plt.title(f"Coeficiente Silhouette - {nombre_analisis}")
    plt.xlabel("Número de clústeres (k)")
    plt.ylabel("Coeficiente Silhouette")
    plt.grid(True, alpha=0.3)

    ruta_silhouette = os.path.join(carpeta_out, "02_silhouette_score.png")
    plt.savefig(ruta_silhouette, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nResultados de validación:")

    for _, fila in tabla_silhouette.iterrows():
        print(
            f"k = {int(fila['k'])}: "
            f"silhouette = {fila['silhouette_promedio']:.3f}; "
            f"tamaños = {fila['tamanos_clusters']}"
        )

    print(f"k con mayor silhouette: {k_max_silhouette}")
    print(f"k seleccionado finalmente: {k_seleccionado}")

    # --------------------------------------------------------
    # Aplicar K-Means final
    # --------------------------------------------------------

    kmeans_final = KMeans(
        n_clusters=k_seleccionado,
        random_state=42,
        n_init=20
    )

    labels_originales = kmeans_final.fit_predict(X_scaled)

    # Reordenar clusters por carga parasitaria:
    # Cluster 0 = mayor carga
    # Cluster k-1 = menor carga
    labels_reordenados, mapa_clusters, tabla_mapa_clusters = reordenar_clusters_por_carga(
        labels_originales,
        df_modelo
    )

    df_modelo["cluster_original"] = labels_originales
    df_modelo["cluster"] = labels_reordenados

    print("\nReordenamiento de clusters según carga parasitaria:")
    print("Cluster original -> Cluster nuevo")

    for _, fila in tabla_mapa_clusters.iterrows():
        original = int(fila["cluster_original"])
        nuevo = int(fila["cluster_nuevo"])
        promedio = fila["promedio_parasitos_totales"]
        cantidad = int(np.sum(labels_originales == original))

        print(
            f"{original} -> {nuevo} | "
            f"promedio parásitos totales = {promedio:.3f} | "
            f"n = {cantidad}"
        )

    ruta_mapa_clusters = os.path.join(
        carpeta_out,
        "00_mapa_reordenamiento_clusters.csv"
    )

    tabla_mapa_clusters.to_csv(
        ruta_mapa_clusters,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # PCA para visualización
    # --------------------------------------------------------

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    varianza_explicada = pca.explained_variance_ratio_

    resumen_pca = pd.DataFrame({
        "componente": ["PCA1", "PCA2"],
        "proporcion_varianza_explicada": varianza_explicada,
        "porcentaje_varianza_explicada": varianza_explicada * 100
    }).round(4)

    ruta_resumen_pca = os.path.join(
        carpeta_out,
        "03_varianza_explicada_pca.csv"
    )

    resumen_pca.to_csv(
        ruta_resumen_pca,
        index=False,
        encoding="utf-8-sig"
    )

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
    plt.xlabel(f"Componente principal 1 ({varianza_explicada[0] * 100:.1f} %)")
    plt.ylabel(f"Componente principal 2 ({varianza_explicada[1] * 100:.1f} %)")
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
    plt.xlabel(f"Componente principal 1 ({varianza_explicada[0] * 100:.1f} %)")
    plt.ylabel(f"Componente principal 2 ({varianza_explicada[1] * 100:.1f} %)")
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
            promedio_parasitos_totales=("parasitos_totales", "mean"),
            promedio_temperatura=("temperatura", "mean"),
            promedio_salinidad=("salinidad", "mean")
        )
        .reset_index()
        .sort_values("cluster")
    )

    for col in resumen_cluster.select_dtypes(include="number").columns:
        resumen_cluster[col] = resumen_cluster[col].round(3)

    # Como los clusters ya están ordenados por carga:
    # Cluster 0 = carga más alta
    # Cluster k-1 = carga más baja
    resumen_cluster["orden_carga"] = resumen_cluster["cluster"] + 1

    resumen_cluster["interpretacion"] = resumen_cluster["cluster"].apply(
        lambda c: (
            "Mayor carga parasitaria"
            if c == resumen_cluster["cluster"].min()
            else (
                "Menor carga parasitaria"
                if c == resumen_cluster["cluster"].max()
                else "Carga parasitaria intermedia"
            )
        )
    )

    ruta_resumen = os.path.join(carpeta_out, "05_resumen_clusters.csv")

    resumen_cluster.to_csv(
        ruta_resumen,
        index=False,
        encoding="utf-8-sig"
    )

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

    ruta_region_cluster = os.path.join(
        carpeta_out,
        "06_tabla_region_cluster.csv"
    )

    tabla_region_cluster.to_csv(
        ruta_region_cluster,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Serie temporal de parásitos totales por clúster
    # --------------------------------------------------------

    df_modelo = (
        df_modelo
        .sort_values(["pais", "region", "anio", "semana"])
        .reset_index(drop=True)
    )

    df_modelo["tiempo_decimal"] = (
        df_modelo["anio"] + (df_modelo["semana"] - 1) / 53
    )

    plt.figure(figsize=(15, 6))

    for cluster in sorted(df_modelo["cluster"].unique()):
        sub = df_modelo[df_modelo["cluster"] == cluster]

        plt.scatter(
            sub["tiempo_decimal"],
            sub["parasitos_totales"],
            label=f"Cluster {cluster}",
            s=25
        )

    # Las líneas se trazan separadamente para evitar unir regiones distintas.
    for region in sorted(df_modelo["region"].unique()):
        sub_region = df_modelo[df_modelo["region"] == region]

        plt.plot(
            sub_region["tiempo_decimal"],
            sub_region["parasitos_totales"],
            linewidth=0.7,
            alpha=0.35
        )

    plt.title(f"Evolución temporal de parásitos totales por clúster - {nombre_analisis}")
    plt.xlabel("Año")
    plt.ylabel("Parásitos totales")
    plt.legend()
    plt.grid(True, alpha=0.3)

    ruta_serie = os.path.join(carpeta_out, "07_serie_temporal_clusters.png")
    plt.savefig(ruta_serie, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Boxplot de parásitos totales por clúster
    # --------------------------------------------------------

    clusters_ordenados = sorted(df_modelo["cluster"].unique())

    datos_boxplot = [
        df_modelo[df_modelo["cluster"] == c]["parasitos_totales"]
        for c in clusters_ordenados
    ]

    plt.figure(figsize=(10, 6))

    plt.boxplot(
        datos_boxplot,
        tick_labels=[f"Clúster {c}" for c in clusters_ordenados]
    )

    plt.title(f"Distribución de parásitos totales por clúster - {nombre_analisis}")
    plt.xlabel("Cluster")
    plt.ylabel("Parásitos totales")
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
        "promedio_parasitos_totales"
    ]

    resumen_parasitos = resumen_cluster.set_index("cluster")[variables_parasitarias]

    plt.figure(figsize=(12, 6))

    resumen_parasitos.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(f"Promedio de variables parasitarias por cluster - {nombre_analisis}")
    plt.xlabel("Cluster")
    plt.ylabel("Valor promedio")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ruta_barras_parasitos = os.path.join(
        carpeta_out,
        "09_promedios_parasitarios.png"
    )

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

    resumen_ambiental.plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title(f"Promedio de temperatura y salinidad por cluster - {nombre_analisis}")
    plt.xlabel("Cluster")
    plt.ylabel("Valor promedio")
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    plt.legend(["Temperatura (°C)", "Salinidad (PSU)"])

    ruta_barras_ambiente = os.path.join(
        carpeta_out,
        "10_promedios_ambientales.png"
    )

    plt.savefig(ruta_barras_ambiente, dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # Guardar dataset con clusters
    # --------------------------------------------------------

    ruta_dataset_cluster = os.path.join(
        carpeta_out,
        "11_dataset_con_clusters.csv"
    )

    df_modelo.to_csv(
        ruta_dataset_cluster,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\nArchivos guardados en: {carpeta_out}")

# ============================================================
# 8. Cargar datos
# ============================================================

chile = cargar_pais(carpeta_chile, "Chile")
noruega = cargar_pais(carpeta_noruega, "Noruega")

chile = preparar_variables(chile)
noruega = preparar_variables(noruega)

todo = pd.concat([chile, noruega], ignore_index=True)
todo = preparar_variables(todo)

# ------------------------------------------------------------
# Guardar bases unificadas
# ------------------------------------------------------------

chile.to_csv(
    os.path.join(carpeta_salida, "Base_Chile_Clustering.csv"),
    index=False,
    encoding="utf-8-sig"
)

noruega.to_csv(
    os.path.join(carpeta_salida, "Base_Noruega_Clustering.csv"),
    index=False,
    encoding="utf-8-sig"
)

todo.to_csv(
    os.path.join(carpeta_salida, "Base_Chile_Noruega_Clustering.csv"),
    index=False,
    encoding="utf-8-sig"
)

# ============================================================
# 9. Ejecutar clustering en 3 niveles
# ============================================================

ejecutar_clustering(
    df=chile,
    nombre_analisis="Chile",
    carpeta_base_salida=carpeta_salida,
    k_seleccionado=2
)

ejecutar_clustering(
    df=noruega,
    nombre_analisis="Noruega",
    carpeta_base_salida=carpeta_salida,
    k_seleccionado=3
)

ejecutar_clustering(
    df=todo,
    nombre_analisis="Chile-Noruega",
    carpeta_base_salida=carpeta_salida,
    k_seleccionado=3
)

print("\n====================================================")
print("PROCESO FINALIZADO CORRECTAMENTE")
print("====================================================")
print(f"Resultados guardados en: {carpeta_salida}")