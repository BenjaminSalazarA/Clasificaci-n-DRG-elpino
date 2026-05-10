from pathlib import Path
import re
import unicodedata

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

OUTPUT_DIR = "output/"

GRD = "grd"
EDAD = "edad_en_anos"
SEXO = "sexo_desc"
DIAGNOSTICO = ["diag_"]
PROCEDIMIENTO = ["proced_"]


#Función para normalizar texto
def normalizar(texto):
    texto = str(texto).strip().lower()
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    texto = re.sub(r"_+", "_", texto).strip("_")
    return texto


#Funcion que estandariza todo el csv
def normalizar_csv(df):
    df = df.copy()
    df.columns = [normalizar(c) for c in df.columns]
    return df


# Limpia datos
def limpiar_campos_basicos(df):
    df = df.copy()

    df[EDAD] = pd.to_numeric(df[EDAD], errors="coerce")
    df[SEXO] = df[SEXO].astype(str).str.strip()
    df[GRD] = df[GRD].astype(str).str.strip()

    return df


# Funcion para buscar diagnosticos y procedimientos
def obtener_columnas_clinicas(df):
    diag_cols = [col for col in df.columns if col.startswith("diag_")]
    proc_cols = [col for col in df.columns if col.startswith("proced_")]
    return diag_cols, proc_cols


# funcion para realizar un analisis de calidad de datos
def resumen_calidad(df, diag_cols, proc_cols):
    filas = []

    columnas_revisar = [GRD, EDAD, SEXO] + diag_cols + proc_cols

    for col in columnas_revisar:
        nulos = df[col].isna().sum()
        vacios = (df[col].astype(str).str.strip() == "").sum() if df[col].dtype == "object" else 0
        unicos = df[col].nunique(dropna=True)

        filas.append({
            "columna": col,
            "tipo": str(df[col].dtype),
            "nulos": int(nulos),
            "vacios": int(vacios),
            "unicos": int(unicos),
        })

    return pd.DataFrame(filas)


# Busca errores en los datos
def buscar_errores(df):
    reglas = []

    edad_negativa = int((df[EDAD] < 0).sum())
    edad_mayor_120 = int((df[EDAD] > 120).sum())
    grd_vacio = int((df[GRD].astype(str).str.strip() == "").sum())
    sexo_vacio = int((df[SEXO].astype(str).str.strip() == "").sum())

    reglas.append({"regla": "edad < 0", "cantidad": edad_negativa})
    reglas.append({"regla": "edad > 120", "cantidad": edad_mayor_120})
    reglas.append({"regla": "grd vacío", "cantidad": grd_vacio})
    reglas.append({"regla": "sexo vacío", "cantidad": sexo_vacio})

    return pd.DataFrame(reglas)


# Obtiene datos de las edades
def estadisticas_descriptivas(df):
    return df[[EDAD]].describe().T


# Calcula extremos mediante IQR
def calculo_extremos(df):
    q1 = df[EDAD].quantile(0.25)
    q3 = df[EDAD].quantile(0.75)
    iqr = q3 - q1
    li = q1 - 1.5 * iqr
    ls = q3 + 1.5 * iqr

    extremos = df[(df[EDAD] < li) | (df[EDAD] > ls)]

    return pd.DataFrame([{
        "variable": EDAD,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "limite_inferior": li,
        "limite_superior": ls,
        "cantidad_extremos": len(extremos),
    }])


# Grafico de Edades
def grafico_edad(df, outdir):
    plt.figure(figsize=(10, 5))
    sns.histplot(df[EDAD].dropna(), bins=30, kde=True)
    plt.title("Distribución de edad")
    plt.xlabel("Edad")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(outdir / "Grafico_Edades.png", dpi=200)
    plt.close()


#Diagrama Caja y Bigotes de Edades
def diagrama_caja_bigotes(df, outdir):
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[EDAD])
    plt.title("Diagrama Caja y Bigotes de edad")
    plt.xlabel("Edad")
    plt.tight_layout()
    plt.savefig(outdir / "Caja_Bigotes_Edad.png", dpi=200)
    plt.close()


# Distribucion de clases GRD
def grafico_grd(df, outdir, top_n=20):
    top = df[GRD].value_counts().head(top_n)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top.index, y=top.values)
    plt.title(f"Top {top_n} clases GRD")
    plt.xlabel("GRD")
    plt.ylabel("Cantidad")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(outdir / "Distribucion GRD.png", dpi=200)
    plt.close()


# guarda estructura del dataset utilizado
def guardar_esquema(diag_cols, proc_cols, outdir):
    contenido = [
        f"target: {GRD}",
        f"age: {EDAD}",
        f"sex: {SEXO}",
        f"diag_cols: {diag_cols}",
        f"proc_cols: {proc_cols}",
    ]
    (outdir / "estructura_dataset.txt").write_text("\n".join(contenido), encoding="utf-8")


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("dataset_elpino.csv", sep=";")
    df = normalizar_csv(df)
    df = limpiar_campos_basicos(df)

    diag_cols, proc_cols = obtener_columnas_clinicas(df)

    quality_df = resumen_calidad(df, diag_cols, proc_cols)
    quality_df.to_csv(outdir / "calidad_datos.csv", index=False)

    correctness_df = buscar_errores(df)
    correctness_df.to_csv(outdir / "extremos.csv", index=False)

    stats_df = estadisticas_descriptivas(df)
    stats_df.to_csv(outdir / "datos_edades).csv")

    extremos_df = calculo_extremos(df)
    extremos_df.to_csv(outdir / "casos_extremos.csv", index=False)

    grafico_edad(df, outdir)
    diagrama_caja_bigotes(df, outdir)
    grafico_grd(df, outdir)

    guardar_esquema(diag_cols, proc_cols, outdir)

    print("EDA finalizado.")
    print("Dataset usado: dataset_elpino.csv")
    print("Target:", GRD)
    print("Edad:", EDAD)
    print("Sexo:", SEXO)
    print("Diagnósticos detectados:", len(diag_cols))
    print("Procedimientos detectados:", len(proc_cols))
    print("Archivos guardados en:", outdir)


if __name__ == "__main__":
    main()