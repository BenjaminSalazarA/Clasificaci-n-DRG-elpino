import re
import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import joblib

# Normaliza texto
def normalizar(texto):
    texto = str(texto).lower().strip()
    texto = "".join(
        c for c in unicodedata.normalize("NFKD", texto)
        if not unicodedata.combining(c)
    )
    texto = re.sub(r"[^a-z0-9]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# Normaliza nombres de columnas
def normalizar_columnas(df):
    df = df.copy()
    df.columns = [normalizar(col).replace(" ", "_") for col in df.columns]
    return df


# Une diagnosticos y procedimientos
def crear_texto_clinico(df):
    diag_cols = [col for col in df.columns if col.startswith("diag_")]
    proc_cols = [col for col in df.columns if col.startswith("proced_")]

    df["texto_clinico"] = df[diag_cols + proc_cols].fillna("").agg(" ".join, axis=1)
    df["texto_clinico"] = df["texto_clinico"].apply(normalizar)

    return df


def main():
    # Cargar datos
    df = pd.read_csv("dataset_elpino.csv", sep=";")

    # Normaliza columnas
    df = normalizar_columnas(df)

    # Limpiar variables basicas
    df["edad_en_anos"] = pd.to_numeric(df["edad_en_anos"], errors="coerce")
    df["sexo_desc"] = df["sexo_desc"].astype(str).str.strip()
    df["grd"] = df["grd"].astype(str).str.strip()

    # Crear texto clinico
    df = crear_texto_clinico(df)

    # Eliminar filas vacias
    df = df[(df["grd"] != "") & (df["texto_clinico"] != "")].copy()

    # Filtrar clases con pocos ejemplos
    conteo = df["grd"].value_counts()
    clases_validas = conteo[conteo >= 5].index
    df = df[df["grd"].isin(clases_validas)]
    
    # Variables
    X = df["texto_clinico"]
    y = df["grd"]

    # Separa entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Vectorizar texto
    vectorizador = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X_train_vec = vectorizador.fit_transform(X_train)
    X_test_vec = vectorizador.transform(X_test)

    # Modelo
    modelo = modelo = LogisticRegression(max_iter=1000, class_weight='balanced')
    modelo.fit(X_train_vec, y_train)

    #Guardar modelo
    joblib.dump(modelo, "modelo_LR.pkl")

    # Prediccion
    y_pred = modelo.predict(X_test_vec)

    #matriz de confusion
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation=90)

    plt.title("Matriz de Confusión - Regresión Logística")
    plt.xlabel("Predicha")
    plt.ylabel("Real")
    plt.show()

    # obtener las 10 clases mas frecuentes en test
    top_clases = y_test.value_counts().head(10).index

    # filtrar datos
    mask = y_test.isin(top_clases)
    y_test_top = y_test[mask]
    y_pred_top = y_pred[mask]

    # matriz
    cm_top = confusion_matrix(y_test_top, y_pred_top, labels=top_clases)

    plt.figure(figsize=(8, 6))
    disp_top = ConfusionMatrixDisplay(
        confusion_matrix=cm_top,
        display_labels=top_clases
    )
    disp_top.plot(cmap="Blues", xticks_rotation=90)
    plt.title("Matriz de Confusión - Top 10 GRD")
    plt.xlabel("Predicha")
    plt.ylabel("Real")
    plt.show()

    # obtener clases menos frecuentes
    bottom_clases = y_test.value_counts().sort_values().head(10).index

    # filtrar datos
    mask = y_test.isin(bottom_clases)
    y_test_bottom = y_test[mask]
    y_pred_bottom = y_pred[mask]

    # matriz
    cm_bottom = confusion_matrix(y_test_bottom, y_pred_bottom, labels=bottom_clases)

    plt.figure(figsize=(8, 6))
    disp_bottom = ConfusionMatrixDisplay(
        confusion_matrix=cm_bottom,
        display_labels=bottom_clases
    )
    disp_bottom.plot(cmap="Reds", xticks_rotation=90)

    plt.xlabel("Predicha")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - peores 10 GRD")

    plt.show()

    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("\nResultados - Regresión Logística")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score ponderado: {f1:.4f}")

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))


if __name__ == "__main__":
    main()