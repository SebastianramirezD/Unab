"""
download_dataset.py
-------------------
Descarga el dataset SMS Spam Collection desde el UCI Repository
y lo guarda en datasets/raw/spam.csv
"""

import os
import urllib.request
import zipfile
import shutil

# Seed global para reproducibilidad
RANDOM_SEED = 42

RAW_DIR = os.path.join(os.path.dirname(__file__), "../../datasets/raw")
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
ZIP_PATH = os.path.join(RAW_DIR, "smsspamcollection.zip")
OUTPUT_CSV = os.path.join(RAW_DIR, "spam.csv")


def download():
    os.makedirs(RAW_DIR, exist_ok=True)

    print(f"Descargando dataset desde {URL} ...")
    urllib.request.urlretrieve(URL, ZIP_PATH)
    print("Descarga completa.")

    print("Extrayendo archivo...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(RAW_DIR)

    # El archivo original se llama SMSSpamCollection (sin extensión)
    raw_file = os.path.join(RAW_DIR, "SMSSpamCollection")
    if os.path.exists(raw_file):
        # Convertir a CSV con encabezados
        import pandas as pd
        df = pd.read_csv(raw_file, sep="\t", header=None, names=["label", "text"])
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Dataset guardado en: {OUTPUT_CSV}")
        print(f"Total registros: {len(df)}")
        print(df["label"].value_counts())
    else:
        print("ERROR: No se encontró el archivo SMSSpamCollection tras la extracción.")

    # Limpiar zip
    os.remove(ZIP_PATH)


if __name__ == "__main__":
    download()
