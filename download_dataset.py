from pathlib import Path
import shutil
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

# HIER anpassen:
DATASET = "arturillenseer/rakuten-product-images-ml"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "_kaggle_download"


def remove_if_exists(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def main() -> None:
    print("Starte Dataset-Download...")

    DATA_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        print("Fehler bei der Kaggle-Anmeldung.")
        print("Prüfe, ob deine Datei 'kaggle.json' hier liegt:")
        print(rf"{Path.home()}\.kaggle\kaggle.json")
        print(f"Details: {e}")
        sys.exit(1)

    try:
        print(f"Lade Dataset '{DATASET}' herunter...")
        api.dataset_download_files(
            DATASET,
            path=str(TEMP_DIR),
            unzip=True
        )
    except Exception as e:
        print("Fehler beim Herunterladen des Datasets.")
        print("Prüfe:")
        print("- Ist die Dataset-ID korrekt?")
        print("- Hast du Zugriff auf das Dataset?")
        print(f"Details: {e}")
        sys.exit(1)

    # Alles aus dem temporären Ordner nach data verschieben
    for item in TEMP_DIR.iterdir():
        target = DATA_DIR / item.name
        remove_if_exists(target)
        shutil.move(str(item), str(target))

    remove_if_exists(TEMP_DIR)

    print("Fertig.")
    print(f"Deine Daten liegen jetzt in: {DATA_DIR}")


if __name__ == "__main__":
    main()
