from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import ZipFile


DATA_LOCAL = Path("data_local")
DATA_DIR = Path("data")
CSV_NAME = "loan_data_2007_2014.csv"
ZIP_NAME = CSV_NAME + ".zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_csv = DATA_LOCAL / CSV_NAME
    local_zip = DATA_LOCAL / ZIP_NAME
    out_csv = DATA_DIR / CSV_NAME

    if local_csv.exists():
        shutil.copy2(local_csv, out_csv)
        print(f"Copied {local_csv} -> {out_csv}")
        return

    if local_zip.exists():
        with ZipFile(local_zip) as zf:
            if CSV_NAME in zf.namelist():
                zf.extract(CSV_NAME, DATA_DIR)
                print(f"Extracted {CSV_NAME} from {local_zip} -> {out_csv}")
                return
            else:
                zf.extractall(DATA_DIR)
                print(f"Extracted all contents of {local_zip} into {DATA_DIR}")
                return

    print(
        "No local data found. Place CSV or ZIP under 'data_local/' and rerun."
    )


if __name__ == "__main__":
    main()

