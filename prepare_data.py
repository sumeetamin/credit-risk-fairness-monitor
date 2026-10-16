"""Download and convert the cited UCI XLS workbook to the runner's CSV format."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DATA_DIR = Path("data")
ARCHIVE_PATH = DATA_DIR / "uci-default-credit-clients.zip"
CSV_PATH = DATA_DIR / "credit.csv"
METADATA_PATH = DATA_DIR / "source.json"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE_PATH.is_file() or not zipfile.is_zipfile(ARCHIVE_PATH):
        temporary_path = ARCHIVE_PATH.with_suffix(".download")
        request = Request(DATA_URL, headers={"User-Agent": "credit-risk-fairness-monitor/1.0"})
        with urlopen(request, timeout=60) as response:
            temporary_path.write_bytes(response.read())
        if not zipfile.is_zipfile(temporary_path):
            temporary_path.unlink(missing_ok=True)
            raise ValueError("UCI download was not a valid ZIP archive")
        temporary_path.replace(ARCHIVE_PATH)

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        workbook_name = next(name for name in archive.namelist() if name.lower().endswith(".xls"))
        workbook = archive.read(workbook_name)

    # UCI's workbook has a variable-code row followed by the real column names.
    frame = pd.read_excel(io.BytesIO(workbook), header=1)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame.to_csv(CSV_PATH, index=False)
    fingerprint = hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest()
    metadata = {
        "source": "UCI Machine Learning Repository",
        "source_url": "https://archive.ics.uci.edu/dataset/350/defaultofcreditcardclients",
        "download_url": DATA_URL,
        "citation": "Yeh, I. (2009). Default of Credit Card Clients. DOI: 10.24432/C55S3H.",
        "dataset_license": "CC BY 4.0",
        "workbook_name": workbook_name,
        "archive_sha256": fingerprint,
        "csv_sha256": hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
        "rows": int(len(frame)),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns)}")
    print(f"Target: default payment next month")
    print(f"Source archive SHA-256: {fingerprint}")
    print(f"Prepared local CSV: {CSV_PATH}")


if __name__ == "__main__":
    main()

