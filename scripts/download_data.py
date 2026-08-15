from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

# Allow running scripts directly from the repository root without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.config import (  # noqa: E402
    BLS_NATIONAL_XLSX,
    BLS_NATIONAL_ZIP_URL,
    ONET_FILES,
    RAW_DIR,
)

HEADERS = {"User-Agent": "CareerVector/0.1 educational-project"}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...")
    with requests.get(url, stream=True, timeout=120, headers=HEADERS) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_bls(destination_dir: Path) -> None:
    print(f"Downloading BLS OEWS archive for {BLS_NATIONAL_XLSX} ...")
    response = requests.get(BLS_NATIONAL_ZIP_URL, timeout=120, headers=HEADERS)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        xlsx_members = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        if not xlsx_members:
            raise RuntimeError("BLS archive contained no XLSX file")
        # The archive currently contains the national_M2025_dl.xlsx workbook.
        preferred = next((m for m in xlsx_members if Path(m).name == BLS_NATIONAL_XLSX), xlsx_members[0])
        output = destination_dir / BLS_NATIONAL_XLSX
        output.write_bytes(archive.read(preferred))
        print(f"Extracted {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--skip-bls", action="store_true", help="Skip optional BLS salary data")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in ONET_FILES.items():
        destination = RAW_DIR / filename
        if destination.exists() and not args.force:
            print(f"Keeping existing {filename}")
        else:
            download(url, destination)

    wage_path = RAW_DIR / BLS_NATIONAL_XLSX
    if not args.skip_bls:
        if wage_path.exists() and not args.force:
            print(f"Keeping existing {wage_path.name}")
        else:
            download_bls(RAW_DIR)

    print("Data download complete.")


if __name__ == "__main__":
    main()
