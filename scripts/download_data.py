from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from careervector.config import (  # noqa: E402
    BLS_NATIONAL_XLSX,
    BLS_NATIONAL_ZIP_URL,
    BLS_PROJECTIONS_URL,
    BLS_PROJECTIONS_XLSX,
    ESCO_DIR,
    NCES_CIP_SOC_URL,
    NCES_CIP_SOC_XLSX,
    ONET_FILES,
    RAW_DIR,
)

HEADERS = {"User-Agent": "CareerVector/0.4 educational-project"}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {destination.name} ...")
    with requests.get(url, stream=True, timeout=180, headers=HEADERS) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def download_bls_wages(destination_dir: Path) -> None:
    print(f"Downloading BLS OEWS archive for {BLS_NATIONAL_XLSX} ...")
    response = requests.get(BLS_NATIONAL_ZIP_URL, timeout=180, headers=HEADERS)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        xlsx_members = [name for name in archive.namelist() if name.lower().endswith(".xlsx")]
        if not xlsx_members:
            raise RuntimeError("BLS archive contained no XLSX file")
        preferred = next((m for m in xlsx_members if Path(m).name == BLS_NATIONAL_XLSX), xlsx_members[0])
        output = destination_dir / BLS_NATIONAL_XLSX
        output.write_bytes(archive.read(preferred))
        print(f"Extracted {output.name}")


def maybe_download(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Keeping existing {destination.name}")
    else:
        download(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--skip-wages", action="store_true", help="Skip BLS OEWS wage archive")
    parser.add_argument("--skip-enrichment", action="store_true", help="Skip NCES CIP-SOC and BLS projections")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in ONET_FILES.items():
        maybe_download(url, RAW_DIR / filename, force=args.force)

    if not args.skip_wages:
        wage_path = RAW_DIR / BLS_NATIONAL_XLSX
        if wage_path.exists() and not args.force:
            print(f"Keeping existing {wage_path.name}")
        else:
            download_bls_wages(RAW_DIR)

    if not args.skip_enrichment:
        maybe_download(NCES_CIP_SOC_URL, RAW_DIR / NCES_CIP_SOC_XLSX, force=args.force)
        maybe_download(BLS_PROJECTIONS_URL, RAW_DIR / BLS_PROJECTIONS_XLSX, force=args.force)

    print("\nData download complete.")
    print(
        "Optional ESCO enrichment: download the official ESCO English CSV classification package "
        "and unzip it under data/raw/esco/. CareerVector will detect it automatically."
    )
    ESCO_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
