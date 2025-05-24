#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from datetime import timedelta
from typing import Dict, List
import pandas as pd
import shutil
import sys
import argparse
import logging
import tqdm
from lxml import etree as ET

# Configure logging
tqdm.tqdm.pandas()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_xml(path: Path) -> pd.DataFrame:
    """Return tidy hourly DataFrame from one IESO XML."""
    records: List[Dict] = []
    root = ET.parse(path).getroot()

    for daily in root.xpath(".//*[local-name()='DailyData']"):
        day = pd.to_datetime(daily.xpath("./*[local-name()='Day']/text()")[0])

        for hd in daily.xpath("./*[local-name()='HourlyData']"):
            hour = int(hd.xpath("./*[local-name()='Hour']/text()")[0])
            ts = day + timedelta(hours=hour - 1)

            row: Dict[str, float] = {"Datetime": ts}
            for ft in hd.xpath("./*[local-name()='FuelTotal']"):
                fuel = ft.xpath("./*[local-name()='Fuel']/text()")[0].title()
                out = ft.xpath(".//*[local-name()='Output']/text()")
                qual = ft.xpath(".//*[local-name()='OutputQuality']/text()")
                val = float(out[0]) if out else 0.0
                if qual and qual[0] == "-1":
                    val = 0.0
                row[fuel] = val

            records.append(row)

    if not records:
        raise ValueError(f"No hourly data in {path.name}")

    df = pd.DataFrame(records)
    df = (
        df.set_index("Datetime")
          .sort_index()
          .asfreq("h")
          .fillna(0.0)
    )
    return df


def build_master(
    out_dir: Path,
    pattern: str,
    out_name: str,
    header_row: int = 0
) -> None:
    """
    Merge all CSVs matching <pattern> in <out_dir> into <out_name>.

    • If header_row == 0   → assume a normal header, parse 'Datetime'.
    • If header_row == 3   → skip 3 metadata lines (Demand files),
                             build Datetime = Date + (Hour-1) h.
    """
    csvs = sorted(out_dir.glob(pattern))
    if not csvs:
        logger.warning(f"No files match pattern {pattern} in {out_dir}")
        return

    frames = []
    for p in csvs:
        if header_row == 0:
            df = pd.read_csv(
                p,
                parse_dates=["Datetime"],
                index_col="Datetime"
            )
        else:
            tmp = pd.read_csv(p, header=header_row)
            tmp["Datetime"] = (
                pd.to_datetime(tmp["Date"]) +
                pd.to_timedelta(tmp["Hour"] - 1, "h")
            )
            df = (
                tmp.drop(columns=["Date", "Hour"] )
                   .set_index("Datetime")
            )
        frames.append(df)

    combined = pd.concat(frames)
    master = (
        combined.sort_index()
                .loc[~combined.index.duplicated()]
    )

    out_path = out_dir / out_name
    master.to_csv(out_path, float_format="%.1f")
    logger.info(f"Master file written: {out_path} ({len(master):,} rows)")


def main(src_xml: str, out_csv: str) -> None:
    src_dir = Path(src_xml).expanduser().resolve()
    out_dir = Path(out_csv).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert XML → CSV
    xml_files = sorted(src_dir.glob("PUB_GenOutputbyFuelHourly_*.xml"))
    for xml_path in tqdm.tqdm(xml_files, desc="Converting XML → CSV"):
        try:
            df = parse_xml(xml_path)
            dest = out_dir / xml_path.with_suffix('.csv').name
            df.to_csv(dest, float_format="%.1f")
        except Exception as e:
            logger.error(f"Failed to parse {xml_path.name}: {e}")

    # 2. Move stray CSVs
    stray_csvs = [p for p in src_dir.glob("*.csv") if p.parent != out_dir]
    for f in tqdm.tqdm(stray_csvs, desc="Moving stray CSVs"):
        shutil.move(str(f), out_dir / f.name)

    # 3. Build masters
    build_master(
        out_dir,
        pattern="PUB_GenOutputbyFuelHourly_*.csv",
        out_name="GenOutput_ALL.csv",
        header_row=0
    )
    build_master(
        out_dir,
        pattern="PUB_Demand_*.csv",
        out_name="Demand_ALL.csv",
        header_row=3
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert IESO XML to per-fuel CSVs and merge into master files"
    )
    parser.add_argument(
        "src_xml",
        help="Directory containing IESO XML files and stray CSVs"
    )
    parser.add_argument(
        "out_csv",
        help="Directory to write output CSVs and master files"
    )
    args = parser.parse_args()

    main(args.src_xml, args.out_csv)

    # Cleanup non-master CSV files
    out_dir = Path(args.out_csv).expanduser().resolve()
    master_files = {"GenOutput_ALL.csv", "Demand_ALL.csv"}
    all_csvs = sorted(out_dir.glob("*.csv"))
    to_delete = [p for p in all_csvs if p.name not in master_files]
    if to_delete:
        print("\nCleanup: the following non-master CSV files were generated:")
        for p in to_delete:
            print(f"  - {p.name}")
        ans = input("Do you want to delete these files? [y/N]: ").strip().lower()
        if ans == 'y':
            for p in to_delete:
                try:
                    p.unlink()
                    logger.info(f"Deleted {p.name}")
                except Exception as e:
                    logger.error(f"Failed to delete {p.name}: {e}")
            print("Non-master CSV files deleted.")
        else:
            print("Non-master CSV files retained.")
    else:
        print("No non-master CSV files found to clean up.")

    # Cleanup raw XMLs
    raw_dir = Path(args.src_xml).expanduser().resolve()
    raw_files = list(raw_dir.glob("*") )
    if raw_files:
        print("\nRaw folder contains the following files:")
        for p in raw_files:
            print(f"  - {p.name}")
        ans2 = input("Do you want to delete ALL files in the raw folder? [y/N]: ").strip().lower()
        if ans2 == 'y':
            for p in raw_files:
                try:
                    if p.is_file():
                        p.unlink()
                    else:
                        shutil.rmtree(p)
                    logger.info(f"Deleted raw file/folder {p.name}")
                except Exception as e:
                    logger.error(f"Failed to delete {p.name}: {e}")
            print("All raw files deleted.")
        else:
            print("Raw folder contents retained.")
    else:
        print("Raw folder is already empty.")