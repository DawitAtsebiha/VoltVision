from __future__ import annotations
from pathlib import Path
from datetime import timedelta
from typing import Dict, List
import pandas as pd
import shutil, sys, tqdm

# ───────────────────────── XML PARSER ──────────────────────────
from lxml import etree as ET          # full XPath support

def parse_xml(path: Path) -> pd.DataFrame:
    """Return tidy hourly DataFrame from one IESO XML."""
    records: List[Dict] = []
    root = ET.parse(path).getroot()

    for daily in root.xpath(".//*[local-name()='DailyData']"):
        day = pd.to_datetime(daily.xpath("./*[local-name()='Day']/text()")[0])

        for hd in daily.xpath("./*[local-name()='HourlyData']"):
            hour = int(hd.xpath("./*[local-name()='Hour']/text()")[0])
            ts   = day + timedelta(hours=hour - 1)   # hour-ending → start

            row = {"Datetime": ts}
            for ft in hd.xpath("./*[local-name()='FuelTotal']"):
                fuel  = ft.xpath("./*[local-name()='Fuel']/text()")[0].title()
                out   = ft.xpath(".//*[local-name()='Output']/text()")
                qual  = ft.xpath(".//*[local-name()='OutputQuality']/text()")
                val   = float(out[0]) if out else 0.0
                if qual and qual[0] == "-1":          # bad data ⇒ 0 MW
                    val = 0.0
                row[fuel] = val

            records.append(row)

    if not records:
        raise ValueError(f"No hourly data in {path.name}")

    return (pd.DataFrame(records)
              .set_index("Datetime")
              .sort_index()
              .asfreq("h")
              .fillna(0.0))

# 3. Master File Function
def build_master(out_dir: Path,
                 pattern: str,
                 out_name: str,
                 header_row: int = 0) -> None:
    """
    Merge all CSVs matching <pattern> in <out_dir> into <out_name>.

    • If header_row == 0   → assume a normal header, parse 'Datetime'.
    • If header_row == 3   → skip 3 metadata lines (Demand files),
                             build Datetime = Date + (Hour-1) h.
    """
    csvs = sorted(out_dir.glob(pattern))
    if not csvs:
        print(f"⚠ No files match {pattern}")
        return

    frames = []
    for p in csvs:
        if header_row == 0:                         # Generation files
            df = pd.read_csv(p,
                             parse_dates=["Datetime"],
                             index_col="Datetime")
        else:                                       # Demand files
            tmp = pd.read_csv(p, header=header_row)
            tmp["Datetime"] = (pd.to_datetime(tmp["Date"]) +
                               pd.to_timedelta(tmp["Hour"] - 1, "h"))
            df = (tmp.drop(columns=["Date", "Hour"])
                     .set_index("Datetime"))
        frames.append(df)

    master = (pd.concat(frames)
                .sort_index()
                .loc[~pd.concat(frames).index.duplicated()])

    out_path = out_dir / out_name
    master.to_csv(out_path, float_format="%.1f")
    print(f"✓ Master file written → {out_path}  ({len(master):,} rows)")

# ───────────────────────── MAIN SCRIPT ─────────────────────────
def main(src_xml: str, out_csv: str):
    src_dir = Path(src_xml).expanduser().resolve()
    out_dir = Path(out_csv).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. convert every XML → CSV into out_dir
    xml_files = sorted(src_dir.glob("PUB_GenOutputbyFuelHourly_*.xml"))
    for xml_path in tqdm.tqdm(xml_files, desc="Converting XML → CSV"):
        df = parse_xml(xml_path)
        dest = out_dir / xml_path.with_suffix(".csv").name
        df.to_csv(dest, float_format="%.1f")

    # 2. move any existing CSVs that were sitting in src_dir
    stray_csvs = [p for p in src_dir.glob("*.csv") if p.parent != out_dir]
    for f in tqdm.tqdm(stray_csvs, desc="Moving stray CSVs"):
        shutil.move(str(f), out_dir / f.name)

    # Generation master – no header skip
    build_master(out_dir,
                pattern="PUB_GenOutputbyFuelHourly_*.csv",
                out_name="GenOutput_ALL.csv",
                header_row=0)

    # Demand master – skip first 3 rows
    build_master(out_dir,
                pattern="PUB_Demand_*.csv",
                out_name="Demand_ALL.csv",
                header_row=3)



# ────────────────────────── CLI ENTRY ──────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python XML_to_CSV_Converter.py <src_xml_dir> <out_csv_dir>")
    main(sys.argv[1], sys.argv[2])
