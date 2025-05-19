from lxml import etree as ET
import pandas as pd
from datetime import timedelta
from pathlib import Path

def parse_xml(path: Path) -> pd.DataFrame:
    records = []
    root = ET.parse(path).getroot()

    for daily in root.xpath(".//*[local-name()='DailyData']"):
        day_date = pd.to_datetime(daily.xpath("./*[local-name()='Day']/text()")[0])

        for hd in daily.xpath("./*[local-name()='HourlyData']"):
            hour_num = int(hd.xpath("./*[local-name()='Hour']/text()")[0])
            ts = day_date + timedelta(hours=hour_num - 1)

            row = {"Datetime": ts}
            for ft in hd.xpath("./*[local-name()='FuelTotal']"):
                fuel = ft.xpath("./*[local-name()='Fuel']/text()")[0].title()

                out_nodes = ft.xpath(".//*[local-name()='Output']/text()")
                if not out_nodes:
                    row[fuel] = 0.0
                    continue

                val = float(out_nodes[0])
                qflag = ft.xpath(".//*[local-name()='OutputQuality']/text()")
                if qflag and qflag[0] == "-1":
                    val = 0.0        # bad quality ⇒ treat as 0 MW
                row[fuel] = val
            records.append(row)

    if not records:
        raise ValueError(f"No hourly records in {path.name}")

    return (pd.DataFrame(records)
              .set_index("Datetime")
              .sort_index()
              .asfreq("H")
              .fillna(0.0))          # fill missing fuels with 0 MW


if __name__ == "__main__":
    import sys, pathlib, tqdm, os

    if len(sys.argv) != 3:
        sys.exit("Usage: python XML_to_CSV_Converter.py <src_xml_dir> <out_csv_dir>")

    src = pathlib.Path(sys.argv[1]).expanduser().resolve()
    out = pathlib.Path(sys.argv[2]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(src.glob("PUB_GenOutputbyFuelHourly_*.xml"))
    if not xml_files:
        sys.exit(f"No XML files found in {src}")

    for xml_path in tqdm.tqdm(xml_files, desc="Converting"):
        df = parse_xml(xml_path)
        csv_path = out / xml_path.with_suffix(".csv").name
        df.to_csv(csv_path, float_format="%.1f")