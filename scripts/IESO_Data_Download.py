import requests, pathlib, re
from bs4 import BeautifulSoup

BASE = "https://reports-public.ieso.ca/public"
out = pathlib.Path("ieso_raw"); out.mkdir(exist_ok=True)

def fetch_dir(url):
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    return [a["href"] for a in soup.select("a") if re.match(r"PUB_.*\.(csv|xml)$", a["href"])]

# 1. Demand CSVs
for fn in fetch_dir(f"{BASE}/Demand/"):
    r = requests.get(f"{BASE}/Demand/{fn}", timeout=60)
    (out / fn).write_bytes(r.content)

# 2. Generation-by-fuel XMLs
for fn in fetch_dir(f"{BASE}/GenOutputbyFuelHourly/"):
    r = requests.get(f"{BASE}/GenOutputbyFuelHourly/{fn}", timeout=60)
    (out / fn).write_bytes(r.content)

print("Downloaded", len(list(out.iterdir())), "files")
