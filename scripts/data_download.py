import requests, pathlib, re
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE = "https://reports-public.ieso.ca/public"
out = pathlib.Path("data_raw"); out.mkdir(exist_ok=True)

def fetch_dir(url) -> list[str]:
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    return [
        a["href"] 
        for a in soup.select("a") 
        if re.match(r"PUB_.*\.(csv|xml)$", a["href"])]

# 1. Demand CSVs
demand_files = fetch_dir(f"{BASE}/Demand/")
for fn in tqdm(demand_files, desc="Downloading Ontario/Market Demand Data"):
    r = requests.get(f"{BASE}/Demand/{fn}", timeout=60)
    (out / fn).write_bytes(r.content)

# 2. Generation-by-fuel XMLs
gen_files = fetch_dir(f"{BASE}/GenOutputbyFuelHourly/")
for fn in tqdm(gen_files, desc="Downloading Generation Output Type Data"):
    r = requests.get(f"{BASE}/GenOutputbyFuelHourly/{fn}", timeout=60)
    (out / fn).write_bytes(r.content)

print("Downloaded", len(list(out.iterdir())), "files")
