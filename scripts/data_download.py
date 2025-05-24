import tkinter.messagebox
import requests, pathlib, re, tkinter
from bs4 import BeautifulSoup
from tkinter.scrolledtext import ScrolledText

BASE = "https://reports-public.ieso.ca/public"
out = pathlib.Path("data_raw"); out.mkdir(exist_ok=True)

def fetch_dir(url) -> list[str]:
    soup = BeautifulSoup(requests.get(url).text, "html.parser")
    return [
        a["href"] 
        for a in soup.select("a") 
        if re.match(r"PUB_.*\.(csv|xml)$", a["href"])]

class ProgressPopup:
    def __init__(self, title: str, total: int):
        self.root = tkinter.Tk()
        self.root.title(title)
        self.root.geometry("400x200")
        self.text = ScrolledText(self.root, state='disabled')
        self.text.pack(expand=True, fill='both', padx=20, pady=20)
        self.count = 0
        self.total = total
    
    def update(self, filename: str):
        self.count += 1
        self.text.config(state='normal')
        self.text.insert('end', f"[{self.count}/{self.total}] {filename}\n")
        self.text.see('end')
        self.text.config(state='disabled')
        self.root.update()

    def close(self):
        self.root.destroy()

def download_phase(phase_name: str, url_dir: str):
    files = fetch_dir(f"{BASE}/{url_dir}/")
    popup = ProgressPopup(f"Downloading {phase_name}", total=len(files))
    
    for fn in files:
        r = requests.get(f"{BASE}/{url_dir}/{fn}", timeout=60)
        (out / fn).write_bytes(r.content)
        popup.update(fn)

    popup.close()

def main():
    download_phase("Ontario/Market Demand Data", "Demand")
    download_phase("Generation Output by Type Data", "GenOutputbyFuelHourly")

    summary_root = tkinter.Tk()
    summary_root.withdraw()
    total = len(list(out.iterdir()))
    tkinter.messagebox.showinfo(
        "Finished!",
        f"Downloaded {total} files into:\n{out.resolve()}"
    )

    summary_root.destroy()

if __name__ == "__main__":
    main()