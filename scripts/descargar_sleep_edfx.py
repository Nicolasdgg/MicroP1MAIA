"""
Descarga los registros de Sleep-EDF Database Expanded v1.0.0 (PhysioNet) que falten en data/raw.

- La lista de archivos y sus hashes salen de data/raw/SHA256SUMS.txt (publicado por PhysioNet).
- Solo descarga los EDF que no existen o cuyo SHA256 no coincide; es idempotente y se puede re-ejecutar.
- Cada archivo se descarga a un .part y se renombra solo si el SHA256 coincide (nunca deja archivos corruptos).
- Al final imprime cuántos archivos hay, cuántos se descargaron y cuántos fallaron.

Uso: python scripts/descargar_sleep_edfx.py [--workers 4]
"""

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(rel: str, expected: str, retries: int = 4) -> str:
    dest = RAW / rel
    if dest.exists() and sha256(dest) == expected:
        return "ok"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(BASE_URL + rel, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if sha256(tmp) != expected:
                raise ValueError("SHA256 no coincide")
            tmp.replace(dest)
            return "descargado"
        except Exception as e:  # red inestable: se reintenta con espera creciente
            if attempt == retries:
                tmp.unlink(missing_ok=True)
                return f"error: {e}"
            time.sleep(5 * attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    entries = []
    for line in (RAW / "SHA256SUMS.txt").read_text().splitlines():
        h, rel = line.split(maxsplit=1)
        if rel.endswith(".edf"):
            entries.append((rel, h))
    print(f"[descarga] {len(entries)} archivos EDF en la lista oficial")

    t0 = time.time()
    status = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, rel, h): rel for rel, h in entries}
        for i, fut in enumerate(as_completed(futures), 1):
            status[futures[fut]] = fut.result()
            if i % 20 == 0 or i == len(entries):
                print(f"[descarga] {i}/{len(entries)} procesados · {time.time() - t0:.0f} s", flush=True)

    errors = {k: v for k, v in status.items() if v.startswith("error")}
    print(f"[descarga] ya existían: {sum(v == 'ok' for v in status.values())} · "
          f"descargados: {sum(v == 'descargado' for v in status.values())} · errores: {len(errors)} · "
          f"tiempo: {(time.time() - t0) / 60:.1f} min")
    for k, v in errors.items():
        print(f"  {k}: {v}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
