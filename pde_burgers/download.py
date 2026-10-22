"""Download only 1D_Burgers_Sols_Nu0.01.hdf5 from DaRUS.

Follows PDEBench's official download path (pdebench/data_download): the URL and MD5 are
taken from PDEBench's `pdebench_data_urls.csv`; the values below are a fallback copy of
that row in case GitHub is unreachable from the server.
"""
import argparse
import csv
import hashlib
import io
import os
import urllib.request

FILENAME = "1D_Burgers_Sols_Nu0.01.hdf5"
CSV_URL = "https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_download/pdebench_data_urls.csv"
FALLBACK = {"URL": "https://darus.uni-stuttgart.de/api/access/datafile/281363",
            "Path": "1D/Burgers/Train/", "MD5": "e6d9a4f62baf9a29121a816b919e2770"}


def lookup_row():
    try:
        with urllib.request.urlopen(CSV_URL, timeout=30) as r:
            rows = list(csv.DictReader(io.StringIO(r.read().decode())))
        for row in rows:
            if row["Filename"] == FILENAME:
                return row
        print(f"[warn] {FILENAME} not in PDEBench CSV; using fallback")
    except Exception as e:
        print(f"[warn] could not fetch PDEBench CSV ({e}); using fallback")
    return FALLBACK


def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_folder", default="data/pdebench")
    p.add_argument("--skip_md5", action="store_true")
    args = p.parse_args()

    row = lookup_row()
    out_dir = os.path.join(args.root_folder, row["Path"])
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, FILENAME)
    print(f"URL : {row['URL']}\nMD5 : {row['MD5']}\nDest: {out}")

    if not os.path.exists(out):
        tmp = out + ".part"
        # wget -c resumes partial downloads of this ~8 GB file
        rc = os.system(f"wget -c -O '{tmp}' '{row['URL']}'")
        if rc != 0:
            raise SystemExit("download failed (re-run to resume)")
        os.replace(tmp, out)
    if not args.skip_md5:
        got = md5sum(out)
        print(f"md5 {got} ({'OK' if got == row['MD5'] else 'MISMATCH'})")
        if got != row["MD5"]:
            raise SystemExit("MD5 mismatch: delete the file and re-download")
    print(out)


if __name__ == "__main__":
    main()
