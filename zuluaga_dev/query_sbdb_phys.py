#!/usr/bin/env python
"""Download JPL SBDB physical properties in batches.

The single-object API (one HTTP request per asteroid) is what got refused
with 403. The query API returns many objects per request. This script pages
that catalog in batches of BATCH objects, then keeps the asteroids listed in
data/mpc_designations.csv (every provid in df_set).

Provids already stored in data/sbdb_physical_properties.csv are not requested
again. The batch API does not include uncertainties or literature references,
so sigma, reference, and note are left empty for newly added rows.
"""

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_FILE = DATA_DIR / "sbdb_physical_properties.csv"
DESIGNATIONS = DATA_DIR / "mpc_designations.csv"

API = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
BATCH = 50000
PAUSE_S = 1.0
RETRIES = 3
USER_AGENT = "medellin-workshop-solarsystem research"

PHYS_FIELDS = [
    "diameter", "albedo", "rot_per", "H", "GM", "density",
    "BV", "UB", "spec_T", "spec_B", "extent", "pole",
]
QUERY_FIELDS = ["pdes", "full_name", "name"] + PHYS_FIELDS
COLUMNS = ["provid", "fullname", "parameter", "value", "sigma", "reference", "note"]
NUMBERED = re.compile(r"^\((\d+)\)")
PARENS = re.compile(r"\(([^)]+)\)")


def format_elapsed(seconds):
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def already_queried():
    """Provids that already have a saved property row. Failures are not reused."""
    if not OUT_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    saved = pd.read_csv(OUT_FILE)
    notes = saved["note"].fillna("").astype(str)
    return saved.loc[~notes.str.startswith("query failed")].copy()


def load_targets():
    """provid lookup from mpc_designations.csv, plus the MPC number when numbered."""
    df = pd.read_csv(DESIGNATIONS)
    df["provid"] = df["provid"].astype(str).str.strip()
    by_provid = {}
    by_number = {}
    for provid, designation in zip(df["provid"], df["mpc_designation"].fillna("").astype(str)):
        by_provid[provid] = designation.strip()
        match = NUMBERED.match(designation.strip())
        if match:
            by_number[match.group(1)] = provid
    return by_provid, by_number


def match_provids(pdes, full_name, by_provid, by_number):
    found = []
    pdes = "" if pdes is None else str(pdes).strip()
    if pdes in by_provid:
        found.append(pdes)
    if pdes.isdigit() and pdes in by_number:
        found.append(by_number[pdes])
    for token in PARENS.findall(full_name or ""):
        token = token.strip()
        if token in by_provid:
            found.append(token)
    return list(dict.fromkeys(found))


def fetch_batch(limit_from):
    params = urllib.parse.urlencode({
        "fields": ",".join(QUERY_FIELDS),
        "sb-kind": "a",
        "limit": BATCH,
        "limit-from": limit_from,
    })
    request = urllib.request.Request(API + "?" + params, headers={"User-Agent": USER_AGENT})
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            time.sleep(5 * attempt)
    raise RuntimeError(f"SBDB batch at {limit_from} failed: {last_error}")


def rows_from_batch(payload, by_provid, by_number, done):
    fields = payload["fields"]
    index = {name: i for i, name in enumerate(fields)}
    rows = []
    for record in payload["data"]:
        full_name = record[index["full_name"]] or ""
        fullname = " ".join(str(full_name).split())
        provids = match_provids(record[index["pdes"]], full_name, by_provid, by_number)
        for provid in provids:
            if provid in done:
                continue
            for field in PHYS_FIELDS:
                value = record[index[field]]
                if value in (None, ""):
                    continue
                rows.append({
                    "provid": provid,
                    "fullname": fullname,
                    "parameter": field,
                    "value": value,
                    "sigma": "",
                    "reference": "",
                    "note": "",
                })
    return rows


def main():
    kept = already_queried()
    done = set(kept["provid"].dropna().astype(str).str.strip())
    by_provid, by_number = load_targets()
    pending = [provid for provid in by_provid if provid not in done]
    print(f"{len(by_provid)} provids in mpc_designations.csv")
    print(f"{len(done)} already saved, {len(pending)} still missing")
    if not pending:
        print(OUT_FILE)
        return

    t_start = time.perf_counter()
    new_rows = []
    limit_from = 0
    total = None
    while total is None or limit_from < total:
        payload = fetch_batch(limit_from)
        total = int(payload["count"])
        got = len(payload["data"])
        if got == 0:
            break
        new_rows.extend(rows_from_batch(payload, by_provid, by_number, done))
        limit_from += got
        elapsed = time.perf_counter() - t_start
        print(
            f"batch {limit_from}/{total} | "
            f"matched rows so far {len(new_rows)} | "
            f"elapsed {format_elapsed(elapsed)}"
        )
        if limit_from < total:
            time.sleep(PAUSE_S)

    added = pd.DataFrame(new_rows, columns=COLUMNS)
    # A provid can be reached by its number and by its provisional designation.
    added = added.drop_duplicates(["provid", "parameter"])
    combined = pd.concat([kept, added], ignore_index=True)
    tmp = OUT_FILE.with_suffix(".csv.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(OUT_FILE)
    n_provid = combined["provid"].nunique()
    print(f"Wrote {len(combined)} rows for {n_provid} asteroids to {OUT_FILE}")
    print(f"Added {added['provid'].nunique()} asteroids in {format_elapsed(time.perf_counter() - t_start)}")


if __name__ == "__main__":
    main()
