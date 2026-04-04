#!/usr/bin/env python3
"""Validate OPFData + PowerGraph and generate preprocessing quality report."""
from pathlib import Path
import json
import csv
from collections import defaultdict
import numpy as np
from scipy import io as sio
import h5py

DATA_ROOT = Path("data")
OPFDATA_DIR = DATA_ROOT / "opfdata"
POWERGRAPH_MAT_DIR = DATA_ROOT / "powergraph" / "dataset_cascades_extracted" / "dataset_cascades"
REPORT_DIR = DATA_ROOT / "processed" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _sum_len_dict_lists(obj):
    if isinstance(obj, dict):
        total = 0
        for v in obj.values():
            if isinstance(v, list):
                total += len(v)
        return total
    if isinstance(obj, list):
        return len(obj)
    return 0


def validate_opf(limit=None):
    files = list(OPFDATA_DIR.rglob("*.json"))
    if limit:
        files = files[:limit]

    out = {
        "total": len(files),
        "valid": 0,
        "invalid": 0,
        "cases": defaultdict(int),
        "n_nodes": [],
        "n_edges": [],
        "objective": [],
    }

    for f in files:
        try:
            d = json.loads(f.read_text())
            grid = d.get("grid", {})
            nodes = _sum_len_dict_lists(grid.get("nodes", {}))
            edges = _sum_len_dict_lists(grid.get("edges", {}))
            obj = d.get("metadata", {}).get("objective", None)

            case = "unknown"
            for p in f.parts:
                if "case" in p.lower():
                    case = p
                    break

            out["cases"][case] += 1
            out["n_nodes"].append(nodes)
            out["n_edges"].append(edges)
            if obj is not None:
                out["objective"].append(float(obj))
            out["valid"] += 1
        except Exception:
            out["invalid"] += 1

    return out


def _read_mat_any(path: Path):
    try:
        d = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        keys = [k for k in d.keys() if not k.startswith("__")]
        meta = {}
        for k in keys:
            v = d.get(k)
            if hasattr(v, "shape"):
                meta[k] = {"shape": tuple(v.shape), "dtype": str(v.dtype)}
            else:
                meta[k] = {"type": type(v).__name__}
        return meta
    except NotImplementedError:
        meta = {}
        with h5py.File(str(path), "r") as hf:
            for k in hf.keys():
                try:
                    arr = hf[k][()]
                    meta[k] = {"shape": tuple(arr.shape), "dtype": str(arr.dtype)}
                except Exception:
                    meta[k] = {"type": "h5obj"}
        return meta


def validate_powergraph():
    mats = list(POWERGRAPH_MAT_DIR.rglob("*.mat"))
    out = {"total": len(mats), "valid": 0, "invalid": 0, "files": []}

    for f in mats:
        try:
            vars_meta = _read_mat_any(f)
            out["valid"] += 1
            out["files"].append({
                "source": str(f.relative_to(DATA_ROOT)),
                "size_mb": f.stat().st_size / (1024 * 1024),
                "n_vars": len(vars_meta),
                "vars": vars_meta,
            })
        except Exception as e:
            out["invalid"] += 1
            out["files"].append({"source": str(f.relative_to(DATA_ROOT)), "error": type(e).__name__})

    return out


def summarize_stats(vals):
    if not vals:
        return None
    a = np.array(vals)
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
    }


def main():
    opf = validate_opf()
    pg = validate_powergraph()

    # write txt report
    txt = REPORT_DIR / "preprocessing_report.txt"
    with open(txt, "w") as f:
        f.write("PREPROCESSING QUALITY REPORT\n")
        f.write("=" * 72 + "\n\n")

        f.write("OPFDATA\n")
        f.write("-" * 72 + "\n")
        f.write(f"total: {opf['total']}\nvalid: {opf['valid']}\ninvalid: {opf['invalid']}\n")
        f.write(f"cases: {dict(opf['cases'])}\n")
        for k, v in [("n_nodes", opf["n_nodes"]), ("n_edges", opf["n_edges"]), ("objective", opf["objective"])]:
            s = summarize_stats(v)
            if s:
                f.write(f"{k}: {s}\n")

        f.write("\nPOWERGRAPH\n")
        f.write("-" * 72 + "\n")
        f.write(f"total: {pg['total']}\nvalid: {pg['valid']}\ninvalid: {pg['invalid']}\n")
        f.write("sample files:\n")
        for row in pg["files"][:8]:
            f.write(f"  - {row.get('source')} vars={row.get('n_vars', 'NA')} error={row.get('error','')}\n")

    # csv outputs
    with open(REPORT_DIR / "opf_cases.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_name", "count"])
        for c, n in sorted(opf["cases"].items(), key=lambda x: x[1], reverse=True):
            w.writerow([c, n])

    with open(REPORT_DIR / "powergraph_files.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "size_mb", "n_vars", "error"])
        for r in pg["files"]:
            w.writerow([r.get("source"), r.get("size_mb"), r.get("n_vars"), r.get("error")])

    print(f"Wrote {txt}")


if __name__ == "__main__":
    main()
