from __future__ import annotations

import json
import shutil
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
import xml.etree.ElementTree as ET

import requests

# =========================
# Config
# =========================

WORKDIR = Path("power_demo_work")
OUT_ZIP = Path("DEMO.zip")

# 是否清空工作目录
CLEAN_WORKDIR = False

# 1) PGLib
PGLIB_CASE = "pglib_opf_case118_ieee.m"
PGLIB_RAW_BASE = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master"

# 2) OPFData public bucket
OPFDATA_BUCKET_BASE = "https://storage.googleapis.com/gridopt-dataset"

# 自动发现时会尝试的前缀
OPFDATA_PREFIX_HINTS = [
    "",
    "dataset_release_1/",
    "case14",
    "case118",
    "14",
    "118",
    "ieee14",
    "ieee118",
]

# 优先关键词
OPFDATA_PREFERRED_KEYWORDS = [
    "case14",
    "14_ieee",
    "ieee14",
    "case118",
    "118_ieee",
    "ieee118",
    "sample",
    "train",
    "valid",
    "test",
]

# 允许的后缀
OPFDATA_ALLOWED_EXTS = {
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
    ".pt",
    ".pkl",
    ".pickle",
    ".npz",
    ".npy",
    ".mat",
    ".gz",
    ".txt",
    ".md",
}

OPFDATA_DEPRIORITIZE_NAMES = {"readme", "license"}

# demo 下载限制
OPFDATA_MAX_FILES = 1
OPFDATA_MAX_BYTES_PER_FILE = 40 * 1024 * 1024  # 40MB
OPFDATA_DISCOVERY_MAX_PAGES = 20
OPFDATA_DISCOVERY_MAX_PREFIXES = 200

# 最稳妥：直接手动指定对象
OPFDATA_MANUAL_OBJECTS: List[str] = [
    "dataset_release_1/pglib_opf_case14_ieee_0.tar.gz",
]

# 3) PowerGraph figshare
POWERGRAPH_ARTICLE_ID = 22820534
POWERGRAPH_KEYWORDS = ["ieee24", "ieee39", "ieee118", "uk", "metadata", "readme"]
POWERGRAPH_MAX_FILES = 1
POWERGRAPH_MAX_BYTES_PER_FILE = 200 * 1024 * 1024
POWERGRAPH_AUTO_EXTRACT_ARCHIVE = True

# Network
TIMEOUT = 60
HEADERS = {"User-Agent": "power-demo-builder/3.0"}

# =========================
# Helpers
# =========================

def log(msg: str) -> None:
    print(msg, flush=True)

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clean_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def request_json(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def download_file(url: str, dest: Path, max_bytes: Optional[int] = None) -> Path:
    safe_mkdir(dest.parent)
    with requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True) as r:
        r.raise_for_status()
        total = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise RuntimeError(
                        f"Aborted download because file exceeded max_bytes={max_bytes}: {url}"
                    )
                f.write(chunk)
    return dest

def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f} {units[i]}"

def write_json(path: Path, obj: dict) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_text(path: Path, text: str) -> None:
    safe_mkdir(path.parent)
    path.write_text(text, encoding="utf-8")

def sanitize_relname(name: str) -> str:
    return name.replace("/", "__")

def maybe_extract_archive(path: Path, dest_dir: Path) -> Optional[Path]:
    lower = path.name.lower()

    # zip
    if lower.endswith(".zip"):
        extract_dir = dest_dir / f"{path.stem}_extracted"
        ensure_dir(extract_dir)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    # tar.gz
    if lower.endswith(".tar.gz"):
        stem = path.name[:-7]
        extract_dir = dest_dir / f"{stem}_extracted"
        ensure_dir(extract_dir)
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(extract_dir)
        return extract_dir

    # tgz
    if lower.endswith(".tgz"):
        stem = path.name[:-4]
        extract_dir = dest_dir / f"{stem}_extracted"
        ensure_dir(extract_dir)
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(extract_dir)
        return extract_dir

    return None

# =========================
# PGLib
# =========================

def download_pglib_case(root: Path, case_name: str) -> Dict:
    out_dir = root / "pglib"
    safe_mkdir(out_dir)

    url = f"{PGLIB_RAW_BASE}/{quote(case_name)}"
    dest = out_dir / case_name

    if dest.exists():
        log(f"[PGLib] exists, skip: {case_name}")
    else:
        log(f"[PGLib] downloading {case_name}")
        download_file(url, dest)

    return {
        "source": "PGLib-OPF",
        "url": url,
        "saved_as": str(dest.relative_to(root)),
        "bytes": dest.stat().st_size,
    }

# =========================
# OPFData
# =========================

def parse_gcs_list_xml(xml_text: str) -> Tuple[List[Dict], List[str], Optional[str]]:
    root = ET.fromstring(xml_text)

    items = []
    prefixes = []

    for contents in root.findall(".//{*}Contents"):
        key_el = contents.find("{*}Key")
        size_el = contents.find("{*}Size")
        if key_el is None:
            continue
        items.append({
            "name": key_el.text or "",
            "size": int(size_el.text) if size_el is not None and size_el.text else None,
        })

    for cp in root.findall(".//{*}CommonPrefixes"):
        p = cp.find("{*}Prefix")
        if p is not None and p.text:
            prefixes.append(p.text)

    next_marker_el = root.find(".//{*}NextMarker")
    next_marker = next_marker_el.text if next_marker_el is not None and next_marker_el.text else None

    return items, prefixes, next_marker

def list_gcs_once(prefix: str = "", delimiter: Optional[str] = None, marker: Optional[str] = None, max_keys: int = 1000) -> Tuple[List[Dict], List[str], Optional[str]]:
    params = {"prefix": prefix, "max-keys": str(max_keys)}
    if delimiter is not None:
        params["delimiter"] = delimiter
    if marker:
        params["marker"] = marker

    r = requests.get(OPFDATA_BUCKET_BASE, params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return parse_gcs_list_xml(r.text)

def list_gcs_all_objects_for_prefix(prefix: str, max_pages: int = OPFDATA_DISCOVERY_MAX_PAGES) -> List[Dict]:
    all_items: List[Dict] = []
    marker = None
    pages = 0

    while pages < max_pages:
        items, _, next_marker = list_gcs_once(prefix=prefix, delimiter=None, marker=marker, max_keys=1000)
        all_items.extend(items)
        pages += 1
        if not next_marker:
            break
        marker = next_marker

    return all_items

def discover_gcs_prefixes(seed_prefixes: List[str]) -> List[str]:
    seen = set()
    queue = list(seed_prefixes)
    discovered = []

    while queue and len(seen) < OPFDATA_DISCOVERY_MAX_PREFIXES:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        discovered.append(cur)

        try:
            _, child_prefixes, _ = list_gcs_once(prefix=cur, delimiter="/", marker=None, max_keys=1000)
            for cp in child_prefixes:
                if cp not in seen:
                    queue.append(cp)
        except Exception as e:
            log(f"[OPFData] prefix discovery failed for {cur!r}: {e}")

    return discovered

def choose_opfdata_objects() -> List[Dict]:
    # 手动指定优先
    if OPFDATA_MANUAL_OBJECTS:
        selected = [{"name": x, "size": None, "manual": True} for x in OPFDATA_MANUAL_OBJECTS]
        log("[OPFData] using manual objects:")
        for x in selected:
            log(f"  - {x['name']}")
        return selected

    seed_prefixes = list(dict.fromkeys(OPFDATA_PREFIX_HINTS))
    all_prefixes = discover_gcs_prefixes(seed_prefixes)

    log(f"[OPFData] discovered {len(all_prefixes)} prefixes")

    candidates: List[Dict] = []
    for p in all_prefixes:
        try:
            objs = list_gcs_all_objects_for_prefix(p)
            candidates.extend(objs)
        except Exception as e:
            log(f"[OPFData] object listing failed for prefix={p!r}: {e}")

    if not candidates:
        try:
            candidates.extend(list_gcs_all_objects_for_prefix(""))
        except Exception as e:
            raise RuntimeError(f"Could not list OPFData bucket recursively: {e}")

    uniq = {}
    for obj in candidates:
        name = obj["name"]
        if name and name not in uniq:
            uniq[name] = obj

    objs = list(uniq.values())
    log(f"[OPFData] total unique discovered objects: {len(objs)}")

    log("[OPFData] first 80 discovered object names:")
    for x in objs[:80]:
        log(f"  - {x['name']} | size={x.get('size')}")

    def object_score(obj: Dict) -> Tuple[int, int, str]:
        name = obj["name"]
        lower = name.lower()
        size = obj.get("size")
        size_rank = size if size is not None else 10**12

        score = 0

        for kw in OPFDATA_PREFERRED_KEYWORDS:
            if kw in lower:
                score -= 20

        suffix = Path(lower).suffix
        if suffix in {".json", ".jsonl", ".csv", ".parquet", ".pt", ".pkl", ".pickle", ".npz", ".npy", ".mat", ".gz"}:
            score -= 80
        elif suffix in {".txt", ".md"}:
            score += 30

        if "readme" in lower or "license" in lower:
            score += 200

        if size is not None:
            if size <= OPFDATA_MAX_BYTES_PER_FILE:
                score -= 30
            else:
                score += 300

        return (score, size_rank, lower)

    filtered = []
    for obj in objs:
        name = obj["name"]
        lower = name.lower()
        size = obj.get("size")

        if not name or name.endswith("/"):
            continue
        if size is not None and size > OPFDATA_MAX_BYTES_PER_FILE:
            continue
        if "readme" in lower or "license" in lower:
            continue

        filtered.append(obj)

    filtered.sort(key=object_score)

    if not filtered:
        log("[OPFData] strict selection got 0 files, fallback to loose selection")
        for obj in objs:
            name = obj["name"]
            size = obj.get("size")
            if not name or name.endswith("/"):
                continue
            if size is not None and size > OPFDATA_MAX_BYTES_PER_FILE:
                continue
            filtered.append(obj)
        filtered.sort(key=object_score)

    selected = filtered[:OPFDATA_MAX_FILES]

    log("[OPFData] selected objects:")
    for x in selected:
        log(f"  - {x['name']} ({human_size(x['size']) if x.get('size') else 'unknown'})")

    return selected

def download_opfdata_objects(root: Path) -> List[Dict]:
    out_dir = root / "opfdata"
    safe_mkdir(out_dir)

    selected = choose_opfdata_objects()
    if not selected:
        raise RuntimeError(
            "No suitable OPFData files were found. You may need to adjust OPFDATA_PREFIX_HINTS "
            "or set OPFDATA_MANUAL_OBJECTS explicitly."
        )

    results = []
    for obj in selected:
        key = obj["name"]
        url = f"{OPFDATA_BUCKET_BASE}/{quote(key)}"
        fname = sanitize_relname(key)
        dest = out_dir / fname

        if dest.exists():
            log(f"[OPFData] exists, skip: {key}")
        else:
            log(f"[OPFData] downloading {key}")
            try:
                download_file(url, dest, max_bytes=OPFDATA_MAX_BYTES_PER_FILE)
            except Exception as e:
                log(f"[OPFData] skipped {key}: {e}")
                continue

        item = {
            "source": "OPFData",
            "object_name": key,
            "url": url,
            "saved_as": str(dest.relative_to(root)),
            "bytes": dest.stat().st_size if dest.exists() else None,
        }

        try:
            extract_dir = maybe_extract_archive(dest, out_dir)
            if extract_dir:
                item["extracted_to"] = str(extract_dir.relative_to(root))
        except Exception as e:
            log(f"[OPFData] extract failed for {key}: {e}")

        results.append(item)

    if not results:
        raise RuntimeError("All selected OPFData files failed to download.")
    return results

# =========================
# PowerGraph
# =========================

def get_figshare_article_files(article_id: int) -> List[Dict]:
    data = request_json(f"https://api.figshare.com/v2/articles/{article_id}")
    return data.get("files", [])

def choose_powergraph_files(files: List[Dict]) -> List[Dict]:
    ranked = []

    for f in files:
        name = (f.get("name") or "").lower()
        size = int(f.get("size") or 0)
        if size <= 0:
            continue
        if size > POWERGRAPH_MAX_BYTES_PER_FILE:
            continue

        score = 100
        for kw in POWERGRAPH_KEYWORDS:
            if kw in name:
                score -= 30

        if "readme" in name or "meta" in name:
            score -= 20

        if name.endswith((".csv", ".json", ".txt", ".md", ".pkl", ".pt", ".zip", ".7z", ".mat")):
            score -= 5

        ranked.append((score, size, name, f))

    ranked.sort(key=lambda x: (x[0], x[1], x[2]))
    return [x[3] for x in ranked[:POWERGRAPH_MAX_FILES]]

def download_powergraph_files(root: Path) -> List[Dict]:
    out_dir = root / "powergraph"
    safe_mkdir(out_dir)

    files = get_figshare_article_files(POWERGRAPH_ARTICLE_ID)
    if not files:
        raise RuntimeError("No files found in Figshare article response.")

    selected = choose_powergraph_files(files)
    if not selected:
        raise RuntimeError(
            "No suitable PowerGraph files selected. You may need to relax "
            "POWERGRAPH_MAX_BYTES_PER_FILE or change POWERGRAPH_KEYWORDS."
        )

    results = []
    for f in selected:
        name = f.get("name") or f"file_{f.get('id', 'unknown')}"
        download_url = f.get("download_url")
        if not download_url:
            continue

        dest = out_dir / sanitize_relname(name)

        if dest.exists():
            log(f"[PowerGraph] exists, skip: {name}")
        else:
            log(f"[PowerGraph] downloading {name}")
            try:
                download_file(download_url, dest, max_bytes=POWERGRAPH_MAX_BYTES_PER_FILE)
            except Exception as e:
                log(f"[PowerGraph] skipped {name}: {e}")
                continue

        item = {
            "source": "PowerGraph",
            "file_id": f.get("id"),
            "name": name,
            "url": download_url,
            "saved_as": str(dest.relative_to(root)),
            "bytes": dest.stat().st_size if dest.exists() else None,
        }

        if POWERGRAPH_AUTO_EXTRACT_ARCHIVE:
            try:
                extract_dir = maybe_extract_archive(dest, out_dir)
                if extract_dir:
                    item["extracted_to"] = str(extract_dir.relative_to(root))
            except Exception as e:
                log(f"[PowerGraph] extract failed for {name}: {e}")

        results.append(item)

    if not results:
        raise RuntimeError("All selected PowerGraph files failed to download.")
    return results

# =========================
# Demo metadata / zip
# =========================

def build_manifest(root: Path, items: List[Dict]) -> Dict:
    manifest = {
        "created_at_unix": int(time.time()),
        "note": (
            "This is a small demo package assembled from public sources. "
            "It is intended for exploration / Kaggle demo, not full training."
        ),
        "files": items,
        "summary": {
            "total_files": len(items),
            "total_bytes": sum(
                int(x.get("bytes", 0) or 0)
                for x in items
                if x.get("bytes") is not None
            ),
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest

def build_readme(root: Path, manifest: Dict) -> None:
    lines = []
    lines.append("# DEMO package")
    lines.append("")
    lines.append("This package was built automatically from public data sources:")
    lines.append("- PGLib-OPF: grid case template")
    lines.append("- OPFData: small sample of public objects from the public GCS bucket")
    lines.append("- PowerGraph: small sample of public Figshare files")
    lines.append("")
    lines.append("## Included files")
    for f in manifest["files"]:
        extra = ""
        if f.get("extracted_to"):
            extra = f" -> extracted to `{f['extracted_to']}`"
        lines.append(
            f"- {f['source']}: `{f['saved_as']}` ({human_size(int(f.get('bytes', 0) or 0))}){extra}"
        )
    lines.append("")
    lines.append("## Next step suggestion")
    lines.append(
        "Use the PGLib case as topology template, parse the OPFData files into node/edge features, "
        "and use PowerGraph as downstream benchmark/demo."
    )
    write_text(root / "README.md", "\n".join(lines))

def zip_dir(src_dir: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(src_dir)))

# =========================
# Main
# =========================

def main() -> int:
    if CLEAN_WORKDIR:
        clean_dir(WORKDIR)
    else:
        ensure_dir(WORKDIR)

    all_items: List[Dict] = []

    try:
        item = download_pglib_case(WORKDIR, PGLIB_CASE)
        all_items.append(item)
    except Exception as e:
        log(f"[ERROR] PGLib download failed: {e}")

    try:
        items = download_opfdata_objects(WORKDIR)
        all_items.extend(items)
    except Exception as e:
        log(f"[ERROR] OPFData partial download failed: {e}")

    try:
        items = download_powergraph_files(WORKDIR)
        all_items.extend(items)
    except Exception as e:
        log(f"[ERROR] PowerGraph partial download failed: {e}")

    if not all_items:
        log("[FATAL] Nothing was downloaded.")
        return 1

    manifest = build_manifest(WORKDIR, all_items)
    build_readme(WORKDIR, manifest)
    zip_dir(WORKDIR, OUT_ZIP)

    log("")
    log(f"Done. Created: {OUT_ZIP.resolve()}")
    log(f"Total files: {manifest['summary']['total_files']}")
    log(f"Total size : {human_size(manifest['summary']['total_bytes'])}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())