#!/usr/bin/env python3
"""
Complete Dataset Download & Organization Script

Downloads and organizes OPFData, PGLib-OPF, and PowerGraph datasets.
Standardized folder structure with no zip packaging.
Uses built-in libraries only.

Author: Auto-generated
Date: 2026-04-03
"""

import json
import shutil
import tarfile
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

# =========================
# Config
# =========================

# 数据根目录
DATA_ROOT = Path("data")

# 子目录
OPFDATA_DIR = DATA_ROOT / "opfdata"
PGLIB_DIR = DATA_ROOT / "pglib"
POWERGRAPH_DIR = DATA_ROOT / "powergraph"
LOGS_DIR = DATA_ROOT / "logs"

# 1) PGLib
PGLIB_CASE = "pglib_opf_case118_ieee.m"
PGLIB_RAW_BASE = "https://raw.githubusercontent.com/power-grid-lib/pglib-opf/master"

# 2) OPFData - 下载完整数据集
OPFDATA_BUCKET_BASE = "https://storage.googleapis.com/gridopt-dataset"

# 3) PowerGraph - 完整下载
POWERGRAPH_ARTICLE_ID = 22820534
POWERGRAPH_MAX_BYTES_PER_FILE = 500 * 1024 * 1024  # 500MB

# Network
TIMEOUT = 120

# =========================
# Helpers
# =========================

def log(msg: str) -> None:
    """Log message with timestamp"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest: Path, max_bytes: Optional[int] = None) -> bool:
    """Download file with progress and size limit"""
    safe_mkdir(dest.parent)
    
    try:
        headers = {'User-Agent': 'power-dataset-builder/4.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            total = 0
            content_length = int(response.headers.get('Content-Length', 0))
            
            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    
                    total += len(chunk)
                    
                    if max_bytes is not None and total > max_bytes:
                        dest.unlink()
                        log(f"    ❌ File too large ({human_size(total)}), skipped")
                        return False
                    
                    f.write(chunk)
                    
                    # Show progress
                    if content_length > 0:
                        pct = (total / content_length) * 100
                        print(f"      └─ {pct:.1f}% ({human_size(total)}/{human_size(content_length)})", 
                              end='\r', flush=True)
        
        print()  # Newline after progress
        return True
    except urllib.error.URLError as e:
        log(f"    ❌ Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False
    except Exception as e:
        log(f"    ❌ Error: {e}")
        if dest.exists():
            dest.unlink()
        return False

def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    i = 0
    while x >= 1024 and i < len(units) - 1:
        x /= 1024
        i += 1
    return f"{x:.2f}{units[i]}"

def write_json(path: Path, obj: dict) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def sanitize_filename(name: str) -> str:
    """Sanitize filename"""
    return name.replace("/", "__").replace("\\", "__")

def maybe_extract_archive(path: Path, dest_dir: Path) -> Optional[Path]:
    """Extract archive if supported"""
    lower = path.name.lower()

    try:
        if lower.endswith(".zip"):
            extract_dir = dest_dir / f"{path.stem}_extracted"
            ensure_dir(extract_dir)
            log(f"    📦 Extracting {path.name}...")
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)
            return extract_dir

        if lower.endswith(".tar.gz"):
            stem = path.name[:-7]
            extract_dir = dest_dir / f"{stem}_extracted"
            ensure_dir(extract_dir)
            log(f"    📦 Extracting {path.name}...")
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(extract_dir)
            return extract_dir

        if lower.endswith(".tgz"):
            stem = path.name[:-4]
            extract_dir = dest_dir / f"{stem}_extracted"
            ensure_dir(extract_dir)
            log(f"    📦 Extracting {path.name}...")
            with tarfile.open(path, "r:gz") as tf:
                tf.extractall(extract_dir)
            return extract_dir
    except Exception as e:
        log(f"    ⚠️  Extract failed: {e}")

    return None

# =========================
# PGLib Download
# =========================

def download_pglib_case(case_name: str) -> Optional[Dict]:
    """Download PGLib case"""
    log(f"📥 Downloading PGLib: {case_name}")
    
    url = f"{PGLIB_RAW_BASE}/{case_name}"
    dest = PGLIB_DIR / case_name

    if dest.exists():
        log(f"  ✅ Already exists: {case_name}")
        return {
            "source": "PGLib-OPF",
            "name": case_name,
            "url": url,
            "path": str(dest.relative_to(DATA_ROOT)),
            "size": dest.stat().st_size,
        }

    if download_file(url, dest):
        log(f"  ✅ Downloaded: {case_name} ({human_size(dest.stat().st_size)})")
        return {
            "source": "PGLib-OPF",
            "name": case_name,
            "url": url,
            "path": str(dest.relative_to(DATA_ROOT)),
            "size": dest.stat().st_size,
        }
    
    return None

# =========================
# OPFData - Manual Selection
# =========================

def download_opfdata_manual() -> List[Dict]:
    """Download selected OPFData files (with manual configuration)"""
    log("📥 Downloading OPFData (Selected Files)")
    
    # 手动指定要下载的关键文件
    manual_files = [
        "dataset_release_1/pglib_opf_case14_ieee_0.tar.gz",
        "dataset_release_1/pglib_opf_case30_ieee_0.tar.gz",
        "dataset_release_1/pglib_opf_case57_ieee_0.tar.gz",
        "dataset_release_1/pglib_opf_case118_ieee_0.tar.gz",
    ]

    results = []
    downloaded = 0
    skipped = 0

    for i, key in enumerate(manual_files, 1):
        url = f"{OPFDATA_BUCKET_BASE}/{key}"
        fname = sanitize_filename(key)
        dest = OPFDATA_DIR / fname

        log(f"  📥 [{i}/{len(manual_files)}] {key}")

        if dest.exists():
            log(f"    ✅ Already exists ({human_size(dest.stat().st_size)})")
            results.append({
                "source": "OPFData",
                "name": key,
                "url": url,
                "path": str(dest.relative_to(DATA_ROOT)),
                "size": dest.stat().st_size,
            })
            downloaded += 1
            continue

        if download_file(url, dest):
            # 尝试解压
            extract_dir = maybe_extract_archive(dest, OPFDATA_DIR)
            
            result = {
                "source": "OPFData",
                "name": key,
                "url": url,
                "path": str(dest.relative_to(DATA_ROOT)),
                "size": dest.stat().st_size if dest.exists() else 0,
            }
            if extract_dir:
                result["extracted"] = str(extract_dir.relative_to(DATA_ROOT))
            
            results.append(result)
            downloaded += 1
        else:
            skipped += 1

    log(f"  📊 OPFData Summary: {downloaded} downloaded, {skipped} skipped")
    return results

# =========================
# PowerGraph Download
# =========================

def get_figshare_article_files(article_id: int) -> List[Dict]:
    """Get all files from Figshare article"""
    log(f"📥 Querying Figshare article {article_id}...")
    
    try:
        url = f"https://api.figshare.com/v2/articles/{article_id}"
        headers = {'User-Agent': 'power-dataset-builder/4.0'}
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        files = data.get("files", [])
        log(f"  📊 Found {len(files)} files")
        return files
    except Exception as e:
        log(f"  ❌ Failed to query Figshare: {e}")
        return []

def download_powergraph_complete() -> List[Dict]:
    """Download PowerGraph dataset"""
    log("📥 Downloading PowerGraph")
    
    files = get_figshare_article_files(POWERGRAPH_ARTICLE_ID)
    if not files:
        return []

    results = []
    downloaded = 0
    skipped = 0

    for i, f in enumerate(files, 1):
        name = f.get("name") or f"file_{f.get('id', 'unknown')}"
        size = int(f.get("size") or 0)
        download_url = f.get("download_url")

        if not download_url:
            log(f"  ⏭️  [{i}/{len(files)}] No download URL: {name}")
            skipped += 1
            continue

        # 文件大小过滤
        if size > POWERGRAPH_MAX_BYTES_PER_FILE:
            log(f"  ⏭️  [{i}/{len(files)}] Too large: {name} ({human_size(size)})")
            skipped += 1
            continue

        dest = POWERGRAPH_DIR / sanitize_filename(name)

        log(f"  📥 [{i}/{len(files)}] {name} ({human_size(size)})")

        if dest.exists():
            log(f"    ✅ Already exists")
            results.append({
                "source": "PowerGraph",
                "name": name,
                "file_id": f.get("id"),
                "url": download_url,
                "path": str(dest.relative_to(DATA_ROOT)),
                "size": dest.stat().st_size,
            })
            downloaded += 1
            continue

        if download_file(download_url, dest, max_bytes=POWERGRAPH_MAX_BYTES_PER_FILE):
            # 尝试解压
            extract_dir = maybe_extract_archive(dest, POWERGRAPH_DIR)
            
            result = {
                "source": "PowerGraph",
                "name": name,
                "file_id": f.get("id"),
                "url": download_url,
                "path": str(dest.relative_to(DATA_ROOT)),
                "size": dest.stat().st_size if dest.exists() else 0,
            }
            if extract_dir:
                result["extracted"] = str(extract_dir.relative_to(DATA_ROOT))
            
            results.append(result)
            downloaded += 1
        else:
            skipped += 1

    log(f"  📊 PowerGraph Summary: {downloaded} downloaded, {skipped} skipped")
    return results

# =========================
# Manifest & Summary
# =========================

def create_manifest(items: List[Dict]) -> Dict:
    """Create manifest of downloaded data"""
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "created_at_unix": int(time.time()),
        "description": "Complete power grid dataset for self-supervised cascading failure learning",
        "datasets": {
            "pglib": [],
            "opfdata": [],
            "powergraph": [],
        },
        "summary": {
            "total_files": 0,
            "total_bytes": 0,
            "by_source": {},
        },
    }

    for item in items:
        source = item["source"].lower()
        
        # 分类统计
        if source == "pglib-opf":
            manifest["datasets"]["pglib"].append(item)
        elif source == "opfdata":
            manifest["datasets"]["opfdata"].append(item)
        elif source == "powergraph":
            manifest["datasets"]["powergraph"].append(item)

        # 总体统计
        manifest["summary"]["total_files"] += 1
        manifest["summary"]["total_bytes"] += item.get("size", 0)
        
        if source not in manifest["summary"]["by_source"]:
            manifest["summary"]["by_source"][source] = {
                "count": 0,
                "bytes": 0,
            }
        manifest["summary"]["by_source"][source]["count"] += 1
        manifest["summary"]["by_source"][source]["bytes"] += item.get("size", 0)

    return manifest

def save_manifest(manifest: Dict) -> None:
    """Save manifest to file"""
    log("📝 Saving manifest...")
    write_json(DATA_ROOT / "MANIFEST.json", manifest)

def print_summary(manifest: Dict) -> None:
    """Print download summary"""
    log("")
    log("=" * 70)
    log("📊 DOWNLOAD SUMMARY")
    log("=" * 70)
    
    summary = manifest["summary"]
    log(f"Total files downloaded: {summary['total_files']}")
    log(f"Total size: {human_size(summary['total_bytes'])}")
    log("")
    log("By source:")
    for source, stats in summary["by_source"].items():
        log(f"  • {source}: {stats['count']} files, {human_size(stats['bytes'])}")
    
    log("")
    log("Folder structure:")
    log(f"  {DATA_ROOT}/")
    log(f"  ├── pglib/          ({len(manifest['datasets']['pglib'])} files)")
    log(f"  ├── opfdata/        ({len(manifest['datasets']['opfdata'])} files)")
    log(f"  ├── powergraph/     ({len(manifest['datasets']['powergraph'])} files)")
    log(f"  └── MANIFEST.json   (metadata)")
    log("=" * 70)

# =========================
# Main
# =========================

def main() -> int:
    """Main download script"""
    ensure_dir(DATA_ROOT)
    ensure_dir(LOGS_DIR)

    log("🚀 Starting dataset download...")
    log(f"   Target directory: {DATA_ROOT.resolve()}")
    log("")

    all_items: List[Dict] = []

    # Download PGLib
    try:
        item = download_pglib_case(PGLIB_CASE)
        if item:
            all_items.append(item)
    except Exception as e:
        log(f"❌ PGLib download failed: {e}")

    log("")

    # Download OPFData
    try:
        items = download_opfdata_manual()
        all_items.extend(items)
    except Exception as e:
        log(f"❌ OPFData download failed: {e}")

    log("")

    # Download PowerGraph
    try:
        items = download_powergraph_complete()
        all_items.extend(items)
    except Exception as e:
        log(f"❌ PowerGraph download failed: {e}")

    log("")

    if not all_items:
        log("❌ Nothing was downloaded!")
        return 1

    # Create and save manifest
    manifest = create_manifest(all_items)
    save_manifest(manifest)
    print_summary(manifest)

    log("")
    log("✅ Download complete!")
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        log("⚠️  Interrupted by user")
        exit_code = 1
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    exit(exit_code)
