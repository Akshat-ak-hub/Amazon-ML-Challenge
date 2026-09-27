"""Assemble the final submission zip in the exact required structure.

<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── tests/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Usage:  python package_submission.py --team YOUR_TEAM_NAME
"""
import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUTPUT_SRC = Path(r"D:\CODE SNAP\ml_challenge\output")
CODE_SRC = REPO / "code" / "business_entity_resolution"
DOC_SRC = REPO / "Documentation_template.md"

# Exclude scratch/diagnostic files and caches from the shipped code.
EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}


def should_include(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    name = path.name
    if name.startswith("_"):          # scratch probes/diagnostics
        return False
    if name.endswith((".pyc", ".pyo", ".log")):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True, help="team name for the zip filename")
    ap.add_argument("--out", default=None, help="output zip path")
    args = ap.parse_args()

    zip_path = Path(args.out) if args.out else REPO / f"{args.team}_submission.zip"

    # sanity checks
    missing = []
    for f in ("matching_results.tsv", "candidate_pairs.tsv"):
        if not (OUTPUT_SRC / f).exists():
            missing.append(str(OUTPUT_SRC / f))
    if not DOC_SRC.exists():
        missing.append(str(DOC_SRC))
    if not (CODE_SRC / "README.md").exists():
        missing.append(str(CODE_SRC / "README.md"))
    if not (CODE_SRC / "requirements.txt").exists():
        missing.append(str(CODE_SRC / "requirements.txt"))
    if missing:
        print("ERROR - missing required files:")
        for m in missing:
            print("  ", m)
        return 1

    print(f"Building {zip_path.name} ...")
    n = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        # 1) output/
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            src = OUTPUT_SRC / f
            print(f"  adding output/{f}  ({src.stat().st_size/1024**2:.1f} MB) ...", flush=True)
            z.write(src, f"output/{f}")
            n += 1
        # 2) code/business_entity_resolution/
        for root, dirs, files in os.walk(CODE_SRC):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in files:
                p = Path(root) / fname
                if not should_include(p.relative_to(CODE_SRC)):
                    continue
                rel = p.relative_to(CODE_SRC)
                z.write(p, f"code/business_entity_resolution/{rel.as_posix()}")
                n += 1
        # 3) methodology doc
        z.write(DOC_SRC, "Documentation_template.md")
        n += 1

    size = zip_path.stat().st_size / 1024**2
    print(f"\nDONE: {zip_path}")
    print(f"  {n} files, {size:.1f} MB")
    print("\nContents:")
    with zipfile.ZipFile(zip_path) as z:
        for info in sorted(z.infolist(), key=lambda i: i.filename):
            print(f"  {info.file_size/1024**2:>8.2f} MB  {info.filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
