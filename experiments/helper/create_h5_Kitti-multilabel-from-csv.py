#!/usr/bin/env python3
"""
Convert a flat folder of PNGs + a CSV of multi-labels into a single HDF5 file.

CSV format expected (per-line examples):
  000000.txt,0,0,1,1,0
  000001.png,1,0,0,0,0
The first column is the image basename (with or without extension). The remaining
columns are binary labels for each class (0/1).

Output HDF5 layout (single file):
 - 'images' dataset: variable-length uint8 arrays containing .npy bytes (RGB uint8 arrays).
 - 'labels' dataset: NxC uint8 array with full binary vectors (C = #classes).
 - attributes: 'filenames' (JSON list of basenames), 'label_names' (JSON list)
"""

import argparse
import io
import json
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import sys

DEFAULT_LABEL_NAMES = [
    "non_vulnerable_present",
    "non_vulnerable_nearby",
    "vulnerable_present",
    "vulnerable_nearby",
    "crowded_critical",
]

def parse_csv_labels(csv_path: Path, expected_num_labels: int = None):
    """
    Read CSV and return dict: basename_no_ext -> list[int] (binary label vector).
    Accepts CSV rows where first column is filename or basename, remaining columns are labels.
    Skips blank lines and header-like lines that do not have numeric label columns.
    """
    mapping = {}
    with csv_path.open("r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # strip whitespace and ignore empty cells
            row = [c.strip() for c in row if c is not None and c.strip() != ""]
            if len(row) < 2:
                continue
            name = row[0]
            # If the name contains a path, take the stem
            stem = Path(name).stem
            # parse label columns as ints (ignore rows that can't be parsed)
            try:
                labels = [int(float(x)) for x in row[1:]]
            except Exception:
                # skip header or malformed row
                continue
            if expected_num_labels is not None and len(labels) != expected_num_labels:
                raise ValueError(f"Row for {name} has {len(labels)} labels but expected {expected_num_labels}")
            mapping[stem] = labels
    return mapping

def image_to_npy_bytes(path: Path) -> bytes:
    """Open image, convert to RGB numpy array (H,W,3 uint8), serialize as .npy bytes and return bytes."""
    with Image.open(path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def main():
    ap = argparse.ArgumentParser(description="Convert flat PNG folder + CSV labels -> single HDF5 (images + labels).")
    ap.add_argument("--images-dir", default="../data/KITTI_Distance_Multiclass/train/", help="Directory containing PNG images (flat).")
    ap.add_argument("--labels-csv", default="../data/KITTI_Distance_Multiclass/multilabel_annotations.csv", help="CSV file with per-image multi-labels.")
    ap.add_argument("--out-h5", default="../data/KITTI_Distance_Multiclass/KITTI_Distance_Multiclass.h5",help="Output HDF5 filepath.")
    ap.add_argument("--ext", default=".png", help="Image extension to look for (default '.png').")
    ap.add_argument("--label-names", "-n", default=None,
                    help="Comma-separated label names in order, or path to a JSON file. "
                         f"Default: {','.join(DEFAULT_LABEL_NAMES)}")
    ap.add_argument("--compression", action="store_true", help="Use LZF compression for the images dataset.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    csv_path = Path(args.labels_csv)
    out_h5 = Path(args.out_h5)
    ext = args.ext if args.ext.startswith('.') else '.' + args.ext

    if not images_dir.exists():
        raise SystemExit(f"Images dir {images_dir} does not exist")
    if not csv_path.exists():
        raise SystemExit(f"CSV labels file {csv_path} does not exist")

    # load label names
    if args.label_names:
        ln = args.label_names
        try:
            p = Path(ln)
            if p.exists():
                label_names = json.loads(p.read_text())
            else:
                # parse comma-separated
                label_names = [s.strip() for s in ln.split(",") if s.strip()]
        except Exception:
            raise SystemExit("Failed to parse --label-names (expect JSON file or comma-separated list)")
    else:
        label_names = DEFAULT_LABEL_NAMES.copy()

    num_labels = len(label_names)
    mapping = parse_csv_labels(csv_path, expected_num_labels=num_labels)
    if not mapping:
        raise SystemExit(f"No valid rows parsed from {csv_path}")

    # For each csv entry, try to find the image file
    found_files = []
    multilabels = []
    basenames = []
    missing = []
    for stem, labels in mapping.items():
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            found_files.append(candidate)
            multilabels.append([int(x) for x in labels])
            basenames.append(candidate.name)
        else:
            # If not found, try any extension in the directory (common fallback)
            found_any = None
            for p in images_dir.iterdir():
                if p.is_file() and p.stem == stem:
                    found_any = p
                    break
            if found_any:
                found_files.append(found_any)
                multilabels.append([int(x) for x in labels])
                basenames.append(found_any.name)
            else:
                missing.append(stem)

    if missing:
        print(f"[warning] {len(missing)} CSV entries did not match any image file (skipped). Example missing: {missing[:5]}", file=sys.stderr)

    N = len(found_files)
    if N == 0:
        raise SystemExit("No images found to write. Check images-dir, extension, and CSV basename matching.")

    # Prepare HDF5
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    compression = "lzf" if args.compression else None

    multilabels_arr = np.array(multilabels, dtype=np.uint8)  # NxC

    with h5py.File(out_h5, "w") as f:
        dset_imgs = f.create_dataset("images", (N,), dtype=vlen_uint8, compression=compression)
        dset_labels = f.create_dataset("labels", data=multilabels_arr, dtype=np.uint8)

        # attributes (store JSON lists)
        f.attrs["filenames"] = json.dumps(basenames)
        f.attrs["label_names"] = json.dumps(label_names)
        f.attrs["multilabel_format"] = "binary_matrix"  # informational

        # write images as .npy bytes (so X dataset loader can read them)
        for i, p in enumerate(tqdm(found_files, desc=f"Writing {out_h5.name}", unit="img")):
            b = image_to_npy_bytes(p)
            arr = np.frombuffer(b, dtype=np.uint8)
            dset_imgs[i] = arr

        f.flush()

    print(f"✅ Wrote {out_h5} ({N} images). compression={compression}.")
    if missing:
        print(f"[info] {len(missing)} CSV rows referenced missing images (skipped).")

if __name__ == "__main__":
    main()
