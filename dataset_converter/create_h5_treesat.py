#!/usr/bin/env python3
"""
Convert an ImageFolder-like folder (class subfolders) with TIFF images
into separate train/test HDF5 files, using train_filenames.lst and test_filenames.lst
to assign files to splits.

Assumptions & behavior:
- The .lst files contain file paths relative to the root (e.g. "n01440764/ILSVRC2012_val_00002138.tif"),
  but if they contain only basenames the script will fallback to matching basenames.
- Only files that are found under the class subfolders are processed.
- Writes two HDF5 files (train & test) if the corresponding list file is provided.
- Keeps the same class_to_idx mapping for both outputs.
- Stores images as variable-length uint8 arrays (numpy .npy bytes) identical to your original format.
"""
import argparse
import io
import json
from pathlib import Path
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm
import sys
from typing import List, Tuple, Dict, Set

def read_list_file(list_path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Read list file and return two sets:
    - full_paths: normalized POSIX-style strings (as in file lines)
    - basenames: just the filename component for fallback matching
    """
    full_paths = set()
    basenames = set()
    if not list_path.exists():
        return full_paths, basenames
    with list_path.open("r") as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            # Normalize and store both the posix relative form and basename
            p = Path(s)
            full_paths.add(p.as_posix())
            basenames.add(p.name)
    return full_paths, basenames

def find_all_image_files(root: Path, exts: Tuple[str, ...]=(".tif", ".tiff")):
    """
    Walk class subfolders and collect files with given extensions.
    Returns:
      files: list[Path] sorted
      labels: list[int] (same order)
      class_to_idx: dict
    """
    root = Path(root)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    files = []
    labels = []
    for c in classes:
        class_dir = root / c
        # gather files with extensions (case-insensitive)
        for ext in exts:
            for p in sorted(class_dir.rglob(f"*{ext}")):
                files.append(p)
                labels.append(class_to_idx[c])
    # ensure stable order
    paired = sorted(zip(files, labels), key=lambda x: str(x[0]))
    files, labels = zip(*paired) if paired else ([], [])
    return list(files), list(labels), class_to_idx

def select_split(files: List[Path], labels: List[int],
                 root: Path,
                 split_full_paths: Set[str],
                 split_basenames: Set[str]) -> Tuple[List[Path], List[int]]:
    """
    From all files, select those that belong to the split defined by the lists.
    Matching is attempted in this order:
      1) relative POSIX path (relative to root) matches an entry in split_full_paths
      2) basename matches an entry in split_basenames (fallback)
    """
    selected_files = []
    selected_labels = []
    for p, lab in zip(files, labels):
        try:
            rel = p.relative_to(root)
            rel_posix = rel.as_posix()
        except Exception:
            # if for some reason p is not under root, fall back to name-only
            rel_posix = p.name
        if rel_posix in split_full_paths:
            selected_files.append(p)
            selected_labels.append(lab)
        elif p.name in split_basenames:
            selected_files.append(p)
            selected_labels.append(lab)
        # else: file not in list -> ignored
    return selected_files, selected_labels

def image_to_npy_bytes(path: Path) -> bytes:
    """Open image, convert to RGB numpy array, serialize as .npy bytes and return bytes."""
    with Image.open(path) as img:
        # convert to RGB explicitly (keeps shape H,W,3) and ensures uint8
        arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()

def write_h5(files: List[Path], labels: List[int], class_to_idx: Dict[str,int],
             root: Path, out_path: Path, use_compression: bool):
    """
    Write an h5 file given lists of files and labels.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(files)
    if N == 0:
        print(f"[warning] No files to write for {out_path}. Skipping.", file=sys.stderr)
        return

    vlen_uint8 = h5py.vlen_dtype(np.uint8)
    compression = "lzf" if use_compression else None

    with h5py.File(out_path, "w") as f:
        dset_imgs = f.create_dataset("images", (N,), dtype=vlen_uint8, compression=compression)
        dset_labels = f.create_dataset("labels", data=np.array(labels, dtype=np.int32))

        # store filenames for reference (relative to root)
        rel_fnames = []
        for p in files:
            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = p.name
            rel_fnames.append(rel)
        f.attrs["filenames"] = json.dumps(rel_fnames)
        f.attrs["class_to_idx"] = json.dumps(class_to_idx)

        for i, p in enumerate(tqdm(files, desc=f"Writing {out_path.name}", unit="img")):
            b = image_to_npy_bytes(p)
            arr = np.frombuffer(b, dtype=np.uint8)
            dset_imgs[i] = arr
        f.flush()

    print(f"✅ Wrote {out_path} ({N} images). Compression: {compression}")

def main(args):
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Source root {src} does not exist")

    # Read lists (if provided)
    train_full, train_base = read_list_file(Path(args.train_list)) if args.train_list else (set(), set())
    test_full, test_base = read_list_file(Path(args.test_list)) if args.test_list else (set(), set())

    files, labels, class_to_idx = find_all_image_files(src, exts=tuple(args.ext.split(",")))
    print(f"Found {len(files)} images across {len(class_to_idx)} classes under {src}.")

    # select splits
    train_files, train_labels = select_split(files, labels, src, train_full, train_base)
    test_files, test_labels = select_split(files, labels, src, test_full, test_base)

    # If lists were empty, we will not do automatic splitting, instead we warn and exit (safer). 
    if args.train_list is None and args.test_list is None:
        raise SystemExit("No train/test list provided. Please provide at least --train-list or --test-list.")

    # write outputs
    if args.train_list:
        write_h5(train_files, train_labels, class_to_idx, src, Path(args.out_train), args.compression)
    if args.test_list:
        write_h5(test_files, test_labels, class_to_idx, src, Path(args.out_test), args.compression)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert ImageFolder (tif images) into HDF5 files by split lists")
    ap.add_argument("--src", default="../data/TreeSAT", help="root (class subfolders).")
    ap.add_argument("--train-list", default="../data/TreeSAT/train_filenames.lst", help="Path to train_filenames.lst (relative paths or basenames). Optional.")
    ap.add_argument("--test-list", default="../data/TreeSAT/test_filenames.lst", help="Path to test_filenames.lst (relative paths or basenames). Optional.")
    ap.add_argument("--out-train", default="../data/TreeSAT/TreeSAT_train.h5", help="Output HDF5 path for train split.")
    ap.add_argument("--out-test", default="../data/TreeSAT/TreeSAT_test.h5", help="Output HDF5 path for test split.")
    ap.add_argument("--compression", action="store_true", help="Use LZF compression for the dataset (fast).")
    ap.add_argument("--ext", default=".tif", help="Comma-separated extensions to include (default '.tif,.tiff').")
    args = ap.parse_args()
    main(args)
