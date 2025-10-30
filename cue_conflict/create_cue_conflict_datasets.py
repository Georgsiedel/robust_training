"""
Creates a reduced copy of a dataset based on a mapping file.

Behavior:
- Reads mapping file (default: C10-mapping.txt located in dataset folder or given by --mapping-file).
  Lines are expected like:
      airplane=['airplane']
      truck=['truck']
  The script extracts the text before '=' (e.g. "airplane", "truck") -> allowed classes.
- In the dataset main folder, iterate top-level subfolders.
  - If a top-level subfolder name is NOT in allowed classes, it will NOT be copied.
  - If a top-level subfolder IS allowed, create that subfolder inside the output dataset and copy
    only image files whose "class2" (the part after the last '-') is in the allowed set.
    - class2 is obtained from the filename stem after the last '-', then trailing digits are removed.
    - Example: cat12-dog3.png -> class2 = "dog"
- By default only files directly inside each top-level class folder are considered; use --recursive to
  process files nested in subdirectories while preserving relative subpaths.
- Non-image files are ignored.
- Use --dry-run to preview actions without copying.

Usage:
    python make_reduced_dataset.py path/to/dataset \
        --mapping-file C10-mapping.txt \
        --output-name dataset_reduced \
        [--dry-run] [--overwrite] [--recursive] [--verbose]
"""
from pathlib import Path
import argparse
import re
import shutil
import sys

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'}

def parse_mapping_file(mapping_path: Path):
    """
    Parse mapping file and return a set of class names appearing before '='.
    Accepts lines like: airplane=['airplane'] or truck=['truck']
    """
    allowed = set()
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    pattern = re.compile(r'^([^=]+)=')
    with mapping_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            m = pattern.match(line)
            if m:
                left = m.group(1).strip()
                if left:
                    allowed.add(left)
    return allowed

def extract_class2_from_filename(filename: str):
    """
    Given a filename like "class1x-class2y.png", extract the second class (after last '-'),
    stripping trailing digits. Returns None if no '-' present or result empty.
    Examples:
        "cat12-dog3.png" -> "dog"
        "automobile1-bird45.png" -> "bird"
        "nohyphen.png" -> None
    """
    name_no_ext = Path(filename).stem
    if '-' not in name_no_ext:
        return None
    class2_raw = name_no_ext.rsplit('-', 1)[-1]
    # strip trailing digits
    class2 = re.sub(r'\d+$', '', class2_raw)
    class2 = class2.strip()
    return class2 if class2 else None

def is_image_file(path: Path):
    return path.suffix.lower() in IMAGE_EXTS

def copy_allowed_files(src_dir: Path, dst_dir: Path, allowed: set, dry_run: bool, recursive: bool, verbose: bool):
    """
    Copy image files from src_dir into dst_dir according to rules:
    - only copy images whose class2 is in allowed
    - if recursive: walk recursively and preserve relative subpaths
    - if not recursive: only consider files directly in src_dir
    """
    if recursive:
        # iterate recursively
        for p in src_dir.rglob('*'):
            if p.is_file() and is_image_file(p):
                rel = p.relative_to(src_dir)
                class2 = extract_class2_from_filename(p.name)
                if class2 is None:
                    if verbose:
                        print(f"Skipping (no class2): {p}")
                    continue
                if class2 not in allowed:
                    if verbose:
                        print(f"Skipping (class2 not allowed '{class2}'): {p}")
                    continue
                dst_path = dst_dir / rel
                if dry_run:
                    print(f"[DRY-RUN] Would copy: {p} -> {dst_path}")
                else:
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst_path)
                    if verbose:
                        print(f"Copied: {p} -> {dst_path}")
    else:
        # non-recursive: only files directly under src_dir
        for p in sorted(src_dir.iterdir()):
            if p.is_file() and is_image_file(p):
                class2 = extract_class2_from_filename(p.name)
                if class2 is None:
                    if verbose:
                        print(f"Skipping (no class2): {p}")
                    continue
                if class2 not in allowed:
                    if verbose:
                        print(f"Skipping (class2 not allowed '{class2}'): {p}")
                    continue
                dst_path = dst_dir / p.name
                if dry_run:
                    print(f"[DRY-RUN] Would copy: {p} -> {dst_path}")
                else:
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst_path)
                    if verbose:
                        print(f"Copied: {p} -> {dst_path}")
            else:
                if verbose and p.is_file():
                    # non-image file
                    print(f"Ignoring non-image file: {p}")
                # skip directories (unless recursive is True)

def main():
    parser = argparse.ArgumentParser(description="Make reduced copy of dataset based on mapping file.")
    parser.add_argument('--dataset_dir', default='../data/cue-conflict/cue-conflict', help='Path to dataset main folder (relative or absolute)')
    parser.add_argument('--mapping-file', default='./cue_conflict/TIN-mapping.txt',
                        help='Mapping file. If relative, first looked up as given, then inside dataset_dir.')
    parser.add_argument('--output-name', default='cue-conflict-TIN',
                        help='Name for the output folder (created next to dataset_dir). Default: <datasetname>_reduced')
    parser.add_argument('--dry-run', action='store_true', help='Print what would be done without copying files.')
    parser.add_argument('--overwrite', action='store_true', help='If output folder exists, remove it first.')
    parser.add_argument('--recursive', action='store_true', help='Process files recursively inside allowed class folders (preserve subpaths).')
    parser.add_argument('--verbose', action='store_true', help='Print more information.')
    args = parser.parse_args()

    dataset = Path(args.dataset_dir).resolve()
    if not dataset.exists() or not dataset.is_dir():
        print(f"ERROR: dataset directory does not exist or is not a directory: {dataset}")
        sys.exit(1)

    mapping_path = Path(args.mapping_file)
    if not mapping_path.exists():
        candidate = dataset / args.mapping_file
        if candidate.exists():
            mapping_path = candidate

    try:
        allowed = parse_mapping_file(mapping_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    if not allowed:
        print("WARNING: No class names parsed from mapping file. Nothing will be copied.")
    if args.verbose:
        print(f"Allowed classes: {sorted(allowed)}")
        print(f"Dataset folder: {dataset}")
        print(f"Mapping file used: {mapping_path}")

    # set output folder next to dataset folder
    parent = dataset.parent
    default_outname = dataset.name + '_reduced'
    outname = args.output_name if args.output_name else default_outname
    output_dir = parent / outname

    if output_dir.exists():
        if args.overwrite:
            if args.dry_run:
                print(f"[DRY-RUN] Would remove existing output folder: {output_dir}")
            else:
                if args.verbose:
                    print(f"Removing existing output folder: {output_dir}")
                shutil.rmtree(output_dir)
        else:
            print(f"ERROR: output folder already exists: {output_dir}")
            print("Either remove it manually or run with --overwrite to replace it.")
            sys.exit(1)

    if args.dry_run:
        print(f"[DRY-RUN] Output folder would be: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"Created output folder: {output_dir}")

    # iterate top-level entries in dataset
    for entry in sorted(dataset.iterdir()):
        if entry.is_dir():
            dir_name = entry.name
            if dir_name not in allowed:
                if args.verbose or args.dry_run:
                    print(f"Skipping top-level folder (not allowed): {entry}")
                continue
            # allowed top-level folder -> copy allowed files
            dst_subdir = output_dir / dir_name
            if args.dry_run:
                print(f"[DRY-RUN] Would create folder: {dst_subdir}")
            else:
                dst_subdir.mkdir(parents=True, exist_ok=True)
            copy_allowed_files(entry, dst_subdir, allowed, args.dry_run, args.recursive, args.verbose)
        else:
            # top-level files: ignore
            if args.verbose:
                print(f"Ignoring top-level file: {entry}")

    print("Done.")
    if args.dry_run:
        print("No files were copied because --dry-run was used.")

if __name__ == '__main__':
    main()
