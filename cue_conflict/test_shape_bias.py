#!/usr/bin/env python3
"""
Evaluate reduced cue-conflict dataset with WideResNet model (batchwise evaluation).

This is the same script as before but the evaluation now iterates over the DataLoader
in batches (controlled by --batch-size) instead of assuming the whole dataset fits
in a single batch.
"""
from pathlib import Path
import argparse
import sys
import os
import ast
import re
import warnings
import importlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

# -------------------------
# Helpers
# -------------------------
_digits_re = re.compile(r'\d+$')

def resolve_dataset_path(path_str: str, relative_to_script: bool) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        if relative_to_script:
            script_dir = Path(__file__).resolve().parent
            p = (script_dir / p).resolve()
        else:
            p = p.resolve()
    return p

def extract_shape_texture_from_filename(filename: str):
    """
    filename: 'airplane2-bird3.png' -> returns ('airplane','bird')
    returns (shape, texture) or (None, None) if not parsable
    """
    stem = Path(filename).stem
    if '-' not in stem:
        return (None, None)
    left, right = stem.rsplit('-', 1)
    shape = _digits_re.sub('', left).strip()
    texture = _digits_re.sub('', right).strip()
    if not shape or not texture:
        return (None, None)
    return (shape, texture)

def parse_mapping_file(mapping_path: Path):
    """
    Parse mapping like:
      keyboard=['computer keyboard']
      elephant=['elephant']
    Returns dict: left -> [list of right-side strings]
    """
    mapping = {}
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    with mapping_path.open('r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                left = line.strip()
                mapping[left] = []
                continue
            left, right = line.split('=', 1)
            left = left.strip()
            right = right.strip()
            # try literal eval
            try:
                parsed = ast.literal_eval(right)
                if isinstance(parsed, (list, tuple)):
                    synonyms = [str(x).strip() for x in parsed if x]
                else:
                    synonyms = [str(parsed).strip()]
            except Exception:
                # fallback: remove brackets and split
                r = right.strip()
                if r.startswith('[') and r.endswith(']'):
                    inner = r[1:-1].strip()
                    if inner:
                        parts = [p.strip().strip("'\"") for p in inner.split(',') if p.strip()]
                        synonyms = parts
                    else:
                        synonyms = []
                else:
                    token = r.strip().strip("'\"")
                    synonyms = [token] if token else []
            mapping[left] = synonyms
    return mapping

# -------------------------
# Model import & loading
# -------------------------
def setup_paths(base_path: str):
    """
    Dynamically adjust sys.path to include the repo's base and experiments folder.

    Use this to ensure imports like 'experiments.models.WideResNet' resolve correctly.
    We prepend the paths to sys.path to prioritize the repo.
    """
    if not base_path:
        return
    base = Path(base_path).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Provided repo-root does not exist: {base}")
    base_str = str(base)
    exp_str = os.path.join(base_str, 'experiments')
    # Prepend (prefer repo paths); avoid duplicates
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
    if exp_str not in sys.path:
        sys.path.insert(0, exp_str)

def load_wideresnet_checkpoint(checkpoint_path: Path, model, device):
    """
    Tries multiple common checkpoint dict formats.
    """
    # load checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    model = model.to(device) #torch.nn.DataParallel(

    state_dict = torch.load(str(checkpoint_path), weights_only=False, map_location=device)['model_state_dict']
    
    # Remove "module." prefix from keys
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # Remove "module." prefix
        new_state_dict[new_key] = v

    # Load the modified state_dict into the original model
    model.load_state_dict(new_state_dict, strict=False)

    model.eval()
    
    return model

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate reduced cue-conflict dataset with WideResNet model.")
    parser.add_argument('--dataset_dir', default="../data/cue-conflict/cue-conflict-TIN", help='Path to reduced dataset (ImageFolder).')
    parser.add_argument('--image-size-factor', type=int, choices=[1,2], default=2, help='Resize images to this size.')
    parser.add_argument('--relative-to-script', action='store_true', help='Resolve dataset path relative to script.')
    parser.add_argument('--original-dataset', default='TinyImageNet', choices=['CIFAR100','CIFAR10', 'TinyImageNet'], help='Original dataset to obtain class names from torchvision (default CIFAR100).')
    parser.add_argument('--mapping-file', default="./cue_conflict/C100-mapping.txt", help='Mapping file path.')
    parser.add_argument('--checkpoint', default="../trained_models/TinyImageNet/WideResNet_28_4/config547_run_2.pth", help='Path to WideResNet checkpoint.')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for loading. If not set, the whole dataset is one batch.')
    parser.add_argument('--device', default=None, help='Device string (cuda or cpu). If not set, uses cuda if available.')
    parser.add_argument('--repo-root', default="../model-based-data-augmentation", help='Path to repo root; will be prepended to sys.path along with repo/experiments.')
    args = parser.parse_args()

    # If user provided a base/repo root, add it to sys.path so repository imports resolve
    if args.repo_root:
        try:
            setup_paths(args.repo_root)
        except Exception as e:
            print(f"ERROR: failed to setup repo paths from --repo-root '{args.repo_root}': {e}", file=sys.stderr)
            sys.exit(1)

    # resolve dataset path
    dataset_path = resolve_dataset_path(args.dataset_dir, args.relative_to_script)
    if not dataset_path.exists() or not dataset_path.is_dir():
        print(f"ERROR: dataset path not found or not a directory: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    # load original torchvision dataset to get class names (order)
    root_for_torchvision = Path('../data').resolve()
    if args.original_dataset.upper() == 'CIFAR100':
        tv_dataset = datasets.CIFAR100(root=str(root_for_torchvision), train=True, download=False, transform=transforms.ToTensor())
    elif args.original_dataset.upper() == 'CIFAR10':
        tv_dataset = datasets.CIFAR10(root=str(root_for_torchvision), train=True, download=False, transform=transforms.ToTensor())
    elif args.original_dataset.upper() == 'TINYIMAGENET':
        tv_dataset = datasets.ImageFolder(root=f'{root_for_torchvision}/TinyImageNet/train', transform=transforms.ToTensor())
    else:
        raise ValueError(f"Unsupported original_dataset: {args.original_dataset}")
    original_class_names = tv_dataset.classes  # list where index -> class name
    num_classes = len(original_class_names)
    print(f"Loaded original dataset classes ({args.original_dataset}): {num_classes} classes (using torchvision)")

    # parse mapping file
    mapping_path = Path(args.mapping-file) if False else Path(args.mapping_file)  # safe access in older scripts
    mapping_path = Path(args.mapping_file)
    if not mapping_path.exists():
        print(f"ERROR: mapping file not found: {mapping_path}", file=sys.stderr)
        sys.exit(1)
    mapping = parse_mapping_file(mapping_path)

    # transforms: include normalization appropriate for CIFAR datasets
    #mean = [0.50707516, 0.48654887, 0.44091784] if args.original_dataset.upper() == 'CIFAR100' else [0.4914, 0.4822, 0.4465]
    #std  = [0.26733429, 0.25643846, 0.27615047] if args.original_dataset.upper() == 'CIFAR100' else [0.247, 0.243, 0.261]

    transform = transforms.Compose([
        transforms.Resize((args.image_size_factor * 32, args.image_size_factor * 32)),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    # load reduced dataset with ImageFolder
    dataset = ImageFolder(root=str(dataset_path), transform=transform)
    n = len(dataset)
    print(f"Found {n} images across {len(dataset.classes)} top-level folders (shapes).")
    if n == 0:
        print("No images found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # prepare sample info list (order matches dataset.samples and DataLoader with shuffle=False)
    samples = dataset.samples  # list of (path, class_idx)
    sample_info = []
    for path, class_idx in samples:
        fname = Path(path).name
        folder_shape = dataset.classes[class_idx]
        shape_from_fname, texture_from_fname = extract_shape_texture_from_filename(fname)
        if shape_from_fname is None:
            shape_from_fname = folder_shape
        sample_info.append({
            'path': path,
            'folder_shape': folder_shape,
            'shape_ds': shape_from_fname,
            'texture_ds': texture_from_fname
        })

    # import WideResNet class
    try:
        from experiments.models.wideresnet import WideResNet_28_4
    except Exception as e:
        print(f"ERROR importing WideResNet_28_4: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        model_arch = WideResNet_28_4(num_classes, args.image_size_factor, args.original_dataset, True)
    except Exception as e:
        print(f"ERROR importing WideResNet_28_4: {e}", file=sys.stderr)
        sys.exit(1)

    # set device
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    print(f"Using device: {device}")

    # load checkpoint into model
    ckpt_path = Path(args.checkpoint)
    print(ckpt_path)

    try:
        model = load_wideresnet_checkpoint(ckpt_path, model_arch, device=device)
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}", file=sys.stderr)
        sys.exit(1)
    print("Model loaded and set to eval mode.")

    # DataLoader and evaluation in batches
    batch_size = args.batch_size if args.batch_size else n
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    shape_count = 0
    texture_count = 0
    neither_count = 0
    skipped_no_conflict = 0
    total_conflict = 0
    per_entry = []
    from experiments.utils import plot_images

    model.eval()
    sample_idx = 0  # pointer into sample_info

    mean_tensor = torch.tensor([0,0,0]).view(1, 3, 1, 1)
    std_tensor = torch.tensor([1,1,1]).view(1, 3, 1, 1)

    with torch.no_grad():
        for batch_images, _batch_labels in loader:
            b = batch_images.to(device)
            plot_images(10, mean_tensor, std_tensor, b)
            logits = model(b)
            probs = F.softmax(logits, dim=1)
            top1 = torch.argmax(probs, dim=1).cpu().tolist()  # indices for this batch

            for j, pred_idx in enumerate(top1):
                if sample_idx >= len(sample_info):
                    # safety check
                    break
                info = sample_info[sample_idx]
                pred = original_class_names[pred_idx].lower().strip()
                shape_ds = info['shape_ds']
                texture_ds = info['texture_ds']

                # if texture wasn't parseable from filename, treat as no-conflict and skip
                if texture_ds is None:
                    skipped_no_conflict += 1
                    per_entry.append((info['path'], shape_ds, texture_ds, pred, 'skipped_no_texture'))
                    sample_idx += 1
                    continue

                # skip images where there's no conflict
                if shape_ds == texture_ds:
                    skipped_no_conflict += 1
                    per_entry.append((info['path'], shape_ds, texture_ds, pred, 'skipped_no_conflict'))
                    sample_idx += 1
                    continue

                total_conflict += 1
                # find mapping values (list) for shape and texture
                shape_map_list = mapping.get(shape_ds, [])
                texture_map_list = mapping.get(texture_ds, [])
                # normalize map entries to lowercase stripped
                shape_map_set = {s.lower().strip() for s in shape_map_list if s}
                texture_map_set = {s.lower().strip() for s in texture_map_list if s}
                # check if model predicted matches any of the map values
                is_shape = pred in shape_map_set
                is_texture = pred in texture_map_set
                if is_shape and not is_texture:
                    shape_count += 1
                    outcome = 'shape'
                elif is_texture and not is_shape:
                    texture_count += 1
                    outcome = 'texture'
                elif is_shape and is_texture:
                    neither_count += 1
                    outcome = 'both'
                else:
                    neither_count += 1
                    outcome = 'neither'
                per_entry.append((info['path'], shape_ds, texture_ds, pred, outcome))
                sample_idx += 1

    # sanity: sample_idx should equal total samples processed
    if sample_idx != len(sample_info):
        warnings.warn(f"Processed {sample_idx} samples but expected {len(sample_info)}. Check DataLoader ordering!")

    # print summary
    print("\n=== Summary ===")
    print(f"Total images processed: {n}")
    print(f"Skipped (no texture parsed or no conflict): {skipped_no_conflict}")
    print(f"Total cue-conflict images considered: {total_conflict}")
    print(f"Model predicted shape-mapped label: {shape_count}")
    print(f"Model predicted texture-mapped label: {texture_count}")
    print(f"Model predicted neither/both: {neither_count}")

    denom = shape_count + texture_count + neither_count
    if denom > 0:
        print(f"Fraction shape: {shape_count/denom:.4f}, texture: {texture_count/denom:.4f}, neither: {neither_count/denom:.4f}")

    # print a few example lines
    print("\nExamples (first 10):")
    for e in per_entry[:10]:
        print(e)

if __name__ == '__main__':
    main()