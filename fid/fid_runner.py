import os
import sys
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torch.utils.data import DataLoader, TensorDataset

# Dynamically adjust sys.path to include the repo's base and experiments folder
def setup_paths(base_path):
    sys.path.append(base_path)
    sys.path.append(os.path.join(base_path, 'experiments'))

def load_models(device, base_path):
    from adaIN.model import vgg as original_vgg, decoder as original_decoder
    vgg = original_vgg
    decoder = original_decoder
    vgg.load_state_dict(torch.load(f'{base_path}/experiments/adaIN/vgg_normalised.pth', map_location=device))
    vgg = torch.nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(f'{base_path}/experiments/adaIN/decoder.pth', map_location=device))
    return vgg.to(device).eval(), decoder.to(device).eval()

def stylize_images(args, device):
    print(f"Loading and sampling {args.num_to_sample} images...")
    #Set the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    npz_map = {'c10': args.npz_c10, 'c100': args.npz_c100, 'tin': args.npz_tin}
    npz_path = npz_map[args.dataset]
    save_dir = os.path.join(args.output_dir, f'styled_fid_images/{args.dataset}_{args.num_to_sample}')
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(npz_path)
    all_images = data['image']
    indices = torch.randperm(len(all_images))[:args.num_to_sample]
    sampled_images = all_images[indices.numpy()]

    images_tensor = torch.from_numpy(sampled_images).permute(0, 3, 1, 2).float() / 255.0
    loader = DataLoader(TensorDataset(images_tensor), batch_size=128, shuffle=False)

    print("Stylizing images...")
    vgg, decoder = load_models(device, args.base_path)
    from experiments.style_transfer import NSTTransform
    style_feats = torch.from_numpy(np.load(args.style_feats_path)).to(device)

    nst = NSTTransform(
        style_feats=style_feats,
        vgg=vgg,
        decoder=decoder,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        probability=args.probability,
        pixels=args.pixels
    )

    with torch.no_grad():
        count = 0
        for batch in tqdm(loader):
            x = batch[0].to(device)
            y = nst(x).cpu()
            y = (y * 255).clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
            for img in y:
                Image.fromarray(img).save(os.path.join(save_dir, f"{count:06d}.png"))
                count += 1

    print(f"Saved {count} stylized images to: {save_dir}")
    return save_dir

def extract_training_set(args):
    print("Extracting real training images...")
    output_dir = os.path.join(args.output_dir, f"{args.dataset}-training")
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset == "c10":
        dataset = CIFAR10(root=args.output_dir, train=True, download=True)
    elif args.dataset == "c100":
        dataset = CIFAR100(root=args.output_dir, train=True, download=True)
    elif args.dataset == "tin":
        dataset = ImageFolder(root=args.tiny_imagenet_path, transform=None)
    else:
        raise ValueError("Invalid dataset name")

    for idx, (img, _) in tqdm(enumerate(dataset), total=len(dataset)):
        Image.fromarray(np.array(img)).save(os.path.join(output_dir, f"{idx:06d}.png"))

    print(f"Saved real training set to: {output_dir}")
    return output_dir

def calculate_ref_stats(real_train_dir, dataset_name, edm_fid_script, output_dir):
    ref_out = os.path.join(output_dir, f"{dataset_name}-training-ref.npz")
    print(f"Calculating FID reference stats...")
    os.system(f"python {edm_fid_script} ref --data {real_train_dir} --dest {ref_out}")
    return ref_out

def calculate_fid(styled_dir, ref_path, dataset_name, num_to_sample, edm_fid_script):
    print(f"Calculating FID for {dataset_name}...")
    os.system(f"python {edm_fid_script} calc --images {styled_dir} "
              f"--ref {ref_path} --num {num_to_sample} --seed 42 --batch 64")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['c10', 'c100', 'tin'])
    parser.add_argument('--alpha_min', type=float, default=1.0)
    parser.add_argument('--alpha_max', type=float, default=1.0)
    parser.add_argument('--probability', type=float, default=1.0)
    parser.add_argument('--num_to_sample', type=int)
    parser.add_argument('--pixels', type=int)
    parser.add_argument('--base_path', default="/kaggle/working/model-based-data-augmentation")
    parser.add_argument('--style_feats_path', default="/kaggle/input/style-feats-adain-1000/style_feats_adain_1000.npy")
    parser.add_argument('--tiny_imagenet_path', default="/kaggle/input/tinyimagenet/tiny-imagenet-200/train")
    parser.add_argument('--npz_c10', default="/kaggle/input/1m-cifar10/1mcifar10.npz")
    parser.add_argument('--npz_c100', default="/kaggle/input/1m-cifar100/1mcifar100.npz")
    parser.add_argument('--npz_tin', default="/kaggle/input/1m-tiny/tiny_edm_1m.npz")
    parser.add_argument('--edm_fid_script', default="/kaggle/working/model-based-data-augmentation/fid/edm/fid.py")
    parser.add_argument('--output_dir', default="/kaggle/working")
    parser.add_argument('--seed', type=int, default=42, help="Seed for reproducibility")

    args = parser.parse_args()

    args.num_to_sample = args.num_to_sample if args.num_to_sample else (100000 if args.dataset == 'tin' else 50000)
    args.pixels = args.pixels if args.pixels else (64 if args.dataset == 'tin' else 32)

    setup_paths(args.base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    styled_dir = stylize_images(args, device)
    real_dir = extract_training_set(args)
    ref_path = calculate_ref_stats(real_dir, args.dataset, args.edm_fid_script, args.output_dir)
    calculate_fid(styled_dir, ref_path, args.dataset, args.num_to_sample, args.edm_fid_script)

if __name__ == "__main__":
    main()
