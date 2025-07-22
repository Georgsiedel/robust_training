#!/usr/bin/env python3
import torch
import argparse
from pathlib import Path

from PIL import Image
from torchvision import transforms
import style_transfer as style_transfer
import torchvision.transforms.v2 as transforms_v2

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_img = Image.open(out_dir / "orig.jpg").convert("RGB")
    gen_img = Image.open(out_dir / "generated.jpg").convert("RGB")

    _to_pil = transforms.ToPILImage()

    def save(x, path):
        """
        Save x to disk at `path`.
        x may be either:
        - a PIL.Image
        - a torch.Tensor of shape (C,H,W), dtype float in [0,1] or uint8
        """
        if isinstance(x, torch.Tensor):
            t = x.detach().cpu()
            # if float, clamp to [0,1]
            if t.dtype.is_floating_point:
                t = t.clamp(0,1)
            img = _to_pil(t)
            img.save(path)
        else:
            # assume PIL
            x.save(path)

    # save base images
    #save(orig_img,      "orig.jpg")
    #save(gen_img,       "generated.jpg")

    # 4) stylize both
    vgg, decoder = style_transfer.load_models()
    style_feats  = style_transfer.load_feat_files()
    nst = style_transfer.NSTTransform(
        style_feats, vgg, decoder,
        alpha_min=0.8, alpha_max=0.8,
        probability=1.0, pixels=64
    )

    orig_st, style_img_orig  = nst(transforms.ToTensor()(orig_img))
    gen_st, style_img_gen   = nst(transforms.ToTensor()(gen_img))
    save(orig_st, out_dir / "orig_stylized.jpg")
    save(gen_st, out_dir / "generated_stylized.jpg")
    save(style_img_orig, out_dir / "orig_style.jpg")
    save(style_img_gen, out_dir / "generated_style.jpg")

    # 5) trivial + random erase
    re = transforms_v2.RandomErasing(
        p=args.rand_erase_p,
        scale=(0.02, 0.2)
    )
    ta = transforms_v2.TrivialAugmentWide()
    ta._AUGMENTATION_SPACE = {
        "ShearX": (lambda num_bins, height, width: torch.linspace(0.1, 0.5, num_bins), True),
        "ShearY": (lambda num_bins, height, width: torch.linspace(0.1, 0.5, num_bins), True),
        "Rotate": (lambda num_bins, height, width: torch.linspace(15.0, 50.0, num_bins), True),
    }

    tf = transforms_v2.Compose([
        ta,
        re
    ])

    orig_st_tf = tf(orig_st)
    gen_st_tf  = tf(gen_st)
    save(orig_st_tf,  out_dir / "orig_stylized_transformed.jpg")
    save(gen_st_tf,   out_dir / "generated_stylized_transformed.jpg")

    print("Saved 8 images to", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--generated-npz",  default="../data/TinyImageNet-add-1m-dm.npz")
    p.add_argument("--original-folder",default="../data/TinyImageNet/train")
    p.add_argument("--class-name",    default="n01443537",
                   help="Name of class to sample, e.g. 'n01443537'")
    p.add_argument("--rand-erase-p",    type=float, default=0.0)
    p.add_argument("--output-dir",      default="experiments/example_outputs")
    args = p.parse_args()
    main(args)
