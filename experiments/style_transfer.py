import os
import sys
import numpy as np

current_dir = os.path.dirname(__file__)
module_path = os.path.abspath(current_dir)

import math
import random

if module_path not in sys.path:
    sys.path.append(module_path)

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import adaIN.model as adaINmodel
import adaIN.utils as utils
from run_0 import device as nst_device
from experiments.utils import plot_images

encoder_rel_path = 'adaIN/vgg_normalised.pth'
decoder_rel_path = 'adaIN/decoder.pth'
encoder_path = os.path.abspath(os.path.join(current_dir, encoder_rel_path))
decoder_path = os.path.abspath(os.path.join(current_dir, decoder_rel_path))

def load_models():

    vgg = adaINmodel.vgg
    decoder = adaINmodel.decoder
    vgg.load_state_dict(torch.load(encoder_path, weights_only=True))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    decoder.load_state_dict(torch.load(decoder_path, weights_only=True))

    vgg.to(nst_device)
    decoder.to(nst_device)

    vgg.eval()
    decoder.eval()
    return vgg, decoder

def load_feat_files(path):

    style_feats_path = os.path.abspath(os.path.join(os.path.dirname(current_dir), path))
    style_feats_np = np.load(style_feats_path)
    style_feats_tensor = torch.from_numpy(style_feats_np)
    style_feats_tensor = style_feats_tensor.to(nst_device)
    return style_feats_tensor


class NSTTransform(transforms.Transform):
    """
    A class to apply neural style transfer with AdaIN to datasets in the training pipeline.
    Now supports both RGB (3-channel) and grayscale (1-channel) images.
    Parameters:

    style_feats: Style features extracted from the style images using adaIN Encoder
    vgg: AdaIN Encoder
    decoder: AdaIN Decoder
    alpha = Strength of style transfer [between 0 and 1]
    probability = Probability of applying style transfer [between 0 and 1]
    randomize = randomly selected strength of alpha from a given range
    rand_min = Minimum value of alpha if randomized
    rand_max = Maximum value of alpha if randomized
    """

    def __init__(self, style_feats, vgg, decoder,
                 alpha_min=1.0, alpha_max=1.0,
                 probability=0.5):
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        self.style_features = style_feats
        self.num_styles = len(style_feats)
        self.probability = probability
        self.to_pil_img = transforms.ToPILImage()

    @torch.no_grad()
    def __call__(self, x):
        single_image = x.ndimension() == 3
        if single_image:
            x = x.unsqueeze(0)  # [C,H,W] → [1,C,H,W]

        ex_device = x.device

        batchsize = x.size(0)
        ratio = int(math.floor(batchsize * self.probability + random.random()))
        if ratio == 0:
            return x.squeeze(0) if single_image else x

        # Detect grayscale
        was_grayscale = x.shape[1] == 1
        if was_grayscale:
            # Repeat channel → RGB
            x = x.repeat(1, 3, 1, 1)
        
        # Choose random subset to stylize
        idy = torch.randperm(self.num_styles)[:ratio]
        idx = torch.randperm(batchsize)[:ratio]

        x_selected = x[idx]

        _, _, H, W = x.shape
        if (H, W) != (224, 224):
            x_selected = self.upsample(x_selected)

        x_selected = x_selected.to(nst_device)
        x_selected = self.style_transfer(self.vgg, self.decoder, x_selected, self.style_features[idy])
        x_selected = x_selected.to(ex_device)

        if (H, W) != (224, 224):
            x_selected = nn.Upsample(size=(H, W), mode='bilinear', align_corners=False)(x_selected)

        x[idx] = x_selected

        #normalized tensor, does not appear necessary in practice
        #stl_imgs = self.norm_style_tensor(stl_imgs)

        # Convert back to grayscale if needed
        if was_grayscale:
            x = F.rgb_to_grayscale(x)

        if single_image:
            x = x.squeeze(0)  # Back to [C,H,W]

        return x

    @torch.no_grad()
    def norm_style_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()

        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        #scaled_tensor = normalized_tensor * 255
        #scaled_tensor = scaled_tensor.byte() # converts dtype to torch.uint8 between 0 and 255 #here
        return normalized_tensor

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style):
        alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max)
        content_f = vgg(content)
        style_f = style
        feat = utils.adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)

class NSTTransform_rectangular(transforms.Transform):
    """
    A class to apply neural style transfer with AdaIN to datasets in the training pipeline.
    Supports rectangular images (both horizontal and vertical as longer sides) by resizing 
    the shorter side to target resolution 224px, patching the longer side into 32px overlapping patches,
    stylizing all patches of selected images in one batch and reconstructing all full images with blending.
    
    Parameters:

    style_feats: Style features extracted from the style images using adaIN Encoder
    vgg: AdaIN Encoder
    decoder: AdaIN Decoder
    alpha = Strength of style transfer [between 0 and 1]
    probability = Probability of applying style transfer [between 0 and 1]
    randomize = randomly selected strength of alpha from a given range
    rand_min = Minimum value of alpha if randomized
    rand_max = Maximum value of alpha if randomized

     """

    def __init__(self, style_feats, vgg, decoder, alpha_min=1.0, alpha_max=1.0, probability=0.5, overlap=32):
        super().__init__()
        self.vgg = vgg
        self.decoder = decoder
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        self.to_tensor = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True)
        ])
        self.style_features = style_feats
        self.num_styles = len(style_feats)
        self.probability = probability
        self.to_pil_img = transforms.ToPILImage()
        self.overlap = overlap

    @torch.no_grad()
    def __call__(self, x):
        single_image = x.ndimension() == 3
        
        if single_image:
            x = x.unsqueeze(0)  # [C,H,W] → [1,C,H,W]

        batchsize = x.size(0)
        ratio = int(math.floor(batchsize * self.probability + random.random()))
        if ratio == 0:
            return x.squeeze(0) if single_image else x
        
        # --- Preselection up-front ---
        idy = torch.randperm(self.num_styles)[:ratio].tolist()   # style indices
        idx = torch.randperm(batchsize)[:ratio].tolist()         # image indices to stylize
        stylize_map = {int(img_idx): int(style_idx) for img_idx, style_idx in zip(idx, idy)}

        selected_x = x[idx]  # shape [R, C, H, W] on CPU

        # Detect grayscale and convert to RGB for processing
        was_grayscale = selected_x.shape[1] == 1
        if was_grayscale:
            selected_x = selected_x.repeat(1, 3, 1, 1)

        # Constants
        patch_size = 224
        overlap = 32
        stride = patch_size - overlap

        metas_selected = []
        resized_selected_images = []        # CPU tensors [3, new_H, new_W] in same order as selected indices
        patches_to_process = []             # list of CPU [3,224,224] patches in-order (image-major)
        patches_per_selected_image = []     # number of patches per selected image in same order as selected indices
        selected_image_order = idx[:]       # the original image indices in the order we process them

        # Extract patches for each selected image (loop only over selected_x_rgb)
        for r, img in enumerate(selected_x):  # img shape [3,H,W] on CPU
            img = img.unsqueeze(0)  # [1,3,H,W]
            _, _, H, W = img.shape
            orig_H, orig_W = int(H), int(W)

            # Resize short side -> 224 (preserve aspect)
            scale = 224.0 / float(min(H, W))
            new_H = int(round(H * scale))
            new_W = int(round(W * scale))
            img_resized = F.resize(img, size=[new_H, new_W], interpolation=F.InterpolationMode.BILINEAR)  # [1,3,new_H,new_W]

            resized_selected_images.append(img_resized.squeeze(0).cpu())  # keep CPU copy for reconstruction

            # compute tiling grid
            num_h = max(1, math.ceil((new_H - overlap) / stride))
            num_w = max(1, math.ceil((new_W - overlap) / stride))
            num_patches = int(num_h * num_w)
            metas_selected.append((selected_image_order[r], orig_H, orig_W, new_H, new_W, int(num_h), int(num_w), int(num_patches)))

            # extract patches (CPU) in deterministic order and append to patches_to_process
            cnt = 0
            for i in range(num_h):
                for j in range(num_w):
                    top = min(int(i * stride), max(0, new_H - patch_size))
                    left = min(int(j * stride), max(0, new_W - patch_size))
                    patch = img_resized[:, :, top:top+patch_size, left:left+patch_size].squeeze(0).cpu()  # [3,224,224] CPU
                    patches_to_process.append(patch)
                    cnt += 1
            patches_per_selected_image.append(cnt)

        # Build big batches for stylization
        patches_tensor = torch.stack(patches_to_process, dim=0).to(nst_device)  # [N_proc,3,224,224] on device

        # Repeat style_feats for each selected image according to its patch count and concat
        style_feats_batch_list = []
        for orig_img_idx, cnt in zip(selected_image_order, patches_per_selected_image):
            style_idx = stylize_map[orig_img_idx]
            sf = self.style_features[style_idx]        # shape [C_f, H_f, W_f]
            sf_rep = sf.unsqueeze(0).repeat(cnt, 1, 1, 1)  # [cnt, C_f, H_f, W_f]
            style_feats_batch_list.append(sf_rep)
        
        style_feats_batch = torch.cat(style_feats_batch_list, dim=0).to(nst_device)  # [N_proc, C_f, H_f, W_f] on device

        # Single call to style_transfer (on-device)
        stylized_patches_proc = self.style_transfer(self.vgg, self.decoder, patches_tensor, style_feats_batch)
        stylized_patches_proc = stylized_patches_proc.cpu()  # bring back to CPU for reassembly

        # Precompute cosine/Hann blending mask (CPU)
        y = torch.linspace(0, math.pi, patch_size)
        hann = 0.5 - 0.5 * torch.cos(y)
        mask2d = (hann.unsqueeze(1) * hann.unsqueeze(0))  # [224,224]
        mask2d = mask2d / mask2d.max()
        mask3 = mask2d.unsqueeze(0).repeat(3, 1, 1)  # [3,224,224]

        # Reconstruct each selected image from its stylized patches and write back into x at positions idx
        proc_ptr = 0
        for meta in metas_selected:
            img_idx, orig_H, orig_W, new_H, new_W, num_h, num_w, num_patches = meta

            recon = torch.zeros((3, new_H, new_W), dtype=torch.float32)
            weight = torch.zeros_like(recon)

            for i in range(num_h):
                for j in range(num_w):
                    top = min(int(i * stride), max(0, new_H - patch_size))
                    left = min(int(j * stride), max(0, new_W - patch_size))

                    if proc_ptr >= stylized_patches_proc.shape[0]:
                        print("ERROR: stylized patches exhausted unexpectedly.")
                        raise RuntimeError("Stylized patches exhausted unexpectedly.")

                    patch = stylized_patches_proc[proc_ptr]  # [3,224,224] CPU
                    proc_ptr += 1

                    recon[:, top:top+patch_size, left:left+patch_size] += patch * mask3
                    weight[:, top:top+patch_size, left:left+patch_size] += mask3

            # Normalize overlapping regions
            recon = recon / torch.clamp(weight, min=1e-5)

            # Insert the reconstructed selected image back into original tensor x (on CPU)
            expected_channels = x[img_idx].shape[0]
            if recon.shape[0] != expected_channels:
                print(f"ERROR: channel mismatch for image {img_idx}: expected {expected_channels}, got {recon.shape[0]}")
                raise RuntimeError("Channel mismatch during insertion of stylized image.")

            # Resize back to original image size
            recon_resized_back = F.resize(recon, size=[orig_H, orig_W], interpolation=F.InterpolationMode.BILINEAR)

            if was_grayscale:
                recon_resized_back = F.rgb_to_grayscale(recon_resized_back)

            x[img_idx] = recon_resized_back

        # Sanity: ensure we consumed all stylized patches
        if proc_ptr != stylized_patches_proc.shape[0]:
            print("ERROR: Not all stylized patches were consumed:", proc_ptr, "of", stylized_patches_proc.shape[0])
            raise RuntimeError("Not all stylized patches were consumed during reconstruction.")

        # Return result on same device as input (CPU) and same shape as input
        if single_image:
            x = x.squeeze(0)

        return x

    @torch.no_grad()
    def norm_style_tensor(self, tensor):
        min_val = tensor.min()
        max_val = tensor.max()

        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        #scaled_tensor = normalized_tensor * 255
        #scaled_tensor = scaled_tensor.byte() # converts dtype to torch.uint8 between 0 and 255 #here
        return normalized_tensor

    @torch.no_grad()
    def style_transfer(self, vgg, decoder, content, style):
        alpha = np.random.uniform(low=self.alpha_min, high=self.alpha_max)
        content_f = vgg(content)
        style_f = style
        feat = utils.adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        return decoder(feat)