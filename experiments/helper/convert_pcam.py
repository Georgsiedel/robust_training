
import os
import torch
from torchvision import transforms
import torchvision

def convert_pcam_to_imagefolder(self, pcam_dataset, split_name):
    split_dir = os.path.join(self.data_path, f"PCAM_{split_name}_images")
    os.makedirs(os.path.join(split_dir, "0"), exist_ok=True)  # Class 0 folder
    os.makedirs(os.path.join(split_dir, "1"), exist_ok=True)  # Class 1 folder

    print(f"Converting {split_name} split to ImageFolder at: {split_dir}")
    for idx in range(len(pcam_dataset)):
        img, label = pcam_dataset[idx]  # img can be PIL.Image or Tensor

        if isinstance(img, torch.Tensor):  # Convert tensor to PIL if needed
            img = transforms.ToPILImage()(img)

        # Now img is guaranteed to be a PIL Image, so we can save directly
        img_path = os.path.join(split_dir, str(int(label)), f"{idx}.png")
        img.save(img_path)

    return split_dir

#load once from torchvision and convert to imagefolder to allow pickle with multiple workers
data_path = '../data/pcam/'
base_trainset = torchvision.datasets.PCAM(root=os.path.abspath(f'{data_path}'), split='train', download=True)
testset = torchvision.datasets.PCAM(root=os.path.abspath(f'{data_path}'), split='val', download=True)
test = torchvision.datasets.PCAM(root=os.path.abspath(f'{data_path}'), split='test', download=True)

# Convert train and test sets
train_dir = convert_pcam_to_imagefolder(base_trainset, "train")
val_dir = convert_pcam_to_imagefolder(testset, "val")
test_dir = convert_pcam_to_imagefolder(test, "test") 