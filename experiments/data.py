import random
import os
import time
import json
import gc

import torch
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, ConcatDataset, RandomSampler, BatchSampler, DataLoader, TensorDataset
import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
import experiments.custom_transforms as custom_transforms
from run_0 import device
from experiments.utils import plot_images, CsvHandler
from experiments.custom_datasets import SubsetWithTransform, NumpyDataset, AugmentedDataset, ListDataset, CustomDataset 
from experiments.custom_datasets import BalancedRatioSampler, GroupedAugmentedDataset, ReproducibleBalancedRatioSampler, StyleDataset

def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1, verbose=False):

    if manifold:
        mean = torch.mean(batch, dim=(0, 2, 3), keepdim=True).to(device)
        std = torch.std(batch, dim=(0, 2, 3), keepdim=True).to(device)
        mean = mean.view(1, batch.size(1), 1, 1)
        std = ((1 / std) / manifold_factor).view(1, batch.size(1), 1, 1)
    elif normalized:
        if dataset == 'CIFAR10':
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.247, 0.243, 0.261]).view(1, 3, 1, 1).to(device)
        elif dataset == 'CIFAR100':
            mean = torch.tensor([0.50707516, 0.48654887, 0.44091784]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.26733429, 0.25643846, 0.27615047]).view(1, 3, 1, 1).to(device)
        elif dataset in ['ImageNet', 'TinyImageNet', 'ImageNet-100']:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(device)
            if verbose:
                print('no normalization values set for this dataset, scaling to [-1,1]')
    else:
        mean = 0
        std = 1

    return mean, std

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    global fixed_worker_rng #impulse noise augmentations sk-learn function needs a separate rng for reproducibility
    fixed_worker_rng = np.random.default_rng()

def extract_labels(dataset):
    """
    Return a flat list of labels (one entry per sample) for any dataset.
    """
    if hasattr(dataset, 'targets'):
        # most vision datasets put labels here
        return list(dataset.targets)
    elif hasattr(dataset, 'labels'):
        return list(dataset.labels)
    elif hasattr(dataset, 'samples'):
        # ImageFolder and friends
        return [s[1] for s in dataset.samples]
    else:
        # worst case: iterate—but still O(N), same as splitting
        return [dataset[i][1] for i in range(len(dataset))]

def extract_num_classes(dataset, labels=None):
    """
    Return the number of classes in the dataset.
    Supports scalar or multilabel (multi-hot vector) labels.
    """
    # Use dataset info if available
    if hasattr(dataset, 'classes'):
        return len(dataset.classes)
    if hasattr(dataset, 'class_to_idx'):
        return len(dataset.class_to_idx)
    
    # Otherwise get labels if not provided
    if labels is None:
        labels = extract_labels(dataset)

    # If labels are multilabel vectors (list of arrays/tensors)
    if (
        isinstance(labels, (list, tuple))
        and len(labels) > 0
        and (
            (hasattr(labels[0], 'ndim') and labels[0].ndim == 1)  # np.ndarray or tensor
            or (isinstance(labels[0], (list, tuple)) and all(isinstance(x, (int,float)) for x in labels[0]))  # list/tuple of numbers
        )
    ):
        return len(labels[0])  # number of classes from length of vector

    # Otherwise treat as scalar labels, count unique
    unique_labels = set()
    for lbl in labels:
        # If tensor or numpy scalar convert to Python scalar
        if hasattr(lbl, 'item'):
            unique_labels.add(lbl.item())
        else:
            unique_labels.add(lbl)
    return len(unique_labels)

class DataLoading():
    def __init__(self, dataset, validontest=True, epochs=200, 
                 resize = False, run=0, number_workers=0, kaggle=False):
        self.dataset = dataset
        self.resize = resize
        self.run = run
        self.epochs = epochs
        self.validontest = validontest
        self.number_workers = number_workers
        self.kaggle = kaggle

        if dataset in ['CIFAR10', 'CIFAR100', 'GTSRB','ImageNet', 'ImageNet-100']:
            self.factor = 1
        elif dataset in ['TinyImageNet', 'EuroSAT', 'Wafermap']:
            self.factor = 2
        elif dataset in ['PCAM']:
            self.factor = 3
        else:
            self.factor = 1

        
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paths.json")
        with open(file_path, "r") as f:
            paths = json.load(f)

        suffix = "_kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else ""  # use env to detect Kaggle

        def resolve_path(key, suffix=""):
            # First try the suffixed path
            p = paths.get(f"{key}{suffix}")
            if not p:
                p = paths[key]
            
            # Make absolute
            if not os.path.isabs(p):
                # Relative to the repository root
                repo_root = os.path.dirname(os.path.dirname(__file__))
                p = os.path.abspath(os.path.join(repo_root, p))
            return p

        self.data_path = resolve_path("data", suffix)
        self.c_labels_path = resolve_path("c_labels", suffix)
        self.trained_models_path = resolve_path("trained_models", suffix)
        self.style_feats_path = resolve_path("style_feats", suffix)
        self.write_data_path = resolve_path("write_data", suffix)

    def create_transforms(self, train_aug_strat_orig, train_aug_strat_gen, RandomEraseProbability=0.0, 
                          grouped_stylization=False):
        self.train_aug_strat_orig = train_aug_strat_orig
        self.train_aug_strat_gen = train_aug_strat_gen
        self.grouped_stylization = grouped_stylization
        self.RandomEraseProbability = RandomEraseProbability
        # list of all data transformations used
        t = transforms.ToTensor()
        c32 = transforms.RandomCrop(32, padding=4)
        c64 = transforms.RandomCrop(64, padding=8)
        c64_WM = transforms.RandomCrop(64, padding=6)
        c96 = transforms.RandomCrop(96, padding=12)
        c224 = transforms.RandomCrop(224, padding=28)
        flip = transforms.RandomHorizontalFlip()
        flip_v = transforms.RandomVerticalFlip()
        r32 = transforms.Resize((32,32), antialias=True)
        r224 = transforms.Resize(224, antialias=True)
        r256 = transforms.Resize(256, antialias=True)
        cc224 = transforms.CenterCrop(224)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        re = transforms.RandomErasing(p=self.RandomEraseProbability, scale=(0.02, 0.4)) #, value='random' --> normally distributed and out of bounds 0-1

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset in ['ImageNet', 'ImageNet-100']:
            self.transforms_preprocess = transforms.Compose([t])
            self.transforms_preprocess_additional_test = transforms.Compose([r256, cc224])
        elif self.dataset == 'GTSRB':
            self.transforms_preprocess = transforms.Compose([t, r32])
        elif self.dataset == 'WaferMap':
            #https://github.com/Junliangwangdhu/WaferMap/tree/master
            #preprocessing once upon load_base_data, not on the fly (small set fits in memory, lots of operations)
            self.transforms_preprocess = transforms.Compose([
                #t,
                #custom_transforms.ToFloat32(),
                #custom_transforms.DivideBy2(),
                #custom_transforms.ExpandGrayscaleTensorTo3Channels(), #directly converts to 3 channels
                #c64_WM
            ])
        else:
            self.transforms_preprocess = transforms.Compose([t])
        
        if self.resize == True and self.dataset not in ['ImageNet', 'ImageNet-100']:
            self.transforms_preprocess = transforms.Compose([t, r224])

        # standard augmentations of training set, without tensor transformation
        if self.dataset in ['ImageNet', 'ImageNet-100']:
            self.transforms_basic = transforms.Compose([flip, rrc224])
        elif self.dataset in ['CIFAR10', 'CIFAR100', 'GTSRB']:
            self.transforms_basic = transforms.Compose([flip, c32])
        elif self.dataset in ['TinyImageNet', 'EuroSAT']:
            self.transforms_basic = transforms.Compose([flip, c64])
        elif self.dataset in ['PCAM']:
            self.transforms_basic = transforms.Compose([flip, flip_v, c96])
        elif self.dataset in ['WaferMap']:
            self.transforms_basic = transforms.Compose([c64_WM, flip, flip_v])

        if self.resize == True and self.dataset not in ['ImageNet', 'ImageNet-100']:
            self.transforms_basic = transforms.Compose([flip, c224])

        self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = custom_transforms.get_transforms_map(train_aug_strat_orig, re, self.dataset, self.factor, grouped_stylization, self.style_feats_path)
        self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = custom_transforms.get_transforms_map(train_aug_strat_gen, re, self.dataset, self.factor, grouped_stylization, self.style_feats_path)

    def update_transforms(self, stylize_prob_orig=None, stylize_prob_syn=None, alpha_min_orig=None, 
                          alpha_min_syn=None, RandomEraseProbability=None):
        
        if RandomEraseProbability is None:
            RandomEraseProbability = self.RandomEraseProbability
        re = transforms.RandomErasing(p=RandomEraseProbability, scale=(0.02, 0.4)) #, value='random' --> normally distributed and out of bounds 0-1

        self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = custom_transforms.get_transforms_map(self.train_aug_strat_orig, re, self.dataset, self.factor, self.grouped_stylization, self.style_feats_path)
        self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = custom_transforms.get_transforms_map(self.train_aug_strat_gen, re, self.dataset, self.factor, self.grouped_stylization, self.style_feats_path)

        if stylize_prob_orig is not None:
            self.stylization_orig.stylized_ratio = stylize_prob_orig
        if stylize_prob_syn is not None:
            self.stylization_gen.stylized_ratio = stylize_prob_syn
        if alpha_min_orig is not None:
            self.stylization_orig.transform_style.alpha_min = alpha_min_orig
        if alpha_min_syn is not None:
            self.stylization_gen.transform_style.alpha_min = alpha_min_syn
    
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
    
    def load_base_data(self, test_only=False):

        if self.validontest:

            if self.dataset in ['ImageNet', 'ImageNet-100']:
                self.testset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/val'),
                                                                transform=transforms.Compose([self.transforms_preprocess, self.transforms_preprocess_additional_test]))
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'))
            
            elif self.dataset == 'TinyImageNet':
                self.testset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/val'),
                                                                transform=self.transforms_preprocess)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'))

            elif self.dataset in ['CIFAR10', 'CIFAR100']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.testset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=False, download=True,
                                        transform=self.transforms_preprocess)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=True, download=True)
            
            elif self.dataset in ['GTSRB']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.testset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='test', download=True,
                                        transform=self.transforms_preprocess)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='train', download=True)
            elif self.dataset in ['PCAM']:
                #load once from torchvision and convert to imagefolder to allow pickle with multiple workers
                #load_helper = getattr(torchvision.datasets, self.dataset)
                #self.base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='train', download=True)
                #self.testset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='val', download=True)
                #test = load_helper(root=os.path.abspath(f'{self.data_path}'), split='test', download=True)

                # Convert train and test sets
                #train_dir = self.convert_pcam_to_imagefolder(self.base_trainset, "train")
                #val_dir = self.convert_pcam_to_imagefolder(self.testset, "val")
                #test_dir = self.convert_pcam_to_imagefolder(test, "test")                
                
                self.testset = ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/PCAM_test_images'), transform=self.transforms_preprocess)

                if test_only:
                    self.base_trainset = None
                else:
                    valset = ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/PCAM_val_images'))
                    trainset = ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/PCAM_train_images'))
                    self.base_trainset = ConcatDataset([trainset, valset])
            
            elif self.dataset == 'EuroSAT':
                print('EuroSAT has no predefined test split. Using a custom seeded random split.')
                load_helper = getattr(torchvision.datasets, self.dataset)
                full_set = load_helper(root=os.path.abspath(f'{self.data_path}'), download=True)

                all_labels = extract_labels(full_set)
                
                train_indices, val_indices, _, _ = train_test_split(
                range(len(full_set)),
                all_labels,
                stratify=all_labels,
                test_size=0.2,
                random_state=0) #always with 0 seed - testset split should always be the same

                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = Subset(full_set, train_indices)
                self.testset = SubsetWithTransform(Subset(full_set, val_indices), transforms.Compose([self.transforms_preprocess]))
                
                self.num_classes = extract_num_classes(self.testset, labels=all_labels)
                return    
                        
            elif self.dataset == 'WaferMap':
                print('WaferMap has no predefined test split. Using a custom seeded random split.')
                data=np.load(os.path.join(f'{self.data_path}/MixedWM38.npz'))
                x = data["arr_0"]
                y = data["arr_1"]
                
                x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                stratify=y,
                test_size=0.2,
                random_state=0)

                x_transform = transforms.Compose([transforms.ToTensor(),
                                    custom_transforms.ToFloat32(),
                                    custom_transforms.DivideBy2(),
                                    custom_transforms.ExpandGrayscaleTensorTo3Channels(),
                                    transforms.RandomCrop(64, padding=6)])

                x_train = torch.stack([x_transform(img) for img in x_train])
                x_test = torch.stack([x_transform(img) for img in x_test])
                y_train = torch.from_numpy(y_train).float()
                y_test = torch.from_numpy(y_test).float()

                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = TensorDataset(x_train, y_train)
                self.testset = TensorDataset(x_test, y_test) #transforms already done
            
            else:
                print('Dataset not loadable')

            all_labels = extract_labels(self.testset)
            self.num_classes = extract_num_classes(self.testset, labels=all_labels)

        else:
            if self.dataset in ['ImageNet', 'TinyImageNet', 'ImageNet-100']:
                base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'))
            elif self.dataset in ['CIFAR10', 'CIFAR100']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=True, download=True)
            elif self.dataset in ['GTSRB']:
                load_helper = getattr(torchvision.datasets, self.dataset)
                base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), split='train', download=True)
            elif self.dataset in ['PCAM']:
                #Convert to ImageFolder from torchvision once, see above

                self.base_trainset = ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/PCAM_train_images'))
                self.testset = ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/PCAM_val_images'), transform=self.transforms_preprocess)

                self.num_classes = len(self.base_trainset.classes)
                return  #PCAM already features train/val split, so we can return
            
            elif self.dataset in ['EuroSAT']:
                print('EuroSAT has no predefined test split. Using a custom seeded random split.')
                load_helper = getattr(torchvision.datasets, self.dataset)
                full_set = load_helper(root=os.path.abspath(f'{self.data_path}'), download=True)
                
                all_labels = extract_labels(full_set)
                
                train_indices, val_indices, _, _ = train_test_split(
                range(len(full_set)),
                all_labels,
                stratify=all_labels,
                test_size=0.2,
                random_state=0) #always with 0 seed - testset split should always be the same

                base_trainset = Subset(full_set, train_indices)

            elif self.dataset == 'WaferMap':
                print('WaferMap has no predefined test split. Using a custom seeded random split.')
                data=np.load(os.path.join(f'{self.data_path}/MixedWM38.npz'))
                x = data["arr_0"]
                y = data["arr_1"]
                
                x_train, _, y_train, _ = train_test_split(
                x,
                y,
                stratify=y,
                test_size=0.2,
                random_state=0)

                base_trainset = NumpyDataset(x_train, y_train)

            else:
                print('Dataset not loadable')  

            all_labels = extract_labels(base_trainset)

            #use 20% of training set as validation set, but at most 10000 images
            testsplit = 0.2 if len(base_trainset) <= 50000 else 10000

            train_indices, val_indices, _, _ = train_test_split(
                range(len(base_trainset)),
                all_labels,
                stratify=all_labels,
                test_size=testsplit,
                random_state=self.run)  # same validation split for same runs, but new validation on multiple runs
            
            if test_only == False:
                self.base_trainset = Subset(base_trainset, train_indices)

            if self.dataset in ['ImageNet', 'ImageNet-100']:
                self.testset = SubsetWithTransform(Subset(base_trainset, val_indices), transforms.Compose([self.transforms_preprocess, self.transforms_preprocess_additional_test]))
            else:
                self.testset = SubsetWithTransform(Subset(base_trainset, val_indices), transforms.Compose([self.transforms_preprocess]))
            
            self.num_classes = extract_num_classes(self.testset, labels=all_labels)
    
    def load_style_dataloader(self, style_dir, batch_size):
        style_dataset = StyleDataset(style_dir, dataset_type=self.dataset)
        style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=False)
        return style_loader

        
    def load_augmented_traindata(self, target_size, generated_ratio, epoch=0, robust_samples=0, grouped_stylization=False):
        self.robust_samples = robust_samples
        self.target_size = target_size
        try:
            self.generated_dataset = np.load(os.path.abspath(f'{self.data_path}/{self.dataset}-add-1m-dm.npz'),
                                    mmap_mode='r') if generated_ratio > 0.0 else None
            self.generated_ratio = generated_ratio
        except:
            print(f'No synthetic data found for this dataset in {self.data_path}/{self.dataset}-add-1m-dm.npz')
            self.generated_ratio = 0.0
            self.generated_dataset = None

        self.epoch = epoch

        torch.manual_seed(self.epoch + self.epochs * self.run)
        torch.cuda.manual_seed(self.epoch + self.epochs * self.run)
        np.random.seed(self.epoch + self.epochs * self.run)
        random.seed(self.epoch + self.epochs * self.run)

        self.num_generated = int(target_size * self.generated_ratio)
        self.num_original = target_size - self.num_generated

        if grouped_stylization == False:

            if self.num_original > 0:
                original_indices = torch.randperm(self.target_size)[:self.num_original]
                original_subset = SubsetWithTransform(Subset(self.base_trainset, original_indices), self.transforms_preprocess)

                if self.stylization_orig is not None:
                    stylized_original_subset, style_mask_orig = self.stylization_orig(original_subset)
                else: 
                    stylized_original_subset, style_mask_orig = original_subset, [False] * len(original_subset)
            else:
                stylized_original_subset, style_mask_orig = None, []
            
            if self.num_generated > 0 and self.generated_dataset is not None:
                generated_indices = np.random.choice(len(self.generated_dataset['label']), size=self.num_generated, replace=False)

                generated_subset = NumpyDataset(
                    self.generated_dataset['image'][generated_indices],
                    self.generated_dataset['label'][generated_indices],
                    transform=self.transforms_preprocess
                )

                if self.stylization_gen is not None:
                    stylized_generated_subset, style_mask_gen = self.stylization_gen(generated_subset)
                else:
                    stylized_generated_subset, style_mask_gen = generated_subset, [False] * len(generated_subset)
            else:
                stylized_generated_subset, style_mask_gen = None, []
            
            style_mask = style_mask_orig + style_mask_gen
            
            self.trainset = AugmentedDataset(stylized_original_subset, stylized_generated_subset, style_mask,
                                            self.transforms_basic, self.transforms_orig_after_style, self.transforms_gen_after_style, 
                                            self.transforms_orig_after_nostyle, self.transforms_gen_after_nostyle, self.robust_samples)

        else:
            if self.num_original > 0:
                original_indices = torch.randperm(len(self.base_trainset))[:self.num_original]
                original_subset = SubsetWithTransform(Subset(self.base_trainset, original_indices), self.transforms_preprocess)
            else:
                original_subset = None
            
            if self.num_generated > 0 and self.generated_dataset is not None:
                generated_indices = np.random.choice(len(self.generated_dataset['label']), size=self.num_generated, replace=False)

                generated_subset = NumpyDataset(
                    self.generated_dataset['image'][generated_indices],
                    self.generated_dataset['label'][generated_indices],
                    transform=self.transforms_preprocess
                )
            else:
                generated_subset = None
            
            self.trainset = GroupedAugmentedDataset(original_subset, generated_subset, self.transforms_basic, self.stylization_orig, 
                                    self.stylization_gen, self.transforms_orig_after_style, self.transforms_gen_after_style, 
                                    self.transforms_orig_after_nostyle, self.transforms_gen_after_nostyle, self.robust_samples, epoch)
    
    def precompute_and_append_c_data(self, set, c_datasets, corruption, csv_handler, subset, subsetsize, valid_run):
        random_corrupted_testset = SubsetWithTransform(self.testset, 
                                                    transform=custom_transforms.RandomCommonCorruptionTransform(set, corruption, self.dataset, csv_handler, self.resize))
        if subset == True:
            selected_indices = np.random.choice(len(self.testset), subsetsize, replace=False)
            random_corrupted_testset = Subset(random_corrupted_testset, selected_indices)
        
        # If valid_run, precompute the transformed outputs and wrap them as a standard dataset. (we do not want to tranform every epoch)
        if valid_run:

            batch_size = min(100, subsetsize)

            r = torch.Generator()
            r.manual_seed(0) #ensure that the same testset is always used when generating random corruptions

            precompute_loader = DataLoader(
                random_corrupted_testset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=0, #because of some pickle error with multiprocessing
                worker_init_fn=seed_worker,
                generator=r,
                drop_last=False
            )
            
            # Collect all batches into tensors (no double for loop needed!)
            all_samples = []
            all_labels = []
            
            for batch_samples, batch_labels in precompute_loader:
                all_samples.append(batch_samples)
                all_labels.append(batch_labels)
            
            # Concatenate all batches into single tensors
            all_samples_tensor = torch.cat(all_samples, dim=0)
            all_labels_tensor = torch.cat(all_labels, dim=0)
            
            # Use TensorDataset - much more efficient than ListDataset
            random_corrupted_testset = TensorDataset(all_samples_tensor, all_labels_tensor)

                                
        c_datasets.append(random_corrupted_testset)

        return c_datasets

    def load_data_c(self, subset, subsetsize, valid_run):

        c_datasets = []
        #c-corruption benchmark: https://github.com/hendrycks/robustness
        corruptions_c = np.asarray(np.loadtxt(os.path.join(self.c_labels_path, "c-labels.txt"), dtype=list))
        
        np.random.seed(self.run) # to make subsamples reproducible
        torch.manual_seed(self.run)
        random.seed(self.run)
        global fixed_worker_rng #impulse noise augmentations sk-learn function needs a separate rng for reproducibility
        fixed_worker_rng = np.random.default_rng()

        if self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption
            
            csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/cifar_c_bar.csv'))
            corruptions_bar = csv_handler.read_corruptions()

            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            
            for corruption, set in corruptions:

                if self.validontest:
                    subtestset = self.testset
                    np_data_c = np.load(os.path.abspath(f'{self.data_path}/{self.dataset}-{set}/{corruption}.npy'), mmap_mode='r')
                    np_data_c = np.array(np.array_split(np_data_c, 5))

                    if subset == True:
                        selected_indices = np.random.choice(len(self.testset), subsetsize, replace=False)
                        subtestset = Subset(self.testset, selected_indices)
                        np_data_c = [intensity_dataset[selected_indices] for intensity_dataset in np_data_c]
                    concat_intensities = ConcatDataset([CustomDataset(intensity_data_c, subtestset, self.resize, self.transforms_preprocess) for intensity_data_c in np_data_c])
                    c_datasets.append(concat_intensities)

                else:
                    c_datasets = self.precompute_and_append_c_data(set, c_datasets, corruption, csv_handler, subset, subsetsize, valid_run)
                    
        elif self.dataset in ['ImageNet', 'TinyImageNet', 'ImageNet-100']:

            csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/imagenet_c_bar.csv'))
            corruptions_bar = np.asarray(csv_handler.read_corruptions())
            
            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            
            for corruption, set in corruptions:
                
                if self.validontest:
                    intensity_datasets = [torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}-{set}/' + corruption + '/' + str(intensity)),
                                                                        transform=self.transforms_preprocess) for intensity in range(1, 6)]
                    if subset == True:
                        selected_indices = np.random.choice(len(intensity_datasets[0]), subsetsize, replace=False)
                        intensity_datasets = [Subset(intensity_dataset, selected_indices) for intensity_dataset in intensity_datasets]
                    concat_intensities = ConcatDataset(intensity_datasets)
                    c_datasets.append(concat_intensities)

                else:
                    c_datasets = self.precompute_and_append_c_data(set, c_datasets, corruption, csv_handler, subset, subsetsize, valid_run)

        else:
            if self.validontest:
                print('No c- and c-bar-benchmark available for this dataset. ' \
                'Computing custom corruptions as in CIFAR-C and CIFAR-C-bar.')

            csv_handler = CsvHandler(os.path.abspath(f'{self.c_labels_path}/cifar_c_bar.csv'))
            corruptions_bar = csv_handler.read_corruptions()

            corruptions = [(string, 'c') for string in corruptions_c] + [(string, 'c-bar') for string in corruptions_bar]
            
            for corruption, set in corruptions:
                c_datasets = self.precompute_and_append_c_data(set, c_datasets, corruption, csv_handler, subset, subsetsize, valid_run)

        if valid_run == True:
            c_datasets = ConcatDataset(c_datasets)
            self.c_datasets_dict = {'combined': c_datasets}
        else:
            self.c_datasets_dict = {label: dataset for label, dataset in zip([corr for corr, _ in corruptions], c_datasets)}

        return self.c_datasets_dict

    def get_loader(self, batchsize, grouped_stylization=False):

        self.batchsize = batchsize

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)

        if grouped_stylization == False:
            if self.generated_ratio > 0.0:
                self.CustomSampler = BalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                    batch_size=batchsize)
            else:
                self.CustomSampler = BatchSampler(RandomSampler(self.trainset), batch_size=batchsize, drop_last=False)            
        else:
            self.CustomSampler = ReproducibleBalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                 batch_size=batchsize, epoch=self.epoch)

        self.trainloader = DataLoader(self.trainset, pin_memory=True, batch_sampler=self.CustomSampler,
                                    num_workers=self.number_workers, worker_init_fn=seed_worker, 
                                    generator=g, persistent_workers=False)
        
        val_workers = self.number_workers if self.dataset in ['ImageNet', 'ImageNet-100'] else 0
        self.testloader = DataLoader(self.testset, batch_size=batchsize, pin_memory=True, num_workers=val_workers)

        return self.trainloader, self.testloader
    

    def update_set(self, epoch, start_epoch, grouped_stylization=False, config=None):
        
        if config:
            self.update_transforms(stylize_prob_orig=config.get("stylize_prob_real", None), 
                            stylize_prob_syn=config.get("stylize_prob_synth", None), 
                            alpha_min_orig=config.get("alpha_min_real", None), 
                            alpha_min_syn=config.get("alpha_min_synth", None), 
                            RandomEraseProbability=config.get('RandomEraseProbability', None))

        if grouped_stylization == False:
            if ((self.generated_ratio != 0.0 or self.stylization_gen is not None or self.stylization_orig is not None) and epoch != 0 and epoch != start_epoch) or config is not None:
                # This should be updated when config gives new transforms parameters, when there is generated data or when there is stylization
                del self.trainset

                self.load_augmented_traindata(self.target_size, generated_ratio=self.generated_ratio, epoch=epoch, robust_samples=self.robust_samples, grouped_stylization=False)
        else:    
            if ((self.generated_ratio != 0.0) and epoch != 0 and epoch != start_epoch) or config is not None:
                # This should be updated when config gives new transforms parameters, when there is generated data or when there is stylization
                self.load_augmented_traindata(self.target_size, generated_ratio=config["synth_ratio"], epoch=epoch, robust_samples=self.robust_samples, grouped_stylization=True)
            elif (self.stylization_gen is not None or self.stylization_orig is not None) and epoch != 0 and epoch != start_epoch:
                self.trainset.set_epoch(epoch)

        del self.trainloader
        gc.collect()

        g = torch.Generator()
        g.manual_seed(self.epoch + self.epochs * self.run)
        self.trainloader = DataLoader(self.trainset, batch_sampler=self.CustomSampler, pin_memory=True, 
                                      num_workers=self.number_workers, worker_init_fn=seed_worker,
                                      generator=g, persistent_workers=False)
        
        return self.trainloader