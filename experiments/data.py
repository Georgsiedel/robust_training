import random
import os
import time
import json
import gc

import torch
import torch.cuda.amp
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import Subset, ConcatDataset, RandomSampler, BatchSampler, DataLoader
import numpy as np
import experiments.custom_transforms as custom_transforms
from run_exp import device
from experiments.utils import plot_images, CsvHandler
from experiments.custom_datasets import SubsetWithTransform, GeneratedDataset, AugmentedDataset, ListDataset, CustomDataset 
from experiments.custom_datasets import BalancedRatioSampler, GroupedAugmentedDataset, ReproducibleBalancedRatioSampler, StyleDataset

def normalization_values(batch, dataset, normalized, manifold=False, manifold_factor=1):

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
        elif (dataset == 'ImageNet' or dataset == 'TinyImageNet'):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        else:
            print('no normalization values set for this dataset')
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

class DataLoading():
    def __init__(self, dataset, validontest=True, epochs=200, generated_ratio=0.0, 
                 resize = False, run=0, number_workers=0, kaggle=False):
        self.dataset = dataset
        self.generated_ratio = generated_ratio
        self.resize = resize
        self.run = run
        self.epochs = epochs
        self.validontest = validontest
        self.number_workers = number_workers
        self.kaggle = kaggle

        if dataset == 'CIFAR10':
            self.factor = 1
        elif dataset == 'CIFAR100':
            self.factor = 1
        elif dataset == 'ImageNet':
            self.factor = 1
        elif dataset == 'TinyImageNet':
            self.factor = 2
        
        file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paths.json")
        with open(file_path, "r") as f:
            self.path = json.load(f)
            suffix = '_kaggle' if self.kaggle else ''

            self.data_path = self.path.get(f"data{suffix}")
            self.c_labels_path = self.path.get(f"c_labels{suffix}")
            self.trained_models_path = self.path.get(f"trained_models{suffix}")
            self.style_feats_path = self.path.get(f"style_feats{suffix}")

    def create_transforms(self, train_aug_strat_orig, train_aug_strat_gen, RandomEraseProbability=0.0, grouped_stylization=False):
        # list of all data transformations used
        t = transforms.ToTensor()
        c32 = transforms.RandomCrop(32, padding=4)
        c64 = transforms.RandomCrop(64, padding=8)
        c224 = transforms.RandomCrop(224, padding=28)
        flip = transforms.RandomHorizontalFlip()
        r224 = transforms.Resize(224, antialias=True)
        r256 = transforms.Resize(256, antialias=True)
        cc224 = transforms.CenterCrop(224)
        rrc224 = transforms.RandomResizedCrop(224, antialias=True)
        re = transforms.RandomErasing(p=RandomEraseProbability, scale=(0.02, 0.4)) #, value='random' --> normally distributed and out of bounds 0-1

        # transformations of validation/test set and necessary transformations for training
        # always done (even for clean images while training, when using robust loss)
        if self.dataset == 'ImageNet':
            self.transforms_preprocess = transforms.Compose([t, r256, cc224])
        elif self.resize == True:
            self.transforms_preprocess = transforms.Compose([t, r224])
        else:
            self.transforms_preprocess = transforms.Compose([t])

        # standard augmentations of training set, without tensor transformation
        if self.dataset == 'ImageNet':
            self.transforms_basic = transforms.Compose([flip])
        elif self.resize:
            self.transforms_basic = transforms.Compose([flip, c224])
        elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
            self.transforms_basic = transforms.Compose([flip, c32])
        elif self.dataset == 'TinyImageNet':
            self.transforms_basic = transforms.Compose([flip, c64])

        self.stylization_orig, self.transforms_orig_after_style, self.transforms_orig_after_nostyle = custom_transforms.get_transforms_map(train_aug_strat_orig, re, self.dataset, self.factor, grouped_stylization, self.style_feats_path)
        self.stylization_gen, self.transforms_gen_after_style, self.transforms_gen_after_nostyle = custom_transforms.get_transforms_map(train_aug_strat_gen, re, self.dataset, self.factor, grouped_stylization, self.style_feats_path)

    def load_base_data(self, test_only=False):

        if self.validontest:

            if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                self.testset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/val'),
                                                                transform=self.transforms_preprocess)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'))

            elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                load_helper = getattr(torchvision.datasets, self.dataset)
                self.testset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=False, download=True,
                                        transform=self.transforms_preprocess)
                if test_only:
                    self.base_trainset = None
                else:
                    self.base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=True, download=True)
                
            else:
                print('Dataset not loadable')
            
            self.num_classes = len(self.testset.classes)
        
        else:
            if self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
                base_trainset = torchvision.datasets.ImageFolder(root=os.path.abspath(f'{self.data_path}/{self.dataset}/train'))
            elif self.dataset == 'CIFAR10' or self.dataset == 'CIFAR100':
                load_helper = getattr(torchvision.datasets, self.dataset)
                base_trainset = load_helper(root=os.path.abspath(f'{self.data_path}'), train=True, download=True)
            else:
                print('Dataset not loadable')  
        
            validsplit = 0.2
            train_indices, val_indices, _, _ = train_test_split(
                range(len(base_trainset)),
                base_trainset.targets,
                stratify=base_trainset.targets,
                test_size=validsplit,
                random_state=self.run)  # same validation split for same runs, but new validation on multiple runs
            self.base_trainset = Subset(base_trainset, train_indices)
            validset = Subset(base_trainset, val_indices)

            self.testset = [(self.transforms_preprocess(data), target) for data, target in validset]
                
            self.num_classes = len(base_trainset.classes)
    
    def load_style_dataloader(self, style_dir, batch_size):
        style_dataset = StyleDataset(style_dir, dataset_type=self.dataset)
        style_loader = DataLoader(style_dataset, batch_size=batch_size, shuffle=False)
        return style_loader

        

    def load_augmented_traindata(self, target_size, epoch=0, robust_samples=0, grouped_stylization=False):
        self.robust_samples = robust_samples
        self.target_size = target_size
        self.generated_dataset = np.load(os.path.abspath(f'{self.data_path}/{self.dataset}-add-1m-dm.npz'),
                                    mmap_mode='r') if self.generated_ratio > 0.0 else None
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

                generated_subset = GeneratedDataset(
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

                generated_subset = GeneratedDataset(
                    self.generated_dataset['image'][generated_indices],
                    self.generated_dataset['label'][generated_indices],
                    transform=self.transforms_preprocess
                )
            else:
                generated_subset = None
            
            self.trainset = GroupedAugmentedDataset(original_subset, generated_subset, self.transforms_basic, self.stylization_orig, 
                                    self.stylization_gen, self.transforms_orig_after_style, self.transforms_gen_after_style, 
                                    self.transforms_orig_after_nostyle, self.transforms_gen_after_nostyle, self.robust_samples, epoch)


    def load_data_c(self, subset, subsetsize, valid_run):

        c_datasets = []
        #c-corruption benchmark: https://github.com/hendrycks/robustness
        corruptions_c = np.asarray(np.loadtxt(os.path.abspath(f'{self.c_labels_path}/c-labels.txt'), dtype=list))
        
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
                    random_corrupted_testset = SubsetWithTransform(self.testset, 
                                                    transform=custom_transforms.RandomCommonCorruptionTransform(set, corruption, self.dataset, csv_handler))
                    if subset == True:
                        selected_indices = np.random.choice(len(self.testset), subsetsize, replace=False)
                        random_corrupted_testset = Subset(random_corrupted_testset, selected_indices)
                    
                    # If valid_run, precompute the transformed outputs and wrap them as a standard dataset. (we do not want to tranform every epoch)
                    if valid_run:
                        if corruption in ['caustic_refraction', 'sparkles']: #compute heavier corruptions

                            r = torch.Generator()
                            r.manual_seed(0) #ensure that the same testset is always used when generating random corruptions

                            precompute_loader = DataLoader(
                                random_corrupted_testset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=self.number_workers,
                                worker_init_fn=seed_worker,
                                generator=r
                            )
                            
                            precomputed_samples = [(sample[0], label[0]) for sample, label in precompute_loader]
                            # Wrap the precomputed samples in a dataset so that further processing sees a standard Dataset object.
                            random_corrupted_testset = ListDataset(precomputed_samples)
                        
                        else:
                            precomputed_samples = [sample for sample in random_corrupted_testset]
                            #Wrap the precomputed samples in a dataset so that further processing sees a standard Dataset object.
                            random_corrupted_testset = ListDataset(precomputed_samples)
                                            
                    c_datasets.append(random_corrupted_testset)
                    

        elif self.dataset == 'ImageNet' or self.dataset == 'TinyImageNet':
            #c-bar-corruption benchmark: https://github.com/facebookresearch/augmentation-corruption

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
                    random_corrupted_testset = SubsetWithTransform(self.testset, 
                                                    transform=custom_transforms.RandomCommonCorruptionTransform(set, corruption, self.dataset, csv_handler))
                    if subset == True:
                        selected_indices = np.random.choice(len(self.testset), subsetsize, replace=False)
                        random_corrupted_testset = Subset(random_corrupted_testset, selected_indices)
                    

                    # If valid_run, precompute the transformed outputs and wrap them as a standard dataset (we do not want to online tranform every epoch)
                    if valid_run:
                        if corruption in ['caustic_refraction', 'sparkles']:  #compute heavier corruptions

                            r = torch.Generator()
                            r.manual_seed(0) #ensure that the same testset is always (run, epoch) used when generating random corruptions

                            precompute_loader = DataLoader(
                                random_corrupted_testset,
                                batch_size=1,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=self.number_workers,
                                worker_init_fn=seed_worker,
                                generator=r
                            )
                            
                            precomputed_samples = [(sample[0], label[0]) for sample, label in precompute_loader]
                            # Wrap the precomputed samples in a dataset so that further processing sees a standard Dataset object.
                            random_corrupted_testset = ListDataset(precomputed_samples)
                        
                        else:
                            precomputed_samples = [sample for sample in random_corrupted_testset]
                            #Wrap the precomputed samples in a dataset so that further processing sees a standard Dataset object.
                            random_corrupted_testset = ListDataset(precomputed_samples)
                                            
                    c_datasets.append(random_corrupted_testset)

        else:
            print('No corrupted benchmark available other than CIFAR10-c, CIFAR100-c, TinyImageNet-c and ImageNet-c.')
            return

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

            self.trainloader = DataLoader(self.trainset, pin_memory=True, batch_sampler=self.CustomSampler,
                                        num_workers=self.number_workers, worker_init_fn=seed_worker, 
                                            generator=g, persistent_workers=False)
            
        else:
            self.CustomSampler = ReproducibleBalancedRatioSampler(self.trainset, generated_ratio=self.generated_ratio,
                                                 batch_size=batchsize, epoch=self.epoch)

            self.trainloader = DataLoader(self.trainset, pin_memory=True, batch_sampler=self.CustomSampler,
                                      num_workers=self.number_workers, worker_init_fn=seed_worker, 
                                        generator=g, persistent_workers=False)
        
        val_workers = self.number_workers if self.dataset=='ImageNet' else 0
        self.testloader = DataLoader(self.testset, batch_size=batchsize, pin_memory=True, num_workers=val_workers)

        return self.trainloader, self.testloader
    

    def update_set(self, epoch, start_epoch, grouped_stylization=False):

        if grouped_stylization == False:
            if (self.generated_ratio != 0.0 or self.stylization_gen is not None or self.stylization_orig is not None) and epoch != 0 and epoch != start_epoch:
                            
                del self.trainset

                self.load_augmented_traindata(self.target_size, epoch=epoch, robust_samples=self.robust_samples, grouped_stylization=False)
        else:    
            if (self.generated_ratio != 0.0) and epoch != 0 and epoch != start_epoch:
                    self.load_augmented_traindata(self.target_size, epoch=epoch, robust_samples=self.robust_samples, grouped_stylization=True)
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