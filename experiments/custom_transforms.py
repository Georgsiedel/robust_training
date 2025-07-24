import random
import torch
import torch.cuda.amp
import torch.utils
from torch.utils.data import DataLoader, Subset
import torchvision.transforms.v2 as transforms_v2
import torchvision.transforms as transforms
from run_0 import device
import gc
import experiments.eval_corruption_transforms as c
import torch
from PIL import Image
import numpy as np
import time
from experiments.utils import plot_images
import experiments.style_transfer as style_transfer
from experiments.custom_datasets import StylizedTensorDataset
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2


def get_transforms_map(strat_name, re, dataset, factor, grouped=False, style_path="../data/style_feats_adain_1000.npy"):
    transform_manager = TransformFactory(dataset, factor, re, style_path)
    if grouped:
        transf = transform_manager.get_transforms_grouped(strat_name)
    else:
        transf = transform_manager.get_transforms(strat_name)
    return transf

class TransformFactory:
    def __init__(self, dataset, factor, re, style_path):
        self.dataset = dataset
        self.factor = factor
        self.re = re
        self.TAc = CustomTA_color()
        self.TAg = CustomTA_geometric()
        self.style_path = style_path

    def _stylization(self, probability=0.95, alpha_min=0.2, alpha_max=1.0):
        vgg, decoder = style_transfer.load_models()
        style_feats = style_transfer.load_feat_files(self.style_path)
        pixels = 224 if self.dataset == 'ImageNet' else 32 * self.factor
        return style_transfer.NSTTransform(style_feats, vgg, decoder, alpha_min=alpha_min, alpha_max=alpha_max, probability=probability, pixels=pixels)

    def get_transforms_grouped(self, strat_name):
        if strat_name == "StyleTransfer50alpha00":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.0, alpha_max=0.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha02":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.2, alpha_max=0.2)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha04":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.4)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha06":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.6, alpha_max=0.6)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha08":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.8, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha01-10":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha05-10":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha04-07":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha05-08":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha08-10":
            return (BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.8, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer100alpha10":
            return (BatchStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer90alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer80alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer70alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer60alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer40alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer30alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer20alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer10alpha10":
            return (BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer100alpha05-08":
            return (BatchStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer90alpha05-08":
            return (BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer80alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer70alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer60alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer40alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer30alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer20alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer10alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer100alpha04-07":
            return BatchStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer90alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer80alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer70alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer60alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer40alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer30alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer20alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "StyleTransfer10alpha04-07":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re
        elif strat_name == "TAorStyle90alpha10":
            return BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=0.95, alpha_min=1.0, alpha_max=1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle80alpha10":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle70alpha10":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle60alpha10":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle50alpha10":
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle40alpha10":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle30alpha10":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle20alpha10":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle10alpha10":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle90alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle80alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle70alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle60alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle50alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle40alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle30alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle20alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle10alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle90alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle80alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle70alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle60alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle50alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle40alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle30alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle20alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle10alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle10alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle20alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle30alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle40alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle50alpha05-08":
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle10alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle20alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle30alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle40alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle50alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle60alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle70alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle80alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle90alpha01-10":
            return BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle100alpha01-10":
            return BatchStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle10alpha10":
            return BatchStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle20alpha10":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle30alpha10":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle40alpha10":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle50alpha10":
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle60alpha10":
            return BatchStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle70alpha10":
            return BatchStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle80alpha10":
            return BatchStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle90alpha10":
            return BatchStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle100alpha10":
            return BatchStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "AugMixandStyle20alpha10":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.AugMix(), self.re]), transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixandStyle30alpha10":
            return BatchStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.AugMix(), self.re]), transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixandStyle40alpha1-10":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.AugMix(), self.re]), transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixorStyle20alpha1-10":
            return BatchStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixorStyle40alpha1-10":
            return BatchStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "StyleorColorAndGeometricOrRE":
            random_choice = RandomChoiceTransforms([self.TAg, transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'), transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)], [0.7, 0.15, 0.15])
            combined_transform = transforms.Compose([self.TAc, random_choice])
            return BatchStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.5, 1.0)), random_choice, combined_transform
        elif strat_name == "TAorRE":
            return None, None, RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(), transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'), transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)], [0.8, 0.1, 0.1])
        elif strat_name == "TrivialAugmentWide":
            return None, None, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "RandAugment":
            return None, None, transforms.Compose([transforms_v2.RandAugment(), self.re])
        elif strat_name == "AutoAugment":
            return None, None, transforms.Compose([transforms_v2.AutoAugment(), self.re])
        elif strat_name == "AugMix":
            return None, None, transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == 'None':
            return None, None, self.re
        else:
            print('Training augmentation strategy', strat_name, 'could not be found. Proceeding without augmentation strategy.')
            return None, self.re, self.re
        
    def get_transforms(self, strat_name):
        if strat_name == "StyleTransfer50alpha00":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.0, alpha_max=0.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha02":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.2, alpha_max=0.2)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha04":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.4)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha06":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.6, alpha_max=0.6)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha08":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.8, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha01-10":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha05-10":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha04-07":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.4, alpha_max=0.7)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha05-08":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer50alpha08-10":
            return (DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.8, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer100alpha10":
            return (DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer90alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer80alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer70alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer60alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer40alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer30alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer20alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer10alpha10":
            return (DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=1.0, alpha_max=1.0)), self.re, self.re)
        elif strat_name == "StyleTransfer100alpha05-08":
            return (DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer90alpha05-08":
            return (DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re)
        elif strat_name == "StyleTransfer80alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer70alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer60alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer40alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer30alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer20alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer10alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.5, alpha_max=0.8)), self.re, self.re
        elif strat_name == "StyleTransfer100alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer90alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer80alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer70alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer60alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer40alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer30alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer20alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "StyleTransfer10alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(probability=1.0, alpha_min=0.1, alpha_max=1.0)), self.re, self.re
        elif strat_name == "TAorStyle90alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(probability=0.95, alpha_min=1.0, alpha_max=1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle80alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle70alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle60alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle50alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle40alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle30alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle20alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle10alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle90alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle80alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle70alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle60alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle50alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle40alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle30alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle20alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle10alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle90alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle80alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle70alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle60alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle50alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle40alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle30alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle20alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAorStyle10alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), self.re, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle10alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle20alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle30alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle40alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle50alpha05-08":
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.5, 0.8)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle10alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle20alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle30alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle40alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle50alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle60alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle70alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle80alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle90alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle100alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle10alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.1, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle20alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle30alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle40alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle50alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle60alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.6, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle70alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.7, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle80alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.8, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle90alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.9, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "TAandStyle100alpha10":
            return DatasetStyleTransforms(stylized_ratio=1.0, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re]), transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "AugMixandStyle20alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.AugMix(), self.re]), transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixandStyle30alpha10":
            return DatasetStyleTransforms(stylized_ratio=0.3, batch_size=50, transform_style=self._stylization(0.95, 1.0, 1.0)), transforms.Compose([transforms_v2.AugMix(), self.re]), transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixandStyle40alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), transforms.Compose([transforms_v2.AugMix(), self.re]), transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixorStyle20alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.2, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "AugMixorStyle40alpha01-10":
            return DatasetStyleTransforms(stylized_ratio=0.4, batch_size=50, transform_style=self._stylization(0.95, 0.1, 1.0)), self.re, transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == "StyleorColorAndGeometricOrRE":
            random_choice = RandomChoiceTransforms([self.TAg, transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'), transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)], [0.7, 0.15, 0.15])
            combined_transform = transforms.Compose([self.TAc, random_choice])
            return DatasetStyleTransforms(stylized_ratio=0.5, batch_size=50, transform_style=self._stylization(0.95, 0.5, 1.0)), random_choice, combined_transform
        elif strat_name == "TAorRE":
            return None, None, RandomChoiceTransforms([transforms_v2.TrivialAugmentWide(), transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value='random'), transforms_v2.RandomErasing(p=1.0, scale=(0.02, 0.4), value=0)], [0.8, 0.1, 0.1])
        elif strat_name == "TrivialAugmentWide":
            return None, None, transforms.Compose([transforms_v2.TrivialAugmentWide(), self.re])
        elif strat_name == "RandAugment":
            return None, None, transforms.Compose([transforms_v2.RandAugment(), self.re])
        elif strat_name == "AutoAugment":
            return None, None, transforms.Compose([transforms_v2.AutoAugment(), self.re])
        elif strat_name == "AugMix":
            return None, None, transforms.Compose([transforms_v2.AugMix(), self.re])
        elif strat_name == 'None':
            return None, None, self.re
        else:
            print('Training augmentation strategy', strat_name, 'could not be found. Proceeding without augmentation strategy.')
            return None, self.re, self.re


class PilToNumpy(object):
    def __init__(self, as_float=False, scaled_to_one=False):
        self.as_float = as_float
        self.scaled_to_one = scaled_to_one
        assert (not scaled_to_one) or (as_float and scaled_to_one),\
                "Must output a float if rescaling to one."

    def __call__(self, image):
        if not self.as_float:
            return np.array(image).astype(np.uint8)
        elif not self.scaled_to_one:
            return np.array(image).astype(np.float32)
        else:
            return np.array(image).astype(np.float32) / 255

class NumpyToPil(object):
    def __init__(self):
        pass

    def __call__(self, image):
        return Image.fromarray(image)

def build_transform_c_bar(name, severity, dataset, resize):
    assert dataset in ['CIFAR10', 'CIFAR100', 'ImageNet', 'TinyImageNet', 'GTSRB', 'PCAM', 'EuroSAT'],\
            "Only cifar and imagenet image resolutions are supported."
    
    if dataset in ['CIFAR10', 'CIFAR100', 'GTSRB']: 
        im_size = 32
    elif dataset in ['TinyImageNet', 'EuroSAT']: 
        im_size = 64
    elif dataset in ['EuroSAT']:
        im_size = 96
    else:
        im_size = 224

    if resize:
        im_size = 224

    transform_c_bar_list = [
    c.SingleFrequencyGreyscale,
    c.CocentricSineWaves,
    c.PlasmaNoise,
    c.CausticNoise,
    c.PerlinNoise,
    c.BlueNoise,
    c.BrownishNoise,
    c.TransverseChromaticAbberation,
    c.CircularMotionBlur,
    c.CheckerBoardCutOut,
    c.Sparkles,
    c.InverseSparkles,
    c.Lines,
    c.BlueNoiseSample,
    c.PinchAndTwirl,
    c.CausticRefraction,
    c.Ripple
    ]   

    transform_c_bar_dict = {t.name : t for t in transform_c_bar_list}
    
    return transform_c_bar_dict[name](severity=severity, im_size=im_size)


def transform_c(image, severity=1, corruption_name=None, corruption_number=-1):
    """This function returns a corrupted version of the given image.
    
    Args:
        image (numpy.ndarray):      image to corrupt; a numpy array in [0, 255], expected datatype is np.uint8
                                    expected shape is either (height x width x channels) or (height x width); 
                                    width and height must be at least 32 pixels;
                                    channels must be 1 or 3;
        severity (int):             strength with which to corrupt the image; an integer in [1, 5]
        corruption_name (str):      specifies which corruption function to call, must be one of
                                        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                                        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                                        'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                                        'speckle_noise', 'gaussian_blur', 'spatter', 'saturate';
                                    the last four are validation corruptions
        corruption_number (int):    the position of the corruption_name in the above list; an integer in [0, 18]; 
                                        useful for easy looping; 15, 16, 17, 18 are validation corruption numbers
    Returns:
        numpy.ndarray:              the image corrupted by a corruption function at the given severity; same shape as input
    """
    corruption_tuple = (c.gaussian_noise, 
                        c.shot_noise, 
                        c.impulse_noise, 
                        c.defocus_blur,
                        c.glass_blur, 
                        c.motion_blur, 
                        c.zoom_blur, 
                        c.snow, 
                        c.frost, 
                        c.fog,
                        c.brightness, 
                        c.contrast, 
                        c.elastic_transform, 
                        c.pixelate,
                        c.jpeg_compression, 
                        c.speckle_noise, 
                        c.gaussian_blur, 
                        c.spatter,
                        c.saturate)

    corruption_dict = {corr_func.__name__: corr_func for corr_func in
                   corruption_tuple}

    if not isinstance(image, np.ndarray):
        raise AttributeError('Expecting type(image) to be numpy.ndarray')
    if not (image.dtype.type is np.uint8):
        raise AttributeError('Expecting image.dtype.type to be numpy.uint8')
        
    if not (image.ndim in [2,3]):
        raise AttributeError('Expecting image.shape to be either (height x width) or (height x width x channels)')
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    
    height, width, channels = image.shape

    if height == 32:
        scale = 'cifar'
    elif 32 < height <= 96:
        scale = 'tin'
    else: 
        scale = 'in'
    
    if (height < 32 or width < 32):
        raise AttributeError('Image width and height must be at least 32 pixels')
    
    if not (channels in [1,3]):
        raise AttributeError('Expecting image to have either 1 or 3 channels (last dimension)')
        
    if channels == 1:
        image = np.stack((np.squeeze(image),)*3, axis=-1)
    
    if not severity in [1,2,3,4,5]:
        raise AttributeError('Severity must be an integer in [1, 5]')
    
    if not (corruption_name is None):
        image_corrupted = corruption_dict[corruption_name](Image.fromarray(image),
                                                       severity, scale)
    elif corruption_number != -1:
        image_corrupted = corruption_tuple[corruption_number](Image.fromarray(image),
                                                          severity, scale)
    else:
        raise ValueError("Either corruption_name or corruption_number must be passed")

    return np.uint8(image_corrupted)

class RandomCommonCorruptionTransform:
    def __init__(self, set, corruption_name, dataset, csv_handler, resize):
        self.corruption_name = corruption_name
        self.set = set
        self.dataset = dataset
        self.csv_handler = csv_handler
        self.TtoPIL = transforms.ToPILImage()
        self.PILtoNP = PilToNumpy()
        self.NPtoPIL = NumpyToPil()
        self.ToTensor = transforms.ToTensor()
        self.resize = resize

    def __call__(self, img):
        severity = random.randint(1, 5)

        if self.set == 'c':
            img_np = self.PILtoNP(self.TtoPIL(img))
            corrupted_img = self.ToTensor(self.NPtoPIL(transform_c(img_np, severity=severity, corruption_name=self.corruption_name)))
        elif self.set == 'c-bar':
            severity_value = self.csv_handler.get_value(self.corruption_name, severity)
            
            comp = transforms.Compose([self.TtoPIL,
                                self.PILtoNP,
                build_transform_c_bar(self.corruption_name, severity_value, self.dataset, self.resize),
                self.NPtoPIL,
                self.ToTensor
                ])
            corrupted_img = comp(img)

        return corrupted_img

class DatasetStyleTransforms:
    def __init__(self, transform_style, batch_size, stylized_ratio):
        """
        Args:
            transform_style: Callable to apply stylization.
            batch_size: Batch size for tensors passed to transform_style.
            stylized_ratio: Fraction of images to stylize (0 to 1).
        """
        self.transform_style = transform_style
        self.batch_size = batch_size
        self.stylized_ratio = stylized_ratio

    def __call__(self, dataset):
        """
        Stylize a fraction of images in the dataset and return a new dataset.

        Args:
            dataset: PyTorch Dataset to process.

        Returns:
            stylized_dataset: A new TensorDataset with stylized images.
        """

        num_images = len(dataset)
        num_stylized = int(num_images * self.stylized_ratio)
        stylized_indices = torch.randperm(num_images)[:num_stylized]
        
        # Create a Subset with the stylized indices
        stylized_subset = Subset(dataset, stylized_indices)

        # DataLoader for processing the stylized subset
        loader = DataLoader(stylized_subset, batch_size=self.batch_size, shuffle=False)
        
        # Use zeros as placeholders for non-stylized images and labels
        sample_image, _ = dataset[0]  # Get sample shape from the dataset
        stylized_images = torch.zeros((num_stylized, *sample_image.shape), dtype=sample_image.dtype)

        # Iterate over the DataLoader and process stylized images
        for batch_indices, (images, _) in zip(loader.batch_sampler, loader):  
            # Apply the transformation to the batch
            transformed_images = self.transform_style(images)

            # Store the transformed images and labels in their original positions
            stylized_images[batch_indices] = transformed_images

        # Delete intermediary variables to save memory
        del loader, stylized_subset
        gc.collect()

        style_mask = torch.zeros(num_images, dtype=torch.bool)
        style_mask[stylized_indices] = True
        style_mask = style_mask.tolist()

        # Return the stylized dataset
        return StylizedTensorDataset(dataset, stylized_images, stylized_indices), style_mask

class BatchStyleTransforms:
    def __init__(self, transform_style, batch_size, stylized_ratio):
        """
        Args:
            transform_style: Callable to apply stylization.
            batch_size: Batch size for tensors passed to transform_style.
            stylized_ratio: Fraction of images to stylize (0 to 1).
        """
        self.transform_style = transform_style
        self.batch_size = batch_size
        self.stylized_ratio = stylized_ratio

    def __call__(self, images):
        """
        Stylize a tensor batch of images.

        Args:
            images (torch.Tensor): A tensor batch of images with shape (batch_size, *image_shape).

        Returns:
            Tuple[torch.Tensor, List[bool]]: 
                - A tensor batch of images where a fraction is stylized, with the same shape as input.
                - A boolean list indicating which images were stylized.
        """

        num_images = len(images)
        num_stylized = int(num_images * self.stylized_ratio)

        if num_stylized > 0:
            # Select indices of images to stylize
            stylized_indices = torch.randperm(num_images)[:num_stylized]
            images_to_stylize = images[stylized_indices]

            # Process the subset of images in smaller batches
            for i in range(0, len(images_to_stylize), self.batch_size):
                # Apply the style transform to the batch
                batch = images_to_stylize[i:i + self.batch_size]
                images_to_stylize[i:i + self.batch_size] = self.transform_style(batch)

            # Replace the original images with the stylized ones
            images[stylized_indices] = images_to_stylize

            # Create the style mask
            style_mask = torch.zeros(num_images, dtype=torch.bool)
            style_mask[stylized_indices] = True
        else:
            # If no images are stylized, create an all-false style mask
            style_mask = torch.zeros(num_images, dtype=torch.bool)

        # Return the modified images and style mask
        return images, style_mask


class RandomChoiceTransforms:
    def __init__(self, transforms, p):
        assert len(transforms) == len(p), "The number of transforms and probabilities must match."

        self.transforms = transforms
        self.p = p

    def __call__(self, x):
        choice = random.choices(self.transforms, self.p)[0]
        return choice(x)

class EmptyTransforms:
    def __init__(self):
        pass  # No operations needed for empty transforms.

    def __call__(self, x):
        return x

class StylizedChoiceTransforms:
    def __init__(self, transforms, probabilities):
        assert len(transforms) == len(probabilities) == 2, "The number of transforms and probabilities must be 2, one before Stylization and one without Stylization."
        self.transforms = transforms
        self.probabilities = probabilities

    def __call__(self, x):
        choice = random.choices(list(self.transforms.items()), list(self.probabilities.values()))[0]
        type, function = choice[0], choice[1]
        if type == "before_stylization":
            return function(x), True
        elif type == "before_no_stylization":
            return function(x), False
        else:
            raise ValueError("Invalid dict key for stylized choice transform.")

class CustomTA_color(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "Brightness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Color": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Contrast": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Sharpness": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "Posterize": (lambda num_bins, height, width: (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6))).round().int(), False),
    "Solarize": (lambda num_bins, height, width: torch.linspace(1.0, 0.0, num_bins), False),
    "AutoContrast": (lambda num_bins, height, width: None, False),
    "Equalize": (lambda num_bins, height, width: None, False)
    }

class CustomTA_geometric(transforms_v2.TrivialAugmentWide):
    _AUGMENTATION_SPACE = {
    "Identity": (lambda num_bins, height, width: None, False),
    "ShearX": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "ShearY": (lambda num_bins, height, width: torch.linspace(0.0, 0.99, num_bins), True),
    "TranslateX": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "TranslateY": (lambda num_bins, height, width: torch.linspace(0.0, 32.0, num_bins), True),
    "Rotate": (lambda num_bins, height, width: torch.linspace(0.0, 135.0, num_bins), True),
    }

class On_GPU_Transforms():
    def __init__(self, transforms_orig_gpu, transforms_orig_post, transforms_gen_gpu, transforms_gen_post):

        self.transforms_orig_gpu = transforms_orig_gpu
        self.transforms_orig_post = transforms_orig_post
        self.transforms_gen_gpu = transforms_gen_gpu
        self.transforms_gen_post = transforms_gen_post

    def __call__(self, x, sources, apply):
        
        if self.transforms_orig_gpu == None and self.transforms_gen_gpu == None:
            return x

        x = x.to(device)

        if x.size(0) == 2 * sources.size(0):
            sources = torch.cat([sources, sources], dim=0)
        
        orig_mask = (sources) & (apply)
        if orig_mask.any():
            if apply[sources].sum().item() > 200:
                #split the batch into chunks if the number of images to be stylized is more than 180 cause VRAM
                chunks = torch.split(x[orig_mask], 200)
                processed_chunks = [self.transforms_orig_gpu(chunk) for chunk in chunks]
                x[orig_mask] = torch.cat(processed_chunks, dim=0)
            else:
                x[orig_mask] = self.transforms_orig_gpu(x[orig_mask])
        
        gen_mask = (~sources) & (apply)
        if gen_mask.any():
            if apply[~sources].sum().item() > 200:
                #split the batch into chunks if the number of images to be stylized is more than 180 cause VRAM
                chunks = torch.split(x[gen_mask], 200)
                processed_chunks = [self.transforms_gen_gpu(chunk) for chunk in chunks]
                x[gen_mask] = torch.cat(processed_chunks, dim=0)
            else:
                x[gen_mask] = self.transforms_gen_gpu(x[gen_mask])
        
        x = x.cpu()
        if orig_mask.any():
            x[orig_mask] = torch.stack([self.transforms_orig_post(image) for image in x[orig_mask]])
        if gen_mask.any():
            x[gen_mask] = torch.stack([self.transforms_gen_post(image) for image in x[gen_mask]])
        x = x.to(device)

        return x