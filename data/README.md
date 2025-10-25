The current path in the repository to the data folder leads outside the workspace, one directory level higher.
CIFAR10 / CIFAR100 are downloaded automatically to there, all ImageNet, TinyImageNet, -c and -c-bar datasets need to be added.

Generated data usage requires the respective images in this folder in numpy format: "{dataset}-add-1m-dm.npz" 
as can be downloaded from here: https://github.com/wzekai99/DM-Improves-AT 
or generated here: https://github.com/NVlabs/edm

Stylemix in our implementation requires encoded image features from the painter-by-numbers dataset to be put into the data repository, named "style_feats_adain_1000.npy", as can be downloaded from here: ttps://zenodo.org/records/16279015

This folder contains the label datasets for the c- and c-bar datasets, which must also be put into the data repository
