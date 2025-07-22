run_exp.py calls train.py and eval.py modules from the experiments folder for one or multiple experiment IDs, the seetup of which need to be defined in expereiments/configs/config_{ID}.py

The paths defined in this project reference a "data" and a "trained_models" folder on the same directory level as this project (one directory up from here).

"trained_models" and the "results"-directory use an internal folder structure: /results/{'datasetname'}/{'modelname'} and ./trained_models/{'datasetname'}/{'modelname'}.

The model architectures in /experiments/models contain a parameter "factor" for TinyImageNet's 64x64 images. This model uses the same architecture as for CIFAR 32x32 images, just with a stride=factor=2 in the first convolution. All models inherit a forward pass from ct_model.py to allow normalization as well as noise injections and mixup within the forward pass (and in deeper layers).
