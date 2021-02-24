# Deep Learning Training Reference

This repository has one train.py file that has a traning implementation that can be used as reference to train image classification models

Requirements
* Numpy
* Pytorch

Installation
------------
Use pip/conda to install numpy 
```
conda install numpy
```
or
```
pip install numpy
```

See https://numpy.org/install/ on how to install numpy if you are having trouble.

Use https://pytorch.org/get-started/locally/ to install pytorch
the command would look like
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

How to run
----------
Create a folder "./pretrained/\<dataset name\>" and "./pretrained/\<dataset name\>/temp"
i.e. 
```
mkdir pretrained
mkdir pretrained/cifar10
mkdir pretrained/cifar10/temp
```

Run the training program
``` 
python train.py --dataset=cifar10 --epochs=100 --loss=crossentropy --optimizer=adam --arch=resnet18
```

Features
--------
* Customizable train and validation split
* Resume training after stopping (keeps optimizer state, and best model)
* Model naming convention
* Saves models in "./pretrained/\<dataset name\>" and state in "./pretrained/\<dataset name\>/temp"
