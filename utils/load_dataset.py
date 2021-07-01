"""
@author: Deepak Ravikumar Tatachar, Sangamesh Kodge
@copyright: Nanoelectronics Research Laboratory
"""

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms.transforms import Resize
from utils.tinyimagenet import TinyImageNet
from utils.noise import UniformNoise, GaussianNoise

class Dict_To_Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_transform(test_transform,
                  train_transform,
                  val_transform,
                  mean,
                  std,
                  augment,
                  img_dim,
                  padding_crop):

    if(test_transform == None):
        test_transform = transforms.Compose([
                                                transforms.Resize((img_dim, img_dim)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std),
                                            ])

    if(val_transform == None):    
        val_transform = test_transform

    if(train_transform == None):
        if augment:
            train_transform = transforms.Compose([
                                                    transforms.RandomCrop(img_dim, padding=padding_crop),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean, std),
                                                ])
        else:
            train_transform = test_transform

    return train_transform, val_transform, test_transform

def load_dataset(dataset='CIFAR10',
                 train_batch_size=128,
                 test_batch_size=128,
                 val_split=0.0,
                 augment=True,
                 padding_crop=4,
                 shuffle=True,
                 random_seed=None,
                 device='cpu',
                 resize_shape=None,
                 mean=None,
                 std=None,
                 train_transform=None,
                 test_transform=None,
                 val_transform=None):
    '''
    Inputs
    dataset -> CIFAR10, CIFAR100, TinyImageNet, ImageNet
    train_batch_size -> batch size for training dataset
    test_batch_size -> batch size for testing dataset
    val_split -> percentage of training data split as validation dataset
    augment -> bool flag for Random horizontal flip and shift with padding
    padding_crop -> units of pixel shift
    shuffle -> bool flag for shuffling the training and testing dataset
    random_seed -> fixes the shuffle seed for reproducing the results
    device -> cuda device or cpu
    return -> bool for returning the mean, std, img_size
    '''
    # Load dataset
    # Use the following transform for training and testing
    if (dataset.lower() == 'mnist'):
        if(mean == None):
            mean = [0.1307]
        if(std == None):
            std = [0.3081]
        img_dim = 28
        img_ch = 1
        num_classes = 10
        num_worker = 4
        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\MNIST\\'

        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.MNIST(root=root,
                                              train=True,
                                              download=True,
                                              transform=train_transform)

        valset = torchvision.datasets.MNIST(root=root,
                                            train=True,
                                            download=True,
                                            transform=val_transform)

        testset = torchvision.datasets.MNIST(root=root,
                                             train=False,
                                             download=True,
                                             transform=test_transform)
        
    elif(dataset.lower() == 'cifar10'):
        if(mean == None):
            mean = [0.4914, 0.4822, 0.4465]
        if(std == None):
            std = [0.2023, 0.1994, 0.2010]

        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 2
        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\CIFAR10\\'

        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.CIFAR10(root=root,
                                                train=True,
                                                download=True,
                                                transform=train_transform)

        valset = torchvision.datasets.CIFAR10(root=root,
                                              train=True,
                                              download=True,
                                              transform=val_transform)

        testset = torchvision.datasets.CIFAR10(root=root,
                                               train=False,
                                               download=True,
                                               transform=test_transform)

    # http://cs231n.stanford.edu/tiny-imagenet-200.zip
    elif(dataset.lower() == 'tinyimagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]
        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\TinyImageNet\\'
        img_dim = 64
        if(resize_shape == None):
            resize_shape = (32, 32)
            img_dim = 32
        img_ch = 3
        num_classes = 200
        num_worker = 4
        
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = TinyImageNet(root=root, transform=test_transform, train=True) 
        valset = TinyImageNet(root=root, transform=test_transform, train=True) 
        testset = TinyImageNet(root=root, transform=test_transform, train=False)
  
    elif(dataset.lower() == 'svhn'):
        if(mean == None):
            mean = [0.4376821,  0.4437697,  0.47280442]
        if(std == None):
            std = [0.19803012, 0.20101562, 0.19703614]
        img_dim = 32
        img_ch = 3
        num_classes = 10
        num_worker = 2
        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\SVHN\\'

        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.SVHN(root=root,
                                             split='train',
                                             download=True,
                                             transform=train_transform)

        valset = torchvision.datasets.SVHN(root=root,
                                           split='train',
                                           download=True,
                                           transform=val_transform)

        testset = torchvision.datasets.SVHN(root=root,
                                            split='test',
                                            download=True, 
                                            transform=test_transform)

    elif(dataset.lower() == 'lsun'):
        if(mean == None):
            mean = [0.5071, 0.4699, 0.4326]
        if(std == None):
            std = [0.2485, 0.2492, 0.2673]
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 10
        num_worker = 0
       
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\LSUN\\'
        trainset = torchvision.datasets.LSUN(root=root,
                                             classes='val',
                                             transform=train_transform)

        valset = torchvision.datasets.LSUN(root=root,
                                           classes='val',
                                           transform=val_transform)

        testset = torchvision.datasets.LSUN(root=root,
                                            classes='val',
                                            transform=test_transform)
    elif(dataset.lower() == 'places365'):
        if(mean == None):
            mean = [0.4578, 0.4413, 0.4078]
        if(std == None):
            std = [0.2435, 0.2418, 0.2622]
        
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 365
        num_worker = 4

        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\Places365\\'
        trainset = torchvision.datasets.Places365(root=root,
                                                  split='train-standard',
                                                  transform=train_transform,
                                                  download=False)

        valset = torchvision.datasets.Places365(root=root,
                                                split='train-standard',
                                                transform=val_transform,
                                                download=False)

        testset = torchvision.datasets.Places365(root=root,
                                                 split='val',
                                                 transform=test_transform,
                                                 download=False)

    elif(dataset.lower() == 'cifar100'):
        if(mean == None):
            mean = [0.5071, 0.4867, 0.4408]
        if(std == None):
            std = [0.2675, 0.2565, 0.2761]

        img_dim = 32
        img_ch = 3
        num_classes = 100
        num_worker = 4
        root = 'C:\\Users\\dravikum\\Documents\\Datasets\\CIFAR100\\'

        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.CIFAR100(root=root,
                                                 train=True,
                                                 download=True,
                                                 transform=train_transform)

        valset = torchvision.datasets.CIFAR100(root=root,
                                               train=True,
                                               download=True,
                                               transform=val_transform)

        testset = torchvision.datasets.CIFAR100(root=root,
                                                train=False,
                                                download=True,
                                                transform=test_transform)  

    elif(dataset.lower() == 'textures'):
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
        img_dim = 32
        if(resize_shape == None):
            resize_shape = (img_dim, img_dim)
        img_ch = 3
        num_classes = 47
        num_worker = 4
        datapath ='C:\\Users\\dravikum\\Documents\\Datasets\\Textures\\'
                
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.ImageFolder(root=datapath + 'images',
                                                    transform=train_transform)

        valset = torchvision.datasets.ImageFolder(root=datapath + 'images',
                                                  transform=val_transform)

        testset = torchvision.datasets.ImageFolder(root=datapath + 'images',
                                                   transform=test_transform)

    elif(dataset.lower() == 'u-noise'):
        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4

        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
                
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)
        trainset = UniformNoise(size=(img_ch, img_dim, img_dim))
        valset = UniformNoise(size=(img_ch, img_dim, img_dim))
        testset = UniformNoise(size=(img_ch, img_dim, img_dim))

    elif(dataset.lower() == 'g-noise'):
        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4
                
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")
                
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = GaussianNoise(size=(img_ch, img_dim, img_dim))
        valset = GaussianNoise(size=(img_ch, img_dim, img_dim))
        testset = GaussianNoise(size=(img_ch, img_dim, img_dim))

    elif(dataset.lower() == 'isun'):
        if(mean == None or std == None):
            raise ValueError("Mean and std for textures no supported yet!")

        img_dim = 32
        img_ch = 3
        num_classes = 1
        num_worker = 4
        datapath ='C:\\Users\\dravikum\\Documents\\Datasets\\iSUN\\'
                
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.ImageFolder(root=datapath,
                                                    transform=train_transform)

        valset = torchvision.datasets.ImageFolder(root=datapath,
                                                  transform=val_transform)

        testset = torchvision.datasets.ImageFolder(root=datapath,
                                                   transform=test_transform)

    elif(dataset.lower() == 'imagenet'):
        if(mean == None):
            mean = [0.485, 0.456, 0.406]
        if(std == None):
            std = [0.229, 0.224, 0.225]

        img_dim = 224
        img_ch = 3
        num_classes = 1000
        num_worker = 40
        datapath ='/local/a/imagenet/imagenet2012/'
        #datapath = 'Path for image net goes here' # Set path here
                
        train_transform, val_transform, test_transform = get_transform(test_transform,
                                                                       train_transform,
                                                                       val_transform,
                                                                       mean,
                                                                       std,
                                                                       augment,
                                                                       img_dim,
                                                                       padding_crop)

        trainset = torchvision.datasets.ImageFolder(root=datapath + 'train',
                                                    transform=train_transform)

        valset = torchvision.datasets.ImageFolder(root=datapath + 'train',
                                                  transform=val_transform)

        testset = torchvision.datasets.ImageFolder(root=datapath + 'val',
                                                   transform=test_transform)
    else:
        # Right way to handle exception in python 
        # see https://stackoverflow.com/questions/2052390/manually-raising-throwing-an-exception-in-python
        # Explains all the traps of using exception, does a good job!! I mean the link :)
        raise ValueError("Unsupported dataset")
    
    # Split the training dataset into training and validation sets
    print('\nForming the sampler for train and validation split')
    num_train = len(trainset)
    ind = list(range(num_train))
    split = int(val_split * num_train)

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(ind)
    
    train_idx, val_idx = ind[split:], ind[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Load dataloader
    print('Loading data to the dataloader \n')
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_size=train_batch_size,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=num_worker)

    val_loader =  torch.utils.data.DataLoader(valset,
                                              batch_size=train_batch_size,
                                              sampler=val_sampler,
                                              pin_memory=True,
                                              num_workers=num_worker)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_worker)

    transforms_dict = {
                          'train': train_transform,
                          'val': val_transform,
                          'test': test_transform
                      }

    return_dict = {
                    'name': dataset,
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'test_loader': test_loader,
                    'num_classes': num_classes,
                    'mean' : mean,
                    'std': std,
                    'img_dim': img_dim,
                    'img_ch': img_ch,
                    'train_batch_size': train_batch_size,
                    'test_batch_size': test_batch_size,
                    'val_split': val_split,
                    'padding_crop': padding_crop,
                    'augment': augment,
                    'random_seed': random_seed,
                    'shuffle': shuffle,
                    'transforms': Dict_To_Obj(**transforms_dict),
                  }

    dataset_obj = Dict_To_Obj(**return_dict)
    return dataset_obj
