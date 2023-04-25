""" 
Manage importing the correct Dataset
"""
from Attack_fig6.constants import BATCH_SIZE_TRAIN, BATCH_SIZE_TEST
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision import transforms


def import_dataset(dataset_to_import):
    if dataset_to_import in datasets_dict.keys():
        train_loader, test_loader = datasets_dict[dataset_to_import]()
    else:
        print(f'Dataset "{dataset_to_import}" is not currently available.\n \
                Dataset available: {datasets_dict.keys()}')
    return train_loader, test_loader


def import_RetinaMNIST():
    """ RetinaMNIST """

    info = INFO['retinamnist']
    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([transforms.ToTensor()])
    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=True)
    test_dataset = DataClass(split='test', transform=data_transform, download=True)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)
    return train_loader, test_loader


def import_MNIST():
    """ MNIST """
    train_loader = DataLoader(
        datasets.MNIST('/files/',
                       train=True,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                           #  transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True
    )
    test_loader = DataLoader(
        datasets.MNIST('/files/',
                       train=False,
                       download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE_TEST,
        shuffle=True
    )
    return train_loader, test_loader


datasets_dict = {'retinamnist': import_RetinaMNIST,
                 'mnist': import_MNIST}
