import constants as cons
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import datasets, transforms


def import_ISIC():
    """ Import ISIC Dataset """

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    train_dataset = datasets.ImageFolder("path/to/train_data", transform=train_transform)
    val_dataset = datasets.ImageFolder("path/to/val_data", transform=val_transform)

    train_loader = DataLoader(train_dataset, 
                              batch_size=cons.BATCH_SIZE_TRAIN,
                              shuffle=True, 
                              num_workers=4
                              )
    val_loader = DataLoader(val_dataset, 
                            batch_size=cons.BATCH_SIZE_VAL,
                            shuffle=False, 
                            num_workers=4
                            )
    
    return train_loader, val_loader







