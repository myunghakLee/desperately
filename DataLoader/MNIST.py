# +
import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import torch.utils.data as data_utils

# -

def get_data(batch_size, num_workers=4, resize=256, crop=224, percent = 1.0):

    normalize = transforms.Normalize(mean=[0.5074, 0.4867, 0.4411],
                                     std=[0.2675, 0.2566, 0.2763])

    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        normalize,
    ])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3,1,1)),
            normalize,
    ])
    
    dataset_train = torchvision.datasets.MNIST(root="./Dataset/", train=True, transform=transform_train, download=True)
    indices = np.random.choice(range(len(dataset_train)), int(len(dataset_train) * percent))
    dataset_train = data_utils.Subset(dataset_train, indices)
    
    
    dataset_val = torchvision.datasets.MNIST(root="./Dataset/", train=False, transform=transform_test, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    return train_loader, val_loader
