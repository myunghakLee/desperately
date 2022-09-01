import torch
import torchvision
import torchvision.transforms as transforms

def get_data(batch_size, num_workers=4, resize=256, crop=224):

    normalize = transforms.Normalize(mean=[0.4464, 0.3856, 0.2927],
                                     std=[0.3013, 0.2488, 0.2729])

    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        normalize,
    ])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])
    
    dataset_train = torchvision.datasets.Flowers102(root="./Dataset/", split='test', transform=transform_train, download=True)
    dataset_val = torchvision.datasets.Flowers102(root="./Dataset/", split='val', transform=transform_test, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    
    return train_loader, val_loader
