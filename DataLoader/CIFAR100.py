import torch
import torchvision
import torchvision.transforms as transforms

# +
def get_data(batch_size, num_workers=4, resize=256, test_resize = 256, crop=224, degree = 0, augmix = False, use_original = False):

#     normalize = transforms.Normalize(mean=[0.5074, 0.4867, 0.4411],
#                                      std=[0.2675, 0.2566, 0.2763])
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.2761504713256840])

    if use_original:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        transform_train = transforms.Compose([
            #transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            normalize
        ])        
    else:
    
        transform_test = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ])

        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(crop),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=color_jitter[0], contrast=color_jitter[1], 
                                       saturation=color_jitter[2], hue=color_jitter[3]),
                transforms.ToTensor(),
                normalize,
        ])
    
    dataset_train = torchvision.datasets.CIFAR100(root="./Dataset/", train=True, transform=transform_train, download=True)
    dataset_val = torchvision.datasets.CIFAR100(root="./Dataset/", train=False, transform=transform_test, download=True)
    
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)
    
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    return train_loader, val_loader
