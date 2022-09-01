import torch
import torchvision
import torchvision.transforms as transforms

def get_data(batch_size, num_workers=4, resize=256, crop=224):

    normalize = transforms.Normalize(mean=[0.4919, 0.4827, 0.4472],
                                     std=[0.2470, 0.2434, 0.2616])

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

    dataset_train = torchvision.datasets.CIFAR10(root="./Dataset/", train=True, transform=transform_train, download=True)
    dataset_val = torchvision.datasets.CIFAR10(root="./Dataset/", train=False, transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    val_loader = torch.utils.data.DataLoader(dataset_val,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    return train_loader, val_loader
