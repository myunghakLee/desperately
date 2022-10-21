import torch
import torchvision
import torchvision.transforms as transforms


def get_data(batch_size, num_workers=4, resize=256, crop=224):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
    ])

    dataset_val = torchvision.datasets.ImageNet(root= "./Dataset/ImageNet", split='val', transform = transform)
    dataset_train = torchvision.datasets.ImageNet(root= "./Dataset/ImageNet", split='train', transform = transform_train)

    val_loader = torch.utils.data.DataLoader(dataset_val,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                              batch_size=128,
                                              shuffle=True,
                                              num_workers=4)

    return train_loader, val_loader
