import numpy as np
import torch
from torchvision import datasets, transforms

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()


def get_loaders(
    dir_, batch_size, batch_size_test=None, num_workers=0, transform_fn=None, data_augmentation=True
):
    if batch_size_test is None:
        batch_size_test = batch_size

    if data_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    train_dataset = datasets.CIFAR10(dir_, train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(dir_, train=False, transform=test_transform, download=True)

    if transform_fn is not None:
        data, y = transform_fn(train_dataset.data, train_dataset.targets)
        train_dataset.targets = y
        train_dataset.data = data

        data, y = transform_fn(test_dataset.data, test_dataset.targets)
        test_dataset.targets = y
        test_dataset.data = data

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, test_loader


