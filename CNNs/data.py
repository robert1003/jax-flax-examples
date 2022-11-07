import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

def NumpyDataLoader(dataset, **kwargs):
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    return DataLoader(dataset, collate_fn=numpy_collate, **kwargs)

def get_transform(mean, std):
    def img2np(img):
        img = np.array(img, dtype=np.float32)
        img = (img/255.0 - mean) / std
        return img

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        img2np
    ])
    test_transform = img2np

    return train_transform, test_transform

def get_data(val_split=0.2, train_transform=None, test_transform=None):
    train_dataset = CIFAR10(root='./', train=True, 
            transform=train_transform, download=True)
    test_dataset = CIFAR10(root='./', train=False,
            transform=test_transform, download=True)

    val_size = int(len(train_dataset) * val_split)
    train_size = int(len(train_dataset) - val_size)
    train_set, val_set = random_split(train_dataset, (train_size, val_size),
            generator=torch.Generator().manual_seed(0))

    return train_set, val_set, test_dataset

def get_stats():
    # XXX: indices is only an attribute of SubsetDataset, should make it more general
    train_subset, _, _ = get_data()
    data = np.stack([train_subset.dataset.data[i] for i in train_subset.indices])
    mean = (data/255.0).mean(axis=(0, 1, 2))
    std = (data/255.0).std(axis=(0, 1, 2))
    
    return mean, std

def load_CIFAR10(batch_size, num_workers):
    mean, std = get_stats()
    train_transform, test_transform = get_transform(mean, std)
    train_set, val_set, test_set = get_data(
            0.2, train_transform, test_transform)

    train_loader = NumpyDataLoader(train_set, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)
    val_loader = NumpyDataLoader(val_set, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
    test_loader = NumpyDataLoader(test_set, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def main():
    # TODO: plot and visualize to make sure that transform works
    pass

if __name__ == '__main__':
    main()
