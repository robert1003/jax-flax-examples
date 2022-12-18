import numpy as np
import torch
import torchvision.datasets
from torch.utils.data import DataLoader, random_split
from torch.utils.data import random_split

def NumpyDataLoader(dataset, **kwargs):
    '''
    A Numpy wrapper over torch.utils.data.DataLoader
        
    Args:
        dataset: torch.utils.data.Dataset instance
        kwargs: argument for torch.utils.data.DataLoader

    Return:
        torch.utils.data.DataLoader
    '''
    def numpy_collate(batch):
        if isinstance(batch[0], np.ndarray):
            return np.stack(batch)
        elif isinstance(batch[0], (tuple, list)):
            transposed = zip(*batch)
            return [numpy_collate(samples) for samples in transposed]
        else:
            return np.array(batch)

    return DataLoader(dataset, collate_fn=numpy_collate, **kwargs)

def load_img_data(data_name, transform, split_arr, batch_size=128, num_workers=8):
    '''
    Helper function to load data from torchvision.datasets

    Args:
        dataset: dataset name specified in torchvision.datasets
        transform: transform to apply on data
        split_arr: a list contains the split size e.g. [50000, 10000]
        batch_size: batch size of dataloader
        num workers: num of workers for dataloader

    Return:
        train, val, test dataset and its corresponding dataloader
    '''
    train_dataset = getattr(torchvision.datasets, data_name)(
            root='../data', train=True, transform=transform, download=True)
    train_set, val_set = random_split(train_dataset, split_arr,
            generator=torch.Generator().manual_seed(42))
    test_set = getattr(torchvision.datasets, data_name)(
            root='../data', train=False, transform=transform, download=True)

    train_dataloader = NumpyDataLoader(train_set, batch_size=batch_size,
            shuffle=True, num_workers=num_workers)
    val_dataloader = NumpyDataLoader(val_set, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)
    test_dataloader = NumpyDataLoader(test_set, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)

    return (train_set, val_set, test_set), (train_dataloader, val_dataloader, test_dataloader)

