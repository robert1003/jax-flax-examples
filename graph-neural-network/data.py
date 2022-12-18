import numpy as np

import torch_geometric.datasets as datasets
import torch_geometric.transforms as transforms
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj

class NumpyGeoDataLoader:
    def __init__(self, *args, **kwargs):
        self.dataloader = DataLoader(*args, **kwargs)
        self.cur_iter = None

    def __iter__(self):
        self.cur_iter = iter(self.dataloader)
        return self

    def __next__(self):
        batch_ = next(self.cur_iter)
       
        batch = np.array(batch_.batch)
        x = np.array(batch_.x)
        G = np.array(to_dense_adj(batch_.edge_index))
        y = np.array(batch_.y)

        msk = {}
        msk['train_mask'] = np.array(batch_.train_mask)
        msk['val_mask'] = np.array(batch_.val_mask)
        msk['test_mask'] = np.array(batch_.test_mask)

        return (batch, msk, x, G, y)

def load_graph_data(data_name, name, batch_size=128, num_workers=8):
    assert data_name == 'Planetoid' and name == 'Cora'
    dataset = getattr(datasets, data_name)(
            root='../data', name=name, split='full')
   
    dataloader = NumpyGeoDataLoader(dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers)

    return dataset, dataloader

if __name__ == '__main__':
    dataset, dataloader = load_graph_data('Planetoid', 'Cora', batch_size=1)

    print(dataset[0])
    print(next(iter(dataloader)))
