# Jax / Flax Tips

* Preventing JAX from using all GPU mem
```python3
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
```

* Pytorch dataloader to Numpy dataloader ([ref](https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html))
```python3
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
```

* Flax save/load model: [ref](https://github.com/google/flax/discussions/1876)
