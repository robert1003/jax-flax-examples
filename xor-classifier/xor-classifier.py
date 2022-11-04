### (Optional) prevent jax from preallocating GPU mem
### ref: https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from torch.utils.data import Dataset, DataLoader

### Data - use dataset from pytorch
class XORDataset(Dataset):
    def __init__(self, size, seed, std=0.1):
        super().__init__()
        self.size = size
        self._gen_data(np.random.RandomState(seed=seed), std)

    def _gen_data(self, rng, std):
        self.data = rng.randint(low=0, high=2, size=(self.size, 2)).astype(np.float32)
        self.data += rng.normal(loc=0.0, scale=std, size=self.data.shape)
        self.label = (self.data[:, 0] != self.data[:, 1]).astype(np.int32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

### DataLoader - also use DataLoader from pytorch, convert output to NumPy
### ref: https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html
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

### Model
class XORClassifier(nn.Module):
    hidden_dim : int
    output_dim : int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

### Optimizer
def SGDOptimizer(lr):
    return optax.sgd(learning_rate=lr)

### Loss
def calc_loss_acc(state, param, batch):
    X, y = batch
    logits = state.apply_fn(param, X).squeeze(axis=-1)
    y_hat = (logits > 0).astype(jnp.int32) # sigmoid(x)>0.5 -> x>0

    loss = optax.sigmoid_binary_cross_entropy(logits, y).mean()
    acc = (y_hat == y).astype(jnp.float32).mean()

    return loss, acc

### Train Loop
@jax.jit
def train_one_step(state, batch):
    grad_fn = jax.value_and_grad(calc_loss_acc, argnums=1, has_aux=True)
    (loss, acc), grads = grad_fn(state, state.params, batch)
    state = state.apply_gradients(grads=grads)

    return state, loss, acc

def train_one_epoch(state, dataloader):
    batch_loss = []
    batch_acc = []
    for batch in dataloader:
        state, loss, acc = train_one_step(state, batch)
        batch_loss.append(loss.item())
        batch_acc.append(acc.item())

    return state, batch_loss, batch_acc

### Eval Loop
@jax.jit
def eval_one_step(state, batch):
    loss, acc = calc_loss_acc(state, state.params, batch)

    return loss, acc

def eval(state, dataloader):
    tot_loss, tot_acc, tot_cnt = 0, 0, 0
    for batch in dataloader:
        loss, acc = eval_one_step(state, batch)
        batch_size = batch[0].shape[0]

        tot_loss += loss.item() * batch_size
        tot_acc += acc.item() * batch_size
        tot_cnt += batch_size

    return tot_loss / tot_cnt, tot_acc / tot_cnt

### Arguments
def parse_args():
    parser = argparse.ArgumentParser(description='xor-classifier')

    parser.add_argument('-d', '--data_size', type=int, default=10000,
            help='number of training data')
    parser.add_argument('-l', '--lr', type=float, default=1e-3,
            help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=10,
            help='number of train epochs')
    parser.add_argument('-s', '--seed', type=int, default=0,
            help='seed for random generator')
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpt',
            help='checkpoint folder')

    return parser.parse_args()

def main():
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    # data
    rng, train_data_rng, eval_data_rng = jax.random.split(rng, 3)
    train_dataset = XORDataset(size=args.data_size, seed=train_data_rng, std=0.1)
    train_dataloader = NumpyDataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_dataset = XORDataset(size=500, seed=eval_data_rng, std=0.1)
    eval_dataloader = NumpyDataLoader(eval_dataset, batch_size=64, shuffle=False)

    # model
    model = XORClassifier(hidden_dim=10, output_dim=1)
    rng, model_rng = jax.random.split(rng, 2)
    sample_input = next(iter(train_dataloader))[0]
    params = model.init(model_rng, sample_input)

    # optimizer
    optimizer = SGDOptimizer(args.lr)

    # train state
    model_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # train!
    print('='*10, 'Train', '='*10)
    for epoch in range(1, args.epochs+1):
        model_state, loss, acc = train_one_epoch(model_state, train_dataloader)
        print('Epoch {:02}: Loss={:.2f} Acc={:.2f}'.format(epoch, loss[-1], acc[-1]))

    # eval!
    print('='*10, 'Eval', '='*10)
    loss, acc = eval(model_state, eval_dataloader)
    print('Loss={:.2f} Acc={:.2f}'.format(loss, acc))

    # save model
    checkpoints.save_checkpoint(ckpt_dir=args.ckpt_dir, target=model_state, 
            step=len(train_dataloader)*args.epochs, prefix='XORClassifier', overwrite=True)

    # load model and test if eval result is same
    # note: target here is optional, just to match object keys
    model_state = checkpoints.restore_checkpoint(ckpt_dir=args.ckpt_dir, target=model_state, prefix='XORClassifier')

    print('='*10, 'Eval from checkpoint', '='*10)
    loss, acc = eval(model_state, eval_dataloader)
    print('Loss={:.2f} Acc={:.2f}'.format(loss, acc))

if __name__ == '__main__':
    main()
