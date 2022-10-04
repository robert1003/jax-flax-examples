import argparse
from typing import Any, Callable, Sequence

import jax
from jax import random
from jax import numpy as jnp
import optax
from flax import linen as nn

def gen_data(key, n_samples, x_dim, y_dim):
    def predict(W, b, x):
        return jnp.dot(x, W) + b

    key_param, key_sample = random.split(key)

    key_W, key_b = random.split(key_param)
    W = random.normal(key_W, (x_dim, y_dim))
    b = random.normal(key_b, (y_dim,))

    key_sample, key_noise = random.split(key_sample)
    x_samples = random.normal(key_sample, (n_samples, x_dim))
    y_samples = predict(W, b, x_samples) + random.normal(key_noise, (n_samples, y_dim))

    return W, b, x_samples, y_samples

class SimpleDense(nn.Module):
    dim: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.dim))
        bias = self.param('bias', self.bias_init, (self.dim,))

        return jnp.dot(x, kernel) + bias

class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, dim in enumerate(self.features):
            x = SimpleDense(dim, name=f"dense{i}")(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--x_dim', type=int, default=10, help='')
    parser.add_argument('--y_dim', type=int, default=5, help='')
    parser.add_argument('--num_samples', type=int, default=20, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='')
    parser.add_argument('--train_epochs', type=int, default=100, help='')

    return parser.parse_args()

def main():
    args = parse_args()
    key = random.PRNGKey(args.seed)
    key_data, key_model = random.split(key)
    W, b, x_samples, y_samples = gen_data(key_data, args.num_samples, args.x_dim, args.y_dim)

    model = MLP(features=[args.y_dim])
    x = random.uniform(key_data, (args.x_dim,)) # dummy data
    output, params = model.init_with_output(key_model, x)
    print('initialized parameters:')
    print(jax.tree_util.tree_map(lambda x: x.shape, params))
    print('output:', output)

    # optimizer
    optimizer = optax.sgd(learning_rate=args.learning_rate)
    opt_state = optimizer.init(params)

    # criterion (loss function) and grad function
    @jax.jit
    def mse(params, x_batch, y_batch):
        def l2_error(x, y):
            y_pred = model.apply(params, x)
            return jnp.inner(y - y_pred, y - y_pred) / 2.0

        return jnp.mean(jax.vmap(l2_error)(x_batch, y_batch), axis=0)
    loss_grad_fn = jax.value_and_grad(mse)

    # train loop
    for epoch in range(args.train_epochs):
        loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if (epoch + 1) % 100 == 0:
            print('Loss step {}: {:.5f}'.format(epoch + 1, loss_val))

    from flax.core import freeze, unfreeze
    true_params = freeze({'params': {'dense0': {'kernel': W, 'bias': b}}})
    print('Loss for true W, b: {:.5f}'.format(mse(true_params, x_samples, y_samples)))  

if __name__ == '__main__':
    main()
