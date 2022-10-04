import argparse
import jax
from jax import random
from jax import numpy as jnp
import numpy as np

import time

def predict(W, b, x):
    return jnp.dot(x, W) + b

@jax.jit
def mse(W, b, x_batch, y_batch):
    def l2_error(x, y):
        y_pred = predict(W, b, x)
        return jnp.inner(y - y_pred, y - y_pred) / 2.0

    return jnp.mean(jax.vmap(l2_error)(x_batch, y_batch), axis=0)

def gen_data(key, n_samples, x_dim, y_dim):
    key_param, key_sample = random.split(key)

    key_W, key_b = random.split(key_param)
    W = random.normal(key_W, (x_dim, y_dim))
    b = random.normal(key_b, (y_dim,))

    key_sample, key_noise = random.split(key_sample)
    x_samples = random.normal(key_sample, (n_samples, x_dim))
    y_samples = predict(W, b, x_samples) + random.normal(key_noise, (n_samples, y_dim))

    return W, b, x_samples, y_samples

@jax.jit
def update(W, b, x_batch, y_batch, lr):
    loss, (grad_W, grad_b) = jax.value_and_grad(mse, (0, 1))(W, b, x_batch, y_batch)
    W, b = W - grad_W * lr, b - grad_b * lr

    return loss, W, b
    
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
    W, b, x_samples, y_samples = gen_data(key, args.num_samples, args.x_dim, args.y_dim)

    W_hat, b_hat = jnp.zeros_like(W), jnp.zeros_like(b)
    for epoch in range(args.train_epochs):
        loss, W_hat, b_hat = update(W_hat, b_hat, x_samples, y_samples, args.learning_rate)

        if (epoch + 1) % 100 == 0:
            print('Loss step {}: {:.5f}'.format(epoch + 1, loss))

    print('Loss for true W, b: {:.5f}'.format(mse(W, b, x_samples, y_samples)))
    print('L2 between W and W_hat', ((W - W_hat)**2).mean())
    print('L2 beteen b and b_hat', ((b - b_hat)**2).mean())
    print('Linf between W and W_hat', jnp.abs(W - W_hat).max())
    print('Linf between b and b_hat', jnp.abs(b - b_hat).max())


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()

    print('Execution finished in {} seconds'.format(end_time - start_time))
