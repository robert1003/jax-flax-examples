'''
In flax.linen.nn.Module, `kernel_init` is a function with type
`Callable[[PRNGKey, Shape, Dtype], Array] which will be used to
initialize each pytree leaf.

Math of Xavier init and Kaming init is given at 
https://pouannes.github.io/blog/initialization/#mjx-eqn-eqfwd_K
'''
import jax
import jax.numpy as jnp

import numpy as np

'''
Const weight
'''
def const_init(c=0.0):
    return lambda key, shape, dtype: c * jnp.ones(shape, dtype=dtype)

'''
Const variance
'''
def const_var_init(std=0.01):
    return lambda key, shape, dtype: std * jax.random.normal(key, shape, dtype=dtype)

'''
Xavier initialization - this initialization try to maintain
constant variance of output and gradient of each layer.

(suitable for activation: Identity, Tanh)
'''
def xavier_normal_init(key, shape, dtype):
    std = np.sqrt(2/(shape[0]+shape[-1]))
    return std * jax.random.normal(key, shape, dtype=dtype)

'''
Kaming initialization - same goal as Xavier, however the
derivation take the effect of activation function into account.

Note that first layer should not use this, since input did not
pass through an activation.

(suitable for activation: ReLU)
'''
def kaming_normal_init(key, shape, dtype):
    std = np.sqrt(2/shape[0])
    return std * jax.random.normal(key, shape, dtype=dtype)

def main():
    # TODO: do visualization of activation & gradients of each initialization

if __name__ == '__main__':
    main()
