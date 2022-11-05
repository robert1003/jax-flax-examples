import jax.numpy as jnp
from flax import linen as nn

'''
f(x) = 1 / (1 + exp(-x))
'''
class Sigmoid(nn.Module):
    def __call__(self, x):
        return 1 / (1 + jnp.exp(-x))

'''
f(x) = sinh(x) / cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
'''
class Tanh(nn.Module):
    def __call__(self, x):
        return (jnp.exp(x) - jnp.exp(-x)) / (jnp.exp(x) + jnp.exp(-x))

'''
f(x) = max(0, x)
'''
class ReLU(nn.Module):
    def __call__(self, x):
        return jnp.maximum(0, x)

'''
f(x) = x if x >= 0 else -c*x
'''
class LeakyReLU(nn.Module):
    alpha: float=0.1
    
    def __call__(self, x):
        return jnp.where(x >= 0, x, self.alpha*x)

'''
f(x) = x if x > 0 else c*(exp(x)-1)
'''
class ELU(nn.Module):
    alpha: float=1.0

    def __call__(self, x):
        return alpha * (jnp.exp(x) - 1)

'''
f(x) = 0 if x <= -3 else (x if x >= 3 else x*(x+3)/6)
'''
class Hardswish(nn.Module):
    def __call__(self, x):
        x = jnp.where(x <= -3, 0, x)
        x = jnp.where(x <= 3, x*(x+3)/6, x)
        return x

def main():
    # TODO: visualize activation functions

    # 1. plot activation function and their corresponding gradient

    # 2. use activation funciton in a network and examine gradient
    # magnitude & output value after activation

    # 3. fully train the previous network and output the same 
    # info above

    # 4. show dying ReLU problem by counting the percentage of dead
    # neuron in each layer (for a deep network)

if __name__ == '__main__':
    main()
