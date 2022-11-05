import jax
from jax.tree_util import tree_map
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple

PyTree = Any
Params = PyTree
ParamsUpdate = PyTree
Gradient = PyTree
OptState = Tuple

'''
`init`: Setup OptState as a tuple with the given Params
`update`: Calculate ParamsUpdate and new OptState given raw Gradient and Params
'''
@dataclass
class Optimizer:
    init: Callable[[Params], OptState]
    update: Callable[[Gradient, OptState, Optional[Params]], Tuple[ParamsUpdate, OptState]]

'''
SGD:
    w_t = w_{t-1} - \lr * g_t
'''
def SGD(lr):
    def init(params):
        return tuple() # no internal state needed

    def update(gradient, optState, params=None):
        paramUpdate = tree_map(lambda g: -lr*g, gradient)
        return paramUpdate, optState

    return Optimizer(init, update)

'''
SGD with momentum:
    m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g_t
    w_t = w_{t-1} - \lr * m_t
'''
def SGD_momentum(lr, beta_1):
    def init(params):
        m = tree_map(jnp.zeros_like, params)
        return m

    def update(gradient, optState, params=None):
        optState = tree_map(lambda m, g: beta_1*m + (1-beta_1)*g, optState, gradient)
        paramUpdate = tree_map(lambda m: -lr*m, optState)
        return paramUpdate, optState

    return Optimizer(init, update)

'''
RMSprop:
    v_t = \alpha * v_{t-1} + (1 - \alpha) * g_t^2
    w_t = w_{t-1} - \lr * g_t / (sqrt(v_t) + \eps)
'''
def RMSprop(lr, alpha, eps):
    def init(params):
        v = tree_map(jnp.zeros_like, params)
        return v

    def update(gradient, optState, params=None):
        optState = tree_map(lambda v, g: alpha*v + (1-alpha)*jnp.power(g, 2), optState, gradient)
        paramUpdate = tree_map(lambda v, g: -lr*g_t / (jnp.sqrt(v) + eps), optState, gradient)
        return paramUpdate, optState

    return Optimizer(init, update)

'''
Adam:
    m_t = \beta_1 * m_{t-1} + (1 - \beta_1) * g_t
    v_t = \beta_2 * v_{t-1} + (1 - \beta_2) * g_t^2
    w_t = w_{t-1} - \lr * (m_t / (1 - \beta_1^t)) / (sqrt((v_t / (1 - \beta_2^t)) + \eps)
'''
def Adam(lr, beta_1, beta_2, eps):
    def init(params):
        step = 0.0
        m = tree_map(jnp.zeros_like, params)
        v = tree_map(jnp.zeros_like, params)
        return (step, m, v)

    def update(gradient, optState, params=None):
        step, m, v = optState
        step += 1
        m = tree_map(lambda m, g: beta_1*m + (1-beta_1)*g, m, gradient)
        v = tree_map(lambda v, g: beta_2*v + (1-beta_2)*jnp.power(g, 2), v, gradient)

        m_corr, v_corr = 1.0 - beta_1**step, 1.0 - beta_2**step
        def _update(m, v):
            m /= m_corr
            v /= v_corr
            return -lr * m / (jnp.sqrt(v) + eps)
        paramUpdate = tree_map(_update, m, v)

        return paramUpdate, (step, m, v)

    return Optimizer(init, update)

def main():
    # TODO: train our optimizer on FashionMNIST or CIFAR10
    # TODO: create some artificial pathological loss surface and illustrate each optimizer's performance

if __name__ == '__main__':
    main()
