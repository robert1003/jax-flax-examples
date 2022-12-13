import jax
import jax.numpy as jnp
import numpy as np

from model import CouplingLayer, GatedConvNet, VariationalDequantization, \
        Dequantization, ImageFlow, SqueezeFlow, SplitFlow

def img2np(img):
    img = np.array(img, dtype=np.int32)
    img = np.expand_dims(img, axis=-1)
    return img

def create_checkerboard_mask(h, w, invert=False):
    x, y = jnp.arange(h, dtype=jnp.int32), jnp.arange(w, dtype=jnp.int32)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    mask = jnp.fmod(xx+yy, 2)
    mask = mask.astype(jnp.float32).reshape(1, h, w, 1)
    if invert:
        mask = 1 - mask

    return mask

def create_channel_mask(c_in, invert=False):
    mask = jnp.concatenate([
            jnp.ones((c_in//2,), dtype=jnp.float32),
            jnp.zeros((c_in-c_in//2,), dtype=jnp.float32)
        ])
    mask = mask.reshape(1, 1, 1, c_in)
    if invert:
        mask = 1 - mask

    return mask

def create_simple_flow(use_vardeq=True):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(
            network=GatedConvNet(c_out=2, c_hidden=16),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
            c_in=1) for i in range(4)
        ]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]

    for i in range(8):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=32),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
            c_in=1)
        ]

    flow_model = ImageFlow(flow_layers)
    return flow_model

def create_multiscale_flow(use_vardeq=True):
    flow_layers = []
    if use_vardeq:
        vardeq_layers = [CouplingLayer(
            network=GatedConvNet(c_out=2, c_hidden=16),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
            c_in=1) for i in range(4)
        ]
        flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
    else:
        flow_layers += [Dequantization()]

    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=2, c_hidden=32),
            mask=create_checkerboard_mask(h=28, w=28, invert=(i%2==1)),
            c_in=1)
        ]
    flow_layers += [SqueezeFlow()]

    for i in range(2):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=8, c_hidden=48),
            mask=create_channel_mask(c_in=4, invert=(i%2==1)),
            c_in=4)
        ]
    flow_layers += [SplitFlow(), SqueezeFlow()]

    for i in range(4):
        flow_layers += [CouplingLayer(network=GatedConvNet(c_out=16, c_hidden=64),
            mask=create_channel_mask(c_in=8, invert=(i%2==1)),
            c_in=8)
        ]

    flow_model = ImageFlow(flow_layers)
    return flow_model


if __name__ == '__main__':
    print(create_checkerboard_mask(h=8, w=8).repeat(2, -1))
    print(create_channel_mask(c_in=10))
