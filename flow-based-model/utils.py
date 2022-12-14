import torch
import torchvision
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt

from model import CouplingLayer, GatedConvNet, VariationalDequantization, \
        Dequantization, ImageFlow, SqueezeFlow, SplitFlow

def img2np(img):
    img = np.array(img, dtype=np.int32)
    img = np.expand_dims(img, axis=-1)
    return img

def show_imgs(imgs, file_name, title=None, row_size=4):
    # Form a grid of pictures (we use max. 8 columns)
    imgs = np.copy(jax.device_get(imgs))
    num_imgs = imgs.shape[0]
    is_int = (imgs.dtype==np.int32)
    nrow = min(num_imgs, row_size)
    ncol = int(math.ceil(num_imgs/nrow))
    imgs_torch = torch.from_numpy(imgs).permute(0, 3, 1, 2)
    imgs = torchvision.utils.make_grid(imgs_torch, nrow=nrow, pad_value=128 if is_int else 0.5)
    np_imgs = imgs.cpu().numpy()
    # Plot the grid
    plt.figure(figsize=(1.5*nrow, 1.5*ncol))
    plt.imshow(np.transpose(np_imgs, (1,2,0)), interpolation='nearest')
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()

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
