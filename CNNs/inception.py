import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, Callable

'''
Each block consists of four branches:
    1x1 conv
    1x1 conv -> 3x3 conv
    1x1 conv -> 5x5 conv
    3x3 maxpool -> 1x1 conv
then we concatenate the output of those four branches.

The first layer is used for dimensionality reduction.
(1x1 conv will keep the height & weight of img, only
chaning its depth)
'''
class InceptionBlock(nn.Module):
    c_reduce: Dict # require 3x3, 5x5
    c_out: Dict # require 1x1, 3x3, 5x5, max
    act_fn: Callable # an callable activation function

    def _conv_bn(self, out_channel, kernel_size, x, train):
        x = nn.Conv(out_channel, kernel_size, 
                kernel_init=nn.initializers.kaiming_normal(),
                use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)

        return x

    @nn.compact
    def __call__(self, x, train=True):
        # 1x1 conv
        x_1x1 = self._conv_bn(self.c_out['1x1'], (1, 1), x, train)

        # 1x1 conv -> 3x3 conv
        x_3x3 = self._conv_bn(self.c_reduce['3x3'], (1, 1),
                    x, train)
        x_3x3 = self._conv_bn(self.c_out['3x3'], (3, 3),
                    x_3x3, train)

        # 1x1 conv -> 5x5 conv
        x_5x5 = self._conv_bn(self.c_reduce['5x5'], (1, 1),
                    x, train)
        x_5x5 = self._conv_bn(self.c_out['5x5'], (5, 5),
                    x_5x5, train)

        # 3x3 maxpool -> 1x1 conv
        x_max = nn.max_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        x_max = self._conv_bn(self.c_out['max'], (1, 1),
                    x_max, train)

        # concat all outputs
        x_out = jnp.concatenate([x_1x1, x_3x3, x_5x5, x_max], axis=-1)

        return x_out

'''
GoogleNet consists of two part:
    1. A conv-batchnorm layer that convert the input channel
       to the desired channel.
    2. Nine InceptionBlock stacked sequentially.

Since we are classifying CIFAR 10 instead of ImageNet, 
we can reduce the number of filters in 5x5 layers.
'''
class GoogleNet_cifar10(nn.Module):
    num_classes: int
    act_fn: Callable

    def _build_args(self,
            reduce_3x3,
            reduce_5x5,
            out_1x1,
            out_3x3,
            out_5x5,
            out_max):
        c_reduce = {'3x3': reduce_3x3, '5x5': reduce_5x5}
        c_out = {'1x1': out_1x1, '3x3': out_3x3, '5x5': out_5x5, 
                'max': out_max}
        return c_reduce, c_out, self.act_fn

    @nn.compact
    def __call__(self, x, train=True):
        # Convert input channel to desired channel
        x = nn.Conv(64, kernel_size=(3, 3), 
            kernel_init=nn.initializers.kaiming_normal(), use_bias=False)(x)
        x = nn.BatchNorm()(x, use_running_average=not train)
        x = self.act_fn(x)

        # Nine InceptionBlock
        inceptionBlocks = [
            InceptionBlock(*self._build_args(32, 16, 16, 32, 8, 8)),
            InceptionBlock(*self._build_args(32, 16, 24, 48, 12, 12)),
            lambda x, train: nn.max_pool(x, (3, 3), strides=(2, 2)),
            InceptionBlock(*self._build_args(32, 16, 24, 48, 12, 12)),
            InceptionBlock(*self._build_args(32, 16, 24, 48, 16, 16)),
            InceptionBlock(*self._build_args(32, 16, 16, 48, 16, 16)),
            InceptionBlock(*self._build_args(32, 16, 32, 48, 24, 24)),
            lambda x, train: nn.max_pool(x, (3, 3), strides=(2, 2)),
            InceptionBlock(*self._build_args(8, 16, 32, 64, 16, 16)),
            InceptionBlock(*self._build_args(8, 16, 32, 64, 16, 16))
        ]
        for block in inceptionBlocks:
            x = block(x, train=train)

        x = x.mean(axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)

        return x

def main():
    from data import load_CIFAR10
    train_loader, val_loader, test_loader = load_CIFAR10(batch_size=128, num_workers=8)

    from module import FlaxModule
    import jax
    model = GoogleNet_cifar10(10, nn.relu)
    key = jax.random.PRNGKey(0)
    rng, key = jax.random.split(key, 2)
    module = FlaxModule(model, rng, next(iter(train_loader))[0])

    from trainer import FlaxTrainer
    trainer = FlaxTrainer()
    train_state = trainer.fit(module, train_loader, val_loader, epochs=200)


if __name__ == '__main__':
    main()
