from typing import Union, Tuple, Callable
from tqdm import tqdm

import optax
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state, checkpoints

class FlowModule:
    def __init__(self, name: str):
        self.name = name
    
    @staticmethod
    def build_train_step_fn() -> Callable:
        def train_step(
                state: train_state.TrainState,
                rng: Union[jax.Array, jax.random.PRNGKeyArray],
                batch: Tuple[jax.Array]
                ):
            imgs, _ = batch
            loss_fn = lambda params: state.apply_fn({'params': params}, imgs, rng, testing=False)
            (loss, rng), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)

            return state, rng, loss

        return jax.jit(train_step)
        
    @staticmethod
    def build_val_step_fn():
        def val_step(state, rng, batch):
            return state.apply_fn({'params': state.params}, batch[0], rng, testing=False)

        return jax.jit(val_step)

    @staticmethod
    def build_pred_step_fn():
        def pred_step(state, rng, batch):
            return state.apply_fn({'params': state.params}, batch[0], rng, testing=True)

        return jax.jit(pred_step)

    @staticmethod
    def build_lr_scheduler(lr, steps):
        return optax.exponential_decay(init_value=lr, transition_steps=steps,
                decay_rate=0.99, end_value=0.01*lr)

    @staticmethod
    def build_optimizer(lr_scheduler):
        return optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(lr_scheduler)
            )
    
    @staticmethod
    def build_model(rng, model, example):
        init_rng, flow_rng, rng = jax.random.split(rng, 3)
        return model.init(init_rng, example, flow_rng)['params']

    @staticmethod
    def build_state(model, params, opt):
        return train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=opt)

    @staticmethod
    def sample(state, sample_fn, rng, img_shape, z_init=None):
        imgs = state.apply_fn({'params': state.params}, img_shape, rng, z_init, method=sample_fn)

        return imgs

    @staticmethod
    def encode(state, encode_fn, rng, imgs):
        z, ldj, rng = state.apply_fn({'params': state.params}, imgs, rng, method=encode_fn)

        return z, ldj, rng

    @staticmethod
    def train_epoch(state, rng, dataloader, train_step, epoch):
        avg_loss = 0.
        tqdm_bar = tqdm(dataloader, leave=False, desc=f'Epoch {epoch}')
        for batch in tqdm_bar:
            state, rng, loss = train_step(state, rng, batch)
            tqdm_bar.set_postfix({'loss': loss.item()})
            avg_loss += loss
        avg_loss /= len(dataloader)

        return state, avg_loss.item()

    @staticmethod
    def val_epoch(state, rng, dataloader, val_step):
        tot_loss, tot_size = [], []
        for batch in tqdm(dataloader, desc='Val', leave=False):
            loss, rng = val_step(state, rng, batch)
            tot_loss.append(loss)
            tot_size.append(batch[0].shape[0])

        tot_loss = np.stack(jax.device_get(tot_loss))
        tot_size = np.stack(tot_size)
        avg_loss = (tot_loss * tot_size).sum() / tot_size.sum()

        return avg_loss

    @staticmethod
    def pred_epoch(state, rng, dataloader, pred_step):
        tot_loss, tot_size = [], []
        for batch in tqdm(dataloader, desc='Pred', leave=False):
            loss, rng = pred_step(state, rng, batch)
            tot_loss.append(loss)
            tot_size.append(batch[0].shape[0])

        tot_loss = np.stack(jax.device_get(tot_loss))
        tot_size = np.stack(tot_size)
        avg_loss = (tot_loss * tot_size).sum() / tot_size.sum()

        return avg_loss

    @staticmethod
    def save_model(ckpt_dir, state, step):
        checkpoints.save_checkpoint(
                ckpt_dir=ckpt_dir, target=state.params, step=step)

    @staticmethod
    def load_model(ckpt_dir, state, model):
        params = checkpoints.restore_checkpoint(
                ckpt_dir=ckpt_dir, target=state.params)

        return train_state.TrainState.create(
                apply_fn=model.apply, params=params, tx=state.tx)
