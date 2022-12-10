import optax
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from flax.training import train_state
from flax.core.frozen_dict import unfreeze

class FlaxModule:
    def __init__(self, model, rng, sample_input):
        self.model = model
        self.rng = rng
        self.sample_input = sample_input
        self.criterion = optax.softmax_cross_entropy_with_integer_labels

    def _forward_pass(self, state, param, X):
        logits, mutated_vars = state.apply_fn(param, X,
                mutable=['batch_stats'])
        y_hat = jnp.argmax(logits, axis=-1).astype(jnp.int32)

        return logits, y_hat

    def _calc_loss_acc(self, state, param, batch):
        X, y = batch
        logits, y_hat = self._forward_pass(state, param, X)
        loss = self.criterion(logits, y).mean()
        acc = (y_hat == y).astype(jnp.float32).mean()

        return loss, acc

    @partial(jax.jit, static_argnums=(0,))
    def training_step(self, state, batch, batch_idx):
        grad_fn = jax.value_and_grad(self._calc_loss_acc, argnums=1, has_aux=True)
        (loss, acc), grads = grad_fn(state, state.params, batch)
        return grads, (loss.item(), acc.item())

    @partial(jax.jit, static_argnums=(0,))
    def validation_step(self, state, batch, batch_idx):
        loss, acc = self._calc_loss_acc(state, batch)
        return loss.item(), acc.item()

    def predict_step(self, state, batch, batch_idx):
        X = batch
        logits = state.apply_fn(state.param, X)
        y_hat = jnp.argmax(logits, axis=-1).astype(jnp.int32)

        return y_hat

    def configure_train_state(self):
        optimizer = optax.adam(learning_rate=1e-3)
        self.rng, model_rng = jax.random.split(self.rng, 2)
        params = self.model.init(model_rng, self.sample_input)
        state = train_state.TrainState.create(apply_fn=self.model.apply,
                params=params, tx=optimizer)

        return state
