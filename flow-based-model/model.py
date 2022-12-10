from typing import Sequence
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn

class ImageFlow(nn.Module):
    flows: Sequence[nn.Module]
    import_samples: int=8

    def __call__(self, x, rng, testing=False):
        if not testing:
            bpd, rng = self._get_likelihood(x, rng)
        else:
            img_ll, rng = self._get_likelihood(
                    x.repeat(self.import_samples,0), rng, return_ll=True)
            img_ll = img_ll.reshape(-1, self.import_samples)
            img_ll = jax.nn.logsumexp(img_ll, axis=-1) - \
                    np.log(self.import_samples)

            bpd = -img_ll*np.log2(np.exp(1)) / np.prod(x.shape[1:])
            bpd = bpd.mean()

        return bpd, rng

    def encode(self, imgs, rng):
        # img -> latent
        z, ldj = imgs, jnp.zeros(imgs.shape[0])
        for flow in self.flows:
            z, ldj, rng = flow(z, ldj, rng, reverse=False)

        return z, ldj, rng

    def _get_likelihood(self, imgs, rng, return_ll=False):
        # img -> latent -> likelihood of latent
        # assume prior is multivariate normal here
        z, ldj, rng = self.encode(imgs, rng)
        log_pz = jax.scipy.stats.norm.logpdf(z).sum(axis=(1,2,3))
        log_px = ldj + log_pz
        nll = -log_px

        npd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
        return (bpd.mean() if not return_ll else log_px), rng

    def sample(self, img_shape, rng, z_init=None):
        # latent (given or gen from prior) -> img
        if z_init is None:
            rng, normal_rng = random.split(rng)
            z = random.normal(normal_rng, shape=img_shape)
        else:
            z = z_init

        ldj = jnp.zeros(img_shape[0])
        for flow in reversed(self.flows):
            z, ldj, rng = flow(z, ldj, rng, reverse=True)

        return z, rng

class Dequantization(nn.Module):
    alpha: float=1e-5
    quants: int=256

    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, ldj, rng = self.dequant(z, ldj, rng)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.quants
            ldj += np.log(self.quants) * np.prod(z.shape[1:])
            z = jnp.floor(z)
            z = jax.lax.clamp(min=0., x=z, max=self.quants-1.).astype(jnp.int32)

        return z, ldj, rng
    
    def sigmoid(self, z, ldj, reverse=False):
        if not reverse:
            ldj += (-z - 2 * jax.nn.softplus(-z)).sum(axis=[1,2,3])
            z = nn.sigmoid(z)
            ldj -= np.log(1 - self.alpha) * np.prod(z.shape[1:])
            z = (z - 0.5*self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-jnp.log(z) - jnp.log(1-z)).sum(axis=[1,2,3])
            z = jnp.log(z) - jnp.log(1-z)

        return z, ldj

    def dequant(self, z, ldj, rng):
        # discrete -> continuous
        z = z.astype(jnp.float32)
        rng, uniform_rng = random.split(rng)
        z = z + random.uniform(uniform_rng, z.shape)
        z = z / self.quants
        ldj -= np.log(self.quants) * np.prod(z.shape[1:])

        return z, ldj, rng

class VariationalDequantization(Dequantization):
    var_flows: Sequence[nn.Module]=None

    def dequant(self, z, ldj, rng):
        z = z.astype(jnp.float32)
        img = (z / 255.0) * 2 - 1

        # uniform prior
        rng, uniform_rng = random.split(rng)
        deq_noise = random.uniform(uniform_rng, z.shape)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        if self.var_flows is not None:
            for flow in self.var_flows:
                deq_noise, ldj, rng = flow(deq_noise, ldj, rng, reverse=False, orig_img=img)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        z = (z + deq_noise) / 256.0
        ldj -= np.log(256.0) * np.prod(z.shape[1:])

        return z, ldj, rng

if __name__ == '__main__':
    orig_img = np.random.random_integers(low=0, high=255, size=(1, 24, 24, 1))
    ldj = jnp.zeros(1,)
    dequant_model = Dequantization()
    dequant_rng = random.PRNGKey(5)
    deq_img, ldj, dequant_rng = dequant_model(orig_img, ldj,
            dequant_rng, reverse=False)
    reconst_img, ldj, dequant_rng = dequant_model(deq_img, ldj,
            dequant_rng, reverse=True)

    print(deq_img.shape, reconst_img.shape)

    d1, d2 = jnp.where(orig_img.squeeze() != reconst_img.squeeze())
    if len(d1) != 0:
        print("Dequantization was not invertible.")
        #for i in range(d1.shape[0]):
        #    print("Original value:", orig_img[0,d1[i],d2[i],0].item())
        #    print("Reconstructed value:", reconst_img[0,d1[i],d2[i],0].item())
    else:
        print("Successfully inverted dequantization")

    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    def visualize_dequantization(quants, prior=None):
        """
        Function for visualizing the dequantization values of discrete values in continuous space
        """
        # Prior over discrete values. If not given, a uniform is assumed
        if prior is None:
            prior = np.ones(quants, dtype=np.float32) / quants
        prior = prior / prior.sum()  # Ensure proper categorical distribution

        inp = jnp.arange(-4, 4, 0.01).reshape(-1, 1, 1, 1)  # Possible continuous values we want to consider
        ldj = jnp.zeros(inp.shape[0])
        dequant_module = Dequantization(quants=quants)
        # Invert dequantization on continuous values to find corresponding discrete value
        out, ldj, _ = dequant_module(inp, ldj, rng=None, reverse=True)
        inp, out, prob = inp.squeeze(), out.squeeze(), jnp.exp(ldj)
        prob = prob * prior[out] # Probability scaled by categorical prior

        # Plot volumes and continuous distribution
        sns.set_style("white")
        fig = plt.figure(figsize=(6,3))
        x_ticks = []
        for v in np.unique(out):
            indices = np.where(out==v)
            color = to_rgb(f"C{v}")
            plt.fill_between(inp[indices], prob[indices], np.zeros(indices[0].shape[0]), color=color+(0.5,), label=str(v))
            plt.plot([inp[indices[0][0]]]*2,  [0, prob[indices[0][0]]],  color=color)
            plt.plot([inp[indices[0][-1]]]*2, [0, prob[indices[0][-1]]], color=color)
            x_ticks.append(inp[indices[0][0]])
        x_ticks.append(inp.max())
        plt.xticks(x_ticks, [f"{x:.1f}" for x in x_ticks])
        plt.plot(inp,prob, color=(0.0,0.0,0.0))
        # Set final plot properties
        plt.ylim(0, prob.max()*1.1)
        plt.xlim(inp.min(), inp.max())
        plt.xlabel("z")
        plt.ylabel("Probability")
        plt.title(f"Dequantization distribution for {quants} discrete values")
        plt.legend()
        plt.savefig("dequant", bbox_inches="tight")

    visualize_dequantization(quants=8)
    visualize_dequantization(quants=8, prior=np.array([0.075, 0.2, 0.4, 0.2, 0.075, 0.025, 0.0125, 0.0125]))
