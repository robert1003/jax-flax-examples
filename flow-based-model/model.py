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

        bpd = nll * np.log2(np.exp(1)) / np.prod(imgs.shape[1:])
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

class CouplingLayer(nn.Module):
    network: nn.Module # f(x) -> mu, sigma
    mask: np.ndarray
    c_in: int

    def setup(self):
        self.scaling_factor = self.param('scaling_factor',
                nn.initializers.zeros, (self.c_in,))

    def __call__(self, z, ldj, rng, reverse=False, orig_img=None):
        z_in = z*self.mask # mask out 1:d
        if orig_img is None:
            nn_out = self.network(z_in)
        else:
            nn_out = self.network(jnp.concatenate([z_in, orig_img], axis=-1)) # condition on orig image
        s, t = nn_out.split(2, axis=-1) # mu, log(sigma)

        # scale sigma into [-scaling_fac, scaling_fac]
        s_fac = jnp.exp(self.scaling_factor).reshape(1, 1, 1, -1)
        s = nn.tanh(s / s_fac) * s_fac

        # apply mask on s and t
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        if not reverse:
            z = (z + t) * jnp.exp(s)
            ldj += s.sum(axis=[1,2,3])
        else:
            z = (z * jnp.exp(-s)) - t
            ldj -= s.sum(axis=[1,2,3])

        return z, ldj, rng

class ConcatELU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return jnp.concatenate([nn.elu(x), nn.elu(-x)], axis=-1)

class GatedConv(nn.Module):
    c_in: int
    c_hidden: int

    @nn.compact
    def __call__(self, x):
        out = nn.Sequential([
            ConcatELU(),
            nn.Conv(self.c_hidden, kernel_size=(3,3)),
            ConcatELU(),
            nn.Conv(2*self.c_in, kernel_size=(1,1))
        ])(x)
        val, gate = out.split(2, axis=-1)
        return x + val*nn.sigmoid(gate)

class GatedConvNet(nn.Module):
    c_hidden: int
    c_out: int
    num_layers: int=3

    def setup(self):
        layers = []
        layers += [nn.Conv(self.c_hidden, kernel_size=(3,3))]
        for layer_index in range(self.num_layers):
            layers += [GatedConv(self.c_hidden, self.c_hidden), nn.LayerNorm()]
        layers += [ConcatELU(), nn.Conv(self.c_out, kernel_size=(3,3),
            kernel_init=nn.initializers.zeros)]
        self.nn = nn.Sequential(layers)

    def __call__(self, x):
        return self.nn(x)

class SqueezeFlow(nn.Module):
    def __call__(self, z, ldj, rng, reverse=False):
        B, H, W, C = z.shape
        if not reverse:
            # H x W x C -> H/2 x W/2 x 4C
            z = z.reshape(B, H//2, 2, W//2, 2, C)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H//2, W//2, 4*C)
        else:
            # H x W x C -> 2H x 2W x C/4
            z = z.reshape(B, H, W, 2, 2, C//4)
            z = z.transpose((0, 1, 3, 2, 4, 5))
            z = z.reshape(B, H*2, W*2, C//4)

        return z, ldj, rng

class SplitFlow(nn.Module):
    def __call__(self, z, ldj, rng, reverse=False):
        if not reverse:
            z, z_split = z.split(2, axis=-1)
            ldj += jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1,2,3]) # norm prior
        else:
            z_split = random.normal(rng, z.shape)
            z = jnp.concatenate([z, z_split], axis=-1)
            ldj -= jax.scipy.stats.norm.logpdf(z_split).sum(axis=[1,2,3])

        return z, ldj, rng

if __name__ == '__main__':
    # test Dequantization
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
    def visualize_dequantization(quants, fig_name, prior=None):
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
        plt.savefig(fig_name, bbox_inches="tight")

    visualize_dequantization(quants=8, fig_name="dequant_uniform.png")
    visualize_dequantization(quants=8, fig_name="dequant_normal.png",
            prior=np.array([0.075, 0.2, 0.4, 0.2, 0.075, 0.025, 0.0125, 0.0125]))

    # test scaling
    x = jnp.arange(-5,5,0.01)
    scaling_factors = [0.5, 1, 2]
    sns.set()
    fig, ax = plt.subplots(1, 3, figsize=(12,3))
    for i, scale in enumerate(scaling_factors):
        y = nn.tanh(x / scale) * scale
        ax[i].plot(x, y)
        ax[i].set_title("Scaling factor: " + str(scale))
        ax[i].set_ylim(-3, 3)
    plt.subplots_adjust(wspace=0.4)
    sns.reset_orig()
    plt.savefig("scaling_factor.png", bbox_inches='tight')

    # test Conv Net
    main_rng = random.PRNGKey(10)
    main_rng, x_rng = random.split(main_rng)
    x = random.normal(x_rng, (3, 32, 32, 16))
    gcn = GatedConvNet(c_hidden=32, c_out=18, num_layers=3)
    main_rng, init_rng = random.split(main_rng)
    params = gcn.init(init_rng, x)['params']
    # Apply attention with parameters on the inputs
    out = gcn.apply({'params': params}, x)
    print('Out', out.shape)

    # test SqueezeFlow
    sq_flow = SqueezeFlow()
    rand_img = jnp.arange(1,17).reshape(1, 4, 4, 1)
    print("Image (before)\n", rand_img.transpose(0, 3, 1, 2)) # Permute for readability
    forward_img, _, _ = sq_flow(rand_img, ldj=None, rng=None, reverse=False)
    print("\nImage (forward)\n", forward_img)
    reconst_img, _, _ = sq_flow(forward_img, ldj=None, rng=None, reverse=True)
    print("\nImage (reverse)\n", reconst_img.transpose(0, 3, 1, 2))

