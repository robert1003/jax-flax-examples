# Flow-based Generative Model (for MNIST)

## Tricks

### Dequantization

Recaull that the goal for our MNIST flow model here transform a discrete distribution (pixel value distribution) to a multivariate gaussian normal distribution. In other words, we want to model discrete distribution (e.g. categorical distribution) with our flow model here. The problem is that a discrete distribution is not well-defined in a continuous space: the pdf of it will become a function that is mostly zero but full of delta spikes with no width (Figure 1). This will cause the flow model to place infinite likelihood on those points, and the resulting model cannot be used since it tells nothing about the "true" discrete pixel value distribution.

[TODO figure 1]

An intuitive way to fix this is to add a uniform noise to the discrete distribution, which is the "dequantize" process. The reverse process ("quantize" process, continuous -> discrete) can be a simple $floor$ function. Suppose the discrete distribution we want to model is $p(x)$, then
```math
p(x) = \int p(x+u)du = \int \frac{q(u|x)}{q(u|x)} du = \mathbb{E}\_{u\sim q(u|x)} \left[ \frac{p(x+u)}{q(u|x)} \right]
```
where $q(u|x)$ is any continuous distribution. If we add a uniform noise to fix this, then $q(u|x)=Unif(0, 1)$ and the "true" $p(x)$ we want to modify will become Figure 2. Notice that this distribution, though better than Figure 1, is still hard to model it with continuous distribution because of the discontinuity between values.

[TODO figure 2]

To solve this, we can make the noise $u$ depends on $x$ i.e. $u\sim q(u|x)$. An intuitive way of thinking this is: consider a categorical distribution with $p(0)=0.8, p(1)=0.2$. We should make the "width" that corresponds to label $0$ larger (since it has a larger probability) and "width" that corresponds to label $1$ smaller. Why? Because the resulting pdf will be much more smooth, as seen in Figure 3.

[TODO figure 3]

There are many ways to parameterize $q(u|x)$, such as conditional VAE or conditional Flow. We choose to use conditional Flow here.

## Analysis

Run on 12th Gen Intel(R) Core(TM) i7-12700F / NVIDIA GeForce RTX 3090

### Quantitative Result

| Model | Train Bpd | Val Bpd | Test Bpd | Inference Time (ms) | Num Params |
| - | - | - | - | - | - |
| simple | 1.079357 | 1.080054 | 1.078568 | 4.1067 | 556312 |
| multi-simple | 1.081993 | 1.087551 | 1.085518 | 2.6471 | 628388 |
| vardeq | 1.038131 | 1.041275 | 1.039238 | 5.5446 | 1639742 |
| multi-vardeq | 1.005583 | 1.021777 | 1.020105 | 4.0657 | 1711818 |

[TODO]

### Example Images

[TODO]

### Interpolation

[TODO]

### Dequant distribution

[TODO]
