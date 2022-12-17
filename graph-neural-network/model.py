import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

class GCNLayer(nn.Module):
    c_out: int

    @nn.compact
    def __call__(self, x, G):
        '''
        Input:
            x: node features, shape=[batch_size, num_nodes, c_in]
            G: adjacency matrix, shape=[batch_size, num_nodes, num_nodes]

        Transformation formula:
            h_out = h_in -> param -> G -> avg by degree
        '''
        deg = G.sum(axis=-1, keepdims=True) # include itself
        x = nn.Dense(features=self.c_out, name='proj')(x)
        x = jax.lax.batch_matmul(G, x) / deg

        return x

class GATLayer(nn.Module):
    c_out: int
    num_heads: int
    concat: bool=False # avg / concat over multiple heads
    alpha: float=0.2 # param for LeakyReLU
    
    def setup(self):
        if self.concat:
            assert self.c_out % self.num_heads == 0
            c_out_per_head = self.c_out // self.num_heads
        else:
            c_out_per_head = self.c_out

        self.proj = nn.Dense(c_out_per_head*self.num_heads,
                kernel_init=nn.initializers.glorot_uniform())
        self.attn_fst = self.param('attn_fst',
                nn.initializers.glorot_uniform(),
                (self.num_heads, c_out_per_head))
        self.attn_snd = self.param('attn_snd',
                nn.initializers.glorot_uniform(),
                (self.num_heads, c_out_per_head))

    def __call__(self, x, G, print_attn=False):
        '''
        Input:
            x: node features, shape=[batch_size, num_nodes, c_in]
            G: adjacency matrix, shape=[batch_size, num_nodes, num_nodes]

        Transformation formula:
            h_out = attn weighted sum of (W @ h_in -> G)
            where attn is calculated as softmax over 
            (a @ W @ (h_in || h_neighbor))
        '''
        batch_size, num_nodes = x.shape[0], x.shape[1]
        x = self.proj(x)
        x = x.reshape((batch_size, num_nodes, self.num_heads, -1)) 
        # [batch_size, num_nodes, num_heads, c_out_per_head]
        
        attn_logit_fst = (x * self.attn_fst).sum(axis=-1)
        # [batch_size, num_nodes, num_heads]
        attn_logit_snd = (x * self.attn_snd).sum(axis=-1)
        # [batch_size, num_nodes, num_heads]
        attn_logit = \
            attn_logit_fst[:, :, None, :].repeat(num_nodes, axis=2) + \
            attn_logit_snd[:, None, :, :].repeat(num_nodes, axis=1)
        # [batch_size, num_nodes, num_nodes, num_heads]
        attn_logit = nn.leaky_relu(attn_logit, self.alpha)

        # mask out nodes that is not connected
        attn_logit = jnp.where(G[...,None] == 1, attn_logit,
                jnp.ones_like(attn_logit) * (-9e15))
        
        # convert to prob
        attn_prob = nn.softmax(attn_logit, axis=2)
        # [batch_size, num_nodes, num_nodes, num_heads]
        if print_attn:
            print(attn_prob.transpose(0, 3, 1, 2))

        # magic! https://ajcr.net/Basic-guide-to-einsum/
        x = jnp.einsum('bijh,bjhc->bihc', attn_prob, x)
        # [batch_size, num_nodes, num_heads, c_out_per_head]
        if not self.concat:
            x = x.mean(axis=2) # avg out vector from different heat
            # [batch_size, num_nodes, c_out]
        else:
            x = x.reshape(batch_size, num_nodes, -1)
            # [batch_size, num_nodes, c_out]

        return x

if __name__ == '__main__':
    # test GCNLayer
    x = jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2))
    G = jnp.array([[[1,1,0,0],[1,1,1,1],[0,1,1,1],[0,1,1,1]]]).astype(
            jnp.float32)

    gcn_layer = GCNLayer(c_out=2)
    params = {'proj': {
        'kernel': jnp.array([[1.,0.],[0.,1.]]), # Identity func
        'bias': jnp.array([0.,0.]),
    }}
    print('(In) Node features:\n', x)
    print('(Out) Node features:\n', gcn_layer.apply({'params':params}, x, G))
    print('Adjaency matrix:\n', G)

    # test GATLayer
    x = jnp.arange(8, dtype=jnp.float32).reshape((1, 4, 2))
    G = jnp.array([[[1,1,0,0],[1,1,1,1],[0,1,1,1],[0,1,1,1]]]).astype(
            jnp.float32)

    gat_layer = GATLayer(c_out=2, num_heads=2, concat=False, alpha=0.2)
    params = {
        'proj': {
            'kernel': jnp.array([[1.,0.,1.,0.],[0.,1.,0.,1.]]),
            'bias': jnp.array([0.,0.,0.,0.])
        },
        'attn_fst': jnp.array([[-0.2, 0.3],[0.5,-0.1]]),
        'attn_snd': jnp.array([[-0.2, 0.3],[0.5,-0.1]])
    }
    out = gat_layer.apply({'params': params}, x, G)

    print('(In) Node features:\n', x)
    print('(Out) Node features:\n', gat_layer.apply({'params':params}, x, G))
    print('Adjaency matrix:\n', G)

    

