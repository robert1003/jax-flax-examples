import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PositionEncoding:
    def __init__(self, dim, base):
        self.dim = dim
        self.base = base

    def _omega(self, k):
        return 1.0 / (self.base ** (2*k / self.dim))

    def __call__(self, pos):
        encoding = []
        for i in range(self.dim):
            k = i // 2
            if i % 2 == 0:
                encoding.append(np.sin(self._omega(k)*pos))
            else:
                encoding.append(np.cos(self._omega(k)*pos))

        return np.array(encoding)

if __name__ == '__main__':
    N, dim, base = 50, 128, 10000
    posEncode = PositionEncoding(dim, base)
    encodings = []
    for i in range(N):
        encodings.append(posEncode(i))
    encodings = np.stack(encodings)
    assert encodings.shape == (N, dim)

    # visualize the position encoding
    plt.figure(figsize=(15, 5))
    sns.heatmap(encodings, cmap='RdBu')
    plt.xlabel('Depth')
    xticks = list(range(0, 128, 20))
    plt.xticks(xticks, xticks)
    plt.ylabel('Position')
    yticks = list(range(0, N+1, 10))
    plt.yticks(yticks, yticks)
    plt.tight_layout()
    plt.savefig('position-encoding.png', bbox_inches='tight')
    plt.clf()
    #plt.show()

    # demonstrate that position encoding are time-step symmetrical and decays with time
    def l2_dis(v1, v2):
        return np.dot(v1, v2)

    dis = np.stack([np.array([l2_dis(v1, v2) for v2 in encodings]) for v1 in encodings])
    plt.figure(figsize=(5, 5))
    sns.heatmap(dis, cmap='Blues')
    plt.xlabel('Position') 
    plt.ylabel('Position')
    xyticks = list(range(0, N+1, 10))
    plt.xticks(xyticks, xyticks)
    plt.yticks(xyticks, xyticks)
    plt.tight_layout()
    plt.savefig('pos-encoding-dot-prod.png', bbox_inches='tight')
    plt.clf()
    #plt.show()
