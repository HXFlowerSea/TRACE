import matplotlib.pyplot as plt
from sklearn import decomposition as sk_dec
from sklearn import preprocessing as sk_prep
import torch

def plot_vectors(latent: torch.Tensor, labels: torch.Tensor):
    latent = sk_prep.normalize(X=latent, norm="l2")
    z2d = sk_dec.PCA(n_components=2).fit_transform(latent)

    fig, ax = plt.subplots(figsize=(10, 10))

    for y in labels.unique():
        ax.scatter(
            z2d[labels == y, 0], z2d[labels == y, 1],
            marker=".", label=y.item(),
        )

    fig.legend()

    return fig