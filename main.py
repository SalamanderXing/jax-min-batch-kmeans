import jax.numpy as np
import jax.random as jrn
from numpy import genfromtxt
import ipdb
from jax import jit
import random
from jax import lax

@jit
def get_distortion_cost(
    feature_data: np.DeviceArray,
    assigned_centroids: np.DeviceArray,
    centroids: np.DeviceArray,
):
    # we subtract, for each instance, the its closes centroid. Then we square the result and sum over all instances.
    return np.sum((feature_data - centroids[assigned_centroids]) ** 2)


@jit
def update_centroids(vs, centroids, assigned_centroids, batch):
    for i in range(batch.shape[0]):
        c_index = assigned_centroids[i]
        vs = vs.at[c_index].add(1)
        lr = 1 / vs[c_index]
        centroids = centroids.at[c_index].add(lr * (batch[i] - centroids[c_index]))
    return centroids, vs

def body_fun(i, vals):
    batch, assigned_centroids, centroids, vs = vals
    c_index = assigned_centroids[i]
    vs = vs.at[c_index].add(1)
    lr = 1 / vs[c_index]
    centroids = centroids.at[c_index].add(lr * (batch[i] - centroids[c_index]))
    return batch, assigned_centroids, centroids, vs

def update_centroids_func(vs, centroids, assigned_centroids, batch):
    # while faster to compile, for some reason this performs worse than the for loop
    _, _, centroids, vs = lax.fori_loop(0, batch.shape[0], body_fun, (batch, assigned_centroids, centroids, vs))
    return centroids, vs

@jit
def calculate_distance(feature_data, query_instance):
    return np.sum(
        np.abs(feature_data - query_instance), axis=1
    )  # directly applies the formula for manhattan distance. axis=1 means sum over rows


@jit
def assign_centroids(feature_data, centroids):
    distances = np.stack(
        tuple(calculate_distance(feature_data, c) for c in centroids)
    )  # compute distances for each centroid with respect
    assigned_centroids = np.argmin(
        distances, axis=0
    )  # assign each instance to the closest centroid
    return assigned_centroids

def update_centroids_vectorized(v, centroids, assigned_centroids, batch, k):
    # attepmt to vectorize the update centroids function, but it's much slower to converge
    tmp = assigned_centroids[None].repeat(k, axis=0)
    arange = np.arange(k)[None].T
    mask = tmp == arange
    current_v = mask.cumsum(axis=1)
    v = v + current_v
    lrs = (1 / v).T.flatten()
    # filters lrs = inf
    lrs = np.where(lrs == np.inf, 0, lrs)
    uba = assigned_centroids + np.arange(0, k*batch.shape[0], k)
    correct_lrs = lrs[uba, None]
    # centroids = centroids + correct_lrs * (batch - centroids)
    counts = np.zeros(k, dtype=np.int32)
    vals, cs = np.unique(np.sort(assigned_centroids), return_counts=True)
    counts = counts.at[vals].set(cs)
    max_count = np.max(counts)
    to_add = max_count - counts
    tot_to_add = np.sum(to_add)
    empty = np.arange(max_count)[None].repeat(k, 0)
    mask2 = (empty < to_add[:, None]).astype(int)
    # allaccio = mask2.at[mask2 > 0].set(np.arange(1, k + 1)[:, None]).flatten()
    allaccio = (mask2 * np.arange(1, k + 1)[:, None]).flatten()
    alloccio = allaccio[allaccio > 0] - 1
    assigned_centroids_ciao = np.concatenate([assigned_centroids, alloccio])
    toll = np.zeros((tot_to_add, batch.shape[1]))
    batch_ciao = np.concatenate([batch, toll])
    yalla = assigned_centroids_ciao.argsort()
    yolla = batch_ciao[yalla].reshape(k, max_count, batch.shape[1])
    youla_lrs = (
        np.concatenate((correct_lrs, np.zeros(max_count)[:, None]))[yalla]
        .flatten()
        .reshape(k, max_count)[:, :, None]
    )
    cuntroids = centroids[:, None]
    centroids = ((1 - youla_lrs)*cuntroids  + youla_lrs*yolla).mean(axis=1) # (cuntroids + youla_lrs * (yolla - cuntroids)).sum(axis=1)
    return centroids, v



class MiniBatchKMeansOptimizer:
    def __init__(
        self,
        xs: np.DeviceArray,
        k: int,
        *,
        batch_size: int = 1000,
        n_init: int = 10,
        iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 0,
    ):
        self.batch_size = batch_size
        self.xs = xs
        self.k = k
        self.n_init = n_init
        self.iter = iter
        self.tol = tol
        self.rn_key = jrn.PRNGKey(random_state)

    def fit(self):
        centroids = jrn.choice(self.rn_key, self.xs, shape=(self.k,))
        vs = np.zeros(self.k)
        for i in range(self.iter):
            rn_key, subkey = jrn.split(self.rn_key)
            batch = jrn.choice(subkey, self.xs, shape=(self.batch_size,), replace=False)
            assigned_centroids = assign_centroids(batch, centroids)
            distortion_cost = get_distortion_cost(
                batch, assigned_centroids, centroids
            ).item()
            print(f"{distortion_cost=:.2f}")
            # centroids, vs = update_centroids_weird(vs, centroids, assigned_centroids, batch, self.k) 
            centroids, vs = update_centroids(vs, centroids, assigned_centroids, batch)
            #centroids, vs = update_centroids_func(vs, centroids, assigned_centroids, batch)
        self.centroids = centroids


def main():
    xs: np.DeviceArray = np.asarray(genfromtxt("clusteringData.csv", delimiter=","))
    kmeans = MiniBatchKMeansOptimizer(
        xs, k=4, batch_size=1000, iter=1000, random_state=random.randint(0, 1000)
    )
    kmeans.fit()


if __name__ == "__main__":
    main()
