import jax.numpy as jnp
import numpy as np
import jax.random as jrn
from jax import jit
from jax import lax


@jit
def get_distortion_cost(
    feature_data: jnp.DeviceArray,
    assigned_centroids: jnp.DeviceArray,
    centroids: jnp.DeviceArray,
):
    # we subtract, for each instance, the its closes centroid. Then we square the result and sum over all instances.
    return jnp.sum((feature_data - centroids[assigned_centroids]) ** 2)


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
    return jnp.sum(
        jnp.abs(feature_data - query_instance), axis=1
    )  # directly applies the formula for manhattan distance. axis=1 means sum over rows


@jit
def assign_centroids(feature_data, centroids):
    distances = jnp.stack(
        tuple(calculate_distance(feature_data, c) for c in centroids)
    )  # compute distances for each centroid with respect
    assigned_centroids = jnp.argmin(
        distances, axis=0
    )  # assign each instance to the closest centroid
    return assigned_centroids

def update_centroids_vectorized(v, centroids, assigned_centroids, batch, k):
    # attepmt to vectorize the update centroids function, but it's much slower to converge
    tmp = assigned_centroids[None].repeat(k, axis=0)
    arange = jnp.arange(k)[None].T
    mask = tmp == arange
    current_v = mask.cumsum(axis=1)
    v = v + current_v
    lrs = (1 / v).T.flatten()
    # filters lrs = inf
    lrs = jnp.where(lrs == jnp.inf, 0, lrs)
    uba = assigned_centroids + jnp.arange(0, k*batch.shape[0], k)
    correct_lrs = lrs[uba, None]
    # centroids = centroids + correct_lrs * (batch - centroids)
    counts = jnp.zeros(k, dtype=jnp.int32)
    vals, cs = jnp.unique(jnp.sort(assigned_centroids), return_counts=True)
    counts = counts.at[vals].set(cs)
    max_count = jnp.max(counts)
    to_add = max_count - counts
    tot_to_add = jnp.sum(to_add)
    empty = jnp.arange(max_count)[None].repeat(k, 0)
    mask2 = (empty < to_add[:, None]).astype(int)
    # allaccio = mask2.at[mask2 > 0].set(np.arange(1, k + 1)[:, None]).flatten()
    allaccio = (mask2 * jnp.arange(1, k + 1)[:, None]).flatten()
    alloccio = allaccio[allaccio > 0] - 1
    assigned_centroids_ciao = jnp.concatenate([assigned_centroids, alloccio])
    toll = jnp.zeros((tot_to_add, batch.shape[1]))
    batch_ciao = jnp.concatenate([batch, toll])
    yalla = assigned_centroids_ciao.argsort()
    yolla = batch_ciao[yalla].reshape(k, max_count, batch.shape[1])
    youla_lrs = (
        jnp.concatenate((correct_lrs, jnp.zeros(max_count)[:, None]))[yalla]
        .flatten()
        .reshape(k, max_count)[:, :, None]
    )
    cuntroids = centroids[:, None]
    centroids = ((1 - youla_lrs)*cuntroids  + youla_lrs*yolla).mean(axis=1) # (cuntroids + youla_lrs * (yolla - cuntroids)).sum(axis=1)
    return centroids, v



class MiniBatchKMeans:
    def __init__(
        self,
        xs: jnp.DeviceArray | np.ndarray,
        k: int,
        *,
        batch_size: int = 1000,
        iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 0,
    ):
        self.batch_size: int= batch_size
        self.xs:jnp.DeviceArray = jnp.array(xs)
        self.k:int = k
        self.iter:int = iter
        self.tol = tol
        self.rn_key:jnp.DeviceArray = jrn.PRNGKey(random_state)
        self.centroids:Optional[DeviceArray] = None

    def fit(self):
        centroids = jrn.choice(self.rn_key, self.xs, shape=(self.k,))
        vs = jnp.zeros(self.k)
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
    xs: jnp.DeviceArray = jnp.asarray(genfromtxt("clusteringData.csv", delimiter=","))
    kmeans = MiniBatchKMeans(
        xs, k=4, batch_size=1000, iter=1000, random_state=0
    )
    kmeans.fit()


if __name__ == "__main__":
    main()
