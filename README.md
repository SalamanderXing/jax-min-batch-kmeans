# Mini-Batch KMeans written in JAX

Just-in-time compiled implementation of the algorithm [Mini-Batch KMeans](https://doi.org/10.1145/1772690.1772862)[1]

## Installation
```bash
git clone https://github.com/GiulioZani/jax-min-batch-kmeans

cd jax-mini-batch-kmeans

```

## Usage

```python
from mini_batch_kmeans import MiniBatchKMeans

def main():
   xs = # some array of shape (number of samples, number of features)
   mini_batch_kmeans = MiniBatchKMeans(
	xs, # can be a numpy or jax array
	k=4, # number of clusters
	batch_size=1000, # batch size
	iter=1000, # number of iterations
	random_state=0
   )
   mini_batch_kmeans.fit()

   print(f"{mini_batch_kmeans.centroids=}")
```

## References

[1] D. Sculley. 2010. Web-scale k-means clustering. In Proceedings of the 19th international conference on World wide web (WWW '10). Association for Computing Machinery, New York, NY, USA, 1177–1178. https://doi.org/10.1145/1772690.1772862
