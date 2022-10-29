import numpy as np


def min_batch_k_means(
    features_data, k: int, restarts: int, epsilon: float, batch_size: int
):
    # same as in k-means implementation, we want to restart many times
    for restart in range(restarts):
        # we initialize the centroids by randomly selecting k points from the dataset
        centroids = create_centroids(features_data, k)  # same function as in kmeans
        # v contains the per-center counts and its used to compute the learning rate
        vs = np.zeros(k)  # initialize vs to 0, one for each centroids.
        # patience is the number of times we can go through the loop without significant improvement
        patience = 2
        no_change = 0  # number of times we have not seen improvement
        while no_change < patience:
            # we sample a batch of size batch_size
            batch = np.random.choice(features_data, batch_size, replace=False)
            # we assign each point in the batch to the closest centroid
            assigned_centroids = assign_centroids(
                batch, centroids
            )  # same function as in kmeans
            # we iterate over the batch items in order to update the centroids
            new_centroids = centroids.copy()
            for i in range(batch_size):
                # get the centroid index assigned to the current data point
                c_index = assigned_centroids[i]
                # update the count for the centroid
                vs[c_index] += 1
                # compute the learning rate
                # note that the learning rate is inversely proportional to the count
                # this because we want to change it less if it has many points assigned to it
                # Notice that the learning rate reduces with each iteration.
                # This guarrantees convergence, as each new centroid will be changed less and less
                lr = 1 / vs[c_index]
                # udpate the centroid, we use the learning rate to control how much we change it
                # to keep it stable, we subtract the centroid itself multiplied by the learning rate
                # then we add the current data point multiplied by the learning rate
                new_centroids[c_index] = (1 - lr) * centroids[c_index] + lr * batch[i]
            # as described in the paper: K-means vs Mini Batch K-means: A comparison
            # we detect convergence when approximately no change is made to the centroids
            # for a number of successive iterations (determined by patience)
            diff = mean_distance(
                centroids, new_centroids
            )  # we can use euclidean distance
            centroids = new_centroids
            if diff < epsilon:
                no_change += 1
            else:
                no_change = 0

        # in the end we need to evaulate the distortion cost, same as for the regular kmeans
        distortion_cost = get_distortion_cost(features_data, centroids)
        print(f"Restart {restart} distortion cost: {distortion_cost}")


def mean_distance(old_centroids, centroids):
    return np.mean(np.linalg.norm(old_centroids - centroids, axis=1))


def get_distortion_cost(features_data, centroids):
    pass


def create_centroids(features_data, k):
    return features_data[np.random.choice(features_data.shape[0], k, replace=False)]


def assign_centroids(batch, centroids):
    return np.argmin(np.linalg.norm(batch - centroids, axis=1), axis=1)
