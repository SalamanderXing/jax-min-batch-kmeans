import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_centroids(feature_data, k: int) -> np.ndarray:
    """Calculate initial centroids as randomly select k feature instances (k rows of the dataset)"""
    indices = np.arange(k)  # generate indices for k centroids
    np.random.shuffle(indices)  # shuffle indices
    centroids = feature_data[indices[:k]]  # select k rows from feature_data
    return centroids


# manhattan distance
def calculate_distance(feature_data, query_instance):
    return np.sum(
        np.abs(feature_data - query_instance), axis=1
    )  # directly applies the formula for manhattan distance. axis=1 means sum over rows


def assign_centroids(feature_data, centroids):
    distances = np.stack(
        tuple(calculate_distance(feature_data, c) for c in centroids)
    )  # compute distances for each centroid with respect
    assigned_centroids = np.argmin(
        distances, axis=0
    )  # assign each instance to the closest centroid
    return assigned_centroids


def move_centroids(feature_data, assigned_centroids, centroids):
    """
    the line below can be interpreted as follows:
    For each centroid, compute the mean of all instances assigned to that centroid.
    Those means are the new centroids.
    """
    return np.stack(
        tuple(
            np.mean(feature_data[np.where(assigned_centroids == i)[0]], axis=0)
            for i in range(len(centroids))
        )
    )


# squared euclidean distance
def distortion_cost(feature_data, assigned_centroids, centroids):
    # we subtract, for each instance, the its closes centroid. Then we square the result and sum over all instances.
    return np.sum((feature_data - centroids[assigned_centroids]) ** 2)


def restart_KMeans(data, k: int, iterations: int = 10, restarts: int = 10):
    best_centroids = np.zeros([])  # initialize with empty array
    best_cost = float("inf")  # initialize with infinity (cannot be higher than that)
    log_cost_per_step = False
    for n_restart in range(restarts):  # each restart we need to start from scratch
        print(f"\n\n{n_restart=}")
        centroids = create_centroids(data, k)  # initialize centroids
        for i in range(iterations):  # iterate to make the centroids converge
            assigned_centroids = assign_centroids(
                data, centroids
            )  # assign centroids to each instance
            if log_cost_per_step:
                cost = distortion_cost(
                    data, assigned_centroids, centroids
                )  # compute distorion cost
                print(f"{i=} {cost=}")
                centroids = move_centroids(
                    data, assigned_centroids, centroids
                )  # update the centroids
        assigned_centroids = assign_centroids(
            data, centroids
        )  # assign centroids to compute distortion cost
        cost = distortion_cost(data, assigned_centroids, centroids)
        if (
            cost < best_cost
        ):  # if the cost is lower than the best cost, we update the best cost and the best centroids
            best_cost = cost
            best_centroids = centroids
            print(f"found: {best_cost=}")
    return best_cost, best_centroids


def elbow_plot(data, min_k=1, max_k=10):
    """
    Compute the distortion cost for different values of k and plot the result.
    To make it quicker, we set iterations=2 and restarts=2. This because we noticed the for this problem the algorithm converges almost immediately.
    """
    costs = [
        restart_KMeans(data, k, 2, 2)[0] for k in range(min_k, max_k + 1)
    ]  # get distortion cost for each value of k
    plt.plot(costs)
    plt.xticks(np.arange(min_k, max_k + 1))  # set xticks to be the values of k
    plt.xlabel("Number of clusters")
    plt.ylabel("Distortion cost")
    plt.show()


def main():
    data = pd.read_csv("clusteringData.csv", header=None).values
    elbow_plot(data)


if __name__ == "__main__":
    main()
