import numpy as np
import matplotlib.pyplot as plt

# load data
def load_data(file_path):
    return np.loadtxt(file_path)

def initialize_centroids(data, k):
    # initialize cluster centroids randomly with k data points
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def compute_distances(data, centroids):
    return np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)

def assign_clusters(data, centroids):
    # assign each data point to the nearest centroid
    distances = compute_distances(data, centroids)
    # return an array where each element is the index of the nearest centroid corresponding to the data point
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    # return an array of the updated centroids as a mean of assigned points
    return np.array([data[labels == i].mean(axis=0) for i in range(k)])

# k-means clustering algorithm
def k_means_clustering(data, k, i=100, tolerance=1e-4):
    # initialize centroids with randomly selected k data points
    centroids = initialize_centroids(data, k)
    # initialize empty list to keep track of centroids position for each iteration
    cluster_history = []
    
    for _ in range(i):
        # assign data point to nearest centroid
        labels = assign_clusters(data, centroids)
        # copy the current centroids and add to history
        cluster_history.append(centroids.copy())
        
        # update centroids by calculating the mean of all data points assigned to each cluster
        new_centroids = update_centroids(data, labels, k)
        # check if the change in centroids is less than the default tolerance
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break   # less than tolerance means convergence, break
        centroids = new_centroids   # otherwise, update centroids to the new value
    
    # return the labels for each data point, the final centroids, and the centroid history
    return labels, centroids, cluster_history

def plot_clusters(data, k):
    # run k-means on the provided data with k clusters
    labels, centroids, history = k_means_clustering(data, k)
    
    # plot the data points cluster
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b'][:k]
    for i in range(k):
        plt.scatter(data[labels == i, 0], data[labels == i, 1], c=colors[i], label=f'Cluster {i}')
    # plot movement
    for i, step in enumerate(history):
        plt.scatter(step[:, 0], step[:, 1], c='black', marker='x', alpha=0.5, label=f'Step {i}' if i == 0 else "")
    # plot centroids of the cluster
    plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='o', edgecolors='k', s=200, label='Centroids')
    plt.legend()
    plt.title(f'K-Means Clustering with k={k}')
    plt.show()

    # print(f"Converged after {len(history)-1} iterations")
    print(f"Cluster centroids:\n{centroids}")
    for i in range(k):
        count = np.sum(labels == i)
        print(f"Cluster {i+1}: {count} points")

def main():
    # load and process the data
    file_path = 'cluster_data_2D.dat'
    data = load_data(file_path)

    # run for k = 2 and k = 3
    for k in [2, 3]:
        print(f"\nk={k}")
        plot_clusters(data, k)

if __name__ == "__main__":
    main()