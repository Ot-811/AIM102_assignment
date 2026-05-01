import numpy as np
import matplotlib.pyplot as plt

points = np.array([
    [1,1],[1.5,2],[3,4],[5,7],[3.5,5],[4.5,5],[3.5,4.5]
])

def kmeans(X, k, max_iters=10):
    indices = np.random.choice(len(X), k, replace=False)
    centroids = X[indices]

    for _ in range(max_iters):
        clusters = [[] for _ in range(k)]

        for point in X:
            distances = [np.linalg.norm(point - c) for c in centroids]
            idx = np.argmin(distances)
            clusters[idx].append(point)

        new_centroids = []
        for cluster in clusters:
            new_centroids.append(np.mean(cluster, axis=0))

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

def compute_sse(centroids, clusters):
    sse = 0
    for i, cluster in enumerate(clusters):
        for point in cluster:
            sse += np.linalg.norm(point - centroids[i])**2
    return sse

for i in range(3):
    centroids, clusters = kmeans(points, 2)

    print(f"\nRun {i+1}")
    print("Centroids:", centroids)
    print("SSE:", compute_sse(centroids, clusters))

    plt.figure()

    for cluster in clusters:
        cluster = np.array(cluster)
        plt.scatter(cluster[:,0], cluster[:,1])

    plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200)

    plt.title(f"K-Means Run {i+1}")
    plt.grid()

    plt.savefig(f"q4_kmeans_run_{i+1}.png", dpi=300, bbox_inches='tight')
    plt.show()