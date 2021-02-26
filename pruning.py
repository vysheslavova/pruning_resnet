import torch
from sklearn.cluster import KMeans


def pruning(weights, n, device):
    with torch.no_grad():
        size = weights.shape
        weights = weights.flatten(1).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=n, random_state=23).fit(weights)
        centroids = kmeans.cluster_centers_
        result = torch.tensor(centroids[kmeans.fit_predict(weights)])
        result = torch.reshape(result, size).to(device, dtype=torch.float)
        return torch.nn.Parameter(result)