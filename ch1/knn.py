''' Implementation of KNN (exact) in Python 3, to facilitate exercises in Ch 1.
of ML-APA without needing MATLAB, and for some practice.
'''
from heapq import heappush, heappop, heappushpop

def exact_knn(x_test, x_training, y_training, k=5):
    '''Computes exact k-nearest neighbour label for test input x.

    parameters:
        x_test: a length D list representing a test datapoint.
        x_training: NxD list of lists representing training data.
        y_training: a length N list of training labels.
        k: The number of nearest neighbours to use.
    returns:
        A probability density represent as a dictionary,
            P(x_test | x_training, y_training, k), for every label.
    Examples:
    >>> exact_knn([0], [[0],[1],[2],[3],[4],[5]], [1,1,1,0,0,0], 3)
    1
    >>> exact_knn([2], [[0],[1],[2],[3],[4],[5]], [1,1,1,0,0,0], 3)
    1
    >>> exact_knn([3], [[0],[1],[2],[3],[4],[5]], [1,1,1,0,0,0], 3)
    0
    >>> exact_knn([5], [[0],[1],[2],[3],[4],[5]], [1,1,1,0,0,0], 3)
    0
    '''
    heap = [] #heap of pairs of distances and indices.
    for i in range(len(x_training)):
        dists = ((i-j)**2 for i, j in zip(x_test, x_training[i]))
        dist_sum = 0
        for dist in dists:
            dist_sum -= dist # use negative distances to work with min-heap
            if len(heap) == k and dist_sum < heap[0][0]:
                break
        if len(heap) < k:
            heappush(heap, (dist_sum, y_training[i]))
        elif dist_sum > heap[0][0]:
            heappushpop(heap, (dist_sum, y_training[i]))

    labels = {}
    for item in heap:
        labels[item[1]] = 1 if item[1] not in labels else 1 + labels[item[1]]
    best_label = sorted(labels.items(), key=lambda x: -x[1])[0][0]
    return best_label

if __name__ == "__main__":
    import doctest
    doctest.testmod()
