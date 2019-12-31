''' Implementation of KNN (exact) in Python 3, to facilitate exercises in Ch 1.
of ML-APA without needing MATLAB, and for some practice.
'''
import doctest
from heapq import heappush, heappushpop
import numpy as np

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
    for i, item in enumerate(x_training):
        dist_sum = 0
        for j, feature in enumerate(x_test):
            dist = (int(x_test[j]) - int(item[j]))**2
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

def read_data(fname):
    '''Reads a binary dataformat used in the MNIST sets'''
    with open(fname, 'r') as data_file:
        bits = np.fromfile(data_file, dtype=np.ubyte)
        bits = bits[16:] # cut off the formating header.
        image_size = 28*28
        return np.reshape(bits, [len(bits)//image_size, image_size])

def read_labels(fname):
    '''Reads a binary label format used in the MNIST sets'''
    with open(fname, 'r') as data_file:
        bits = np.fromfile(data_file, dtype=np.ubyte)
        return bits[8:] # cut off the formating header.

if __name__ == "__main__":
    doctest.testmod()
    training = read_data('data/train-images-idx3-ubyte')
    test = read_data('data/t10k-images-idx3-ubyte')
    training_labels = read_labels('data/train-labels-idx1-ubyte')
    test_labels = read_labels('data/t10k-labels-idx1-ubyte')

    num_correct = 0
    counter = 0
    for example, truth in zip(test[0:1000], test_labels[0:1000]):
        label = exact_knn(example, training, training_labels)
        num_correct += 1 if label == truth else 0
        counter += 1
        if counter % (len(test)//100) == 0:
            print('*', end='', flush=True)
    print('done!')
    print(f'Accuracy: {100*num_correct/len(test)}%')



