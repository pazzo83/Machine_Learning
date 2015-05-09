from matplotlib import pyplot as plt
import numpy as np

# We load the data with load_iris from sklearn
from sklearn.datasets import load_iris
data = load_iris()

# load_iris returns an object with several fields
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

for t in range(3):
    if t == 0:
        c = 'r'
        marker = '>'
    elif t == 1:
        c = 'g'
        marker = 'o'
    elif t == 2:
        c = 'b'
        marker = 'x'
    plt.scatter(features[target == t,0],
                features[target == t,1],
                marker=marker,
                c=c)

# numpy indexing
labels = target_names[target]

# petal length is at position 2
plength = features[:, 2]

# bool array
is_setosa = (labels == 'setosa')

max_setosa = plength[is_setosa ].max()
min_non_setosa = plength[~is_setosa].min()

# print data
print('Maximum of setosa: {0}'.format(max_setosa))
print('Minimum of others: {0}'.format(min_non_setosa))


# select non-setosa data
features = features[~is_setosa]
labels = labels[~is_setosa]

# new target table, is virginica
is_virginica = (labels == 'virginica')

# testing accuracy

def is_virginica_test(fi, t, reverse, example):
    # Apply threshold model to new example
    test = example[fi] > t
    if reverse:
        test = not test

    return test

def fit_model(features, labels):
    '''Learn a simple threshold model'''
    best_acc = -1.0
    # Loop over all the features:
    for fi in range(features.shape[1]):
        thresh = features[:, fi].copy()
        # test all feature values in order:
        thresh.sort()
        for t in thresh:
            pred = (features[:, fi] > t)

            # Measure the accuracy of this
            acc = (pred == labels).mean()

            rev_acc = (pred == ~labels).mean()
            if rev_acc > acc:
                acc = rev_acc
                reverse = True
            else:
                reverse = False
            if acc > best_acc:
                best_acc = acc
                best_fi = fi
                best_t = t
                best_reverse = reverse

    # A model is a threshold and an index
    return best_t, best_fi, best_reverse

def predict(model, features):
    '''Apply a learned model'''
    # A model is a pair as returned by fit_model
    t, fi, reverse = model
    if reverse:
        return features[:, fi] <= t
    else:
        return features[:, fi] > t

def accuracy(features, labels, model):
    '''Compute the accuracy of the model'''
    preds = predict(model, features)
    return np.mean(preds == labels)

def cross_validation(features, labels):
    correct = 0.0
    for ei in range(len(features)):
        # select all but one position at 'ei'
        training = np.ones(len(features), bool)
        training[ei] = False
        testing = ~training
        model = fit_model(features[training], labels)
        predictions = predict(model, features[training])
        correct += np.sum(predictions == labels)
    acc = correct/float(len(features))

    return acc

