import cv2
import numpy as np
import train_and_test as t
import matplotlib.pyplot as plt

features = ['extent', 'solidity', 'd/D',
            '4A/(pi*d^2)', 'pi*d/P', '4*pi*A/P^2', 'P/(pi*D)',
            'A1', 'A2', 'B', 'F4', 'F5', 'F6', 'F10', 'F11', 'F12',
            'F13', 'F14', 'F15']

features_left = features.copy()

dropped_features_indices = []

x = np.load("./dataset/balanced/features.npy")[:3000]
y = np.load("./dataset/balanced/GroundTruth.npy")[:3000]


def remove_one_feature():
    max_acc = {
        'index': -1,
        'accurancy': 0
    }

    current_x = np.delete(x, dropped_features_indices, 1)

    for i in range(0, len(features) - len(dropped_features_indices)):
        test_features = np.delete(current_x, i, 1)
        acc = t.train_and_test(test_features, y, 4)
        if (acc > max_acc['accurancy']):
            max_acc['index'] = i
            max_acc['accurancy'] = acc
    print(max_acc)
    return max_acc['index'], max_acc['accurancy']


results = []

for _ in range(0, 18):
    index, acc = remove_one_feature()
    dropped_feature_index = features.index(features_left.pop(index))
    print(features[dropped_feature_index])
    dropped_features_indices.append(dropped_feature_index)
    results.append(acc)

plt.plot(results)
plt.show()
