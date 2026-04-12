from src.classifier.utils import *
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

if __name__ == "__main__":
    x = get_images_fast("dataset/train-images.idx3-ubyte")
    y = get_labels_fast("dataset/train-labels.idx1-ubyte")

    x_test = get_images_fast("dataset/t10k-images.idx3-ubyte")
    y_test = get_labels_fast("dataset/t10k-labels.idx1-ubyte")

    # reshape and normalize
    x = x.reshape(x.shape[0], -1)/255

    # reshape and normalize
    x_test = x_test.reshape(x_test.shape[0], -1)/255

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        solver="sgd",
        alpha=10e-4,
        momentum=0.9,
        max_iter=200,
        verbose=True
    )
    model.fit(x, y)
    print([W.shape for W in model.coefs_])
    print([b.shape for b in model.intercepts_])
    print("Test accuracy:", model.score(x_test, y_test))
    #print(model.coefs_)
    #print(model.intercepts_)
    #np.savez("weights/sklearn_weights.npz",
    #         weight=model.coefs_,
    #         bias=model.intercepts_,
    #         allow_pickle=True)
    
    #with open("weights/sklearn_weights_and_biases.pkl", 'wb') as file:
    #    pickle.dump(
    #        {
    #            "weights":model.coefs_,
    #            "biases":model.intercepts_,
    #            "classes":model.classes_
    #        },
    #        file
    #    )
