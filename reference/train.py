from src.read_dataset import get_images
from src.read_dataset import get_labels
import numpy as np
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    images = get_images("dataset/train-images.idx3-ubyte")
    labels = get_labels("dataset/train-labels.idx1-ubyte")

    test_images = get_images("dataset/t10k-images.idx3-ubyte")
    test_labels = get_labels("dataset/t10k-labels.idx1-ubyte")

    x = np.array(images)
    y = np.array(labels)

    x_test = np.array(test_images)
    y_test = np.array(test_labels)

    x = x.reshape(x.shape[0], -1)
    # normalize
    x = x / 255

    x_test = x_test.reshape(x_test.shape[0], -1)
    x_test = x_test / 255

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=100,
        verbose=True
    )

    model.fit(x, y)

    print("Test accuracy:", model.score(x_test, y_test))



