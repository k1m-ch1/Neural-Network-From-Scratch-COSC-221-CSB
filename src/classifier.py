import pickle

from matplotlib import text
from src.utils import *
import copy

class MLPClassifier:

    activations = {
        "relu":(relu,d_relu),
        "logistic":(sigmoid, d_sigmoid),
        "tanh":(tanh, d_tanh),
        "identity":(identity, d_identity)
    }

    def __init__(self, hidden_layer_sizes:tuple[int,...], max_iter:int=20, activation:str='relu', verbose:bool=False) -> None:
        """
        We basically are implementing a stripped down version of the scikit learn MLPClassifier class. No solvers, we're only going to use stochastic gradient descent.

        Args:
            hidden_layer_sizes (tuple): a tuple representing the size of each hidden layer
            max_iter (int): the number of passes through the whole dataset

            verbose (bool): a boolean representing whether we want to display progress to the user

        Return:
            None
        """
        if hidden_layer_sizes == tuple():
            raise ValueError("The hidden layer must exist")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.verbose = verbose

        if activation not in self.activations:
            raise ValueError(f"The activation function called {activation} doesn't exist")

        self.activation, self.d_activation = self.activations[activation]

    def load_weights(self, coefs, intercepts, classes):
        """
        Load the weights and biases that has already been trained
        """

        # weights contains a list of weight matrix
        expected_weights_shape = tuple((self.hidden_layer_sizes[i],self.hidden_layer_sizes[i + 1])
            for i in range(len(self.hidden_layer_sizes) - 1))
        # (256, 128), (128, 64), (64, 10)
        actual_weights_shape = tuple(w.shape for w in coefs[1:-1])
        if actual_weights_shape != expected_weights_shape:
            raise ValueError(f"Please load weights of the right shape for this model. Expected hidden layer weight shape {expected_weights_shape}")

        expected_biases_shape = tuple((b, )
            for b in self.hidden_layer_sizes)
        actual_biases_shape = tuple(b.shape for b in intercepts[:-1])
        if actual_biases_shape != expected_biases_shape:
            raise ValueError(f"Please load biases of the right shape for this model. Expected hidden layer weight shape {expected_biases_shape}, got {actual_biases_shape}"
)
        if (coefs[-1].shape[1] != intercepts[-1].shape[0]):
            raise ValueError("The weights you passed doesn't match that shape of the biases.")

        if (intercepts[-1].shape[0] != len(classes)):
            raise ValueError("The classes you picked doesn't the size of the biases")

        self.coefs_ = copy.deepcopy(coefs)
        self.intercepts_ = copy.deepcopy(intercepts)
        self.classes_ = copy.deepcopy(classes)


    def predict_proba(self, X):
        """
        Given a list of samples, we do forward propagation according to the weights and biases to arrive at a prediction.

        Args:
            X (numpy darray): a numpy array with shape (number of samples, number of features)

        Return:
            Y (numpy darray): a numpy array with shape (number of sample, number of types of labels) which represents a probability distribution
        """

        try:
            # forward propagation
            Y = []
            for x in X:
                # need to flatten the image, or turn 2d list into 1d list
                y = x.flatten()
                for weight, bias in zip(self.coefs_[:-1], self.intercepts_[:-1]):
                    y = self.activation(y@weight + bias)

                # at the final layer, use softmax
                weight = self.coefs_[-1]
                bias = self.intercepts_[-1]
                y = softmax(y@weight + bias)
                Y.append(y)
            return Y
        except NameError as e:
            raise NameError(f"You can't do forward propagation before acquiring the weights: {e}")

    def predict(self, X):
        """
        A simple wrapper around predict_proba where it automatically maps the probability distribution to the best match among the classes.
        """
        Y = self.predict_proba(X)
        return [self.classes_[y.argmax()] for y in Y]

    def score(self, X, Y):
        """
        Just gives the percentage at which we predicted the right label.

        Args:
            X (numpy darray): a list of features
            Y (numpy darray): a list of labels for those features

        Return:
            result (tuple): (score, another tuple of index of failed prediction)
        """

        boolean_array_result = np.array(self.predict(X)) == np.array(Y)
        incorrect_indicies = np.where(~boolean_array_result)[0]
        score = sum(boolean_array_result)/len(boolean_array_result)
        return score, incorrect_indicies

    def fit(self, X, Y) -> None:
        """
        This is to train the model, in order to adjust the weights and biases.

        Args:
            X (numpy darray): a numpy array with shape (number of samples, number of features)
            Y (numpy darray): the labels with shape (number of sampels, )
        Return:
            None
        """
        pass

if __name__ == "__main__":
    #images = get_images_fast("dataset/train-images.idx3-ubyte")
    #labels = get_labels_fast("dataset/train-labels.idx1-ubyte")
    import matplotlib.pyplot as plt
    test_images = get_images_fast("dataset/t10k-images.idx3-ubyte")
    test_labels = get_labels_fast("dataset/t10k-labels.idx1-ubyte")

    #x = images.reshape(images.shape[0], -1)/255

    x_test = test_labels.reshape(test_labels.shape[0], -1)/255

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        max_iter=20,
        verbose=True,
    )

    with open("weights/sklearn_weights_and_biases.pkl", 'rb') as file:
        weights_and_biases = pickle.load(file)
        model.load_weights(weights_and_biases["weights"],
                           weights_and_biases["biases"],
                           weights_and_biases["classes"])
        N = 1_000
        #predicted_dist = model.predict_proba(test_images[:N])
        #predicted_label = model.predict(test_images[:N])
        #for i in range(N):
        #    plt.imshow(test_images[i])
        #    plt.title(f"this is an image of {test_labels[i]}, predicted dist: {predicted_dist[i]}, predicted label: {predicted_label[i]}")
        #    plt.show()
        score, incorrect_indicies = model.score(test_images[:N], test_labels[:N])

        print("score: ", score)
        print("incorrect indicies: ", incorrect_indicies)
        for i in incorrect_indicies:
            predicted_label = model.predict([test_images[i]])[0]
            plt.imshow(test_images[i])
            plt.title(f"actual label: {test_labels[i]}, predicted label: {predicted_label}")
            plt.show()
