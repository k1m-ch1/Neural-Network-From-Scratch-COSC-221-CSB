import pickle

from classifier.utils import *
import copy

class MLPClassifier:

    activations = {
        "relu":(relu,d_relu),
        "logistic":(sigmoid, d_sigmoid),
        "tanh":(tanh, d_tanh),
        "identity":(identity, d_identity)
    }

    learning_rates = ["constant",
                     "adaptive"]

    def __init__(self, hidden_layer_sizes:tuple[int,...],
                 max_iter:int=20,
                 activation:str='relu',
                 batch_size:int=200,
                 momentum:float=0.9,
                 alpha:float=1e-2,
                 learning_rate:str='constant',
                 beta:float=0.9,
                 epsilon:float=10e-8,
                 verbose:bool=False,
                 ) -> None:
        """
        We basically are implementing a stripped down version of the scikit learn MLPClassifier class. No solvers, we're only going to use stochastic gradient descent.

        Args:
            hidden_layer_sizes (tuple): a tuple representing the size of each hidden layer
            max_iter (int): the number of passes through the whole dataset
            batch_size (int): the number of batches that will go into the forward prop such that it will be averaged and updated in a single backpropagation step
            activation (string): the activation function, can be relu, sigmoid, tanh, identity
            alpha (float): the learning rate. Default is 1e-10.
            momentum (float): the momentum (using classical momentum).
            learning_rate (str): can either be 'constant' or 'adaptive'. If it's 'adaptive', we will use RMSProp to adust the effective learning rate.
            beta (float): the decay rate for RMSProp.
            epsilon (float): in order to prevent a division by zero
            verbose (bool): a boolean representing whether we want to display progress to the user
        Return:
            None
        """
        if hidden_layer_sizes == tuple():
            raise ValueError("The hidden layer must exist")

        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.verbose = verbose
        self.alpha = alpha
        self.batch_size = batch_size
        self.momentum = momentum
        self.beta = beta
        self.epsilon = epsilon

        if activation not in self.activations:
            raise ValueError(f"The activation function called {activation} doesn't exist")

        self.activation, self.d_activation = self.activations[activation]

        if learning_rate not in self.learning_rates:
            raise ValueError(f"The learning rate called {learning_rate} doesn't exist. Select one from {self.learning_rates}.")

        self.learning_rate = learning_rate

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
        self.coefs_momentum_ = []
        self.intercepts_momentum_ = []
        
        for i in range(len(self.intercepts_)):
                self.coefs_momentum_.append(np.zeros(self.coefs_momentum_[i].shape))
                self.intercepts_momentum_.append(np.zeros(self.intercepts_momentum_[i].shape))

    def forward(self, X):
        """
        Given a list of samples, we do forward propagation according to the weights and biases to arrive at a prediction.

        Args:
            X (numpy darray): a numpy array with shape (number of samples, number of features)

        Return:
            Y (list): a list that contains matrices representing a^{[l]} at layer l, for all m samples.
            Z (list): a list that contains matrices representing z^{[l]} at layer l, for all m samples.
        """

        try:
            # forward propagation
            Z = [X]
            Y = [X]
            for weight, bias in zip(self.coefs_[:-1], self.intercepts_[:-1]):
                z = Y[-1]@weight + bias
                Y.append(self.activation(z))
                Z.append(z)
            weight = self.coefs_[-1]
            bias = self.intercepts_[-1]
            z = Y[-1]@weight + bias
            Y.append(softmax(z))
            Z.append(z)
            return Y, Z

        except NameError as e:
            raise NameError(f"You can't do forward propagation before acquiring the weights: {e}")

    def predict_proba(self, X):
        """
        Given a list of samples, we do forward propagation according to the weights and biases to arrive at a prediction.

        Args:
            X (numpy darray): a numpy array with shape (number of samples, number of features)

        Return:
            Y (numpy darray): a numpy array with shape (number of sample, number of types of labels) which represents a probability distribution
        """
        return self.forward(X)[0][-1]

    def predict(self, X):
        """
        A simple wrapper around predict_proba where it automatically maps the probability distribution to the best match among the classes.
        """
        Y = self.predict_proba(X)
        print(Y.shape)
        return np.argmax(Y, axis=1)

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

    def fit(self, X, y, save_path=None) -> None:
        """
        This is to train the model, in order to adjust the weights and biases.

        Args:
            X (numpy darray): a numpy array with shape (number of samples, number of features) 
            y (numpy darray): the labels with shape (number of sampels, )
            save_path (string): path to save (optional)
        Return:
            None
        """
        try:
            if (len(X) != len(y)):
                raise ValueError("The number of example features should match that of the number of labels given")
            # first divide into batches
            # i hate naming conventions
            Y = y
            self.classes_ = np.unique(Y)
            self.coefs_ = []
            self.coefs_momentum_ = []
            self.intercepts_= []
            self.intercepts_momentum_ = []
            self.r_W = []
            self.r_b = []
            for i in range(len(self.hidden_layer_sizes) + 1):
                n = int()
                n_prev = int()
                if i == len(self.hidden_layer_sizes):
                    n = len(self.classes_)
                else:
                    n = self.hidden_layer_sizes[i]

                if i == 0:
                    n_prev = X.shape[1]
                else:
                    n_prev = self.hidden_layer_sizes[i - 1]
                # He's initialization sets variance = 2/n^{[l - 1]}
                self.coefs_.append(np.random.normal(0, np.sqrt(2/n_prev), size=(n_prev, n)))
                self.coefs_momentum_.append(np.zeros((n_prev, n)))
                self.intercepts_.append(np.zeros(n))
                self.intercepts_momentum_.append(np.zeros(n))
                self.r_W.append(np.zeros((n_prev, n)))
                self.r_b.append(np.zeros(n))

            L = len(self.hidden_layer_sizes) + 1
            for iter_num in range(self.max_iter):
                J = np.array([])
                for i in range(int(np.ceil(len(X)/self.batch_size))):
                    m = self.batch_size
                    x = X[self.batch_size*i:min(self.batch_size*(i + 1), len(X))]
                    y = Y[self.batch_size*i:min(self.batch_size*(i + 1), len(Y))]
                    # one hot labeling
                    indices= np.searchsorted(self.classes_, y)
                    y = np.eye(len(self.classes_))[indices]

                    # so we now need to propagate backwards

                    # first need to compute delta[L]
                    # forward propagation to get a^{[L]}
                    a, z = self.forward(x)

                    # J for the batch loss matrix
                    J = cross_entropy_loss(y, a[-1])
                    # first calculate the batch change in bias
                    delta = a[-1] - y
                    for l in reversed(range(1, L)): # l goes from L - 1 to 0
                        dW = (a[l].T @ delta)/m
                        db = np.mean(delta, axis=0)

                        # adding momentum
                        self.coefs_momentum_[l] = self.momentum*self.coefs_momentum_[l] + dW
                        self.intercepts_momentum_[l] = self.momentum*self.intercepts_momentum_[l] + db

                        if self.learning_rate == "adaptive":
                            #self.r_W[l] = self.beta*self.r_W[l] + (1 - self.beta)*(dW*dW)
                            #self.r_b[l] = self.beta*self.r_b[l] + (1 - self.beta)*(db*db)
                            
                            self.r_W[l] = self.r_W[l] + dW*dW
                            self.r_b[l] = self.r_b[l] + db*db

                            self.coefs_[l] = update(self.coefs_[l],
                                                    self.alpha/(np.sqrt(self.r_W[l]) + self.epsilon),
                                                    self.coefs_momentum_[l])

                            self.intercepts_[l] = update(self.intercepts_[l],
                                                         self.alpha/(np.sqrt(self.r_b[l]) + self.epsilon),
                                                         self.intercepts_momentum_[l])
                        else:
                            self.coefs_[l] = update(self.coefs_[l], self.alpha, self.coefs_momentum_[l])
                            self.intercepts_[l] = update(self.intercepts_[l], self.alpha, self.intercepts_momentum_[l])

                        if l > 0:
                            delta = (delta @ self.coefs_[l].T) * self.d_activation(z[l])

                if self.verbose:
                    print(f"Finished iteration {iter_num + 1}/{self.max_iter}. Loss: {np.mean(J, axis=0)}")

        except KeyboardInterrupt as e:
            if save_path != None:
                self.save(save_path)
            raise InterruptedError(f"Training canceled! Has saved weights to {save_path} as specified")


    def save(self, path):
        """
        Saving weights as a pickle file. Path is just the file path to save at.
        """
        with open(path, 'wb') as file:
                pickle.dump(
                    {
                        "weights":self.coefs_,
                        "biases":self.intercepts_,
                        "classes":self.classes_
                    },
                    file
                )


if __name__ == "__main__":
    images = get_images_fast("../dataset/train-images.idx3-ubyte")
    labels = get_labels_fast("../dataset/train-labels.idx1-ubyte")
    import matplotlib.pyplot as plt
    test_images = get_images_fast("../dataset/t10k-images.idx3-ubyte")
    test_labels = get_labels_fast("../dataset/t10k-labels.idx1-ubyte")

    X = images.reshape(images.shape[0], -1)/255

    X_test = test_images.reshape(test_images.shape[0], -1)/255

    model = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu',
        max_iter=200,
        learning_rate="adaptive",
        batch_size=200,
        verbose=True,
    )

    SAVE_PATH = "../weights/self_trained_momentum_4.pkl"

    model.fit(X, labels, save_path=SAVE_PATH)

    #with open("weights/sklearn_weights_and_biases.pkl", 'rb') as file:
    #    weights_and_biases = pickle.load(file)
    #    model.load_weights(weights_and_biases["weights"],
    #                       weights_and_biases["biases"],
    #                       weights_and_biases["classes"])

    N = 10_000
    score, incorrect_indicies = model.score(X_test[:N], test_labels[:N])
    print("score: ", score)
    model.save(SAVE_PATH)
    #print("incorrect indicies: ", incorrect_indicies)
    #predict, Z = model.forward(X_test[:N])
    #print([i.shape for i in Z])
    #predicted_label = model.predict(np.array([x_test[i] for i in incorrect_indicies]))
    #print(predicted_label)
    #for i in incorrect_indicies:
    #    predicted_label = model.predict(np.array([x_test[i]]))[0]
    #    plt.imshow(test_images[i])
    #    plt.title(f"actual label: {test_labels[i]}, predicted label: {predicted_label}")
    #    plt.show()
