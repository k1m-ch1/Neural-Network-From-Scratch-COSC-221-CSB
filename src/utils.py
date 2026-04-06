import numpy as np

# functions for importing datasets

def get_images(filename):
    with open(filename, 'rb') as file:
        magic_bytes = file.read(4)
        number_of_images = int.from_bytes(file.read(4))
        number_of_rows = int.from_bytes(file.read(4))
        number_of_columns = int.from_bytes(file.read(4))

        return [[[int.from_bytes(file.read(1)) for _ in range(number_of_columns)]
            for _ in range(number_of_rows)]
            for _ in range(number_of_images)]

def get_labels(filename):
    with open(filename, 'rb') as file:
        magic_bytes = file.read(4)
        number_of_labels = int.from_bytes(file.read(4))
        return [int.from_bytes(file.read(1)) for _ in range(number_of_labels)]

def get_images_fast(filename):
    with open(filename, 'rb') as file:
        magic_bytes = int.from_bytes(file.read(4))
        if magic_bytes != 2051:
            raise ValueError("Magic bytes doesn't match that of idx3")
        number_of_images = int.from_bytes(file.read(4))
        number_of_rows = int.from_bytes(file.read(4))
        number_of_columns = int.from_bytes(file.read(4))

        images = np.frombuffer(file.read(), dtype=np.uint8)
        images = images[:number_of_images*number_of_rows*number_of_columns]
        return images.reshape(number_of_images, number_of_rows, number_of_columns)

def get_labels_fast(filename):
    with open(filename, 'rb') as file:
        magic_bytes = int.from_bytes(file.read(4))
        if magic_bytes != 2049:
            raise ValueError("Magic bytes doesn't match that of idx1")
        number_of_labels = int.from_bytes(file.read(4))
        return np.frombuffer(file.read(), dtype=np.uint8)

# some useful functions

def relu(z):
    """
    The ReLU function makes all input less than 0, 0. Leaves the rest unchanged.
    """
    return np.maximum(0, z)

def d_relu(z):
    """
    The derivative of the ReLU function is the unit step function.
    """
    return int(z > 0)

def sigmoid(z):
    """
    The sigmoid function
    """
    return 1/(1 + np.exp(-z))

def d_sigmoid(z):
    """
    The derivative of the sigmoid function
    """
    return sigmoid(z)*(1 - sigmoid(z))

def tanh(z):
    """
    The hyperbolic tangent function
    """
    # kinda redundant lol
    return np.tanh(z)

def d_tanh(z):
    return 1 - tanh(z)**2

def identity(z):
    return z

def d_identity(z):
    return 1

def softmax(z):
    exp_z = np.exp(z - np.max(z))  # stable softmax
    #exp_z = np.exp(z)  # stable softmax
    return exp_z / np.sum(exp_z)

def cross_entropy_loss(y, y_hat):
    epsilon = 1e-12
    y_hat = np.clip(y_hat, epsilon, 1 - epsilon) # in case that y_hat is so small that we run out of precision and it becomes 0
    return - np.sum(y*np.log(y_hat), y_hat)

if __name__ == "__main__":
    #import matplotlib.pyplot as plt
    #x = np.linspace(-5, 5, 100)
    #plt.plot(x, tanh(x), label="g(x)")
    #plt.plot(x, d_tanh(x), label="g'(x)")
    #plt.legend()
    #plt.show()
    #s = softmax(np.array([1,2,3]))
    #print(s)
    #print(np.sum(s))
    pass



