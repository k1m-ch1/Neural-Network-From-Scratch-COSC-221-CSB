import pickle
import numpy as np

if __name__ == "__main__":
    with open("weights/sklearn_weights_and_biases.pkl", 'rb') as file:
        weights_and_biases = pickle.load(file)
        print([x.shape for x in weights_and_biases["weights"]])
        print([x.shape for x in weights_and_biases["biases"]])
        print(weights_and_biases["classes"])
        #print(weights_and_biases)

