from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=20,
        verbose=True
    )

    model.fit

    print(model.coefs_)
    print(model.intercepts_)
