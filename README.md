# Neural-Network-From-Scratch-COSC-221-CSB

Links:

- [github link](https://github.com/k1m-ch1/Neural-Network-From-Scratch-COSC-221-CSB)
- [pypi link](https://pypi.org/project/mlpclassifier/)

Neural Network to classify handwritten digits (a rite of passage project at this point lol).

We will try to re-implement a stripped down version of the `MLPClassifier` class from `scikit-learn` from first principles. With this, we can then train a general classifier using the Multi-Layered Perceptron model.

# To run

So since we've re-implemented an MLP using `scikit-learn`'s `MLPClassifier` as a template, the API should be familiar.

Downloading the package

```
pip install mlpclassifier
```

Or create a virtual environment to download it.

Using `uv`:

```
uv add mlpclassifier
```

To use the library:

```python
from classifier.classifier import *
# to load the MNIST dataset.
images = get_images_fast("dataset/train-images.idx3-ubyte")
labels = get_labels_fast("dataset/train-labels.idx1-ubyte")

test_images = get_images_fast("dataset/t10k-images.idx3-ubyte")
test_labels = get_labels_fast("dataset/t10k-labels.idx1-ubyte")

# pre-processing
X = images.reshape(images.shape[0], -1)/255
X_test = test_images.reshape(test_images.shape[0], -1)/255

model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        max_iter=1000,
        alpha=1e-3,
        batch_size=200,
        verbose=True,
)

SAVE_PATH = "weights/self_trained.pkl"

# if Ctrl-C, then it will save to that file path
model.fit(X, labels, save_path=SAVE_PATH)

N = 10_000
score, incorrect_indicies = model.score(X_test[:N], test_labels[:N])
print("score: ", score)
model.save(SAVE_PATH)
```


# Dataset

Download the dataset from Kaggle

```
curl -L https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset -o ./dataset.zip
```

Then just unzip it into a directory called `./dataset`

```
unzip -d dataset ./dataset.zip
```

Optional, but clean redundancy:

```
rm -r *-idx*-ubyte
```

I've removed some duplicates, so currently I have:

```
$ ls ./dataset/
 t10k-images.idx3-ubyte   train-images.idx3-ubyte
 t10k-labels.idx1-ubyte   train-labels.idx1-ubyte
```

So it seems like by convention:

- we divide our dataset into training data, and then testing data
- currently, it seems like we have 60k training examples and 10k testing examples
- they do this to see how well the model has generalized

# Reference model

For now, we'll use a reference model through `scikit-learn`. 

# TODO

- [x] debug all the row vector stuff
- [] package it in pip
- [] document the API

## Forward propagation

- [x] variable L for layer
- [x] a list $n^{[l]}$ for the size at each layer
- [] initialize using He's initalization
- [x] forward propagation step using that forward propagation formula

## Backward propagation

- [x] He's initialization
- [x] back propagation
- [x] scoring
- [x] saving
- [] make the learn rate $\alpha$ more adjustable
