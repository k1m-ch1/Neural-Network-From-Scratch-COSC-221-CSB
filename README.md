# Neural-Network-From-Scratch-COSC-221-CSB-

Neural Network to classify handwritten digits (a rite of passage project at this point lol)

# Dataset

Download the dataset from Kaggle

```
curl -L https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset -o ./dataset.zip ```
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

# TODO

## Forward propagation

- [] variable L for layer
- [] a list $n^{[l]}$ for the size at each layer
- [] initialize using He's initalization
- [] forward propagation step using that forward propagation formula

## Backward propagation
