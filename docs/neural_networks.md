# Neurons

So Neural networks are composed of neurons chained together.

So what's a neuron?

A neuron is a function that takes in inputs and collapse it to an output according to some weight plus a bias, going into an activation function $g$.

$$
y = g(w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b)
$$

Since Neural networks has so many neurons, we need to have some language to describe the network itself, so we'll use the following notation:

- a neural network has $L$ layers (in fact, it's $L + 1$ layers, but we say that it's $L$ layers if we don't count the input layer)
- the output layer $l$ represented as a vector $a^{[l]}$
- the activation function at layer $l$ is $g^{[l]}$
- the pre-activation vector $z^{[l]}$
- the weight matrix $W^{[L]}$
- the bias matrix $b^{[L]}$
- each layer $l$ outputs a vector of size $n^{[L]}$
- the loss function is written as $\mathcal{L}$

\begin{align*}
&a^{[l]} = g^{[l]}(z^{[l]})\\
&z^{[l]} = W^{[l]}a^{[l - 1]} + b^{[l]}\\
\end{align*}

So you start an input layer $a^{[0]}$ with dimensions $n^{[0]}$, then you just run the it through this recurrence relation.

Usually:

- $W^{[l]}$ is to be determined in all layers
- $g^{[L]}$ (the activation function of the last layer) is a softmax (for classification, which is what we'll be doing)
- $g^{[l]}$ is usually just the $\text{ReLU}$ function
- $\mathcal{L}$ is the cross-entropy loss function

Note that the cross-entropy is written like:

$$
\mathcal{L} (y, \hat{y}) = - \sum_{i = 1}^{n^{[L]}} y_i \ln (\hat{y_i})
$$

Also, $\hat{y} = a^{[L]}$ and $y$ are just the labels. Similarly, $x = a^{[0]}$ is the input.

Now the $\text{ReLU}$ function is defined as such:

$$
\text{ReLU} (x) = \begin{cases}
& x \text{ if } x < 0\\
& x \text{ otherwise } 
\end{cases}
$$

Now the softmax function is defined as such:

$$
s(x_i) = \frac{e^{x_i}}{\sum_{j = 1}^{n^{[L]}} e^{x_j}}
$$

So for now, we use something called "stochastic gradient descent", which means to only use the output of the loss function directly.

If we want to adjust weights in batches, that is, if we divide our whole dataset to batches of size $m$, we simply take the average of all the result of the loss function in that batch.

Basically, we say that:

- $\hat{y}^{(i)}$ is the predicted output from the $i$th input in a total of $m$ batches
- $y^{(i)}$ is the label of the $i$th element of the batch

The cost function $J$ is simply the average of the result of each of the loss function in that batch.

$$
J = \frac{1}{m} \sum_{i = 1}^m \mathcal{L} (y^{(i)}, \hat{y}^{(i)})
$$

We're not going to use that because that's gonna add extra complexity.

So that's pretty much forward propagation done, in fact, to recap, I'll write another section on forward propagation.

# Forward propagation

So we're already given all the weights $W^{[l]}$, and also the bias $b^{[l]}$ (we set it to some random value at the start).

In fact, it seems like the standard values for initialization seem to be:

- $b^{[0]} = 0$
- $W^{[l]}$ according to the following distribution:

$$
W^{[l]} \sim \mathcal{N} (0, \frac{2}{n^{[l - 1]}})
$$

This is called the He initialization.

Now, we're also given input $x = a^{[0]}$ and also labels $y = a^{[L]}$.

So, basically, from $0 < l < L$ just run it through the following formula:

\begin{align*}
& a^{[l]} = g^{[l]} (z^{[l]})
& z^{[l]} = W^{[l]} a^{[l - 1]} + b^{[l]}
\end{align*}

Where:

- $g^{[l]}(z^{[l]}) = \text{ReLU}(z^{[l]})$ from $0 < l < L$

Once you've arrived at $a^{[L - 1]}$, now just change $g^{[L]}(z^{[L]}) = s(z^{[L]})$ where $s(z^{[L]})$ is the softmax function.

Finally, we have $\hat{y} = a^{[L]}$.

Now we need to compute the actual loss function itself

$$
\mathcal{L}(y, \hat{y}) = - \sum_{i = 1}^{n^{[L]}} y_i \ln (\hat{y}_i)
$$

Notice that since $0 \leq \hat{y}_i \leq 1$ since we ran it through a softmax earlier (it's probably never going to be 0, but in theory, it could be 0).

In practice, we just shift it by some small value $\epsilon$ in case that it's $0$. Maybe we should just set it to $\epsilon = 10^{-8}$ or sth.

$$
\mathcal(y, \hat{y}) = \sum_{i = 1}^{n^{[L]}} y_i \ln(\hat{y_i} + \epsilon)
$$

That's pretty much it for forward propagation, to go from $x = a^{[0]}$ to $\mathcal{L}(y, \hat{y})$ where $y = a^{[L]}$.

# Back propagation

So that was the easy stuff done, back propagation is where things start to get more complicated.

The goal of back propagation is to adjust all the weights and biases automatically such that our loss $\mathcal{L}(y, \hat{y})$ is minimized (for stochastic gradient descent). More generally, you'd want to minimize the cost, but it's just the average of the loss for batches, so the main idea is still the same.

The tools used in calculus is incredibly helpful since we're dealing with an optimization problem here.

So we want to minimize:

$$
\mathcal{L}(y, \hat{y})
$$

The way the function is written kinda misleading, it actually depends on all the following:

- input $a^{[0]}$
- the labels $y$
- weights $W^{[l]}$ for $0 < l < L$
- biases $b^{[l]}$ for $0 < l < L$

So basically, this loss function $\mathcal{L}$ depends on so many variables!

Now luckily, there's a fact in calculus that goes like this:

> The gradient of a multi-variate function "points" in the "direction" of greatest ascent

Now, as humans, we can't visualize dimensions higher than 3 (average humans), so this theorem is a bit more general than that.

This theorem basically says something like:

$$
\max_{\hat{u} \in n, |\hat{u}| = 1} (D_{\hat{u}} \mathcal{L}) \implies \hat{u} = \frac{\nabla \mathcal {L}}{|\nabla \mathcal{L}|}
$$

Now, we're not trying to maximize the loss function, we're trying to minimize it. For some reason, even in higher dimensions, the direction of steepest descent is opposite of the direction of steepest ascent.

> [!NOTE]
> If you ever want to prove this, the fact that the directional derivative $D_{\hat{u}} \mathcal{L} = \hat{u} \nabla \mathcal{L}$ is really helpful. Of course, you'd need to prove that too. 

$$
\min_{\hat{u} \in n, |\hat{u}| = 1} (D_{\hat{u}} \mathcal{L}) \implies \hat{u} = - \frac{\nabla \mathcal {L}}{|\nabla \mathcal{L}|}
$$

So that sounds simple in principle, it's too bad this loss function $\mathcal{L}$ has so many variables. Luckily, we can just use the chain rule to propagate backwards in order to find how each weight in the weight matrices and each bias in the bias vectors affect our loss function $\mathcal{L}$.
