Neural networks are essentially mathematical expressions.

At the most fundamental level, a neural network consists of layers of neurons. Each neuron has its own weights and biases. It takes the input values, computes their weighted sum, adds the bias, and typically applies a non-linear activation function such as ReLU or tanh before passing the result to the next layer. During training, these parameters are continuously updated according to a loss function, which measures the difference between the network's prediction and the ground truth.

The optimization process can be understood through two main stages: **forward propagation** and **backward propagation**.

During the forward pass, the network processes the input through each layer and produces its prediction. At the same time, a **computation graph** of the mathematical operations is constructed dynamically. Operations such as addition, multiplication, and power become nodes in a **Directed Acyclic Graph (DAG)**. The prediction is eventually compared with the ground truth to produce the loss, which becomes the root of this computation graph.

The objective is to minimize the loss. To determine how each parameter (weights and biases) contributes to the loss, we calculate the gradient of the loss with respect to each parameter. This can be done efficiently by recursively applying the **chain rule** through the computation graph. This is essentially what backpropagation does: it propagates gradients from the loss back toward the parameter nodes.

The gradient at a particular point points in the direction of steepest ascent. Therefore, to minimize the loss, we move in the opposite direction. This is the fundamental idea behind **gradient descent**. Parameters are iteratively updated in the opposite direction of their gradients, with the learning rate controlling the magnitude of each update. In practice, this optimization is usually performed iteratively over mini-batches, repeatedly performing a forward pass, computing the loss, propagating the gradients backward, and updating the parameters.
