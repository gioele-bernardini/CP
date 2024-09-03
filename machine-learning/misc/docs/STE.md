### **Fully Connected Layer (Dense Layer)**

A **fully connected layer**, also known as a **dense layer**, is a type of layer in an artificial neural network where every neuron in the layer is connected to every neuron in the subsequent layer. This is a common layer type in traditional neural networks, particularly in feedforward networks, where each input is transformed into an output through a series of multiplication and addition operations followed by an activation function.

Mathematically, a fully connected layer performs the following operation:

\[
y = \sigma(Wx + b)
\]

Where:
- \(x\) is the input vector,
- \(W\) is the weight matrix,
- \(b\) is the bias vector,
- \(\sigma\) is the activation function,
- \(y\) is the output vector.

### **Quantization and Its Non-Differentiability**

**Quantization** is a process used to map a large set of input values to a smaller set. In the context of neural networks, quantization typically refers to reducing the precision of the network's weights and activations. For example, instead of using floating-point values, which have high precision, we might use only a few discrete values, such as -1 and 1 in binary neural networks (BNNs).

The quantization function can be mathematically described as:

\[
\text{quantize}(x) = \text{round}(x)
\]

For binary quantization, this might be something like:

\[
\text{quantize}(x) = 
\begin{cases} 
1 & \text{if } x \geq 0 \\
-1 & \text{if } x < 0 
\end{cases}
\]

This function is non-differentiable at \(x = 0\) because the gradient does not exist at this point—the function makes a sharp "jump" from one value to another. In addition, the gradient for any value of \(x\) that isn't exactly 0 is 0 because the function outputs constant values (-1 or 1), leading to no slope (derivative).

This non-differentiability poses a significant challenge for training neural networks using gradient-based methods, which rely on the calculation of derivatives to update the model's parameters.

### **Quantization via STE (Straight-Through Estimator) in Larq**

The **Straight-Through Estimator (STE)** is a technique used to overcome the problem of non-differentiability in quantization functions during the training of neural networks. Specifically, in Binary Neural Networks (BNNs), where weights and activations are quantized to binary values, STE allows the network to be trained using standard backpropagation.

In practice, the STE works by handling the forward and backward passes differently:
- **Forward pass**: The weights or activations are quantized using a non-differentiable function, such as the one described above.
- **Backward pass**: Instead of using the true gradient (which doesn't exist or is zero for quantization), the gradient is approximated using the gradient of the original, non-quantized function.

For example:
- During the forward pass, the output might be computed as \( \text{quantize}(x) \).
- During the backward pass, the gradient is calculated as if the function were simply \( y = x \) (i.e., the identity function), effectively allowing the gradient to "pass through" the quantization step without being affected by its non-differentiability.

This approach allows the training process to proceed as if the quantization step were smooth and differentiable, even though it is not.

### **Why Is It Called "Straight-Through Estimator"?**

The method is called the **Straight-Through Estimator (STE)** because, during backpropagation, it "passes through" the quantization function as if it doesn't exist, allowing the gradient to be computed as if the operation was a smooth and differentiable function.

### Why "Straight-Through"?
- **Straight-Through** literally means "pass straight through," which accurately describes what happens during backpropagation: the gradient is not blocked or altered by the non-differentiable quantization function. Instead of halting due to the non-existence of a gradient, the STE method allows gradients to "pass straight through" the quantization step by treating it as if it were an identity operation.

### Summary
In summary, the **Straight-Through Estimator** technique is crucial for training Binary Neural Networks, as it enables the use of standard backpropagation methods despite the non-differentiable nature of the quantization function. The method allows the network to be trained as if the quantization were a smooth, differentiable function, thereby combining the computational efficiency of quantization with the power of gradient-based optimization.

