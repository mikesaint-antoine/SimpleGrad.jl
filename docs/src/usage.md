



## *Value* Objects

TEST TEST TEST

Here's how you define a *Value*

```julia
using TestPackage
x = Value(4.0);

println(x)
```

*Value* objects can store numbers, perform operations, and automatically track the gradients of the outputs.

Here's how you take a look at the number a *Value* is storing, and it's gradient:

```julia
println(x.data) # the number, which is 4 in this case
```

```julia
println(x.grad) # the gradient, which is 0 for now, but will be automatically tracked as we do operations
```

Next let's try an operation. We'll define another *Value* called **y**, add it to **x**, and save the result as **z**

```julia
y = Value(3.0);
z = x + y;

println(z)
```

Pretty simple so far, right? But here's the cool part -- we can now do a backward pass to calculate the derivative of **z** with respect to **x** and **y**. Here's how we do that:


```julia
backward(z)
```

Now, the **grad** fields of **x** and **y** are populated, and will tell us the derivative of **z** with respect to each of the inputs **x** and **y**.


```julia
println(x.grad) # dz/dx = 1, meaning an increase of 1 in x will lead to an increase of 1 in z.
```

```julia
println(y.grad) # dy/dx = 1, meaning an increase of 1 in y will lead to an increase of 1 in z.
```

Pretty cool, right? So that's the basic functionality of the *Value* class. You can store store numbers, do operations, and track the derivative of the output with respect to all of the inputs. This allows you to, for example, minimize a loss function through gradient-descent.

Here's a list of the operations currently supported:
* **Addition**
* **Subtraction**
* **Multiplication**
* **Division**
* **Exponents**
* **e^x**
* **log()**
* **tanh()**

These are basically the same as in Karpathy's Python Micrograd, and basically implemented the same way. Let's test a couple of them out. We've already done addition, so let's try subtraction.

```julia
x = Value(10.0);
y = Value(3.0);
z = x - y;

println(z)
```

If you want, you can try **backward(z)**, and you should be able to find **x.grad** = *dz/dx* = 1 and **y.grad** = *dz/dy* = -1. But I'll skip over that for now.

Next let's try multiplication.


```julia
x = Value(6.0);
y = Value(2.0);
z = x * y;

println(z)
```

And again, we can get the derivative with of **z** with respect to **x** and **y**.

```julia
# backward pass for multiplication
backward(z)
println(x.grad) # dz/dx = y = 2
println(y.grad) # dz/dy = x = 6
```

Alright, so far so good! Let's try division now...

```julia
x = Value(15.0);
y = Value(5.0);
z = x / y;

println(z)
```

```julia
# backward pass for division
backward(z)
println(x.grad) # dz/dx = 1/5 = 0.2
println(y.grad) # dz/dy = -15 / x^2 = -0.6
```

Ok, now let's try exponents. **NOTE:** just as in the original Micrograd, the exponents here must be an int or float, NOT a *Value* object. Might work on fixing this later.

```julia
# exponents
x = Value(5.0);
y = 2; # NOTE - exponent can't be Value, must be int or float
z = x^y;

println(z)
```

```julia
# backward pass for exponent
backward(z)
println(x.grad) # dz/dx = 2x = 10
```


Ok, now for the exponential function e^x, which we will call **exp()**.

```julia
# e^x
x = Value(2.0);
z = exp(x);

println(z)
```

```julia
# backward pass for e^x
backward(z)
println(x.grad) # dz/dx = e^x = (same thing we got for above)
```

Ok, now for the natural logarithm, which we call **log()**.

```julia
# natural log
x = Value(10.0);
z = log(x);

println(z)
```

```julia
# backward pass for natural log
backward(z)
println(x.grad) # dz/dx = 1/x = 0.1
```


Lastly, the **tanh()** function. Personally my trig is pretty rusty and I don't use this function very often, but I'm including it because it was in the original Micrograd. I think Karpathy included it to use as a possible activation function for a linear layer of neurons, to add nonlinearity and bound the layer outputs on [-1, 1].

```julia
# tanh()
x = Value(3.0);
z = tanh(x);

println(z)
```

```julia
# backward pass for tanh()
backward(z)
println(x.grad) # dz/dx = 1 - tanh^2(x) = ????
```


So far these examples have been pretty simple. But as long as we're using these simple functions, we can combine them in pretty complicated ways. The gradients can still be calculated for all the inputs, using backpropagation and the chain rule of derivatives.

Let's try out a complicated example to see this...


```julia
input1 = Value(2.3);
input2 = Value(-3.5);
input3 = Value(3.9);

weight1 = Value(-0.8);
weight2 = Value(1.8);
weight3 = Value(3.0);

bias = Value(-3.2);

y_pred = tanh(input1*weight1 + input2*weight2 + input3*weight3 + bias);
y_true = Value(0.8);

loss = (y_pred - y_true)^2;

println(loss)
```

Here we're using 3 inputs, 3 weights, a bias, and a tanh() activation function to come up with some prediction in a regression problem, and calculating a loss by comparing it to the target value.

Even though this looks pretty complicated, we can still use **backward(loss)** to calculate the derivative of the loss with respect to everything.


```julia
backward(loss)

println(weight1.grad) # dloss/dweight1
println(weight2.grad) # dloss/dweight2
println(weight3.grad) # dloss/dweight3
println(bias.grad) # dloss/dbias

# if you wanted, you could also see the derivatives of the loss with respect to the inputs, y_pred, or y_true
# although in a typically neural net situation, those variables would not be updated in the gradient descent
```





## *Tensor* Class

The *Value* class from the original Micrograd is a great tool for understanding how backpropagation works and implementing gradient descent for simple problems like linear regression. Unfortunately though, it's far too slow to use for even simple neural net problems. So, we'll define a new *Tensor* object for those calculations.


```julia
x = Tensor([2.0, 3.0, 4.0]);
println(x)
```


```julia
println(x.data)
println(x.grad)
```

Right now the *Tensor* class pretty much has the bare minimum needed to implement a simple neural network. Here's a list of the operations currently supported:
* **Addition**
* **Matrix Multiplication / Dot Product**
* **Relu**
* **Softmax Activation / Cross Entropy Loss Combination**

Rather than testing out all of these individually, let's see if we can save some time by testing them all out at once:

```julia
# Tensor test -- attempting a forward pass of a simple neural net

# using Statistics
# using Random
# do we need these?

inputs = Tensor(rand(2, 3)); # Matrix with shape (2,3) -- 2 batches, 3 input features per batch
weights1 = Tensor(rand(3, 4)); # Matrix with shape (3,4) -- takes 3 inputs, has 4 neurons
weights2 = Tensor(rand( 4, 5)); # Matrix with shape (4,5) -- takes 4 inputs, has 5 neurons
biases1 = Tensor([1.0,1.0,1.0,1.0]); # Bias vector for first layer neurons
biases2 = Tensor([1.0,1.0,1.0,1.0,1.0]); # Bias vector for second layer neurons


layer1_out = relu(inputs * weights1 + biases1);

layer2_out = layer1_out * weights2 + biases2;


# important -- correct classes should be one-hot encoded and NOT a Tensor, just a regular matrix.
y_true = [0 1 0 0 0;
          0 0 0 1 0]

loss = softmax_crossentropy(layer2_out,y_true)



println(loss)
```

Now we can find the derivative of the loss with respect to the weights and biases (and inputs although that isn't as relevant).

```julia
backward(loss)

println("weights1 gradient:")
println(weights1.grad)
println()
println("weights2 gradient:")
println(weights2.grad)
println()
println("biases1 gradient:")
println(biases1.grad)
println()
println("biases2 gradient:")
println(biases2.grad)
println()
```


Pretty cool!

