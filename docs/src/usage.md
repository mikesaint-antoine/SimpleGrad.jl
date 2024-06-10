## *Values*

Let's start with the *Value* composite type. Here's how you define a *Value*

```julia
using SimpleGrad

x = Value(4.0)

println(x)
# output: Value(4.0)
```

*Values* can store numbers, perform operations, and automatically track the gradients of the variables involved.

Here's how you take a look at the number a *Value* is storing (called ```Value.data```), and its gradient (called ```Value.grad```):

```julia
println(x.data) # the number
# output: 4.0

println(x.grad) # the gradient
# output: 0.0
```

Here, ```x.data == 4.0``` because the *Value* ```x``` is storing the number ```4.0```, and ```x.grad == 0.0``` is a placeholder for the gradient, which could eventually change if we do some operations and eventually back-calculate the gradient.


Next let's try an operation. We'll define another *Value* called ```y```, add it to ```x```, and save the result as ```z```.

```julia
y = Value(3.0)
z = x + y

println(z)
# output: Value(7.0)
```

Pretty simple so far, right? But here's the cool part -- we can now do a backward pass to calculate the derivative of ```z``` with respect to ```x``` and ```y```. Here's how we do that:


```julia
backward(z)
```

Now, the ```grad``` fields of ```x``` and ```y``` are populated, and will tell us the derivative of ```z``` with respect to each of the inputs ```x``` and ```y```.


```julia
println(x.grad) # dz/dx = 1, meaning an increase of 1 in x will lead to an increase of 1 in z.
# output: 1.0

println(y.grad) # dz/dy = 1, meaning an increase of 1 in y will lead to an increase of 1 in z.
# output: 1.0
```

In mathematical terms, we're considering the equation ``z = x + y`` and are interested in the derivatives ``\frac{dz}{dx}`` and ``\frac{dz}{dy}``. ```x.grad == 1``` tells us that ``\frac{dz}{dx} = 1`` and ```y.grad == 1``` tells us that ``\frac{dz}{dy} = 1`` for the values of ```x``` and ```y``` that we've defined in our code (and in this specific example, for all values of ```x``` and ```y```). If you're rusty on the calculus, you can also think of it this way: increasing ```x``` by 1 will cause ```z``` to increase by 1, and increasing ```y``` by 1 will also cause ```z``` to increase by 1.

So that's the basic functionality of the *Value* class. We can store store numbers, do operations, and track the derivative of the output with respect to all of the inputs. This allows us to, for example, minimize a loss function through gradient-descent, by tracking the derivative of the loss with respect to the model parameters, and then updating those parameters so that the loss decreases.

Here's a list of the operations currently supported:
* **Addition**
* **Subtraction**
* **Multiplication**
* **Division**
* **Exponents**
* **e^x**
* **log()**
* **tanh()**

Let's test a couple of them out. We've already done addition, so let's try subtraction.

```julia
x = Value(10.0)
y = Value(3.0)
z = x - y

println(z)
# output: Value(7.0)
```

If you want, you can try ```backward(z)```, and you should be able to find ```x.grad == 1``` meaning that  ``\frac{dz}{dx} = 1``, and ```y.grad == -1``` meaning that ``\frac{dz}{dy} = -1``. But I'll skip over that for now.

Next let's try multiplication.


```julia
x = Value(6.0)
y = Value(2.0)
z = x * y

println(z)
# output: Value(12.0)
```

And again, we can get the derivative with of ```z``` with respect to ```x``` and ```y```.

```julia
backward(z)

println(x.grad) # dz/dx = y = 2
# output: 2.0

println(y.grad) # dz/dy = x = 6
# output: 6.0
```

Alright, so far so good! Let's try division now:

```julia
x = Value(15.0)
y = Value(5.0)
z = x / y

println(z)
# output: Value(3.0)
```

And the backward pass:

```julia
backward(z)

println(x.grad) # dz/dx = 1/5 = 0.2
# output: 0.2

println(y.grad) # dz/dy = -15 / x^2 = -0.6
# output: -0.6
```

Ok, now let's try exponents. **NOTE:** for this function, the exponents here must be a regular number, NOT a *Value*. Might work on fixing this later.

```julia
x = Value(5.0)
y = 2 # NOTE - exponent can't be Value, must be int or float.
z = x^y

println(z)
# output: Value(25.0)
```

And here's the backward pass:

```julia
backward(z)

println(x.grad) # dz/dx = 2x = 10
# output: 10.0
```


Ok, now for the exponential function ``e^x``, which we'll call `exp()`.

```julia
x = Value(2.0)
z = exp(x)

println(z)
# output: Value(7.38905609893065)
```

And here's the backward pass:

```julia
backward(z)

println(x.grad) # dz/dx = e^x = (same thing we got for above)
# output: 7.38905609893065
```

Ok, now for the natural logarithm, which we call **log()**.

```julia
x = Value(10.0)
z = log(x)

println(z)
# output: Value(2.302585092994046)
```

And here's the backward pass:

```julia
backward(z)

println(x.grad) # dz/dx = 1/x = 0.1
# output: 0.1
```

Lastly, the **tanh()** function. Personally my trig is pretty rusty and I don't use this function very often, but I'm including it because it was in Andrej Karpathy's [Micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0), which the SimpleGrad *Value* is based on. **tanh()**  is useful as a possible activation function for a linear layer of neurons, to add nonlinearity and bound the layer outputs on [-1, 1].

```julia
x = Value(3.0)
z = tanh(x)

println(z)
# output: Value(0.9950547536867305)
```

And here's the backward pass:

```julia
backward(z)
println(x.grad) # dz/dx = 1 - tanh^2(x) = ????
# output: 0.009866037165440211
```

So far these examples have been pretty simple. But as long as we're using these simple functions, we can combine them in pretty complicated ways. The gradients can still be calculated for all the inputs, using backpropagation and the chain rule of derivatives.

Let's try out a complicated example to see this...


```julia
input1 = Value(2.3)
input2 = Value(-3.5)
input3 = Value(3.9)

weight1 = Value(-0.8)
weight2 = Value(1.8)
weight3 = Value(3.0)

bias = Value(-3.2)

y_pred = tanh(input1*weight1 + input2*weight2 + input3*weight3 + bias)
y_true = Value(0.8)

loss = (y_pred - y_true)^2

println(loss)
# output: Value(0.20683027474728832)
```

Here we're using 3 inputs, 3 weights, a bias, and a tanh() activation function to come up with some prediction in a regression problem, and calculating a loss by comparing it to the target value.

Even though this looks pretty complicated, we can still use **backward(loss)** to calculate the derivative of the loss with respect to everything.


```julia
backward(loss)

println(weight1.grad) # dloss/dweight1
# output: -1.8427042527651991

println(weight2.grad) # dloss/dweight2
# output: 2.80411516725139

println(weight3.grad) # dloss/dweight3
# output: -3.12458547208012

println(bias.grad) # dloss/dbias
# output: -0.8011757620718257
```





## *Tensors*

*Values* are pretty useful for some specific cases, but unfortunately their scalar-valued calculations will be too slow when it comes to implementing even a pretty basic neural network. So in addition to *Values*, we also have our *Tensor* composite type, which stores data in array format (either one-dimensional or two-dimensional).

We can define a *Tensor* like this:

```julia
x = Tensor([2.0, 3.0, 4.0])

println(x)
# output: Tensor([2.0, 3.0, 4.0])
```

Similarly to *Values*, *Tensors* also have fields called ```data``` and ```grad``` that store their arrays of numbers and gradients.

```julia
println(x.data)
# output: [2.0, 3.0, 4.0]

println(x.grad)
# output: [0.0, 0.0, 0.0]
```

Right now the *Tensor* class pretty much has the bare minimum needed to implement a simple neural network, although I'm probably going to add more in the future. Here's a list of the operations currently supported:
* **Addition**
* **Matrix Multiplication**
* **ReLU**
* **Softmax Activation / Cross Entropy Loss Combination**

Rather than testing out all of these individually, let's see if we can save some time by testing them all out at once:

```julia
using Random
Random.seed!(1234)

inputs = Tensor(rand(2, 3)) # Matrix with shape (2,3) -- 2 batches, 3 input features per batch
weights1 = Tensor(rand(3, 4)) # Matrix with shape (3,4) -- takes 3 inputs, has 4 neurons
weights2 = Tensor(rand( 4, 5)) # Matrix with shape (4,5) -- takes 4 inputs, has 5 neurons
biases1 = Tensor([1.0,1.0,1.0,1.0]) # Bias vector for first layer neurons
biases2 = Tensor([1.0,1.0,1.0,1.0,1.0]) # Bias vector for second layer neurons


layer1_out = relu(inputs * weights1 + biases1)

layer2_out = layer1_out * weights2 + biases2


# important -- correct classes should be one-hot encoded and NOT a Tensor, just a regular matrix.
y_true = [0 1 0 0 0;
          0 0 0 1 0]

loss = softmax_crossentropy(layer2_out,y_true)



println(loss)
# output: Tensor([1.9662258101705288])
```

Now we can find the derivative of the loss with respect to the weights and biases (and inputs if we want although that isn't as relevant).

```julia
backward(loss)

println(weights1.grad)
# output: [0.15435974752037773 -0.15345737221995426 0.2758968460269525 0.10323749643003427; 0.10696292189737254 -0.18148549954816842 0.20715095141049542 0.12882715523280347; 0.16664054851985355 -0.23974576071873882 0.31358944957671503 0.16792563848560238] 

println(weights2.grad)
# output: [1.4368011084584609 -1.2194506134059484 0.01035073085763216 -0.28468347036857006 0.05698224445842545; 1.1107416179804015 -0.8773320376457919 0.008372825855784966 -0.28744229473297594 0.04565988854258152; 1.0101174661066419 -0.8246890949782356 0.007462028540626175 -0.233753522609643 0.04086312294061065; 0.9055652666627538 -0.7839803418601996 0.00643629203797843 -0.16355609507053923 0.035534878230006596]


println(biases1.grad)
# output: [0.1994372624495202, -0.31780293172407714, 0.38186796081101293, 0.22451103170524483]

println(biases2.grad)
# output: [0.5783840763706425, -0.4259635644934768, 0.0045351214402561, -0.18149144684303034, 0.024535813525608553]
```

Pretty cool! To see how all of this actually works, check out the [Under the Hood](under_the_hood.md) section. For more extensive tutorials, check out the [linear regression](tutorials/linear_regression.md) and [MNIST](tutorials/mnist.md) sections.