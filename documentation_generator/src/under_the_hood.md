

## *Value* composite type

The basic *Value* composite type looks like this:

```julia
mutable struct Value{opType}
    data::Float64
    grad::Float64
    op::opType
end
```

There are three fields: `data`, `grad`, and `op`. We've seen *two* of these fields before, in the Usage section -- `Value.data` and `Value.grad`, representing the number being stored in the *Value* and its gradient. 

`Value.op` is something new that we'll be using behind the scenes as part of the gradient tracking. Basically, we'll use it to keep track of what operations and operands were used to create a *Value* object. To do this, we'll also need to define a new composite type of keep track of these operations. Here's what that looks like:


```julia
struct Operation{FuncType,ArgTypes}
    op::FuncType
    args::ArgTypes
end
```

`Operation.op` will tell us the operation type (addition, multiplication, etc) and `Operation.args` will point to the operands used in the operation, so that we can access them if we want to.

Next, we need a constructor so that we can initialize *Values*:

```julia
# constructor -- Value(data, grad, op)
Value(x::Number) = Value(Float64(x), 0.0, nothing)
```

Looks a bit complicated, I know, but let's break this down. We can initialize a *Value* object with `Value(x)` where `x` is some number. The `Value(Float64(x), 0.0, nothing)` part means that when we initialze a *Value* with `Value(x)`, this will set `Value.data = Float64(x)` (casting `x` to a Float64 if it's not already), `Value.grad = 0.0` and `Value.op = nothing`. The reason that the operation is set to "nothing" here is because we have initialized this *Value* ourselves rather than creating it as the result of an operation.

Next, a bit of code so that we can print out values and take a look at them.

```julia
import Base: show
function show(io::IO, value::Value)
    print(io, "Value(",value.data, ")")
end
```

This lets us print a *Value* and see the number that it's storing. The `import Base: show` at the top means that we're using a base Julia function called "show" and definining what it will do when we pass a *Value* as an input. We'll be doing this a lot, for many different base functions.

Ok, so that's our basic setup for *Values*. At this point, we should be able to run the following code:

```julia
x = Value(4.0)

println(x)
# output: Value(4.0)

println(x.data)
# output: 4.0

println(x.grad)
# output: 0.0

println(x.op)
# output:
```

## Defining *Value* addition

Alright, so we have our basic building block, but now we want to be able to actually do some calculations with it.

Let's start with addition. Bear with me for a second, I'm gonna give you the full block of code and then we'll go through it bit by bit:

```julia
import Base.+
function +(a::Value, b::Value)
    out = a.data + b.data

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(+, (a, b) ))

    return result
end
```

The `import Base.+` means that we're importing the base addition function, and `+(a::Value, b::Value)` means that we're defining what the `+` operator will do when used on two *Values*, which we call `a` and `b` for the purpose of the function definition. Basically this will allow us to do `x + y` where `x` and `y` are *Values* rather than regular numbers.

`out = a.data + b.data` is how we calculate the actual sum of the two input *Values* that will be stored in the output *Value*. Then we create a new *Value* with `result = Value(out, 0.0, Operation(+, (a, b) ))`. Hopefully this part looks familiar, since we're using the same constructor syntax as before. This will set `result.data = out` and `result.grad = 0.0`. The only new part here is that instead of setting `result.op = nothing`, we're setting `result.op = Operation(+, (a, b) )` to specify that this *Value* was created from an addition operation, and pointing to `a` and `b` as the operands, so that we can access them if we want to. 

Alright, I know things are getting a little complicated, but setting things up like this will give us a lot of power to go backwards through operations. For example, using only the parts we've written so far you should be able to run this code:

```julia
# define two Values
x = Value(2.0)
y = Value(3.0)

# add them together to get a new Value
z = x + y

println(z)
# output: Value(5.0)

# inspect the new Value to see what operation produced it
println(z.op.op)
# output: + (generic function with 194 methods)

# access the Values that were used as operands
println(z.op.args)
# output: (Value(2.0), Value(3.0))
```


## Defining *Value* backpropagation

Alright, now let's try to implement backpropagation for the addition operation. Basically the goal here is to be able to calculate the derivative of the output with respect to each of the inputs in the operation.

Before we actually write the code for this, I'll first show you what we want the end result to look like:

```julia
# define two Values
x = Value(2.0)
y = Value(3.0)

# add them together to get a new Value
z = x + y

# calculate the derivative of z with respect to the inputs
backward(z)

# the gradient of x tells us the derivative of z with respect to x
println(x.grad)
# output: 1.0

# dz/dx = 1, meaning an increase of 1 in x will lead to an increase of 1 in z.

# we can also check y.grad if we want to
println(y.grad)
# output: 1.0
```

Alright so that's how the end result should look, but now we need to actually write the code to get there. To do this, we're going to define a function called `backprop!()` that takes in a *Value* as an input, and then computes the gradients of the operands that were used to create the *Value*. This will be an internal function (not actually called by the user), but pretty soon we'll also define another function called `backward()` which will perform the full backward pass, calling `backprop!()` along the way. 

One of the cool things about Julia is something called "multiple dispatch" -- this means that you can define functions with the same name that do things differently based on the type of input that's passed in. If you recall, when we originally defined our *Value* object, we made it so that the object type contains information about the operation that was used to create it: `Value{opType}`.  

For example:
```julia
x = Value(2.0)
println(typeof(x))
# output: Value{Nothing}
```

We'll begin with the `backprop!()` function for this simple case, where the *Value* was not created by an operation, but rather defined by the user. In this case, we will just have the `backprop!()` function do nothing:

```julia
backprop!(val::Value{Nothing}) = nothing
```

Now we'll do the harder case, where `backprop!()` is applied to the result of an addition operation, to calculate the gradients of the operands. Let's look at the full code first, and then we'll discuss what each part is doing:

```julia
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}

    # update gradient of first operand
    val.op.args[1].grad += val.grad

    # update gradient of second operand
    val.op.args[2].grad += val.grad
end
```
I know, looks pretty confusing! Let's start with the function definition line: `backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}`. This is just saying that we're definining what the ``backprop!()` function will do when the input is a *Value* called `val` that was created in an addition operation.

Then, the function updates two things: `val.op.args[1].grad` and `val.op.args[2].grad`. This is how we access the gradients of the operands that were used to create `val`, so that we can update their gradients.

So how do we update the gradients? Well as we mentioned before, for a simple addition operation ``z = x + y`` the derivatives of ``z`` with respect to both variables are ``\frac{dz}{dx} = 1`` and ``\frac{dz}{dy} = 1``. This is because increasing either variable by some amount will cause ``z`` to increase by the same amount.

But wait a minute... the code in our `backprop!()` function looks way more complicated than that. We're *not* saying `val.op.args[1].grad = 1` and `val.op.args[2].grad = 1` (setting both gradients equal to 1). Instead we're saying `val.op.args[1].grad += val.grad` and `val.op.args[2].grad += val.grad` -- incrementing the operand gradients by the current value of the input *Value* gradient. The reason we're doing this is because we need to think ahead a little bit. All this complication isn't necessary for our simple ``z = x + y`` example, but we're trying to write this function in a general way so that it'll also work for more complicated examples in the futures. 

Here's a more complicated example we want to be able to handle (although still using only addition):


```julia
x = Value(2.0)
y = Value(3.0)

z = x + y

w = z + x # using x a second time

backward(w)

println(x.grad)
# output: 2.0

println(y.grad)
# output: 1.0

println(z.grad)
# output: 1.0
```

This introduces two complications: it has two layers to the calulation, and ``x`` is used *twice*. We use ``z`` as an intermediate variable to store the result of ``x+y``, but ultimately we're interested in ``w = z + x``, and we want to find the derivatives ``\frac{dw}{dx}``and ``\frac{dw}{dy}``.

This example helps explain the rational for writing `backprop!()` the way we did. We're calculating the gradients of the operands using the gradient of the input *Value* so that we can take advantage of the [chain rule](https://www.youtube.com/watch?v=YG15m2VwSjA): we can calculate ``\frac{dw}{dy} = \frac{dw}{dz}\frac{dz}{dy}``. This will involve two calls to `backprop!()`. First, we'll call `backprop!(w)` which will calculate the gradient of `z`, and then we'll call `backprop!(z)`, which will calculate the gradient of `y`.

Getting the gradient of `x` is a little more complicated. First, let's quicly prove to ourselves that ``\frac{dw}{dx} = 2``. The full equation for ``w`` is ``w = z + x``. We know that ``z = x + y``, so we can rewrite the equation for ``w`` as ``w = (x + y) + x``. Then, taking the derivative with respect to ``x`` gives us ``\frac{dw}{dx} = 2``. Since `x` contributes to the value of `w` twice, increasing `x` by some amount will increase `w` by *twice* that amount.

This is the rationale for why our `backprop!()` function increments the gradients its updating, rather than just setting them to some number. This lets us account for situations where the same *Value* contributes more than once to the final sum. In our example, `x.grad` will be updated twice by the `backprop!()` function -- once during the `backprop!(w)` call and once during the `backprop!(z)` call. Both of these updates will increase `x.grad` by 1, leaving us with our final answer of ``\frac{dw}{dx} = 2``.

Still with me? Alright, last part. We said before that `backprop!()` is an internal function that won't actually be called by the user. Rather, the user will call a wrapper function `backward()` on the final sum *Value*, and that function will do the full backward pass by calling `backprop!()` as many times as required to calculate the derivatives for all of the input *Values*. So now we need to write the `backward()` function.

First let's take a look at the full code, and then we'll discuss what each part is doing:

```julia
function backward(a::Value)

    function build_topo(v::Value, visited=Value[], topo=Value[])
        if !(v in visited)
            push!(visited, v)

            if v.op != nothing
                for operand in v.op.args
                    if operand isa Value
                        build_topo(operand, visited, topo)
                    end
                end
            end

            push!(topo, v)
        end
        return topo
    end
    
    topo = build_topo(a)

    a.grad = 1.0
    for node in reverse(topo)
        backprop!(node)
    end

end
```

When we call `backward(a)` (where `a` is the final result of some operations between *Values*) we want two things to happen. First, we want to fill up an array with `a` itself, and all of the other *Values* that were used to create `a`, sorted in topological order so that a *Value* comes after all of the dependencies used to calculate that *Value* -- meaning that `a` should be the last element in the array since everything else is a dependency of `a`. 

We can build this array with a recursive [depth-first search.] (https://en.wikipedia.org/wiki/Depth-first_search) This is what the nested function `build_topo()` is doing. `build_topo()` returns the topologically sorted array of all the *Values*, with `a` at the end. Then we set `a.grad = 1`, since the derivative of a variable with respect to itself is 1. Finally, we iterate backwards through the list of *Values* and call `backprop!()` on each one to update the gradients of its operands. 

That's it! We're done! With the code we've written up to this point, we can do as many addition operations between *Values* as we want, then do a backward pass on the final sum to calculate its derivative with respect to all the inputs that went into it. We did it!

Now some of you are probably thinking *Wait a minute, we're not done! All we have is an addition operation for Values, and we haven't even started with Tensors yet!* Ok, yeah, that's true. I guess what I mean is we're done with the *difficult part* -- the *Value* object structure, the logic of operation-tracking, and gradient-updating through backpropagation. Now that we're done with all that, adding more *Value* operations is easy. All we need to know is what the operation does, and how to calculate the derivative for it. Then we can use almost the exact same code we've already written, with only those parts changed. When we finally get to *Tensors*, the code will be almost exactly the same, except the operations and derivative calculations will be for matrix/vector form.


## Adding some robustness

Ok, time for a short digression. With our code so far, we can do addition operations between *Values*. This is a good start, but with a couple more lines of code we can add some robustness, so that we'll be able to do operations between *Values* and regular numbers.



First let's take the *Value* + *Number* case, like `Value(2.0) + 3.0` for example. Here's the code:

```julia
# addition for Value + Number
function +(a::Value, b::Number)
    b_value = Value(Float64(b)) # cast b to Value
    return a + b_value  # use the existing method for Value + Value
end
```


In this case, we just cast the *Number* to a *Value*, and then return the result of the *Value* + *Value* operation -- easy! Next, let's take the case where we have *Number* + *Value*, like `3.0 + Value(2.0)` for example. Here's the code:

```julia
# addition for number + Value
function +(a::Number, b::Value)
    return b + a # use Value + Number, which then casts the number to Value and does Value + Value
end
```

In this case, we just switch them around so that it becomes a *Value* + *Number* operation, which we've already covered. Now we should be able to run the following code without a problem:

```julia
test = Value(2.0) + Value(3.0) # Value + Value
println(test)
# output: Value(5.0)

test = Value(2.0) + 3.0 # Value + Number
println(test)
# output: Value(5.0)

test = 2.0 + Value(3.0) # Number + Value
println(test)
# output: Value(5.0)
```



## More *Value* operations

Alright, so let's add some more *Value* operations. We'll start with multiplication. Here's the code, for both the operation and the backward pass:

```julia
import Base.*
function *(a::Value, b::Value)
    out = a.data * b.data

    # Value(data, grad, op)
    result = Value(out, 0.0, Operation(*, (a, b) ))

    return result
end

# backprop for multiplication operation
function backprop!(val::Value{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}

    val.op.args[1].grad += val.op.args[2].data * val.grad

    val.op.args[2].grad += val.op.args[1].data * val.grad

end
```

That's it! Told you it was easy! The `*(a::Value, b::Value)` function is almost exactly the same as the addition function we wrote before, except that we're setting `out = a.data * b.data` and recording the operation as `Operation(*, (a, b) )`. 

The `backprop!()` function is also very similar to the one we wrote for addition, with just a couple small changes. First of all, we're now using `where {FunType<:typeof(*), ArgTypes}` in the funciton definition to specify that this is the version of `backprop!()` to use when the input variable was created with a multiplication operation (again, the cool thing about multiple dispatch is that we can define several versions of a function with different input types).

The second minor difference is that we need to change the way the derivates are calculated, since we're dealing with multiplication rather than addition. For a multiplicaiton operation ``z = xy`` the derivatives of ``z`` with respect to ``x`` and ``y`` are ``\frac{dz}{dx} = y``and ``\frac{dz}{dy} = x``. The two lines inside `backprop!()` are just saying this in code -- the gradient of each operand is incremented by the value of the other operand multiplied by the `val.grad` (the result of the operation) to allow for the chain rule. 

With the code we've written so far, we can do things like this:

```julia
x = Value(2.0)
m = Value(4.0)
b = Value(7.0)

y = m*x + b

backward(y)

println(m.grad)
# output: 2.0

println(x.grad)
# output: 4.0

println(b.grad)
# output: 1.0
```

By the way, Julia will still take care of the order of operations for us here, so we could have written `y = b + m * x` and gotten the same answer.

Just like in the addition case, we can also add some code make our multiplication robust to *Value* * *Number* and *Number* * *Value* cases:

```julia
# Value * number
function *(a::Value, b::Number)
    b_value = Value(Float64(b))  # cast b to Value
    return a * b_value # use the existing method for Value * Value
end

# number * Value
function *(a::Number, b::Value)
    return b * a # use Value * Number, which then casts the number to Value and does Value * Value
end
```

A lot of the operations will be like the multiplication case, where we'll need to write a new `backprop!()` operation. However, sometimes we can find a clever way to do things that avoids this. For example, this is how we'll implement *Value* subtraction:


```julia
import Base.-

# negation
function -(a::Value)
    return a * -1
end

# subtraction: Value - Value
function -(a::Value, b::Value)
    return a + (-b)
end
```

The first function `-(a::Value)` allows us to negate *Values* with a minus sign. This can be done by multiplying the *Value* by ``-1``, an operation we can already do with our `*(a::Value, b::Number)` function. The second function `-(a::Value, b::Value)` allows us to do subtraction with *Values* by negating the second *Value* and then adding them together. 

Pretty clever, right? This way we don't need to write a new `backprop!()` function for subtraction, because we've turned the subtraction operation into a combination of multiplication and addition. 

Anyway, from here it's just a matter of adding more operations so that we can do more calculations with our *Values*. There are the operations currently supported:

* **Addition**
* **Subtraction**
* **Multiplication**
* **Division**
* **Exponents**
* **e^x**
* **log()**
* **tanh()**

If you've understood everything up to this point, you should be able to read all the source code for the *Values* and make sense of it. If there are any operations you'd like to see added, either let me know and I'll try to add them, or you can also write them yourself and submit a pull request!

## *Tensor* composite type

*Tensors* work almost exactly the same way as *Values*, except with a little bit of extra complications that come with dealing with vectors and matrices. But the fundamentals are basically the same. We'll track operations with our *Operation* objects, override several base Julia functions to work for *Tensor* operations, and implement the backward pass with an internal `backprop!()` function and a user-facing `backward()` function.

The *Operation* object structure is the same as before:

```julia
struct Operation{FuncType,ArgTypes}
    op::FuncType
    args::ArgTypes
end
```

And here's our definition of the *Tensor* object structure:

```julia
mutable struct Tensor{opType}
    data::Union{Array{Float64,1},Array{Float64,2}}
    grad::Union{Array{Float64,1},Array{Float64,2}}
    op::opType
end
```

As you can see, it's very similary to the *Value* object, except that the `Tensor.data` and `Tensor.grad` fields are arrays (either one-dimensional or two-dimensional) rather than numbers. 

Here's the *Tensor* constructor:

```julia
Tensor(x::Union{Array{Float64,1},Array{Float64,2}}) = Tensor(x, zeros(Float64, size(x)), nothing)
```

Again, same basic idea as the *Value* constructor, except that we're dealing with arrays instead of numbers.

Just a couple more quick things to take care of before we start defining operations. The following code lets us print out *Tensors* and also sets the `backprop!()` function to be `nothing` in cases where a *Tensor* was defined by the user rather than being created in an operation (again, same as with *Values*):


```julia
import Base.show
function show(io::IO, tensor::Tensor)
    print(io, "Tensor(",tensor.data, ")")
end

backprop!(tensor::Tensor{Nothing}) = nothing
```

Ok, now let's try defining a *Tensor* operation. When we were learning about how *Values* work we started with addition because that seemed like the easiest. But for *Tensors*, addition will actually be a little tough because of some shape-broadcasting we'll need to do. So we'll start with matrix multiplication, since that will be easier. Here's the code:

```julia
import Base.*
function *(a::Tensor, b::Tensor)

    out = a.data * b.data

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(*, (a, b)))

    return result
end
```

Very similar to what we were doing with *Values*, except that this time it's matrix multiplication. But same idea. We do the matrix multiplication with `out = a.data * b.data`. Then we store the resulting matrix `out` along with an emptry gradient `size(out))` in a new *Tensor* called `result`, and record the operation as `Operation(*, (a, b))`. 

And here's the `backprop!()` function:

```julia
function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(*), ArgTypes}

    tensor.op.args[1].grad += tensor.grad * transpose(tensor.op.args[2].data)

    tensor.op.args[2].grad += transpose(tensor.op.args[1].data) * tensor.grad 

end
```

Again, basically the same idea as with the *Value* `backprop!()` functions. The only difficult part is that now everything has to be done for matrices, which makes the actual calculations more complicated and less intuitive. If you want, you can always work out a simple example on paper to prove to yourself that the gradient updates in this `backprop!()` function are actually correct. 

Finally, here's the code for the full backward pass:

```julia
function backward(a::Tensor)

    function build_topo(v::Tensor, visited=Tensor[], topo=Tensor[])
        if !(v in visited)
            push!(visited, v)

            if v.op != nothing
                for operand in v.op.args
                    if operand isa Tensor
                        build_topo(operand, visited, topo)
                    end
                end
            end

            push!(topo, v)
        end
        return topo
    end
    
    topo = build_topo(a)

    a.grad .= 1.0
    for node in reverse(topo)
        backprop!(node)
    end

end
```

Again, almost exactly the same as for the *Values*.




## More *Tensor* operations

For the sake of completeness, I'm going to give you the code for all of the *Tensor* operations needed to write a basic neural network. For now, working through the details to understand them will be left as an exercise for the reader, although I'll probably try to come back to this section to write a more complete description when I have some more free time.

Here's addition:

```julia
import Base.+
function +(a::Tensor, b::Tensor)

    if length(size(a.data)) == length(size(b.data))
        out = a.data .+ b.data
    elseif length(size(a.data)) > length(size(b.data))
        # a is 2D, b is 1D
        out = a.data .+ transpose(b.data)
    else
        # a is 1D, b is 2D
        out = b.data .+ transpose(a.data)
    end

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(+, (a, b)))

    return result
end

function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(+), ArgTypes}


    if length(size(tensor.grad)) > length(size(tensor.op.args[1].data))
        tensor.op.args[1].grad += dropdims(sum(tensor.grad, dims=1), dims=1) # need dropdims to make this size (x,) rather than size (1,x)
    else
        tensor.op.args[1].grad += ones(size(tensor.op.args[1].data)) .* tensor.grad
    end

    if length(size(tensor.grad)) > length(size(tensor.op.args[2].data))
        tensor.op.args[2].grad += dropdims(sum(tensor.grad, dims=1), dims=1) # need dropdims to make this size (x,) rather than size (1,x)
    else
        tensor.op.args[2].grad += ones(size(tensor.op.args[2].data)) .* tensor.grad
    end

end
```

Hint about this one: the complicated parts are related to broadcasting so that we can add add a one-dimensional vector to a two-dimensional matrix (like adding biases in a neural net).

Here's the ReLU activation function:

```julia
function relu(a::Tensor)

    out = max.(a.data,0)

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(relu, (a,)))

    return result
end

function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(relu), ArgTypes}

    tensor.op.args[1].grad += (tensor.op.args[1].data .> 0) .* tensor.grad

end
```


Here's the combined softmax activation and cross entropy loss:

```julia
function softmax_crossentropy(a::Tensor,y_true::Union{Array{Int,2},Array{Float64,2}}; grad::Bool=true)

    ## implementing softmax activation and cross entropy loss separately leads to very complicated gradients
    ## but combining them makes the gradient a lot easier to deal with

    ## credit to Sendex and his textbook for teaching me this part
    ## great textbook for doing this stuff in Python, you can get it here:
    ## https://nnfs.io/

    # softmax activation
    exp_values = exp.(a.data .- maximum(a.data, dims=2))
    probs = exp_values ./ sum(exp_values, dims=2)
    
    ## crossentropy - sample losses
    samples = size(probs, 1)
    probs_clipped = clamp.(probs, 1e-7, 1 - 1e-7)
    # deal with 0s


    # basically just returns an array with the probability of the correct answer for each batch
    correct_confidences = sum(probs_clipped .* y_true, dims=2)

    # negative log likelihood
    sample_losses = -log.(correct_confidences)


    # loss_mean
    out = [mean(sample_losses)]



    if grad

        # it's easier to do the grad calculation here because doing it seperately will involve redoing a lot of calculations

        samples = size(probs, 1)

        # convert from one-hot to index list
        y_true_argmax = argmax(y_true, dims=2)

        a.grad = copy(probs)
        for samp_ind in 1:samples
            a.grad[samp_ind, y_true_argmax[samp_ind][2]] -= 1
            ## this syntax y_true_argmax[i][2] is just to get the column index of the true value
        end
        a.grad ./= samples


    end

    # Tensor(data, grad, op)
    result = Tensor(out, zeros(Float64, size(out)), Operation(softmax_crossentropy, (a,)))

    return result
end



function backprop!(tensor::Tensor{Operation{FunType, ArgTypes}}) where {FunType<:typeof(softmax_crossentropy), ArgTypes}

end
```

Note: this is the most complicated one by far, and is also the odd-one-out in that the gradient is actually calculated during the forward pass (unless told not to), which the `backprop!()` function just does nothing. 

So yeah, sorry to leave you guys with a complicated one here. I'll probably come back later and try to write a more thorough description of this one when I have some more free time in the future.





