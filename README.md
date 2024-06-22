# SimpleGrad.jl

[[Documentation Site](https://mikesaint-antoine.github.io/SimpleGrad.jl)]

Hi, thanks for checking out `SimpleGrad.jl`! This is a machine learning package that can be used for a variety of applications involving gradient-tracking and backpropagation, including neural networks. 

However, unlike most other machine learning packages, the primary goal of SimpleGrad is to be *educational*. So if you're looking for the best possible speed and performance, you probably *shouldn't* use SimpleGrad (maybe check out [Flux](https://fluxml.ai/) instead). But if you're new to machine learning or Julia (or both) and want to understand how things work, then I think you'll find SimpleGrad useful! 

In fact, I started this project because I'm a beginner at machine learning and Julia myself, and figured the best way to learn was by doing -- or at least, attempting. This package is the result of my own attempt to learn about machine learning, gradient-tracking, and [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/) (Julia's alternative to object oriented programming), and my hope is that it will be useful for other beginners who are trying to learn. The idea is that the [source code](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) should be useful for some basic applications, while also being simple enough to read, understand, and edit/customize if you want to.

SimpleGrad includes two main [composite types](https://docs.julialang.org/en/v1/manual/types/#Composite-Types) (basically Julia's version of objects/classes): *Values* and *Tensors*. *Values* store single numbers and *Tensors* store arrays of numbers. Both *Values* and *Tensors* support a variety of operations, which are automatically tracked so that the gradients can be calculated with a backward pass.

The [documentation](https://mikesaint-antoine.github.io/SimpleGrad.jl) for this package contains a couple different parts. In the [Usage](https://mikesaint-antoine.github.io/SimpleGrad.jl/usage/) section, we cover how to actually use *Values* and *Tensors* to do calculations and compute gradients. In the [Under the Hood](https://mikesaint-antoine.github.io/SimpleGrad.jl/under_the_hood/) section, we take a look at the source code and talk about how it works. I've also included two tutorials for extra practice (and will probably add more later): [linear regression](https://mikesaint-antoine.github.io/SimpleGrad.jl/tutorials/linear_regression/) and [MNIST](https://mikesaint-antoine.github.io/SimpleGrad.jl/tutorials/mnist/).

By the way, this project is an ongoing work-in-progress and I'm open to suggestions, criticisms, questions, and pull-requests. You can reach me by email at *mikest@udel.edu*.

## Installation

If everything's working properly, you should be able to install SimpleGrad right from the Julia package manager like this:

```julia
using Pkg
Pkg.add("SimpleGrad")
```
As an alternative, you can also just grab the [source code file](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) and put it on your computer. Then, just direct Julia to its location in order to use it. Like this, but replace the location with its location on your computer:

```julia
push!(LOAD_PATH, "/Users/mikesaint-antoine/Desktop/") 
# change this to the location of the folder where SimpleGrad.jl is on your computer

using SimpleGrad
```
In fact, even if installation through ```Pkg``` is working, grabbing the source code directly might be the better approach for educational purposes, so that you can easily edit it and play around with it.


## Credits

Huge thanks to these people for teaching me how to do this stuff:

- Andrej Karpathy for his [Micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Harrison Kinsley (Sentdex on Youtube) for his [Neural Net From Scratch in Python textbook](https://nnfs.io/)
- u/Bob_Dieter for his [reddit comments](https://www.reddit.com/r/Julia/comments/18knzll/comment/kdytys3)