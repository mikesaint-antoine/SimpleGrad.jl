## SimpleGrad.jl Basics

Hi, thanks for checking out `SimpleGrad.jl`! This is a machine learning package that can be used for a variety of applications involving gradient-tracking and backpropagation, including neural networks. 

However, unlike most other machine learning packages, the primary goal of SimpleGrad is to be *educational*. So if you're looking for the best possible speed and performance, you probably *shouldn't* use SimpleGrad (maybe check out [Flux](https://fluxml.ai/) instead). But if you're new to machine learning or Julia (or both) and want to understand how things work, then I think you'll find SimpleGrad useful! 

In fact, I started this project because I'm a beginner at machine learning and Julia myself, and figured the best way to learn was by doing -- or at least, attempting. This package is the result of my own attempt to learn about machine learning, gradient-tracking, and [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/) (Julia's alternative to object oriented programming), and my hope is that it will be useful for other beginners who are trying to learn. The idea is that the [source code](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) should be useful for some basic applications, while also being simple enough to read, understand, and edit/customize if you want to.

SimpleGrad includes two main [composite types](https://docs.julialang.org/en/v1/manual/types/#Composite-Types) (basically Julia's version of objects/classes): *Values* and *Tensors*. *Values* store single numbers and *Tensors* store arrays of numbers. Both *Values* and *Tensors* support a variety of operations, which are automatically tracked so that the gradients can be calculated with a backward pass.

In the [Usage](usage.md) section, we'll cover how to actually use *Values* and *Tensors* to do calculations and compute gradients. Then in the [Under the Hood](under_the_hood.md) section, we'll take a look at the source code and talk about how it works. Lastly, I've also included two tutorials for extra practice (and will probably add more later): [linear regression](tutorials/linear_regression.md) and [MNIST](tutorials/mnist.md).

By the way, this project is an ongoing work-in-progress and I'm open to suggestions, criticisms, questions, and pull-requests. You can reach me by email at *mikest@udel.edu*.

## Installation

If everything's working properly, you should be able to install SimpleGrad right from the Julia package manager like this:

```julia
using Pkg
Pkg.add("SimpleGrad")
```
However, if that gives you any problems then it probably means that I screwed something up. Sorry! I'm a beginner here and this is my first time trying to make a Julia package. So as a backup, you can also just grab the [source code file](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) and put it on your computer. Then, just direct Julia to its location in order to use it. Like this, but replace the location with its location on your computer:

```julia
push!(LOAD_PATH, "/Users/mikesaint-antoine/Desktop/") 
# change this to the location of the folder where SimpleGrad.jl is on your computer

using SimpleGrad
```
In fact, even if installation through ```Pkg``` is working, grabbing the source code directly might be the better approach for educational purposes, so that you can easily edit it and play around with it.


## Credits

In this section I'd like to credit/cite a couple people who taught me all this stuff.

- Andrej Karpathy for his [Micrograd tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0)
- Harrison Kinsley (Sentdex on Youtube) for his [Neural Net From Scratch in Python textbook](https://nnfs.io/)
- u/Bob_Dieter for his [reddit comments](https://www.reddit.com/r/Julia/comments/18knzll/comment/kdytys3/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) -- my first attempt at writing a Julia gradient tracker was this project called [mikerograd](https://github.com/mikesaint-antoine/mikerograd). At the time I was a total beginner, coming from Python, and really didn't know how to use Julia's multiple dispatch, so I tried to write it similarly to how it would look in OOP / Python style. I posted it on Reddit, and u/Bob_Dieter replied with a series of very helpful comments about how it could be done better with multiple dispatch / function overloading. Thanks Bob!