## SimpleGrad.jl

Hi, thanks for checking out `SimpleGrad.jl`! This is a machine learning package that can be used for a variety of applications involving gradient-tracking and backpropagation, including neural networks. 

However, unlike most other machine learning packages, the primary goal of SimpleGrad is to be *educational*. So if you're looking for the best possible speed and performance, you probably *shouldn't* use SimpleGrad (maybe check out [Flux](https://fluxml.ai/) instead). But if you're new to machine learning or Julia (or both) and want to understand how things work, then I think you'll find SimpleGrad useful! 

In fact, I started this project because I'm a beginner at machine learning and Julia myself, and figured the best way to learn was by doing -- or at least, attempting. This package is the result of my own attempt to learn about machine learning, gradient-tracking, and [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/) (Julia's alternative to object oriented programming), and my hope is that it will be useful for other beginners who are trying to learn. The idea is that the [source code](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) should be useful for some basic applications, while also being simple enough to read, understand, and edit/customize if you want to.

SimpleGrad includes two main [composite types](https://docs.julialang.org/en/v1/manual/types/#Composite-Types) (basically Julia's version of objects/classes): *Values* and *Tensors*. *Values* store single numbers and *Tensors* store arrays of numbers. Both *Values* and *Tensors* support a variety of operations, which are automatically tracked so that the gradients can be calculated with a backward pass.

In the [Usage](docs/src/usage.md) section, we'll cover how to actually use *Values* and *Tensors* to do calculations and compute gradients. Then in the [Under the Hood](under_the_hood.md) section, we'll take a look at the source code and talk about how it works. Lastly, I've also included two tutorials for extra practice (and will probably add more later): [linear regression](tutorials/linear_regression.md) and [MNIST](tutorials/mnist.md).

By the way, this project is an ongoing work-in-progress and I'm open to suggestions, criticisms, questions, and pull-requests. You can reach me by email at *mikest@udel.edu*.