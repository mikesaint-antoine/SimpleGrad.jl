# SimpleGrad.jl

Hi, thanks for checking out `SimpleGrad.jl`! This is a machine learning package that can be used for a variety of applications that involve gradient-tracking and backpropagation, including neural networks. 

However, unlike most other machine learning packages, the primary goal of SimpleGrad is to be *educational*. So if you're looking for the best possible speed and performance, you probably *shouldn't* use SimpleGrad (maybe check out [Flux.jl](https://fluxml.ai/) instead). But if you're new to machine learning or Julia (or both) and want to understand how things work, then I think you'll find SimpleGrad useful! 

In fact, I started this project because I'm a beginner at machine learning and Julia myself, and figured the best way to learn was by doing (or at least, attempting). This package is the result of my own attempt to learn about machine learning, gradient-tracking, and [multiple dispatch](https://docs.julialang.org/en/v1/manual/methods/) (Julia's alternative to object oriented programming), and my hope is that it will be useful for other beginners who are trying to learn. The idea is that the [source code](https://github.com/mikesaint-antoine/SimpleGrad.jl/blob/main/src/SimpleGrad.jl) should be useful for some basic applications, while also being simple enough to read, understand, and edit/customize if you want to.

SimpleGrad includes two main [composite types](https://docs.julialang.org/en/v1/manual/types/#Composite-Types) (basically Julia's version of objects/classes): *Values* and *Tensors*. *Values* store single numbers, and *Tensors* store arrays of numbers. Both *Values* and *Tensors* support a variety of operations, which are automatically tracked so that the gradients can be calculated with a backward pass.

In the [Usage](usage.md) section, we'll cover how to actually use *Values* and *Tensors* to do calculations and calculate gradients. Then in the [Under the Hood](under_the_hood.md) section, we'll take a look at the source code and talk about how it works. 






## Introduction

Hi! Thanks for checking out this Julia library!

This library is based on Andrej Karpathy's Micrograd (which is in Python), and has a similar goal: to provide basic gradient-tracking, with code **simple enough to read** and **easy enough to edit and customize.** 

Like the original Python Micrograd, we'll be using a *Value* class for scalar-based gradient tracking. 

In addition, I've also added a *Tensor* class for array/matrix-based gradient tracking. Even though this wasn't included in the original Micrograd, I decided to add it so we could do simple examples like MNIST without the code taking forever.

I'm hoping that this will be a useful resource for people who are just getting started with Julia and trying to learn the basics. 


## Installation

To install the package, you need to...

```julia
using Pkg
Pkg.add("TestPackage")
```


Right now there's no fancy installation with a package manager or anything. You need to actually download the *mikerograd.jl* file form this repo, put it somewhere on your computer, and then direct Julia to it's location in order to use it. Like this, but replace the location with its location on your computer:

```julia
push!(LOAD_PATH, "/Users/mikesaint-antoine/Desktop/new_mikerograd") 
# change this to the location of the folder where mikerograd.jl is on your computer

using mikerograd
```

## Very Basics
There are two types of objects in mikerograd: *Values* and *Tensors*.

